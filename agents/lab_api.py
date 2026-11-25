"""
lab_api.py - the one door the future agents use to reach the simulated lab.

LabAPI holds the fake optics lab (optics_simulation) and the temperature-controlled box
(box_simulation.thermostat) on one shared clock. An agent can only do what an experimenter
could: wait, read the thermometer (Kalman and raw, never the true temperature), change the
box target, switch the control on or off, measure a map with GGS, check it and focus with it.

With a LabState (state.py), every public action is also written to the lab database, together
with a thermometer row every temperature_every_s; api.actor says who did it.

The sample follows the true box temperature as in drift_test: X_true = sample.at(T_true), a
lab.TemperatureSample that is exactly X0 at OPTICS T_ref_C (fixed; set_target never changes it).
Row convention (as in lab.py): E_out = E_in @ X, (L x M) = (L x N) @ (N x M).
"""

from __future__ import annotations

import time

import numpy as np

from box_simulation import thermostat
from config import BASELINE, DRIFT_TEST, LAB_API, OPTICS, STATE
from optics_simulation import ggs, lab, metrics

SETTINGS = {**DRIFT_TEST, **LAB_API}   # same lab, camera, GGS, box and sample settings as drift_test


class LabAPI:
    """The simulated lab behind one interface; every call adds one line to the log."""

    def __init__(self, seed: int = 0, state=None, run_name: str = "run", box_settings: dict | None = None,
                 lab_settings: dict | None = None):
        """Build the lab and the box; seed shifts every seed in SETTINGS (seed=0 gives the drift_test seeds).

        state: optional LabState; then a run named run_name is started and every action is recorded.
        box_settings: optional THERMOSTAT overrides for this lab only (e.g. other PID gains).
        lab_settings: optional OPTICS overrides for this lab only (e.g. time_drift_tau_min).
        """
        s = self.s = SETTINGS
        self.fake = lab.FakeLab(
            N=s["N"], M=s["M"], seed=s["seed"] + seed,
            noise_level=s["noise_level"], bit_depth=s["bit_depth"], full_scale=s["full_scale"],
            phase_error=s["phase_error"], flicker=s["flicker"],
        )
        self.X0 = self.fake.X_true.copy()                 # the sample at T_ref_C
        optics = {**OPTICS, **(lab_settings or {})}
        tau_min = optics["time_drift_tau_min"]
        self.sample = lab.TemperatureSample(self.X0, OPTICS["T_ref_C"], s["T_c"], OPTICS["T_step_C"], s["y_seed"] + seed,
                                            None if tau_min is None else 60.0 * tau_min, optics["time_drift_step_s"])
        self.E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"] + seed))
        self.ggs_rng = np.random.default_rng(s["ggs_seed"] + seed)

        # The box at T0, already settled by the same warm-up as drift_test, control on.
        self.box = thermostat.Thermostat({**thermostat.SETTINGS, **(box_settings or {})}, s["box_seed"] + seed,
                                         s["T0"], s["T0"])
        self.clock = 0.0
        self.events = []                                  # (time_s, function) world events, see scenario.py
        self.state = None                                 # nothing is recorded during the warm-up
        self._advance(s["warmup_s"])
        self.clock = 0.0                                  # the clock starts after the warm-up

        self.actor = "operator"
        self.state = state
        self.run_id = None if state is None else state.start_run(run_name, seed, s)

        self.X_est = None
        self.map_time = None
        self.last_check = None
        self.last_eta = None
        self.log = []

    def _advance(self, seconds: float) -> None:
        """Step the box forward 1 s at a time; after each second the sample follows the true temperature
        and every world event whose time has come runs once."""
        steps_per_s = int(round(1.0 / thermostat.SETTINGS["dt"]))
        for _ in range(int(round(seconds))):
            for _ in range(steps_per_s):
                self.box.step()
            self.clock += 1.0
            if self.state is not None and int(self.clock) % STATE["temperature_every_s"] == 0:
                self.state.add_temperature(self.run_id, self.clock, float(self.box.x), float(self.box.z),
                                           float(self.box.target), int(self.box.duty))
            self.fake.X_true = self.sample.at(self.box.T, self.clock)
            for event in [e for e in self.events if e[0] <= self.clock]:
                self.events.remove(event)
                event[1]()

    def _log(self, text: str) -> None:
        """Add one line 't=...s text' to the log."""
        self.log.append(f"t={self.clock:.0f}s {text}")

    def _record(self, t_s: float, action: str, args: dict, result) -> None:
        """Write one action (started at lab time t_s) to the lab database, if there is one, and commit."""
        if self.state is not None:
            self.state.add_action(self.run_id, t_s, self.actor, action, args, result)
            self.state.commit()

    def note(self, text: str, level: str = "info") -> None:
        """Write a note ("info" or "warning") to the log and to the lab database."""
        self._log(f"{level.upper()} {text}")
        if self.state is not None:
            self.state.add_note(self.run_id, self.clock, self.actor, level, text)
            self.state.commit()

    def wait(self, seconds: float) -> None:
        """Let lab time pass (whole seconds)."""
        t_start = self.clock
        self._advance(seconds)
        self._log(f"wait {seconds:g} s")
        self._record(t_start, "wait", {"seconds": seconds}, None)

    def get_temperature(self) -> dict:
        """Last thermometer values in C: Kalman estimate, raw reading and the target (never the true temperature)."""
        reading = {"kalman": float(self.box.x), "raw": float(self.box.z), "target": float(self.box.target)}
        self._log(f"get_temperature kalman {reading['kalman']:.3f} raw {reading['raw']:.3f}")
        return reading

    def set_target(self, T: float) -> None:
        """New box target in C."""
        self.box.target = T
        self._log(f"set_target {T:g}")
        self._record(self.clock, "set_target", {"T": T}, None)

    def set_control(self, on: bool) -> None:
        """Switch the PID on or off (off: duty 0, the box drifts toward room temperature)."""
        self.box.control = on
        self._log(f"set_control {'on' if on else 'off'}")
        self._record(self.clock, "set_control", {"on": on}, None)

    def measure_map(self) -> dict:
        """Take L photos of the current sample and recover the map with GGS; takes measure_map_s of lab time.

        Returns: {"n_iter": GGS iterations, "lab_time_s": lab time, "compute_time_s": GGS run time}.
        """
        s = self.s
        t_start = self.clock
        E_in = self.fake.patterns(s["L_ratio"] * s["N"])          # (L x N)
        A = lab.amplitude(self.fake.measure(E_in))               # (L x M)
        t0 = time.perf_counter()
        self.X_est, info = ggs.recover(E_in, A, self.ggs_rng, max_iter=s["max_iter"], tol=s["tol"],
                                       n_restarts=s["n_restarts"], restart_threshold=s["restart_threshold"])
        compute = time.perf_counter() - t0
        self.map_time = self.clock
        self._advance(s["measure_map_s"])
        self._log(f"measure_map {info['n_iter']} iterations")
        result = {"n_iter": int(info["n_iter"]), "lab_time_s": s["measure_map_s"], "compute_time_s": compute}
        if self.state is not None:
            self.state.add_map(self.run_id, t_start, "measure", None, self.clock - t_start)
        self._record(t_start, "measure_map", {}, result)
        return result

    def check_map(self) -> float:
        """Prediction correlation (Pearson) of the stored map on the current sample; takes check_map_s of lab time."""
        t_start = self.clock
        self.last_check = metrics.prediction_correlation(self.E_test, self.X_est, self.fake.measure(self.E_test))
        self._advance(self.s["check_map_s"])
        self._log(f"check_map {self.last_check:.3f}")
        if self.state is not None:
            self.state.add_map(self.run_id, t_start, "check", self.last_check, self.clock - t_start)
        self._record(t_start, "check_map", {}, self.last_check)
        return self.last_check

    def focus(self, point: int) -> float:
        """Focus on camera point `point` with the stored map; returns the enhancement eta. Takes focus_s of lab time."""
        # A focus spot would saturate the fixed exposure; use auto scale for this photo, as in main.py.
        t_start = self.clock
        full_scale, self.fake.full_scale = self.fake.full_scale, None
        self.last_eta = metrics.focus_test(self.fake, self.X_est, targets=[point])["eta_mean"]
        self.fake.full_scale = full_scale
        self._advance(self.s["focus_s"])
        self._log(f"focus {point} eta {self.last_eta:.1f}")
        if self.state is not None:
            self.state.add_focus(self.run_id, t_start, point, self.last_eta)
        self._record(t_start, "focus", {"point": point}, self.last_eta)
        return self.last_eta

    def get_status(self) -> dict:
        """Time, Kalman temperature, target, control, duty, map age (s), last check_map value and last eta."""
        status = {
            "time_s": self.clock,
            "kalman": float(self.box.x),
            "target": float(self.box.target),
            "control": self.box.control,
            "duty": int(self.box.duty),
            "map_age_s": None if self.map_time is None else self.clock - self.map_time,
            "last_check": self.last_check,
            "last_eta": self.last_eta,
        }
        self._log("get_status")
        return status

    def get_log(self) -> list[str]:
        """All log lines so far, like 't=1200s check_map 0.871'."""
        return list(self.log)


if __name__ == "__main__":
    api = LabAPI(seed=0)

    print("measure_map:", api.measure_map())
    r1 = api.check_map()
    print(f"check_map: {r1:.3f} (expect about 0.87)")
    eta = api.focus(SETTINGS["focus_target"])
    print(f"focus({SETTINGS['focus_target']}): eta {eta:.1f} (expect about 55-60 for this point)")

    api.set_control(False)
    api.wait(1800)
    print("temperature after 30 min without control:", api.get_temperature())
    r2 = api.check_map()
    print(f"check_map after 30 min without control: {r2:.3f} (limit < {BASELINE['check_threshold']:g}, "
          f"the re-measure threshold of the rules operator)")

    api.set_control(True)
    api.wait(1800)
    print("temperature after 30 min with control:", api.get_temperature())
    print("measure_map:", api.measure_map())
    r3 = api.check_map()
    print(f"check_map with a new map: {r3:.3f} (expect about 0.87)")

    print("get_status:", api.get_status())
    print("last 10 log lines:")
    for line in api.get_log()[-10:]:
        print(f"  {line}")

    assert r1 > 0.8 and r3 > 0.8, "a fresh map should predict the speckle well"
    # Without control the box only warms toward T_room + P_internal / G (about 20.9 C), about +0.5 C in 30 min,
    # small against T_c = 1.6 C, so the old map keeps about 0.6. It must still count as stale [estimate].
    assert r2 < BASELINE["check_threshold"], "the old map should be stale after the box warmed up"
    print("lab_api.py self-test passed")
