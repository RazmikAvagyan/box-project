"""
thermostat.py - the temperature-controlled box and its thermostat.

One loop step (every dt seconds):
    BME280 sensor reading z -> Kalman filter -> PID -> PWM duty -> cooler -> box temperature

Box model (one temperature for the whole box, a "lumped" model):
    C * dT/dt = G * (T_room - T) + P_internal - P_max * duty / 1023 + dP(t)
dP(t) is a slowly varying extra heat (Ornstein-Uhlenbeck process) in the true box only;
the Kalman filter's model (BoxModel.step) does not know it.

Sources of the numbers (see the comments in config.THERMOSTAT):
- Thermostat document / thesis: dt (20 loops per second), 10-bit cooling-only
  PWM, target 20 C, sensor noise and spike size, PID gains.
- Kalman R from the sensor settings (sensor_std squared); Kalman Q_T and Q_d tuned on the simulated box;
  the spike gate, the lockout guard and the start uncertainty of d are estimates.
- S-2-12 cooler sheet: P_max (250 to 300 W).
- Estimates (the cooled box itself is not described in the sources): C, G,
  T_room, P_internal, the heat disturbance dP, and how often spikes happen.
"""

from __future__ import annotations

import numpy as np

from config import THERMOSTAT as SETTINGS


class BoxModel:
    """The box as one temperature: heat leaks in from the room, the optics heat it, the cooler removes heat."""

    def __init__(self, s: dict):
        """Store the box constants from the settings dict s."""
        self.s = s

    def step(self, T: float, duty: int) -> float:
        """One Euler step of length dt: new box temperature (C) for temperature T and PWM duty (0..1023)."""
        s = self.s
        # Heat flow in W: leak from the warmer room + internal heat - cooler.
        power = s["G"] * (s["T_room"] - T) + s["P_internal"] - s["P_max"] * duty / s["duty_max"]
        # The heat capacity C turns a power into a rate of temperature change.
        return T + power / s["C"] * s["dt"]


class Sensor:
    """BME280 temperature sensor: true temperature + Gaussian noise + rare spikes."""

    def __init__(self, s: dict, rng: np.random.Generator):
        """Store the noise settings from s and the random generator rng."""
        self.s = s
        self.rng = rng

    def read(self, T_true: float) -> float:
        """One reading (C) of the true temperature T_true."""
        s = self.s
        z = T_true + self.rng.normal(0.0, s["sensor_std"])
        # Occasionally a reading jumps by 0.5 to 1.0 C (glitch), up or down.
        if self.rng.random() < s["spike_prob"]:
            z += self.rng.choice([-1.0, 1.0]) * self.rng.uniform(s["spike_min"], s["spike_max"])
        return z


class KalmanFilter:
    """Kalman filter with two unknowns: the box temperature T and an unknown extra heat d (W).

    In simple words: the filter keeps a best guess of T and of d (a warmer room, a weaker
    cooler, more laser heat - anything the box model does not know), and how sure it is of each.
    Predict: the box model says where T goes in one step for the duty the PID sent last step,
    plus the extra heat d (T_pred = BoxModel.step(T, duty) + d / C * dt); d is expected to stay
    the same. Both guesses get a little less sure every step (Q_T for T, Q_d for d).
    Update: only T is measured (noise R). The gain says how far to move toward the reading; if
    the readings keep coming out warmer than predicted, part of that difference goes into d, so
    the model learns the missing heat and the estimate no longer lags behind.
    Spike rejection: a reading more than gate_sigma * sqrt(P_TT + R) away from the prediction
    is skipped (counted in n_skipped). Lockout guard: real spikes are single readings, so after
    max_skips_in_row skipped readings in a row the box itself must have changed; then the filter
    forgets how sure it was (P back to the start values) and takes the reading (counted in n_resets).

    The uncertainty P is the 2 x 2 covariance of (T, d), kept as P_TT, P_Td and P_dd.
    """

    def __init__(self, s: dict, box: BoxModel):
        """Settings s (Q_T, Q_d, R, P0, P0_d, gate_sigma, max_skips_in_row), box: model for the predict step."""
        self.s = s
        self.box = box
        # The model is linear in T: T_next = a * T + (terms without T); b turns heat into temperature.
        self.a = box.step(1.0, 0) - box.step(0.0, 0)
        self.b = s["dt"] / s["C"]
        self.x = None
        self.d = 0.0
        self.n_skipped = 0
        self.n_resets = 0
        self.skips_in_row = 0
        self.last_skipped = False

    def _reset_P(self) -> None:
        """Start uncertainty: P0 for T, P0_d for d, no correlation."""
        self.P_TT, self.P_Td, self.P_dd = self.s["P0"], 0.0, self.s["P0_d"]

    def update(self, z: float, duty_prev: int) -> float:
        """Filter one reading z (C), with duty_prev the duty of the last step; returns the estimate x (C)."""
        s = self.s
        if self.x is None:
            # Start from the first reading.
            self.x = z
            self._reset_P()
            return self.x
        a, b = self.a, self.b
        # Predict with the box model plus the extra heat d, one dt ahead.
        x_pred = self.box.step(self.x, duty_prev) + b * self.d
        P_TT = a * a * self.P_TT + 2 * a * b * self.P_Td + b * b * self.P_dd + s["Q_T"]
        P_Td = a * self.P_Td + b * self.P_dd
        P_dd = self.P_dd + s["Q_d"]
        # Spike rejection: a reading far outside the expected spread is a glitch; keep the prediction.
        self.last_skipped = abs(z - x_pred) > s["gate_sigma"] * np.sqrt(P_TT + s["R"])
        if self.last_skipped:
            self.skips_in_row += 1
            if self.skips_in_row < s["max_skips_in_row"]:
                self.n_skipped += 1
                self.x, self.P_TT, self.P_Td, self.P_dd = x_pred, P_TT, P_Td, P_dd
                return self.x
            # Lockout guard: too many "spikes" in a row means the box changed; start over from this reading.
            self.n_resets += 1
            self.last_skipped = False
            self._reset_P()
            P_TT, P_Td, P_dd = self.P_TT, self.P_Td, self.P_dd
        self.skips_in_row = 0
        # Update: only T is measured; the gains for T and d move both guesses toward the reading.
        K_T, K_d = P_TT / (P_TT + s["R"]), P_Td / (P_TT + s["R"])
        innovation = z - x_pred
        self.x = x_pred + K_T * innovation
        self.d = self.d + K_d * innovation
        self.P_TT, self.P_Td, self.P_dd = (1 - K_T) * P_TT, (1 - K_T) * P_Td, P_dd - K_d * P_Td
        return self.x

    def steady_state_gain(self, n: int = 200000) -> float:
        """Gain K_T after n updates without spikes (the uncertainty no longer changes), for the printout."""
        a, b, s = self.a, self.b, self.s
        P_TT, P_Td, P_dd = s["P0"], 0.0, s["P0_d"]
        for _ in range(n):
            P_TT, P_Td, P_dd = (a * a * P_TT + 2 * a * b * P_Td + b * b * P_dd + s["Q_T"],
                                a * P_Td + b * P_dd, P_dd + s["Q_d"])
            K_T, K_d = P_TT / (P_TT + s["R"]), P_Td / (P_TT + s["R"])
            P_TT, P_Td, P_dd = (1 - K_T) * P_TT, (1 - K_T) * P_Td, P_dd - K_d * P_Td
        return K_T


class PID:
    """PID controller: more cooling when too warm (P), when warm for long (I), when warming fast (D)."""

    def __init__(self, Kp: float, Ki: float, Kd: float, duty_max: int):
        """Gains Kp, Ki, Kd (per loop step, source: thermostat code) and the maximum duty."""
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.duty_max = duty_max
        self.duty_last = None
        self.reset()

    def reset(self) -> None:
        """Forget the error history (used when the target changes)."""
        self.sum_e = 0.0
        self.e_prev = None

    def update(self, e: float) -> int:
        """Duty (0..duty_max) for the error e = T_filtered - T_target (C, positive = too warm)."""
        # Anti-windup: the sum cannot grow beyond what the integral term could ever use.
        limit = self.duty_max / self.Ki
        # Conditional integration: while the cooler is already full on and it is still too warm (or
        # already off and still too cold), more integral cannot help, so do not let it build up.
        saturated = (self.duty_last == self.duty_max and e > 0) or (self.duty_last == 0 and e < 0)
        if not saturated:
            self.sum_e = float(np.clip(self.sum_e + e, -limit, limit))
        de = 0.0 if self.e_prev is None else e - self.e_prev
        self.e_prev = e
        u = self.Kp * e + self.Ki * self.sum_e + self.Kd * de
        # The cooler can only cool: duty is clipped to 0..duty_max.
        self.duty_last = int(np.clip(u, 0, self.duty_max))
        return self.duty_last


class Thermostat:
    """The true box with its sensor, Kalman filter and PID, advanced one control loop step (dt) at a time."""

    def __init__(self, s: dict, seed: int, T_start: float, target: float, control: bool = True,
                 use_kalman: bool = True):
        """Settings s, seed for the sensor noise (and, separately, the heat disturbance dP), start state.

        control=False: duty stays 0. use_kalman=False: the PID gets the raw reading.
        """
        self.s = s
        # The true box gets its own copy of the settings, so a change to the real box (e.g. a
        # warmer room or a failed cooler) does not reach the Kalman filter's model.
        self.box = BoxModel(dict(s))
        self.sensor = Sensor(s, np.random.default_rng(seed))
        self.kalman = KalmanFilter(s, BoxModel(s))
        self.pid = PID(s["Kp"], s["Ki"], s["Kd"], s["duty_max"])
        self.T, self.target, self.control, self.use_kalman = T_start, target, control, use_kalman
        self.duty = 0
        self.z = self.x = None
        # Slow extra heat dP (Ornstein-Uhlenbeck): each step keeps a fraction a of the old value and
        # adds fresh noise, so dP stays at std P_dist_std and forgets itself over P_dist_tau_s.
        # Its own generator, so the sensor readings stay the same.
        self.rng_dist = np.random.default_rng([seed, 1])
        self.a = np.exp(-s["dt"] / s["P_dist_tau_s"])
        self.dP = self.rng_dist.normal(0.0, s["P_dist_std"])

    def step(self) -> None:
        """One loop: read the sensor (z), filter (x), set the duty, then move the true temperature T by dt."""
        s = self.s
        self.z = self.sensor.read(self.T)
        self.x = self.kalman.update(self.z, self.duty)      # duty is still the one sent last step
        measured = self.x if self.use_kalman else self.z
        self.duty = self.pid.update(measured - self.target) if self.control else 0
        self.T = self.box.step(self.T, self.duty) + self.dP / s["C"] * s["dt"]
        self.dP = self.a * self.dP + np.sqrt(1 - self.a * self.a) * self.rng_dist.normal(0.0, s["P_dist_std"])


def simulate(
    duration_s: float,
    control: bool = True,
    T_start: float = SETTINGS["target"],
    target: float = SETTINGS["target"],
    target_steps: list[tuple[float, float]] | None = None,
    settings: dict | None = None,
    seed: int = 0,
    warmup_s: float = 0.0,
    use_kalman: bool = True,
    box_changes: list[tuple[float, str, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate the box and thermostat for duration_s seconds.

    control=False: the cooler stays off (duty 0). target_steps: optional list of
    (time_s, new_target); the PID integral is reset at every change.
    warmup_s: run this long first without recording; then time restarts at 0
    (the PID integral and the Kalman state are kept), like a box already settled.
    use_kalman=False: the PID gets the raw sensor reading instead of the Kalman estimate
    (the filter still runs, so "T_filtered" is still recorded).
    box_changes: optional list of (time_s, setting, value) changes to the true box only
    (e.g. (100, "P_max", 0.0) for a failed cooler); the Kalman model does not see them.
    Returns: dict of arrays (n_steps,): "t_s", "T_true", "T_raw", "T_filtered",
             "duty", "target", "sum_e" (PID error sum, to check anti-windup) and
             "skipped" (1 where the Kalman filter rejected the reading as a spike) and
             "resets" (number of lockout-guard resets so far, counted from the start of the warm-up).
    """
    s = SETTINGS if settings is None else settings
    n = int(round(duration_s / s["dt"]))
    box = Thermostat(s, seed, T_start, target, control, use_kalman)
    steps = sorted(target_steps or [])
    changes = sorted(box_changes or [])

    out = {k: np.empty(n) for k in ["t_s", "T_true", "T_raw", "T_filtered", "duty", "target", "sum_e", "skipped",
                                    "resets"]}
    n_warm = int(round(warmup_s / s["dt"]))
    for k in range(n_warm + n):
        i = k - n_warm                    # recorded index; negative during the warm-up
        t = i * s["dt"]
        while steps and t >= steps[0][0]:
            box.target = steps.pop(0)[1]
            box.pid.reset()
        while changes and t >= changes[0][0]:
            _, key, value = changes.pop(0)
            box.box.s[key] = value
        T = box.T
        box.step()
        if i >= 0:
            out["t_s"][i], out["T_true"][i], out["T_raw"][i], out["T_filtered"][i] = t, T, box.z, box.x
            out["duty"][i], out["target"][i], out["sum_e"][i] = box.duty, box.target, box.pid.sum_e
            out["skipped"][i] = box.kalman.last_skipped
            out["resets"][i] = box.kalman.n_resets
    return out


if __name__ == "__main__":
    from pathlib import Path

    from analysis import plots

    out = Path("results/thermostat")
    s = SETTINGS

    # 1. Hold at 20 C, over the whole recorded time.
    # After a 300 s warm-up the PID integral has already built up, as in a box that has been running.
    hold = simulate(300, control=True, T_start=20.0, target=20.0, seed=0, warmup_s=300)
    T_hold = hold["T_true"]
    dev = np.abs(T_hold - 20).max()
    print(f"hold: min {T_hold.min():.4f}, max {T_hold.max():.4f}, std {T_hold.std():.4f} C, "
          f"max |T - 20| {dev:.4f} C (limit 0.08), mean duty {hold['duty'].mean():.0f}")
    lim = s["limit_C"]
    print(f"hold: share of time inside ±{lim:g} °C: {100 * np.mean(np.abs(T_hold - 20) <= lim):.0f}%")
    print(f"Kalman: skipped {int(hold['skipped'].sum())} of {len(hold['skipped'])} readings as spikes")
    assert dev < 0.08, "hold: deviation too large"
    assert T_hold.std() < 0.02, "hold: temperature too unsteady"

    # 3. Kalman filter during the hold.
    ratio = np.std(hold["T_raw"] - hold["T_true"]) / np.std(hold["T_filtered"] - hold["T_true"])
    print(f"Kalman noise reduction: std(raw - true) / std(filtered - true) = {ratio:.2f} (limit > 2.5)")
    assert ratio > 2.5, "Kalman filter reduces noise too little"
    K_ss = KalmanFilter(s, BoxModel(s)).steady_state_gain()
    print(f"Kalman steady-state gain for T: K_ss = {K_ss:.4f} (about {1 / K_ss:.0f} readings averaged)")
    r_kalman = np.corrcoef(hold["T_filtered"], hold["T_true"])[0, 1]
    print(f"Kalman vs true (hold): correlation {r_kalman:.3f}, "
          f"error std {np.std(hold['T_filtered'] - hold['T_true']):.4f} C")

    # Choice of Q_T: the same hold for several Q_T, each over seeds 0..4. Best = smallest mean hold std;
    # among Q_T within 0.0002 C of that, the one with the smallest mean max |T - 20|.
    sweep = {}
    for Q in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        runs = [simulate(300, control=True, T_start=20.0, target=20.0, seed=seed, warmup_s=300,
                         settings={**s, "Q_T": Q})["T_true"] for seed in range(5)]
        sweep[Q] = (np.mean([T.std() for T in runs]), np.mean([np.abs(T - 20).max() for T in runs]))
        print(f"Q_T sweep (5 seeds): Q_T {Q:.0e}: mean hold std {sweep[Q][0]:.4f} C, "
              f"mean max |T - 20| {sweep[Q][1]:.4f} C")
    best_std = min(v[0] for v in sweep.values())
    Q_best = min((Q for Q, v in sweep.items() if v[0] <= best_std + 0.0002), key=lambda Q: sweep[Q][1])
    print(f"Q_T sweep: best Q_T {Q_best:.0e} (SETTINGS uses {s['Q_T']:.0e})")

    # Choice of Q_d: the normal hold and a hold where the room warms by 3 C at once at 150 s (a stress
    # test, no +-limit_C requirement), seeds 0..4.
    # Best = smallest mean max |T - 20| over both tests.
    warm = [(150.0, "T_room", s["T_room"] + 3.0)]
    sweep_d = {}
    for Q_d in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
        max_devs = [np.abs(simulate(300, control=True, T_start=20.0, target=20.0, seed=seed, warmup_s=300,
                                    settings={**s, "Q_d": Q_d}, box_changes=changes)["T_true"] - 20).max()
                    for changes in ([], warm) for seed in range(5)]
        sweep_d[Q_d] = (np.mean(max_devs[:5]), np.mean(max_devs[5:]), np.mean(max_devs))
        print(f"Q_d sweep (5 seeds): Q_d {Q_d:.0e}: mean max |T - 20| hold {sweep_d[Q_d][0]:.4f} C, "
              f"warm room {sweep_d[Q_d][1]:.4f} C, both {sweep_d[Q_d][2]:.4f} C")
    Q_d_best = min(sweep_d, key=lambda Q_d: sweep_d[Q_d][2])
    print(f"Q_d sweep: best Q_d {Q_d_best:.0e} (SETTINGS uses {s['Q_d']:.0e})")

    # 2. Step from 21 C to 20 C.
    step = simulate(200, control=True, T_start=21.0, target=20.0, seed=1)
    outside = np.flatnonzero(np.abs(step["T_true"] - 20) >= 0.05)
    settling = step["t_s"][outside[-1]] + s["dt"] if outside.size else 0.0
    undershoot = 20 - step["T_true"].min()
    print(f"step 21 -> 20 C: settling time {settling:.1f} s (limit 45), undershoot {undershoot:.4f} C (limit 0.05)")
    # Limit 45 s [estimate; with conditional integration the step settles without undershoot].
    assert settling < 45, "step: settles too slowly"
    assert undershoot < 0.05, "step: undershoot too large"
    w = step["t_s"] <= 60.0
    print(f"Kalman vs true (cooldown 0-60 s): correlation {np.corrcoef(step['T_filtered'][w], step['T_true'][w])[0, 1]:.4f}, "
          f"error std {np.std(step['T_filtered'][w] - step['T_true'][w]):.4f} C")

    # 4. Cooler off.
    off = simulate(300, control=False, T_start=20.0, seed=2)
    rise = off["T_true"][-1] - 20
    print(f"cooler off for 300 s: rise {rise:.2f} C (limit 1.5..2.5)")
    assert 1.5 < rise < 2.5, "cooler off: rise outside the expected range"

    # 5. Anti-windup: an unreachable target must not let the error sum run away.
    windup = simulate(120, control=True, T_start=20.0, target=5.0, seed=3)
    limit = s["duty_max"] / s["Ki"]
    print(f"anti-windup (target 5 C): max |sum_e| {np.abs(windup['sum_e']).max():.1f} (limit {limit:.1f})")
    assert np.abs(windup["sum_e"]).max() <= limit, "anti-windup: error sum exceeded its limit"

    # 6. The same hold without the Kalman filter: the PID sees the raw, noisy readings.
    hold_raw = simulate(300, control=True, T_start=20.0, target=20.0, seed=0, warmup_s=300, use_kalman=False)
    dev_raw = np.abs(hold_raw["T_true"] - 20)
    print(f"hold without Kalman: std {hold_raw['T_true'].std():.4f} °C, "
          f"max |T - 20| {dev_raw.max():.4f} °C, share inside ±{lim:g} °C {100 * np.mean(dev_raw <= lim):.0f}%")

    # 7. The cooler fails (P_max = 0) for 60 s at 100 s of a 400 s hold; the Kalman model does not know.
    fail = simulate(400, control=True, T_start=20.0, target=20.0, seed=0, warmup_s=300,
                    box_changes=[(100.0, "P_max", 0.0), (160.0, "P_max", s["P_max"])])
    after = fail["t_s"] >= 160.0
    outside = fail["t_s"][after][np.abs(fail["T_true"][after] - 20) >= lim]
    back = outside[-1] + s["dt"] - 160.0 if outside.size else 0.0
    print(f"cooler fails 100 to 160 s: max true temperature {fail['T_true'].max():.3f} C, "
          f"Kalman resets {int(fail['resets'][-1] - fail['resets'][0])}, "
          f"back within ±{lim:g} C {back:.1f} s after the cooler returns (limit 120)")
    assert back < 120, "cooler failure: the box does not come back in time"

    paths = [
        # Symlog axis: the 1 °C start and the few-millidegree wobble afterwards are both visible.
        plots.plot_box_deviation(step["t_s"], step["T_true"], step["target"], out / "01_pid_cooldown_21_to_20C.png",
                                 f"PID cooldown from 21 °C to 20 °C: settles in {settling:.1f} s"),
        plots.plot_kalman_error(hold["t_s"], hold["T_true"], hold["T_raw"], hold["T_filtered"],
                                out / "02_kalman_error_vs_raw_error.png"),
        plots.plot_deviation_comparison(
            hold["t_s"], {"PID + Kalman": (hold["T_true"], hold["target"]),
                          "PID on raw sensor": (hold_raw["T_true"], hold_raw["target"])},
            out / "03_pid_hold_20C_with_vs_without_kalman.png", "PID holding 20 °C: with and without the Kalman filter",
            colors={"PID + Kalman": "blue", "PID on raw sensor": "red"}),
    ]
    for p in paths:
        print(p)
    print("thermostat.py self-test passed")
