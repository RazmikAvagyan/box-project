"""
watchdog.py - plain-code alarms for the Map agent, read only from the lab database (no lab time).

Watchdog(state, run_id).check(now) looks at the thermometer, map and focus rows and returns the
alarms that fire now; each alarm fires at most once per alarm_cooldown_s. `active` holds every
alarm condition that is true right now (also those in cooldown). The agent never sees the true
temperature, so a cooler failure shows up only as "cooler at full power" and "box off target".
"""

from __future__ import annotations

import numpy as np

from agents.core.state import LabState
from config import STATE, TARGET_C, THERMOSTAT
from config import WATCHDOG as SETTINGS


def kalman_at(state: LabState, run_id: int, t_s: float) -> float | None:
    """The Kalman reading of the thermometer row closest to lab time t_s (None if there are no rows)."""
    rows = state.query("SELECT kalman FROM temperatures WHERE run_id = ? ORDER BY ABS(t_s - ?), t_s LIMIT 1",
                       (run_id, t_s))
    return rows[0]["kalman"] if rows else None


def last_map_time(state: LabState, run_id: int, before_s: float | None = None) -> float | None:
    """Start time of the last measure_map (at or before before_s, if given); None if there is none."""
    rows = state.query("SELECT MAX(t_s) AS t FROM maps WHERE run_id = ? AND kind = 'measure' AND t_s <= ?",
                       (run_id, float("inf") if before_s is None else before_s))
    return rows[0]["t"]


class Watchdog:
    """Alarms from the database rows of one run."""

    def __init__(self, state: LabState, run_id: int):
        self.state, self.run_id = state, run_id
        self.last_fired = {}         # alarm -> lab time it last fired
        self.active = []

    def _window(self, now: float, seconds: float) -> list[dict]:
        """Thermometer rows of the last `seconds` up to now, or [] if they do not yet cover that whole time."""
        rows = [r for r in self.state.since("temperatures", self.run_id, now - seconds) if r["t_s"] <= now]
        return rows if rows and rows[0]["t_s"] <= now - seconds + 1 else []

    def target_changed_at(self, now: float) -> float:
        """Lab time of the last target change up to now (-inf if the target never changed)."""
        last = self.state.query("SELECT target FROM temperatures WHERE run_id = ? AND t_s <= ? ORDER BY t_s DESC LIMIT 1",
                                (self.run_id, now))
        if not last:
            return -np.inf
        rows = self.state.query("SELECT MAX(t_s) AS t FROM temperatures WHERE run_id = ? AND t_s <= ? AND target != ?",
                                (self.run_id, now, last[0]["target"]))
        return -np.inf if rows[0]["t"] is None else rows[0]["t"] + STATE["temperature_every_s"]

    def conditions(self, now: float) -> list[str]:
        """Every alarm condition that is true at lab time now."""
        s = SETTINGS
        alarms = []
        rows = self._window(now, s["off_target_s"])
        if rows and all(abs(r["kalman"] - r["target"]) > THERMOSTAT["swing_band_C"] for r in rows):
            alarms.append("box off target")
        t_map = last_map_time(self.state, self.run_id, now)
        rows = self._window(now, s["moved_avg_s"])
        # The mean over about one swing period, so the normal PID swing peaks do not count as a move.
        if t_map is not None and rows and now - self.target_changed_at(now) >= s["target_change_quiet_s"]:
            k_mean = np.mean([r["kalman"] for r in rows])
            if abs(k_mean - kalman_at(self.state, self.run_id, t_map)) > s["moved_C"]:
                alarms.append("temperature moved since map")
        rows = self._window(now, s["full_power_s"])
        # Full power while the box holds its target is normal in a warm room; alarm only if it still warms above the band.
        if rows and all(r["duty"] >= s["duty_full_share"] * THERMOSTAT["duty_max"] for r in rows):
            t, k = np.array([r["t_s"] for r in rows]), np.array([r["kalman"] for r in rows])
            if k[-1] > rows[-1]["target"] + THERMOSTAT["swing_band_C"] and np.polyfit(t, k, 1)[0] > 0:
                alarms.append("cooler at full power")
        rows = self._window(now, s["stuck_s"])
        if rows and len({r["raw"] for r in rows}) == 1:
            alarms.append("sensor stuck")
        if t_map is not None:
            focus = [r for r in self.state.since("focus", self.run_id, t_map) if r["t_s"] <= now]
            ref = [r["eta"] for r in focus if r["t_s"] < t_map + s["eta_ref_s"]]
            if ref and focus and focus[-1]["t_s"] >= t_map + s["eta_ref_s"] and focus[-1]["eta"] < s["eta_drop_share"] * np.mean(ref):
                alarms.append("focus dropped")
        return alarms

    def check(self, now: float) -> list[str]:
        """The alarms that fire at lab time now: true now and not fired in the last alarm_cooldown_s."""
        self.active = self.conditions(now)
        fired = [a for a in self.active if now - self.last_fired.get(a, -np.inf) >= SETTINGS["alarm_cooldown_s"]]
        for alarm in fired:
            self.last_fired[alarm] = now
        return fired


if __name__ == "__main__":
    s = SETTINGS
    target, full, band = TARGET_C, THERMOSTAT["duty_max"], THERMOSTAT["swing_band_C"]

    def make_run(kalman, raw, duty, eta, aim=lambda t: target, map_times=(0.0,), seconds=1200):
        """A database with one made-up run: functions of t for kalman, raw, duty and the target (aim);
        eta(t) for a focus photo every 60 s; measure_map rows at map_times. Returns (state, run_id)."""
        state = LabState(":memory:")
        run_id = state.start_run("watchdog_test", 0, {})
        for t in range(1, seconds + 1):
            state.add_temperature(run_id, t, kalman(t), raw(t), aim(t), duty(t))
            if t % 60 == 0:
                state.add_focus(run_id, t, 7, eta(t))
        for t in map_times:
            state.add_map(run_id, t, "measure", None, 60)
        return state, run_id

    rng = np.random.default_rng(0)
    noise = rng.normal(0, THERMOSTAT["sensor_std"], 2000)

    def swing(t, amplitude=0.3):
        """A normal hold: PID swing with a 5 min period."""
        return target + amplitude * np.sin(2 * np.pi * t / 300)

    def warming(t):
        """Cooler gone at t = 700: the box warms 0.005 C/s from target + 0.3 C."""
        return target + (0.3 + 0.005 * (t - 700) if t > 700 else 0.0)

    normal = dict(kalman=swing, raw=lambda t: round(swing(t) + noise[t], 2), duty=lambda t: 500, eta=lambda t: 60.0)
    cases = {
        "normal": (normal, []),
        "swing peaks, map at a trough": ({**normal, "kalman": lambda t: swing(t, 0.45)}, [], (225.0,)),
        "box off target": ({**normal, "kalman": lambda t: target + (0.6 if t > 500 else 0.0)}, ["box off target"]),
        "temperature moved since map": ({**normal, "kalman": lambda t: target + (0.9 if t > 500 else 0.0)},
                                        ["box off target", "temperature moved since map"]),
        "target change (quiet, then moved)": ({**normal, "kalman": lambda t: target + (0.9 if t > 300 else 0.0),
                                               "aim": lambda t: target + (0.9 if t > 300 else 0.0)},
                                              ["temperature moved since map"]),
        "full power, box holds (warm room)": ({**normal, "duty": lambda t: full if t > 600 else 500}, []),
        "cooler at full power": ({**normal, "kalman": warming, "duty": lambda t: full if t > 700 else 500},
                                 ["cooler at full power", "box off target", "temperature moved since map"]),
        "sensor stuck": ({**normal, "raw": lambda t: 20.31 if t > 700 else round(swing(t) + noise[t], 2)}, ["sensor stuck"]),
        "focus dropped": ({**normal, "eta": lambda t: 60.0 if t < 600 else 30.0}, ["focus dropped"]),
    }
    for name, (rows, expect, *maps) in cases.items():
        state, run_id = make_run(**rows, **({"map_times": maps[0]} if maps else {}))
        dog = Watchdog(state, run_id)
        fired = [(now, a) for now in range(60, 1200, 60) for a in dog.check(now)]   # < one cooldown after 600 s
        print(f"{name:34} fired {fired}")
        assert [a for _, a in fired] == expect, fired
        state.close()

    # Cooldown: a condition that stays true fires again only after alarm_cooldown_s.
    state, run_id = make_run(**{**normal, "raw": lambda t: 20.31}, seconds=1800)
    dog = Watchdog(state, run_id)
    fired = [now for now in range(60, 1801, 60) if dog.check(now)]
    print(f"{'cooldown (sensor stuck 30 min)':34} fired at {fired}, active {dog.active}")
    assert fired == [60, 60 + s["alarm_cooldown_s"], 60 + 2 * s["alarm_cooldown_s"]]
    print("watchdog: each alarm once, none on normal rows, cooldown kept: ok")
