"""
safety.py - a plain-code check of every plan (rules or AI) between plan() and act().

SafetyCheck(state, run_id).review(plan, obs, now) drops the actions that break a rule and says why:
no measure_map while the box is not stable, none sooner than min_map_gap_s after the last one,
at most max_map_time_per_hour_s of measure_map lab time per hour, and no wait while a watchdog
alarm is active (obs["alarms"]). It reads only the lab database and the observations.
"""

from __future__ import annotations

from agents.state import LabState
from agents.watchdog import last_map_time
from config import LAB_API
from config import SAFETY as SETTINGS


class SafetyCheck:
    """Rules every plan must pass."""

    def __init__(self, state: LabState, run_id: int):
        self.state, self.run_id = state, run_id

    def map_time_last_hour(self, now: float) -> float:
        """Lab time (s) spent on measure_map that started in the last hour."""
        rows = self.state.query("SELECT SUM(duration_s) AS s FROM maps WHERE run_id = ? AND kind = 'measure' "
                                "AND t_s > ? AND t_s <= ?", (self.run_id, now - 3600, now))
        return rows[0]["s"] or 0.0

    def why_not(self, name: str, obs: dict, now: float, planned_map_s: float) -> str | None:
        """The reason an action is refused, or None if it may run. planned_map_s: map time already allowed in this plan."""
        s = SETTINGS
        if name == "measure_map":
            if not obs["stable"]:
                return "the box is not stable"
            t_map = last_map_time(self.state, self.run_id, now)
            if t_map is not None and now - t_map < s["min_map_gap_s"]:
                return f"the last map is only {now - t_map:.0f} s old (< {s['min_map_gap_s']:g} s)"
            used = self.map_time_last_hour(now) + planned_map_s
            if used + LAB_API["measure_map_s"] > s["max_map_time_per_hour_s"]:
                return f"{used:.0f} s of map time in the last hour (max {s['max_map_time_per_hour_s']:g} s)"
        if name == "wait" and obs.get("alarms"):
            return f"watchdog alarm active ({', '.join(obs['alarms'])})"
        return None

    def review(self, plan: list[tuple[str, dict]], obs: dict, now: float) -> tuple[list[tuple[str, dict]], list[str]]:
        """(allowed plan, refusals "refused <action> because <why>") for a plan at lab time now."""
        allowed, refusals, planned_map_s = [], [], 0.0
        for name, args in plan:
            why = self.why_not(name, obs, now, planned_map_s)
            if why is None:
                allowed.append((name, args))
                planned_map_s += LAB_API["measure_map_s"] if name == "measure_map" else 0.0
            else:
                refusals.append(f"refused {name}{f'({args})' if args else ''} because {why}")
        return allowed, refusals


if __name__ == "__main__":
    def make_state(map_times: list[float]) -> tuple[LabState, int]:
        """A database with measure_map rows at the given lab times (60 s each)."""
        state = LabState(":memory:")
        run_id = state.start_run("safety_test", 0, {})
        for t in map_times:
            state.add_map(run_id, t, "measure", None, 60.0)
        return state, run_id

    stable, moving = {"stable": True, "alarms": []}, {"stable": False, "alarms": []}
    alarm = {"stable": True, "alarms": ["cooler at full power"]}
    hour_full = [3600 - 300 * k for k in range(1, 11)]          # 10 maps = 600 s in the last hour, last 300 s ago
    cases = [
        ("measure_map, stable box", [0.0], stable, [("measure_map", {})], True),
        ("measure_map, box not stable", [0.0], moving, [("measure_map", {})], False),
        ("measure_map, last map 100 s ago", [3500.0], stable, [("measure_map", {})], False),
        ("measure_map, hour budget used", hour_full, stable, [("measure_map", {})], False),
        ("wait, watchdog alarm active", [0.0], alarm, [("wait", {"seconds": 60})], False),
        ("wait, no alarm", [0.0], stable, [("wait", {"seconds": 60})], True),
    ]
    for label, maps, obs, plan, ok in cases:
        state, run_id = make_state(maps)
        allowed, refusals = SafetyCheck(state, run_id).review(plan, obs, 3600.0)
        print(f"{label:32} allowed {allowed}  {'; '.join(refusals)}")
        assert (allowed == plan and not refusals) if ok else (allowed == [] and len(refusals) == 1)
        state.close()
    print("safety: each refusal case refuses, normal actions pass: ok")
