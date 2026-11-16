"""
loop.py - the agent loop Monitor -> Diagnose -> Plan -> Act -> Verify, with a first Map agent.

The Map agent keeps the map useful: it checks the map, and measures a new one only when the old
one is stale AND the box is stable (a map taken while the temperature still moves is soon stale
again). While the box moves it skips the check and looks again sooner; a target change wakes it. The brain is plan(): simple rules for now; an AI brain will replace it later. The agent
only uses LabAPI and the lab database (state.py), never the true temperature, and writes one
short note per phase that did something, so the database shows why it acted.
"""

from __future__ import annotations

import numpy as np

from agents import scenario
from agents.lab_api import LabAPI
from agents.state import LabState
from config import AGENT_LOOP as SETTINGS
from config import BASELINE, DRIFT_TEST, SCENARIO, THERMOSTAT

ALLOWED = ("measure_map", "check_map", "focus", "set_target", "set_control", "wait")


class Agent:
    """One agent: monitor, diagnose, plan, act and verify through the LabAPI and the lab database."""

    def __init__(self, api: LabAPI, state: LabState, name: str):
        """api: the lab; state: the lab database the api writes to; name: written as the actor."""
        self.api, self.state, self.name = api, state, name
        api.actor = name
        self.warnings = 0

    def _warn(self, text: str) -> None:
        """Write a warning note and count it."""
        self.api.note(text, level="warning")
        self.warnings += 1

    def monitor(self) -> dict:
        """Observations: status, the last stable_s of thermometer rows, and a map check (None while the box moves)."""
        status = self.api.get_status()
        temps = self.state.since("temperatures", self.api.run_id, self.api.clock - SETTINGS["stable_s"])
        stable = all(abs(row["kalman"] - row["target"]) < THERMOSTAT["limit_C"] for row in temps)
        # A check on a moving box says little and costs lab time; check only a stable box.
        check = self.api.check_map() if stable else None
        return {"status": status, "temperatures": temps, "stable": stable, "check": check}

    def diagnose(self, obs: dict) -> list[str]:
        """Problems found in the observations: "box not stable", or else "map stale"."""
        if not obs["stable"]:
            return ["box not stable"]
        return ["map stale"] if obs["check"] < BASELINE["check_threshold"] else []

    def plan(self, problems: list[str]) -> list[tuple[str, dict]]:
        """The brain: actions (function name, args) for the problems. Simple rules for now."""
        if "map stale" in problems and "box not stable" not in problems:
            return [("measure_map", {})]
        return []

    def act(self, plan: list[tuple[str, dict]]) -> list:
        """Run each planned action through the api; names outside ALLOWED are refused with a warning note."""
        results = []
        for name, args in plan:
            if name not in ALLOWED:
                self._warn(f"refused action {name}({args}): not allowed")
                continue
            results.append(getattr(self.api, name)(**args))
        return results

    def verify(self, plan: list[tuple[str, dict]]) -> float | None:
        """After a measure_map, check the new map; warn if it is still stale. Returns the check or None."""
        if not any(name == "measure_map" for name, _ in plan):
            return None
        check = self.api.check_map()
        if check < BASELINE["check_threshold"]:
            self._warn(f"verify: new map still stale (check {check:.3f} < {BASELINE['check_threshold']:g})")
        return check

    def run_once(self) -> dict:
        """One loop through the five phases; returns what happened in each."""
        obs = self.monitor()
        problems = self.diagnose(obs)
        stable = "box not stable" not in problems
        checked = "check skipped" if obs["check"] is None else f"check {obs['check']:.3f}"
        self.api.note(f"monitor: {checked}, box {'stable' if stable else 'not stable'}")
        if problems:
            self.api.note(f"diagnose: {', '.join(problems)}")
        plan = self.plan(problems)
        if plan:
            self.api.note(f"plan: {', '.join(name for name, _ in plan)}")
        elif "box not stable" in problems:
            self.api.note("plan: box not stable, look again soon")
        results = self.act(plan)
        if results:
            self.api.note(f"act: done {', '.join(name for name, _ in plan)}")
        check = self.verify(plan)
        if check is not None:
            self.api.note(f"verify: new map check {check:.3f}")
        return {"obs": obs, "problems": problems, "plan": plan, "results": results, "verify": check}


def run(seed: int = 0, state: LabState | None = None) -> dict:
    """The 2-hour scenario with a focus photo every step_s, and agent.run_once() every check_every_s
    (counted from the start of the last loop), after retry_after_s while the box is not stable, and at once
    when the target changes.

    state: the lab database (an in-memory one if None); the run is saved as "map_agent_rules".
    Returns the same keys as baseline.run ("checked" counts the check_map calls in each step).
    """
    s = BASELINE
    state = LabState(":memory:") if state is None else state
    state.delete_runs("map_agent_rules")
    api = LabAPI(seed=seed, state=state, run_name="map_agent_rules")
    agent = Agent(api, state, "map_agent")
    scenario.add_events(api)
    api.measure_map()
    rec = {k: [] for k in ["t_s", "eta", "kalman", "T_true", "target", "measured", "checked"]}
    lab_time, last_loop, next_after = 0.0, api.clock, s["check_every_s"]
    last_target = api.get_status()["target"]

    for k in range(1, int(SCENARIO["duration_s"] // s["step_s"]) + 1):
        # Each step starts on the step_s grid (later only if the last step's work took longer).
        if api.clock < k * s["step_s"]:
            api.wait(k * s["step_s"] - api.clock)
        t_step = api.clock
        eta = api.focus(DRIFT_TEST["focus_target"])
        reading = api.get_temperature()
        measured, checked = False, 0
        target = api.get_status()["target"]
        if api.clock - last_loop >= next_after or target != last_target:
            t0 = api.clock
            done = agent.run_once()
            measured = any(name == "measure_map" for name, _ in done["plan"])
            checked = (done["obs"]["check"] is not None) + (done["verify"] is not None)
            lab_time += api.clock - t0
            last_loop, last_target = t0, target
            retry = "box not stable" in done["problems"]
            next_after = SETTINGS["retry_after_s"] if retry else s["check_every_s"]
        for key, value in [("t_s", t_step), ("eta", eta), ("kalman", reading["kalman"]), ("T_true", api.box.T),
                           ("target", reading["target"]), ("measured", measured), ("checked", checked)]:
            rec[key].append(value)

    out = {key: np.array(values) for key, values in rec.items()}
    out.update({"lab_time_s": lab_time, "warnings": agent.warnings, "api": api})
    return out


if __name__ == "__main__":
    from pathlib import Path

    from agents import baseline
    from analysis import plots
    from config import STATE

    state = LabState(STATE["db_path"])
    runs = {"never": baseline.run("never"), "rules": baseline.run("rules"), "map agent": run(state=state)}
    state.close()
    summary = {name: baseline.summarize(r) for name, r in runs.items()}

    rows = [("mean eta", "mean_eta", ".1f"), ("min eta", "min_eta", ".1f"),
            ("share eta > 30", "share_eta_above_30", ".2f"), ("measure_map calls", "measure_map_calls", "d"),
            ("check_map calls", "check_map_calls", "d"), ("lab time on maps (s)", "lab_time_on_maps_s", ".0f"),
            ("warnings", "warnings", "d"), ("new maps while box off target", "maps_while_box_off_target", "d")]
    print(f"{'':30} {'never':>8} {'rules':>8} {'map agent':>10}")
    for label, key, fmt in rows:
        print(f"{label:30} " + " ".join(f"{summary[n][key]:{fmt}}".rjust(w)
                                        for n, w in [("never", 8), ("rules", 8), ("map agent", 10)]))
    after = runs["rules"]["t_s"] >= SCENARIO["target_change_at_s"]
    fail = (runs["rules"]["t_s"] >= SCENARIO["cooler_fail_at_s"]) & (runs["rules"]["t_s"] < SCENARIO["target_change_at_s"])
    for label, mask in [("cooler failure to target change", fail), ("after the target change", after)]:
        print(f"mean eta {label}: " + ", ".join(f"{n} {r['eta'][mask].mean():.1f}" for n, r in runs.items()))
    print("map agent notes (plans and warnings):")
    for line in runs["map agent"]["api"].get_log():
        if any(word in line for word in ("plan:", "WARNING", "verify:", "EVENT")):
            print(f"  {line}")

    out = Path("results/agents")
    print(plots.plot_policy_comparison(
        {name: (r["t_s"], r["eta"], r["measured"]) for name, r in runs.items()}, baseline.scenario_events(),
        out / "01_never_rules_agent_focus_brightness.png",
        colors={"never": "red", "rules": "blue", "map agent": "green"},
        title="Focus brightness during the scenario: never vs. fixed rules vs. Map agent"))
