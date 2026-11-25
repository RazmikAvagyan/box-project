"""
loop.py - the agent loop Monitor -> Diagnose -> Plan -> Act -> Verify, with a first Map agent.

The loop is a LangGraph graph with one node per phase (monitor -> diagnose -> plan -> act -> verify; with no
problems the loop ends after diagnose); the graph state holds the observations, problems, planned actions, act
results and the verify check of one loop. With use_messages the agent reads its unread messages at the start of
each loop: after "box unstable" or "cooler problem" it does no check_map or measure_map until "box stable again".
The simulation clock (LabAPI) and the lab database stay outside the graph.

The Map agent keeps the map useful: it checks the map, and measures a new one only when the old
one is stale AND the box is stable, i.e. inside target +- swing_band_C (a map taken while the
temperature still moves is soon stale again). While the box moves it skips the check and looks again sooner; a target change wakes it. The brain is plan(), which asks a brain from brain.py: the simple rules
or an AI (Claude, Gemini, OpenAI, or a fake one for tests) that falls back to the rules on any error.

Three plain-code helpers, each behind a switch in AGENT_LOOP (all off = the basic Map agent, all on = the full one):
the watchdog (watchdog.py) reads the database every step and wakes the agent on an alarm; the safety
check (safety.py) reviews every plan before act(); the predictor (predict.py) lets monitor() skip
check_map when the predicted check is clearly good. The agent only uses LabAPI and the lab database
(state.py), never the true temperature, and writes one short note per phase that did something, so the
database shows why it woke, what it planned, what safety refused and when a check was skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from agents import brain as brains
from agents import scenario
from agents.brain import ALLOWED
from agents.lab_api import LabAPI
from agents.predict import MapPredictor
from agents.safety import SafetyCheck
from agents.state import LabState, TOPICS
from agents.watchdog import Watchdog
from box_simulation import thermostat
from config import AGENT_LOOP, ANALYST, OPTICS
from config import AGENT_LOOP as SETTINGS
from config import AI_BRAIN
from config import BASELINE, DRIFT_TEST, PREDICTION, SCENARIO, THERMOSTAT

SWITCHES = ("use_watchdog", "use_safety", "use_prediction", "use_fresh_level", "use_messages", "use_analyst")
HOLD_TOPICS = ("box unstable", "cooler problem")          # no check or new map after these ...
RELEASE_TOPIC = "box stable again"                        # ... until this one
assert set(HOLD_TOPICS) | {RELEASE_TOPIC} <= set(TOPICS)
HELPERS = ("use_watchdog", "use_safety", "use_prediction")      # all off = the basic Map agent, all on = the full one


class LoopState(TypedDict, total=False):
    """The graph state of one loop (node names and state keys must differ in LangGraph)."""
    obs: dict                           # monitor: observations
    problems: list[str]                 # diagnose
    actions: list[tuple[str, dict]]     # plan: the reviewed plan
    results: list                       # act
    new_check: float | None             # verify: check of a new map, or None


class Agent:
    """One agent: monitor, diagnose, plan, act and verify through the LabAPI and the lab database."""

    def __init__(self, api: LabAPI, state: LabState, name: str, brain: str = AI_BRAIN["provider"],
                 switches: dict | None = None):
        """api: the lab; state: the lab database the api writes to; name: written as the actor;
        brain: provider name for brain.make ("rules", "fake", "claude", "gemini", "openai");
        switches: {"use_watchdog", "use_safety", "use_prediction", "use_fresh_level": bool}; missing keys come
        from AGENT_LOOP."""
        self.api, self.state, self.name = api, state, name
        self.brain = brains.make(brain)
        on = {**{key: SETTINGS[key] for key in SWITCHES}, **(switches or {})}
        self.use_fresh = on["use_fresh_level"]
        self.use_messages = on["use_messages"]
        self.hold, self.last_read = False, -np.inf    # waiting for "box stable again"; messages read up to this time
        self.fresh_checks = []          # verify check of every new map, oldest first
        self.backoff_s, self.no_map_until = None, -np.inf
        self.watchdog = Watchdog(state, api.run_id) if on["use_watchdog"] else None
        self.safety = SafetyCheck(state, api.run_id) if on["use_safety"] else None
        self.predictor = MapPredictor(state, api.run_id) if on["use_prediction"] else None
        api.actor = name
        self.warnings = 0
        self.graph = self._build_graph()

    def _warn(self, text: str) -> None:
        """Write a warning note and count it."""
        self.api.note(text, level="warning")
        self.warnings += 1

    def fresh_level(self) -> float | None:
        """Median verify check of the last fresh_n new maps (None until there are fresh_n, or if switched off)."""
        n = SETTINGS["fresh_n"]
        return float(np.median(self.fresh_checks[-n:])) if self.use_fresh and len(self.fresh_checks) >= n else None

    def stale_limit(self) -> float:
        """The map is stale below this: stale_share * fresh level, or BASELINE check_threshold without a fresh level."""
        level = self.fresh_level()
        return BASELINE["check_threshold"] if level is None else SETTINGS["stale_share"] * level

    def _check(self) -> float:
        """check_map, and with the predictor: test its prediction for this check and note the error."""
        predicted = None if self.predictor is None else self.predictor.predict(self.api.clock)
        check = self.api.check_map()
        if predicted is not None:
            error = self.predictor.test(check, predicted[0])
            self.api.note(f"predict: predicted {predicted[0]:.3f} for dT {predicted[1]:.3f} C, checked {check:.3f}, "
                          f"error {error:.3f}")
            if not self.predictor.trusted:
                self._warn(f"predict: error above {PREDICTION['max_error']:g} {PREDICTION['max_bad']} times, "
                           "always check from now on")
        return check

    def monitor(self) -> dict:
        """Observations: status, the last stable_s of thermometer rows, the active watchdog alarms, and a map check
        (None while the box moves; the predicted value if the predictor is sure enough, then "predicted" is True)."""
        if self.use_messages:
            for message in self.state.unread(self.api.run_id, self.name, self.last_read):
                self.hold = True if message["topic"] in HOLD_TOPICS else \
                    (False if message["topic"] == RELEASE_TOPIC else self.hold)
                self.api.note(f"read message: {message['topic']} ({message['text']})")
            self.last_read = self.api.clock
        status = self.api.get_status()
        temps = self.state.since("temperatures", self.api.run_id, self.api.clock - SETTINGS["stable_s"])
        # The real PID swings about +-0.3 C, so "stable" means inside the swing band, not the thesis limit.
        stable = all(abs(row["kalman"] - row["target"]) < THERMOSTAT["swing_band_C"] for row in temps)
        alarms = [] if self.watchdog is None else list(self.watchdog.active)
        check, predicted, limit = None, False, self.stale_limit()
        # A check on a moving box says little and costs lab time; check only a stable box (and not while waiting
        # for "box stable again" from the Temperature agent).
        if stable and not self.hold:
            guess = None if self.predictor is None else self.predictor.predict(self.api.clock)
            if guess is not None and guess[0] > limit + PREDICTION["margin"]:
                check, predicted = guess[0], True
                self.api.note(f"predict: check skipped, predicted {guess[0]:.3f} for dT {guess[1]:.3f} C")
            else:
                check = self._check()
        return {"status": status, "temperatures": temps, "stable": stable, "check": check, "predicted": predicted,
                "alarms": alarms, "hold": self.hold, "stale_limit": limit, "fresh_level": self.fresh_level(),
                "backoff_left_s": max(0.0, self.no_map_until - self.api.clock)}

    def diagnose(self, obs: dict) -> list[str]:
        """Problems found in the observations: "box not stable", or else "map stale" (below the stale limit),
        written "map stale (backing off)" while recent new maps did not help (then the rules do not re-measure)."""
        if not obs["stable"]:
            return ["box not stable"]
        if obs["hold"]:
            return ["waiting for box stable again"]
        if obs["check"] >= obs["stale_limit"]:
            return []
        return ["map stale (backing off)"] if obs["backoff_left_s"] > 0 else ["map stale"]

    def review(self, plan: list[tuple[str, dict]], obs: dict, problems: list[str]) -> list[tuple[str, dict]]:
        """The safety check between plan() and act(): drop refused actions (warning "safety: refused ...");
        if an AI plan is refused completely, the rules plan (also reviewed) runs instead."""
        if self.safety is None:
            return plan
        allowed, refusals = self.safety.review(plan, obs, self.api.clock)
        for refusal in refusals:
            self._warn(f"safety: {refusal}")
        if plan and not allowed and self.brain.provider != "rules":
            allowed, refusals = self.safety.review(brains.rules_plan(problems), obs, self.api.clock)
            for refusal in refusals:
                self._warn(f"safety: {refusal} (rules plan)")
            self.api.note(f"safety: AI plan refused, rules plan: {', '.join(n for n, _ in allowed) or 'no action'}")
        return allowed

    def plan(self, obs: dict, problems: list[str]) -> list[tuple[str, dict]]:
        """The brain: actions (function name, args) for the problems, from self.brain.decide; writes the plan note
        (an AI brain's own note with its reason, or a warning "brain fallback: <why>" and the rules note)."""
        decision = self.brain.decide(obs, problems)
        plan = decision["plan"]
        if decision["fallback"]:
            self._warn(f"brain fallback: {decision['fallback']}")
        if decision["note"]:
            self.api.note(decision["note"])
        elif plan:
            self.api.note(f"plan: {', '.join(name for name, _ in plan)}")
        elif "box not stable" in problems:
            self.api.note("plan: box not stable, look again soon")
        return plan

    def act(self, plan: list[tuple[str, dict]]) -> list:
        """Run each planned action through the api; names outside ALLOWED are refused and a failing action is
        skipped, each with a warning note, so no plan can stop the run."""
        results = []
        for name, args in plan:
            if name not in ALLOWED:
                self._warn(f"refused action {name}({args}): not allowed")
                continue
            try:
                results.append(getattr(self.api, name)(**args))
            except Exception as e:
                self._warn(f"act failed: {name}({args}): {type(e).__name__}: {e}")
        return results

    def verify(self, plan: list[tuple[str, dict]], before: float | None) -> float | None:
        """After a measure_map, check the new map; warn if it is still stale. With the fresh level: if the new map is
        not min_gain better than the check before it (`before`), double the wait before the next re-measure (up to
        max_backoff_s); a map that helps resets it. Returns the check or None."""
        if not any(name == "measure_map" for name, _ in plan):
            return None
        limit = self.stale_limit()
        check = self._check()
        self.fresh_checks.append(check)
        if check < limit:
            self._warn(f"verify: new map still stale (check {check:.3f} < {limit:.3g})")
        if self.use_fresh and before is not None:
            if check < before + SETTINGS["min_gain"]:
                self.backoff_s = min(2 * (self.backoff_s or BASELINE["check_every_s"]), SETTINGS["max_backoff_s"])
                self.no_map_until = self.api.clock + self.backoff_s
                self.api.note(f"map stale but a new map would not be better, backing off {self.backoff_s:.0f} s "
                              f"(new {check:.3f} vs before {before:.3f})")
            else:
                self.backoff_s, self.no_map_until = None, -np.inf
        return check

    def _monitor_node(self, state: LoopState) -> LoopState:
        """Graph node: observations."""
        return {"obs": self.monitor()}

    def _diagnose_node(self, state: LoopState) -> LoopState:
        """Graph node: problems, with the monitor and diagnose notes."""
        obs = state["obs"]
        problems = self.diagnose(obs)
        stable = "box not stable" not in problems
        checked = "check skipped" if obs["check"] is None else \
            f"check {'predicted ' if obs['predicted'] else ''}{obs['check']:.3f}"
        self.api.note(f"monitor: {checked}, box {'stable' if stable else 'not stable'}")
        if problems:
            self.api.note(f"diagnose: {', '.join(problems)}")
        return {"problems": problems}

    def _plan_node(self, state: LoopState) -> LoopState:
        """Graph node: the brain's plan after the safety check."""
        obs, problems = state["obs"], state["problems"]
        return {"actions": self.review(self.plan(obs, problems), obs, problems)}

    def _act_node(self, state: LoopState) -> LoopState:
        """Graph node: run the plan."""
        results = self.act(state["actions"])
        if results:
            self.api.note(f"act: done {', '.join(name for name, _ in state['actions'])}")
        return {"results": results}

    def _verify_node(self, state: LoopState) -> LoopState:
        """Graph node: check a new map."""
        check = self.verify(state["actions"], state["obs"]["check"])
        if check is not None:
            self.api.note(f"verify: new map check {check:.3f}")
        return {"new_check": check}

    def _build_graph(self):
        """The compiled LangGraph graph monitor -> diagnose -> plan -> act -> verify."""
        graph = StateGraph(LoopState)
        phases = [("monitor", self._monitor_node), ("diagnose", self._diagnose_node), ("plan", self._plan_node),
                  ("act", self._act_node), ("verify", self._verify_node)]
        for name, node in phases:
            graph.add_node(name, node)
        graph.add_edge(START, "monitor")
        graph.add_edge("monitor", "diagnose")
        # No problems: nothing to plan, act on or verify.
        graph.add_conditional_edges("diagnose", lambda state: "plan" if state["problems"] else END, ["plan", END])
        graph.add_edge("plan", "act")
        graph.add_edge("act", "verify")
        graph.add_edge("verify", END)
        return graph.compile()

    def run_once(self) -> dict:
        """One loop through the five phases (one run of the graph); returns what happened in each."""
        state = self.graph.invoke({})
        return {"obs": state["obs"], "problems": state["problems"], "plan": state.get("actions", []),
                "results": state.get("results", []), "verify": state.get("new_check")}


def first_reaction(log: list[str]) -> float | None:
    """Seconds from the cooler failure to the operator's first warning, "not stable" note or sent message (None if none)."""
    for line in log:
        t = float(line.split("s ", 1)[0][2:])
        if t >= SCENARIO["cooler_fail_at_s"] and ("WARNING" in line or "not stable" in line or "message to" in line):
            return t - SCENARIO["cooler_fail_at_s"]
    return None


class MapTask:
    """The Map agent as a task for the team graph (team.py) and for run(): its per-step work, its watchdog, when its
    loop is due, one loop, and its per-step records. Another task agent can replace it in the team."""

    name = "map_agent"
    wake_topics = ("box stable again", "target changed")      # messages that make its loop run at once

    def __init__(self, api: LabAPI, state: LabState, brain: str, switches: dict | None):
        self.api = api
        self.agent = Agent(api, state, self.name, brain, switches)
        self.rec = {k: [] for k in ["t_s", "eta", "kalman", "T_true", "target", "measured", "checked"]}
        self.lab_time, self.last_loop, self.next_after = 0.0, api.clock, BASELINE["check_every_s"]
        self.wakes, self.skipped = [], 0    # watchdog wakes (t_s, alarms, acted) and checks skipped by the prediction
        self.last_target = api.get_status()["target"]
        self.step = {}

    def start(self) -> None:
        """The first map, before the first step."""
        self.api.measure_map()
        self.last_loop = self.api.clock

    def on_step(self) -> None:
        """The task's work at the start of every step: a focus photo and the thermometer."""
        self.step = {"t_step": self.api.clock, "eta": self.api.focus(DRIFT_TEST["focus_target"]),
                     "reading": self.api.get_temperature()}
        self.step["target"] = self.api.get_status()["target"]

    def alarms(self) -> list[str]:
        """The watchdog alarms that fire now (each written as a warning note)."""
        agent = self.agent
        alarms = [] if agent.watchdog is None else agent.watchdog.check(self.api.clock)
        for alarm in alarms:
            agent._warn(f"watchdog: {alarm}")
        return alarms

    def due(self, alarms: list[str], messages: list[dict]) -> str | None:
        """Why the loop runs now: "schedule", "target changed", "watchdog ...", "message ..." (only with
        use_messages and a wake topic), or None."""
        if self.api.clock - self.last_loop >= self.next_after:
            return "schedule"
        if self.step["target"] != self.last_target:
            return "target changed"
        if alarms:
            return "watchdog " + ", ".join(alarms)
        wake = [m["topic"] for m in messages if m["topic"] in self.wake_topics] if self.agent.use_messages else []
        return "message " + ", ".join(wake) if wake else None

    def run_loop(self, reason: str) -> None:
        """One loop of the Map agent (reason from due()), then the step records."""
        api = self.api
        if reason != "schedule":
            api.note(f"wake: {reason}")
        t0 = api.clock
        done = self.agent.run_once()
        self.skipped += done["obs"]["predicted"]
        if reason.startswith("watchdog"):
            self.wakes.append((t0, reason.removeprefix("watchdog ").split(", "), bool(done["plan"])))
        measured = any(name == "measure_map" for name, _ in done["plan"])
        checked = (done["obs"]["check"] is not None and not done["obs"]["predicted"]) + (done["verify"] is not None)
        self.lab_time += api.clock - t0
        self.last_loop, self.last_target = t0, self.step["target"]
        retry = "box not stable" in done["problems"]
        self.next_after = SETTINGS["retry_after_s"] if retry else BASELINE["check_every_s"]
        self.record(measured, checked)

    def record(self, measured: bool = False, checked: int = 0) -> None:
        """The records of this step (a step without a loop: record() with the defaults)."""
        reading = self.step["reading"]
        for key, value in [("t_s", self.step["t_step"]), ("eta", self.step["eta"]), ("kalman", reading["kalman"]),
                           ("T_true", self.api.box.T), ("target", reading["target"]), ("measured", measured),
                           ("checked", checked)]:
            self.rec[key].append(value)

    def result(self) -> dict:
        """Per-step arrays and run totals (the keys of baseline.run, plus wakes, skipped and fresh_checks)."""
        out = {key: np.array(values) for key, values in self.rec.items()}
        out.update({"lab_time_s": self.lab_time, "warnings": self.agent.warnings, "api": self.api, "wakes": self.wakes,
                    "skipped": self.skipped, "fresh_checks": np.array(self.agent.fresh_checks)})
        return out


def run_info(task: MapTask, operator: str, script: str, brain: str, seed: int, box_settings: dict | None,
             switches: dict | None, lab_settings: dict | None) -> dict:
    """The header of an analyst report: operator, script, brain, PID, seed, map rule, time drift, lab time, mean eta."""
    on = {**SETTINGS, **(switches or {})}
    tau = {**OPTICS, **(lab_settings or {})}["time_drift_tau_min"]
    return {"script": script, "operator": operator, "brain": brain, "PID": thermostat.pid_text(box_settings),
            "seed": seed, "map rule": "fresh level" if on["use_fresh_level"] else "fixed 0.8",
            "time drift": "none" if tau is None else f"tau {tau:g} min", "lab time of the run (s)": float(task.api.clock),
            "mean eta": float(np.mean(task.rec["eta"]))}


def operator_name(switches: dict | None) -> str:
    """"basic Map agent" (watchdog, safety and prediction off), "full Map agent" (all on), or "Map agent (mixed)"."""
    on = {**SETTINGS, **(switches or {})}
    helpers = {on[k] for k in HELPERS}
    return "Map agent (mixed)" if len(helpers) == 2 else ("full Map agent" if helpers == {True} else "basic Map agent")


def run(seed: int = 0, state: LabState | None = None, brain: str = AI_BRAIN["provider"],
        switches: dict | None = None, run_name: str | None = None, box_settings: dict | None = None,
        lab_settings: dict | None = None, report_dir: str | Path | None = None, script: str = "agents/loop.py",
        operator: str | None = None) -> dict:
    """The 2-hour scenario with a focus photo every step_s, and agent.run_once() every check_every_s
    (counted from the start of the last loop), after retry_after_s while the box is not stable, and at once
    when the target changes or (with the watchdog) an alarm fires. The Map agent alone (no team).

    state: the lab database (an in-memory one if None); brain: provider name; switches: see Agent;
    the run is saved as run_name (default "map_agent_<brain>"); box_settings / lab_settings: THERMOSTAT / OPTICS overrides (see LabAPI);
    report_dir: where the analyst writes its report (default results/analysis/map_agent); script and operator (default from
    the switches): for the report header.
    Returns the same keys as baseline.run ("checked" counts the check_map calls in each step).
    """
    s = BASELINE
    state = LabState(":memory:") if state is None else state
    run_name = f"map_agent_{brain}" if run_name is None else run_name
    state.delete_runs(run_name)
    api = LabAPI(seed=seed, state=state, run_name=run_name, box_settings=box_settings, lab_settings=lab_settings)
    task = MapTask(api, state, brain, switches)
    scenario.add_events(api)
    task.start()
    for k in range(1, int(SCENARIO["duration_s"] // s["step_s"]) + 1):
        # Each step starts on the step_s grid (later only if the last step's work took longer).
        if api.clock < k * s["step_s"]:
            api.wait(k * s["step_s"] - api.clock)
        task.on_step()
        reason = task.due(task.alarms(), [])
        if reason:
            task.run_loop(reason)
        else:
            task.record()
    out = task.result()
    if {**SETTINGS, **(switches or {})}["use_analyst"]:
        from agents import data_analyst
        info = run_info(task, operator or operator_name(switches), script, brain, seed, box_settings, switches, lab_settings)
        out["findings"] = data_analyst.analyze_and_save(
            state, api.run_id, brain, report_dir or Path(ANALYST["out_dir"]) / "map_agent", info)
    return out


def run_team(seed: int = 0, state: LabState | None = None, box_settings: dict | None = None,
             brain: str = AI_BRAIN["provider"], switches: dict | None = None, run_name: str | None = None,
             duration_s: float = SCENARIO["duration_s"], lab_settings: dict | None = None,
             report_dir: str | Path | None = None, script: str = "agents/loop.py") -> dict:
    """The scenario with the team graph (team.py): the Temperature agent and the Map agent as its task.
    Returns the keys of run(), plus "messages" (dicts t_s, sender, receiver, topic, text) and
    "temperature_refusals". report_dir: where the analyst writes its report (default results/analysis/team)."""
    from agents.team import Team
    from agents.temperature_agent import TemperatureAgent

    state = LabState(":memory:") if state is None else state
    run_name = f"team_{brain}" if run_name is None else run_name
    state.delete_runs(run_name)
    api = LabAPI(seed=seed, state=state, run_name=run_name, box_settings=box_settings, lab_settings=lab_settings)
    task = MapTask(api, state, brain, switches)
    temperature = TemperatureAgent(api, state, receiver=task.name)
    api.actor = task.name
    scenario.add_events(api)
    task.start()
    analyst = None
    if {**SETTINGS, **(switches or {})}["use_analyst"]:
        from agents import data_analyst
        team_op = "team" + ("" if {**SETTINGS, **(switches or {})}["use_messages"] else " without messages")
        analyst = lambda: data_analyst.analyze_and_save(
            state, api.run_id, brain, report_dir or Path(ANALYST["out_dir"]) / "team",
            run_info(task, team_op, script, brain, seed, box_settings, switches, lab_settings))
    team = Team(api, task, temperature, int(duration_s // BASELINE["step_s"]), BASELINE["step_s"], analyst)
    messages = team.run()
    out = task.result()
    out.update({"messages": messages, "temperature_refusals": temperature.refusals})
    if analyst:
        out["findings"] = team.findings
    return out


if __name__ == "__main__":
    from agents import data_analyst
    import json
    import os

    from agents import baseline
    from analysis import plots
    from config import STATE

    class BadBrain(brains.FakeBrain):
        """A fake AI that asks for forbidden or out-of-range actions, one answer per loop."""

        answers = [("set_control", {"on": False}), ("set_target", {"T": 25.0}), ("focus", {"point": 99999}),
                   ("wait", {"seconds": -50}), ("wait", {"seconds": 100000})]

        def ask(self, text, problems):
            name, args = self.answers.pop(0)
            return json.dumps({"actions": [{"name": name, "args": args}], "reason": "test"})

    test_state = LabState(":memory:")
    api = LabAPI(seed=0, state=test_state, run_name="bad_brain_test")
    api.measure_map()
    agent = Agent(api, test_state, "map_agent", "rules", dict.fromkeys(SWITCHES, False))   # the plain loop
    agent.brain = BadBrain()
    # The brain is asked only when diagnose finds a problem, so each loop gets one ("map stale").
    for _ in range(len(BadBrain.answers)):
        obs, problems = agent.monitor(), ["map stale"]
        plan = agent.review(agent.plan(obs, problems), obs, problems)
        agent.act(plan)
        assert plan == brains.rules_plan(problems)
    assert agent.warnings == 5 and api.box.control and api.box.target == THERMOSTAT["target"]
    agent.act([("focus", {"point": 99999}), ("wait", {"seconds": 1})])   # act() itself, without parse()
    assert agent.warnings == 6
    test_state.close()
    print("bad AI answers (set_control, set_target, focus(99999), wait(-50), wait(100000)) fell back; "
          "a failing action in act() warned; the run went on: ok")
    for line in api.get_log():
        if "WARNING" in line:
            print(f"  {line}")

    from box_simulation.thermostat import pid_gains, pid_label, plot_pids
    from config import PLOTS, WATCHDOG

    # Two thermostats: the PID the real box runs today (THERMOSTAT) and the main PID of the plots (override only).
    pids = {pid_label(n): pid_gains(n) or None for n in plot_pids()}
    real, main = pid_label("real"), pid_label(PLOTS["main_pid"])
    rules_set = {"fixed 0.8": False, "fresh level": True}
    report_dir = Path(ANALYST["out_dir"]) / "agents"
    state = LabState(STATE["db_path"])
    runs = {}                                       # (stale rule or "-", PID, operator) -> run
    for pid, box in pids.items():
        for policy in ("never", "rules"):
            runs[("-", pid, policy)] = baseline.run(policy, box_settings=box)
        for rule, fresh in rules_set.items():
            for name, helpers in [("basic", False), ("full", True)]:
                switches = {**dict.fromkeys(HELPERS, helpers), "use_fresh_level": fresh}
                run_name = f"{name}_map_agent_{thermostat.pid_tag(box)}_{'fresh_level' if fresh else 'fixed_stale'}"
                runs[(rule, pid, name)] = run(state=state, brain="rules", switches=switches, run_name=run_name,
                                              box_settings=box, report_dir=report_dir)
    # The fake brain is a test: an in-memory database and no report.
    fake = run(state=LabState(":memory:"), brain="fake", switches={"use_analyst": False})
    provider = AI_BRAIN["test_provider"]
    brains.load_env()
    if os.environ.get(AI_BRAIN["key_env"][provider]):
        print(f"{provider}: key found, running the Map agent with the {provider} brain")
        runs[("fresh level", real, provider)] = run(
            state=state, brain=provider, run_name=f"full_map_agent_{thermostat.pid_tag(None)}_fresh_level_{provider}",
            report_dir=report_dir)
    else:
        print(f"{provider}: no API key, skipped")
    state.close()
    print(f"analyst reports: {data_analyst.write_index(report_dir)}")

    event_times = [t for t, _ in baseline.scenario_events()]

    def useful(wake: tuple) -> bool:
        """A watchdog wake that led to an action or came within useful_within_s after a scenario event."""
        t, _, acted = wake
        return acted or any(0 <= t - t_event <= WATCHDOG["useful_within_s"] for t_event in event_times)

    def agent_only(f):
        """A row value only the Map agents have."""
        return lambda r: "-" if "wakes" not in r else f(r)

    rows = [("mean eta", "mean_eta", ".1f"), ("min eta", "min_eta", ".1f"),
            ("share eta > 30", "share_eta_above_30", ".2f"), ("measure_map calls", "measure_map_calls", "d"),
            ("check_map calls", "check_map_calls", "d"), ("lab time on maps (s)", "lab_time_on_maps_s", ".0f"),
            ("warnings", "warnings", "d"), ("new maps while box off target", "maps_while_box_off_target", "d"),
            (f"share time within ±{THERMOSTAT['limit_C']:g} C", "share_within_limit", ".2f")]
    extra = [("first reaction to cooler (s)", lambda r: "-" if first_reaction(r["api"].get_log()) is None
              else f"{first_reaction(r['api'].get_log()):.0f}"),
             ("checks skipped by prediction", agent_only(lambda r: f"{r['skipped']}")),
             ("watchdog wakes (useful)", agent_only(lambda r: f"{len(r['wakes'])} ({sum(map(useful, r['wakes']))})")),
             ("fresh map check (median)", agent_only(lambda r: f"{np.median(r['fresh_checks']):.3f}")),
             ("back-offs", agent_only(lambda r: f"{sum('backing off' in line for line in r['api'].get_log())}"))]
    # A map measured and checked at once, right after the warm-up, with each PID.
    fresh_now = {}
    for pid, box in pids.items():
        api = LabAPI(seed=0, box_settings=box)
        api.measure_map()
        fresh_now[pid] = api.check_map()
    for rule in rules_set:
        cols = [(pid, op) for pid in pids for op in ("never", "rules", "basic", "full")]
        cell = {(pid, op): runs[("-" if op in ("never", "rules") else rule, pid, op)] for pid, op in cols}
        summary = {key: baseline.summarize(r) for key, r in cell.items()}
        print(f"\nstale rule: {rule}" + ("" if rule == "fixed 0.8" else
              f" (stale below {AGENT_LOOP['stale_share']:g} x median of the last {AGENT_LOOP['fresh_n']} new maps, back-off)"))
        print(f"{'':30} " + " ".join(pid[:4 * 9 + 3].center(4 * 9 + 3) for pid in pids))
        print(f"{'':30} " + " ".join(op.rjust(9) for _, op in cols))
        for label, key, fmt in rows:
            print(f"{label:30} " + " ".join(f"{summary[c][key]:{fmt}}".rjust(9) for c in cols))
        for label, value in extra:
            print(f"{label:30} " + " ".join(value(cell[c]).rjust(9) for c in cols))
    print("\nfresh map check right after the warm-up (measure_map, then check_map at once): "
          + ", ".join(f"{pid} {value:.3f}" for pid, value in fresh_now.items()))
    default = runs[("fresh level", real, "full")]
    # A refused AI plan also gets its rules plan reviewed, which can add a warning; everything else must match.
    a, b = baseline.summarize(fake), baseline.summarize(default)
    same = all(a[key] == b[key] for key in a if key != "warnings")
    print(f"full Map agent, fake brain = rules brain (default settings, all but warnings): {'yes' if same else 'NO'}, "
          f"warnings {a['warnings']} vs {b['warnings']}")
    for pid in pids:
        print(f"full Map agent ({pid}, fresh level) back-off and verify notes:")
        for line in runs[("fresh level", pid, "full")]["api"].get_log():
            if any(word in line for word in ("backing off", "verify:", "safety:", "check skipped", "EVENT")):
                print(f"  {line}")

    # Plot 01 with the main PID; 02: the full Map agent with the real box PID and the main PID.
    out = Path("results/agents")
    full_main = runs[("fresh level", main, "full")]
    base = {"never": runs[("-", main, "never")], "rules": runs[("-", main, "rules")], "full Map agent": full_main}
    figures = [
        (base, "01_never_rules_agent_focus_brightness.png", {"never": "red", "rules": "blue", "full Map agent": "green"},
         f"Focus brightness during the scenario: never vs. fixed rules vs. full Map agent\n{main}, seed 0"),
        ({real: default, main: full_main}, "02_full_map_agent_real_vs_main_pid_focus_brightness.png",
         {real: "purple", main: "darkorange"},
         "Focus brightness: full Map agent with the real box PID and the main PID (seed 0)"),
    ]
    for curves, name, colors, title in figures:
        print(plots.plot_policy_comparison({label: (r["t_s"], r["eta"], r["measured"]) for label, r in curves.items()},
                                           baseline.scenario_events(), out / name, colors=colors, title=title))
