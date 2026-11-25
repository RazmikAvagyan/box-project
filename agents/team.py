"""
team.py - the team graph (LangGraph): the Temperature agent and one task agent on one lab clock.

One tick = step_s of lab time:
  lab_step (clock, the task's step work, e.g. a focus photo and the thermometer) -> watchdog ->
  temperature_agent -> "task due?" -> <task> or skip -> "end of run?" -> lab_step again, or (after the last tick)
  data_analyst -> END (the analyst is an optional callable, so this file stays task-independent).

The graph knows nothing about the task: the task is any object with
  name, on_step(), alarms() -> list[str], due(alarms, messages) -> reason or None, run_loop(reason), record().
Today the task is the Map agent (loop.MapTask, put together with the lab in loop.run_team); another task agent
can replace it without changing this file. This module imports nothing optics-specific.
The messages of the run are kept in the team state (and the Temperature agent writes them to the database).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class TeamState(TypedDict, total=False):
    """The team graph state, one tick at a time."""
    k: int                                              # tick number, 1 .. n_ticks
    alarms: list[str]                                   # watchdog alarms that fired this tick
    new_messages: list[dict]                            # messages sent this tick
    messages: Annotated[list[dict], operator.add]       # every message of the run


class Team:
    """The team graph over a lab (clock and wait), a task agent and the Temperature agent."""

    def __init__(self, lab, task, temperature_agent, n_ticks: int, step_s: float, analyst=None):
        """analyst: optional callable run once after the last tick (the data_analyst node); it returns findings."""
        self.lab, self.task, self.temperature_agent, self.analyst = lab, task, temperature_agent, analyst
        self.findings = []
        self.n_ticks, self.step_s = n_ticks, step_s
        self.graph = self._build_graph()

    def lab_step(self, state: TeamState) -> TeamState:
        """Wait for the step_s grid, then the task's work of this step."""
        k = state.get("k", 0) + 1
        # Each step starts on the step_s grid (later only if the last step's work took longer).
        if self.lab.clock < k * self.step_s:
            self.lab.wait(k * self.step_s - self.lab.clock)
        self.task.on_step()
        return {"k": k}

    def watchdog(self, state: TeamState) -> TeamState:
        """The alarms that fire now (the task's watchdog); they go to the Temperature agent and the task."""
        return {"alarms": self.task.alarms()}

    def temperature(self, state: TeamState) -> TeamState:
        """One loop of the Temperature agent subgraph; its messages go into the team state."""
        sent = self.temperature_agent.run_once(state["alarms"])
        return {"new_messages": sent, "messages": sent}

    def task_due(self, state: TeamState) -> str:
        """Conditional edge: the task node if the task says its loop is due, else skip."""
        self.reason = self.task.due(state["alarms"], state["new_messages"])
        return self.task.name if self.reason else "skip"

    def task_loop(self, state: TeamState) -> TeamState:
        """One loop of the task agent."""
        self.task.run_loop(self.reason)
        return {}

    def skip(self, state: TeamState) -> TeamState:
        """No task loop this tick."""
        self.task.record()
        return {}

    def end_of_run(self, state: TeamState) -> str:
        """Conditional edge: the next tick, or (after n_ticks) the data analyst, or END without one."""
        if state["k"] < self.n_ticks:
            return "lab_step"
        return "data_analyst" if self.analyst else END

    def data_analyst(self, state: TeamState) -> TeamState:
        """After the last tick: the data analyst reads the run from the lab database."""
        self.findings = self.analyst()
        return {}

    def _build_graph(self):
        """The compiled team graph."""
        graph = StateGraph(TeamState)
        for name, node in [("lab_step", self.lab_step), ("watchdog", self.watchdog),
                           ("temperature_agent", self.temperature), (self.task.name, self.task_loop), ("skip", self.skip)]:
            graph.add_node(name, node)
        if self.analyst:
            graph.add_node("data_analyst", self.data_analyst)
            graph.add_edge("data_analyst", END)
        graph.add_edge(START, "lab_step")
        graph.add_edge("lab_step", "watchdog")
        graph.add_edge("watchdog", "temperature_agent")
        graph.add_conditional_edges("temperature_agent", self.task_due, [self.task.name, "skip"])
        for name in (self.task.name, "skip"):
            graph.add_conditional_edges(name, self.end_of_run, ["lab_step", "data_analyst" if self.analyst else END])
        return graph.compile()

    def run(self) -> list[dict]:
        """All ticks through the graph; returns the messages of the run (from the team state)."""
        final = self.graph.invoke({"k": 0, "messages": []}, {"recursion_limit": 6 * self.n_ticks + 10})
        return final["messages"]

    def mermaid(self) -> str:
        """The team graph as Mermaid text (drawn by LangGraph itself, no drawing library)."""
        return self.graph.get_graph().draw_mermaid()


if __name__ == "__main__":
    import sys

    # 1. The graph with a made-up task and Temperature agent (nothing optics-specific): routing, messages, end.
    class Clock:
        """A lab that only has a clock."""
        clock = 0.0

        def wait(self, seconds):
            self.clock += seconds

    class CountTask:
        """A made-up task: due every third tick or on a "box stable again" message; counts its loops."""
        name = "count_task"

        def __init__(self, lab):
            self.lab, self.loops, self.skips, self.ticks = lab, [], 0, 0

        def on_step(self):
            self.ticks += 1

        def alarms(self):
            return ["test alarm"] if self.ticks == 5 else []

        def due(self, alarms, messages):
            if any(m["topic"] == "box stable again" for m in messages):
                return "message"
            return "schedule" if self.ticks % 3 == 0 else ("watchdog" if alarms else None)

        def run_loop(self, reason):
            self.loops.append((self.lab.clock, reason))

        def record(self):
            self.skips += 1

    class Talker:
        """A made-up Temperature agent: "box stable again" at tick 4."""
        def __init__(self, lab):
            self.lab, self.seen = lab, []

        def run_once(self, alarms):
            self.seen += alarms
            return [{"t_s": self.lab.clock, "topic": "box stable again"}] if self.lab.clock == 240 else []

    lab = Clock()
    task, talker = CountTask(lab), Talker(lab)
    team = Team(lab, task, talker, n_ticks=7, step_s=60)
    messages = team.run()
    print(f"made-up task: loops {task.loops}, skips {task.skips}, alarms seen {talker.seen}, messages {messages}")
    assert task.loops == [(180, "schedule"), (240, "message"), (300, "watchdog"), (360, "schedule")]
    assert task.skips == 3 and talker.seen == ["test alarm"] and messages == [{"t_s": 240, "topic": "box stable again"}]
    assert "count_task" in team.mermaid() and "optics" not in sys.modules and "optics_simulation" not in sys.modules
    print("team graph with a made-up task (no optics imported): ok")

    # 2. The real team: Temperature agent + Map agent (put together in agents/loop.py).
    from pathlib import Path

    import numpy as np

    from agents import baseline, loop
    from agents.state import LabState
    from box_simulation.thermostat import pid_gains, pid_label
    from config import BASELINE, PLOTS, STATE

    from agents import data_analyst
    from box_simulation.thermostat import pid_tag
    from config import ANALYST

    box = pid_gains(PLOTS["main_pid"])            # the main PID of the plots
    tag = pid_tag(box)
    report_dir = Path(ANALYST["out_dir"]) / "team"
    state = LabState(STATE["db_path"])
    with_messages = loop.run_team(state=state, brain="rules", box_settings=box, run_name=f"team_{tag}",
                                  report_dir=report_dir, script="agents/team.py")
    without = loop.run_team(state=state, brain="rules", switches={"use_messages": False}, run_name=f"team_{tag}_no_messages",
                            box_settings=box, report_dir=report_dir, script="agents/team.py")
    full = loop.run(state=state, brain="rules", box_settings=box, run_name=f"team_reference_full_map_agent_{tag}_fresh_level",
                    report_dir=report_dir, script="agents/team.py", operator="full Map agent (team reference)")

    # With use_messages False the team equals the full Map agent (loop.run) exactly.
    a, b = baseline.summarize(without), baseline.summarize(full)
    same = a == b and np.array_equal(without["eta"], full["eta"]) and np.array_equal(without["t_s"], full["t_s"])
    print(f"team without messages = full Map agent ({pid_label(PLOTS['main_pid'])}): {'yes' if same else 'NO'}")
    assert same, (a, b)

    # While "box unstable" or "cooler problem" is open, the Map agent does no check_map or measure_map.
    run_id = with_messages["api"].run_id
    msgs = state.query("SELECT t_s, topic, text FROM messages WHERE run_id = ? ORDER BY t_s, rowid", (run_id,))
    print("messages:")
    for m in msgs:
        print(f"  t={m['t_s']:.0f}s {m['topic']} ({m['text']})")
    holds, start = [], None
    for m in msgs:
        if m["topic"] in ("box unstable", "cooler problem") and start is None:
            start = m["t_s"]
        elif m["topic"] == "box stable again" and start is not None:
            holds.append((start, m["t_s"]))
            start = None
    if start is not None:
        holds.append((start, np.inf))
    maps = state.query("SELECT t_s, kind FROM maps WHERE run_id = ?", (run_id,))
    during = [m for m in maps if any(t0 < m["t_s"] < t1 for t0, t1 in holds)]
    print(f"hold periods {holds}: map rows inside them {during}")
    assert holds and not during
    assert [(m["t_s"], m["topic"]) for m in with_messages["messages"]] == [(m["t_s"], m["topic"]) for m in msgs]
    s, f = baseline.summarize(with_messages), baseline.summarize(full)
    print(f"seed 0: team mean eta {s['mean_eta']:.1f}, min {s['min_eta']:.1f}, maps {s['measure_map_calls']}, "
          f"checks {s['check_map_calls']}, lab time {s['lab_time_on_maps_s']:.0f} s; full Map agent mean eta "
          f"{f['mean_eta']:.1f}, min {f['min_eta']:.1f}, maps {f['measure_map_calls']}, checks {f['check_map_calls']}, "
          f"lab time {f['lab_time_on_maps_s']:.0f} s")
    state.close()

    out = Path("results/agents")
    lab_api = loop.LabAPI(seed=0)            # a fresh lab, only to draw the graph with the Map agent as its task
    real = Team(lab_api, loop.MapTask(lab_api, LabState(":memory:"), "rules", None), talker, 1, BASELINE["step_s"])
    (out / "team_graph.mmd").write_text(real.mermaid())
    print(f"saved {out / 'team_graph.mmd'}")
    print(f"analyst reports: {data_analyst.write_index(report_dir)}")
    print("team.py self-test passed")
