"""
temperature_agent.py - the Temperature agent: it owns the box (target and control) and tells the Map agent when the
box is not fit for maps.

A LangGraph subgraph with the same phases as the Map agent: monitor -> diagnose -> plan -> safety check -> act ->
verify (no problems: the loop ends after diagnose). Rules brain only. It reads only the thermometer rows of the
lab database (never the true temperature) and the watchdog alarms it is given.

Messages to the Map agent, each once per event (written to the database and returned to the team graph):
  "box unstable"     |kalman - target| > swing_band_C for longer than WATCHDOG off_target_s
  "cooler problem"   the watchdog alarm "cooler at full power"
  "box stable again" after one of the two above, once the box is back inside the band for AGENT_LOOP stable_s
  "target changed"   after every change of the target (the user changes it; this agent never does)
  "target too close to room"  with a target change, if the new target is above room - LAB_LIMITS room_margin_C (the
                     box can only cool); with the room temperature unknown the check is skipped with a note
Allowed actions: wait, send_message, and set_control(on=True) only (it may switch the PID back on, never off). It reaches
the lab only through SafeLab (temperature_agent role); with a human (human.py) its graph stops before act for approval.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from agents.core.human import Human, compile_graph, invoke
from agents.core.safe_lab import SafeLab, target_vs_room
from agents.core.state import LabState, TOPICS
from config import AGENT_LOOP, TARGET_C, THERMOSTAT, WATCHDOG

TEMPERATURE_ALLOWED = ("wait", "send_message", "set_control")


class TemperatureState(TypedDict, total=False):
    """The graph state of one loop."""
    alarms: list[str]                   # watchdog alarms that fired this step
    obs: dict                           # monitor
    problems: list[str]                 # diagnose
    proposed: list[tuple[str, dict]]    # plan
    actions: list[tuple[str, dict]]     # safety check: the allowed plan
    sent: list[dict]                    # act: messages sent
    verified: bool                      # verify


class TemperatureAgent:
    """Rules-brain Temperature agent; api gives the clock, notes, the thermometer and set_control (any lab_api.Lab or a
    stand-in), always through SafeLab with the temperature_agent role."""

    def __init__(self, api, state: LabState, name: str = "temperature_agent", receiver: str = "map_agent",
                 human: Human | None = None):
        """human: the human in the loop (default HUMAN in config.py, mode "off")."""
        self.api = api if isinstance(api, SafeLab) else SafeLab(api, name, role="temperature_agent")
        self.state, self.name, self.receiver = state, name, receiver
        self.human = human or Human()
        self.room_detail = ""
        self.unstable = self.cooler = False         # a "box unstable" / "cooler problem" event is open
        self.last_target = None
        self.control_on_t = -np.inf                 # last set_control(on=True), retried at most once per off_target_s
        self.sent = []                              # every message of the run, oldest first
        self.refusals = 0
        self.graph = self._build_graph()

    def _note(self, text: str, level: str = "info") -> None:
        """A note in the lab database with this agent as the actor (SafeLab writes it as this agent)."""
        self.api.note(text, level=level)

    def _window(self, rows: list[dict], now: float, seconds: float) -> list[dict]:
        """The rows of the last `seconds`, or [] if they do not yet cover that whole time."""
        rows = [r for r in rows if r["t_s"] > now - seconds - 1]
        return rows if rows and rows[0]["t_s"] <= now - seconds + 1 else []

    def _monitor_node(self, state: TemperatureState) -> TemperatureState:
        """Thermometer rows of the last few minutes (Kalman, raw, target, duty) and the alarms."""
        now = self.api.clock
        seconds = max(WATCHDOG["off_target_s"], AGENT_LOOP["stable_s"])
        rows = [r for r in self.state.since("temperatures", self.api.run_id, now - seconds - 1) if r["t_s"] <= now]
        return {"obs": {"now": now, "rows": rows, "alarms": state.get("alarms", [])}}

    def _diagnose_node(self, state: TemperatureState) -> TemperatureState:
        """Events to report (topics) and "control seems off" (box warm above the band with duty 0 all the time)."""
        obs, band = state["obs"], THERMOSTAT["swing_band_C"]
        rows, now = obs["rows"], obs["now"]
        if not rows:
            return {"problems": []}
        problems = []
        if self.last_target is not None and rows[-1]["target"] != self.last_target:
            problems.append("target changed")
            room = self.api.get_temperature().get("room")
            warning = target_vs_room(rows[-1]["target"], room)
            if room is None:
                self._note("target vs room check skipped: room temperature unknown")
            elif warning:
                problems.append("target too close to room")
                self.room_detail = warning
        off = self._window(rows, now, WATCHDOG["off_target_s"])
        if not self.unstable and off and all(abs(r["kalman"] - r["target"]) > band for r in off):
            problems.append("box unstable")
        if not self.cooler and "cooler at full power" in obs["alarms"]:
            problems.append("cooler problem")
        calm = self._window(rows, now, AGENT_LOOP["stable_s"])
        if (self.unstable or self.cooler) and calm and all(abs(r["kalman"] - r["target"]) < band for r in calm):
            problems.append("box stable again")
        if off and now - self.control_on_t >= WATCHDOG["off_target_s"] and \
                all(r["duty"] == 0 and r["kalman"] > r["target"] + band for r in off):
            problems.append("control seems off")
        if self.last_target is None:
            self.last_target = rows[-1]["target"]
        return {"problems": problems}

    def _plan_node(self, state: TemperatureState) -> TemperatureState:
        """Rules brain: a message for every event, and switch the PID back on if it seems off."""
        last = state["obs"]["rows"][-1]
        detail = f"kalman {last['kalman']:.2f} C, target {last['target']:g} C"
        plan = [("send_message", {"topic": p, "text": self.room_detail if p == "target too close to room" else detail})
                for p in state["problems"] if p in TOPICS]
        if "control seems off" in state["problems"]:
            plan.append(("set_control", {"on": True}))
        return {"proposed": plan}

    def _safety_node(self, state: TemperatureState) -> TemperatureState:
        """Only wait, send_message and set_control(on=True); everything else is refused with a warning note."""
        allowed = []
        for name, args in state["proposed"]:
            if name not in TEMPERATURE_ALLOWED or (name == "set_control" and args.get("on") is not True):
                self._note(f"safety: refused {name}({args}) because the Temperature agent may not do that", "warning")
                self.refusals += 1
            else:
                allowed.append((name, args))
        return {"actions": allowed}

    def _act_node(self, state: TemperatureState) -> TemperatureState:
        """Send the messages (database and team state) and run the other actions."""
        sent = []
        for name, args in state["actions"]:
            if name == "send_message":
                message = {"t_s": self.api.clock, "sender": self.name, "receiver": self.receiver, **args}
                self.state.add_message(self.api.run_id, message["t_s"], self.name, self.receiver, args["topic"], args["text"])
                self._note(f"message to {self.receiver}: {args['topic']} ({args['text']})")
                sent.append(message)
                topic = args["topic"]
                self.unstable = True if topic == "box unstable" else (False if topic == "box stable again" else self.unstable)
                self.cooler = True if topic == "cooler problem" else (False if topic == "box stable again" else self.cooler)
                if topic == "target changed":
                    self.last_target = state["obs"]["rows"][-1]["target"]
            elif name == "set_control":
                self.api.set_control(on=True)
                self.control_on_t = self.api.clock
                self._note("act: control switched back on")
            else:
                self.api.wait(**args)
        self.sent += sent
        return {"sent": sent}

    def _verify_node(self, state: TemperatureState) -> TemperatureState:
        """Every sent message is in the database."""
        stored = self.state.unread(self.api.run_id, self.receiver, self.api.clock - 1)
        ok = all(any(r["topic"] == m["topic"] for r in stored) for m in state["sent"])
        if not ok:
            self._note("verify: a sent message is missing in the database", "warning")
        return {"verified": ok}

    def _build_graph(self):
        """The compiled subgraph monitor -> diagnose -> (no problems: END) plan -> safety -> act -> verify."""
        graph = StateGraph(TemperatureState)
        for name, node in [("monitor", self._monitor_node), ("diagnose", self._diagnose_node), ("plan", self._plan_node),
                           ("safety", self._safety_node), ("act", self._act_node), ("verify", self._verify_node)]:
            graph.add_node(name, node)
        graph.add_edge(START, "monitor")
        graph.add_edge("monitor", "diagnose")
        graph.add_conditional_edges("diagnose", lambda state: "plan" if state["problems"] else END, ["plan", END])
        for a, b in [("plan", "safety"), ("safety", "act"), ("act", "verify")]:
            graph.add_edge(a, b)
        graph.add_edge("verify", END)
        return compile_graph(graph, self.human)

    def _review(self, values: dict) -> list[tuple[str, dict]]:
        """The human's review at the interrupt before act (rules brain only: nothing counts as an AI extra)."""
        return self.human.review(self.api, self.name, values["actions"], ", ".join(values["problems"]))

    def run_once(self, alarms: list[str]) -> list[dict]:
        """One loop at the current lab time with this step's watchdog alarms; returns the messages sent."""
        return invoke(self.graph, {"alarms": alarms}, self.human, self._review).get("sent", [])


if __name__ == "__main__":
    band, target = THERMOSTAT["swing_band_C"], TARGET_C

    class StandIn:
        """The few lab parts the agent uses, with a clock and a room temperature set by the test (no lab, no box)."""
        def __init__(self, state, run_id, room=25.0):
            self.state, self.run_id, self.clock, self.actor, self.control_calls = state, run_id, 0.0, "test", []
            self.room = room

        def get_temperature(self):
            return {"room": self.room}

        def note(self, text, level="info"):
            self.state.add_note(self.run_id, self.clock, self.actor, level, text)

        def set_control(self, on):
            self.control_calls.append(on)

        def wait(self, seconds):
            self.clock += seconds

    def make_run(kalman, aim=lambda t: target, duty=lambda t: 500, seconds=2400):
        """A database with made-up thermometer rows (kalman(t), target aim(t), duty(t)) for one run."""
        state = LabState(":memory:")
        run_id = state.start_run("temperature_agent_test", 0, {})
        for t in range(1, seconds + 1):
            state.add_temperature(run_id, t, kalman(t), kalman(t), aim(t), duty(t))
        return state, run_id

    def play(rows: dict, alarms_at: dict | None = None, seconds: int = 2400, room=25.0) -> tuple[list, StandIn]:
        """Run the agent every 60 s over made-up rows; alarms_at maps time -> alarms. Returns (topics, stand-in)."""
        state, run_id = make_run(**rows, seconds=seconds)
        api = StandIn(state, run_id, room)
        agent = TemperatureAgent(api, state)
        for now in range(60, seconds + 1, 60):
            api.clock = float(now)
            agent.run_once((alarms_at or {}).get(now, []))
        return [(m["t_s"], m["topic"]) for m in agent.sent], api

    def swing(t):
        """A normal hold: +-0.3 C PID swing, period 5 min."""
        return target + 0.3 * np.sin(2 * np.pi * t / 300)

    cases = {
        "normal hold": ({"kalman": swing}, None, []),
        "box off target, then back": ({"kalman": lambda t: target + (0.8 if 500 < t < 1100 else 0.0)}, None,
                                      ["box unstable", "box stable again"]),
        "cooler alarm, box off, back": ({"kalman": lambda t: target + (0.9 if 600 < t < 1300 else 0.0)},
                                        {660: ["cooler at full power"], 1260: ["cooler at full power"]},
                                        ["cooler problem", "box unstable", "box stable again"]),
        "target change": ({"kalman": lambda t: target + (0.5 if t > 700 else 0.0),
                           "aim": lambda t: target + (0.5 if t > 700 else 0.0)}, None, ["target changed"]),
    }
    for name, (rows, alarms_at, expect) in cases.items():
        topics, api = play(rows, alarms_at)
        print(f"{name:30} messages {topics}")
        assert [topic for _, topic in topics] == expect, topics
        assert api.control_calls == []

    # Target vs room (the box can only cool): a new target above room - room_margin_C gets its own message;
    # with the room temperature unknown only a note says the check was skipped.
    from config import LAB_LIMITS
    step = {"kalman": lambda t: target + (0.5 if t > 700 else 0.0), "aim": lambda t: target + (0.5 if t > 700 else 0.0)}
    close_room = target + 0.5 + LAB_LIMITS["room_margin_C"] - 0.2
    topics, api = play(step, room=close_room)
    print(f"{'target 0.2 C too close to room':30} messages {topics}")
    assert [topic for _, topic in topics] == ["target changed", "target too close to room"]
    assert "can only cool" in api.state.query("SELECT text FROM messages WHERE topic = 'target too close to room'")[0]["text"]
    topics, api = play(step, room=None)
    skipped = api.state.query("SELECT actor, text FROM notes WHERE text LIKE '%room temperature unknown%'")
    print(f"{'room temperature unknown':30} messages {topics}, notes {[n['text'] for n in skipped]}")
    assert [topic for _, topic in topics] == ["target changed"] and skipped[0]["actor"] == "temperature_agent"

    # Control off: duty 0 and the box warm above the band -> set_control(on=True), never off.
    topics, api = play({"kalman": lambda t: target + (0.6 + 0.001 * (t - 600) if t > 600 else 0.0),
                        "duty": lambda t: 0 if t > 600 else 500}, seconds=1500)
    print(f"{'PID off (duty 0, box warms)':30} messages {topics}, set_control calls {api.control_calls}")
    assert api.control_calls and all(on is True for on in api.control_calls)
    assert [topic for _, topic in topics] == ["box unstable"]

    # The safety check refuses set_control(on=False) and set_target even if a plan asks for them.
    state, run_id = make_run(swing, seconds=300)
    api = StandIn(state, run_id)
    agent = TemperatureAgent(api, state)
    api.clock = 300.0
    out = agent._safety_node({"proposed": [("set_control", {"on": False}), ("set_target", {"T": 25.0}),
                                           ("send_message", {"topic": "box unstable", "text": "test"})]})
    print(f"{'safety check':30} allowed {out['actions']}, refused {agent.refusals}")
    assert out["actions"] == [("send_message", {"topic": "box unstable", "text": "test"})] and agent.refusals == 2
    print("temperature_agent.py: each message once, none on a normal hold, never off or set_target: ok")
