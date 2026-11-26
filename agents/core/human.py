"""
human.py - the human in the loop: a LangGraph interrupt before the act node of every agent.

HUMAN["mode"] in config.py:
  "off"          no interrupt; everything exactly as without a human (default)
  "suggest"      the agents only propose: every planned action waits for approval, nothing runs without it
  "approve_big"  only big actions wait: set_target, set_control, measure_map, and anything an AI brain proposes that
                 the rules would not; waits, checks, focus photos and messages go through without asking
The approval comes from a function approver(proposal) -> bool, proposal = {"agent", "action", "args", "reason"}:
in the terminal ask_terminal (y/n with a timeout; no answer = refused, so nothing changes and the lab keeps its safe
state), in tests a scripted one. Every proposal, approval and refusal is a note with the actor "human".
The agents' graphs are compiled with compile_graph() and run with invoke(); with mode "off" both are the plain
LangGraph calls. This module imports nothing optics-specific.
"""

from __future__ import annotations

import itertools
import select
import sys
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver

from agents.core.safe_lab import SafeLab
from config import HUMAN

MODES = ("off", "suggest", "approve_big")
BIG = ("set_target", "set_control", "measure_map")
_threads = itertools.count()


def ask_terminal(proposal: dict, timeout_s: float = HUMAN["timeout_s"]) -> bool:
    """Ask y/n in the terminal; True only for an answer starting with "y" within timeout_s."""
    print(f"\n[{proposal['agent']}] wants {proposal['action']}({proposal['args']}): {proposal['reason']}")
    print(f"approve? y/n (no answer in {timeout_s:g} s = refused): ", end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        print("\nno answer: refused")
        return False
    return sys.stdin.readline().strip().lower().startswith("y")


class Human:
    """The mode and the approver; reviews an agent's planned actions before its act node."""

    def __init__(self, mode: str = HUMAN["mode"], approver: Callable[[dict], bool] | None = None):
        """mode: one of MODES; approver: proposal dict -> bool (default ask_terminal)."""
        if mode not in MODES:
            raise ValueError(f"unknown human mode {mode!r}, expected one of {MODES}")
        self.mode, self.approver = mode, approver or ask_terminal
        self.asked = []                         # every proposal shown to the human, with the answer

    def needs_approval(self, name: str, args: dict, rules_actions: list | None) -> bool:
        """Must this action wait? suggest: always; approve_big: big actions and AI actions the rules would not plan."""
        if self.mode == "suggest":
            return True
        if self.mode == "approve_big":
            return name in BIG or (rules_actions is not None and (name, args) not in rules_actions)
        return False

    def review(self, lab, agent: str, actions: list[tuple[str, dict]], reason: str,
               rules_actions: list | None = None) -> list[tuple[str, dict]]:
        """The approved actions (in order). lab: the agent's lab (a SafeLab or any lab) for the human's notes;
        rules_actions: what the rules would plan (None = the actions come from the rules)."""
        safe = lab if isinstance(lab, SafeLab) else SafeLab(lab, "human", role="operator")
        note = lambda text, level="info": safe.note_as("human", text, level)
        approved = []
        for name, args in actions:
            if not self.needs_approval(name, args, rules_actions):
                approved.append((name, args))
                continue
            proposal = {"agent": agent, "action": name, "args": args, "reason": reason}
            note(f"proposal from {agent}: {name}({args}), reason: {reason}")
            ok = bool(self.approver(proposal))
            self.asked.append({**proposal, "approved": ok})
            note(f"{'approved' if ok else 'refused'}: {name}({args}) from {agent}", level="info" if ok else "warning")
            if ok:
                approved.append((name, args))
        return approved


def compile_graph(graph, human: Human | None):
    """The compiled graph; with a human (mode not "off") it stops before its "act" node (needs a checkpointer)."""
    if human is None or human.mode == "off":
        return graph.compile()
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["act"])


def invoke(compiled, inputs: dict, human: Human | None, review: Callable[[dict], list]) -> dict:
    """Run the graph once; at the interrupt before "act", review(state values) gives the approved "actions", then
    the graph goes on. Returns the final state values (as compiled.invoke does)."""
    if human is None or human.mode == "off":
        return compiled.invoke(inputs)
    config = {"configurable": {"thread_id": f"loop-{next(_threads)}"}}
    compiled.invoke(inputs, config)
    snapshot = compiled.get_state(config)
    if "act" in snapshot.next:
        compiled.update_state(config, {"actions": review(snapshot.values)})
        compiled.invoke(None, config)
    return compiled.get_state(config).values


if __name__ == "__main__":
    import numpy as np

    from agents.team import baseline

    from agents.map_agent import loop
    from agents.core.state import LabState

    class NoteLab:
        """A lab that only keeps notes (for the review tests)."""
        clock, actor, run_id = 0.0, "operator", None

        def __init__(self):
            self.notes = []

        def note(self, text, level="info"):
            self.notes.append((self.actor, level, text))

    def recorder(answer: bool):
        """A scripted approver that answers `answer` and remembers what it was asked."""
        asked = []
        return asked, lambda proposal: asked.append(proposal["action"]) or answer

    plan = [("wait", {"seconds": 60}), ("check_map", {}), ("focus", {"point": 27}), ("measure_map", {}),
            ("send_message", {"topic": "box unstable", "text": "test"}), ("set_control", {"on": True})]
    for mode, expect in [("off", []), ("approve_big", ["measure_map", "set_control"]),
                         ("suggest", [name for name, _ in plan])]:
        lab, (asked, approver) = NoteLab(), recorder(True)
        approved = Human(mode, approver).review(lab, "map_agent", plan, "test")
        print(f"{mode:12} asked about {asked}")
        assert asked == expect and approved == plan
        assert len(lab.notes) == 2 * len(expect) and all(actor == "human" for actor, _, _ in lab.notes)
    # approve_big: an AI action the rules would not plan (check_map) also waits; refused actions are dropped.
    lab, (asked, approver) = NoteLab(), recorder(False)
    approved = Human("approve_big", approver).review(lab, "map_agent", [("check_map", {}), ("wait", {"seconds": 60})],
                                                     "AI", rules_actions=[("wait", {"seconds": 60})])
    assert asked == ["check_map"] and approved == [("wait", {"seconds": 60})]
    assert [level for _, level, _ in lab.notes] == ["info", "warning"]
    print("approve_big: an AI-only action waits for approval, a refused one is dropped: ok")

    # The team over the first 40 min of the scenario with each mode (rules brain, in-memory database).
    def team(human: Human | None) -> tuple[dict, LabState]:
        state = LabState(":memory:")
        r = loop.run_team(state=state, brain="rules", duration_s=2400, switches={"use_analyst": False}, human=human,
                          run_name="human_test")
        return r, state

    def same(a: dict, b: dict) -> bool:
        """Identical focus brightness, records and summaries."""
        return baseline.summarize(a) == baseline.summarize(b) and np.array_equal(a["eta"], b["eta"]) and \
            [(m["t_s"], m["topic"]) for m in a["messages"]] == [(m["t_s"], m["topic"]) for m in b["messages"]]

    off, _ = team(None)
    off_explicit, _ = team(Human("off"))
    assert same(off, off_explicit)

    # suggest, refuse all: the agents only propose; no measure_map after the first map, no control or target change,
    # no message; every proposal is refused in a note by "human".
    asked, refuse_all = recorder(False)
    refused, state = team(Human("suggest", refuse_all))
    actions = state.query("SELECT actor, action FROM actions WHERE action IN ('measure_map', 'set_control', 'set_target')")
    notes = state.query("SELECT level, text FROM notes WHERE actor = 'human'")
    proposals = [n for n in notes if n["text"].startswith("proposal")]
    refusals = [n for n in notes if n["text"].startswith("refused")]
    print(f"suggest, refuse all: asked {len(asked)} times ({sorted(set(asked))}), lab actions {[dict(a) for a in actions]}, "
          f"messages {len(refused['messages'])}, human notes: {len(proposals)} proposals, {len(refusals)} refusals")
    assert asked and len(actions) == 1 and actions[0]["action"] == "measure_map"     # only the first map, before the loop
    assert refused["measured"].sum() == 0 and not refused["messages"]
    assert len(proposals) == len(refusals) == len(asked)

    # approve_big, approve all: waits, checks and messages go through without asking; the numbers equal "off".
    asked, approve_all = recorder(True)
    big, _ = team(Human("approve_big", approve_all))
    print(f"approve_big, approve all: asked {len(asked)} times ({sorted(set(asked))}), same numbers as off: "
          f"{'yes' if same(big, off) else 'NO'}")
    assert set(asked) <= set(BIG) and same(big, off)
    asked, approve_all = recorder(True)
    suggest_all, _ = team(Human("suggest", approve_all))
    assert same(suggest_all, off) and asked      # (no message in the first 40 min; the review test covers messages)
    print("suggest, approve all: same numbers as off: yes")
    print("human.py self-test passed")
