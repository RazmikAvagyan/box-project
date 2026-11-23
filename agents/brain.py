"""
brain.py - the brains of the Map agent: what to do about the problems found in one loop.

RulesBrain: the simple rules (re-measure only when the map is stale and the box is stable).
AIBrain(provider): asks a language model (Claude, Gemini or OpenAI, plain HTTPS with urllib) for a
JSON plan {"actions": [{"name": ..., "args": {...}}], "reason": "..."}. If the key is missing, the
call fails or times out, or the answer is bad (not JSON, unknown action, wrong args), the rules
decide this loop instead and the agent writes a warning note "brain fallback: <why>".
FakeBrain: an AIBrain without network that answers the rules plan as JSON text (for self-tests).

API keys come from the environment or from the .env file in the project root (env.load_env);
they are never printed or written anywhere.
"""

from __future__ import annotations

import inspect
import json
import os
import urllib.error
import urllib.request

from agents.lab_api import LabAPI
from env import load_env
from config import AGENT_LOOP, BASELINE, DRIFT_TEST, LAB_API, TARGET_C, THERMOSTAT
from config import AI_BRAIN as SETTINGS

ALLOWED = ("measure_map", "check_map", "focus", "set_target", "set_control", "wait")
# The Map agent's AI may only touch the map and the clock; the target belongs to a later Temperature agent.
MAP_AGENT_ALLOWED = ("measure_map", "check_map", "wait")

# What each allowed action does and costs in lab time, for the prompt.
ACTIONS = {
    "measure_map": f"measure_map(): take photos and recover a new map with GGS; {LAB_API['measure_map_s']:g} s",
    "check_map": f"check_map(): prediction correlation of the stored map; {LAB_API['check_map_s']:.1f} s",
    "focus": f"focus(point: int): one focus photo on a camera point; {LAB_API['focus_s']:g} s",
    "set_target": "set_target(T: float): new box target in C; 0 s",
    "set_control": "set_control(on: bool): switch the box PID on or off; 0 s",
    "wait": f"wait(seconds: float): let lab time pass, 0 < seconds <= {SETTINGS['max_wait_s']:g}; the given seconds",
}


def rules_plan(problems: list[str]) -> list[tuple[str, dict]]:
    """The rules: measure a new map only when it is stale and the box is stable."""
    if "map stale" in problems and "box not stable" not in problems:
        return [("measure_map", {})]
    return []


def prompt(obs: dict, problems: list[str]) -> str:
    """The question for the model, built from the loop's observations and problems."""
    status = obs["status"]
    check = obs["check"]
    check_text = f"{check:.3f}" + (" (predicted from the temperature change, not measured)" if obs.get("predicted")
                                   else "") if check is not None else (
        "not checked this loop (box not stable)" + ("" if status["last_check"] is None
                                                     else f"; last value {status['last_check']:.3f}"))
    age = "no map yet" if status["map_age_s"] is None else f"{status['map_age_s'] / 60:.1f}"
    lines = [
        "You are the Map agent of a simulated optics lab. A camera sees light through a scattering sample;",
        "the map (transmission matrix) lets us focus light. The sample changes with temperature, so the map",
        "goes stale. A map measured while the box temperature still moves goes stale again soon.",
        "Goal: keep the map useful (check_map >= threshold) while using little lab time.",
        "",
        "Observations:",
        f"- box stable (|Kalman - target| < {THERMOSTAT['swing_band_C']:g} C for the last "
        f"{AGENT_LOOP['stable_s']:g} s): {'yes' if obs['stable'] else 'no'}",
        f"- box temperature (Kalman) {status['kalman']:.3f} C, target {status['target']:g} C",
        f"- last check_map value: {check_text}",
        f"- stale limit: {obs.get('stale_limit', BASELINE['check_threshold']):.3f} (below = map stale)"
        + ("" if obs.get("fresh_level") is None else
           f"; a fresh map now checks about {obs['fresh_level']:.3f} (median of the last new maps)"),
        *([f"- the last new maps did not help: no re-measure for another {obs['backoff_left_s']:.0f} s"]
          if obs.get("backoff_left_s") else []),
        f"- minutes since the last map: {age}",
        f"- problems found: {', '.join(problems) if problems else 'none'}",
        f"- watchdog alarms active: {', '.join(obs.get('alarms') or []) or 'none'}",
        "",
        "Allowed actions (lab time cost):",
        *[f"- {ACTIONS[name]}" for name in MAP_AGENT_ALLOWED],
        "",
        'Answer ONLY with JSON, no other text: {"actions": [{"name": "...", "args": {}}], "reason": "..."}',
        'An empty "actions" list means do nothing this loop. Keep the reason to one short sentence.',
        f"Limits per answer: at most {SETTINGS['max_actions']} actions, at most {SETTINGS['max_measure_map']} "
        f"measure_map, and all waits together at most {SETTINGS['max_wait_total_s']:g} s.",
    ]
    return "\n".join(lines)


def check_values(name: str, args: dict) -> None:
    """Raise ValueError if an argument value is out of range (wait seconds, focus point)."""
    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if name == "wait" and not (number(args["seconds"]) and 0 < args["seconds"] <= SETTINGS["max_wait_s"]):
        raise ValueError(f"wait seconds {args['seconds']!r} not in (0, {SETTINGS['max_wait_s']:g}]")
    if name == "focus" and not (isinstance(args["point"], int) and not isinstance(args["point"], bool)
                                and 0 <= args["point"] < DRIFT_TEST["M"]):
        raise ValueError(f"focus point {args['point']!r} not in 0..{DRIFT_TEST['M'] - 1}")


def check_limits(plan: list[tuple[str, dict]]) -> None:
    """Raise ValueError "plan too long: ..." if the plan has too many actions, measure_maps or wait seconds."""
    n_measure = sum(name == "measure_map" for name, _ in plan)
    waited = sum(args["seconds"] for name, args in plan if name == "wait")
    if len(plan) > SETTINGS["max_actions"]:
        raise ValueError(f"plan too long: {len(plan)} actions > {SETTINGS['max_actions']}")
    if n_measure > SETTINGS["max_measure_map"]:
        raise ValueError(f"plan too long: {n_measure} measure_map > {SETTINGS['max_measure_map']}")
    if waited > SETTINGS["max_wait_total_s"]:
        raise ValueError(f"plan too long: waits {waited:g} s > {SETTINGS['max_wait_total_s']:g} s")


def parse(text: str, allowed: tuple[str, ...] = MAP_AGENT_ALLOWED) -> tuple[list[tuple[str, dict]], str]:
    """Plan and reason from the model's answer; raises ValueError if it is not a valid plan
    (bad JSON, an action outside `allowed`, wrong args or an out-of-range value)."""
    start, end = text.find("{"), text.rfind("}")    # models sometimes wrap the JSON in ``` fences
    try:
        answer = json.loads(text[start:end + 1] if start >= 0 else text)
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON ({e.msg})") from None
    if not isinstance(answer, dict) or not isinstance(answer.get("actions"), list):
        raise ValueError("answer has no 'actions' list")
    plan = []
    for action in answer["actions"]:
        name, args = action.get("name"), action.get("args") or {}
        if name not in allowed:
            raise ValueError(f"action {name!r} not allowed for this agent")
        try:
            inspect.signature(getattr(LabAPI, name)).bind(None, **args)
        except TypeError as e:
            raise ValueError(f"bad args for {name}: {e}") from None
        check_values(name, args)
        plan.append((name, args))
    check_limits(plan)
    return plan, str(answer.get("reason", "")).strip()


def _post(url: str, headers: dict, body: dict) -> dict:
    """POST JSON and return the JSON answer; HTTP errors keep a short part of the error body."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=SETTINGS["timeout_s"]) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from None


def ask_claude(text: str, key: str, model: str) -> str:
    """One Anthropic Messages API call; returns the answer text."""
    out = _post("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": SETTINGS["max_tokens"], "temperature": SETTINGS["temperature"],
                 "messages": [{"role": "user", "content": text}]})
    return "".join(block["text"] for block in out["content"] if block["type"] == "text")


def ask_gemini(text: str, key: str, model: str) -> str:
    """One Gemini generateContent call; returns the answer text (thought parts left out)."""
    out = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": key},
                {"contents": [{"role": "user", "parts": [{"text": text}]}],
                 "generationConfig": {"temperature": SETTINGS["temperature"],
                                      "maxOutputTokens": SETTINGS["max_tokens"],
                                      "responseMimeType": "application/json"}})
    parts = out["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts if not part.get("thought"))


def ask_openai(text: str, key: str, model: str) -> str:
    """One OpenAI Chat Completions call; returns the answer text."""
    out = _post("https://api.openai.com/v1/chat/completions", {"authorization": f"Bearer {key}"},
                {"model": model, "max_completion_tokens": SETTINGS["max_tokens"],
                 "temperature": SETTINGS["temperature"], "response_format": {"type": "json_object"},
                 "messages": [{"role": "user", "content": text}]})
    return out["choices"][0]["message"]["content"]


CALLERS = {"claude": ask_claude, "gemini": ask_gemini, "openai": ask_openai}


class RulesBrain:
    """The simple rules as a brain."""

    provider = "rules"

    def decide(self, obs: dict, problems: list[str]) -> dict:
        """{"plan": [(name, args)], "note": None, "fallback": None}; the agent writes its usual plan note."""
        return {"plan": rules_plan(problems), "note": None, "fallback": None}


class AIBrain:
    """A language model as the brain; falls back to the rules for one loop on any error."""

    def __init__(self, provider: str, caller=None):
        """provider: "claude", "gemini" or "openai"; caller(text, key, model) -> answer text (default: the real API)."""
        load_env()
        self.provider = provider
        self.caller = caller or CALLERS.get(provider)

    def ask(self, text: str, problems: list[str]) -> str:
        """The model's answer text to the prompt `text`; raises on a missing key or a failed call."""
        key = os.environ.get(SETTINGS["key_env"][self.provider])
        if not key:
            raise RuntimeError(f"no API key ({SETTINGS['key_env'][self.provider]} not set)")
        return self.caller(text, key, SETTINGS["models"][self.provider])

    def decide(self, obs: dict, problems: list[str]) -> dict:
        """{"plan": [(name, args)], "note": "plan (<provider>): ... because ..." or None, "fallback": why or None}."""
        try:
            plan, reason = parse(self.ask(prompt(obs, problems), problems))
        except Exception as e:      # any brain error: the rules decide this loop, the run goes on
            why = str(e) if isinstance(e, (ValueError, RuntimeError)) else f"{type(e).__name__}: {e}"
            return {"plan": rules_plan(problems), "note": None, "fallback": why}
        actions = ", ".join(name + (f"({args})" if args else "") for name, args in plan) or "no action"
        return {"plan": plan, "note": f"plan ({self.provider}): {actions} because {reason[:200]}", "fallback": None}


class FakeBrain(AIBrain):
    """An AI brain without network: answers the rules plan as JSON text, so the parsing path runs."""

    def __init__(self):
        super().__init__("fake")

    def ask(self, text: str, problems: list[str]) -> str:
        """The rules plan as the JSON a model would give."""
        actions = [{"name": name, "args": args} for name, args in rules_plan(problems)]
        return json.dumps({"actions": actions, "reason": ", ".join(problems) or "no problems"})


def make(provider: str = SETTINGS["provider"]):
    """The brain for a provider name: "rules", "fake", or a real one ("claude", "gemini", "openai")."""
    return {"rules": RulesBrain, "fake": FakeBrain}.get(provider, lambda: AIBrain(provider))()


if __name__ == "__main__":

    def obs_for(stable: bool, check: float | None, map_age_s: float) -> dict:
        """Made-up observations as Agent.monitor returns them."""
        status = {"time_s": 1000.0, "kalman": TARGET_C + (0.05 if stable else 1.2), "target": TARGET_C, "control": True,
                  "duty": 500, "map_age_s": map_age_s, "last_check": 0.79, "last_eta": 55.0}
        return {"status": status, "temperatures": [], "stable": stable, "check": check}

    def problems_for(obs: dict) -> list[str]:
        """Same as Agent.diagnose."""
        if not obs["stable"]:
            return ["box not stable"]
        return ["map stale"] if obs["check"] < BASELINE["check_threshold"] else []

    cases = [obs_for(True, 0.87, 300), obs_for(True, 0.75, 900), obs_for(False, None, 600),
             obs_for(True, 0.80, 60), obs_for(False, None, 3000)]
    rules, fake = RulesBrain(), FakeBrain()
    for obs in cases:
        problems = problems_for(obs)
        r, f = rules.decide(obs, problems), fake.decide(obs, problems)
        print(f"{str(problems):20} rules {r['plan']}  fake {f['plan']}  note: {f['note']}")
        assert r["plan"] == f["plan"] and f["fallback"] is None
    print("RulesBrain and FakeBrain give the same plans: ok")
    print("prompt for the stale case:\n  " + prompt(cases[1], problems_for(cases[1])).replace("\n", "\n  "))

    def answer(text):
        return lambda prompt_text, key, model: text

    def time_out(prompt_text, key, model):
        raise TimeoutError("timed out")

    key_name = SETTINGS["key_env"]["claude"]
    real_key = os.environ.pop(key_name, None)       # the real key (if any) is put back below, never shown
    os.environ[key_name] = "test-key"
    obs, problems = cases[1], problems_for(cases[1])
    for label, brain in [("bad JSON", AIBrain("claude", answer("measure the map please"))),
                         ("unknown action", AIBrain("claude", answer('{"actions": [{"name": "open_box"}], "reason": "x"}'))),
                         ("bad args", AIBrain("claude", answer('{"actions": [{"name": "wait", "args": {"s": 1}}], "reason": "x"}'))),
                         ("timeout", AIBrain("claude", time_out)),
                         *[(label, AIBrain("claude", answer(json.dumps({"actions": [{"name": n, "args": a}], "reason": "x"}))))
                           for label, n, a in [("set_control", "set_control", {"on": False}),
                                               ("set_target", "set_target", {"T": 25.0}),
                                               ("focus(99999)", "focus", {"point": 99999}),
                                               ("wait(-50)", "wait", {"seconds": -50}),
                                               ("wait(100000)", "wait", {"seconds": 100000})]]]:
        d = brain.decide(obs, problems)
        print(f"{label:15} -> plan {d['plan']}, brain fallback: {d['fallback']}")
        assert d["plan"] == rules_plan(problems) and d["fallback"]
    no_key = AIBrain("claude", answer("{}"))     # built first: its load_env() could bring back a key from .env
    del os.environ[key_name]
    d = no_key.decide(obs, problems)
    print(f"{'missing key':15} -> plan {d['plan']}, brain fallback: {d['fallback']}")
    assert d["plan"] == rules_plan(problems) and "no API key" in d["fallback"]
    good = AIBrain("claude", answer('```json\n{"actions": [{"name": "measure_map", "args": {}}], "reason": "stale"}\n```'))
    os.environ[key_name] = "test-key"
    d = good.decide(obs, problems)
    assert d["plan"] == [("measure_map", {})] and d["note"] == "plan (claude): measure_map because stale"
    print(f"{'fenced JSON':15} -> {d['note']}")
    d = AIBrain("claude", answer('{"actions": [{"name": "wait", "args": {"seconds": 100}}], "reason": "settle"}')).decide(obs, problems)
    assert d["plan"] == [("wait", {"seconds": 100})] and d["fallback"] is None
    print(f"{'wait(100)':15} -> {d['note']}")
    for point, ok in [(0, True), (DRIFT_TEST["M"] - 1, True), (DRIFT_TEST["M"], False), (99999, False), (1.5, False)]:
        try:
            check_values("focus", {"point": point})
            assert ok
        except ValueError:
            assert not ok
    print("focus point range check (for later agents that may focus): ok")

    def plan_answer(*actions):
        """Model answer text for a list of (name, args)."""
        return answer(json.dumps({"actions": [{"name": n, "args": a} for n, a in actions], "reason": "test"}))

    mm, wait200 = ("measure_map", {}), ("wait", {"seconds": 200})
    for label, actions, ok in [("6x measure_map", [mm] * 6, False), ("2x measure_map", [mm] * 2, False),
                               ("2x wait(200)", [wait200] * 2, False),
                               ("measure+check", [mm, ("check_map", {})], True),
                               ("wait(100)", [("wait", {"seconds": 100})], True)]:
        d = AIBrain("claude", plan_answer(*actions)).decide(obs, problems)
        print(f"{label:15} -> " + (d["note"] if ok else f"plan {d['plan']}, brain fallback: {d['fallback']}"))
        assert (d["fallback"] is None and d["plan"] == actions) if ok else \
            (d["plan"] == rules_plan(problems) and d["fallback"].startswith("plan too long"))
    print("plan limits: ok")
    del os.environ[key_name]
    if real_key is not None:
        os.environ[key_name] = real_key
    print("fallbacks: ok")

    load_env()
    for provider in CALLERS:
        found = bool(os.environ.get(SETTINGS["key_env"][provider]))
        print(f"{provider}: {'key found' if found else 'no API key, skipped'}")
