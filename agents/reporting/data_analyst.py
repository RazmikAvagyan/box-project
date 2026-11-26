"""
data_analyst.py - the data analyst agent: it reads one finished run from the lab database and writes findings.

It reads only the lab database (LabState), never the simulation, so it works the same on the real lab. It writes
only to the findings and notes tables, plus results/analysis/<run name>.json and <run name>.md.

Every analysis is plain code (numpy) and a LangChain tool: temperature, events, lab time, agents, and the
task-specific map staleness (TASK_TOOLS; another task adds its own analysis there; it also fits every staleness form
of staleness.py, reports each form's held-out error and the form chosen in code, and remembers the choice for the
run's setup so the predictor with PREDICTION form "auto" can use it); comparing groups of runs is a
tool too. Each tool returns findings: dicts kind, text (one plain sentence), value, unit, evidence.

The brain is chosen like the Map agent's (AI_BRAIN provider; "fake" for tests; "rules"):
  rules: run every tool that has data and keep the plain findings text.
  AI:    the model gets a short description of the run, calls the tools it wants (also after seeing results, at
         most AGENT_LOOP analyst_max_tool_calls), and writes one sentence per finding, addressed by the finding's id
         (e.g. "temperature-3"). It never writes a number itself: every number in a sentence must match that one
         finding (its value, its evidence, its plain text, also as minutes or percent; checked in code). On a missing key, a timeout,
         a bad answer or a made-up number the rules text is used and a warning note "analyst fallback: <why>" is
         written.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents.map_agent import staleness
from agents.map_agent.predict import MapPredictor
from agents.core.state import LabState
from config import AGENT_LOOP, AI_BRAIN, THERMOSTAT, WATCHDOG
from config import ANALYST as SETTINGS
from env import load_env

NAME = "data_analyst"


def finding(kind: str, text: str, value: float | None, unit: str, evidence) -> dict:
    """One finding; numbers as plain floats so it can be stored as JSON."""
    return {"kind": kind, "text": text, "value": None if value is None else float(value), "unit": unit,
            "evidence": evidence}


def load(state: LabState, run_id: int) -> dict:
    """Everything the analyses need from the lab database for one run (rows only, no simulation)."""
    q = lambda sql: state.query(sql, (run_id,))
    run = state.query("SELECT name, settings FROM runs WHERE id = ?", (run_id,))[0]
    temps = q("SELECT t_s, kalman, target, duty FROM temperatures WHERE run_id = ? ORDER BY t_s, rowid")
    data = {"state": state, "run_id": run_id, "name": run["name"], "settings": json.loads(run["settings"]),
            "maps": q("SELECT t_s, kind, correlation, duration_s FROM maps WHERE run_id = ? ORDER BY t_s, rowid"),
            "focus": q("SELECT t_s, eta FROM focus WHERE run_id = ? ORDER BY t_s, rowid"),
            "actions": q("SELECT t_s, actor, action, args FROM actions WHERE run_id = ? ORDER BY t_s, rowid"),
            "notes": q("SELECT t_s, actor, level, text FROM notes WHERE run_id = ? AND actor != 'data_analyst' "
                       "ORDER BY t_s, rowid"),
            "messages": q("SELECT t_s, sender, receiver, topic FROM messages WHERE run_id = ? ORDER BY t_s, rowid")}
    data["t"] = np.array([r["t_s"] for r in temps])
    data["dev"] = np.array([r["kalman"] - r["target"] for r in temps])
    data["target"] = np.array([r["target"] for r in temps])
    data["duty"] = np.array([r["duty"] for r in temps])
    data["counts"] = {"temperatures": len(temps), **{k: len(data[k]) for k in ("maps", "focus", "actions", "notes", "messages")}}
    return data


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True in mask."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _back_in_band(data: dict, after_s: float) -> float | None:
    """First lab time >= after_s, after the box has been outside the band, when it is inside for stable_s (None if never)."""
    t, inside = data["t"], np.abs(data["dev"]) < THERMOSTAT["swing_band_C"]
    left = np.flatnonzero((t >= after_s) & ~inside)
    if not left.size:
        return None
    for i in np.flatnonzero((t > t[left[0]]) & inside):
        window = (t >= t[i] - AGENT_LOOP["stable_s"]) & (t <= t[i])
        if t[i] - AGENT_LOOP["stable_s"] >= t[left[0]] and inside[window].all():
            return float(t[i])
    return None


def _first_reaction(data: dict, after_s: float) -> dict | None:
    """The first agent note after after_s that reacts (a warning, "not stable", a message or a wake), as a row."""
    for note in data["notes"]:
        text = note["text"]
        if note["t_s"] >= after_s and (note["level"] == "warning" or "not stable" in text or text.startswith(("message to", "wake:"))):
            return note
    return None


def temperature(data: dict) -> list[dict]:
    """Share of time within the thesis limit and the swing band, the largest deviation, periods outside the band."""
    t, dev, band, limit = data["t"], data["dev"], THERMOSTAT["swing_band_C"], THERMOSTAT["limit_C"]
    within = float(np.mean(np.abs(dev) <= limit))
    in_band = float(np.mean(np.abs(dev) < band))
    i = int(np.argmax(np.abs(dev)))
    out = [finding("share within limit", f"The box (Kalman) was within ±{limit:g} °C of the target {100 * within:.0f} % "
                   f"of the time.", within, "share", {"table": "temperatures", "rows": len(t), "limit_C": limit}),
           finding("share within band", f"The box was within the swing band ±{band:g} °C {100 * in_band:.0f} % of the "
                   f"time.", in_band, "share", {"table": "temperatures", "rows": len(t), "band_C": band}),
           finding("largest deviation", f"The largest deviation was {dev[i]:+.2f} °C at {t[i] / 60:.1f} min.",
                   dev[i], "C", {"table": "temperatures", "t_s": float(t[i])})]
    periods = [(t[a], t[b]) for a, b in _runs_of(np.abs(dev) >= band) if t[b] - t[a] + 1 >= SETTINGS["min_period_s"]]
    out.append(finding("periods outside band", f"The box was outside the band {len(periods)} times for at least "
                       f"{SETTINGS['min_period_s']:g} s.", len(periods), "count",
                       {"table": "temperatures", "periods_s": [[float(a), float(b)] for a, b in periods]}))
    for a, b in periods:
        out.append(finding("period outside band", f"Outside the band from {a / 60:.1f} to {b / 60:.1f} min "
                           f"({b - a + 1:.0f} s).", b - a + 1, "s", {"table": "temperatures", "from_s": float(a),
                                                                       "to_s": float(b)}))
    return out


def events(data: dict) -> list[dict]:
    """Cooler failure and target changes as seen in the data, the first agent reaction, the time back in the band."""
    t, dev, duty, out = data["t"], data["dev"], data["duty"], []
    # Cooler failure: the longest run at full power during which the box left the band (in a normal hold the
    # cooler is at full power only briefly and the box stays in the band).
    full = duty >= WATCHDOG["duty_full_share"] * THERMOSTAT["duty_max"]
    runs = [(a, b) for a, b in _runs_of(full) if np.any(np.abs(dev[a:b + 1]) >= THERMOSTAT["swing_band_C"])]
    if runs:
        a, b = max(runs, key=lambda r: t[r[1]] - t[r[0]])
        start = float(t[a])
        out.append(finding("cooler failure seen", f"A cooler failure shows at {start / 60:.1f} min: full power for "
                           f"{t[b] - t[a] + 1:.0f} s while the box left the band.", start, "s",
                           {"table": "temperatures", "full_power_from_s": start, "to_s": float(t[b])}))
        events_at = [("cooler failure", start)]
    else:
        events_at = []
    for i in np.flatnonzero(np.diff(data["target"]) != 0):
        when = float(t[i + 1])
        out.append(finding("target change seen", f"The target changed from {data['target'][i]:g} to "
                           f"{data['target'][i + 1]:g} °C at {when / 60:.1f} min.", when, "s",
                           {"table": "temperatures", "t_s": when}))
        events_at.append(("target change", when))
    for event, when in events_at:
        note = _first_reaction(data, when)
        if note is not None:
            out.append(finding(f"first reaction to {event}", f"The first agent reaction to the {event} came after "
                               f"{note['t_s'] - when:.0f} s ({note['actor']}).", note["t_s"] - when, "s",
                               {"table": "notes", "t_s": note["t_s"], "text": note["text"][:80]}))
        back = _back_in_band(data, when)
        if back is not None:
            out.append(finding(f"back in band after {event}", f"The box was back in the band {back - when:.0f} s after "
                               f"the {event}.", back - when, "s", {"table": "temperatures", "t_s": back}))
    return out


def lab_time(data: dict) -> list[dict]:
    """Lab time per action (each action lasts until the next one starts), and time wasted on maps or checks
    started while the box was outside the band."""
    actions, end = data["actions"], float(data["t"][-1])
    starts = [a["t_s"] for a in actions] + [end]
    spent, wasted = {}, 0.0
    for a, (t0, t1) in zip(actions, zip(starts, starts[1:])):
        spent[a["action"]] = spent.get(a["action"], 0.0) + (t1 - t0)
        if a["action"] in ("measure_map", "check_map"):
            i = min(np.searchsorted(data["t"], t0), len(data["t"]) - 1)
            wasted += (t1 - t0) if abs(data["dev"][i]) >= THERMOSTAT["swing_band_C"] else 0.0
    total = end - starts[0]
    out = [finding("lab time total", f"The run lasted {total:.0f} s of lab time.", total, "s",
                   {"table": "actions", "rows": len(actions), "from_s": starts[0], "to_s": end})]
    for name in ("measure_map", "check_map", "focus", "wait"):
        out.append(finding(f"lab time {name}", f"{name} took {spent.get(name, 0.0):.0f} s of lab time "
                           f"({100 * spent.get(name, 0.0) / total:.1f} %).", spent.get(name, 0.0), "s",
                           {"table": "actions", "action": name, "share": spent.get(name, 0.0) / total}))
    other = total - sum(spent.get(n, 0.0) for n in ("measure_map", "check_map", "focus", "wait"))
    out.append(finding("lab time other", f"Other actions took {other:.0f} s.", other, "s", {"table": "actions"}))
    out.append(finding("lab time wasted", f"{wasted:.0f} s went to maps or checks started while the box was outside "
                       "the band.", wasted, "s", {"table": "actions + temperatures"}))
    return out


def agents(data: dict) -> list[dict]:
    """Alarms, messages, refusals, warnings and AI fallbacks, counted from the notes and messages."""
    notes = data["notes"]
    count = lambda test: sum(1 for n in notes if test(n))
    numbers = [("alarms", "watchdog alarms", count(lambda n: n["text"].startswith("watchdog:"))),
               ("messages", "messages between agents", len(data["messages"])),
               ("refusals", "refused actions", count(lambda n: "refused" in n["text"])),
               ("warnings", "warning notes", count(lambda n: n["level"] == "warning")),
               ("ai fallbacks", "AI brain fallbacks", count(lambda n: "brain fallback" in n["text"]))]
    return [finding(kind, f"There were {n} {label}.", n, "count", {"table": "notes" if kind != "messages" else "messages"})
            for kind, label, n in numbers]


def _fit(dT: np.ndarray, check: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares ln(check) = ln(c0) - a * x; returns c0, its standard error, a, its standard error."""
    design = np.column_stack([np.ones_like(x), -x])
    y = np.log(np.clip(check, 1e-3, 1.0))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    cov = np.linalg.inv(design.T @ design) * (resid @ resid) / max(len(y) - 2, 1)
    c0 = float(np.exp(coef[0]))
    return c0, c0 * float(np.sqrt(cov[0, 0])), float(coef[1]), float(np.sqrt(cov[1, 1]))


def map_staleness(data: dict) -> list[dict]:
    """Task-specific (Map agent): how fast the map goes stale with the temperature change since the last map, and
    the focus brightness before and after each new map."""
    dT, check = MapPredictor(data["state"], data["run_id"]).pairs()
    out = []
    if len(dT) >= 3:
        evidence = {"table": "maps + temperatures", "pairs": len(dT), "dT_max_C": float(dT.max())}
        c0, c0_se, a, a_se = _fit(dT, check, dT ** 2)
        if a > 0:
            T_c, T_c_se = 1 / np.sqrt(a), 0.5 * a_se / a ** 1.5
            out.append(finding("map T_c gaussian", f"Fit c0·exp(-(dT/T_c)²): T_c {T_c:.2f} ± {T_c_se:.2f} °C, c0 "
                               f"{c0:.3f} ± {c0_se:.3f} ({len(dT)} checks).", T_c, "C",
                               {**evidence, "T_c_se": T_c_se, "c0": c0, "c0_se": c0_se}))
        # The camera check is an intensity correlation, about the square of the field correlation exp(-dT/T_c).
        c0, c0_se, a, a_se = _fit(dT, check, dT)
        if a > 0:
            T_c, T_c_se = 2 / a, 2 * a_se / a ** 2
            out.append(finding("map T_c exponential", f"Fit c0·exp(-2·dT/T_c): T_c {T_c:.2f} ± {T_c_se:.2f} °C, c0 "
                               f"{c0:.3f} ± {c0_se:.3f} ({len(dT)} checks).", T_c, "C",
                               {**evidence, "T_c_se": T_c_se, "c0": c0, "c0_se": c0_se}))
    out += staleness_forms(data)
    window, gains = SETTINGS["focus_window_s"], []
    focus_t = np.array([f["t_s"] for f in data["focus"]])
    eta = np.array([f["eta"] for f in data["focus"]])
    for m in data["maps"]:
        if m["kind"] != "measure" or m["t_s"] == data["maps"][0]["t_s"]:
            continue
        end = m["t_s"] + m["duration_s"]
        before = eta[(focus_t >= m["t_s"] - window) & (focus_t < m["t_s"])]
        after = eta[(focus_t >= end) & (focus_t < end + window)]
        if before.size and after.size:
            gains.append(after.mean() - before.mean())
            out.append(finding("focus around new map", f"New map at {m['t_s'] / 60:.1f} min: mean eta "
                               f"{before.mean():.1f} before, {after.mean():.1f} after.", after.mean() - before.mean(),
                               "eta", {"table": "maps + focus", "t_s": m["t_s"], "before": float(before.mean()),
                                       "after": float(after.mean())}))
    if gains:
        out.append(finding("focus gain per new map", f"A new map changed the mean eta by {np.mean(gains):+.1f} on "
                           f"average ({len(gains)} maps).", np.mean(gains), "eta", {"table": "maps + focus",
                                                                                     "maps": len(gains)}))
    return out


def staleness_forms(data: dict) -> list[dict]:
    """Every candidate staleness form (staleness.py) with its leave-one-out error, score and parameters, and the form
    chosen in code (lowest score); nothing with fewer than PREDICTION min_points checks."""
    dT, dt, check = staleness.pairs(data["state"], data["run_id"])
    choice = staleness.choose(dT, dt, check)
    if choice is None:
        return []
    finite = lambda x: None if x is None or not np.isfinite(x) else float(x)
    out = []
    for name, f in choice["forms"].items():
        r = f["readable"]
        extra = f", beta {r['beta']:.2f}" if staleness.FORMS[name]["beta"] is None else ""
        extra += "" if r["tau_s"] is None else (", time term none" if finite(r["tau_s"]) is None
                                                else f", tau {r['tau_s']:.0f} s")
        out.append(finding(f"staleness fit {name}", f"Form {name}: leave-one-out error {f['loo_rmse']:.4f}, score "
                           f"{f['score']:.1f} ({f['k']} parameters), c0 {r['c0']:.3f}{extra} ({choice['n']} checks).",
                           f["loo_rmse"], "check", {"table": "maps + temperatures", "checks": choice["n"],
                                                     "score": f["score"], "parameters": f["k"], "c0": r["c0"],
                                                     "T_c_C": finite(r["T_c_C"]), "beta": r["beta"],
                                                     "tau_s": finite(r["tau_s"])}))
    best = choice["forms"][choice["chosen"]]
    out.append(finding("staleness form chosen", f"Chosen staleness form: {choice['chosen']} (lowest score "
                       f"{best['score']:.1f} of {len(choice['forms'])} forms; leave-one-out error {best['loo_rmse']:.4f}).",
                       best["loo_rmse"], "check", {"table": "maps + temperatures", "form": choice["form"],
                                                   "checks": choice["n"], "score": best["score"],
                                                   "loo_rmse": {n: f["loo_rmse"] for n, f in choice["forms"].items()}}))
    return out


# name -> (analysis, description, has data?). TASK_TOOLS holds the task-specific analyses.
GENERAL_TOOLS = {
    "temperature": (temperature, "Share of time within ±0.03 °C and the swing band, the largest deviation and every "
                    "period outside the band (temperature rows).", lambda d: d["counts"]["temperatures"] > 0),
    "events": (events, "Cooler failure and target changes as seen in the data, the first agent reaction and the time "
               "until the box was back in the band.", lambda d: d["counts"]["temperatures"] > 0),
    "lab_time": (lab_time, "Lab time on measure_map, check_map, focus, waiting and other actions, and the time wasted "
                 "on maps or checks while the box was outside the band.", lambda d: d["counts"]["actions"] > 0),
    "agents": (agents, "Counts of watchdog alarms, messages, refusals, warnings and AI fallbacks.",
               lambda d: d["counts"]["notes"] + d["counts"]["messages"] > 0),
}
TASK_TOOLS = {
    "map_staleness": (map_staleness, "How fast the map goes stale: T_c and c0 fitted to the map checks against the "
                      "temperature change since the last map, and the focus brightness before and after each new map.",
                      lambda d: any(m["kind"] == "check" for m in d["maps"])),
}


def with_ids(tool: str, findings: list[dict]) -> list[dict]:
    """The findings of one tool, each with a unique id "<tool>-<index>" (the AI answers by id)."""
    return [{"id": f"{tool}-{i}", **f} for i, f in enumerate(findings)]


def make_tools(data: dict) -> dict[str, StructuredTool]:
    """The analyses of one run as LangChain tools (no arguments; each returns its findings with ids as JSON)."""
    return {name: StructuredTool.from_function(func=lambda f=f, name=name: json.dumps(with_ids(name, f(data))),
                                               name=name, description=description)
            for name, (f, description, _) in {**GENERAL_TOOLS, **TASK_TOOLS}.items()}


def describe(data: dict) -> str:
    """The short description of the run the AI gets."""
    s = data["settings"]
    t_end = float(data["t"][-1]) if len(data["t"]) else 0.0
    rows = ", ".join(f"{table} {n}" for table, n in data["counts"].items())
    return (f"Run '{data['name']}', {t_end / 60:.0f} min of lab time. Settings: N {s.get('N')}, M {s.get('M')}, "
            f"box target {s.get('T0')} °C. Rows in the lab database: {rows}.")


def numbers_in(value) -> list[float]:
    """Every number in a tool result (values and evidence), also as minutes (s / 60) and percent (share * 100)."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value), float(value) / 60, float(value) * 100]
    if isinstance(value, dict):
        return [x for v in value.values() for x in numbers_in(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in numbers_in(v)]
    if isinstance(value, str):
        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return []


def made_up_numbers(text: str, one: dict) -> list[str]:
    """Numbers in text that match nothing of this one finding (its value, evidence and plain text; also as minutes or
    percent). A number matches if it rounds from one: 41.47 -> "41.5"."""
    known = numbers_in([one["value"], one["evidence"], one["text"]])
    bad = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        n, decimals = float(token), len(token.split(".")[1]) if "." in token else 0
        if not any(abs(n - k) <= 0.5 * 10 ** -decimals + 1e-9 or abs(abs(n) - abs(k)) <= 0.5 * 10 ** -decimals + 1e-9
                   for k in known):
            bad.append(token)
    return bad


SYSTEM = ("You are the data analyst of a lab. Call the analysis tools you need (you may call more after seeing "
          "results), then answer ONLY with JSON: {\"findings\": [{\"id\": \"<id of a finding from a tool result>\", "
          "\"text\": \"<one plain sentence about that finding>\"}]}, each id at most once. A sentence may only use "
          "numbers of its own finding; findings you leave out keep their plain text.")


class FakeAnalystModel(GenericFakeChatModel):
    """A chat model without network for tests: it plays the given replies in order; bind_tools returns itself."""

    def bind_tools(self, tools, **kwargs):
        return self


def _rules_replies(data: dict) -> list[AIMessage]:
    """The fake brain's default replies: call every tool with data, then the plain findings text as the answer."""
    names = [n for n, (_, _, has) in {**GENERAL_TOOLS, **TASK_TOOLS}.items() if has(data)]
    calls = [{"name": n, "args": {}, "id": f"call_{i}"} for i, n in enumerate(names)]
    with_id = [f for n in names for f in with_ids(n, {**GENERAL_TOOLS, **TASK_TOOLS}[n][0](data))]
    answer = {"findings": [{"id": f["id"], "text": f["text"]} for f in with_id]}
    return [AIMessage(content="", tool_calls=calls), AIMessage(content=json.dumps(answer))]


def chat_model(provider: str, data: dict, replies: list[AIMessage] | None = None):
    """The chat model for the analyst: a fake one (replies, or the default rules-like replies) or a real provider."""
    if provider == "fake":
        return FakeAnalystModel(messages=iter(replies or _rules_replies(data)))
    from agents.core.brain import chat_model as provider_model
    load_env()
    key = os.environ.get(AI_BRAIN["key_env"][provider])
    if not key:
        raise RuntimeError(f"no API key ({AI_BRAIN['key_env'][provider]} not set)")
    return provider_model(provider, key, SETTINGS["max_tokens"], SETTINGS["timeout_s"])   # the analyst's own limits


def rules_findings(data: dict) -> tuple[list[dict], list[str]]:
    """Rules brain: every tool that has data, plain text. Returns findings and the tools used."""
    names = [n for n, (_, _, has) in {**GENERAL_TOOLS, **TASK_TOOLS}.items() if has(data)]
    return [f for n in names for f in {**GENERAL_TOOLS, **TASK_TOOLS}[n][0](data)], names


def ai_findings(model, data: dict) -> tuple[list[dict], list[str]]:
    """The AI chooses tools and writes the text. Raises ValueError on a bad answer or a made-up number."""
    tools = make_tools(data)
    messages = [SystemMessage(SYSTEM), HumanMessage(describe(data))]
    bound = model.bind_tools(list(tools.values()))
    results, called = {}, []
    while True:
        reply = bound.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            if call["name"] not in tools:
                raise ValueError(f"unknown tool {call['name']!r}")
            if len(called) >= AGENT_LOOP["analyst_max_tool_calls"]:
                raise ValueError(f"more than {AGENT_LOOP['analyst_max_tool_calls']} tool calls")
            output = tools[call["name"]].invoke({})
            results[call["name"]] = json.loads(output)
            called.append(call["name"])
            messages.append(ToolMessage(output, tool_call_id=call["id"]))
    text = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
    start, end = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[start:end + 1])["findings"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("bad JSON answer") from None
    found = {f["id"]: f for name in dict.fromkeys(called) for f in results[name]}
    texts = {}
    for item in answer:
        fid = item.get("id") if isinstance(item, dict) else None
        if fid not in found:
            raise ValueError(f"unknown finding id {fid!r}")
        if fid in texts:
            raise ValueError(f"finding id {fid!r} answered twice")
        bad = made_up_numbers(str(item.get("text", "")), found[fid])
        if bad:
            raise ValueError(f"made-up number(s) {', '.join(bad)} for {fid}: {str(item['text'])[:80]}")
        texts[fid] = item["text"]
    # The AI's sentence replaces only the finding with that id; the others keep their plain text.
    return [{**{k: v for k, v in f.items() if k != "id"}, "text": texts.get(fid, f["text"])} for fid, f in found.items()], called


def analyze_run(state: LabState, run_id: int, brain: str = AI_BRAIN["provider"],
                replies: list[AIMessage] | None = None) -> list[dict]:
    """The findings of one run (brain: "rules", "fake" or a provider; replies: scripted fake replies for tests).
    Writes notes only (which tools, and any fallback); save() stores the findings."""
    data = load(state, run_id)
    t_end = float(data["t"][-1]) if len(data["t"]) else 0.0
    note = lambda text, level="info": state.add_note(run_id, t_end, NAME, level, text)
    if brain == "rules":
        findings, called = rules_findings(data)
    else:
        try:
            findings, called = ai_findings(chat_model(brain, data, replies), data)
        except Exception as e:      # any brain error: the rules text, the run goes on
            note(f"analyst fallback: {e}", "warning")
            findings, called = rules_findings(data)
            brain = "rules"
    note(f"analyst ({brain}): {len(findings)} findings from tools {', '.join(called)}")
    state.commit()
    return findings


def save(state: LabState, run_id: int, findings: list[dict], out_dir: str | Path,
         run_info: dict | None = None) -> list[Path]:
    """Store the findings in the findings table and as <out_dir>/<run name>.json ("run", "run_info", "findings") and
    .md (a short header from run_info, then one line per finding). run_info comes from the caller (script, operator,
    brain, PID, seed, map rule, time drift, lab time, mean eta); the run name and the time written are added here."""
    name = state.query("SELECT name FROM runs WHERE id = ?", (run_id,))[0]["name"]
    state.db.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
    for f in findings:
        state.add_finding(run_id, f)
    state.commit()
    info = {"run name": name, **(run_info or {}), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"run": name, "run_info": info, "findings": findings}, indent=2))
    header = "\n".join(f"- {key}: {value:.1f}" if isinstance(value, float) else f"- {key}: {value}"
                       for key, value in info.items())
    (out / f"{name}.md").write_text(f"# Findings: {name}\n\n{header}\n\n## Findings\n\n"
                                    + "\n".join(f"- {f['text']}" for f in findings) + "\n")
    return [out / f"{name}.json", out / f"{name}.md"]


def analyze_and_save(state: LabState, run_id: int, brain: str = AI_BRAIN["provider"],
                     out_dir: str | Path = SETTINGS["out_dir"], run_info: dict | None = None) -> list[dict]:
    """analyze_run, then save (into out_dir, given by the caller); what loop.run and loop.run_team call at the end."""
    findings = analyze_run(state, run_id, brain)
    save(state, run_id, findings, out_dir, run_info)
    # The chosen staleness form (made in code, whatever the brain wrote) is remembered for the run's setup.
    chosen = [f for f in findings if f["kind"] == "staleness form chosen"]
    if chosen and run_info and "PID" in run_info and "time drift" in run_info:
        e = chosen[0]["evidence"]
        name = state.query("SELECT name FROM runs WHERE id = ?", (run_id,))[0]["name"]
        staleness.save_choice(staleness.setup_key(run_info["PID"], run_info["time drift"]), e["form"], e["checks"],
                              e["loo_rmse"], name)
    return findings


def write_index(out_dir: str | Path) -> Path:
    """<out_dir>/index.md: one line per saved report (name, operator, PID, mean eta, number of findings), overwritten."""
    out = Path(out_dir)
    rows = []
    for path in sorted(out.glob("*.json")):
        report = json.loads(path.read_text())
        info = report.get("run_info", {})
        eta = info.get("mean eta")
        rows.append(f"| {report['run']} | {info.get('operator', '')} | {info.get('PID', '')} | "
                    f"{'' if eta is None else f'{eta:.1f}'} | {len(report['findings'])} |")
    text = (f"# Analyst reports in {out}\n\n| report | operator | PID | mean η | findings |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")
    (out / "index.md").write_text(text)
    return out / "index.md"


def compare_groups(state: LabState, group_a: list[int], group_b: list[int] | None = None) -> list[dict]:
    """Mean ± standard error over runs of every finding kind that has one value per run (from the stored findings),
    and, with a second group, every kind whose means differ by more than se_factor standard errors."""
    def table(ids):
        values = {}
        for run_id in ids:
            for f in state.findings(run_id):
                values.setdefault(f["kind"], {}).setdefault(run_id, []).append(f["value"])
        return {k: np.array([v[0] for v in per.values()]) for k, per in values.items()
                if len(per) == len(ids) and all(len(v) == 1 and v[0] is not None for v in per.values())}

    a, out = table(group_a), []
    stats = lambda x: (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))
    for kind, x in a.items():
        mean, se = stats(x)
        out.append(finding(f"mean {kind}", f"{kind}: {mean:.3g} ± {se:.2g} over {len(x)} runs.", mean, "",
                           {"runs": list(group_a), "se": se}))
    if group_b:
        b = table(group_b)
        for kind in sorted(set(a) & set(b)):
            (ma, sa), (mb, sb) = stats(a[kind]), stats(b[kind])
            se = float(np.hypot(sa, sb))
            if abs(ma - mb) > SETTINGS["se_factor"] * se:
                out.append(finding(f"difference {kind}", f"{kind} differs: {ma:.3g} vs {mb:.3g} (difference "
                                   f"{ma - mb:+.3g}, more than {SETTINGS['se_factor']:g} standard errors {se:.2g}).",
                                   ma - mb, "", {"runs_a": list(group_a), "runs_b": list(group_b), "se": se}))
    return out


def analyze_runs(state: LabState, run_ids: list[int], other_ids: list[int] | None = None,
                 brain: str = AI_BRAIN["provider"], replies: list[AIMessage] | None = None) -> list[dict]:
    """Compare several runs (findings must be stored); the comparison is a tool too, so an AI brain can write the
    text (numbers checked as for one run); on any error the rules text is used."""
    findings = compare_groups(state, run_ids, other_ids)
    if brain == "rules":
        return findings
    with_id = with_ids("compare_runs", findings)
    tool = StructuredTool.from_function(func=lambda: json.dumps(with_id), name="compare_runs",
                                        description="Mean ± standard error of the main numbers over the runs, and "
                                                    "every difference beyond 2 standard errors between two groups.")
    try:
        if brain == "fake":
            answer = {"findings": [{"id": f["id"], "text": f["text"]} for f in with_id]}
            model = FakeAnalystModel(messages=iter(replies or [
                AIMessage(content="", tool_calls=[{"name": "compare_runs", "args": {}, "id": "c"}]),
                AIMessage(content=json.dumps(answer))]))
        else:
            model = chat_model(brain, {})
        bound = model.bind_tools([tool])
        messages = [SystemMessage(SYSTEM), HumanMessage(f"Compare {len(run_ids)} runs with {len(other_ids or [])} others.")]
        reply = bound.invoke(messages)
        while reply.tool_calls:
            messages += [reply] + [ToolMessage(tool.invoke({}), tool_call_id=c["id"]) for c in reply.tool_calls]
            reply = bound.invoke(messages)
        text = reply.content
        items = json.loads(text[text.find("{"):text.rfind("}") + 1])["findings"]
        by_id, texts = {f["id"]: f for f in with_id}, {}
        for item in items:
            if item["id"] not in by_id or item["id"] in texts or made_up_numbers(item["text"], by_id[item["id"]]):
                raise ValueError(f"bad answer for {item['id']!r}")
            texts[item["id"]] = item["text"]
        return [{**f, "text": texts.get(fid, f["text"])} for fid, f in zip(by_id, findings)]
    except Exception:
        return findings


if __name__ == "__main__":
    import tempfile

    from agents.map_agent import loop
    from config import DRIFT_TEST, PID_TUNING, SCENARIO

    state = LabState(":memory:")
    tuned = dict(zip(("Kp", "Ki", "Kd"), PID_TUNING["refined"]))
    off = {"use_analyst": False}
    runs = {"PID 200/4": loop.run_team(state=state, brain="rules", switches=off, run_name="team_pid_200_4"),
            "PID 3000/0.36": loop.run_team(state=state, brain="rules", switches=off, box_settings=tuned,
                                           run_name="team_pid_3000_0.36")}
    run_id = runs["PID 200/4"]["api"].run_id

    # The analyst writes only findings and notes.
    tables = ["runs", "temperatures", "maps", "focus", "actions", "notes", "messages", "findings"]
    count = lambda: {t: state.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in tables}
    before = count()
    with tempfile.TemporaryDirectory() as tmp:
        all_findings = {pid: analyze_and_save(state, r["api"].run_id, "rules", tmp) for pid, r in runs.items()}
        after = count()
        changed = {t for t in tables if before[t] != after[t]}
        print(f"tables written by the analyst: {sorted(changed)}")
        assert changed <= {"findings", "notes"}

        # Saved and read back (database and JSON).
        stored = state.findings(run_id)
        saved = json.loads((Path(tmp) / "team_pid_200_4.json").read_text())["findings"]
        assert [f["kind"] for f in stored] == [f["kind"] for f in all_findings["PID 200/4"]] == [f["kind"] for f in saved]
        md = (Path(tmp) / "team_pid_200_4.md").read_text()
        assert md.split("## Findings")[1].count("\n- ") == len(saved) and "- run name: team_pid_200_4" in md
        assert json.loads((Path(tmp) / "team_pid_200_4.json").read_text())["run_info"]["run name"] == "team_pid_200_4"
        index = write_index(tmp).read_text()
        assert index.count("\n| team_pid_") == 2
        print(f"findings saved and read back: {len(stored)} rows, JSON and .md with header; index.md with 2 reports")

    for pid, findings in all_findings.items():
        print(f"\nfindings, team seed 0, {pid}:")
        for f in findings:
            print(f"  [{f['kind']}] {f['text']}")

    by_kind = {f["kind"]: f for f in all_findings["PID 200/4"]}
    T_true = DRIFT_TEST["T_c"]
    T_exp, T_gauss = by_kind["map T_c exponential"]["value"], by_kind["map T_c gaussian"]["value"]
    print(f"\nT_c: exponential fit {T_exp:.2f} °C, gaussian fit {T_gauss:.2f} °C, lab model {T_true:g} °C "
          f"(exponential within {100 * SETTINGS['test_tc_rel_tol']:.0f} % required)")
    assert abs(T_exp - T_true) <= SETTINGS["test_tc_rel_tol"] * T_true
    tol = SETTINGS["test_event_tol_s"]
    cooler, target = by_kind["cooler failure seen"]["value"], by_kind["target change seen"]["value"]
    print(f"cooler failure seen at {cooler / 60:.1f} min (scenario {SCENARIO['cooler_fail_at_s'] / 60:g} min), target "
          f"change at {target / 60:.1f} min (scenario {SCENARIO['target_change_at_s'] / 60:g} min)")
    assert abs(cooler - SCENARIO["cooler_fail_at_s"]) <= tol and abs(target - SCENARIO["target_change_at_s"]) <= tol
    parts = sum(by_kind[f"lab time {n}"]["value"] for n in ("measure_map", "check_map", "focus", "wait", "other"))
    print(f"lab time parts {parts:.0f} s = total {by_kind['lab time total']['value']:.0f} s; "
          f"other {by_kind['lab time other']['value']:.0f} s")
    assert abs(parts - by_kind["lab time total"]["value"]) < 1e-6 and abs(by_kind["lab time other"]["value"]) < 1e-6

    # The rules text passes the number check: each sentence only uses numbers of its own finding.
    rules, _ = rules_findings(load(state, run_id))
    assert not [f for f in rules if made_up_numbers(f["text"], f)]

    # Fake brain (answers by finding id): only some tools, a follow-up call, same-kind findings, made-up numbers.
    data = load(state, run_id)
    call = lambda *names: AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": f"{n}_{i}"}
                                                            for i, n in enumerate(names)])
    answer = lambda items: AIMessage(content=json.dumps({"findings": items}))
    temp = {f["id"]: f for f in with_ids("temperature", temperature(data))}
    band = temp["temperature-1"]
    good_text = f"The box stayed within the band {100 * band['value']:.0f} % of the time."
    findings = analyze_run(state, run_id, "fake", [call("temperature"), answer([{"id": "temperature-1", "text": good_text}])])
    assert [f["text"] for f in findings] == [good_text if fid == "temperature-1" else f["text"] for fid, f in temp.items()]
    print(f"\nfake brain, only the temperature tool: {len(findings)} findings, AI text kept for temperature-1: {good_text!r}")
    findings = analyze_run(state, run_id, "fake", [call("events"), call("lab_time"), answer([])])
    assert {f["kind"] for f in findings} >= {"cooler failure seen", "lab time total"} and \
        not any(f["kind"].startswith("share") for f in findings)
    print("fake brain, follow-up call (events, then lab_time): both used, nothing else")

    # Two findings of the same kind get two different sentences, each on its own finding.
    periods = [fid for fid, f in temp.items() if f["kind"] == "period outside band"][:2]
    sentences = {fid: f"The box left the band for {temp[fid]['value']:.0f} s starting at "
                      f"{temp[fid]['evidence']['from_s'] / 60:.1f} min." for fid in periods}
    findings = analyze_run(state, run_id, "fake", [call("temperature"),
                                                   answer([{"id": fid, "text": t} for fid, t in sentences.items()])])
    got = [f["text"] for f in findings if f["kind"] == "period outside band"]
    assert got[:2] == list(sentences.values()) and got[0] != got[1]
    print(f"fake brain, two findings of one kind: {got[:2]}")

    warnings = lambda: state.query("SELECT COUNT(*) AS n FROM notes WHERE run_id = ? AND actor = ? AND level = 'warning'",
                                   (run_id, NAME))[0]["n"]
    plain = [f["text"] for f in rules]
    limit_pct = f"{100 * temp['temperature-0']['value']:.0f}"
    agent_counts = {f["id"]: f for f in with_ids("agents", agents(data))}
    n_messages = int(agent_counts["agents-1"]["value"])
    wrong = n_messages + 1
    cases = [("made-up number 4321.7", "temperature", "temperature-1", "The box was in the band 4321.7 % of the time."),
             (f"number {limit_pct} % from another finding (share within limit)", "temperature", "temperature-1",
              f"The box was in the band {limit_pct} % of the time."),
             (f"wrong whole number ({wrong} messages, there were {n_messages})", "agents", "agents-1",
              f"There were {wrong} messages between agents.")]
    for label, tool, fid, text in cases:
        w0 = warnings()
        findings = analyze_run(state, run_id, "fake", [call(tool), answer([{"id": fid, "text": text}])])
        assert warnings() == w0 + 1 and [f["text"] for f in findings] == plain
        print(f"fake brain, {label}: caught, rules text used, warning note written")
    for label, replies in [("bad JSON", [call("temperature"), AIMessage(content="no json")]),
                           ("unknown tool", [call("open_box")]),
                           ("unknown finding id", [call("temperature"), answer([{"id": "temperature-99", "text": "x"}])]),
                           ("repeated finding id", [call("temperature"), answer([{"id": "temperature-1", "text": good_text}] * 2)]),
                           ("too many tool calls", [call(*["agents"] * (AGENT_LOOP["analyst_max_tool_calls"] + 1))])]:
        findings = analyze_run(state, run_id, "fake", replies)
        assert [f["text"] for f in findings] == plain
        print(f"fake brain, {label}: fell back to the rules")
    key = AI_BRAIN["key_env"]["claude"]
    if not os.environ.get(key):
        findings = analyze_run(state, run_id, "claude")
        assert len(findings) == len(rules)
        print("claude without an API key: fell back to the rules")
    default = analyze_run(state, run_id, "fake")
    assert [f["text"] for f in default] == plain
    print(f"default fake brain (calls every tool, answers with the rules text by id): {len(default)} findings")

    # Comparing runs: made-up stored findings, three runs per group, one clear difference.
    cmp = LabState(":memory:")
    ids = [cmp.start_run(f"r{i}", i, {}) for i in range(6)]
    for i, rid in enumerate(ids):
        cmp.add_finding(rid, finding("share within limit", "x", 0.07 + 0.01 * (i % 3) if i < 3 else 0.69 + 0.01 * (i % 3),
                                     "share", {}))
        cmp.add_finding(rid, finding("lab time wasted", "x", 20.0 + 5 * (i % 3), "s", {}))   # same in both groups
    result = analyze_runs(cmp, ids[:3], ids[3:], brain="rules")
    diffs = [f["kind"] for f in result if f["kind"].startswith("difference")]
    print(f"compare groups: {[f['text'] for f in result]}")
    assert diffs == ["difference share within limit"]
    assert [f["text"] for f in analyze_runs(cmp, ids[:3], ids[3:], brain="fake")] == [f["text"] for f in result]
    print("data_analyst.py self-test passed")
