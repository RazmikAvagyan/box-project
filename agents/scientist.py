"""
scientist.py - the Scientist agent: reads all results, picks open questions, runs a few small extra experiments within
a fixed budget, and writes one report in which every claim is checked against computed numbers.

A LangGraph graph: read -> questions -> plan -> run -> check -> write.
  read       code only: the analyst reports (results/analysis/*/*.json), the comparisons (results/compare/*.csv),
             and the summaries of the drift test, main, the real-box fit and the thermostat, plus a few settings.
             Every number gets a source id (file + key).
  questions  the brain picks up to max_questions open questions (rules brain: a fixed list built from the data).
  plan       the brain proposes experiments; code accepts only whitelisted kinds ("compare": operators, PID, time
             drift tau, seeds; "drift_test": long test with a time drift tau), refuses anything else (logged) and cuts
             what does not fit the budget (max_experiments, max_seeds, max_compute_min).
  run        code runs the accepted experiments and computes their numbers ("beats"/"differs" only beyond 2 standard
             errors of the paired seed differences).
  check      every sentence must cite source ids, and every number in it must match a cited source within rounding
             (also as minutes or percent); failing sentences are dropped and logged, never rewritten.
  write      results/report/report.md and report.json.
The brain is chosen like the analyst's (AI_BRAIN provider, "fake" for tests, "rules"); code computes every number.

python -m agents.scientist          the real run (rules brain unless an API key is set), writes results/report/
python -m agents.scientist --test   self-test with a fake brain (scripted replies), output in a temporary folder
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from config import AI_BRAIN, COMPARE, DRIFT_TEST, SCENARIO, THERMOSTAT
from config import SCIENTIST as SETTINGS

ROOT = Path(".")
TAU_FILES = {None: "results.csv", 120: "results_time_drift_120min.csv", 30: "results_time_drift_30min.csv"}
OPERATORS = ["never", "rules", "basic", "full", "team"]
NAMES = {"never": "never", "rules": "fixed rules", "basic": "basic Map agent", "full": "full Map agent", "team": "team"}
SECTIONS = ["What was simulated", "Thermostat", "Optics and map staleness", "Operators compared",
            "What the agents add, and where they do not", "New experiments and their answers",
            "What must be measured on the real setup", "Limitations"]


def flatten(value, prefix: str) -> dict[str, float]:
    """Numeric leaves of a JSON value as {"<prefix>:<key.path>": number}."""
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if not prefix.endswith(":") else f"{prefix}{k}"))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
        out[prefix] = float(value)
    return out


def tau_text(tau) -> str:
    """"no time drift" or "time drift 30 min"."""
    return "no time drift" if tau is None else f"time drift {tau:g} min"


def stats(values: np.ndarray) -> tuple[float, float, int]:
    """Mean, standard error of the mean and n of the finite values."""
    v = values[np.isfinite(values)]
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")), len(v)


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int]:
    """Mean and standard error of a - b over the seeds both have."""
    d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))])
    return float(d.mean()), (float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")), len(d)


def verdict(mean: float, se: float) -> str:
    """"higher", "lower" (beyond se_factor standard errors) or "no clear difference"."""
    if se == 0.0 and mean == 0.0:
        return "identical"
    if np.isfinite(se) and abs(mean) > COMPARE["se_factor"] * se:
        return "higher" if mean > 0 else "lower"
    return "no clear difference"


def read_compare(path: Path) -> dict:
    """{(pid, operator): {metric: {seed: value}}} from a results<tag>.csv."""
    table = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            per = table.setdefault((row["pid"], row["operator"]), {})
            for key in ("mean_eta", "lab_time_on_maps_s", "first_reaction_s"):
                per.setdefault(key, {})[int(row["seed"])] = float(row[key])
    return table


def compare_sources(table: dict, name: str) -> dict[str, float]:
    """Source numbers of one comparison: mean, SE, n per (pid, operator, metric) and paired differences."""
    out = {}
    for (pid, op), metrics in table.items():
        for key, per_seed in metrics.items():
            m, se, n = stats(np.array(list(per_seed.values())))
            out.update({f"{name}:{pid}|{op}|{key}.mean": m, f"{name}:{pid}|{op}|{key}.se": se,
                        f"{name}:{pid}|{op}|{key}.n": n})
    for pid in {p for p, _ in table}:
        for a, b in [("full", "rules"), ("basic", "rules"), ("team", "full"), ("rules", "never")]:
            if (pid, a) in table and (pid, b) in table:
                m, se, n = paired(table[(pid, a)]["mean_eta"], table[(pid, b)]["mean_eta"])
                out.update({f"{name}:{pid}|{a}-{b}|mean_eta.diff": m, f"{name}:{pid}|{a}-{b}|mean_eta.diff_se": se,
                            f"{name}:{pid}|{a}-{b}|mean_eta.diff_n": n})
    return out


def read_all() -> tuple[dict[str, float], dict]:
    """All source numbers {id: value} and the comparison tables {tau: table}."""
    sources = {}
    for path in [Path("results/thermostat/summary.json"), Path("results/drift_test/summary.json"),
                 Path("results/main/summary.json"), Path("results/real_box/fit/summary.json")]:
        if path.exists():
            sources.update(flatten(json.loads(path.read_text()), f"{path.relative_to('results')}:"))
    for path in sorted(Path("results/analysis").glob("*/*.json")):
        report = json.loads(path.read_text())
        name = f"analysis/{path.parent.name}/{path.stem}"
        seen = {}
        for f in report["findings"]:
            if f.get("value") is not None:
                k = seen[f["kind"]] = seen.get(f["kind"], -1) + 1
                sources[f"{name}:{f['kind']}" + (f"#{k}" if k else "")] = float(f["value"])
                sources.update(flatten(f.get("evidence", {}), f"{name}:{f['kind']}" + (f"#{k}" if k else "") + ".evidence."))
        sources.update(flatten(report.get("run_info", {}), f"{name}:run_info."))
    tables = {}
    for tau, file in TAU_FILES.items():
        path = Path("results/compare") / file
        if path.exists():
            tables[tau] = read_compare(path)
            sources.update(compare_sources(tables[tau], f"compare/{file}"))
    sources.update(flatten({"T_c": DRIFT_TEST["T_c"], "limit_C": THERMOSTAT["limit_C"], "sensor_std": THERMOSTAT["sensor_std"],
                            "swing_band_C": THERMOSTAT["swing_band_C"], "se_factor": COMPARE["se_factor"],
                            "time_drift_taus": [t for t in COMPARE["time_drift_taus"] if t],
                            "seeds": len(COMPARE["seeds"]), "room_warm_by_C": SCENARIO["room_warm_by_C"],
                            "cooler_fail_for_s": SCENARIO["cooler_fail_for_s"], "new_target_C": SCENARIO["new_target_C"],
                            "max_experiments": SETTINGS["max_experiments"], "max_seeds": SETTINGS["max_seeds"],
                            "max_compute_min": SETTINGS["max_compute_min"]}, "config.py:"))
    return sources, tables


def pid_labels() -> dict[str, str]:
    """{"real": label, "main": label} of the comparison PIDs."""
    from box_simulation import thermostat
    from config import PLOTS
    return {"real": thermostat.pid_label("real"), "main": thermostat.pid_label(PLOTS["main_pid"])}


def rules_questions(sources: dict, tables: dict) -> list[dict]:
    """The rules brain's fixed questions, each with the experiment it needs (or none)."""
    pid = pid_labels()
    return [
        {"id": "full_vs_rules_tau30_real_pid",
         "text": f"Does the full Map agent beat the fixed rules at time drift 30 min with {pid['real']}?",
         "experiment": {"kind": "compare", "operators": ["rules", "full"], "pid": pid["real"], "tau": 30,
                        "seeds": list(range(len(COMPARE["seeds"]), len(COMPARE["seeds"]) + SETTINGS["max_seeds"]))}},
        {"id": "tau_where_agents_beat_rules",
         "text": f"At which time drift tau do the agents start to beat the fixed rules ({pid['main']})?",
         "experiment": {"kind": "compare", "operators": ["rules", "full"], "pid": pid["main"], "tau": 60,
                        "seeds": list(range(SETTINGS["max_seeds"]))}},
        {"id": "team_vs_full", "text": "Does the team differ from the full Map agent?", "experiment": None},
    ]


def check_experiment(exp) -> str | None:
    """Why an experiment is not allowed (None if it is on the whitelist)."""
    from experiments import compare
    if not isinstance(exp, dict) or exp.get("kind") not in ("compare", "drift_test"):
        return f"kind {exp.get('kind') if isinstance(exp, dict) else exp!r} is not on the whitelist (compare, drift_test)"
    if exp["kind"] == "compare":
        if exp.get("pid") not in compare.PIDS:
            return f"PID {exp.get('pid')!r} is not one of the comparison PIDs"
        if not exp.get("operators") or any(op not in OPERATORS for op in exp["operators"]):
            return f"operators {exp.get('operators')!r} not all in {OPERATORS}"
        if not all(isinstance(s, int) and s >= 0 for s in exp.get("seeds", [])) or not exp.get("seeds"):
            return "seeds must be a non-empty list of non-negative whole numbers"
    tau = exp.get("tau")
    if tau is not None and not (isinstance(tau, (int, float)) and tau > 0):
        return f"tau {tau!r} must be None or a positive number of minutes"
    return None


def plan_budget(experiments: list) -> tuple[list, list, list]:
    """(accepted, refused (exp, why), cut (exp, why)) for the proposed experiments, in order."""
    accepted, refused, cut, minutes = [], [], [], 0.0
    for exp in experiments:
        why = check_experiment(exp)
        if why:
            refused.append((exp, why))
            continue
        exp = dict(exp)
        if exp["kind"] == "compare" and len(exp["seeds"]) > SETTINGS["max_seeds"]:
            exp["seeds"] = exp["seeds"][:SETTINGS["max_seeds"]]
            exp["note"] = f"seeds cut to the first {SETTINGS['max_seeds']}"
        runs = len(exp["seeds"]) * len(exp["operators"]) if exp["kind"] == "compare" else 6
        cost = runs * SETTINGS["est_min_per_run"]
        if len(accepted) >= SETTINGS["max_experiments"]:
            cut.append((exp, f"over max_experiments {SETTINGS['max_experiments']}"))
        elif minutes + cost > SETTINGS["max_compute_min"]:
            cut.append((exp, f"estimated {cost:.1f} min would pass max_compute_min {SETTINGS['max_compute_min']}"))
        else:
            accepted.append({**exp, "estimated_min": cost})
            minutes += cost
    return accepted, refused, cut


def run_experiment(k: int, exp: dict, tables: dict) -> tuple[dict, dict]:
    """Run one accepted experiment; returns (result summary, new source numbers {id: value})."""
    from experiments import compare, drift_test
    start = time.time()
    name = f"experiment{k}"
    if exp["kind"] == "compare":
        tasks = [(seed, exp["pid"], op, exp["tau"]) for seed in exp["seeds"] for op in exp["operators"]]
        with Pool() as pool:
            rows = pool.map(compare.run_one, tasks)
        table = {}
        for r in rows:
            per = table.setdefault((r["pid"], r["operator"]), {})
            for key in ("mean_eta", "lab_time_on_maps_s", "first_reaction_s"):
                per.setdefault(key, {})[r["seed"]] = float(r[key])
        new = compare_sources(table, name)
        # With saved seeds for the same PID and time drift, the paired test also runs on all seeds together.
        if exp["tau"] in tables:
            both = {}
            for key in table:
                old = tables[exp["tau"]].get(key, {})
                both[key] = {m: {**old.get(m, {}), **v} for m, v in table[key].items()}
            new.update(compare_sources(both, f"{name}+saved"))
        result = {"runs": len(rows)}
    else:
        t, out = drift_test.long_test(tau_min=exp["tau"])
        new = {f"{name}:{case}|room {room:g}|r_end": float(v["r"][-1]) for (case, room), v in out.items()}
        result = {"runs": len(out)}
    result["wall_min"] = (time.time() - start) / 60.0
    new[f"{name}:wall_min"] = result["wall_min"]
    if exp.get("tau") is not None:
        new[f"{name}:tau"] = float(exp["tau"])
    return result, new


def numbers(text: str) -> list[str]:
    """The numbers written in a sentence (PID names such as "PID 3000/0.36" are names, not numbers)."""
    text = re.sub(r"PID \d+(?:\.\d+)?/\d+(?:\.\d+)?", "PID", text)
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


VERDICT_WORDS = {"higher": [r"\bhigher\b", r"\bbeats?\b", r"\bahead\b", r"\bbetter\b"],
                 "lower": [r"\blower\b", r"\bbehind\b", r"\bworse\b"],
                 "no clear difference": [r"\bno clear difference\b", r"\blevel\b", r"\bsame\b"],
                 "identical": [r"\bidentical\b"]}


def check_claim(claim: dict, sources: dict) -> str | None:
    """Why a sentence fails (None if it passes). Every number must match a cited source within its rounding (also as
    min or %); a signed number (+0.7, -0.8) only a source of the same sign, an unsigned one also the absolute value.
    A verdict word (VERDICT_WORDS: beats/higher/ahead/better, lower/behind/worse, no clear difference/level/same,
    identical) needs a cited .diff with its .diff_se, and each word used must be the verdict of one of the cited pairs;
    an identical pair (exactly 0 ± 0) also allows the no-clear-difference words. Questions (up to "?") are not checked."""
    ids = claim.get("sources") or []
    if not ids:
        return "no source cited"
    missing = [i for i in ids if i not in sources]
    if missing:
        return f"unknown source {missing[0]!r}"
    known = [x for i in ids for x in (sources[i], sources[i] / 60, sources[i] * 100) if np.isfinite(sources[i])]
    text = claim.get("text", "")
    for token in numbers(text):
        n, decimals = float(token), len(token.split(".")[1]) if "." in token else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        signed = token[0] in "+-"
        if not any(abs(n - k) <= tol if signed else abs(abs(n) - abs(k)) <= tol for k in known):
            return f"number {token} matches no cited source" + (" with its sign" if signed and any(
                abs(abs(n) - abs(k)) <= tol for k in known) else "")
    claim_text = re.sub(r"Question:[^?]*\?", "", text)
    used = {v for v, patterns in VERDICT_WORDS.items() if any(re.search(p, claim_text) for p in patterns)}
    if used:
        pairs = [(i, i[:-len(".diff")] + ".diff_se") for i in ids if i.endswith(".diff")]
        pairs = [(d, s) for d, s in pairs if s in ids]
        if not pairs:
            return f"verdict word ({', '.join(sorted(used))}) without a cited .diff and its .diff_se"
        allowed = {verdict(sources[d], sources[s]) for d, s in pairs}
        if "identical" in allowed:
            allowed.add("no clear difference")
        wrong = used - allowed
        if wrong:
            return f"verdict word {sorted(wrong)[0]!r} disagrees with the sources (verdict {', '.join(sorted(allowed))})"
    return None


FILE_NAMES = {f"compare/{f}": f"comparison, {tau_text(t)}" for t, f in TAU_FILES.items()}


def source_name(sid: str) -> str:
    """A short readable name of a source id, e.g. "comparison, time drift 30 min: PID 200/4, full-rules, mean_eta diff"."""
    file, key = sid.split(":", 1)
    file = FILE_NAMES.get(file, re.sub(r"^experiment(\d+)", r"new experiment \1", file))
    return f"{file}: " + key.replace("|", ", ").replace("_", " ")


def sources_text(sources: dict) -> str:
    """All sources for an AI prompt: one line per source (id, value, name); grouped by file when that would pass
    max_tokens (about 4 characters per token)."""
    flat = "\n".join(f"{sid} = {v:.6g} ({source_name(sid)})" for sid, v in sources.items())
    if len(flat) / 4 <= SETTINGS["max_tokens"]:
        return flat
    groups = {}
    for sid, v in sources.items():
        file, key = sid.split(":", 1)
        groups.setdefault(file, []).append(f"  {key} = {v:.6g}")
    return "Grouped by file; a source id is '<file>:<key>'.\n" + "\n".join(
        f"{file} ({FILE_NAMES.get(file, file)}):\n" + "\n".join(lines) for file, lines in groups.items())


CODE_ANSWERS = ["full_vs_rules_tau30_real_pid", "tau_where_agents_beat_rules", "team_vs_full"]


def rules_sentences(sources: dict, answers: dict) -> dict[str, list[dict]]:
    """The rules brain's report sentences, each with its source ids (numbers formatted from those sources)."""
    S = sources
    pid = pid_labels()
    out = {h: [] for h in SECTIONS}
    def say(section, text, *ids):
        out[section].append({"text": text, "sources": list(ids)})

    m = "main/summary.json:"
    say(SECTIONS[0], f"A phase-only SLM with N = {S[m + 'settings.N']:.0f} squares shapes light through a scattering sample "
        f"seen by M = {S[m + 'settings.M']:.0f} camera points; GGS learns the sample's transmission matrix (the map) from "
        f"{S[m + 'results.L']:.0f} photos, and a fresh map predicts the camera with correlation "
        f"{S[m + 'results.prediction_correlation']:.3f}.", m + "settings.N", m + "settings.M", m + "results.L",
        m + "results.prediction_correlation")
    say(SECTIONS[0], f"The box is held at its target by a PID and a Kalman filter; in the two-hour scenario the room warms "
        f"by {S['config.py:room_warm_by_C']:g} °C, the cooler fails for {S['config.py:cooler_fail_for_s'] / 60:.0f} min "
        f"and the user asks for a new target of {S['config.py:new_target_C']:g} °C.",
        "config.py:room_warm_by_C", "config.py:cooler_fail_for_s", "config.py:new_target_C")

    t = "thermostat/summary.json:"
    labels = list(json.loads(Path("results/thermostat/summary.json").read_text())["pids"])
    for label in labels:
        key = f"{t}pids.{label}."
        if key + "cooldown_inside_limit_s" in S:
            text = f"stays inside ±{S[t + 'limit_C']:g} °C {S[key + 'cooldown_inside_limit_s'] / 60:.1f} min after a target step"
            ids = [t + "limit_C", key + "cooldown_inside_limit_s"]
        else:
            text = f"never stays inside ±{S[t + 'limit_C']:g} °C after a step (it swings)"
            ids = [t + "limit_C"]
        say(SECTIONS[1], f"With {label} the box {text}; in a hold its temperature std is "
            f"{S[key + 'hold_std_C.Kalman']:.4f} °C.", *ids, key + "hold_std_C.Kalman")
    k = f"{t}pids.{pid['main']}."
    say(SECTIONS[1], f"The Kalman filter hardly changes the box temperature (hold std {S[k + 'hold_std_C.Kalman']:.4f} vs. "
        f"{S[k + 'hold_std_C.raw sensor']:.4f} °C on the raw sensor with {pid['main']}), but the cooler is much calmer: the duty "
        f"change per control step has a std of {S[k + 'duty_change_std_pct.Kalman']:.3f} % instead of "
        f"{S[k + 'duty_change_std_pct.raw sensor']:.3f} %.", k + "hold_std_C.Kalman", k + "hold_std_C.raw sensor",
        k + "duty_change_std_pct.Kalman", k + "duty_change_std_pct.raw sensor")
    say(SECTIONS[1], f"Its readings are also more accurate: in the {S[t + 'kalman_error_std_C.hold_s'] / 60:.0f} min hold "
        f"at {S[t + 'target_C']:g} °C with {pid['real']}, the error std of the temperature reading is "
        f"{S[t + 'kalman_error_std_C.Kalman']:.4f} °C with the Kalman filter vs. "
        f"{S[t + 'kalman_error_std_C.raw sensor']:.4f} °C for the raw sensor.", t + "kalman_error_std_C.hold_s",
        t + "target_C", t + "kalman_error_std_C.Kalman", t + "kalman_error_std_C.raw sensor")

    a = "analysis/agents/full_map_agent_pid_200_4_fresh_level:"
    if a + "map T_c exponential" in S:
        say(SECTIONS[2], f"The analyst's fit of the map checks against the temperature change gives T_c = "
            f"{S[a + 'map T_c exponential']:.2f} °C for the exponential intensity model, close to the lab model's "
            f"{S['config.py:T_c']:g} °C; the Gaussian form gives {S[a + 'map T_c gaussian']:.2f} °C because the model is "
            "not Gaussian.", a + "map T_c exponential", "config.py:T_c", a + "map T_c gaussian")
    c = lambda tau, p, op, key: f"compare/{TAU_FILES[tau]}:{p}|{op}|{key}"
    say(SECTIONS[2], f"Without time drift the sample comes back when the temperature comes back, so never re-measuring "
        f"keeps a mean η of {S[c(None, pid['real'], 'never', 'mean_eta.mean')]:.1f} with {pid['real']}; with time drift of "
        f"{S['config.py:time_drift_taus[0]']:g} and {S['config.py:time_drift_taus[1]']:g} min it falls to "
        f"{S[c(120, pid['real'], 'never', 'mean_eta.mean')]:.1f} and {S[c(30, pid['real'], 'never', 'mean_eta.mean')]:.1f}.",
        c(None, pid['real'], 'never', 'mean_eta.mean'), "config.py:time_drift_taus[0]", "config.py:time_drift_taus[1]",
        c(120, pid['real'], 'never', 'mean_eta.mean'), c(30, pid['real'], 'never', 'mean_eta.mean'))

    say(SECTIONS[3], f"Mean η and lab time on maps over {S['config.py:seeds']:.0f} seeds are in the table below; a "
        f"difference counts only beyond {S['config.py:se_factor']:g} standard errors of the paired seed differences.",
        "config.py:seeds", "config.py:se_factor")

    d = lambda tau, p, pair: f"compare/{TAU_FILES[tau]}:{p}|{pair}|mean_eta"
    tau_id = {None: [], 120: ["config.py:time_drift_taus[0]"], 30: ["config.py:time_drift_taus[1]"]}
    for agent, pair in [("Full", "full-rules"), ("Basic", "basic-rules")]:
        for tau in TAU_FILES:
            for p in pid.values():
                mean, se = S[d(tau, p, pair) + '.diff'], S[d(tau, p, pair) + '.diff_se']
                say(SECTIONS[4], f"{agent} Map agent minus fixed rules, {p}, {tau_text(tau)}: {mean:+.1f} ± {se:.1f} η "
                    f"({verdict(mean, se)}).", d(tau, p, pair) + '.diff', d(tau, p, pair) + '.diff_se', *tau_id[tau])
    r, f = c(None, pid['real'], 'rules', 'first_reaction_s.mean'), c(None, pid['real'], 'full', 'first_reaction_s.mean')
    say(SECTIONS[4], f"The agents notice the cooler failure much sooner: the full Map agent reacts after {S[f]:.0f} s "
        f"on average, the fixed rules after {S[r]:.0f} s ({pid['real']}, no time drift).", f, r)
    lr, lf = c(30, pid['main'], 'rules', 'lab_time_on_maps_s.mean'), c(30, pid['main'], 'full', 'lab_time_on_maps_s.mean')
    say(SECTIONS[4], f"Their gain at time drift 30 min costs lab time: {S[lf]:.0f} s on maps for the full Map agent "
        f"vs. {S[lr]:.0f} s for the fixed rules ({pid['main']}).", lf, lr, "config.py:time_drift_taus[1]")
    tm = d(None, pid['main'], 'team-full')
    say(SECTIONS[4], f"The team gives exactly the full Map agent's numbers (difference {S[tm + '.diff']:+.1f} ± "
        f"{S[tm + '.diff_se']:.1f} η), because both use the same stability rule.", tm + '.diff', tm + '.diff_se')

    for q in answers.values():
        for sentence in q["sentences"]:
            out[SECTIONS[5]].append(sentence)

    say(SECTIONS[6], f"T_c of the real sample: the simulation assumes {S['config.py:T_c']:g} °C (an estimate from the "
        "thesis drift curve); measure the map check against the temperature change on the real setup.", "config.py:T_c")
    say(SECTIONS[6], f"Time drift of the real sample at a constant temperature: the comparison tried tau "
        f"{S['config.py:time_drift_taus[0]']:g} and {S['config.py:time_drift_taus[1]']:g} min, and whether the agents "
        "help depends on it.", "config.py:time_drift_taus[0]", "config.py:time_drift_taus[1]")
    say(SECTIONS[6], f"The PID and the Kalman filter on the real box: {pid['real']} swings and never stays inside "
        f"±{S[t + 'limit_C']:g} °C in the simulation; test {pid['main']} on the box, and the sensor noise "
        f"({S['config.py:sensor_std']:.4f} °C from the real test) that the Kalman filter is tuned for.",
        t + "limit_C", "config.py:sensor_std")
    return out


class ScientistState(TypedDict, total=False):
    """The graph state."""
    brain: str
    replies: list
    sources: dict
    tables: dict
    questions: list
    plan: list
    refused: list
    cut: list
    results: list
    answers: dict
    sentences: dict
    claims: dict
    dropped: list
    log: list


def ask(state: ScientistState, prompt: str) -> dict:
    """One brain call returning JSON (fake: the next scripted reply; else a LangChain chat model); raises on errors."""
    if state["brain"] == "fake":
        reply = state["replies"].pop(0)
    else:
        from agents.brain import chat_model
        from env import load_env
        load_env()
        key = os.environ.get(AI_BRAIN["key_env"][state["brain"]])
        if not key:
            raise RuntimeError(f"no API key ({AI_BRAIN['key_env'][state['brain']]} not set)")
        reply = chat_model(state["brain"], key, SETTINGS["max_tokens"], SETTINGS["timeout_s"]).invoke(
            [HumanMessage(prompt)])
    text = reply.content if isinstance(reply, AIMessage) else str(reply)
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def node_read(state: ScientistState) -> ScientistState:
    """Read every result file (code only)."""
    sources, tables = read_all()
    return {"sources": sources, "tables": tables, "log": state.get("log", []) + [f"read {len(sources)} source numbers"]}


def node_questions(state: ScientistState) -> ScientistState:
    """Up to max_questions open questions (rules: the fixed list)."""
    log = list(state["log"])
    questions = rules_questions(state["sources"], state["tables"])
    if state["brain"] != "rules":
        try:
            answer = ask(state, "Pick up to {} open questions from these results. Code can answer the question ids {}; "
                         "other questions need a matching experiment. Answer JSON "
                         '{{"questions": [{{"id": ..., "text": ...}}]}}.\n\nSources:\n{}'.format(
                             SETTINGS["max_questions"], CODE_ANSWERS, sources_text(state["sources"])))
            questions = [{"id": q["id"], "text": q["text"], "experiment": None} for q in answer["questions"]]
        except Exception as e:
            log.append(f"scientist fallback (questions): {e}")
            state = {**state, "brain": "rules"}
    return {"questions": questions[:SETTINGS["max_questions"]], "log": log, "brain": state["brain"]}


def node_plan(state: ScientistState) -> ScientistState:
    """Proposed experiments, checked against the whitelist and the budget."""
    log = list(state["log"])
    proposed = [q["experiment"] for q in state["questions"] if q.get("experiment")]
    if state["brain"] != "rules":
        try:
            proposed = ask(state, "Propose experiments for these questions: {}. Allowed kinds: compare (operators, pid, "
                           "tau, seeds) and drift_test (tau). Answer JSON {{\"experiments\": [...]}}.".format(
                               json.dumps(state["questions"])))["experiments"]
        except Exception as e:
            log.append(f"scientist fallback (plan): {e}")
    accepted, refused, cut = plan_budget(proposed)
    log += [f"refused experiment {e}: {why}" for e, why in refused] + [f"cut experiment {e}: {why}" for e, why in cut]
    return {"plan": accepted, "refused": refused, "cut": cut, "log": log}


def node_run(state: ScientistState) -> ScientistState:
    """Run the accepted experiments (stop when the compute budget is used)."""
    sources, results, used, log = dict(state["sources"]), [], 0.0, list(state["log"])
    per_run, runs_done = SETTINGS["est_min_per_run"], 0
    for k, exp in enumerate(state["plan"], start=1):
        runs = len(exp["seeds"]) * len(exp["operators"]) if exp["kind"] == "compare" else 6
        if used + runs * per_run > SETTINGS["max_compute_min"]:
            log.append(f"experiment {k} not run: estimated {runs * per_run:.1f} min ({per_run:.3f} min per run) "
                       f"would pass max_compute_min after {used:.1f} min")
            continue
        result, new = run_experiment(k, exp, state["tables"])
        result["estimated_min"] = runs * per_run
        used += result["wall_min"]
        runs_done += result["runs"]
        per_run = used / runs_done  # measured wall time per run so far, for the next estimate
        sources.update(new)
        results.append({**exp, **result, "id": f"experiment{k}"})
        log.append(f"experiment {k} ({exp['kind']}): {result['runs']} runs in {result['wall_min']:.1f} min")
    return {"sources": sources, "results": results, "answers": answer_questions(state["questions"], results, sources),
            "log": log}


def answer_questions(questions: list, results: list, S: dict) -> dict:
    """Code answers for the rules questions (numbers and verdicts from the sources), as cited sentences."""
    pid = pid_labels()
    answers = {}
    for q in questions:
        sentences = []
        if q["id"] == "full_vs_rules_tau30_real_pid":
            base = f"compare/{TAU_FILES[30]}:{pid['real']}|full-rules|mean_eta"
            sentences.append({"text": f"Question: {q['text']} With the saved {S[base + '.diff_n']:.0f} seeds the difference "
                              f"is {S[base + '.diff']:+.1f} ± {S[base + '.diff_se']:.1f} η "
                              f"({verdict(S[base + '.diff'], S[base + '.diff_se'])}).",
                              "sources": [base + ".diff_n", base + ".diff", base + ".diff_se", "config.py:time_drift_taus[1]"]})
            for r in results:
                if r["kind"] == "compare" and r["pid"] == pid["real"] and r["tau"] == 30:
                    b = f"{r['id']}+saved:{pid['real']}|full-rules|mean_eta"
                    sentences.append({"text": f"Answer: with {S[b + '.diff_n']:.0f} seeds (saved and new) the full Map agent "
                                      f"minus the fixed rules is {S[b + '.diff']:+.1f} ± {S[b + '.diff_se']:.1f} η, so "
                                      f"{verdict(S[b + '.diff'], S[b + '.diff_se'])}.",
                                      "sources": [b + ".diff_n", b + ".diff", b + ".diff_se"]})
        elif q["id"] == "tau_where_agents_beat_rules":
            parts, ids = [], []
            points = [(None, f"compare/{TAU_FILES[None]}:", None), (120, f"compare/{TAU_FILES[120]}:", "config.py:time_drift_taus[0]")]
            points += [(r["tau"], f"{r['id']}:", f"{r['id']}:tau") for r in results
                       if r["kind"] == "compare" and r["pid"] == pid["main"]]
            points += [(30, f"compare/{TAU_FILES[30]}:", "config.py:time_drift_taus[1]")]
            per_tau = []
            for tau, prefix, tau_id in points:
                key = f"{prefix}{pid['main']}|full-rules|mean_eta"
                if key + ".diff" not in S:
                    continue
                v = verdict(S[key + ".diff"], S[key + ".diff_se"])
                parts.append(f"{tau_text(tau)} {S[key + '.diff']:+.1f} ± {S[key + '.diff_se']:.1f} ({v})")
                ids += [key + ".diff", key + ".diff_se"] + ([tau_id] if tau_id else [])
                per_tau.append((tau, v))
            sentences.append({"text": f"Question: {q['text']} Full Map agent minus fixed rules in η: " + "; ".join(parts) + ".",
                              "sources": ids})
            # The answer's wording comes from the verdicts (2 standard error rule), fastest drift first.
            per_tau.sort(key=lambda p: -1 if p[0] is None else -p[0])
            at = lambda tau: "without time drift" if tau is None else f"at time drift {tau:g} min"
            phrase = lambda tau, v: at(tau) + (" there is no clear difference" if v == "no clear difference"
                                               else f" it is {v}")
            higher = [tau for tau, v in per_tau if v == "higher"]
            rest = ", ".join(phrase(tau, v) for tau, v in reversed(per_tau) if v != "higher")
            if higher:
                taus = " and ".join(f"{tau:g}" for tau in sorted(higher, reverse=True))
                text = (f"Answer: the full Map agent beats the fixed rules (beyond the 2 standard errors) at time drift "
                        f"{taus} min" + (f"; {rest}." if rest else "."))
            else:
                text = f"Answer: compared with the fixed rules (2 standard error rule), {rest}."
            sentences.append({"text": text, "sources": ids + ["config.py:se_factor"]})
        elif q["id"] == "team_vs_full":
            ids, same = [], True
            for tau in TAU_FILES:
                for p in pid.values():
                    key = f"compare/{TAU_FILES[tau]}:{p}|team-full|mean_eta"
                    if key + ".diff" in S:
                        ids += [key + ".diff", key + ".diff_se"]
                        same &= S[key + ".diff"] == 0 and S[key + ".diff_se"] == 0
            sentences.append({"text": f"Question: {q['text']} Answer: " + ("no; in every saved comparison the difference "
                              "is exactly zero, seed by seed." if same else "yes, in some comparisons (see the table)."),
                              "sources": ids})
        answers[q["id"]] = {"question": q["text"], "sentences": sentences}
        if q["id"] not in CODE_ANSWERS:
            answers[q["id"]]["not_answered"] = "no matching experiment or data"
    return answers


def node_check(state: ScientistState) -> ScientistState:
    """The sentences (from the brain, or the rules) checked against their sources; failures dropped and logged."""
    log = list(state["log"])
    sentences = rules_sentences(state["sources"], state["answers"])
    if state["brain"] != "rules":
        try:
            sentences = ask(state, "Write the report sections {} (cite the source ids of every number in 'sources'; "
                            "a difference counts only beyond 2 standard errors). Answer JSON "
                            "{{\"sections\": {{heading: [{{\"text\": ..., \"sources\": [...]}}]}}}}.\n\n"
                            "Code answers to the questions: {}\n\nSources:\n{}".format(
                                SECTIONS, json.dumps(state["answers"]), sources_text(state["sources"])))["sections"]
        except Exception as e:
            log.append(f"scientist fallback (write): {e}")
    claims, dropped = {h: [] for h in SECTIONS}, []
    for heading, items in sentences.items():
        for item in items:
            why = check_claim(item, state["sources"])
            if why:
                dropped.append({"section": heading, "text": item.get("text", ""), "why": why})
                log.append(f"dropped sentence ({why}): {item.get('text', '')[:80]}")
            else:
                claims.setdefault(heading, []).append(item)
    return {"claims": claims, "dropped": dropped, "log": log}


def operators_table(S: dict) -> list[str]:
    """Markdown table rows (made by code from the comparison sources): mean eta ± SE and lab time on maps."""
    pid = pid_labels()
    rows = ["| operator | PID | time drift | mean η ± SE | lab time on maps (s) ± SE |", "|---|---|---|---|---|"]
    for tau in TAU_FILES:
        for p in pid.values():
            for op in OPERATORS:
                key = f"compare/{TAU_FILES[tau]}:{p}|{op}|"
                if key + "mean_eta.mean" in S:
                    rows.append(f"| {NAMES[op]} | {p} | {tau_text(tau)} | {S[key + 'mean_eta.mean']:.1f} ± "
                                f"{S[key + 'mean_eta.se']:.1f} | {S[key + 'lab_time_on_maps_s.mean']:.0f} ± "
                                f"{S[key + 'lab_time_on_maps_s.se']:.0f} |")
    return rows


PLOTS_BY_SECTION = {
    "Thermostat": ["results/thermostat/01_pid_cooldown_21_to_20C.png", "results/thermostat/03_pid_hold_20C_with_vs_without_kalman.png"],
    "Optics and map staleness": ["results/drift_test/02_long_test_map_accuracy.png"],
    "Operators compared": ["results/compare/01_mean_eta_and_lab_time_per_operator_time_drift.png"],
    "What the agents add, and where they do not": ["results/agents/02_full_map_agent_real_vs_main_pid_focus_brightness.png",
                                                   "results/scenario/01_box_and_room_temperature.png"],
}


def write_report(state: ScientistState, out_dir: Path, brain_note: str) -> list[Path]:
    """results/report/report.md and report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    S, claims = state["sources"], state["claims"]
    cited = sorted({i for items in claims.values() for c in items for i in c["sources"]})
    number = {sid: k + 1 for k, sid in enumerate(cited)}
    used = sum(r["wall_min"] for r in state["results"])
    lines = ["# Scientist report", "", f"Written {time.strftime('%Y-%m-%d %H:%M')}; {brain_note}. Every sentence "
             "cites its sources [S…] (listed at the end); every number in it was checked against them. Tables and plot "
             "paths are made by code from the same sources.", ""]
    for heading in SECTIONS:
        lines += [f"## {heading}", ""]
        for c in claims.get(heading, []):
            lines.append(f"- {c['text']} [" + ", ".join(f"S{number[i]}" for i in c["sources"]) + "]")
        if heading == "Operators compared":
            lines += [""] + operators_table(S) + ["", "Sources of the table: results/compare/results*.csv (mean and "
                                                   "standard error over the seeds)."]
        if heading == "New experiments and their answers":
            over = used - SETTINGS["max_compute_min"]
            lines += ["", f"Budget: {len(state['results'])} of {SETTINGS['max_experiments']} experiments, "
                      f"{sum(r['runs'] for r in state['results'])} runs, {used:.1f} of {SETTINGS['max_compute_min']} min"
                      + (f", over budget by {over:.1f} min." if over > 0 else ".")]
            lines += [f"- {r['id']}: {r['kind']}, operators {', '.join(r.get('operators', []))}, "
                      f"{r.get('pid', '')}, {tau_text(r.get('tau'))}, seeds {r['seeds'][0]}–{r['seeds'][-1]}"
                      for r in state["results"] if r["kind"] == "compare"]
            lines += [f"- Question not answered: {a['question']} ({a['not_answered']})"
                      for a in state["answers"].values() if a.get("not_answered")]
            lines += [f"- Refused: {e} ({why})" for e, why in state["refused"]]
            lines += [f"- Cut: {e} ({why})" for e, why in state["cut"]]
        if heading == "Limitations":
            lines += ["- Simulation only: no number here is measured on the real box (the real-box fit is in "
                      "results/real_box/fit/).", f"- Brain: {brain_note}."]
        for plot in PLOTS_BY_SECTION.get(heading, []):
            lines += ["", f"![{Path(plot).stem}](../../{plot})"]
        lines.append("")
    lines += ["## Sources", ""] + [f"- S{number[i]}: `{i}` = {S[i]:.6g}" for i in cited]
    if state["dropped"]:
        lines += ["", "## Dropped sentences", ""] + [f"- ({d['why']}) {d['text']}" for d in state["dropped"]]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    report = {"brain": brain_note, "claims": claims, "sources": {i: S[i] for i in cited}, "dropped": state["dropped"],
              "questions": state["questions"], "experiments": state["results"],
              "not_answered": [a["question"] for a in state["answers"].values() if a.get("not_answered")],
              "refused": [{"experiment": e, "why": w} for e, w in state["refused"]],
              "cut": [{"experiment": e, "why": w} for e, w in state["cut"]],
              "budget_used": {"experiments": len(state["results"]), "runs": sum(r["runs"] for r in state["results"]),
                              "wall_min": used, "over_budget_min": max(0.0, used - SETTINGS["max_compute_min"]),
                              "max_experiments": SETTINGS["max_experiments"],
                              "max_seeds": SETTINGS["max_seeds"], "max_compute_min": SETTINGS["max_compute_min"]},
              "log": state["log"]}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return [out_dir / "report.md", out_dir / "report.json"]


def run_scientist(brain: str, out_dir: Path, replies: list | None = None) -> dict:
    """The graph read -> questions -> plan -> run -> check -> write; returns the final state."""
    graph = StateGraph(ScientistState)
    for name, node in [("read", node_read), ("questions", node_questions), ("plan", node_plan), ("run", node_run),
                       ("check", node_check)]:
        graph.add_node(name, node)
    note = {"rules": "rules brain (no AI)", "fake": "fake brain (scripted test replies)"}.get(brain, f"AI brain {brain}")
    graph.add_node("write", lambda state: {"log": state["log"] + [f"wrote {p}" for p in write_report(state, out_dir, note)]})
    for a, b in [(START, "read"), ("read", "questions"), ("questions", "plan"), ("plan", "run"), ("run", "check"),
                 ("check", "write"), ("write", END)]:
        graph.add_edge(a, b)
    return graph.compile().invoke({"brain": brain, "replies": list(replies or []), "log": []})


def choose_brain() -> str:
    """The AI_BRAIN provider if it is a real one with an API key, else the rules brain."""
    from env import load_env
    load_env()
    provider = AI_BRAIN["provider"]
    return provider if provider in AI_BRAIN["key_env"] and os.environ.get(AI_BRAIN["key_env"][provider]) else "rules"


if __name__ == "__main__":
    import sys
    import tempfile

    if sys.argv[1:] != ["--test"]:
        brain = choose_brain()
        final = run_scientist(brain, Path(SETTINGS["out_dir"]))
        print("\n".join(final["log"]))
        sys.exit(0)

    # Self-test: a fake brain with scripted replies (one forbidden experiment, a plan over budget, a question with no
    # code answer, sentences with a wrong number, no source, a wrong sign and a wrong verdict word).
    from box_simulation import thermostat
    real = thermostat.pid_label("real")
    small = lambda seed: {"kind": "compare", "operators": ["rules", "full"], "pid": real, "tau": 30, "seeds": [seed]}
    plan = [small(100), {"kind": "set_target", "T": 25.0}, small(101), small(102), small(103)]
    claim_ok = {"text": f"With {real} the fixed rules at time drift 30 min reach a mean η of {{v:.1f}}.",
                "sources": [f"compare/{TAU_FILES[30]}:{real}|rules|mean_eta.mean", "config.py:time_drift_taus[1]"]}
    sources, _ = read_all()
    claim_ok["text"] = claim_ok["text"].format(v=sources[claim_ok["sources"][0]])
    wrong = {"text": f"With {real} the fixed rules at time drift 30 min reach a mean η of 99.9.", "sources": claim_ok["sources"]}
    unsourced = {"text": "The agents are always better.", "sources": []}
    main = thermostat.pid_label(__import__("config").PLOTS["main_pid"])
    pair = f"compare/{TAU_FILES[30]}:{main}|full-rules|mean_eta"
    diff, se = sources[pair + ".diff"], sources[pair + ".diff_se"]
    cite = [pair + ".diff", pair + ".diff_se", "config.py:time_drift_taus[1]"]
    verdict_ok = {"text": f"At time drift 30 min the full Map agent minus the fixed rules is {diff:+.1f} ± {se:.1f} η "
                  f"({verdict(diff, se)}).", "sources": cite}
    wrong_sign = {"text": f"At time drift 30 min the full Map agent minus the fixed rules is {-diff:+.1f} ± {se:.1f} η.",
                  "sources": cite}
    wrong_word = {"text": f"At time drift 30 min the full Map agent minus the fixed rules is {diff:+.1f} ± {se:.1f} η "
                  "(no clear difference).", "sources": cite}
    low = f"compare/{TAU_FILES[120]}:{main}|full-rules|mean_eta"
    level = {"text": "At time drift 120 min the full Map agent is level with the fixed rules.",
             "sources": [low + ".diff", low + ".diff_se", "config.py:time_drift_taus[0]"]}
    no_pair = {"text": "The full Map agent beats the fixed rules at time drift 30 min.",
               "sources": ["config.py:time_drift_taus[1]"]}
    replies = [AIMessage(json.dumps({"questions": [{"id": "q1", "text": "Do the agents beat the rules?"}]})),
               AIMessage(json.dumps({"experiments": plan})),
               AIMessage(json.dumps({"sections": {SECTIONS[4]: [claim_ok, wrong, unsourced, verdict_ok, wrong_sign,
                                                                wrong_word, no_pair, level]}}))]
    with tempfile.TemporaryDirectory() as tmp:
        final = run_scientist("fake", Path(tmp), replies)
        report = json.loads((Path(tmp) / "report.json").read_text())
        assert (Path(tmp) / "report.md").exists()
    refused = [e for e, _ in final["refused"]]
    print(f"refused: {final['refused']}")
    print(f"cut: {[(e['seeds'], why) for e, why in final['cut']]}")
    print(f"ran: {[(r['id'], r['seeds'], r['runs']) for r in final['results']]}")
    print(f"dropped: {[(d['why'], d['text'][:50]) for d in final['dropped']]}")
    assert refused == [{"kind": "set_target", "T": 25.0}]
    assert len(final["results"]) == SETTINGS["max_experiments"] and [e["seeds"] for e, _ in final["cut"]] == [[103]]
    assert verdict(diff, se) == "higher"
    assert [d["why"] for d in final["dropped"]] == [
        "number 99.9 matches no cited source", "no source cited",
        f"number {-diff:+.1f} matches no cited source with its sign",
        "verdict word 'no clear difference' disagrees with the sources (verdict higher)",
        "verdict word (higher) without a cited .diff and its .diff_se",
        "verdict word 'no clear difference' disagrees with the sources (verdict lower)"]
    assert verdict(sources[low + ".diff"], sources[low + ".diff_se"]) == "lower"
    assert final["claims"][SECTIONS[4]] == [claim_ok, verdict_ok]
    assert report["budget_used"]["experiments"] == SETTINGS["max_experiments"]
    assert report["not_answered"] == ["Do the agents beat the rules?"]
    # A budget overrun: a tiny budget with a too low first estimate; the first experiment passes the budget, the
    # second is not run (its estimate now uses the measured time per run), and the report says by how much.
    saved = dict(SETTINGS)
    SETTINGS.update(est_min_per_run=1e-4, max_compute_min=0.01)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            over = run_scientist("fake", Path(tmp), [AIMessage(json.dumps({"questions": []})),
                                                     AIMessage(json.dumps({"experiments": [small(104), small(105)]})),
                                                     AIMessage(json.dumps({"sections": {}}))])
            budget_line = [l for l in (Path(tmp) / "report.md").read_text().splitlines() if l.startswith("Budget")][0]
            over_min = json.loads((Path(tmp) / "report.json").read_text())["budget_used"]["over_budget_min"]
    finally:
        SETTINGS.update(saved)
    print(f"overrun: {budget_line}")
    assert len(over["results"]) == 1 and over_min > 0 and f"over budget by {over_min:.1f} min" in budget_line
    assert any("experiment 2 not run" in line for line in over["log"])
    # The checker on its own: a number as minutes, a PID name, a rounding step.
    assert check_claim({"text": "It lasts 10 min.", "sources": ["config.py:cooler_fail_for_s"]}, sources) is None
    assert check_claim({"text": f"With {real} it lasts 11 min.", "sources": ["config.py:cooler_fail_for_s"]}, sources)
    assert check_claim({"text": f"The gap is {abs(diff):.1f} η.", "sources": cite[:1]}, sources) is None
    print("scientist.py self-test passed")
