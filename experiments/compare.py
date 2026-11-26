"""
compare.py - the operators over many seeds: never, rules, the basic and the full Map agent (rules brain, fresh level on;
the full one with watchdog, safety check and prediction), the team (Temperature agent + full Map agent) and the team
whose Map agent relies on the Temperature agent's messages instead of its own stability check (team_no_own_check),
each with the box PID of today (THERMOSTAT) and with the main PID of the plots (PLOTS main_pid, an override only),
and (python -m experiments.compare full) for each time drift setting in COMPARE time_drift_taus.

Each run is the 2-hour scenario with LabAPI seed s (lab, sample, sensor and GGS seeds all shift with it).
For each operator and PID: mean ± std over seeds. Differences are paired by seed (same lab, same sample):
a difference counts as real only if |mean difference| > se_factor standard errors of the paired differences,
otherwise "no clear difference". All runs are spread over the CPU cores (multiprocessing), as in tune_pid.py.

python -m experiments.compare        self-test: test_seeds only, output in a temporary folder
python -m experiments.compare full   all seeds, writes results/compare/
python -m experiments.compare forms  the predictor's staleness form "gaussian" vs "auto" (staleness.py) for the basic and
                                     the full Map agent, all seeds, both PIDs, every time drift; writes results/compare/forms.*
"""

from __future__ import annotations

import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from agents.team import baseline

from agents.map_agent import loop, staleness
from agents.core.state import LabState
from analysis import plots
from box_simulation import thermostat
from config import AGENT_LOOP, LAB_API, SCENARIO, THERMOSTAT
from config import COMPARE as SETTINGS

# The real box PID and the main PID of the plots (PLOTS main_pid), by label.
PIDS = {thermostat.pid_label(n): thermostat.pid_gains(n) or None for n in thermostat.plot_pids()}
METRICS = [("mean eta", "mean_eta", ".1f"), ("min eta", "min_eta", ".1f"), ("measure_map calls", "measure_map_calls", ".1f"),
           ("lab time on maps (s)", "lab_time_on_maps_s", ".0f"),
           (f"share within ±{THERMOSTAT['limit_C']:g} C", "share_within_limit", ".2f"),
           ("first reaction to cooler (s)", "first_reaction_s", ".0f"),
           ("lab time wasted in cooler failure (s)", "wasted_in_failure_s", ".0f"),
           (f"mean eta {SETTINGS['after_cooler_s'] / 60:g} min after cooler back", "eta_after_cooler", ".1f"),
           ("messages per run", "messages", ".1f"),
           ("new maps while box off target", "maps_while_box_off_target", ".1f"),
           ("checks while box off target", "checks_while_box_off_target", ".1f")]
# Paired comparisons: operators within one PID, and each operator between the PIDs.
PAIRS = [("rules", "never"), ("basic", "rules"), ("full", "rules"), ("full", "basic"), ("team", "full"), ("team", "rules"),
         ("team_no_own_check", "full"), ("team_no_own_check", "team")]
LABELS = {"never": "never", "rules": "rules", "basic": "basic Map agent", "full": "full Map agent",
          "team": "team", "team_no_own_check": "team, no own check"}   # for the plot


def stable_again_s(r: dict) -> float:
    """Lab time when the box is stable again after the cooler failure: the first step after the cooler is back
    with |kalman - target| < swing_band_C on every step of the last stable_s (inf if never)."""
    fail_end = SCENARIO["cooler_fail_at_s"] + SCENARIO["cooler_fail_for_s"]
    inside = np.abs(r["kalman"] - r["target"]) < THERMOSTAT["swing_band_C"]
    for i, t in enumerate(r["t_s"]):
        window = (r["t_s"] >= t - AGENT_LOOP["stable_s"]) & (r["t_s"] <= t)
        if t >= fail_end and inside[window].all():
            return float(t)
    return np.inf


def wasted_in_failure(r: dict) -> float:
    """Lab time (s) of the measure_map and check_map calls that started between the cooler failure and the moment
    the box is stable again (maps measured or checked on a moving box)."""
    start, end = SCENARIO["cooler_fail_at_s"], stable_again_s(r)
    cost = {"measure_map": LAB_API["measure_map_s"], "check_map": LAB_API["check_map_s"]}
    total = 0.0
    for line in r["api"].get_log():
        t_end, text = line.split("s ", 1)
        name = text.split(" ", 1)[0]
        # Log lines carry the end time; the call started cost[name] earlier.
        if name in cost and start <= float(t_end[2:]) - cost[name] < end:
            total += cost[name]
    return total


def run_one(task: tuple[int, str, str, float | None]) -> dict:
    """One scenario run (seed, PID label, operator, time drift tau in min or None); returns its summary plus the first
    reaction to the cooler failure and the other rows of METRICS."""
    seed, pid, operator, tau = task
    box, lab_settings = PIDS[pid], {"time_drift_tau_min": tau}
    if operator in ("never", "rules"):
        r = baseline.run(operator, seed=seed, box_settings=box, lab_settings=lab_settings)
    elif operator in ("team", "team_no_own_check"):
        # The analyst is off here: parallel runs share run names, so its files would overwrite each other.
        switches = {"use_analyst": False, **({"own_stability_check": False} if operator == "team_no_own_check" else {})}
        r = loop.run_team(seed=seed, brain="rules", box_settings=box, switches=switches, lab_settings=lab_settings)
    else:
        switches = {**dict.fromkeys(loop.HELPERS, operator == "full"), "use_fresh_level": True, "use_analyst": False}
        r = loop.run(seed=seed, brain="rules", switches=switches, box_settings=box, lab_settings=lab_settings)
    reaction = loop.first_reaction(r["api"].get_log())
    fail_end = SCENARIO["cooler_fail_at_s"] + SCENARIO["cooler_fail_for_s"]
    after = (r["t_s"] >= fail_end) & (r["t_s"] < fail_end + SETTINGS["after_cooler_s"])
    return {"seed": seed, "pid": pid, "operator": operator, **baseline.summarize(r),
            "first_reaction_s": np.nan if reaction is None else reaction,
            "wasted_in_failure_s": wasted_in_failure(r), "eta_after_cooler": float(r["eta"][after].mean()),
            "messages": float(len(r["messages"])) if "messages" in r else np.nan,
            "checks_while_box_off_target": float(np.sum((r["checked"] > 0) & off_band(r)))}


def off_band(r: dict) -> np.ndarray:
    """Per step (n,): the box is outside target +- swing_band_C (Kalman, as the agents see it) at the step start."""
    return np.abs(r["kalman"] - r["target"]) > THERMOSTAT["swing_band_C"]


FORM_METRICS = [("prediction error (mean |check - predicted|)", "prediction_error", ".4f"),
                ("checks skipped by prediction", "skipped", ".1f"), ("check_map calls", "check_map_calls", ".1f"),
                ("mean eta", "mean_eta", ".2f"), ("lab time on maps (s)", "lab_time_on_maps_s", ".0f")]


def map_agent_switches(operator: str) -> dict:
    """The basic or full Map agent (rules brain, fresh level on, no analyst: parallel runs)."""
    return {**dict.fromkeys(loop.HELPERS, operator == "full"), "use_fresh_level": True, "use_analyst": False}


def calibrate_form(task: tuple[str, float | None]) -> tuple[str, float | None, dict]:
    """The staleness form chosen (staleness.choose, as the data analyst does) on one full Map agent run of a setup
    (PID label, time drift tau) with the calibration seed."""
    pid, tau = task
    state = LabState(":memory:")
    r = loop.run(seed=SETTINGS["form_calibration_seed"], state=state, brain="rules", switches=map_agent_switches("full"),
                 box_settings=PIDS[pid], lab_settings={"time_drift_tau_min": tau}, run_name="calibration")
    choice = staleness.choose(*staleness.pairs(state, r["api"].run_id))
    return pid, tau, choice


def run_form(task: tuple[int, str, str, float | None, str, dict | None]) -> dict:
    """One scenario run of the basic or full Map agent with a predictor form ("gaussian", or "auto" = the form dict
    chosen for the setup); returns the FORM_METRICS."""
    seed, pid, operator, tau, name, form = task
    r = loop.run(seed=seed, brain="rules", switches=map_agent_switches(operator), box_settings=PIDS[pid],
                 lab_settings={"time_drift_tau_min": tau}, prediction_form=form if name == "auto" else "gaussian")
    errors = [float(line.rsplit("error ", 1)[1]) for line in r["api"].get_log() if "predict: predicted" in line
              and ", checked " in line]
    sm = baseline.summarize(r)
    return {"seed": seed, "pid": pid, "operator": operator, "tau": tau, "form": name,
            "prediction_error": float(np.mean(errors)) if errors else np.nan, "skipped": float(r["skipped"]),
            "check_map_calls": float(sm["check_map_calls"]), "mean_eta": sm["mean_eta"],
            "lab_time_on_maps_s": sm["lab_time_on_maps_s"]}


def compare_forms(seeds: list[int], out: Path) -> Path:
    """"gaussian" vs "auto" for the basic and full Map agent over seeds, both PIDs and every time drift; prints and
    saves forms.csv and forms.txt (mean ± SE, paired auto - gaussian with the se_factor rule)."""
    start, lines = time.time(), []
    say = lambda text="": (print(text), lines.append(text))
    taus = SETTINGS["time_drift_taus"]
    with Pool() as pool:
        chosen = {(pid, tau): choice for pid, tau, choice in pool.map(calibrate_form, [(p, t) for p in PIDS for t in taus])}
        say(f"form chosen per setup (one full Map agent run, seed {SETTINGS['form_calibration_seed']}):")
        for (pid, tau), c in chosen.items():
            say(f"  {pid}, time drift {tau or 'none'}: {c['chosen']} (beta {c['form']['beta']:.2f}, {c['n']} checks); "
                "LOO RMSE " + ", ".join(f"{n} {f['loo_rmse']:.4f}" for n, f in c["forms"].items()))
        tasks = [(seed, pid, op, tau, name, chosen[(pid, tau)]["form"] if name == "auto" else None)
                 for seed in seeds for pid in PIDS for tau in taus for op in ("basic", "full") for name in ("gaussian", "auto")]
        rows = pool.map(run_form, tasks)
    say(f"{len(rows)} runs ({len(seeds)} seeds) in {time.time() - start:.0f} s")
    get = lambda pid, tau, op, name, key: np.array([r[key] for r in sorted(rows, key=lambda r: r["seed"])
                                                    if (r["pid"], r["tau"], r["operator"], r["form"]) == (pid, tau, op, name)])
    for pid in PIDS:
        for tau in taus:
            say(f"\n{pid}, time drift {tau or 'none'} (auto = {chosen[(pid, tau)]['chosen']}): mean ± SE over seeds; "
                f"auto - gaussian paired, real only beyond {SETTINGS['se_factor']:g} SE")
            for op in ("basic", "full"):
                for label, key, fmt in FORM_METRICS:
                    g, a = get(pid, tau, op, "gaussian", key), get(pid, tau, op, "auto", key)
                    cell = lambda v: "-" if np.isnan(v).all() else \
                        f"{np.nanmean(v):{fmt}} ± {np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))):{fmt}}"
                    mean, se, n = paired(a, g)
                    diff = "-" if n < 2 else f"{mean:+{fmt}} ± {se:{fmt}}  {verdict(mean, se)}"
                    say(f"  {op:5} {label:44} gaussian {cell(g):>18}  auto {cell(a):>18}  auto - gaussian {diff}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "forms.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["pid"], str(r["tau"]), r["operator"], r["form"], r["seed"])))
    (out / "forms.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {out / 'forms.csv'}, {out / 'forms.txt'}")
    return out / "forms.csv"


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Mean and standard error of a - b over the seeds where both are known; returns (mean, se, n)."""
    d = (a - b)[~np.isnan(a - b)]
    if len(d) < 2:
        return np.nan, np.nan, len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def verdict(mean: float, se: float) -> str:
    """'real' text for a paired difference larger than se_factor standard errors, else 'no clear difference'."""
    if np.isnan(mean):
        return "n/a"
    if se == 0.0 and mean == 0.0:
        return "no clear difference (identical)"
    return "real difference" if abs(mean) > SETTINGS["se_factor"] * se else "no clear difference"


def tag(tau: float | None) -> str:
    """File name part of a time drift setting: "" (none) or "_time_drift_<tau>min"."""
    return "" if tau is None else f"_time_drift_{tau:g}min"


def compare(seeds: list[int], out: Path, tau: float | None = None) -> dict:
    """Run every (seed, PID, operator) with time drift tau (min, None = off), print the tables and save them as
    results<tag>.csv and results<tag>.txt in out.

    Returns: {(pid, operator, metric key): values (n_seeds,)}.
    """
    start = time.time()
    lines = []
    say = lambda text="": (print(text), lines.append(text))
    tasks = [(seed, pid, op, tau) for seed in seeds for pid in PIDS for op in SETTINGS["operators"]]
    with Pool() as pool:
        rows = pool.map(run_one, tasks)
    drift = "no time drift" if tau is None else f"time drift of the sample, tau {tau:g} min"
    say(f"{len(rows)} runs ({len(seeds)} seeds), {drift}, in {time.time() - start:.0f} s")

    values = {}
    for pid in PIDS:
        for op in SETTINGS["operators"]:
            mine = sorted((r for r in rows if r["pid"] == pid and r["operator"] == op), key=lambda r: r["seed"])
            for _, key, _ in METRICS:
                values[(pid, op, key)] = np.array([r[key] for r in mine], dtype=float)

    say(f"\nmean ± std over {len(seeds)} seeds, {drift} (basic / full = basic / full Map agent)")
    cols = [(pid, op) for pid in PIDS for op in SETTINGS["operators"]]
    say(f"{'':38} " + " ".join(pid.center(len(SETTINGS["operators"]) * 15 - 1) for pid in PIDS))
    say(f"{'':38} " + " ".join(op.rjust(14) for _, op in cols))
    for label, key, fmt in METRICS:
        cells = []
        for pid, op in cols:
            v = values[(pid, op, key)]
            cells.append("-" if np.isnan(v).all() else f"{np.nanmean(v):{fmt}}±{np.nanstd(v, ddof=1) if len(v) > 1 else 0:{fmt}}")
        say(f"{label:38} " + " ".join(c.rjust(14) for c in cells))

    say(f"\npaired differences (same seeds): mean ± standard error; real only if > {SETTINGS['se_factor']:g} SE")
    comparisons = [(f"{a} - {b} ({pid.split(' (')[0]})", (pid, a), (pid, b)) for pid in PIDS for a, b in PAIRS]
    if len(PIDS) == 2:
        old, new = PIDS
        comparisons += [(f"{op}: {new.split(' (')[0]} - {old.split(' (')[0]}", (new, op), (old, op))
                        for op in SETTINGS["operators"]]
    for label, key, fmt in METRICS:
        say(f"{label}:")
        for name, (pa, oa), (pb, ob) in comparisons:
            mean, se, n = paired(values[(pa, oa, key)], values[(pb, ob, key)])
            if n < 2:
                continue
            say(f"  {name:34} {mean:+{fmt}} ± {se:{fmt}}  {verdict(mean, se)}")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"results{tag(tau)}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "pid", "operator"] + [key for _, key, _ in METRICS])
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["pid"], r["operator"], r["seed"])):
            writer.writerow({k: r[k] for k in writer.fieldnames})
    (out / f"results{tag(tau)}.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {csv_path}, {out / f'results{tag(tau)}.txt'}")
    return values


def read_values(path: Path) -> dict:
    """{(pid, operator, metric key): values (n_seeds,)} from a saved results<tag>.csv."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {(pid, op, key): np.array([float(r[key]) for r in rows if r["pid"] == pid and r["operator"] == op])
            for pid in PIDS for op in SETTINGS["operators"] for _, key, _ in METRICS}


def mean_and_se(values: dict, n_seeds: int, key: str = "mean_eta") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per PID: (mean (n_ops,), standard error (n_ops,)) of one metric over the seeds."""
    return {pid: (np.array([values[(pid, op, key)].mean() for op in SETTINGS["operators"]]),
                  np.array([values[(pid, op, key)].std(ddof=1) / np.sqrt(n_seeds) for op in SETTINGS["operators"]]))
            for pid in PIDS}


if __name__ == "__main__":
    import sys
    import tempfile

    if sys.argv[1:] == ["forms"]:
        compare_forms(SETTINGS["seeds"], Path(SETTINGS["out_dir"]))
    elif sys.argv[1:] in (["full"], ["plots"]):
        # "plots": redraw the plot from the saved results<tag>.csv files, without running anything.
        out = Path(SETTINGS["out_dir"])
        n = len(SETTINGS["seeds"])
        taus = SETTINGS["time_drift_taus"]
        if sys.argv[1:] == ["full"]:
            values = {tau: compare(SETTINGS["seeds"], out, tau) for tau in taus}
        else:
            values = {tau: read_values(out / f"results{tag(tau)}.csv") for tau in taus}
        name = lambda tau: "no time drift" if tau is None else f"time drift {tau:g} min"
        rows = [(ylabel, fmt, {pid: {name(tau): mean_and_se(values[tau], n, key)[pid] for tau in taus} for pid in PIDS})
                for ylabel, fmt, key in [("mean η", ".1f", "mean_eta"), ("lab time on maps (s)", ".0f", "lab_time_on_maps_s")]]
        print(plots.plot_operator_panels(
            [LABELS[op] for op in SETTINGS["operators"]], rows, out / "01_mean_eta_and_lab_time_per_operator_time_drift.png",
            "Focus brightness and lab time on maps per operator: no time drift vs. "
            + " vs. ".join(f"{t:g} min" for t in taus if t) + f"\n{n} seeds; error bars: standard error of the mean"))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            values = compare(SETTINGS["test_seeds"], Path(tmp))
            assert (Path(tmp) / "results.csv").exists() and (Path(tmp) / "results.txt").exists()
        n = len(SETTINGS["test_seeds"])
        assert all(len(v) == n for v in values.values())
        # Seed 0 with the real box PID is the agents.map_agent.loop run: the same scenario must give the same numbers.
        assert np.isclose(values[(thermostat.pid_label("real"), "rules", "mean_eta")][0], 43.2, atol=0.05)
        print(f"compare self-test ({n} seeds): ok")
