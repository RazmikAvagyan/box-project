"""
compare.py - the operators over many seeds: never, rules, the basic and the full Map agent (rules brain, fresh level on;
the full one with watchdog, safety check and prediction),
each with the box PID of today (THERMOSTAT) and with the refined PID (PID_TUNING["refined"], an override only).

Each run is the 2-hour scenario with LabAPI seed s (lab, sample, sensor and GGS seeds all shift with it).
For each operator and PID: mean ± std over seeds. Differences are paired by seed (same lab, same sample):
a difference counts as real only if |mean difference| > se_factor standard errors of the paired differences,
otherwise "no clear difference". All runs are spread over the CPU cores (multiprocessing), as in tune_pid.py.

python -m experiments.compare        self-test: test_seeds only, output in a temporary folder
python -m experiments.compare full   all seeds, writes results/compare/
"""

from __future__ import annotations

import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from agents import baseline, loop
from analysis import plots
from config import COMPARE as SETTINGS
from config import PID_TUNING, THERMOSTAT

PIDS = {"PID 200/4": None, "PID 3000/0.36": dict(zip(("Kp", "Ki", "Kd"), PID_TUNING["refined"]))}
METRICS = [("mean eta", "mean_eta", ".1f"), ("min eta", "min_eta", ".1f"), ("measure_map calls", "measure_map_calls", ".1f"),
           ("lab time on maps (s)", "lab_time_on_maps_s", ".0f"),
           (f"share within ±{THERMOSTAT['limit_C']:g} C", "share_within_limit", ".2f"),
           ("first reaction to cooler (s)", "first_reaction_s", ".0f")]
# Paired comparisons: operators within one PID, and each operator between the PIDs.
PAIRS = [("rules", "never"), ("basic", "rules"), ("full", "rules"), ("full", "basic")]
LABELS = {"never": "never", "rules": "rules", "basic": "basic Map agent", "full": "full Map agent"}   # for the plot


def run_one(task: tuple[int, str, str]) -> dict:
    """One scenario run (seed, PID label, operator); returns its summary plus the first reaction to the cooler failure."""
    seed, pid, operator = task
    box = PIDS[pid]
    if operator in ("never", "rules"):
        r = baseline.run(operator, seed=seed, box_settings=box)
    else:
        switches = {**dict.fromkeys(loop.HELPERS, operator == "full"), "use_fresh_level": True}
        r = loop.run(seed=seed, brain="rules", switches=switches, box_settings=box)
    reaction = loop.first_reaction(r["api"].get_log())
    return {"seed": seed, "pid": pid, "operator": operator, **baseline.summarize(r),
            "first_reaction_s": np.nan if reaction is None else reaction}


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


def compare(seeds: list[int], out: Path) -> dict:
    """Run every (seed, PID, operator), print the tables, save results.csv and the plot in out.

    Returns: {(pid, operator, metric key): values (n_seeds,)}.
    """
    start = time.time()
    tasks = [(seed, pid, op) for seed in seeds for pid in PIDS for op in SETTINGS["operators"]]
    with Pool() as pool:
        rows = pool.map(run_one, tasks)
    print(f"{len(rows)} runs ({len(seeds)} seeds) in {time.time() - start:.0f} s")

    values = {}
    for pid in PIDS:
        for op in SETTINGS["operators"]:
            mine = sorted((r for r in rows if r["pid"] == pid and r["operator"] == op), key=lambda r: r["seed"])
            for _, key, _ in METRICS:
                values[(pid, op, key)] = np.array([r[key] for r in mine], dtype=float)

    print(f"\nmean ± std over {len(seeds)} seeds (basic / full = basic / full Map agent)")
    cols = [(pid, op) for pid in PIDS for op in SETTINGS["operators"]]
    print(f"{'':30} " + " ".join(pid.center(4 * 14 + 3) for pid in PIDS))
    print(f"{'':30} " + " ".join(op.rjust(14) for _, op in cols))
    for label, key, fmt in METRICS:
        cells = []
        for pid, op in cols:
            v = values[(pid, op, key)]
            cells.append("-" if np.isnan(v).all() else f"{np.nanmean(v):{fmt}}±{np.nanstd(v, ddof=1) if len(v) > 1 else 0:{fmt}}")
        print(f"{label:30} " + " ".join(c.rjust(14) for c in cells))

    print(f"\npaired differences (same seeds): mean ± standard error; real only if > {SETTINGS['se_factor']:g} SE")
    comparisons = [(f"{a} - {b} ({pid})", (pid, a), (pid, b)) for pid in PIDS for a, b in PAIRS]
    old, new = PIDS
    comparisons += [(f"{op}: {new} - {old}", (new, op), (old, op)) for op in SETTINGS["operators"]]
    for label, key, fmt in METRICS:
        print(f"{label}:")
        for name, (pa, oa), (pb, ob) in comparisons:
            mean, se, n = paired(values[(pa, oa, key)], values[(pb, ob, key)])
            if n < 2:
                continue
            print(f"  {name:32} {mean:+{fmt}} ± {se:{fmt}}  {verdict(mean, se)}")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "pid", "operator"] + [key for _, key, _ in METRICS])
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["pid"], r["operator"], r["seed"])):
            writer.writerow({k: r[k] for k in writer.fieldnames})
    groups = {pid: (np.array([values[(pid, op, "mean_eta")].mean() for op in SETTINGS["operators"]]),
                    np.array([values[(pid, op, "mean_eta")].std(ddof=1) for op in SETTINGS["operators"]]))
              for pid in PIDS}
    plot = plots.plot_operator_comparison(
        [LABELS[op] for op in SETTINGS["operators"]], groups, out / "01_mean_eta_per_operator_old_vs_tuned_pid.png",
        f"Mean focus brightness per operator over {len(seeds)} seeds (error bars: ±1 std)", "mean η over the scenario")
    print(f"saved {csv_path} and {plot}")
    return values


if __name__ == "__main__":
    import sys
    import tempfile

    if sys.argv[1:] == ["full"]:
        compare(SETTINGS["seeds"], Path(SETTINGS["out_dir"]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            values = compare(SETTINGS["test_seeds"], Path(tmp))
            assert (Path(tmp) / "results.csv").exists()
        n = len(SETTINGS["test_seeds"])
        assert all(len(v) == n for v in values.values())
        # Seed 0 with the old PID is the agents.loop run: the same scenario must give the same numbers.
        assert np.isclose(values[("PID 200/4", "rules", "mean_eta")][0], 43.2, atol=0.05)
        print(f"compare self-test ({n} seeds): ok")
