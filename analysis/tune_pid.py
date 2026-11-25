"""
tune_pid.py - PID search on the simulated box (fitted to the real box), with robustness and a refinement.

Every run is thermostat.simulate with the THERMOSTAT settings and other gains, so the same firmware rules as the
real PID (20 Hz, sum without dt, clip, integral not frozen while saturated) and readings in 0.01 C steps; the
Kalman filter's box model is off the fit by kalman_model_factors. Cases, 60 min each, one seed:
(a) hold 20 C, room 21.1 C; (b) hold 20 C, room 23.8 C, both after a settling warm-up, scored after 15 min by the
RMS of (T - 20), the share within +-limit_C and max |T - 20| (true temperature); (d) like the real test 3: box at
20.53 C, walls at it, cooler off, room 21.1 C, target 20 C: lowest temperature and the time until the box stays
within +-limit_C for good.
Cooler term: the std of the duty step from one loop to the next (% of duty_max, after 15 min) times
duty_noise_weight, so a nervous cooler costs score.
1. Grid over Kp and Ki (Kd fixed); grid score = the worse RMS of (a) and (b) + the cooler term (worse of a and b).
2. Robustness for the best grid PIDs: (a) and (b) again with P_max or tau_cooler of the true box changed (the Kalman
   model stays as it is); score = the worst RMS of these 10 runs + the cooler term (worst of the 10 runs).
3. Refinement: Nelder-Mead on (log Kp, log Ki) with this score from the best robust PID (own numpy version,
   since the project uses numpy and matplotlib only), within Kp_range and Ki_range, then rounded to a whole Kp
   and Ki with 2 decimals.
All runs are spread over the CPU cores (multiprocessing). THERMOSTAT is not changed.
Writes results/real_box/pid_tuning/ (results.csv, the case (d) plot and the grid-score heat map).
"""

from __future__ import annotations

import itertools
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analysis import plots
from analysis.fit_real_box import load
from box_simulation.thermostat import simulate
from config import FIT_REAL_BOX
from config import PID_TUNING as SETTINGS
from config import THERMOSTAT


def run(task: tuple, s: dict = SETTINGS) -> dict[str, np.ndarray]:
    """thermostat.simulate of one task (gains (Kp, Ki, Kd), case "a", "b" or "d", change None or (setting, factor)).

    change scales one setting of the true box only, from the start of the warm-up on.
    """
    (Kp, Ki, Kd), case, change = task
    settings = {**THERMOSTAT, "Kp": Kp, "Ki": Ki, "Kd": Kd, "kalman_model_factors": s["kalman_model_factors"]}
    if case == "d":
        return simulate(s["duration_s"], control=True, T_start=s["d_T_start"], target=s["target"],
                        settings={**settings, "T_room": s["d_room_C"]}, seed=s["seed"])
    changes = [] if change is None else [(-s["warmup_s"], change[0], THERMOSTAT[change[0]] * change[1])]
    return simulate(s["duration_s"], control=True, T_start=s["target"], target=s["target"],
                    settings={**settings, "T_room": dict(s["holds"])[case]}, seed=s["seed"], warmup_s=s["warmup_s"],
                    box_changes=changes)


def settle_time(t: np.ndarray, T: np.ndarray, target: float, limit: float) -> float:
    """Time (s) after which T (n,) over t (n,) stays within target +- limit; NaN if it is outside at the end."""
    outside = np.flatnonzero(np.abs(T - target) > limit)
    if outside.size == 0:
        return 0.0
    return float("nan") if outside[-1] == len(t) - 1 else float(t[outside[-1] + 1])


def metrics(task: tuple, s: dict = SETTINGS) -> dict[str, float]:
    """Scores of one task: hold cases rms, share, max and duty_noise (std of the duty step per loop, % of duty_max),
    all after score_from_s; case d lowest temperature and settle time."""
    r = run(task)
    if task[1] == "d":
        return {"lowest": float(r["T_true"].min()),
                "settle_s": settle_time(r["t_s"], r["T_true"], s["target"], THERMOSTAT["limit_C"])}
    keep = r["t_s"] >= s["score_from_s"]
    err = (r["T_true"] - r["target"])[keep]
    return {"rms": float(np.sqrt(np.mean(err ** 2))), "share": float(np.mean(np.abs(err) <= THERMOSTAT["limit_C"])),
            "max": float(np.abs(err).max()),
            "duty_noise": float(np.std(np.diff(r["duty"][keep])) / THERMOSTAT["duty_max"] * 100.0)}


def score_of(runs: list[dict], s: dict = SETTINGS) -> tuple[float, float]:
    """(worst RMS, cooler term) of a list of hold-case metrics; the score is their sum."""
    return max(r["rms"] for r in runs), s["duty_noise_weight"] * max(r["duty_noise"] for r in runs)


def robust_tasks(gains: tuple, s: dict = SETTINGS) -> list[tuple]:
    """The 8 robustness tasks of one PID: cases a and b for each changed box."""
    return [(gains, case, change) for change in s["robust_factors"] for case, _ in s["holds"]]


def full_tasks(gains: tuple, s: dict = SETTINGS) -> list[tuple]:
    """All 11 tasks of one PID: a, b, d and the 8 robustness runs."""
    return [(gains, "a", None), (gains, "b", None), (gains, "d", None)] + robust_tasks(gains, s)


def row(gains: tuple, m: dict[tuple, dict]) -> dict:
    """Table row of one PID from the metrics m of its tasks (task -> metrics)."""
    a, b, d = (m[(gains, c, None)] for c in "abd")
    robust = [m[t] for t in robust_tasks(gains)]
    rms, cooler = score_of([a, b] + robust)
    return {"Kp": gains[0], "Ki": gains[1], "Kd": gains[2], "rms_a": a["rms"], "rms_b": b["rms"],
            "share_a": a["share"], "share_b": b["share"], "robust_rms": rms, "cooler_term": cooler,
            "score": rms + cooler, "robust_max": max(x["max"] for x in robust),
            "d_lowest": d["lowest"], "d_settle_s": d["settle_s"]}


class Budget(Exception):
    """Raised when the Nelder-Mead evaluation budget is used up."""


def nelder_mead(f, x0: np.ndarray, step: float, max_evals: int) -> dict:
    """Minimise f over x (k,) with the Nelder-Mead simplex (reflection 1, expansion 2, contraction 0.5,
    shrink 0.5), start simplex x0 and x0 + step along each axis, at most max_evals calls of f.

    Returns "x", "f" (best), "x0", "f0", "evals".
    """
    calls = []

    def g(x):
        if len(calls) >= max_evals:
            raise Budget
        calls.append((x, f(x)))
        return calls[-1][1]

    simplex = [np.array(x0, float)] + [np.array(x0, float) + step * e for e in np.eye(len(x0))]
    try:
        values = [g(x) for x in simplex]
        while True:
            order = np.argsort(values)
            simplex, values = [simplex[i] for i in order], [values[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = g(xr)
            if values[0] <= fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            elif fr < values[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = g(xe)
                simplex[-1], values[-1] = (xe, fe) if fe < fr else (xr, fr)
            else:
                # Contraction outside (reflected point better than the worst) or inside; else shrink to the best.
                xc = centroid + 0.5 * ((xr if fr < values[-1] else simplex[-1]) - centroid)
                fc = g(xc)
                if fc < min(fr, values[-1]):
                    simplex[-1], values[-1] = xc, fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = g(simplex[i])
    except Budget:
        pass
    best = min(calls, key=lambda c: c[1])
    return {"x": best[0], "f": best[1], "x0": np.array(x0, float), "f0": calls[0][1], "evals": len(calls)}


def load_test3(s: dict = SETTINGS) -> dict[str, np.ndarray]:
    """The real test 3 from the target change until FIT_REAL_BOX hold20_new_pid_end_clock (the cooler cable came loose
    after that); t_s from the target change."""
    d = load(FIT_REAL_BOX["hold20_new_pid_csv"], s["test3_from_clock"])
    end = FIT_REAL_BOX["hold20_new_pid_end_clock"]
    until = (datetime.fromisoformat(end) - datetime.fromisoformat(s["test3_from_clock"])).total_seconds()
    keep = d["t_s"] <= until
    return {k: v[keep] for k, v in d.items()}


def main(s: dict = SETTINGS) -> list[dict]:
    """Grid, robustness, refinement; prints the table, saves results.csv and the two plots; returns the table rows."""
    start = time.time()
    out = Path(s["out_dir"])
    Kd = float(s["Kd"])
    grid = [(float(kp), float(ki), Kd) for kp, ki in itertools.product(s["Kp"], s["Ki"])]
    m = {}
    with Pool() as pool:
        evaluate = lambda tasks: m.update(zip(tasks, pool.map(metrics, tasks)))

        evaluate([(g, c, None) for g in grid for c in "abd"])
        grid_score = {g: sum(score_of([m[(g, "a", None)], m[(g, "b", None)]], s)) for g in grid}
        top = sorted(grid, key=grid_score.get)[:s["n_robust"]]
        print(f"grid: {len(grid)} PIDs done, best grid score {grid_score[top[0]]:.4f} C at Kp {top[0][0]:g}, "
              f"Ki {top[0][1]:g} ({time.time() - start:.0f} s)", flush=True)
        evaluate([t for g in top for t in robust_tasks(g, s)])
        rows = sorted((row(g, m) for g in top), key=lambda r: r["score"])
        print(f"robustness: {len(top)} PIDs done, best score {rows[0]['score']:.4f} C at Kp {rows[0]['Kp']:g}, "
              f"Ki {rows[0]['Ki']:g} ({time.time() - start:.0f} s)", flush=True)

        # Nelder-Mead works on (log Kp, log Ki), limited to Kp_range and Ki_range.
        lo, hi = np.log([s["Kp_range"][0], s["Ki_range"][0]]), np.log([s["Kp_range"][1], s["Ki_range"][1]])
        gains_of = lambda x: (*map(float, np.exp(np.clip(x, lo, hi))), Kd)

        def score(x):
            gains = gains_of(x)
            tasks = [(gains, c, None) for c, _ in s["holds"]] + robust_tasks(gains, s)
            rms, cooler = score_of(pool.map(metrics, tasks), s)
            print(f"  Nelder-Mead evaluation: Kp {gains[0]:.1f}, Ki {gains[1]:.3f}: RMS {rms:.4f} + cooler {cooler:.4f} "
                  f"= {rms + cooler:.4f} C ({time.time() - start:.0f} s)", flush=True)
            return rms + cooler

        x0 = np.log([rows[0]["Kp"], rows[0]["Ki"]])
        nm = nelder_mead(score, x0, s["nm_step"], s["nm_max_evals"])
        end = gains_of(nm["x"])
        refined = (float(np.round(end[0])), float(np.round(end[1], 2)), Kd)
        extra = [refined] + [tuple(map(float, g)) for g in s["compare"]]
        evaluate([t for g in extra for t in full_tasks(g, s) if t not in m])
    print(f"Nelder-Mead: start Kp {rows[0]['Kp']:g}, Ki {rows[0]['Ki']:g} (score {nm['f0']:.4f} C) "
          f"-> end Kp {end[0]:.1f}, Ki {end[1]:.3f} (score {nm['f']:.4f} C), "
          f"{nm['evals']} evaluations; rounded to Kp {refined[0]:g}, Ki {refined[1]:g}", flush=True)

    labels = [f"robust {i + 1}" for i in range(len(rows))] + ["refined", "test 3 (800/0.5)", "old (200/4)"]
    rows = rows + [row(g, m) for g in extra]
    lim = THERMOSTAT["limit_C"]
    print(f"{'':17}{'Kp':>6}{'Ki':>6} | {'RMS a':>7}{'RMS b':>7} | {'in±' + format(lim, 'g') + ' a':>9}{'b':>5} | "
          f"{'rob RMS':>8}{'cooler':>8}{'score':>8}{'rob max':>8} | {'d lowest':>9}{'d settle':>9}")
    for label, r in zip(labels, rows):
        settle = "never" if np.isnan(r["d_settle_s"]) else f"{r['d_settle_s'] / 60:.1f} min"
        print(f"{label:17}{r['Kp']:6g}{r['Ki']:6g} | {r['rms_a']:7.4f}{r['rms_b']:7.4f} | {r['share_a']:9.2f}"
              f"{r['share_b']:5.2f} | {r['robust_rms']:8.4f}{r['cooler_term']:8.4f}{r['score']:8.4f}"
              f"{r['robust_max']:8.3f} | {r['d_lowest']:9.3f}{settle:>9}")
    t3 = load_test3(s)
    print(f"measured test 3 (800/0.5, until {FIT_REAL_BOX['hold20_new_pid_end_clock'][11:]}): lowest {t3['T'].min():.2f} C, "
          f"settle (raw sensor, ±{lim:g} C) "
          + (lambda v: "not within the record" if np.isnan(v) else f"{v / 60:.1f} min")(
              settle_time(t3["t_s"], t3["T"], s["target"], lim)))

    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    (out / "results.csv").write_text("label," + ",".join(keys) + "\n" + "\n".join(
        label + "," + ",".join(f"{r[k]:.6g}" for k in keys) for label, r in zip(labels, rows)) + "\n")

    runs = {f"simulated, refined: Kp {refined[0]:g}, Ki {refined[1]:g}": run((refined, "d", None)),
            "simulated, test 3: Kp 800, Ki 0.5": run(((800.0, 0.5, Kd), "d", None))}
    curves = {k: (r["t_s"] / 60.0, r["T_true"], 100.0 * r["duty"] / THERMOSTAT["duty_max"]) for k, r in runs.items()}
    curves["measured test 3: Kp 800, Ki 0.5"] = (t3["t_s"] / 60.0, t3["T"], 100.0 * t3["duty"] / THERMOSTAT["duty_max"])
    first = next(iter(runs.values()))
    score_map = np.array([[grid_score[(float(kp), float(ki), Kd)] for kp in s["Kp"]] for ki in s["Ki"]])
    paths = [
        plots.plot_temperature_and_duty(
            curves, (first["t_s"] / 60.0, first["target"]), out / "01_case_d_start_like_test3.png",
            f"Case d (box {s['d_T_start']:g} °C, cooler off, target {s['target']:g} °C, room {s['d_room_C']:g} °C): "
            f"refined vs. test 3 PID",
            colors=dict(zip(curves, ["tab:blue", "tab:orange", "black"]))),
        plots.plot_score_map(np.array(s["Kp"], float), np.array(s["Ki"], float), score_map, refined[:2],
                             f"refined: Kp {refined[0]:g}, Ki {refined[1]:g}", out / "02_grid_score_heat_map.png",
                             f"PID grid score (worse RMS of cases a and b + cooler term, Kd {Kd:g})"),
    ]
    for path in [*paths, out / "results.csv"]:
        print(path)
    print(f"total run time {time.time() - start:.0f} s")
    return rows


if __name__ == "__main__":
    main()
