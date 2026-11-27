"""
fit_real_box.py - fit the box model of box_simulation/thermostat.py to the real box test.

Reads the logged CSV from start_clock on, finds the phases from the target column and drives the model
with the real duty and the room temperature (interpolated between the room samples). The model runs
open-loop through the whole test (the hold too, with its real duty), but only the rest, cooling and
warm-up phases count in the fit; the hold is the check. C stays fixed at its config value.

Three models are compared: one mass; one mass with a cooler that gets weaker when the box is colder
(P_max * u * (1 - k * (T_room - T))); and two masses (air + walls and contents, as in thermostat.py).
Then: sensor noise and step size from the rest phase, the heat disturbance that explains the rest-phase residual, the PID loop rate from the hold, and the PID swing
of the real box vs. thermostat.simulate with the fitted numbers and the real PID gains (the simulated true
temperature is compared, since the simulated sensor has spikes that the real log does not show).
Writes results/real_box/fit/ (three plots and summary.json).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis import plots
from box_simulation import thermostat
from box_simulation.thermostat import simulate
from config import FIT_REAL_BOX as SETTINGS
from config import THERMOSTAT

MODELS = {
    "one mass": ["G", "P_internal", "P_max", "tau_cooler"],
    "one mass + weaker cooler": ["G", "P_internal", "P_max", "tau_cooler", "k"],
    "two masses": ["G", "P_internal", "P_max", "tau_cooler", "C_w", "G_w"],
}
LOG_PARAMS = {"tau_cooler", "C_w"}      # always positive: fitted as logarithms
UNITS = {"G": "W/K", "P_internal": "W", "P_max": "W", "tau_cooler": "s", "C_w": "J/K", "G_w": "W/K", "k": "1/K"}


def load(path: str | Path, start_clock: str) -> dict[str, np.ndarray]:
    """CSV rows from start_clock on as arrays (n,): t_s (s since start_clock), T (raw), T_filtered, duty, target,
    T_room (the room sensor's probe in the air, room_probe_temp, or its case sensor room_temp where the probe is empty;
    interpolated linearly between the sample times)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t0 = datetime.fromisoformat(start_clock).timestamp()
    clock = lambda text: datetime.fromisoformat(text).timestamp() - t0
    room_value = lambda r: r["room_probe_temp"] if r["room_probe_temp"] else r["room_temp"]
    room = sorted({(clock(r["room_sample_time"]), float(room_value(r))) for r in rows if room_value(r)})
    rows = [r for r in rows if r["clock_time"] >= start_clock]
    col = lambda k: np.array([float(r[k]) for r in rows])
    t = col("time_s") - col("time_s")[0]
    room_t, room_T = np.array(room).T
    return {"t_s": t, "T": col("temperature"), "T_filtered": col("filtered"), "duty": col("duty"),
            "target": col("target"), "T_room": np.interp(t, room_t, room_T)}


def find_phases(target: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Index arrays of the runs of constant target (n,), named in order by names."""
    runs = np.split(np.arange(len(target)), np.flatnonzero(np.diff(target)) + 1)
    return dict(zip(names, runs))


def simulate_box(p: dict, d: dict) -> dict[str, np.ndarray]:
    """Open-loop model run with the real duty and room temperature of d; returns "T", "T_w", "u" (n,).

    p: G, P_internal, P_max, tau_cooler, and optionally k (weaker cooler) or C_w, G_w (walls).
    One Euler step per logged interval (about 2 s; the slowest box time constants are minutes).
    The box starts settled: walls at the first measured temperature, cooler state 0.
    """
    t, duty, T_room = d["t_s"], d["duty"] / THERMOSTAT["duty_max"], d["T_room"]
    C, k, C_w, G_w = THERMOSTAT["C"], p.get("k", 0.0), p.get("C_w", 1.0), p.get("G_w", 0.0)
    T, T_w, u = np.empty_like(t), np.empty_like(t), np.zeros_like(t)
    T[0] = T_w[0] = d["T"][0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        u[i + 1] = u[i] + (1.0 - np.exp(-dt / p["tau_cooler"])) * (duty[i] - u[i])
        cooler = p["P_max"] * u[i + 1] * (1.0 - k * (T_room[i] - T[i]))
        power = p["G"] * (T_room[i] - T[i]) + p["P_internal"] - cooler + G_w * (T_w[i] - T[i])
        T[i + 1] = T[i] + power / C * dt
        T_w[i + 1] = T_w[i] + G_w * (T[i] - T_w[i]) / C_w * dt
    return {"T": T, "T_w": T_w, "u": u}


def levenberg_marquardt(residuals, x0: np.ndarray, max_iter: int) -> np.ndarray:
    """Least squares: x (k,) that minimises sum(residuals(x)**2), starting at x0 (k,)."""
    x, lam = np.array(x0, float), 1e-2
    r = residuals(x)
    for _ in range(max_iter):
        # Forward-difference Jacobian (m, k).
        h = 1e-4 * np.maximum(np.abs(x), 1e-2)
        J = np.column_stack([(residuals(x + h[j] * np.eye(len(x))[j]) - r) / h[j] for j in range(len(x))])
        A, g = J.T @ J, J.T @ r
        while lam < 1e8:
            dx = -np.linalg.lstsq(A + lam * np.diag(np.diag(A)), g, rcond=None)[0]
            r_new = residuals(x + dx)
            # A step that leaves the physical range gives NaN residuals; it counts as worse.
            if np.all(np.isfinite(r_new)) and r_new @ r_new < r @ r:
                x, r, lam = x + dx, r_new, lam / 3
                break
            lam *= 4
        if lam >= 1e8 or np.max(np.abs(dx)) < 1e-7:
            break
    return x


def fit(d: dict, phases: dict, names: list[str], s: dict = SETTINGS) -> dict:
    """Fit the parameters names (see MODELS) on the fit phases; returns {param: value} (start values from s)."""
    mask = np.zeros(len(d["t_s"]), bool)
    for name in s["fit_phases"]:
        mask[phases[name]] = True
    start = {**s["start"], "k": s["start_k"]}
    to_p = lambda x: {n: np.exp(v) if n in LOG_PARAMS else v for n, v in zip(names, x)}
    residuals = lambda x: (simulate_box(to_p(x), d)["T"] - d["T"])[mask]
    x0 = [np.log(start[n]) if n in LOG_PARAMS else start[n] for n in names]
    return {n: float(v) for n, v in to_p(levenberg_marquardt(residuals, x0, s["max_iter"])).items()}


def rms(x: np.ndarray) -> float:
    """Root mean square of x (n,)."""
    return float(np.sqrt(np.mean(x ** 2)))


def sensor_noise(t: np.ndarray, T: np.ndarray) -> dict:
    """Noise std (C) of T (n,) around a straight-line trend over t (n,), and the smallest step between readings."""
    noise = T - np.polyval(np.polyfit(t, T, 1), t)
    steps = np.abs(np.diff(T))
    return {"sigma_C": float(noise.std()), "step_C": float(np.round(steps[steps > 0].min(), 4))}


def pid_loop_rate(d: dict, hold: np.ndarray, Ki: float, duty_max: int) -> dict:
    """PID loops per second from the hold: d(duty)/dt = Kp * dT/dt + Ki * f * e while the duty is unsaturated.

    A least-squares fit over the unsaturated 2 s intervals (filtered temperature, e = T - target)
    gives Kp and Ki * f; f = (Ki * f) / Ki is the number of times per second the integral adds Ki * e.
    """
    t, T, duty, target = d["t_s"][hold], d["T_filtered"][hold], d["duty"][hold], d["target"][hold]
    ok = (duty[:-1] > 0) & (duty[:-1] < duty_max) & (duty[1:] > 0) & (duty[1:] < duty_max)
    dt = np.diff(t)[ok]
    A = np.column_stack([np.diff(T)[ok] / dt, ((T[:-1] + T[1:]) / 2 - target[:-1])[ok]])
    Kp, Ki_f = np.linalg.lstsq(A, np.diff(duty)[ok] / dt, rcond=None)[0]
    return {"n_intervals": int(ok.sum()), "Kp_fit": float(Kp), "Ki_times_f": float(Ki_f),
            "loops_per_s": float(Ki_f / Ki)}


def swing(t: np.ndarray, T: np.ndarray, target: float, smooth_s: float, edge_s: float, band: float) -> dict:
    """PID swing of T (n,) over t (n,) in s: the highest point of each run above target and the lowest of each
    run below it (after a smooth_s moving average); peaks within edge_s of the start or end are not counted.
    A run changes side only when the smoothed T passes target +- band, so noise near the target is no new run.

    Returns the peak lists (time_s, C), their means and the period (mean spacing of the maxima and of the minima).
    """
    n = max(1, int(round(smooth_s / np.median(np.diff(t)))))
    window = np.ones(n)
    smooth = np.convolve(T, window, mode="same") / np.convolve(np.ones_like(T), window, mode="same")
    side = np.empty(len(t), bool)
    side[0] = smooth[0] > target
    for i in range(1, len(t)):
        side[i] = smooth[i] > target + band or (side[i - 1] and smooth[i] >= target - band)
    runs = np.split(np.arange(len(t)), np.flatnonzero(np.diff(side)) + 1)
    maxima, minima = [], []
    for run in runs:
        above = side[run[0]]
        i = run[np.argmax(smooth[run]) if above else np.argmin(smooth[run])]
        if t[0] + edge_s <= t[i] <= t[-1] - edge_s:
            (maxima if above else minima).append((float(t[i]), float(smooth[i])))
    gaps = [b[0] - a[0] for peaks in (maxima, minima) for a, b in zip(peaks, peaks[1:])]
    return {"maxima": maxima, "minima": minima,
            "mean_max_C": float(np.mean([v for _, v in maxima])) if maxima else None,
            "mean_min_C": float(np.mean([v for _, v in minima])) if minima else None,
            "period_s": float(np.mean(gaps)) if gaps else None}


def simulate_hold(p: dict, d: dict, hold: np.ndarray, state: dict, sigma: float, duration_s: float,
                  seed: int, freeze: bool) -> dict:
    """thermostat.simulate of the hold with the fitted box p and the measured sensor noise sigma (C; Kalman R = sigma^2):
    starts at the measured temperature, with the walls and the cooler state of the fitted model at the hold start
    (state: "T_w", "u" arrays), room = mean over the hold. freeze: freeze_integral_when_saturated of the PID."""
    i = hold[0]
    s = {**THERMOSTAT, **p, "T_room": float(d["T_room"][hold].mean()), "sensor_std": sigma, "R": sigma ** 2,
         "freeze_integral_when_saturated": freeze}
    return thermostat.simulate(duration_s, control=True, T_start=float(d["T"][i]), target=float(d["target"][i]),
                               settings=s, seed=seed, T_w_start=float(state["T_w"][i]), u_start=float(state["u"][i]))


def disturbance_std(p: dict, sigma: float, rest_rms: float, duration_s: float, T_start: float, T_room: float,
                    seeds: list[int]) -> dict:
    """Heat disturbance P_dist_std (W) that explains the rest-phase fit residual: rest_rms^2 = sigma^2 + (k * P_dist_std)^2.

    k (C per W) is the RMS change of the true temperature that a 1 W disturbance (P_dist_tau_s as in config)
    causes over duration_s with the cooler off, from runs with and without it (fitted box p, room T_room, mean over
    seeds). The model is linear in the disturbance, so k * P_dist_std is its effect for any P_dist_std.
    """
    s = {**THERMOSTAT, **p, "T_room": T_room, "P_dist_std": 1.0}
    runs = [simulate(duration_s, control=False, T_start=T_start, settings=s, seed=seed)["T_true"]
            - simulate(duration_s, control=False, T_start=T_start, settings={**s, "P_dist_std": 0.0}, seed=seed)["T_true"]
            for seed in seeds]
    k = rms(np.concatenate(runs))
    return {"C_per_W": k, "rest_rms_C": rest_rms, "sigma_C": sigma,
            "P_dist_std_W": float(np.sqrt(max(rest_rms ** 2 - sigma ** 2, 0.0)) / k)}


SWING_KEYS = ["period_s", "mean_max_C", "mean_min_C"]


def mean_swing(runs: list[dict]) -> dict:
    """Mean period, mean max and mean min over the swing dicts in runs (missing values left out)."""
    return {k: float(np.mean([r[k] for r in runs if r[k] is not None])) for k in SWING_KEYS}


def hold20_check(p: dict, sigma: float, swing_args: tuple, s: dict = SETTINGS) -> dict:
    """The real 20 C hold vs. thermostat.simulate with the fitted box p, the real PID and sensor noise sigma (C).

    Same start as the real test: the measured temperature at hold20_start_clock with the walls at it (a settled box),
    cooler off, room hold20_room_C. Returns the data, the simulated runs (one per seed) and the swing of both.
    """
    d = load(s["hold20_csv"], s["hold20_start_clock"])
    target = float(d["target"][0])
    st = {**THERMOSTAT, **p, "T_room": s["hold20_room_C"], "sensor_std": sigma, "R": sigma ** 2}
    sims = [simulate(d["t_s"][-1], control=True, T_start=float(d["T"][0]), target=target, settings=st, seed=seed)
            for seed in s["seeds"]]
    return {"data": d, "sims": sims, "target": target, "measured": swing(d["t_s"], d["T"], target, *swing_args),
            "simulated": mean_swing([swing(x["t_s"], x["T_true"], target, *swing_args) for x in sims])}


def main(s: dict = SETTINGS) -> dict:
    """Run the whole fit and the checks; save the plots and summary.json in out_dir; returns the summary."""
    out = Path(s["out_dir"])
    d = load(s["csv"], s["start_clock"])
    phases = find_phases(d["target"], s["phases"])
    hold = phases["hold"]
    fit_mask = np.isin(np.arange(len(d["t_s"])), np.concatenate([phases[n] for n in s["fit_phases"]]))

    models = {}
    for model, names in MODELS.items():
        p = fit(d, phases, names, s)
        run = simulate_box(p, d)
        err = run["T"] - d["T"]
        models[model] = {"params": p, "run": run, "err": err, "rms_fit_C": rms(err[fit_mask]),
                         "rms_hold_open_loop_C": rms(err[hold]),
                         "rms_by_phase_C": {n: rms(err[i]) for n, i in phases.items()}}
        print(f"{model:26s} fit RMS {models[model]['rms_fit_C']:.4f} C, hold (not fitted) "
              f"{models[model]['rms_hold_open_loop_C']:.4f} C, "
              + ", ".join(f"{n} {v:.4g} {UNITS[n]}" for n, v in p.items()))
    best = models["two masses"]
    p = best["params"]

    rest = phases["rest"]
    noise = sensor_noise(d["t_s"][rest], d["T"][rest])
    print(f"sensor (rest phase, linear trend removed): sigma {noise['sigma_C']:.4f} C, "
          f"step {noise['step_C']:.2f} C, R = sigma^2 = {noise['sigma_C'] ** 2:.3g} C^2")
    dist = disturbance_std(p, noise["sigma_C"], best["rms_by_phase_C"]["rest"], d["t_s"][rest[-1]] - d["t_s"][rest[0]],
                           float(d["T"][rest[0]]), float(d["T_room"][rest].mean()), s["dist_seeds"])
    print(f"heat disturbance: rest RMS {dist['rest_rms_C']:.4f} C, sensor {dist['sigma_C']:.4f} C, "
          f"1 W gives {dist['C_per_W']:.4f} C -> P_dist_std {dist['P_dist_std_W']:.2f} W")
    loop = pid_loop_rate(d, hold, THERMOSTAT["Ki"], THERMOSTAT["duty_max"])
    print(f"PID from the hold ({loop['n_intervals']} unsaturated intervals): Ki * f = {loop['Ki_times_f']:.1f} "
          f"per C*s -> {loop['loops_per_s']:.1f} integral updates per s (Ki {THERMOSTAT['Ki']:g}); "
          f"Kp {loop['Kp_fit']:.0f} (set {THERMOSTAT['Kp']:g})")

    t_hold = d["t_s"][hold] - d["t_s"][hold][0]
    target = float(d["target"][hold][0])
    swing_args = (s["swing_smooth_s"], s["swing_edge_s"], s["swing_band_C"])
    measured = swing(t_hold, d["T"][hold], target, *swing_args)
    print(f"16.5 C hold measured: period {measured['period_s']:.0f} s, max {measured['mean_max_C']:.2f} C, "
          f"min {measured['mean_min_C']:.2f} C, peaks {measured['maxima'] + measured['minima']}")
    compares, sims = {}, None
    # Old PID model (integral frozen while saturated) and the real firmware (integral keeps counting).
    for label, freeze in [("old, integral frozen", True), ("new, integral counts on", False)]:
        runs = [simulate_hold(p, d, hold, best["run"], noise["sigma_C"], t_hold[-1], seed, freeze) for seed in s["seeds"]]
        longs = [simulate_hold(p, d, hold, best["run"], noise["sigma_C"], s["hold_long_s"], seed, freeze)
                 for seed in s["seeds"]]
        compares[label] = {
            "measured": measured,
            "simulated_same_length": mean_swing([swing(x["t_s"], x["T_true"], target, *swing_args) for x in runs]),
            "simulated_long": mean_swing([swing(x["t_s"], x["T_true"], target, *swing_args) for x in longs]),
        }
        sims = runs
        for key in ["simulated_same_length", "simulated_long"]:
            c = compares[label][key]
            print(f"16.5 C hold {label}: {key} (mean of {len(s['seeds'])} seeds): period {c['period_s']:.0f} s, "
                  f"max {c['mean_max_C']:.2f} C, min {c['mean_min_C']:.2f} C")
    compare = compares["new, integral counts on"]

    h20 = hold20_check(p, noise["sigma_C"], swing_args, s)
    m20, s20 = h20["measured"], h20["simulated"]
    for label, w in [("measured", m20), (f"simulated (mean of {len(s['seeds'])} seeds)", s20)]:
        print(f"20 C hold {label}: min {w['mean_min_C']:.2f} C, max {w['mean_max_C']:.2f} C, "
              f"period {w['period_s'] / 60:.1f} min")
    print(f"20 C hold simulated - measured: min {s20['mean_min_C'] - m20['mean_min_C']:+.3f} C, "
          f"max {s20['mean_max_C'] - m20['mean_max_C']:+.3f} C, period {(s20['period_s'] - m20['period_s']) / 60:+.1f} min")

    t_min = d["t_s"] / 60.0
    spans = [(t_min[i[0]], t_min[i[-1]], n) for n, i in phases.items()]
    paths = [
        plots.plot_fit_overview(t_min, d["T"], best["run"]["T"], 100.0 * d["duty"] / THERMOSTAT["duty_max"],
                                100.0 * best["run"]["u"], spans, out / "01_measured_vs_fitted_temperature.png",
                                f"Real box test 2026-11-18: measured vs. fitted box temperature "
                                f"(fit RMS {best['rms_fit_C']:.3f} °C, hold not fitted)"),
        plots.plot_fit_error(t_min, {m: v["err"] for m, v in models.items()}, spans,
                             out / "02_fit_error_over_time.png",
                             "Fit error over time (model - measured), three box models"),
        plots.plot_hold_swing(t_hold / 60.0, d["T"][hold], sims[0]["t_s"] / 60.0, sims[0]["T_true"], target,
                              out / "03_pid_hold_measured_vs_simulated.png",
                              f"PID hold at {target:g} °C: real box vs. simulated box (Kp {THERMOSTAT['Kp']:g}, "
                              f"Ki {THERMOSTAT['Ki']:g}, Kd {THERMOSTAT['Kd']:g})"),
        plots.plot_temperature_and_duty(
            {"measured (real box)": (h20["data"]["t_s"] / 60.0, h20["data"]["T"],
                                     100.0 * h20["data"]["duty"] / THERMOSTAT["duty_max"]),
             "simulated (fitted box + real PID)": (h20["sims"][0]["t_s"] / 60.0, h20["sims"][0]["T_true"],
                                                   100.0 * h20["sims"][0]["duty"] / THERMOSTAT["duty_max"])},
            (h20["data"]["t_s"] / 60.0, h20["data"]["target"]), out / "04_hold_20C_measured_vs_simulated.png",
            f"PID hold at 20 °C, room {s['hold20_room_C']:g} °C: real box vs. simulated box "
            f"(Kp {THERMOSTAT['Kp']:g}, Ki {THERMOSTAT['Ki']:g}, Kd {THERMOSTAT['Kd']:g})",
            colors={"measured (real box)": "black", "simulated (fitted box + real PID)": "tab:orange"}),
    ]

    summary = {
        "data": s["csv"], "start_clock": s["start_clock"],
        "phases_s": {n: [float(d["t_s"][i[0]]), float(d["t_s"][i[-1]])] for n, i in phases.items()},
        "C_fixed_J_per_K": THERMOSTAT["C"],
        "units": UNITS,
        "models": {m: {k: v[k] for k in ["params", "rms_fit_C", "rms_hold_open_loop_C", "rms_by_phase_C"]}
                   for m, v in models.items()},
        "chosen_model": "two masses",
        "sensor": {**noise, "R_C2": noise["sigma_C"] ** 2},
        "heat_disturbance": dist,
        "pid_loop_rate": loop,
        "swing": compare,
        "swing_old_integral_frozen": compares["old, integral frozen"],
        "hold20": {"data": s["hold20_csv"], "measured": {k: m20[k] for k in SWING_KEYS}, "simulated": s20},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for path in [*paths, out / "summary.json"]:
        print(path)
    return summary


if __name__ == "__main__":
    main()
