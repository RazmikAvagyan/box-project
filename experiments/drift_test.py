"""
drift_test.py - does the map still work while the temperature changes?

The map X_est is measured once at t = 0, then for 5 minutes the same test
patterns are shown and the camera photos are compared with what X_est predicts.
Two cases, both from the box simulation (box_simulation.thermostat): without
temperature control (the cooler is off and the box warms up) and with
temperature control (PID + Kalman thermostat, already settled by a warm-up).

The sample at temperature T is lab.sample_at_temperature(X0, Y, T - T(t=0), T_c).
Row convention (as in lab.py): E_out = E_in @ X, (L x M) = (L x N) @ (N x M).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import plots
from box_simulation import thermostat
from config import DRIFT_TEST as SETTINGS
from experiments import main
from optics_simulation import lab, metrics

CASES = {"no control": "no control (cooler off)", "with control": "with PID + Kalman control"}
COLORS = {"no control (cooler off)": "red", "with PID + Kalman control": "blue"}


def box_runs(s: dict) -> dict[str, dict[str, np.ndarray]]:
    """Box simulation (20 Hz) for both cases; each value is the dict from thermostat.simulate."""
    duration_s = s["duration_min"] * 60
    return {
        "no control": thermostat.simulate(duration_s, control=False, T_start=s["T0"], seed=s["box_seed"]),
        "with control": thermostat.simulate(duration_s, control=True, T_start=s["T0"], target=s["T0"],
                                            warmup_s=s["warmup_s"], seed=s["box_seed"]),
    }


def at_photo_times(run: dict[str, np.ndarray], t_min: np.ndarray) -> np.ndarray:
    """True box temperature (len(t_min),) at the time step nearest to each photo time t_min (minutes)."""
    i = np.clip(np.searchsorted(run["t_s"], t_min * 60.0), 1, len(run["t_s"]) - 1)
    nearest = np.where(np.abs(run["t_s"][i - 1] - t_min * 60.0) <= np.abs(run["t_s"][i] - t_min * 60.0), i - 1, i)
    return run["T_true"][nearest]


def run_case(
    fake: lab.FakeLab,
    X_est: np.ndarray,
    X0: np.ndarray,
    Y: np.ndarray,
    temperatures: np.ndarray,
    E_test: np.ndarray,
    s: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[float, float]]:
    """Follow the sample through one temperature curve (T,) with the fixed map X_est (N, M).

    The sample changes with T - temperatures[0]: the map was measured at t = 0.
    Returns: prediction correlation (T,), output correlation vs a reference photo
             at t = 0 (T,), the focus photos (M,) at the first and the last time point,
             and the background level (mean intensity of the test photos) at those two times.
    """
    T_map = temperatures[0]
    # Reference photo: a separate photo of the test patterns at the first
    # temperature. Comparing with it (not with the t = 0 photo itself) shows the
    # real photo-to-photo agreement, limited by fresh SLM error, flicker and noise.
    # It uses its own random generator, so all other photos stay exactly the same.
    fake.X_true = lab.sample_at_temperature(X0, Y, 0.0, s["T_c"])
    rng, fake.rng = fake.rng, np.random.default_rng(s["ref_seed"])
    I_ref = fake.measure(E_test)
    fake.rng = rng

    r_pred = np.empty(len(temperatures))
    r_out = np.empty(len(temperatures))
    focus, background = [], []
    for i, T in enumerate(temperatures):
        # The real sample changes with temperature; the map does not.
        fake.X_true = lab.sample_at_temperature(X0, Y, T - T_map, s["T_c"])
        I_test = fake.measure(E_test)
        r_pred[i] = metrics.prediction_correlation(E_test, X_est, I_test)
        r_out[i] = metrics.output_correlation(I_test, I_ref)
        if i in (0, len(temperatures) - 1):
            # The test patterns are random, so their mean intensity is the unfocused background.
            background.append(float(I_test.mean()))
            # A focus spot would saturate the fixed exposure, as in main.py.
            full_scale, fake.full_scale = fake.full_scale, None
            focus.append(fake.measure(metrics.focus_pattern(X_est, s["focus_target"]))[0])
            fake.full_scale = full_scale
    return r_pred, r_out, (focus[0], focus[1]), (background[0], background[1])


def main_drift() -> dict:
    """Run the drift test with SETTINGS; save plots and data; return the summary dict."""
    s = SETTINGS
    out = Path(s["out_dir"])

    # 1. Same fake lab and GGS as main.py; the map is measured once, at t = 0.
    fake, _, _, X_est, _, _ = main.build_lab_and_map(s)

    # 2. Temperature model and fixed test patterns.
    X0 = fake.X_true.copy()
    Y = lab.make_transmission_matrix(s["N"], s["M"], np.random.default_rng(s["y_seed"]))
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))

    # Box temperatures from the thermostat simulation, read at each photo time.
    t = np.arange(0, s["duration_min"] * 60 + s["dt_s"] / 2, s["dt_s"]) / 60.0   # minutes
    runs = box_runs(s)
    temps = {case: at_photo_times(runs[case], t) for case in CASES}

    # 3.-4. Follow both cases over time.
    r_pred, r_out, focus, eta = {}, {}, {}, {}
    for case in CASES:
        r_pred[case], r_out[case], focus[case], bg = run_case(fake, X_est, X0, Y, temps[case], E_test, s)
        # Enhancement of each focus photo: target intensity / background.
        eta[case] = tuple(I[s["focus_target"]] / b for I, b in zip(focus[case], bg))

    # 5. Put the sample back to its reference state.
    fake.X_true = X0

    # 6. Plots. ΔT is measured from the temperature at t = 0, when the map was measured.
    run_on = runs["with control"]
    saved = [
        plots.plot_correlation_vs_time(t, {CASES[c]: r_pred[c] for c in CASES}, out / "01_map_accuracy_over_time.png",
                                       title=f"How well the map predicts the camera over {s['duration_min']:g} minutes",
                                       colors=COLORS),
        plots.plot_focus_grid(
            {CASES[c]: focus[c] for c in CASES}, {CASES[c]: eta[c] for c in CASES}, s["image_shape"],
            s["focus_target"], out / "02_focus_spot_start_vs_end.png",
            title=f"Focus spot at the start and after {s['duration_min']:g} minutes",
            col_titles=("start (t = 0)", f"end (t = {s['duration_min']:g} min)")),
        # Correlation and temperature on one time axis (two y-axes), one plot per case.
        plots.plot_correlation_and_temperature(
            t, r_pred["no control"], runs["no control"]["t_s"] / 60.0,
            runs["no control"]["T_true"] - runs["no control"]["T_true"][0], out / "03_no_control_temperature_and_accuracy.png",
            "No control (cooler off): the box warms up and the map accuracy drops", COLORS[CASES["no control"]],
            "ΔT (°C)", right_ylim=(0.0, 2.5)),
        plots.plot_correlation_and_temperature(
            t, r_pred["with control"], run_on["t_s"] / 60.0, run_on["T_true"] - s["T0"],
            out / "04_pid_control_temperature_and_accuracy.png",
            f"With PID + Kalman control: the box stays at {s['T0']:g} °C and the map stays accurate\n"
            f"max |T - {s['T0']:g}| {np.abs(run_on['T_true'] - s['T0']).max():.4f} °C "
            f"(limit ±{plots.LIMIT_C:g} °C)", COLORS[CASES["with control"]],
            f"distance from {s['T0']:g} °C (°C)", right_ylim=(-0.045, 0.045)),
    ]

    # 7. Data.
    csv_path = out / "results.csv"
    np.savetxt(
        csv_path,
        np.column_stack([t, temps["no control"], temps["with control"],
                         r_pred["no control"], r_pred["with control"],
                         r_out["no control"], r_out["with control"]]),
        delimiter=",", fmt="%.6f", comments="",
        header="t_min,T_no_control,T_with_control,r_pred_no_control,r_pred_with_control,"
               "r_out_no_control,r_out_with_control",
    )
    curves = {f"{name}_{case.replace(' ', '_')}": values
              for name, d in [("T", temps), ("r_pred", r_pred), ("r_out", r_out)] for case, values in d.items()}
    summary = {
        name: {"start": float(v[0]), "end": float(v[-1]), "min": float(v.min())} for name, v in curves.items()
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps({"settings": s, "results": summary}, indent=2))
    saved += [csv_path, summary_path]

    # 8. Table once per minute, then one line per case.
    per_minute = int(round(60 / s["dt_s"]))
    print(f"{'t (min)':>7} {'T no ctrl':>9} {'T ctrl':>7} {'r no ctrl':>9} {'r ctrl':>7} "
          f"{'out no ctrl':>11} {'out ctrl':>8}")
    for i in range(0, len(t), per_minute):
        print(f"{t[i]:>7.1f} {temps['no control'][i]:>9.2f} {temps['with control'][i]:>7.3f} "
              f"{r_pred['no control'][i]:>9.3f} {r_pred['with control'][i]:>7.3f} "
              f"{r_out['no control'][i]:>11.3f} {r_out['with control'][i]:>8.3f}")
    for case in CASES:
        r = r_pred[case]
        print(f"{CASES[case]}: prediction correlation {r[0]:.3f} -> {r[-1]:.3f} (min {r.min():.3f})")
    print(f"Saved in {out}/:")
    for p in saved:
        print(f"  {p}")

    for case in CASES:
        assert 0.80 < r_pred[case][0] < 0.95, f"{case}: start {r_pred[case][0]:.3f} not in (0.80, 0.95)"
    assert r_pred["no control"][-1] < 0.15, f"no control: end {r_pred['no control'][-1]:.3f} >= 0.15"
    assert r_pred["with control"].min() > 0.75, f"with control: min {r_pred['with control'].min():.3f} <= 0.75"
    assert r_pred["with control"][-1] > 0.80, f"with control: end {r_pred['with control'][-1]:.3f} <= 0.80"
    print("drift_test.py finished")
    return summary


if __name__ == "__main__":
    main_drift()
