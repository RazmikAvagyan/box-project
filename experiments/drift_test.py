"""
drift_test.py - does the map still work while the temperature changes?

In each case the map X_est is measured at t = 0 (at the box temperature then), then for 5 minutes
the same test patterns are shown and the camera photos are compared with what X_est predicts.
Two cases, both from the box simulation (box_simulation.thermostat): without
temperature control (the cooler is off and the box warms up) and with
temperature control (PID + Kalman thermostat, already settled by a warm-up).

The sample at temperature T is sample.at(T), a lab.TemperatureSample that is exactly X0 at
OPTICS T_ref_C; the similarity of two temperatures depends only on their difference.
Row convention (as in lab.py): E_out = E_in @ X, (L x M) = (L x N) @ (N x M).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import plots
from box_simulation import thermostat
from config import DRIFT_TEST as SETTINGS
from config import OPTICS
from experiments import main
from optics_simulation import lab, metrics

CASES = {"no control": "no control (cooler off)", "with control": "with PID + Kalman control"}
COLORS = {"no control (cooler off)": "red", "with PID + Kalman control": "blue"}


def box_runs(s: dict, box_settings: dict | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Box simulation (20 Hz) for both cases; each value is the dict from thermostat.simulate.
    box_settings: optional THERMOSTAT overrides (e.g. other PID gains)."""
    duration_s = s["duration_min"] * 60
    box = {**thermostat.SETTINGS, **(box_settings or {})}
    return {
        "no control": thermostat.simulate(duration_s, control=False, T_start=s["T0"], seed=s["box_seed"], settings=box),
        "with control": thermostat.simulate(duration_s, control=True, T_start=s["T0"], target=s["T0"],
                                            warmup_s=s["warmup_s"], seed=s["box_seed"], settings=box),
    }


def at_photo_times(run: dict[str, np.ndarray], t_min: np.ndarray) -> np.ndarray:
    """True box temperature (len(t_min),) at the time step nearest to each photo time t_min (minutes)."""
    i = np.clip(np.searchsorted(run["t_s"], t_min * 60.0), 1, len(run["t_s"]) - 1)
    nearest = np.where(np.abs(run["t_s"][i - 1] - t_min * 60.0) <= np.abs(run["t_s"][i] - t_min * 60.0), i - 1, i)
    return run["T_true"][nearest]


def run_case(
    fake: lab.FakeLab,
    X_est: np.ndarray,
    sample: lab.TemperatureSample,
    temperatures: np.ndarray,
    E_test: np.ndarray,
    s: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[float, float]]:
    """Follow the sample through one temperature curve (T,) with the fixed map X_est (N, M), measured at
    temperatures[0]; the sample is sample.at(T).
    Returns: prediction correlation (T,), output correlation vs a reference photo
             at t = 0 (T,), the focus photos (M,) at the first and the last time point,
             and the background level (mean intensity of the test photos) at those two times.
    """
    T_map = temperatures[0]
    # Reference photo: a separate photo of the test patterns at the first
    # temperature. Comparing with it (not with the t = 0 photo itself) shows the
    # real photo-to-photo agreement, limited by fresh SLM error, flicker and noise.
    # It uses its own random generator, so all other photos stay exactly the same.
    fake.X_true = sample.at(T_map)
    rng, fake.rng = fake.rng, np.random.default_rng(s["ref_seed"])
    I_ref = fake.measure(E_test)
    fake.rng = rng

    r_pred = np.empty(len(temperatures))
    r_out = np.empty(len(temperatures))
    focus, background = [], []
    for i, T in enumerate(temperatures):
        # The real sample changes with temperature; the map does not.
        fake.X_true = sample.at(T)
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


def main_drift(s: dict = SETTINGS, box_settings: dict | None = None) -> tuple[dict, dict]:
    """Run the drift test with settings s (and THERMOSTAT overrides box_settings); save plots and data in s["out_dir"].

    Returns: the summary dict and the prediction correlations {case: (T,)}.
    """
    out = Path(s["out_dir"])

    # 1.-2. Same fake lab as main.py, the temperature model and fixed test patterns.
    fake = main.build_lab(s)
    X0 = fake.X_true.copy()
    sample = lab.TemperatureSample(X0, OPTICS["T_ref_C"], s["T_c"], OPTICS["T_step_C"], s["y_seed"])
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))

    # Box temperatures from the thermostat simulation, read at each photo time.
    t = np.arange(0, s["duration_min"] * 60 + s["dt_s"] / 2, s["dt_s"]) / 60.0   # minutes
    runs = box_runs(s, box_settings)
    temps = {case: at_photo_times(runs[case], t) for case in CASES}

    # 3.-4. Follow both cases over time.
    r_pred, r_out, focus, eta = {}, {}, {}, {}
    for case in CASES:
        # Same GGS as main.py; the map is measured at t = 0 of this case, at the box temperature then.
        fake.X_true = sample.at(temps[case][0])
        X_est = main.measure_map(fake, s)[2]
        r_pred[case], r_out[case], focus[case], bg = run_case(fake, X_est, sample, temps[case], E_test, s)
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
            f"With PID + Kalman control: box temperature and map accuracy\n"
            f"max |T - {s['T0']:g}| {np.abs(run_on['T_true'] - s['T0']).max():.4f} °C "
            f"(limit ±{plots.LIMIT_C:g} °C)", COLORS[CASES["with control"]],
            f"distance from {s['T0']:g} °C (°C)",
            right_ylim=(-1.1 * np.abs(run_on["T_true"] - s["T0"]).max(), 1.1 * np.abs(run_on["T_true"] - s["T0"]).max()),
            limit=plots.LIMIT_C),
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
    return summary, r_pred


def long_test(s: dict = SETTINGS) -> dict[tuple[str, float], dict]:
    """Long test: long_min minutes, one map at t = 0, for no control, the real PID and the tuned PID, in each room
    of PID_TUNING["holds"] (T_room as a THERMOSTAT override). Saves 05_long_test_map_accuracy.png in s["out_dir"].

    Returns: {(case, room C): {"r": prediction correlation (T,), "T_range": max - min of the true box temperature}}.
    """
    from config import PID_TUNING, THERMOSTAT

    fake = main.build_lab(s)
    sample = lab.TemperatureSample(fake.X_true.copy(), OPTICS["T_ref_C"], s["T_c"], OPTICS["T_step_C"], s["y_seed"])
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))
    duration_s = s["long_min"] * 60
    t = np.arange(0, duration_s + s["long_dt_s"] / 2, s["long_dt_s"]) / 60.0          # minutes
    cases = {"no control": None, f"PID {THERMOSTAT['Kp']:g}/{THERMOSTAT['Ki']:g}": {},
             "PID {:g}/{:g} (tuned)".format(*PID_TUNING["refined"]): dict(zip(("Kp", "Ki", "Kd"), PID_TUNING["refined"]))}
    out = {}
    for _, room in PID_TUNING["holds"]:
        for case, gains in cases.items():
            box = {**thermostat.SETTINGS, "T_room": room, **(gains or {})}
            if gains is None:
                run = thermostat.simulate(duration_s, control=False, T_start=s["T0"], seed=s["box_seed"], settings=box)
            else:
                run = thermostat.simulate(duration_s, control=True, T_start=s["T0"], target=s["T0"],
                                          warmup_s=s["long_warmup_s"], seed=s["box_seed"], settings=box)
            temps = at_photo_times(run, t)
            fake.X_true = sample.at(temps[0])
            X_est = main.measure_map(fake, s)[2]
            r = run_case(fake, X_est, sample, temps, E_test, s)[0]
            out[(case, room)] = {"r": r, "T_range": float(run["T_true"].max() - run["T_true"].min())}
            print(f"long test, room {room:g} C, {case:22}: r {r[0]:.3f} -> {r[-1]:.3f} (min {r.min():.3f}), "
                  f"box range {out[(case, room)]['T_range']:.3f} C")
    colors = dict(zip(cases, ["red", "blue", "green"]))
    rooms = [room for _, room in PID_TUNING["holds"]]
    label = lambda case, room: f"{case}, room {room:g} °C"
    print(plots.plot_correlation_vs_time(
        t, {label(c, room): v["r"] for (c, room), v in out.items()}, Path(s["out_dir"]) / "05_long_test_map_accuracy.png",
        title=f"Map measured once at t = 0: accuracy over {s['long_min']:g} minutes (target {s['T0']:g} °C)",
        colors={label(c, room): colors[c] for c, room in out},
        styles={label(c, room): "-" if room == rooms[0] else "--" for c, room in out}))
    return out


def check(r_pred: dict) -> list[str]:
    """The drift test requirements on the prediction correlations {case: (T,)}. The start range and "no control
    ends lower" are asserted; returns the failed limits that depend on the PID (min > 0.75, end > 0.80 with control)."""
    for case in CASES:
        assert 0.80 < r_pred[case][0] < 0.95, f"{case}: start {r_pred[case][0]:.3f} not in (0.80, 0.95)"
    # The fitted box warms only about 0.2 C in 5 min without the cooler, so the map stays fairly good;
    # it must still end worse than with control [estimate].
    print(f"end of the test: no control {r_pred['no control'][-1]:.3f}, with control {r_pred['with control'][-1]:.3f} "
          f"(no control must end lower)")
    assert r_pred["no control"][-1] < r_pred["with control"][-1], "no control does not end lower than with control"
    r = r_pred["with control"]
    return ([f"with control: min {r.min():.3f} <= 0.75"] if r.min() <= 0.75 else []) + \
        ([f"with control: end {r[-1]:.3f} <= 0.80"] if r[-1] <= 0.80 else [])


if __name__ == "__main__":
    import tempfile

    from config import PID_TUNING, THERMOSTAT

    real = f"PID {THERMOSTAT['Kp']:g}/{THERMOSTAT['Ki']:g}"
    tuned = "PID {:g}/{:g}".format(*PID_TUNING["refined"])
    pids = {real: None, tuned: dict(zip(("Kp", "Ki", "Kd"), PID_TUNING["refined"]))}
    # Default settings write results/drift_test/; the other runs go to a temporary folder. The 19 C runs check
    # another target (above about 20.9 C the box settles without cooling and the PID would do nothing);
    # T_ref_C stays at its config value.
    r = {}
    with tempfile.TemporaryDirectory() as tmp:
        for target in (SETTINGS["T0"], 19.0):
            for pid, box in pids.items():
                default = target == SETTINGS["T0"] and box is None
                print(f"\n{pid}, target {target:g} C (T_ref_C {OPTICS['T_ref_C']:g} C):")
                s = SETTINGS if default else {**SETTINGS, "T0": target, "out_dir": str(Path(tmp) / f"{pid}_{target}")}
                r[(pid, target)] = main_drift(s, box)[1]
    print()
    for (pid, target), rp in r.items():
        failed = check(rp)
        print(f"{pid}, {target:g} C: " + "; ".join(
            f"{CASES[c]} start {rp[c][0]:.3f} end {rp[c][-1]:.3f} min {rp[c].min():.3f}" for c in CASES))
        if pid == tuned:
            assert not failed, f"{pid}, {target:g} C: " + "; ".join(failed)
        elif failed:
            print(f"  {'; '.join(failed)}: expected to fail with the real {pid} (swings ±0.3 C)")

    print(f"\nLong test ({SETTINGS['long_min']:g} min, map measured once at t = 0):")
    long = long_test()
    warm = max(room for _, room in PID_TUNING["holds"])
    end_tuned = long[(f"{tuned} (tuned)", warm)]["r"][-1]
    end_off = long[("no control", warm)]["r"][-1]
    print(f"room {warm:g} C: end correlation tuned PID {end_tuned:.3f}, no control {end_off:.3f} "
          f"(tuned must be at least {SETTINGS['long_margin']:g} higher)")
    assert end_tuned > end_off + SETTINGS["long_margin"], "long test: tuned PID not clearly better than no control"
    print("drift_test.py finished")
