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
from config import OPTICS, PLOTS
from experiments import main
from optics_simulation import lab, metrics

CASES = {"no control": "no control (cooler off)", "with control": "with PID + Kalman control"}


def case_labels(pid: str) -> dict[str, str]:
    """Legend labels of the two cases, naming the PID (a name for thermostat.pid_label)."""
    return {"no control": CASES["no control"], "with control": f"with {thermostat.pid_label(pid)} + Kalman"}


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
    times_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], tuple[float, float]]:
    """Follow the sample through one temperature curve (T,) with the fixed map X_est (N, M), measured at
    temperatures[0]; the sample is sample.at(T, t) with t from times_s (T,) in s (only used with time drift).
    Returns: prediction correlation (T,), output correlation vs a reference photo
             at t = 0 (T,), the focus photos (M,) at the first and the last time point,
             and the background level (mean intensity of the test photos) at those two times.
    """
    T_map = temperatures[0]
    # Reference photo: a separate photo of the test patterns at the first
    # temperature. Comparing with it (not with the t = 0 photo itself) shows the
    # real photo-to-photo agreement, limited by fresh SLM error, flicker and noise.
    # It uses its own random generator, so all other photos stay exactly the same.
    times_s = np.zeros(len(temperatures)) if times_s is None else times_s
    fake.X_true = sample.at(T_map, times_s[0])
    rng, fake.rng = fake.rng, np.random.default_rng(s["ref_seed"])
    I_ref = fake.measure(E_test)
    fake.rng = rng

    r_pred = np.empty(len(temperatures))
    r_out = np.empty(len(temperatures))
    focus, background = [], []
    for i, T in enumerate(temperatures):
        # The real sample changes with temperature; the map does not.
        fake.X_true = sample.at(T, times_s[i])
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


def main_drift(s: dict = SETTINGS, pid: str = "real", tau_min: float | None = OPTICS["time_drift_tau_min"]) -> tuple[dict, dict]:
    """Run the drift test with settings s, the PID named pid (see thermostat.pid_gains) and time drift tau_min (min,
    None = off); save plots and data in s["out_dir"].

    Returns: the summary dict and the prediction correlations {case: (T,)}.
    """
    out = Path(s["out_dir"])

    # 1.-2. Same fake lab as main.py, the temperature model and fixed test patterns.
    fake = main.build_lab(s)
    X0 = fake.X_true.copy()
    sample = lab.TemperatureSample(X0, OPTICS["T_ref_C"], s["T_c"], OPTICS["T_step_C"], s["y_seed"],
                                   None if tau_min is None else 60.0 * tau_min, OPTICS["time_drift_step_s"])
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))
    names = case_labels(pid)

    # Box temperatures from the thermostat simulation, read at each photo time.
    t = np.arange(0, s["duration_min"] * 60 + s["dt_s"] / 2, s["dt_s"]) / 60.0   # minutes
    runs = box_runs(s, thermostat.pid_gains(pid))
    temps = {case: at_photo_times(runs[case], t) for case in CASES}

    # 3.-4. Follow both cases over time.
    r_pred, r_out, focus, eta = {}, {}, {}, {}
    for case in CASES:
        # Same GGS as main.py; the map is measured at t = 0 of this case, at the box temperature then.
        fake.X_true = sample.at(temps[case][0], 0.0)
        X_est = main.measure_map(fake, s)[2]
        r_pred[case], r_out[case], focus[case], bg = run_case(fake, X_est, sample, temps[case], E_test, s, t * 60.0)
        # Enhancement of each focus photo: target intensity / background.
        eta[case] = tuple(I[s["focus_target"]] / b for I, b in zip(focus[case], bg))

    # 5. Put the sample back to its reference state.
    fake.X_true = X0

    # 6. Plot (the map accuracy over time is in results.csv and in the long test plot).
    saved = [
        plots.plot_focus_grid(
            {names[c]: focus[c] for c in CASES}, {names[c]: eta[c] for c in CASES}, s["image_shape"],
            s["focus_target"], out / "01_focus_spot_start_vs_end.png",
            title=f"Focus spot at the start and after {s['duration_min']:g} minutes (seed 0)",
            col_titles=("start (t = 0)", f"end (t = {s['duration_min']:g} min)")),
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


def long_test(s: dict = SETTINGS, tau_min: float | None = OPTICS["time_drift_tau_min"]) -> tuple[np.ndarray, dict]:
    """Long test: long_min minutes, one map at t = 0, for no control, the real box PID and the main PID (PLOTS), in each
    room of PID_TUNING["holds"] (T_room as a THERMOSTAT override), with time drift tau_min (min, None = off).

    Returns: the photo times t (T,) in minutes and {(case, room C): {"r": prediction correlation (T,),
    "T_range": max - min of the true box temperature}}.
    """
    from config import PID_TUNING

    fake = main.build_lab(s)
    sample = lab.TemperatureSample(fake.X_true.copy(), OPTICS["T_ref_C"], s["T_c"], OPTICS["T_step_C"], s["y_seed"],
                                   None if tau_min is None else 60.0 * tau_min, OPTICS["time_drift_step_s"])
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))
    duration_s = s["long_min"] * 60
    t = np.arange(0, duration_s + s["long_dt_s"] / 2, s["long_dt_s"]) / 60.0          # minutes
    cases = {"no control": None, **{thermostat.pid_label(n): thermostat.pid_gains(n) for n in thermostat.plot_pids()}}
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
            fake.X_true = sample.at(temps[0], 0.0)
            X_est = main.measure_map(fake, s)[2]
            r = run_case(fake, X_est, sample, temps, E_test, s, t * 60.0)[0]
            out[(case, room)] = {"r": r, "T_range": float(run["T_true"].max() - run["T_true"].min())}
            print(f"long test, room {room:g} C, {case:38}: r {r[0]:.3f} -> {r[-1]:.3f} (min {r.min():.3f}), "
                  f"box range {out[(case, room)]['T_range']:.3f} C")
    return t, out


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

    from config import PID_TUNING

    main_pid = PLOTS["main_pid"]
    names = list(dict.fromkeys(["real", "refined", main_pid]))
    # The main PID writes results/drift_test/; the other runs go to a temporary folder. The 19 C runs check another
    # target (above about 20.9 C the box settles without cooling and the PID would do nothing); T_ref_C stays.
    r = {}
    with tempfile.TemporaryDirectory() as tmp:
        for target in (SETTINGS["T0"], 19.0):
            for pid in names:
                default = target == SETTINGS["T0"] and pid == main_pid
                print(f"\n{thermostat.pid_label(pid)}, target {target:g} C (T_ref_C {OPTICS['T_ref_C']:g} C):")
                s = SETTINGS if default else {**SETTINGS, "T0": target, "out_dir": str(Path(tmp) / f"{pid}_{target}")}
                r[(pid, target)] = main_drift(s, pid)[1]
    print()
    for (pid, target), rp in r.items():
        failed = check(rp)
        print(f"{thermostat.pid_label(pid)}, {target:g} C: " + "; ".join(
            f"{CASES[c]} start {rp[c][0]:.3f} end {rp[c][-1]:.3f} min {rp[c].min():.3f}" for c in CASES))
        if pid == "refined":
            assert not failed, f"{thermostat.pid_label(pid)}, {target:g} C: " + "; ".join(failed)
        elif failed:
            print(f"  {'; '.join(failed)}: expected to fail with {thermostat.pid_label(pid)} (swings ±0.3 C)")

    print(f"\nLong test ({SETTINGS['long_min']:g} min, map measured once at t = 0):")
    t, long = long_test()
    warm = max(room for _, room in PID_TUNING["holds"])
    end_main = long[(thermostat.pid_label(main_pid), warm)]["r"][-1]
    end_off = long[("no control", warm)]["r"][-1]
    print(f"room {warm:g} C: end correlation {thermostat.pid_label(main_pid)} {end_main:.3f}, no control {end_off:.3f} "
          f"(the main PID must be at least {SETTINGS['long_margin']:g} higher)")
    assert end_main > end_off + SETTINGS["long_margin"], "long test: the main PID not clearly better than no control"
    tau = 30
    print(f"\nLong test with time drift of the sample, tau {tau} min:")
    _, drifting = long_test(tau_min=tau)
    # One figure: no time drift | time drift, the same cases and colours.
    out = Path(SETTINGS["out_dir"])
    cases = list(dict.fromkeys(c for c, _ in long))
    rooms = list(dict.fromkeys(room for _, room in long))
    label = lambda case, room: f"{case}, room {room:g} °C"
    print(plots.plot_correlation_panels(
        t, {"no time drift": {label(c, r): v["r"] for (c, r), v in long.items()},
            f"time drift of the sample, tau {tau} min": {label(c, r): v["r"] for (c, r), v in drifting.items()}},
        out / "02_long_test_map_accuracy.png",
        f"Map measured once at t = 0: accuracy over {SETTINGS['long_min']:g} minutes (target {SETTINGS['T0']:g} °C, seed 0)",
        colors={label(c, r): color for c, color in zip(cases, ["red", "blue", "green"]) for r in rooms},
        styles={label(c, r): "-" if r == rooms[0] else "--" for c in cases for r in rooms}))
    print("drift_test.py finished")
