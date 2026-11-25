"""
main.py - one full simulated experiment, start to end.

fake lab -> SLM patterns -> camera photos -> GGS -> checks -> plots and saved data.

Row convention (as in lab.py): E_out = E_in @ X, (L x M) = (L x N) @ (N x M).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from analysis import plots
from config import MAIN as SETTINGS
from optics_simulation import ggs, lab, metrics


def build_lab(s: dict) -> lab.FakeLab:
    """The fake lab from settings s: a random scattering sample, an imperfect SLM and laser, a realistic camera."""
    return lab.FakeLab(
        N=s["N"], M=s["M"], seed=s["seed"],
        noise_level=s["noise_level"], bit_depth=s["bit_depth"], full_scale=s["full_scale"],
        phase_error=s["phase_error"], flicker=s["flicker"],
    )


def measure_map(fake: lab.FakeLab, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float]:
    """Measure L patterns on the current sample and recover the map with GGS.

    Returns: E_in (L, N), I (L, M), X_est (N, M), GGS info dict, GGS run time in s.
    """
    # Show L random patterns on the SLM and take a camera photo of each.
    E_in = fake.patterns(s["L_ratio"] * s["N"])        # (L x N)
    I = fake.measure(E_in)                             # (L x M)
    A = lab.amplitude(I)                               # GGS works with amplitudes

    # Recover the transmission matrix from patterns and photos only.
    t0 = time.perf_counter()
    X_est, info = ggs.recover(
        E_in, A, np.random.default_rng(s["ggs_seed"]),
        max_iter=s["max_iter"], tol=s["tol"],
        n_restarts=s["n_restarts"], restart_threshold=s["restart_threshold"],
    )
    return E_in, I, X_est, info, time.perf_counter() - t0


def build_lab_and_map(s: dict) -> tuple[lab.FakeLab, np.ndarray, np.ndarray, np.ndarray, dict, float]:
    """Build the fake lab from settings s, measure L patterns and recover the map with GGS.

    Returns: fake (FakeLab), E_in (L, N), I (L, M), X_est (N, M), GGS info dict, GGS run time in s.
    """
    fake = build_lab(s)
    return (fake, *measure_map(fake, s))


def main() -> dict:
    """Run the full simulation with SETTINGS; save plots and data; return the results dict."""
    s = SETTINGS
    out = Path(s["out_dir"])
    L = s["L_ratio"] * s["N"]

    # 1.-3. Fake lab, L photos and the GGS map.
    fake, E_in, I, X_est, info, run_time = build_lab_and_map(s)

    # 4a. Compare with the true matrix (only possible in simulation).
    quality = metrics.map_quality(X_est, fake.X_true)

    # 4b. Does the map predict the speckle of new patterns? (possible in a real lab)
    E_test = lab.make_slm_patterns(s["n_test"], s["N"], np.random.default_rng(s["test_seed"]))
    r_pred = metrics.prediction_correlation(E_test, X_est, fake.measure(E_test))

    # 4c. Focusing. A focus spot is ~50x brighter than speckle, so with the fixed
    # full_scale the camera would saturate and clip the spot. Like an experimenter
    # shortening the exposure, switch to auto scale (brightest pixel = full range).
    fake.full_scale = None
    focus = metrics.focus_test(fake, X_est)

    # 5. Plots.
    target = s["focus_target"]
    I_random = fake.measure(fake.patterns(1))[0]
    I_focus = fake.measure(metrics.focus_pattern(X_est, target))[0]
    saved = [
        plots.plot_ggs_convergence(info["history"], out / "01_map_learning.png", f" (seed {s['seed']})"),
        plots.plot_focus(I_random, I_focus, s["image_shape"], target, out / "02_focusing_example.png",
                         f"Random vs. focusing SLM pattern on the camera (seed {s['seed']})"),
        plots.plot_enhancement(focus["eta"], focus["eta_theory"], s["N"],
                              out / "03_focus_brightness_all_points.png", f" (seed {s['seed']})"),
    ]

    # 6. Data. Plain Python numbers so json can save them.
    results = {
        "L": L,
        "ggs_iterations": int(info["n_iter"]),
        "converged": bool(info["converged"]),
        "restarted_columns": int(info["restarted_columns"]),
        "run_time_s": float(run_time),
        "corr_mean": float(quality["corr_mean"]),
        "corr_min": float(quality["corr_min"]),
        "prediction_correlation": float(r_pred),
        "eta_mean": float(focus["eta_mean"]),
        "eta_theory": float(focus["eta_theory"]),
        "ratio": float(focus["ratio"]),
    }
    np.savez_compressed(
        out / "data.npz",
        E_in=E_in, I=I, X_est=X_est, X_true=fake.X_true, eta=focus["eta"], history=np.array(info["history"]),
    )
    saved.append(out / "data.npz")
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps({"settings": s, "results": results}, indent=2))
    saved.append(summary_path)

    # 7. Readable summary.
    print(f"Settings: N={s['N']}, M={s['M']}, L={L}, noise={s['noise_level']}, "
          f"phase_error={s['phase_error']} rad, flicker={s['flicker']}, {s['bit_depth']}-bit camera")
    print(f"GGS: {results['ggs_iterations']} iterations, "
          f"{'converged' if results['converged'] else 'NOT converged'}, "
          f"{results['restarted_columns']} columns restarted, {run_time:.1f} s")
    print(f"Map quality (simulation only): corr_mean {results['corr_mean']:.3f}, corr_min {results['corr_min']:.3f}")
    print(f"Prediction correlation (new patterns): {r_pred:.3f}")
    print(f"Focusing: eta_mean {results['eta_mean']:.1f} / theory {results['eta_theory']:.1f} "
          f"(ratio {results['ratio']:.2f})")
    print(f"Saved in {out}/:")
    for p in saved:
        print(f"  {p}")

    return results


if __name__ == "__main__":
    main()
