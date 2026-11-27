"""
metrics.py - how good is a recovered transmission matrix?

Two ways to judge X_est (N x M), with the row convention of lab.py:
    E_out = E_in @ X        (L x M) = (L x N) @ (N x M)

1. Focusing test (works in a real lab): use X_est to build an SLM pattern that
   focuses light on chosen camera points, send it through the real sample, and
   measure the enhancement eta = I_target / mean background intensity.
2. Map quality (simulation only): compare X_est with the known X_true using
   lab.column_correlation.
"""

from __future__ import annotations

import numpy as np

from config import OPTICS
from optics_simulation import lab


def focus_pattern(X_est: np.ndarray, targets: int | list[int], quantize: bool = True, levels: int = 255) -> np.ndarray:
    """Phase-only SLM pattern that focuses light on the target camera point(s).

    E_in[n] = exp(-1j * angle(sum over targets m of X_est[n, m]))

    Input: X_est (N, M); targets: one camera index or a list of them.
           quantize: round the phase to 8-bit SLM levels (`levels` gray values per 2*pi, see lab.gray_to_phase).
    Returns: E_in, shape (1, N), complex128.
    """
    cols = np.atleast_1d(targets)
    # Phase conjugation: each SLM square cancels the delay of its path to the
    # target, so all N contributions arrive in step and add up coherently.
    # For several targets, conjugate the sum of their columns.
    phase = -np.angle(X_est[:, cols].sum(axis=1))
    if quantize:
        phase = lab.gray_to_phase(lab.phase_to_gray(phase, levels), levels)
    return np.exp(1j * phase)[np.newaxis, :]


def enhancement(I_out: np.ndarray, targets: int | list[int], background: float) -> np.ndarray:
    """Enhancement eta = I_out[target] / background for each target.

    Input: I_out, shape (1, M) or (M,); targets: int or list of ints.
    Returns: shape (number of targets,), float64. eta = 1 means no focusing.
    """
    return np.ravel(I_out)[np.atleast_1d(targets)] / background


def theoretical_enhancement(N: int) -> float:
    """Expected eta for a phase-only SLM with N squares: (pi / 4) * (N - 1) + 1."""
    # Phase-only control gets pi/4 of the (N - 1) gain a full amplitude+phase
    # modulator would give, because the random path amplitudes cannot be matched.
    return np.pi / 4 * (N - 1) + 1


def focus_test(
    fake_lab: lab.FakeLab,
    X_est: np.ndarray,
    targets: list[int] | None = None,
    n_background: int = OPTICS["focus_n_background"],
    quantize: bool = True,
) -> dict:
    """Focus on each target one at a time and measure the enhancement in the simulation.

    The pattern is built from X_est (our found map), but the light goes through
    fake_lab.X_true (the real sample), exactly as in a real lab.

    Input: fake_lab (holds X_true, N x M), X_est (N, M);
           targets: camera indices, None = all M points.
    Returns: dict with "eta" (array, one per target), "eta_mean", "eta_theory",
             "ratio" (eta_mean / eta_theory), "background" (float).
    """
    if targets is None:
        targets = list(range(fake_lab.M))

    # Background: with random patterns the contributions add with random phases,
    # so the mean intensity over random patterns is the unfocused background level.
    background = fake_lab.measure(fake_lab.patterns(n_background)).mean()

    eta = np.empty(len(targets))
    for i, m in enumerate(targets):
        I_out = fake_lab.measure(focus_pattern(X_est, m, quantize=quantize))   # (1, M)
        eta[i] = enhancement(I_out, m, background)[0]

    eta_theory = theoretical_enhancement(fake_lab.N)
    return {
        "eta": eta,
        "eta_mean": float(eta.mean()),
        "eta_theory": eta_theory,
        "ratio": float(eta.mean()) / eta_theory,
        "background": background,
    }


def map_quality(X_est: np.ndarray, X_true: np.ndarray) -> dict:
    """Simulation only: column correlation between X_est and X_true (both N x M).

    Returns: dict with "corr_mean" and "corr_min" (floats, 1 = perfect up to a phase per column).
    """
    corr = lab.column_correlation(X_est, X_true)
    return {"corr_mean": float(corr.mean()), "corr_min": float(corr.min())}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation r of two arrays of the same shape (e.g. L x M), flattened.

    r = 1: bright and dark spots match exactly; r = 0: no relation.
    """
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def prediction_correlation(E_test: np.ndarray, X_est: np.ndarray, I_measured: np.ndarray) -> float:
    """Pearson r between the intensities X_est predicts and what the camera measured.

    Input: E_test (L, N) test patterns NOT used to find X_est,
           X_est (N, M), I_measured (L, M) camera intensities for E_test.
    Returns: r (float). This is the check possible in a real lab (Pearson correlation).
    """
    # If the map is right, it predicts the speckle (which pixels are bright or
    # dark) for new patterns. A map that is wrong, or a sample that has since
    # changed, predicts a different speckle, and r drops towards 0.
    I_pred = np.abs(E_test @ X_est) ** 2               # (L x M)
    return pearson(I_pred, I_measured)


def output_correlation(I_now: np.ndarray, I_start: np.ndarray) -> float:
    """Pearson r between camera intensities for the same test patterns at two times.

    Input: I_now, I_start, both (L, M). Returns: r (float). Needs no map at all.
    """
    # The speckle is a fingerprint of the sample: if the sample has not
    # changed, the same patterns give the same speckle and r = 1.
    return pearson(I_now, I_start)


if __name__ == "__main__":
    from optics_simulation import ggs

    N = M = 64
    fake = lab.FakeLab(N=N, M=M, seed=10)
    E_in = fake.patterns(8 * N)
    A = lab.amplitude(fake.measure(E_in))
    X_est, _ = ggs.recover(E_in, A, np.random.default_rng(1))

    q = map_quality(X_est, fake.X_true)
    print(f"GGS map quality: corr_mean {q['corr_mean']:.4f}, corr_min {q['corr_min']:.4f}")

    eta_theory = theoretical_enhancement(N)
    X_random = lab.make_transmission_matrix(N, M, np.random.default_rng(999))

    print(f"\n{'map':<12} {'eta_mean':>9} {'eta_theory':>10} {'ratio':>6}")
    for name, X_map in [("true X", fake.X_true), ("GGS X_est", X_est), ("random X", X_random)]:
        res = focus_test(fake, X_map)
        print(f"{name:<12} {res['eta_mean']:>9.2f} {res['eta_theory']:>10.2f} {res['ratio']:>6.3f}")
        if name == "random X":
            assert 0.5 < res["eta_mean"] < 2, f"random map focuses: eta_mean {res['eta_mean']:.2f}"
        else:
            assert abs(res["ratio"] - 1) < 0.1, f"{name}: eta_mean {res['eta_mean']:.2f} not within 10% of {eta_theory:.2f}"

    # Several spots at once share the light, so each gets roughly eta_theory / 4
    # on average. The spots are unequal because the columns add with random
    # relative phases, so only the mean is compared with eta_theory / 8; each
    # single spot just has to be clearly above the background (eta = 1).
    spots = [5, 20, 35, 50]
    # Mean intensity over random patterns = unfocused background level.
    background = fake.measure(fake.patterns(200)).mean()
    eta_multi = enhancement(fake.measure(focus_pattern(X_est, spots)), spots, background)
    print(f"\n4 spots {spots}: eta = " + ", ".join(f"{e:.2f}" for e in eta_multi))
    print(f"  mean {eta_multi.mean():.2f} (limit > {eta_theory / 8:.2f}), min {eta_multi.min():.2f} (limit > 2)")
    assert eta_multi.mean() > eta_theory / 8, "4 spots are too dim on average"
    assert np.all(eta_multi > 2), "a spot is not clearly above the background"

    # Drift: the sample changes, but we keep focusing with the old X_est.
    # Expected: eta ~ (1 - a^2) * (eta_theory - 1) + 1, since the old map only
    # matches the part sqrt(1 - a^2) * X_true that is left of the old sample.
    drift_rng = np.random.default_rng(3)
    random_patterns = lab.make_slm_patterns(200, N, np.random.default_rng(4))
    print(f"\n{'drift a':>7} {'eta_mean':>9}")
    eta_drift = []
    for a in [0.0, 0.3, 0.6, 0.9]:
        X_drift = lab.drift_step(fake.X_true, a, drift_rng)
        bg = lab.measure(random_patterns, X_drift).mean()   # unfocused background level
        eta = [enhancement(lab.measure(focus_pattern(X_est, m), X_drift), m, bg)[0] for m in range(M)]
        eta_drift.append(float(np.mean(eta)))
        print(f"{a:>7.1f} {eta_drift[-1]:>9.2f}")
    assert all(np.diff(eta_drift) < 0), f"eta did not decrease with drift: {eta_drift}"

    # Correlation of predicted vs measured speckle, on fresh test patterns
    # that GGS never saw (the only check possible in a real lab).
    E_test = lab.make_slm_patterns(100, N, np.random.default_rng(5))
    I_test = fake.measure(E_test)
    r_est = prediction_correlation(E_test, X_est, I_test)
    r_random = prediction_correlation(E_test, X_random, I_test)
    noisy = lab.FakeLab(N=N, M=M, seed=10, noise_level=0.05)       # same X_true as fake
    r_noisy = prediction_correlation(E_test, X_est, noisy.measure(E_test))
    print(f"\nprediction correlation: GGS map {r_est:.4f}, random map {r_random:.4f}, "
          f"GGS map with 5% noise {r_noisy:.4f}")
    assert r_est > 0.99, f"GGS map predicts badly: r = {r_est:.4f}"
    assert abs(r_random) < 0.1, f"random map predicts too well: r = {r_random:.4f}"
    assert r_noisy > 0.95, f"GGS map with noise predicts badly: r = {r_noisy:.4f}"

    # Drift: speckle intensities of a circular Gaussian field keep a fraction
    # |sqrt(1 - a^2)|^2 = 1 - a^2 of their correlation with the old sample.
    drift_rng = np.random.default_rng(6)
    I_start = lab.measure(E_test, fake.X_true)
    print(f"\n{'drift a':>7} {'pred corr':>9} {'output corr':>11} {'1 - a^2':>7}")
    r_pred = []
    for a in [0.0, 0.3, 0.6, 0.9]:
        I_now = lab.measure(E_test, lab.drift_step(fake.X_true, a, drift_rng))
        r_pred.append(prediction_correlation(E_test, X_est, I_now))
        r_out = output_correlation(I_now, I_start)
        print(f"{a:>7.1f} {r_pred[-1]:>9.4f} {r_out:>11.4f} {1 - a**2:>7.4f}")
        assert abs(r_pred[-1] - (1 - a**2)) < 0.05, f"a = {a}: r = {r_pred[-1]:.4f}, theory {1 - a**2:.4f}"
    assert all(np.diff(r_pred) < 0), f"prediction correlation did not decrease with drift: {r_pred}"

    print("\nmetrics.py self-test passed")
