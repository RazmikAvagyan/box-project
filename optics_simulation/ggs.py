"""
ggs.py - recover the transmission matrix X from intensity-only measurements.

Generalized Gerchberg-Saxton (GGS). Sources: Huang et al., Photonics Research
2021; E. Arzumanyan's thesis.

Row convention (same as lab.py):
    E_out = E_in @ X
    (L x M) = (L x N) @ (N x M)

Known:   E_in (L x N), the SLM patterns (all entries of size 1).
         A = sqrt(I) (L x M), the measured camera amplitudes.
Wanted:  X (N x M) such that |E_in @ X| = A.

Each column of X (one camera point) can only be found up to one constant
phase factor, because the camera never sees absolute phase.
lab.column_correlation ignores that factor.
"""

from __future__ import annotations

import numpy as np

from config import MAIN
from optics_simulation import lab


def fit_error(E_pred: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Per-column fit error, computable without knowing the true X.

    err_j = || |E_pred[:, j]| - A[:, j] || / || A[:, j] ||, with E_pred = E_in @ X.

    Input: E_pred (L, M) complex, A (L, M) measured amplitudes.
    Returns: shape (M,), float64.
    """
    return np.linalg.norm(np.abs(E_pred) - A, axis=0) / np.linalg.norm(A, axis=0)


def ggs(
    E_in: np.ndarray,
    E_pinv: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict]:
    """GGS iterations on all columns at once, starting from the guess X.

    Input: E_in (L, N), E_pinv = pinv(E_in) (N, L), A (L, M), start X (N, M).
    Stops when mean column_correlation(X_k, X_{k-2}) >= 1 - tol, or after max_iter.
    Returns: X_est (N, M) complex128, and info dict with
             "n_iter" (int), "converged" (bool), "history" (list of mean fit error per iteration).
    """
    history: list[float] = []
    X_prev2 = None   # X_{k-2}
    X_prev1 = None   # X_{k-1}
    converged = False

    # Predict: the field the camera would see if the current guess were true.
    E_pred = E_in @ X                                  # (L x M)

    for _ in range(max_iter):
        # Correct: the measured amplitude is trusted; the phase is unknown, so we
        # keep the phase of our prediction and put back the measured size.
        E_corr = A * np.exp(1j * np.angle(E_pred))     # (L x M)

        # Update: find the X that best maps E_in to E_corr in the least-squares
        # sense. With L > N there are more equations than unknowns, so this
        # pulls the guess towards a matrix that fits all patterns at once.
        X_prev2, X_prev1 = X_prev1, X
        X = E_pinv @ E_corr                            # (N x M)

        # Predict again for the new X; reused for the fit error and the next step.
        E_pred = E_in @ X
        history.append(float(fit_error(E_pred, A).mean()))

        # Stop when X no longer changes (up to a phase per column) over two steps.
        if X_prev2 is not None and lab.column_correlation(X, X_prev2).mean() >= 1.0 - tol:
            converged = True
            break

    return X, {"n_iter": len(history), "converged": converged, "history": history}


def recover(
    E_in: np.ndarray,
    A: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = MAIN["max_iter"],
    tol: float = MAIN["tol"],
    n_restarts: int = MAIN["n_restarts"],
    restart_threshold: float = MAIN["restart_threshold"],
) -> tuple[np.ndarray, dict]:
    """GGS from a random start, with restarts for columns stuck in a local minimum.

    Runs ggs() on all columns, then up to n_restarts times re-runs GGS (new random
    start) only on columns with fit error > restart_threshold, keeping a new
    column only if its fit error is lower.

    Input: E_in (L, N), A (L, M), rng for the random starts.
    Returns: X_est (N, M), and info dict with everything from the first ggs() run plus
             "restarted_columns" (int, total over all rounds) and
             "final_fit_error" (array of shape (M,)).
    """
    if E_in.shape[0] != A.shape[0]:
        raise ValueError(
            f"E_in has L = {E_in.shape[0]} rows (patterns) but A has L = {A.shape[0]} rows; they must match."
        )
    # The pseudo-inverse only depends on the patterns: one for the main run and all restarts.
    E_pinv = np.linalg.pinv(E_in)                      # (N x L)
    N, M = E_in.shape[1], A.shape[1]

    X, info = ggs(E_in, E_pinv, A, lab.make_transmission_matrix(N, M, rng), max_iter, tol)

    err = fit_error(E_in @ X, A)
    restarted = 0
    for _ in range(n_restarts):
        bad = np.flatnonzero(err > restart_threshold)
        if bad.size == 0:
            break
        restarted += bad.size
        # Columns are independent, so we can re-run GGS on just the stuck ones.
        X_start = lab.make_transmission_matrix(N, bad.size, rng)
        X_new, _ = ggs(E_in, E_pinv, A[:, bad], X_start, max_iter, tol)
        err_new = fit_error(E_in @ X_new, A[:, bad])
        better = err_new < err[bad]
        X[:, bad[better]] = X_new[:, better]
        err[bad[better]] = err_new[better]

    info["restarted_columns"] = restarted
    info["final_fit_error"] = err
    return X, info


if __name__ == "__main__":
    import time

    N = M = 64
    cases = [
        # (L/N, noise_level, seed, min mean correlation or None)
        (8, 0.0, 10, 0.95),
        (8, 0.05, 11, 0.9),
        (3, 0.0, 12, None),
    ]

    print(f"{'L/N':>4} {'noise':>6} {'iters':>6} {'restarted':>9} {'mean corr':>9} {'min corr':>9} {'time [s]':>8}")
    for ratio, noise, seed, limit in cases:
        fake = lab.FakeLab(N=N, M=M, seed=seed, noise_level=noise)
        E_in = fake.patterns(ratio * N)
        A = lab.amplitude(fake.measure(E_in))

        t0 = time.perf_counter()
        X_est, info = recover(E_in, A, np.random.default_rng(seed + 100))
        dt = time.perf_counter() - t0

        corr = lab.column_correlation(X_est, fake.X_true)
        print(f"{ratio:>4} {noise:>6.2f} {info['n_iter']:>6} {info['restarted_columns']:>9} "
              f"{corr.mean():>9.4f} {corr.min():>9.4f} {dt:>8.2f}")
        if limit is None:
            print(f"note: L = {ratio}*N gives mean correlation {corr.mean():.4f}; too few patterns "
                  "make the phase problem under-determined, so a poor result is expected.")
        else:
            assert corr.mean() > limit, f"L = {ratio}*N, noise {noise}: mean correlation {corr.mean():.4f} <= {limit}"

    print("ggs.py self-test passed")
