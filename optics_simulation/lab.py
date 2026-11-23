"""
lab.py - a "fake lab" that produces simulated optics data.

Physical picture:
Light hits a phase-only SLM (spatial light modulator) split into N squares,
passes through a strongly scattering sample (e.g. white paint), and a camera
records the intensity at M output points.

Light fields are complex numbers: |E| is the amplitude, angle(E) is the phase.
The sample is fully described by a complex transmission matrix X.

Row convention (GGS paper, Huang et al., Photonics Research 2021; E. Arzumanyan's thesis):
    E_out = E_in @ X
    (L x M) = (L x N) @ (N x M)

    N = number of SLM squares (inputs)
    M = number of camera points (outputs)
    L = number of SLM test patterns (one pattern per ROW of E_in)

The camera cannot see phase; it only measures I = |E_out|**2  (L x M).

This file only GENERATES data. Recovering X (the GGS algorithm) lives in ggs.py.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def make_transmission_matrix(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Random transmission matrix of a strongly scattering sample.

    Each entry is an independent circular complex Gaussian
    (normal + 1j*normal)/sqrt(2), so E[|X_nm|^2] = 1.

    Returns: X, shape (N, M), complex128.
    """
    # In a strongly scattering medium, every output is a sum of many random
    # paths, so by the central limit theorem each entry is complex Gaussian.
    X = (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / np.sqrt(2.0)
    return X.astype(np.complex128)


def gray_to_phase(gray: np.ndarray) -> np.ndarray:
    """Convert 8-bit SLM gray values to phase: phase = 2*pi*g/255.

    Input: gray, any shape, values in [0, 255].
    Returns: phase in radians, same shape, float64.
    Note: g = 0 and g = 255 both mean "no delay" (0 and 2*pi).
    """
    g = np.asarray(gray, dtype=np.float64)
    return TWO_PI * g / 255.0


def phase_to_gray(phase: np.ndarray) -> np.ndarray:
    """Convert phase (radians) to 8-bit SLM gray values.

    g = round((phase mod 2*pi) / (2*pi) * 255), always in [0, 255].

    Input: phase, any shape (any real values; wrapped mod 2*pi).
    Returns: gray, same shape, uint8.
    """
    p = np.mod(np.asarray(phase, dtype=np.float64), TWO_PI)
    g = np.round(p / TWO_PI * 255.0)
    return g.astype(np.uint8)


def make_slm_patterns(L: int, N: int, rng: np.random.Generator, quantize: bool = True) -> np.ndarray:
    """Random phase-only SLM patterns, one pattern per row.

    Each entry has amplitude 1 (phase-only SLM) and a random phase in [0, 2*pi).
    If quantize=True the phase is rounded to what an 8-bit SLM can show
    (phase -> gray -> phase).

    Returns: E_in, shape (L, N), complex128.
    """
    phase = rng.uniform(0.0, TWO_PI, size=(L, N))
    if quantize:
        phase = gray_to_phase(phase_to_gray(phase))
    # Phase-only modulation: |E| = 1 everywhere, only the delay changes.
    return np.exp(1j * phase).astype(np.complex128)


def patterns_to_gray_images(E_in: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Gray values that drive a real SLM to show the patterns E_in.

    Input: E_in, shape (L, N), complex.
           shape: optional (h, w) with h*w == N, to get 2D images.
    Returns: uint8 array of shape (L, N), or (L, h, w) if shape is given.
    """
    E_in = np.asarray(E_in)
    gray = phase_to_gray(np.angle(E_in))
    if shape is None:
        return gray
    h, w = shape
    L, N = E_in.shape
    if h * w != N:
        raise ValueError(f"shape {shape} has h*w = {h * w}, but E_in has N = {N} columns.")
    return gray.reshape(L, h, w)


def measure(
    E_in: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_level: float = 0.0,
    bit_depth: int | None = None,
    full_scale: float | None = None,
    phase_error: float = 0.0,
    flicker: float = 0.0,
) -> np.ndarray:
    """Simulate SLM, sample and camera: I = |E_in @ X|^2 plus optional imperfections.

    Order: SLM phase error -> E_out -> intensity -> laser flicker -> camera noise
           -> clip at 0 -> bit depth / full_scale.

    Input: E_in, shape (L, N), complex.
           X, shape (N, M), complex.
           rng: required if noise_level, phase_error or flicker > 0.
           phase_error: std (radians) of a fresh random phase error on every SLM
               square for every call, E_actual = E_in * exp(1j * delta).
           flicker: relative laser brightness change; each photo (row of I) is
               multiplied by its own factor max(0, 1 + flicker * g), g ~ N(0, 1).
           noise_level: Gaussian noise std = noise_level * mean(I).
           bit_depth: e.g. 8 or 12 to simulate an integer camera; None = ideal.
           full_scale: with bit_depth set, the fixed intensity that maps to
               2**bit_depth - 1 counts; brighter pixels saturate (clipped to
               full_scale). None = use I.max() of this batch. Ignored if
               bit_depth is None.
    Returns: I, shape (L, M), float64, non-negative.
    """
    E_in = np.asarray(E_in)
    X = np.asarray(X)
    if E_in.shape[1] != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: E_in is {E_in.shape} (L x N) so N = {E_in.shape[1]}, "
            f"but X is {X.shape} (N x M) so N = {X.shape[0]}."
        )
    if full_scale is not None and not full_scale > 0:
        raise ValueError(f"full_scale must be > 0 or None, got {full_scale!r}.")
    if (noise_level > 0 or phase_error > 0 or flicker > 0) and rng is None:
        raise ValueError(
            "noise_level, phase_error or flicker > 0 needs a random generator: "
            "pass rng=np.random.default_rng(seed)."
        )

    if phase_error > 0:
        # The SLM never shows exactly the asked phase; the error is new every
        # time a pattern is shown, so GGS cannot learn it as part of X.
        E_in = E_in * np.exp(1j * rng.normal(0.0, phase_error, size=E_in.shape))

    # Each output field is a coherent sum over all N SLM squares.
    E_out = E_in @ X                      # (L x M)
    # The camera only sees intensity; the phase is lost.
    I = np.abs(E_out) ** 2                # (L x M), float64

    if flicker > 0:
        # The laser is a little brighter or dimmer for each photo (one row).
        factor = np.clip(1.0 + flicker * rng.standard_normal(I.shape[0]), 0.0, None)
        I = I * factor[:, np.newaxis]

    if noise_level > 0:
        I = I + rng.normal(0.0, noise_level * I.mean(), size=I.shape)
        I = np.clip(I, 0.0, None)         # a camera cannot record negative light

    if bit_depth is not None:
        if not isinstance(bit_depth, (int, np.integer)) or bit_depth < 1:
            raise ValueError(f"bit_depth must be a positive int or None, got {bit_depth!r}.")
        max_counts = 2 ** int(bit_depth) - 1
        if full_scale is not None:
            # Fixed camera exposure: the same intensity always gives the same
            # count, and anything brighter than full_scale saturates.
            I = np.clip(I, 0.0, full_scale)
            scale = max_counts / full_scale
            I = np.round(I * scale) / scale
        else:
            I_max = I.max()
            if I_max > 0:
                # Brightest pixel -> 2**bit_depth - 1 counts, then back to original units.
                scale = max_counts / I_max
                I = np.round(I * scale) / scale

    return I.astype(np.float64)


def amplitude(I: np.ndarray) -> np.ndarray:
    """Amplitude from intensity: sqrt(I). Same shape as I (e.g. L x M)."""
    return np.sqrt(I)


def drift_step(X: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    """One step of slow sample drift: X_new = sqrt(1 - a^2) * X + a * Y.

    Y is a fresh random matrix from make_transmission_matrix, and a in [0, 1]
    sets the drift speed (a = 0: no change, a = 1: a completely new sample).

    Input: X, shape (N, M), complex.
    Returns: X_new, shape (N, M), complex128.
    """
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"drift rate a must be in [0, 1], got {a!r}.")
    N, M = X.shape
    Y = make_transmission_matrix(N, M, rng)
    # The weights satisfy (sqrt(1 - a^2))^2 + a^2 = 1, so E[|X_nm|^2] stays 1:
    # the sample keeps the same statistics, it just slowly "forgets" its old X.
    return np.sqrt(1.0 - a**2) * X + a * Y


class TemperatureSample:
    """The sample's transmission matrix at any temperature T (C); exactly X0 at T_ref.

    The similarity of two temperatures depends only on their difference:
    field correlation corr(X(T1), X(T2)) = exp(-|T1 - T2| / T_c).
    Built as an AR(1) chain on the grid T_k = T_ref + k * step, outward from X_0 = X0 in both directions:
    X_k = rho * X_(k-1) + sqrt(1 - rho^2) * Y_k (k > 0; k < 0 uses X_(k+1)), rho = exp(-step / T_c).
    Going up and going down from T_ref therefore give different samples, and the sample comes back
    only if the temperature comes back to the same value. Y_k has its own seed (seed, k), so the chain is
    the same whatever order the grid is extended in. The grid grows lazily to any temperature asked for.
    Between grid points: linear interpolation, rescaled so that E[|X_nm|^2] stays 1.
    T_c is set by hand (source: thesis drift curve); the model is not measured on our sample.
    """

    def __init__(self, X0: np.ndarray, T_ref: float, T_c: float, step: float, seed: int):
        """X0 (N, M): the sample at T_ref (C); T_c: correlation temperature (C); step: grid step (C)."""
        self.T_ref, self.step, self.seed = T_ref, step, seed
        self.rho = np.exp(-step / T_c)
        self.grid = {0: X0.copy()}
        self.k_min = self.k_max = 0

    def _extend(self, k: int) -> None:
        """Build the chain out to grid index k."""
        N, M = self.grid[0].shape
        a = np.sqrt(1.0 - self.rho ** 2)
        while self.k_max < k:
            self.k_max += 1
            Y = make_transmission_matrix(N, M, np.random.default_rng([self.seed, 0, self.k_max]))
            self.grid[self.k_max] = self.rho * self.grid[self.k_max - 1] + a * Y
        while self.k_min > k:
            self.k_min -= 1
            Y = make_transmission_matrix(N, M, np.random.default_rng([self.seed, 1, -self.k_min]))
            self.grid[self.k_min] = self.rho * self.grid[self.k_min + 1] + a * Y

    def at(self, T: float) -> np.ndarray:
        """The sample at temperature T (C). Returns: X, shape (N, M), complex128 (a new array)."""
        x = (T - self.T_ref) / self.step
        k = int(np.floor(x))
        w = x - k
        self._extend(k)
        if w == 0.0:
            return self.grid[k].copy()
        self._extend(k + 1)
        # Neighbours have E[<X_k, X_k+1>] = rho per entry, so this keeps E[|X_nm|^2] = 1.
        norm = np.sqrt((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * self.rho)
        return ((1 - w) * self.grid[k] + w * self.grid[k + 1]) / norm


def column_correlation(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """Per-column similarity of two transmission matrices.

    For each column j: |<X_a[:, j], X_b[:, j]>| / (||X_a[:, j]|| * ||X_b[:, j]||).
    The absolute value ignores one constant phase per column (per camera
    point), which GGS cannot recover from intensities anyway.

    Input: X_a, X_b, shape (N, M), complex.
    Returns: shape (M,), float64, values in [0, 1] (1 = identical up to phase).
    """
    if X_a.shape != X_b.shape:
        raise ValueError(f"X_a has shape {X_a.shape} but X_b has shape {X_b.shape}.")
    # Inner product <a, b> = sum(conj(a) * b) over the N inputs, per column.
    inner = np.sum(np.conj(X_a) * X_b, axis=0)
    norms = np.linalg.norm(X_a, axis=0) * np.linalg.norm(X_b, axis=0)
    return np.abs(inner) / norms


class FakeLab:
    """Holds the settings, the random generator and the true matrix X_true (N x M)."""

    def __init__(
        self,
        N: int,
        M: int,
        seed: int = 0,
        noise_level: float = 0.0,
        bit_depth: int | None = None,
        full_scale: float | None = None,
        phase_error: float = 0.0,
        flicker: float = 0.0,
    ):
        """Create the lab: rng = default_rng(seed), X_true of shape (N, M).

        noise_level, bit_depth, full_scale, phase_error and flicker are passed
        to measure() on every call.
        """
        self.N = N
        self.M = M
        self.seed = seed
        self.noise_level = noise_level
        self.bit_depth = bit_depth
        self.full_scale = full_scale
        self.phase_error = phase_error
        self.flicker = flicker
        self.rng = np.random.default_rng(seed)
        self.X_true = make_transmission_matrix(N, M, self.rng)

    def patterns(self, L: int, quantize: bool = True) -> np.ndarray:
        """Random SLM patterns E_in, shape (L, N), complex128."""
        return make_slm_patterns(L, self.N, self.rng, quantize=quantize)

    def measure(self, E_in: np.ndarray) -> np.ndarray:
        """Camera intensities I, shape (L, M), for patterns E_in of shape (L, N)."""
        return measure(
            E_in,
            self.X_true,
            rng=self.rng,
            noise_level=self.noise_level,
            bit_depth=self.bit_depth,
            full_scale=self.full_scale,
            phase_error=self.phase_error,
            flicker=self.flicker,
        )

    def drift(self, a: float) -> None:
        """Let the sample drift by one step of rate a; updates X_true (N x M) in place."""
        self.X_true = drift_step(self.X_true, a, self.rng)


if __name__ == "__main__":
    N, M = 64, 64
    L = 5 * N

    lab = FakeLab(N=N, M=M, seed=0)
    E_in = lab.patterns(L, quantize=True)
    I = lab.measure(E_in)

    print("X_true shape:", lab.X_true.shape)
    print("E_in shape:  ", E_in.shape)
    print("I shape:     ", I.shape)
    assert lab.X_true.shape == (N, M)
    assert E_in.shape == (L, N)
    assert I.shape == (L, M)

    # Phase-only SLM: every entry has amplitude 1.
    max_amp_err = np.max(np.abs(np.abs(E_in) - 1.0))
    print(f"max | |E_in| - 1 |: {max_amp_err:.2e}")
    assert max_amp_err < 1e-12, "E_in entries are not unit amplitude"

    # Gray round trip: error must be within one gray step (compared mod 2*pi).
    p = np.random.default_rng(1).uniform(-10.0, 10.0, size=10_000)
    p_back = gray_to_phase(phase_to_gray(p))
    wrapped_err = np.abs(np.angle(np.exp(1j * (p_back - p))))
    print(f"max phase round-trip error: {wrapped_err.max():.4f} rad (limit {TWO_PI / 255:.4f})")
    assert wrapped_err.max() <= TWO_PI / 255, "phase <-> gray round trip too inaccurate"

    # Each output is a sum of N unit-size random terms -> mean intensity ~ N.
    mean_I = I.mean()
    print(f"mean(I): {mean_I:.2f} (expected ~ {N})")
    assert abs(mean_I - N) < 0.1 * N, "mean intensity not close to N"

    # Reproducibility: same seed -> identical results.
    lab2 = FakeLab(N=N, M=M, seed=0)
    I2 = lab2.measure(lab2.patterns(L, quantize=True))
    assert np.array_equal(lab.X_true, lab2.X_true), "X_true differs for the same seed"
    assert np.array_equal(I, I2), "I differs for the same seed"
    print("same seed -> identical X_true and I: OK")

    # Fixed full_scale: every batch is quantized with the same step
    # full_scale / (2**bit_depth - 1), unlike the per-batch I.max() scaling.
    bits, fs = 8, 4.0 * N
    step = fs / (2**bits - 1)
    cam = FakeLab(N=N, M=M, seed=0, bit_depth=bits, full_scale=fs)
    I_a = cam.measure(cam.patterns(L))
    I_b = cam.measure(cam.patterns(L))
    for I_batch in (I_a, I_b):
        counts = I_batch / step
        assert np.allclose(counts, np.round(counts), atol=1e-9), "values are not multiples of the step"
        assert I_batch.max() <= fs, "values above full_scale were not clipped"
    # Measured step = smallest gap between distinct recorded levels.
    step_a = np.diff(np.unique(I_a)).min()
    step_b = np.diff(np.unique(I_b)).min()
    print(f"quantization step: batch A {step_a:.6f}, batch B {step_b:.6f} (expected {step:.6f})")
    assert np.isclose(step_a, step) and np.isclose(step_b, step), "batches use different quantization steps"

    # Drift: correlation with the starting matrix should fall slowly.
    # Theory: after n steps the "memory" of X0 is sqrt(1 - a^2)^n.
    a = 0.05
    drift_rng = np.random.default_rng(2)
    X0 = lab.X_true.copy()
    X = X0
    corr_history = [column_correlation(X0, X).mean()]
    print(f"drift a = {a}: step  0, mean column correlation {corr_history[0]:.4f}")
    for n in range(1, 51):
        X = drift_step(X, a, drift_rng)
        if n % 10 == 0:
            corr = column_correlation(X0, X).mean()
            corr_history.append(corr)
            print(f"drift a = {a}: step {n:2d}, mean column correlation {corr:.4f} "
                  f"(theory ~ {np.sqrt(1 - a**2) ** n:.4f})")
    assert np.isclose(corr_history[0], 1.0), "correlation of X0 with itself is not 1"
    assert all(np.diff(corr_history) < 0), "correlation did not decrease steadily"
    assert X.shape == X0.shape

    # Temperature model: exactly X0 at T_ref; the correlation depends only on the temperature difference.
    from config import DRIFT_TEST, OPTICS
    T_c, T_ref = DRIFT_TEST["T_c"], OPTICS["T_ref_C"]
    sample = TemperatureSample(X0, T_ref, T_c, OPTICS["T_step_C"], seed=9)
    assert np.array_equal(sample.at(T_ref), X0), "T_ref does not give X0"

    def corr(T1: float, T2: float) -> float:
        """Mean column correlation of the sample at T1 and T2."""
        return float(column_correlation(sample.at(T1), sample.at(T2)).mean())

    pairs = [(19.6, 20.4), (20.0, 20.4), (20.4, 20.8), (19.6, 19.2), (20.013, 20.413), (T_ref, T_ref + T_c)]
    c = {pair: corr(*pair) for pair in pairs}
    for (T1, T2), value in c.items():
        print(f"TemperatureSample: corr({T1:g}, {T2:g}) = {value:.4f} (theory exp(-{abs(T2 - T1):g}/{T_c:g}) = "
              f"{np.exp(-abs(T2 - T1) / T_c):.4f})")
    same_gap = [c[(20.0, 20.4)], c[(20.4, 20.8)], c[(19.6, 19.2)], c[(20.013, 20.413)]]
    assert max(same_gap) - min(same_gap) < 0.05, "equal differences do not give equal correlation"
    assert c[(19.6, 20.4)] < c[(20.0, 20.4)] - 0.1, "19.6 vs 20.4 is not clearly lower than 20.0 vs 20.4"
    assert abs(c[(T_ref, T_ref + T_c)] - np.exp(-1)) < 0.05, "corr at a difference of T_c is not exp(-1)"
    # Up and down from T_ref are different samples (the old model gave the same one).
    assert corr(T_ref - 0.4, T_ref + 0.4) < 0.7, "up and down from T_ref give the same sample"
    X_far, X_cold = sample.at(30.0), sample.at(5.0)
    print(f"TemperatureSample: 30 C and 5 C work (grid {sample.T_ref + sample.k_min * sample.step:.2f} .. "
          f"{sample.T_ref + sample.k_max * sample.step:.2f} C, {len(sample.grid)} points), "
          f"mean |X|^2 at 30 C {np.mean(np.abs(X_far) ** 2):.3f}, corr(20, 30) {corr(20.0, 30.0):.3f}")
    assert X_far.shape == X0.shape and abs(np.mean(np.abs(X_cold) ** 2) - 1) < 0.1
    # The same chain whatever order the grid is built in.
    other = TemperatureSample(X0, T_ref, T_c, OPTICS["T_step_C"], seed=9)
    assert np.allclose(other.at(19.6), sample.at(19.6)) and np.allclose(other.at(20.4), sample.at(20.4))
    print("TemperatureSample: equal differences give equal correlation, up != down, any temperature: OK")

    # Imperfections are random but still reproducible with the same seed.
    imperfect = [FakeLab(N=N, M=M, seed=0, phase_error=0.3, flicker=0.05) for _ in range(2)]
    I_imp = [f.measure(f.patterns(L)) for f in imperfect]
    assert np.array_equal(I_imp[0], I_imp[1]), "imperfect lab is not reproducible"
    print("phase_error + flicker, same seed -> identical I: OK")

    # Phase error is fresh for every call: the same pattern gives a slightly
    # different photo each time, but the speckle is still mostly the same.
    pe_lab = FakeLab(N=N, M=M, seed=0, phase_error=0.3)
    E_pe = pe_lab.patterns(L)
    I_pe1, I_pe2 = pe_lab.measure(E_pe), pe_lab.measure(E_pe)
    r_pe = np.corrcoef(I_pe1.ravel(), I_pe2.ravel())[0, 1]
    print(f"phase_error 0.3: same pattern twice, Pearson r = {r_pe:.4f} (limit > 0.8)")
    assert not np.array_equal(I_pe1, I_pe2), "phase error is not fresh per call"
    assert r_pe > 0.8, "phase error changes the photos too much"

    # Flicker scales whole photos: row mean / ideal row mean has std ~ flicker.
    fl_lab = FakeLab(N=N, M=M, seed=0, flicker=0.05)
    E_fl = fl_lab.patterns(L)
    row_ratio = fl_lab.measure(E_fl).mean(axis=1) / measure(E_fl, fl_lab.X_true).mean(axis=1)
    print(f"flicker 0.05: std of photo brightness ratio = {row_ratio.std():.4f} (limit 0.03..0.07)")
    assert 0.03 < row_ratio.std() < 0.07, "flicker strength is wrong"

    # Shape check must catch a mismatch.
    try:
        measure(E_in[:, :10], lab.X_true)
        raise AssertionError("shape mismatch was not detected")
    except ValueError:
        pass

    print("lab.py self-test passed")
