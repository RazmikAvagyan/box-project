"""
plots.py - plotting functions only (no GGS, no FakeLab, no physics).

Each function takes plain numbers/arrays, draws ONE plot, saves it as a PNG
(dpi=150), closes the figure and returns the saved path. The focus figures
(camera images side by side) are the only figures with more than one axes.

Camera images: the simulation has M camera points; to show them as an image
the M values are reshaped to (h, w) with h * w == M (8 x 8 for M = 64).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # files only, no windows; also works without a display
import matplotlib.pyplot as plt
import numpy as np

from config import THERMOSTAT

FIGSIZE = (10, 5)       # inches, the same for every figure
GRID_ALPHA = 0.3
CMAP = "inferno"
LIMIT_C = THERMOSTAT["limit_C"]
plt.rcParams["font.size"] = 12


def to_image(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape M camera values (M,) to an image of shape (h, w), with h * w == M."""
    return np.reshape(values, shape)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Save fig as PNG at path (creating the folder), close it, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _y_range(ax: plt.Axes, *arrays: np.ndarray, center: float | None = None,
             min_span: float | None = None, margin: float = 0.2) -> tuple[float, float]:
    """Set y-limits from the 1st..99th percentile of all arrays together, plus margin * span.

    min_span: the range is never smaller than this (tiny noise does not look huge).
    center: make the range symmetric around this value.
    Returns the (low, high) limits that were set.
    """
    values = np.concatenate([np.ravel(a) for a in arrays])
    lo, hi = np.percentile(values, [1, 99])
    if center is not None:
        half = max(hi - center, center - lo)
        lo, hi = center - half, center + half
    if min_span is not None and hi - lo < min_span:
        mid = (lo + hi) / 2 if center is None else center
        lo, hi = mid - min_span / 2, mid + min_span / 2
    pad = margin * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    return lo - pad, hi + pad


def _stats(values: np.ndarray, fmt: str = ".3f", unit: str = "") -> str:
    """Short 'min / max / mean' text for a legend or title."""
    return (f"min {values.min():{fmt}}{unit}, max {values.max():{fmt}}{unit}, "
            f"mean {values.mean():{fmt}}{unit}")


def _new_plot(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    """One figure with one axes in the common style."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=GRID_ALPHA)
    return fig, ax


def _show_image(ax: plt.Axes, image: np.ndarray, vmin: float, vmax: float):
    """Draw one camera image (h, w) without axis ticks; returns the image handle for a colorbar."""
    im = ax.imshow(image, cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_ggs_convergence(history: list[float], path: str | Path) -> Path:
    """Mean fit error per GGS iteration (len = number of iterations), linear y axis."""
    fig, ax = _new_plot("Learning the sample map (GGS): error per iteration", "iteration",
                        "fit error (lower is better)")
    ax.plot(np.arange(1, len(history) + 1), history,
            label=f"start {history[0]:.3f}, end {history[-1]:.3f}")
    lo, hi = _y_range(ax, np.asarray(history))
    # The first iterations lie above the 99th percentile; keep the start value visible.
    ax.set_ylim(lo, max(hi, 1.05 * history[0]))
    ax.legend(loc="best")
    return _save(fig, path)


def plot_focus(
    I_random: np.ndarray,
    I_focus: np.ndarray,
    shape: tuple[int, int],
    target: int,
    path: str | Path,
) -> Path:
    """Camera images (M,) for a random pattern and a focus pattern, side by side.

    Each image has its own colour bar, so the speckle structure is visible too;
    the titles give mean and max to compare the brightness. The target pixel
    (index into M) is circled on the focus image.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, I, name in [(axes[0], I_random, "Random SLM pattern: speckle"),
                          (axes[1], I_focus, "Focusing SLM pattern: bright spot")]:
        im = _show_image(ax, to_image(I, shape), 0.0, I.max())
        ax.set_title(f"{name}\nmean {I.mean():.0f}, max {I.max():.0f}")
        fig.colorbar(im, ax=ax, label="intensity (a.u.)", fraction=0.046, pad=0.04)
    row, col = divmod(target, shape[1])   # row-major reshape: index = row * w + col
    axes[1].plot(col, row, "o", markersize=14, markerfacecolor="none", markeredgecolor="cyan")
    return _save(fig, path)


def plot_focus_grid(
    images: dict[str, tuple[np.ndarray, np.ndarray]],
    etas: dict[str, tuple[float, float]],
    shape: tuple[int, int],
    target: int,
    path: str | Path,
    title: str = "Focus spot at start and end",
    col_titles: tuple[str, str] = ("start", "end"),
) -> Path:
    """Grid of focus photos: one row per case (label -> (I_start, I_end), each (M,)), two columns.

    One colour scale for all images (plus one colour bar), so a fading spot is visible.
    etas maps label -> (eta_start, eta_end), shown in each image title. The target is circled.
    """
    n_rows = len(images)
    fig, axes = plt.subplots(n_rows, 2, figsize=(FIGSIZE[0], 4.5 * n_rows), squeeze=False)
    vmax = max(I.max() for pair in images.values() for I in pair)
    row, col = divmod(target, shape[1])   # row-major reshape: index = row * w + col
    for (label, pair), row_axes in zip(images.items(), axes):
        for ax, I, eta, col_title in zip(row_axes, pair, etas[label], col_titles):
            im = _show_image(ax, to_image(I, shape), 0.0, vmax)
            ax.plot(col, row, "o", markersize=14, markerfacecolor="none", markeredgecolor="cyan")
            ax.set_title(f"{col_title}: η = {eta:.1f}")
        row_axes[0].set_ylabel(label)   # row title
    fig.colorbar(im, ax=axes, label="intensity (a.u.)")
    fig.suptitle(title)
    return _save(fig, path)


def plot_enhancement(eta: np.ndarray, eta_theory: float, N: int, path: str | Path) -> Path:
    """Enhancement eta (M,) per camera point, with the theory level eta_theory for N SLM squares.

    Bars start at 0 (fixed range), so the bar heights compare honestly.
    """
    fig, ax = _new_plot("Focus brightness at each camera point", "camera point index",
                        "enhancement η (spot / background)")
    ax.bar(np.arange(len(eta)), eta, label=f"measured η ({_stats(eta, '.1f')})")
    ax.axhline(eta_theory, linestyle="--", color="black", label=f"theory for N = {N}: {eta_theory:.1f}")
    # Head room so the legend sits above the bars instead of on them.
    ax.set_ylim(0, 1.45 * max(eta.max(), eta_theory))
    ax.legend(loc="upper center", ncol=2, fontsize=10)
    return _save(fig, path)


def plot_correlation_vs_time(
    t: np.ndarray,
    curves: dict[str, np.ndarray],
    path: str | Path,
    title: str = "Correlation over time",
    colors: dict[str, str] | None = None,
) -> Path:
    """Correlation curves against time t (T,) in minutes; curves maps label -> r (T,).

    Fixed y-range 0 to 1. The legend shows start -> end and the minimum of each curve.
    """
    fig, ax = _new_plot(title, "time (min)", "prediction correlation r (1 = perfect)")
    for label, r in curves.items():
        ax.plot(t, r, marker="o", markersize=4, color=(colors or {}).get(label),
                label=f"{label}: {r[0]:.3f} -> {r[-1]:.3f}, min {r.min():.3f}")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    return _save(fig, path)


def plot_correlation_and_temperature(
    t: np.ndarray,
    r: np.ndarray,
    t_curve: np.ndarray,
    y_curve: np.ndarray,
    path: str | Path,
    title: str,
    color: str,
    right_label: str,
    right_ylim: tuple[float, float] | None = None,
) -> Path:
    """Correlation and temperature over time on two y-axes (time in minutes).

    Left axis: correlation r (T,) at the photo times t (T,), as a line with dots, 0 to 1.
    Right axis: a temperature curve y_curve (n,) over t_curve (n,), e.g. ΔT in °C or the
    deviation from the target in °C (right_label says which), as a grey dashed line. One legend for both.
    """
    fig, ax = _new_plot(title, "time (min)", "prediction correlation r")
    line_r, = ax.plot(t, r, marker="o", markersize=4, color=color,
                      label=f"prediction correlation: {r[0]:.3f} -> {r[-1]:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("prediction correlation r", color=color)
    ax.tick_params(axis="y", colors=color)

    ax_T = ax.twinx()
    line_T, = ax_T.plot(t_curve, y_curve, linestyle="--", color="gray", label=right_label)
    ax_T.set_ylabel(right_label, color="gray")
    ax_T.tick_params(axis="y", colors="gray")
    if right_ylim is not None:
        ax_T.set_ylim(*right_ylim)
    else:
        _y_range(ax_T, y_curve, min_span=0.1)
    ax.legend([line_r, line_T], [line_r.get_label(), line_T.get_label()], loc="best")
    return _save(fig, path)


def plot_box_deviation(
    t_s: np.ndarray,
    T_true: np.ndarray,
    target: np.ndarray,
    path: str | Path,
    title: str,
    t_from: float = 0.0,
) -> Path:
    """Deviation (T_true - target) in °C over time t_s in seconds, from t_from on, with the +-0.03 °C limit.

    Symlog axis, linear within +-0.03 °C and logarithmic beyond, so a 1 °C step and the
    few-millidegree wobble after it are both visible; y from -0.06 to 1.2 °C.
    """
    keep = t_s >= t_from
    t, dev = t_s[keep], T_true[keep] - target[keep]
    fig, ax = _new_plot(title, "time (s)", f"distance from target (°C, log scale above {LIMIT_C:g})")
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.plot(t, dev, linewidth=2.0, color="black", label="true temperature")
    ax.axhline(LIMIT_C, linestyle="--", color="green", label=f"limit ±{LIMIT_C:g} °C")
    ax.axhline(-LIMIT_C, linestyle="--", color="green")
    ax.set_yscale("symlog", linthresh=LIMIT_C, linscale=1.5)
    ax.set_ylim(-0.06, 1.2)
    ticks = [-0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3, 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_kalman_error(
    t_s: np.ndarray,
    T_true: np.ndarray,
    T_raw: np.ndarray,
    T_filtered: np.ndarray,
    path: str | Path,
) -> Path:
    """Error (measured - true) in °C of the raw readings and of the Kalman estimate (all (n_steps,)) over t_s (s).

    Raw: grey dots (every 5th reading). Kalman: blue line. A black line at 0 marks no error.
    Symlog y axis from -1 to 1 °C: linear within +-0.02 °C (a small band around 0, thin dotted
    lines mark the change), logarithmic beyond, so the spikes fit too. The second title line
    gives both error stds over the whole run and their ratio. Legend below the axes.
    """
    raw_err, kal_err = T_raw - T_true, T_filtered - T_true
    std_raw, std_kal = np.std(raw_err), np.std(kal_err)
    fig, ax = _new_plot("How far off is each measurement? Raw sensor vs Kalman filter\n"
                        f"raw error std {std_raw:.4f} °C, Kalman error std {std_kal:.4f} °C "
                        f"({std_raw / std_kal:.1f} times smaller)",
                        "time (s)", "error = measured - true (°C, log scale above 0.02)")
    ax.scatter(t_s[::5], raw_err[::5], s=6, alpha=0.4, color="gray", label="raw sensor - true")
    ax.plot(t_s, kal_err, linewidth=1.5, color="tab:blue", label="Kalman filtered - true")
    ax.axhline(0.0, linewidth=1.0, color="black", label="perfect (no error)")
    for y in (-0.02, 0.02):
        ax.axhline(y, linewidth=0.8, linestyle=":", color="gray")
    ax.set_yscale("symlog", linthresh=0.02, linscale=0.5)
    ax.set_ylim(-1.0, 1.0)
    ticks = [-1, -0.3, -0.1, -0.02, 0, 0.02, 0.1, 0.3, 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # Below the axes, so it never covers the dots.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    return _save(fig, path)


def plot_deviation_comparison(
    t_s: np.ndarray,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """Deviation (T_true - target) in °C of several runs over time t_s (n_steps,) in seconds, with +-0.03 °C.

    runs maps label -> (T_true (n_steps,), target (n_steps,)). The legend gives std and
    max |deviation| of each run.
    """
    fig, ax = _new_plot(title, "time (s)", "distance from target (°C)")
    ax.axhline(0.0, linewidth=0.8, color="black")
    devs = []
    for label, (T_true, target) in runs.items():
        dev = T_true - target
        devs.append(dev)
        ax.plot(t_s, dev, linewidth=1.2, color=(colors or {}).get(label),
                label=f"{label}: std {dev.std():.4f} °C, max |dev| {np.abs(dev).max():.4f} °C")
    ax.axhline(LIMIT_C, linestyle="--", color="green", label=f"limit ±{LIMIT_C:g} °C")
    ax.axhline(-LIMIT_C, linestyle="--", color="green")
    lim = max(1.5 * LIMIT_C, 1.2 * max(np.abs(d).max() for d in devs))
    ax.set_ylim(-lim, lim)
    # Below the axes, so it never covers the curves or the limit lines.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    return _save(fig, path)


def _event_lines(ax: plt.Axes, events: list[tuple[float, str]], key_y: float, top: bool = False) -> None:
    """Grey vertical lines at the event times (s, x axis in minutes), numbered 1, 2, ... at the bottom
    (or at the top if top=True), and one key line 'number label, ...' in grey at axes height key_y."""
    for number, (t_event, _) in enumerate(events, start=1):
        ax.axvline(t_event / 60.0, linewidth=1.0, color="gray")
        # A small number at the bottom or top of each line (y in axes coordinates).
        ax.text(t_event / 60.0, 0.98 if top else 0.02, f"{number} ", va="top" if top else "bottom", ha="right",
                fontsize=10, color="gray", transform=ax.get_xaxis_transform())
    key = ", ".join(f"{number} {label}" for number, (_, label) in enumerate(events, start=1))
    ax.text(0.5, key_y, key, ha="center", va="top", fontsize=10, color="gray", transform=ax.transAxes)


def plot_scenario_overview(
    t_s: np.ndarray,
    T_true: np.ndarray,
    T_kalman: np.ndarray,
    target: np.ndarray,
    T_room: np.ndarray,
    events: list[tuple[float, str]],
    broken: tuple[float, float],
    path: str | Path,
) -> Path:
    """Box (true and Kalman), target and room temperature (all (n,)) in °C over t_s (n,) in seconds, linear axis.

    y from 19 °C to about 1 °C above the highest value; time in minutes. broken: (start_s, end_s) of the
    cooler failure, shaded light red. events: numbered grey lines with a key line, as in plot_scenario.
    """
    fig, ax = _new_plot("Box and room temperature during the 2-hour scenario (normal scale)",
                        "time (min)", "temperature (°C)")
    t = t_s / 60.0
    ax.axvspan(broken[0] / 60.0, broken[1] / 60.0, color="red", alpha=0.15, label="cooler broken")
    ax.plot(t, T_true, linewidth=1.5, color="black", label="true temperature")
    ax.plot(t, T_kalman, linewidth=0.8, color="tab:blue", label="Kalman estimate")
    ax.plot(t, target, linestyle="--", linewidth=1.2, color="green", label="target")
    ax.plot(t, T_room, linestyle=":", linewidth=2.0, color="tab:orange", label="room temperature")
    ax.set_ylim(19.0, max(T_true.max(), T_room.max()) + 1.0)
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10)
    _event_lines(ax, events, key_y=-0.32)
    return _save(fig, path)


def plot_scenario(
    t_s: np.ndarray,
    T_true: np.ndarray,
    T_kalman: np.ndarray,
    target: np.ndarray,
    events: list[tuple[float, str]],
    path: str | Path,
    T_room: np.ndarray | None = None,
) -> Path:
    """True and Kalman temperature (n,) over time t_s (n,) in seconds, with the target (n,) and labelled events.

    Shown as distance from the first target in °C on a symlog axis: linear within +-LIMIT_C, so the
    small wobble during the hold is visible, and logarithmic beyond, so a failure of a few °C fits.
    T_room (n,), optional: the room temperature as a dotted orange line on the same axis.
    events: (time_s, label) pairs, drawn as grey vertical lines numbered 1, 2, ... at the bottom;
    a key line below the legend says what each number is. Time axis in minutes.
    """
    ref = target[0]
    fig, ax = _new_plot(f"Box temperature during the 2-hour scenario "
                        f"(zoom: linear within ±{LIMIT_C:g} °C, log scale above)",
                        "time (min)", f"distance from {ref:g} °C (°C, log scale above {LIMIT_C:g})")
    t = t_s / 60.0
    ax.plot(t, T_true - ref, linewidth=1.5, color="black", label="true temperature")
    ax.plot(t, T_kalman - ref, linewidth=1.0, color="tab:blue", alpha=0.7, label="Kalman estimate")
    ax.plot(t, target - ref, linestyle="--", linewidth=1.2, color="green", label="target")
    highest = np.max(T_true - ref)
    if T_room is not None:
        ax.plot(t, T_room - ref, linestyle=":", linewidth=2.0, color="tab:orange", label="room temperature")
        highest = max(highest, np.max(T_room - ref))
    ax.set_yscale("symlog", linthresh=LIMIT_C, linscale=1.5)
    top = max(1.0, 1.5 * highest)
    bottom = min(-0.1, 1.5 * min(np.min(T_true - ref), np.min(T_kalman - ref)))
    ax.set_ylim(bottom, top)
    ticks = [v for v in [-0.3, -0.1, -0.03, 0, 0.03, 0.1, 0.5, 1, 3, 10] if bottom <= v <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    _event_lines(ax, events, key_y=-0.32)
    return _save(fig, path)


def plot_cooler_power(
    t_s: np.ndarray,
    power_pct: np.ndarray,
    real_pct: np.ndarray,
    events: list[tuple[float, str]],
    broken: tuple[float, float],
    path: str | Path,
) -> Path:
    """Cooler power over time t_s (n,) in seconds (axis in minutes), in % of full power (n,):
    power_pct = what the PID asks for (dark blue dashed line, on top), real_pct = what the cooler really gives
    (thick orange line). Both are smoothed with a centered 60 s moving average first, so the plot is easy to read.

    broken: (start_s, end_s) when the cooler gives nothing, shaded light red.
    events: numbered grey vertical lines with a key line below the legend, as in plot_scenario.
    """
    fig, ax = _new_plot("Cooler power: asked by the PID vs. really given (0 % while the cooler is broken)",
                        "time (min)", "cooler power (% of full power)")
    # Centered moving average over 60 s, one point per sample; near the ends only the samples that exist count.
    n = int(round(60.0 / (t_s[1] - t_s[0])))
    window = np.ones(n)
    count = np.convolve(np.ones_like(power_pct), window, mode="same")
    asked, real = [np.convolve(x, window, mode="same") / count for x in (power_pct, real_pct)]
    t = t_s / 60.0
    ax.axvspan(broken[0] / 60.0, broken[1] / 60.0, color="red", alpha=0.15, label="cooler broken")
    ax.plot(t, real, linewidth=3.0, color="tab:orange", label="real cooling (what the cooler gives)")
    ax.plot(t, asked, linestyle="--", linewidth=1.5, color="darkblue", label="cooling asked by the PID")
    ax.set_ylim(0, 105)
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    _event_lines(ax, events, key_y=-0.32)
    return _save(fig, path)

def plot_policy_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    events: list[tuple[float, str]],
    path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Focus brightness during the scenario: never re-measure vs. fixed rules",
) -> Path:
    """Focus enhancement eta over time for several operator policies.

    curves maps label -> (t_s (n,), eta (n,), measured (n,) bool): small markers where that policy
    measured a new map. events: numbered grey lines with a key line, as in plot_scenario.
    The legend gives the mean eta of each policy. Time axis in minutes.
    """
    fig, ax = _new_plot(title,
                        "time (min)", "enhancement η (spot / background)")
    for label, (t_s, eta, measured) in curves.items():
        color = (colors or {}).get(label)
        ax.plot(t_s / 60.0, eta, linewidth=1.5, color=color, label=f"{label}: mean η {eta.mean():.1f}")
        if measured.any():
            ax.plot(t_s[measured] / 60.0, eta[measured], "o", markersize=6, markerfacecolor="none",
                    markeredgecolor=color, label=f"{label}: new map measured")
    t_all = np.concatenate([c[0] for c in curves.values()]) / 60.0
    ax.set_xlim(0, t_all.max())
    ax.set_ylim(0, 1.15 * max(c[1].max() for c in curves.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    _event_lines(ax, events, key_y=-0.32, top=True)
    return _save(fig, path)


if __name__ == "__main__":
    import tempfile

    from box_simulation import thermostat
    from optics_simulation import ggs, lab, metrics

    N = M = 64
    shape = (8, 8)

    fake = lab.FakeLab(N=N, M=M, seed=10)
    E_in = fake.patterns(8 * N)
    X_est, info = ggs.recover(E_in, lab.amplitude(fake.measure(E_in)), np.random.default_rng(1))
    I_random = fake.measure(fake.patterns(1))[0]
    I_focus = fake.measure(metrics.focus_pattern(X_est, 27))[0]
    res = metrics.focus_test(fake, X_est)

    # Demo time series: drift a little at every step; the old X_est stays fixed.
    t = np.linspace(0, 5, 21)
    E_test = lab.make_slm_patterns(100, N, np.random.default_rng(5))
    drift_rng = np.random.default_rng(7)
    correlations = {}
    for label, a in [("no temperature control", 0.15), ("with temperature control", 0.02)]:
        X = fake.X_true
        r = []
        for step in range(len(t)):
            if step > 0:
                X = lab.drift_step(X, a, drift_rng)
            r.append(metrics.prediction_correlation(E_test, X_est, lab.measure(E_test, X)))
        correlations[label] = np.array(r)

    hold = thermostat.simulate(60, control=True, T_start=20.0, target=20.0, seed=0, warmup_s=60)
    off = thermostat.simulate(60, control=False, T_start=20.0, seed=0)
    step = thermostat.simulate(60, control=True, T_start=21.0, target=20.0, seed=1)
    colors = {"no temperature control": "red", "with temperature control": "blue"}

    # Test plots go into a temporary folder that is deleted at the end.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        paths = [
            plot_ggs_convergence(info["history"], out / "ggs_convergence.png"),
            plot_focus(I_random, I_focus, shape, 27, out / "focus.png"),
            plot_enhancement(res["eta"], res["eta_theory"], N, out / "enhancement.png"),
            plot_focus_grid({"case A": (I_focus, I_random), "case B": (I_focus, I_focus)},
                            {"case A": (50.0, 1.0), "case B": (50.0, 50.0)}, shape, 27, out / "focus_grid.png"),
            plot_correlation_vs_time(t, correlations, out / "correlation.png", colors=colors),
            plot_correlation_and_temperature(t, correlations["no temperature control"], off["t_s"] / 60,
                                             off["T_true"] - off["T_true"][0], out / "correlation_and_temperature.png",
                                             "Correlation and temperature", "red", "ΔT (°C)"),
            plot_correlation_and_temperature(t, correlations["with temperature control"], hold["t_s"] / 60,
                                             hold["T_true"] - 20.0, out / "correlation_and_deviation.png",
                                             "Correlation and deviation", "blue", "distance from 20 °C (°C)",
                                             right_ylim=(-0.045, 0.045)),
            plot_box_deviation(step["t_s"], step["T_true"], step["target"], out / "box_step_deviation.png",
                               "Step deviation"),
            plot_kalman_error(hold["t_s"], hold["T_true"], hold["T_raw"], hold["T_filtered"], out / "kalman_error.png"),
            plot_scenario(hold["t_s"], hold["T_true"], hold["T_filtered"], hold["target"], [(30.0, "event")],
                          out / "scenario.png", T_room=np.full_like(hold["t_s"], 22.0)),
            plot_cooler_power(hold["t_s"], hold["duty"] / 10.23, hold["duty"] / 10.23, [(30.0, "event")], (20.0, 40.0),
                              out / "cooler_power.png"),
            plot_policy_comparison({"a": (t * 60, correlations["no temperature control"] * 50, t > 2),
                                    "b": (t * 60, correlations["with temperature control"] * 50, t < 0)},
                                   [(60.0, "event")], out / "policy_comparison.png"),
            plot_scenario_overview(hold["t_s"], hold["T_true"], hold["T_filtered"], hold["target"],
                                   np.full_like(hold["t_s"], 22.0), [(30.0, "event")], (20.0, 40.0),
                                   out / "scenario_overview.png"),
            plot_deviation_comparison(hold["t_s"], {"a": (hold["T_true"], hold["target"]),
                                                    "b": (off["T_true"], off["target"])},
                                      out / "deviation_comparison.png", "Comparison"),
        ]
        for p in paths:
            assert p.exists() and p.stat().st_size > 0, f"{p.name} is missing or empty"
            print(f"ok: {p.name}")
    print("plots.py self-test passed")
