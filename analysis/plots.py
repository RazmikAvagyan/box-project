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

from config import TARGET_C, THERMOSTAT

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
    styles: dict[str, str] | None = None,
) -> Path:
    """Correlation curves against time t (T,) in minutes; curves maps label -> r (T,); optional colour and
    line style per label.

    Fixed y-range 0 to 1. The legend shows start -> end and the minimum of each curve (below the plot for
    more than four curves).
    """
    fig, ax = _new_plot(title, "time (min)", "prediction correlation r (1 = perfect)")
    for label, r in curves.items():
        ax.plot(t, r, marker="o", markersize=4, color=(colors or {}).get(label),
                linestyle=(styles or {}).get(label, "-"), label=f"{label}: {r[0]:.3f} -> {r[-1]:.3f}, min {r.min():.3f}")
    ax.set_ylim(0, 1)
    if len(curves) > 4:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    else:
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
    limit: float | None = None,
) -> Path:
    """Correlation and temperature over time on two y-axes (time in minutes).

    Left axis: correlation r (T,) at the photo times t (T,), as a line with dots, 0 to 1.
    Right axis: a temperature curve y_curve (n,) over t_curve (n,), e.g. ΔT in °C or the
    deviation from the target in °C (right_label says which), as a grey dashed line. One legend for both.
    limit: optional, thin grey dotted lines at +-limit (°C) on the right axis, in the legend.
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
    lines = [line_r, line_T]
    if limit is not None:
        line_lim = ax_T.axhline(limit, linestyle=":", linewidth=0.8, color="gray", label=f"limit ±{limit:g} °C")
        ax_T.axhline(-limit, linestyle=":", linewidth=0.8, color="gray")
        lines.append(line_lim)
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    return _save(fig, path)


def plot_box_deviation(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
    t_from: float = 0.0,
) -> Path:
    """Deviation (T_true - target) in °C over time, from t_from on, with the +-0.03 °C limit;
    curves maps legend label -> (t_s (n,), T_true (n,), target (n,)), times in seconds.

    Symlog axis, linear within +-0.03 °C and logarithmic beyond, so a 1 °C step and the
    few-millidegree wobble after it are both visible; y from the data (1.2 x the lowest and highest
    deviation of all curves, at least +-2 x LIMIT_C), so every curve shows on both sides.
    """
    fig, ax = _new_plot(title, "time (s)", f"distance from target (°C, log scale above {LIMIT_C:g})")
    ax.axhline(0.0, linewidth=0.8, color="black")
    devs = []
    for label, (t_s, T_true, target) in curves.items():
        keep = t_s >= t_from
        devs.append(T_true[keep] - target[keep])
        ax.plot(t_s[keep], devs[-1], linewidth=2.0, color=(colors or {}).get(label, "black"), label=label)
    ax.axhline(LIMIT_C, linestyle="--", color="green", label=f"limit ±{LIMIT_C:g} °C")
    ax.axhline(-LIMIT_C, linestyle="--", color="green")
    ax.set_yscale("symlog", linthresh=LIMIT_C, linscale=1.5)
    dev = np.concatenate(devs)
    bottom, top = min(-2 * LIMIT_C, 1.2 * dev.min()), max(2 * LIMIT_C, 1.2 * dev.max())
    ax.set_ylim(bottom, top)
    ticks = [v for v in [-1, -0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3, 1] if bottom <= v <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10)
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
    (or at the top if top=True), and one key line 'number label, ...' in grey at axes height key_y, or lower
    if the legend reaches below that, so the legend never covers the key line (savefig's tight box adds the space)."""
    legend = ax.get_legend()
    if legend is not None:
        ax.figure.canvas.draw()
        box = legend.get_window_extent().transformed(ax.transAxes.inverted())
        key_y = min(key_y, box.y0 - 0.03)
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


def plot_real_box_test(
    t_s: np.ndarray,
    T_box: np.ndarray,
    target: np.ndarray,
    T_room: np.ndarray,
    T_probe: np.ndarray,
    power_pct: np.ndarray,
    path: str | Path,
    title: str = "Real box test",
) -> Path:
    """Logged real box: box temperature, target, room temperature (dashed) and room probe (dotted),
    all (n,) in °C over t_s (n,) in seconds (axis in minutes); cooler power (n,) in % on a second axis.

    Room values may be NaN (failed reads). Grey vertical lines at each target change.
    """
    fig, ax = _new_plot(title, "time (min)", "temperature (°C)")
    t = t_s / 60.0
    ax.plot(t, T_box, linewidth=1.5, color="black", label="box temperature")
    ax.plot(t, target, linewidth=1.2, color="green", label="target")
    ax.plot(t, T_room, linestyle="--", linewidth=1.5, color="tab:orange", label="room (next to box)")
    ax.plot(t, T_probe, linestyle=":", linewidth=2.0, color="tab:red", label="room probe (far side)")
    for i in np.flatnonzero(np.diff(target) != 0) + 1:
        ax.axvline(t[i], linewidth=1.0, color="gray")
        ax.text(t[i], 0.98, f" {target[i - 1]:g} → {target[i]:g} °C", va="top", ha="left", fontsize=9,
                color="gray", transform=ax.get_xaxis_transform())
    _y_range(ax, *[a[np.isfinite(a)] for a in (T_box, target, T_room, T_probe)], min_span=1.0, margin=0.1)
    ax.set_xlim(t[0], t[-1])
    ax2 = ax.twinx()
    ax2.plot(t, power_pct, linewidth=1.0, color="tab:blue", alpha=0.6, label="cooler power")
    ax2.set_ylabel("cooler power (% of full PWM)", color="tab:blue")
    ax2.set_ylim(0, 105)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=5, fontsize=10)
    return _save(fig, path)


def _phase_spans(ax: plt.Axes, spans: list[tuple[float, float, str]], label: bool = True) -> None:
    """Grey lines at the phase starts (x in minutes) and the phase names at the top; the hold is shaded."""
    for start, end, name in spans:
        ax.axvline(start, linewidth=1.0, color="gray")
        if name == "hold":
            ax.axvspan(start, end, color="gray", alpha=0.12)
        if label:
            ax.text((start + end) / 2, 0.98, name + (" (not fitted)" if name == "hold" else ""), ha="center",
                    va="top", fontsize=10, color="gray", transform=ax.get_xaxis_transform())


def plot_fit_overview(
    t_min: np.ndarray,
    T_meas: np.ndarray,
    T_fit: np.ndarray,
    duty_pct: np.ndarray,
    u_pct: np.ndarray,
    spans: list[tuple[float, float, str]],
    path: str | Path,
    title: str,
) -> Path:
    """Measured (raw) and fitted box temperature (n,) in °C over t_min (n,) in minutes; below, on its own axes,
    the cooler power (n,) in %: duty sent by the PID and the lagged cooling the box feels (fitted).
    spans: (start_min, end_min, name) of the phases; the hold is shaded.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 7), sharex=True, height_ratios=[3, 1])
    ax.set_title(title)
    ax.plot(t_min, T_meas, linewidth=1.0, color="black", label="measured (raw sensor)")
    ax.plot(t_min, T_fit, linewidth=1.5, linestyle="--", color="tab:orange", label="fitted model (real duty and room as input)")
    ax.set_ylabel("box temperature (°C)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(loc="lower left", fontsize=10)
    _phase_spans(ax, spans)
    ax2.plot(t_min, duty_pct, linewidth=1.0, color="tab:blue", label="duty sent (PWM)")
    ax2.plot(t_min, u_pct, linewidth=1.5, linestyle="--", color="darkblue", label="cooling felt (fitted lag)")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("time since the rest phase started (min)")
    ax2.set_ylabel("cooler (%)")
    ax2.grid(alpha=GRID_ALPHA)
    ax2.legend(loc="center right", fontsize=9)
    _phase_spans(ax2, spans, label=False)
    ax.set_xlim(t_min[0], t_min[-1])
    return _save(fig, path)


def plot_fit_error(
    t_min: np.ndarray,
    errors: dict[str, np.ndarray],
    spans: list[tuple[float, float, str]],
    path: str | Path,
    title: str,
) -> Path:
    """Fit error model - measured (n,) in °C of each model in errors over t_min (n,) in minutes; the label
    shows the RMS outside the hold. spans: phases as in plot_fit_overview."""
    fig, ax = _new_plot(title, "time since the rest phase started (min)", "model - measured (°C)")
    fitted = np.ones(len(t_min), bool)
    for start, end, name in spans:
        if name == "hold":
            fitted &= (t_min < start) | (t_min > end)
    colors = ["tab:red", "tab:purple", "black"]
    for (label, err), color in zip(errors.items(), colors):
        ax.plot(t_min, err, linewidth=1.5 if color == "black" else 1.0, color=color,
                label=f"{label} (RMS {np.sqrt(np.mean(err[fitted] ** 2)):.3f} °C outside the hold)")
    ax.axhline(0.0, linewidth=0.8, color="gray")
    _phase_spans(ax, spans)
    ax.set_xlim(t_min[0], t_min[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
    return _save(fig, path)


def plot_hold_swing(
    t_meas_min: np.ndarray,
    T_meas: np.ndarray,
    t_sim_min: np.ndarray,
    T_sim: np.ndarray,
    target: float,
    path: str | Path,
    title: str,
) -> Path:
    """PID hold: measured temperature T_meas (n,) over t_meas_min (n,) and simulated box temperature T_sim (m,) over
    t_sim_min (m,), both in °C and minutes since the hold started, with the target."""
    fig, ax = _new_plot(title, "time since the hold started (min)", "box temperature (°C)")
    ax.plot(t_sim_min, T_sim, linewidth=1.5, linestyle="--", color="tab:orange",
            label="simulated (fitted box + PID + Kalman)")
    ax.plot(t_meas_min, T_meas, linewidth=1.5, color="black", label="measured (real box)")
    ax.axhline(target, linestyle="--", linewidth=1.2, color="green", label=f"target {target:g} °C")
    ax.set_xlim(0, max(t_meas_min[-1], t_sim_min[-1]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    return _save(fig, path)


def plot_temperature_and_duty(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target: tuple[np.ndarray, np.ndarray],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """Box temperature (top) and cooler duty (bottom, own axes) over time for several runs.

    curves: label -> (t_min (n,), T (n,) in °C, duty_pct (n,) in % of full PWM); target: (t_min (m,), target (m,) in °C).
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 7), sharex=True, height_ratios=[3, 1])
    ax.set_title(title)
    styles = ["-", "--", ":"]
    for (label, (t, T, duty)), style in zip(curves.items(), styles):
        color = None if colors is None else colors.get(label)
        ax.plot(t, T, linestyle=style, linewidth=1.5, color=color, label=label)
        ax2.plot(t, duty, linestyle=style, linewidth=1.0, color=color, label=label)
    ax.plot(*target, linestyle="--", linewidth=1.0, color="green", label="target")
    ax.set_ylabel("box temperature (°C)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("duty (%)")
    ax2.grid(alpha=GRID_ALPHA)
    ax.set_xlim(min(c[0][0] for c in curves.values()), max(c[0][-1] for c in curves.values()))
    return _save(fig, path)


def plot_score_map(
    Kp: np.ndarray,
    Ki: np.ndarray,
    score: np.ndarray,
    point: tuple[float, float],
    point_label: str,
    path: str | Path,
    title: str,
) -> Path:
    """Heat map of a score (nKi, nKp) in °C over the grid Kp (nKp,) and Ki (nKi,), both log axes, log colour
    scale (darker = lower = better), with one point (Kp, Ki) marked."""
    fig, ax = _new_plot(title, "Kp", "Ki")
    # Cell edges halfway (in log) between the grid values.
    edges = lambda v: np.exp(np.concatenate([[1.5 * np.log(v[0]) - 0.5 * np.log(v[1])],
                                             (np.log(v[:-1]) + np.log(v[1:])) / 2,
                                             [1.5 * np.log(v[-1]) - 0.5 * np.log(v[-2])]]))
    mesh = ax.pcolormesh(edges(Kp), edges(Ki), score, cmap="Blues_r",
                         norm=matplotlib.colors.LogNorm(vmin=score.min(), vmax=score.max()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    # The axes always include the marked point (10 % margin), also when it lies beyond the last cell.
    ax.set_xlim(min(edges(Kp)[0], point[0] / 1.1), max(edges(Kp)[-1], point[0] * 1.1))
    ax.set_ylim(min(edges(Ki)[0], point[1] / 1.1), max(edges(Ki)[-1], point[1] * 1.1))
    ax.set_xticks(Kp)
    ax.set_xticklabels([f"{v:g}" for v in Kp], fontsize=9)
    ax.set_yticks(Ki)
    ax.set_yticklabels([f"{v:g}" for v in Ki], fontsize=9)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="grid score (°C)")
    ax.plot(*point, marker="x", markersize=12, markeredgewidth=3, color="tab:red", linestyle="none", label=point_label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    return _save(fig, path)


def plot_operator_comparison(
    operators: list[str],
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    title: str,
    ylabel: str,
) -> Path:
    """Grouped bars: one group per operator, one bar per entry of groups, label -> (mean (n_ops,), error (n_ops,));
    the error bars are given by the caller (e.g. std over seeds)."""
    fig, ax = _new_plot(title, "operator", ylabel)
    x = np.arange(len(operators))
    width = 0.8 / len(groups)
    for i, (label, (mean, err)) in enumerate(groups.items()):
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, mean, width, yerr=err, capsize=4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(operators)
    ax.legend()
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

    hold = thermostat.simulate(60, control=True, T_start=TARGET_C, target=TARGET_C, seed=0, warmup_s=60)
    off = thermostat.simulate(60, control=False, T_start=TARGET_C, seed=0)
    step = thermostat.simulate(60, control=True, T_start=TARGET_C + 1.0, target=TARGET_C, seed=1)
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
            plot_correlation_vs_time(t, {f"{k} {i}": v for i in range(3) for k, v in correlations.items()},
                                     out / "correlation_many.png", styles={"no temperature control 1": "--"}),
            plot_correlation_and_temperature(t, correlations["no temperature control"], off["t_s"] / 60,
                                             off["T_true"] - off["T_true"][0], out / "correlation_and_temperature.png",
                                             "Correlation and temperature", "red", "ΔT (°C)"),
            plot_correlation_and_temperature(t, correlations["with temperature control"], hold["t_s"] / 60,
                                             hold["T_true"] - TARGET_C, out / "correlation_and_deviation.png",
                                             "Correlation and deviation", "blue", f"distance from {TARGET_C:g} °C (°C)",
                                             right_ylim=(-0.045, 0.045)),
            plot_box_deviation({"step": (step["t_s"], step["T_true"], step["target"]),
                                "hold": (hold["t_s"], hold["T_true"], hold["target"])}, out / "box_step_deviation.png",
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
            plot_real_box_test(hold["t_s"], hold["T_filtered"], np.where(hold["t_s"] < 30, TARGET_C, TARGET_C + 0.5),
                               np.full_like(hold["t_s"], 22.0), np.full_like(hold["t_s"], np.nan),
                               hold["duty"] / 10.23, out / "real_box_test.png"),
            plot_fit_overview(hold["t_s"] / 60, hold["T_raw"], hold["T_true"], hold["duty"] / 10.23, hold["duty"] / 10.23,
                              [(0.0, 2.0, "rest"), (2.0, 4.0, "hold")], out / "fit_overview.png", "Fit"),
            plot_fit_error(hold["t_s"] / 60, {"a": hold["T_raw"] - hold["T_true"]}, [(2.0, 4.0, "hold")],
                           out / "fit_error.png", "Fit error"),
            plot_hold_swing(hold["t_s"] / 60, hold["T_raw"], step["t_s"] / 60, step["T_true"], TARGET_C,
                            out / "hold_swing.png", "Hold swing"),
            plot_temperature_and_duty({"a": (hold["t_s"] / 60, hold["T_true"], hold["duty"] / 10.23),
                                       "b": (step["t_s"] / 60, step["T_true"], step["duty"] / 10.23)},
                                      (hold["t_s"] / 60, hold["target"]), out / "temperature_and_duty.png", "T and duty"),
            plot_operator_comparison(["a", "b"], {"x": (np.array([40.0, 50.0]), np.array([2.0, 3.0])),
                                                  "y": (np.array([45.0, 52.0]), np.array([1.0, 2.0]))},
                                     out / "operator_comparison.png", "Operators", "mean η"),
            plot_score_map(np.array([100.0, 200.0, 400.0]), np.array([0.1, 1.0]),
                           np.array([[0.1, 0.05, 0.02], [0.2, 0.1, 0.01]]), (300.0, 0.5), "point",
                           out / "score_map.png", "Score map"),
        ]
        for p in paths:
            assert p.exists() and p.stat().st_size > 0, f"{p.name} is missing or empty"
            print(f"ok: {p.name}")
    print("plots.py self-test passed")
