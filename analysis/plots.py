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

from config import PLOTS, TARGET_C, THERMOSTAT

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


def plot_ggs_convergence(history: list[float], path: str | Path, title_note: str = "") -> Path:
    """Mean fit error per GGS iteration (len = number of iterations), linear y axis."""
    fig, ax = _new_plot("Learning the sample map (GGS): error per iteration" + title_note, "iteration",
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
    title: str = "",
) -> Path:
    """Camera images (M,) for a random pattern and a focus pattern, side by side.

    Each image has its own colour bar, so the speckle structure is visible too;
    the titles give mean and max to compare the brightness. The target pixel
    (index into M) is circled on the focus image.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    if title:
        fig.suptitle(title)
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


def plot_enhancement(eta: np.ndarray, eta_theory: float, N: int, path: str | Path, title_note: str = "") -> Path:
    """Enhancement eta (M,) per camera point, with the theory level eta_theory for N SLM squares.

    Bars start at 0 (fixed range), so the bar heights compare honestly.
    """
    fig, ax = _new_plot("Focus brightness at each camera point" + title_note, "camera point index",
                        "enhancement η (spot / background)")
    ax.bar(np.arange(len(eta)), eta, label=f"measured η ({_stats(eta, '.1f')})")
    ax.axhline(eta_theory, linestyle="--", color="black", label=f"theory for N = {N}: {eta_theory:.1f}")
    # Head room so the legend sits above the bars instead of on them.
    ax.set_ylim(0, 1.45 * max(eta.max(), eta_theory))
    ax.legend(loc="upper center", ncol=2, fontsize=10)
    return _save(fig, path)


def plot_correlation_panels(
    t: np.ndarray,
    panels: dict[str, dict[str, np.ndarray]],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
    styles: dict[str, str] | None = None,
) -> Path:
    """Correlation curves against time t (T,) in minutes, one panel per entry of panels (panel title -> label ->
    r (T,)), side by side on the same 0 to 1 axis, with one shared legend below (colour and line style per label)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), FIGSIZE[1]), sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle(title, y=1.02)
    for ax, (panel, curves) in zip(axes, panels.items()):
        for label, r in curves.items():
            ax.plot(t, r, marker="o", markersize=3, color=(colors or {}).get(label),
                    linestyle=(styles or {}).get(label, "-"), label=label)
        ax.set_title(f"{panel} (scale 0 to 1)", fontsize=12)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (min)")
        ax.grid(alpha=GRID_ALPHA)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("prediction correlation r (1 = perfect)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10)
    return _save(fig, path)


def plot_box_deviation(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """Deviation (T_true - target) in °C over time (s) for several runs on one panel with a scale that shows every
    line in full, with the +-0.03 °C limit; curves maps legend label -> (t_s, T_true, target) (n,)."""
    fig, ax = _new_plot(title, "time (s)", "distance from target (°C)")
    ax.axhline(0.0, linewidth=0.8, color="black")
    devs = []
    for label, (t_s, T_true, target) in curves.items():
        devs.append(T_true - target)
        ax.plot(t_s, devs[-1], linewidth=1.8, color=(colors or {}).get(label), label=label)
    ax.axhline(LIMIT_C, linestyle="--", linewidth=1.0, color="gray", label=f"limit ±{LIMIT_C:g} °C")
    ax.axhline(-LIMIT_C, linestyle="--", linewidth=1.0, color="gray")
    lo, hi = min(d.min() for d in devs), max(d.max() for d in devs)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, max(c[0][-1] for c in curves.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    return _save(fig, path)


def plot_kalman_error(
    t_s: np.ndarray,
    T_true: np.ndarray,
    T_raw: np.ndarray,
    T_filtered: np.ndarray,
    path: str | Path,
    title_note: str = "",
) -> Path:
    """Error (measured - true) in °C of the raw readings and of the Kalman estimate (all (n_steps,)) over t_s (s).

    Linear y axis of +-4 raw stds (at least +-0.05 °C); raw readings beyond it (sensor spikes) are drawn as
    triangles on the edge and counted in the legend. Raw: grey dots (every 5th reading); Kalman: blue line.
    """
    raw_err, kal_err = T_raw - T_true, T_filtered - T_true
    std_raw, std_kal = np.std(raw_err), np.std(kal_err)
    fig, ax = _new_plot("How far off is each measurement? Raw sensor vs Kalman filter" + title_note + "\n"
                        f"raw error std {std_raw:.4f} °C, Kalman error std {std_kal:.4f} °C "
                        f"({std_raw / std_kal:.1f} times smaller)",
                        "time (s)", "error = measured - true (°C)")
    lim = max(0.05, 4 * std_raw)
    inside = np.abs(raw_err) <= lim
    ax.scatter(t_s[::5][inside[::5]], raw_err[::5][inside[::5]], s=6, alpha=0.4, color="gray", label="raw sensor - true")
    spikes = np.flatnonzero(~inside)
    if spikes.size:
        ax.plot(t_s[spikes], np.sign(raw_err[spikes]) * 0.97 * lim, "^", color="tab:red", markersize=7, linestyle="none",
                label=f"{spikes.size} spikes beyond ±{lim:.2g} °C (on the edge)")
    ax.plot(t_s, kal_err, linewidth=1.5, color="tab:blue", label="Kalman filtered - true")
    ax.axhline(0.0, linewidth=1.0, color="black", label="perfect (no error)")
    ax.set_ylim(-lim, lim)
    ax.set_xlim(0, t_s[-1])
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


def message_label_places(messages: list[tuple[float, str]], widths: dict[str, float]) -> list[tuple[str, float, int, float, float]]:
    """Where the scenario plot writes each message label: (topic, triangle minute, row, start minute, end minute).
    widths: the measured width of each label (min on the time axis), padded by scenario_label_pad_min here.
    Row 0 (the triangles' row): right of the triangle, then left, scenario_label_gap_min away; lower rows (each
    scenario_label_step_C lower): starting exactly at the triangle. The first free place is used, touching no triangle
    (half width 0.5 min) and no earlier label."""
    gap = PLOTS["scenario_label_gap_min"]
    ordered = sorted(messages)
    rows = {0: [(m / 60.0 - 0.5, m / 60.0 + 0.5) for m, _ in ordered]}       # occupied minutes per row
    places = []
    for t_msg, topic in ordered:
        minutes, width = t_msg / 60.0, widths[topic] + PLOTS["scenario_label_pad_min"]
        for row in range(len(ordered) + 1):
            taken = rows.setdefault(row, [])
            options = ([(minutes + gap, minutes + gap + width), (minutes - gap - width, minutes - gap)] if row == 0
                       else [(minutes, minutes + width)])
            free = next(((a, b) for a, b in options if a >= 0 and all(b < lo or a > hi for lo, hi in taken)), None)
            if free:
                taken.append(free)
                places.append((topic, minutes, row, *free))
                break
    return places


def plot_scenario(
    t_s: np.ndarray,
    T_true: np.ndarray | None,
    T_kalman: np.ndarray,
    target: np.ndarray,
    T_room: np.ndarray,
    events: list[tuple[float, str]],
    path: str | Path,
    y_range: tuple[float, float],
    title: str,
    broken: tuple[float, float] | None = None,
    messages: list[tuple[float, str]] | None = None,
    places_out: list | None = None,
) -> Path:
    """Absolute temperatures (n,) in °C over t_s (n,) in s on one panel with the fixed range y_range: box (true and
    Kalman, dashed; no true line when T_true is None or all nan, as on the real lab), target and room; the cooler
    failure broken (start_s, end_s) shaded; numbered event lines.
    messages: optional (t_s, topic), each a small black triangle at PLOTS scenario_message_y_C with its label placed by
    message_label_places, with each label's width measured on the drawn figure (a lower label starts at its triangle
    and is joined to it by a thin grey line). places_out: optional list that receives the label places. Time in min."""
    fig, ax = _new_plot(title, "time (min)", "temperature (°C)")
    t = t_s / 60.0
    if T_true is not None and not np.all(np.isnan(np.asarray(T_true, dtype=float))):
        ax.plot(t, T_true, linewidth=1.5, color="black", label="box, true temperature")
    ax.plot(t, T_kalman, linewidth=1.2, linestyle="--", color="tab:blue", label="box, Kalman estimate (dashed)")
    ax.plot(t, target, linewidth=1.2, linestyle="-.", color="green", label="target")
    ax.plot(t, T_room, linewidth=2.0, linestyle=":", color="tab:orange", label="room")
    if broken is not None:
        ax.axvspan(broken[0] / 60.0, broken[1] / 60.0, color="red", alpha=0.12, label="cooler broken")
    y, step = PLOTS["scenario_message_y_C"], PLOTS["scenario_label_step_C"]
    ax.set_ylim(*y_range)
    ax.set_xlim(0, t[-1])
    # Each label's real width: draw it once, measure it with the renderer, convert to minutes on the time axis.
    renderer = fig.canvas.get_renderer()
    widths = {}
    for topic in {topic for _, topic in messages or []}:
        probe = ax.text(0, y, topic, fontsize=8, va="center", ha="left")
        box = probe.get_window_extent(renderer)
        (x0, _), (x1, _) = ax.transData.inverted().transform([(box.x0, box.y0), (box.x1, box.y0)])
        widths[topic] = float(x1 - x0)
        probe.remove()
    for k, (t_msg, _) in enumerate(sorted(messages or [])):
        ax.plot(t_msg / 60.0, y, "v", markersize=7, color="black", linestyle="none",
                label="Temperature agent message" if k == 0 else None)
    places = message_label_places(messages or [], widths)
    if places_out is not None:
        places_out.extend(places)
    for topic, minutes, row, start, end in places:
        label_y = y - row * step
        if row > 0:
            # A lower label starts at its own triangle; a thin line joins them.
            ax.plot([minutes, minutes], [y - 0.07, label_y + 0.08], linewidth=0.8, color="gray")
        ax.text(start, label_y, topic, fontsize=8, va="center", ha="left")
    ax.set_ylim(*y_range)
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    _event_lines(ax, events, key_y=-0.32)
    return _save(fig, path)


def plot_cooler_power(
    t_s: np.ndarray,
    power_pct: np.ndarray,
    real_pct: np.ndarray,
    events: list[tuple[float, str]],
    broken: tuple[float, float],
    path: str | Path,
    title_note: str = "",
) -> Path:
    """Cooler power over time t_s (n,) in seconds (axis in minutes), in % of full power (n,):
    power_pct = what the PID asks for (dark blue dashed line, on top), real_pct = what the cooler really gives
    (thick orange line). Both are smoothed with a centered 60 s moving average first, so the plot is easy to read.

    broken: (start_s, end_s) when the cooler gives nothing, shaded light red.
    events: numbered grey vertical lines with a key line below the legend, as in plot_scenario.
    """
    fig, ax = _new_plot("Cooler power: asked by the PID vs. really given (0 % while the cooler is broken)" + title_note,
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
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    _event_lines(ax, events, key_y=-0.32)
    return _save(fig, path)

def _by_topic(messages: list[tuple[float, str]]) -> dict[str, list[float]]:
    """Message times grouped by topic, in order of first appearance."""
    groups = {}
    for t_s, topic in messages:
        groups.setdefault(topic, []).append(t_s)
    return groups


def moving_average(t_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Centred moving average (n,) of values (n,) over window_s seconds of t_s (n,)."""
    return np.array([values[np.abs(t_s - t) <= window_s / 2].mean() for t in t_s])


def plot_policy_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    events: list[tuple[float, str]],
    path: str | Path,
    colors: dict[str, str] | None = None,
    title: str = "Focus brightness during the scenario: never re-measure vs. fixed rules",
    window_s: float = PLOTS["moving_average_s"],
) -> Path:
    """Focus enhancement eta over time for several operator policies.

    curves maps label -> (t_s (n,), eta (n,), measured (n,) bool). The main line is a window_s moving average,
    the per-step values faint behind it; circles mark new maps. A curve equal to an earlier one is drawn dashed
    and marked "(identical)". events: numbered grey lines, the key line below the legend. Time in minutes.
    """
    fig, ax = _new_plot(title, f"time (min); main lines: {window_s / 60:g}-min moving average, faint: every step",
                        "enhancement η (spot / background)")
    drawn = []
    for label, (t_s, eta, measured) in curves.items():
        color = (colors or {}).get(label)
        same = next((other for other, e in drawn if np.array_equal(e, eta)), None)
        style, note = ("--", f" (identical to {same})") if same else ("-", "")
        avg = moving_average(t_s, eta, window_s)
        ax.plot(t_s / 60.0, eta, linewidth=0.8, color=color, alpha=0.25)
        ax.plot(t_s / 60.0, avg, linewidth=2.0, linestyle=style, color=color,
                label=f"{label}: mean η {eta.mean():.1f}{note}")
        if measured.any():
            ax.plot(t_s[measured] / 60.0, avg[measured], "o", markersize=6, markerfacecolor="none",
                    markeredgecolor=color, label=f"{label}: new map measured")
        drawn.append((label, eta))
    t_all = np.concatenate([c[0] for c in curves.values()]) / 60.0
    ax.set_xlim(0, t_all.max())
    ax.set_ylim(0, 1.15 * max(c[1].max() for c in curves.values()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
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
    """Logged real box, two stacked panels over time (minutes, from 0): box temperature, target, room (dashed) and
    room probe (dotted), all (n,) in °C; below, the cooler power (n,) in % of full PWM (0 to 100).
    Room values may be NaN (failed reads). Grey vertical lines at each target change."""
    fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 1.4 * FIGSIZE[1]), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    fig.suptitle(title)
    t = t_s / 60.0
    ax.plot(t, T_box, linewidth=1.5, color="black", label="box temperature")
    ax.plot(t, target, linewidth=1.2, linestyle="-.", color="green", label="target")
    ax.plot(t, T_room, linestyle="--", linewidth=1.5, color="tab:orange", label="room (next to box)")
    ax.plot(t, T_probe, linestyle=":", linewidth=2.0, color="tab:red", label="room probe (far side)")
    for a in (ax, ax_p):
        for i in np.flatnonzero(np.diff(target) != 0) + 1:
            a.axvline(t[i], linewidth=1.0, color="gray")
        a.grid(alpha=GRID_ALPHA)
    for i in np.flatnonzero(np.diff(target) != 0) + 1:
        right = t[i] > 0.8 * t[-1]          # near the right end the label goes left of the line
        ax.text(t[i], 0.98, f" {target[i - 1]:g} → {target[i]:g} °C ", va="top", ha="right" if right else "left",
                fontsize=9, color="gray", transform=ax.get_xaxis_transform())
    # The scale follows the box and the room; a target far outside them (e.g. 30 °C = cooler off) leaves the panel.
    lo, hi = _y_range(ax, *[a[np.isfinite(a)] for a in (T_box, T_room, T_probe)], min_span=1.0, margin=0.1)
    off = sorted({float(v) for v in target if v < lo or v > hi})
    ax.set_ylabel("temperature (°C)")
    ax.set_title(f"temperatures (scale {lo:.1f} to {hi:.1f} °C"
                 + (f"; target {', '.join(f'{v:g}' for v in off)} °C off the panel" if off else "") + ")", fontsize=11)
    ax_p.plot(t, power_pct, linewidth=1.2, color="tab:blue", label="cooler power")
    ax_p.set_ylim(0, 105)
    ax_p.set_ylabel("cooler (%)")
    ax_p.set_title("cooler power (scale 0 to 100 % of full PWM)", fontsize=11)
    ax_p.set_xlabel("time (min)")
    ax_p.set_xlim(0, t[-1])
    for a in (ax, ax_p):
        a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)      # outside, never on the curves
    _phase_spans(ax, spans)
    ax2.plot(t_min, duty_pct, linewidth=1.0, color="tab:blue", label="duty sent (PWM)")
    ax2.plot(t_min, u_pct, linewidth=1.5, linestyle="--", color="darkblue", label="cooling felt (fitted lag)")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("time since the rest phase started (min)")
    ax2.set_ylabel("cooler (%)")
    lo, hi = ax.get_ylim()
    ax.set_title(f"{ax.get_title()}\nbox temperature (scale {lo:.1f} to {hi:.1f} °C)", fontsize=11)
    ax2.set_title("cooler (scale 0 to 100 %)", fontsize=10)
    ax2.grid(alpha=GRID_ALPHA)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    _phase_spans(ax2, spans, label=False)
    ax.set_xlim(0, t_min[-1])
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
    ax.set_xlim(0, t_min[-1])
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)      # outside, never on the curves
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("time (min)")
    ax2.set_ylabel("duty (%)")
    lo, hi = ax.get_ylim()
    ax.set_title(f"{ax.get_title()}\nbox temperature (scale {lo:.1f} to {hi:.1f} °C)", fontsize=11)
    ax2.set_title("duty (scale 0 to 100 %)", fontsize=10)
    ax2.grid(alpha=GRID_ALPHA)
    ax.set_xlim(0, max(c[0][-1] for c in curves.values()))
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


def plot_operator_panels(
    operators: list[str],
    rows: list[tuple[str, str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]],
    path: str | Path,
    title: str,
) -> Path:
    """Grouped bars: one row per entry of rows (y label, number format, panel title -> groups) and one panel per
    panel title; groups map label -> (mean (n_ops,), error (n_ops,)), one bar per group for each operator (no number
    labels: the numbers are in the results tables). Panels share the y range within a row; titles on the first row."""
    n_cols = len(rows[0][2])
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(max(FIGSIZE[0], 6.5 * n_cols), 4.6 * len(rows) + 0.8),
                             squeeze=False, gridspec_kw={"hspace": 0.6})
    fig.suptitle(title, y=1.01)
    x = np.arange(len(operators))
    for r, (ylabel, fmt, panels) in enumerate(rows):
        top = max(float(np.max(m + e)) for groups in panels.values() for m, e in groups.values())
        for c, (panel, groups) in enumerate(panels.items()):
            ax = axes[r, c]
            width = 0.8 / len(groups)
            for i, (label, (mean, err)) in enumerate(groups.items()):
                ax.bar(x + (i - (len(groups) - 1) / 2) * width, mean, width, yerr=err, capsize=3, label=label)
            ax.set_ylim(0, 1.15 * top)
            ax.set_xticks(x)
            ax.set_xticklabels(operators, rotation=20, ha="right")
            ax.grid(alpha=GRID_ALPHA, axis="y")
            ax.set_title(f"{panel}: {ylabel}" if r == 0 else ylabel, fontsize=11)
            if c == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=10)
    return _save(fig, path)


def plot_kalman_summary(
    columns: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    windows: dict[str, tuple[float, float]],
    path: str | Path,
    title: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """One column per PID (column title -> run label -> (t_s, T_true, target, duty %), all (n,)), each with its own
    y ranges. Top: box temperature minus target over the whole run, legend with std and max |deviation|. Bottom: the
    cooler duty in the time window windows[column] (s), legend with the std of the duty change per control step
    (whole run). Coinciding temperatures: the first run thick and light, later runs dashed on top; duty: the first run
    on top, later runs thin underneath."""
    n = len(columns)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 10), squeeze=False, gridspec_kw={"hspace": 0.85})
    fig.suptitle(title, y=0.98)
    for c, (column, runs) in enumerate(columns.items()):
        ax_T, ax_d = axes[0, c], axes[1, c]
        devs, window_duty = [], []
        w0, w1 = windows[column]
        for i, (label, (t_s, T_true, target, duty)) in enumerate(runs.items()):
            color = (colors or {}).get(label)
            dev = T_true - target
            devs.append(dev)
            # Temperature: the lines coincide, so the first is thick and light, later ones dashed on top.
            ax_T.plot(t_s, dev, linewidth=3.5 if i == 0 else 1.3, alpha=0.45 if i == 0 else 1.0,
                      linestyle="-" if i == 0 else "--", color=color,
                      label=f"{label}: std {dev.std():.4f} °C, max |dev| {np.abs(dev).max():.4f} °C")
            # Duty: the first (calm) line on top, later (noisy) ones thin underneath.
            w = (t_s >= w0) & (t_s <= w1)
            window_duty.append(duty[w])
            ax_d.plot(t_s[w], duty[w], linewidth=2.5 if i == 0 else 0.8, zorder=3 if i == 0 else 2, color=color,
                      label=f"{label}: duty change per step, std {np.diff(duty).std():.3f} %")
        ax_T.axhline(LIMIT_C, linestyle="--", linewidth=1.0, color="gray", label=f"limit ±{LIMIT_C:g} °C")
        ax_T.axhline(-LIMIT_C, linestyle="--", linewidth=1.0, color="gray")
        lim = max(1.5 * LIMIT_C, 1.15 * max(np.abs(d).max() for d in devs))
        ax_T.set_ylim(-lim, lim)
        ax_T.set_xlim(0, max(r[0][-1] for r in runs.values()))
        ax_T.set_title(f"{column}\nbox temperature, whole run (scale ±{lim:.2g} °C)", fontsize=11)
        ax_T.set_xlabel("time (s)")
        lo = min(d.min() for d in window_duty)
        hi = max(d.max() for d in window_duty)
        pad = 0.1 * (hi - lo) + 0.5
        ax_d.set_ylim(max(0.0, lo - pad), hi + pad)
        ax_d.set_xlim(w0, w1)
        ax_d.set_title(f"cooler duty, {w1 - w0:.0f} s with the cooler working "
                       f"(scale {max(0.0, lo - pad):.0f} to {hi + pad:.0f} %)", fontsize=11)
        ax_d.set_xlabel("time (s)")
        for ax in (ax_T, ax_d):
            ax.grid(alpha=GRID_ALPHA)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
    axes[0, 0].set_ylabel("from target (°C)")
    axes[1, 0].set_ylabel("cooler duty (%)")
    return _save(fig, path)


def plot_hardware_test(rows: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, title: str) -> Path:
    """Hardware test frames, one row per kind (e.g. "raw" (96, 96) and "intensity" (48, 48)): blank SLM and test
    pattern on one 0..255 scale, and their difference (pattern - blank) on a symmetric scale; one colorbar per
    quantity and row."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(FIGSIZE[0], 0.8 * FIGSIZE[1] * len(rows)), layout="constrained",
                             squeeze=False)
    fig.suptitle(title)
    for row, (kind, (blank, pattern)) in zip(axes, rows.items()):
        diff = pattern.astype(float) - blank.astype(float)
        for ax, image, name in [(row[0], blank, "blank SLM"), (row[1], pattern, "test pattern")]:
            im = _show_image(ax, image, 0, 255)
            ax.set_title(f"{kind}, {name} {image.shape[1]}×{image.shape[0]}\nmean {image.mean():.1f}, max {image.max():.0f}",
                         fontsize=10)
        fig.colorbar(im, ax=row[:2], shrink=0.8, label=f"{kind} (counts)")
        span = max(float(np.abs(diff).max()), 1.0)
        im = row[2].imshow(diff, cmap="RdBu_r", vmin=-span, vmax=span)
        row[2].set_xticks([])
        row[2].set_yticks([])
        row[2].set_title(f"{kind}, pattern - blank\nmean |diff| {np.abs(diff).mean():.1f}", fontsize=10)
        fig.colorbar(im, ax=row[2], shrink=0.8, label="difference (counts)")
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
            plot_box_deviation({"step": (step["t_s"], step["T_true"], step["target"]),
                                "hold": (hold["t_s"], hold["T_true"], hold["target"])}, out / "box_step_deviation.png",
                               "Step deviation"),
            plot_kalman_error(hold["t_s"], hold["T_true"], hold["T_raw"], hold["T_filtered"], out / "kalman_error.png"),
            plot_scenario(hold["t_s"], hold["T_true"], hold["T_filtered"], hold["target"], np.full_like(hold["t_s"], 22.0),
                          [(30.0, "event")], out / "scenario.png", (19.5, 24.5), "Scenario (seed 0)", broken=(20.0, 40.0),
                          messages=[(25.0, "box unstable"), (45.0, "box stable again")]),
            plot_cooler_power(hold["t_s"], hold["duty"] / 10.23, hold["duty"] / 10.23, [(30.0, "event")], (20.0, 40.0),
                              out / "cooler_power.png"),
            plot_policy_comparison({"a": (t * 60, correlations["no temperature control"] * 50, t > 2),
                                    "b": (t * 60, correlations["with temperature control"] * 50, t < 0)},
                                   [(60.0, "event")], out / "policy_comparison.png"),
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
            plot_policy_comparison({"a": (t * 60, correlations["no temperature control"] * 50, t > 2),
                                    "b": (t * 60, correlations["no temperature control"] * 50, t > 2)},
                                   [(60.0, "event")], out / "policy_identical.png"),
            plot_score_map(np.array([100.0, 200.0, 400.0]), np.array([0.1, 1.0]),
                           np.array([[0.1, 0.05, 0.02], [0.2, 0.1, 0.01]]), (300.0, 0.5), "point",
                           out / "score_map.png", "Score map"),
            plot_correlation_panels(t, {"no drift": correlations, "drift": {k: v / 2 for k, v in correlations.items()}},
                                    out / "correlation_panels.png", "Correlation panels", colors=colors),
            plot_operator_panels(["a", "b"], [("mean η", ".1f", {"PID 1": {"x": (np.array([40.0, 50.0]), np.array([2.0, 3.0]))},
                                                                 "PID 2": {"x": (np.array([45.0, 52.0]), np.array([1.0, 2.0]))}}),
                                              ("lab time (s)", ".0f", {"PID 1": {"x": (np.array([400.0, 900.0]), np.array([20.0, 30.0]))},
                                                                       "PID 2": {"x": (np.array([450.0, 520.0]), np.array([10.0, 20.0]))}})],
                                 out / "operator_panels.png", "Operators (error bars: standard error)"),
            plot_kalman_summary({"PID a": {"Kalman": (hold["t_s"], hold["T_true"], hold["target"], hold["duty"] / 10.23),
                                           "raw": (off["t_s"], off["T_true"], off["target"], off["duty"] / 10.23)},
                                 "PID b": {"Kalman": (hold["t_s"], hold["T_true"], hold["target"], hold["duty"] / 10.23)}},
                                {"PID a": (10.0, 40.0), "PID b": (5.0, 30.0)}, out / "kalman_summary.png",
                                "Kalman filter: same temperature, much calmer cooler"),
            plot_hardware_test({"raw": (np.full((96, 96), 20.0), np.random.default_rng(0).uniform(0, 255, (96, 96))),
                                "intensity": (np.full((48, 48), 20.0), np.random.default_rng(1).uniform(0, 255, (48, 48)))},
                               out / "hardware_test.png", "Hardware test"),
        ]
        for p in paths:
            assert p.exists() and p.stat().st_size > 0, f"{p.name} is missing or empty"
            print(f"ok: {p.name}")
    print("plots.py self-test passed")
