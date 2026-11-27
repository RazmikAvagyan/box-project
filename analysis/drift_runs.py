"""
drift_runs.py - plots the real setup's stability runs of 2026-11-27 (real_lab/drift.py and the lab PC's probe script):
the correlation of each reference frame to the run's first frame, and the SLM, box and room temperatures.

The runs (results/real_box/stability/): 11:57 the first run (frames and SLM temperature only), 12:19 with the box
closed, 12:32 with the box open. The correlation is computed here from the saved frames (Pearson, all raw pixels), so
all runs use the same definition. Writes results/real_box/stability/01_reference_correlation_and_temperatures.png.

python -m analysis.drift_runs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis import plots
from config import DRIFT_RUN

RUNS = {"11:57 first run": "2026-11-27_11-57-40.npz", "12:19 box closed": "2026-11-27_12-19-19_box.npz",
        "12:32 box open": "2026-11-27_12-32-38_box.npz"}


def load_run(path: Path) -> dict[str, np.ndarray]:
    """t_s, corr (to the first frame), slm_C and, where saved, box_C and room_C (n,) of one run's .npz."""
    saved = np.load(path)
    frames = saved["frames"].reshape(len(saved["frames"]), -1).astype(float)
    corr = np.array([np.corrcoef(frames[0], f)[0, 1] for f in frames])
    if "table" in saved.files:
        table = dict(zip([str(c) for c in saved["columns"]], saved["table"].T))
        return {"t_s": table["t_s"], "corr": corr, "slm_C": table["slm_C"], "box_C": table["box_C"],
                "room_C": table["room_C"]}
    return {"t_s": saved["t_s"], "corr": corr, "slm_C": saved["slm_temp"]}


def main() -> Path:
    """Load the runs, print a summary line per run, and save the plot."""
    out = Path(DRIFT_RUN["out_dir"])
    runs = {label: load_run(out / name) for label, name in RUNS.items()}
    for label, run in runs.items():
        dT = lambda key: f"{np.nanmax(run[key]) - np.nanmin(run[key]):.2f}" if key in run else "-"
        print(f"{label:18} {len(run['t_s'])} frames over {run['t_s'][-1] / 60:.1f} min: corr to the first frame "
              f"{run['corr'][-1]:.4f} at the end (min {run['corr'].min():.4f}); temperature range SLM {dT('slm_C')} C, "
              f"box {dT('box_C')} C, room {dT('room_C')} C")
    path = plots.plot_drift_runs(runs, out / "01_reference_correlation_and_temperatures.png",
                                 "Real setup, reference SLM image every 10 s (2026-11-27): correlation to the first "
                                 "frame\nand temperatures; first run, box closed, box open")
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    main()
