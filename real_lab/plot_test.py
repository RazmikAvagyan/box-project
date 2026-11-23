"""
plot_test.py - plot one real box CSV from real_lab.logger; the PNG is saved next to the CSV.

Usage: python -m real_lab.plot_test results/real_box/test_<date>_<time>.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from analysis import plots


def load(path: str | Path) -> dict[str, np.ndarray]:
    """CSV columns as float arrays (n,); empty cells become NaN, text columns are left out."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keys = ["time_s", "filtered", "target", "power_pct", "room_temp", "room_probe_temp"]
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in keys}


def main(path: str | Path) -> Path:
    """Save the plot of the CSV at path as <same name>.png and return the PNG path."""
    path = Path(path)
    d = load(path)
    return plots.plot_real_box_test(d["time_s"], d["filtered"], d["target"], d["room_temp"],
                                    d["room_probe_temp"], d["power_pct"], path.with_suffix(".png"),
                                    title=f"Real box test {path.stem.removeprefix('test_')}")


if __name__ == "__main__":
    print(f"saved {main(sys.argv[1])}")
