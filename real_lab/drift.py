"""
drift.py - stability of the real setup: the same reference SLM image (the bias grating) every interval_s for duration_s.

For each frame: its Pearson correlation to the first frame (all raw camera pixels), its mean, and the SLM, box and room
temperatures (room: the case sensor and the probe in the air). The box is only read (GET requests); nothing is set.
With the fake devices nothing is read from the network (box and room temperatures are NaN).
Saves frames, the table and its column names as results/real_box/stability/<date_time>.npz (fake devices:
results/hardware_test_fake/stability/, not committed). Settings in config.py DRIFT_RUN. Based on the lab PC's probe
script of 2026-11-27 (results/real_box/lab_scripts/probe4.py).

python -m real_lab.drift                                         on the lab PC (laser by hand, goggles)
python -m real_lab.drift --fake --duration_s 3 --interval_s 1    a short run with the fake devices (any machine)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DRIFT_RUN as SETTINGS
from config import HARDWARE, REAL_BOX
from real_lab.hardware import FakeSLM, Camera, SLM, measure, reference_frame, warm_up
from real_lab.logger import get_json, read_room

COLUMNS = ["t_s", "slm_C", "box_C", "box_duty", "room_C", "room_probe_C", "mean", "corr"]


def box_and_room(timeout_s: float) -> tuple[float, float, float, float]:
    """(box temperature, box duty, room case sensor, room probe); NaN for anything not readable or not configured."""
    box_C = box_duty = room_C = probe_C = np.nan
    if REAL_BOX["url"]:
        try:
            box = get_json(REAL_BOX["url"], timeout_s)
            box_C, box_duty = float(box["temperature"]), float(box.get("duty", np.nan))
        except Exception:
            pass
    if REAL_BOX["room_url"]:
        try:
            room = read_room(REAL_BOX["room_url"], REAL_BOX["room_channel"], timeout_s)
            room_C = float(room["room_temp"]) if room["room_temp"] not in (None, "") else np.nan
            probe_C = float(room["room_probe_temp"]) if room["room_probe_temp"] not in (None, "") else np.nan
        except Exception:
            pass
    return box_C, box_duty, room_C, probe_C


def run(duration_s: float, interval_s: float, fake: bool | None = None) -> Path:
    """One drift run; returns the saved .npz."""
    frames, rows = [], []
    with SLM(fake, HARDWARE) as slm, Camera(fake, slm, HARDWARE) as camera:
        warm_up(slm, camera)
        reference = reference_frame(HARDWARE)
        kind = "fake" if isinstance(slm, FakeSLM) else "real"
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            t_frame = time.monotonic()
            frame = measure(slm, camera, reference)
            frames.append(frame)
            box_C, box_duty, room_C, probe_C = (np.nan,) * 4 if kind == "fake" else box_and_room(SETTINGS["read_timeout_s"])
            corr = float(np.corrcoef(frames[0].ravel(), frame.ravel())[0, 1])
            rows.append([t_frame - t0, slm.temperature(), box_C, box_duty, room_C, probe_C, float(frame.mean()), corr])
            print(f"{rows[-1][0]:6.0f} s  corr {corr:.4f}  SLM {rows[-1][1]:.2f}  box {box_C:.2f}  room {room_C:.2f} "
                  f"(probe {probe_C:.2f})  mean {frame.mean():.1f}")
            time.sleep(max(0.0, interval_s - (time.monotonic() - t_frame)))
    out = Path(SETTINGS["fake_out_dir" if kind == "fake" else "out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.npz"
    np.savez(path, frames=np.array(frames), table=np.array(rows), columns=np.array(COLUMNS))
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference-frame drift of the real setup")
    parser.add_argument("--fake", action="store_true", help="the fake devices (any machine)")
    parser.add_argument("--duration_s", type=float, default=SETTINGS["duration_s"])
    parser.add_argument("--interval_s", type=float, default=SETTINGS["interval_s"])
    args = parser.parse_args()
    print("Laser is switched by hand. Wear goggles.")
    path = run(args.duration_s, args.interval_s, True if args.fake else None)
    saved = np.load(path)
    assert list(saved["columns"]) == COLUMNS and len(saved["frames"]) == len(saved["table"]) >= 1
    assert np.isclose(saved["table"][0, COLUMNS.index("corr")], 1.0)
