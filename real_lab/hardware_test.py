"""
hardware_test.py - a first check of the real SLM and camera together (real_lab/hardware.py).

Opens both, shows a blank SLM image and grabs a frame, then shows a test pattern (a vertical phase ramp: the phase rises
from left to right), waits HARDWARE settle_s and grabs again. Saves one PNG (both frames and their difference) and the
raw arrays (.npz) in results/real_box/hardware_test/<date_time>/ (with the fake devices in
results/hardware_test_fake/<date_time>/, which is not committed), prints min/max/mean of each image, and always closes
both devices (SLM blank, then off; camera acquisition off).

python -m real_lab.hardware_test          on the lab PC (the fake devices on a machine that is not Windows)
python -m real_lab.hardware_test --fake   with the fake devices on any machine
"""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis import plots
from config import HARDWARE as SETTINGS
from real_lab.hardware import FakeSLM, Camera, SLM


def phase_ramp(shape: tuple[int, int]) -> np.ndarray:
    """A vertical phase ramp (H x W uint8): 0 at the left edge rising to 255 at the right."""
    height, width = shape
    return np.tile((np.arange(width) * 256 // width).astype(np.uint8), (height, 1))


def describe(name: str, image: np.ndarray) -> str:
    """'name: min .., max .., mean ..' of one image."""
    return f"{name:12} min {image.min():7.1f}  max {image.max():7.1f}  mean {image.mean():7.2f}"


def main() -> Path:
    """Run the test; returns the output folder."""
    parser = argparse.ArgumentParser(description="SLM + camera hardware test")
    parser.add_argument("--fake", action="store_true", help="use the fake devices (any machine)")
    args = parser.parse_args()
    print("Laser is switched by hand. Wear goggles.")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with ExitStack() as stack:                   # closes both devices in any case (also after an error)
        slm = stack.enter_context(SLM(fake=True if args.fake else None))
        camera = stack.enter_context(Camera(fake=True if args.fake else None, slm=slm))
        kind = "fake" if isinstance(slm, FakeSLM) else "real"
        out = Path(SETTINGS["fake_out_dir" if kind == "fake" else "out_dir"]) / stamp
        print(f"devices: {kind} SLM {slm.shape[1]} x {slm.shape[0]}, {kind} camera {camera.shape[1]} x {camera.shape[0]}")

        slm.blank()
        time.sleep(SETTINGS["settle_s"])
        blank = camera.grab()
        pattern_phase = phase_ramp(slm.shape)
        slm.show(pattern_phase)
        time.sleep(SETTINGS["settle_s"])
        pattern = camera.grab()

    diff = pattern.astype(np.int16) - blank.astype(np.int16)
    for name, image in [("blank", blank), ("test pattern", pattern), ("difference", diff)]:
        print(describe(name, image))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "frames.npz", blank=blank, pattern=pattern, difference=diff, pattern_phase=pattern_phase)
    png = plots.plot_hardware_test(blank, pattern, out / "01_camera_blank_vs_phase_ramp.png",
                                   f"Hardware test ({kind} devices): camera with a blank SLM and with a phase ramp\n"
                                   f"{out.name}")
    print(f"saved {png}")
    print(f"saved {out / 'frames.npz'}")
    return out


if __name__ == "__main__":
    main()
