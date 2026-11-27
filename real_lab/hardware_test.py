"""
hardware_test.py - a first check of the real SLM and camera together (real_lab/hardware.py).

Opens both, prints the SLM temperature, takes the warm-up frames (warm_up: an SLM image, then a frame), shows a blank SLM image and grabs a frame, then shows a test pattern (a vertical
phase ramp: the phase rises from left to right), waits HARDWARE settle_s and grabs again. Each frame is kept raw (the
96 x 96 polarization mosaic) and as intensity (48 x 48, the mean of the four polarizer sub-images). Saves one PNG (raw
and intensity: both frames and their difference) and the arrays (.npz) in results/real_box/hardware_test/<date_time>/ (with the fake devices in
results/hardware_test_fake/<date_time>/, which is not committed), prints min/max/mean of each image, and always closes
both devices (SLM blank, then off; camera acquisition off).

python -m real_lab.hardware_test          on the lab PC (the fake devices on a machine that is not Windows)
python -m real_lab.hardware_test --fake   with the fake devices on any machine
python -m real_lab.hardware_test --no-trigger   free-running camera, no SLM pulse (if triggered frames time out)
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
from real_lab.hardware import Camera, FakeSLM, SLM, intensity, warm_up


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
    parser.add_argument("--no-trigger", action="store_true",
                        help="trigger off: free-running camera, no SLM output pulse (if triggered frames time out)")
    args = parser.parse_args()
    s = {**SETTINGS, "trigger": SETTINGS["trigger"] and not args.no_trigger}
    print("Laser is switched by hand. Wear goggles.")
    print("trigger: " + (f"external (SLM output pulse -> camera input {s['trigger_input']}, delay {s['trigger_delay']})"
                         if s["trigger"] else "off (free-running camera, no SLM pulse)"))

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with ExitStack() as stack:                   # closes both devices in any case (also after an error)
        slm = stack.enter_context(SLM(fake=True if args.fake else None, s=s))
        camera = stack.enter_context(Camera(fake=True if args.fake else None, slm=slm, s=s))
        kind = "fake" if isinstance(slm, FakeSLM) else "real"
        out = Path(SETTINGS["fake_out_dir" if kind == "fake" else "out_dir"]) / stamp
        print(f"devices: {kind} SLM {slm.shape[1]} x {slm.shape[0]}, {kind} camera {camera.shape[1]} x {camera.shape[0]}")
        print(f"SLM temperature: {slm.temperature():.2f} C")
        warm_up(slm, camera)                     # an SLM image before each warm-up frame (needed with the trigger)

        slm.blank()
        time.sleep(SETTINGS["settle_s"])
        blank = camera.grab()
        pattern_phase = phase_ramp(slm.shape)
        slm.show(pattern_phase)
        time.sleep(SETTINGS["settle_s"])
        pattern = camera.grab()

    blank_i, pattern_i = intensity(blank), intensity(pattern)
    for name, image in [("blank raw", blank), ("ramp raw", pattern), ("diff raw", pattern - blank),
                        ("blank int.", blank_i), ("ramp int.", pattern_i), ("diff int.", pattern_i - blank_i)]:
        print(describe(name, image))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "frames.npz", blank_raw=blank, pattern_raw=pattern, blank_intensity=blank_i,
             pattern_intensity=pattern_i, pattern_phase=pattern_phase)
    png = plots.plot_hardware_test({"raw": (blank, pattern), "intensity": (blank_i, pattern_i)},
                                   out / "01_camera_blank_vs_phase_ramp.png",
                                   f"Hardware test ({kind} devices): camera with a blank SLM and with a phase ramp\n"
                                   f"{out.name}")
    print(f"saved {png}")
    print(f"saved {out / 'frames.npz'}")
    return out


if __name__ == "__main__":
    main()
