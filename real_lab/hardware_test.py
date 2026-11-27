"""
hardware_test.py - a first check of the real SLM and camera together (real_lab/hardware.py).

Opens both, prints the SLM temperature, takes the warm-up frames (warm_up), then a frame with the reference SLM image
(the bias grating alone) and one with a pattern (random 4 x 4 macro-pixel values, seed 0, built by slm_frame), each
through measure() (the SLM image, then the frame); prints mean, max and the share of saturated pixels (255). Each frame
is kept raw (the 96 x 96 polarization mosaic) and as intensity (48 x 48, the mean of the four polarizer sub-images).
Saves one PNG (raw and intensity: both frames and their difference) and the arrays (.npz) in
results/real_box/hardware_test/<date_time>/ (with the fake devices in results/hardware_test_fake/<date_time>/, which is
not committed), and always closes both devices (SLM blank, then off; camera acquisition off).

python -m real_lab.hardware_test          on the lab PC (the fake devices on a machine that is not Windows)
python -m real_lab.hardware_test --fake   with the fake devices on any machine
python -m real_lab.hardware_test --no-trigger   free-running camera, no SLM pulse (if triggered frames time out)
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis import plots
from config import HARDWARE as SETTINGS
from real_lab.hardware import Camera, FakeSLM, SLM, intensity, measure, reference_frame, slm_frame, warm_up


def describe(name: str, image: np.ndarray, saturation: bool = True) -> str:
    """'name: mean .., max .., (% of pixels at 255)' of one image."""
    text = f"{name:16} mean {image.mean():7.2f}  max {image.max():7.1f}"
    return text + (f"  at 255: {100 * np.mean(image >= 255):5.1f} %" if saturation else "")


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
        warm_up(slm, camera)                     # measure() with the reference image, warmup_frames times
        grid = s["slm_macro_grid"]
        values = np.random.default_rng(0).integers(0, 256, (grid, grid)).astype(np.uint8)
        reference = measure(slm, camera, reference_frame(s))
        pattern = measure(slm, camera, slm_frame(values, s))

    reference_i, pattern_i = intensity(reference), intensity(pattern)
    for name, image, saturation in [("reference raw", reference, True), ("pattern raw", pattern, True),
                                    ("difference raw", pattern - reference, False),
                                    ("reference int.", reference_i, False), ("pattern int.", pattern_i, False)]:
        print(describe(name, image, saturation))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "frames.npz", reference_raw=reference, pattern_raw=pattern, reference_intensity=reference_i,
             pattern_intensity=pattern_i, pattern_values=values)
    png = plots.plot_hardware_test({"raw": (reference, pattern), "intensity": (reference_i, pattern_i)},
                                   out / "01_camera_reference_vs_pattern.png",
                                   f"Hardware test ({kind} devices): bias grating alone vs. random {grid}x{grid} "
                                   f"values (seed 0)\n{out.name}")
    print(f"saved {png}")
    print(f"saved {out / 'frames.npz'}")
    return out


if __name__ == "__main__":
    main()
