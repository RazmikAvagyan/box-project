"""
hardware.py - the real SLM and camera as a small hardware layer (no optics model, no agents).

SLM: Meadowlark PCIe SLM (1024 x 1024) through the Blink Plus SDK's C wrapper (ctypes). Camera: Alkeria Celera C5S-MP
over USB3 through the alkusb3 Python module. Both are used only on the lab PC (Windows); the DLL and alkusb3 are
imported inside open(), so this module imports on any machine.
Based on the lab's own utils.py and on Meadowlark's Blink_SDK_example.py (the call sequence of the SDK). All settings
are in config.py HARDWARE.

SLM(fake=...) and Camera(fake=...) give FakeSLM / FakeCamera with fake=True, and automatically on a machine that is not
Windows: the fake camera sees a speckle pattern from a fixed random transmission of the fake SLM's phase, so a test
pattern changes the image as on the real setup. Every SDK return value is checked; a failure raises HardwareError
(a LabError, so the team graph handles it as a hardware error) with the SDK's own message. Both classes are context
managers and always close in __exit__ (SLM: blank image, then Delete_SDK; camera: acquire off).
"""

from __future__ import annotations

import sys

import numpy as np

from agents.core.lab_api import LabError
from config import HARDWARE as SETTINGS


class HardwareError(LabError):
    """The SLM or camera SDK reported a failure."""


def use_fake(fake: bool | None) -> bool:
    """fake as given, or (None) True on any machine that is not Windows."""
    return sys.platform != "win32" if fake is None else fake


class RealSLM:
    """The Meadowlark SLM through the Blink Plus C wrapper (ctypes)."""

    def __init__(self, s: dict = SETTINGS):
        self.s, self.lib, self.board, self.shape = s, None, s["slm_board"], None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _error(self, what: str) -> HardwareError:
        """HardwareError with the SDK's last error message."""
        message = self.lib.Get_last_error_message() if self.lib is not None else b""
        return HardwareError(f"SLM {what} failed: {(message or b'').decode(errors='replace')}")

    def open(self) -> "RealSLM":
        """Load the SDK, find the board, check an 8-bit image depth and load the LUT."""
        from ctypes import CDLL, byref, c_char_p, c_uint, cdll
        cdll.LoadLibrary(self.s["slm_wrapper"])
        self.lib = CDLL("Blink_C_wrapper")
        self.lib.Get_last_error_message.restype = c_char_p
        num_boards, ok = c_uint(0), c_uint(0)
        self.lib.Create_SDK(byref(num_boards), byref(ok))
        if ok.value == 0 or num_boards.value < 1:
            raise self._error(f"Create_SDK (boards found {num_boards.value})")
        width, height = self.lib.Get_image_width(self.board), self.lib.Get_image_height(self.board)
        depth = self.lib.Get_image_depth(self.board)
        if depth // 8 != 1:
            self.close()
            raise HardwareError(f"SLM image depth {depth} bits: this layer writes 8-bit images only")
        self.shape = (height, width)
        # Load_LUT_file returns 0 on failure (as in Meadowlark's Blink_SDK_example.py).
        if self.lib.Load_LUT_file(self.board, self.s["slm_lut"].encode()) == 0:
            raise self._error(f"Load_LUT_file ({self.s['slm_lut']})")
        return self

    def show(self, phase: np.ndarray) -> None:
        """Write one phase image (H x W, uint8, the SLM's size) and wait until it is written."""
        from ctypes import POINTER, c_ubyte
        image = np.ascontiguousarray(phase, dtype=np.uint8)
        if image.shape != self.shape:
            raise ValueError(f"SLM image shape {image.shape}, expected {self.shape}")
        timeout = self.s["slm_timeout_ms"]
        if self.lib.Write_image(self.board, image.ctypes.data_as(POINTER(c_ubyte)), timeout) != 1:
            raise self._error("Write_image")
        if self.lib.ImageWriteComplete(self.board, timeout) != 1:
            raise self._error("ImageWriteComplete")

    def blank(self) -> None:
        """A flat (all zero) phase image."""
        self.show(np.zeros(self.shape, dtype=np.uint8))

    def close(self) -> None:
        """Blank image, then Delete_SDK (safe to call twice)."""
        if self.lib is None:
            return
        try:
            if self.shape is not None:
                self.blank()
        finally:
            self.lib.Delete_SDK()
            self.lib = None


class FakeSLM:
    """A stand-in SLM that keeps the last image (any machine)."""

    def __init__(self, s: dict = SETTINGS):
        self.s, self.shape, self.image, self.is_open = s, tuple(s["slm_size"]), None, False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "FakeSLM":
        self.is_open = True
        self.image = np.zeros(self.shape, dtype=np.uint8)
        return self

    def show(self, phase: np.ndarray) -> None:
        """Keep the image (same checks as the real SLM)."""
        if not self.is_open:
            raise HardwareError("SLM is not open")
        image = np.ascontiguousarray(phase, dtype=np.uint8)
        if image.shape != self.shape:
            raise ValueError(f"SLM image shape {image.shape}, expected {self.shape}")
        self.image = image

    def blank(self) -> None:
        self.show(np.zeros(self.shape, dtype=np.uint8))

    def close(self) -> None:
        if self.is_open:
            self.blank()
        self.is_open = False


class RealCamera:
    """The Alkeria camera through alkusb3: one 8-bit raw frame per grab."""

    def __init__(self, s: dict = SETTINGS):
        self.s, self.cam, self.shape = s, None, (s["image_size_y"], s["image_size_x"])

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "RealCamera":
        """Init the camera, set the window and settings (acquire off while setting), then start acquiring."""
        if self.s["trigger"]:
            raise HardwareError("external trigger: the alkusb3 trigger settings are not in this layer yet (HARDWARE "
                                "trigger must be False)")
        import alkusb3
        s = self.s
        try:
            self.cam = alkusb3.generic_camera()
            self.cam.camera = s["camera_index"]
            self.cam.init()
            self.cam.acquire = False
            for name in ("image_start_x", "image_start_y", "image_size_x", "image_size_y", "video_mode"):
                setattr(self.cam, name, s[name])
            self.cam.ADCResolution = s["adc_resolution"]
            self.cam.digital_gain = s["digital_gain"]
            self.cam.color_coding = getattr(alkusb3.color_coding, s["color_coding"])
            self.cam.shutter = s["shutter"]
            self.cam.frame_rate = s["frame_rate"]
            self.cam.acquire = True
        except Exception as error:
            self.close()
            raise HardwareError(f"camera open failed: {type(error).__name__}: {error}") from error
        return self

    def grab(self) -> np.ndarray:
        """One frame (image_size_y x image_size_x, uint8)."""
        try:
            data = self.cam.get_raw_data()
        except Exception as error:
            raise HardwareError(f"camera grab failed: {type(error).__name__}: {error}") from error
        if data is None or len(data) != self.shape[0] * self.shape[1]:
            raise HardwareError(f"camera frame has {0 if data is None else len(data)} bytes, expected "
                                f"{self.shape[0] * self.shape[1]} (8-bit {self.shape[1]} x {self.shape[0]})")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.shape).copy()

    def close(self) -> None:
        """Stop acquiring (safe to call twice)."""
        if self.cam is not None:
            try:
                self.cam.acquire = False
            finally:
                self.cam = None


class FakeCamera:
    """A stand-in camera: speckle from a fixed random transmission of the fake SLM's phase (16 x 16 macro pixels),
    with camera noise, as 8-bit frames of the configured size."""

    def __init__(self, slm: FakeSLM | None = None, s: dict = SETTINGS, seed: int = 0):
        self.s, self.slm, self.shape, self.is_open = s, slm, (s["image_size_y"], s["image_size_x"]), False
        self.rng = np.random.default_rng(seed)
        n_in, n_out = 16 * 16, self.shape[0] * self.shape[1]
        self.T = (self.rng.normal(size=(n_in, n_out)) + 1j * self.rng.normal(size=(n_in, n_out))) / np.sqrt(2 * n_in)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "FakeCamera":
        self.is_open = True
        return self

    def grab(self) -> np.ndarray:
        """One frame (image_size_y x image_size_x, uint8)."""
        if not self.is_open:
            raise HardwareError("camera is not open")
        phase = np.zeros(self.s["slm_size"], dtype=np.uint8) if self.slm is None or self.slm.image is None \
            else self.slm.image
        h, w = phase.shape
        macro = phase[: h - h % 16, : w - w % 16].reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))   # (16 x 16)
        E_in = np.exp(2j * np.pi * macro.reshape(1, -1) / 256)        # (1 x N), row convention E_out = E_in @ T
        intensity = np.abs(E_in @ self.T).reshape(self.shape) ** 2
        frame = 200 * intensity / 3.0 + self.rng.normal(0, 2, self.shape) + 10
        return np.clip(np.round(frame), 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.is_open = False


def SLM(fake: bool | None = None, s: dict = SETTINGS):
    """The SLM: RealSLM on the lab PC, FakeSLM with fake=True or on a machine that is not Windows."""
    return FakeSLM(s) if use_fake(fake) else RealSLM(s)


def Camera(fake: bool | None = None, slm=None, s: dict = SETTINGS):
    """The camera: RealCamera on the lab PC, FakeCamera (seeing the fake slm) with fake=True or not on Windows."""
    return FakeCamera(slm if isinstance(slm, FakeSLM) else None, s) if use_fake(fake) else RealCamera(s)


if __name__ == "__main__":
    # Self-test with the fake devices (any machine): shapes, checks, closing.
    with SLM(fake=True) as slm, Camera(fake=True, slm=slm) as cam:
        dark = cam.grab()
        ramp = np.tile((np.arange(slm.shape[1]) * 256 // slm.shape[1]).astype(np.uint8), (slm.shape[0], 1))
        slm.show(ramp)
        lit = cam.grab()
        print(f"fake SLM {slm.shape}, fake camera frames {dark.shape} {dark.dtype}: blank mean {dark.mean():.1f}, "
              f"ramp mean {lit.mean():.1f}, mean |difference| {np.abs(lit.astype(int) - dark).mean():.1f}")
        assert dark.shape == (SETTINGS["image_size_y"], SETTINGS["image_size_x"]) and dark.dtype == np.uint8
        assert np.abs(lit.astype(int) - dark).mean() > 1, "a test pattern must change the fake speckle"
        try:
            slm.show(np.zeros((10, 10), dtype=np.uint8))
            raise AssertionError("a wrong image shape must raise")
        except ValueError as error:
            print(f"wrong shape: {error}")
    assert not slm.is_open and not cam.is_open and not slm.image.any(), "closed, SLM blank"
    try:
        RealCamera({**SETTINGS, "trigger": True}).open()
        raise AssertionError("trigger mode must raise")
    except HardwareError as error:
        print(f"trigger on: {error}")
    assert use_fake(None) == (sys.platform != "win32")
    print("hardware.py self-test passed (fake devices)")
