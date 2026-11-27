"""
hardware.py - the real SLM and camera as a small hardware layer (no optics model, no agents).

SLM: Meadowlark PCIe SLM (1024 x 1024), SDK build 870 ("blink_pcie_870"), through its C wrapper DLL with ctypes; every
function gets its argtypes and restype, as in the lab's utils.py. Camera: Alkeria Celera C5S-MP (a 2 x 2 polarization
mosaic) over USB3 through the alkusb3 Python module. Both are used only on the lab PC (Windows); the DLL and alkusb3
are imported inside open(), so this module imports on any machine.
Based on the lab PC's working code (tm_engineering/utils/utils.py and config4.json, 2026-11-26) and on Meadowlark's
Blink_SDK_example.py. All settings are in config.py HARDWARE.

SLM(fake=...) and Camera(fake=...) give FakeSLM / FakeCamera with fake=True, and automatically on a machine that is not
Windows: the fake camera sees a speckle pattern from a fixed random transmission of the fake SLM's phase, so a test
pattern changes the image as on the real setup. Every SDK return value is checked; a failure raises HardwareError
(a LabError, so the team graph handles it as a hardware error) with the SDK's own message. Both classes are context
managers and always close in __exit__ (SLM: blank image, then Delete_SDK; camera: acquire off).
External trigger (HARDWARE trigger, on by default as in the lab's working setup): the SLM sends an output pulse on each
flip and the camera starts a frame on its external input; with trigger False the camera runs free and the SLM sends no
pulse. SLM images are built with slm_frame(values) (the lab's prepare_slm_data: macro pixels, bias grating, gap mask);
the camera sits in the bias grating's first order, so without the grating it sees no light. reference_frame() is the
grating alone. All measurement code takes frames only through measure(slm, camera, phase): the SLM image, then at once the
camera frame (with the trigger the frame is taken at the SLM write and the camera trigger delay does the settling; a
sleep in between loses it; without the trigger measure waits settle_s). warm_up(slm, camera) runs measure() with a
blank image warmup_frames times (as in the lab code) and is called once both devices are open.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from agents.core.lab_api import LabError
from config import HARDWARE as SETTINGS


class HardwareError(LabError):
    """The SLM or camera SDK reported a failure."""


def use_fake(fake: bool | None) -> bool:
    """fake as given, or (None) True on any machine that is not Windows."""
    return sys.platform != "win32" if fake is None else fake


# Polarization mosaic of the camera: polarizer angle (deg) -> (row, column) offset of its sub-image raw[r::2, c::2]
# [source: lab PC utils.py, 2026-11-26]: raw[::2, ::2] 90 deg, raw[::2, 1::2] 45 deg, raw[1::2, ::2] 135 deg,
# raw[1::2, 1::2] 0 deg.
MOSAIC = {"90": (0, 0), "45": (0, 1), "135": (1, 0), "0": (1, 1)}


def channel(raw: np.ndarray, angle: str) -> np.ndarray:
    """The sub-image (H/2 x W/2, float32) of one polarizer angle (a MOSAIC key) of a mosaic frame (H x W)."""
    r, c = MOSAIC[angle]
    return np.asarray(raw, dtype=np.float32)[r::2, c::2]


def intensity(raw: np.ndarray) -> np.ndarray:
    """Intensity (H/2 x W/2) of a polarization mosaic frame (H x W): the mean of its four polarizer sub-images
    (MOSAIC)."""
    return sum(channel(raw, angle) for angle in MOSAIC) / 4.0


class RealSLM:
    """The Meadowlark SLM (SDK build 870) through its C wrapper DLL (ctypes)."""

    def __init__(self, s: dict = SETTINGS):
        self.s, self.lib, self.board, self.shape, self._dll_dir = s, None, s["slm_board"], None, None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _message(self) -> str:
        """The SDK's last error message ("" if none or no SDK)."""
        if self.lib is None:
            return ""
        return (self.lib.Get_last_error_message() or b"").decode(errors="replace")

    def _fail(self, what: str) -> HardwareError:
        """HardwareError for a failed step, with the SDK's last error message."""
        return HardwareError(f"SLM {what} failed: {self._message() or 'no SDK message'}")

    def _check(self, result: int, what: str) -> None:
        """Raise HardwareError unless an SDK call returned 1."""
        if result != 1:
            raise self._fail(f"{what} (returned {result})")

    def open(self) -> "RealSLM":
        """Load the DLL (its folder on the DLL search path), create the SDK, check the board, size and 8-bit depth,
        set wait-for-trigger off, flip-immediate and the output pulse (HARDWARE trigger), and load the LUT. Anything
        that fails after Create_SDK calls Delete_SDK before the error is raised."""
        import ctypes
        from ctypes import POINTER, byref, c_bool, c_char_p, c_double, c_int, c_ubyte, c_uint
        wrapper = self.s["slm_wrapper"]
        self._dll_dir = os.add_dll_directory(os.path.dirname(wrapper))      # keep the handle while the DLL is used
        lib = ctypes.CDLL(wrapper)
        for name, argtypes, restype in [
            ("Create_SDK", [POINTER(c_uint), POINTER(c_int)], None), ("Delete_SDK", [], None),
            ("Get_last_error_message", [], c_char_p),
            ("Get_image_width", [c_int], c_int), ("Get_image_height", [c_int], c_int), ("Get_image_depth", [c_int], c_int),
            ("Read_SLM_temperature", [c_int], c_double), ("Load_LUT_file", [c_int, c_char_p], c_int),
            ("Write_image", [c_int, POINTER(c_ubyte), c_uint], c_int), ("ImageWriteComplete", [c_int, c_uint], c_int),
            ("SetWaitForTrigger", [c_int, c_bool], c_int), ("SetFlipImmediate", [c_int, c_bool], c_int),
            ("SetOutputPulse", [c_int, c_bool], c_int)]:
            getattr(lib, name).argtypes, getattr(lib, name).restype = argtypes, restype
        self.lib = lib
        num_boards, ok = c_uint(0), c_int(0)
        lib.Create_SDK(byref(num_boards), byref(ok))
        try:
            if ok.value <= 0 or not 1 <= self.board <= num_boards.value:
                raise self._fail(f"Create_SDK (ok {ok.value}, boards {num_boards.value}, board {self.board})")
            width, height = lib.Get_image_width(self.board), lib.Get_image_height(self.board)
            if (height, width) != tuple(self.s["slm_size"]):
                raise self._fail(f"size check ({width} x {height}, expected {self.s['slm_size'][1]} x "
                                 f"{self.s['slm_size'][0]})")
            depth = lib.Get_image_depth(self.board)
            if depth // 8 != 1:
                raise self._fail(f"depth check ({depth} bits; this layer writes 8-bit images only)")
            self.shape = (height, width)
            self._check(lib.SetWaitForTrigger(self.board, False), "SetWaitForTrigger(False)")
            self._check(lib.SetFlipImmediate(self.board, bool(self.s["slm_flip_immediate"])), "SetFlipImmediate")
            self._check(lib.SetOutputPulse(self.board, bool(self.s["trigger"])), "SetOutputPulse")
            self._check(lib.Load_LUT_file(self.board, os.fsencode(self.s["slm_lut"])), f"Load_LUT_file ({self.s['slm_lut']})")
        except Exception:
            self._shut_down()
            raise
        return self

    def show(self, phase: np.ndarray) -> None:
        """Write one phase image (H x W, uint8, the SLM's size) and wait until it is written."""
        from ctypes import POINTER, c_ubyte
        image = np.ascontiguousarray(phase, dtype=np.uint8)
        if image.shape != self.shape:
            raise ValueError(f"SLM image shape {image.shape}, expected {self.shape}")
        timeout = self.s["slm_timeout_ms"]
        self._check(self.lib.Write_image(self.board, image.ctypes.data_as(POINTER(c_ubyte)), timeout), "Write_image")
        self._check(self.lib.ImageWriteComplete(self.board, timeout), "ImageWriteComplete")

    def blank(self) -> None:
        """A flat (all zero) phase image."""
        self.show(np.zeros(self.shape, dtype=np.uint8))

    def temperature(self) -> float:
        """The SLM temperature in C (Read_SLM_temperature)."""
        return float(self.lib.Read_SLM_temperature(self.board))

    def _shut_down(self) -> None:
        """Delete_SDK and release the DLL folder (safe to call twice)."""
        if self.lib is not None:
            self.lib.Delete_SDK()
            self.lib = None
        if self._dll_dir is not None:
            self._dll_dir.close()
            self._dll_dir = None

    def close(self) -> None:
        """Blank image, then Delete_SDK (safe to call twice)."""
        try:
            if self.lib is not None and self.shape is not None:
                self.blank()
        finally:
            self._shut_down()


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

    def temperature(self) -> float:
        """A fixed 25.0 C."""
        return 25.0

    def close(self) -> None:
        if self.is_open:
            self.blank()
        self.is_open = False


class RealCamera:
    """The Alkeria polarization camera through alkusb3: one raw 8-bit mosaic frame per grab."""

    def __init__(self, s: dict = SETTINGS):
        self.s, self.cam, self.shape = s, None, (s["image_size_y"], s["image_size_x"])

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "RealCamera":
        """Open the camera in the order of the lab's working open_camera() (tm_engineering utils.py): camera index before
        and after init, acquire off; video mode and window size BEFORE the window start (start + size may not exceed the
        sensor); ADC resolution and gain; the frame start trigger (debounce on its input port); colour coding and
        shutter; acquire on, then the frame rate. Without the trigger it drops warmup_frames frames here; with the
        trigger a frame needs an SLM image first, so use warm_up(slm, camera) after both are open."""
        import alkusb3
        s = self.s
        if s["trigger"] and s["trigger_edge"] is None:
            raise HardwareError("external trigger on, but HARDWARE trigger_edge is not set")
        try:
            self.cam = alkusb3.generic_camera()
            self.cam.camera = s["camera_index"]
            self.cam.init()
            self.cam.acquire = False
            self.cam.camera = s["camera_index"]
            # The window size before its start: the camera refuses a start + (old, full) size beyond the sensor.
            for name in ("video_mode", "image_size_x", "image_size_y", "image_start_x", "image_start_y"):
                setattr(self.cam, name, s[name])
            self.cam.ADCResolution = s["adc_resolution"]
            self.cam.digital_gain = s["digital_gain"]
            trigger = self.cam.frame_start_trigger
            if s["trigger"]:
                self.cam.pio_ports[s["trigger_input"]].debounce_time = s["trigger_debounce"]
                trigger.source = alkusb3.trigger_source.external
                trigger.external_input = s["trigger_input"]
                trigger.detect_external_input_edge = s["trigger_edge"]
                trigger.invert_external_input = True
                trigger.delay = s["trigger_delay"]
                trigger.enabled = True
            else:
                trigger.enabled = False
            self.cam.color_coding = getattr(alkusb3.color_coding, s["color_coding"])
            self.cam.shutter = s["shutter"]
            self.cam.acquire = True
            self.cam.frame_rate = s["frame_rate"]                 # after acquire, as in the lab code
            if not s["trigger"]:                                   # with the trigger: warm_up(slm, camera) instead
                for _ in range(s["warmup_frames"]):
                    self.grab()
        except HardwareError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise HardwareError(f"camera open failed: {type(error).__name__}: {error}") from error
        return self

    def grab(self) -> np.ndarray:
        """One raw mosaic frame (image_size_y x image_size_x, float32 counts). With the trigger the camera has a frame
        only after an SLM image (its output pulse); take frames through measure()."""
        # Frames are read as (image_size_y, image_size_x) rows x columns; the lab code reads them with PIL using
        # (size_y, size_x) as (width, height). For a non-square window check the axes on the real camera; the 96 x 96
        # window is square, so it does not matter now.
        no_frame = ("no camera frame: with the external trigger the camera only takes a frame after an SLM image (output "
                    "pulse); check the trigger cable or run with --no-trigger" if self.s["trigger"] else
                    "no camera frame (get_raw_data gave nothing)")
        try:
            data = self.cam.get_raw_data()
        except AttributeError as error:          # alkusb3 without a frame: "'NoneType' object has no attribute 'to_bytes'"
            raise HardwareError(no_frame) from error
        except Exception as error:
            raise HardwareError(f"camera grab failed: {type(error).__name__}: {error}") from error
        if data is None:
            raise HardwareError(no_frame)
        if len(data) != self.shape[0] * self.shape[1]:
            raise HardwareError(f"camera frame has {0 if data is None else len(data)} bytes, expected "
                                f"{self.shape[0] * self.shape[1]} (8-bit {self.shape[1]} x {self.shape[0]})")
        return np.frombuffer(data, dtype=np.uint8).reshape(self.shape).astype(np.float32)

    def grab_intensity(self) -> np.ndarray:
        """One frame as intensity (half size, the mean of the four polarizer sub-images)."""
        return intensity(self.grab())

    def close(self) -> None:
        """Stop acquiring (safe to call twice)."""
        if self.cam is not None:
            try:
                self.cam.acquire = False
            finally:
                self.cam = None


class FakeCamera:
    """A stand-in camera in the first diffraction order of the SLM's column grating: the field of each of 16 x 16 blocks
    is the mean of exp(i phase) * (-1)^column (zero for a flat image, full for the bias grating), sent through a fixed
    random transmission to speckle grains of fake_grain_px pixels, plus camera noise; raw frames of the configured size (float32 counts 0..255). So a blank SLM
    gives a dark frame and the grating a bright speckle, as on the real setup."""

    def __init__(self, slm: FakeSLM | None = None, s: dict = SETTINGS, seed: int = 0):
        self.s, self.slm, self.shape, self.is_open = s, slm, (s["image_size_y"], s["image_size_x"]), False
        self.rng = np.random.default_rng(seed)
        self.grain = s["fake_grain_px"]                           # speckle grain: one random output per grain
        n_in, n_out = 16 * 16, (self.shape[0] // self.grain) * (self.shape[1] // self.grain)
        self.T = (self.rng.normal(size=(n_in, n_out)) + 1j * self.rng.normal(size=(n_in, n_out))) / np.sqrt(2 * n_in)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "FakeCamera":
        self.is_open = True
        return self

    def grab(self) -> np.ndarray:
        """One raw frame (image_size_y x image_size_x, float32 counts)."""
        if not self.is_open:
            raise HardwareError("camera is not open")
        phase = np.zeros(self.s["slm_size"], dtype=np.uint8) if self.slm is None or self.slm.image is None \
            else self.slm.image
        h, w = phase.shape
        # First order of the period-2 column grating: each pixel's field with the sign of its column.
        first_order = np.exp(2j * np.pi * phase / 256.0) * np.where(np.arange(w) % 2 == 0, 1.0, -1.0)
        blocks = first_order[: h - h % 16, : w - w % 16].reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        E_in = blocks.reshape(1, -1)                                  # (1 x N), row convention E_out = E_in @ T
        coarse = np.abs(E_in @ self.T).reshape(self.shape[0] // self.grain, self.shape[1] // self.grain) ** 2
        speckle = np.kron(coarse, np.ones((self.grain, self.grain)))  # grains of grain x grain pixels
        frame = 200 * speckle / 3.0 + self.rng.normal(0, 1, self.shape) + 2
        return np.clip(np.round(frame), 0, 255).astype(np.float32)

    def grab_intensity(self) -> np.ndarray:
        """One frame as intensity (half size), as on the real camera."""
        return intensity(self.grab())

    def close(self) -> None:
        self.is_open = False


def slm_frame(values: np.ndarray, s: dict = SETTINGS) -> np.ndarray:
    """The SLM image (H x W uint8) for values (grid x grid, 0..255) on the macro pixels, built like the lab's
    prepare_slm_data for a square SLM: each value repeated over its macro pixel, plus the bias grating on every second
    column ([:, ::2], uint8 wrap), times a mask that is 0 on the first gap_px rows and columns of every macro pixel."""
    grid, (height, width) = s["slm_macro_grid"], s["slm_size"]
    values = np.asarray(values)
    if values.shape != (grid, grid):
        raise ValueError(f"SLM values shape {values.shape}, expected ({grid}, {grid})")
    if values.min() < 0 or values.max() > 255:
        raise ValueError("SLM values must be within 0 .. 255")
    macro = height // grid
    data = np.repeat(np.repeat(values.astype(np.uint8), macro, axis=0), macro, axis=1)         # (H x W)
    grating = np.zeros((height, width), dtype=np.uint8)
    grating[:, ::2] = s["slm_bias_grating"]
    data = data + grating                                                                       # uint8: wraps at 256
    keep_rows = (np.arange(height) % macro) >= s["slm_gap_px"]
    keep_cols = (np.arange(width) % macro) >= s["slm_gap_px"]
    mask = np.outer(keep_rows, keep_cols).astype(np.uint8)
    return data * mask


def reference_frame(s: dict = SETTINGS) -> np.ndarray:
    """The bias grating alone (all values 0): the reference SLM image (H x W uint8)."""
    return slm_frame(np.zeros((s["slm_macro_grid"], s["slm_macro_grid"]), dtype=np.uint8), s)


def measure(slm, camera, phase: np.ndarray) -> np.ndarray:
    """The camera frame (raw, float32, the camera's shape) for one SLM phase image (H x W uint8): the image, then the
    frame at once with the trigger (the frame is taken at the SLM write; a sleep in between loses it), or after
    settle_s without it. The only way later measurement code takes frames."""
    slm.show(phase)
    if not camera.s["trigger"]:
        time.sleep(camera.s["settle_s"])
    return camera.grab()


def warm_up(slm, camera, n: int = SETTINGS["warmup_frames"]) -> None:
    """n warm-up frames through measure() with the reference image (the bias grating), discarded (as in the lab code).
    Call it once both devices are open, before the first measured frame."""
    for _ in range(n):
        measure(slm, camera, reference_frame(camera.s))


def SLM(fake: bool | None = None, s: dict = SETTINGS):
    """The SLM: RealSLM on the lab PC, FakeSLM with fake=True or on a machine that is not Windows."""
    return FakeSLM(s) if use_fake(fake) else RealSLM(s)


def Camera(fake: bool | None = None, slm=None, s: dict = SETTINGS):
    """The camera: RealCamera on the lab PC, FakeCamera (seeing the fake slm) with fake=True or not on Windows."""
    return FakeCamera(slm if isinstance(slm, FakeSLM) else None, s) if use_fake(fake) else RealCamera(s)


if __name__ == "__main__":
    import types

    # 1. The fake devices (any machine): shapes, the intensity of the mosaic, checks, closing.
    # slm_frame: shape, the bias grating on even columns, the gap mask, the uint8 wrap.
    grid, gap, bias = SETTINGS["slm_macro_grid"], SETTINGS["slm_gap_px"], SETTINGS["slm_bias_grating"]
    macro = SETTINGS["slm_size"][0] // grid
    values = np.full((grid, grid), 200, dtype=np.uint8)
    values[0, 1] = 50
    frame = slm_frame(values)
    inside = (gap, gap)                                             # first pixel after the gap (row, even column)
    assert frame.shape == tuple(SETTINGS["slm_size"]) and frame.dtype == np.uint8
    assert frame[inside] == (200 + bias) % 256 and frame[gap, gap + 1] == 200, "grating on even columns, uint8 wrap"
    assert frame[gap, macro + gap] == (50 + bias) % 256 and frame[gap, macro + gap + 1] == 50, "the macro pixel's value"
    assert not frame[:gap].any() and not frame[:, macro:macro + gap].any(), "gap rows and columns are 0"
    ref = reference_frame()
    assert (ref[gap, gap::2][:macro // 2 - gap // 2] == bias).all() and not ref[gap, gap + 1::2].any()
    for bad in (np.zeros((3, 3)), np.full((grid, grid), 300)):
        try:
            slm_frame(bad)
            raise AssertionError("bad SLM values must raise")
        except ValueError:
            pass
    print(f"slm_frame: {frame.shape}, value + {bias} on even columns (200 -> {frame[inside]}), gap {gap} px masked")

    with SLM(fake=True) as slm, Camera(fake=True, slm=slm) as cam:
        dark = cam.grab()                                         # blank SLM: no grating, no light
        slm.show(reference_frame())
        lit, lit_i = cam.grab(), cam.grab_intensity()
        slm.show(slm_frame(np.random.default_rng(0).integers(0, 256, (grid, grid))))
        other = cam.grab()
        print(f"fake SLM {slm.shape} at {slm.temperature():g} C, fake camera raw {dark.shape} {dark.dtype}, intensity "
              f"{lit_i.shape}: blank mean {dark.mean():.1f}, grating mean {lit.mean():.1f}, random values mean "
              f"{other.mean():.1f}")
        assert dark.shape == (SETTINGS["image_size_y"], SETTINGS["image_size_x"]) and dark.dtype == np.float32
        assert lit_i.shape == (dark.shape[0] // 2, dark.shape[1] // 2) and slm.temperature() == 25.0
        assert dark.mean() < 5 and lit.mean() > 30, "blank dark, grating bright (the camera is in the first order)"
        assert np.abs(other - lit).mean() > 5, "other values change the speckle"
        try:
            slm.show(np.zeros((10, 10), dtype=np.uint8))
            raise AssertionError("a wrong image shape must raise")
        except ValueError as error:
            print(f"wrong shape: {error}")
    assert not slm.is_open and not cam.is_open and not slm.image.any(), "closed, SLM blank"
    mosaic = np.array([[90, 45], [135, 0]], dtype=np.float32)
    assert np.array_equal(intensity(np.tile(mosaic, (2, 2))), np.full((2, 2), 67.5, dtype=np.float32))

    # 2. RealCamera.open with a stand-in alkusb3 module: every setting and the trigger are passed through, the
    # warm-up frames are dropped, and the camera stops acquiring when closed.
    SENSOR = {"x": 1232, "y": 1028}     # x from the real camera's error message; y assumed (full height)
    FRAME_WINDOW_S = 0.01               # s, a triggered frame must be read this soon after the SLM write (real: ~10 ms)

    class StandInCamera:
        """Like the real camera: a full-sensor window at first, start + size may never exceed the sensor; records the
        order of every setting; get_raw_data counts the frames."""
        def __init__(self):
            for name, value in [("order", []), ("frames", 0), ("frame_start_trigger", types.SimpleNamespace()),
                                ("pio_ports", [types.SimpleNamespace(debounce_time=None) for _ in range(4)]),
                                ("image_start_x", 0), ("image_start_y", 0), ("image_size_x", SENSOR["x"]),
                                ("image_size_y", SENSOR["y"]), ("slm_written_t", None)]:
                object.__setattr__(self, name, value)

        def __setattr__(self, name, value):
            for axis in ("x", "y"):
                if name in (f"image_start_{axis}", f"image_size_{axis}"):
                    start = value if name.startswith("image_start") else getattr(self, f"image_start_{axis}")
                    size = value if name.startswith("image_size") else getattr(self, f"image_size_{axis}")
                    if start + size > SENSOR[axis]:
                        raise ValueError(f"ImageStart{axis.upper()} + ImageSize{axis.upper()} cannot exceed {SENSOR[axis]}")
            self.order.append((name, value))
            object.__setattr__(self, name, value)

        def init(self):
            self.order.append(("init", None))

        def get_raw_data(self):
            """A frame; with the trigger enabled it is taken at the SLM write and read only right after it (within
            FRAME_WINDOW_S, no other read since), else no frame, as alkusb3 on the lab PC."""
            if getattr(self.frame_start_trigger, "enabled", False):
                if self.slm_written_t is None or time.monotonic() - self.slm_written_t > FRAME_WINDOW_S:
                    object.__setattr__(self, "slm_written_t", None)
                    return None.to_bytes()                           # AttributeError, as on the lab PC
                object.__setattr__(self, "slm_written_t", None)
            object.__setattr__(self, "frames", self.frames + 1)        # a frame, not a setting
            return bytes(SETTINGS["image_size_x"] * SETTINGS["image_size_y"])

    stand_in = types.SimpleNamespace(generic_camera=StandInCamera,
                                     color_coding=types.SimpleNamespace(raw8="raw8 coding"),
                                     trigger_source=types.SimpleNamespace(external="external source"))
    sys.modules["alkusb3"] = stand_in
    assert SETTINGS["trigger"] is True and SETTINGS["trigger_edge"] is True, "the default is the lab's triggered setup"
    for on in (True, False):
        s = SETTINGS if on else {**SETTINGS, "trigger": False}
        camera = RealCamera(s).open()
        cam, trigger = camera.cam, camera.cam.frame_start_trigger
        settings_ok = (cam.image_start_x, cam.image_size_y, cam.ADCResolution, cam.color_coding, cam.shutter,
                       cam.frame_rate) == (s["image_start_x"], s["image_size_y"], s["adc_resolution"], "raw8 coding",
                                           s["shutter"], s["frame_rate"])
        names = [name for name, _ in cam.order]
        warm_open = cam.frames                                     # frames grabbed inside open()
        if on:
            # Without an SLM image no frame: the clear error. Then warm_up: an SLM image before each frame.
            try:
                camera.grab()
                raise AssertionError("with the trigger, a grab without an SLM image must fail")
            except HardwareError as error:
                print(f"stand-in camera, trigger on, grab without an SLM image: {error}")
                assert "only takes a frame after an SLM image" in str(error)

        class TriggeringSLM(FakeSLM):
            """A fake SLM whose every image sends the stand-in camera a trigger pulse (the moment of the write)."""
            def show(self, phase):
                super().show(phase)
                object.__setattr__(cam, "slm_written_t", time.monotonic())
        with TriggeringSLM() as pulse_slm:
            warm_up(pulse_slm, camera)
            frame = measure(pulse_slm, camera, np.full(pulse_slm.shape, 128, dtype=np.uint8))
            assert frame.shape == camera.shape and cam.frames == warm_open + s["warmup_frames"] + 1
            print(f"stand-in camera, trigger {on}: warm_up and measure() give frames")
            if on:
                # show, sleep settle_s, grab: the triggered frame came during the sleep and is lost.
                pulse_slm.show(np.zeros(pulse_slm.shape, dtype=np.uint8))
                time.sleep(s["settle_s"])
                try:
                    camera.grab()
                    raise AssertionError("with the trigger, show + sleep + grab must lose the frame")
                except HardwareError as error:
                    print(f"stand-in camera, trigger on, show + sleep {s['settle_s']} s + grab: {error}")
        print(f"stand-in camera, trigger {on}: settings passed {settings_ok}, warm-up frames in open() {warm_open}, "
              f"trigger {vars(trigger)}, debounce on input {s['trigger_input']}: "
              f"{cam.pio_ports[s['trigger_input']].debounce_time}")
        assert settings_ok and warm_open == (0 if on else s["warmup_frames"]) and cam.acquire is True
        # The lab's order: camera, init, acquire off, camera again; size before start; frame rate after acquire on.
        assert names[:4] == ["camera", "init", "acquire", "camera"], names[:4]
        assert names.index("image_size_x") < names.index("image_start_x") and \
            names.index("image_size_y") < names.index("image_start_y")
        assert names.index("frame_rate") == len(names) - 1 and cam.order[-2] == ("acquire", True)
        if on:
            assert vars(trigger) == {"source": "external source", "external_input": s["trigger_input"],
                                     "detect_external_input_edge": True, "invert_external_input": True,
                                     "delay": s["trigger_delay"], "enabled": True}
            assert cam.pio_ports[s["trigger_input"]].debounce_time == s["trigger_debounce"]
        else:
            assert vars(trigger) == {"enabled": False} and cam.pio_ports[s["trigger_input"]].debounce_time is None
        camera.close()
        assert cam.acquire is False and camera.cam is None
    old_order = StandInCamera()
    try:
        for name in ("image_start_x", "image_start_y", "image_size_x", "image_size_y"):     # the old order: start first
            setattr(old_order, name, SETTINGS[name])
        raise AssertionError("the old order (start before size) must fail on the stand-in camera")
    except ValueError as error:
        print(f"old order (start before size): {error}")
    try:
        RealCamera({**SETTINGS, "trigger": True, "trigger_edge": None}).open()
        raise AssertionError("trigger on without trigger_edge must raise")
    except HardwareError as error:
        print(f"trigger on, no edge: {error}")
    del sys.modules["alkusb3"]

    # 3. RealSLM.open with a stand-in DLL: argtypes/restype set, return values checked (1 = success), the trigger
    # flag reaches SetOutputPulse, Delete_SDK after a failure, and close() blanks the SLM before Delete_SDK.
    import ctypes

    class StandInFunction:
        """One DLL function: returns its value and records its calls (argtypes/restype can be set)."""
        def __init__(self, log, name, value):
            self.log, self.name, self.value = log, name, value

        def __call__(self, *args):
            self.log.append((self.name, args))
            return self.value(*args) if callable(self.value) else self.value

    def stand_in_dll(log, lut_result=1):
        """A stand-in Blink_C_wrapper with a 1024 x 1024, 8-bit board 1."""
        def create(num_boards, ok):
            num_boards._obj.value, ok._obj.value = 1, 1
        values = {"Create_SDK": create, "Delete_SDK": None, "Get_last_error_message": b"stand-in message",
                  "Get_image_width": 1024, "Get_image_height": 1024, "Get_image_depth": 8,
                  "Read_SLM_temperature": 31.5, "Load_LUT_file": lut_result, "Write_image": 1, "ImageWriteComplete": 1,
                  "SetWaitForTrigger": 1, "SetFlipImmediate": 1, "SetOutputPulse": 1}
        return types.SimpleNamespace(**{n: StandInFunction(log, n, v) for n, v in values.items()})

    class DllDir:
        def close(self):
            pass

    real_cdll, had_add = ctypes.CDLL, hasattr(os, "add_dll_directory")
    os.add_dll_directory = lambda path: DllDir()
    try:
        log = []
        ctypes.CDLL = lambda path: stand_in_dll(log)
        slm = RealSLM(SETTINGS).open()                      # the default: trigger on
        assert slm.lib.Write_image.argtypes[1] == ctypes.POINTER(ctypes.c_ubyte) and slm.lib.Write_image.restype == ctypes.c_int
        assert ("SetOutputPulse", (1, True)) in log and ("SetWaitForTrigger", (1, False)) in log
        slm.show(np.zeros((1024, 1024), dtype=np.uint8))
        print(f"stand-in SLM: open, show, temperature {slm.temperature():g} C")
        slm.close()
        names = [name for name, _ in log]
        assert names[-3:] == ["Write_image", "ImageWriteComplete", "Delete_SDK"], names[-3:]
        log = []
        ctypes.CDLL = lambda path: stand_in_dll(log, lut_result=0)
        try:
            RealSLM(SETTINGS).open()
            raise AssertionError("Load_LUT_file 0 must raise")
        except HardwareError as error:
            print(f"stand-in SLM, LUT fails: {error}")
            assert "stand-in message" in str(error) and [n for n, _ in log][-1] == "Delete_SDK"
    finally:
        ctypes.CDLL = real_cdll
        if not had_add:
            del os.add_dll_directory
    assert use_fake(None) == (sys.platform != "win32")
    print("hardware.py self-test passed (fake devices, stand-in alkusb3 and SLM DLL)")
