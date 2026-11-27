"""
lab.py - RealLab: the real box behind the same interface as the simulated lab (agents.core.lab_api.Lab).

The thermometer, the status, set_target and set_control talk to the box web API (REAL_BOX "url" = BOX_URL from .env,
the status JSON; targets are POSTed as {"target": T} to REAL_BOX "target_path" on the same host). The box has no on/off
endpoint: set_control(False) sets REAL_BOX "off_target_C" (above the box temperature, so the cooler stays idle, PWM 0)
and keeps the last target; set_control(True) sets that target back. The PID gains are never written. The room temperature
comes from ROOM_URL (the room channel, read at most every room_interval_s).

measure_map, check_map and focus use the real SLM and camera (real_lab/hardware.py), opened on the first optics call
(then warm_up) and kept open until close(); box-only use never touches them. N = slm_macro_grid^2 SLM inputs (random
quantized phase patterns as in the simulation, shown with slm_frame: macro pixels, bias grating, gaps; phase <-> gray
with HARDWARE slm_phase_levels = 256, the SLM's LUT), M = out_grid^2 outputs (camera_outputs: single pixels of one
polarization channel). The maths is the simulation's (ggs.recover, metrics.prediction_correlation, metrics.focus_pattern,
metrics.enhancement); lab times are the real wall-clock durations. A map is saved with its patterns, intensities,
reference frame and SLM temperature in HARDWARE maps_dir (fake devices: fake_maps_dir). wait() uses the wall clock and writes a thermometer
row to the lab database every STATE temperature_every_s, as the simulated lab does. truth() is None: nobody knows the
true temperature of the real box. Box request failures raise LabError (a hardware error the team graph handles).
The self-test uses a stand-in web server on 127.0.0.1; it never contacts the box.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

from agents.core.lab_api import SETTINGS as SIM
from agents.core.lab_api import Lab, LabError
from config import HARDWARE, OPTICS, REAL_BOX, STATE
from optics_simulation import ggs, metrics
from optics_simulation import lab as optics
from real_lab import hardware
from real_lab.logger import SSL_CONTEXT, read_room


def camera_outputs(raw: np.ndarray, s: dict = HARDWARE) -> np.ndarray:
    """The M = out_grid^2 outputs of one raw camera frame (H x W): single pixels of one polarization channel
    (polarization_channel's 48 x 48 sub-image, every output_stride-th pixel), row by row. The one place that decides
    what a camera "point" is. Each output must be one field (one speckle grain, one polarization): GGS fits a rank-1
    field per output, so a mean over several grains or over the four polarizations cannot be fitted (on the lab PC a
    6 x 6 block mean gave check_map 0.05, single 0-deg pixels 0.82)."""
    points = hardware.channel(raw, s["polarization_channel"])[:: s["output_stride"], :: s["output_stride"]]
    if points.shape != (s["out_grid"], s["out_grid"]):
        raise ValueError(f"camera outputs {points.shape}, expected out_grid x out_grid = {s['out_grid']} x "
                         f"{s['out_grid']} (window / 2 / output_stride)")
    return points.ravel()


class RealLab(Lab):
    """The real box (web API), room sensor, SLM and camera."""

    def __init__(self, state=None, run_name: str = "real_lab", box_url: str | None = None, room_url: str | None = None,
                 settings: dict | None = None, hardware_settings: dict | None = None, fake_hardware: bool | None = None,
                 maps_dir: str | Path | None = None):
        """state: optional LabState (actions, notes and thermometer rows are recorded there under run_name);
        box_url / room_url: default REAL_BOX url / room_url (BOX_URL / ROOM_URL from .env); hardware_settings:
        HARDWARE overrides; fake_hardware: the fake SLM and camera (None: fake on a machine that is not Windows);
        maps_dir: where maps are saved (default HARDWARE maps_dir, or fake_maps_dir with fake devices)."""
        super().__init__()
        self.s = {**REAL_BOX, **(settings or {})}
        self.box_url = box_url or self.s["url"]
        self.room_url = room_url if room_url is not None else self.s["room_url"]
        if not self.box_url:
            raise ValueError("no box URL: set BOX_URL in .env (see .env.example)")
        self.hw = {**HARDWARE, **(hardware_settings or {})}
        self.N, self.M = self.hw["slm_macro_grid"] ** 2, self.hw["out_grid"] ** 2
        self.n_points = self.M                     # camera points for focus (SafeLab checks the range)
        self.fake_hardware, self.maps_dir = fake_hardware, maps_dir
        self._slm = self._camera = None            # opened on the first optics call
        self.rng = np.random.default_rng(self.hw["pattern_seed"])
        self.ggs_rng = np.random.default_rng(self.hw["ggs_seed"])
        self.X_est = self.map_time = self.last_check = self.last_eta = self._background = None
        self._start = time.monotonic()
        self._room, self._room_t = None, -float("inf")
        self.state = state
        self.run_id = None if state is None else state.start_run(run_name, 0, {"box_url": "BOX_URL from .env"})
        self.last_row_t = -float("inf")
        self.target = None                         # the last target set (restored by set_control(True))

    @property
    def clock(self) -> float:
        """Wall-clock seconds since this lab was made."""
        return time.monotonic() - self._start

    def _request(self, url: str, body: dict | None = None) -> dict:
        """GET (body None) or POST JSON to url; the parsed JSON answer ({} if empty). Raises LabError on failure."""
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers={"User-Agent": self.s["user_agent"],
                                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.s["timeout_s"], context=SSL_CONTEXT) as response:
                text = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LabError(f"box request failed: {error}") from error
        return json.loads(text) if text else {}

    def _box(self) -> dict:
        """The box status JSON (temperature, filtered, target, duty, ...)."""
        return self._request(self.box_url)

    def room(self) -> float | None:
        """Room temperature in C from the room sensor's case sensor (room_temp; the logger also saves its probe, which
        reads about 0.65 C warmer and made the box fit need an unphysical -3.2 W internal heat), read at most every
        room_interval_s; None if the read fails or there is no ROOM_URL."""
        if not self.room_url:
            return None
        if self.clock - self._room_t >= self.s["room_interval_s"]:
            try:
                self._room = float(read_room(self.room_url, self.s["room_channel"], self.s["room_timeout_s"])["room_temp"])
            except Exception as error:
                self._log(f"room read failed ({error})")
                self._room = None
            self._room_t = self.clock
        return self._room

    def _record(self, t_s: float, action: str, args: dict, result) -> None:
        """One action in the lab database, if there is one."""
        if self.state is not None:
            self.state.add_action(self.run_id, t_s, self.actor, action, args, result)
            self.state.commit()

    def _thermometer_row(self) -> None:
        """A thermometer row (Kalman = the box's filtered value, raw, target, duty) in the lab database."""
        box = self._box()
        if self.state is not None:
            self.state.add_temperature(self.run_id, self.clock, float(box["filtered"]), float(box["temperature"]),
                                       float(box["target"]), int(box["duty"]))
            self.state.commit()
        self.last_row_t = self.clock

    def note(self, text: str, level: str = "info") -> None:
        """A note in the log and the lab database."""
        self._log(f"{level.upper()} {text}")
        if self.state is not None:
            self.state.add_note(self.run_id, self.clock, self.actor, level, text)
            self.state.commit()

    def wait(self, seconds: float) -> None:
        """Wait on the wall clock; a thermometer row every temperature_every_s on the way."""
        t_start, end = self.clock, self.clock + seconds
        every = STATE["temperature_every_s"]
        while self.clock < end:
            time.sleep(max(0.0, min(end, self.last_row_t + every) - self.clock))
            if self.clock - self.last_row_t >= every:
                self._thermometer_row()
        self._log(f"wait {seconds:g} s")
        self._record(t_start, "wait", {"seconds": seconds}, None)

    def get_temperature(self) -> dict:
        """Box filtered (as "kalman"), raw and target, and the room temperature (None if unknown), in C."""
        box = self._box()
        reading = {"kalman": float(box["filtered"]), "raw": float(box["temperature"]), "target": float(box["target"]),
                   "room": self.room()}
        self._log(f"get_temperature kalman {reading['kalman']:.3f} raw {reading['raw']:.3f}")
        return reading

    def get_status(self) -> dict:
        """Time, filtered temperature, target, control (None if the box does not say), duty; no map yet."""
        box = self._box()
        status = {"time_s": self.clock, "kalman": float(box["filtered"]), "target": float(box["target"]),
                  "control": box.get("control"), "duty": int(box["duty"]),
                  "map_age_s": None if self.map_time is None else self.clock - self.map_time,
                  "last_check": self.last_check, "last_eta": self.last_eta}
        self._log("get_status")
        return status

    def _post_target(self, T: float) -> None:
        """POST {"target": T} to target_path."""
        self._request(urljoin(self.box_url, self.s["target_path"]), {"target": T})

    def set_target(self, T: float) -> None:
        """New box target in C (POST to target_path); remembered for set_control(True)."""
        self._post_target(T)
        self.target = T
        self._log(f"set_target {T:g}")
        self._record(self.clock, "set_target", {"T": T}, None)

    def set_control(self, on: bool) -> None:
        """Control off: target off_target_C (the cooler stays idle), the last target is kept; on: the last target back.
        Without a known target (none set and the box already at off_target_C) switching on raises LabError."""
        if self.target is None:
            current = float(self._box()["target"])
            if current != self.s["off_target_C"]:
                self.target = current
        if on:
            if self.target is None:
                raise LabError("control on: no target to restore (the box is at the off target); set_target first")
            self._post_target(self.target)
        else:
            self._post_target(self.s["off_target_C"])
        self._log(f"set_control {'on' if on else 'off'}")
        self._record(self.clock, "set_control", {"on": on}, None)

    def open_hardware(self, slm=None, camera=None) -> None:
        """Open the SLM and camera (given ones, e.g. for a test, or HARDWARE's) and take the warm-up frames; nothing if
        they are open already. Called by the first optics call."""
        if self._slm is not None:
            return
        slm = slm or hardware.SLM(self.fake_hardware, self.hw)
        camera = camera or hardware.Camera(self.fake_hardware, slm, self.hw)
        try:
            slm.open()
            camera.open()
            hardware.warm_up(slm, camera)
        except Exception:
            camera.close()
            slm.close()
            raise
        self._slm, self._camera = slm, camera
        fake = isinstance(slm, hardware.FakeSLM)
        if self.maps_dir is None:
            self.maps_dir = self.hw["fake_maps_dir" if fake else "maps_dir"]
        self._log(f"hardware open ({'fake' if fake else 'real'} SLM and camera)")

    def close(self) -> None:
        """Close the camera and the SLM (SLM blank, then off); safe to call twice."""
        try:
            if self._camera is not None:
                self._camera.close()
        finally:
            if self._slm is not None:
                self._slm.close()
            self._slm = self._camera = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _patterns(self, n: int) -> np.ndarray:
        """n random phase patterns (n x N), quantized to the SLM's gray levels."""
        return optics.make_slm_patterns(n, self.N, self.rng, quantize=True, levels=self.hw["slm_phase_levels"])

    def _show_and_read(self, E_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each pattern (rows of E_in, n x N) on the SLM, one frame each: (outputs I (n x M), saturated share of the
        raw pixels per frame (n,))."""
        grid, levels = self.hw["slm_macro_grid"], self.hw["slm_phase_levels"]
        I, saturated = np.empty((len(E_in), self.M)), np.empty(len(E_in))
        for k, row in enumerate(E_in):
            values = optics.phase_to_gray(np.angle(row), levels).reshape(grid, grid)
            raw = hardware.measure(self._slm, self._camera, hardware.slm_frame(values, self.hw))
            I[k], saturated[k] = camera_outputs(raw, self.hw), np.mean(raw >= 255)
        return I, saturated

    def _need_map(self) -> None:
        """ValueError without a map."""
        if self.X_est is None:
            raise ValueError("no map yet: run measure_map first")

    def measure_map(self) -> dict:
        """A new map: a reference frame (light check), then L = L_ratio * N random patterns, A = sqrt(I), X_est by GGS
        (as in the simulation). Saved with its data in maps_dir. Returns n_iter, lab_time_s (wall clock),
        compute_time_s and saturated_pct."""
        t_start = self.clock
        self.open_hardware()
        reference = hardware.measure(self._slm, self._camera, hardware.reference_frame(self.hw))
        if reference.mean() < self.hw["min_light"]:
            raise hardware.HardwareError(f"no light on the camera: laser off, beam blocked or box open? (reference "
                                         f"mean {reference.mean():.1f} < {self.hw['min_light']:g})")
        E_in = self._patterns(SIM["L_ratio"] * self.N)                                  # (L x N)
        I, saturated = self._show_and_read(E_in)                                       # (L x M)
        t0 = time.perf_counter()
        X_est, info = ggs.recover(E_in, optics.amplitude(I), self.ggs_rng, max_iter=SIM["max_iter"], tol=SIM["tol"],
                                  n_restarts=SIM["n_restarts"], restart_threshold=SIM["restart_threshold"])
        compute = time.perf_counter() - t0
        self.X_est, self.map_time, self._background = X_est, t_start, None
        result = {"n_iter": int(info["n_iter"]), "lab_time_s": self.clock - t_start, "compute_time_s": compute,
                  "saturated_pct": float(100 * saturated.mean())}
        path = Path(self.maps_dir) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, E_in=E_in, I=I, X_est=X_est, reference=reference, slm_temperature_C=self._slm.temperature(),
                 lab_time_s=t_start, saved=str(datetime.now()), N=self.N, M=self.M)
        self._log(f"measure_map {result['n_iter']} iterations, {result['saturated_pct']:.1f} % saturated, {path.name}")
        if self.state is not None:
            self.state.add_map(self.run_id, t_start, "measure", None, result["lab_time_s"])
        self._record(t_start, "measure_map", {}, result)
        return result

    def check_map(self) -> float:
        """Prediction correlation of the map on n_test fresh random patterns (as in the simulation)."""
        self._need_map()
        t_start = self.clock
        self.open_hardware()
        E_test = self._patterns(SIM["n_test"])
        I, _ = self._show_and_read(E_test)
        self.last_check = float(metrics.prediction_correlation(E_test, self.X_est, I))
        self._log(f"check_map {self.last_check:.3f}")
        if self.state is not None:
            self.state.add_map(self.run_id, t_start, "check", self.last_check, self.clock - t_start)
        self._record(t_start, "check_map", {}, self.last_check)
        return self.last_check

    def focus(self, point: int) -> float:
        """Focus on camera point `point` with the map; returns eta = I[point] / background, the background being the
        mean output over focus_n_background random patterns (once per map). A saturated focus frame gives a warning
        note (eta is then too low)."""
        self._need_map()
        t_start = self.clock
        self.open_hardware()
        levels = self.hw["slm_phase_levels"]
        if self._background is None:
            I_background, _ = self._show_and_read(self._patterns(OPTICS["focus_n_background"]))
            self._background = float(I_background.mean())
        I, saturated = self._show_and_read(metrics.focus_pattern(self.X_est, point, quantize=True, levels=levels))
        self.last_eta = float(metrics.enhancement(I, point, self._background)[0])
        self._log(f"focus {point} eta {self.last_eta:.1f}")
        if saturated[0] > 0:
            self.note(f"focus {point}: {100 * saturated[0]:.1f} % of the camera pixels saturated, eta "
                      f"{self.last_eta:.1f} is a lower bound", level="warning")
        if self.state is not None:
            self.state.add_focus(self.run_id, t_start, point, self.last_eta)
        self._record(t_start, "focus", {"point": point}, self.last_eta)
        return self.last_eta


if __name__ == "__main__":
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from agents.core.safe_lab import SafeLab
    from agents.core.state import LabState

    # A stand-in for the box web API and the room server on 127.0.0.1 (no network, never the real box).
    box = {"temperature": 20.31, "filtered": 20.25, "target": 20.0, "duty": 412}
    room = {"channels": [{"name": REAL_BOX["room_channel"], "latest": {
        "temperature": 21.4, "external_temperature": 21.0, "humidity": 40.0, "at": "2026-11-25T10:00:00Z"}}]}
    requests = []

    class StandInBox(BaseHTTPRequestHandler):
        """GET /api/status and /room; POST {"target": T} to the target path changes the status; other POSTs get 404."""

        def _send(self, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(("GET", self.path, None))
            self._send(box if self.path == "/api/status" else room)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", self.path, body))
            if self.path != REAL_BOX["target_path"]:                # the box has no other write path we use
                self.send_response(404)
                self.end_headers()
                return
            box["target"] = body["target"]
            self._send({})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), StandInBox)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    state = LabState(":memory:")
    lab = RealLab(state, "real_lab_test", box_url=f"{base}/api/status", room_url=f"{base}/room")

    reading = lab.get_temperature()
    print(f"get_temperature: {reading}")
    assert reading == {"kalman": 20.25, "raw": 20.31, "target": 20.0, "room": 21.4}
    status = lab.get_status()
    assert status["duty"] == 412 and status["control"] is None and status["map_age_s"] is None
    assert lab.truth() is None, "nobody knows the true temperature of the real box"

    # Through SafeLab as the operator: a target above room - margin is sent, with a warning note.
    operator = SafeLab(lab, "operator")
    from config import LAB_LIMITS
    near_room = round(room["channels"][0]["latest"]["temperature"] - LAB_LIMITS["room_margin_C"] + 0.1, 6)
    operator.set_target(near_room)
    operator.set_control(True)
    posts = [(path, body) for kind, path, body in requests if kind == "POST"]
    print(f"POSTs: {posts}")
    assert posts == [(REAL_BOX["target_path"], {"target": near_room}), (REAL_BOX["target_path"], {"target": near_room})]
    assert lab.get_temperature()["target"] == near_room

    # Control off, then on: the box gets the off target (cooler idle), then the remembered target back.
    operator.set_control(False)
    assert lab.get_temperature()["target"] == REAL_BOX["off_target_C"]
    operator.set_control(True)
    off_on = [(path, body) for kind, path, body in requests if kind == "POST"][-2:]
    print(f"off, then on: {off_on}")
    assert off_on == [(REAL_BOX["target_path"], {"target": REAL_BOX["off_target_C"]}),
                      (REAL_BOX["target_path"], {"target": near_room})]
    assert lab.get_temperature()["target"] == near_room
    assert [a["args"] for a in state.query("SELECT args FROM actions WHERE action = 'set_control'")] == \
        ['{"on": true}', '{"on": false}', '{"on": true}']
    warnings = state.query("SELECT actor, text FROM notes WHERE level = 'warning'")
    assert len(warnings) == 1 and "can only cool" in warnings[0]["text"] and warnings[0]["actor"] == "operator"

    # Box-only use never touches the SLM or camera; the optics without a map: a clear error.
    assert lab._slm is None and lab._camera is None, "box calls must not open the hardware"
    try:
        lab.check_map()
        raise AssertionError("check_map without a map must raise")
    except ValueError as error:
        print(f"check_map without a map: {error}")

    # wait() uses the wall clock and writes thermometer rows on the way.
    t0 = lab.clock
    lab.wait(1.5)
    rows = state.query("SELECT * FROM temperatures")
    print(f"wait(1.5): {lab.clock - t0:.2f} s of wall clock, {len(rows)} thermometer rows")
    assert lab.clock - t0 >= 1.5 and len(rows) >= 2 and rows[-1]["target"] == near_room

    # Room unknown: the target vs room check is skipped with a note. Box down: LabError.
    no_room = RealLab(state, "real_lab_test_no_room", box_url=f"{base}/api/status", room_url="")
    SafeLab(no_room, "operator").set_target(25.0)
    assert state.query("SELECT text FROM notes WHERE text LIKE '%room temperature unknown%'")
    server.shutdown()
    server.server_close()
    try:
        lab.get_temperature()
        raise AssertionError("a dead box should raise LabError")
    except LabError as error:
        print(f"box down: LabError ({error})")
    assert all(url.startswith("/") for _, url, _ in requests), "only the stand-in server was asked"

    # The optics with the fake SLM and camera (any machine): measure_map, check_map, focus, maps saved; again with
    # an 8 x 8 grid; and the no-light error with an SLM that shows nothing (blank).
    import tempfile
    eta_grid_4 = None
    with tempfile.TemporaryDirectory() as tmp:
        for grid in (4, 8):
            with RealLab(state, f"real_lab_optics_{grid}", box_url=f"{base}/api/status", room_url="",
                         hardware_settings={"slm_macro_grid": grid}, fake_hardware=True, maps_dir=tmp) as optic:
                result = optic.measure_map()
                check = optic.check_map()
                eta = optic.focus(9) if grid == 4 else None
                eta_grid_4 = eta if grid == 4 else eta_grid_4
                L = SIM["L_ratio"] * optic.N
                print(f"fake optics, grid {grid}: N {optic.N}, M {optic.M}, L {L}: measure_map {result['lab_time_s']:.1f} s "
                      f"(GGS {result['compute_time_s']:.1f} s, {result['n_iter']} iterations, "
                      f"{result['saturated_pct']:.2f} % saturated), check_map {check:.3f}"
                      + ("" if eta is None else f", focus(9) eta {eta:.1f}"))
                assert check > 0.5 and (eta is None or eta > 1)
                assert optic._slm is not None
            assert optic._slm is None, "closed by the context manager"
        maps = sorted(Path(tmp).glob("*.npz"))
        saved = np.load(maps[0])
        assert len(maps) >= 1 and {"E_in", "I", "X_est", "reference", "slm_temperature_C"} <= set(saved.files)
        rows = state.query("SELECT kind, correlation, duration_s FROM maps WHERE run_id = "
                           "(SELECT id FROM runs WHERE name = 'real_lab_optics_4')")
        focus_rows = state.query("SELECT point, eta FROM focus WHERE run_id = "
                                 "(SELECT id FROM runs WHERE name = 'real_lab_optics_4')")
        assert [r["kind"] for r in rows] == ["measure", "check"] and all(r["duration_s"] > 0 for r in rows)
        assert focus_rows[0]["point"] == 9 and focus_rows[0]["eta"] == eta_grid_4

        class BlankSLM(hardware.FakeSLM):
            """An SLM that shows nothing (as with the laser off: the camera stays dark)."""
            def show(self, phase):
                super().show(np.zeros(self.shape, dtype=np.uint8))
        dark = RealLab(state, "real_lab_dark", box_url=f"{base}/api/status", room_url="", fake_hardware=True,
                       maps_dir=tmp)
        blank_slm = BlankSLM()
        dark.open_hardware(blank_slm, hardware.FakeCamera(blank_slm))
        try:
            dark.measure_map()
            raise AssertionError("a dark camera must raise")
        except hardware.HardwareError as error:
            print(f"dark camera: {error}")
            assert "no light on the camera" in str(error)
        dark.close()
    print("real_lab/lab.py self-test passed (stand-in web server and fake hardware only)")
