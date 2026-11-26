"""
lab.py - RealLab: the real box behind the same interface as the simulated lab (agents.core.lab_api.Lab).

The thermometer, the status, set_target and set_control talk to the box web API (REAL_BOX "url" = BOX_URL from .env,
the status JSON; targets are POSTed as {"target": T} to REAL_BOX "target_path" on the same host). The box has no on/off
endpoint: set_control(False) sets REAL_BOX "off_target_C" (above the box temperature, so the cooler stays idle, PWM 0)
and keeps the last target; set_control(True) sets that target back. The PID gains are never written. The room temperature
comes from ROOM_URL (the room channel, read at most every room_interval_s). The SLM and the camera are not connected
yet: measure_map, check_map and focus raise NotImplementedError. wait() uses the wall clock and writes a thermometer
row to the lab database every STATE temperature_every_s, as the simulated lab does. truth() is None: nobody knows the
true temperature of the real box. Box request failures raise LabError (a hardware error the team graph handles).
The self-test uses a stand-in web server on 127.0.0.1; it never contacts the box.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin

from agents.core.lab_api import Lab, LabError
from config import REAL_BOX, STATE
from real_lab.logger import SSL_CONTEXT, read_room

NOT_CONNECTED = "needs the real SLM/camera"


class RealLab(Lab):
    """The real box (web API) and room sensor; the SLM and camera methods are not connected yet."""

    def __init__(self, state=None, run_name: str = "real_lab", box_url: str | None = None, room_url: str | None = None,
                 settings: dict | None = None):
        """state: optional LabState (actions, notes and thermometer rows are recorded there under run_name);
        box_url / room_url: default REAL_BOX url / room_url (BOX_URL / ROOM_URL from .env)."""
        super().__init__()
        self.s = {**REAL_BOX, **(settings or {})}
        self.box_url = box_url or self.s["url"]
        self.room_url = room_url if room_url is not None else self.s["room_url"]
        if not self.box_url:
            raise ValueError("no box URL: set BOX_URL in .env (see .env.example)")
        self.n_points = None                       # the camera is not connected yet
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
        """Room temperature in C (read at most every room_interval_s; None if the read fails or there is no ROOM_URL)."""
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
                  "control": box.get("control"), "duty": int(box["duty"]), "map_age_s": None, "last_check": None,
                  "last_eta": None}
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

    def measure_map(self) -> dict:
        """Not connected yet."""
        raise NotImplementedError(NOT_CONNECTED)

    def check_map(self) -> float:
        """Not connected yet."""
        raise NotImplementedError(NOT_CONNECTED)

    def focus(self, point: int) -> float:
        """Not connected yet."""
        raise NotImplementedError(NOT_CONNECTED)


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

    # The SLM and camera methods are not connected yet.
    for name, args in [("measure_map", ()), ("check_map", ()), ("focus", (3,))]:
        try:
            getattr(lab, name)(*args)
            raise AssertionError(f"{name} should raise")
        except NotImplementedError as error:
            assert str(error) == NOT_CONNECTED

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
    print("real_lab/lab.py self-test passed (stand-in web server only)")
