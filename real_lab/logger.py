"""
logger.py - read-only logger for the real EDO thermostat box and the lab room.

Every interval_s it GETs the box status JSON and appends one row to
results/real_box/test_<date>_<time>.csv (flushed at once). A background thread GETs the room
readings every room_interval_s; their last values are repeated in the rows between reads and left
empty after a failed read. Only GET requests are sent, so no box setting is ever changed.
Stop with Ctrl+C.
"""

from __future__ import annotations

import csv
import json
import ssl
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from config import REAL_BOX as SETTINGS

BOX_FIELDS = ["temperature", "filtered", "target", "duty", "humidity", "pressure", "kp", "ki", "kd",
              "at_active", "at_status"]
ROOM_FIELDS = ["room_temp", "room_probe_temp", "room_humidity", "room_sample_time"]
COLUMNS = ["time_s", "clock_time", *BOX_FIELDS, "power_pct", *ROOM_FIELDS]
SSL_CONTEXT = ssl.create_default_context(cafile=SETTINGS["ca_file"])


def get_json(url: str, timeout: float) -> dict:
    """GET url and return the parsed JSON."""
    request = urllib.request.Request(url, headers={"User-Agent": SETTINGS["user_agent"]})
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        return json.load(response)


def read_room(url: str, channel: str, timeout: float) -> dict:
    """Latest sample of the named room channel as the ROOM_FIELDS dict; sample time in local clock time."""
    latest = next(c for c in get_json(url, timeout)["channels"] if c["name"] == channel)["latest"]
    at = datetime.fromisoformat(latest["at"].replace("Z", "+00:00")).astimezone()
    return {"room_temp": latest["temperature"], "room_probe_temp": latest["external_temperature"],
            "room_humidity": latest["humidity"], "room_sample_time": at.strftime("%Y-%m-%d %H:%M:%S")}


def room_worker(room: dict, stop: threading.Event, s: dict) -> None:
    """Refresh the shared dict room every room_interval_s until stop is set; empty values after a failed read."""
    while not stop.is_set():
        try:
            room.update(read_room(s["room_url"], s["room_channel"], s["room_timeout_s"]))
        except Exception as error:
            room.update(dict.fromkeys(ROOM_FIELDS, ""))
            print(f"warning: room read failed ({error})")
        stop.wait(s["room_interval_s"])


def run(s: dict = SETTINGS) -> Path:
    """Log until Ctrl+C; returns the CSV path."""
    start_clock = datetime.now()
    path = Path(s["out_dir"]) / f"test_{start_clock:%Y-%m-%d_%H-%M-%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    room = dict.fromkeys(ROOM_FIELDS, "")
    stop = threading.Event()
    threading.Thread(target=room_worker, args=(room, stop, s), daemon=True).start()
    print(f"logging to {path} (Ctrl+C to stop)")

    start = time.monotonic()
    tick, rows, last_target, last_print = 0, 0, None, -float("inf")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        f.flush()
        try:
            while True:
                # Fixed time grid, so a slow request does not shift the later rows.
                time.sleep(max(0.0, start + tick * s["interval_s"] - time.monotonic()))
                tick += 1
                try:
                    box = get_json(s["url"], s["timeout_s"])
                except Exception as error:
                    print(f"warning: box read failed ({error}), row skipped")
                    continue
                t_s = time.monotonic() - start
                clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                power_pct = 100.0 * box["duty"] / s["duty_max"]
                writer.writerow([f"{t_s:.2f}", clock, *[box.get(k, "") for k in BOX_FIELDS], f"{power_pct:.1f}",
                                 *[room[k] for k in ROOM_FIELDS]])
                f.flush()
                rows += 1
                if last_target is not None and box["target"] != last_target:
                    print(f"{clock}  target changed {last_target:.2f} -> {box['target']:.2f} °C")
                last_target = box["target"]
                if t_s - last_print >= s["print_every_s"]:
                    last_print = t_s
                    print(f"{clock}  T {box['filtered']:.2f} °C  target {box['target']:.2f} °C  "
                          f"power {power_pct:5.1f} %  room {room['room_temp'] or '--'} °C")
        except KeyboardInterrupt:
            stop.set()
            print(f"\nstopped: {rows} rows in {path}")
    return path


if __name__ == "__main__":
    if not SETTINGS["url"] or not SETTINGS["room_url"]:
        raise SystemExit("set BOX_URL and ROOM_URL in .env (see .env.example)")
    run()
