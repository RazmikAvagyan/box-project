"""
safe_lab.py - permissions in one place: SafeLab(lab, actor) is the only way an agent (and later the MCP server) reaches
a lab (any lab of the lab_api.Lab interface: the simulated lab or the real one).

Per role (the actor's name, or the role given):
  map_agent          wait, read, measure_map, check_map, focus
  temperature_agent  wait, read, set_control(on=True)
  operator           everything
Read (get_temperature, get_status, truth, get_log) and note are allowed for everyone. Hard limits for everyone:
a target only within LAB_LIMITS target_min_C .. target_max_C, a focus point only inside the camera (0 .. n_points - 1).
A refused call writes a warning note and raises PermissionError.

Target vs room: the box can only cool. A new target above (room - room_margin_C) is allowed but writes a warning note
(the Temperature agent then sends "target too close to room"); with the room temperature unknown the check is skipped
and a note says so.
Every call runs with the lab's actor set to this SafeLab's actor (and put back afterwards), so the lab database shows
who did it. This module imports nothing optics-specific.
"""

from __future__ import annotations

from config import LAB_LIMITS

LAB_ACTIONS = ("wait", "set_target", "set_control", "measure_map", "check_map", "focus")
PERMISSIONS = {
    "map_agent": ("wait", "measure_map", "check_map", "focus"),
    "temperature_agent": ("wait", "set_control"),         # set_control(on=True) only, see set_control
    "operator": LAB_ACTIONS,
}


def target_vs_room(T: float, room: float | None) -> str | None:
    """The warning text if target T (C) is above room - room_margin_C (the box can only cool), else None."""
    if room is not None and T > room - LAB_LIMITS["room_margin_C"]:
        return (f"target {T:g} C is above room {room:.2f} C - {LAB_LIMITS['room_margin_C']:g} C: "
                "the box can only cool, it may not reach it")
    return None


class SafeLab:
    """One agent's door to a lab: the lab's methods behind the permissions of one role and the hard limits."""

    def __init__(self, lab, actor: str, role: str | None = None):
        """lab: any lab of the lab_api.Lab interface; actor: written to the lab database; role: PERMISSIONS key
        (default: the actor's name)."""
        self._lab, self.actor, self.role = lab, actor, role or actor          # the raw lab stays private
        if self.role not in PERMISSIONS:
            raise ValueError(f"unknown role {self.role!r}, expected one of {list(PERMISSIONS)}")

    @property
    def clock(self) -> float:
        """Lab time in s."""
        return self._lab.clock

    @property
    def run_id(self):
        """The lab's run id in the lab database (None without one)."""
        return self._lab.run_id

    def _as_actor(self, call):
        """call() with the lab's actor set to this actor (put back afterwards)."""
        before, self._lab.actor = self._lab.actor, self.actor
        try:
            return call()
        finally:
            self._lab.actor = before

    def _refuse(self, name: str, args: dict, why: str) -> None:
        """Warning note and PermissionError."""
        text = f"permission: refused {name}({args}) for {self.actor}: {why}"
        self.note(text, level="warning")
        raise PermissionError(text)

    def _allow(self, name: str, args: dict) -> None:
        """Refuse if the role may not call name at all."""
        if name not in PERMISSIONS[self.role]:
            self._refuse(name, args, f"the {self.role} role may not call {name}")

    def note(self, text: str, level: str = "info") -> None:
        """A note ("info" or "warning") in the lab log and database, with this actor."""
        self._as_actor(lambda: self._lab.note(text, level=level))

    def note_as(self, actor: str, text: str, level: str = "info") -> None:
        """A note under another actor's name (e.g. "human"); no other call is possible as that actor."""
        before, self._lab.actor = self._lab.actor, actor
        try:
            self._lab.note(text, level=level)
        finally:
            self._lab.actor = before

    def get_temperature(self) -> dict:
        """Thermometer: Kalman, raw, target and room temperature (room None if unknown), in C."""
        return self._as_actor(self._lab.get_temperature)

    def get_status(self) -> dict:
        """The lab status (see the lab)."""
        return self._as_actor(self._lab.get_status)

    def truth(self) -> dict | None:
        """{"T_true": ...} on the simulated lab (for plots only), None on the real lab."""
        return self._lab.truth()

    def get_log(self) -> list[str]:
        """The lab's log lines."""
        return self._lab.get_log()

    def wait(self, seconds: float) -> None:
        """Let lab time pass."""
        self._allow("wait", {"seconds": seconds})
        return self._as_actor(lambda: self._lab.wait(seconds))

    def set_target(self, T: float) -> None:
        """New box target in C, within the hard limits; warns if it is too close to the room (allowed anyway)."""
        args = {"T": T}
        self._allow("set_target", args)
        if not LAB_LIMITS["target_min_C"] <= T <= LAB_LIMITS["target_max_C"]:
            self._refuse("set_target", args, f"target outside {LAB_LIMITS['target_min_C']:g} .. "
                                             f"{LAB_LIMITS['target_max_C']:g} C")
        room = self.get_temperature().get("room")
        if room is None:
            self.note(f"set_target {T:g} C: room temperature unknown, target vs room check skipped")
        elif target_vs_room(T, room):
            self.note(f"set_target: {target_vs_room(T, room)}", level="warning")
        return self._as_actor(lambda: self._lab.set_target(T))

    def set_control(self, on: bool) -> None:
        """Switch the PID on or off (the Temperature agent may only switch it on)."""
        args = {"on": on}
        self._allow("set_control", args)
        if self.role == "temperature_agent" and on is not True:
            self._refuse("set_control", args, "the temperature_agent role may only switch the control on")
        return self._as_actor(lambda: self._lab.set_control(on))

    def measure_map(self) -> dict:
        """A new map (see the lab)."""
        self._allow("measure_map", {})
        return self._as_actor(self._lab.measure_map)

    def check_map(self) -> float:
        """Check the stored map (see the lab)."""
        self._allow("check_map", {})
        return self._as_actor(self._lab.check_map)

    def focus(self, point: int) -> float:
        """Focus on camera point `point`, which must be inside the camera (0 .. n_points - 1)."""
        args = {"point": point}
        self._allow("focus", args)
        n = getattr(self._lab, "n_points", None)
        if not isinstance(point, int) or isinstance(point, bool) or point < 0 or (n is not None and point >= n):
            self._refuse("focus", args, f"point outside the camera (0 .. {'?' if n is None else n - 1})")
        return self._as_actor(lambda: self._lab.focus(point))


if __name__ == "__main__":
    import sys

    class StandInLab:
        """A lab that only records which method ran and as which actor (no optics, no box)."""

        def __init__(self, room: float | None = 21.0):
            self.clock, self.actor, self.run_id, self.n_points, self.room = 0.0, "operator", None, 64, room
            self.calls, self.notes = [], []

        def note(self, text, level="info"):
            self.notes.append((self.actor, level, text))

        def get_temperature(self):
            return {"kalman": 20.0, "raw": 20.01, "target": 20.0, "room": self.room}

        def get_status(self):
            return {"time_s": self.clock}

        def truth(self):
            return None

        def get_log(self):
            return []

        def __getattr__(self, name):
            if name not in LAB_ACTIONS:
                raise AttributeError(name)
            return lambda *args: self.calls.append((self.actor, name, args))

    def tried(safe: SafeLab, name: str, *args) -> bool:
        """True if the call ran, False if SafeLab refused it (PermissionError)."""
        try:
            getattr(safe, name)(*args)
            return True
        except PermissionError:
            return False

    calls = {"wait": (10,), "measure_map": (), "check_map": (), "focus": (27,), "set_target": (20.0,),
             "set_control": (True,)}
    expected = {"map_agent": {"wait", "measure_map", "check_map", "focus"},
                "temperature_agent": {"wait", "set_control"},
                "operator": set(calls)}
    for role, allowed in expected.items():
        lab = StandInLab()
        safe = SafeLab(lab, role)
        ran = {name for name, args in calls.items() if tried(safe, name, *args)}
        refused = set(calls) - ran
        print(f"{role:18} ran {sorted(ran)}, refused {sorted(refused)}")
        assert ran == allowed, (role, ran)
        assert all(actor == role for actor, _, _ in lab.calls), "every call runs as the SafeLab's actor"
        assert len([n for n in lab.notes if n[1] == "warning" and "permission: refused" in n[2]]) == len(refused)
        assert lab.actor == "operator", "the lab's own actor is put back after each call"
        assert safe.get_temperature()["kalman"] == 20.0 and safe.truth() is None

    # The Temperature agent may switch the control on, never off.
    lab = StandInLab()
    assert tried(SafeLab(lab, "temperature_agent"), "set_control", True)
    assert not tried(SafeLab(lab, "temperature_agent"), "set_control", False)

    # Hard limits for everyone: the target range and the focus point inside the camera.
    lab = StandInLab()
    operator = SafeLab(lab, "operator")
    for T, ok in [(LAB_LIMITS["target_min_C"], True), (LAB_LIMITS["target_max_C"] - 10, True),
                  (LAB_LIMITS["target_min_C"] - 0.1, False), (LAB_LIMITS["target_max_C"] + 0.1, False)]:
        assert tried(operator, "set_target", T) == ok, T
    for point, ok in [(0, True), (63, True), (64, False), (-1, False), (2.5, False), (True, False)]:
        assert tried(operator, "focus", point) == ok, point
    print(f"limits: target only {LAB_LIMITS['target_min_C']:g} .. {LAB_LIMITS['target_max_C']:g} C, focus point "
          "0 .. 63: ok")

    # Target vs room: allowed with a warning above room - margin; skipped with a note when the room is unknown.
    lab = StandInLab(room=21.0)
    operator = SafeLab(lab, "operator")
    operator.set_target(21.0 - LAB_LIMITS["room_margin_C"] - 0.1)
    assert not [n for n in lab.notes if n[1] == "warning"]
    operator.set_target(21.0 - LAB_LIMITS["room_margin_C"] + 0.1)
    warning = [n for n in lab.notes if n[1] == "warning"]
    assert len(warning) == 1 and "can only cool" in warning[0][2] and lab.calls[-1][1] == "set_target"
    lab = StandInLab(room=None)
    SafeLab(lab, "operator").set_target(25.0)
    assert lab.calls[-1][1] == "set_target" and "room temperature unknown" in lab.notes[-1][2]
    print(f"target vs room: warning note: {warning[0][2]}")
    assert "optics_simulation" not in sys.modules
    print("safe_lab.py self-test passed")
