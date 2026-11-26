"""
server.py - the MCP server of the lab: an MCP client (Claude Desktop, Claude Code, ...) can read the lab and act on it.

Built with the official Python MCP SDK (MCPServer, called FastMCP before SDK 2.x), stdio transport.
One lab lives for the whole server process: MCP["lab"] = "sim" (the simulated lab, its time moves only with the
actions and wait; its PID from MCP["pid"]) or "real" (real_lab.lab.RealLab, needs BOX_URL). The server holds only a SafeLab(lab, "mcp",
MCP["role"]) and calls only SafeLab's methods: the permissions, hard limits and target vs room warning of
agents/core/safe_lab.py apply to every call, and the raw lab is never reachable.

Lab tools: get_temperature, get_status, get_log, wait (at most MCP max_wait_s), set_target, set_control, measure_map,
check_map, focus. HUMAN["mode"] applies as for the agents: "approve_big" / "suggest" ask y/n on the terminal the server
runs in (/dev/tty; no terminal or no answer in HUMAN timeout_s = refused); "off" adds no ask (the MCP client's own
approval prompt is the human check).
Read tools (lab database, read-only): list_runs, run_summary, findings, messages; the Scientist report is the resource
lab://scientist-report. A refusal or failure comes back as a clear tool error text, never a crash. Every tool call
writes a note with the actor "mcp" (tool, arguments, result or error) to the server's run in the lab database.

python -m mcp_server.server          the server (stdio; started by the MCP client)
python -m mcp_server.server --test   self-test: every tool in-process on the simulated lab, in-memory database
"""

from __future__ import annotations

import json
import select
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agents.core.human import Human
from agents.core.safe_lab import SafeLab, target_vs_room
from agents.core.state import LabState
from config import HUMAN, STATE, THERMOSTAT
from config import MCP as SETTINGS

# The MCP client may start the server in any folder: the lab database and the report are found from the project root.
ROOT = Path(__file__).resolve().parent.parent


def plain(value):
    """value as plain JSON types (numpy numbers become floats)."""
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def ask_tty(proposal: dict, timeout_s: float = HUMAN["timeout_s"]) -> bool:
    """Ask y/n on the server's terminal (/dev/tty, not stdin: stdin and stdout carry the MCP messages).
    No terminal, no answer within timeout_s, or anything but "y..." = refused."""
    try:
        tty = open("/dev/tty", "r+")
    except OSError:
        return False
    with tty:
        tty.write(f"\n[MCP client] wants {proposal['action']}({proposal['args']}): {proposal['reason']}\n"
                  f"approve? y/n (no answer in {timeout_s:g} s = refused): ")
        tty.flush()
        ready, _, _ = select.select([tty], [], [], timeout_s)
        return bool(ready) and tty.readline().strip().lower().startswith("y")


def make_lab(state: LabState):
    """The lab of MCP["lab"], recording to state under MCP run_name (the caller wraps it in SafeLab at once)."""
    if SETTINGS["lab"] == "real":
        from real_lab.lab import RealLab
        return RealLab(state, SETTINGS["run_name"])
    from agents.core.lab_api import SimLab
    from box_simulation.thermostat import pid_gains
    return SimLab(seed=SETTINGS["seed"], state=state, run_name=SETTINGS["run_name"],
                  box_settings=pid_gains(SETTINGS["pid"]) or None)


def build(door: SafeLab, state: LabState, human: Human | None = None, report: str | Path = ROOT / SETTINGS["report"],
          max_wait_s: float = SETTINGS["max_wait_s"]) -> MCPServer:
    """The MCP server over one SafeLab (door) and the lab database (state, read-only for the read tools). The tools
    are async so they run on the server's own thread (SQLite connections may not change threads; one client)."""
    server = MCPServer("box-lab", instructions=(
        "A temperature-controlled box with a scattering sample, an SLM and a camera (simulated unless configured "
        "otherwise). Read the thermometer and status, wait, set the box target or control, measure/check the "
        "transmission map and focus; read past runs, their analyst findings, the Temperature agent's messages and "
        "the Scientist report. Every action goes through the lab's permissions and limits."))
    human = human or Human(approver=ask_tty)

    def note(tool: str, args: dict, text: str, level: str = "info") -> None:
        """The note of one tool call (actor "mcp")."""
        door.note(f"tool {tool}({args}): {text[:SETTINGS['note_max_chars']]}", level=level)

    def call(tool: str, args: dict, run):
        """Run one tool body: the result as plain JSON with a note, or a ToolError with a clear text and a note."""
        try:
            result = plain(run())
        except ToolError:
            raise
        except PermissionError as error:
            note(tool, args, f"refused: {error}", "warning")
            raise ToolError(f"refused: {error}") from error
        except Exception as error:
            note(tool, args, f"failed: {type(error).__name__}: {error}", "warning")
            raise ToolError(f"failed: {type(error).__name__}: {error}") from error
        note(tool, args, f"result {json.dumps(result)}")
        return result

    def act(name: str, args: dict, run):
        """A lab action: the human's approval first (HUMAN mode), then SafeLab."""
        if not human.review(door, "mcp client", [(name, args)], "requested by the MCP client"):
            raise PermissionError(f"{name} refused by the human (HUMAN mode {human.mode!r})")
        return run()

    def run_id(run_name: str) -> int:
        """The latest run with this name (ValueError if there is none)."""
        rows = state.query("SELECT MAX(id) AS id FROM runs WHERE name = ?", (run_name,))
        if rows[0]["id"] is None:
            raise ValueError(f"no run named {run_name!r} (see list_runs)")
        return rows[0]["id"]

    @server.tool()
    async def get_temperature() -> dict:
        """Box thermometer in °C: Kalman estimate, raw reading, target, and the room temperature (null if unknown)."""
        return call("get_temperature", {}, door.get_temperature)

    @server.tool()
    async def get_status() -> dict:
        """Lab time (s), Kalman temperature, target, control on/off, cooler duty, map age (s), last map check, last
        focus brightness eta."""
        return call("get_status", {}, door.get_status)

    @server.tool()
    async def get_log(last_n: int = 20) -> dict:
        """The last last_n lines of the lab log ("t=<s>s <what happened>")."""
        return call("get_log", {"last_n": last_n}, lambda: {"lines": door.get_log()[-max(0, last_n):]})

    @server.tool()
    async def wait(seconds: float) -> dict:
        """Let lab time pass (0 < seconds <= the server's max_wait_s); returns the lab time afterwards."""
        args = {"seconds": seconds}

        def run():
            if not 0 < seconds <= max_wait_s:
                raise PermissionError(f"wait {seconds:g} s is outside 0 .. {max_wait_s:g} s (MCP max_wait_s)")
            act("wait", args, lambda: door.wait(seconds))
            return {"time_s": door.clock}
        return call("wait", args, run)

    @server.tool()
    async def set_target(T: float) -> dict:
        """New box target in °C (only within the lab limits; the box can only cool, so a target close to or above
        the room temperature is allowed with a warning)."""
        args = {"T": T}

        def run():
            act("set_target", args, lambda: door.set_target(T))
            warning = target_vs_room(T, door.get_temperature().get("room"))
            return {"target": T, **({"warning": warning} if warning else {})}
        return call("set_target", args, run)

    @server.tool()
    async def set_control(on: bool) -> dict:
        """Switch the box temperature control (PID) on or off."""
        args = {"on": on}
        return call("set_control", args, lambda: act("set_control", args, lambda: door.set_control(on)) or {"control": on})

    @server.tool()
    async def measure_map() -> dict:
        """Measure a new transmission map of the sample (takes lab time); returns GGS iterations and times."""
        return call("measure_map", {}, lambda: act("measure_map", {}, door.measure_map))

    @server.tool()
    async def check_map() -> dict:
        """Check the stored map on the current sample: prediction correlation (about 0.86 fresh, lower when stale)."""
        return call("check_map", {}, lambda: {"check": act("check_map", {}, door.check_map)})

    @server.tool()
    async def focus(point: int) -> dict:
        """Focus the light on camera point `point` with the stored map; returns the enhancement eta."""
        args = {"point": point}
        return call("focus", args, lambda: {"eta": act("focus", args, lambda: door.focus(point))})

    @server.tool()
    async def list_runs() -> dict:
        """Every run in the lab database: id, name, seed, start time."""
        return call("list_runs", {}, lambda: {"runs": state.query("SELECT id, name, seed, started FROM runs ORDER BY id")})

    @server.tool()
    async def run_summary(run_name: str) -> dict:
        """Summary of the latest run with this name: mean focus brightness eta, lab time on maps (s), share of
        thermometer readings (Kalman) within the ± limit of the target, number of new maps and map checks."""
        def run():
            rid = run_id(run_name)
            q = lambda sql: state.query(sql, (rid,))[0]
            limit = THERMOSTAT["limit_C"]
            temps = state.query(f"SELECT AVG(ABS(kalman - target) <= {limit}) AS share, COUNT(*) AS n, MAX(t_s) AS end_s "
                                "FROM temperatures WHERE run_id = ?", (rid,))[0]
            return {"run": run_name, "run_id": rid, "mean_eta": q("SELECT AVG(eta) AS v FROM focus WHERE run_id = ?")["v"],
                    "focus_photos": q("SELECT COUNT(*) AS v FROM focus WHERE run_id = ?")["v"],
                    "lab_time_on_maps_s": q("SELECT COALESCE(SUM(duration_s), 0) AS v FROM maps WHERE run_id = ?")["v"],
                    f"share_within_{limit:g}_C": temps["share"], "thermometer_rows": temps["n"], "end_s": temps["end_s"],
                    "new_maps": q("SELECT COUNT(*) AS v FROM maps WHERE run_id = ? AND kind = 'measure'")["v"],
                    "map_checks": q("SELECT COUNT(*) AS v FROM maps WHERE run_id = ? AND kind = 'check'")["v"]}
        return call("run_summary", {"run_name": run_name}, run)

    @server.tool()
    async def findings(run_name: str) -> dict:
        """The data analyst's findings of the latest run with this name (kind, text, value, unit)."""
        return call("findings", {"run_name": run_name}, lambda: {"findings": [
            {k: f[k] for k in ("kind", "text", "value", "unit")} for f in state.findings(run_id(run_name))]})

    @server.tool()
    async def messages(run_name: str) -> dict:
        """The Temperature agent's messages of the latest run with this name (time, sender, receiver, topic, text)."""
        return call("messages", {"run_name": run_name}, lambda: {"messages": state.query(
            "SELECT t_s, sender, receiver, topic, text FROM messages WHERE run_id = ? ORDER BY t_s, rowid",
            (run_id(run_name),))})

    @server.resource("lab://scientist-report", name="Scientist report", mime_type="text/markdown",
                     description="The Scientist's report (results/report/report.md): every claim checked against "
                                 "computed numbers.")
    async def scientist_report() -> str:
        """The report text (or a short note if it was not written yet)."""
        path = Path(report)
        text = path.read_text() if path.exists() else f"No Scientist report yet ({path}); run python -m agents.reporting.scientist."
        note("scientist_report", {}, f"read {len(text)} characters")
        return text

    return server


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--test"]:
        lab_state = LabState(ROOT / STATE["db_path"])
        build(SafeLab(make_lab(lab_state), "mcp", SETTINGS["role"]), lab_state).run("stdio")
        sys.exit(0)

    import asyncio
    import socket
    import tempfile

    from agents.core.lab_api import Lab, SimLab
    from config import LAB_LIMITS

    # No network in the test: every socket connection fails loudly.
    def no_network(*args, **kwargs):
        raise AssertionError("the MCP self-test must not use the network")
    socket.socket.connect = socket.socket.connect_ex = no_network
    socket.create_connection = socket.getaddrinfo = no_network

    state = LabState(":memory:")
    other = state.start_run("team_test", 0, {})
    state.add_message(other, 3720.0, "temperature_agent", "map_agent", "cooler problem", "kalman 20.40 C, target 20 C")
    state.add_finding(other, {"kind": "time within limit", "text": "The box was within ±0.03 °C for 68 % of the time.",
                              "value": 0.68, "unit": "share", "evidence": {}})
    state.commit()
    lab = SimLab(seed=0, state=state, run_name=SETTINGS["run_name"])
    door = SafeLab(lab, "mcp", "operator")
    tmp = tempfile.TemporaryDirectory()
    report = Path(tmp.name) / "report.md"
    report.write_text("# Scientist report\n\ntest\n")
    server = build(door, state, Human("off"), report)
    calls = 0

    async def call(name: str, args: dict | None = None):
        """One tool call in-process: (ok, result dict or error text)."""
        global calls
        calls += 1
        try:
            result = await server.call_tool(name, args or {})
        except ToolError as error:
            return False, str(error)
        return not result.is_error, (result.structured_content if result.structured_content is not None
                                     else json.loads(result.content[0].text))

    async def main():
        tools = sorted(t.name for t in await server.list_tools())
        print(f"tools: {tools}")
        assert tools == sorted(["get_temperature", "get_status", "get_log", "wait", "set_target", "set_control",
                                "measure_map", "check_map", "focus", "list_runs", "run_summary", "findings", "messages"])
        ok, t = await call("get_temperature")
        ok2, status = await call("get_status")
        print(f"get_temperature: {t}")
        assert ok and ok2 and {"kalman", "raw", "target", "room"} <= set(t) and status["time_s"] == 0
        ok, waited = await call("wait", {"seconds": 120})
        print(f"wait(120): {waited}")
        assert ok and waited["time_s"] == 120 and (await call("get_status"))[1]["time_s"] == 120
        ok, m = await call("measure_map")
        ok2, c = await call("check_map")
        ok3, f = await call("focus", {"point": 27})
        print(f"measure_map {ok}, check_map {c}, focus(27) {f}")
        assert ok and ok2 and ok3 and c["check"] > 0.8 and f["eta"] > 10
        ok, log = await call("get_log", {"last_n": 3})
        assert ok and len(log["lines"]) == 3 and "focus" in log["lines"][-1]
        assert (await call("set_control", {"on": True}))[0]

        # Refusals come back as clear tool errors, not crashes.
        for name, args, expect in [
            ("set_target", {"T": LAB_LIMITS["target_max_C"] + 10}, "outside"),
            ("wait", {"seconds": SETTINGS["max_wait_s"] + 1}, "max_wait_s"),
            ("focus", {"point": 999}, "outside the camera"),
        ]:
            ok, text = await call(name, args)
            print(f"{name}({args}): {text}")
            assert not ok and "refused" in text and expect in text
        room = t["room"]
        ok, result = await call("set_target", {"T": room - LAB_LIMITS["room_margin_C"] + 0.5})
        print(f"set_target near the room ({room:g} C): {result}")
        assert ok and "can only cool" in result["warning"]

        # Read tools.
        ok, runs = await call("list_runs")
        assert ok and [r["name"] for r in runs["runs"]] == ["team_test", SETTINGS["run_name"]]
        ok, summary = await call("run_summary", {"run_name": SETTINGS["run_name"]})
        print(f"run_summary: {summary}")
        assert ok and summary["new_maps"] == 1 and summary["map_checks"] == 1 and summary["focus_photos"] == 1
        ok, found = await call("findings", {"run_name": "team_test"})
        ok2, msgs = await call("messages", {"run_name": "team_test"})
        assert ok and ok2 and found["findings"][0]["value"] == 0.68 and msgs["messages"][0]["topic"] == "cooler problem"
        ok, text = await call("run_summary", {"run_name": "no_such_run"})
        assert not ok and "no run named" in text
        contents = list(await server.read_resource("lab://scientist-report"))
        assert "Scientist report" in contents[0].content

        # Every tool call wrote one note with the actor "mcp" (plus the resource read).
        notes = state.query("SELECT text, level FROM notes WHERE actor = 'mcp' AND text LIKE 'tool %'")
        print(f"tool calls {calls}, notes by mcp {len(notes)} ({sum(n['level'] == 'warning' for n in notes)} warnings)")
        assert len(notes) == calls + 1

        # Another role: a map_agent client may not set the target.
        map_server = build(SafeLab(lab, "mcp", "map_agent"), state, Human("off"), report)
        try:
            await map_server.call_tool("set_target", {"T": 20.0})
            raise AssertionError("map_agent role must refuse set_target")
        except ToolError as error:
            print(f"map_agent role, set_target: {error}")
            assert "may not call set_target" in str(error)

        # The human in the loop: approve_big with a refuse-all approver refuses measure_map, lets wait through.
        human_server = build(door, state, Human("approve_big", lambda proposal: False), report)
        assert (await human_server.call_tool("wait", {"seconds": 10})).is_error is False
        try:
            await human_server.call_tool("measure_map", {})
            raise AssertionError("approve_big must ask before measure_map")
        except ToolError as error:
            print(f"approve_big, refuse all, measure_map: {error}")
            assert "refused by the human" in str(error)

        # The raw lab is not reachable: SafeLab keeps it private, no tool returns it, and no tool function holds it.
        assert not hasattr(door, "lab")
        for tool in server._tool_manager.list_tools():
            cells = [cell.cell_contents for cell in (tool.fn.__closure__ or ())]
            assert not any(isinstance(value, Lab) for value in cells), tool.name

    asyncio.run(main())
    tmp.cleanup()
    print("server.py self-test passed (simulated lab, in-memory database, no network)")
