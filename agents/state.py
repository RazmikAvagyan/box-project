"""
state.py - the shared lab database: a local SQLite file (built-in sqlite3, no internet).

LabAPI writes what a real lab would record: the thermometer (Kalman and raw, never the true
temperature), map measurements and checks, focus photos, every action and notes. Agents read
it back with query, last and since.

Tables (all rows carry run_id and the lab time t_s in seconds):
  runs:         id, name, seed, settings (JSON), started (wall-clock time)
  temperatures: run_id, t_s, kalman, raw, target, duty
  maps:         run_id, t_s, kind ("measure" or "check"), correlation, duration_s
  focus:        run_id, t_s, point, eta
  actions:      run_id, t_s, actor, action, args (JSON), result (JSON)
  notes:        run_id, t_s, actor, level ("info" or "warning"), text

Export one run as CSV files: python -m agents.state export <run name>  (writes results/export/<run name>/)
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path

from config import STATE

TABLES = ["temperatures", "maps", "focus", "actions", "notes"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, seed INTEGER,
                                 settings TEXT, started TEXT);
CREATE TABLE IF NOT EXISTS temperatures (run_id INTEGER, t_s REAL, kalman REAL, raw REAL, target REAL,
                                         duty INTEGER);
CREATE TABLE IF NOT EXISTS maps (run_id INTEGER, t_s REAL, kind TEXT, correlation REAL, duration_s REAL);
CREATE TABLE IF NOT EXISTS focus (run_id INTEGER, t_s REAL, point INTEGER, eta REAL);
CREATE TABLE IF NOT EXISTS actions (run_id INTEGER, t_s REAL, actor TEXT, action TEXT, args TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS notes (run_id INTEGER, t_s REAL, actor TEXT, level TEXT, text TEXT);
"""


class LabState:
    """The lab database in one SQLite file; rows are written by LabAPI and read by the agents."""

    def __init__(self, path: str | Path):
        """Open (or create) the database file at path and create missing tables."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def start_run(self, name: str, seed: int, settings: dict) -> int:
        """Add a run; returns its id."""
        cur = self.db.execute("INSERT INTO runs (name, seed, settings, started) VALUES (?, ?, ?, ?)",
                              (name, seed, json.dumps(settings), time.strftime("%Y-%m-%d %H:%M:%S")))
        self.db.commit()
        return cur.lastrowid

    def delete_runs(self, name: str) -> None:
        """Delete every run with this name and all its rows in the other tables, then commit."""
        ids = [(row["id"],) for row in self.query("SELECT id FROM runs WHERE name = ?", (name,))]
        for table in ["temperatures", "maps", "focus", "actions", "notes"]:
            self.db.executemany(f"DELETE FROM {table} WHERE run_id = ?", ids)
        self.db.execute("DELETE FROM runs WHERE name = ?", (name,))
        self.db.commit()

    def add_temperature(self, run_id: int, t_s: float, kalman: float, raw: float, target: float, duty: int) -> None:
        """One thermometer row (C)."""
        self.db.execute("INSERT INTO temperatures VALUES (?, ?, ?, ?, ?, ?)", (run_id, t_s, kalman, raw, target, duty))

    def add_map(self, run_id: int, t_s: float, kind: str, correlation: float | None, duration_s: float) -> None:
        """One map row: kind "measure" (correlation None) or "check" (prediction correlation)."""
        self.db.execute("INSERT INTO maps VALUES (?, ?, ?, ?, ?)", (run_id, t_s, kind, correlation, duration_s))

    def add_focus(self, run_id: int, t_s: float, point: int, eta: float) -> None:
        """One focus photo: camera point and enhancement eta."""
        self.db.execute("INSERT INTO focus VALUES (?, ?, ?, ?)", (run_id, t_s, point, eta))

    def add_action(self, run_id: int, t_s: float, actor: str, action: str, args: dict, result) -> None:
        """One call of a LabAPI function, with its arguments and result as JSON."""
        self.db.execute("INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, t_s, actor, action, json.dumps(args), json.dumps(result)))

    def add_note(self, run_id: int, t_s: float, actor: str, level: str, text: str) -> None:
        """One note: level "info" or "warning"."""
        self.db.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?)", (run_id, t_s, actor, level, text))

    def commit(self) -> None:
        """Write the pending rows to the file."""
        self.db.commit()

    def close(self) -> None:
        """Commit and close the file."""
        self.db.commit()
        self.db.close()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read query; returns the rows as dicts."""
        return [dict(row) for row in self.db.execute(sql, params)]

    def last(self, table: str, run_id: int, n: int) -> list[dict]:
        """The last n rows of a table for one run, oldest first."""
        rows = self.query(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY t_s DESC, rowid DESC LIMIT ?",
                          (run_id, n))
        return rows[::-1]

    def since(self, table: str, run_id: int, t_s: float) -> list[dict]:
        """All rows of a table for one run from lab time t_s on, oldest first."""
        return self.query(f"SELECT * FROM {table} WHERE run_id = ? AND t_s >= ? ORDER BY t_s, rowid", (run_id, t_s))


    def export_run(self, run_id: int, folder: str | Path) -> dict[str, int]:
        """Write one CSV per table with this run's rows, plus run.json (name, seed, settings, row counts).

        Returns: the row count of each table.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        counts = {}
        for table in TABLES:
            cur = self.db.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY t_s, rowid", (run_id,))
            rows = cur.fetchall()
            with open(folder / f"{table}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([c[0] for c in cur.description])
                writer.writerows(tuple(row) for row in rows)
            counts[table] = len(rows)
        run = self.query("SELECT * FROM runs WHERE id = ?", (run_id,))[0]
        info = {"id": run_id, "name": run["name"], "seed": run["seed"], "started": run["started"],
                "settings": json.loads(run["settings"]), "rows": counts}
        (folder / "run.json").write_text(json.dumps(info, indent=2))
        return counts

if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["export"]:
        # Export the newest run with this name from the lab database.
        name = sys.argv[2]
        state = LabState(STATE["db_path"])
        run_id = state.query("SELECT id FROM runs WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))[0]["id"]
        folder = Path("results/export") / name
        counts = state.export_run(run_id, folder)
        state.close()
        print(f"run {name} (id {run_id}) exported to {folder}/:")
        for table, n in counts.items():
            print(f"  {folder / (table + '.csv')}: {n} rows")
        print(f"  {folder / 'run.json'}")
    else:
        import tempfile

        from agents import baseline

        with tempfile.TemporaryDirectory() as tmp:
            state = LabState(Path(tmp) / "lab_state.sqlite")
            r = baseline.run("rules", state=state)
            summary = baseline.summarize(r)
            run_id = r["api"].run_id

            for table in ["runs", "temperatures", "maps", "focus", "actions", "notes"]:
                print(f"{table}: {state.query(f'SELECT COUNT(*) AS n FROM {table}')[0]['n']} rows")
            measures = state.query("SELECT * FROM maps WHERE run_id = ? AND kind = 'measure'", (run_id,))
            checks = state.query("SELECT * FROM maps WHERE run_id = ? AND kind = 'check'", (run_id,))
            print("measure rows of maps:")
            for row in measures:
                print(f"  {row}")
            print("warning notes:")
            for row in state.query("SELECT * FROM notes WHERE run_id = ? AND level = 'warning'", (run_id,)):
                print(f"  {row}")
            print("last 5 temperatures:")
            for row in state.last("temperatures", run_id, 5):
                print(f"  {row}")

            # The baseline counts only the maps the rules added; the first map at the start is one more row.
            print(f"measure rows {len(measures)} = measure_map_calls {summary['measure_map_calls']} + 1 (start map), "
                  f"check rows {len(checks)} = check_map_calls {summary['check_map_calls']}")
            assert len(measures) == summary["measure_map_calls"] + 1, "measure rows do not match the baseline"
            assert len(checks) == summary["check_map_calls"], "check rows do not match the baseline"
            state.close()
        print("state.py self-test passed")
