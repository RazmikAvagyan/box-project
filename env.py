"""
env.py - read the local .env file (API keys, lab addresses) into os.environ, standard library only.

config.py calls load_env() before it reads the lab addresses, so every module sees the .env values.
.env stays local (it is in .gitignore); .env.example lists the names without values.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"     # the project root


def load_env(path: str | Path = ENV_FILE) -> None:
    """Put KEY=value lines of the .env file into os.environ (keys already set win). No file: do nothing."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        env = Path(folder) / ".env"
        env.write_text("# test file\n\nENV_TEST_A=plain\nENV_TEST_B = \"quoted value\"\nENV_TEST_C='kept'\n")
        os.environ["ENV_TEST_C"] = "already set"
        load_env(env)
        assert os.environ["ENV_TEST_A"] == "plain" and os.environ["ENV_TEST_B"] == "quoted value"
        assert os.environ["ENV_TEST_C"] == "already set"
        load_env(Path(folder) / "missing.env")
        for name in ("ENV_TEST_A", "ENV_TEST_B", "ENV_TEST_C"):
            del os.environ[name]
    print("load_env: reads KEY=value, skips comments and empty lines, strips quotes, keeps set keys: ok")
    load_env()
    for name in ("BOX_URL", "ROOM_URL"):
        print(f"{name}: {'set' if os.environ.get(name) else 'not set (fill in .env, see .env.example)'}")
