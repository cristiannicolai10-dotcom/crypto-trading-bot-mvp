from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Allow direct execution:
#   python bot/automation/test_git_manager_porcelain.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.automation.git_manager import run_git


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        subprocess.run(
            ["git", "init", "-q"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=root,
            check=True,
        )

        (root / "base.txt").write_text(
            "base\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "base.txt"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-qm", "base"],
            cwd=root,
            check=True,
        )

        path = root / "research" / "experiments" / "x.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "print('ok')\n",
            encoding="utf-8",
        )

        subprocess.run(
            ["git", "add", "-N", "."],
            cwd=root,
            check=True,
        )

        out = run_git(
            ["status", "--porcelain"],
            root,
        )
        first = out.splitlines()[0]

        assert first[3:] == "research/experiments/x.py", first
        print("porcelain path preservation: OK")


if __name__ == "__main__":
    main()
