from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .settings import (
    AI_WORKTREE_ROOT,
    BASE_DIR,
)


class GitError(RuntimeError):
    pass


def run_git(
    args: list[str],
    cwd: Path,
    check: bool = True,
) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout.strip()
    if check and process.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({process.returncode}):\n{output}"
        )
    return output


class GitManager:
    def ensure_repo(self) -> None:
        if not (BASE_DIR / ".git").exists():
            raise GitError(f"Not a git repository: {BASE_DIR}")

    def current_commit(self) -> str:
        self.ensure_repo()
        return run_git(["rev-parse", "HEAD"], BASE_DIR)

    def current_branch(self) -> str:
        return run_git(["branch", "--show-current"], BASE_DIR)

    def main_is_clean(self) -> bool:
        return not bool(run_git(["status", "--porcelain"], BASE_DIR))

    def worktree_path(self, proposal_id: str) -> Path:
        return AI_WORKTREE_ROOT / proposal_id.upper()

    def branch_name(self, proposal_id: str) -> str:
        return f"research/{proposal_id.lower()}"

    def create_worktree(self, proposal_id: str) -> tuple[Path, str, str]:
        self.ensure_repo()

        if self.current_branch() != "main":
            raise GitError(
                "Main repository must be on branch 'main' before creating AI proposals."
            )

        if not self.main_is_clean():
            raise GitError(
                "Main repository is not clean. Commit/push Stage 2 and any local changes first."
            )

        AI_WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)

        proposal_id = proposal_id.upper()
        path = self.worktree_path(proposal_id)
        branch = self.branch_name(proposal_id)
        base_commit = self.current_commit()

        if path.exists():
            raise GitError(f"Worktree already exists: {path}")

        existing = run_git(
            ["branch", "--list", branch],
            BASE_DIR,
            check=False,
        )
        if existing:
            raise GitError(f"Branch already exists: {branch}")

        run_git(
            [
                "worktree",
                "add",
                "-b",
                branch,
                str(path),
                base_commit,
            ],
            BASE_DIR,
        )

        return path, branch, base_commit

    def apply_files(
        self,
        worktree: Path,
        files: list[dict],
    ) -> None:
        root = worktree.resolve()

        for item in files:
            relative = Path(item["path"])
            target = (worktree / relative).resolve()

            try:
                target.relative_to(root)
            except ValueError:
                raise GitError(f"Unsafe target outside worktree: {target}")

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and target.is_symlink():
                raise GitError(f"Refusing to overwrite symlink: {target}")

            target.write_text(
                str(item["content"]),
                encoding="utf-8",
            )

        # Make new untracked files visible in git diff without staging
        # their contents for commit yet.
        run_git(
            ["add", "-N", "--", "."],
            worktree,
        )

    def diff(self, worktree: Path) -> str:
        return run_git(
            ["diff", "--no-ext-diff", "--"],
            worktree,
            check=False,
        )

    def diff_stat(self, worktree: Path) -> str:
        return run_git(
            ["diff", "--stat", "--"],
            worktree,
            check=False,
        )

    def diff_check(self, worktree: Path) -> None:
        run_git(["diff", "--check", "--"], worktree)

    def changed_files(self, worktree: Path) -> list[str]:
        output = run_git(
            ["status", "--porcelain"],
            worktree,
            check=False,
        )
        result = []
        for line in output.splitlines():
            if not line.strip():
                continue
            value = line[3:].strip()
            if " -> " in value:
                value = value.split(" -> ", 1)[1]
            result.append(value)
        return result

    def commit_and_push(
        self,
        worktree: Path,
        branch: str,
        title: str,
    ) -> str:
        run_git(["add", "--all"], worktree)

        staged = run_git(
            ["diff", "--cached", "--name-only"],
            worktree,
        )
        if not staged:
            raise GitError("No staged changes to commit.")

        message = f"Research proposal: {title}"[:200]
        run_git(["commit", "-m", message], worktree)

        commit = run_git(["rev-parse", "HEAD"], worktree)
        run_git(
            ["push", "-u", "origin", branch],
            worktree,
        )
        return commit

    def promote_ff_only(self, proposal_id: str) -> str:
        self.ensure_repo()

        if self.current_branch() != "main":
            raise GitError(
                "Main worktree must be on branch 'main' before promotion."
            )

        if not self.main_is_clean():
            raise GitError(
                "Main worktree is not clean. Commit/stash local changes first."
            )

        branch = self.branch_name(proposal_id)

        run_git(["fetch", "origin"], BASE_DIR)
        run_git(["pull", "--ff-only", "origin", "main"], BASE_DIR)
        run_git(["merge", "--ff-only", branch], BASE_DIR)
        run_git(["push", "origin", "main"], BASE_DIR)

        return self.current_commit()
