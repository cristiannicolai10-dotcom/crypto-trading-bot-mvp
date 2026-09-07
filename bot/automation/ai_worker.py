from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai_queue import AIJobQueue, atomic_json_write, load_json, utc_now
from .git_manager import GitManager
from .openai_agent import OpenAIResearchAgent
from .proposal_manager import ProposalManager
from .result_context import build_result_context
from .safety_validator import validate_proposal_payload
from .settings import (
    AI_EXPERIMENT_TIMEOUT_SECONDS,
    AI_WORKER_POLL_SECONDS,
    BASE_DIR,
    LOG_DIR,
    STATE_DIR,
    validate_runtime,
)
from .telegram_api import TelegramAPI


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                LOG_DIR / "ai_worker.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )


def proposal_id() -> str:
    now = datetime.now(timezone.utc)
    return (
        f"P-{now.strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:4].upper()}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def proposal_buttons(pid: str) -> str:
    return json.dumps(
        {
            "inline_keyboard": [
                [
                    {
                        "text": "🧾 VIEW DIFF",
                        "callback_data": f"ai_diff:{pid}",
                    }
                ],
                [
                    {
                        "text": "✅ APPROVE CODE",
                        "callback_data": f"ai_approve:{pid}",
                    },
                    {
                        "text": "❌ REJECT",
                        "callback_data": f"ai_reject:{pid}",
                    },
                ],
            ]
        },
        ensure_ascii=False,
    )


def completion_buttons(
    pid: str,
    has_next: bool,
) -> str:
    keyboard = [
        [
            {
                "text": "🚀 PROMOTE TO MAIN",
                "callback_data": f"ai_promote:{pid}",
            },
            {
                "text": "📌 KEEP BRANCH",
                "callback_data": f"ai_keep:{pid}",
            },
        ]
    ]

    if has_next:
        keyboard.append(
            [
                {
                    "text": "🧪 CREATE NEXT",
                    "callback_data": f"ai_next:{pid}",
                }
            ]
        )

    return json.dumps(
        {"inline_keyboard": keyboard},
        ensure_ascii=False,
    )


def verify_proposal_files(
    proposal: dict[str, Any],
    worktree: Path,
) -> None:
    hashes = proposal.get("file_hashes", {})
    if not hashes:
        raise RuntimeError(
            "Proposal has no immutable file hashes."
        )

    for rel, expected in hashes.items():
        path = worktree / rel
        if not path.exists() or not path.is_file():
            raise RuntimeError(
                f"Approved proposal file missing: {rel}"
            )

        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Proposal file changed after approval preview: {rel}"
            )


def compile_changed_python(
    worktree: Path,
    changed_files: list[str],
) -> None:
    python_bin = BASE_DIR / "venv" / "bin" / "python"
    if not python_bin.exists():
        raise RuntimeError(
            f"Project venv Python missing: {python_bin}"
        )

    py_files = [
        worktree / rel
        for rel in changed_files
        if rel.endswith(".py")
    ]

    if not py_files:
        return

    process = subprocess.run(
        [
            str(python_bin),
            "-m",
            "py_compile",
            *[str(path) for path in py_files],
        ],
        cwd=worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Python compile failed:\n"
            + process.stdout[-5000:]
        )


class AIResearchWorker:
    def __init__(self) -> None:
        self.queue = AIJobQueue()
        self.proposals = ProposalManager()
        self.git = GitManager()
        self.tg = TelegramAPI()

    def send(self, text: str, reply_markup: str | None = None) -> None:
        self.tg.send_message(
            text,
            reply_markup=reply_markup,
        )

    def generate_proposal(self, job: dict[str, Any]) -> None:
        request_text = str(job["request_text"]).strip()
        if not request_text:
            raise RuntimeError("Empty proposal request.")

        self.send("🧠 Building research proposal…")

        agent = OpenAIResearchAgent()
        plan = agent.plan(request_text)
        generated = agent.generate_code_proposal(
            request_text,
            plan,
        )
        generated = validate_proposal_payload(generated)

        pid = proposal_id()
        worktree, branch, base_commit = self.git.create_worktree(pid)

        try:
            self.git.apply_files(
                worktree,
                generated["files"],
            )
            self.git.diff_check(worktree)

            changed = self.git.changed_files(worktree)
            if not changed:
                raise RuntimeError(
                    "OpenAI proposal produced no repository changes."
                )

            diff = self.git.diff(worktree)
            if not diff.strip():
                raise RuntimeError(
                    "Proposal diff is empty."
                )

            proposal_dir = (
                self.proposals.path(pid).parent
            )
            proposal_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            diff_path = proposal_dir / "proposal.diff"
            diff_path.write_text(
                diff,
                encoding="utf-8",
            )

            file_hashes = {}
            for item in generated["files"]:
                rel = item["path"]
                file_hashes[rel] = file_sha256(
                    worktree / rel
                )

            proposal = {
                **generated,
                "proposal_id": pid,
                "request_text": request_text,
                "planner": plan,
                "status": "ready_for_approval",
                "branch": branch,
                "base_commit": base_commit,
                "worktree": str(worktree),
                "changed_files": changed,
                "diff_path": str(diff_path),
                "diff_stat": self.git.diff_stat(
                    worktree
                ),
                "file_hashes": file_hashes,
                "requested_by": job.get(
                    "requested_by"
                ),
            }

            self.proposals.create(proposal)

        except Exception:
            # Keep worktree on failure if it exists; it helps debugging.
            raise

        warning = ""
        if proposal.get(
            "requires_frozen_change"
        ):
            warning = "\n⚠️ Changes a frozen research rule."

        self.send(
            f"🧠 {pid} READY\n"
            f"{proposal['title']}\n"
            f"Files changed: {len(changed)}"
            f"{warning}\n\n"
            f"Use /diff {pid} to inspect code.",
            reply_markup=proposal_buttons(pid),
        )

        job["proposal_id"] = pid

    def execute_proposal(self, job: dict[str, Any]) -> None:
        pid = str(job["proposal_id"]).upper()
        proposal = self.proposals.get(pid)

        if proposal.get("status") not in {
            "ready_for_approval",
            "approved",
        }:
            raise RuntimeError(
                f"{pid} cannot execute from status "
                f"{proposal.get('status')!r}"
            )

        proposal = validate_proposal_payload(
            proposal
        )

        worktree = Path(proposal["worktree"])
        if not worktree.exists():
            raise RuntimeError(
                f"Proposal worktree missing: {worktree}"
            )

        verify_proposal_files(
            proposal,
            worktree,
        )

        self.proposals.update(
            pid,
            status="validating",
            approved_at=utc_now(),
            approved_by=job.get("requested_by"),
        )

        self.git.diff_check(worktree)
        compile_changed_python(
            worktree,
            proposal["changed_files"],
        )

        self.proposals.update(
            pid,
            status="committing",
        )

        commit = self.git.commit_and_push(
            worktree,
            proposal["branch"],
            proposal["title"],
        )

        self.proposals.update(
            pid,
            status="running",
            proposal_commit=commit,
        )

        self.send(
            f"🟡 {pid} RUNNING\n"
            f"{proposal['title']}"
        )

        python_bin = BASE_DIR / "venv" / "bin" / "python"
        run_script = worktree / proposal["run_script"]
        if not run_script.exists():
            raise RuntimeError(
                f"run_script does not exist: "
                f"{proposal['run_script']}"
            )

        log_path = LOG_DIR / f"{pid}.log"
        started = time.time()

        with log_path.open(
            "w",
            encoding="utf-8",
            errors="replace",
        ) as log_handle:
            process = subprocess.run(
                [
                    str(python_bin),
                    str(run_script),
                    *[
                        str(value)
                        for value in proposal.get(
                            "run_args",
                            [],
                        )
                    ],
                ],
                cwd=worktree,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=AI_EXPERIMENT_TIMEOUT_SECONDS,
                env={
                    **os.environ,
                    "RESEARCH_PROPOSAL_ID": pid,
                    "CRYPTO_BOT_SHARED_DATA_DIR": str(
                        BASE_DIR / "data"
                    ),
                },
            )

        duration = round(
            time.time() - started,
            2,
        )

        if process.returncode != 0:
            self.proposals.update(
                pid,
                status="failed",
                duration_seconds=duration,
                exit_code=int(
                    process.returncode
                ),
                log_path=str(log_path),
            )
            raise RuntimeError(
                f"Experiment returned "
                f"{process.returncode}. "
                f"Use /logs {pid}."
            )

        result_context, outputs = (
            build_result_context(
                worktree,
                proposal["output_globs"],
                log_path,
            )
        )

        if not result_context.strip():
            raise RuntimeError(
                "Experiment completed but no readable result context "
                "was found from its declared outputs/log."
            )

        self.proposals.update(
            pid,
            status="analyzing",
            duration_seconds=duration,
            exit_code=0,
            log_path=str(log_path),
            output_paths=outputs,
        )

        agent = OpenAIResearchAgent()
        analysis = agent.analyze_result(
            proposal=proposal,
            result_context=result_context,
        )

        final = self.proposals.update(
            pid,
            status="completed",
            analysis=analysis,
            completed_at=utc_now(),
        )

        runtime_state = {
            "proposal_id": pid,
            "title": proposal["title"],
            "completed_at": final.get(
                "completed_at"
            ),
            "verdict": analysis.get(
                "verdict"
            ),
            "telegram_text": analysis.get(
                "telegram_text"
            ),
            "next_request": analysis.get(
                "next_request"
            ),
            "branch": proposal["branch"],
            "proposal_commit": commit,
        }

        atomic_json_write(
            STATE_DIR / "last_ai_result.json",
            runtime_state,
        )

        verdict = analysis.get(
            "verdict",
            "NEEDS_REVIEW",
        )
        text = str(
            analysis.get(
                "telegram_text",
                "",
            )
        ).strip()

        self.send(
            f"🟢 {pid} COMPLETED\n"
            f"AI verdict: {verdict}\n\n"
            f"{text}",
            reply_markup=completion_buttons(
                pid,
                bool(
                    str(
                        analysis.get(
                            "next_request",
                            "",
                        )
                    ).strip()
                ),
            ),
        )

    def promote_proposal(self, job: dict[str, Any]) -> None:
        pid = str(job["proposal_id"]).upper()
        proposal = self.proposals.get(pid)

        if proposal.get("status") not in {
            "completed",
            "kept_branch",
        }:
            raise RuntimeError(
                f"{pid} can only be promoted after a successful run."
            )

        commit = self.git.promote_ff_only(
            pid
        )

        self.proposals.update(
            pid,
            status="promoted",
            promoted_at=utc_now(),
            main_commit=commit,
        )

        self.send(
            f"🚀 {pid} PROMOTED TO MAIN\n"
            f"Main: {commit[:12]}"
        )

    def process(self, running_path: Path) -> None:
        job = load_json(running_path)
        job["status"] = "running"
        job["started_at"] = utc_now()
        atomic_json_write(
            running_path,
            job,
        )

        try:
            kind = job["kind"]

            if kind == "generate_proposal":
                self.generate_proposal(job)
            elif kind == "execute_proposal":
                self.execute_proposal(job)
            elif kind == "promote_proposal":
                self.promote_proposal(job)
            else:
                raise RuntimeError(
                    f"Unknown AI job kind: {kind}"
                )

            job["status"] = "completed"
            job["finished_at"] = utc_now()
            self.queue.finish(
                running_path,
                job,
                success=True,
            )

        except subprocess.TimeoutExpired:
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["error"] = (
                "Experiment timed out after "
                f"{AI_EXPERIMENT_TIMEOUT_SECONDS} seconds."
            )
            self.queue.finish(
                running_path,
                job,
                success=False,
            )
            self.send(
                "🔴 AI research job failed\n"
                + job["error"]
            )

        except Exception as exc:
            logging.exception(
                "AI research job failed"
            )
            job["status"] = "failed"
            job["finished_at"] = utc_now()
            job["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            self.queue.finish(
                running_path,
                job,
                success=False,
            )
            self.send(
                "🔴 AI research job failed\n"
                + job["error"]
            )

    def run(self) -> None:
        self.send(
            "AI research worker is online."
        )

        while True:
            try:
                path = self.queue.claim()
                if path is None:
                    time.sleep(
                        AI_WORKER_POLL_SECONDS
                    )
                    continue

                self.process(path)

            except KeyboardInterrupt:
                raise
            except Exception:
                logging.exception(
                    "AI worker loop error"
                )
                time.sleep(5)


def main() -> None:
    validate_runtime()
    setup_logging()
    AIResearchWorker().run()


if __name__ == "__main__":
    main()
