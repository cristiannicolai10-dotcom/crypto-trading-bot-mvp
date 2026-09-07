from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

RESEARCH_DIR = BASE_DIR / "research"
EXPERIMENT_DIR = RESEARCH_DIR / "experiments"
STATE_DIR = RESEARCH_DIR / "state"
PROPOSAL_DIR = RESEARCH_DIR / "proposals"
JOB_DIR = RESEARCH_DIR / "jobs"
PENDING_DIR = JOB_DIR / "pending"
RUNNING_DIR = JOB_DIR / "running"
COMPLETED_DIR = JOB_DIR / "completed"
FAILED_DIR = JOB_DIR / "failed"
LOG_DIR = BASE_DIR / "logs" / "automation"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TELEGRAM_POLL_TIMEOUT = int(os.getenv("RESEARCH_TG_POLL_TIMEOUT", "30"))
WORKER_POLL_SECONDS = float(os.getenv("RESEARCH_WORKER_POLL_SECONDS", "2"))
MAX_TELEGRAM_FILE_MB = float(os.getenv("RESEARCH_MAX_TELEGRAM_FILE_MB", "45"))
DEFAULT_EXPERIMENT_TIMEOUT_SECONDS = int(
    os.getenv("RESEARCH_EXPERIMENT_TIMEOUT_SECONDS", "7200")
)



# Stage 2 — AI proposal / Git worktree automation.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol").strip()
OPENAI_REASONING_EFFORT = os.getenv(
    "OPENAI_REASONING_EFFORT",
    "high",
).strip()
OPENAI_TIMEOUT_SECONDS = int(
    os.getenv("OPENAI_TIMEOUT_SECONDS", "180")
)
OPENAI_MAX_OUTPUT_TOKENS = int(
    os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "32000")
)

AI_PROPOSAL_DIR = RESEARCH_DIR / "ai_proposals"
AI_JOB_DIR = RESEARCH_DIR / "ai_jobs"
AI_PENDING_DIR = AI_JOB_DIR / "pending"
AI_RUNNING_DIR = AI_JOB_DIR / "running"
AI_COMPLETED_DIR = AI_JOB_DIR / "completed"
AI_FAILED_DIR = AI_JOB_DIR / "failed"

AI_WORKTREE_ROOT = Path(
    os.getenv(
        "RESEARCH_AI_WORKTREE_ROOT",
        "/home/botadmin/research_worktrees",
    )
)
AI_WORKER_POLL_SECONDS = float(
    os.getenv("RESEARCH_AI_WORKER_POLL_SECONDS", "2")
)
AI_EXPERIMENT_TIMEOUT_SECONDS = int(
    os.getenv("RESEARCH_AI_EXPERIMENT_TIMEOUT_SECONDS", "7200")
)

def ensure_directories() -> None:
    for path in [
        EXPERIMENT_DIR,
        STATE_DIR,
        PROPOSAL_DIR,
        PENDING_DIR,
        RUNNING_DIR,
        COMPLETED_DIR,
        FAILED_DIR,
        LOG_DIR,
        AI_PROPOSAL_DIR,
        AI_PENDING_DIR,
        AI_RUNNING_DIR,
        AI_COMPLETED_DIR,
        AI_FAILED_DIR,
        AI_WORKTREE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_runtime() -> None:
    ensure_directories()

    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(
            "Missing required environment variables in "
            f"{ENV_FILE}: {', '.join(missing)}"
        )
