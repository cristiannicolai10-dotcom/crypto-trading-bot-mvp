from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .settings import BASE_DIR


ALLOWED_PREFIXES = (
    "bot/backtest/",
    "bot/analysis/",
    "bot/data_engine/",
    "research/experiments/",
)

FORBIDDEN_PATH_PARTS = (
    ".env",
    ".git",
    "bot/automation/",
    "bot/execution/",
    "deploy/",
    "venv/",
    "secrets",
    "credentials",
)

FORBIDDEN_CODE_PATTERNS = (
    r"\bos\.system\s*\(",
    r"\bsubprocess\.",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bparamiko\b",
    r"\bsocket\b",
    r"\bsmtplib\b",
    r"\bftplib\b",
    r"\bpickle\.loads\b",
    r"\bmarshal\.loads\b",
    r"\brequests\.(post|put|patch|delete)\s*\(",
    r"\bhttpx\.(post|put|patch|delete)\s*\(",
    r"\bOPENAI_API_KEY\b",
    r"\bTELEGRAM_TOKEN\b",
    r"\bSUPABASE_KEY\b",
    r"\bBYBIT_SECRET\b",
    r"\bplace_order\b",
    r"/v5/order/create",
    r"\bcreate_order\b",
    r"\bwithdraw\b",
    r"\bset_leverage\b",
)


def normalize_repo_path(value: str) -> str:
    raw = str(value).strip().replace("\\", "/")
    if not raw or raw.startswith("/"):
        raise ValueError(f"Absolute/empty path not allowed: {value!r}")

    parts = Path(raw).parts
    if ".." in parts:
        raise ValueError(f"Path traversal not allowed: {value!r}")

    normalized = "/".join(parts)

    if any(part in normalized for part in FORBIDDEN_PATH_PARTS):
        raise ValueError(f"Forbidden path: {normalized}")

    if not normalized.startswith(ALLOWED_PREFIXES):
        raise ValueError(
            f"Path outside AI allowlist: {normalized}. "
            f"Allowed prefixes: {ALLOWED_PREFIXES}"
        )

    return normalized


def validate_output_glob(value: str) -> str:
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        raise ValueError("Empty output glob.")
    if raw.startswith("/") or ".." in Path(raw).parts:
        raise ValueError(f"Unsafe output glob: {raw}")
    if not (
        raw.startswith("reports/")
        or raw.startswith("research/results/")
    ):
        raise ValueError(
            f"Output glob must be under reports/ or research/results/: {raw}"
        )
    return raw


def validate_proposal_payload(proposal: dict[str, Any]) -> dict[str, Any]:
    files = proposal.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("Proposal must contain at least one file.")
    if len(files) > 10:
        raise ValueError("Proposal may change at most 10 files.")

    total_bytes = 0
    seen = set()

    for item in files:
        path = normalize_repo_path(item["path"])
        item["path"] = path

        if path in seen:
            raise ValueError(f"Duplicate proposed path: {path}")
        seen.add(path)

        content = str(item.get("content", ""))
        total_bytes += len(content.encode("utf-8"))

        if len(content.encode("utf-8")) > 600_000:
            raise ValueError(f"Proposed file is too large: {path}")

        if path.endswith(".py"):
            for pattern in FORBIDDEN_CODE_PATTERNS:
                if re.search(pattern, content, flags=re.IGNORECASE):
                    raise ValueError(
                        f"Forbidden code pattern {pattern!r} in {path}"
                    )

    if total_bytes > 1_500_000:
        raise ValueError("Total proposed code exceeds 1.5 MB.")

    run_script = normalize_repo_path(proposal["run_script"])
    if not run_script.endswith(".py"):
        raise ValueError("run_script must be a Python file.")
    proposal["run_script"] = run_script

    args = proposal.get("run_args", [])
    if not isinstance(args, list):
        raise ValueError("run_args must be a list.")
    if len(args) > 30:
        raise ValueError("Too many run_args.")
    for arg in args:
        value = str(arg)
        if "\n" in value or "\r" in value or len(value) > 500:
            raise ValueError("Unsafe run argument.")

    output_globs = proposal.get("output_globs", [])
    if not isinstance(output_globs, list) or not output_globs:
        raise ValueError("At least one output_glob is required.")
    proposal["output_globs"] = [
        validate_output_glob(value)
        for value in output_globs
    ]

    return proposal
