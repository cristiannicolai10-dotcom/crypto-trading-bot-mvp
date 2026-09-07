from __future__ import annotations

import csv
import glob
from pathlib import Path


MAX_CONTEXT_CHARS = 180_000
MAX_FILE_CHARS = 70_000


def tail_text(path: Path, max_chars: int = 20_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    return text[-max_chars:]


def csv_context(path: Path) -> str:
    size = path.stat().st_size

    if size <= 1_000_000:
        text = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )
        return text[:MAX_FILE_CHARS]

    rows = []
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            rows.append(",".join(row))
            if idx >= 80:
                break

    return "\n".join(rows)[:MAX_FILE_CHARS]


def build_result_context(
    worktree: Path,
    output_globs: list[str],
    log_path: Path,
) -> tuple[str, list[str]]:
    parts = []
    found_files: list[Path] = []

    for pattern in output_globs:
        for match in glob.glob(str(worktree / pattern)):
            path = Path(match)
            if path.is_file():
                found_files.append(path)

    found_files = sorted(
        set(found_files),
        key=lambda p: p.stat().st_mtime,
    )

    for path in found_files[-12:]:
        rel = path.relative_to(worktree)
        header = (
            f"\n--- OUTPUT FILE: {rel} "
            f"({path.stat().st_size} bytes) ---\n"
        )
        if path.suffix.lower() == ".csv":
            body = csv_context(path)
        else:
            body = path.read_text(
                encoding="utf-8",
                errors="replace",
            )[:MAX_FILE_CHARS]

        parts.append(header + body)

    log_tail = tail_text(log_path)
    if log_tail:
        parts.append(
            "\n--- EXECUTION LOG TAIL ---\n"
            + log_tail
        )

    combined = "\n".join(parts)
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[-MAX_CONTEXT_CHARS:]

    return combined, [
        str(path.relative_to(worktree))
        for path in found_files
    ]
