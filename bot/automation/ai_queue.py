from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import (
    AI_COMPLETED_DIR,
    AI_FAILED_DIR,
    AI_PENDING_DIR,
    AI_RUNNING_DIR,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class AIJobQueue:
    def create(
        self,
        kind: str,
        requested_by: str,
        **payload: Any,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        job_id = (
            f"AI_{now.strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        data = {
            "job_id": job_id,
            "kind": kind,
            "requested_by": requested_by,
            "created_at": utc_now(),
            "status": "pending",
            **payload,
        }

        path = AI_PENDING_DIR / f"{job_id}.json"
        atomic_json_write(path, data)
        return data

    def claim(self) -> Path | None:
        pending = sorted(
            AI_PENDING_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        for source in pending:
            target = AI_RUNNING_DIR / source.name
            try:
                os.replace(source, target)
                return target
            except FileNotFoundError:
                continue
        return None

    def finish(
        self,
        running_path: Path,
        payload: dict[str, Any],
        success: bool,
    ) -> Path:
        destination_dir = AI_COMPLETED_DIR if success else AI_FAILED_DIR
        destination = destination_dir / running_path.name
        atomic_json_write(running_path, payload)
        os.replace(running_path, destination)
        return destination

    def status(self) -> dict[str, int]:
        return {
            "pending": len(list(AI_PENDING_DIR.glob("*.json"))),
            "running": len(list(AI_RUNNING_DIR.glob("*.json"))),
            "completed": len(list(AI_COMPLETED_DIR.glob("*.json"))),
            "failed": len(list(AI_FAILED_DIR.glob("*.json"))),
        }
