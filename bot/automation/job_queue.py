from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import (
    COMPLETED_DIR,
    FAILED_DIR,
    PENDING_DIR,
    PROPOSAL_DIR,
    RUNNING_DIR,
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


class JobQueue:
    def create_proposal(
        self,
        experiment: dict[str, Any],
        requested_by: str,
    ) -> Path:
        payload = {
            "experiment_id": experiment["experiment_id"],
            "requested_by": requested_by,
            "created_at": utc_now(),
            "status": "pending_approval",
        }
        path = PROPOSAL_DIR / f"{experiment['experiment_id']}.json"
        atomic_json_write(path, payload)
        return path

    def get_proposal(self, experiment_id: str) -> dict[str, Any] | None:
        path = PROPOSAL_DIR / f"{experiment_id.upper()}.json"
        if not path.exists():
            return None
        return load_json(path)

    def reject_proposal(self, experiment_id: str) -> bool:
        path = PROPOSAL_DIR / f"{experiment_id.upper()}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def approve_proposal(
        self,
        experiment: dict[str, Any],
        approved_by: str,
    ) -> dict[str, Any]:
        experiment_id = experiment["experiment_id"]
        proposal_path = PROPOSAL_DIR / f"{experiment_id}.json"
        if not proposal_path.exists():
            raise RuntimeError(
                f"No pending proposal for {experiment_id}. Use /run {experiment_id} first."
            )

        proposal = load_json(proposal_path)

        job_id = (
            f"{experiment_id}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        payload = {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "requested_by": proposal.get("requested_by"),
            "approved_by": approved_by,
            "created_at": proposal.get("created_at"),
            "approved_at": utc_now(),
            "status": "pending",
        }

        pending_path = PENDING_DIR / f"{job_id}.json"
        atomic_json_write(pending_path, payload)
        proposal_path.unlink(missing_ok=True)
        return payload

    @staticmethod
    def list_paths(directory: Path) -> list[Path]:
        return sorted(
            directory.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def queue_status(self) -> dict[str, int]:
        return {
            "pending": len(self.list_paths(PENDING_DIR)),
            "running": len(self.list_paths(RUNNING_DIR)),
            "completed": len(self.list_paths(COMPLETED_DIR)),
            "failed": len(self.list_paths(FAILED_DIR)),
        }

    def latest_job(
        self,
        experiment_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        experiment_id = experiment_id.upper()
        candidates: list[tuple[str, Path]] = []

        for status, directory in [
            ("running", RUNNING_DIR),
            ("completed", COMPLETED_DIR),
            ("failed", FAILED_DIR),
            ("pending", PENDING_DIR),
        ]:
            for path in self.list_paths(directory):
                try:
                    data = load_json(path)
                except Exception:
                    continue
                if data.get("experiment_id", "").upper() == experiment_id:
                    candidates.append((status, path))

        if not candidates:
            return None

        status, path = max(
            candidates,
            key=lambda item: item[1].stat().st_mtime,
        )
        return status, load_json(path)
