from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import AI_PROPOSAL_DIR


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


class ProposalManager:
    def path(self, proposal_id: str) -> Path:
        return AI_PROPOSAL_DIR / proposal_id.upper() / "proposal.json"

    def create(self, proposal: dict[str, Any]) -> Path:
        proposal_id = proposal["proposal_id"].upper()
        proposal["proposal_id"] = proposal_id
        proposal.setdefault("created_at", utc_now())
        proposal.setdefault("status", "ready_for_approval")
        path = self.path(proposal_id)
        atomic_json_write(path, proposal)
        return path

    def get(self, proposal_id: str) -> dict[str, Any]:
        path = self.path(proposal_id)
        if not path.exists():
            raise KeyError(f"Proposal not found: {proposal_id}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def update(self, proposal_id: str, **changes: Any) -> dict[str, Any]:
        proposal = self.get(proposal_id)
        proposal.update(changes)
        proposal["updated_at"] = utc_now()
        atomic_json_write(self.path(proposal_id), proposal)
        return proposal

    def latest(self, limit: int = 10) -> list[dict[str, Any]]:
        paths = sorted(
            AI_PROPOSAL_DIR.glob("P-*/proposal.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        result = []
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    result.append(json.load(handle))
            except Exception:
                continue
        return result
