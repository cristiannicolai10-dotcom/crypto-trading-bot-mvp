from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import EXPERIMENT_DIR


class ExperimentRegistry:
    def __init__(self) -> None:
        self.experiment_dir = EXPERIMENT_DIR

    def _load_file(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        required = [
            "experiment_id",
            "title",
            "command",
        ]
        missing = [
            key
            for key in required
            if key not in data
        ]
        if missing:
            raise RuntimeError(
                f"{path.name} missing fields: {', '.join(missing)}"
            )

        if not isinstance(data["command"], list) or not data["command"]:
            raise RuntimeError(
                f"{path.name}: command must be a non-empty JSON list"
            )

        return data

    def all(self) -> list[dict[str, Any]]:
        experiments = []
        for path in sorted(self.experiment_dir.glob("E-*.json")):
            data = self._load_file(path)
            if data.get("enabled", True):
                experiments.append(data)
        return experiments

    def get(self, experiment_id: str) -> dict[str, Any]:
        experiment_id = experiment_id.strip().upper()

        direct = self.experiment_dir / f"{experiment_id}.json"
        if direct.exists():
            data = self._load_file(direct)
            if not data.get("enabled", True):
                raise RuntimeError(f"{experiment_id} is disabled.")
            return data

        for data in self.all():
            if data["experiment_id"].upper() == experiment_id:
                return data

        raise KeyError(f"Experiment not found: {experiment_id}")

    @staticmethod
    def proposal_text(exp: dict[str, Any]) -> str:
        lines = [
            f"EXPERIMENT {exp['experiment_id']}",
            f"Title: {exp['title']}",
        ]

        hypothesis = exp.get("hypothesis")
        if hypothesis:
            lines.extend(
                [
                    f"Hypothesis: {hypothesis}",
                ]
            )

        lines.extend(
            [
                "",
                "Command:",
                "  " + " ".join(str(x) for x in exp["command"]),
            ]
        )

        return "\n".join(lines)
