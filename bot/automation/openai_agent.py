from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .safety_validator import (
    ALLOWED_PREFIXES,
    normalize_repo_path,
    validate_proposal_payload,
)
from .settings import (
    BASE_DIR,
    OPENAI_ANALYSIS_MAX_OUTPUT_TOKENS,
    OPENAI_API_KEY,
    OPENAI_CODE_MAX_OUTPUT_TOKENS,
    OPENAI_MODEL,
    OPENAI_PLANNER_MAX_OUTPUT_TOKENS,
    OPENAI_RATE_LIMIT_RETRIES,
    OPENAI_RATE_LIMIT_SLEEP_SECONDS,
    OPENAI_REASONING_EFFORT,
    OPENAI_TIMEOUT_SECONDS,
    STATE_DIR,
)


RESPONSES_URL = "https://api.openai.com/v1/responses"


PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "plan": {"type": "string"},
        "files_to_read": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "new_files": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
    },
    "required": [
        "title",
        "plan",
        "files_to_read",
        "new_files",
    ],
    "additionalProperties": False,
}


CODE_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "hypothesis": {"type": "string"},
        "requires_frozen_change": {"type": "boolean"},
        "frozen_change_reason": {"type": "string"},
        "analysis_focus": {"type": "string"},
        "files": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "run_script": {"type": "string"},
        "run_args": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 30,
        },
        "output_globs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {"type": "string"},
        },
    },
    "required": [
        "title",
        "hypothesis",
        "requires_frozen_change",
        "frozen_change_reason",
        "analysis_focus",
        "files",
        "run_script",
        "run_args",
        "output_globs",
    ],
    "additionalProperties": False,
}


RESULT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "KEEP",
                "HOLD",
                "REJECT",
                "NEEDS_REVIEW",
            ],
        },
        "telegram_text": {"type": "string"},
        "next_request": {"type": "string"},
    },
    "required": [
        "verdict",
        "telegram_text",
        "next_request",
    ],
    "additionalProperties": False,
}


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    texts: list[str] = []

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue

            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)

            if content.get("type") == "refusal":
                refusal = content.get("refusal")
                if isinstance(refusal, str):
                    raise RuntimeError(
                        f"OpenAI refused the request: {refusal}"
                    )

    if texts:
        return "\n".join(texts)

    raise RuntimeError(
        "OpenAI response did not contain output_text."
    )


class OpenAIResearchAgent:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from .env"
            )

    def _structured_response(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": OPENAI_MODEL,
            "input": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": max_output_tokens,
        }

        if OPENAI_REASONING_EFFORT:
            payload["reasoning"] = {
                "effort": OPENAI_REASONING_EFFORT,
            }

        response = None

        for attempt in range(
            OPENAI_RATE_LIMIT_RETRIES + 1
        ):
            response = requests.post(
                RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )

            if response.status_code != 429:
                break

            body = response.text[:3000]

            # Retrying cannot fix a single request whose declared
            # prompt + output budget is itself above the TPM ceiling.
            if "Request too large" in body:
                raise RuntimeError(
                    "OpenAI request exceeds the current TPM limit even "
                    "before execution. Reduce prompt/context or the "
                    "max_output_tokens budget. "
                    f"API response: {body}"
                )

            if attempt >= OPENAI_RATE_LIMIT_RETRIES:
                break

            retry_after = response.headers.get(
                "retry-after"
            )
            try:
                wait_seconds = float(
                    retry_after
                ) if retry_after else (
                    OPENAI_RATE_LIMIT_SLEEP_SECONDS
                    * (attempt + 1)
                )
            except ValueError:
                wait_seconds = (
                    OPENAI_RATE_LIMIT_SLEEP_SECONDS
                    * (attempt + 1)
                )

            time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError(
                "OpenAI API request was not attempted."
            )

        if response.status_code >= 400:
            body = response.text[:3000]
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {body}"
            )

        data = response.json()
        text = _extract_output_text(data)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Structured OpenAI output was not valid JSON."
            ) from exc

        return parsed

    @staticmethod
    def _read_text(path: Path, limit: int = 35_000) -> str:
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if len(text) <= limit:
            return text
        return (
            text[: limit // 2]
            + "\n\n... [TRUNCATED] ...\n\n"
            + text[-limit // 2 :]
        )

    def _state_context(self) -> str:
        parts = []

        for name in [
            "current_strategy.json",
            "frozen_rules.json",
            "last_ai_result.json",
            "research_decisions.md",
        ]:
            path = STATE_DIR / name
            if path.exists():
                parts.append(
                    f"\n--- research/state/{name} ---\n"
                    + self._read_text(path, 30_000)
                )

        return "\n".join(parts)

    def _repo_tree(self) -> str:
        rows = []

        for prefix in ALLOWED_PREFIXES:
            root = BASE_DIR / prefix
            if not root.exists():
                continue

            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts:
                    continue
                if path.suffix in {
                    ".pyc",
                    ".parquet",
                    ".csv",
                    ".log",
                }:
                    continue

                rel = path.relative_to(BASE_DIR)
                rows.append(
                    f"{rel} ({path.stat().st_size} bytes)"
                )

        return "\n".join(sorted(rows)[:700])

    def plan(self, request_text: str) -> dict[str, Any]:
        system = """You are the planning component of an automated quantitative crypto research system.

Your task is ONLY to decide which existing repository files should be inspected before a code proposal is generated.

Security and research constraints:
- Research only. Never propose live order placement, withdrawals, leverage changes, credential handling, or secret access.
- Existing frozen strategy state is authoritative.
- Prefer a new research/backtest script over editing a previously frozen experiment.
- Only inspect code under:
  bot/backtest/
  bot/analysis/
  bot/data_engine/
  research/experiments/
- Never request .env, credentials, bot/automation, bot/execution, deploy, git internals, or venv.
- Keep files_to_read minimal and relevant.
- new_files are only suggested paths; the next generation step decides final contents.
"""

        user = f"""USER RESEARCH REQUEST
{request_text}

CURRENT RESEARCH STATE
{self._state_context()}

ALLOWED REPOSITORY TREE
{self._repo_tree()}
"""

        plan = self._structured_response(
            schema_name="research_plan",
            schema=PLANNER_SCHEMA,
            system=system,
            user=user,
            max_output_tokens=OPENAI_PLANNER_MAX_OUTPUT_TOKENS,
        )

        clean_files = []
        for raw in plan.get("files_to_read", []):
            try:
                rel = normalize_repo_path(raw)
            except Exception:
                continue
            path = BASE_DIR / rel
            if path.exists() and path.is_file():
                clean_files.append(rel)

        plan["files_to_read"] = clean_files[:8]

        clean_new = []
        for raw in plan.get("new_files", []):
            try:
                clean_new.append(
                    normalize_repo_path(raw)
                )
            except Exception:
                continue
        plan["new_files"] = clean_new[:6]

        return plan

    def _selected_file_context(
        self,
        files_to_read: list[str],
    ) -> str:
        total_limit = 90_000
        used = 0
        parts = []

        for rel in files_to_read:
            path = BASE_DIR / rel
            text = self._read_text(path, 30_000)

            if used + len(text) > total_limit:
                remaining = total_limit - used
                if remaining <= 2000:
                    break
                text = text[:remaining]

            parts.append(
                f"\n--- FILE: {rel} ---\n{text}"
            )
            used += len(text)

        return "\n".join(parts)

    def generate_code_proposal(
        self,
        request_text: str,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        system = """You are the code-generation component of an automated quantitative crypto research system.

Produce ONE executable, falsifiable research experiment.

Hard constraints:
- Research only. No live orders, no exchange account mutations, no withdrawals, no secret access.
- Do not modify automation, deployment, .env, execution/live-trading code, credentials, or git internals.
- Allowed code paths only:
  bot/backtest/
  bot/analysis/
  bot/data_engine/
  research/experiments/
- Prefer creating a new research script rather than changing a frozen script.
- All features/signals must be causal. Avoid look-ahead leakage.
- Preserve chronological Discovery/Validation separation when applicable.
- Do not retune a frozen parameter using Validation results.
- If the proposal changes a frozen strategy rule, set requires_frozen_change=true and explain why.
- run_script must be a Python file under the allowed paths.
- Output files must be written under reports/ or research/results/.
- Do not use shell execution, subprocess, eval, exec, network POST/PUT/PATCH/DELETE calls, sockets, SMTP, SSH, or credential environment variables.
- Public market-data GET requests are allowed when genuinely required.
- Code must be suitable for Ubuntu 24.04 and the existing project venv.
- Gitignored historical datasets are not copied into the proposal worktree.
  For READ-ONLY access to existing local datasets, use:
  Path(os.environ["CRYPTO_BOT_SHARED_DATA_DIR"])
  which points to /home/botadmin/crypto-bot/data at execution time.
  Never write into CRYPTO_BOT_SHARED_DATA_DIR.
- Write experiment outputs only under reports/ or research/results/ in the worktree.
- Return complete file contents for every file you propose to create or replace.
"""

        user = f"""USER RESEARCH REQUEST
{request_text}

PLANNER OUTPUT
{json.dumps(plan, ensure_ascii=False, indent=2)}

CURRENT RESEARCH STATE
{self._state_context()}

SELECTED EXISTING FILES
{self._selected_file_context(plan.get("files_to_read", []))}

Create the smallest robust experiment that answers the request.
"""

        proposal = self._structured_response(
            schema_name="code_proposal",
            schema=CODE_PROPOSAL_SCHEMA,
            system=system,
            user=user,
            max_output_tokens=OPENAI_CODE_MAX_OUTPUT_TOKENS,
        )

        return validate_proposal_payload(proposal)

    def analyze_result(
        self,
        *,
        proposal: dict[str, Any],
        result_context: str,
    ) -> dict[str, Any]:
        system = """You are the result-analysis component of an automated quantitative crypto research system.

Evaluate the completed experiment without inventing metrics.

Rules:
- Distinguish Discovery from Validation.
- Do not select or retune parameters using Validation.
- Be skeptical of tiny samples, overlapping labels, survivorship bias, regime dependence, cost fragility, and correlated features.
- If costs invalidate the edge, say so.
- KEEP means evidence is strong enough to preserve the candidate for the next pre-defined validation stage, NOT permission for live trading.
- REJECT means the tested hypothesis failed.
- HOLD means interesting but insufficient.
- NEEDS_REVIEW means the experiment itself is methodologically or technically ambiguous.
- telegram_text must be concise, practical, and contain the important numeric evidence.
- Do not include file paths, CSV attachments, implementation chatter, or a heading called 'Summary'.
- next_request should be one specific next research experiment, or an empty string if no next experiment is justified.
"""

        user = f"""PROPOSAL
Title: {proposal.get('title')}
Hypothesis: {proposal.get('hypothesis')}
Analysis focus: {proposal.get('analysis_focus')}
Frozen-rule change requested: {proposal.get('requires_frozen_change')}
Frozen-rule reason: {proposal.get('frozen_change_reason')}

EXPERIMENT OUTPUT
{result_context}
"""

        return self._structured_response(
            schema_name="result_analysis",
            schema=RESULT_ANALYSIS_SCHEMA,
            system=system,
            user=user,
            max_output_tokens=OPENAI_ANALYSIS_MAX_OUTPUT_TOKENS,
        )
