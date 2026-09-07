#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Stage 2 V2.3: reduce OpenAI TPM footprint =="

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/settings.py \
  bot/automation/openai_agent.py \
  bot/automation/ai_worker.py \
  bot/automation/research_bot.py

append_if_missing() {
  key="$1"
  value="$2"
  if ! grep -q "^${key}=" "$BASE/.env"; then
    printf '%s=%s\n' "$key" "$value" >> "$BASE/.env"
  fi
}

append_if_missing OPENAI_PLANNER_MAX_OUTPUT_TOKENS 3000
append_if_missing OPENAI_CODE_MAX_OUTPUT_TOKENS 16000
append_if_missing OPENAI_ANALYSIS_MAX_OUTPUT_TOKENS 4000
append_if_missing OPENAI_RATE_LIMIT_RETRIES 2
append_if_missing OPENAI_RATE_LIMIT_SLEEP_SECONDS 65

sudo systemctl restart crypto-research-telegram.service
sudo systemctl restart crypto-research-ai-worker.service

echo
echo "OpenAI budgets:"
grep -E '^OPENAI_(MODEL|REASONING_EFFORT|PLANNER_MAX_OUTPUT_TOKENS|CODE_MAX_OUTPUT_TOKENS|ANALYSIS_MAX_OUTPUT_TOKENS|RATE_LIMIT_RETRIES|RATE_LIMIT_SLEEP_SECONDS)=' "$BASE/.env" || true

echo
echo "Services:"
for svc in \
  crypto-research-telegram \
  crypto-research-ai-worker
do
  systemctl is-active --quiet "$svc" && \
    echo "  $svc: active" || \
    echo "  $svc: NOT ACTIVE"
done

echo
echo "V2.3 active."
echo "Planner/code/analysis now use separate token budgets and 429 backoff."
