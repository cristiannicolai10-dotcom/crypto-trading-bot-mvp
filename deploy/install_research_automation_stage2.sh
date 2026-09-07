#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Installing Research Automation Stage 2 =="

if [ ! -x "$BASE/venv/bin/python" ]; then
  echo "ERROR: project venv missing."
  exit 1
fi

if [ ! -f "$BASE/.env" ]; then
  echo "ERROR: $BASE/.env missing."
  exit 1
fi

if ! grep -q '^OPENAI_API_KEY=' "$BASE/.env"; then
  echo "ERROR: OPENAI_API_KEY missing in .env"
  echo "Add it, then run this installer again."
  exit 1
fi

if ! grep -q '^OPENAI_MODEL=' "$BASE/.env"; then
  printf '\nOPENAI_MODEL=gpt-5.6-sol\n' >> "$BASE/.env"
fi

if ! grep -q '^OPENAI_REASONING_EFFORT=' "$BASE/.env"; then
  printf 'OPENAI_REASONING_EFFORT=high\n' >> "$BASE/.env"
fi

mkdir -p \
  research/ai_proposals \
  research/ai_jobs/pending \
  research/ai_jobs/running \
  research/ai_jobs/completed \
  research/ai_jobs/failed \
  logs/automation \
  /home/botadmin/research_worktrees

touch .gitignore

for line in \
  "research/jobs/" \
  "research/proposals/" \
  "research/ai_jobs/" \
  "research/ai_proposals/" \
  "research/state/last_ai_result.json" \
  "logs/" \
  "*.log"
do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/settings.py \
  bot/automation/telegram_api.py \
  bot/automation/ai_queue.py \
  bot/automation/proposal_manager.py \
  bot/automation/safety_validator.py \
  bot/automation/git_manager.py \
  bot/automation/result_context.py \
  bot/automation/openai_agent.py \
  bot/automation/research_bot.py \
  bot/automation/research_worker.py \
  bot/automation/ai_worker.py

echo "Python compile: OK"

# Validate API authentication without generating model output.
set +u
source "$BASE/.env"
set -u

HTTP_CODE="$(
  curl -sS -o /tmp/openai_stage2_models.json \
    -w "%{http_code}" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    https://api.openai.com/v1/models
)"

if [ "$HTTP_CODE" != "200" ]; then
  echo "ERROR: OpenAI API authentication test returned HTTP $HTTP_CODE"
  cat /tmp/openai_stage2_models.json
  exit 1
fi

echo "OpenAI API authentication: OK"

sudo cp \
  deploy/crypto-research-ai-worker.service \
  /etc/systemd/system/crypto-research-ai-worker.service

sudo systemctl daemon-reload
sudo systemctl enable crypto-research-ai-worker.service
sudo systemctl restart crypto-research-telegram.service
sudo systemctl restart crypto-research-worker.service
sudo systemctl restart crypto-research-ai-worker.service

echo
echo "Services:"
for svc in \
  crypto-research-telegram \
  crypto-research-worker \
  crypto-research-ai-worker
do
  systemctl is-active --quiet "$svc" && \
    echo "  $svc: active" || \
    echo "  $svc: NOT ACTIVE"
done

echo
echo "Stage 2 installed."
echo
echo "IMPORTANT:"
echo "Before /propose can create a Git worktree, commit and push"
echo "the Stage 2 installation so the main repository is clean."
echo
echo "Example:"
echo "  git status"
echo "  git add bot/automation deploy research/experiments research/state .gitignore README_STAGE2.md"
echo '  git commit -m "Add research automation stage 2"'
echo "  git push origin main"
