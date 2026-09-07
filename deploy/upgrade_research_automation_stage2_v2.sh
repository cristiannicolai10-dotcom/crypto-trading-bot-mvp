#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Upgrading Research Automation Stage 2 -> V2 =="

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/ai_worker.py \
  bot/automation/research_bot.py \
  bot/automation/telegram_api.py

sudo systemctl restart crypto-research-telegram.service
sudo systemctl restart crypto-research-ai-worker.service

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
echo "V2 active."
echo "New AI proposals now include a VIEW DIFF button."
echo "For an existing proposal, use /proposal P-... to get the new buttons."
