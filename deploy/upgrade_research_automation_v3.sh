#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Upgrading Research Automation to V3 =="

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/experiment_registry.py \
  bot/automation/telegram_api.py \
  bot/automation/research_bot.py \
  bot/automation/research_worker.py

sudo systemctl restart crypto-research-telegram.service
sudo systemctl restart crypto-research-worker.service

echo
echo "Telegram bot:"
sudo systemctl --no-pager --full status crypto-research-telegram.service | sed -n '1,12p'

echo
echo "Worker:"
sudo systemctl --no-pager --full status crypto-research-worker.service | sed -n '1,12p'

echo
echo "V3 active."
echo "Telegram messages are now minimal: no summary, expected outputs, or frozen parameters."
