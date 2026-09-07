#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Research Automation Stage 1 installer =="

if [ ! -x "$BASE/venv/bin/python" ]; then
  echo "ERROR: venv python not found at $BASE/venv/bin/python"
  exit 1
fi

if [ ! -f "$BASE/.env" ]; then
  echo "ERROR: $BASE/.env not found"
  exit 1
fi

if ! grep -q '^TELEGRAM_TOKEN=' "$BASE/.env"; then
  echo "ERROR: TELEGRAM_TOKEN missing in .env"
  exit 1
fi

if ! grep -q '^TELEGRAM_CHAT_ID=' "$BASE/.env"; then
  echo "ERROR: TELEGRAM_CHAT_ID missing in .env"
  exit 1
fi

mkdir -p \
  research/experiments \
  research/state \
  research/proposals \
  research/jobs/pending \
  research/jobs/running \
  research/jobs/completed \
  research/jobs/failed \
  logs/automation \
  bot/automation

# Install the current E-0001 research script only if it is not already present.
if [ ! -f "bot/backtest/build_frozen_A_B_D_signal_book.py" ]; then
  cp \
    "bootstrap_files/build_frozen_A_B_D_signal_book.py" \
    "bot/backtest/build_frozen_A_B_D_signal_book.py"
  echo "Installed bot/backtest/build_frozen_A_B_D_signal_book.py"
else
  echo "Keeping existing bot/backtest/build_frozen_A_B_D_signal_book.py"
fi

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/settings.py \
  bot/automation/telegram_api.py \
  bot/automation/experiment_registry.py \
  bot/automation/job_queue.py \
  bot/automation/research_bot.py \
  bot/automation/research_worker.py \
  bot/backtest/build_frozen_A_B_D_signal_book.py

echo "Python compile: OK"

sudo cp \
  deploy/crypto-research-telegram.service \
  /etc/systemd/system/crypto-research-telegram.service

sudo cp \
  deploy/crypto-research-worker.service \
  /etc/systemd/system/crypto-research-worker.service

sudo systemctl daemon-reload
sudo systemctl enable --now crypto-research-telegram.service
sudo systemctl enable --now crypto-research-worker.service

echo
echo "Services:"
sudo systemctl --no-pager --full status crypto-research-telegram.service | sed -n '1,12p'
echo
sudo systemctl --no-pager --full status crypto-research-worker.service | sed -n '1,12p'

echo
echo "Stage 1 installed."
echo "Open Telegram and send: /help"
