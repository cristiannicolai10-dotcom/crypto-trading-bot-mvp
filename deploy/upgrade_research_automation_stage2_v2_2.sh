#!/usr/bin/env bash
set -euo pipefail

BASE="/home/botadmin/crypto-bot"
cd "$BASE"

echo "== Stage 2 V2.2: Git path parsing + regression test fix =="

"$BASE/venv/bin/python" -m py_compile \
  bot/automation/git_manager.py \
  bot/automation/ai_worker.py \
  bot/automation/research_bot.py \
  bot/automation/test_git_manager_porcelain.py

# Run as a module so the project root is on sys.path.
"$BASE/venv/bin/python" -m bot.automation.test_git_manager_porcelain

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
echo "V2.2 active."
echo "Git porcelain parsing regression test passed."
