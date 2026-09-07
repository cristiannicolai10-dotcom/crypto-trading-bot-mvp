# Crypto Research Automation — Stage 1

Stage 1 replaces the manual flow:

Chat -> download script -> upload server -> run -> download CSV -> upload result

with:

Telegram -> proposal -> APPROVE -> Hetzner worker -> run -> CSV/logs -> Telegram

## Safety model

- The Telegram bot accepts commands only from `TELEGRAM_CHAT_ID`.
- Telegram text is never executed as a shell command.
- Only commands declared in `research/experiments/E-*.json` may execute.
- `/run E-xxxx` creates a proposal only.
- `/approve E-xxxx` queues the experiment.
- The worker is a separate systemd service.
- No live trading or order placement exists in Stage 1.

## Telegram commands

- `/status`
- `/current`
- `/experiments`
- `/run E-0001`
- `/approve E-0001`
- `/reject E-0001`
- `/result E-0001`
- `/logs E-0001`
- `/help`

## First experiment

`E-0001` runs the frozen A/B/D 60-minute cost diagnostic.

## Installation

From `/home/botadmin/crypto-bot`, after extracting the ZIP:

```bash
chmod +x deploy/install_research_automation.sh
./deploy/install_research_automation.sh
```

Then in Telegram:

```text
/help
/experiments
/run E-0001
/approve E-0001
```

The worker will execute the experiment and send its result CSV files back to Telegram automatically.

## Logs

```bash
journalctl -u crypto-research-telegram -f
journalctl -u crypto-research-worker -f
```

Local logs are also stored in:

```text
/home/botadmin/crypto-bot/logs/automation/
```

## Stop / restart

```bash
sudo systemctl restart crypto-research-telegram
sudo systemctl restart crypto-research-worker

sudo systemctl stop crypto-research-telegram
sudo systemctl stop crypto-research-worker
```

## Stage 2

After Stage 1 is stable, Stage 2 adds:

OpenAI API -> proposed code patch -> git branch/diff -> Telegram approval -> validation -> GitHub push -> worker run -> AI result analysis -> Telegram report.

Stage 2 should not be enabled until Stage 1 reliably executes and returns several experiments.


## V2 Telegram behavior

- `/run E-xxxx` includes real APPROVE / REJECT buttons.
- Result CSV files are not sent to Telegram.
- CSV files remain on the Hetzner server.
- Completion messages are concise and omit Exit code, Git hash, and output file lists.
- `/result E-xxxx` returns only status, duration, and summary.
- `/logs E-xxxx` remains available for troubleshooting.


## V3 Telegram behavior

Removed from Telegram messages:
- Summary
- Expected outputs
- Frozen parameters

`/run E-xxxx` now shows only experiment identity, title/hypothesis, command, and APPROVE / REJECT buttons.

Completion messages now show only:
- COMPLETED / FAILED
- Duration
- A short error message only on failure
- Confirmation that results remain on the server

`/current` still intentionally shows the current/frozen research state when explicitly requested.
