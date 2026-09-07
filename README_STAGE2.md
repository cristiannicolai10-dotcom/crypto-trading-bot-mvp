# Crypto Research Automation — Stage 2

Stage 2 adds an AI research loop on top of the working Telegram/worker Stage 1.

## Flow

Telegram `/propose <request>`
→ OpenAI plans relevant repository context
→ OpenAI generates one structured code proposal
→ isolated Git worktree + research branch
→ static safety validation + Git diff
→ Telegram APPROVE / REJECT
→ Python compile + `git diff --check`
→ commit + push research branch to GitHub
→ experiment runs in the isolated worktree
→ result files stay on Hetzner
→ OpenAI analyzes the results
→ concise analysis in Telegram
→ optional PROMOTE TO MAIN / KEEP BRANCH / CREATE NEXT

## Safety

AI-generated code is NOT allowed to modify:
- `.env`
- `bot/automation/`
- `bot/execution/`
- `deploy/`
- credentials/secrets
- git internals

AI-generated code is limited to:
- `bot/backtest/`
- `bot/analysis/`
- `bot/data_engine/`
- `research/experiments/`

The execution command is not arbitrary shell. The worker constructs:

`/home/botadmin/crypto-bot/venv/bin/python <approved run_script> <approved args>`

Generated research code is statically rejected if it includes dangerous shell/process primitives, secret names, live order calls, or mutation-oriented network calls.

Main promotion is `git merge --ff-only`; if main changed meanwhile, promotion stops rather than forcing a merge.

## Telegram

- `/propose <research request>`
- `/proposals`
- `/proposal P-...`
- `/diff P-...`
- `/approve P-...`
- `/reject P-...`
- `/promote P-...`
- `/result P-...`
- `/logs P-...`
- `/status`
- `/current`

Existing E-xxxx frozen experiments still work.

## Required `.env`

Do not share the key in Telegram or ChatGPT.

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-sol
OPENAI_REASONING_EFFORT=high
```

## Install

Extract the Stage 2 ZIP over `/home/botadmin/crypto-bot`, then:

```bash
chmod +x deploy/install_research_automation_stage2.sh
./deploy/install_research_automation_stage2.sh
```

After installation, commit and push Stage 2 so `main` is clean:

```bash
git status
git add bot/automation deploy research/experiments research/state .gitignore README_STAGE2.md
git commit -m "Add research automation stage 2"
git push origin main
```

Do not add `.env`.

## First safe test

Use a small research-only proposal:

```text
/propose Create a diagnostic script that reads the latest frozen A/B/D cost-summary results and checks whether each candidate remains positive after 2 bps slippage per side. Do not change any frozen rule.
```

The AI worker should return a P-... proposal with APPROVE / REJECT.

Inspect:

```text
/diff P-...
```

Approve only after checking the diff.

## Logs

```bash
journalctl -u crypto-research-telegram -f
journalctl -u crypto-research-ai-worker -f
```

## Notes

- OpenAI API usage is billed through the API account/project and is separate from the ChatGPT application subscription.
- The default model is configurable through `OPENAI_MODEL`.
- Result CSVs remain on the server and are not sent to Telegram.
- A KEEP verdict is a research decision, never authorization for live trading.


## Stage 2 V2 — VIEW DIFF button

AI proposal messages now contain:

- `🧾 VIEW DIFF`
- `✅ APPROVE CODE`
- `❌ REJECT`

Pressing `VIEW DIFF` performs the same action as `/diff P-...` and displays the code diff directly in Telegram.

For a proposal created before this upgrade, send `/proposal P-...`; the refreshed proposal message will include the new buttons.


## Stage 2 V2.1 — Git porcelain path fix

Fixed a bug where `run_git()` used `.strip()` on Git output. For the first
`git status --porcelain` line this removed the leading status-column space,
causing `research/...` to become `esearch/...` during Python compile.

V2.1 preserves leading whitespace with `.rstrip()` and includes a regression
check for the exact failure mode.


## Stage 2 V2.2 — regression test import fix

V2.1 fixed the real Git porcelain parsing bug, but its installer executed the
regression test as a file path, so Python did not place the project root on
`sys.path` and raised `ModuleNotFoundError: No module named 'bot'`.

V2.2:
- keeps the `.rstrip()` Git parsing fix;
- makes the regression test directly executable;
- runs it safely as `python -m bot.automation.test_git_manager_porcelain`.


## Stage 2 V2.3 — OpenAI TPM protection

The previous Stage 2 used one 32K `max_output_tokens` allowance for planner,
code-generation, and result-analysis calls. On a 60K TPM organization this
could make a single code-generation request exceed the rate limit before it
started.

V2.3 uses separate budgets:
- Planner: 3,000 output tokens
- Code proposal: 16,000 output tokens
- Result analysis: 4,000 output tokens

It also reduces repository/context text sent to the model and retries temporary
429 rate-limit errors with backoff. A truly oversized single request is not
blindly retried; it returns a clear error instead.
