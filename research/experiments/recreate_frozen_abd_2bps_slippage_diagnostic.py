from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


# The experiment is located at research/experiments/<file>, so parents[2]
# is the proposal worktree root.
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (WORKTREE_ROOT / "reports").resolve()
EXPECTED_SOURCE_REPORT_DIR = Path(
    "/home/botadmin/crypto-bot/reports"
).resolve()

if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

# Import only; do not modify frozen.REPORT_DIR or any frozen constants.
from bot.backtest import build_frozen_A_B_D_signal_book as frozen


STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def verify_frozen_source_configuration():
    configured = Path(frozen.REPORT_DIR).resolve()
    if configured != EXPECTED_SOURCE_REPORT_DIR:
        raise RuntimeError(
            "Frozen REPORT_DIR is not the required existing source directory: "
            f"{configured} != {EXPECTED_SOURCE_REPORT_DIR}"
        )


def verify_source_file(path, label):
    source = Path(path).resolve()
    if source.parent != EXPECTED_SOURCE_REPORT_DIR:
        raise RuntimeError(
            f"{label} loader returned an unexpected source path: {source}"
        )
    if not source.is_file():
        raise RuntimeError(f"{label} source report does not exist: {source}")


def output_path(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR:
        raise RuntimeError(f"Refusing to write outside worktree reports: {path}")
    return path


def main():
    verify_frozen_source_configuration()

    # These are the unchanged frozen loaders. They resolve their own latest
    # source reports using frozen.REPORT_DIR, which remains /home/.../reports.
    a_rows, a_source = frozen.load_A()
    bd_rows, bd_source = frozen.load_B_D()
    verify_source_file(a_source, "A")
    verify_source_file(bd_source, "B/D")

    signal_book = (
        pd.concat([a_rows, bd_rows], ignore_index=True)
        .sort_values(["available_at", "strategy_id"])
        .reset_index(drop=True)
    )

    # Reuse the frozen cost formula and scenario construction. The frozen
    # function creates its fixed grid; this diagnostic selects only 2 bps.
    all_cost_rows = frozen.add_cost_scenarios(signal_book)
    cost_rows = all_cost_rows.loc[
        all_cost_rows["slippage_bps_per_side"].eq(2.0)
    ].copy()

    if cost_rows.empty:
        raise RuntimeError("The frozen cost calculation produced no 2 bps rows.")

    if set(cost_rows["slippage_bps_per_side"].unique()) != {2.0}:
        raise RuntimeError("2 bps diagnostic contains an unexpected slippage value.")

    cost_summary = frozen.summarize_costs(cost_rows)
    break_even = frozen.strategy_break_even_slippage(signal_book)
    overlaps = frozen.overlap_report(signal_book)

    signal_file = output_path(
        f"recreated_frozen_A_B_D_signal_book_{STAMP}.csv"
    )
    costs_file = output_path(
        f"recreated_frozen_A_B_D_cost_rows_2bps_{STAMP}.csv"
    )
    summary_file = output_path(
        f"recreated_frozen_A_B_D_cost_summary_2bps_{STAMP}.csv"
    )
    break_even_file = output_path(
        f"recreated_frozen_A_B_D_break_even_slippage_{STAMP}.csv"
    )
    overlaps_file = output_path(
        f"recreated_frozen_A_B_D_overlaps_{STAMP}.csv"
    )

    signal_book.to_csv(signal_file, index=False)
    cost_rows.to_csv(costs_file, index=False)
    cost_summary.to_csv(summary_file, index=False)
    break_even.to_csv(break_even_file, index=False)
    overlaps.to_csv(overlaps_file, index=False)

    print("Frozen A/B/D 2 bps diagnostic complete.")
    print(f"A source:       {a_source}")
    print(f"B/D source:     {bd_source}")
    print(f"Signal rows:    {len(signal_book)}")
    print(f"Cost rows:      {len(cost_rows)}")
    print(f"Signal book:    {signal_file}")
    print(f"Cost rows:      {costs_file}")
    print(f"Cost summary:   {summary_file}")
    print(f"Break-even:     {break_even_file}")
    print(f"Overlaps:       {overlaps_file}")


if __name__ == "__main__":
    main()
