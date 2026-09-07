#!/usr/bin/env python3
"""Recreate the frozen A/B/D 2 bps-per-side slippage diagnostic.

This script deliberately does not import or modify the frozen backtest
modules. It reproduces their frozen loader and cost methodology while using
an explicit read-only source directory and a separate worktree output
目录.
"""

from datetime import datetime, timezone
from pathlib import Path
import glob

import numpy as np
import pandas as pd


# Read-only frozen source reports. This is intentionally not configurable.
FROZEN_SOURCE_REPORT_DIR = Path("/home/botadmin/crypto-bot/reports")

# The script is research-only and writes beneath the proposal worktree.
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_REPORT_DIR = WORKTREE_ROOT / "reports"

# Frozen A/B/D methodology and risk-cost constants.
LEVERAGE = 10.0
MAX_TOTAL_MARGIN_FRACTION = 0.20
TAKER_FEE_BPS_PER_SIDE = 5.5
SLIPPAGE_BPS_PER_SIDE = 2.0
PRIMARY_EVENT_MODE = "NEW_EPISODE"

A_TOP_N = 6
B_FEATURE = "4h_trend_score"
B_ORIENTATION = "LOWER_BETTER"
B_TOP_N = 5
D_FEATURE = "5m_ret_1bar_pct"
D_ORIENTATION = "HIGHER_BETTER"
D_TOP_N = 1


def latest_file(pattern: str) -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern:\n{pattern}")
    return Path(files[-1])


def profit_factor(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    positive = values.loc[values > 0].sum()
    negative = -values.loc[values < 0].sum()
    if negative <= 0:
        return float("inf") if positive > 0 else np.nan
    return float(positive / negative)


def load_a() -> tuple[pd.DataFrame, Path]:
    path = latest_file(
        str(FROZEN_SOURCE_REPORT_DIR / "market_trigger_A_top30_rank_rows_*.csv")
    )
    df = pd.read_csv(path)
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )

    required = {
        "universe",
        "event_mode",
        "event_id",
        "available_at",
        "split",
        "symbol",
        "coin_rank",
        "return_60m_pct",
        "return_60m_atr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Trigger A rank rows missing columns:\n" + "\n".join(missing))

    selected = df.loc[
        (df["universe"] == "ALL_AVAILABLE_TOP30")
        & (df["event_mode"] == PRIMARY_EVENT_MODE)
        & (pd.to_numeric(df["coin_rank"], errors="coerce") <= A_TOP_N)
    ].copy()

    if selected.empty:
        raise RuntimeError("No Trigger A Top6 NEW_EPISODE rows found.")

    rows = []
    for (event_id, event_time, split), group in selected.groupby(
        ["event_id", "available_at", "split"], observed=True
    ):
        ordered = group.sort_values("coin_rank")
        rows.append(
            {
                "strategy_id": "A_LONG_TOP6_5M_TREND",
                "trigger": "A",
                "direction": "LONG",
                "split": split,
                "available_at": event_time,
                "event_id": f"A_{int(event_id)}",
                "selected_count": int(group["symbol"].nunique()),
                "selected_symbols": ",".join(ordered["symbol"].astype(str).tolist()),
                "gross_direction_return_pct": float(group["return_60m_pct"].mean()),
                "gross_direction_return_atr": float(group["return_60m_atr"].mean()),
                "source_file": str(path),
            }
        )

    return pd.DataFrame(rows), path


def load_b_and_d() -> tuple[pd.DataFrame, Path]:
    path = latest_file(
        str(FROZEN_SOURCE_REPORT_DIR / "trigger_B_D_selector_absolute_rank_events_*.csv")
    )
    df = pd.read_csv(path)
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )

    required = {
        "trigger_name",
        "direction",
        "feature",
        "orientation",
        "split",
        "event_mode",
        "available_at",
        "top_n",
        "selected_symbols",
        "mean_direction_return_pct",
        "mean_direction_return_atr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "B/D absolute-rank events missing columns:\n" + "\n".join(missing)
        )

    specs = [
        {
            "trigger_name": "TRIGGER_B_LONG",
            "strategy_id": "B_LONG_BOTTOM5_4H_TREND",
            "trigger": "B",
            "direction": "LONG",
            "feature": B_FEATURE,
            "orientation": B_ORIENTATION,
            "top_n": B_TOP_N,
        },
        {
            "trigger_name": "TRIGGER_D_SHORT",
            "strategy_id": "D_SHORT_TOP1_LAST5M_RETURN",
            "trigger": "D",
            "direction": "SHORT",
            "feature": D_FEATURE,
            "orientation": D_ORIENTATION,
            "top_n": D_TOP_N,
        },
    ]

    parts = []
    for spec in specs:
        selected = df.loc[
            (df["trigger_name"] == spec["trigger_name"])
            & (df["event_mode"] == PRIMARY_EVENT_MODE)
            & (df["feature"] == spec["feature"])
            & (df["orientation"] == spec["orientation"])
            & (pd.to_numeric(df["top_n"], errors="coerce") == spec["top_n"])
        ].copy()

        if selected.empty:
            raise RuntimeError(f"No rows for {spec['strategy_id']}")

        selected["strategy_id"] = spec["strategy_id"]
        selected["trigger"] = spec["trigger"]
        selected["direction"] = spec["direction"]
        selected["event_id"] = [
            f"{spec['trigger']}_{i + 1}" for i in range(len(selected))
        ]
        selected["selected_count"] = spec["top_n"]
        selected["gross_direction_return_pct"] = selected[
            "mean_direction_return_pct"
        ]
        selected["gross_direction_return_atr"] = selected[
            "mean_direction_return_atr"
        ]
        selected["source_file"] = str(path)
        parts.append(
            selected[
                [
                    "strategy_id",
                    "trigger",
                    "direction",
                    "split",
                    "available_at",
                    "event_id",
                    "selected_count",
                    "selected_symbols",
                    "gross_direction_return_pct",
                    "gross_direction_return_atr",
                    "source_file",
                ]
            ]
        )

    return pd.concat(parts, ignore_index=True), path


def add_two_bps_costs(signal_book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    roundtrip_cost_pct = 2.0 * (
        TAKER_FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE
    ) * 0.01

    for _, row in signal_book.iterrows():
        gross_pct = float(row["gross_direction_return_pct"])
        net_pct = gross_pct - roundtrip_cost_pct
        out = row.to_dict()
        out.update(
            {
                "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
                "fee_bps_per_side": TAKER_FEE_BPS_PER_SIDE,
                "roundtrip_cost_pct": roundtrip_cost_pct,
                "net_direction_return_pct": net_pct,
                "net_margin_roe_pct_10x": net_pct * LEVERAGE,
                "equity_return_pct_if_full_20pct_margin": (
                    net_pct * LEVERAGE * MAX_TOTAL_MARGIN_FRACTION
                ),
            }
        )
        rows.append(out)

    return pd.DataFrame(rows)


def summarize_costs(cost_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "strategy_id",
        "trigger",
        "direction",
        "split",
        "slippage_bps_per_side",
    ]

    for keys, group in cost_rows.groupby(group_columns, observed=True):
        strategy_id, trigger, direction, split, slip_bps = keys
        net = group["net_direction_return_pct"]
        gross = group["gross_direction_return_pct"]
        rows.append(
            {
                "strategy_id": strategy_id,
                "trigger": trigger,
                "direction": direction,
                "split": split,
                "slippage_bps_per_side": slip_bps,
                "events": int(len(group)),
                "avg_selected_count": float(group["selected_count"].mean()),
                "mean_gross_direction_return_pct": float(gross.mean()),
                "median_gross_direction_return_pct": float(gross.median()),
                "gross_positive_rate": float((gross > 0).mean()),
                "mean_net_direction_return_pct": float(net.mean()),
                "median_net_direction_return_pct": float(net.median()),
                "net_positive_rate": float((net > 0).mean()),
                "net_profit_factor": profit_factor(net),
                "mean_net_margin_roe_pct_10x": float(
                    group["net_margin_roe_pct_10x"].mean()
                ),
                "mean_equity_return_pct_if_full_20pct_margin": float(
                    group["equity_return_pct_if_full_20pct_margin"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def strategy_break_even_slippage(signal_book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["strategy_id", "trigger", "direction", "split"]
    for keys, group in signal_book.groupby(group_columns, observed=True):
        strategy_id, trigger, direction, split = keys
        mean_gross_pct = float(group["gross_direction_return_pct"].mean())
        gross_total_bps = mean_gross_pct / 0.01
        rows.append(
            {
                "strategy_id": strategy_id,
                "trigger": trigger,
                "direction": direction,
                "split": split,
                "events": len(group),
                "mean_gross_direction_return_pct": mean_gross_pct,
                "break_even_slippage_bps_per_side_after_fees": (
                    gross_total_bps - 2.0 * TAKER_FEE_BPS_PER_SIDE
                )
                / 2.0,
            }
        )
    return pd.DataFrame(rows)


def overlap_report(signal_book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_time, group in signal_book.groupby("available_at", observed=True):
        strategies = sorted(group["strategy_id"].unique())
        directions = sorted(group["direction"].unique())
        if len(strategies) <= 1:
            continue
        rows.append(
            {
                "available_at": event_time,
                "strategies": ",".join(strategies),
                "triggers": ",".join(sorted(group["trigger"].unique())),
                "directions": ",".join(directions),
                "strategy_count": len(strategies),
                "direction_conflict": len(directions) > 1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    a_rows, a_source = load_a()
    bd_rows, bd_source = load_b_and_d()
    signal_book = pd.concat([a_rows, bd_rows], ignore_index=True).sort_values(
        ["available_at", "strategy_id"]
    )

    cost_rows = add_two_bps_costs(signal_book)
    cost_summary = summarize_costs(cost_rows)
    break_even = strategy_break_even_slippage(signal_book)
    overlaps = overlap_report(signal_book)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outputs = {
        "signal_book": OUTPUT_REPORT_DIR
        / f"recreated_frozen_A_B_D_signal_book_{stamp}.csv",
        "cost_rows": OUTPUT_REPORT_DIR
        / f"recreated_frozen_A_B_D_2bps_cost_rows_{stamp}.csv",
        "cost_summary": OUTPUT_REPORT_DIR
        / f"recreated_frozen_A_B_D_2bps_cost_summary_{stamp}.csv",
        "break_even": OUTPUT_REPORT_DIR
        / f"recreated_frozen_A_B_D_break_even_slippage_{stamp}.csv",
        "overlaps": OUTPUT_REPORT_DIR
        / f"recreated_frozen_A_B_D_overlaps_{stamp}.csv",
    }

    signal_book.to_csv(outputs["signal_book"], index=False)
    cost_rows.to_csv(outputs["cost_rows"], index=False)
    cost_summary.to_csv(outputs["cost_summary"], index=False)
    break_even.to_csv(outputs["break_even"], index=False)
    overlaps.to_csv(outputs["overlaps"], index=False)

    display_columns = [
        "strategy_id",
        "split",
        "slippage_bps_per_side",
        "events",
        "mean_gross_direction_return_pct",
        "mean_net_direction_return_pct",
        "net_positive_rate",
        "net_profit_factor",
        "mean_net_margin_roe_pct_10x",
    ]

    print("FROZEN A/B/D 2 BPS-PER-SIDE DIAGNOSTIC")
    print(f"A source:       {a_source}")
    print(f"B/D source:     {bd_source}")
    print(f"Output dir:     {OUTPUT_REPORT_DIR}")
    print()
    print(cost_summary[display_columns].sort_values(
        ["strategy_id", "split", "slippage_bps_per_side"]
    ).to_string(index=False))
    print()
    print("BREAK-EVEN SLIPPAGE")
    print(break_even.to_string(index=False))
    print()
    print("OVERLAPS")
    print("No same-timestamp strategy overlaps." if overlaps.empty else overlaps.to_string(index=False))
    print()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
