import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")

HORIZONS = [5, 15, 30, 45, 60]
RANK_FEATURE = "5m_trend_score"


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern:\n{pattern}")
    return files[-1]


def load_inputs():
    outcomes_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_coin_outcomes_*.csv",
        )
    )

    events_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_independent_events_*.csv",
        )
    )

    outcomes = pd.read_csv(outcomes_file)
    events = pd.read_csv(events_file)

    outcomes["available_at"] = pd.to_datetime(
        outcomes["available_at"],
        utc=True,
        errors="coerce",
    )

    events["available_at"] = pd.to_datetime(
        events["available_at"],
        utc=True,
        errors="coerce",
    )

    events["new_episode"] = (
        events["new_episode"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    outcomes["full_history_universe"] = (
        outcomes["full_history_universe"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    required = {
        "event_id",
        "available_at",
        "split",
        "symbol",
        RANK_FEATURE,
        "return_60m_atr",
        "mfe_60m_atr",
        "mae_60m_atr",
        "trend_up_60m",
        "positive_60m",
    }

    for h in HORIZONS:
        required.add(f"return_{h}m_pct")

    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise RuntimeError(
            "Outcome file missing columns:\n" + "\n".join(missing)
        )

    return outcomes, events, outcomes_file, events_file


def rank_all_coins_per_event(frame):
    """
    Rank ALL available coins in the frozen Top30 universe.
    Rank 1 = highest 5m trend_score.
    No percentile conversion is used.
    """
    out = frame.copy()

    out["coin_rank"] = (
        out.groupby("event_id")[RANK_FEATURE]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    return out


def build_rank_rows(outcomes, events):
    event_modes = {
        "COOLDOWN_60M": set(events["event_id"].astype(int)),
        "NEW_EPISODE": set(
            events.loc[
                events["new_episode"],
                "event_id",
            ].astype(int)
        ),
    }

    universes = {
        "ALL_AVAILABLE_TOP30": outcomes.copy(),
        "FULL_HISTORY_ONLY": outcomes.loc[
            outcomes["full_history_universe"]
        ].copy(),
    }

    rows = []

    for universe_name, base in universes.items():
        for event_mode, event_ids in event_modes.items():
            subset = base.loc[
                base["event_id"].astype(int).isin(event_ids)
            ].copy()

            ranked = rank_all_coins_per_event(subset)

            for _, row in ranked.iterrows():
                record = {
                    "universe": universe_name,
                    "event_mode": event_mode,
                    "event_id": int(row["event_id"]),
                    "available_at": row["available_at"],
                    "split": row["split"],
                    "symbol": row["symbol"],
                    "coin_rank": int(row["coin_rank"]),
                    RANK_FEATURE: row[RANK_FEATURE],
                    "return_60m_atr": row["return_60m_atr"],
                    "mfe_60m_atr": row["mfe_60m_atr"],
                    "mae_60m_atr": row["mae_60m_atr"],
                    "trend_up_60m": row["trend_up_60m"],
                    "positive_60m": row["positive_60m"],
                }

                for h in HORIZONS:
                    record[f"return_{h}m_pct"] = row[
                        f"return_{h}m_pct"
                    ]

                rows.append(record)

    return pd.DataFrame(rows)


def summarize_exact_rank(rank_rows):
    rows = []

    group_cols = [
        "universe",
        "event_mode",
        "split",
        "coin_rank",
    ]

    for keys, group in rank_rows.groupby(
        group_cols,
        observed=True,
    ):
        universe, event_mode, split, coin_rank = keys

        row = {
            "universe": universe,
            "event_mode": event_mode,
            "split": split,
            "coin_rank": int(coin_rank),
            "observations": int(len(group)),
            "events": int(group["event_id"].nunique()),
            "symbols_seen": int(group["symbol"].nunique()),
            "mean_return_60m_atr": float(
                group["return_60m_atr"].mean()
            ),
            "median_return_60m_atr": float(
                group["return_60m_atr"].median()
            ),
            "mean_mfe_60m_atr": float(
                group["mfe_60m_atr"].mean()
            ),
            "mean_mae_60m_atr": float(
                group["mae_60m_atr"].mean()
            ),
            "trend_up_rate_60m": float(
                group["trend_up_60m"].mean()
            ),
            "positive_rate_60m": float(
                group["positive_60m"].mean()
            ),
        }

        for h in HORIZONS:
            col = f"return_{h}m_pct"
            row[f"mean_return_{h}m_pct"] = float(
                group[col].mean()
            )
            row[f"median_return_{h}m_pct"] = float(
                group[col].median()
            )
            row[f"positive_rate_{h}m"] = float(
                (group[col] > 0).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_cumulative_top_n(rank_rows):
    """
    Equal-weight baskets Top1, Top2, ... Top30.
    This analyzes absolute rank counts, not percentages.
    """
    rows = []

    max_rank = int(rank_rows["coin_rank"].max())

    for (
        universe,
        event_mode,
        split,
        event_id,
        event_time,
    ), event_group in rank_rows.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "event_id",
            "available_at",
        ],
        observed=True,
    ):
        event_group = event_group.sort_values("coin_rank")
        available = len(event_group)

        for top_n in range(1, min(max_rank, available) + 1):
            selected = event_group.loc[
                event_group["coin_rank"] <= top_n
            ]

            row = {
                "universe": universe,
                "event_mode": event_mode,
                "split": split,
                "event_id": int(event_id),
                "available_at": event_time,
                "top_n": int(top_n),
                "available_coins": int(available),
                "selected_coins": int(len(selected)),
                "selected_symbols": ",".join(
                    selected.sort_values("coin_rank")["symbol"]
                    .astype(str)
                    .tolist()
                ),
                "mean_return_60m_atr": float(
                    selected["return_60m_atr"].mean()
                ),
                "mean_mfe_60m_atr": float(
                    selected["mfe_60m_atr"].mean()
                ),
                "mean_mae_60m_atr": float(
                    selected["mae_60m_atr"].mean()
                ),
                "trend_up_rate_60m": float(
                    selected["trend_up_60m"].mean()
                ),
                "positive_rate_60m": float(
                    selected["positive_60m"].mean()
                ),
            }

            for h in HORIZONS:
                col = f"return_{h}m_pct"
                row[f"mean_return_{h}m_pct"] = float(
                    selected[col].mean()
                )
                row[f"positive_rate_{h}m"] = float(
                    (selected[col] > 0).mean()
                )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_cumulative_top_n(cumulative):
    rows = []

    for (
        universe,
        event_mode,
        split,
        top_n,
    ), group in cumulative.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "top_n",
        ],
        observed=True,
    ):
        row = {
            "universe": universe,
            "event_mode": event_mode,
            "split": split,
            "top_n": int(top_n),
            "events": int(group["event_id"].nunique()),
            "avg_available_coins": float(
                group["available_coins"].mean()
            ),
            "mean_return_60m_atr": float(
                group["mean_return_60m_atr"].mean()
            ),
            "median_event_return_60m_atr": float(
                group["mean_return_60m_atr"].median()
            ),
            "mean_mfe_60m_atr": float(
                group["mean_mfe_60m_atr"].mean()
            ),
            "mean_mae_60m_atr": float(
                group["mean_mae_60m_atr"].mean()
            ),
            "mean_trend_up_rate_60m": float(
                group["trend_up_rate_60m"].mean()
            ),
            "mean_positive_rate_60m": float(
                group["positive_rate_60m"].mean()
            ),
        }

        for h in HORIZONS:
            row[f"mean_return_{h}m_pct"] = float(
                group[f"mean_return_{h}m_pct"].mean()
            )
            row[f"mean_positive_rate_{h}m"] = float(
                group[f"positive_rate_{h}m"].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def add_vs_all30_uplift(summary):
    out_parts = []

    for (
        universe,
        event_mode,
        split,
    ), group in summary.groupby(
        [
            "universe",
            "event_mode",
            "split",
        ],
        observed=True,
    ):
        g = group.copy()

        # Baseline = largest available Top-N row for that slice.
        baseline_row = g.loc[
            g["top_n"].idxmax()
        ]

        baseline_atr = float(
            baseline_row["mean_return_60m_atr"]
        )

        g["baseline_top_n"] = int(
            baseline_row["top_n"]
        )

        g["uplift_vs_all_available_60m_atr"] = (
            g["mean_return_60m_atr"]
            - baseline_atr
        )

        for h in HORIZONS:
            baseline_h = float(
                baseline_row[f"mean_return_{h}m_pct"]
            )

            g[f"uplift_vs_all_available_{h}m_pct"] = (
                g[f"mean_return_{h}m_pct"]
                - baseline_h
            )

        out_parts.append(g)

    return pd.concat(
        out_parts,
        ignore_index=True,
    )


def build_symbol_frequency(rank_rows):
    rows = []

    for (
        universe,
        event_mode,
        split,
        symbol,
    ), group in rank_rows.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "symbol",
        ],
        observed=True,
    ):
        rows.append(
            {
                "universe": universe,
                "event_mode": event_mode,
                "split": split,
                "symbol": symbol,
                "observations": int(len(group)),
                "mean_rank": float(group["coin_rank"].mean()),
                "median_rank": float(group["coin_rank"].median()),
                "times_rank_1": int(
                    (group["coin_rank"] == 1).sum()
                ),
                "times_top_3": int(
                    (group["coin_rank"] <= 3).sum()
                ),
                "times_top_5": int(
                    (group["coin_rank"] <= 5).sum()
                ),
                "mean_return_60m_atr": float(
                    group["return_60m_atr"].mean()
                ),
                "positive_rate_60m": float(
                    group["positive_60m"].mean()
                ),
                "trend_up_rate_60m": float(
                    group["trend_up_60m"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    (
        outcomes,
        events,
        outcomes_file,
        events_file,
    ) = load_inputs()

    print()
    print("=" * 120)
    print("MARKET TRIGGER A - TOP30 ABSOLUTE RANK ANALYSIS")
    print("=" * 120)
    print(f"Outcomes source: {outcomes_file}")
    print(f"Events source:   {events_file}")
    print()
    print("IMPORTANT:")
    print("  No percentages are used for coin selection.")
    print("  Every available coin in the frozen Top30 universe is ranked.")
    print("  Rank 1 = strongest 5m trend_score.")
    print("  We analyze exact Rank 1..30 and cumulative Top1..Top30.")
    print("=" * 120)

    rank_rows = build_rank_rows(
        outcomes,
        events,
    )

    exact_rank_summary = summarize_exact_rank(
        rank_rows
    )

    cumulative = build_cumulative_top_n(
        rank_rows
    )

    cumulative_summary = summarize_cumulative_top_n(
        cumulative
    )

    cumulative_summary = add_vs_all30_uplift(
        cumulative_summary
    )

    symbol_frequency = build_symbol_frequency(
        rank_rows
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    rank_rows_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_rank_rows_{stamp}.csv",
    )

    exact_rank_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_exact_rank_summary_{stamp}.csv",
    )

    cumulative_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_cumulative_event_results_{stamp}.csv",
    )

    cumulative_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_cumulative_summary_{stamp}.csv",
    )

    symbol_frequency_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_symbol_frequency_{stamp}.csv",
    )

    rank_rows.to_csv(
        rank_rows_file,
        index=False,
    )

    exact_rank_summary.to_csv(
        exact_rank_file,
        index=False,
    )

    cumulative.to_csv(
        cumulative_file,
        index=False,
    )

    cumulative_summary.to_csv(
        cumulative_summary_file,
        index=False,
    )

    symbol_frequency.to_csv(
        symbol_frequency_file,
        index=False,
    )

    print()
    print("=" * 170)
    print("FULL-HISTORY / NEW EPISODE / CUMULATIVE TOP-N")
    print("=" * 170)

    show = cumulative_summary.loc[
        (
            cumulative_summary["universe"]
            == "FULL_HISTORY_ONLY"
        )
        &
        (
            cumulative_summary["event_mode"]
            == "NEW_EPISODE"
        )
    ].copy()

    cols = [
        "split",
        "top_n",
        "events",
        "avg_available_coins",
        "mean_return_15m_pct",
        "mean_return_30m_pct",
        "mean_return_45m_pct",
        "mean_return_60m_pct",
        "mean_return_60m_atr",
        "uplift_vs_all_available_60m_atr",
        "mean_positive_rate_60m",
        "mean_trend_up_rate_60m",
        "mean_mfe_60m_atr",
        "mean_mae_60m_atr",
    ]

    print(
        show[cols]
        .sort_values(
            ["split", "top_n"]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 170)
    print("EXACT INDIVIDUAL RANK PERFORMANCE - FULL HISTORY / NEW EPISODE")
    print("=" * 170)

    show_rank = exact_rank_summary.loc[
        (
            exact_rank_summary["universe"]
            == "FULL_HISTORY_ONLY"
        )
        &
        (
            exact_rank_summary["event_mode"]
            == "NEW_EPISODE"
        )
    ].copy()

    rank_cols = [
        "split",
        "coin_rank",
        "events",
        "mean_return_60m_pct",
        "mean_return_60m_atr",
        "positive_rate_60m",
        "trend_up_rate_60m",
        "mean_mfe_60m_atr",
        "mean_mae_60m_atr",
    ]

    print(
        show_rank[rank_cols]
        .sort_values(
            ["split", "coin_rank"]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Rank rows:          {rank_rows_file}")
    print(f"Exact rank summary: {exact_rank_file}")
    print(f"Cumulative events:  {cumulative_file}")
    print(f"Cumulative summary: {cumulative_summary_file}")
    print(f"Symbol frequency:   {symbol_frequency_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
