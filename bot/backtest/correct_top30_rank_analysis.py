import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")

HORIZONS = [5, 15, 30, 45, 60]

# Absolute rank bands. No percentile conversion.
RANK_BANDS = [
    (1, 5, "RANK_01_05"),
    (6, 10, "RANK_06_10"),
    (11, 15, "RANK_11_15"),
    (16, 20, "RANK_16_20"),
    (21, 25, "RANK_21_25"),
    (26, 30, "RANK_26_30"),
]


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No files found for pattern:\n{pattern}"
        )
    return files[-1]


def load_inputs():
    cumulative_event_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_top30_cumulative_event_results_*.csv",
        )
    )

    rank_rows_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_top30_rank_rows_*.csv",
        )
    )

    cumulative = pd.read_csv(cumulative_event_file)
    ranks = pd.read_csv(rank_rows_file)

    for df in [cumulative, ranks]:
        df["available_at"] = pd.to_datetime(
            df["available_at"],
            utc=True,
            errors="coerce",
        )

    return cumulative, ranks, cumulative_event_file, rank_rows_file


def add_event_matched_baseline(cumulative):
    """
    Critical correction:
    compare Top-N to ALL AVAILABLE coins from the SAME event.
    Never compare a 12-event Top-N average to a 1- or 3-event
    Top29 average.
    """
    key_cols = [
        "universe",
        "event_mode",
        "split",
        "event_id",
        "available_at",
    ]

    # The largest top_n present for each event is exactly the
    # all-available basket for that event.
    baseline = (
        cumulative
        .sort_values("top_n")
        .groupby(
            key_cols,
            observed=True,
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    keep = key_cols + [
        "top_n",
        "mean_return_60m_atr",
    ]

    for h in HORIZONS:
        keep.append(
            f"mean_return_{h}m_pct"
        )

    baseline = baseline[keep].copy()

    rename = {
        "top_n":
            "event_baseline_all_coins",
        "mean_return_60m_atr":
            "event_baseline_return_60m_atr",
    }

    for h in HORIZONS:
        rename[
            f"mean_return_{h}m_pct"
        ] = (
            f"event_baseline_return_{h}m_pct"
        )

    baseline = baseline.rename(
        columns=rename
    )

    out = cumulative.merge(
        baseline,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    out[
        "corrected_uplift_vs_all_60m_atr"
    ] = (
        out["mean_return_60m_atr"]
        - out[
            "event_baseline_return_60m_atr"
        ]
    )

    for h in HORIZONS:
        out[
            f"corrected_uplift_vs_all_{h}m_pct"
        ] = (
            out[
                f"mean_return_{h}m_pct"
            ]
            - out[
                f"event_baseline_return_{h}m_pct"
            ]
        )

    return out


def summarize_corrected_cumulative(corrected):
    rows = []

    group_cols = [
        "universe",
        "event_mode",
        "split",
        "top_n",
    ]

    for keys, group in corrected.groupby(
        group_cols,
        observed=True,
    ):
        universe, event_mode, split, top_n = keys

        row = {
            "universe": universe,
            "event_mode": event_mode,
            "split": split,
            "top_n": int(top_n),
            "events": int(
                group["event_id"].nunique()
            ),
            "avg_available_coins": float(
                group["available_coins"].mean()
            ),
            "mean_return_60m_pct": float(
                group["mean_return_60m_pct"].mean()
            ),
            "mean_return_60m_atr": float(
                group["mean_return_60m_atr"].mean()
            ),
            "median_event_return_60m_atr": float(
                group["mean_return_60m_atr"].median()
            ),
            "mean_corrected_uplift_60m_atr": float(
                group[
                    "corrected_uplift_vs_all_60m_atr"
                ].mean()
            ),
            "median_corrected_uplift_60m_atr": float(
                group[
                    "corrected_uplift_vs_all_60m_atr"
                ].median()
            ),
            "beats_all_available_rate": float(
                (
                    group[
                        "corrected_uplift_vs_all_60m_atr"
                    ]
                    > 0
                ).mean()
            ),
            "mean_positive_rate_60m": float(
                group["positive_rate_60m"].mean()
            ),
            "mean_trend_up_rate_60m": float(
                group["trend_up_rate_60m"].mean()
            ),
            "mean_mfe_60m_atr": float(
                group["mean_mfe_60m_atr"].mean()
            ),
            "mean_mae_60m_atr": float(
                group["mean_mae_60m_atr"].mean()
            ),
        }

        for h in HORIZONS:
            row[
                f"mean_return_{h}m_pct"
            ] = float(
                group[
                    f"mean_return_{h}m_pct"
                ].mean()
            )

            row[
                f"mean_corrected_uplift_{h}m_pct"
            ] = float(
                group[
                    f"corrected_uplift_vs_all_{h}m_pct"
                ].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def add_rank_band(ranks):
    out = ranks.copy()
    out["rank_band"] = None

    for lo, hi, label in RANK_BANDS:
        mask = (
            (out["coin_rank"] >= lo)
            & (out["coin_rank"] <= hi)
        )
        out.loc[
            mask,
            "rank_band"
        ] = label

    return out


def summarize_rank_bands(ranks):
    ranked = add_rank_band(ranks)

    ranked = ranked.loc[
        ranked["rank_band"].notna()
    ].copy()

    rows = []

    for keys, group in ranked.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "rank_band",
        ],
        observed=True,
    ):
        universe, event_mode, split, rank_band = keys

        row = {
            "universe": universe,
            "event_mode": event_mode,
            "split": split,
            "rank_band": rank_band,
            "observations": int(len(group)),
            "events": int(
                group["event_id"].nunique()
            ),
            "mean_return_60m_pct": float(
                group["return_60m_pct"].mean()
            ),
            "mean_return_60m_atr": float(
                group["return_60m_atr"].mean()
            ),
            "median_return_60m_atr": float(
                group["return_60m_atr"].median()
            ),
            "positive_rate_60m": float(
                group["positive_60m"].mean()
            ),
            "trend_up_rate_60m": float(
                group["trend_up_60m"].mean()
            ),
            "mean_mfe_60m_atr": float(
                group["mfe_60m_atr"].mean()
            ),
            "mean_mae_60m_atr": float(
                group["mae_60m_atr"].mean()
            ),
        }

        for h in HORIZONS:
            row[f"mean_return_{h}m_pct"] = float(
                group[
                    f"return_{h}m_pct"
                ].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def spearman_strength_ic(group):
    """
    Positive IC means stronger rank (Rank 1 strongest)
    tends to have higher future return.
    """
    g = group[
        [
            "coin_rank",
            "return_60m_atr",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(g) < 10:
        return np.nan

    if g["coin_rank"].nunique() < 5:
        return np.nan

    strength = -g["coin_rank"].astype(float)

    return float(
        strength.rank(pct=True)
        .corr(
            g["return_60m_atr"]
            .rank(pct=True)
        )
    )


def rank_ic_summary(ranks):
    event_rows = []

    for keys, group in ranks.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "event_id",
            "available_at",
        ],
        observed=True,
    ):
        (
            universe,
            event_mode,
            split,
            event_id,
            available_at,
        ) = keys

        ic = spearman_strength_ic(group)

        event_rows.append(
            {
                "universe": universe,
                "event_mode": event_mode,
                "split": split,
                "event_id": int(event_id),
                "available_at": available_at,
                "coins": int(
                    group["symbol"].nunique()
                ),
                "rank_strength_ic_60m": ic,
            }
        )

    event_ic = pd.DataFrame(event_rows)

    summary_rows = []

    for keys, group in event_ic.groupby(
        [
            "universe",
            "event_mode",
            "split",
        ],
        observed=True,
    ):
        universe, event_mode, split = keys

        valid = group[
            "rank_strength_ic_60m"
        ].dropna()

        if valid.empty:
            continue

        summary_rows.append(
            {
                "universe": universe,
                "event_mode": event_mode,
                "split": split,
                "events_used": int(len(valid)),
                "mean_rank_strength_ic_60m": float(
                    valid.mean()
                ),
                "median_rank_strength_ic_60m": float(
                    valid.median()
                ),
                "positive_ic_event_rate": float(
                    (valid > 0).mean()
                ),
                "mean_abs_ic": float(
                    valid.abs().mean()
                ),
            }
        )

    return event_ic, pd.DataFrame(summary_rows)


def main():
    (
        cumulative,
        ranks,
        cumulative_event_file,
        rank_rows_file,
    ) = load_inputs()

    print()
    print("=" * 120)
    print("TOP30 ABSOLUTE RANK ANALYSIS - CORRECTED EVENT-MATCHED BASELINE")
    print("=" * 120)
    print(f"Cumulative source: {cumulative_event_file}")
    print(f"Rank source:       {rank_rows_file}")
    print()
    print("Correction:")
    print("  Every Top-N basket is compared against ALL AVAILABLE")
    print("  coins from the SAME event.")
    print("  No percentile selection is used.")
    print("=" * 120)

    corrected = add_event_matched_baseline(
        cumulative
    )

    corrected_summary = (
        summarize_corrected_cumulative(
            corrected
        )
    )

    rank_bands = summarize_rank_bands(
        ranks
    )

    event_ic, ic_summary = rank_ic_summary(
        ranks
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    corrected_events_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_corrected_event_results_{stamp}.csv",
    )

    corrected_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_corrected_summary_{stamp}.csv",
    )

    rank_bands_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_rank_bands_{stamp}.csv",
    )

    rank_ic_events_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_rank_ic_events_{stamp}.csv",
    )

    rank_ic_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_top30_rank_ic_summary_{stamp}.csv",
    )

    corrected.to_csv(
        corrected_events_file,
        index=False,
    )

    corrected_summary.to_csv(
        corrected_summary_file,
        index=False,
    )

    rank_bands.to_csv(
        rank_bands_file,
        index=False,
    )

    event_ic.to_csv(
        rank_ic_events_file,
        index=False,
    )

    ic_summary.to_csv(
        rank_ic_summary_file,
        index=False,
    )

    print()
    print("=" * 160)
    print("ALL AVAILABLE TOP30 - NEW EPISODE - TOP1..TOP12")
    print("=" * 160)

    show = corrected_summary.loc[
        (
            corrected_summary["universe"]
            == "ALL_AVAILABLE_TOP30"
        )
        &
        (
            corrected_summary["event_mode"]
            == "NEW_EPISODE"
        )
        &
        (
            corrected_summary["top_n"]
            <= 12
        )
    ].copy()

    print(
        show[
            [
                "split",
                "top_n",
                "events",
                "avg_available_coins",
                "mean_return_60m_pct",
                "mean_return_60m_atr",
                "mean_corrected_uplift_60m_atr",
                "median_corrected_uplift_60m_atr",
                "beats_all_available_rate",
                "mean_positive_rate_60m",
                "mean_trend_up_rate_60m",
            ]
        ]
        .sort_values(
            [
                "split",
                "top_n",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 160)
    print("ALL AVAILABLE TOP30 - ABSOLUTE RANK BANDS")
    print("=" * 160)

    show_bands = rank_bands.loc[
        rank_bands[
            "universe"
        ]
        == "ALL_AVAILABLE_TOP30"
    ].copy()

    print(
        show_bands[
            [
                "event_mode",
                "split",
                "rank_band",
                "events",
                "mean_return_60m_pct",
                "mean_return_60m_atr",
                "positive_rate_60m",
                "trend_up_rate_60m",
            ]
        ]
        .sort_values(
            [
                "event_mode",
                "split",
                "rank_band",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("RANK INFORMATION COEFFICIENT")
    print("=" * 120)
    print(
        ic_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Corrected events:  {corrected_events_file}")
    print(f"Corrected summary: {corrected_summary_file}")
    print(f"Rank bands:        {rank_bands_file}")
    print(f"Rank IC events:    {rank_ic_events_file}")
    print(f"Rank IC summary:   {rank_ic_summary_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
