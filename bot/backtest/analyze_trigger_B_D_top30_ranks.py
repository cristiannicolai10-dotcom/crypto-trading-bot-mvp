import gc
import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
DATA_DIR = os.path.join(BASE_DIR, "data", "trend_research_3m")
FEATURE_DIR = os.path.join(DATA_DIR, "features_90d_lowmem")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Frozen trigger candidates from the trigger-library discovery.
TRIGGERS = {
    "TRIGGER_B_LONG": {
        "candidate_id": "LONG_D4_L12",
        "direction": "LONG",
    },
    "TRIGGER_D_SHORT": {
        "candidate_id": "SHORT_D3_L9",
        "direction": "SHORT",
    },
}

RANK_FEATURE = "5m_trend_score"

RANK_BANDS = [
    (1, 5, "RANK_01_05"),
    (6, 10, "RANK_06_10"),
    (11, 15, "RANK_11_15"),
    (16, 20, "RANK_16_20"),
    (21, 25, "RANK_21_25"),
    (26, 30, "RANK_26_30"),
]


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern:\n{pattern}")
    return files[-1]


def load_candidate_events():
    path = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_library_candidate_events_*.csv",
        )
    )

    df = pd.read_csv(path)

    df["available_at"] = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce",
    )

    wanted_ids = {
        cfg["candidate_id"]
        for cfg in TRIGGERS.values()
    }

    df = df.loc[
        df["candidate_id"].isin(wanted_ids)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No rows found for frozen Trigger B / Trigger D."
        )

    # Attach readable trigger name.
    reverse = {
        cfg["candidate_id"]: name
        for name, cfg in TRIGGERS.items()
    }

    df["trigger_name"] = df["candidate_id"].map(reverse)

    return df, path


def feature_files():
    files = sorted(
        glob.glob(
            os.path.join(
                FEATURE_DIR,
                "*_features.parquet",
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No feature files in {FEATURE_DIR}"
        )

    return files


def build_coin_event_panel(events):
    event_times = set(
        events["available_at"].dropna().tolist()
    )

    files = feature_files()
    parts = []

    needed = [
        "available_at",
        RANK_FEATURE,
        "future_1h_return_pct",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
        "trend_class",
    ]

    print()
    print("=" * 120)
    print("BUILDING TRIGGER B / D x TOP30 PANEL")
    print("=" * 120)

    for idx, path in enumerate(files, start=1):
        symbol = (
            os.path.basename(path)
            .replace("_features.parquet", "")
        )

        df = pd.read_parquet(
            path,
            columns=needed,
        )

        df["available_at"] = pd.to_datetime(
            df["available_at"],
            utc=True,
            errors="coerce",
        )

        df = df.loc[
            df["available_at"].isin(event_times)
        ].copy()

        if df.empty:
            continue

        df["symbol"] = symbol

        parts.append(df)

        print(
            f"[{idx:02d}/{len(files):02d}] "
            f"{symbol}: {len(df)} event rows"
        )

    if not parts:
        raise RuntimeError(
            "No coin rows matched trigger event timestamps."
        )

    coins = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    # Merge trigger metadata. A timestamp may belong to both B and D,
    # which is fine and should create separate trigger observations.
    trigger_meta = events[
        [
            "candidate_id",
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "available_at",
        ]
    ].drop_duplicates()

    panel = trigger_meta.merge(
        coins,
        on="available_at",
        how="inner",
        validate="many_to_many",
    )

    return panel


def add_directional_metrics(panel):
    out = panel.copy()

    is_long = out["direction"] == "LONG"

    out["direction_return_pct"] = np.where(
        is_long,
        out["future_1h_return_pct"],
        -out["future_1h_return_pct"],
    )

    out["direction_return_atr"] = np.where(
        is_long,
        out["future_1h_return_atr"],
        -out["future_1h_return_atr"],
    )

    # Favorable/adverse excursion expressed from the trigger direction.
    out["direction_mfe_atr"] = np.where(
        is_long,
        out["future_1h_mfe_atr"],
        -out["future_1h_mae_atr"],
    )

    out["direction_mae_atr"] = np.where(
        is_long,
        out["future_1h_mae_atr"],
        -out["future_1h_mfe_atr"],
    )

    trend = out["trend_class"].astype(str)

    out["direction_hit"] = np.where(
        is_long,
        trend == "UP",
        trend == "DOWN",
    ).astype(np.int8)

    out["direction_positive"] = (
        out["direction_return_atr"] > 0
    ).astype(np.int8)

    return out


def rank_all_top30(panel):
    out_parts = []

    keys = [
        "trigger_name",
        "candidate_id",
        "direction",
        "split",
        "event_mode",
        "available_at",
    ]

    for _, group in panel.groupby(
        keys,
        observed=True,
        sort=False,
    ):
        g = group.copy()

        g = g.dropna(
            subset=[
                RANK_FEATURE,
                "direction_return_atr",
            ]
        )

        if len(g) < 10:
            continue

        # Rank 1 = highest 5m trend_score.
        g["coin_rank"] = (
            g[RANK_FEATURE]
            .rank(
                ascending=False,
                method="first",
            )
            .astype(int)
        )

        g["coins_available"] = len(g)

        out_parts.append(g)

    if not out_parts:
        raise RuntimeError(
            "No rankable event groups were created."
        )

    out = pd.concat(
        out_parts,
        ignore_index=True,
    )

    del out_parts
    gc.collect()

    return out


def add_rank_band(rank_rows):
    out = rank_rows.copy()
    out["rank_band"] = None

    for lo, hi, label in RANK_BANDS:
        mask = (
            (out["coin_rank"] >= lo)
            & (out["coin_rank"] <= hi)
        )
        out.loc[mask, "rank_band"] = label

    return out


def exact_rank_summary(rank_rows):
    rows = []

    for keys, group in rank_rows.groupby(
        [
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "coin_rank",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            split,
            event_mode,
            coin_rank,
        ) = keys

        rows.append(
            {
                "trigger_name": trigger_name,
                "direction": direction,
                "split": split,
                "event_mode": event_mode,
                "coin_rank": int(coin_rank),
                "observations": len(group),
                "events": int(
                    group["available_at"].nunique()
                ),
                "symbols_seen": int(
                    group["symbol"].nunique()
                ),
                "mean_direction_return_pct": float(
                    group["direction_return_pct"].mean()
                ),
                "mean_direction_return_atr": float(
                    group["direction_return_atr"].mean()
                ),
                "median_direction_return_atr": float(
                    group["direction_return_atr"].median()
                ),
                "positive_direction_rate": float(
                    group["direction_positive"].mean()
                ),
                "direction_hit_rate": float(
                    group["direction_hit"].mean()
                ),
                "mean_direction_mfe_atr": float(
                    group["direction_mfe_atr"].mean()
                ),
                "mean_direction_mae_atr": float(
                    group["direction_mae_atr"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def rank_band_summary(rank_rows):
    ranked = add_rank_band(rank_rows)

    ranked = ranked.loc[
        ranked["rank_band"].notna()
    ].copy()

    rows = []

    for keys, group in ranked.groupby(
        [
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "rank_band",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            split,
            event_mode,
            rank_band,
        ) = keys

        rows.append(
            {
                "trigger_name": trigger_name,
                "direction": direction,
                "split": split,
                "event_mode": event_mode,
                "rank_band": rank_band,
                "observations": len(group),
                "events": int(
                    group["available_at"].nunique()
                ),
                "mean_direction_return_pct": float(
                    group["direction_return_pct"].mean()
                ),
                "mean_direction_return_atr": float(
                    group["direction_return_atr"].mean()
                ),
                "median_direction_return_atr": float(
                    group["direction_return_atr"].median()
                ),
                "positive_direction_rate": float(
                    group["direction_positive"].mean()
                ),
                "direction_hit_rate": float(
                    group["direction_hit"].mean()
                ),
                "mean_direction_mfe_atr": float(
                    group["direction_mfe_atr"].mean()
                ),
                "mean_direction_mae_atr": float(
                    group["direction_mae_atr"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def spearman_strength_ic(group):
    g = group[
        [
            "coin_rank",
            "direction_return_atr",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(g) < 10:
        return np.nan

    # Positive IC = stronger 5m trend_score coins are better
    # for the trigger direction.
    strength = -g["coin_rank"].astype(float)

    return float(
        strength.rank(pct=True)
        .corr(
            g["direction_return_atr"]
            .rank(pct=True)
        )
    )


def rank_ic_summary(rank_rows):
    event_rows = []

    for keys, group in rank_rows.groupby(
        [
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "available_at",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            split,
            event_mode,
            available_at,
        ) = keys

        ic = spearman_strength_ic(group)

        event_rows.append(
            {
                "trigger_name": trigger_name,
                "direction": direction,
                "split": split,
                "event_mode": event_mode,
                "available_at": available_at,
                "coins_available": int(
                    group["coins_available"].max()
                ),
                "rank_strength_ic": ic,
            }
        )

    event_ic = pd.DataFrame(event_rows)

    rows = []

    for keys, group in event_ic.groupby(
        [
            "trigger_name",
            "direction",
            "split",
            "event_mode",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            split,
            event_mode,
        ) = keys

        s = group["rank_strength_ic"].dropna()

        if s.empty:
            continue

        rows.append(
            {
                "trigger_name": trigger_name,
                "direction": direction,
                "split": split,
                "event_mode": event_mode,
                "events_used": len(s),
                "mean_rank_strength_ic": float(s.mean()),
                "median_rank_strength_ic": float(s.median()),
                "positive_ic_event_rate": float(
                    (s > 0).mean()
                ),
                "negative_ic_event_rate": float(
                    (s < 0).mean()
                ),
                "mean_abs_ic": float(
                    s.abs().mean()
                ),
            }
        )

    return event_ic, pd.DataFrame(rows)


def cumulative_baskets(rank_rows):
    rows = []

    group_keys = [
        "trigger_name",
        "candidate_id",
        "direction",
        "split",
        "event_mode",
        "available_at",
    ]

    for keys, group in rank_rows.groupby(
        group_keys,
        observed=True,
        sort=False,
    ):
        (
            trigger_name,
            candidate_id,
            direction,
            split,
            event_mode,
            available_at,
        ) = keys

        g = group.sort_values("coin_rank").copy()
        available = len(g)

        # Same-event baseline = every available coin.
        baseline_return = float(
            g["direction_return_atr"].mean()
        )

        for side in [
            "STRONGEST",
            "WEAKEST",
        ]:
            for n in range(1, available + 1):
                if side == "STRONGEST":
                    selected = g.head(n)
                else:
                    selected = g.tail(n)

                mean_dir = float(
                    selected["direction_return_atr"].mean()
                )

                rows.append(
                    {
                        "trigger_name": trigger_name,
                        "candidate_id": candidate_id,
                        "direction": direction,
                        "split": split,
                        "event_mode": event_mode,
                        "available_at": available_at,
                        "selector_side": side,
                        "top_n": n,
                        "available_coins": available,
                        "selected_coins": len(selected),
                        "selected_symbols": ",".join(
                            selected["symbol"].astype(str).tolist()
                        ),
                        "mean_direction_return_pct": float(
                            selected["direction_return_pct"].mean()
                        ),
                        "mean_direction_return_atr": mean_dir,
                        "median_direction_return_atr": float(
                            selected["direction_return_atr"].median()
                        ),
                        "positive_direction_rate": float(
                            selected["direction_positive"].mean()
                        ),
                        "direction_hit_rate": float(
                            selected["direction_hit"].mean()
                        ),
                        "mean_direction_mfe_atr": float(
                            selected["direction_mfe_atr"].mean()
                        ),
                        "mean_direction_mae_atr": float(
                            selected["direction_mae_atr"].mean()
                        ),
                        "all_coins_direction_return_atr":
                            baseline_return,
                        "uplift_vs_all_coins_atr":
                            mean_dir - baseline_return,
                    }
                )

    return pd.DataFrame(rows)


def cumulative_summary(cumulative):
    rows = []

    for keys, group in cumulative.groupby(
        [
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "selector_side",
            "top_n",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            split,
            event_mode,
            selector_side,
            top_n,
        ) = keys

        rows.append(
            {
                "trigger_name": trigger_name,
                "direction": direction,
                "split": split,
                "event_mode": event_mode,
                "selector_side": selector_side,
                "top_n": int(top_n),
                "events": int(
                    group["available_at"].nunique()
                ),
                "avg_available_coins": float(
                    group["available_coins"].mean()
                ),
                "mean_direction_return_pct": float(
                    group["mean_direction_return_pct"].mean()
                ),
                "mean_direction_return_atr": float(
                    group["mean_direction_return_atr"].mean()
                ),
                "median_event_direction_return_atr": float(
                    group["mean_direction_return_atr"].median()
                ),
                "mean_uplift_vs_all_coins_atr": float(
                    group["uplift_vs_all_coins_atr"].mean()
                ),
                "median_uplift_vs_all_coins_atr": float(
                    group["uplift_vs_all_coins_atr"].median()
                ),
                "beats_all_coins_rate": float(
                    (
                        group["uplift_vs_all_coins_atr"] > 0
                    ).mean()
                ),
                "mean_positive_direction_rate": float(
                    group["positive_direction_rate"].mean()
                ),
                "mean_direction_hit_rate": float(
                    group["direction_hit_rate"].mean()
                ),
                "mean_direction_mfe_atr": float(
                    group["mean_direction_mfe_atr"].mean()
                ),
                "mean_direction_mae_atr": float(
                    group["mean_direction_mae_atr"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    ensure_dirs()

    events, event_source = (
        load_candidate_events()
    )

    print()
    print("=" * 120)
    print("TRIGGER B / D - TOP30 ABSOLUTE RANK TEST")
    print("=" * 120)
    print(f"Candidate events: {event_source}")
    print()
    print("Frozen market triggers:")
    for name, cfg in TRIGGERS.items():
        print(
            f"  {name}: {cfg['candidate_id']} "
            f"({cfg['direction']})"
        )
    print()
    print("Ranking:")
    print(
        "  Rank 1 = highest 5m trend_score among all available Top30 coins"
    )
    print("  Exact rank 1..30")
    print("  Absolute bands 1-5, 6-10, ..., 26-30")
    print("  STRONGEST Top1..Top30")
    print("  WEAKEST   Bottom1..Bottom30")
    print()
    print(
        "Positive rank IC means stronger coins are better for the trigger direction."
    )
    print(
        "Negative rank IC means weaker coins are better for the trigger direction."
    )
    print("=" * 120)

    panel = build_coin_event_panel(
        events
    )

    panel = add_directional_metrics(
        panel
    )

    rank_rows = rank_all_top30(
        panel
    )

    exact = exact_rank_summary(
        rank_rows
    )

    bands = rank_band_summary(
        rank_rows
    )

    ic_events, ic_summary = (
        rank_ic_summary(
            rank_rows
        )
    )

    cumulative = cumulative_baskets(
        rank_rows
    )

    cum_summary = cumulative_summary(
        cumulative
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    rank_rows_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_rank_rows_{stamp}.csv",
    )

    exact_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_exact_rank_summary_{stamp}.csv",
    )

    bands_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_rank_bands_{stamp}.csv",
    )

    ic_events_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_rank_ic_events_{stamp}.csv",
    )

    ic_summary_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_rank_ic_summary_{stamp}.csv",
    )

    cumulative_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_cumulative_events_{stamp}.csv",
    )

    cumulative_summary_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_top30_cumulative_summary_{stamp}.csv",
    )

    rank_rows.to_csv(
        rank_rows_file,
        index=False,
    )

    exact.to_csv(
        exact_file,
        index=False,
    )

    bands.to_csv(
        bands_file,
        index=False,
    )

    ic_events.to_csv(
        ic_events_file,
        index=False,
    )

    ic_summary.to_csv(
        ic_summary_file,
        index=False,
    )

    cumulative.to_csv(
        cumulative_file,
        index=False,
    )

    cum_summary.to_csv(
        cumulative_summary_file,
        index=False,
    )

    print()
    print("=" * 150)
    print("RANK IC SUMMARY")
    print("=" * 150)
    print(
        ic_summary.to_string(index=False)
    )

    print()
    print("=" * 150)
    print("ABSOLUTE RANK BANDS")
    print("=" * 150)
    print(
        bands[
            [
                "trigger_name",
                "direction",
                "split",
                "event_mode",
                "rank_band",
                "events",
                "mean_direction_return_pct",
                "mean_direction_return_atr",
                "positive_direction_rate",
                "direction_hit_rate",
            ]
        ]
        .sort_values(
            [
                "trigger_name",
                "split",
                "event_mode",
                "rank_band",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 150)
    print("CUMULATIVE N=1..10")
    print("=" * 150)

    show = cum_summary.loc[
        cum_summary["top_n"] <= 10
    ].copy()

    print(
        show[
            [
                "trigger_name",
                "direction",
                "split",
                "event_mode",
                "selector_side",
                "top_n",
                "events",
                "mean_direction_return_pct",
                "mean_direction_return_atr",
                "mean_uplift_vs_all_coins_atr",
                "beats_all_coins_rate",
                "mean_positive_direction_rate",
                "mean_direction_hit_rate",
            ]
        ]
        .sort_values(
            [
                "trigger_name",
                "split",
                "event_mode",
                "selector_side",
                "top_n",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Rank rows:          {rank_rows_file}")
    print(f"Exact ranks:        {exact_file}")
    print(f"Rank bands:         {bands_file}")
    print(f"Rank IC events:     {ic_events_file}")
    print(f"Rank IC summary:    {ic_summary_file}")
    print(f"Cumulative events:  {cumulative_file}")
    print(f"Cumulative summary: {cumulative_summary_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
