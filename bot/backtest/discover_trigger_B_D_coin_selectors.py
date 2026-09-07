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

PRIMARY_EVENT_MODE = "NEW_EPISODE"
TOP_DISCOVERY_FEATURES_PER_TRIGGER = 12
MIN_COINS_PER_EVENT = 10
MIN_DISCOVERY_EVENTS = {
    "TRIGGER_B_LONG": 20,
    "TRIGGER_D_SHORT": 80,
}

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


def load_trigger_events():
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

    reverse = {
        cfg["candidate_id"]: name
        for name, cfg in TRIGGERS.items()
    }

    wanted = set(reverse)

    df = df.loc[
        df["candidate_id"].isin(wanted)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No Trigger B / D events found."
        )

    df["trigger_name"] = (
        df["candidate_id"].map(reverse)
    )

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
            f"No feature files found in {FEATURE_DIR}"
        )

    return files


def is_candidate_feature(col, dtype):
    if not pd.api.types.is_numeric_dtype(dtype):
        return False

    excluded_exact = {
        "future_1h_return_pct",
        "future_1h_mfe_pct",
        "future_1h_mae_pct",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
        "target_up",
        "target_down",
    }

    if col in excluded_exact:
        return False

    # Coin-specific current features only.
    # Common BTC state is identical across coins at one event
    # and cannot select among Top30.
    allowed_prefixes = (
        "5m_",
        "1h_",
        "4h_",
        "btc_correlation_",
        "relative_strength_",
    )

    if not col.startswith(allowed_prefixes):
        return False

    # Do not use already-percentile/rank engineered columns.
    if col.startswith("rank_"):
        return False

    # Avoid raw price-scale levels if present.
    raw_suffixes = (
        "_ema50",
        "_ema200",
        "_alligator_jaw",
        "_alligator_teeth",
        "_alligator_lips",
        "_last_fractal_high",
        "_last_fractal_low",
        "_atr",
    )

    if col.endswith(raw_suffixes) and not col.endswith(
        "_atr_pct"
    ):
        return False

    return True


def discover_feature_columns(path):
    sample = pd.read_parquet(path)

    cols = []

    for col in sample.columns:
        if is_candidate_feature(
            col,
            sample[col].dtype,
        ):
            cols.append(col)

    required = [
        "available_at",
        "future_1h_return_pct",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
        "trend_class",
    ]

    missing = [
        c for c in required
        if c not in sample.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    del sample
    gc.collect()

    return sorted(set(cols)), required


def build_event_coin_panel(
    trigger_events,
):
    files = feature_files()

    feature_cols, required = (
        discover_feature_columns(
            files[0]
        )
    )

    event_times = set(
        trigger_events[
            "available_at"
        ].dropna().tolist()
    )

    read_cols = list(
        dict.fromkeys(
            required + feature_cols
        )
    )

    parts = []

    print()
    print("=" * 120)
    print("BUILDING B / D CONDITIONAL COIN FEATURE PANEL")
    print("=" * 120)
    print(f"Candidate coin features: {len(feature_cols)}")

    for idx, path in enumerate(
        files,
        start=1,
    ):
        symbol = (
            os.path.basename(path)
            .replace("_features.parquet", "")
        )

        df = pd.read_parquet(
            path,
            columns=read_cols,
        )

        df["available_at"] = pd.to_datetime(
            df["available_at"],
            utc=True,
            errors="coerce",
        )

        df = df.loc[
            df["available_at"].isin(
                event_times
            )
        ].copy()

        if df.empty:
            continue

        df["symbol"] = symbol

        parts.append(df)

        print(
            f"[{idx:02d}/{len(files):02d}] "
            f"{symbol}: {len(df)} rows"
        )

    if not parts:
        raise RuntimeError(
            "No feature rows matched event timestamps."
        )

    coin = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    meta = trigger_events[
        [
            "candidate_id",
            "trigger_name",
            "direction",
            "split",
            "event_mode",
            "available_at",
        ]
    ].drop_duplicates()

    panel = meta.merge(
        coin,
        on="available_at",
        how="inner",
        validate="many_to_many",
    )

    is_long = (
        panel["direction"] == "LONG"
    )

    panel["direction_return_pct"] = np.where(
        is_long,
        panel["future_1h_return_pct"],
        -panel["future_1h_return_pct"],
    )

    panel["direction_return_atr"] = np.where(
        is_long,
        panel["future_1h_return_atr"],
        -panel["future_1h_return_atr"],
    )

    panel["direction_mfe_atr"] = np.where(
        is_long,
        panel["future_1h_mfe_atr"],
        -panel["future_1h_mae_atr"],
    )

    panel["direction_mae_atr"] = np.where(
        is_long,
        panel["future_1h_mae_atr"],
        -panel["future_1h_mfe_atr"],
    )

    panel["direction_positive"] = (
        panel["direction_return_atr"]
        > 0
    ).astype(np.int8)

    trend = (
        panel["trend_class"]
        .astype(str)
    )

    panel["direction_hit"] = np.where(
        is_long,
        trend == "UP",
        trend == "DOWN",
    ).astype(np.int8)

    return (
        panel,
        feature_cols,
    )


def spearman_ic(x, y):
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(pair) < MIN_COINS_PER_EVENT:
        return np.nan

    if pair["x"].nunique() < 3:
        return np.nan

    return float(
        pair["x"]
        .rank(
            pct=True,
            method="average",
        )
        .corr(
            pair["y"]
            .rank(
                pct=True,
                method="average",
            )
        )
    )


def feature_ic_scan(
    panel,
    feature_cols,
):
    event_rows = []

    primary = panel.loc[
        panel["event_mode"]
        == PRIMARY_EVENT_MODE
    ].copy()

    for trigger_name in TRIGGERS:
        trig = primary.loc[
            primary["trigger_name"]
            == trigger_name
        ]

        print()
        print(
            f"IC scan {trigger_name}: "
            f"{len(feature_cols)} features"
        )

        for idx, feature in enumerate(
            feature_cols,
            start=1,
        ):
            if idx % 10 == 0:
                print(
                    f"  {idx}/{len(feature_cols)}"
                )

            for split in [
                "DISCOVERY_60D",
                "VALIDATION_30D",
            ]:
                split_df = trig.loc[
                    trig["split"] == split
                ]

                for (
                    event_time,
                    group,
                ) in split_df.groupby(
                    "available_at",
                    observed=True,
                ):
                    ic = spearman_ic(
                        group[feature],
                        group[
                            "direction_return_atr"
                        ],
                    )

                    if np.isfinite(ic):
                        event_rows.append(
                            {
                                "trigger_name":
                                    trigger_name,
                                "direction":
                                    TRIGGERS[
                                        trigger_name
                                    ]["direction"],
                                "feature":
                                    feature,
                                "split":
                                    split,
                                "available_at":
                                    event_time,
                                "coins":
                                    int(
                                        group[
                                            "symbol"
                                        ].nunique()
                                    ),
                                "ic":
                                    ic,
                            }
                        )

    event_ic = pd.DataFrame(
        event_rows
    )

    if event_ic.empty:
        raise RuntimeError(
            "No feature IC observations were created."
        )

    summary_rows = []

    for (
        trigger_name,
        feature,
        split,
    ), group in event_ic.groupby(
        [
            "trigger_name",
            "feature",
            "split",
        ],
        observed=True,
    ):
        s = group["ic"].dropna()

        summary_rows.append(
            {
                "trigger_name":
                    trigger_name,
                "direction":
                    TRIGGERS[
                        trigger_name
                    ]["direction"],
                "feature":
                    feature,
                "split":
                    split,
                "events_used":
                    len(s),
                "mean_ic":
                    float(s.mean()),
                "median_ic":
                    float(s.median()),
                "positive_ic_rate":
                    float(
                        (s > 0).mean()
                    ),
                "negative_ic_rate":
                    float(
                        (s < 0).mean()
                    ),
                "mean_abs_ic":
                    float(
                        s.abs().mean()
                    ),
            }
        )

    summary_long = pd.DataFrame(
        summary_rows
    )

    pivot = summary_long.pivot(
        index=[
            "trigger_name",
            "direction",
            "feature",
        ],
        columns="split",
        values=[
            "events_used",
            "mean_ic",
            "median_ic",
            "positive_ic_rate",
            "negative_ic_rate",
            "mean_abs_ic",
        ],
    )

    pivot.columns = [
        f"{metric}_{split}"
        for metric, split
        in pivot.columns
    ]

    summary = pivot.reset_index()

    disc = (
        "mean_ic_DISCOVERY_60D"
    )
    val = (
        "mean_ic_VALIDATION_30D"
    )

    summary[
        "orientation"
    ] = np.where(
        summary[disc] >= 0,
        "HIGHER_BETTER",
        "LOWER_BETTER",
    )

    summary[
        "discovery_abs_mean_ic"
    ] = summary[
        disc
    ].abs()

    summary[
        "validation_oriented_ic"
    ] = np.where(
        summary[
            "orientation"
        ]
        == "HIGHER_BETTER",
        summary[val],
        -summary[val],
    )

    summary[
        "same_ic_sign"
    ] = (
        np.sign(
            summary[disc]
        )
        ==
        np.sign(
            summary[val]
        )
    )

    summary[
        "discovery_sign_event_rate"
    ] = np.where(
        summary[
            "orientation"
        ]
        == "HIGHER_BETTER",
        summary[
            "positive_ic_rate_DISCOVERY_60D"
        ],
        summary[
            "negative_ic_rate_DISCOVERY_60D"
        ],
    )

    summary[
        "validation_sign_event_rate"
    ] = np.where(
        summary[
            "orientation"
        ]
        == "HIGHER_BETTER",
        summary[
            "positive_ic_rate_VALIDATION_30D"
        ],
        summary[
            "negative_ic_rate_VALIDATION_30D"
        ],
    )

    # Discovery-only eligibility. Validation is never used here.
    summary[
        "discovery_eligible"
    ] = (
        (
            summary[
                "events_used_DISCOVERY_60D"
            ]
            >= summary[
                "trigger_name"
            ].map(
                MIN_DISCOVERY_EVENTS
            )
        )
        &
        (
            summary[
                "discovery_sign_event_rate"
            ]
            >= 0.55
        )
    )

    summary = summary.sort_values(
        [
            "trigger_name",
            "discovery_eligible",
            "discovery_abs_mean_ic",
            "discovery_sign_event_rate",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return (
        event_ic,
        summary_long,
        summary,
    )


def choose_discovery_features(
    feature_summary,
):
    selected = []

    for trigger_name in TRIGGERS:
        g = feature_summary.loc[
            (
                feature_summary[
                    "trigger_name"
                ]
                == trigger_name
            )
            &
            (
                feature_summary[
                    "discovery_eligible"
                ]
            )
        ].copy()

        g = g.sort_values(
            [
                "discovery_abs_mean_ic",
                "discovery_sign_event_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )

        g = g.head(
            TOP_DISCOVERY_FEATURES_PER_TRIGGER
        )

        selected.append(g)

    if not selected:
        return pd.DataFrame()

    return pd.concat(
        selected,
        ignore_index=True,
    )


def rank_feature_event(
    group,
    feature,
    orientation,
):
    g = group.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            feature,
            "direction_return_atr",
        ]
    ).copy()

    if len(g) < MIN_COINS_PER_EVENT:
        return None

    ascending = (
        orientation
        == "LOWER_BETTER"
    )

    # Rank 1 = best according to the discovery-frozen orientation.
    g["feature_rank"] = (
        g[feature]
        .rank(
            method="first",
            ascending=ascending,
        )
        .astype(int)
    )

    g["coins_available"] = len(g)

    return g


def evaluate_selected_features(
    panel,
    selected_features,
):
    event_rows = []

    for _, sel in selected_features.iterrows():
        trigger_name = sel[
            "trigger_name"
        ]
        feature = sel[
            "feature"
        ]
        orientation = sel[
            "orientation"
        ]

        trig = panel.loc[
            panel[
                "trigger_name"
            ]
            == trigger_name
        ]

        print(
            f"Absolute-rank test: "
            f"{trigger_name} | {feature} | {orientation}"
        )

        for (
            split,
            event_mode,
            event_time,
        ), group in trig.groupby(
            [
                "split",
                "event_mode",
                "available_at",
            ],
            observed=True,
        ):
            ranked = rank_feature_event(
                group,
                feature,
                orientation,
            )

            if ranked is None:
                continue

            baseline = float(
                ranked[
                    "direction_return_atr"
                ].mean()
            )

            available = len(ranked)

            for top_n in range(
                1,
                available + 1,
            ):
                selected = ranked.loc[
                    ranked[
                        "feature_rank"
                    ]
                    <= top_n
                ]

                mean_ret = float(
                    selected[
                        "direction_return_atr"
                    ].mean()
                )

                event_rows.append(
                    {
                        "trigger_name":
                            trigger_name,
                        "direction":
                            sel[
                                "direction"
                            ],
                        "feature":
                            feature,
                        "orientation":
                            orientation,
                        "split":
                            split,
                        "event_mode":
                            event_mode,
                        "available_at":
                            event_time,
                        "top_n":
                            top_n,
                        "available_coins":
                            available,
                        "selected_symbols":
                            ",".join(
                                selected
                                .sort_values(
                                    "feature_rank"
                                )["symbol"]
                                .astype(str)
                                .tolist()
                            ),
                        "mean_direction_return_pct":
                            float(
                                selected[
                                    "direction_return_pct"
                                ].mean()
                            ),
                        "mean_direction_return_atr":
                            mean_ret,
                        "median_direction_return_atr":
                            float(
                                selected[
                                    "direction_return_atr"
                                ].median()
                            ),
                        "positive_direction_rate":
                            float(
                                selected[
                                    "direction_positive"
                                ].mean()
                            ),
                        "direction_hit_rate":
                            float(
                                selected[
                                    "direction_hit"
                                ].mean()
                            ),
                        "mean_direction_mfe_atr":
                            float(
                                selected[
                                    "direction_mfe_atr"
                                ].mean()
                            ),
                        "mean_direction_mae_atr":
                            float(
                                selected[
                                    "direction_mae_atr"
                                ].mean()
                            ),
                        "all_coins_direction_return_atr":
                            baseline,
                        "uplift_vs_all_coins_atr":
                            mean_ret - baseline,
                    }
                )

    event_df = pd.DataFrame(
        event_rows
    )

    rows = []

    for keys, group in event_df.groupby(
        [
            "trigger_name",
            "direction",
            "feature",
            "orientation",
            "split",
            "event_mode",
            "top_n",
        ],
        observed=True,
    ):
        (
            trigger_name,
            direction,
            feature,
            orientation,
            split,
            event_mode,
            top_n,
        ) = keys

        rows.append(
            {
                "trigger_name":
                    trigger_name,
                "direction":
                    direction,
                "feature":
                    feature,
                "orientation":
                    orientation,
                "split":
                    split,
                "event_mode":
                    event_mode,
                "top_n":
                    int(top_n),
                "events":
                    int(
                        group[
                            "available_at"
                        ].nunique()
                    ),
                "avg_available_coins":
                    float(
                        group[
                            "available_coins"
                        ].mean()
                    ),
                "mean_direction_return_pct":
                    float(
                        group[
                            "mean_direction_return_pct"
                        ].mean()
                    ),
                "mean_direction_return_atr":
                    float(
                        group[
                            "mean_direction_return_atr"
                        ].mean()
                    ),
                "median_event_direction_return_atr":
                    float(
                        group[
                            "mean_direction_return_atr"
                        ].median()
                    ),
                "mean_uplift_vs_all_coins_atr":
                    float(
                        group[
                            "uplift_vs_all_coins_atr"
                        ].mean()
                    ),
                "median_uplift_vs_all_coins_atr":
                    float(
                        group[
                            "uplift_vs_all_coins_atr"
                        ].median()
                    ),
                "beats_all_coins_rate":
                    float(
                        (
                            group[
                                "uplift_vs_all_coins_atr"
                            ]
                            > 0
                        ).mean()
                    ),
                "mean_positive_direction_rate":
                    float(
                        group[
                            "positive_direction_rate"
                        ].mean()
                    ),
                "mean_direction_hit_rate":
                    float(
                        group[
                            "direction_hit_rate"
                        ].mean()
                    ),
                "mean_direction_mfe_atr":
                    float(
                        group[
                            "mean_direction_mfe_atr"
                        ].mean()
                    ),
                "mean_direction_mae_atr":
                    float(
                        group[
                            "mean_direction_mae_atr"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return (
        event_df,
        summary,
    )


def add_cross_split_robustness(
    absolute_summary,
):
    primary = absolute_summary.loc[
        absolute_summary[
            "event_mode"
        ]
        == PRIMARY_EVENT_MODE
    ].copy()

    metrics = [
        "events",
        "mean_direction_return_atr",
        "mean_uplift_vs_all_coins_atr",
        "beats_all_coins_rate",
        "mean_positive_direction_rate",
    ]

    pivot = primary.pivot(
        index=[
            "trigger_name",
            "direction",
            "feature",
            "orientation",
            "top_n",
        ],
        columns="split",
        values=metrics,
    )

    pivot.columns = [
        f"{metric}_{split}"
        for metric, split
        in pivot.columns
    ]

    out = pivot.reset_index()

    disc_uplift = (
        "mean_uplift_vs_all_coins_atr_DISCOVERY_60D"
    )
    val_uplift = (
        "mean_uplift_vs_all_coins_atr_VALIDATION_30D"
    )

    if (
        disc_uplift in out.columns
        and val_uplift in out.columns
    ):
        out[
            "positive_uplift_both"
        ] = (
            (out[disc_uplift] > 0)
            &
            (out[val_uplift] > 0)
        )

        out[
            "min_uplift_across_splits"
        ] = out[
            [
                disc_uplift,
                val_uplift,
            ]
        ].min(
            axis=1
        )

        out[
            "avg_uplift_across_splits"
        ] = out[
            [
                disc_uplift,
                val_uplift,
            ]
        ].mean(
            axis=1
        )

        out = out.sort_values(
            [
                "trigger_name",
                "positive_uplift_both",
                "min_uplift_across_splits",
                "avg_uplift_across_splits",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )

    return out


def main():
    ensure_dirs()

    trigger_events, event_source = (
        load_trigger_events()
    )

    print()
    print("=" * 120)
    print("B / D CONDITIONAL COIN SELECTOR DISCOVERY")
    print("=" * 120)
    print(f"Trigger event source: {event_source}")
    print()
    print(
        "Selection is ABSOLUTE Rank 1..Top30."
    )
    print(
        "No Top-% selection is used."
    )
    print(
        "Feature orientation and feature shortlist are selected from Discovery only."
    )
    print("=" * 120)

    panel, feature_cols = (
        build_event_coin_panel(
            trigger_events
        )
    )

    (
        ic_events,
        ic_long,
        ic_summary,
    ) = feature_ic_scan(
        panel,
        feature_cols,
    )

    selected_features = (
        choose_discovery_features(
            ic_summary
        )
    )

    if selected_features.empty:
        raise RuntimeError(
            "No discovery-eligible selector features."
        )

    print()
    print("=" * 140)
    print("DISCOVERY-SELECTED FEATURES")
    print("=" * 140)

    print(
        selected_features[
            [
                "trigger_name",
                "feature",
                "orientation",
                "events_used_DISCOVERY_60D",
                "mean_ic_DISCOVERY_60D",
                "discovery_sign_event_rate",
                "mean_ic_VALIDATION_30D",
                "validation_oriented_ic",
                "validation_sign_event_rate",
                "same_ic_sign",
            ]
        ].to_string(
            index=False
        )
    )

    (
        absolute_events,
        absolute_summary,
    ) = evaluate_selected_features(
        panel,
        selected_features,
    )

    robustness = (
        add_cross_split_robustness(
            absolute_summary
        )
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    ic_events_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_feature_ic_events_{stamp}.csv",
    )

    ic_summary_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_feature_ic_summary_{stamp}.csv",
    )

    selected_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_discovery_selected_features_{stamp}.csv",
    )

    abs_events_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_absolute_rank_events_{stamp}.csv",
    )

    abs_summary_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_absolute_rank_summary_{stamp}.csv",
    )

    robustness_file = os.path.join(
        REPORT_DIR,
        f"trigger_B_D_selector_robustness_{stamp}.csv",
    )

    ic_events.to_csv(
        ic_events_file,
        index=False,
    )

    ic_summary.to_csv(
        ic_summary_file,
        index=False,
    )

    selected_features.to_csv(
        selected_file,
        index=False,
    )

    absolute_events.to_csv(
        abs_events_file,
        index=False,
    )

    absolute_summary.to_csv(
        abs_summary_file,
        index=False,
    )

    robustness.to_csv(
        robustness_file,
        index=False,
    )

    print()
    print("=" * 160)
    print("TOP ROBUST ABSOLUTE-RANK CONFIGS")
    print("Primary event mode = NEW_EPISODE")
    print("=" * 160)

    display_cols = [
        "trigger_name",
        "feature",
        "orientation",
        "top_n",
        "positive_uplift_both",
        "min_uplift_across_splits",
        "avg_uplift_across_splits",
        "mean_direction_return_atr_DISCOVERY_60D",
        "mean_direction_return_atr_VALIDATION_30D",
        "mean_uplift_vs_all_coins_atr_DISCOVERY_60D",
        "mean_uplift_vs_all_coins_atr_VALIDATION_30D",
        "beats_all_coins_rate_DISCOVERY_60D",
        "beats_all_coins_rate_VALIDATION_30D",
    ]

    display_cols = [
        c for c in display_cols
        if c in robustness.columns
    ]

    for trigger_name in TRIGGERS:
        print()
        print(trigger_name)
        print(
            robustness.loc[
                robustness[
                    "trigger_name"
                ]
                == trigger_name,
                display_cols,
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"IC events:        {ic_events_file}")
    print(f"IC summary:       {ic_summary_file}")
    print(f"Selected features:{selected_file}")
    print(f"Absolute events:  {abs_events_file}")
    print(f"Absolute summary: {abs_summary_file}")
    print(f"Robustness:       {robustness_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
