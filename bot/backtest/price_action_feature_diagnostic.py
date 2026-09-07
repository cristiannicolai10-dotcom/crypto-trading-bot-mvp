import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# IMPORT EXISTING INDICATOR ENGINE
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )

import backtest_inverse as bt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "historical"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"

PRE_SAMPLE_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

DEVELOPMENT_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_3y.parquet"
)


# ============================================================
# NEW FEATURE SET
# ============================================================

NEW_FEATURES = [
    # Signal candle anatomy
    "candle_range_atr",
    "body_to_range",
    "signed_body_to_range",
    "lower_wick_to_range",
    "upper_wick_to_range",
    "close_location_in_range",
    "signal_candle_return_pct",

    # ATR dynamics
    "atr_change_24h_pct",
    "atr_change_3d_pct",
    "atr_vs_7d_median",

    # Recent structure / breakout quality
    "prior_24h_range_atr",
    "prior_3d_range_atr",
    "break_below_prior_24h_low_atr",
    "break_below_prior_3d_low_atr",
    "distance_from_prior_3d_high_atr",

    # Fractal age
    "fractal_confirmation_age_bars",
    "fractal_candidate_age_bars",

    # Fully closed prior daily regime
    "prev_daily_return_pct",
    "prev_daily_ema20_vs_ema50_pct",
    "prev_daily_close_vs_ema20_pct",
    "prev_daily_ema20_slope_5d_pct",
]


# ============================================================
# INPUT PAIRED RESULTS
# ============================================================

def find_paired_file():

    if len(sys.argv) > 1:

        candidate = sys.argv[1]

        if not os.path.isabs(
            candidate
        ):

            candidate = os.path.join(
                BASE_DIR,
                candidate
            )

        if not os.path.exists(
            candidate
        ):

            raise RuntimeError(
                f"Paired result file not found: {candidate}"
            )

        return candidate

    pattern = os.path.join(
        REPORT_DIR,
        f"paired_signal_results_{SYMBOL}_{TIMEFRAME}_*.csv"
    )

    files = sorted(
        glob.glob(
            pattern
        )
    )

    if not files:

        raise RuntimeError(
            "No paired_signal_results CSV found in reports/."
        )

    return files[-1]


# ============================================================
# LOAD RAW HISTORY
# ============================================================

def load_raw_file(
    filepath,
    label
):

    if not os.path.exists(
        filepath
    ):

        raise RuntimeError(
            f"Missing {label}: {filepath}"
        )

    df = pd.read_parquet(
        filepath
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"{label}: "
        f"{len(df):,} candles | "
        f"{df.iloc[0]['timestamp']} -> "
        f"{df.iloc[-1]['timestamp']}"
    )

    return df


def load_combined_history():

    print()
    print("=" * 110)
    print("PRICE ACTION FEATURE DIAGNOSTIC")
    print("=" * 110)
    print("No ML.")
    print("No threshold optimization.")
    print("Goal: test whether NEW causal price-action features")
    print("separate TRADE vs NO_TRADE and INVERSE vs BREAKOUT.")
    print("=" * 110)

    pre = load_raw_file(
        PRE_SAMPLE_FILE,
        "Pre-sample"
    )

    development = load_raw_file(
        DEVELOPMENT_FILE,
        "Development"
    )

    df = pd.concat(
        [
            pre,
            development
        ],
        ignore_index=True
    )

    duplicate_count = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Combined candles: {len(df):,}"
    )

    print(
        f"Duplicates removed: {duplicate_count}"
    )

    print(
        f"From: {df.iloc[0]['timestamp']}"
    )

    print(
        f"To:   {df.iloc[-1]['timestamp']}"
    )

    df = bt.remove_open_candle(
        df,
        TIMEFRAME
    )

    print()
    print("Calculating existing indicators...")

    df = bt.calculate_indicators(
        df
    )

    return df


# ============================================================
# DAILY CLOSED FEATURES
# ============================================================

def add_previous_closed_daily_features(
    df
):

    work = df.copy()

    temp = (
        work[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ]
        .set_index(
            "timestamp"
        )
        .sort_index()
    )

    daily = pd.DataFrame(
        {
            "open":
                temp["open"]
                .resample("1D")
                .first(),

            "high":
                temp["high"]
                .resample("1D")
                .max(),

            "low":
                temp["low"]
                .resample("1D")
                .min(),

            "close":
                temp["close"]
                .resample("1D")
                .last(),

            "volume":
                temp["volume"]
                .resample("1D")
                .sum(),
        }
    ).dropna(
        subset=["close"]
    )

    daily[
        "daily_return_pct"
    ] = (
        daily["close"]
        .pct_change(1)
        * 100
    )

    daily[
        "ema20"
    ] = (
        daily["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    daily[
        "ema50"
    ] = (
        daily["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    daily[
        "daily_ema20_vs_ema50_pct"
    ] = (
        (
            daily["ema20"]
            / daily["ema50"]
        )
        - 1
    ) * 100

    daily[
        "daily_close_vs_ema20_pct"
    ] = (
        (
            daily["close"]
            / daily["ema20"]
        )
        - 1
    ) * 100

    daily[
        "daily_ema20_slope_5d_pct"
    ] = (
        daily["ema20"]
        .pct_change(5)
        * 100
    )

    # Causal alignment:
    # for any signal during calendar day D,
    # use only the fully completed day D-1.
    causal_daily = daily[
        [
            "daily_return_pct",
            "daily_ema20_vs_ema50_pct",
            "daily_close_vs_ema20_pct",
            "daily_ema20_slope_5d_pct",
        ]
    ].shift(1)

    causal_daily = causal_daily.rename(
        columns={
            "daily_return_pct":
                "prev_daily_return_pct",

            "daily_ema20_vs_ema50_pct":
                "prev_daily_ema20_vs_ema50_pct",

            "daily_close_vs_ema20_pct":
                "prev_daily_close_vs_ema20_pct",

            "daily_ema20_slope_5d_pct":
                "prev_daily_ema20_slope_5d_pct",
        }
    )

    work[
        "calendar_day"
    ] = (
        work["timestamp"]
        .dt
        .floor("1D")
    )

    causal_daily = (
        causal_daily
        .reset_index()
        .rename(
            columns={
                "timestamp":
                    "calendar_day"
            }
        )
    )

    work = work.merge(
        causal_daily,
        on="calendar_day",
        how="left"
    )

    return work


# ============================================================
# PRICE ACTION FEATURES
# ============================================================

def add_price_action_features(
    df
):

    work = df.copy()

    # --------------------------------------------------------
    # Candle anatomy
    # --------------------------------------------------------

    candle_range = (
        work["high"]
        - work["low"]
    )

    candle_range_safe = (
        candle_range
        .replace(
            0,
            np.nan
        )
    )

    body = (
        work["close"]
        - work["open"]
    )

    abs_body = body.abs()

    lower_body_edge = pd.concat(
        [
            work["open"],
            work["close"]
        ],
        axis=1
    ).min(
        axis=1
    )

    upper_body_edge = pd.concat(
        [
            work["open"],
            work["close"]
        ],
        axis=1
    ).max(
        axis=1
    )

    lower_wick = (
        lower_body_edge
        - work["low"]
    )

    upper_wick = (
        work["high"]
        - upper_body_edge
    )

    work[
        "candle_range_atr"
    ] = (
        candle_range
        / work["atr"]
    )

    work[
        "body_to_range"
    ] = (
        abs_body
        / candle_range_safe
    )

    work[
        "signed_body_to_range"
    ] = (
        body
        / candle_range_safe
    )

    work[
        "lower_wick_to_range"
    ] = (
        lower_wick
        / candle_range_safe
    )

    work[
        "upper_wick_to_range"
    ] = (
        upper_wick
        / candle_range_safe
    )

    work[
        "close_location_in_range"
    ] = (
        (
            work["close"]
            - work["low"]
        )
        / candle_range_safe
    )

    work[
        "signal_candle_return_pct"
    ] = (
        (
            work["close"]
            / work["open"]
        )
        - 1
    ) * 100

    # --------------------------------------------------------
    # ATR dynamics
    # --------------------------------------------------------

    work[
        "atr_change_24h_pct"
    ] = (
        work["atr"]
        .pct_change(6)
        * 100
    )

    work[
        "atr_change_3d_pct"
    ] = (
        work["atr"]
        .pct_change(18)
        * 100
    )

    work[
        "atr_vs_7d_median"
    ] = (
        work["atr"]
        / (
            work["atr"]
            .rolling(
                42,
                min_periods=20
            )
            .median()
        )
    )

    # --------------------------------------------------------
    # Prior structure
    #
    # SHIFT(1) is intentional: recent high/low excludes
    # the signal candle itself.
    # --------------------------------------------------------

    prior_high = (
        work["high"]
        .shift(1)
    )

    prior_low = (
        work["low"]
        .shift(1)
    )

    prior_24h_high = (
        prior_high
        .rolling(
            6,
            min_periods=6
        )
        .max()
    )

    prior_24h_low = (
        prior_low
        .rolling(
            6,
            min_periods=6
        )
        .min()
    )

    prior_3d_high = (
        prior_high
        .rolling(
            18,
            min_periods=18
        )
        .max()
    )

    prior_3d_low = (
        prior_low
        .rolling(
            18,
            min_periods=18
        )
        .min()
    )

    work[
        "prior_24h_range_atr"
    ] = (
        (
            prior_24h_high
            - prior_24h_low
        )
        / work["atr"]
    )

    work[
        "prior_3d_range_atr"
    ] = (
        (
            prior_3d_high
            - prior_3d_low
        )
        / work["atr"]
    )

    # Positive = signal close is below previous low.
    work[
        "break_below_prior_24h_low_atr"
    ] = (
        (
            prior_24h_low
            - work["close"]
        )
        / work["atr"]
    )

    work[
        "break_below_prior_3d_low_atr"
    ] = (
        (
            prior_3d_low
            - work["close"]
        )
        / work["atr"]
    )

    work[
        "distance_from_prior_3d_high_atr"
    ] = (
        (
            prior_3d_high
            - work["close"]
        )
        / work["atr"]
    )

    # --------------------------------------------------------
    # Fractal age
    #
    # Mirrors the prior diagnostic semantics:
    # fractal_low is confirmation on current bar for a
    # candidate two bars earlier.
    # --------------------------------------------------------

    bar_index = np.arange(
        len(work),
        dtype=float
    )

    if "fractal_low" in work.columns:

        confirmed = (
            work["fractal_low"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        confirmation_bar = np.where(
            confirmed,
            bar_index,
            np.nan
        )

        confirmation_bar = (
            pd.Series(
                confirmation_bar,
                index=work.index
            )
            .ffill()
        )

        work[
            "fractal_confirmation_age_bars"
        ] = (
            bar_index
            - confirmation_bar
        )

        work[
            "fractal_candidate_age_bars"
        ] = (
            work[
                "fractal_confirmation_age_bars"
            ]
            + 2
        )

    else:

        work[
            "fractal_confirmation_age_bars"
        ] = np.nan

        work[
            "fractal_candidate_age_bars"
        ] = np.nan

    # --------------------------------------------------------
    # Previous fully closed daily information
    # --------------------------------------------------------

    work = add_previous_closed_daily_features(
        work
    )

    return work


# ============================================================
# LOAD LABELS AND MERGE
# ============================================================

def load_paired_labels(
    filepath
):

    paired = pd.read_csv(
        filepath
    )

    paired[
        "signal_timestamp"
    ] = pd.to_datetime(
        paired[
            "signal_timestamp"
        ],
        utc=True,
        errors="coerce"
    )

    required = [
        "signal_number",
        "signal_timestamp",
        "year",
        "window",
        "sample_group",
        "pair_class",
        "preferred_mode",
        "breakout_result",
        "inverse_result",
        "breakout_r_multiple",
        "inverse_r_multiple",
    ]

    missing = [
        column
        for column in required
        if column
        not in paired.columns
    ]

    if missing:

        raise RuntimeError(
            "Paired CSV missing columns: "
            + ", ".join(
                missing
            )
        )

    paired = paired[
        paired[
            "preferred_mode"
        ]
        .isin(
            [
                "BREAKOUT",
                "INVERSE",
                "NO_TRADE"
            ]
        )
    ].copy()

    return paired


def build_signal_dataset(
    history,
    paired
):

    feature_source = history[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
        ]
        + NEW_FEATURES
    ].copy()

    feature_source = feature_source.rename(
        columns={
            "timestamp":
                "signal_timestamp"
        }
    )

    merged = paired.merge(
        feature_source,
        on="signal_timestamp",
        how="left",
        validate="one_to_one"
    )

    missing_rows = int(
        merged[
            NEW_FEATURES
        ]
        .isna()
        .all(
            axis=1
        )
        .sum()
    )

    print()
    print(
        f"Paired labels: {len(paired)}"
    )

    print(
        f"Merged signal rows: {len(merged)}"
    )

    print(
        f"Rows with all new features missing: "
        f"{missing_rows}"
    )

    return merged


# ============================================================
# SEPARATION HELPERS
# ============================================================

def describe_values(
    values
):

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    if len(values) == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
        }

    q25 = values.quantile(
        0.25
    )

    q75 = values.quantile(
        0.75
    )

    return {
        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                values.median()
            ),

        "q25":
            float(
                q25
            ),

        "q75":
            float(
                q75
            ),

        "iqr":
            float(
                q75
                - q25
            ),
    }


def separation_score(
    a,
    b
):

    a_stats = describe_values(
        a
    )

    b_stats = describe_values(
        b
    )

    pooled_iqr = (
        (
            a_stats["iqr"]
            + b_stats["iqr"]
        )
        / 2
    )

    median_difference = (
        a_stats["median"]
        - b_stats["median"]
    )

    if (
        pd.isna(
            pooled_iqr
        )
        or
        pooled_iqr == 0
    ):

        score = np.nan

    else:

        score = (
            abs(
                median_difference
            )
            / pooled_iqr
        )

    if pd.isna(
        median_difference
    ):

        direction = "NA"

    elif median_difference > 0:

        direction = "A_HIGHER"

    elif median_difference < 0:

        direction = "B_HIGHER"

    else:

        direction = "EQUAL"

    return {
        "a_n":
            a_stats["n"],

        "b_n":
            b_stats["n"],

        "a_mean":
            a_stats["mean"],

        "b_mean":
            b_stats["mean"],

        "a_median":
            a_stats["median"],

        "b_median":
            b_stats["median"],

        "a_q25":
            a_stats["q25"],

        "a_q75":
            a_stats["q75"],

        "b_q25":
            b_stats["q25"],

        "b_q75":
            b_stats["q75"],

        "median_difference":
            median_difference,

        "separation_score":
            score,

        "direction":
            direction,
    }


# ============================================================
# CLASS FEATURE SUMMARY
# ============================================================

def build_class_summary(
    df
):

    rows = []

    for mode in [
        "BREAKOUT",
        "INVERSE",
        "NO_TRADE"
    ]:

        subset = df[
            df[
                "preferred_mode"
            ]
            == mode
        ]

        for feature in NEW_FEATURES:

            stats = describe_values(
                subset[
                    feature
                ]
            )

            rows.append(
                {
                    "preferred_mode":
                        mode,

                    "feature":
                        feature,

                    **stats,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STAGE SEPARATION
# ============================================================

def stage_subsets(
    df,
    stage
):

    if stage == "STAGE1_TRADE_VS_NO_TRADE":

        a = df[
            df[
                "preferred_mode"
            ]
            .isin(
                [
                    "BREAKOUT",
                    "INVERSE"
                ]
            )
        ]

        b = df[
            df[
                "preferred_mode"
            ]
            == "NO_TRADE"
        ]

        a_label = "TRADE"
        b_label = "NO_TRADE"

    elif stage == "STAGE2_INVERSE_VS_BREAKOUT":

        a = df[
            df[
                "preferred_mode"
            ]
            == "INVERSE"
        ]

        b = df[
            df[
                "preferred_mode"
            ]
            == "BREAKOUT"
        ]

        a_label = "INVERSE"
        b_label = "BREAKOUT"

    else:

        raise ValueError(
            stage
        )

    return (
        a,
        b,
        a_label,
        b_label
    )


def build_separation_table(
    df
):

    rows = []

    scopes = [
        (
            "FULL",
            df
        ),
        (
            "PRESAMPLE",
            df[
                df[
                    "sample_group"
                ]
                == "PRESAMPLE"
            ]
        ),
        (
            "DEVELOPMENT",
            df[
                df[
                    "sample_group"
                ]
                == "DEVELOPMENT"
            ]
        ),
    ]

    stages = [
        "STAGE1_TRADE_VS_NO_TRADE",
        "STAGE2_INVERSE_VS_BREAKOUT",
    ]

    for scope_name, scope_df in scopes:

        for stage in stages:

            (
                a,
                b,
                a_label,
                b_label
            ) = stage_subsets(
                scope_df,
                stage
            )

            for feature in NEW_FEATURES:

                result = separation_score(
                    a[
                        feature
                    ],
                    b[
                        feature
                    ]
                )

                rows.append(
                    {
                        "scope":
                            scope_name,

                        "stage":
                            stage,

                        "feature":
                            feature,

                        "group_a":
                            a_label,

                        "group_b":
                            b_label,

                        **result,
                    }
                )

    result_df = pd.DataFrame(
        rows
    )

    result_df[
        "rank_within_scope_stage"
    ] = (
        result_df
        .groupby(
            [
                "scope",
                "stage"
            ]
        )[
            "separation_score"
        ]
        .rank(
            method="min",
            ascending=False
        )
    )

    return result_df


# ============================================================
# STABILITY SUMMARY
# ============================================================

def build_stability_summary(
    separation_df
):

    rows = []

    for stage in separation_df[
        "stage"
    ].unique():

        stage_df = separation_df[
            separation_df[
                "stage"
            ]
            == stage
        ]

        for feature in NEW_FEATURES:

            feature_df = stage_df[
                stage_df[
                    "feature"
                ]
                == feature
            ]

            def get_row(
                scope
            ):

                match = feature_df[
                    feature_df[
                        "scope"
                    ]
                    == scope
                ]

                if match.empty:
                    return None

                return match.iloc[
                    0
                ]

            full = get_row(
                "FULL"
            )

            pre = get_row(
                "PRESAMPLE"
            )

            dev = get_row(
                "DEVELOPMENT"
            )

            if (
                full is None
                or
                pre is None
                or
                dev is None
            ):

                continue

            pre_diff = pre[
                "median_difference"
            ]

            dev_diff = dev[
                "median_difference"
            ]

            if (
                pd.isna(
                    pre_diff
                )
                or
                pd.isna(
                    dev_diff
                )
            ):

                direction_stable = False

            else:

                direction_stable = (
                    np.sign(
                        pre_diff
                    )
                    ==
                    np.sign(
                        dev_diff
                    )
                )

            split_scores = [
                pre[
                    "separation_score"
                ],
                dev[
                    "separation_score"
                ],
            ]

            valid_split_scores = [
                value
                for value
                in split_scores
                if not pd.isna(
                    value
                )
            ]

            min_split_score = (
                min(
                    valid_split_scores
                )
                if valid_split_scores
                else np.nan
            )

            mean_split_score = (
                float(
                    np.mean(
                        valid_split_scores
                    )
                )
                if valid_split_scores
                else np.nan
            )

            rows.append(
                {
                    "stage":
                        stage,

                    "feature":
                        feature,

                    "full_separation":
                        full[
                            "separation_score"
                        ],

                    "presample_separation":
                        pre[
                            "separation_score"
                        ],

                    "development_separation":
                        dev[
                            "separation_score"
                        ],

                    "presample_median_difference":
                        pre_diff,

                    "development_median_difference":
                        dev_diff,

                    "direction_stable":
                        direction_stable,

                    "min_split_separation":
                        min_split_score,

                    "mean_split_separation":
                        mean_split_score,
                }
            )

    stability = pd.DataFrame(
        rows
    )

    if stability.empty:
        return stability

    stability[
        "stable_rank"
    ] = np.nan

    for stage in stability[
        "stage"
    ].unique():

        mask = (
            stability[
                "stage"
            ]
            == stage
        )

        stable_mask = (
            mask
            &
            stability[
                "direction_stable"
            ]
        )

        stability.loc[
            stable_mask,
            "stable_rank"
        ] = (
            stability.loc[
                stable_mask,
                "min_split_separation"
            ]
            .rank(
                method="min",
                ascending=False
            )
        )

    return stability


# ============================================================
# PRINT TOP RESULTS
# ============================================================

def print_stage_top(
    stability,
    stage,
    top_n=10
):

    subset = stability[
        (
            stability[
                "stage"
            ]
            == stage
        )
        &
        (
            stability[
                "direction_stable"
            ]
        )
    ].copy()

    subset = subset.sort_values(
        [
            "min_split_separation",
            "full_separation"
        ],
        ascending=False
    )

    print()
    print("=" * 120)
    print(stage)
    print("Stable direction in PRESAMPLE and DEVELOPMENT")
    print("=" * 120)

    if subset.empty:

        print(
            "No direction-stable features."
        )

        return

    print(
        subset[
            [
                "feature",
                "full_separation",
                "presample_separation",
                "development_separation",
                "min_split_separation",
                "presample_median_difference",
                "development_median_difference",
            ]
        ]
        .head(
            top_n
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    paired_file = find_paired_file()

    print()
    print(
        f"Paired labels source: "
        f"{paired_file}"
    )

    history = load_combined_history()

    history = add_price_action_features(
        history
    )

    paired = load_paired_labels(
        paired_file
    )

    signals = build_signal_dataset(
        history,
        paired
    )

    class_summary = build_class_summary(
        signals
    )

    separation = build_separation_table(
        signals
    )

    stability = build_stability_summary(
        separation
    )

    print()
    print("=" * 100)
    print("SIGNAL CLASS COUNTS")
    print("=" * 100)

    print(
        signals[
            "preferred_mode"
        ]
        .value_counts()
        .to_string()
    )

    print_stage_top(
        stability,
        "STAGE1_TRADE_VS_NO_TRADE"
    )

    print_stage_top(
        stability,
        "STAGE2_INVERSE_VS_BREAKOUT"
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    signal_file = os.path.join(
        REPORT_DIR,
        f"price_action_signal_features_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    class_file = os.path.join(
        REPORT_DIR,
        f"price_action_feature_summary_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    separation_file = os.path.join(
        REPORT_DIR,
        f"price_action_feature_separation_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    stability_file = os.path.join(
        REPORT_DIR,
        f"price_action_feature_stability_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    signals.to_csv(
        signal_file,
        index=False
    )

    class_summary.to_csv(
        class_file,
        index=False
    )

    separation.to_csv(
        separation_file,
        index=False
    )

    stability.to_csv(
        stability_file,
        index=False
    )

    print()
    print("=" * 110)
    print("PRICE ACTION DIAGNOSTIC FINISHED")
    print("=" * 110)
    print(f"Signal features: {signal_file}")
    print(f"Class summary:   {class_file}")
    print(f"Separation:      {separation_file}")
    print(f"Stability:       {stability_file}")
    print("=" * 110)

    print()
    print(
        "Interpretation rule for this diagnostic:"
    )

    print(
        "A useful feature should have BOTH meaningful separation "
        "and the same median-difference direction in PRESAMPLE "
        "and DEVELOPMENT. Do not select a feature only because "
        "FULL-sample separation looks large."
    )


if __name__ == "__main__":
    main()
