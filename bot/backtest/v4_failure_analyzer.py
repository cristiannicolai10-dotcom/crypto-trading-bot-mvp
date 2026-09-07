import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# IMPORT EXISTING BACKTEST ENGINE
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


# V4_B — FIXED
SL_ATR = 1.25
TP_ATR = 3.00


# Walk-forward classification
GOOD_WINDOWS = {
    3,
    4,
    6
}

BAD_WINDOWS = {
    2,
    5
}

WINDOW_MONTHS = 6


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    filepath = os.path.join(
        DATA_DIR,
        f"{SYMBOL}_{TIMEFRAME}_3y.parquet"
    )

    if not os.path.exists(
        filepath
    ):

        raise RuntimeError(
            f"File not found: {filepath}"
        )

    print()
    print("=" * 100)
    print("V4 FAILURE ANALYZER")
    print("=" * 100)

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Timeframe: {TIMEFRAME}"
    )

    print(
        f"SL: {SL_ATR} ATR"
    )

    print(
        f"TP: {TP_ATR} ATR"
    )

    print(
        "GOOD windows: W3 + W4 + W6"
    )

    print(
        "BAD windows: W2 + W5"
    )

    print(
        "W1 excluded"
    )

    print(
        "Supabase: NOT USED"
    )

    print("=" * 100)

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
        .drop_duplicates(
            subset=[
                "timestamp"
            ]
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    df = bt.remove_open_candle(
        df,
        TIMEFRAME
    )

    print(
        f"Candles: {len(df):,}"
    )

    print(
        f"From: {df.iloc[0]['timestamp']}"
    )

    print(
        f"To:   {df.iloc[-1]['timestamp']}"
    )

    print()
    print(
        "Calculating indicators..."
    )

    df = bt.calculate_indicators(
        df
    )

    print(
        "Calculating original V3 signals..."
    )

    df = bt.calculate_scores(
        df
    )

    return df


# ============================================================
# WINDOW NUMBERS
# ============================================================

def assign_windows(
    df
):

    df = df.copy()

    dataset_start = (
        df[
            "timestamp"
        ]
        .min()
    )

    dataset_end = (
        df[
            "timestamp"
        ]
        .max()
    )

    df[
        "window"
    ] = np.nan

    start = dataset_start
    window_number = 1

    while start <= dataset_end:

        end = (
            start
            + pd.DateOffset(
                months=WINDOW_MONTHS
            )
        )

        if end > dataset_end:

            end = (
                dataset_end
                + pd.Timedelta(
                    microseconds=1
                )
            )

        mask = (
            (
                df[
                    "timestamp"
                ]
                >= start
            )
            &
            (
                df[
                    "timestamp"
                ]
                < end
            )
        )

        df.loc[
            mask,
            "window"
        ] = window_number

        start = end
        window_number += 1

    df[
        "window"
    ] = (
        df[
            "window"
        ]
        .astype(
            "Int64"
        )
    )

    return df


# ============================================================
# BUILD FAILURE FEATURES
# ============================================================

def add_features(
    df
):

    df = df.copy()


    # --------------------------------------------------------
    # ATR %
    # --------------------------------------------------------

    df[
        "atr_pct"
    ] = (
        df[
            "atr"
        ]
        / df[
            "close"
        ]
        * 100
    )


    # --------------------------------------------------------
    # EMA50 ↔ EMA200 distance
    # --------------------------------------------------------

    df[
        "ema_spread_pct"
    ] = (
        (
            df[
                "ema_50"
            ]
            - df[
                "ema_200"
            ]
        )
        .abs()
        / df[
            "ema_200"
        ]
        * 100
    )


    # --------------------------------------------------------
    # EMA200 slope
    #
    # 3 candles = 12 hours
    # 6 candles = 24 hours
    # --------------------------------------------------------

    df[
        "ema200_slope_12h_pct"
    ] = (
        df[
            "ema_200"
        ]
        .pct_change(
            3
        )
        * 100
    )

    df[
        "ema200_slope_24h_pct"
    ] = (
        df[
            "ema_200"
        ]
        .pct_change(
            6
        )
        * 100
    )


    # --------------------------------------------------------
    # PRICE ↔ EMA200
    #
    # Negative = price below EMA200
    # --------------------------------------------------------

    df[
        "price_vs_ema200_pct"
    ] = (
        (
            df[
                "close"
            ]
            - df[
                "ema_200"
            ]
        )
        / df[
            "ema_200"
        ]
        * 100
    )


    # --------------------------------------------------------
    # ALLIGATOR STATE
    # --------------------------------------------------------

    bullish_alligator = (
        (
            df[
                "alligator_lips"
            ]
            > df[
                "alligator_teeth"
            ]
        )
        &
        (
            df[
                "alligator_teeth"
            ]
            > df[
                "alligator_jaw"
            ]
        )
    )

    bearish_alligator = (
        (
            df[
                "alligator_lips"
            ]
            < df[
                "alligator_teeth"
            ]
        )
        &
        (
            df[
                "alligator_teeth"
            ]
            < df[
                "alligator_jaw"
            ]
        )
    )

    df[
        "alligator_state"
    ] = "tangled"

    df.loc[
        bullish_alligator,
        "alligator_state"
    ] = "bullish"

    df.loc[
        bearish_alligator,
        "alligator_state"
    ] = "bearish"


    # --------------------------------------------------------
    # ALLIGATOR SPREAD
    # --------------------------------------------------------

    alligator_max = (
        df[
            [
                "alligator_lips",
                "alligator_teeth",
                "alligator_jaw"
            ]
        ]
        .max(
            axis=1
        )
    )

    alligator_min = (
        df[
            [
                "alligator_lips",
                "alligator_teeth",
                "alligator_jaw"
            ]
        ]
        .min(
            axis=1
        )
    )

    df[
        "alligator_spread_pct"
    ] = (
        (
            alligator_max
            - alligator_min
        )
        / df[
            "close"
        ]
        * 100
    )


    # --------------------------------------------------------
    # FRACTAL LOW LEVEL
    #
    # Williams fractal is confirmed two bars later.
    # --------------------------------------------------------

    if (
        "fractal_low"
        in df.columns
    ):

        confirmed_fractal_low = (
            df[
                "low"
            ]
            .shift(
                2
            )
            .where(
                df[
                    "fractal_low"
                ]
                .fillna(
                    False
                )
            )
        )

        df[
            "last_fractal_low"
        ] = (
            confirmed_fractal_low
            .ffill()
        )

    elif (
        "fractal_low_level"
        in df.columns
    ):

        df[
            "last_fractal_low"
        ] = (
            df[
                "fractal_low_level"
            ]
            .ffill()
        )

    else:

        print(
            "WARNING: fractal low column not found."
        )

        df[
            "last_fractal_low"
        ] = np.nan


    # --------------------------------------------------------
    # BREAKDOWN DEPTH
    # --------------------------------------------------------

    df[
        "breakdown_depth_pct"
    ] = (
        (
            df[
                "last_fractal_low"
            ]
            - df[
                "close"
            ]
        )
        / df[
            "last_fractal_low"
        ]
        * 100
    )


    # --------------------------------------------------------
    # V4 SIGNAL
    # --------------------------------------------------------

    df[
        "v4_signal"
    ] = (
        (
            df[
                "direction"
            ]
            == "SHORT"
        )
        &
        (
            df[
                "market_regime"
            ]
            == "bearish"
        )
        &
        (
            df[
                "volatility_regime"
            ]
            == "normal"
        )
    )


    # --------------------------------------------------------
    # GOOD / BAD GROUP
    # --------------------------------------------------------

    df[
        "period_group"
    ] = "EXCLUDED"

    df.loc[
        df[
            "window"
        ]
        .isin(
            GOOD_WINDOWS
        ),
        "period_group"
    ] = "GOOD"

    df.loc[
        df[
            "window"
        ]
        .isin(
            BAD_WINDOWS
        ),
        "period_group"
    ] = "BAD"


    return df


# ============================================================
# BACKTEST GOOD / BAD GROUPS
# ============================================================

def run_group_backtest(
    df,
    group_name
):

    test_df = (
        df.copy()
    )

    valid = (
        test_df[
            "v4_signal"
        ]
        &
        (
            test_df[
                "period_group"
            ]
            == group_name
        )
    )

    signals = int(
        valid.sum()
    )

    test_df.loc[
        ~valid,
        "direction"
    ] = "HOLD"

    old_sl = (
        bt.ATR_STOP_MULTIPLIER
    )

    old_tp = (
        bt.ATR_TARGET_MULTIPLIER
    )

    old_warmup = (
        bt.WARMUP_BARS
    )

    try:

        bt.ATR_STOP_MULTIPLIER = (
            SL_ATR
        )

        bt.ATR_TARGET_MULTIPLIER = (
            TP_ATR
        )

        bt.WARMUP_BARS = 250

        (
            trades,
            final_balance,
            max_drawdown
        ) = bt.run_backtest(
            test_df
        )

        summary = bt.build_summary(
            trades=trades,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            df=test_df,
            final_balance=final_balance,
            max_drawdown=max_drawdown
        )

    finally:

        bt.ATR_STOP_MULTIPLIER = (
            old_sl
        )

        bt.ATR_TARGET_MULTIPLIER = (
            old_tp
        )

        bt.WARMUP_BARS = (
            old_warmup
        )

    return {

        "group":
            group_name,

        "signals":
            signals,

        "trades":
            int(
                summary.get(
                    "total_trades",
                    0
                )
            ),

        "win_rate_pct":
            summary.get(
                "win_rate_pct"
            ),

        "profit_factor":
            summary.get(
                "profit_factor"
            ),

        "return_pct":
            summary.get(
                "total_return_pct"
            ),

        "max_drawdown_pct":
            summary.get(
                "max_drawdown_pct"
            ),

        "average_trade_pct":
            summary.get(
                "average_trade_pct"
            ),

        "max_consecutive_losses":
            summary.get(
                "max_consecutive_losses"
            )
    }


# ============================================================
# NUMERIC FEATURE COMPARISON
# ============================================================

def compare_numeric_features(
    signal_df
):

    features = [

        "atr_pct",

        "ema_spread_pct",

        "ema200_slope_12h_pct",

        "ema200_slope_24h_pct",

        "price_vs_ema200_pct",

        "volume_ratio",

        "alligator_spread_pct",

        "breakdown_depth_pct"
    ]

    rows = []

    good_df = (
        signal_df[
            signal_df[
                "period_group"
            ]
            == "GOOD"
        ]
    )

    bad_df = (
        signal_df[
            signal_df[
                "period_group"
            ]
            == "BAD"
        ]
    )

    for feature in features:

        good = (
            pd.to_numeric(
                good_df[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
        )

        bad = (
            pd.to_numeric(
                bad_df[
                    feature
                ],
                errors="coerce"
            )
            .dropna()
        )

        if (
            good.empty
            or bad.empty
        ):

            continue

        good_q25 = (
            good.quantile(
                0.25
            )
        )

        good_q75 = (
            good.quantile(
                0.75
            )
        )

        bad_q25 = (
            bad.quantile(
                0.25
            )
        )

        bad_q75 = (
            bad.quantile(
                0.75
            )
        )

        good_iqr = (
            good_q75
            - good_q25
        )

        bad_iqr = (
            bad_q75
            - bad_q25
        )

        pooled_iqr = (
            (
                abs(
                    good_iqr
                )
                +
                abs(
                    bad_iqr
                )
            )
            / 2
        )

        median_difference = (
            good.median()
            - bad.median()
        )

        if pooled_iqr > 0:

            separation_score = (
                abs(
                    median_difference
                )
                / pooled_iqr
            )

        else:

            separation_score = np.nan

        rows.append(
            {

                "feature":
                    feature,

                "good_n":
                    len(
                        good
                    ),

                "bad_n":
                    len(
                        bad
                    ),

                "good_mean":
                    good.mean(),

                "bad_mean":
                    bad.mean(),

                "good_median":
                    good.median(),

                "bad_median":
                    bad.median(),

                "good_q25":
                    good_q25,

                "good_q75":
                    good_q75,

                "bad_q25":
                    bad_q25,

                "bad_q75":
                    bad_q75,

                "median_difference":
                    median_difference,

                "separation_score":
                    separation_score
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    if not result_df.empty:

        result_df = (
            result_df
            .sort_values(
                "separation_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    return result_df


# ============================================================
# ALLIGATOR CATEGORICAL COMPARISON
# ============================================================

def compare_alligator_state(
    signal_df
):

    rows = []

    for group_name in [
        "GOOD",
        "BAD"
    ]:

        subset = (
            signal_df[
                signal_df[
                    "period_group"
                ]
                == group_name
            ]
        )

        total = len(
            subset
        )

        counts = (
            subset[
                "alligator_state"
            ]
            .value_counts()
        )

        for state in [
            "bearish",
            "bullish",
            "tangled"
        ]:

            count = int(
                counts.get(
                    state,
                    0
                )
            )

            pct = (
                count
                / total
                * 100
                if total > 0
                else 0
            )

            rows.append(
                {

                    "group":
                        group_name,

                    "alligator_state":
                        state,

                    "count":
                        count,

                    "pct":
                        pct
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WINDOW FEATURE SUMMARY
# ============================================================

def build_window_summary(
    signal_df
):

    numeric_features = [

        "atr_pct",

        "ema_spread_pct",

        "ema200_slope_24h_pct",

        "price_vs_ema200_pct",

        "volume_ratio",

        "alligator_spread_pct",

        "breakdown_depth_pct"
    ]

    summary = (
        signal_df
        .groupby(
            [
                "window",
                "period_group"
            ]
        )[
            numeric_features
        ]
        .median()
        .reset_index()
    )

    counts = (
        signal_df
        .groupby(
            [
                "window",
                "period_group"
            ]
        )
        .size()
        .reset_index(
            name="signals"
        )
    )

    summary = (
        counts
        .merge(
            summary,
            on=[
                "window",
                "period_group"
            ],
            how="left"
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    df = assign_windows(
        df
    )

    df = add_features(
        df
    )


    signal_df = (
        df[
            df[
                "v4_signal"
            ]
            &
            (
                df[
                    "period_group"
                ]
                .isin(
                    [
                        "GOOD",
                        "BAD"
                    ]
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    print()
    print("=" * 100)
    print("V4 SIGNAL COUNTS")
    print("=" * 100)

    print(
        signal_df[
            [
                "window",
                "period_group"
            ]
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    good_result = (
        run_group_backtest(
            df,
            "GOOD"
        )
    )

    bad_result = (
        run_group_backtest(
            df,
            "BAD"
        )
    )


    performance_df = pd.DataFrame(
        [
            good_result,
            bad_result
        ]
    )


    print()
    print("=" * 100)
    print("GOOD vs BAD BACKTEST")
    print("=" * 100)

    print(
        performance_df
        .to_string(
            index=False
        )
    )


    comparison_df = (
        compare_numeric_features(
            signal_df
        )
    )


    print()
    print("=" * 120)
    print("NUMERIC FEATURE DIFFERENCES")
    print("=" * 120)

    if not comparison_df.empty:

        print(
            comparison_df[
                [
                    "feature",
                    "good_median",
                    "bad_median",
                    "median_difference",
                    "separation_score",
                    "good_q25",
                    "good_q75",
                    "bad_q25",
                    "bad_q75"
                ]
            ]
            .to_string(
                index=False
            )
        )


    alligator_df = (
        compare_alligator_state(
            signal_df
        )
    )


    print()
    print("=" * 100)
    print("ALLIGATOR STATE")
    print("=" * 100)

    print(
        alligator_df
        .to_string(
            index=False
        )
    )


    window_summary_df = (
        build_window_summary(
            signal_df
        )
    )


    print()
    print("=" * 120)
    print("MEDIAN FEATURES BY WINDOW")
    print("=" * 120)

    print(
        window_summary_df
        .to_string(
            index=False
        )
    )


    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    signals_file = os.path.join(
        REPORT_DIR,
        f"v4_failure_signals_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    comparison_file = os.path.join(
        REPORT_DIR,
        f"v4_failure_comparison_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    windows_file = os.path.join(
        REPORT_DIR,
        f"v4_failure_windows_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    performance_file = os.path.join(
        REPORT_DIR,
        f"v4_failure_performance_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    alligator_file = os.path.join(
        REPORT_DIR,
        f"v4_failure_alligator_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    signal_columns = [

        "timestamp",

        "window",

        "period_group",

        "close",

        "atr_pct",

        "ema_spread_pct",

        "ema200_slope_12h_pct",

        "ema200_slope_24h_pct",

        "price_vs_ema200_pct",

        "volume_ratio",

        "alligator_state",

        "alligator_spread_pct",

        "last_fractal_low",

        "breakdown_depth_pct"
    ]


    signal_df[
        signal_columns
    ].to_csv(
        signals_file,
        index=False
    )


    comparison_df.to_csv(
        comparison_file,
        index=False
    )


    window_summary_df.to_csv(
        windows_file,
        index=False
    )


    performance_df.to_csv(
        performance_file,
        index=False
    )


    alligator_df.to_csv(
        alligator_file,
        index=False
    )


    print()
    print("=" * 100)
    print("FAILURE ANALYSIS FINISHED")
    print("=" * 100)

    print(
        f"Signals:     {signals_file}"
    )

    print(
        f"Comparison:  {comparison_file}"
    )

    print(
        f"Windows:     {windows_file}"
    )

    print(
        f"Performance: {performance_file}"
    )

    print(
        f"Alligator:   {alligator_file}"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
