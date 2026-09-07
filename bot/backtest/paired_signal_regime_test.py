import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# IMPORT EXISTING INDICATOR / SIGNAL ENGINE
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

SL_ATR = 1.25
TP_ATR = 3.00

TAKER_FEE_RATE = 0.00055

WINDOW_MONTHS = 6

DEVELOPMENT_START = pd.Timestamp(
    "2023-09-07T16:00:00Z"
)


PRE_SAMPLE_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

DEVELOPMENT_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_3y.parquet"
)


# ============================================================
# LOAD HISTORY
# ============================================================

def load_one_file(
    filepath,
    label
):

    if not os.path.exists(
        filepath
    ):

        raise RuntimeError(
            f"Missing {label} file: {filepath}"
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

    if df.empty:

        raise RuntimeError(
            f"{label} dataset is empty."
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
    print("PAIRED SIGNAL REGIME TEST")
    print("=" * 110)

    print(
        "Every candidate signal is tested independently in BOTH directions."
    )

    print()
    print("Candidate setup is FROZEN:")
    print("  Original V3 signal = SHORT")
    print("  Market regime      = bearish")
    print("  Volatility regime  = normal")

    print()
    print("Paired execution:")
    print("  BREAKOUT = SHORT")
    print("  INVERSE  = LONG")

    print()
    print(f"Entry: next 4h candle OPEN")
    print(f"SL:    {SL_ATR} ATR")
    print(f"TP:    {TP_ATR} ATR")
    print(f"RR:    {TP_ATR / SL_ATR:.2f}")
    print(
        "Same-candle SL+TP ambiguity: conservative SL-first."
    )
    print(
        f"Fee diagnostic: {TAKER_FEE_RATE * 100:.3f}% per side."
    )
    print("Funding: NOT modeled.")
    print("No one-position-at-a-time restriction.")
    print("No V5 filters.")
    print("Supabase: NOT USED")
    print("=" * 110)

    pre = load_one_file(
        PRE_SAMPLE_FILE,
        "Pre-sample"
    )

    development = load_one_file(
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

    duplicates_before = int(
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
        f"Duplicates removed: {duplicates_before}"
    )

    print(
        f"From: {df.iloc[0]['timestamp']}"
    )

    print(
        f"To:   {df.iloc[-1]['timestamp']}"
    )

    expected_step = pd.Timedelta(
        hours=4
    )

    diffs = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    gaps = diffs[
        diffs > expected_step
    ]

    irregular = diffs[
        diffs != expected_step
    ]

    print(
        f"Irregular 4h steps: {len(irregular)}"
    )

    print(
        f"Gaps > 4h: {len(gaps)}"
    )

    if len(gaps) > 0:

        gap_table = pd.DataFrame(
            {
                "timestamp": df.loc[
                    gaps.index,
                    "timestamp"
                ],
                "gap": gaps.values,
            }
        )

        print()
        print("Largest gaps:")

        print(
            gap_table
            .sort_values(
                "gap",
                ascending=False
            )
            .head(10)
            .to_string(
                index=False
            )
        )

    df = bt.remove_open_candle(
        df,
        TIMEFRAME
    )

    print()
    print("Calculating indicators...")

    df = bt.calculate_indicators(
        df
    )

    print(
        "Calculating original V3 signals..."
    )

    df = bt.calculate_scores(
        df
    )

    return add_features(
        df
    )


# ============================================================
# PRE-TRADE FEATURES
# ============================================================

def add_features(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # Exact V4 candidate setup
    # --------------------------------------------------------

    df[
        "candidate_signal"
    ] = (
        (
            df["direction"]
            == "SHORT"
        )
        &
        (
            df["market_regime"]
            == "bearish"
        )
        &
        (
            df["volatility_regime"]
            == "normal"
        )
    )

    # --------------------------------------------------------
    # ATR %
    # --------------------------------------------------------

    df[
        "atr_pct"
    ] = (
        df["atr"]
        / df["close"]
        * 100
    )

    # --------------------------------------------------------
    # EMA features
    # --------------------------------------------------------

    df[
        "ema_spread_pct"
    ] = (
        (
            df["ema_50"]
            - df["ema_200"]
        )
        / df["ema_200"]
        * 100
    )

    df[
        "price_vs_ema200_pct"
    ] = (
        (
            df["close"]
            - df["ema_200"]
        )
        / df["ema_200"]
        * 100
    )

    # 6 x 4h = 24h
    df[
        "ema200_slope_24h_pct"
    ] = (
        df["ema_200"]
        .pct_change(
            6
        )
        * 100
    )

    # 18 x 4h = 72h = 3d
    df[
        "ema200_slope_3d_pct"
    ] = (
        df["ema_200"]
        .pct_change(
            18
        )
        * 100
    )

    # --------------------------------------------------------
    # BTC returns available at signal close
    # --------------------------------------------------------

    df[
        "return_24h_pct"
    ] = (
        df["close"]
        .pct_change(
            6
        )
        * 100
    )

    df[
        "return_3d_pct"
    ] = (
        df["close"]
        .pct_change(
            18
        )
        * 100
    )

    df[
        "return_7d_pct"
    ] = (
        df["close"]
        .pct_change(
            42
        )
        * 100
    )

    # --------------------------------------------------------
    # Alligator state + spread
    # --------------------------------------------------------

    bullish_alligator = (
        (
            df["alligator_lips"]
            > df["alligator_teeth"]
        )
        &
        (
            df["alligator_teeth"]
            > df["alligator_jaw"]
        )
    )

    bearish_alligator = (
        (
            df["alligator_lips"]
            < df["alligator_teeth"]
        )
        &
        (
            df["alligator_teeth"]
            < df["alligator_jaw"]
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
        / df["close"]
        * 100
    )

    # --------------------------------------------------------
    # Confirmed fractal low
    # --------------------------------------------------------

    if "fractal_low" in df.columns:

        confirmed_fractal_low = (
            df["low"]
            .shift(
                2
            )
            .where(
                df["fractal_low"]
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

    elif "fractal_low_level" in df.columns:

        df[
            "last_fractal_low"
        ] = (
            df["fractal_low_level"]
            .ffill()
        )

    else:

        print(
            "WARNING: fractal low column not available."
        )

        df[
            "last_fractal_low"
        ] = np.nan

    df[
        "breakdown_depth_pct"
    ] = (
        (
            df["last_fractal_low"]
            - df["close"]
        )
        / df["last_fractal_low"]
        * 100
    )

    return df


# ============================================================
# WINDOW ID
# ============================================================

def build_window_lookup(
    df
):

    dataset_start = (
        df["timestamp"].min()
    )

    dataset_end = (
        df["timestamp"].max()
    )

    windows = []

    start = dataset_start
    number = 1

    while start <= dataset_end:

        end = (
            start
            + pd.DateOffset(
                months=WINDOW_MONTHS
            )
        )

        windows.append(
            {
                "window": f"W{number}",
                "start": start,
                "end": end,
            }
        )

        start = end
        number += 1

    return windows


def get_window_name(
    timestamp,
    windows
):

    for window in windows:

        if (
            timestamp
            >= window["start"]
            and
            timestamp
            < window["end"]
        ):

            return window[
                "window"
            ]

    return None


# ============================================================
# INDEPENDENT TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction,
    atr_value
):

    if (
        entry_index < 0
        or
        entry_index >= len(df)
    ):

        return {
            "result": "NO_ENTRY",
            "exit_reason": "NO_NEXT_CANDLE",
            "entry_price": np.nan,
            "stop_price": np.nan,
            "target_price": np.nan,
            "exit_price": np.nan,
            "exit_timestamp": pd.NaT,
            "bars_held": np.nan,
            "gross_return_pct": np.nan,
            "net_return_pct_after_fees": np.nan,
            "r_multiple": np.nan,
            "same_candle_ambiguity": False,
        }

    if (
        atr_value is None
        or
        pd.isna(
            atr_value
        )
        or
        atr_value <= 0
    ):

        return {
            "result": "NO_ENTRY",
            "exit_reason": "INVALID_ATR",
            "entry_price": np.nan,
            "stop_price": np.nan,
            "target_price": np.nan,
            "exit_price": np.nan,
            "exit_timestamp": pd.NaT,
            "bars_held": np.nan,
            "gross_return_pct": np.nan,
            "net_return_pct_after_fees": np.nan,
            "r_multiple": np.nan,
            "same_candle_ambiguity": False,
        }

    entry_price = float(
        df.iloc[
            entry_index
        ]["open"]
    )

    if direction == "LONG":

        stop_price = (
            entry_price
            - SL_ATR * atr_value
        )

        target_price = (
            entry_price
            + TP_ATR * atr_value
        )

    elif direction == "SHORT":

        stop_price = (
            entry_price
            + SL_ATR * atr_value
        )

        target_price = (
            entry_price
            - TP_ATR * atr_value
        )

    else:

        raise ValueError(
            f"Unsupported direction: {direction}"
        )

    # --------------------------------------------------------
    # Scan forward independently for this one signal.
    #
    # No one-position-at-a-time restriction.
    # Overlapping trades are intentionally allowed.
    # --------------------------------------------------------

    for j in range(
        entry_index,
        len(df)
    ):

        row = df.iloc[
            j
        ]

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        if direction == "LONG":

            stop_hit = (
                low
                <= stop_price
            )

            target_hit = (
                high
                >= target_price
            )

        else:

            stop_hit = (
                high
                >= stop_price
            )

            target_hit = (
                low
                <= target_price
            )

        same_candle = (
            stop_hit
            and
            target_hit
        )

        # Conservative rule:
        # if both SL and TP are touched in one candle,
        # count the stop first.
        if same_candle:

            result = "LOSS"
            exit_reason = "SL_FIRST_SAME_CANDLE"
            exit_price = stop_price
            r_multiple = -1.0

        elif stop_hit:

            result = "LOSS"
            exit_reason = "SL"
            exit_price = stop_price
            r_multiple = -1.0

        elif target_hit:

            result = "WIN"
            exit_reason = "TP"
            exit_price = target_price
            r_multiple = (
                TP_ATR
                / SL_ATR
            )

        else:

            continue

        if direction == "LONG":

            gross_return_pct = (
                (
                    exit_price
                    - entry_price
                )
                / entry_price
                * 100
            )

        else:

            gross_return_pct = (
                (
                    entry_price
                    - exit_price
                )
                / entry_price
                * 100
            )

        fees_pct = (
            2
            * TAKER_FEE_RATE
            * 100
        )

        net_return_pct = (
            gross_return_pct
            - fees_pct
        )

        return {
            "result": result,
            "exit_reason": exit_reason,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_price": exit_price,
            "exit_timestamp": row[
                "timestamp"
            ],
            "bars_held": (
                j
                - entry_index
                + 1
            ),
            "gross_return_pct":
                gross_return_pct,
            "net_return_pct_after_fees":
                net_return_pct,
            "r_multiple":
                r_multiple,
            "same_candle_ambiguity":
                same_candle,
        }

    # --------------------------------------------------------
    # Dataset ended before SL or TP
    # --------------------------------------------------------

    last_row = df.iloc[
        -1
    ]

    last_close = float(
        last_row["close"]
    )

    if direction == "LONG":

        mark_to_market_pct = (
            (
                last_close
                - entry_price
            )
            / entry_price
            * 100
        )

    else:

        mark_to_market_pct = (
            (
                entry_price
                - last_close
            )
            / entry_price
            * 100
        )

    return {
        "result": "UNRESOLVED",
        "exit_reason": "END_OF_DATA",
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": last_close,
        "exit_timestamp": last_row[
            "timestamp"
        ],
        "bars_held": (
            len(df)
            - entry_index
        ),
        "gross_return_pct":
            mark_to_market_pct,
        "net_return_pct_after_fees":
            (
                mark_to_market_pct
                - 2
                * TAKER_FEE_RATE
                * 100
            ),
        "r_multiple": np.nan,
        "same_candle_ambiguity": False,
    }


# ============================================================
# CLASSIFY PAIR
# ============================================================

def classify_pair(
    breakout_result,
    inverse_result
):

    b = breakout_result
    i = inverse_result

    if (
        b == "WIN"
        and
        i == "LOSS"
    ):

        return (
            "BREAKOUT_ONLY",
            "BREAKOUT"
        )

    if (
        b == "LOSS"
        and
        i == "WIN"
    ):

        return (
            "INVERSE_ONLY",
            "INVERSE"
        )

    if (
        b == "LOSS"
        and
        i == "LOSS"
    ):

        return (
            "BOTH_LOSS",
            "NO_TRADE"
        )

    if (
        b == "WIN"
        and
        i == "WIN"
    ):

        return (
            "BOTH_WIN",
            "AMBIGUOUS"
        )

    return (
        "UNRESOLVED",
        "UNRESOLVED"
    )


# ============================================================
# BUILD PAIRED SIGNAL DATASET
# ============================================================

def build_paired_dataset(
    df
):

    windows = build_window_lookup(
        df
    )

    candidate_indices = (
        df.index[
            df[
                "candidate_signal"
            ]
        ]
        .tolist()
    )

    print()
    print(
        f"Candidate signals found: "
        f"{len(candidate_indices)}"
    )

    rows = []

    for number, signal_index in enumerate(
        candidate_indices,
        start=1
    ):

        if (
            signal_index
            + 1
            >= len(df)
        ):

            continue

        signal_row = df.iloc[
            signal_index
        ]

        entry_index = (
            signal_index
            + 1
        )

        atr_value = float(
            signal_row["atr"]
        )

        breakout = simulate_trade(
            df=df,
            entry_index=entry_index,
            direction="SHORT",
            atr_value=atr_value
        )

        inverse = simulate_trade(
            df=df,
            entry_index=entry_index,
            direction="LONG",
            atr_value=atr_value
        )

        pair_class, preferred_mode = (
            classify_pair(
                breakout_result=breakout[
                    "result"
                ],
                inverse_result=inverse[
                    "result"
                ]
            )
        )

        timestamp = signal_row[
            "timestamp"
        ]

        sample_group = (
            "PRESAMPLE"
            if timestamp
            < DEVELOPMENT_START
            else "DEVELOPMENT"
        )

        row = {
            "signal_number":
                number,

            "signal_timestamp":
                timestamp,

            "entry_timestamp":
                df.iloc[
                    entry_index
                ]["timestamp"],

            "year":
                int(
                    timestamp.year
                ),

            "window":
                get_window_name(
                    timestamp,
                    windows
                ),

            "sample_group":
                sample_group,

            "signal_close":
                signal_row[
                    "close"
                ],

            "atr":
                atr_value,

            "atr_pct":
                signal_row[
                    "atr_pct"
                ],

            "ema_spread_pct":
                signal_row[
                    "ema_spread_pct"
                ],

            "price_vs_ema200_pct":
                signal_row[
                    "price_vs_ema200_pct"
                ],

            "ema200_slope_24h_pct":
                signal_row[
                    "ema200_slope_24h_pct"
                ],

            "ema200_slope_3d_pct":
                signal_row[
                    "ema200_slope_3d_pct"
                ],

            "volume_ratio":
                signal_row[
                    "volume_ratio"
                ],

            "alligator_state":
                signal_row[
                    "alligator_state"
                ],

            "alligator_spread_pct":
                signal_row[
                    "alligator_spread_pct"
                ],

            "last_fractal_low":
                signal_row[
                    "last_fractal_low"
                ],

            "breakdown_depth_pct":
                signal_row[
                    "breakdown_depth_pct"
                ],

            "return_24h_pct":
                signal_row[
                    "return_24h_pct"
                ],

            "return_3d_pct":
                signal_row[
                    "return_3d_pct"
                ],

            "return_7d_pct":
                signal_row[
                    "return_7d_pct"
                ],

            "pair_class":
                pair_class,

            "preferred_mode":
                preferred_mode,

            # BREAKOUT SHORT
            "breakout_result":
                breakout[
                    "result"
                ],

            "breakout_exit_reason":
                breakout[
                    "exit_reason"
                ],

            "breakout_entry_price":
                breakout[
                    "entry_price"
                ],

            "breakout_stop_price":
                breakout[
                    "stop_price"
                ],

            "breakout_target_price":
                breakout[
                    "target_price"
                ],

            "breakout_exit_price":
                breakout[
                    "exit_price"
                ],

            "breakout_exit_timestamp":
                breakout[
                    "exit_timestamp"
                ],

            "breakout_bars_held":
                breakout[
                    "bars_held"
                ],

            "breakout_r_multiple":
                breakout[
                    "r_multiple"
                ],

            "breakout_gross_return_pct":
                breakout[
                    "gross_return_pct"
                ],

            "breakout_net_return_pct":
                breakout[
                    "net_return_pct_after_fees"
                ],

            "breakout_same_candle_ambiguity":
                breakout[
                    "same_candle_ambiguity"
                ],

            # INVERSE LONG
            "inverse_result":
                inverse[
                    "result"
                ],

            "inverse_exit_reason":
                inverse[
                    "exit_reason"
                ],

            "inverse_entry_price":
                inverse[
                    "entry_price"
                ],

            "inverse_stop_price":
                inverse[
                    "stop_price"
                ],

            "inverse_target_price":
                inverse[
                    "target_price"
                ],

            "inverse_exit_price":
                inverse[
                    "exit_price"
                ],

            "inverse_exit_timestamp":
                inverse[
                    "exit_timestamp"
                ],

            "inverse_bars_held":
                inverse[
                    "bars_held"
                ],

            "inverse_r_multiple":
                inverse[
                    "r_multiple"
                ],

            "inverse_gross_return_pct":
                inverse[
                    "gross_return_pct"
                ],

            "inverse_net_return_pct":
                inverse[
                    "net_return_pct_after_fees"
                ],

            "inverse_same_candle_ambiguity":
                inverse[
                    "same_candle_ambiguity"
                ],
        }

        rows.append(
            row
        )

        if (
            number % 50
            == 0
        ):

            print(
                f"Processed "
                f"{number}/{len(candidate_indices)} "
                f"signals..."
            )

    result_df = pd.DataFrame(
        rows
    )

    return result_df


# ============================================================
# CLASS SUMMARY
# ============================================================

def build_class_summary(
    paired_df
):

    counts = (
        paired_df[
            "pair_class"
        ]
        .value_counts()
        .rename_axis(
            "pair_class"
        )
        .reset_index(
            name="signals"
        )
    )

    counts[
        "pct_of_signals"
    ] = (
        counts[
            "signals"
        ]
        / len(
            paired_df
        )
        * 100
    )

    return counts


# ============================================================
# FEATURE SUMMARY BY PREFERRED MODE
# ============================================================

def build_feature_summary(
    paired_df
):

    numeric_features = [
        "atr_pct",
        "ema_spread_pct",
        "price_vs_ema200_pct",
        "ema200_slope_24h_pct",
        "ema200_slope_3d_pct",
        "volume_ratio",
        "alligator_spread_pct",
        "breakdown_depth_pct",
        "return_24h_pct",
        "return_3d_pct",
        "return_7d_pct",
    ]

    rows = []

    for preferred_mode in [
        "BREAKOUT",
        "INVERSE",
        "NO_TRADE",
        "AMBIGUOUS"
    ]:

        subset = paired_df[
            paired_df[
                "preferred_mode"
            ]
            == preferred_mode
        ]

        if subset.empty:
            continue

        for feature in numeric_features:

            values = pd.to_numeric(
                subset[
                    feature
                ],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "preferred_mode":
                        preferred_mode,

                    "feature":
                        feature,

                    "n":
                        int(
                            len(values)
                        ),

                    "mean":
                        values.mean(),

                    "median":
                        values.median(),

                    "q25":
                        values.quantile(
                            0.25
                        ),

                    "q75":
                        values.quantile(
                            0.75
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PERIOD SUMMARY
# ============================================================

def summarize_one_period_group(
    paired_df,
    period_column,
    period_type
):

    rows = []

    for period_value, subset in paired_df.groupby(
        period_column,
        dropna=False
    ):

        total = len(
            subset
        )

        def count_mode(
            value
        ):

            return int(
                (
                    subset[
                        "preferred_mode"
                    ]
                    == value
                )
                .sum()
            )

        breakout = count_mode(
            "BREAKOUT"
        )

        inverse = count_mode(
            "INVERSE"
        )

        no_trade = count_mode(
            "NO_TRADE"
        )

        ambiguous = count_mode(
            "AMBIGUOUS"
        )

        unresolved = count_mode(
            "UNRESOLVED"
        )

        rows.append(
            {
                "period_type":
                    period_type,

                "period":
                    period_value,

                "signals":
                    total,

                "breakout_only":
                    breakout,

                "inverse_only":
                    inverse,

                "no_trade_both_loss":
                    no_trade,

                "ambiguous_both_win":
                    ambiguous,

                "unresolved":
                    unresolved,

                "breakout_pct":
                    (
                        breakout
                        / total
                        * 100
                        if total
                        else 0
                    ),

                "inverse_pct":
                    (
                        inverse
                        / total
                        * 100
                        if total
                        else 0
                    ),

                "no_trade_pct":
                    (
                        no_trade
                        / total
                        * 100
                        if total
                        else 0
                    ),

                "ambiguous_pct":
                    (
                        ambiguous
                        / total
                        * 100
                        if total
                        else 0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_period_summary(
    paired_df
):

    frames = [
        summarize_one_period_group(
            paired_df,
            "year",
            "year"
        ),
        summarize_one_period_group(
            paired_df,
            "window",
            "window"
        ),
        summarize_one_period_group(
            paired_df,
            "sample_group",
            "sample_group"
        ),
    ]

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# ALLIGATOR SUMMARY
# ============================================================

def build_alligator_summary(
    paired_df
):

    grouped = (
        paired_df
        .groupby(
            [
                "preferred_mode",
                "alligator_state"
            ]
        )
        .size()
        .reset_index(
            name="signals"
        )
    )

    totals = (
        grouped
        .groupby(
            "preferred_mode"
        )[
            "signals"
        ]
        .transform(
            "sum"
        )
    )

    grouped[
        "pct_within_mode"
    ] = (
        grouped[
            "signals"
        ]
        / totals
        * 100
    )

    return grouped


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    paired_df,
    class_summary,
    period_summary
):

    print()
    print("=" * 100)
    print("PAIR CLASS SUMMARY")
    print("=" * 100)

    print(
        class_summary
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("SAMPLE GROUP SUMMARY")
    print("=" * 120)

    sample = (
        period_summary[
            period_summary[
                "period_type"
            ]
            == "sample_group"
        ]
    )

    print(
        sample
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 140)
    print("CALENDAR YEAR LABEL DISTRIBUTION")
    print("=" * 140)

    years = (
        period_summary[
            period_summary[
                "period_type"
            ]
            == "year"
        ]
    )

    print(
        years
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 140)
    print("6-MONTH WINDOW LABEL DISTRIBUTION")
    print("=" * 140)

    windows = (
        period_summary[
            period_summary[
                "period_type"
            ]
            == "window"
        ]
    )

    print(
        windows
        .to_string(
            index=False
        )
    )

    same_candle_breakout = int(
        paired_df[
            "breakout_same_candle_ambiguity"
        ]
        .sum()
    )

    same_candle_inverse = int(
        paired_df[
            "inverse_same_candle_ambiguity"
        ]
        .sum()
    )

    print()
    print("=" * 100)
    print("EXECUTION DIAGNOSTICS")
    print("=" * 100)

    print(
        f"Signals evaluated: "
        f"{len(paired_df)}"
    )

    print(
        f"BREAKOUT same-candle SL+TP cases: "
        f"{same_candle_breakout}"
    )

    print(
        f"INVERSE same-candle SL+TP cases: "
        f"{same_candle_inverse}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_combined_history()

    paired_df = build_paired_dataset(
        df
    )

    if paired_df.empty:

        raise RuntimeError(
            "No paired signals were produced."
        )

    class_summary = (
        build_class_summary(
            paired_df
        )
    )

    feature_summary = (
        build_feature_summary(
            paired_df
        )
    )

    period_summary = (
        build_period_summary(
            paired_df
        )
    )

    alligator_summary = (
        build_alligator_summary(
            paired_df
        )
    )

    print_summary(
        paired_df,
        class_summary,
        period_summary
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    paired_file = os.path.join(
        REPORT_DIR,
        f"paired_signal_results_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    class_file = os.path.join(
        REPORT_DIR,
        f"paired_signal_class_summary_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    feature_file = os.path.join(
        REPORT_DIR,
        f"paired_signal_feature_summary_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    period_file = os.path.join(
        REPORT_DIR,
        f"paired_signal_period_summary_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    alligator_file = os.path.join(
        REPORT_DIR,
        f"paired_signal_alligator_summary_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    paired_df.to_csv(
        paired_file,
        index=False
    )

    class_summary.to_csv(
        class_file,
        index=False
    )

    feature_summary.to_csv(
        feature_file,
        index=False
    )

    period_summary.to_csv(
        period_file,
        index=False
    )

    alligator_summary.to_csv(
        alligator_file,
        index=False
    )

    print()
    print("=" * 100)
    print("PAIRED SIGNAL TEST FINISHED")
    print("=" * 100)
    print(f"Signal-level results: {paired_file}")
    print(f"Class summary:        {class_file}")
    print(f"Feature summary:      {feature_file}")
    print(f"Period summary:       {period_file}")
    print(f"Alligator summary:    {alligator_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()
