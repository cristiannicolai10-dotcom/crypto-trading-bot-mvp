import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# IMPORT V3 INVERSE ENGINE
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
    print("=" * 90)
    print("V4_B WALK-FORWARD STABILITY TEST")
    print("=" * 90)

    print(
        f"File: {filepath}"
    )

    print(
        f"SL: {SL_ATR} ATR"
    )

    print(
        f"TP: {TP_ATR} ATR"
    )

    print(
        f"Window: {WINDOW_MONTHS} months"
    )

    print(
        "Supabase: NOT USED"
    )

    print("=" * 90)

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
# V4_B SIGNAL
# ============================================================

def get_v4_signal_mask(
    df
):

    return (

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


# ============================================================
# BUILD 6-MONTH WINDOWS
# ============================================================

def build_windows(
    df
):

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

    windows = []

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

        windows.append(
            {
                "window_number":
                    window_number,

                "start":
                    start,

                "end":
                    end
            }
        )

        start = end

        window_number += 1

    return windows


# ============================================================
# RUN WINDOW
# ============================================================

def run_window(
    full_df,
    window
):

    df = full_df.copy()

    window_start = (
        window["start"]
    )

    window_end = (
        window["end"]
    )

    v4_mask = (
        get_v4_signal_mask(
            df
        )
    )

    time_mask = (

        (
            df[
                "timestamp"
            ]
            >= window_start
        )

        &

        (
            df[
                "timestamp"
            ]
            < window_end
        )
    )

    allowed_signal = (
        v4_mask
        & time_mask
    )

    candidate_signals = int(
        allowed_signal.sum()
    )

    # --------------------------------------------------------
    # Disable all signals outside this time window.
    #
    # We keep the complete candle dataset so a trade entered
    # near the end of a window can still exit naturally on
    # subsequent candles.
    # --------------------------------------------------------

    df.loc[
        ~allowed_signal,
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
            df
        )

        summary = bt.build_summary(
            trades=trades,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            df=df,
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

        "window":
            window[
                "window_number"
            ],

        "start":
            window_start.isoformat(),

        "end":
            (
                window_end
                - pd.Timedelta(
                    microseconds=1
                )
            ).isoformat(),

        "candidate_signals":
            candidate_signals,

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
                "total_return_pct",
                0
            ),

        "max_drawdown_pct":
            summary.get(
                "max_drawdown_pct",
                0
            ),

        "fees_usd":
            summary.get(
                "total_fees_usd",
                0
            ),

        "average_trade_pct":
            summary.get(
                "average_trade_pct"
            ),

        "max_consecutive_losses":
            summary.get(
                "max_consecutive_losses"
            ),

        "final_balance":
            summary.get(
                "final_balance",
                bt.INITIAL_BALANCE
            )
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    pf = (
        result[
            "profit_factor"
        ]
    )

    win_rate = (
        result[
            "win_rate_pct"
        ]
    )

    pf_text = (
        f"{pf:.3f}"
        if pf is not None
        else "N/A"
    )

    win_text = (
        f"{win_rate:.2f}%"
        if win_rate is not None
        else "N/A"
    )

    print(
        f"W{result['window']:<2} | "
        f"{result['start'][:10]} "
        f"→ "
        f"{result['end'][:10]} | "
        f"Signals={result['candidate_signals']:<3} | "
        f"Trades={result['trades']:<3} | "
        f"Win={win_text:<7} | "
        f"PF={pf_text:<6} | "
        f"Return={result['return_pct']:+.2f}% | "
        f"DD={result['max_drawdown_pct']:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    windows = build_windows(
        df
    )

    results = []

    print()
    print("=" * 110)

    print(
        "6-MONTH WINDOWS"
    )

    print("=" * 110)

    for window in windows:

        result = run_window(
            full_df=df,
            window=window
        )

        results.append(
            result
        )

        print_result(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # STABILITY SUMMARY
    # ========================================================

    valid_pf = (
        results_df[
            "profit_factor"
        ]
        .dropna()
    )

    positive_windows = int(
        (
            results_df[
                "return_pct"
            ]
            > 0
        )
        .sum()
    )

    negative_windows = int(
        (
            results_df[
                "return_pct"
            ]
            < 0
        )
        .sum()
    )

    pf_above_one = int(
        (
            valid_pf
            > 1
        )
        .sum()
    )

    total_windows = int(
        len(
            results_df
        )
    )

    total_trades = int(
        results_df[
            "trades"
        ]
        .sum()
    )

    median_pf = (
        valid_pf.median()
        if not valid_pf.empty
        else None
    )

    mean_pf = (
        valid_pf.mean()
        if not valid_pf.empty
        else None
    )

    worst_return = (
        results_df[
            "return_pct"
        ]
        .min()
    )

    best_return = (
        results_df[
            "return_pct"
        ]
        .max()
    )

    worst_dd = (
        results_df[
            "max_drawdown_pct"
        ]
        .max()
    )

    print()
    print("=" * 90)

    print(
        "WALK-FORWARD STABILITY SUMMARY"
    )

    print("=" * 90)

    print(
        f"Windows: {total_windows}"
    )

    print(
        f"Positive windows: "
        f"{positive_windows}/{total_windows}"
    )

    print(
        f"Negative windows: "
        f"{negative_windows}/{total_windows}"
    )

    print(
        f"PF > 1 windows: "
        f"{pf_above_one}/{len(valid_pf)}"
    )

    print(
        f"Total trades across windows: "
        f"{total_trades}"
    )

    if median_pf is not None:

        print(
            f"Median PF: "
            f"{median_pf:.3f}"
        )

        print(
            f"Mean PF: "
            f"{mean_pf:.3f}"
        )

    print(
        f"Best window return: "
        f"{best_return:+.2f}%"
    )

    print(
        f"Worst window return: "
        f"{worst_return:+.2f}%"
    )

    print(
        f"Worst window DD: "
        f"{worst_dd:.2f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    report_file = os.path.join(
        REPORT_DIR,
        f"v4_walk_forward_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    results_df.to_csv(
        report_file,
        index=False
    )

    print()
    print(
        f"Full report: "
        f"{report_file}"
    )

    print("=" * 90)


if __name__ == "__main__":

    main()
