import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# IMPORT EXISTING V3 INVERSE ENGINE
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
# FROZEN STRATEGY CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "historical",
    "BTCUSDT_4h_presample_2019_2023.parquet"
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

DEVELOPMENT_START = pd.Timestamp(
    "2023-09-07T16:00:00Z"
)


# ============================================================
# LOAD + CALCULATE
# ============================================================

def load_dataset():

    if not os.path.exists(
        DATA_FILE
    ):
        raise RuntimeError(
            f"File not found: {DATA_FILE}"
        )

    print()
    print("=" * 100)
    print("V4_B UNTOUCHED PRE-SAMPLE VALIDATION")
    print("=" * 100)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print("Strategy parameters are FROZEN:")
    print("  Original V3 signal = SHORT")
    print("  Market regime      = bearish")
    print("  Volatility regime  = normal")
    print("  Execution           = inverse LONG")
    print(f"  SL                  = {SL_ATR} ATR")
    print(f"  TP                  = {TP_ATR} ATR")
    print("No V5 filters.")
    print("Supabase: NOT USED")
    print("=" * 100)

    df = pd.read_parquet(
        DATA_FILE
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
            subset=["timestamp"]
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
            "Historical dataset is empty."
        )

    overlap = int(
        (
            df["timestamp"]
            >= DEVELOPMENT_START
        )
        .sum()
    )

    if overlap:
        raise RuntimeError(
            f"Dataset contains {overlap} candles "
            f"at/after development start {DEVELOPMENT_START}."
        )

    print(f"Candles: {len(df):,}")
    print(f"From: {df.iloc[0]['timestamp']}")
    print(f"To:   {df.iloc[-1]['timestamp']}")

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

    df["frozen_v4_signal"] = (
        (df["direction"] == "SHORT")
        &
        (df["market_regime"] == "bearish")
        &
        (df["volatility_regime"] == "normal")
    )

    return df


# ============================================================
# PERIODS
# ============================================================

def build_periods(df):

    dataset_start = (
        df["timestamp"].min()
    )

    dataset_end = (
        df["timestamp"].max()
    )

    end_exclusive = (
        dataset_end
        + pd.Timedelta(
            microseconds=1
        )
    )

    periods = [
        {
            "period": "FULL_PRESAMPLE",
            "start": dataset_start,
            "end": end_exclusive,
        }
    ]

    # Calendar years.
    years = sorted(
        df["timestamp"]
        .dt
        .year
        .unique()
        .tolist()
    )

    for year in years:

        start = pd.Timestamp(
            f"{year}-01-01T00:00:00Z"
        )

        end = pd.Timestamp(
            f"{year + 1}-01-01T00:00:00Z"
        )

        actual_start = max(
            start,
            dataset_start
        )

        actual_end = min(
            end,
            end_exclusive
        )

        if actual_start < actual_end:

            periods.append(
                {
                    "period": f"YEAR_{year}",
                    "start": actual_start,
                    "end": actual_end,
                }
            )

    # Consecutive six-month windows.
    start = dataset_start
    window_number = 1

    while start <= dataset_end:

        end = (
            start
            + pd.DateOffset(
                months=WINDOW_MONTHS
            )
        )

        if end > end_exclusive:
            end = end_exclusive

        periods.append(
            {
                "period": f"W{window_number}",
                "start": start,
                "end": end,
            }
        )

        start = end
        window_number += 1

    return periods


# ============================================================
# RUN BACKTEST
# ============================================================

def run_period(
    df,
    period
):

    test_df = df.copy()

    time_mask = (
        (
            test_df["timestamp"]
            >= period["start"]
        )
        &
        (
            test_df["timestamp"]
            < period["end"]
        )
    )

    allowed = (
        test_df["frozen_v4_signal"]
        &
        time_mask
    )

    candidate_signals = int(
        allowed.sum()
    )

    test_df.loc[
        ~allowed,
        "direction"
    ] = "HOLD"

    old_sl = bt.ATR_STOP_MULTIPLIER
    old_tp = bt.ATR_TARGET_MULTIPLIER
    old_warmup = bt.WARMUP_BARS

    try:

        bt.ATR_STOP_MULTIPLIER = SL_ATR
        bt.ATR_TARGET_MULTIPLIER = TP_ATR
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

        bt.ATR_STOP_MULTIPLIER = old_sl
        bt.ATR_TARGET_MULTIPLIER = old_tp
        bt.WARMUP_BARS = old_warmup

    return {
        "period":
            period["period"],

        "start":
            period["start"].isoformat(),

        "end":
            (
                period["end"]
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
            ),
    }


# ============================================================
# PRINT HELPERS
# ============================================================

def fmt(value, digits=3):

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass

    return f"{value:.{digits}f}"


def print_row(result):

    print(
        f"{result['period']:<16} | "
        f"Signals={result['candidate_signals']:<3} | "
        f"Trades={result['trades']:<3} | "
        f"Win={fmt(result['win_rate_pct'], 2):>6} | "
        f"PF={fmt(result['profit_factor'], 3):>6} | "
        f"Return={result['return_pct']:+7.2f}% | "
        f"DD={result['max_drawdown_pct']:6.2f}% | "
        f"MaxL={result['max_consecutive_losses']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    periods = build_periods(
        df
    )

    results = []

    print()
    print("=" * 115)
    print("RUNNING FROZEN V4_B VALIDATION")
    print("=" * 115)

    for period in periods:

        result = run_period(
            df,
            period
        )

        results.append(
            result
        )

        print_row(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Stability summary for 6-month windows
    # --------------------------------------------------------

    windows = (
        results_df[
            results_df[
                "period"
            ]
            .str
            .match(r"^W\d+$")
        ]
        .copy()
    )

    valid_pf = (
        windows[
            "profit_factor"
        ]
        .dropna()
    )

    positive_windows = int(
        (
            windows["return_pct"] > 0
        )
        .sum()
    )

    negative_windows = int(
        (
            windows["return_pct"] < 0
        )
        .sum()
    )

    pf_above_one = int(
        (
            valid_pf > 1
        )
        .sum()
    )

    # --------------------------------------------------------
    # Calendar-year summary
    # --------------------------------------------------------

    years = (
        results_df[
            results_df[
                "period"
            ]
            .str
            .startswith(
                "YEAR_"
            )
        ]
        .copy()
    )

    positive_years = int(
        (
            years[
                "return_pct"
            ]
            > 0
        )
        .sum()
    )

    negative_years = int(
        (
            years[
                "return_pct"
            ]
            < 0
        )
        .sum()
    )

    print()
    print("=" * 100)
    print("PRE-SAMPLE STABILITY SUMMARY")
    print("=" * 100)

    print(
        f"6-month windows: "
        f"{len(windows)}"
    )

    print(
        f"Positive windows: "
        f"{positive_windows}/{len(windows)}"
    )

    print(
        f"Negative windows: "
        f"{negative_windows}/{len(windows)}"
    )

    print(
        f"PF > 1 windows: "
        f"{pf_above_one}/{len(valid_pf)}"
    )

    if not valid_pf.empty:

        print(
            f"Median window PF: "
            f"{valid_pf.median():.3f}"
        )

        print(
            f"Mean window PF:   "
            f"{valid_pf.mean():.3f}"
        )

    print(
        f"Positive calendar years: "
        f"{positive_years}/{len(years)}"
    )

    print(
        f"Negative calendar years: "
        f"{negative_years}/{len(years)}"
    )

    if not windows.empty:

        print(
            f"Best window return:  "
            f"{windows['return_pct'].max():+.2f}%"
        )

        print(
            f"Worst window return: "
            f"{windows['return_pct'].min():+.2f}%"
        )

        print(
            f"Worst window DD:     "
            f"{windows['max_drawdown_pct'].max():.2f}%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    report_file = os.path.join(
        REPORT_DIR,
        f"v4_presample_validation_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    results_df.to_csv(
        report_file,
        index=False
    )

    print()
    print("=" * 100)
    print("VALIDATION FINISHED")
    print("=" * 100)
    print(f"Full report: {report_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()
