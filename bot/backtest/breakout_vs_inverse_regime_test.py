import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# IMPORT EXISTING ENGINES
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )

import backtest_engine as bt_breakout
import backtest_inverse as bt_inverse


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

# Apples-to-apples comparison:
# both engines use the exact same candidate signals and
# the exact same ATR distances.
SL_ATR = 1.25
TP_ATR = 3.00

WINDOW_MONTHS = 6


PRE_SAMPLE_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

DEVELOPMENT_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_3y.parquet"
)


# ============================================================
# LOAD + MERGE HISTORY
# ============================================================

def load_one_file(filepath, label):

    if not os.path.exists(filepath):
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
    print("BREAKOUT vs INVERSE REGIME TEST")
    print("=" * 110)

    print(
        "Purpose: test whether the SAME bearish breakdown setup "
        "should sometimes be followed and sometimes faded."
    )

    print()
    print("Frozen candidate setup:")
    print("  Original V3 signal = SHORT")
    print("  Market regime      = bearish")
    print("  Volatility regime  = normal")

    print()
    print("Mode BREAKOUT:")
    print("  Execute original SHORT")

    print()
    print("Mode INVERSE:")
    print("  Execute inverse LONG")

    print()
    print(f"Both modes: SL={SL_ATR} ATR | TP={TP_ATR} ATR")
    print("No V5 filters.")
    print("No parameter optimization.")
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

    duplicate_count_before = int(
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
        f"Duplicates removed at merge: "
        f"{duplicate_count_before}"
    )

    print(
        f"Combined from: "
        f"{df.iloc[0]['timestamp']}"
    )

    print(
        f"Combined to:   "
        f"{df.iloc[-1]['timestamp']}"
    )

    # --------------------------------------------------------
    # Continuity diagnostics
    # --------------------------------------------------------

    expected_step = pd.Timedelta(
        hours=4
    )

    diffs = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    irregular = diffs[
        diffs != expected_step
    ]

    gaps = diffs[
        diffs > expected_step
    ]

    print(
        f"Irregular 4h steps: "
        f"{len(irregular)}"
    )

    print(
        f"Gaps > 4h: "
        f"{len(gaps)}"
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

    # Remove current open candle if the combined dataset
    # accidentally includes one.
    df = bt_inverse.remove_open_candle(
        df,
        TIMEFRAME
    )

    print()
    print("Calculating indicators...")

    df = bt_inverse.calculate_indicators(
        df
    )

    print(
        "Calculating original V3 signals..."
    )

    df = bt_inverse.calculate_scores(
        df
    )

    # Exact V4 regime setup discovered earlier.
    df["candidate_signal"] = (
        (df["direction"] == "SHORT")
        &
        (df["market_regime"] == "bearish")
        &
        (df["volatility_regime"] == "normal")
    )

    print(
        f"Total candidate signals: "
        f"{int(df['candidate_signal'].sum())}"
    )

    return df


# ============================================================
# BUILD PERIODS
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
            "period": "FULL_2020_2026",
            "period_type": "full",
            "start": dataset_start,
            "end": end_exclusive,
        }
    ]

    # --------------------------------------------------------
    # Calendar years
    # --------------------------------------------------------

    years = sorted(
        df["timestamp"]
        .dt
        .year
        .unique()
        .tolist()
    )

    for year in years:

        year_start = pd.Timestamp(
            f"{year}-01-01T00:00:00Z"
        )

        year_end = pd.Timestamp(
            f"{year + 1}-01-01T00:00:00Z"
        )

        start = max(
            year_start,
            dataset_start
        )

        end = min(
            year_end,
            end_exclusive
        )

        if start < end:

            periods.append(
                {
                    "period": f"YEAR_{year}",
                    "period_type": "year",
                    "start": start,
                    "end": end,
                }
            )

    # --------------------------------------------------------
    # Consecutive 6-month windows
    # --------------------------------------------------------

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
                "period_type": "window",
                "start": start,
                "end": end,
            }
        )

        start = end
        window_number += 1

    return periods


# ============================================================
# ENGINE CONFIG HELPERS
# ============================================================

def set_engine_parameters(
    engine,
    sl,
    tp,
    warmup
):

    old_values = {
        "sl":
            engine.ATR_STOP_MULTIPLIER,

        "tp":
            engine.ATR_TARGET_MULTIPLIER,

        "warmup":
            engine.WARMUP_BARS,
    }

    engine.ATR_STOP_MULTIPLIER = sl
    engine.ATR_TARGET_MULTIPLIER = tp
    engine.WARMUP_BARS = warmup

    return old_values


def restore_engine_parameters(
    engine,
    old_values
):

    engine.ATR_STOP_MULTIPLIER = (
        old_values["sl"]
    )

    engine.ATR_TARGET_MULTIPLIER = (
        old_values["tp"]
    )

    engine.WARMUP_BARS = (
        old_values["warmup"]
    )


# ============================================================
# RUN ONE MODE / PERIOD
# ============================================================

def run_one(
    df,
    period,
    mode
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
        test_df["candidate_signal"]
        &
        time_mask
    )

    candidate_signals = int(
        allowed.sum()
    )

    # Keep only the exact shared signal set.
    test_df.loc[
        ~allowed,
        "direction"
    ] = "HOLD"

    if mode == "BREAKOUT":

        engine = bt_breakout

        # Original V3 SHORT remains SHORT.
        # backtest_engine follows the signal.

    elif mode == "INVERSE":

        engine = bt_inverse

        # Original V3 SHORT remains SHORT.
        # backtest_inverse flips it to LONG.

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    old_values = set_engine_parameters(
        engine=engine,
        sl=SL_ATR,
        tp=TP_ATR,
        warmup=250
    )

    try:

        (
            trades,
            final_balance,
            max_drawdown
        ) = engine.run_backtest(
            test_df
        )

        summary = engine.build_summary(
            trades=trades,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            df=test_df,
            final_balance=final_balance,
            max_drawdown=max_drawdown
        )

    finally:

        restore_engine_parameters(
            engine,
            old_values
        )

    return {
        "mode":
            mode,

        "period":
            period["period"],

        "period_type":
            period["period_type"],

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
                engine.INITIAL_BALANCE
            ),

        "sl_atr":
            SL_ATR,

        "tp_atr":
            TP_ATR,

        "risk_reward":
            TP_ATR / SL_ATR,
    }


# ============================================================
# PRINT HELPERS
# ============================================================

def fmt(
    value,
    digits=3
):

    if value is None:
        return "N/A"

    try:

        if pd.isna(value):
            return "N/A"

    except TypeError:
        pass

    return f"{value:.{digits}f}"


def print_result(
    result
):

    print(
        f"{result['mode']:<8} | "
        f"{result['period']:<15} | "
        f"Signals={result['candidate_signals']:<3} | "
        f"Trades={result['trades']:<3} | "
        f"Win={fmt(result['win_rate_pct'], 2):>6} | "
        f"PF={fmt(result['profit_factor'], 3):>6} | "
        f"Return={result['return_pct']:+7.2f}% | "
        f"DD={result['max_drawdown_pct']:6.2f}%"
    )


# ============================================================
# BUILD SIDE-BY-SIDE COMPARISON
# ============================================================

def build_comparison(
    results_df
):

    key_columns = [
        "period",
        "period_type",
        "start",
        "end",
    ]

    breakout = (
        results_df[
            results_df["mode"]
            == "BREAKOUT"
        ]
        .copy()
    )

    inverse = (
        results_df[
            results_df["mode"]
            == "INVERSE"
        ]
        .copy()
    )

    breakout = breakout.rename(
        columns={
            "candidate_signals":
                "candidate_signals_breakout",

            "trades":
                "trades_breakout",

            "win_rate_pct":
                "win_rate_breakout",

            "profit_factor":
                "pf_breakout",

            "return_pct":
                "return_breakout_pct",

            "max_drawdown_pct":
                "dd_breakout_pct",

            "max_consecutive_losses":
                "max_losses_breakout",
        }
    )

    inverse = inverse.rename(
        columns={
            "candidate_signals":
                "candidate_signals_inverse",

            "trades":
                "trades_inverse",

            "win_rate_pct":
                "win_rate_inverse",

            "profit_factor":
                "pf_inverse",

            "return_pct":
                "return_inverse_pct",

            "max_drawdown_pct":
                "dd_inverse_pct",

            "max_consecutive_losses":
                "max_losses_inverse",
        }
    )

    keep_breakout = key_columns + [
        "candidate_signals_breakout",
        "trades_breakout",
        "win_rate_breakout",
        "pf_breakout",
        "return_breakout_pct",
        "dd_breakout_pct",
        "max_losses_breakout",
    ]

    keep_inverse = key_columns + [
        "candidate_signals_inverse",
        "trades_inverse",
        "win_rate_inverse",
        "pf_inverse",
        "return_inverse_pct",
        "dd_inverse_pct",
        "max_losses_inverse",
    ]

    comparison = breakout[
        keep_breakout
    ].merge(
        inverse[
            keep_inverse
        ],
        on=key_columns,
        how="inner"
    )

    comparison[
        "return_spread_breakout_minus_inverse"
    ] = (
        comparison[
            "return_breakout_pct"
        ]
        -
        comparison[
            "return_inverse_pct"
        ]
    )

    comparison[
        "breakout_positive"
    ] = (
        comparison[
            "return_breakout_pct"
        ]
        > 0
    )

    comparison[
        "inverse_positive"
    ] = (
        comparison[
            "return_inverse_pct"
        ]
        > 0
    )

    comparison[
        "inverse_loses_breakout_wins"
    ] = (
        comparison[
            "breakout_positive"
        ]
        &
        ~comparison[
            "inverse_positive"
        ]
    )

    comparison[
        "breakout_loses_inverse_wins"
    ] = (
        ~comparison[
            "breakout_positive"
        ]
        &
        comparison[
            "inverse_positive"
        ]
    )

    def choose_winner(row):

        b = row[
            "return_breakout_pct"
        ]

        i = row[
            "return_inverse_pct"
        ]

        if b > i:
            return "BREAKOUT"

        if i > b:
            return "INVERSE"

        return "TIE"

    comparison[
        "winner_by_return"
    ] = comparison.apply(
        choose_winner,
        axis=1
    )

    return comparison


# ============================================================
# WINDOW DIAGNOSTIC SUMMARY
# ============================================================

def print_window_diagnostic(
    comparison_df
):

    windows = (
        comparison_df[
            comparison_df[
                "period_type"
            ]
            == "window"
        ]
        .copy()
    )

    print()
    print("=" * 130)
    print("6-MONTH WINDOW COMPARISON")
    print("=" * 130)

    display_columns = [
        "period",
        "trades_breakout",
        "pf_breakout",
        "return_breakout_pct",
        "trades_inverse",
        "pf_inverse",
        "return_inverse_pct",
        "winner_by_return",
    ]

    print(
        windows[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    both_positive = int(
        (
            windows[
                "breakout_positive"
            ]
            &
            windows[
                "inverse_positive"
            ]
        )
        .sum()
    )

    breakout_only = int(
        windows[
            "inverse_loses_breakout_wins"
        ]
        .sum()
    )

    inverse_only = int(
        windows[
            "breakout_loses_inverse_wins"
        ]
        .sum()
    )

    both_negative = int(
        (
            ~windows[
                "breakout_positive"
            ]
            &
            ~windows[
                "inverse_positive"
            ]
        )
        .sum()
    )

    print()
    print("=" * 100)
    print("REGIME-SWITCHING DIAGNOSTIC")
    print("=" * 100)

    print(
        f"Windows tested: "
        f"{len(windows)}"
    )

    print(
        f"Both modes positive: "
        f"{both_positive}"
    )

    print(
        f"BREAKOUT positive / INVERSE negative: "
        f"{breakout_only}"
    )

    print(
        f"INVERSE positive / BREAKOUT negative: "
        f"{inverse_only}"
    )

    print(
        f"Both modes negative: "
        f"{both_negative}"
    )

    breakout_wins = int(
        (
            windows[
                "winner_by_return"
            ]
            == "BREAKOUT"
        )
        .sum()
    )

    inverse_wins = int(
        (
            windows[
                "winner_by_return"
            ]
            == "INVERSE"
        )
        .sum()
    )

    print(
        f"BREAKOUT wins by return: "
        f"{breakout_wins}/{len(windows)}"
    )

    print(
        f"INVERSE wins by return: "
        f"{inverse_wins}/{len(windows)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_combined_history()

    periods = build_periods(
        df
    )

    results = []

    print()
    print("=" * 120)
    print("RUNNING BOTH ENGINES")
    print("=" * 120)

    for mode in [
        "BREAKOUT",
        "INVERSE"
    ]:

        print()
        print("-" * 120)
        print(
            f"MODE: {mode}"
        )
        print("-" * 120)

        for period in periods:

            result = run_one(
                df=df,
                period=period,
                mode=mode
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

    comparison_df = build_comparison(
        results_df
    )

    # --------------------------------------------------------
    # Full-period comparison
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("FULL PERIOD COMPARISON")
    print("=" * 120)

    full = (
        results_df[
            results_df[
                "period"
            ]
            == "FULL_2020_2026"
        ]
        .copy()
    )

    print(
        full[
            [
                "mode",
                "candidate_signals",
                "trades",
                "win_rate_pct",
                "profit_factor",
                "return_pct",
                "max_drawdown_pct",
                "max_consecutive_losses",
                "fees_usd",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Year comparison
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("CALENDAR YEAR COMPARISON")
    print("=" * 130)

    years = (
        comparison_df[
            comparison_df[
                "period_type"
            ]
            == "year"
        ]
        .copy()
    )

    print(
        years[
            [
                "period",
                "pf_breakout",
                "return_breakout_pct",
                "pf_inverse",
                "return_inverse_pct",
                "winner_by_return",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print_window_diagnostic(
        comparison_df
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    results_file = os.path.join(
        REPORT_DIR,
        f"breakout_vs_inverse_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    comparison_file = os.path.join(
        REPORT_DIR,
        f"breakout_vs_inverse_comparison_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print()
    print("=" * 100)
    print("TEST FINISHED")
    print("=" * 100)
    print(f"Full results: {results_file}")
    print(f"Comparison:   {comparison_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()
