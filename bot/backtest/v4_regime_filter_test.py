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

RECENT_DAYS = 120


CONFIGURATIONS = [

    {
        "name": "V4_A",
        "sl": 1.50,
        "tp": 3.00
    },

    {
        "name": "V4_B",
        "sl": 1.25,
        "tp": 3.00
    },

    {
        "name": "V4_C",
        "sl": 1.00,
        "tp": 2.50
    }
]


# ============================================================
# LOAD LOCAL DATA
# ============================================================

def load_dataset():

    filepath = os.path.join(
        DATA_DIR,
        f"{SYMBOL}_{TIMEFRAME}_3y.parquet"
    )


    if not os.path.exists(filepath):

        raise RuntimeError(
            f"File not found: {filepath}"
        )


    print()
    print("=" * 80)

    print(
        "V4 REGIME FILTER TEST"
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Timeframe: {TIMEFRAME}"
    )

    print(
        f"File: {filepath}"
    )

    print(
        "Supabase: NOT USED"
    )

    print("=" * 80)


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


    if len(df) < bt.WARMUP_BARS + 10:

        raise RuntimeError(
            "Insufficient historical data"
        )


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
# APPLY V4 REGIME FILTER
# ============================================================

def apply_v4_filter(
    df
):

    filtered = df.copy()


    # --------------------------------------------------------
    # Original V3 SHORT means:
    #
    # bearish fractal breakdown
    #
    # backtest_inverse will later convert:
    #
    # SHORT -> LONG
    # --------------------------------------------------------

    valid_v4_signal = (

        (
            filtered[
                "direction"
            ]
            == "SHORT"
        )

        &

        (
            filtered[
                "market_regime"
            ]
            == "bearish"
        )

        &

        (
            filtered[
                "volatility_regime"
            ]
            == "normal"
        )
    )


    # Everything outside V4 conditions
    # becomes HOLD.

    filtered.loc[
        ~valid_v4_signal,
        "direction"
    ] = "HOLD"


    filtered[
        "v4_valid_signal"
    ] = valid_v4_signal


    return filtered


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    df,
    configuration,
    period_name,
    warmup_bars
):

    old_sl = bt.ATR_STOP_MULTIPLIER
    old_tp = bt.ATR_TARGET_MULTIPLIER
    old_warmup = bt.WARMUP_BARS


    try:

        bt.ATR_STOP_MULTIPLIER = float(
            configuration["sl"]
        )

        bt.ATR_TARGET_MULTIPLIER = float(
            configuration["tp"]
        )

        bt.WARMUP_BARS = int(
            warmup_bars
        )


        filtered_df = apply_v4_filter(
            df
        )


        candidate_signals = int(
            filtered_df[
                "v4_valid_signal"
            ].sum()
        )


        (
            trades,
            final_balance,
            max_drawdown
        ) = bt.run_backtest(
            filtered_df
        )


        summary = bt.build_summary(
            trades=trades,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            df=filtered_df,
            final_balance=final_balance,
            max_drawdown=max_drawdown
        )


    finally:

        bt.ATR_STOP_MULTIPLIER = old_sl
        bt.ATR_TARGET_MULTIPLIER = old_tp
        bt.WARMUP_BARS = old_warmup


    return {

        "configuration":
            configuration["name"],

        "period":
            period_name,

        "sl_atr":
            configuration["sl"],

        "tp_atr":
            configuration["tp"],

        "risk_reward":
            (
                configuration["tp"]
                / configuration["sl"]
            ),

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
                "max_drawdown_pct"
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

        "data_start":
            df.iloc[0][
                "timestamp"
            ].isoformat(),

        "data_end":
            df.iloc[-1][
                "timestamp"
            ].isoformat()
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    pf = result[
        "profit_factor"
    ]

    win = result[
        "win_rate_pct"
    ]

    dd = result[
        "max_drawdown_pct"
    ]


    pf_text = (
        f"{pf:.3f}"
        if pf is not None
        else "N/A"
    )


    win_text = (
        f"{win:.2f}%"
        if win is not None
        else "N/A"
    )


    dd_text = (
        f"{dd:.2f}%"
        if dd is not None
        else "N/A"
    )


    print(
        f"{result['configuration']:<5} | "
        f"{result['period']:<15} | "
        f"SL={result['sl_atr']:<4} | "
        f"TP={result['tp_atr']:<4} | "
        f"Signals={result['candidate_signals']:<4} | "
        f"Trades={result['trades']:<4} | "
        f"Win={win_text:<8} | "
        f"PF={pf_text:<6} | "
        f"Return={result['return_pct']:+.2f}% | "
        f"DD={dd_text}"
    )


# ============================================================
# YEAR-BY-YEAR TEST
# ============================================================

def build_year_periods(
    df
):

    periods = []


    years = sorted(
        df[
            "timestamp"
        ]
        .dt
        .year
        .unique()
        .tolist()
    )


    for year in years:

        year_df = (
            df[
                df[
                    "timestamp"
                ]
                .dt
                .year
                == year
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        if len(year_df) < 50:

            continue


        periods.append(
            (
                f"YEAR_{year}",
                year_df
            )
        )


    return periods


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()


    latest_timestamp = (
        df[
            "timestamp"
        ]
        .max()
    )


    recent_cutoff = (
        latest_timestamp
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )


    unseen_df = (
        df[
            df[
                "timestamp"
            ]
            < recent_cutoff
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    recent_df = (
        df[
            df[
                "timestamp"
            ]
            >= recent_cutoff
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    periods = [

        (
            "FULL_3Y",
            df,
            250
        ),

        (
            "UNSEEN_HISTORY",
            unseen_df,
            250
        ),

        (
            "RECENT_120D",
            recent_df,
            0
        )
    ]


    # Add calendar-year diagnostics.

    for (
        period_name,
        period_df
    ) in build_year_periods(df):

        periods.append(
            (
                period_name,
                period_df,
                0
            )
        )


    results = []


    print()
    print("=" * 100)

    print(
        "V4 FILTER:"
    )

    print(
        "Original V3 SHORT"
    )

    print(
        "+ market_regime = bearish"
    )

    print(
        "+ volatility_regime = normal"
    )

    print(
        "= inverse execution LONG"
    )

    print("=" * 100)


    for configuration in CONFIGURATIONS:

        print()
        print(
            "-" * 100
        )

        print(
            f"{configuration['name']} | "
            f"SL={configuration['sl']} ATR | "
            f"TP={configuration['tp']} ATR"
        )

        print(
            "-" * 100
        )


        for (
            period_name,
            period_df,
            warmup_bars
        ) in periods:


            result = run_test(
                df=period_df,
                configuration=configuration,
                period_name=period_name,
                warmup_bars=warmup_bars
            )


            results.append(
                result
            )


            print_result(
                result
            )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    report_file = os.path.join(
        REPORT_DIR,
        f"v4_regime_filter_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )


    results_df.to_csv(
        report_file,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 100)

    print(
        "MAIN V4 RESULTS"
    )

    print("=" * 100)


    main_periods = (
        results_df[
            results_df[
                "period"
            ]
            .isin(
                [
                    "FULL_3Y",
                    "UNSEEN_HISTORY",
                    "RECENT_120D"
                ]
            )
        ]
    )


    print(
        main_periods[
            [
                "configuration",
                "period",
                "sl_atr",
                "tp_atr",
                "candidate_signals",
                "trades",
                "win_rate_pct",
                "profit_factor",
                "return_pct",
                "max_drawdown_pct",
                "fees_usd"
            ]
        ]
        .to_string(
            index=False
        )
    )


    print()
    print("=" * 100)

    print(
        "YEAR-BY-YEAR RESULTS"
    )

    print("=" * 100)


    yearly = (
        results_df[
            results_df[
                "period"
            ]
            .str
            .startswith(
                "YEAR_"
            )
        ]
    )


    if not yearly.empty:

        print(
            yearly[
                [
                    "configuration",
                    "period",
                    "trades",
                    "win_rate_pct",
                    "profit_factor",
                    "return_pct",
                    "max_drawdown_pct"
                ]
            ]
            .to_string(
                index=False
            )
        )


    print()
    print(
        f"Full report: "
        f"{report_file}"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
