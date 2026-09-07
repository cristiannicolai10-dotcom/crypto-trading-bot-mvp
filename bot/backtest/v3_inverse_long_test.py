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
    sys.path.insert(0, CURRENT_DIR)

import backtest_inverse as bt


# ============================================================
# PATHS
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

RECENT_DAYS = 120


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "1h": [

        {
            "name": "baseline",
            "sl": 1.50,
            "tp": 3.00
        },

        {
            "name": "candidate_A",
            "sl": 1.25,
            "tp": 3.00
        },

        {
            "name": "candidate_B",
            "sl": 1.25,
            "tp": 3.50
        }
    ],


    "4h": [

        {
            "name": "baseline",
            "sl": 1.50,
            "tp": 3.00
        },

        {
            "name": "candidate_A",
            "sl": 1.00,
            "tp": 2.50
        },

        {
            "name": "candidate_B",
            "sl": 1.25,
            "tp": 3.00
        }
    ]
}


# ============================================================
# LOAD LOCAL PARQUET
# ============================================================

def load_dataset(
    symbol,
    timeframe
):

    filepath = os.path.join(
        DATA_DIR,
        f"{symbol}_{timeframe}_3y.parquet"
    )


    if not os.path.exists(
        filepath
    ):

        raise RuntimeError(
            f"File not found: {filepath}"
        )


    print()
    print("=" * 75)

    print(
        f"LOADING LOCAL DATA | "
        f"{symbol} {timeframe}"
    )

    print(
        f"File: {filepath}"
    )

    print("=" * 75)


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


    # Remove current candle if it is not closed.
    df = bt.remove_open_candle(
        df,
        timeframe
    )


    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"From: "
        f"{df.iloc[0]['timestamp']}"
    )

    print(
        f"To:   "
        f"{df.iloc[-1]['timestamp']}"
    )


    if (
        len(df)
        < bt.WARMUP_BARS + 10
    ):

        raise RuntimeError(
            f"Insufficient data: "
            f"{symbol} {timeframe}"
        )


    print(
        "Calculating indicators..."
    )


    df = bt.calculate_indicators(
        df
    )


    print(
        "Calculating V3 signals..."
    )


    # This is the original V3 signal logic.
    # Direction inversion happens later
    # inside backtest_inverse.run_backtest().
    df = bt.calculate_scores(
        df
    )


    return df


# ============================================================
# RUN ONE PERIOD
# ============================================================

def run_period(
    df,
    symbol,
    timeframe,
    config_name,
    sl_atr,
    tp_atr,
    period_name,
    warmup_bars
):

    if df.empty:

        return None


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

        bt.ATR_STOP_MULTIPLIER = float(
            sl_atr
        )

        bt.ATR_TARGET_MULTIPLIER = float(
            tp_atr
        )

        bt.WARMUP_BARS = int(
            warmup_bars
        )


        (
            trades,
            final_balance,
            max_drawdown
        ) = bt.run_backtest(
            df.copy()
        )


        summary = bt.build_summary(
            trades=trades,
            symbol=symbol,
            timeframe=timeframe,
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


    result = {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "configuration":
            config_name,

        "period":
            period_name,

        "sl_atr":
            sl_atr,

        "tp_atr":
            tp_atr,

        "risk_reward":
            tp_atr / sl_atr,

        "data_start":
            df.iloc[0][
                "timestamp"
            ].isoformat(),

        "data_end":
            df.iloc[-1][
                "timestamp"
            ].isoformat(),

        "candles":
            int(
                len(df)
            ),

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
            )
    }


    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    if result is None:
        return


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
        f"{result['timeframe']:<2} | "
        f"{result['configuration']:<12} | "
        f"{result['period']:<15} | "
        f"SL={result['sl_atr']:<4} | "
        f"TP={result['tp_atr']:<4} | "
        f"Trades={result['trades']:<4} | "
        f"Win={win_text:<8} | "
        f"PF={pf_text:<6} | "
        f"Return="
        f"{result['return_pct']:+.2f}% | "
        f"DD={dd_text}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []


    print()
    print("=" * 75)

    print(
        "V3 INVERSE - LONG HISTORICAL VALIDATION"
    )

    print(
        "Data source: LOCAL PARQUET"
    )

    print(
        "Supabase: NOT USED"
    )

    print(
        f"Initial balance: "
        f"${bt.INITIAL_BALANCE:.2f}"
    )

    print(
        f"Position fraction: "
        f"{bt.POSITION_FRACTION * 100:.0f}%"
    )

    print(
        f"Leverage: "
        f"{bt.LEVERAGE}x"
    )

    print("=" * 75)


    for (
        timeframe,
        configurations
    ) in CONFIGURATIONS.items():


        df = load_dataset(
            SYMBOL,
            timeframe
        )


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


        unseen_history = (
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


        recent_history = (
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


        print()
        print("=" * 75)

        print(
            f"{timeframe} PERIOD SPLIT"
        )

        print(
            f"UNSEEN HISTORY: "
            f"{unseen_history.iloc[0]['timestamp']} "
            f"→ "
            f"{unseen_history.iloc[-1]['timestamp']}"
        )

        print(
            f"RECENT 120D:    "
            f"{recent_history.iloc[0]['timestamp']} "
            f"→ "
            f"{recent_history.iloc[-1]['timestamp']}"
        )

        print("=" * 75)


        for config in configurations:

            config_name = (
                config["name"]
            )

            sl_atr = (
                config["sl"]
            )

            tp_atr = (
                config["tp"]
            )


            # =================================================
            # FULL 3 YEARS
            # =================================================

            full_result = run_period(
                df=df,
                symbol=SYMBOL,
                timeframe=timeframe,
                config_name=config_name,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                period_name="FULL_3Y",
                warmup_bars=250
            )


            all_results.append(
                full_result
            )

            print_result(
                full_result
            )


            # =================================================
            # UNSEEN OLD HISTORY
            # =================================================

            unseen_result = run_period(
                df=unseen_history,
                symbol=SYMBOL,
                timeframe=timeframe,
                config_name=config_name,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                period_name="UNSEEN_HISTORY",
                warmup_bars=250
            )


            all_results.append(
                unseen_result
            )

            print_result(
                unseen_result
            )


            # =================================================
            # RECENT PERIOD
            #
            # Indicators were calculated using full history
            # before slicing, therefore no extra warmup is
            # necessary here.
            # =================================================

            recent_result = run_period(
                df=recent_history,
                symbol=SYMBOL,
                timeframe=timeframe,
                config_name=config_name,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                period_name="RECENT_120D",
                warmup_bars=0
            )


            all_results.append(
                recent_result
            )

            print_result(
                recent_result
            )


            print("-" * 75)


    # ========================================================
    # SAVE REPORT
    # ========================================================

    results_df = pd.DataFrame(
        [
            row
            for row in all_results
            if row is not None
        ]
    )


    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    report_file = os.path.join(
        REPORT_DIR,
        f"v3_inverse_long_validation_"
        f"{SYMBOL}_"
        f"{stamp}.csv"
    )


    results_df.to_csv(
        report_file,
        index=False
    )


    # ========================================================
    # SHOW UNSEEN RESULTS
    # ========================================================

    unseen_results = (
        results_df[
            results_df[
                "period"
            ]
            == "UNSEEN_HISTORY"
        ]
        .copy()
    )


    print()
    print("=" * 75)

    print(
        "UNSEEN HISTORY RESULTS"
    )

    print("=" * 75)


    if not unseen_results.empty:

        display_columns = [
            "timeframe",
            "configuration",
            "sl_atr",
            "tp_atr",
            "trades",
            "win_rate_pct",
            "profit_factor",
            "return_pct",
            "max_drawdown_pct",
            "fees_usd"
        ]


        print(
            unseen_results[
                display_columns
            ]
            .sort_values(
                [
                    "timeframe",
                    "profit_factor"
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .to_string(
                index=False
            )
        )


    print()
    print(
        f"Full report: "
        f"{report_file}"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()
