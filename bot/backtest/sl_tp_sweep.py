import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# IMPORT EXISTING V3 INVERSE BACKTEST
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

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS TO TEST
# ============================================================

# Experiment A:
# Keep TP fixed at 3 ATR
# and change only SL.

SL_VALUES = [
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50
]

FIXED_TP = 3.00


# Experiment B:
# Keep SL fixed at 1.5 ATR
# and change only TP.

TP_VALUES = [
    1.50,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
    5.00
]

FIXED_SL = 1.50


TIMEFRAMES = [
    "1h",
    "4h"
]

SYMBOL = "BTCUSDT"


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(
    symbol,
    timeframe
):

    print()
    print(
        "=" * 70
    )

    print(
        f"LOADING DATA | "
        f"{symbol} "
        f"{timeframe}"
    )

    print(
        "=" * 70
    )


    df = bt.get_all_candles(
        symbol,
        timeframe
    )


    if df.empty:

        raise RuntimeError(
            f"No candles found for "
            f"{symbol} {timeframe}"
        )


    df = bt.remove_open_candle(
        df,
        timeframe
    )


    print(
        f"Dataset: "
        f"{len(df):,} candles"
    )


    if (
        len(df)
        < bt.WARMUP_BARS + 10
    ):

        raise RuntimeError(
            f"Insufficient data for "
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

    df = bt.calculate_scores(
        df
    )


    return df


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    df,
    symbol,
    timeframe,
    test_type,
    sl_atr,
    tp_atr
):

    # Change only the global parameters
    # used by existing inverse backtester.

    bt.ATR_STOP_MULTIPLIER = (
        float(sl_atr)
    )

    bt.ATR_TARGET_MULTIPLIER = (
        float(tp_atr)
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


    total_trades = int(
        summary.get(
            "total_trades",
            0
        )
    )


    if total_trades > 0:

        win_rate = summary.get(
            "win_rate_pct"
        )

        profit_factor = summary.get(
            "profit_factor"
        )

        max_dd = summary.get(
            "max_drawdown_pct"
        )

        fees = summary.get(
            "total_fees_usd"
        )

        return_pct = summary.get(
            "total_return_pct"
        )

        final_balance = summary.get(
            "final_balance"
        )

        average_trade = summary.get(
            "average_trade_pct"
        )

        max_consecutive_losses = (
            summary.get(
                "max_consecutive_losses"
            )
        )

    else:

        win_rate = None
        profit_factor = None
        max_dd = None
        fees = 0
        return_pct = 0
        final_balance = (
            bt.INITIAL_BALANCE
        )
        average_trade = None
        max_consecutive_losses = None


    result = {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "test_type":
            test_type,

        "sl_atr":
            sl_atr,

        "tp_atr":
            tp_atr,

        "risk_reward":
            (
                tp_atr / sl_atr
                if sl_atr > 0
                else None
            ),

        "trades":
            total_trades,

        "win_rate_pct":
            win_rate,

        "profit_factor":
            profit_factor,

        "return_pct":
            return_pct,

        "max_drawdown_pct":
            max_dd,

        "fees_usd":
            fees,

        "final_balance":
            final_balance,

        "average_trade_pct":
            average_trade,

        "max_consecutive_losses":
            max_consecutive_losses
    }


    pf_text = (
        f"{profit_factor:.4f}"
        if profit_factor is not None
        else "N/A"
    )


    print(
        f"{test_type:<10} | "
        f"{timeframe:<2} | "
        f"SL={sl_atr:<4} | "
        f"TP={tp_atr:<4} | "
        f"Trades={total_trades:<4} | "
        f"Win={win_rate if win_rate is not None else 0:.2f}% | "
        f"PF={pf_text} | "
        f"Return={return_pct:.2f}% | "
        f"DD={max_dd if max_dd is not None else 0:.2f}%"
    )


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []


    for timeframe in TIMEFRAMES:

        df = prepare_dataset(
            SYMBOL,
            timeframe
        )


        # ====================================================
        # EXPERIMENT A
        # CHANGE ONLY SL
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            f"SL SWEEP | "
            f"{SYMBOL} "
            f"{timeframe} | "
            f"TP FIXED = "
            f"{FIXED_TP} ATR"
        )

        print(
            "=" * 70
        )


        for sl_value in SL_VALUES:

            result = run_test(
                df=df,
                symbol=SYMBOL,
                timeframe=timeframe,
                test_type="SL_SWEEP",
                sl_atr=sl_value,
                tp_atr=FIXED_TP
            )

            all_results.append(
                result
            )


        # ====================================================
        # EXPERIMENT B
        # CHANGE ONLY TP
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            f"TP SWEEP | "
            f"{SYMBOL} "
            f"{timeframe} | "
            f"SL FIXED = "
            f"{FIXED_SL} ATR"
        )

        print(
            "=" * 70
        )


        for tp_value in TP_VALUES:

            result = run_test(
                df=df,
                symbol=SYMBOL,
                timeframe=timeframe,
                test_type="TP_SWEEP",
                sl_atr=FIXED_SL,
                tp_atr=tp_value
            )

            all_results.append(
                result
            )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )


    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    report_file = os.path.join(
        REPORT_DIR,
        f"sl_tp_sweep_"
        f"{SYMBOL}_"
        f"{stamp}.csv"
    )


    results_df.to_csv(
        report_file,
        index=False
    )


    print()
    print(
        "=" * 70
    )

    print(
        "SL / TP SWEEP FINISHED"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # SHOW BEST RESULTS
    # ========================================================

    valid_results = (
        results_df
        .dropna(
            subset=[
                "profit_factor"
            ]
        )
        .copy()
    )


    if not valid_results.empty:

        print()
        print(
            "BEST RESULTS BY PROFIT FACTOR"
        )

        print()


        best = (
            valid_results
            .sort_values(
                [
                    "profit_factor",
                    "return_pct"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .head(10)
        )


        print(
            best[
                [
                    "timeframe",
                    "test_type",
                    "sl_atr",
                    "tp_atr",
                    "risk_reward",
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


if __name__ == "__main__":

    main()
