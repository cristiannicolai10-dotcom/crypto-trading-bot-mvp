import os
import sys
from datetime import datetime, timezone

import pandas as pd


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import backtest_inverse as bt


BASE_DIR = "/home/botadmin/crypto-bot"

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


SYMBOL = "BTCUSDT"

TIMEFRAMES = [
    "1h",
    "4h"
]


SL_VALUES = [
    0.75,
    1.00,
    1.25,
    1.50
]


TP_VALUES = [
    2.00,
    2.50,
    3.00,
    3.50,
    4.00
]


def prepare_dataset(
    symbol,
    timeframe
):

    print()
    print("=" * 70)

    print(
        f"LOADING | "
        f"{symbol} "
        f"{timeframe}"
    )

    print("=" * 70)


    df = bt.get_all_candles(
        symbol,
        timeframe
    )


    if df.empty:

        raise RuntimeError(
            f"No data for "
            f"{symbol} "
            f"{timeframe}"
        )


    df = bt.remove_open_candle(
        df,
        timeframe
    )


    if len(df) < bt.WARMUP_BARS + 10:

        raise RuntimeError(
            f"Insufficient data for "
            f"{symbol} "
            f"{timeframe}"
        )


    print(
        f"Dataset: "
        f"{len(df):,} candles"
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


def run_one_test(
    df,
    symbol,
    timeframe,
    sl_atr,
    tp_atr
):

    bt.ATR_STOP_MULTIPLIER = float(
        sl_atr
    )

    bt.ATR_TARGET_MULTIPLIER = float(
        tp_atr
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


    result = {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "sl_atr":
            sl_atr,

        "tp_atr":
            tp_atr,

        "risk_reward":
            tp_atr / sl_atr,

        "trades":
            total_trades,

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

        "fees_usd":
            summary.get(
                "total_fees_usd"
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
                "final_balance"
            )
    }


    pf = result[
        "profit_factor"
    ]

    ret = result[
        "return_pct"
    ]

    dd = result[
        "max_drawdown_pct"
    ]

    win = result[
        "win_rate_pct"
    ]


    print(
        f"{timeframe:<2} | "
        f"SL={sl_atr:<4} | "
        f"TP={tp_atr:<4} | "
        f"RR={tp_atr/sl_atr:<5.2f} | "
        f"Trades={total_trades:<3} | "
        f"Win={win if win is not None else 0:>6.2f}% | "
        f"PF={pf if pf is not None else 0:>6.3f} | "
        f"Return={ret if ret is not None else 0:>6.2f}% | "
        f"DD={dd if dd is not None else 0:>5.2f}%"
    )


    return result


def main():

    results = []


    for timeframe in TIMEFRAMES:

        df = prepare_dataset(
            SYMBOL,
            timeframe
        )


        print()
        print("=" * 70)

        print(
            f"SL x TP GRID | "
            f"{SYMBOL} "
            f"{timeframe}"
        )

        print("=" * 70)


        for sl_atr in SL_VALUES:

            for tp_atr in TP_VALUES:

                result = run_one_test(
                    df=df,
                    symbol=SYMBOL,
                    timeframe=timeframe,
                    sl_atr=sl_atr,
                    tp_atr=tp_atr
                )

                results.append(
                    result
                )


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
        f"sl_tp_grid_"
        f"{SYMBOL}_"
        f"{stamp}.csv"
    )


    results_df.to_csv(
        report_file,
        index=False
    )


    valid = (
        results_df
        .dropna(
            subset=[
                "profit_factor"
            ]
        )
        .copy()
    )


    print()
    print("=" * 70)

    print(
        "TOP RESULTS"
    )

    print("=" * 70)


    if not valid.empty:

        best = (
            valid
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
            .head(15)
        )


        print(
            best[
                [
                    "timeframe",
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
