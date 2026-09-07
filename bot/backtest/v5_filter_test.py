import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# IMPORT EXISTING V3 INVERSE ENGINE
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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

RECENT_DAYS = 120
WINDOW_MONTHS = 6


# ============================================================
# FILTER VARIANTS
# ============================================================

FILTERS = [
    {
        "name": "V4_BASE",
        "description": "V4 only - no new filter",
        "atr_max": None,
        "price_vs_ema200_min": None,
        "ema200_slope_24h_min": None,
    },
    {
        "name": "V5_A",
        "description": "ATR <= 1.65%",
        "atr_max": 1.65,
        "price_vs_ema200_min": None,
        "ema200_slope_24h_min": None,
    },
    {
        "name": "V5_B",
        "description": "ATR <= 1.65% and price >= -8% vs EMA200",
        "atr_max": 1.65,
        "price_vs_ema200_min": -8.0,
        "ema200_slope_24h_min": None,
    },
    {
        "name": "V5_C",
        "description": "ATR <= 1.65% and EMA200 24h slope >= -0.40%",
        "atr_max": 1.65,
        "price_vs_ema200_min": None,
        "ema200_slope_24h_min": -0.40,
    },
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    filepath = os.path.join(
        DATA_DIR,
        f"{SYMBOL}_{TIMEFRAME}_3y.parquet"
    )

    if not os.path.exists(filepath):
        raise RuntimeError(
            f"Historical file not found: {filepath}"
        )

    print()
    print("=" * 100)
    print("V5 FILTER TEST")
    print("=" * 100)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"SL: {SL_ATR} ATR")
    print(f"TP: {TP_ATR} ATR")
    print(f"Source: {filepath}")
    print("Supabase: NOT USED")
    print("=" * 100)

    df = pd.read_parquet(filepath)

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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = bt.remove_open_candle(
        df,
        TIMEFRAME
    )

    if len(df) < bt.WARMUP_BARS + 10:
        raise RuntimeError(
            "Insufficient historical data"
        )

    print(f"Candles: {len(df):,}")
    print(f"From: {df.iloc[0]['timestamp']}")
    print(f"To:   {df.iloc[-1]['timestamp']}")

    print()
    print("Calculating indicators...")

    df = bt.calculate_indicators(df)

    print("Calculating original V3 signals...")

    df = bt.calculate_scores(df)

    return add_v5_features(df)


# ============================================================
# ADD FEATURES
# ============================================================

def add_v5_features(df):

    df = df.copy()

    df["atr_pct"] = (
        df["atr"]
        / df["close"]
        * 100
    )

    df["price_vs_ema200_pct"] = (
        (
            df["close"]
            - df["ema_200"]
        )
        / df["ema_200"]
        * 100
    )

    # 4h timeframe:
    # 6 candles = 24 hours
    df["ema200_slope_24h_pct"] = (
        df["ema_200"]
        .pct_change(6)
        * 100
    )

    # V4_B base logic:
    # Original V3 SHORT = bearish fractal breakdown.
    # backtest_inverse will execute the opposite side -> LONG.
    df["v4_signal"] = (
        (df["direction"] == "SHORT")
        &
        (df["market_regime"] == "bearish")
        &
        (df["volatility_regime"] == "normal")
    )

    return df


# ============================================================
# BUILD TIME PERIODS
# ============================================================

def build_periods(df):

    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()

    recent_cutoff = (
        dataset_end
        - pd.Timedelta(days=RECENT_DAYS)
    )

    periods = [
        {
            "period": "FULL_3Y",
            "start": dataset_start,
            "end": (
                dataset_end
                + pd.Timedelta(microseconds=1)
            ),
        },
        {
            "period": "UNSEEN_HISTORY",
            "start": dataset_start,
            "end": recent_cutoff,
        },
        {
            "period": "RECENT_120D",
            "start": recent_cutoff,
            "end": (
                dataset_end
                + pd.Timedelta(microseconds=1)
            ),
        },
    ]

    # Add consecutive 6-month windows.
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
# FILTER MASK
# ============================================================

def build_filter_mask(
    df,
    config
):

    mask = df["v4_signal"].copy()

    atr_max = config.get(
        "atr_max"
    )

    price_min = config.get(
        "price_vs_ema200_min"
    )

    slope_min = config.get(
        "ema200_slope_24h_min"
    )

    if atr_max is not None:
        mask = (
            mask
            &
            (df["atr_pct"] <= atr_max)
        )

    if price_min is not None:
        mask = (
            mask
            &
            (
                df["price_vs_ema200_pct"]
                >= price_min
            )
        )

    if slope_min is not None:
        mask = (
            mask
            &
            (
                df["ema200_slope_24h_pct"]
                >= slope_min
            )
        )

    return mask.fillna(False)


# ============================================================
# RUN ONE FILTER / PERIOD
# ============================================================

def run_one(
    df,
    config,
    period
):

    test_df = df.copy()

    filter_mask = build_filter_mask(
        test_df,
        config
    )

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
        filter_mask
        &
        time_mask
    )

    candidate_signals = int(
        allowed.sum()
    )

    # Disable all signals that are not permitted by
    # the selected V5 filter and time period.
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
        "configuration":
            config["name"],

        "description":
            config["description"],

        "period":
            period["period"],

        "period_start":
            period["start"].isoformat(),

        "period_end":
            (
                period["end"]
                - pd.Timedelta(
                    microseconds=1
                )
            ).isoformat(),

        "atr_max":
            config.get(
                "atr_max"
            ),

        "price_vs_ema200_min":
            config.get(
                "price_vs_ema200_min"
            ),

        "ema200_slope_24h_min":
            config.get(
                "ema200_slope_24h_min"
            ),

        "sl_atr":
            SL_ATR,

        "tp_atr":
            TP_ATR,

        "risk_reward":
            TP_ATR / SL_ATR,

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
# PRINT
# ============================================================

def format_number(
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


def print_result(result):

    print(
        f"{result['configuration']:<7} | "
        f"{result['period']:<15} | "
        f"Signals={result['candidate_signals']:<3} | "
        f"Trades={result['trades']:<3} | "
        f"Win={format_number(result['win_rate_pct'], 2):>6} | "
        f"PF={format_number(result['profit_factor'], 3):>6} | "
        f"Return={result['return_pct']:+6.2f}% | "
        f"DD={result['max_drawdown_pct']:5.2f}%"
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def build_stability_summary(
    results_df
):

    rows = []

    window_df = (
        results_df[
            results_df[
                "period"
            ]
            .str
            .match(r"^W\d+$")
        ]
        .copy()
    )

    for config_name in [
        config["name"]
        for config in FILTERS
    ]:

        subset = (
            window_df[
                window_df[
                    "configuration"
                ]
                == config_name
            ]
            .copy()
        )

        valid_pf = (
            subset[
                "profit_factor"
            ]
            .dropna()
        )

        rows.append(
            {
                "configuration":
                    config_name,

                "windows":
                    int(
                        len(subset)
                    ),

                "positive_windows":
                    int(
                        (
                            subset[
                                "return_pct"
                            ]
                            > 0
                        )
                        .sum()
                    ),

                "negative_windows":
                    int(
                        (
                            subset[
                                "return_pct"
                            ]
                            < 0
                        )
                        .sum()
                    ),

                "pf_above_1_windows":
                    int(
                        (
                            valid_pf
                            > 1
                        )
                        .sum()
                    ),

                "pf_valid_windows":
                    int(
                        len(valid_pf)
                    ),

                "median_pf":
                    (
                        valid_pf.median()
                        if not valid_pf.empty
                        else None
                    ),

                "mean_pf":
                    (
                        valid_pf.mean()
                        if not valid_pf.empty
                        else None
                    ),

                "best_window_return_pct":
                    subset[
                        "return_pct"
                    ].max(),

                "worst_window_return_pct":
                    subset[
                        "return_pct"
                    ].min(),

                "worst_window_dd_pct":
                    subset[
                        "max_drawdown_pct"
                    ].max(),

                "total_window_trades":
                    int(
                        subset[
                            "trades"
                        ]
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    periods = build_periods(df)

    results = []

    print()
    print("=" * 110)
    print("RUNNING V4 BASE + V5 FILTER VARIANTS")
    print("=" * 110)

    for config in FILTERS:

        print()
        print("-" * 110)
        print(
            f"{config['name']} | "
            f"{config['description']}"
        )
        print("-" * 110)

        for period in periods:

            result = run_one(
                df=df,
                config=config,
                period=period
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

    stability_df = (
        build_stability_summary(
            results_df
        )
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
        f"v5_filter_test_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    stability_file = os.path.join(
        REPORT_DIR,
        f"v5_filter_stability_"
        f"{SYMBOL}_"
        f"{TIMEFRAME}_"
        f"{stamp}.csv"
    )

    results_df.to_csv(
        report_file,
        index=False
    )

    stability_df.to_csv(
        stability_file,
        index=False
    )

    # ========================================================
    # MAIN PERIOD COMPARISON
    # ========================================================

    print()
    print("=" * 120)
    print("MAIN PERIOD COMPARISON")
    print("=" * 120)

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
        .copy()
    )

    print(
        main_periods[
            [
                "configuration",
                "period",
                "candidate_signals",
                "trades",
                "win_rate_pct",
                "profit_factor",
                "return_pct",
                "max_drawdown_pct",
                "max_consecutive_losses"
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # W5 COMPARISON
    # ========================================================

    print()
    print("=" * 120)
    print("W5 FAILURE PERIOD COMPARISON")
    print("=" * 120)

    w5 = (
        results_df[
            results_df[
                "period"
            ]
            == "W5"
        ]
        .copy()
    )

    print(
        w5[
            [
                "configuration",
                "candidate_signals",
                "trades",
                "win_rate_pct",
                "profit_factor",
                "return_pct",
                "max_drawdown_pct",
                "max_consecutive_losses"
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # STABILITY
    # ========================================================

    print()
    print("=" * 120)
    print("6-MONTH STABILITY SUMMARY")
    print("=" * 120)

    print(
        stability_df
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("V5 FILTER TEST FINISHED")
    print("=" * 100)
    print(f"Full results: {report_file}")
    print(f"Stability:    {stability_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()
