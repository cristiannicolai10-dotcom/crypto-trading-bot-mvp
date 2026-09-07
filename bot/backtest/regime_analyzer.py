import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# IMPORT V3 INVERSE BACKTEST
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

TIMEFRAMES = [
    "1h",
    "4h"
]


# Keep strategy parameters fixed.
# We are analyzing regimes, NOT optimizing SL/TP.

SL_ATR = 1.50
TP_ATR = 3.00


RECENT_DAYS = 120

MIN_TRADES_FOR_RANKING = 20


# ============================================================
# LOAD DATA
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
            f"Historical file not found: "
            f"{filepath}"
        )


    print()
    print("=" * 80)

    print(
        f"LOADING | "
        f"{symbol} "
        f"{timeframe}"
    )

    print(
        f"File: {filepath}"
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
        timeframe
    )


    print(
        f"Candles: "
        f"{len(df):,}"
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
            f"Insufficient data for "
            f"{timeframe}"
        )


    # ========================================================
    # EXISTING INDICATORS
    # ========================================================

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


    # ========================================================
    # ADD REGIME FEATURES
    # ========================================================

    df = add_regime_features(
        df
    )


    return df


# ============================================================
# EXTRA REGIME FEATURES
# ============================================================

def add_regime_features(
    df
):

    df = df.copy()


    # --------------------------------------------------------
    # ATR %
    # --------------------------------------------------------

    df["atr_pct"] = (
        df["atr"]
        / df["close"]
        * 100
    )


    # --------------------------------------------------------
    # EMA SPREAD
    # --------------------------------------------------------

    df["ema_spread_pct"] = (
        (
            df["ema_50"]
            - df["ema_200"]
        )
        .abs()
        / df["close"]
        * 100
    )


    # --------------------------------------------------------
    # TREND STRENGTH
    # --------------------------------------------------------

    conditions = [

        df[
            "ema_spread_pct"
        ] < 0.50,

        (
            df[
                "ema_spread_pct"
            ] >= 0.50
        )
        & (
            df[
                "ema_spread_pct"
            ] < 1.50
        ),

        df[
            "ema_spread_pct"
        ] >= 1.50
    ]


    choices = [
        "compression",
        "moderate",
        "strong"
    ]


    df[
        "trend_strength_regime"
    ] = np.select(
        conditions,
        choices,
        default="unknown"
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
        & (
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
        & (
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
    # VOLUME REGIME
    # --------------------------------------------------------

    df[
        "volume_regime"
    ] = "normal"


    df.loc[
        df[
            "volume_ratio"
        ] < 0.80,
        "volume_regime"
    ] = "low"


    df.loc[
        df[
            "volume_ratio"
        ] >= 1.20,
        "volume_regime"
    ] = "high"


    # --------------------------------------------------------
    # ORIGINAL V3 SIGNAL SIDE
    #
    # Remember:
    #
    # Original LONG  -> inverse executes SHORT
    # Original SHORT -> inverse executes LONG
    # --------------------------------------------------------

    df[
        "signal_side"
    ] = "none"


    df.loc[
        df[
            "direction"
        ] == "LONG",
        "signal_side"
    ] = "breakout_up"


    df.loc[
        df[
            "direction"
        ] == "SHORT",
        "signal_side"
    ] = "breakout_down"


    # --------------------------------------------------------
    # MARKET x VOLATILITY
    # --------------------------------------------------------

    df[
        "market_x_volatility"
    ] = (
        df[
            "market_regime"
        ]
        .astype(str)
        + "_"
        + df[
            "volatility_regime"
        ]
        .astype(str)
    )


    return df


# ============================================================
# RUN FILTERED BACKTEST
# ============================================================

def run_filtered_backtest(
    df,
    mask,
    timeframe,
    period_name,
    dimension,
    regime_value,
    warmup_bars
):

    filtered_df = (
        df.copy()
    )


    # Only allow signals that belong
    # to the regime currently being tested.

    filtered_df.loc[
        ~mask,
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

        bt.WARMUP_BARS = (
            warmup_bars
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
            timeframe=timeframe,
            df=filtered_df,
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


    trade_count = int(
        summary.get(
            "total_trades",
            0
        )
    )


    if trade_count == 0:

        return {

            "timeframe":
                timeframe,

            "period":
                period_name,

            "dimension":
                dimension,

            "regime":
                str(
                    regime_value
                ),

            "trades":
                0,

            "win_rate_pct":
                None,

            "profit_factor":
                None,

            "return_pct":
                0.0,

            "max_drawdown_pct":
                0.0,

            "fees_usd":
                0.0,

            "average_trade_pct":
                None,

            "max_consecutive_losses":
                None
        }


    return {

        "timeframe":
            timeframe,

        "period":
            period_name,

        "dimension":
            dimension,

        "regime":
            str(
                regime_value
            ),

        "trades":
            trade_count,

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
            )
    }


# ============================================================
# ANALYZE ONE DIMENSION
# ============================================================

def analyze_dimension(
    df,
    timeframe,
    period_name,
    dimension,
    warmup_bars
):

    results = []


    values = (
        df[
            dimension
        ]
        .dropna()
        .unique()
        .tolist()
    )


    for value in sorted(
        values
    ):

        mask = (
            df[
                dimension
            ]
            == value
        )


        result = (
            run_filtered_backtest(
                df=df,
                mask=mask,
                timeframe=timeframe,
                period_name=period_name,
                dimension=dimension,
                regime_value=value,
                warmup_bars=warmup_bars
            )
        )


        results.append(
            result
        )


    return results


# ============================================================
# ANALYZE PERIOD
# ============================================================

def analyze_period(
    df,
    timeframe,
    period_name,
    warmup_bars
):

    dimensions = [

        "market_regime",

        "volatility_regime",

        "trend_strength_regime",

        "alligator_state",

        "volume_regime",

        "signal_side",

        "market_x_volatility"
    ]


    results = []


    for dimension in dimensions:

        dimension_results = (
            analyze_dimension(
                df=df,
                timeframe=timeframe,
                period_name=period_name,
                dimension=dimension,
                warmup_bars=warmup_bars
            )
        )


        results.extend(
            dimension_results
        )


    return results


# ============================================================
# PRINT LEADERBOARD
# ============================================================

def print_leaderboard(
    results_df,
    period_name
):

    subset = (
        results_df[
            results_df[
                "period"
            ] == period_name
        ]
        .copy()
    )


    subset = subset[
        subset[
            "trades"
        ] >= MIN_TRADES_FOR_RANKING
    ]


    subset = subset.dropna(
        subset=[
            "profit_factor"
        ]
    )


    if subset.empty:

        print(
            f"No regimes with at least "
            f"{MIN_TRADES_FOR_RANKING} trades."
        )

        return


    # ========================================================
    # BEST
    # ========================================================

    print()
    print("=" * 100)

    print(
        f"BEST REGIMES | "
        f"{period_name}"
    )

    print("=" * 100)


    best = (
        subset
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
                "dimension",
                "regime",
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


    # ========================================================
    # WORST
    # ========================================================

    print()
    print("=" * 100)

    print(
        f"WORST REGIMES | "
        f"{period_name}"
    )

    print("=" * 100)


    worst = (
        subset
        .sort_values(
            [
                "profit_factor",
                "return_pct"
            ],
            ascending=[
                True,
                True
            ]
        )
        .head(15)
    )


    print(
        worst[
            [
                "timeframe",
                "dimension",
                "regime",
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


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []


    print()
    print("=" * 100)

    print(
        "V3 INVERSE REGIME ANALYZER"
    )

    print(
        f"SL = {SL_ATR} ATR"
    )

    print(
        f"TP = {TP_ATR} ATR"
    )

    print(
        "Historical source: LOCAL PARQUET"
    )

    print(
        "Supabase: NOT USED"
    )

    print("=" * 100)


    for timeframe in TIMEFRAMES:

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


        # ====================================================
        # FULL HISTORY
        # ====================================================

        full_df = (
            df.copy()
        )


        # ====================================================
        # UNSEEN / OLDER HISTORY
        # ====================================================

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


        # ====================================================
        # RECENT 120 DAYS
        #
        # Indicators were calculated before slicing,
        # so they retain proper historical context.
        # ====================================================

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


        print()
        print("=" * 100)

        print(
            f"ANALYZING {timeframe}"
        )

        print(
            f"FULL: "
            f"{full_df.iloc[0]['timestamp']} "
            f"→ "
            f"{full_df.iloc[-1]['timestamp']}"
        )

        print(
            f"UNSEEN: "
            f"{unseen_df.iloc[0]['timestamp']} "
            f"→ "
            f"{unseen_df.iloc[-1]['timestamp']}"
        )

        print(
            f"RECENT: "
            f"{recent_df.iloc[0]['timestamp']} "
            f"→ "
            f"{recent_df.iloc[-1]['timestamp']}"
        )

        print("=" * 100)


        full_results = analyze_period(
            df=full_df,
            timeframe=timeframe,
            period_name="FULL_3Y",
            warmup_bars=250
        )


        unseen_results = analyze_period(
            df=unseen_df,
            timeframe=timeframe,
            period_name="UNSEEN_HISTORY",
            warmup_bars=250
        )


        recent_results = analyze_period(
            df=recent_df,
            timeframe=timeframe,
            period_name="RECENT_120D",
            warmup_bars=0
        )


        all_results.extend(
            full_results
        )

        all_results.extend(
            unseen_results
        )

        all_results.extend(
            recent_results
        )


    # ========================================================
    # SAVE REPORT
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
        f"v3_inverse_regime_analysis_"
        f"{SYMBOL}_"
        f"{stamp}.csv"
    )


    results_df.to_csv(
        report_file,
        index=False
    )


    # ========================================================
    # PRINT MAIN RESULTS
    # ========================================================

    print_leaderboard(
        results_df,
        "FULL_3Y"
    )


    print_leaderboard(
        results_df,
        "UNSEEN_HISTORY"
    )


    print_leaderboard(
        results_df,
        "RECENT_120D"
    )


    print()
    print("=" * 100)

    print(
        "REGIME ANALYSIS FINISHED"
    )

    print(
        f"Full report: "
        f"{report_file}"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
