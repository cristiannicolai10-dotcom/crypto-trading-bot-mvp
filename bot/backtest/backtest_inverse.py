import os
import json
import argparse
from datetime import datetime, timezone

import pandas as pd

from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"

ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

STRATEGY_VERSION = "signal_v3_inverse_backtest"

INITIAL_BALANCE = 1000.0

POSITION_FRACTION = 0.20
LEVERAGE = 1.0

MIN_SIGNAL_SCORE = 60
MIN_SCORE_DIFFERENCE = 15

ATR_STOP_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 3.0

TAKER_FEE_RATE = 0.00055

FUNDING_ENABLED = False

PAGE_SIZE = 1000
WARMUP_BARS = 250


TIMEFRAME_MINUTES = {
    "5m": 5,
    "1h": 60,
    "4h": 240
}


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    ENV_FILE
)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY"
)

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL missing"
    )

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_KEY missing"
    )


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ALL CANDLES
# ============================================================

def get_all_candles(
    symbol,
    timeframe
):

    all_rows = []

    last_timestamp = None

    print(
        f"Loading candles | "
        f"{symbol} {timeframe}"
    )


    while True:

        query = (
            supabase
            .table("market_candles")
            .select(
                "timestamp,"
                "open,"
                "high,"
                "low,"
                "close,"
                "volume"
            )
            .eq(
                "symbol",
                symbol
            )
            .eq(
                "timeframe",
                timeframe
            )
            .order(
                "timestamp",
                desc=False
            )
            .limit(
                PAGE_SIZE
            )
        )


        if last_timestamp is not None:

            query = query.gt(
                "timestamp",
                last_timestamp
            )


        response = (
            query
            .execute()
        )


        batch = (
            response.data
            or []
        )


        if not batch:
            break


        all_rows.extend(
            batch
        )


        new_last_timestamp = (
            batch[-1][
                "timestamp"
            ]
        )


        if (
            new_last_timestamp
            == last_timestamp
        ):

            raise RuntimeError(
                "Pagination stuck"
            )


        last_timestamp = (
            new_last_timestamp
        )


        if (
            len(all_rows)
            % 10000
            < PAGE_SIZE
        ):

            print(
                f"Loaded "
                f"{len(all_rows):,} "
                f"candles..."
            )


        if len(batch) < PAGE_SIZE:
            break


    if not all_rows:

        return (
            pd.DataFrame()
        )


    df = pd.DataFrame(
        all_rows
    )


    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True
        )
    )


    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
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


    return df


# ============================================================
# REMOVE CURRENT OPEN CANDLE
# ============================================================

def remove_open_candle(
    df,
    timeframe
):

    if df.empty:

        return df


    minutes = (
        TIMEFRAME_MINUTES[
            timeframe
        ]
    )


    now = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )


    last_timestamp = (
        df.iloc[-1][
            "timestamp"
        ]
    )


    close_time = (
        last_timestamp
        + pd.Timedelta(
            minutes=minutes
        )
    )


    if close_time > now:

        return (
            df
            .iloc[:-1]
            .copy()
        )


    return df


# ============================================================
# SMMA
# ============================================================

def smma(
    series,
    period
):

    return (
        series
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    df
):

    df = (
        df.copy()
    )


    # ========================================================
    # ATR 14
    # ========================================================

    previous_close = (
        df["close"]
        .shift(1)
    )


    tr1 = (
        df["high"]
        - df["low"]
    )


    tr2 = (
        df["high"]
        - previous_close
    ).abs()


    tr3 = (
        df["low"]
        - previous_close
    ).abs()


    true_range = (
        pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        )
        .max(
            axis=1
        )
    )


    df["atr"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )


    # ========================================================
    # EMA
    # ========================================================

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )


    # ========================================================
    # VOLUME RATIO
    # ========================================================

    volume_average = (
        df["volume"]
        .rolling(
            window=20
        )
        .mean()
    )


    df["volume_ratio"] = (
        df["volume"]
        / volume_average
    )


    # ========================================================
    # ALLIGATOR
    # ========================================================

    median_price = (
        (
            df["high"]
            + df["low"]
        )
        / 2
    )


    df[
        "alligator_jaw"
    ] = (
        smma(
            median_price,
            13
        )
        .shift(8)
    )


    df[
        "alligator_teeth"
    ] = (
        smma(
            median_price,
            8
        )
        .shift(5)
    )


    df[
        "alligator_lips"
    ] = (
        smma(
            median_price,
            5
        )
        .shift(3)
    )


    # ========================================================
    # WILLIAMS FRACTALS
    # ========================================================

    high = (
        df["high"]
    )

    low = (
        df["low"]
    )


    center_high = (
        high.shift(2)
    )


    df[
        "fractal_high"
    ] = (
        (
            center_high
            > high.shift(4)
        )
        & (
            center_high
            > high.shift(3)
        )
        & (
            center_high
            > high.shift(1)
        )
        & (
            center_high
            > high
        )
    )


    center_low = (
        low.shift(2)
    )


    df[
        "fractal_low"
    ] = (
        (
            center_low
            < low.shift(4)
        )
        & (
            center_low
            < low.shift(3)
        )
        & (
            center_low
            < low.shift(1)
        )
        & (
            center_low
            < low
        )
    )


    # Fractal level becomes available
    # only after confirmation.

    df[
        "fractal_high_level"
    ] = (
        high
        .shift(2)
        .where(
            df[
                "fractal_high"
            ]
        )
        .ffill()
    )


    df[
        "fractal_low_level"
    ] = (
        low
        .shift(2)
        .where(
            df[
                "fractal_low"
            ]
        )
        .ffill()
    )


    # ========================================================
    # MARKET REGIME
    # ========================================================

    df[
        "market_regime"
    ] = "sideways"


    bullish = (
        (
            df["close"]
            > df["ema_50"]
        )
        & (
            df["ema_50"]
            > df["ema_200"]
        )
    )


    bearish = (
        (
            df["close"]
            < df["ema_50"]
        )
        & (
            df["ema_50"]
            < df["ema_200"]
        )
    )


    df.loc[
        bullish,
        "market_regime"
    ] = "bullish"


    df.loc[
        bearish,
        "market_regime"
    ] = "bearish"


    # ========================================================
    # VOLATILITY REGIME
    # ========================================================

    atr_pct = (
        df["atr"]
        / df["close"]
    )


    atr_median = (
        atr_pct
        .rolling(
            window=50,
            min_periods=20
        )
        .median()
    )


    df[
        "volatility_regime"
    ] = "normal"


    df.loc[
        atr_pct
        > atr_median * 1.5,
        "volatility_regime"
    ] = "high"


    df.loc[
        atr_pct
        < atr_median * 0.75,
        "volatility_regime"
    ] = "low"


    return df


# ============================================================
# SIGNAL SCORE V2
# ============================================================

def calculate_scores(
    df
):

    df = (
        df.copy()
    )


    df[
        "long_score"
    ] = 0


    df[
        "short_score"
    ] = 0


    # ========================================================
    # 1. MARKET REGIME
    # 20 POINTS
    # ========================================================

    df.loc[
        df[
            "market_regime"
        ] == "bullish",
        "long_score"
    ] += 20


    df.loc[
        df[
            "market_regime"
        ] == "bearish",
        "short_score"
    ] += 20


    # ========================================================
    # 2. EMA STRUCTURE
    # 20 POINTS
    # ========================================================

    df.loc[
        df["close"]
        > df["ema_50"],
        "long_score"
    ] += 10


    df.loc[
        df["close"]
        < df["ema_50"],
        "short_score"
    ] += 10


    df.loc[
        df["ema_50"]
        > df["ema_200"],
        "long_score"
    ] += 10


    df.loc[
        df["ema_50"]
        < df["ema_200"],
        "short_score"
    ] += 10


    # ========================================================
    # 3. ALLIGATOR
    # 15 POINTS
    # ========================================================

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


    df.loc[
        bullish_alligator,
        "long_score"
    ] += 15


    df.loc[
        bearish_alligator,
        "short_score"
    ] += 15


    # ========================================================
    # 4. FRACTAL BREAKOUT
    # 20 POINTS
    #
    # V2 MODIFICATION:
    # Only actual crossing is counted.
    # ========================================================

    previous_close = (
        df["close"]
        .shift(1)
    )


    bullish_fractal_breakout = (
        df[
            "fractal_high_level"
        ].notna()
        & (
            previous_close
            <= df[
                "fractal_high_level"
            ]
        )
        & (
            df["close"]
            > df[
                "fractal_high_level"
            ]
        )
    )


    bearish_fractal_breakdown = (
        df[
            "fractal_low_level"
        ].notna()
        & (
            previous_close
            >= df[
                "fractal_low_level"
            ]
        )
        & (
            df["close"]
            < df[
                "fractal_low_level"
            ]
        )
    )


    df.loc[
        bullish_fractal_breakout,
        "long_score"
    ] += 20


    df.loc[
        bearish_fractal_breakdown,
        "short_score"
    ] += 20


    # ========================================================
    # 5. BTC CONTEXT
    #
    # V2 MODIFICATION:
    #
    # BTCUSDT DOES NOT RECEIVE EXTRA +15.
    #
    # Market regime already represents
    # BTC's own trend.
    # ========================================================

    # No additional points here.


    # ========================================================
    # 6. VOLUME CONFIRMATION
    # 10 POINTS
    # ========================================================

    bullish_volume = (
        (
            df[
                "volume_ratio"
            ]
            >= 1.20
        )
        & (
            df["close"]
            > previous_close
        )
    )


    bearish_volume = (
        (
            df[
                "volume_ratio"
            ]
            >= 1.20
        )
        & (
            df["close"]
            < previous_close
        )
    )


    df.loc[
        bullish_volume,
        "long_score"
    ] += 10


    df.loc[
        bearish_volume,
        "short_score"
    ] += 10


    # ========================================================
    # SCORE LIMIT
    # ========================================================

    df[
        "long_score"
    ] = (
        df[
            "long_score"
        ]
        .clip(
            upper=100
        )
    )


    df[
        "short_score"
    ] = (
        df[
            "short_score"
        ]
        .clip(
            upper=100
        )
    )


    # ========================================================
    # FINAL TRADE DECISION - V3
    #
    # A fractal crossing is now MANDATORY.
    #
    # LONG:
    #   bullish fractal breakout
    #   + long score >= threshold
    #   + LONG advantage >= minimum difference
    #
    # SHORT:
    #   bearish fractal breakdown
    #   + short score >= threshold
    #   + SHORT advantage >= minimum difference
    # ========================================================

    df[
        "direction"
    ] = "HOLD"


    long_difference = (
        df["long_score"]
        - df["short_score"]
    )


    short_difference = (
        df["short_score"]
        - df["long_score"]
    )


    valid_long = (
        bullish_fractal_breakout
        & (
            df["long_score"]
            >= MIN_SIGNAL_SCORE
        )
        & (
            long_difference
            >= MIN_SCORE_DIFFERENCE
        )
    )


    valid_short = (
        bearish_fractal_breakdown
        & (
            df["short_score"]
            >= MIN_SIGNAL_SCORE
        )
        & (
            short_difference
            >= MIN_SCORE_DIFFERENCE
        )
    )


    df.loc[
        valid_long,
        "direction"
    ] = "LONG"


    df.loc[
        valid_short,
        "direction"
    ] = "SHORT"


    return df


# ============================================================
# TRADE SIMULATION
# ============================================================

def run_backtest(
    df
):

    trades = []


    balance = (
        INITIAL_BALANCE
    )


    peak_balance = (
        balance
    )


    max_drawdown = 0.0


    i = (
        WARMUP_BARS
    )


    while (
        i
        < len(df) - 1
    ):

        signal = (
            df.iloc[i]
        )


        direction = (
            signal[
                "direction"
            ]
        )

       # ====================================================
        # INVERSE EXPERIMENT
        # LONG signal -> SHORT trade
        # SHORT signal -> LONG trade
        # HOLD remains HOLD
        # ====================================================

        if direction == "LONG":

            direction = "SHORT"

        elif direction == "SHORT":

            direction = "LONG"


        if (
            direction
            == "HOLD"
        ):

            i += 1

            continue


        atr = (
            signal[
                "atr"
            ]
        )


        if (
            pd.isna(
                atr
            )
            or atr <= 0
        ):

            i += 1

            continue


        # Signal becomes known only
        # when candle i closes.
        #
        # Entry is at candle i+1 OPEN.

        entry_index = (
            i + 1
        )


        entry_row = (
            df.iloc[
                entry_index
            ]
        )


        entry_price = float(
            entry_row[
                "open"
            ]
        )


        # ====================================================
        # SL / TP
        # ====================================================

        if (
            direction
            == "LONG"
        ):

            stop_loss = (
                entry_price
                - ATR_STOP_MULTIPLIER
                * atr
            )


            take_profit = (
                entry_price
                + ATR_TARGET_MULTIPLIER
                * atr
            )


        else:

            stop_loss = (
                entry_price
                + ATR_STOP_MULTIPLIER
                * atr
            )


            take_profit = (
                entry_price
                - ATR_TARGET_MULTIPLIER
                * atr
            )


        exit_price = None

        exit_index = None

        exit_reason = None


        j = (
            entry_index
        )


        while (
            j
            < len(df)
        ):

            candle = (
                df.iloc[j]
            )


            high = float(
                candle[
                    "high"
                ]
            )


            low = float(
                candle[
                    "low"
                ]
            )


            # ================================================
            # LONG
            # ================================================

            if (
                direction
                == "LONG"
            ):

                stop_hit = (
                    low
                    <= stop_loss
                )


                target_hit = (
                    high
                    >= take_profit
                )


                # Conservative:
                # if SL and TP occur
                # in same candle,
                # assume SL happened first.

                if stop_hit:

                    exit_price = (
                        stop_loss
                    )


                    exit_index = (
                        j
                    )


                    if target_hit:

                        exit_reason = (
                            "SL_TP_SAME_BAR"
                        )

                    else:

                        exit_reason = (
                            "SL"
                        )


                    break


                if target_hit:

                    exit_price = (
                        take_profit
                    )


                    exit_index = (
                        j
                    )


                    exit_reason = (
                        "TP"
                    )


                    break


            # ================================================
            # SHORT
            # ================================================

            else:

                stop_hit = (
                    high
                    >= stop_loss
                )


                target_hit = (
                    low
                    <= take_profit
                )


                if stop_hit:

                    exit_price = (
                        stop_loss
                    )


                    exit_index = (
                        j
                    )


                    if target_hit:

                        exit_reason = (
                            "SL_TP_SAME_BAR"
                        )

                    else:

                        exit_reason = (
                            "SL"
                        )


                    break


                if target_hit:

                    exit_price = (
                        take_profit
                    )


                    exit_index = (
                        j
                    )


                    exit_reason = (
                        "TP"
                    )


                    break


            j += 1


        # ====================================================
        # POSITION STILL OPEN AT END OF DATASET
        # ====================================================

        if (
            exit_price
            is None
        ):

            exit_index = (
                len(df)
                - 1
            )


            exit_price = float(
                df.iloc[
                    exit_index
                ][
                    "close"
                ]
            )


            exit_reason = (
                "END_OF_DATA"
            )


        # ====================================================
        # RETURN
        # ====================================================

        if (
            direction
            == "LONG"
        ):

            gross_return = (
                (
                    exit_price
                    - entry_price
                )
                / entry_price
            )


        else:

            gross_return = (
                (
                    entry_price
                    - exit_price
                )
                / entry_price
            )


        # Entry + exit fee.

        fee_return = (
            TAKER_FEE_RATE
            + (
                TAKER_FEE_RATE
                * (
                    exit_price
                    / entry_price
                )
            )
        )


        funding_return = 0.0


        net_return = (
            gross_return
            - fee_return
            - funding_return
        )


        # ====================================================
        # POSITION SIZE
        # ====================================================

        margin = (
            balance
            * POSITION_FRACTION
        )


        notional = (
            margin
            * LEVERAGE
        )


        pnl = (
            notional
            * net_return
        )


        fees_usd = (
            notional
            * fee_return
        )


        balance_before = (
            balance
        )


        balance += (
            pnl
        )


        # ====================================================
        # DRAWDOWN
        # ====================================================

        peak_balance = max(
            peak_balance,
            balance
        )


        drawdown = (
            (
                peak_balance
                - balance
            )
            / peak_balance
        )


        max_drawdown = max(
            max_drawdown,
            drawdown
        )


        # ====================================================
        # SAVE TRADE
        # ====================================================

        trades.append(
            {

                "signal_time":
                    signal[
                        "timestamp"
                    ].isoformat(),

                "entry_time":
                    entry_row[
                        "timestamp"
                    ].isoformat(),

                "exit_time":
                    df.iloc[
                        exit_index
                    ][
                        "timestamp"
                    ].isoformat(),

                "direction":
                    direction,

                "long_score":
                    int(
                        signal[
                            "long_score"
                        ]
                    ),

                "short_score":
                    int(
                        signal[
                            "short_score"
                        ]
                    ),

                "entry_price":
                    entry_price,

                "stop_loss":
                    float(
                        stop_loss
                    ),

                "take_profit":
                    float(
                        take_profit
                    ),

                "exit_price":
                    float(
                        exit_price
                    ),

                "exit_reason":
                    exit_reason,

                "gross_return_pct":
                    gross_return
                    * 100,

                "net_return_pct":
                    net_return
                    * 100,

                "fees_usd":
                    fees_usd,

                "funding_usd":
                    0.0,

                "balance_before":
                    balance_before,

                "pnl_usd":
                    pnl,

                "balance_after":
                    balance
            }
        )


        # Only one open position at a time
        # for this symbol/timeframe.

        i = (
            exit_index
            + 1
        )


    return (
        pd.DataFrame(
            trades
        ),
        balance,
        max_drawdown
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    trades,
    symbol,
    timeframe,
    df,
    final_balance,
    max_drawdown
):

    if trades.empty:

        return {

            "strategy_version":
                STRATEGY_VERSION,

            "symbol":
                symbol,

            "timeframe":
                timeframe,

            "candles":
                int(
                    len(df)
                ),

            "initial_balance":
                INITIAL_BALANCE,

            "final_balance":
                INITIAL_BALANCE,

            "net_pnl":
                0.0,

            "total_return_pct":
                0.0,

            "total_trades":
                0
        }


    wins = (
        trades[
            trades[
                "pnl_usd"
            ] > 0
        ]
    )


    losses = (
        trades[
            trades[
                "pnl_usd"
            ] < 0
        ]
    )


    gross_profit = (
        wins[
            "pnl_usd"
        ]
        .sum()
    )


    gross_loss = abs(
        losses[
            "pnl_usd"
        ]
        .sum()
    )


    if (
        gross_loss
        > 0
    ):

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = None


    consecutive_losses = 0

    max_consecutive_losses = 0


    for pnl in (
        trades[
            "pnl_usd"
        ]
    ):

        if pnl < 0:

            consecutive_losses += 1


            max_consecutive_losses = max(
                max_consecutive_losses,
                consecutive_losses
            )


        else:

            consecutive_losses = 0


    return {

        "strategy_version":
            STRATEGY_VERSION,

        "symbol":
            symbol,

        "timeframe":
            timeframe,

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

        "initial_balance":
            INITIAL_BALANCE,

        "final_balance":
            float(
                final_balance
            ),

        "net_pnl":
            float(
                final_balance
                - INITIAL_BALANCE
            ),

        "total_return_pct":
            float(
                (
                    final_balance
                    / INITIAL_BALANCE
                    - 1
                )
                * 100
            ),

        "total_trades":
            int(
                len(
                    trades
                )
            ),

        "long_trades":
            int(
                (
                    trades[
                        "direction"
                    ]
                    == "LONG"
                )
                .sum()
            ),

        "short_trades":
            int(
                (
                    trades[
                        "direction"
                    ]
                    == "SHORT"
                )
                .sum()
            ),

        "wins":
            int(
                len(
                    wins
                )
            ),

        "losses":
            int(
                len(
                    losses
                )
            ),

        "win_rate_pct":
            float(
                len(wins)
                / len(trades)
                * 100
            ),

        "gross_profit":
            float(
                gross_profit
            ),

        "gross_loss":
            float(
                gross_loss
            ),

        "profit_factor":
            (
                float(
                    profit_factor
                )
                if (
                    profit_factor
                    is not None
                )
                else None
            ),

        "average_trade_pct":
            float(
                trades[
                    "net_return_pct"
                ]
                .mean()
            ),

        "max_drawdown_pct":
            float(
                max_drawdown
                * 100
            ),

        "max_consecutive_losses":
            int(
                max_consecutive_losses
            ),

        "total_fees_usd":
            float(
                trades[
                    "fees_usd"
                ]
                .sum()
            ),

        "funding_included":
            FUNDING_ENABLED,

        "position_fraction":
            POSITION_FRACTION,

        "leverage":
            LEVERAGE,

        "atr_stop_multiplier":
            ATR_STOP_MULTIPLIER,

        "atr_target_multiplier":
            ATR_TARGET_MULTIPLIER,

        "minimum_score":
            MIN_SIGNAL_SCORE,

        "minimum_score_difference":
            MIN_SCORE_DIFFERENCE
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--symbol",
        default="BTCUSDT"
    )


    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=[
            "5m",
            "1h",
            "4h"
        ]
    )


    args = (
        parser.parse_args()
    )


    symbol = (
        args.symbol.upper()
    )


    timeframe = (
        args.timeframe
    )


    print(
        "=" * 60
    )


    print(
        f"BACKTEST START | "
        f"{symbol} "
        f"{timeframe}"
    )


    df = (
        get_all_candles(
            symbol,
            timeframe
        )
    )


    if df.empty:

        raise RuntimeError(
            "No candles found"
        )


    df = (
        remove_open_candle(
            df,
            timeframe
        )
    )


    print(
        f"Dataset: "
        f"{len(df):,} "
        f"candles"
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
        < WARMUP_BARS + 10
    ):

        raise RuntimeError(
            "Insufficient historical data"
        )


    print(
        "Calculating indicators..."
    )


    df = (
        calculate_indicators(
            df
        )
    )


    print(
        "Calculating historical signals..."
    )


    df = (
        calculate_scores(
            df
        )
    )


    print(
        "Simulating trades..."
    )


    (
        trades,
        final_balance,
        max_drawdown
    ) = run_backtest(
        df
    )


    summary = (
        build_summary(
            trades=trades,
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            final_balance=final_balance,
            max_drawdown=max_drawdown
        )
    )


    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    base_name = (
        f"backtest_"
        f"{symbol}_"
        f"{timeframe}_"
        f"{STRATEGY_VERSION}_"
        f"{stamp}"
    )


    trades_file = (
        os.path.join(
            REPORT_DIR,
            base_name
            + "_trades.csv"
        )
    )


    summary_file = (
        os.path.join(
            REPORT_DIR,
            base_name
            + "_summary.json"
        )
    )


    trades.to_csv(
        trades_file,
        index=False
    )


    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )


    print()

    print(
        "=" * 60
    )


    print(
        "BACKTEST FINISHED"
    )


    print(
        "=" * 60
    )


    print(
        f"Strategy: "
        f"{STRATEGY_VERSION}"
    )


    print(
        f"Trades: "
        f"{summary.get('total_trades', 0)}"
    )


    if (
        summary.get(
            "total_trades",
            0
        )
        > 0
    ):

        print(
            f"Win rate: "
            f"{summary['win_rate_pct']:.2f}%"
        )


        if (
            summary[
                "profit_factor"
            ]
            is None
        ):

            print(
                "Profit factor: N/A"
            )


        else:

            print(
                f"Profit factor: "
                f"{summary['profit_factor']:.4f}"
            )


        print(
            f"Max drawdown: "
            f"{summary['max_drawdown_pct']:.2f}%"
        )


        print(
            f"Fees: "
            f"${summary['total_fees_usd']:.2f}"
        )


        print(
            f"Initial balance: "
            f"${summary['initial_balance']:.2f}"
        )


        print(
            f"Final balance: "
            f"${summary['final_balance']:.2f}"
        )


        print(
            f"Net PnL: "
            f"${summary['net_pnl']:.2f}"
        )


        print(
            f"Return: "
            f"{summary['total_return_pct']:.2f}%"
        )


    print()


    print(
        f"Trades file: "
        f"{trades_file}"
    )


    print(
        f"Summary file: "
        f"{summary_file}"
    )


if __name__ == "__main__":

    main()
