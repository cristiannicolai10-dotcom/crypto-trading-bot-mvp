import os
import json
import logging

import numpy as np
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

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config",
    "markets.json"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    "signal_engine.log"
)


STRATEGY_VERSION = "signal_v3"


MIN_SIGNAL_SCORE = 60
MIN_SCORE_DIFFERENCE = 15

ATR_STOP_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 3.0


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


# ============================================================
# LOGGING
# ============================================================

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)


logger = logging.getLogger(
    "signal_engine"
)


logging.getLogger(
    "httpx"
).setLevel(
    logging.WARNING
)


logging.getLogger(
    "httpcore"
).setLevel(
    logging.WARNING
)


logging.getLogger(
    "supabase"
).setLevel(
    logging.WARNING
)


# ============================================================
# SUPABASE
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# CONFIG FILE
# ============================================================

def load_config():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# TOP 30
# ============================================================

def get_top30_symbols():

    response = (
        supabase
        .table(
            "market_universe"
        )
        .select(
            "symbol,rank"
        )
        .eq(
            "status",
            "top30"
        )
        .order(
            "rank"
        )
        .execute()
    )


    if not response.data:

        return []


    return [
        row["symbol"]
        for row in response.data
        if row.get("symbol")
    ]


# ============================================================
# GET CANDLES
# ============================================================

def get_candles(
    symbol,
    timeframe,
    limit=300
):

    response = (
        supabase
        .table(
            "market_candles"
        )
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
            desc=True
        )
        .limit(
            limit
        )
        .execute()
    )


    if not response.data:

        return pd.DataFrame()


    df = pd.DataFrame(
        response.data
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
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    return df


# ============================================================
# GET INDICATORS
# ============================================================

def get_indicators(
    symbol,
    timeframe,
    limit=300
):

    response = (
        supabase
        .table(
            "indicators"
        )
        .select(
            "timestamp,"
            "atr,"
            "ema_50,"
            "ema_200,"
            "volume_ratio,"
            "alligator_lips,"
            "alligator_teeth,"
            "alligator_jaw,"
            "fractal_high,"
            "fractal_low,"
            "btc_correlation,"
            "market_regime,"
            "trend_score,"
            "volatility_regime"
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
            desc=True
        )
        .limit(
            limit
        )
        .execute()
    )


    if not response.data:

        return pd.DataFrame()


    df = pd.DataFrame(
        response.data
    )


    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True
        )
    )


    numeric_columns = [
        "atr",
        "ema_50",
        "ema_200",
        "volume_ratio",
        "alligator_lips",
        "alligator_teeth",
        "alligator_jaw",
        "btc_correlation",
        "trend_score"
    ]


    for column in numeric_columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
        )


    df = (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    return df


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    symbol,
    timeframe
):

    candles = (
        get_candles(
            symbol,
            timeframe
        )
    )


    indicators = (
        get_indicators(
            symbol,
            timeframe
        )
    )


    if (
        candles.empty
        or indicators.empty
    ):

        return pd.DataFrame()


    df = (
        candles
        .merge(
            indicators,
            on="timestamp",
            how="inner"
        )
    )


    df = (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    return df


# ============================================================
# BTC REGIME
# ============================================================

def get_btc_regime(
    timeframe
):

    response = (
        supabase
        .table(
            "indicators"
        )
        .select(
            "timestamp,"
            "market_regime"
        )
        .eq(
            "symbol",
            "BTCUSDT"
        )
        .eq(
            "timeframe",
            timeframe
        )
        .order(
            "timestamp",
            desc=True
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return (
        response.data[0]
        .get(
            "market_regime"
        )
    )


# ============================================================
# LAST CONFIRMED FRACTAL LEVELS
# ============================================================

def get_last_fractal_levels(
    df
):

    last_high = None
    last_low = None


    for index in range(
        len(df)
    ):

        if index < 2:

            continue


        if bool(
            df.iloc[index][
                "fractal_high"
            ]
        ):

            value = (
                df.iloc[
                    index - 2
                ][
                    "high"
                ]
            )


            if pd.notna(
                value
            ):

                last_high = float(
                    value
                )


        if bool(
            df.iloc[index][
                "fractal_low"
            ]
        ):

            value = (
                df.iloc[
                    index - 2
                ][
                    "low"
                ]
            )


            if pd.notna(
                value
            ):

                last_low = float(
                    value
                )


    return (
        last_high,
        last_low
    )


# ============================================================
# CALCULATE SCORES
# ============================================================

def calculate_scores(
    df,
    symbol,
    btc_regime
):

    latest = (
        df.iloc[-1]
    )


    previous = (
        df.iloc[-2]
        if len(df) >= 2
        else latest
    )


    long_score = 0
    short_score = 0

    reasons_long = []
    reasons_short = []


    close = float(
        latest["close"]
    )


    previous_close = float(
        previous["close"]
    )


    ema50 = latest[
        "ema_50"
    ]


    ema200 = latest[
        "ema_200"
    ]


    lips = latest[
        "alligator_lips"
    ]


    teeth = latest[
        "alligator_teeth"
    ]


    jaw = latest[
        "alligator_jaw"
    ]


    atr = latest[
        "atr"
    ]


    volume_ratio = latest[
        "volume_ratio"
    ]


    correlation = latest[
        "btc_correlation"
    ]


    market_regime = latest[
        "market_regime"
    ]


    # ========================================================
    # 1. MARKET REGIME
    # 20 POINTS
    # ========================================================

    if (
        market_regime
        == "bullish"
    ):

        long_score += 20

        reasons_long.append(
            "bullish_market_regime"
        )


    elif (
        market_regime
        == "bearish"
    ):

        short_score += 20

        reasons_short.append(
            "bearish_market_regime"
        )


    # ========================================================
    # 2. EMA STRUCTURE
    # 20 POINTS
    # ========================================================

    if (
        pd.notna(
            ema50
        )
        and pd.notna(
            ema200
        )
    ):

        if (
            close
            > ema50
        ):

            long_score += 10

            reasons_long.append(
                "price_above_ema50"
            )


        elif (
            close
            < ema50
        ):

            short_score += 10

            reasons_short.append(
                "price_below_ema50"
            )


        if (
            ema50
            > ema200
        ):

            long_score += 10

            reasons_long.append(
                "ema50_above_ema200"
            )


        elif (
            ema50
            < ema200
        ):

            short_score += 10

            reasons_short.append(
                "ema50_below_ema200"
            )


    # ========================================================
    # 3. ALLIGATOR
    # 15 POINTS
    # ========================================================

    if (
        pd.notna(
            lips
        )
        and pd.notna(
            teeth
        )
        and pd.notna(
            jaw
        )
    ):

        if (
            lips
            > teeth
            > jaw
        ):

            long_score += 15

            reasons_long.append(
                "bullish_alligator"
            )


        elif (
            lips
            < teeth
            < jaw
        ):

            short_score += 15

            reasons_short.append(
                "bearish_alligator"
            )


    # ========================================================
    # 4. FRACTAL CROSSING
    # 20 POINTS
    #
    # V3:
    # crossing is mandatory for trade.
    # ========================================================

    (
        last_fractal_high,
        last_fractal_low
    ) = get_last_fractal_levels(
        df
    )


    bullish_fractal_breakout = False
    bearish_fractal_breakdown = False


    if (
        last_fractal_high
        is not None
    ):

        bullish_fractal_breakout = (
            previous_close
            <= last_fractal_high
            and close
            > last_fractal_high
        )


    if (
        last_fractal_low
        is not None
    ):

        bearish_fractal_breakdown = (
            previous_close
            >= last_fractal_low
            and close
            < last_fractal_low
        )


    if bullish_fractal_breakout:

        long_score += 20

        reasons_long.append(
            "fractal_high_crossing"
        )


    if bearish_fractal_breakdown:

        short_score += 20

        reasons_short.append(
            "fractal_low_crossing"
        )


    # ========================================================
    # 5. BTC CONTEXT
    # 15 POINTS
    #
    # BTCUSDT itself receives NO extra BTC-context points.
    # ========================================================

    if (
        symbol
        != "BTCUSDT"
    ):

        if pd.notna(
            correlation
        ):

            # ------------------------------------------------
            # POSITIVE BTC CORRELATION
            # ------------------------------------------------

            if (
                correlation
                >= 0.40
            ):

                if (
                    btc_regime
                    == "bullish"
                ):

                    long_score += 15

                    reasons_long.append(
                        "positive_btc_correlation_bullish"
                    )


                elif (
                    btc_regime
                    == "bearish"
                ):

                    short_score += 15

                    reasons_short.append(
                        "positive_btc_correlation_bearish"
                    )


            # ------------------------------------------------
            # INVERSE BTC CORRELATION
            # ------------------------------------------------

            elif (
                correlation
                <= -0.40
            ):

                if (
                    btc_regime
                    == "bullish"
                ):

                    short_score += 15

                    reasons_short.append(
                        "inverse_btc_correlation_bullish"
                    )


                elif (
                    btc_regime
                    == "bearish"
                ):

                    long_score += 15

                    reasons_long.append(
                        "inverse_btc_correlation_bearish"
                    )


    # ========================================================
    # 6. VOLUME CONFIRMATION
    # 10 POINTS
    # ========================================================

    if (
        pd.notna(
            volume_ratio
        )
        and volume_ratio
        >= 1.20
    ):

        if (
            close
            > previous_close
        ):

            long_score += 10

            reasons_long.append(
                "bullish_volume_confirmation"
            )


        elif (
            close
            < previous_close
        ):

            short_score += 10

            reasons_short.append(
                "bearish_volume_confirmation"
            )


    long_score = min(
        long_score,
        100
    )


    short_score = min(
        short_score,
        100
    )


    return {

        "long_score":
            long_score,

        "short_score":
            short_score,

        "atr":
            atr,

        "last_fractal_high":
            last_fractal_high,

        "last_fractal_low":
            last_fractal_low,

        "bullish_fractal_breakout":
            bullish_fractal_breakout,

        "bearish_fractal_breakdown":
            bearish_fractal_breakdown,

        "reasons_long":
            reasons_long,

        "reasons_short":
            reasons_short
    }


# ============================================================
# V3 TRADE DECISION
# ============================================================

def determine_direction(
    score_data
):

    long_score = (
        score_data[
            "long_score"
        ]
    )


    short_score = (
        score_data[
            "short_score"
        ]
    )


    bullish_crossing = (
        score_data[
            "bullish_fractal_breakout"
        ]
    )


    bearish_crossing = (
        score_data[
            "bearish_fractal_breakdown"
        ]
    )


    long_difference = (
        long_score
        - short_score
    )


    short_difference = (
        short_score
        - long_score
    )


    # ========================================================
    # LONG
    # ========================================================

    if (
        bullish_crossing
        and long_score
        >= MIN_SIGNAL_SCORE
        and long_difference
        >= MIN_SCORE_DIFFERENCE
    ):

        return "LONG"


    # ========================================================
    # SHORT
    # ========================================================

    if (
        bearish_crossing
        and short_score
        >= MIN_SIGNAL_SCORE
        and short_difference
        >= MIN_SCORE_DIFFERENCE
    ):

        return "SHORT"


    return "HOLD"


# ============================================================
# RISK LEVELS
# ============================================================

def calculate_risk_levels(
    direction,
    entry,
    atr
):

    if (
        direction
        == "HOLD"
    ):

        return (
            None,
            None,
            None
        )


    if (
        pd.isna(
            atr
        )
        or atr <= 0
    ):

        return (
            None,
            None,
            None
        )


    atr = float(
        atr
    )


    # ========================================================
    # LONG
    # ========================================================

    if (
        direction
        == "LONG"
    ):

        stop_loss = (
            entry
            - ATR_STOP_MULTIPLIER
            * atr
        )


        take_profit = (
            entry
            + ATR_TARGET_MULTIPLIER
            * atr
        )


    # ========================================================
    # SHORT
    # ========================================================

    else:

        stop_loss = (
            entry
            + ATR_STOP_MULTIPLIER
            * atr
        )


        take_profit = (
            entry
            - ATR_TARGET_MULTIPLIER
            * atr
        )


    risk = abs(
        entry
        - stop_loss
    )


    reward = abs(
        take_profit
        - entry
    )


    if risk <= 0:

        risk_reward = None


    else:

        risk_reward = (
            reward
            / risk
        )


    return (
        stop_loss,
        take_profit,
        risk_reward
    )


# ============================================================
# JSON SAFE CLEANER
# ============================================================

def clean_value(
    value
):

    if value is None:

        return None


    if isinstance(
        value,
        dict
    ):

        return {
            key: clean_value(
                item
            )
            for key, item
            in value.items()
        }


    if isinstance(
        value,
        list
    ):

        return [
            clean_value(
                item
            )
            for item in value
        ]


    if isinstance(
        value,
        tuple
    ):

        return [
            clean_value(
                item
            )
            for item in value
        ]


    try:

        if pd.isna(
            value
        ):

            return None

    except (
        TypeError,
        ValueError
    ):

        pass


    if isinstance(
        value,
        (
            bool,
            np.bool_
        )
    ):

        return bool(
            value
        )


    if isinstance(
        value,
        (
            int,
            np.integer
        )
    ):

        return int(
            value
        )


    if isinstance(
        value,
        (
            float,
            np.floating
        )
    ):

        if not np.isfinite(
            value
        ):

            return None


        return float(
            value
        )


    return value


# ============================================================
# SAVE SIGNAL
# ============================================================

def save_signal(
    signal
):

    signal = (
        clean_value(
            signal
        )
    )


    (
        supabase
        .table(
            "signals"
        )
        .upsert(
            signal,
            on_conflict=(
                "symbol,"
                "timeframe,"
                "timestamp,"
                "strategy_version"
            )
        )
        .execute()
    )


# ============================================================
# PROCESS SYMBOL + TIMEFRAME
# ============================================================

def process_symbol_timeframe(
    symbol,
    timeframe,
    btc_regime
):

    df = (
        build_dataset(
            symbol,
            timeframe
        )
    )


    if (
        len(df)
        < 50
    ):

        logger.warning(
            f"INSUFFICIENT DATA | "
            f"{symbol} "
            f"{timeframe} | "
            f"rows={len(df)}"
        )

        return None


    latest = (
        df.iloc[-1]
    )


    score_data = (
        calculate_scores(
            df=df,
            symbol=symbol,
            btc_regime=btc_regime
        )
    )


    long_score = (
        score_data[
            "long_score"
        ]
    )


    short_score = (
        score_data[
            "short_score"
        ]
    )


    direction = (
        determine_direction(
            score_data
        )
    )


    confidence = max(
        long_score,
        short_score
    )


    entry_price = float(
        latest[
            "close"
        ]
    )


    (
        stop_loss,
        take_profit,
        risk_reward
    ) = calculate_risk_levels(
        direction=direction,
        entry=entry_price,
        atr=score_data[
            "atr"
        ]
    )


    if (
        direction
        in [
            "LONG",
            "SHORT"
        ]
    ):

        status = (
            "candidate"
        )


    else:

        status = (
            "no_trade"
        )


    details = {

        "market_regime":
            latest[
                "market_regime"
            ],

        "volatility_regime":
            latest[
                "volatility_regime"
            ],

        "btc_regime":
            btc_regime,

        "btc_correlation":
            latest[
                "btc_correlation"
            ],

        "volume_ratio":
            latest[
                "volume_ratio"
            ],

        "last_fractal_high":
            score_data[
                "last_fractal_high"
            ],

        "last_fractal_low":
            score_data[
                "last_fractal_low"
            ],

        "bullish_fractal_breakout":
            score_data[
                "bullish_fractal_breakout"
            ],

        "bearish_fractal_breakdown":
            score_data[
                "bearish_fractal_breakdown"
            ],

        "long_reasons":
            score_data[
                "reasons_long"
            ],

        "short_reasons":
            score_data[
                "reasons_short"
            ]
    }


    signal = {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "timestamp":
            latest[
                "timestamp"
            ].isoformat(),

        "direction":
            direction,

        "score":
            confidence,

        "confidence":
            confidence,

        "long_score":
            long_score,

        "short_score":
            short_score,

        "entry_price":
            entry_price,

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "risk_reward":
            risk_reward,

        "status":
            status,

        "details":
            details,

        "strategy_version":
            STRATEGY_VERSION
    }


    save_signal(
        signal
    )


    logger.info(
        f"SIGNAL | "
        f"{symbol} "
        f"{timeframe} | "
        f"{direction} | "
        f"LONG={long_score} | "
        f"SHORT={short_score} | "
        f"cross_up="
        f"{score_data['bullish_fractal_breakout']} | "
        f"cross_down="
        f"{score_data['bearish_fractal_breakdown']} | "
        f"confidence={confidence}"
    )


    return signal


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "=" * 70
    )


    logger.info(
        "SIGNAL ENGINE V3 STARTED"
    )


    config = (
        load_config()
    )


    timeframes = list(
        config.get(
            "timeframes",
            {}
        ).keys()
    )


    symbols = (
        get_top30_symbols()
    )


    if not symbols:

        raise RuntimeError(
            "No TOP30 symbols found"
        )


    if not timeframes:

        raise RuntimeError(
            "No timeframes configured"
        )


    processed = 0
    long_signals = 0
    short_signals = 0
    hold_signals = 0
    errors = 0


    for timeframe in timeframes:

        btc_regime = (
            get_btc_regime(
                timeframe
            )
        )


        if btc_regime is None:

            logger.warning(
                f"NO BTC REGIME | "
                f"{timeframe}"
            )

            continue


        for symbol in symbols:

            try:

                signal = (
                    process_symbol_timeframe(
                        symbol=symbol,
                        timeframe=timeframe,
                        btc_regime=btc_regime
                    )
                )


                if signal is None:

                    continue


                processed += 1


                if (
                    signal[
                        "direction"
                    ]
                    == "LONG"
                ):

                    long_signals += 1


                elif (
                    signal[
                        "direction"
                    ]
                    == "SHORT"
                ):

                    short_signals += 1


                else:

                    hold_signals += 1


            except Exception as error:

                errors += 1


                logger.exception(
                    f"FAILED | "
                    f"{symbol} "
                    f"{timeframe} | "
                    f"{error}"
                )


    logger.info(
        "SIGNAL ENGINE V3 FINISHED | "
        f"processed="
        f"{processed} | "
        f"LONG="
        f"{long_signals} | "
        f"SHORT="
        f"{short_signals} | "
        f"HOLD="
        f"{hold_signals} | "
        f"errors="
        f"{errors}"
    )


    logger.info(
        "=" * 70
    )


    print(
        "SIGNAL ENGINE V3 FINISHED | "
        f"processed="
        f"{processed} | "
        f"LONG="
        f"{long_signals} | "
        f"SHORT="
        f"{short_signals} | "
        f"HOLD="
        f"{hold_signals} | "
        f"errors="
        f"{errors}"
    )


if __name__ == "__main__":

    try:

        main()


    except Exception as error:

        logger.exception(
            f"FATAL ERROR | "
            f"{error}"
        )


        print(
            f"FATAL ERROR: "
            f"{error}"
        )


        raise
