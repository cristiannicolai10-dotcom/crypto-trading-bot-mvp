import os
import json
import logging

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client


BASE_DIR = "/home/botadmin/crypto-bot"

ENV_FILE = os.path.join(BASE_DIR, ".env")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "markets.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "indicator_engine.log")


load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL missing from .env")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY missing from .env")


os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

logger = logging.getLogger("indicator_engine")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


TIMEFRAME_MINUTES = {
    "5m": 5,
    "1h": 60,
    "4h": 240
}


def load_config():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_top30_symbols():

    response = (
        supabase
        .table("market_universe")
        .select("symbol,rank")
        .eq("status", "top30")
        .order("rank")
        .execute()
    )

    if not response.data:
        return []

    return [
        row["symbol"]
        for row in response.data
        if row.get("symbol")
    ]


def get_candles(
    symbol,
    timeframe,
    limit=500
):

    response = (
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
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)

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
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def remove_open_candle(
    df,
    timeframe
):

    if df.empty:
        return df

    minutes = TIMEFRAME_MINUTES.get(
        timeframe
    )

    if not minutes:
        return df

    now = pd.Timestamp.now(tz="UTC")

    last_timestamp = (
        df.iloc[-1]["timestamp"]
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


def calculate_atr(
    df,
    period=14
):

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

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    return (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )


def calculate_alligator(
    df
):

    median_price = (
        df["high"]
        + df["low"]
    ) / 2

    jaw = (
        smma(
            median_price,
            13
        )
        .shift(8)
    )

    teeth = (
        smma(
            median_price,
            8
        )
        .shift(5)
    )

    lips = (
        smma(
            median_price,
            5
        )
        .shift(3)
    )

    return (
        jaw,
        teeth,
        lips
    )


def calculate_fractals(
    df
):

    high = df["high"]
    low = df["low"]

    center_high = high.shift(2)

    fractal_high = (
        (center_high > high.shift(4))
        & (center_high > high.shift(3))
        & (center_high > high.shift(1))
        & (center_high > high)
    )

    center_low = low.shift(2)

    fractal_low = (
        (center_low < low.shift(4))
        & (center_low < low.shift(3))
        & (center_low < low.shift(1))
        & (center_low < low)
    )

    return (
        fractal_high,
        fractal_low
    )


def calculate_market_regime(
    df
):

    ema50 = df["ema_50"]
    ema200 = df["ema_200"]
    close = df["close"]

    atr_pct = (
        df["atr"]
        / close
    )

    atr_median = (
        atr_pct
        .rolling(
            window=50,
            min_periods=20
        )
        .median()
    )

    regimes = []
    trend_scores = []
    volatility_regimes = []

    for i in range(len(df)):

        if (
            pd.isna(ema50.iloc[i])
            or pd.isna(ema200.iloc[i])
            or pd.isna(atr_pct.iloc[i])
        ):

            regimes.append(None)
            trend_scores.append(None)
            volatility_regimes.append(None)
            continue


        price = close.iloc[i]
        e50 = ema50.iloc[i]
        e200 = ema200.iloc[i]


        ema_distance = (
            (e50 - e200)
            / e200
        )


        if (
            price > e50
            and e50 > e200
        ):

            regime = "bullish"

        elif (
            price < e50
            and e50 < e200
        ):

            regime = "bearish"

        else:

            regime = "sideways"


        trend_score = (
            ema_distance
            * 100
        )


        median_vol = atr_median.iloc[i]

        if pd.isna(median_vol):

            volatility_regime = None

        elif atr_pct.iloc[i] > (
            median_vol * 1.5
        ):

            volatility_regime = "high"

        elif atr_pct.iloc[i] < (
            median_vol * 0.75
        ):

            volatility_regime = "low"

        else:

            volatility_regime = "normal"


        regimes.append(regime)

        trend_scores.append(
            trend_score
        )

        volatility_regimes.append(
            volatility_regime
        )


    return (
        regimes,
        trend_scores,
        volatility_regimes
    )


def calculate_btc_correlation(
    asset_df,
    btc_df,
    window=50
):

    asset = (
        asset_df[
            [
                "timestamp",
                "close"
            ]
        ]
        .rename(
            columns={
                "close":
                    "asset_close"
            }
        )
    )

    btc = (
        btc_df[
            [
                "timestamp",
                "close"
            ]
        ]
        .rename(
            columns={
                "close":
                    "btc_close"
            }
        )
    )

    merged = (
        asset
        .merge(
            btc,
            on="timestamp",
            how="left"
        )
    )


    merged["asset_return"] = (
        merged["asset_close"]
        .pct_change()
    )

    merged["btc_return"] = (
        merged["btc_close"]
        .pct_change()
    )


    merged["btc_correlation"] = (
        merged["asset_return"]
        .rolling(
            window=window,
            min_periods=20
        )
        .corr(
            merged["btc_return"]
        )
    )


    return (
        merged[
            [
                "timestamp",
                "btc_correlation"
            ]
        ]
    )


def calculate_indicators(
    df
):

    df = df.copy()

    df["atr"] = calculate_atr(
        df,
        period=14
    )

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


    (
        df["alligator_jaw"],
        df["alligator_teeth"],
        df["alligator_lips"]
    ) = calculate_alligator(df)


    (
        df["fractal_high"],
        df["fractal_low"]
    ) = calculate_fractals(df)


    (
        df["market_regime"],
        df["trend_score"],
        df["volatility_regime"]
    ) = calculate_market_regime(df)


    return df


def clean_value(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (bool, np.bool_)
    ):
        return bool(value)

    if isinstance(
        value,
        (int, np.integer)
    ):
        return int(value)

    if isinstance(
        value,
        (float, np.floating)
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    return value


def prepare_indicator_rows(
    symbol,
    timeframe,
    df
):

    rows = []

    for _, row in df.iterrows():

        database_row = {

            "symbol":
                symbol,

            "timeframe":
                timeframe,

            "timestamp":
                row["timestamp"].isoformat(),

            "atr":
                row["atr"],

            "alligator_lips":
                row["alligator_lips"],

            "alligator_teeth":
                row["alligator_teeth"],

            "alligator_jaw":
                row["alligator_jaw"],

            "ema_50":
                row["ema_50"],

            "ema_200":
                row["ema_200"],

            "volume_ratio":
                row["volume_ratio"],

            "fractal_high":
                row["fractal_high"],

            "fractal_low":
                row["fractal_low"],

            "btc_correlation":
                row.get(
                    "btc_correlation"
                ),

            "market_regime":
                row.get(
                    "market_regime"
                ),

            "trend_score":
                row.get(
                    "trend_score"
                ),

            "volatility_regime":
                row.get(
                    "volatility_regime"
                )
        }

        # Convert ALL pandas/numpy values
        # into JSON-safe Python values
        database_row = {
            key: clean_value(value)
            for key, value
            in database_row.items()
        }

        rows.append(
            database_row
        )

    return rows


def save_indicators(
    rows
):

    if not rows:
        return

    batch_size = 500

    for start in range(
        0,
        len(rows),
        batch_size
    ):

        batch = rows[
            start:
            start + batch_size
        ]

        (
            supabase
            .table("indicators")
            .upsert(
                batch,
                on_conflict=(
                    "symbol,"
                    "timeframe,"
                    "timestamp"
                )
            )
            .execute()
        )


def process_symbol_timeframe(
    symbol,
    timeframe,
    btc_df
):

    df = get_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=500
    )

    if df.empty:

        logger.warning(
            f"NO DATA | "
            f"{symbol} "
            f"{timeframe}"
        )

        return 0


    df = remove_open_candle(
        df,
        timeframe
    )


    if len(df) < 200:

        logger.warning(
            f"INSUFFICIENT DATA | "
            f"{symbol} "
            f"{timeframe} "
            f"| candles={len(df)}"
        )

        return 0


    df = calculate_indicators(
        df
    )


    if symbol == "BTCUSDT":

        df["btc_correlation"] = 1.0

    else:

        correlation = (
            calculate_btc_correlation(
                df,
                btc_df,
                window=50
            )
        )

        df = (
            df
            .merge(
                correlation,
                on="timestamp",
                how="left"
            )
        )


    rows = prepare_indicator_rows(
        symbol,
        timeframe,
        df
    )


    save_indicators(rows)


    logger.info(
        f"COMPLETED | "
        f"{symbol} "
        f"{timeframe} "
        f"| rows={len(rows)}"
    )


    return len(rows)


def main():

    logger.info("=" * 70)

    logger.info(
        "INDICATOR ENGINE V2 STARTED"
    )


    config = load_config()

    timeframes = list(
        config.get(
            "timeframes",
            {}
        ).keys()
    )

    symbols = get_top30_symbols()


    if not symbols:

        raise RuntimeError(
            "No TOP30 symbols found"
        )


    total_rows = 0
    successful_jobs = 0
    failed_jobs = 0


    for timeframe in timeframes:

        btc_df = get_candles(
            symbol="BTCUSDT",
            timeframe=timeframe,
            limit=500
        )

        btc_df = remove_open_candle(
            btc_df,
            timeframe
        )


        if len(btc_df) < 200:

            logger.warning(
                f"BTC REFERENCE "
                f"INSUFFICIENT | "
                f"{timeframe} "
                f"| candles="
                f"{len(btc_df)}"
            )

            continue


        for symbol in symbols:

            try:

                rows = (
                    process_symbol_timeframe(
                        symbol=symbol,
                        timeframe=timeframe,
                        btc_df=btc_df
                    )
                )

                total_rows += rows

                if rows > 0:

                    successful_jobs += 1


            except Exception as error:

                failed_jobs += 1

                logger.exception(
                    f"FAILED | "
                    f"{symbol} "
                    f"{timeframe} "
                    f"| {error}"
                )


    logger.info(
        "INDICATOR ENGINE V2 FINISHED | "
        f"rows={total_rows} | "
        f"successful="
        f"{successful_jobs} | "
        f"errors="
        f"{failed_jobs}"
    )

    logger.info("=" * 70)


    print(
        "INDICATOR ENGINE V2 FINISHED | "
        f"rows={total_rows} | "
        f"successful="
        f"{successful_jobs} | "
        f"errors="
        f"{failed_jobs}"
    )


if __name__ == "__main__":

    main()
