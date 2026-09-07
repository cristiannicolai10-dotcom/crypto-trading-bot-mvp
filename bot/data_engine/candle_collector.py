import os
import json
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from supabase import create_client


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"
CONFIG_FILE = os.path.join(BASE_DIR, "config", "markets.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "candle_collector.log")
ENV_FILE = os.path.join(BASE_DIR, ".env")


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL missing from .env")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY missing from .env")


# ============================================================
# LOGGING
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

logger = logging.getLogger("candle_collector")

# Disable noisy library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)


# ============================================================
# SUPABASE
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# BYBIT MARKET DATA
# ============================================================

# IMPORTANT:
# Mainnet is used only for real public market data.
# Demo/Testnet execution will be handled separately.
session = HTTP(
    testnet=False
)


# ============================================================
# CONFIG
# ============================================================

def load_config():
    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# GET DYNAMIC TOP 30
# ============================================================

def get_top30_symbols():

    response = (
        supabase
        .table("market_universe")
        .select("symbol,rank,status")
        .eq("status", "top30")
        .order("rank")
        .execute()
    )

    if not response.data:
        return []

    symbols = []

    for row in response.data:

        symbol = row.get("symbol")

        if symbol:
            symbols.append(symbol)

    return symbols


# ============================================================
# GET CANDLES FROM BYBIT
# ============================================================

def get_candles(
    symbol,
    timeframe_value,
    limit=50
):

    response = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=timeframe_value,
        limit=limit
    )

    ret_code = response.get("retCode", -1)

    if ret_code != 0:

        raise RuntimeError(
            f"Bybit error | "
            f"symbol={symbol} | "
            f"retCode={ret_code} | "
            f"message={response.get('retMsg')}"
        )

    return response["result"]["list"]


# ============================================================
# FORMAT BYBIT CANDLE FOR SUPABASE
# ============================================================

def prepare_candle(
    symbol,
    timeframe_name,
    candle
):

    timestamp = datetime.fromtimestamp(
        int(candle[0]) / 1000,
        tz=timezone.utc
    ).isoformat()

    return {
        "symbol": symbol,
        "timeframe": timeframe_name,
        "timestamp": timestamp,
        "open": candle[1],
        "high": candle[2],
        "low": candle[3],
        "close": candle[4],
        "volume": candle[5]
    }


# ============================================================
# SAVE BATCH TO SUPABASE
# ============================================================

def save_candles(rows):

    if not rows:
        return

    (
        supabase
        .table("market_candles")
        .upsert(
            rows,
            on_conflict="symbol,timeframe,timestamp"
        )
        .execute()
    )


# ============================================================
# MAIN COLLECTOR
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("DATA COLLECTION STARTED")

    config = load_config()

    timeframes = config.get("timeframes", {})

    if not timeframes:
        raise RuntimeError(
            "No timeframes configured in config/markets.json"
        )

    # IMPORTANT:
    # Symbols are NOT read from markets.json anymore.
    symbols = get_top30_symbols()

    if not symbols:

        logger.error(
            "No symbols with status=top30 found "
            "in market_universe"
        )

        raise RuntimeError(
            "Top 30 universe is empty. "
            "Run universe_scanner.py first."
        )

    logger.info(
        f"Loaded {len(symbols)} TOP30 symbols"
    )

    logger.info(
        "TOP30: " + ", ".join(symbols)
    )

    logger.info(
        "Timeframes: " +
        ", ".join(timeframes.keys())
    )

    total_rows = 0
    successful_jobs = 0
    failed_jobs = 0

    for symbol in symbols:

        for timeframe_name, timeframe_value in timeframes.items():

            try:

                candles = get_candles(
                    symbol=symbol,
                    timeframe_value=timeframe_value,
                    limit=50
                )

                rows = []

                for candle in candles:

                    row = prepare_candle(
                        symbol=symbol,
                        timeframe_name=timeframe_name,
                        candle=candle
                    )

                    rows.append(row)

                save_candles(rows)

                total_rows += len(rows)
                successful_jobs += 1

                logger.info(
                    f"{symbol} {timeframe_name} completed "
                    f"| candles={len(rows)}"
                )

            except Exception as error:

                failed_jobs += 1

                logger.exception(
                    f"FAILED | "
                    f"{symbol} {timeframe_name} | "
                    f"{error}"
                )

    logger.info(
        "DATA COLLECTION FINISHED | "
        f"symbols={len(symbols)} | "
        f"timeframes={len(timeframes)} | "
        f"rows_processed={total_rows} | "
        f"successful_jobs={successful_jobs} | "
        f"failed_jobs={failed_jobs}"
    )

    logger.info("=" * 60)

    print(
        "DATA COLLECTION FINISHED | "
        f"symbols={len(symbols)} | "
        f"rows={total_rows} | "
        f"successful={successful_jobs} | "
        f"errors={failed_jobs}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        logger.exception(
            f"FATAL ERROR | {error}"
        )

        print(
            f"FATAL ERROR: {error}"
        )

        raise

