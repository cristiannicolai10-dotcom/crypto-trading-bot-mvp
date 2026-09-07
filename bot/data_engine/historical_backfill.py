import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
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
    "historical_backfill.log"
)


# ============================================================
# RETENTION POLICY
# ============================================================

RETENTION_DAYS = {
    "5m": 30,
    "1h": 60,
    "4h": 120
}


# Bybit request limit
BYBIT_LIMIT = 1000

# Delay between API requests
REQUEST_DELAY = 0.20

# Delay after temporary error
ERROR_RETRY_DELAY = 5

# Maximum retries for one page
MAX_PAGE_RETRIES = 5


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
        "SUPABASE_URL missing from .env"
    )


if not SUPABASE_KEY:

    raise RuntimeError(
        "SUPABASE_KEY missing from .env"
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
    "historical_backfill"
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
# CONNECTIONS
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# Public MAINNET data
session = HTTP(
    testnet=False
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
# GET CURRENT TOP 30
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
# GET OLDEST EXISTING CANDLE
# ============================================================

def get_oldest_existing_timestamp(
    symbol,
    timeframe
):

    response = (
        supabase
        .table(
            "market_candles"
        )
        .select(
            "timestamp"
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
            1
        )
        .execute()
    )


    if not response.data:

        return None


    timestamp_text = (
        response.data[0][
            "timestamp"
        ]
    )


    timestamp_text = (
        timestamp_text.replace(
            "Z",
            "+00:00"
        )
    )


    timestamp = (
        datetime.fromisoformat(
            timestamp_text
        )
    )


    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.replace(
                tzinfo=timezone.utc
            )
        )


    return timestamp


# ============================================================
# PREPARE CANDLE
# ============================================================

def prepare_candle(
    symbol,
    timeframe,
    candle
):

    timestamp = (
        datetime.fromtimestamp(
            int(candle[0]) / 1000,
            tz=timezone.utc
        )
        .isoformat()
    )


    return {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "timestamp":
            timestamp,

        "open":
            candle[1],

        "high":
            candle[2],

        "low":
            candle[3],

        "close":
            candle[4],

        "volume":
            candle[5]
    }


# ============================================================
# SAVE CANDLE BATCH
# ============================================================

def save_batch(
    rows
):

    if not rows:

        return


    (
        supabase
        .table(
            "market_candles"
        )
        .upsert(
            rows,
            on_conflict=(
                "symbol,"
                "timeframe,"
                "timestamp"
            )
        )
        .execute()
    )


# ============================================================
# DOWNLOAD ONE PAGE WITH RETRIES
# ============================================================

def get_kline_page(
    symbol,
    timeframe_value,
    end_ms
):

    last_error = None


    for attempt in range(
        1,
        MAX_PAGE_RETRIES + 1
    ):

        try:

            response = (
                session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=timeframe_value,
                    end=end_ms,
                    limit=BYBIT_LIMIT
                )
            )


            ret_code = (
                response.get(
                    "retCode",
                    -1
                )
            )


            if ret_code != 0:

                raise RuntimeError(
                    f"Bybit retCode="
                    f"{ret_code} | "
                    f"{response.get('retMsg')}"
                )


            return (
                response.get(
                    "result",
                    {}
                )
                .get(
                    "list",
                    []
                )
            )


        except Exception as error:

            last_error = error


            logger.warning(
                f"RETRY | "
                f"{symbol} | "
                f"attempt="
                f"{attempt}/"
                f"{MAX_PAGE_RETRIES} | "
                f"{error}"
            )


            if attempt < MAX_PAGE_RETRIES:

                time.sleep(
                    ERROR_RETRY_DELAY
                )


    raise RuntimeError(
        f"Page failed after "
        f"{MAX_PAGE_RETRIES} retries | "
        f"{last_error}"
    )


# ============================================================
# BACKFILL SYMBOL + TIMEFRAME
# ============================================================

def backfill_symbol_timeframe(
    symbol,
    timeframe_name,
    timeframe_value,
    target_start_ms,
    default_end_ms
):

    oldest_existing = (
        get_oldest_existing_timestamp(
            symbol,
            timeframe_name
        )
    )


    # ========================================================
    # RESUME
    # ========================================================

    if oldest_existing:

        oldest_existing_ms = int(
            oldest_existing
            .timestamp()
            * 1000
        )


        # Already reached retention target
        if (
            oldest_existing_ms
            <= target_start_ms
        ):

            logger.info(
                f"ALREADY COMPLETE | "
                f"{symbol} "
                f"{timeframe_name} | "
                f"oldest="
                f"{oldest_existing.isoformat()}"
            )

            return 0


        current_end = (
            oldest_existing_ms
            - 1
        )


        logger.info(
            f"RESUME | "
            f"{symbol} "
            f"{timeframe_name} | "
            f"oldest_existing="
            f"{oldest_existing.isoformat()}"
        )


    else:

        current_end = (
            default_end_ms
        )


        logger.info(
            f"START NEW | "
            f"{symbol} "
            f"{timeframe_name}"
        )


    # ========================================================
    # PAGINATION
    # ========================================================

    total_saved = 0


    while (
        current_end
        > target_start_ms
    ):

        candles = (
            get_kline_page(
                symbol=symbol,
                timeframe_value=(
                    timeframe_value
                ),
                end_ms=current_end
            )
        )


        if not candles:

            logger.info(
                f"NO MORE DATA | "
                f"{symbol} "
                f"{timeframe_name}"
            )

            break


        rows = []

        oldest_batch_ts = None


        for candle in candles:

            candle_ts = int(
                candle[0]
            )


            # Never store data older
            # than retention target.
            if (
                candle_ts
                < target_start_ms
            ):

                continue


            rows.append(
                prepare_candle(
                    symbol=symbol,
                    timeframe=(
                        timeframe_name
                    ),
                    candle=candle
                )
            )


            if (
                oldest_batch_ts is None
                or candle_ts
                < oldest_batch_ts
            ):

                oldest_batch_ts = (
                    candle_ts
                )


        if rows:

            save_batch(
                rows
            )

            total_saved += (
                len(rows)
            )


        # No candle inside requested range
        if oldest_batch_ts is None:

            break


        next_end = (
            oldest_batch_ts
            - 1
        )


        # Safety against infinite loops
        if (
            next_end
            >= current_end
        ):

            logger.error(
                f"PAGINATION STUCK | "
                f"{symbol} "
                f"{timeframe_name}"
            )

            break


        current_end = (
            next_end
        )


        reached = (
            datetime.fromtimestamp(
                current_end / 1000,
                tz=timezone.utc
            )
        )


        logger.info(
            f"{symbol} "
            f"{timeframe_name} | "
            f"session_saved="
            f"{total_saved} | "
            f"reached="
            f"{reached.isoformat()}"
        )


        time.sleep(
            REQUEST_DELAY
        )


    logger.info(
        f"FINISHED | "
        f"{symbol} "
        f"{timeframe_name} | "
        f"session_saved="
        f"{total_saved}"
    )


    return total_saved


# ============================================================
# MAIN
# ============================================================

def main():

    config = (
        load_config()
    )


    timeframes = (
        config.get(
            "timeframes",
            {}
        )
    )


    if not timeframes:

        raise RuntimeError(
            "No timeframes configured "
            "in config/markets.json"
        )


    symbols = (
        get_top30_symbols()
    )


    if not symbols:

        raise RuntimeError(
            "No TOP30 symbols found. "
            "Run universe_scanner.py first."
        )


    now = (
        datetime.now(
            timezone.utc
        )
    )


    default_end_ms = int(
        now.timestamp()
        * 1000
    )


    logger.info(
        "=" * 70
    )


    logger.info(
        "HISTORICAL BACKFILL V3 STARTED"
    )


    logger.info(
        f"symbols="
        f"{len(symbols)}"
    )


    logger.info(
        f"retention="
        f"{RETENTION_DAYS}"
    )


    grand_total = 0

    successful_jobs = 0

    failed_jobs = 0


    # ========================================================
    # PROCESS TIMEFRAMES
    # ========================================================

    for (
        timeframe_name,
        timeframe_value
    ) in timeframes.items():


        retention_days = (
            RETENTION_DAYS.get(
                timeframe_name
            )
        )


        if retention_days is None:

            logger.warning(
                f"SKIPPING UNKNOWN "
                f"TIMEFRAME | "
                f"{timeframe_name}"
            )

            continue


        target_start = (
            now
            - timedelta(
                days=retention_days
            )
        )


        target_start_ms = int(
            target_start
            .timestamp()
            * 1000
        )


        logger.info(
            "-" * 70
        )


        logger.info(
            f"TIMEFRAME START | "
            f"{timeframe_name} | "
            f"retention_days="
            f"{retention_days} | "
            f"target_start="
            f"{target_start.isoformat()}"
        )


        for symbol in symbols:

            try:

                saved = (
                    backfill_symbol_timeframe(
                        symbol=symbol,
                        timeframe_name=(
                            timeframe_name
                        ),
                        timeframe_value=(
                            timeframe_value
                        ),
                        target_start_ms=(
                            target_start_ms
                        ),
                        default_end_ms=(
                            default_end_ms
                        )
                    )
                )


                grand_total += (
                    saved
                )


                successful_jobs += 1


            except Exception as error:

                failed_jobs += 1


                logger.exception(
                    f"JOB FAILED | "
                    f"{symbol} "
                    f"{timeframe_name} | "
                    f"{error}"
                )


    logger.info(
        "=" * 70
    )


    logger.info(
        "HISTORICAL BACKFILL V3 FINISHED | "
        f"new_rows="
        f"{grand_total} | "
        f"successful_jobs="
        f"{successful_jobs} | "
        f"failed_jobs="
        f"{failed_jobs}"
    )


    logger.info(
        "=" * 70
    )


    print()

    print(
        "=" * 70
    )


    print(
        "HISTORICAL BACKFILL V3 FINISHED"
    )


    print(
        f"New rows: "
        f"{grand_total}"
    )


    print(
        f"Successful jobs: "
        f"{successful_jobs}"
    )


    print(
        f"Errors: "
        f"{failed_jobs}"
    )


    print(
        "=" * 70
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
