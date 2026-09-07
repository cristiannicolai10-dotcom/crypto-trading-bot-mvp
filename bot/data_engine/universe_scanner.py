import os
import time
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from supabase import create_client


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"
ENV_FILE = os.path.join(BASE_DIR, ".env")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "universe_scanner.log")

MIN_TURNOVER_24H = 10_000_000
TOP_N = 30


# ============================================================
# ENV
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

logger = logging.getLogger("universe_scanner")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)


# ============================================================
# CONNECTIONS
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Real public Bybit market data.
# No API key required for these endpoints.
session = HTTP(
    testnet=False
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None or value == "":
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# GET ALL ACTIVE USDT LINEAR SYMBOLS
# ============================================================

def get_active_usdt_symbols():

    symbols = []

    cursor = None

    while True:

        params = {
            "category": "linear",
            "limit": 1000
        }

        if cursor:
            params["cursor"] = cursor

        response = session.get_instruments_info(
            **params
        )

        if response.get("retCode") != 0:

            raise RuntimeError(
                f"Bybit instruments error: "
                f"{response.get('retMsg')}"
            )

        result = response.get(
            "result",
            {}
        )

        instruments = result.get(
            "list",
            []
        )

        for instrument in instruments:

            symbol = instrument.get(
                "symbol",
                ""
            )

            status = instrument.get(
                "status",
                ""
            )

            quote_coin = instrument.get(
                "quoteCoin",
                ""
            )

            if (
                status == "Trading"
                and quote_coin == "USDT"
                and symbol.endswith("USDT")
            ):

                symbols.append(
                    symbol
                )

        cursor = result.get(
            "nextPageCursor"
        )

        if not cursor:
            break

        time.sleep(0.1)

    return sorted(
        set(symbols)
    )


# ============================================================
# GET ALL LINEAR TICKERS
# ============================================================

def get_ticker_map():

    response = session.get_tickers(
        category="linear"
    )

    if response.get("retCode") != 0:

        raise RuntimeError(
            f"Bybit ticker error: "
            f"{response.get('retMsg')}"
        )

    tickers = response.get(
        "result",
        {}
    ).get(
        "list",
        []
    )

    return {
        ticker.get("symbol"): ticker
        for ticker in tickers
        if ticker.get("symbol")
    }


# ============================================================
# BUILD MARKET UNIVERSE
# ============================================================

def build_universe():

    active_symbols = (
        get_active_usdt_symbols()
    )

    logger.info(
        f"Active USDT linear symbols: "
        f"{len(active_symbols)}"
    )

    ticker_map = (
        get_ticker_map()
    )

    candidates = []


    for symbol in active_symbols:

        ticker = ticker_map.get(
            symbol
        )

        if not ticker:
            continue


        volume_24h = safe_float(
            ticker.get("volume24h")
        )

        turnover_24h = safe_float(
            ticker.get("turnover24h")
        )

        open_interest = safe_float(
            ticker.get("openInterest")
        )

        open_interest_value = safe_float(
            ticker.get("openInterestValue")
        )

        last_price = safe_float(
            ticker.get("lastPrice")
        )

        # Bybit returns e.g. 0.025 = +2.5%.
        price_change_raw = safe_float(
            ticker.get("price24hPcnt")
        )

        price_change_24h = (
            price_change_raw * 100
        )

        funding_rate = safe_float(
            ticker.get("fundingRate")
        )


        # Temporary MVP liquidity filter.
        if turnover_24h < MIN_TURNOVER_24H:
            continue


        candidates.append(
            {
                "symbol": symbol,
                "volume_24h": volume_24h,
                "turnover_24h": turnover_24h,
                "open_interest": open_interest,
                "open_interest_value": open_interest_value,
                "last_price": last_price,
                "price_change_24h": price_change_24h,
                "funding_rate": funding_rate
            }
        )


    if not candidates:

        raise RuntimeError(
            "No liquid symbols found from Bybit"
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    max_turnover = max(
        row["turnover_24h"]
        for row in candidates
    )

    max_oi_value = max(
        row["open_interest_value"]
        for row in candidates
    )


    for row in candidates:

        if max_turnover > 0:

            liquidity_score = (
                row["turnover_24h"]
                / max_turnover
                * 100
            )

        else:

            liquidity_score = 0.0


        if max_oi_value > 0:

            oi_score = (
                row["open_interest_value"]
                / max_oi_value
                * 100
            )

        else:

            oi_score = 0.0


        volatility_score = min(
            abs(
                row["price_change_24h"]
            )
            * 5,
            100
        )


        # BTC correlation will be handled later
        # by the indicator/regime engine.
        correlation_score = 0.0


        quality_score = (
            liquidity_score * 0.45
            + oi_score * 0.35
            + volatility_score * 0.20
        )


        row[
            "liquidity_score"
        ] = round(
            liquidity_score,
            6
        )

        row[
            "oi_score"
        ] = round(
            oi_score,
            6
        )

        row[
            "volatility_score"
        ] = round(
            volatility_score,
            6
        )

        row[
            "correlation_score"
        ] = correlation_score

        row[
            "quality_score"
        ] = round(
            quality_score,
            6
        )


    # ========================================================
    # RANKING
    # ========================================================

    candidates.sort(
        key=lambda row: row[
            "quality_score"
        ],
        reverse=True
    )


    now = datetime.now(
        timezone.utc
    ).isoformat()


    for index, row in enumerate(
        candidates,
        start=1
    ):

        row["rank"] = index

        if index <= TOP_N:

            row["status"] = "top30"

        else:

            row["status"] = "candidate"

        row["updated_at"] = now


    return candidates


# ============================================================
# RESET PREVIOUS TOP30
# ============================================================

def reset_previous_top30():

    response = (
        supabase
        .table("market_universe")
        .select("symbol")
        .eq("status", "top30")
        .execute()
    )

    old_symbols = (
        response.data
        or []
    )


    for row in old_symbols:

        symbol = row.get(
            "symbol"
        )

        if not symbol:
            continue

        (
            supabase
            .table("market_universe")
            .update(
                {
                    "status": "candidate",
                    "rank": None
                }
            )
            .eq(
                "symbol",
                symbol
            )
            .execute()
        )


# ============================================================
# SAVE UNIVERSE
# ============================================================

def save_universe(rows):

    if not rows:
        return

    batch_size = 100


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
            .table("market_universe")
            .upsert(
                batch,
                on_conflict="symbol"
            )
            .execute()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)

    logger.info(
        "UNIVERSE SCANNER V4 STARTED"
    )


    universe = (
        build_universe()
    )


    reset_previous_top30()


    save_universe(
        universe
    )


    top30 = (
        universe[:TOP_N]
    )


    logger.info(
        f"UNIVERSE SCANNER V4 FINISHED | "
        f"liquid_symbols={len(universe)} | "
        f"top30={len(top30)}"
    )


    print()
    print("=" * 70)

    print(
        "UNIVERSE SCANNER V4 FINISHED"
    )

    print(
        f"Liquid symbols: "
        f"{len(universe)}"
    )

    print(
        f"TOP 30: "
        f"{len(top30)}"
    )

    print("=" * 70)


    for row in top30:

        print(
            f"{row['rank']:>2}. "
            f"{row['symbol']:<16} "
            f"score="
            f"{row['quality_score']:.4f}"
        )


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
