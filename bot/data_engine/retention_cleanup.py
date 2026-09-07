import os
import logging
from datetime import datetime, timezone

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

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    "retention_cleanup.log"
)


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
    "retention_cleanup"
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
# CLEANUP
# ============================================================

def run_cleanup():

    logger.info(
        "=" * 70
    )

    logger.info(
        "RETENTION CLEANUP STARTED"
    )


    response = (
        supabase
        .rpc(
            "cleanup_market_retention"
        )
        .execute()
    )


    result = (
        response.data
        or {}
    )


    logger.info(
        f"CLEANUP RESULT | "
        f"{result}"
    )


    logger.info(
        "RETENTION CLEANUP FINISHED"
    )

    logger.info(
        "=" * 70
    )


    print(
        "=" * 70
    )

    print(
        "RETENTION CLEANUP FINISHED"
    )

    print(
        f"Result: {result}"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_cleanup()

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
