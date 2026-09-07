import os
import json
import time
from datetime import datetime, timezone

import pandas as pd

from dateutil.relativedelta import relativedelta
from pybit.unified_trading import HTTP


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "historical"
)


SYMBOL = "BTCUSDT"

YEARS = 3


TIMEFRAMES = {

    "1h": {
        "bybit_interval": "60",
        "seconds": 3600
    },

    "4h": {
        "bybit_interval": "240",
        "seconds": 14400
    }
}


BYBIT_LIMIT = 1000

REQUEST_DELAY = 0.15

MAX_RETRIES = 5

RETRY_DELAY = 5


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# Public MAINNET data
session = HTTP(
    testnet=False
)


# ============================================================
# DOWNLOAD PAGE
# ============================================================

def get_kline_page(
    symbol,
    interval,
    end_ms
):

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                end=end_ms,
                limit=BYBIT_LIMIT
            )


            ret_code = response.get(
                "retCode",
                -1
            )


            if ret_code != 0:

                raise RuntimeError(
                    f"Bybit error | "
                    f"retCode={ret_code} | "
                    f"{response.get('retMsg')}"
                )


            return (
                response
                .get(
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


            print(
                f"Retry "
                f"{attempt}/"
                f"{MAX_RETRIES} | "
                f"{error}"
            )


            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )


    raise RuntimeError(
        f"Failed after "
        f"{MAX_RETRIES} retries | "
        f"{last_error}"
    )


# ============================================================
# DOWNLOAD TIMEFRAME
# ============================================================

def download_timeframe(
    symbol,
    timeframe_name,
    interval,
    start_time,
    end_time
):

    print()
    print(
        "=" * 70
    )

    print(
        f"DOWNLOAD START | "
        f"{symbol} "
        f"{timeframe_name}"
    )

    print(
        f"From: "
        f"{start_time.isoformat()}"
    )

    print(
        f"To:   "
        f"{end_time.isoformat()}"
    )

    print(
        "=" * 70
    )


    start_ms = int(
        start_time.timestamp()
        * 1000
    )


    current_end_ms = int(
        end_time.timestamp()
        * 1000
    )


    raw_rows = []

    page_number = 0


    while (
        current_end_ms
        > start_ms
    ):

        candles = get_kline_page(
            symbol=symbol,
            interval=interval,
            end_ms=current_end_ms
        )


        if not candles:

            print(
                "No more data."
            )

            break


        page_number += 1

        oldest_timestamp = None

        added_this_page = 0


        for candle in candles:

            candle_timestamp = int(
                candle[0]
            )


            if (
                oldest_timestamp is None
                or candle_timestamp
                < oldest_timestamp
            ):

                oldest_timestamp = (
                    candle_timestamp
                )


            if (
                candle_timestamp
                < start_ms
            ):

                continue


            raw_rows.append(
                {
                    "timestamp_ms":
                        candle_timestamp,

                    "open":
                        candle[1],

                    "high":
                        candle[2],

                    "low":
                        candle[3],

                    "close":
                        candle[4],

                    "volume":
                        candle[5],

                    "turnover":
                        (
                            candle[6]
                            if len(candle) > 6
                            else None
                        )
                }
            )


            added_this_page += 1


        if oldest_timestamp is None:

            break


        reached = datetime.fromtimestamp(
            oldest_timestamp / 1000,
            tz=timezone.utc
        )


        print(
            f"Page {page_number:<3} | "
            f"new={added_this_page:<4} | "
            f"total={len(raw_rows):,} | "
            f"reached={reached.isoformat()}"
        )


        if (
            oldest_timestamp
            <= start_ms
        ):

            break


        next_end_ms = (
            oldest_timestamp
            - 1
        )


        if (
            next_end_ms
            >= current_end_ms
        ):

            raise RuntimeError(
                "Pagination stuck"
            )


        current_end_ms = (
            next_end_ms
        )


        time.sleep(
            REQUEST_DELAY
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    if not raw_rows:

        raise RuntimeError(
            f"No historical data downloaded for "
            f"{symbol} {timeframe_name}"
        )


    df = pd.DataFrame(
        raw_rows
    )


    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp_ms"],
            unit="ms",
            utc=True
        )
    )


    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
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
        .drop(
            columns=[
                "timestamp_ms"
            ]
        )
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
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df,
    expected_seconds
):

    duplicate_count = int(
        df[
            "timestamp"
        ]
        .duplicated()
        .sum()
    )


    differences = (
        df[
            "timestamp"
        ]
        .diff()
        .dt
        .total_seconds()
    )


    gaps = (
        differences
        > expected_seconds * 1.5
    )


    gap_count = int(
        gaps.sum()
    )


    return {

        "rows":
            int(
                len(df)
            ),

        "duplicates":
            duplicate_count,

        "gaps":
            gap_count,

        "first_timestamp":
            (
                df.iloc[0][
                    "timestamp"
                ]
                .isoformat()
            ),

        "last_timestamp":
            (
                df.iloc[-1][
                    "timestamp"
                ]
                .isoformat()
            )
    }


# ============================================================
# SAVE PARQUET
# ============================================================

def save_dataset(
    symbol,
    timeframe,
    df
):

    filename = (
        f"{symbol}_"
        f"{timeframe}_"
        f"{YEARS}y.parquet"
    )


    filepath = os.path.join(
        DATA_DIR,
        filename
    )


    df.to_parquet(
        filepath,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )


    return filepath


# ============================================================
# MAIN
# ============================================================

def main():

    now = datetime.now(
        timezone.utc
    )


    start_time = (
        now
        - relativedelta(
            years=YEARS
        )
    )


    metadata = {

        "symbol":
            SYMBOL,

        "years":
            YEARS,

        "downloaded_at":
            now.isoformat(),

        "target_start":
            start_time.isoformat(),

        "target_end":
            now.isoformat(),

        "datasets":
            {}
    }


    print()
    print(
        "=" * 70
    )

    print(
        "LOCAL HISTORICAL DATA DOWNLOAD"
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Period: {YEARS} years"
    )

    print(
        "Storage: Hetzner / Parquet"
    )

    print(
        "Supabase: NOT USED"
    )

    print(
        "=" * 70
    )


    for (
        timeframe_name,
        settings
    ) in TIMEFRAMES.items():


        df = download_timeframe(
            symbol=SYMBOL,
            timeframe_name=timeframe_name,
            interval=settings[
                "bybit_interval"
            ],
            start_time=start_time,
            end_time=now
        )


        validation = (
            validate_dataset(
                df=df,
                expected_seconds=settings[
                    "seconds"
                ]
            )
        )


        filepath = save_dataset(
            symbol=SYMBOL,
            timeframe=timeframe_name,
            df=df
        )


        metadata[
            "datasets"
        ][
            timeframe_name
        ] = {

            **validation,

            "file":
                filepath
        }


        print()
        print(
            f"SAVED | "
            f"{SYMBOL} "
            f"{timeframe_name}"
        )

        print(
            f"Rows: "
            f"{validation['rows']:,}"
        )

        print(
            f"From: "
            f"{validation['first_timestamp']}"
        )

        print(
            f"To: "
            f"{validation['last_timestamp']}"
        )

        print(
            f"Duplicates: "
            f"{validation['duplicates']}"
        )

        print(
            f"Gaps: "
            f"{validation['gaps']}"
        )

        print(
            f"File: "
            f"{filepath}"
        )


    metadata_file = os.path.join(
        DATA_DIR,
        f"{SYMBOL}_"
        f"{YEARS}y_metadata.json"
    )


    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


    print()
    print(
        "=" * 70
    )

    print(
        "DOWNLOAD FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        f"Metadata: "
        f"{metadata_file}"
    )


if __name__ == "__main__":

    main()
