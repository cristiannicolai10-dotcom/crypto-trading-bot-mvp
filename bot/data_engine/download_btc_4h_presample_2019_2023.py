import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
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

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
BYBIT_INTERVAL = "240"

# Request a deliberately untouched pre-sample period.
# The validation dataset used for strategy development starts at:
# 2023-09-07 16:00 UTC
REQUEST_START = pd.Timestamp(
    "2019-01-01T00:00:00Z"
)

END_EXCLUSIVE = pd.Timestamp(
    "2023-09-07T16:00:00Z"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

METADATA_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023_metadata.json"
)

PAGE_LIMIT = 1000
REQUEST_PAUSE_SECONDS = 0.08
MAX_RETRIES = 5


# ============================================================
# HELPERS
# ============================================================

def to_ms(ts):
    return int(ts.timestamp() * 1000)


def request_page(session, end_ms):

    start_ms = to_ms(REQUEST_START)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get_kline(
                category="linear",
                symbol=SYMBOL,
                interval=BYBIT_INTERVAL,
                start=start_ms,
                end=end_ms,
                limit=PAGE_LIMIT
            )

            ret_code = response.get("retCode")

            if ret_code != 0:
                raise RuntimeError(
                    f"Bybit retCode={ret_code}: "
                    f"{response.get('retMsg')}"
                )

            return (
                response
                .get("result", {})
                .get("list", [])
            )

        except Exception as exc:

            if attempt == MAX_RETRIES:
                raise

            sleep_seconds = min(
                2 ** attempt,
                15
            )

            print(
                f"Request error "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{exc}"
            )

            print(
                f"Retrying in {sleep_seconds}s..."
            )

            time.sleep(
                sleep_seconds
            )


# ============================================================
# DOWNLOAD
# ============================================================

def download_history():

    session = HTTP(
        testnet=False
    )

    rows = []

    # end is inclusive in practice; subtract 1 ms so there is
    # zero overlap with the development dataset.
    cursor_end_ms = (
        to_ms(END_EXCLUSIVE)
        - 1
    )

    requested_start_ms = to_ms(
        REQUEST_START
    )

    page_number = 0

    print()
    print("=" * 90)
    print("BTCUSDT 4H PRE-SAMPLE HISTORY DOWNLOAD")
    print("=" * 90)
    print(f"Requested start: {REQUEST_START}")
    print(f"End exclusive:   {END_EXCLUSIVE}")
    print("Source: Bybit mainnet public linear market data")
    print("=" * 90)

    while cursor_end_ms >= requested_start_ms:

        page_number += 1

        page = request_page(
            session,
            cursor_end_ms
        )

        if not page:

            print(
                "No more candles returned by Bybit."
            )
            break

        page_timestamps = []

        for item in page:

            if len(item) < 6:
                continue

            timestamp_ms = int(
                item[0]
            )

            timestamp = pd.to_datetime(
                timestamp_ms,
                unit="ms",
                utc=True
            )

            if timestamp < REQUEST_START:
                continue

            if timestamp >= END_EXCLUSIVE:
                continue

            page_timestamps.append(
                timestamp_ms
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )

        if not page_timestamps:

            print(
                "Page contained no candles inside requested range."
            )
            break

        oldest_ms = min(
            page_timestamps
        )

        newest_ms = max(
            page_timestamps
        )

        oldest = pd.to_datetime(
            oldest_ms,
            unit="ms",
            utc=True
        )

        newest = pd.to_datetime(
            newest_ms,
            unit="ms",
            utc=True
        )

        print(
            f"Page {page_number:>3}: "
            f"{len(page_timestamps):>4} candles | "
            f"{oldest} -> {newest}"
        )

        if oldest_ms <= requested_start_ms:
            break

        # Move cursor strictly before the oldest candle
        # from the current page.
        next_cursor = (
            oldest_ms
            - 1
        )

        if next_cursor >= cursor_end_ms:
            raise RuntimeError(
                "Pagination cursor did not move backward."
            )

        cursor_end_ms = next_cursor

        time.sleep(
            REQUEST_PAUSE_SECONDS
        )

    if not rows:
        raise RuntimeError(
            "No candles were downloaded."
        )

    df = pd.DataFrame(
        rows
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
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
# VALIDATION
# ============================================================

def validate_history(df):

    expected_step = pd.Timedelta(
        hours=4
    )

    diffs = (
        df["timestamp"]
        .diff()
        .dropna()
    )

    duplicate_count = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    gaps = diffs[
        diffs > expected_step
    ]

    irregular_steps = diffs[
        diffs != expected_step
    ]

    first_ts = df.iloc[0][
        "timestamp"
    ]

    last_ts = df.iloc[-1][
        "timestamp"
    ]

    overlap_count = int(
        (
            df["timestamp"]
            >= END_EXCLUSIVE
        )
        .sum()
    )

    metadata = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "source": "Bybit mainnet public linear",
        "requested_start": REQUEST_START.isoformat(),
        "end_exclusive": END_EXCLUSIVE.isoformat(),
        "actual_start": first_ts.isoformat(),
        "actual_end": last_ts.isoformat(),
        "rows": int(len(df)),
        "duplicates": duplicate_count,
        "irregular_4h_steps": int(len(irregular_steps)),
        "gaps_over_4h": int(len(gaps)),
        "overlap_with_development_period": overlap_count,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    print()
    print("=" * 90)
    print("DOWNLOAD VALIDATION")
    print("=" * 90)
    print(f"Rows:        {len(df):,}")
    print(f"Actual from: {first_ts}")
    print(f"Actual to:   {last_ts}")
    print(f"Duplicates:  {duplicate_count}")
    print(f"Irregular 4h steps: {len(irregular_steps)}")
    print(f"Gaps > 4h:          {len(gaps)}")
    print(f"Overlap >= {END_EXCLUSIVE}: {overlap_count}")

    if first_ts > REQUEST_START:
        print()
        print(
            "NOTE: Bybit returned no earlier BTCUSDT "
            "linear data before the actual start shown above."
        )

    if len(gaps) > 0:

        print()
        print("Largest gaps:")

        gap_table = pd.DataFrame(
            {
                "timestamp": df.loc[
                    gaps.index,
                    "timestamp"
                ],
                "gap": gaps.values,
            }
        )

        print(
            gap_table
            .sort_values(
                "gap",
                ascending=False
            )
            .head(10)
            .to_string(
                index=False
            )
        )

    if overlap_count != 0:
        raise RuntimeError(
            "Downloaded file overlaps with development dataset."
        )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate timestamps remain after cleanup."
        )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_history()

    metadata = validate_history(
        df
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2
        )

    print()
    print("=" * 90)
    print("DOWNLOAD FINISHED")
    print("=" * 90)
    print(f"Parquet:  {OUTPUT_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
