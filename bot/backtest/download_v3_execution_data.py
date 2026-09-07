import glob
import os
import re
import sys
import time
from datetime import timezone

import pandas as pd
from pybit.unified_trading import HTTP


BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")

SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "1"
FIRST_EXEC_WINDOW = 5

OUT_1M = os.path.join(
    DATA_DIR,
    f"{SYMBOL}_1m_signal_windows_W5plus.parquet"
)

OUT_FUNDING = os.path.join(
    DATA_DIR,
    f"{SYMBOL}_funding_W5plus.parquet"
)


def find_signal_file():

    if len(sys.argv) > 1:

        candidate = sys.argv[1]

        if not os.path.isabs(candidate):
            candidate = os.path.join(BASE_DIR, candidate)

        if not os.path.exists(candidate):
            raise RuntimeError(
                f"Signal file not found: {candidate}"
            )

        return candidate

    pattern = os.path.join(
        REPORT_DIR,
        f"price_action_signal_features_{SYMBOL}_4h_*.csv"
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(
            "No price_action_signal_features CSV found."
        )

    return files[-1]


def window_number(value):

    m = re.fullmatch(
        r"W(\d+)",
        str(value)
    )

    if not m:
        return None

    return int(m.group(1))


def load_execution_entries(signal_file):

    df = pd.read_csv(signal_file)

    required = [
        "signal_timestamp",
        "window",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Signal file missing columns: {missing}"
        )

    df["signal_timestamp"] = pd.to_datetime(
        df["signal_timestamp"],
        utc=True,
        errors="coerce"
    )

    df["window_number"] = (
        df["window"]
        .apply(window_number)
    )

    df = df[
        df["window_number"]
        .notna()
    ].copy()

    df["window_number"] = (
        df["window_number"]
        .astype(int)
    )

    df = df[
        df["window_number"]
        >= FIRST_EXEC_WINDOW
    ].copy()

    # Existing 4h backtest convention:
    # signal bar timestamp -> entry at next 4h open.
    df["entry_timestamp"] = (
        df["signal_timestamp"]
        + pd.Timedelta(hours=4)
    )

    entries = (
        df["entry_timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not entries:
        raise RuntimeError(
            "No W5+ entry timestamps found."
        )

    return entries


def call_with_retry(fn, label, attempts=5, **kwargs):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            response = fn(**kwargs)

            if response.get("retCode") != 0:
                raise RuntimeError(
                    f"{label}: "
                    f"{response.get('retCode')} "
                    f"{response.get('retMsg')}"
                )

            return response

        except Exception as exc:

            last_error = exc

            if attempt == attempts:
                break

            wait = min(
                5.0,
                0.5 * attempt
            )

            print(
                f"{label}: retry {attempt}/{attempts} "
                f"after error: {exc}"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"{label} failed after {attempts} attempts: "
        f"{last_error}"
    )


def download_1m_windows(
    session,
    entries
):

    rows = []

    total = len(entries)

    print()
    print("=" * 100)
    print("DOWNLOADING 1-MINUTE EXECUTION WINDOWS")
    print("=" * 100)
    print(
        f"Signals/entry windows: {total}"
    )
    print(
        "Each request covers entry -> entry + 181 minutes."
    )

    for i, entry_ts in enumerate(
        entries,
        start=1
    ):

        end_ts = (
            entry_ts
            + pd.Timedelta(minutes=181)
        )

        response = call_with_retry(
            session.get_kline,
            label=f"1m window {i}/{total}",
            category=CATEGORY,
            symbol=SYMBOL,
            interval=INTERVAL,
            start=int(entry_ts.timestamp() * 1000),
            end=int(end_ts.timestamp() * 1000),
            limit=1000,
        )

        result_rows = (
            response
            .get("result", {})
            .get("list", [])
        )

        rows.extend(result_rows)

        if (
            i == 1
            or i % 20 == 0
            or i == total
        ):

            print(
                f"{i:>4}/{total} windows | "
                f"raw rows collected: {len(rows):,}"
            )

        time.sleep(0.05)

    if not rows:
        raise RuntimeError(
            "Bybit returned no 1m candles."
        )

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        ),
        unit="ms",
        utc=True
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]:

        df[col] = pd.to_numeric(
            df[col],
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
            ]
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df.to_parquet(
        OUT_1M,
        index=False
    )

    print()
    print(
        f"Saved 1m candles: {len(df):,}"
    )
    print(
        f"Range: {df.iloc[0]['timestamp']} "
        f"-> {df.iloc[-1]['timestamp']}"
    )
    print(
        f"File: {OUT_1M}"
    )

    return df


def download_funding(
    session,
    start_ts,
    end_ts
):

    print()
    print("=" * 100)
    print("DOWNLOADING FUNDING HISTORY")
    print("=" * 100)
    print(
        f"Range: {start_ts} -> {end_ts}"
    )

    rows = []

    cursor_end_ms = int(
        end_ts.timestamp()
        * 1000
    )

    start_ms = int(
        start_ts.timestamp()
        * 1000
    )

    page = 0

    while cursor_end_ms >= start_ms:

        page += 1

        response = call_with_retry(
            session.get_funding_rate_history,
            label=f"funding page {page}",
            category=CATEGORY,
            symbol=SYMBOL,
            endTime=cursor_end_ms,
            limit=200,
        )

        result_rows = (
            response
            .get("result", {})
            .get("list", [])
        )

        if not result_rows:
            break

        rows.extend(result_rows)

        timestamps = []

        for item in result_rows:

            try:
                timestamps.append(
                    int(
                        item[
                            "fundingRateTimestamp"
                        ]
                    )
                )
            except Exception:
                pass

        if not timestamps:
            break

        oldest = min(timestamps)

        print(
            f"page {page:>3} | "
            f"rows={len(result_rows):>3} | "
            f"oldest="
            f"{pd.to_datetime(oldest, unit='ms', utc=True)}"
        )

        if oldest <= start_ms:
            break

        new_end = oldest - 1

        if new_end >= cursor_end_ms:
            break

        cursor_end_ms = new_end

        time.sleep(0.08)

    if not rows:

        print(
            "WARNING: no funding rows returned."
        )

        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "funding_rate",
            ]
        )

        empty.to_parquet(
            OUT_FUNDING,
            index=False
        )

        return empty

    normalized = []

    for item in rows:

        normalized.append(
            {
                "timestamp":
                    pd.to_datetime(
                        int(
                            item[
                                "fundingRateTimestamp"
                            ]
                        ),
                        unit="ms",
                        utc=True
                    ),

                "funding_rate":
                    float(
                        item[
                            "fundingRate"
                        ]
                    ),
            }
        )

    df = pd.DataFrame(normalized)

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = df[
        (
            df["timestamp"]
            >= start_ts
        )
        &
        (
            df["timestamp"]
            <= end_ts
        )
    ].copy()

    df.to_parquet(
        OUT_FUNDING,
        index=False
    )

    print()
    print(
        f"Saved funding rows: {len(df):,}"
    )

    if not df.empty:

        print(
            f"Range: {df.iloc[0]['timestamp']} "
            f"-> {df.iloc[-1]['timestamp']}"
        )

    print(
        f"File: {OUT_FUNDING}"
    )

    return df


def main():

    signal_file = find_signal_file()

    print()
    print(
        f"Signal source: {signal_file}"
    )

    entries = load_execution_entries(
        signal_file
    )

    print(
        f"W5+ unique entries: {len(entries)}"
    )

    session = HTTP(
        testnet=False
    )

    one_minute = download_1m_windows(
        session,
        entries
    )

    funding_start = (
        min(entries)
        - pd.Timedelta(days=1)
    )

    funding_end = (
        max(entries)
        + pd.Timedelta(hours=4)
    )

    funding = download_funding(
        session,
        funding_start,
        funding_end
    )

    print()
    print("=" * 100)
    print("DOWNLOAD COMPLETE")
    print("=" * 100)
    print(
        f"1m candles: {len(one_minute):,}"
    )
    print(
        f"Funding rows: {len(funding):,}"
    )
    print(
        OUT_1M
    )
    print(
        OUT_FUNDING
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
