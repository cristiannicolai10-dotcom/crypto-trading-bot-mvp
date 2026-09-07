import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from supabase import create_client


BASE_DIR = "/home/botadmin/crypto-bot"
DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "trend_research_3m"
)
RAW_DIR = os.path.join(
    DATA_DIR,
    "raw_5m"
)
REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

CATEGORY = "linear"
INTERVAL = "5"

# 90 days research + 30 days warmup for slow 4h indicators.
DOWNLOAD_DAYS = 120
TOP_N = 30

MIN_EXPECTED_ROWS_PER_DAY = 270


def ensure_dirs():
    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )
    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )


def load_top30_from_supabase():
    load_dotenv(
        os.path.join(
            BASE_DIR,
            ".env"
        )
    )

    url = os.getenv(
        "SUPABASE_URL"
    )
    key = os.getenv(
        "SUPABASE_KEY"
    )

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY missing from .env"
        )

    client = create_client(
        url,
        key
    )

    response = (
        client
        .table(
            "market_universe"
        )
        .select(
            "symbol,rank,status,"
            "liquidity_score,"
            "volatility_score,"
            "correlation_score,"
            "oi_score,"
            "quality_score"
        )
        .lte(
            "rank",
            TOP_N
        )
        .order(
            "rank"
        )
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "No rows returned from market_universe."
        )

    universe = pd.DataFrame(
        rows
    )

    universe["rank"] = pd.to_numeric(
        universe["rank"],
        errors="coerce"
    )

    universe = (
        universe
        .dropna(
            subset=[
                "symbol",
                "rank",
            ]
        )
        .sort_values(
            "rank"
        )
        .head(
            TOP_N
        )
        .reset_index(
            drop=True
        )
    )

    if len(universe) < TOP_N:
        print(
            f"WARNING: only {len(universe)} ranked symbols found."
        )

    return universe


def call_with_retry(
    fn,
    label,
    attempts=6,
    **kwargs
):
    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):
        try:
            response = fn(
                **kwargs
            )

            if response.get(
                "retCode"
            ) != 0:
                raise RuntimeError(
                    f"{response.get('retCode')} "
                    f"{response.get('retMsg')}"
                )

            return response

        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            wait = min(
                8.0,
                0.8 * attempt
            )

            print(
                f"{label}: retry "
                f"{attempt}/{attempts} "
                f"after {exc}"
            )

            time.sleep(
                wait
            )

    raise RuntimeError(
        f"{label} failed: {last_error}"
    )


def download_symbol_5m(
    session,
    symbol,
    start_ms,
    end_ms
):
    rows = []
    cursor_end = end_ms
    pages = 0

    while cursor_end > start_ms:
        pages += 1

        response = call_with_retry(
            session.get_kline,
            label=f"{symbol} page {pages}",
            category=CATEGORY,
            symbol=symbol,
            interval=INTERVAL,
            start=start_ms,
            end=cursor_end,
            limit=1000,
        )

        page = (
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

        if not page:
            break

        rows.extend(
            page
        )

        timestamps = []

        for item in page:
            try:
                timestamps.append(
                    int(
                        item[0]
                    )
                )
            except Exception:
                pass

        if not timestamps:
            break

        oldest = min(
            timestamps
        )

        if oldest <= start_ms:
            break

        new_end = oldest - 1

        if new_end >= cursor_end:
            break

        cursor_end = new_end

        if pages % 10 == 0:
            print(
                f"  {symbol}: "
                f"{pages} pages, "
                f"{len(rows):,} raw rows"
            )

        time.sleep(
            0.05
        )

    if not rows:
        raise RuntimeError(
            f"{symbol}: no klines returned."
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

    for col in columns[1:]:
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
            subset=[
                "timestamp"
            ],
            keep="last"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def main():
    ensure_dirs()

    universe = (
        load_top30_from_supabase()
    )

    now = pd.Timestamp.now(
        tz="UTC"
    ).floor(
        "5min"
    )

    start = (
        now
        - pd.Timedelta(
            days=DOWNLOAD_DAYS
        )
    )

    start_ms = int(
        start.timestamp()
        * 1000
    )

    end_ms = int(
        now.timestamp()
        * 1000
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    universe_file = os.path.join(
        REPORT_DIR,
        f"trend_research_universe_top30_{stamp}.csv"
    )

    universe[
        "research_download_start"
    ] = start

    universe[
        "research_download_end"
    ] = now

    universe.to_csv(
        universe_file,
        index=False
    )

    print()
    print("=" * 110)
    print("TREND RESEARCH 3M - DATA DOWNLOAD")
    print("=" * 110)
    print(
        f"Frozen universe: {len(universe)} symbols"
    )
    print(
        f"Download range: {start} -> {now}"
    )
    print(
        f"Interval: 5m"
    )
    print(
        f"Warmup + research: {DOWNLOAD_DAYS} days"
    )
    print(
        f"Universe file: {universe_file}"
    )
    print()
    print(
        ", ".join(
            universe[
                "symbol"
            ].tolist()
        )
    )
    print("=" * 110)

    session = HTTP(
        testnet=False
    )

    summary = []

    for idx, row in universe.iterrows():
        symbol = str(
            row[
                "symbol"
            ]
        )

        print()
        print(
            f"[{idx + 1:02d}/{len(universe):02d}] "
            f"{symbol}"
        )

        try:
            df = download_symbol_5m(
                session=session,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms
            )

            path = os.path.join(
                RAW_DIR,
                f"{symbol}_5m_120d.parquet"
            )

            df.to_parquet(
                path,
                index=False
            )

            expected_min = (
                DOWNLOAD_DAYS
                * MIN_EXPECTED_ROWS_PER_DAY
            )

            coverage_ok = (
                len(df)
                >= expected_min
            )

            summary.append(
                {
                    "symbol":
                        symbol,
                    "rank":
                        row[
                            "rank"
                        ],
                    "rows":
                        len(df),
                    "start":
                        df.iloc[0][
                            "timestamp"
                        ],
                    "end":
                        df.iloc[-1][
                            "timestamp"
                        ],
                    "coverage_ok":
                        coverage_ok,
                    "file":
                        path,
                    "status":
                        "OK",
                }
            )

            print(
                f"  rows={len(df):,} | "
                f"{df.iloc[0]['timestamp']} -> "
                f"{df.iloc[-1]['timestamp']} | "
                f"coverage_ok={coverage_ok}"
            )

        except Exception as exc:
            summary.append(
                {
                    "symbol":
                        symbol,
                    "rank":
                        row[
                            "rank"
                        ],
                    "rows":
                        0,
                    "start":
                        None,
                    "end":
                        None,
                    "coverage_ok":
                        False,
                    "file":
                        None,
                    "status":
                        f"ERROR: {exc}",
                }
            )

            print(
                f"  ERROR: {exc}"
            )

    summary_df = pd.DataFrame(
        summary
    )

    summary_file = os.path.join(
        REPORT_DIR,
        f"trend_research_download_summary_{stamp}.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    ok_count = int(
        (
            summary_df[
                "status"
            ]
            == "OK"
        ).sum()
    )

    print()
    print("=" * 110)
    print("DOWNLOAD FINISHED")
    print("=" * 110)
    print(
        f"Successful: {ok_count}/{len(summary_df)}"
    )
    print(
        f"Summary: {summary_file}"
    )
    print("=" * 110)

    if ok_count < 20:
        raise RuntimeError(
            "Too few symbols downloaded for reliable "
            "cross-sectional research."
        )


if __name__ == "__main__":
    main()
