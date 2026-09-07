import gc
import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
DATA_DIR = os.path.join(BASE_DIR, "data", "trend_research_3m")
RAW_DIR = os.path.join(DATA_DIR, "raw_5m")
FEATURE_DIR = os.path.join(DATA_DIR, "features_90d_lowmem")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ============================================================
# FROZEN MARKET TRIGGER A
# ============================================================

BTC_12H_RETURN_MAX = 0.803982
BTC_5M_ATR_MIN = 0.206836
BTC_5M_MARKET_REGIME_MIN = 0.5
BTC_5M_PRICE_VS_EMA200_MIN = 1.2227

EVENT_COOLDOWN_MINUTES = 60
TREND_ATR_THRESHOLD = 0.50

HORIZONS_MINUTES = [5, 15, 30, 45, 60]

# Cross-sectional coin-selection analysis.
MIN_COINS_PER_EVENT_FOR_IC = 10
QUINTILES = 5

# Only coin-specific features are used for "what to trade".
# BTC context is intentionally excluded here because it is common
# to all coins at the same event.
COIN_FEATURE_PREFIXES = (
    "5m_",
    "1h_",
    "4h_",
    "rank_",
    "relative_strength_",
    "btc_correlation_",
)


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def latest_file(pattern, required=True):
    files = sorted(glob.glob(pattern))
    if not files:
        if required:
            raise RuntimeError(f"No files found for pattern: {pattern}")
        return None
    return files[-1]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


# ============================================================
# UNIVERSE
# ============================================================

def load_universe_sets():
    summary_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "trend_research_download_summary_*.csv",
        )
    )

    summary = pd.read_csv(summary_file)

    required = {"symbol", "coverage_ok", "status"}
    missing = required - set(summary.columns)
    if missing:
        raise RuntimeError(
            f"Download summary missing columns: {sorted(missing)}"
        )

    summary["coverage_ok"] = summary["coverage_ok"].apply(parse_bool)

    all_symbols = (
        summary.loc[
            summary["status"].astype(str).str.startswith("OK"),
            "symbol",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    full_symbols = (
        summary.loc[
            summary["coverage_ok"],
            "symbol",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    return summary_file, all_symbols, full_symbols


# ============================================================
# FROZEN BTC EVENT CLOCK
# ============================================================

def load_btc_feature_file():
    path = os.path.join(
        FEATURE_DIR,
        "BTCUSDT_features.parquet",
    )

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing BTC feature file: {path}\n"
            "Run the low-memory trend analyzer first."
        )

    return path


def build_trigger_events(btc_feature_file):
    cols = [
        "available_at",
        "btc_4h_ret_3bar_pct",
        "btc_5m_atr_pct",
        "btc_5m_market_regime",
        "btc_5m_price_vs_ema200_pct",
    ]

    btc = pd.read_parquet(
        btc_feature_file,
        columns=cols,
    )

    btc["available_at"] = pd.to_datetime(
        btc["available_at"],
        utc=True,
        errors="coerce",
    )

    btc = (
        btc
        .dropna(subset=["available_at"])
        .sort_values("available_at")
        .drop_duplicates("available_at", keep="last")
        .reset_index(drop=True)
    )

    btc["trigger_true"] = (
        (btc["btc_4h_ret_3bar_pct"] <= BTC_12H_RETURN_MAX)
        & (btc["btc_5m_atr_pct"] > BTC_5M_ATR_MIN)
        & (btc["btc_5m_market_regime"] > BTC_5M_MARKET_REGIME_MIN)
        & (
            btc["btc_5m_price_vs_ema200_pct"]
            > BTC_5M_PRICE_VS_EMA200_MIN
        )
    )

    btc["new_episode"] = (
        btc["trigger_true"]
        & ~btc["trigger_true"].shift(1, fill_value=False)
    )

    candidates = btc.loc[
        btc["trigger_true"]
    ].copy()

    accepted_rows = []
    last_accepted = None

    cooldown = pd.Timedelta(
        minutes=EVENT_COOLDOWN_MINUTES
    )

    for _, row in candidates.iterrows():
        ts = row["available_at"]

        if (
            last_accepted is None
            or ts >= last_accepted + cooldown
        ):
            accepted_rows.append(row)
            last_accepted = ts

    events = pd.DataFrame(accepted_rows)

    if events.empty:
        raise RuntimeError(
            "Frozen Trigger A produced zero events."
        )

    research_start = btc["available_at"].min()
    research_end = btc["available_at"].max()
    train_end = research_start + pd.Timedelta(days=60)

    events["split"] = np.where(
        events["available_at"] < train_end,
        "DISCOVERY_60D",
        "VALIDATION_30D",
    )

    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(
        1,
        len(events) + 1,
        dtype=int,
    )

    events["minutes_since_previous_event"] = (
        events["available_at"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    events = events[
        [
            "event_id",
            "available_at",
            "split",
            "new_episode",
            "minutes_since_previous_event",
            "btc_4h_ret_3bar_pct",
            "btc_5m_atr_pct",
            "btc_5m_market_regime",
            "btc_5m_price_vs_ema200_pct",
        ]
    ].copy()

    meta = {
        "research_start": research_start,
        "train_end": train_end,
        "research_end": research_end,
        "raw_trigger_bars": int(candidates.shape[0]),
        "independent_events": int(events.shape[0]),
        "discovery_events": int(
            (events["split"] == "DISCOVERY_60D").sum()
        ),
        "validation_events": int(
            (events["split"] == "VALIDATION_30D").sum()
        ),
        "new_episode_events": int(events["new_episode"].sum()),
    }

    del btc, candidates
    gc.collect()

    return events, meta


# ============================================================
# RAW OUTCOMES
# ============================================================

def load_raw_symbol(symbol):
    path = os.path.join(
        RAW_DIR,
        f"{symbol}_5m_120d.parquet",
    )

    if not os.path.exists(path):
        return None

    df = pd.read_parquet(
        path,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return (
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
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def future_metrics_for_event(raw_indexed, event_time):
    if event_time not in raw_indexed.index:
        return None

    entry_row = raw_indexed.loc[event_time]

    # If duplicate timestamps somehow survived, take first.
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[0]

    entry_price = float(entry_row["open"])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    result = {
        "entry_price": entry_price,
    }

    for horizon in HORIZONS_MINUTES:
        end_time = event_time + pd.Timedelta(minutes=horizon)

        bars = raw_indexed.loc[
            (raw_indexed.index >= event_time)
            & (raw_indexed.index < end_time)
        ]

        expected_bars = horizon // 5

        if len(bars) < expected_bars:
            result[f"return_{horizon}m_pct"] = np.nan
            continue

        final_close = float(bars.iloc[-1]["close"])

        result[f"return_{horizon}m_pct"] = (
            final_close / entry_price - 1.0
        ) * 100.0

    bars_60 = raw_indexed.loc[
        (raw_indexed.index >= event_time)
        & (
            raw_indexed.index
            < event_time + pd.Timedelta(minutes=60)
        )
    ]

    if len(bars_60) >= 12:
        future_high = float(bars_60["high"].max())
        future_low = float(bars_60["low"].min())

        result["mfe_60m_pct"] = (
            future_high / entry_price - 1.0
        ) * 100.0
        result["mae_60m_pct"] = (
            future_low / entry_price - 1.0
        ) * 100.0
    else:
        result["mfe_60m_pct"] = np.nan
        result["mae_60m_pct"] = np.nan

    return result


# ============================================================
# EVENT FEATURE ROWS
# ============================================================

def candidate_coin_feature_columns(columns):
    excluded_exact = {
        "available_at",
        "symbol",
        "trend_class",
        "target_up",
        "target_down",
        "future_1h_return_pct",
        "future_1h_mfe_pct",
        "future_1h_mae_pct",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
    }

    out = []

    for col in columns:
        if col in excluded_exact:
            continue

        if col.startswith("btc_") and not col.startswith(
            "btc_correlation_"
        ):
            # Common BTC market context belongs to "when",
            # not the cross-sectional coin selector.
            continue

        if col.startswith(COIN_FEATURE_PREFIXES):
            out.append(col)

    return sorted(set(out))


def load_symbol_features_at_events(
    symbol,
    event_times,
):
    path = os.path.join(
        FEATURE_DIR,
        f"{symbol}_features.parquet",
    )

    if not os.path.exists(path):
        return None, []

    # Read schema first using pandas metadata through a zero-row-like
    # normal read is not available, so read the file once. These symbol
    # files are compact and are released immediately.
    df = pd.read_parquet(path)

    df["available_at"] = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce",
    )

    feature_cols = candidate_coin_feature_columns(
        df.columns
    )

    keep = [
        "available_at",
        "1h_atr_pct",
    ] + feature_cols

    keep = list(dict.fromkeys(
        [c for c in keep if c in df.columns]
    ))

    df = df[keep].copy()

    subset = df.loc[
        df["available_at"].isin(event_times)
    ].copy()

    subset["symbol"] = symbol

    del df
    gc.collect()

    return subset, feature_cols


# ============================================================
# COIN-EVENT PANEL
# ============================================================

def build_coin_event_panel(
    events,
    all_symbols,
    full_symbols,
):
    event_times = set(
        events["available_at"].tolist()
    )

    event_lookup = events.set_index(
        "available_at"
    )[
        ["event_id", "split"]
    ]

    panel_parts = []
    all_feature_names = set()

    total = len(all_symbols)

    for idx, symbol in enumerate(
        all_symbols,
        start=1,
    ):
        print(
            f"[{idx:02d}/{total:02d}] "
            f"{symbol}: event outcomes + features"
        )

        raw = load_raw_symbol(symbol)

        if raw is None or raw.empty:
            print("  raw file missing/empty -> skipped")
            continue

        feature_rows, feature_cols = (
            load_symbol_features_at_events(
                symbol,
                event_times,
            )
        )

        all_feature_names.update(feature_cols)

        if feature_rows is None or feature_rows.empty:
            print("  no event feature rows -> skipped")
            del raw
            gc.collect()
            continue

        raw_indexed = raw.set_index("timestamp")

        outcomes = []

        for event_time in feature_rows["available_at"]:
            metrics = future_metrics_for_event(
                raw_indexed,
                event_time,
            )

            if metrics is None:
                continue

            row = {
                "available_at": event_time,
            }
            row.update(metrics)
            outcomes.append(row)

        outcome_df = pd.DataFrame(outcomes)

        if outcome_df.empty:
            print("  no aligned future outcomes -> skipped")
            del raw, raw_indexed, feature_rows
            gc.collect()
            continue

        merged = feature_rows.merge(
            outcome_df,
            on="available_at",
            how="inner",
            validate="one_to_one",
        )

        merged = merged.merge(
            event_lookup,
            left_on="available_at",
            right_index=True,
            how="left",
            validate="many_to_one",
        )

        merged["full_history_universe"] = (
            symbol in full_symbols
        )

        # ATR-normalized 60m outcome.
        merged["return_60m_atr"] = (
            merged["return_60m_pct"]
            / merged["1h_atr_pct"]
        )

        merged["mfe_60m_atr"] = (
            merged["mfe_60m_pct"]
            / merged["1h_atr_pct"]
        )

        merged["mae_60m_atr"] = (
            merged["mae_60m_pct"]
            / merged["1h_atr_pct"]
        )

        threshold_pct = (
            merged["1h_atr_pct"]
            * TREND_ATR_THRESHOLD
        )

        merged["trend_up_60m"] = (
            merged["return_60m_pct"]
            >= threshold_pct
        ).astype(np.int8)

        merged["trend_down_60m"] = (
            merged["return_60m_pct"]
            <= -threshold_pct
        ).astype(np.int8)

        merged["positive_60m"] = (
            merged["return_60m_pct"] > 0
        ).astype(np.int8)

        panel_parts.append(merged)

        print(
            f"  matched events={len(merged)}"
        )

        del (
            raw,
            raw_indexed,
            feature_rows,
            outcome_df,
            merged,
        )
        gc.collect()

    if not panel_parts:
        raise RuntimeError(
            "No coin-event observations were built."
        )

    panel = pd.concat(
        panel_parts,
        ignore_index=True,
    )

    del panel_parts
    gc.collect()

    return panel, sorted(all_feature_names)


# ============================================================
# EVENT-LEVEL MARKET SUMMARY
# ============================================================

def event_market_summary(panel):
    rows = []

    universes = [
        ("ALL_AVAILABLE", panel),
        (
            "FULL_HISTORY_ONLY",
            panel.loc[
                panel["full_history_universe"]
            ],
        ),
    ]

    for universe_name, frame in universes:
        for (
            event_id,
            split,
            event_time,
        ), group in frame.groupby(
            [
                "event_id",
                "split",
                "available_at",
            ],
            observed=True,
        ):
            row = {
                "universe": universe_name,
                "event_id": int(event_id),
                "split": split,
                "available_at": event_time,
                "coins_available": int(
                    group["symbol"].nunique()
                ),
                "trend_up_breadth":
                    float(group["trend_up_60m"].mean()),
                "trend_down_breadth":
                    float(group["trend_down_60m"].mean()),
                "positive_breadth":
                    float(group["positive_60m"].mean()),
                "mean_return_60m_pct":
                    float(group["return_60m_pct"].mean()),
                "median_return_60m_pct":
                    float(group["return_60m_pct"].median()),
                "mean_return_60m_atr":
                    float(group["return_60m_atr"].mean()),
                "median_return_60m_atr":
                    float(group["return_60m_atr"].median()),
                "mean_mfe_60m_atr":
                    float(group["mfe_60m_atr"].mean()),
                "mean_mae_60m_atr":
                    float(group["mae_60m_atr"].mean()),
            }

            for horizon in HORIZONS_MINUTES:
                col = f"return_{horizon}m_pct"
                row[f"mean_return_{horizon}m_pct"] = (
                    float(group[col].mean())
                )
                row[f"median_return_{horizon}m_pct"] = (
                    float(group[col].median())
                )
                row[f"positive_breadth_{horizon}m"] = (
                    float((group[col] > 0).mean())
                )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_independent_events(event_summary):
    rows = []

    for (
        universe,
        split,
    ), group in event_summary.groupby(
        ["universe", "split"],
        observed=True,
    ):
        n = len(group)

        def mean_ci(series):
            s = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            if len(s) == 0:
                return (
                    np.nan,
                    np.nan,
                    np.nan,
                )

            mean = float(s.mean())

            if len(s) < 2:
                return mean, np.nan, np.nan

            se = float(
                s.std(ddof=1)
                / math.sqrt(len(s))
            )

            return (
                mean,
                mean - 1.96 * se,
                mean + 1.96 * se,
            )

        up_mean, up_lo, up_hi = mean_ci(
            group["trend_up_breadth"]
        )

        pos_mean, pos_lo, pos_hi = mean_ci(
            group["positive_breadth"]
        )

        row = {
            "universe": universe,
            "split": split,
            "independent_events": n,
            "avg_coins_available":
                float(group["coins_available"].mean()),
            "mean_trend_up_breadth":
                up_mean,
            "trend_up_breadth_ci95_low":
                up_lo,
            "trend_up_breadth_ci95_high":
                up_hi,
            "mean_positive_breadth":
                pos_mean,
            "positive_breadth_ci95_low":
                pos_lo,
            "positive_breadth_ci95_high":
                pos_hi,
            "mean_return_60m_pct":
                float(group["mean_return_60m_pct"].mean()),
            "median_event_return_60m_pct":
                float(group["median_return_60m_pct"].median()),
            "mean_return_60m_atr":
                float(group["mean_return_60m_atr"].mean()),
            "mean_mfe_60m_atr":
                float(group["mean_mfe_60m_atr"].mean()),
            "mean_mae_60m_atr":
                float(group["mean_mae_60m_atr"].mean()),
        }

        for horizon in HORIZONS_MINUTES:
            row[f"mean_return_{horizon}m_pct"] = float(
                group[f"mean_return_{horizon}m_pct"].mean()
            )
            row[f"mean_positive_breadth_{horizon}m"] = float(
                group[f"positive_breadth_{horizon}m"].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# COIN SELECTION: CROSS-SECTIONAL IC
# ============================================================

def spearman_corr(x, y):
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(pair) < MIN_COINS_PER_EVENT_FOR_IC:
        return np.nan

    if pair["x"].nunique() < 3:
        return np.nan

    return float(
        pair["x"].rank(pct=True)
        .corr(
            pair["y"].rank(pct=True)
        )
    )


def feature_ic_analysis(
    panel,
    feature_names,
):
    rows = []

    for idx, feature in enumerate(
        feature_names,
        start=1,
    ):
        if feature not in panel.columns:
            continue

        if idx % 10 == 0:
            print(
                f"  IC feature {idx}/{len(feature_names)}"
            )

        for split in [
            "DISCOVERY_60D",
            "VALIDATION_30D",
        ]:
            subset = panel.loc[
                panel["split"] == split
            ]

            ics = []

            for _, group in subset.groupby(
                "event_id",
                observed=True,
            ):
                ic = spearman_corr(
                    group[feature],
                    group["return_60m_atr"],
                )

                if np.isfinite(ic):
                    ics.append(ic)

            if not ics:
                continue

            arr = np.asarray(
                ics,
                dtype=float,
            )

            rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "events_used": len(arr),
                    "mean_ic": float(arr.mean()),
                    "median_ic": float(np.median(arr)),
                    "mean_abs_ic": float(np.abs(arr).mean()),
                    "positive_ic_rate":
                        float((arr > 0).mean()),
                }
            )

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        return long_df, pd.DataFrame()

    pivot = long_df.pivot(
        index="feature",
        columns="split",
        values=[
            "events_used",
            "mean_ic",
            "median_ic",
            "mean_abs_ic",
            "positive_ic_rate",
        ],
    )

    pivot.columns = [
        f"{metric}_{split}"
        for metric, split in pivot.columns
    ]

    summary = pivot.reset_index()

    train_col = "mean_ic_DISCOVERY_60D"
    val_col = "mean_ic_VALIDATION_30D"

    if train_col in summary.columns and val_col in summary.columns:
        summary["same_ic_sign"] = (
            np.sign(summary[train_col])
            == np.sign(summary[val_col])
        )

        summary["discovery_direction"] = np.where(
            summary[train_col] >= 0,
            "HIGHER_BETTER",
            "LOWER_BETTER",
        )

        summary["validation_ic_in_discovery_direction"] = (
            np.where(
                summary[train_col] >= 0,
                summary[val_col],
                -summary[val_col],
            )
        )

        summary["discovery_strength"] = (
            summary[train_col].abs()
        )

        summary = summary.sort_values(
            [
                "same_ic_sign",
                "discovery_strength",
                "validation_ic_in_discovery_direction",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

    return long_df, summary


# ============================================================
# COIN SELECTION: EVENT-RELATIVE QUINTILES
# ============================================================

def assign_event_quintiles(group, feature):
    values = pd.to_numeric(
        group[feature],
        errors="coerce",
    )

    valid = values.notna()

    out = pd.Series(
        np.nan,
        index=group.index,
        dtype=float,
    )

    if valid.sum() < 10:
        return out

    ranks = values.loc[valid].rank(
        method="first",
        pct=True,
    )

    q = np.ceil(
        ranks * QUINTILES
    ).clip(
        1,
        QUINTILES,
    )

    out.loc[valid] = q

    return out


def feature_quintile_analysis(
    panel,
    feature_names,
):
    detail_rows = []
    summary_rows = []

    for idx, feature in enumerate(
        feature_names,
        start=1,
    ):
        if feature not in panel.columns:
            continue

        if idx % 10 == 0:
            print(
                f"  quintile feature {idx}/{len(feature_names)}"
            )

        for split in [
            "DISCOVERY_60D",
            "VALIDATION_30D",
        ]:
            subset = panel.loc[
                panel["split"] == split
            ].copy()

            if subset.empty:
                continue

            q_series = pd.Series(
                np.nan,
                index=subset.index,
                dtype=float,
            )

            for _, group in subset.groupby(
                "event_id",
                observed=True,
            ):
                q_series.loc[group.index] = (
                    assign_event_quintiles(
                        group,
                        feature,
                    )
                )

            subset["_q"] = q_series

            valid = subset.loc[
                subset["_q"].notna()
            ].copy()

            if valid.empty:
                continue

            for q, qgroup in valid.groupby(
                "_q",
                observed=True,
            ):
                detail_rows.append(
                    {
                        "feature": feature,
                        "split": split,
                        "quintile": int(q),
                        "rows": len(qgroup),
                        "events": int(
                            qgroup["event_id"].nunique()
                        ),
                        "mean_return_60m_atr":
                            float(
                                qgroup[
                                    "return_60m_atr"
                                ].mean()
                            ),
                        "median_return_60m_atr":
                            float(
                                qgroup[
                                    "return_60m_atr"
                                ].median()
                            ),
                        "trend_up_rate":
                            float(
                                qgroup[
                                    "trend_up_60m"
                                ].mean()
                            ),
                        "positive_rate":
                            float(
                                qgroup[
                                    "positive_60m"
                                ].mean()
                            ),
                    }
                )

            q1 = valid.loc[
                valid["_q"] == 1,
                "return_60m_atr",
            ]
            q5 = valid.loc[
                valid["_q"] == 5,
                "return_60m_atr",
            ]

            if len(q1) == 0 or len(q5) == 0:
                continue

            q1_mean = float(q1.mean())
            q5_mean = float(q5.mean())

            summary_rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "events_used":
                        int(valid["event_id"].nunique()),
                    "q1_mean_return_60m_atr":
                        q1_mean,
                    "q5_mean_return_60m_atr":
                        q5_mean,
                    "q5_minus_q1_atr":
                        q5_mean - q1_mean,
                    "better_side":
                        (
                            "HIGH"
                            if q5_mean >= q1_mean
                            else "LOW"
                        ),
                    "better_side_mean_return_60m_atr":
                        max(q1_mean, q5_mean),
                    "worse_side_mean_return_60m_atr":
                        min(q1_mean, q5_mean),
                }
            )

            del subset, valid, q_series
            gc.collect()

    detail = pd.DataFrame(detail_rows)
    summary_long = pd.DataFrame(summary_rows)

    if summary_long.empty:
        return detail, summary_long, pd.DataFrame()

    pivot = summary_long.pivot(
        index="feature",
        columns="split",
        values=[
            "events_used",
            "q1_mean_return_60m_atr",
            "q5_mean_return_60m_atr",
            "q5_minus_q1_atr",
            "better_side_mean_return_60m_atr",
        ],
    )

    pivot.columns = [
        f"{metric}_{split}"
        for metric, split in pivot.columns
    ]

    summary = pivot.reset_index()

    train_spread = (
        "q5_minus_q1_atr_DISCOVERY_60D"
    )
    val_spread = (
        "q5_minus_q1_atr_VALIDATION_30D"
    )

    if (
        train_spread in summary.columns
        and val_spread in summary.columns
    ):
        summary["same_quintile_direction"] = (
            np.sign(summary[train_spread])
            == np.sign(summary[val_spread])
        )

        summary["discovery_best_side"] = np.where(
            summary[train_spread] >= 0,
            "HIGH",
            "LOW",
        )

        summary["validation_spread_in_discovery_direction"] = (
            np.where(
                summary[train_spread] >= 0,
                summary[val_spread],
                -summary[val_spread],
            )
        )

        summary["discovery_abs_spread"] = (
            summary[train_spread].abs()
        )

        summary = summary.sort_values(
            [
                "same_quintile_direction",
                "discovery_abs_spread",
                "validation_spread_in_discovery_direction",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

    return (
        detail,
        summary_long,
        summary,
    )


# ============================================================
# TOP COINS PER EVENT
# ============================================================

def add_cross_sectional_outcome_ranks(panel):
    out = panel.copy()

    out["return_rank_60m"] = (
        out.groupby(
            "event_id",
            observed=True,
        )["return_60m_atr"]
        .rank(
            pct=True,
            method="average",
        )
    )

    out["top_quartile_60m"] = (
        out["return_rank_60m"] >= 0.75
    ).astype(np.int8)

    out["bottom_quartile_60m"] = (
        out["return_rank_60m"] <= 0.25
    ).astype(np.int8)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()

    summary_file, all_symbols, full_symbols = (
        load_universe_sets()
    )

    btc_feature_file = (
        load_btc_feature_file()
    )

    print()
    print("=" * 120)
    print("MARKET TRIGGER A - INDEPENDENT EVENT ANALYSIS")
    print("=" * 120)
    print("Frozen Trigger A:")
    print(
        f"  BTC 12h return <= {BTC_12H_RETURN_MAX}%"
    )
    print(
        f"  BTC 5m ATR > {BTC_5M_ATR_MIN}%"
    )
    print(
        f"  BTC 5m market regime > "
        f"{BTC_5M_MARKET_REGIME_MIN}"
    )
    print(
        f"  BTC 5m price_vs_EMA200 > "
        f"{BTC_5M_PRICE_VS_EMA200_MIN}%"
    )
    print(
        f"Independent event cooldown: "
        f"{EVENT_COOLDOWN_MINUTES} minutes"
    )
    print(
        f"Trend target: +/- "
        f"{TREND_ATR_THRESHOLD} x 1h ATR"
    )
    print()
    print(
        f"All downloaded symbols: {len(all_symbols)}"
    )
    print(
        f"Full-history symbols:    {len(full_symbols)}"
    )
    print(
        f"Universe source: {summary_file}"
    )
    print("=" * 120)

    events, trigger_meta = (
        build_trigger_events(
            btc_feature_file
        )
    )

    print()
    print("Trigger compression:")
    for key, value in trigger_meta.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print("=" * 120)
    print("BUILDING EVENT x COIN PANEL")
    print("=" * 120)

    panel, feature_names = (
        build_coin_event_panel(
            events=events,
            all_symbols=all_symbols,
            full_symbols=full_symbols,
        )
    )

    panel = add_cross_sectional_outcome_ranks(
        panel
    )

    print()
    print(
        f"Coin-event rows: {len(panel):,}"
    )
    print(
        f"Events represented: "
        f"{panel['event_id'].nunique()}"
    )
    print(
        f"Symbols represented: "
        f"{panel['symbol'].nunique()}"
    )
    print(
        f"Coin-selection features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Independent event results
    # --------------------------------------------------------

    event_summary = (
        event_market_summary(
            panel
        )
    )

    independent_summary = (
        summarize_independent_events(
            event_summary
        )
    )

    # --------------------------------------------------------
    # "What to trade?" diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("COIN-SELECTION FEATURE ANALYSIS")
    print("=" * 120)
    print(
        "1) Event-by-event Spearman information coefficient"
    )

    ic_long, ic_summary = (
        feature_ic_analysis(
            panel=panel,
            feature_names=feature_names,
        )
    )

    print(
        "2) Event-relative feature quintiles"
    )

    (
        quintile_detail,
        quintile_long,
        quintile_summary,
    ) = feature_quintile_analysis(
        panel=panel,
        feature_names=feature_names,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    events_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_independent_events_{stamp}.csv",
    )

    panel_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_coin_outcomes_{stamp}.csv",
    )

    event_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_event_summary_{stamp}.csv",
    )

    independent_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_independent_summary_{stamp}.csv",
    )

    ic_long_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_coin_feature_ic_long_{stamp}.csv",
    )

    ic_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_coin_feature_ic_summary_{stamp}.csv",
    )

    quintile_detail_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_coin_feature_quintiles_{stamp}.csv",
    )

    quintile_summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_coin_feature_quintile_summary_{stamp}.csv",
    )

    events.to_csv(
        events_file,
        index=False,
    )

    panel.to_csv(
        panel_file,
        index=False,
    )

    event_summary.to_csv(
        event_summary_file,
        index=False,
    )

    independent_summary.to_csv(
        independent_file,
        index=False,
    )

    ic_long.to_csv(
        ic_long_file,
        index=False,
    )

    ic_summary.to_csv(
        ic_summary_file,
        index=False,
    )

    quintile_detail.to_csv(
        quintile_detail_file,
        index=False,
    )

    quintile_summary.to_csv(
        quintile_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 150)
    print("INDEPENDENT EVENT SUMMARY")
    print("=" * 150)

    display_cols = [
        "universe",
        "split",
        "independent_events",
        "avg_coins_available",
        "mean_trend_up_breadth",
        "trend_up_breadth_ci95_low",
        "trend_up_breadth_ci95_high",
        "mean_positive_breadth",
        "mean_return_15m_pct",
        "mean_return_30m_pct",
        "mean_return_45m_pct",
        "mean_return_60m_pct",
        "mean_return_60m_atr",
        "mean_mfe_60m_atr",
        "mean_mae_60m_atr",
    ]

    print(
        independent_summary[
            [
                c for c in display_cols
                if c in independent_summary.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 150)
    print("TOP COIN-SELECTION FEATURES BY DISCOVERY IC")
    print("=" * 150)

    if ic_summary.empty:
        print("No IC results.")
    else:
        top_ic = ic_summary.head(20)
        print(
            top_ic.to_string(index=False)
        )

    print()
    print("=" * 150)
    print("TOP COIN-SELECTION FEATURES BY DISCOVERY QUINTILE SPREAD")
    print("=" * 150)

    if quintile_summary.empty:
        print("No quintile results.")
    else:
        top_q = quintile_summary.head(20)
        print(
            top_q.to_string(index=False)
        )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Independent events: {events_file}")
    print(f"Coin outcomes:      {panel_file}")
    print(f"Event summary:      {event_summary_file}")
    print(f"Independent summary:{independent_file}")
    print(f"IC long:            {ic_long_file}")
    print(f"IC summary:         {ic_summary_file}")
    print(f"Quintiles:          {quintile_detail_file}")
    print(f"Quintile summary:   {quintile_summary_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
