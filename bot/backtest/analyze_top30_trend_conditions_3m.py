import glob
import math
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


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

RESEARCH_DAYS = 90
TRAIN_DAYS = 60
HORIZON_BARS_5M = 12

TREND_ATR_THRESHOLD = 0.50

TREE_MAX_DEPTH = 4
TREE_MIN_LEAF_FRACTION = 0.004
TREE_MIN_LEAF_ABS = 400
TREE_RANDOM_STATE = 42

UNIVARIATE_BINS = 5

# Feature names to use for cross-sectional ranks if present.
CROSS_SECTIONAL_BASES = [
    "5m_atr_pct",
    "5m_volume_ratio20",
    "5m_ema_spread_pct",
    "5m_price_vs_ema200_pct",
    "5m_ret_12bar_pct",
    "1h_atr_pct",
    "1h_ema_spread_pct",
    "1h_price_vs_ema200_pct",
    "4h_atr_pct",
    "4h_ema_spread_pct",
    "4h_price_vs_ema200_pct",
]


# ============================================================
# BASIC INDICATORS
# ============================================================

def rma(series, length):
    return series.ewm(
        alpha=1.0 / length,
        adjust=False,
        min_periods=length
    ).mean()


def true_range(df):
    prev_close = df[
        "close"
    ].shift(1)

    parts = pd.concat(
        [
            (
                df["high"]
                - df["low"]
            ),
            (
                df["high"]
                - prev_close
            ).abs(),
            (
                df["low"]
                - prev_close
            ).abs(),
        ],
        axis=1
    )

    return parts.max(
        axis=1
    )


def add_indicator_features(
    df
):
    out = df.copy()

    out["ema50"] = (
        out["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    out["ema200"] = (
        out["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    out["atr"] = rma(
        true_range(out),
        14
    )

    out["atr_pct"] = (
        out["atr"]
        / out["close"]
        * 100.0
    )

    vol_ma20 = (
        out["volume"]
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    out["volume_ratio20"] = (
        out["volume"]
        / vol_ma20
    )

    out["ema_spread_pct"] = (
        (
            out["ema50"]
            - out["ema200"]
        )
        / out["close"]
        * 100.0
    )

    out["price_vs_ema200_pct"] = (
        (
            out["close"]
            - out["ema200"]
        )
        / out["ema200"]
        * 100.0
    )

    # Causal Williams Alligator-style values.
    jaw_raw = rma(
        out["close"],
        13
    )
    teeth_raw = rma(
        out["close"],
        8
    )
    lips_raw = rma(
        out["close"],
        5
    )

    out["alligator_jaw"] = (
        jaw_raw.shift(8)
    )
    out["alligator_teeth"] = (
        teeth_raw.shift(5)
    )
    out["alligator_lips"] = (
        lips_raw.shift(3)
    )

    out["alligator_spread_pct"] = (
        (
            out[
                [
                    "alligator_jaw",
                    "alligator_teeth",
                    "alligator_lips",
                ]
            ]
            .max(axis=1)
            -
            out[
                [
                    "alligator_jaw",
                    "alligator_teeth",
                    "alligator_lips",
                ]
            ]
            .min(axis=1)
        )
        / out["close"]
        * 100.0
    )

    out["alligator_state"] = np.select(
        [
            (
                out["alligator_lips"]
                > out["alligator_teeth"]
            )
            &
            (
                out["alligator_teeth"]
                > out["alligator_jaw"]
            ),
            (
                out["alligator_lips"]
                < out["alligator_teeth"]
            )
            &
            (
                out["alligator_teeth"]
                < out["alligator_jaw"]
            ),
        ],
        [
            1,
            -1,
        ],
        default=0
    )

    # Confirmed 5-bar fractals: raw pivot is only known
    # two bars later; shift(2) makes it causal.
    raw_fh = (
        (
            out["high"]
            > out["high"].shift(1)
        )
        &
        (
            out["high"]
            > out["high"].shift(2)
        )
        &
        (
            out["high"]
            > out["high"].shift(-1)
        )
        &
        (
            out["high"]
            > out["high"].shift(-2)
        )
    )

    raw_fl = (
        (
            out["low"]
            < out["low"].shift(1)
        )
        &
        (
            out["low"]
            < out["low"].shift(2)
        )
        &
        (
            out["low"]
            < out["low"].shift(-1)
        )
        &
        (
            out["low"]
            < out["low"].shift(-2)
        )
    )

    confirmed_fh = (
        out["high"]
        .where(raw_fh)
        .shift(2)
    )

    confirmed_fl = (
        out["low"]
        .where(raw_fl)
        .shift(2)
    )

    out["last_fractal_high"] = (
        confirmed_fh.ffill()
    )

    out["last_fractal_low"] = (
        confirmed_fl.ffill()
    )

    out["dist_to_fractal_high_atr"] = (
        (
            out["last_fractal_high"]
            - out["close"]
        )
        / out["atr"]
    )

    out["dist_to_fractal_low_atr"] = (
        (
            out["close"]
            - out["last_fractal_low"]
        )
        / out["atr"]
    )

    out["break_above_fractal"] = (
        out["close"]
        > out["last_fractal_high"]
    ).astype(float)

    out["break_below_fractal"] = (
        out["close"]
        < out["last_fractal_low"]
    ).astype(float)

    # Market regime / trend score.
    out["market_regime"] = np.select(
        [
            (
                out["close"]
                > out["ema200"]
            )
            &
            (
                out["ema50"]
                > out["ema200"]
            ),
            (
                out["close"]
                < out["ema200"]
            )
            &
            (
                out["ema50"]
                < out["ema200"]
            ),
        ],
        [
            1,
            -1,
        ],
        default=0
    )

    out["trend_score"] = (
        np.tanh(
            out["ema_spread_pct"]
            / 2.0
        )
        * 50.0
        +
        np.tanh(
            out["price_vs_ema200_pct"]
            / 4.0
        )
        * 30.0
        +
        out["alligator_state"]
        * 20.0
    )

    atr_median_50 = (
        out["atr_pct"]
        .rolling(
            50,
            min_periods=30
        )
        .median()
    )

    out["volatility_regime"] = np.select(
        [
            (
                out["atr_pct"]
                > atr_median_50
                * 1.25
            ),
            (
                out["atr_pct"]
                < atr_median_50
                * 0.80
            ),
        ],
        [
            1,
            -1,
        ],
        default=0
    )

    # Price action.
    candle_range = (
        out["high"]
        - out["low"]
    )

    body = (
        out["close"]
        - out["open"]
    )

    out["body_to_range"] = np.where(
        candle_range > 0,
        body.abs()
        / candle_range,
        np.nan
    )

    out["signed_body_to_range"] = np.where(
        candle_range > 0,
        body
        / candle_range,
        np.nan
    )

    out["upper_wick_to_range"] = np.where(
        candle_range > 0,
        (
            out["high"]
            - out[
                [
                    "open",
                    "close",
                ]
            ].max(axis=1)
        )
        / candle_range,
        np.nan
    )

    out["lower_wick_to_range"] = np.where(
        candle_range > 0,
        (
            out[
                [
                    "open",
                    "close",
                ]
            ].min(axis=1)
            - out["low"]
        )
        / candle_range,
        np.nan
    )

    out["close_location_in_range"] = np.where(
        candle_range > 0,
        (
            out["close"]
            - out["low"]
        )
        / candle_range,
        np.nan
    )

    out["range_atr"] = (
        candle_range
        / out["atr"]
    )

    # Timeframe-native momentum. The prefix applied later
    # makes the meaning explicit:
    # 5m_ret_12bar_pct = 60m return
    # 1h_ret_1bar_pct  = 1h return
    # 4h_ret_1bar_pct  = 4h return
    out["ret_1bar_pct"] = (
        out["close"]
        .pct_change(1)
        * 100.0
    )

    out["ret_3bar_pct"] = (
        out["close"]
        .pct_change(3)
        * 100.0
    )

    out["ret_12bar_pct"] = (
        out["close"]
        .pct_change(12)
        * 100.0
    )

    out["ema200_slope_1bar_pct"] = (
        out["ema200"]
        .pct_change(1)
        * 100.0
    )

    out["ema200_slope_3bar_pct"] = (
        out["ema200"]
        .pct_change(3)
        * 100.0
    )

    out["ema200_slope_12bar_pct"] = (
        out["ema200"]
        .pct_change(12)
        * 100.0
    )

    return out


# ============================================================
# RESAMPLING / AVAILABILITY
# ============================================================

def resample_ohlcv(
    df,
    rule
):
    temp = (
        df
        .set_index(
            "timestamp"
        )
        .sort_index()
    )

    agg = temp.resample(
        rule,
        label="left",
        closed="left"
    ).agg(
        {
            "open":
                "first",
            "high":
                "max",
            "low":
                "min",
            "close":
                "last",
            "volume":
                "sum",
            "turnover":
                "sum",
        }
    )

    agg = (
        agg
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
    )

    return agg


def prefix_feature_frame(
    df,
    prefix,
    bar_minutes
):
    excluded = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
        and c != "timestamp"
    ]

    out = df[
        [
            "timestamp"
        ]
        + feature_cols
    ].copy()

    out["available_at"] = (
        out["timestamp"]
        + pd.Timedelta(
            minutes=bar_minutes
        )
    )

    rename = {
        col: f"{prefix}_{col}"
        for col in feature_cols
    }

    out = out.rename(
        columns=rename
    )

    return out.drop(
        columns=[
            "timestamp"
        ]
    )


def merge_asof_features(
    base,
    feature_frame
):
    return pd.merge_asof(
        base.sort_values(
            "available_at"
        ),
        feature_frame.sort_values(
            "available_at"
        ),
        on="available_at",
        direction="backward",
        allow_exact_matches=True
    )


# ============================================================
# OUTCOMES
# ============================================================

def add_one_hour_outcomes(
    base_5m
):
    df = base_5m.copy()

    entry = (
        df["open"]
        .shift(-1)
    )

    final_close = (
        df["close"]
        .shift(
            -HORIZON_BARS_5M
        )
    )

    high_stack = pd.concat(
        [
            df["high"]
            .shift(-k)
            for k in range(
                1,
                HORIZON_BARS_5M + 1
            )
        ],
        axis=1
    )

    low_stack = pd.concat(
        [
            df["low"]
            .shift(-k)
            for k in range(
                1,
                HORIZON_BARS_5M + 1
            )
        ],
        axis=1
    )

    future_high = (
        high_stack.max(
            axis=1
        )
    )

    future_low = (
        low_stack.min(
            axis=1
        )
    )

    df[
        "outcome_entry_price"
    ] = entry

    df[
        "future_1h_return_pct"
    ] = (
        (
            final_close
            / entry
            - 1.0
        )
        * 100.0
    )

    df[
        "future_1h_mfe_pct"
    ] = (
        (
            future_high
            / entry
            - 1.0
        )
        * 100.0
    )

    df[
        "future_1h_mae_pct"
    ] = (
        (
            future_low
            / entry
            - 1.0
        )
        * 100.0
    )

    return df


# ============================================================
# SYMBOL FEATURE MATRIX
# ============================================================

def build_symbol_matrix(
    symbol,
    raw
):
    raw = (
        raw
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    five = add_indicator_features(
        raw
    )

    one = add_indicator_features(
        resample_ohlcv(
            raw,
            "1h"
        )
    )

    four = add_indicator_features(
        resample_ohlcv(
            raw,
            "4h"
        )
    )

    base = add_one_hour_outcomes(
        five[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
        ]
    )

    base[
        "available_at"
    ] = (
        base[
            "timestamp"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    f5 = prefix_feature_frame(
        five,
        "5m",
        5
    )

    f1 = prefix_feature_frame(
        one,
        "1h",
        60
    )

    f4 = prefix_feature_frame(
        four,
        "4h",
        240
    )

    merged = merge_asof_features(
        base,
        f5
    )

    merged = merge_asof_features(
        merged,
        f1
    )

    merged = merge_asof_features(
        merged,
        f4
    )

    merged[
        "symbol"
    ] = symbol

    return merged


# ============================================================
# CROSS-ASSET CONTEXT
# ============================================================

def add_btc_context(
    matrix
):
    btc = matrix[
        matrix[
            "symbol"
        ]
        == "BTCUSDT"
    ].copy()

    if btc.empty:
        print(
            "WARNING: BTCUSDT not in frozen top30; "
            "BTC context will be absent."
        )
        return matrix

    keep = [
        "available_at",
    ]

    btc_candidates = [
        "5m_atr_pct",
        "5m_volume_ratio20",
        "5m_ema_spread_pct",
        "5m_price_vs_ema200_pct",
        "5m_market_regime",
        "5m_trend_score",
        "5m_ret_12bar_pct",
        "1h_atr_pct",
        "1h_ema_spread_pct",
        "1h_price_vs_ema200_pct",
        "1h_market_regime",
        "1h_trend_score",
        "1h_ret_3bar_pct",
        "4h_atr_pct",
        "4h_ema_spread_pct",
        "4h_price_vs_ema200_pct",
        "4h_market_regime",
        "4h_trend_score",
        "4h_ret_3bar_pct",
    ]

    btc_candidates = [
        c for c in btc_candidates
        if c in btc.columns
    ]

    btc = btc[
        keep
        + btc_candidates
    ].copy()

    btc = btc.rename(
        columns={
            c: f"btc_{c}"
            for c in btc_candidates
        }
    )

    out = matrix.merge(
        btc,
        on="available_at",
        how="left",
        validate="many_to_one"
    )

    if (
        "5m_ret_1h_pct"
        in out.columns
        and "btc_5m_ret_1h_pct"
        in out.columns
    ):
        out[
            "relative_strength_1h_vs_btc"
        ] = (
            out[
                "5m_ret_12bar_pct"
            ]
            - out[
                "btc_5m_ret_12bar_pct"
            ]
        )

    if (
        "1h_ret_3bar_pct"
        in out.columns
        and "btc_1h_ret_3h_pct"
        in out.columns
    ):
        out[
            "relative_strength_3h_vs_btc"
        ] = (
            out[
                "1h_ret_3bar_pct"
            ]
            - out[
                "btc_1h_ret_3bar_pct"
            ]
        )

    return out


def add_cross_sectional_ranks(
    matrix
):
    out = matrix.copy()

    for col in CROSS_SECTIONAL_BASES:
        if col not in out.columns:
            continue

        out[
            f"rank_{col}"
        ] = (
            out
            .groupby(
                "available_at"
            )[
                col
            ]
            .rank(
                pct=True,
                method="average"
            )
        )

    return out


def add_btc_correlations(
    matrix
):
    out = matrix.sort_values(
        [
            "symbol",
            "available_at",
        ]
    ).copy()

    # 5m rolling 50-bar BTC correlation.
    if (
        "5m_ret_1bar_pct"
        in out.columns
        and "btc_5m_ret_1bar_pct"
        in out.columns
    ):
        out[
            "btc_correlation_5m_50"
        ] = (
            out
            .groupby(
                "symbol",
                group_keys=False
            )
            .apply(
                lambda g: (
                    g[
                        "5m_ret_1bar_pct"
                    ]
                    .rolling(
                        50,
                        min_periods=30
                    )
                    .corr(
                        g[
                            "btc_5m_ret_1bar_pct"
                        ]
                    )
                )
            )
            .reset_index(
                level=0,
                drop=True
            )
        )

    # For higher timeframes, compute one observation per
    # completed timeframe bar, then map it back to every 5m row.
    for tf, key_freq in [
        (
            "1h",
            "1h"
        ),
        (
            "4h",
            "4h"
        ),
    ]:
        sym_col = (
            f"{tf}_ret_1bar_pct"
        )

        btc_col = (
            f"btc_{tf}_ret_1bar_pct"
        )

        if (
            sym_col not in out.columns
            or btc_col not in out.columns
        ):
            continue

        key_col = (
            f"_{tf}_key"
        )

        out[
            key_col
        ] = (
            out[
                "available_at"
            ]
            .dt.floor(
                key_freq
            )
        )

        pieces = []

        for symbol, group in out.groupby(
            "symbol"
        ):
            compact = (
                group[
                    [
                        key_col,
                        sym_col,
                        btc_col,
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        key_col
                    ],
                    keep="last"
                )
                .sort_values(
                    key_col
                )
                .copy()
            )

            compact[
                f"btc_correlation_{tf}_50"
            ] = (
                compact[
                    sym_col
                ]
                .rolling(
                    50,
                    min_periods=30
                )
                .corr(
                    compact[
                        btc_col
                    ]
                )
            )

            compact[
                "symbol"
            ] = symbol

            pieces.append(
                compact[
                    [
                        "symbol",
                        key_col,
                        f"btc_correlation_{tf}_50",
                    ]
                ]
            )

        corr_map = pd.concat(
            pieces,
            ignore_index=True
        )

        out = out.merge(
            corr_map,
            on=[
                "symbol",
                key_col,
            ],
            how="left",
            validate="many_to_one"
        )

        out = out.drop(
            columns=[
                key_col
            ]
        )

    return out


# ============================================================
# TARGET
# ============================================================

def add_target(
    matrix
):
    out = matrix.copy()

    if (
        "1h_atr_pct"
        not in out.columns
    ):
        raise RuntimeError(
            "1h_atr_pct unavailable; cannot build "
            "ATR-normalized target."
        )

    threshold = (
        out[
            "1h_atr_pct"
        ]
        * TREND_ATR_THRESHOLD
    )

    out[
        "future_1h_return_atr"
    ] = (
        out[
            "future_1h_return_pct"
        ]
        / out[
            "1h_atr_pct"
        ]
    )

    out[
        "future_1h_mfe_atr"
    ] = (
        out[
            "future_1h_mfe_pct"
        ]
        / out[
            "1h_atr_pct"
        ]
    )

    out[
        "future_1h_mae_atr"
    ] = (
        out[
            "future_1h_mae_pct"
        ]
        / out[
            "1h_atr_pct"
        ]
    )

    out[
        "trend_class"
    ] = np.select(
        [
            (
                out[
                    "future_1h_return_pct"
                ]
                >= threshold
            ),
            (
                out[
                    "future_1h_return_pct"
                ]
                <= -threshold
            ),
        ],
        [
            "UP",
            "DOWN",
        ],
        default="FLAT"
    )

    out[
        "target_up"
    ] = (
        out[
            "trend_class"
        ]
        == "UP"
    ).astype(int)

    out[
        "target_down"
    ] = (
        out[
            "trend_class"
        ]
        == "DOWN"
    ).astype(int)

    return out


# ============================================================
# RULE METRICS
# ============================================================

def wilson_lower_bound(
    wins,
    n,
    z=1.96
):
    if n <= 0:
        return np.nan

    p = wins / n

    denominator = (
        1
        + z * z / n
    )

    center = (
        p
        + z * z / (2 * n)
    )

    margin = z * math.sqrt(
        (
            p * (1 - p) / n
        )
        + (
            z * z
            / (
                4 * n * n
            )
        )
    )

    return (
        center
        - margin
    ) / denominator


def feature_columns(
    df
):
    exclude_exact = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "outcome_entry_price",
        "future_1h_return_pct",
        "future_1h_mfe_pct",
        "future_1h_mae_pct",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
        "target_up",
        "target_down",
    }

    cols = []

    for col in df.columns:
        if col in exclude_exact:
            continue

        if col in {
            "timestamp",
            "available_at",
            "symbol",
            "trend_class",
        }:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        cols.append(
            col
        )

    # Remove raw absolute indicator levels that are
    # price-scale dependent across different coins.
    remove_patterns = [
        "_ema50",
        "_ema200",
        "_alligator_jaw",
        "_alligator_teeth",
        "_alligator_lips",
        "_last_fractal_high",
        "_last_fractal_low",
        "_atr",
    ]

    safe = []

    for col in cols:
        if (
            col.endswith(
                "_atr_pct"
            )
            or "_atr_" in col
            or col.endswith(
                "_range_atr"
            )
        ):
            safe.append(
                col
            )
            continue

        if any(
            col.endswith(
                p
            )
            for p in remove_patterns
        ):
            continue

        safe.append(
            col
        )

    return sorted(
        set(
            safe
        )
    )


# ============================================================
# UNIVARIATE CONDITION SCAN
# ============================================================

def scan_univariate(
    train,
    test,
    features
):
    rows = []

    for feature in features:
        train_vals = pd.to_numeric(
            train[
                feature
            ],
            errors="coerce"
        )

        valid_train = train_vals.dropna()

        if (
            valid_train.nunique()
            < 8
        ):
            continue

        try:
            _, edges = pd.qcut(
                valid_train,
                q=UNIVARIATE_BINS,
                retbins=True,
                duplicates="drop"
            )
        except Exception:
            continue

        edges = np.unique(
            edges
        )

        if len(edges) < 3:
            continue

        edges[0] = -np.inf
        edges[-1] = np.inf

        for bin_idx in range(
            len(edges) - 1
        ):
            lo = edges[
                bin_idx
            ]
            hi = edges[
                bin_idx + 1
            ]

            train_mask = (
                train[feature]
                > lo
            ) & (
                train[feature]
                <= hi
            )

            test_mask = (
                test[feature]
                > lo
            ) & (
                test[feature]
                <= hi
            )

            for direction, target in [
                (
                    "UP",
                    "target_up"
                ),
                (
                    "DOWN",
                    "target_down"
                ),
            ]:
                tr = train.loc[
                    train_mask
                ]

                te = test.loc[
                    test_mask
                ]

                if (
                    len(tr) < 300
                    or len(te) < 100
                ):
                    continue

                tr_wins = int(
                    tr[target].sum()
                )
                te_wins = int(
                    te[target].sum()
                )

                base_test = float(
                    test[
                        target
                    ].mean()
                )

                te_precision = (
                    te_wins
                    / len(te)
                )

                rows.append(
                    {
                        "feature":
                            feature,
                        "bin":
                            bin_idx + 1,
                        "condition":
                            (
                                f"{feature} > {lo:.6g} "
                                f"AND <= {hi:.6g}"
                            ),
                        "direction":
                            direction,
                        "train_n":
                            len(tr),
                        "train_precision":
                            tr_wins
                            / len(tr),
                        "test_n":
                            len(te),
                        "test_precision":
                            te_precision,
                        "test_base_rate":
                            base_test,
                        "test_lift":
                            (
                                te_precision
                                / base_test
                                if base_test > 0
                                else np.nan
                            ),
                        "test_wilson_lb":
                            wilson_lower_bound(
                                te_wins,
                                len(te)
                            ),
                        "test_avg_return_atr":
                            te[
                                "future_1h_return_atr"
                            ].mean(),
                        "test_median_return_atr":
                            te[
                                "future_1h_return_atr"
                            ].median(),
                        "test_coin_count":
                            te[
                                "symbol"
                            ].nunique(),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TREE RULE EXTRACTION
# ============================================================

def tree_paths(
    tree,
    feature_names
):
    t = tree.tree_
    paths = {}

    def recurse(
        node,
        conditions
    ):
        left = t.children_left[
            node
        ]
        right = t.children_right[
            node
        ]

        if left == right:
            paths[node] = list(
                conditions
            )
            return

        feature_idx = t.feature[
            node
        ]

        threshold = t.threshold[
            node
        ]

        name = feature_names[
            feature_idx
        ]

        recurse(
            left,
            conditions
            + [
                (
                    name,
                    "<=",
                    threshold
                )
            ]
        )

        recurse(
            right,
            conditions
            + [
                (
                    name,
                    ">",
                    threshold
                )
            ]
        )

    recurse(
        0,
        []
    )

    return paths


def condition_to_text(
    conditions
):
    return " AND ".join(
        [
            f"{name} {op} {value:.6g}"
            for (
                name,
                op,
                value
            )
            in conditions
        ]
    )


def evaluate_tree_rules(
    train,
    test,
    features,
    target,
    direction
):
    model_train = train[
        features
        + [
            target,
        ]
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan
    )

    medians = (
        model_train[
            features
        ]
        .median()
    )

    x_train = (
        model_train[
            features
        ]
        .fillna(
            medians
        )
    )

    y_train = (
        model_train[
            target
        ].astype(int)
    )

    min_leaf = max(
        TREE_MIN_LEAF_ABS,
        int(
            len(
                x_train
            )
            * TREE_MIN_LEAF_FRACTION
        )
    )

    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=TREE_RANDOM_STATE
    )

    tree.fit(
        x_train,
        y_train
    )

    x_test = (
        test[
            features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan
        )
        .fillna(
            medians
        )
    )

    train_leaf = tree.apply(
        x_train
    )

    test_leaf = tree.apply(
        x_test
    )

    paths = tree_paths(
        tree,
        features
    )

    base_train = float(
        y_train.mean()
    )

    base_test = float(
        test[
            target
        ].mean()
    )

    rows = []

    for leaf_id, conditions in paths.items():
        tr_mask = (
            train_leaf
            == leaf_id
        )

        te_mask = (
            test_leaf
            == leaf_id
        )

        tr_n = int(
            tr_mask.sum()
        )
        te_n = int(
            te_mask.sum()
        )

        if (
            tr_n < min_leaf
            or te_n < 100
        ):
            continue

        tr_subset = train.loc[
            tr_mask
        ]

        te_subset = test.loc[
            te_mask
        ]

        tr_wins = int(
            tr_subset[
                target
            ].sum()
        )

        te_wins = int(
            te_subset[
                target
            ].sum()
        )

        tr_precision = (
            tr_wins
            / tr_n
        )

        te_precision = (
            te_wins
            / te_n
        )

        # Two-week chronological stability across all 90d.
        combined = pd.concat(
            [
                train.assign(
                    _split="TRAIN"
                ),
                test.assign(
                    _split="TEST"
                ),
            ],
            ignore_index=True
        )

        x_combined = (
            combined[
                features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan
            )
            .fillna(
                medians
            )
        )

        combined_leaf = tree.apply(
            x_combined
        )

        leaf_combined = combined.loc[
            combined_leaf
            == leaf_id
        ].copy()

        if not leaf_combined.empty:
            start = (
                combined[
                    "available_at"
                ].min()
            )

            leaf_combined[
                "_block"
            ] = (
                (
                    leaf_combined[
                        "available_at"
                    ]
                    - start
                )
                .dt
                .days
                // 14
            )

            block_stats = (
                leaf_combined
                .groupby(
                    "_block"
                )[
                    target
                ]
                .agg(
                    [
                        "count",
                        "mean",
                    ]
                )
            )

            stable_blocks = int(
                (
                    (
                        block_stats[
                            "count"
                        ]
                        >= 40
                    )
                    &
                    (
                        block_stats[
                            "mean"
                        ]
                        > base_test
                    )
                )
                .sum()
            )

            eligible_blocks = int(
                (
                    block_stats[
                        "count"
                    ]
                    >= 40
                )
                .sum()
            )

        else:
            stable_blocks = 0
            eligible_blocks = 0

        rows.append(
            {
                "direction":
                    direction,
                "leaf_id":
                    int(
                        leaf_id
                    ),
                "condition":
                    condition_to_text(
                        conditions
                    ),
                "train_n":
                    tr_n,
                "train_precision":
                    tr_precision,
                "train_base_rate":
                    base_train,
                "train_lift":
                    (
                        tr_precision
                        / base_train
                        if base_train > 0
                        else np.nan
                    ),
                "test_n":
                    te_n,
                "test_precision":
                    te_precision,
                "test_base_rate":
                    base_test,
                "test_lift":
                    (
                        te_precision
                        / base_test
                        if base_test > 0
                        else np.nan
                    ),
                "test_wilson_lb":
                    wilson_lower_bound(
                        te_wins,
                        te_n
                    ),
                "test_avg_return_atr":
                    te_subset[
                        "future_1h_return_atr"
                    ].mean(),
                "test_median_return_atr":
                    te_subset[
                        "future_1h_return_atr"
                    ].median(),
                "test_avg_mfe_atr":
                    te_subset[
                        "future_1h_mfe_atr"
                    ].mean(),
                "test_avg_mae_atr":
                    te_subset[
                        "future_1h_mae_atr"
                    ].mean(),
                "test_coin_count":
                    te_subset[
                        "symbol"
                    ].nunique(),
                "stable_14d_blocks":
                    stable_blocks,
                "eligible_14d_blocks":
                    eligible_blocks,
                "stability_ratio":
                    (
                        stable_blocks
                        / eligible_blocks
                        if eligible_blocks > 0
                        else np.nan
                    ),
            }
        )

    rules = pd.DataFrame(
        rows
    )

    # AUC diagnostic only.
    test_auc = np.nan

    if (
        test[
            target
        ].nunique()
        > 1
    ):
        try:
            prob = tree.predict_proba(
                x_test
            )[:, 1]

            test_auc = roc_auc_score(
                test[
                    target
                ],
                prob
            )
        except Exception:
            pass

    meta = {
        "direction":
            direction,
        "target":
            target,
        "train_rows":
            len(train),
        "test_rows":
            len(test),
        "min_samples_leaf":
            min_leaf,
        "max_depth":
            TREE_MAX_DEPTH,
        "train_base_rate":
            base_train,
        "test_base_rate":
            base_test,
        "test_auc":
            test_auc,
        "tree_features_used":
            int(
                (
                    tree.feature_importances_
                    > 0
                ).sum()
            ),
    }

    importance = pd.DataFrame(
        {
            "direction":
                direction,
            "feature":
                features,
            "importance":
                tree.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    return (
        rules,
        pd.DataFrame(
            [
                meta
            ]
        ),
        importance
    )


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(
        glob.glob(
            os.path.join(
                RAW_DIR,
                "*_5m_120d.parquet"
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No 5m parquet files found in {RAW_DIR}. "
            "Run download_trend_research_3m.py first."
        )

    print()
    print("=" * 120)
    print("TOP30 MULTI-TIMEFRAME TREND DISCOVERY")
    print("=" * 120)
    print(
        f"Files: {len(files)}"
    )
    print(
        "Signal clock: every closed 5m candle"
    )
    print(
        "Features: 5m + latest closed 1h + latest closed 4h "
        "+ BTC context + cross-sectional ranks"
    )
    print(
        "Prediction horizon: next 60 minutes"
    )
    print(
        f"Trend label: +/- {TREND_ATR_THRESHOLD:.2f} "
        "x latest closed 1h ATR"
    )
    print(
        "Discovery split: first 60d train, last 30d validation"
    )
    print(
        "No future candle may enter feature construction."
    )
    print("=" * 120)

    matrices = []

    for idx, path in enumerate(
        files,
        start=1
    ):
        symbol = (
            os.path.basename(
                path
            )
            .replace(
                "_5m_120d.parquet",
                ""
            )
        )

        print(
            f"[{idx:02d}/{len(files):02d}] "
            f"{symbol}"
        )

        raw = pd.read_parquet(
            path
        )

        raw["timestamp"] = pd.to_datetime(
            raw["timestamp"],
            utc=True,
            errors="coerce"
        )

        raw = (
            raw
            .dropna(
                subset=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        matrix = build_symbol_matrix(
            symbol,
            raw
        )

        matrices.append(
            matrix
        )

    combined = pd.concat(
        matrices,
        ignore_index=True
    )

    combined = add_btc_context(
        combined
    )

    combined = add_cross_sectional_ranks(
        combined
    )

    combined = add_btc_correlations(
        combined
    )

    combined = add_target(
        combined
    )

    latest = combined[
        "available_at"
    ].max()

    research_start = (
        latest
        - pd.Timedelta(
            days=RESEARCH_DAYS
        )
    )

    train_end = (
        research_start
        + pd.Timedelta(
            days=TRAIN_DAYS
        )
    )

    research = combined[
        (
            combined[
                "available_at"
            ]
            >= research_start
        )
        &
        (
            combined[
                "available_at"
            ]
            <= latest
        )
    ].copy()

    research = research[
        research[
            "future_1h_return_pct"
        ].notna()
        &
        research[
            "1h_atr_pct"
        ].notna()
        &
        (
            research[
                "1h_atr_pct"
            ]
            > 0
        )
    ].copy()

    train = research[
        research[
            "available_at"
        ]
        < train_end
    ].copy()

    test = research[
        research[
            "available_at"
        ]
        >= train_end
    ].copy()

    print()
    print("=" * 120)
    print("RESEARCH MATRIX")
    print("=" * 120)
    print(
        f"Research rows: {len(research):,}"
    )
    print(
        f"Train rows:    {len(train):,}"
    )
    print(
        f"Test rows:     {len(test):,}"
    )
    print(
        f"Symbols:       {research['symbol'].nunique()}"
    )
    print(
        f"Research:      {research_start} -> {latest}"
    )
    print(
        f"Train/Test:    {train_end}"
    )
    print()
    print(
        "Target distribution (test):"
    )
    print(
        test[
            "trend_class"
        ]
        .value_counts(
            normalize=True
        )
        .mul(
            100
        )
        .round(
            2
        )
        .to_string()
    )

    features = feature_columns(
        research
    )

    # Drop unusable / nearly constant features.
    usable = []

    for col in features:
        s = train[
            col
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan
        )

        if (
            s.notna().sum()
            < 1000
        ):
            continue

        if (
            s.nunique(
                dropna=True
            )
            < 3
        ):
            continue

        usable.append(
            col
        )

    features = usable

    print()
    print(
        f"Usable numeric features: {len(features)}"
    )

    # --------------------------------------------------------
    # Univariate scan
    # --------------------------------------------------------

    print()
    print(
        "Running univariate condition scan..."
    )

    univariate = scan_univariate(
        train,
        test,
        features
    )

    if not univariate.empty:
        univariate = univariate.sort_values(
            [
                "test_wilson_lb",
                "test_lift",
                "test_n",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )

    # --------------------------------------------------------
    # Tree rules
    # --------------------------------------------------------

    print(
        "Training interpretable UP rule tree..."
    )

    (
        rules_up,
        meta_up,
        imp_up
    ) = evaluate_tree_rules(
        train=train,
        test=test,
        features=features,
        target="target_up",
        direction="UP"
    )

    print(
        "Training interpretable DOWN rule tree..."
    )

    (
        rules_down,
        meta_down,
        imp_down
    ) = evaluate_tree_rules(
        train=train,
        test=test,
        features=features,
        target="target_down",
        direction="DOWN"
    )

    rules = pd.concat(
        [
            rules_up,
            rules_down,
        ],
        ignore_index=True
    )

    if not rules.empty:
        rules = rules.sort_values(
            [
                "test_wilson_lb",
                "test_lift",
                "stability_ratio",
                "test_n",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ]
        )

    meta = pd.concat(
        [
            meta_up,
            meta_down,
        ],
        ignore_index=True
    )

    importance = pd.concat(
        [
            imp_up,
            imp_down,
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Compact top condition summary
    # --------------------------------------------------------

    top_rules = rules.copy()

    if not top_rules.empty:
        top_rules = top_rules[
            (
                top_rules[
                    "test_n"
                ]
                >= 200
            )
            &
            (
                top_rules[
                    "test_coin_count"
                ]
                >= max(
                    10,
                    int(
                        research[
                            "symbol"
                        ].nunique()
                        * 0.4
                    )
                )
            )
        ].copy()

        top_rules = (
            top_rules
            .groupby(
                "direction",
                group_keys=False
            )
            .head(
                20
            )
        )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    matrix_file = os.path.join(
        DATA_DIR,
        f"trend_feature_matrix_90d_{stamp}.parquet"
    )

    univariate_file = os.path.join(
        REPORT_DIR,
        f"trend_univariate_conditions_{stamp}.csv"
    )

    rules_file = os.path.join(
        REPORT_DIR,
        f"trend_combination_rules_{stamp}.csv"
    )

    top_rules_file = os.path.join(
        REPORT_DIR,
        f"trend_top_validated_rules_{stamp}.csv"
    )

    importance_file = os.path.join(
        REPORT_DIR,
        f"trend_rule_feature_importance_{stamp}.csv"
    )

    meta_file = os.path.join(
        REPORT_DIR,
        f"trend_rule_model_summary_{stamp}.csv"
    )

    target_dist_file = os.path.join(
        REPORT_DIR,
        f"trend_target_distribution_{stamp}.csv"
    )

    # Large matrix stays local.
    research.to_parquet(
        matrix_file,
        index=False
    )

    univariate.to_csv(
        univariate_file,
        index=False
    )

    rules.to_csv(
        rules_file,
        index=False
    )

    top_rules.to_csv(
        top_rules_file,
        index=False
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    meta.to_csv(
        meta_file,
        index=False
    )

    (
        research
        .groupby(
            [
                "symbol",
                "trend_class",
            ]
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
        .to_csv(
            target_dist_file,
            index=False
        )
    )

    print()
    print("=" * 120)
    print("TOP VALIDATED RULES")
    print("=" * 120)

    if top_rules.empty:
        print(
            "No rule passed the broad coverage filter."
        )
    else:
        print(
            top_rules[
                [
                    "direction",
                    "condition",
                    "test_n",
                    "test_precision",
                    "test_base_rate",
                    "test_lift",
                    "test_wilson_lb",
                    "test_avg_return_atr",
                    "test_coin_count",
                    "stability_ratio",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(
        f"Feature matrix: {matrix_file}"
    )
    print(
        f"Univariate:     {univariate_file}"
    )
    print(
        f"All rules:      {rules_file}"
    )
    print(
        f"Top rules:      {top_rules_file}"
    )
    print(
        f"Importance:     {importance_file}"
    )
    print(
        f"Model summary:  {meta_file}"
    )
    print(
        f"Target dist:    {target_dist_file}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
