import gc
import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = "/home/botadmin/crypto-bot"
DATA_DIR = os.path.join(BASE_DIR, "data", "trend_research_3m")
RAW_DIR = os.path.join(DATA_DIR, "raw_5m")
FEATURE_DIR = os.path.join(DATA_DIR, "features_90d_lowmem")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

RESEARCH_DAYS = 90
TRAIN_DAYS = 60
HORIZON_5M_BARS = 12
TREND_ATR_THRESHOLD = 0.50

TREE_MAX_DEPTH = 4
TREE_MIN_LEAF_FRACTION = 0.004
TREE_MIN_LEAF_ABS = 400
TREE_RANDOM_STATE = 42

UNIVARIATE_BINS = 5

# Cross-sectional ranks are useful, but are calculated in a separate,
# narrow pass so we never hold the full wide matrix twice in RAM.
RANK_BASES = [
    "5m_atr_pct",
    "5m_volume_ratio20",
    "5m_ema_spread_pct",
    "5m_price_vs_ema200_pct",
    "5m_trend_score",
    "1h_atr_pct",
    "1h_ema_spread_pct",
    "1h_price_vs_ema200_pct",
    "1h_trend_score",
    "4h_atr_pct",
    "4h_ema_spread_pct",
    "4h_price_vs_ema200_pct",
    "4h_trend_score",
]


def ensure_dirs():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def rma(series, length):
    return series.ewm(
        alpha=1.0 / length,
        adjust=False,
        min_periods=length,
    ).mean()


def true_range(df):
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def add_features(df):
    """Compact causal feature set matching the indicators already used by the project."""
    out = df.copy()

    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["atr"] = rma(true_range(out), 14)

    out["atr_pct"] = out["atr"] / out["close"] * 100.0

    vol20 = out["volume"].rolling(20, min_periods=20).mean()
    out["volume_ratio20"] = out["volume"] / vol20

    out["ema_spread_pct"] = (
        (out["ema50"] - out["ema200"]) / out["close"] * 100.0
    )
    out["price_vs_ema200_pct"] = (
        (out["close"] - out["ema200"]) / out["ema200"] * 100.0
    )

    jaw = rma(out["close"], 13).shift(8)
    teeth = rma(out["close"], 8).shift(5)
    lips = rma(out["close"], 5).shift(3)

    out["alligator_state"] = np.select(
        [
            (lips > teeth) & (teeth > jaw),
            (lips < teeth) & (teeth < jaw),
        ],
        [1, -1],
        default=0,
    ).astype(np.int8)

    out["alligator_spread_pct"] = (
        pd.concat([jaw, teeth, lips], axis=1).max(axis=1)
        - pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
    ) / out["close"] * 100.0

    # Williams fractal: a 5-bar pivot becomes known 2 bars later.
    raw_fh = (
        (out["high"] > out["high"].shift(1))
        & (out["high"] > out["high"].shift(2))
        & (out["high"] > out["high"].shift(-1))
        & (out["high"] > out["high"].shift(-2))
    )
    raw_fl = (
        (out["low"] < out["low"].shift(1))
        & (out["low"] < out["low"].shift(2))
        & (out["low"] < out["low"].shift(-1))
        & (out["low"] < out["low"].shift(-2))
    )

    last_fh = out["high"].where(raw_fh).shift(2).ffill()
    last_fl = out["low"].where(raw_fl).shift(2).ffill()

    out["dist_to_fractal_high_atr"] = (last_fh - out["close"]) / out["atr"]
    out["dist_to_fractal_low_atr"] = (out["close"] - last_fl) / out["atr"]
    out["break_above_fractal"] = (out["close"] > last_fh).astype(np.int8)
    out["break_below_fractal"] = (out["close"] < last_fl).astype(np.int8)

    out["market_regime"] = np.select(
        [
            (out["close"] > out["ema200"]) & (out["ema50"] > out["ema200"]),
            (out["close"] < out["ema200"]) & (out["ema50"] < out["ema200"]),
        ],
        [1, -1],
        default=0,
    ).astype(np.int8)

    out["trend_score"] = (
        np.tanh(out["ema_spread_pct"] / 2.0) * 50.0
        + np.tanh(out["price_vs_ema200_pct"] / 4.0) * 30.0
        + out["alligator_state"].astype(float) * 20.0
    )

    atr_med = out["atr_pct"].rolling(50, min_periods=30).median()
    out["volatility_regime"] = np.select(
        [
            out["atr_pct"] > atr_med * 1.25,
            out["atr_pct"] < atr_med * 0.80,
        ],
        [1, -1],
        default=0,
    ).astype(np.int8)

    # Timeframe-native momentum context.
    out["ret_1bar_pct"] = out["close"].pct_change(1) * 100.0
    out["ret_3bar_pct"] = out["close"].pct_change(3) * 100.0
    out["ret_12bar_pct"] = out["close"].pct_change(12) * 100.0

    out["ema200_slope_1bar_pct"] = out["ema200"].pct_change(1) * 100.0
    out["ema200_slope_3bar_pct"] = out["ema200"].pct_change(3) * 100.0

    keep = [
        "timestamp",
        "atr_pct",
        "volume_ratio20",
        "ema_spread_pct",
        "price_vs_ema200_pct",
        "alligator_state",
        "alligator_spread_pct",
        "dist_to_fractal_high_atr",
        "dist_to_fractal_low_atr",
        "break_above_fractal",
        "break_below_fractal",
        "market_regime",
        "trend_score",
        "volatility_regime",
        "ret_1bar_pct",
        "ret_3bar_pct",
        "ret_12bar_pct",
        "ema200_slope_1bar_pct",
        "ema200_slope_3bar_pct",
    ]
    return out[keep].copy()


def resample_ohlcv(df, rule):
    temp = df.set_index("timestamp").sort_index()

    agg = temp.resample(
        rule,
        label="left",
        closed="left",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "turnover": "sum",
        }
    )

    return (
        agg.dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )


def prefix_available(features, prefix, bar_minutes):
    out = features.copy()
    out["available_at"] = (
        out["timestamp"] + pd.Timedelta(minutes=bar_minutes)
    )
    out = out.drop(columns=["timestamp"])

    out = out.rename(
        columns={
            c: f"{prefix}_{c}"
            for c in out.columns
            if c != "available_at"
        }
    )
    return out


def add_outcomes(raw):
    base = raw[
        ["timestamp", "open", "high", "low", "close"]
    ].copy()

    entry = base["open"].shift(-1)
    final_close = base["close"].shift(-HORIZON_5M_BARS)

    future_high = pd.concat(
        [
            base["high"].shift(-k)
            for k in range(1, HORIZON_5M_BARS + 1)
        ],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [
            base["low"].shift(-k)
            for k in range(1, HORIZON_5M_BARS + 1)
        ],
        axis=1,
    ).min(axis=1)

    base["available_at"] = (
        base["timestamp"] + pd.Timedelta(minutes=5)
    )
    base["future_1h_return_pct"] = (
        final_close / entry - 1.0
    ) * 100.0
    base["future_1h_mfe_pct"] = (
        future_high / entry - 1.0
    ) * 100.0
    base["future_1h_mae_pct"] = (
        future_low / entry - 1.0
    ) * 100.0

    return base[
        [
            "timestamp",
            "available_at",
            "future_1h_return_pct",
            "future_1h_mfe_pct",
            "future_1h_mae_pct",
        ]
    ]


def asof_merge(base, feat):
    return pd.merge_asof(
        base.sort_values("available_at"),
        feat.sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )


def load_raw(path):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_tf_feature_frames(raw):
    f5_native = add_features(raw)
    f1_native = add_features(resample_ohlcv(raw, "1h"))
    f4_native = add_features(resample_ohlcv(raw, "4h"))

    f5 = prefix_available(f5_native, "5m", 5)
    f1 = prefix_available(f1_native, "1h", 60)
    f4 = prefix_available(f4_native, "4h", 240)

    return f5, f1, f4


def btc_context_columns():
    wanted = [
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
    return wanted


def build_btc_context(btc_raw):
    f5, f1, f4 = build_tf_feature_frames(btc_raw)

    # Build a 5m clock from BTC itself.
    base = pd.DataFrame(
        {
            "available_at": (
                btc_raw["timestamp"]
                + pd.Timedelta(minutes=5)
            )
        }
    )

    ctx = asof_merge(base, f5)
    ctx = asof_merge(ctx, f1)
    ctx = asof_merge(ctx, f4)

    wanted = [
        c for c in btc_context_columns()
        if c in ctx.columns
    ]

    ctx = ctx[
        ["available_at"] + wanted
    ].copy()

    ctx = ctx.rename(
        columns={
            c: f"btc_{c}"
            for c in wanted
        }
    )

    return ctx


def add_btc_rolling_correlations(frame):
    # 5m correlation: 50 x 5m = 250 minutes.
    if (
        "5m_ret_1bar_pct" in frame.columns
        and "btc_5m_ret_1bar_pct" in frame.columns
    ):
        frame["btc_correlation_5m_50"] = (
            frame["5m_ret_1bar_pct"]
            .rolling(50, min_periods=30)
            .corr(frame["btc_5m_ret_1bar_pct"])
        )

    # Higher-TF correlations are calculated only once per unique
    # completed TF state and forward-filled back to the 5m clock.
    for tf, freq in [("1h", "1h"), ("4h", "4h")]:
        coin_col = f"{tf}_ret_1bar_pct"
        btc_col = f"btc_{tf}_ret_1bar_pct"

        if coin_col not in frame.columns or btc_col not in frame.columns:
            continue

        tmp = frame[
            ["available_at", coin_col, btc_col]
        ].copy()

        tmp["_key"] = tmp["available_at"].dt.floor(freq)

        compact = (
            tmp.drop_duplicates("_key", keep="last")
            .sort_values("_key")
            .reset_index(drop=True)
        )

        compact[f"btc_correlation_{tf}_50"] = (
            compact[coin_col]
            .rolling(50, min_periods=30)
            .corr(compact[btc_col])
        )

        frame["_key"] = frame["available_at"].dt.floor(freq)

        frame = frame.merge(
            compact[
                ["_key", f"btc_correlation_{tf}_50"]
            ],
            on="_key",
            how="left",
            validate="many_to_one",
        ).drop(columns=["_key"])

    return frame


def downcast_frame(df):
    for c in df.columns:
        if c in ["timestamp", "available_at", "symbol", "trend_class"]:
            continue

        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c],
                downcast="float",
            )

        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c],
                downcast="integer",
            )

    return df


def build_one_symbol(
    symbol,
    raw,
    btc_context,
    research_start,
    research_end,
):
    f5, f1, f4 = build_tf_feature_frames(raw)

    base = add_outcomes(raw)

    frame = asof_merge(base, f5)
    frame = asof_merge(frame, f1)
    frame = asof_merge(frame, f4)
    frame = asof_merge(frame, btc_context)

    if (
        "5m_ret_12bar_pct" in frame.columns
        and "btc_5m_ret_12bar_pct" in frame.columns
    ):
        frame["relative_strength_1h_vs_btc"] = (
            frame["5m_ret_12bar_pct"]
            - frame["btc_5m_ret_12bar_pct"]
        )

    if (
        "1h_ret_3bar_pct" in frame.columns
        and "btc_1h_ret_3bar_pct" in frame.columns
    ):
        frame["relative_strength_3h_vs_btc"] = (
            frame["1h_ret_3bar_pct"]
            - frame["btc_1h_ret_3bar_pct"]
        )

    frame = add_btc_rolling_correlations(frame)

    # Keep only actual research window after all warmup calculations.
    frame = frame[
        (frame["available_at"] >= research_start)
        & (frame["available_at"] <= research_end)
    ].copy()

    frame["symbol"] = symbol

    if "1h_atr_pct" not in frame.columns:
        raise RuntimeError(
            f"{symbol}: 1h_atr_pct missing."
        )

    threshold = (
        frame["1h_atr_pct"]
        * TREND_ATR_THRESHOLD
    )

    frame["future_1h_return_atr"] = (
        frame["future_1h_return_pct"]
        / frame["1h_atr_pct"]
    )
    frame["future_1h_mfe_atr"] = (
        frame["future_1h_mfe_pct"]
        / frame["1h_atr_pct"]
    )
    frame["future_1h_mae_atr"] = (
        frame["future_1h_mae_pct"]
        / frame["1h_atr_pct"]
    )

    frame["trend_class"] = np.select(
        [
            frame["future_1h_return_pct"] >= threshold,
            frame["future_1h_return_pct"] <= -threshold,
        ],
        ["UP", "DOWN"],
        default="FLAT",
    )

    frame["target_up"] = (
        frame["trend_class"] == "UP"
    ).astype(np.int8)
    frame["target_down"] = (
        frame["trend_class"] == "DOWN"
    ).astype(np.int8)

    # Drop base signal-bar timestamp; available_at is the correct research clock.
    frame = frame.drop(columns=["timestamp"])

    # Remove raw BTC-only duplicated 5m price-scale features if any.
    frame = downcast_frame(frame)

    return frame


def add_rank_pass(feature_files):
    print()
    print("=" * 110)
    print("CROSS-SECTIONAL RANK PASS")
    print("=" * 110)

    narrow_parts = []

    for path in feature_files:
        df = pd.read_parquet(path)
        symbol = os.path.basename(path).replace("_features.parquet", "")

        cols = [
            c for c in RANK_BASES
            if c in df.columns
        ]

        part = df[
            ["available_at"] + cols
        ].copy()
        part["symbol"] = symbol
        narrow_parts.append(part)

    ranks = pd.concat(
        narrow_parts,
        ignore_index=True,
    )
    del narrow_parts
    gc.collect()

    for col in RANK_BASES:
        if col not in ranks.columns:
            continue

        ranks[f"rank_{col}"] = (
            ranks.groupby("available_at")[col]
            .rank(
                pct=True,
                method="average",
            )
            .astype(np.float32)
        )

    rank_cols = [
        c for c in ranks.columns
        if c.startswith("rank_")
    ]

    for path in feature_files:
        symbol = os.path.basename(path).replace("_features.parquet", "")
        df = pd.read_parquet(path)

        small = ranks.loc[
            ranks["symbol"] == symbol,
            ["available_at"] + rank_cols,
        ]

        df = df.merge(
            small,
            on="available_at",
            how="left",
            validate="one_to_one",
        )

        df = downcast_frame(df)
        df.to_parquet(path, index=False)

        del df, small
        gc.collect()

    del ranks
    gc.collect()


def feature_columns(df):
    exclude = {
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

    for c in df.columns:
        if c in exclude:
            continue
        if c in ["available_at", "symbol", "trend_class"]:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)

    return sorted(cols)


def wilson_lower_bound(wins, n, z=1.96):
    if n <= 0:
        return np.nan

    p = wins / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(
        p * (1 - p) / n
        + z * z / (4 * n * n)
    )
    return (center - margin) / den


def scan_univariate(train, test, features):
    rows = []

    for idx, feature in enumerate(features, start=1):
        if idx % 10 == 0:
            print(
                f"  univariate {idx}/{len(features)}"
            )

        tr_values = pd.to_numeric(
            train[feature],
            errors="coerce",
        )

        valid = tr_values.dropna()

        if valid.nunique() < 8:
            continue

        try:
            _, edges = pd.qcut(
                valid,
                q=UNIVARIATE_BINS,
                retbins=True,
                duplicates="drop",
            )
        except Exception:
            continue

        edges = np.unique(edges)

        if len(edges) < 3:
            continue

        edges[0] = -np.inf
        edges[-1] = np.inf

        for b in range(len(edges) - 1):
            lo = edges[b]
            hi = edges[b + 1]

            tr_mask = (
                (train[feature] > lo)
                & (train[feature] <= hi)
            )
            te_mask = (
                (test[feature] > lo)
                & (test[feature] <= hi)
            )

            tr_n = int(tr_mask.sum())
            te_n = int(te_mask.sum())

            if tr_n < 300 or te_n < 100:
                continue

            for direction, target in [
                ("UP", "target_up"),
                ("DOWN", "target_down"),
            ]:
                tr_wins = int(
                    train.loc[tr_mask, target].sum()
                )
                te_wins = int(
                    test.loc[te_mask, target].sum()
                )

                tr_precision = tr_wins / tr_n
                te_precision = te_wins / te_n
                base = float(test[target].mean())

                te_returns = test.loc[
                    te_mask,
                    "future_1h_return_atr",
                ]

                coin_count = int(
                    test.loc[
                        te_mask,
                        "symbol",
                    ].nunique()
                )

                rows.append(
                    {
                        "feature": feature,
                        "bin": b + 1,
                        "condition":
                            f"{feature} > {lo:.6g} AND <= {hi:.6g}",
                        "direction": direction,
                        "train_n": tr_n,
                        "train_precision": tr_precision,
                        "test_n": te_n,
                        "test_precision": te_precision,
                        "test_base_rate": base,
                        "test_lift":
                            te_precision / base
                            if base > 0
                            else np.nan,
                        "test_wilson_lb":
                            wilson_lower_bound(
                                te_wins,
                                te_n,
                            ),
                        "test_avg_return_atr":
                            float(te_returns.mean()),
                        "test_median_return_atr":
                            float(te_returns.median()),
                        "test_coin_count": coin_count,
                    }
                )

    return pd.DataFrame(rows)


def tree_paths(tree, feature_names):
    t = tree.tree_
    result = {}

    def walk(node, conds):
        left = t.children_left[node]
        right = t.children_right[node]

        if left == right:
            result[node] = list(conds)
            return

        idx = t.feature[node]
        threshold = t.threshold[node]
        name = feature_names[idx]

        walk(
            left,
            conds + [(name, "<=", threshold)],
        )
        walk(
            right,
            conds + [(name, ">", threshold)],
        )

    walk(0, [])
    return result


def condition_text(conditions):
    return " AND ".join(
        [
            f"{name} {op} {value:.6g}"
            for name, op, value in conditions
        ]
    )


def evaluate_tree(train, test, features, target, direction):
    medians = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .median()
        .astype(np.float32)
    )

    x_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .astype(np.float32)
    )
    y_train = train[target].astype(np.int8)

    min_leaf = max(
        TREE_MIN_LEAF_ABS,
        int(len(train) * TREE_MIN_LEAF_FRACTION),
    )

    model = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=TREE_RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    train_leaf = model.apply(x_train)
    del x_train
    gc.collect()

    x_test = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .astype(np.float32)
    )

    test_leaf = model.apply(x_test)

    auc = np.nan
    if test[target].nunique() > 1:
        try:
            auc = roc_auc_score(
                test[target],
                model.predict_proba(x_test)[:, 1],
            )
        except Exception:
            pass

    del x_test
    gc.collect()

    paths = tree_paths(
        model,
        features,
    )

    base_train = float(train[target].mean())
    base_test = float(test[target].mean())

    rules = []

    # Compact leaf/timestamp arrays for 14d stability without rebuilding X.
    stability_train = pd.DataFrame(
        {
            "available_at": train["available_at"].values,
            "target": train[target].values,
            "leaf": train_leaf,
        }
    )
    stability_test = pd.DataFrame(
        {
            "available_at": test["available_at"].values,
            "target": test[target].values,
            "leaf": test_leaf,
        }
    )
    stability = pd.concat(
        [stability_train, stability_test],
        ignore_index=True,
    )
    start = stability["available_at"].min()
    stability["_block"] = (
        (
            stability["available_at"]
            - start
        ).dt.days // 14
    )

    for leaf_id, conditions in paths.items():
        tr_mask = train_leaf == leaf_id
        te_mask = test_leaf == leaf_id

        tr_n = int(tr_mask.sum())
        te_n = int(te_mask.sum())

        if tr_n < min_leaf or te_n < 100:
            continue

        tr_wins = int(
            train.loc[tr_mask, target].sum()
        )
        te_wins = int(
            test.loc[te_mask, target].sum()
        )

        tr_precision = tr_wins / tr_n
        te_precision = te_wins / te_n

        block_stats = (
            stability.loc[
                stability["leaf"] == leaf_id
            ]
            .groupby("_block")["target"]
            .agg(["count", "mean"])
        )

        eligible = int(
            (block_stats["count"] >= 40).sum()
        )
        stable = int(
            (
                (block_stats["count"] >= 40)
                & (block_stats["mean"] > base_test)
            ).sum()
        )

        te_returns = test.loc[
            te_mask,
            "future_1h_return_atr",
        ]

        rules.append(
            {
                "direction": direction,
                "leaf_id": int(leaf_id),
                "condition": condition_text(conditions),
                "train_n": tr_n,
                "train_precision": tr_precision,
                "train_base_rate": base_train,
                "train_lift":
                    tr_precision / base_train
                    if base_train > 0
                    else np.nan,
                "test_n": te_n,
                "test_precision": te_precision,
                "test_base_rate": base_test,
                "test_lift":
                    te_precision / base_test
                    if base_test > 0
                    else np.nan,
                "test_wilson_lb":
                    wilson_lower_bound(
                        te_wins,
                        te_n,
                    ),
                "test_avg_return_atr":
                    float(te_returns.mean()),
                "test_median_return_atr":
                    float(te_returns.median()),
                "test_avg_mfe_atr":
                    float(
                        test.loc[
                            te_mask,
                            "future_1h_mfe_atr",
                        ].mean()
                    ),
                "test_avg_mae_atr":
                    float(
                        test.loc[
                            te_mask,
                            "future_1h_mae_atr",
                        ].mean()
                    ),
                "test_coin_count":
                    int(
                        test.loc[
                            te_mask,
                            "symbol",
                        ].nunique()
                    ),
                "stable_14d_blocks": stable,
                "eligible_14d_blocks": eligible,
                "stability_ratio":
                    stable / eligible
                    if eligible > 0
                    else np.nan,
            }
        )

    meta = pd.DataFrame(
        [
            {
                "direction": direction,
                "target": target,
                "train_rows": len(train),
                "test_rows": len(test),
                "min_samples_leaf": min_leaf,
                "max_depth": TREE_MAX_DEPTH,
                "train_base_rate": base_train,
                "test_base_rate": base_test,
                "test_auc": auc,
                "tree_features_used":
                    int(
                        (
                            model.feature_importances_
                            > 0
                        ).sum()
                    ),
            }
        ]
    )

    importance = pd.DataFrame(
        {
            "direction": direction,
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    return (
        pd.DataFrame(rules),
        meta,
        importance,
    )


def main():
    ensure_dirs()

    raw_files = sorted(
        glob.glob(
            os.path.join(
                RAW_DIR,
                "*_5m_120d.parquet",
            )
        )
    )

    if not raw_files:
        raise RuntimeError(
            f"No raw files found in {RAW_DIR}"
        )

    btc_path = os.path.join(
        RAW_DIR,
        "BTCUSDT_5m_120d.parquet",
    )

    if not os.path.exists(btc_path):
        raise RuntimeError(
            "BTCUSDT raw file is required for BTC context."
        )

    print()
    print("=" * 118)
    print("TOP30 TREND DISCOVERY - LOW MEMORY VERSION")
    print("=" * 118)
    print(f"Raw files: {len(raw_files)}")
    print("Research clock: each closed 5m candle")
    print("Features: compact 5m + 1h + 4h + BTC context + BTC correlations + cross-sectional ranks")
    print("Prediction: next 60 minutes")
    print("Split: first 60d discovery / last 30d validation")
    print("Intermediate symbol matrices are written to disk immediately.")
    print("=" * 118)

    # Common research end = oldest last timestamp across all symbols,
    # so cross-sectional comparisons do not silently use future availability
    # from one symbol beyond another.
    end_candidates = []

    for path in raw_files:
        tmp = pd.read_parquet(
            path,
            columns=["timestamp"],
        )
        ts = pd.to_datetime(
            tmp["timestamp"],
            utc=True,
            errors="coerce",
        ).dropna()

        if not ts.empty:
            end_candidates.append(ts.max())

        del tmp, ts

    if not end_candidates:
        raise RuntimeError("No valid timestamps.")

    research_end = min(end_candidates)
    research_start = (
        research_end
        - pd.Timedelta(days=RESEARCH_DAYS)
    )
    train_end = (
        research_start
        + pd.Timedelta(days=TRAIN_DAYS)
    )

    print()
    print(f"Research:   {research_start} -> {research_end}")
    print(f"Train/Test: {train_end}")

    print()
    print("Building BTC context...")
    btc_raw = load_raw(btc_path)
    btc_context = build_btc_context(btc_raw)
    del btc_raw
    gc.collect()

    feature_files = []

    for idx, path in enumerate(raw_files, start=1):
        symbol = (
            os.path.basename(path)
            .replace("_5m_120d.parquet", "")
        )

        print(
            f"[{idx:02d}/{len(raw_files):02d}] "
            f"{symbol}"
        )

        raw = load_raw(path)

        frame = build_one_symbol(
            symbol=symbol,
            raw=raw,
            btc_context=btc_context,
            research_start=research_start,
            research_end=research_end,
        )

        out_path = os.path.join(
            FEATURE_DIR,
            f"{symbol}_features.parquet",
        )
        frame.to_parquet(
            out_path,
            index=False,
        )
        feature_files.append(out_path)

        mem_mb = (
            frame.memory_usage(
                deep=True
            ).sum()
            / 1024**2
        )

        print(
            f"  rows={len(frame):,} | "
            f"cols={len(frame.columns)} | "
            f"RAM={mem_mb:.1f} MB -> disk"
        )

        del raw, frame
        gc.collect()

    del btc_context
    gc.collect()

    # Narrow rank pass.
    add_rank_pass(feature_files)

    print()
    print("=" * 118)
    print("COMBINING COMPACT 90D MATRICES")
    print("=" * 118)

    parts = []

    for idx, path in enumerate(feature_files, start=1):
        df = pd.read_parquet(path)
        parts.append(df)

        if idx % 5 == 0 or idx == len(feature_files):
            print(
                f"  loaded {idx}/{len(feature_files)}"
            )

    research = pd.concat(
        parts,
        ignore_index=True,
    )
    del parts
    gc.collect()

    research = research[
        research["future_1h_return_pct"].notna()
        & research["1h_atr_pct"].notna()
        & (research["1h_atr_pct"] > 0)
    ].copy()

    research["symbol"] = research["symbol"].astype("category")
    research["trend_class"] = research["trend_class"].astype("category")

    mem_mb = (
        research.memory_usage(
            deep=True
        ).sum()
        / 1024**2
    )

    print(
        f"Combined rows={len(research):,} | "
        f"cols={len(research.columns)} | "
        f"RAM={mem_mb:.1f} MB"
    )

    train = research[
        research["available_at"] < train_end
    ].copy()

    test = research[
        research["available_at"] >= train_end
    ].copy()

    features = feature_columns(research)

    usable = []

    for col in features:
        s = train[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if s.notna().sum() < 1000:
            continue
        if s.nunique(dropna=True) < 3:
            continue

        usable.append(col)

    features = usable

    print()
    print("=" * 118)
    print("RESEARCH MATRIX")
    print("=" * 118)
    print(f"Rows:          {len(research):,}")
    print(f"Train rows:    {len(train):,}")
    print(f"Validation:    {len(test):,}")
    print(f"Symbols:       {research['symbol'].nunique()}")
    print(f"Features:      {len(features)}")
    print()
    print("Validation target distribution:")
    print(
        test["trend_class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print()
    print("Running univariate condition scan...")
    univariate = scan_univariate(
        train,
        test,
        features,
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
            ],
        )

    print()
    print("Training interpretable UP combination tree...")
    rules_up, meta_up, imp_up = evaluate_tree(
        train=train,
        test=test,
        features=features,
        target="target_up",
        direction="UP",
    )
    gc.collect()

    print("Training interpretable DOWN combination tree...")
    rules_down, meta_down, imp_down = evaluate_tree(
        train=train,
        test=test,
        features=features,
        target="target_down",
        direction="DOWN",
    )
    gc.collect()

    rules = pd.concat(
        [rules_up, rules_down],
        ignore_index=True,
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
            ],
        )

    top_rules = rules[
        (rules["test_n"] >= 200)
        & (
            rules["test_coin_count"]
            >= max(
                10,
                int(
                    research["symbol"].nunique()
                    * 0.4
                ),
            )
        )
    ].copy()

    if not top_rules.empty:
        top_rules = (
            top_rules
            .groupby(
                "direction",
                group_keys=False,
            )
            .head(20)
        )

    meta = pd.concat(
        [meta_up, meta_down],
        ignore_index=True,
    )

    importance = pd.concat(
        [imp_up, imp_down],
        ignore_index=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    compact_matrix = os.path.join(
        DATA_DIR,
        f"trend_feature_matrix_90d_lowmem_{stamp}.parquet",
    )
    univariate_file = os.path.join(
        REPORT_DIR,
        f"trend_univariate_conditions_lowmem_{stamp}.csv",
    )
    rules_file = os.path.join(
        REPORT_DIR,
        f"trend_combination_rules_lowmem_{stamp}.csv",
    )
    top_file = os.path.join(
        REPORT_DIR,
        f"trend_top_validated_rules_lowmem_{stamp}.csv",
    )
    importance_file = os.path.join(
        REPORT_DIR,
        f"trend_rule_feature_importance_lowmem_{stamp}.csv",
    )
    meta_file = os.path.join(
        REPORT_DIR,
        f"trend_rule_model_summary_lowmem_{stamp}.csv",
    )
    dist_file = os.path.join(
        REPORT_DIR,
        f"trend_target_distribution_lowmem_{stamp}.csv",
    )

    research.to_parquet(
        compact_matrix,
        index=False,
    )
    univariate.to_csv(
        univariate_file,
        index=False,
    )
    rules.to_csv(
        rules_file,
        index=False,
    )
    top_rules.to_csv(
        top_file,
        index=False,
    )
    importance.to_csv(
        importance_file,
        index=False,
    )
    meta.to_csv(
        meta_file,
        index=False,
    )

    (
        research.groupby(
            ["symbol", "trend_class"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
        .to_csv(
            dist_file,
            index=False,
        )
    )

    print()
    print("=" * 118)
    print("TOP VALIDATED RULES")
    print("=" * 118)

    if top_rules.empty:
        print("No broad rule passed the coverage filter.")
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
            .head(20)
            .to_string(index=False)
        )

    print()
    print("=" * 118)
    print("FILES")
    print("=" * 118)
    print(f"Matrix:      {compact_matrix}")
    print(f"Univariate:  {univariate_file}")
    print(f"All rules:   {rules_file}")
    print(f"Top rules:   {top_file}")
    print(f"Importance:  {importance_file}")
    print(f"Model meta:  {meta_file}")
    print(f"Target dist: {dist_file}")
    print("=" * 118)


if __name__ == "__main__":
    main()
