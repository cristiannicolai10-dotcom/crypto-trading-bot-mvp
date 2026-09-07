import gc
import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = "/home/botadmin/crypto-bot"
DATA_DIR = os.path.join(BASE_DIR, "data", "trend_research_3m")
FEATURE_DIR = os.path.join(DATA_DIR, "features_90d_lowmem")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

TRAIN_DAYS = 60
EVENT_COOLDOWN_MINUTES = 60

# Fixed market-trend definition. These are not optimized in this script.
MARKET_MOVE_ATR = 0.25
MARKET_BREADTH_LONG = 0.60
MARKET_BREADTH_SHORT = 0.40

# Tree discovery is deliberately shallow and interpretable.
TREE_DEPTHS = [3, 4]
TREE_MIN_LEAF_ABS = 300
TREE_MIN_LEAF_FRACTION = 0.02
TREE_RANDOM_STATE = 42

# Candidate gates use discovery only.
MIN_DISCOVERY_COOLDOWN_EVENTS = 8
MIN_DISCOVERY_NEW_EPISODE_EVENTS = 5
MIN_DISCOVERY_RAW_LIFT = 1.15
MIN_DISCOVERY_STABILITY_RATIO = 0.50

# Frozen Trigger A, used only as an overlap benchmark.
TRIGGER_A_BTC_12H_RETURN_MAX = 0.803982
TRIGGER_A_BTC_5M_ATR_MIN = 0.206836
TRIGGER_A_BTC_5M_REGIME_MIN = 0.5
TRIGGER_A_BTC_5M_PRICE_EMA_MIN = 1.2227


# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def wilson_lower_bound(wins, n, z=1.96):
    if n <= 0:
        return np.nan

    p = wins / n
    den = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt(
        p * (1.0 - p) / n
        + z * z / (4.0 * n * n)
    )

    return (center - margin) / den


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def event_times_from_mask(timestamps, mask, mode):
    temp = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                timestamps,
                utc=True,
                errors="coerce",
            ),
            "active": pd.Series(mask).astype(bool).values,
        }
    ).dropna(subset=["available_at"])

    temp = temp.sort_values("available_at").reset_index(drop=True)

    if mode == "NEW_EPISODE":
        rising = (
            temp["active"]
            & ~temp["active"].shift(1, fill_value=False)
        )
        return temp.loc[
            rising,
            "available_at",
        ].tolist()

    if mode != "COOLDOWN_60M":
        raise ValueError(f"Unknown event mode: {mode}")

    active_times = temp.loc[
        temp["active"],
        "available_at",
    ].tolist()

    accepted = []
    last = None
    cooldown = pd.Timedelta(
        minutes=EVENT_COOLDOWN_MINUTES
    )

    for ts in active_times:
        if last is None or ts >= last + cooldown:
            accepted.append(ts)
            last = ts

    return accepted


# ============================================================
# LOAD / BUILD MARKET-LEVEL MATRIX
# ============================================================

def feature_files():
    files = sorted(
        glob.glob(
            os.path.join(
                FEATURE_DIR,
                "*_features.parquet",
            )
        )
    )

    if not files:
        raise RuntimeError(
            f"No feature files found in {FEATURE_DIR}"
        )

    return files


def available_columns(path):
    # Reading a single compact symbol file is cheap and avoids
    # relying on pyarrow internals.
    df = pd.read_parquet(path)
    cols = list(df.columns)
    del df
    gc.collect()
    return cols


def choose_panel_columns(columns):
    required = [
        "available_at",
        "symbol",
        "future_1h_return_atr",
        "future_1h_mfe_atr",
        "future_1h_mae_atr",
        "trend_class",
    ]

    for c in required:
        if c not in columns:
            raise RuntimeError(
                f"Required feature column missing: {c}"
            )

    wanted = list(required)

    # Current, causal market-state features only.
    exact_feature_suffixes = {
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
    }

    for tf in ["5m", "1h", "4h"]:
        for suffix in sorted(exact_feature_suffixes):
            col = f"{tf}_{suffix}"
            if col in columns:
                wanted.append(col)

    return list(dict.fromkeys(wanted))


def build_coin_panel(files):
    first_cols = available_columns(files[0])
    panel_cols = choose_panel_columns(first_cols)

    parts = []

    print()
    print("=" * 120)
    print("BUILDING NARROW TOP30 MARKET PANEL")
    print("=" * 120)

    for idx, path in enumerate(files, start=1):
        symbol = (
            os.path.basename(path)
            .replace("_features.parquet", "")
        )

        df = pd.read_parquet(
            path,
            columns=panel_cols,
        )

        df["available_at"] = pd.to_datetime(
            df["available_at"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "available_at",
                "future_1h_return_atr",
            ]
        )

        # Ensure the actual symbol is correct even if category/string
        # conversion differs between parquet files.
        df["symbol"] = symbol

        # Downcast to keep RAM low.
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = pd.to_numeric(
                    df[c],
                    downcast="float",
                )

        parts.append(df)

        print(
            f"[{idx:02d}/{len(files):02d}] "
            f"{symbol}: {len(df):,} rows"
        )

    panel = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    return panel


def aggregate_market_state(panel):
    g = panel.groupby(
        "available_at",
        observed=True,
        sort=True,
    )

    base = g.agg(
        coins_available=(
            "symbol",
            "nunique",
        ),
        future_market_mean_return_atr=(
            "future_1h_return_atr",
            "mean",
        ),
        future_market_median_return_atr=(
            "future_1h_return_atr",
            "median",
        ),
        future_market_mean_mfe_atr=(
            "future_1h_mfe_atr",
            "mean",
        ),
        future_market_mean_mae_atr=(
            "future_1h_mae_atr",
            "mean",
        ),
    )

    # Future breadth targets.
    tmp = panel[
        [
            "available_at",
            "future_1h_return_atr",
            "trend_class",
        ]
    ].copy()

    tmp["_positive"] = (
        tmp["future_1h_return_atr"] > 0
    ).astype(np.float32)

    tmp["_negative"] = (
        tmp["future_1h_return_atr"] < 0
    ).astype(np.float32)

    tmp["_up"] = (
        tmp["trend_class"].astype(str) == "UP"
    ).astype(np.float32)

    tmp["_down"] = (
        tmp["trend_class"].astype(str) == "DOWN"
    ).astype(np.float32)

    breadth = (
        tmp
        .groupby(
            "available_at",
            observed=True,
        )
        .agg(
            future_positive_breadth=(
                "_positive",
                "mean",
            ),
            future_negative_breadth=(
                "_negative",
                "mean",
            ),
            future_up_breadth=(
                "_up",
                "mean",
            ),
            future_down_breadth=(
                "_down",
                "mean",
            ),
        )
    )

    market = base.join(
        breadth,
        how="inner",
    )

    del tmp, breadth
    gc.collect()

    # Cross-sectional current-market features.
    tf_list = ["5m", "1h", "4h"]

    for tf in tf_list:
        numeric_medians = [
            f"{tf}_atr_pct",
            f"{tf}_volume_ratio20",
            f"{tf}_ema_spread_pct",
            f"{tf}_price_vs_ema200_pct",
            f"{tf}_alligator_spread_pct",
            f"{tf}_dist_to_fractal_high_atr",
            f"{tf}_dist_to_fractal_low_atr",
            f"{tf}_trend_score",
            f"{tf}_ret_1bar_pct",
            f"{tf}_ret_3bar_pct",
            f"{tf}_ret_12bar_pct",
            f"{tf}_ema200_slope_1bar_pct",
            f"{tf}_ema200_slope_3bar_pct",
        ]

        for col in numeric_medians:
            if col not in panel.columns:
                continue

            grouped = panel.groupby(
                "available_at",
                observed=True,
            )[col]

            market[
                f"market_median_{col}"
            ] = grouped.median()

            market[
                f"market_mean_{col}"
            ] = grouped.mean()

            # Dispersion is useful for identifying broad vs narrow moves.
            if col in {
                f"{tf}_trend_score",
                f"{tf}_ret_1bar_pct",
                f"{tf}_ret_3bar_pct",
                f"{tf}_price_vs_ema200_pct",
            }:
                market[
                    f"market_std_{col}"
                ] = grouped.std()

        # Breadth-style current state.
        if f"{tf}_market_regime" in panel.columns:
            col = f"{tf}_market_regime"
            market[
                f"market_{tf}_bullish_regime_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] > 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

            market[
                f"market_{tf}_bearish_regime_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] < 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

        if f"{tf}_alligator_state" in panel.columns:
            col = f"{tf}_alligator_state"

            market[
                f"market_{tf}_bullish_alligator_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] > 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

            market[
                f"market_{tf}_bearish_alligator_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] < 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

        if f"{tf}_price_vs_ema200_pct" in panel.columns:
            col = f"{tf}_price_vs_ema200_pct"

            market[
                f"market_{tf}_above_ema200_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] > 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

        if f"{tf}_ret_1bar_pct" in panel.columns:
            col = f"{tf}_ret_1bar_pct"

            market[
                f"market_{tf}_positive_return_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] > 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

        if f"{tf}_break_above_fractal" in panel.columns:
            market[
                f"market_{tf}_break_above_fractal_breadth"
            ] = (
                panel.groupby(
                    "available_at",
                    observed=True,
                )[
                    f"{tf}_break_above_fractal"
                ]
                .mean()
            )

        if f"{tf}_break_below_fractal" in panel.columns:
            market[
                f"market_{tf}_break_below_fractal_breadth"
            ] = (
                panel.groupby(
                    "available_at",
                    observed=True,
                )[
                    f"{tf}_break_below_fractal"
                ]
                .mean()
            )

        if f"{tf}_volatility_regime" in panel.columns:
            col = f"{tf}_volatility_regime"

            market[
                f"market_{tf}_high_vol_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] > 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

            market[
                f"market_{tf}_low_vol_breadth"
            ] = (
                panel.assign(
                    _x=(
                        panel[col] < 0
                    ).astype(np.float32)
                )
                .groupby(
                    "available_at",
                    observed=True,
                )["_x"]
                .mean()
            )

    market = market.reset_index()

    return market


def add_btc_state(market, files):
    btc_path = None

    for path in files:
        if os.path.basename(path).startswith(
            "BTCUSDT_"
        ):
            btc_path = path
            break

    if btc_path is None:
        raise RuntimeError(
            "BTCUSDT feature file not found."
        )

    btc = pd.read_parquet(
        btc_path
    )

    btc["available_at"] = pd.to_datetime(
        btc["available_at"],
        utc=True,
        errors="coerce",
    )

    btc_cols = ["available_at"]

    for col in btc.columns:
        if col == "available_at":
            continue

        if col.startswith(
            ("5m_", "1h_", "4h_", "rank_")
        ):
            if any(
                bad in col
                for bad in [
                    "future_",
                    "target_",
                ]
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                btc[col]
            ):
                btc_cols.append(col)

    btc = btc[
        list(dict.fromkeys(btc_cols))
    ].drop_duplicates(
        "available_at",
        keep="last",
    )

    rename = {
        c: f"btc_{c}"
        for c in btc.columns
        if c != "available_at"
    }

    btc = btc.rename(
        columns=rename
    )

    out = market.merge(
        btc,
        on="available_at",
        how="left",
        validate="one_to_one",
    )

    return out


def add_targets_and_trigger_a(market):
    out = market.copy()

    out["target_market_long"] = (
        (
            out[
                "future_market_median_return_atr"
            ]
            >= MARKET_MOVE_ATR
        )
        &
        (
            out[
                "future_positive_breadth"
            ]
            >= MARKET_BREADTH_LONG
        )
    ).astype(np.int8)

    out["target_market_short"] = (
        (
            out[
                "future_market_median_return_atr"
            ]
            <= -MARKET_MOVE_ATR
        )
        &
        (
            out[
                "future_positive_breadth"
            ]
            <= MARKET_BREADTH_SHORT
        )
    ).astype(np.int8)

    required_a = [
        "btc_4h_ret_3bar_pct",
        "btc_5m_atr_pct",
        "btc_5m_market_regime",
        "btc_5m_price_vs_ema200_pct",
    ]

    if all(
        c in out.columns
        for c in required_a
    ):
        out["trigger_A_active"] = (
            (
                out[
                    "btc_4h_ret_3bar_pct"
                ]
                <= TRIGGER_A_BTC_12H_RETURN_MAX
            )
            &
            (
                out[
                    "btc_5m_atr_pct"
                ]
                > TRIGGER_A_BTC_5M_ATR_MIN
            )
            &
            (
                out[
                    "btc_5m_market_regime"
                ]
                > TRIGGER_A_BTC_5M_REGIME_MIN
            )
            &
            (
                out[
                    "btc_5m_price_vs_ema200_pct"
                ]
                > TRIGGER_A_BTC_5M_PRICE_EMA_MIN
            )
        )
    else:
        out["trigger_A_active"] = False

    return out


# ============================================================
# MODEL FEATURES
# ============================================================

def model_feature_columns(df):
    exclude = {
        "available_at",
        "coins_available",
        "future_market_mean_return_atr",
        "future_market_median_return_atr",
        "future_market_mean_mfe_atr",
        "future_market_mean_mae_atr",
        "future_positive_breadth",
        "future_negative_breadth",
        "future_up_breadth",
        "future_down_breadth",
        "target_market_long",
        "target_market_short",
        "trigger_A_active",
    }

    features = []

    for c in df.columns:
        if c in exclude:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            continue

        # Avoid raw price-scale values if any slipped into BTC state.
        if any(
            c.endswith(suffix)
            for suffix in [
                "_ema50",
                "_ema200",
                "_alligator_jaw",
                "_alligator_teeth",
                "_alligator_lips",
                "_last_fractal_high",
                "_last_fractal_low",
                "_atr",
            ]
        ):
            if not c.endswith(
                "_atr_pct"
            ):
                continue

        features.append(c)

    return sorted(set(features))


# ============================================================
# TREE RULES
# ============================================================

def tree_paths(model, feature_names):
    t = model.tree_
    result = {}

    def walk(node, conditions):
        left = t.children_left[node]
        right = t.children_right[node]

        if left == right:
            result[node] = list(conditions)
            return

        feature_idx = t.feature[node]
        threshold = t.threshold[node]
        feature_name = feature_names[
            feature_idx
        ]

        walk(
            left,
            conditions
            + [
                (
                    feature_name,
                    "<=",
                    threshold,
                )
            ],
        )

        walk(
            right,
            conditions
            + [
                (
                    feature_name,
                    ">",
                    threshold,
                )
            ],
        )

    walk(0, [])
    return result


def condition_text(conditions):
    return " AND ".join(
        [
            f"{name} {op} {value:.6g}"
            for name, op, value
            in conditions
        ]
    )


def directional_return(
    frame,
    direction,
):
    base = frame[
        "future_market_median_return_atr"
    ]

    if direction == "LONG":
        return base

    return -base


def event_metrics(
    frame,
    active_mask,
    target_col,
    direction,
    mode,
):
    event_times = event_times_from_mask(
        frame["available_at"],
        active_mask,
        mode,
    )

    if not event_times:
        return {
            "events": 0,
            "hit_rate": np.nan,
            "wilson_lb": np.nan,
            "mean_direction_return_atr": np.nan,
            "median_direction_return_atr": np.nan,
            "positive_direction_rate": np.nan,
            "mean_mfe_direction_atr": np.nan,
            "mean_mae_direction_atr": np.nan,
            "trigger_A_active_rate": np.nan,
        }, []

    event_frame = (
        frame
        .set_index("available_at")
        .loc[event_times]
        .reset_index()
    )

    dir_ret = directional_return(
        event_frame,
        direction,
    )

    if direction == "LONG":
        mfe = event_frame[
            "future_market_mean_mfe_atr"
        ]
        mae = event_frame[
            "future_market_mean_mae_atr"
        ]
    else:
        # For shorts, favorable excursion corresponds to
        # the magnitude of the negative MAE.
        mfe = -event_frame[
            "future_market_mean_mae_atr"
        ]
        mae = -event_frame[
            "future_market_mean_mfe_atr"
        ]

    wins = int(
        event_frame[
            target_col
        ].sum()
    )

    n = len(event_frame)

    result = {
        "events":
            n,
        "hit_rate":
            wins / n,
        "wilson_lb":
            wilson_lower_bound(
                wins,
                n,
            ),
        "mean_direction_return_atr":
            safe_mean(dir_ret),
        "median_direction_return_atr":
            safe_median(dir_ret),
        "positive_direction_rate":
            float(
                (dir_ret > 0).mean()
            ),
        "mean_mfe_direction_atr":
            safe_mean(mfe),
        "mean_mae_direction_atr":
            safe_mean(mae),
        "trigger_A_active_rate":
            float(
                event_frame[
                    "trigger_A_active"
                ].mean()
            ),
    }

    return result, event_times


def stability_metrics(
    train,
    active_mask,
    target_col,
    direction,
):
    temp = train.copy()
    temp["_active"] = pd.Series(
        active_mask,
        index=temp.index,
    ).astype(bool)

    start = temp[
        "available_at"
    ].min()

    temp["_block"] = (
        (
            temp[
                "available_at"
            ]
            - start
        )
        .dt.days
        // 15
    )

    base_rate = float(
        temp[
            target_col
        ].mean()
    )

    rows = []

    for block, group in temp.groupby(
        "_block",
        observed=True,
    ):
        active = group.loc[
            group["_active"]
        ]

        if len(active) == 0:
            continue

        hit = float(
            active[
                target_col
            ].mean()
        )

        dir_ret = safe_mean(
            directional_return(
                active,
                direction,
            )
        )

        rows.append(
            {
                "block":
                    int(block),
                "n":
                    len(active),
                "hit_rate":
                    hit,
                "mean_direction_return_atr":
                    dir_ret,
                "beats_base":
                    hit > base_rate,
                "positive_direction":
                    dir_ret > 0,
            }
        )

    if not rows:
        return {
            "blocks_with_signal": 0,
            "blocks_beating_base": 0,
            "blocks_positive_direction": 0,
            "stability_ratio": np.nan,
        }

    block_df = pd.DataFrame(rows)

    stable = (
        block_df[
            "beats_base"
        ]
        &
        block_df[
            "positive_direction"
        ]
    )

    return {
        "blocks_with_signal":
            len(block_df),
        "blocks_beating_base":
            int(
                block_df[
                    "beats_base"
                ].sum()
            ),
        "blocks_positive_direction":
            int(
                block_df[
                    "positive_direction"
                ].sum()
            ),
        "stability_ratio":
            float(
                stable.mean()
            ),
    }


def evaluate_leaf_candidate(
    candidate_id,
    model,
    leaf_id,
    conditions,
    medians,
    features,
    train,
    validation,
    target_col,
    direction,
    tree_depth,
):
    def leaf_mask(frame):
        x = (
            frame[
                features
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(
                medians
            )
            .astype(np.float32)
        )

        return (
            model.apply(x)
            == leaf_id
        )

    train_mask = leaf_mask(train)
    val_mask = leaf_mask(validation)

    train_active = train.loc[
        train_mask
    ]

    validation_active = validation.loc[
        val_mask
    ]

    base_train = float(
        train[
            target_col
        ].mean()
    )

    base_val = float(
        validation[
            target_col
        ].mean()
    )

    train_raw_n = len(train_active)
    val_raw_n = len(validation_active)

    train_raw_precision = (
        float(
            train_active[
                target_col
            ].mean()
        )
        if train_raw_n
        else np.nan
    )

    val_raw_precision = (
        float(
            validation_active[
                target_col
            ].mean()
        )
        if val_raw_n
        else np.nan
    )

    train_raw_lift = (
        train_raw_precision
        / base_train
        if (
            train_raw_n
            and base_train > 0
        )
        else np.nan
    )

    val_raw_lift = (
        val_raw_precision
        / base_val
        if (
            val_raw_n
            and base_val > 0
        )
        else np.nan
    )

    train_cooldown, train_cooldown_times = (
        event_metrics(
            train,
            train_mask,
            target_col,
            direction,
            "COOLDOWN_60M",
        )
    )

    train_episode, train_episode_times = (
        event_metrics(
            train,
            train_mask,
            target_col,
            direction,
            "NEW_EPISODE",
        )
    )

    val_cooldown, val_cooldown_times = (
        event_metrics(
            validation,
            val_mask,
            target_col,
            direction,
            "COOLDOWN_60M",
        )
    )

    val_episode, val_episode_times = (
        event_metrics(
            validation,
            val_mask,
            target_col,
            direction,
            "NEW_EPISODE",
        )
    )

    stability = stability_metrics(
        train,
        train_mask,
        target_col,
        direction,
    )

    record = {
        "candidate_id":
            candidate_id,
        "direction":
            direction,
        "tree_depth":
            tree_depth,
        "leaf_id":
            int(leaf_id),
        "condition":
            condition_text(
                conditions
            ),
        "condition_terms":
            len(conditions),

        "discovery_base_rate":
            base_train,
        "validation_base_rate":
            base_val,

        "discovery_raw_n":
            train_raw_n,
        "discovery_raw_precision":
            train_raw_precision,
        "discovery_raw_lift":
            train_raw_lift,
        "discovery_raw_mean_direction_return_atr":
            safe_mean(
                directional_return(
                    train_active,
                    direction,
                )
            ),

        "validation_raw_n":
            val_raw_n,
        "validation_raw_precision":
            val_raw_precision,
        "validation_raw_lift":
            val_raw_lift,
        "validation_raw_mean_direction_return_atr":
            safe_mean(
                directional_return(
                    validation_active,
                    direction,
                )
            ),
    }

    for prefix, metrics in [
        (
            "discovery_cooldown",
            train_cooldown,
        ),
        (
            "discovery_episode",
            train_episode,
        ),
        (
            "validation_cooldown",
            val_cooldown,
        ),
        (
            "validation_episode",
            val_episode,
        ),
    ]:
        for key, value in metrics.items():
            record[
                f"{prefix}_{key}"
            ] = value

    for key, value in stability.items():
        record[
            f"discovery_{key}"
        ] = value

    event_rows = []

    for split_name, mode_name, event_times in [
        (
            "DISCOVERY_60D",
            "COOLDOWN_60M",
            train_cooldown_times,
        ),
        (
            "DISCOVERY_60D",
            "NEW_EPISODE",
            train_episode_times,
        ),
        (
            "VALIDATION_30D",
            "COOLDOWN_60M",
            val_cooldown_times,
        ),
        (
            "VALIDATION_30D",
            "NEW_EPISODE",
            val_episode_times,
        ),
    ]:
        source = (
            train
            if split_name == "DISCOVERY_60D"
            else validation
        )

        if not event_times:
            continue

        ef = (
            source
            .set_index(
                "available_at"
            )
            .loc[
                event_times
            ]
            .reset_index()
        )

        dir_ret = directional_return(
            ef,
            direction,
        )

        for idx, row in ef.iterrows():
            event_rows.append(
                {
                    "candidate_id":
                        candidate_id,
                    "direction":
                        direction,
                    "split":
                        split_name,
                    "event_mode":
                        mode_name,
                    "available_at":
                        row[
                            "available_at"
                        ],
                    "coins_available":
                        row[
                            "coins_available"
                        ],
                    "target_hit":
                        row[
                            target_col
                        ],
                    "future_market_median_return_atr":
                        row[
                            "future_market_median_return_atr"
                        ],
                    "future_market_mean_return_atr":
                        row[
                            "future_market_mean_return_atr"
                        ],
                    "future_positive_breadth":
                        row[
                            "future_positive_breadth"
                        ],
                    "future_up_breadth":
                        row[
                            "future_up_breadth"
                        ],
                    "future_down_breadth":
                        row[
                            "future_down_breadth"
                        ],
                    "direction_return_atr":
                        dir_ret.iloc[
                            idx
                        ],
                    "trigger_A_active":
                        bool(
                            row[
                                "trigger_A_active"
                            ]
                        ),
                }
            )

    return record, event_rows


def discover_direction(
    train,
    validation,
    features,
    target_col,
    direction,
):
    candidates = []
    events = []
    importances = []

    for depth in TREE_DEPTHS:
        medians = (
            train[
                features
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .median()
            .astype(np.float32)
        )

        x_train = (
            train[
                features
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(
                medians
            )
            .astype(np.float32)
        )

        y_train = train[
            target_col
        ].astype(np.int8)

        min_leaf = max(
            TREE_MIN_LEAF_ABS,
            int(
                len(train)
                * TREE_MIN_LEAF_FRACTION
            ),
        )

        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=TREE_RANDOM_STATE,
        )

        model.fit(
            x_train,
            y_train,
        )

        leaves_train = model.apply(
            x_train
        )

        paths = tree_paths(
            model,
            features,
        )

        for feature, importance in zip(
            features,
            model.feature_importances_,
        ):
            importances.append(
                {
                    "direction":
                        direction,
                    "tree_depth":
                        depth,
                    "feature":
                        feature,
                    "importance":
                        importance,
                }
            )

        base_rate = float(
            y_train.mean()
        )

        for leaf_id, conditions in paths.items():
            leaf_mask = (
                leaves_train
                == leaf_id
            )

            leaf_n = int(
                leaf_mask.sum()
            )

            if leaf_n == 0:
                continue

            leaf_precision = float(
                y_train.loc[
                    leaf_mask
                ].mean()
            )

            leaf_lift = (
                leaf_precision
                / base_rate
                if base_rate > 0
                else np.nan
            )

            # Only positive-class leaves become trigger candidates.
            if (
                not np.isfinite(
                    leaf_lift
                )
                or leaf_lift
                < MIN_DISCOVERY_RAW_LIFT
            ):
                continue

            candidate_id = (
                f"{direction}_D{depth}_L{int(leaf_id)}"
            )

            record, event_rows = (
                evaluate_leaf_candidate(
                    candidate_id=
                        candidate_id,
                    model=
                        model,
                    leaf_id=
                        leaf_id,
                    conditions=
                        conditions,
                    medians=
                        medians,
                    features=
                        features,
                    train=
                        train,
                    validation=
                        validation,
                    target_col=
                        target_col,
                    direction=
                        direction,
                    tree_depth=
                        depth,
                )
            )

            candidates.append(
                record
            )

            events.extend(
                event_rows
            )

        del (
            x_train,
            y_train,
            leaves_train,
            model,
        )
        gc.collect()

    return (
        pd.DataFrame(candidates),
        pd.DataFrame(events),
        pd.DataFrame(importances),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()

    files = feature_files()

    print()
    print("=" * 120)
    print("MARKET TRIGGER LIBRARY DISCOVERY")
    print("=" * 120)
    print(f"Top30 feature files: {len(files)}")
    print("Goal: discover additional LONG/SHORT market states.")
    print("Coin selector is NOT optimized here.")
    print("Validation 30d is never used to fit thresholds.")
    print()
    print(
        f"LONG target: market median +1h return >= "
        f"+{MARKET_MOVE_ATR:.2f} ATR "
        f"AND positive breadth >= "
        f"{MARKET_BREADTH_LONG:.0%}"
    )
    print(
        f"SHORT target: market median +1h return <= "
        f"-{MARKET_MOVE_ATR:.2f} ATR "
        f"AND positive breadth <= "
        f"{MARKET_BREADTH_SHORT:.0%}"
    )
    print("=" * 120)

    panel = build_coin_panel(
        files
    )

    market = aggregate_market_state(
        panel
    )

    del panel
    gc.collect()

    market = add_btc_state(
        market,
        files
    )

    market = add_targets_and_trigger_a(
        market
    )

    market = (
        market
        .dropna(
            subset=[
                "available_at",
                "future_market_median_return_atr",
            ]
        )
        .sort_values(
            "available_at"
        )
        .reset_index(
            drop=True
        )
    )

    # Require a meaningful cross-section.
    market = market.loc[
        market[
            "coins_available"
        ]
        >= 20
    ].copy()

    research_start = market[
        "available_at"
    ].min()

    train_end = (
        research_start
        + pd.Timedelta(
            days=TRAIN_DAYS
        )
    )

    train = market.loc[
        market[
            "available_at"
        ]
        < train_end
    ].copy()

    validation = market.loc[
        market[
            "available_at"
        ]
        >= train_end
    ].copy()

    features = model_feature_columns(
        market
    )

    usable = []

    for c in features:
        s = (
            train[c]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        if s.notna().sum() < 1000:
            continue

        if s.nunique(
            dropna=True
        ) < 3:
            continue

        usable.append(c)

    features = usable

    print()
    print("=" * 120)
    print("MARKET STATE MATRIX")
    print("=" * 120)
    print(
        f"Rows:       {len(market):,}"
    )
    print(
        f"Discovery:  {len(train):,}"
    )
    print(
        f"Validation: {len(validation):,}"
    )
    print(
        f"Features:   {len(features)}"
    )
    print(
        f"Coins/timestamp median: "
        f"{market['coins_available'].median():.0f}"
    )
    print()
    print(
        "Discovery base rates:"
    )
    print(
        f"  LONG  = "
        f"{train['target_market_long'].mean():.2%}"
    )
    print(
        f"  SHORT = "
        f"{train['target_market_short'].mean():.2%}"
    )
    print(
        "Validation base rates:"
    )
    print(
        f"  LONG  = "
        f"{validation['target_market_long'].mean():.2%}"
    )
    print(
        f"  SHORT = "
        f"{validation['target_market_short'].mean():.2%}"
    )

    print()
    print("Discovering LONG trigger candidates...")

    long_candidates, long_events, long_imp = (
        discover_direction(
            train=train,
            validation=validation,
            features=features,
            target_col="target_market_long",
            direction="LONG",
        )
    )

    print(
        "Discovering SHORT trigger candidates..."
    )

    short_candidates, short_events, short_imp = (
        discover_direction(
            train=train,
            validation=validation,
            features=features,
            target_col="target_market_short",
            direction="SHORT",
        )
    )

    candidates = pd.concat(
        [
            long_candidates,
            short_candidates,
        ],
        ignore_index=True,
    )

    candidate_events = pd.concat(
        [
            long_events,
            short_events,
        ],
        ignore_index=True,
    )

    importance = pd.concat(
        [
            long_imp,
            short_imp,
        ],
        ignore_index=True,
    )

    if candidates.empty:
        raise RuntimeError(
            "No candidate leaves passed the discovery lift gate."
        )

    # Discovery-only qualification.
    candidates[
        "discovery_qualified"
    ] = (
        (
            candidates[
                "discovery_cooldown_events"
            ]
            >= MIN_DISCOVERY_COOLDOWN_EVENTS
        )
        &
        (
            candidates[
                "discovery_episode_events"
            ]
            >= MIN_DISCOVERY_NEW_EPISODE_EVENTS
        )
        &
        (
            candidates[
                "discovery_raw_lift"
            ]
            >= MIN_DISCOVERY_RAW_LIFT
        )
        &
        (
            candidates[
                "discovery_stability_ratio"
            ]
            >= MIN_DISCOVERY_STABILITY_RATIO
        )
        &
        (
            candidates[
                "discovery_cooldown_mean_direction_return_atr"
            ]
            > 0
        )
    )

    # Sort using DISCOVERY metrics only.
    candidates = candidates.sort_values(
        [
            "discovery_qualified",
            "discovery_cooldown_wilson_lb",
            "discovery_cooldown_mean_direction_return_atr",
            "discovery_cooldown_events",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # A convenience field for reading, NOT used for fitting.
    candidates[
        "validation_same_direction"
    ] = (
        candidates[
            "validation_cooldown_mean_direction_return_atr"
        ]
        > 0
    )

    candidates[
        "validation_episode_same_direction"
    ] = (
        candidates[
            "validation_episode_mean_direction_return_atr"
        ]
        > 0
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    matrix_file = os.path.join(
        DATA_DIR,
        f"market_trigger_library_state_matrix_{stamp}.parquet",
    )

    candidates_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_library_candidates_{stamp}.csv",
    )

    events_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_library_candidate_events_{stamp}.csv",
    )

    importance_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_library_feature_importance_{stamp}.csv",
    )

    baseline_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_library_baselines_{stamp}.csv",
    )

    market.to_parquet(
        matrix_file,
        index=False,
    )

    candidates.to_csv(
        candidates_file,
        index=False,
    )

    candidate_events.to_csv(
        events_file,
        index=False,
    )

    importance.to_csv(
        importance_file,
        index=False,
    )

    baseline = pd.DataFrame(
        [
            {
                "split":
                    "DISCOVERY_60D",
                "rows":
                    len(train),
                "long_base_rate":
                    train[
                        "target_market_long"
                    ].mean(),
                "short_base_rate":
                    train[
                        "target_market_short"
                    ].mean(),
                "mean_future_market_median_return_atr":
                    train[
                        "future_market_median_return_atr"
                    ].mean(),
            },
            {
                "split":
                    "VALIDATION_30D",
                "rows":
                    len(validation),
                "long_base_rate":
                    validation[
                        "target_market_long"
                    ].mean(),
                "short_base_rate":
                    validation[
                        "target_market_short"
                    ].mean(),
                "mean_future_market_median_return_atr":
                    validation[
                        "future_market_median_return_atr"
                    ].mean(),
            },
        ]
    )

    baseline.to_csv(
        baseline_file,
        index=False,
    )

    print()
    print("=" * 180)
    print("TOP DISCOVERY-QUALIFIED CANDIDATES")
    print("Sorted by discovery only; validation is shown but never used to fit.")
    print("=" * 180)

    show_cols = [
        "candidate_id",
        "direction",
        "condition",
        "discovery_qualified",
        "discovery_raw_lift",
        "discovery_cooldown_events",
        "discovery_cooldown_hit_rate",
        "discovery_cooldown_mean_direction_return_atr",
        "discovery_episode_events",
        "discovery_episode_mean_direction_return_atr",
        "discovery_stability_ratio",
        "discovery_cooldown_trigger_A_active_rate",
        "validation_cooldown_events",
        "validation_cooldown_hit_rate",
        "validation_cooldown_mean_direction_return_atr",
        "validation_episode_events",
        "validation_episode_mean_direction_return_atr",
        "validation_cooldown_trigger_A_active_rate",
    ]

    qualified = candidates.loc[
        candidates[
            "discovery_qualified"
        ]
    ].copy()

    if qualified.empty:
        print(
            "No candidates passed every discovery gate."
        )
        print()
        print(
            candidates[
                show_cols
            ]
            .head(15)
            .to_string(
                index=False
            )
        )
    else:
        print(
            qualified[
                show_cols
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Candidates:  {candidates_file}")
    print(f"Events:      {events_file}")
    print(f"Importance:  {importance_file}")
    print(f"Baselines:   {baseline_file}")
    print(f"State matrix:{matrix_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
