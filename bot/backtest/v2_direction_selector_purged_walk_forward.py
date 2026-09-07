import glob
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "\nscikit-learn is required.\n"
        "Install it inside the project venv with:\n\n"
        "    pip install scikit-learn\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"

FIRST_TEST_WINDOW = 5
WINDOW_MONTHS = 6

# Frozen V2 feature set.
FEATURES = [
    "atr_vs_7d_median",
    "atr_change_3d_pct",
    "break_below_prior_3d_low_atr",
]

RANDOM_STATE = 42
MAX_ITER = 2000
THRESHOLD = 0.50

PRE_SAMPLE_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

DEVELOPMENT_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_3y.parquet"
)


# ============================================================
# INPUT
# ============================================================

def find_input_file():

    # Optional explicit input:
    #
    # python bot/backtest/v2_direction_selector_purged_walk_forward.py \
    # reports/price_action_signal_features_BTCUSDT_4h_....csv
    if len(sys.argv) > 1:

        candidate = sys.argv[1]

        if not os.path.isabs(candidate):
            candidate = os.path.join(
                BASE_DIR,
                candidate
            )

        if not os.path.exists(candidate):
            raise RuntimeError(
                f"Input file does not exist: {candidate}"
            )

        return candidate

    pattern = os.path.join(
        REPORT_DIR,
        f"price_action_signal_features_{SYMBOL}_{TIMEFRAME}_*.csv"
    )

    files = sorted(
        glob.glob(pattern)
    )

    if not files:
        raise RuntimeError(
            "No price_action_signal_features report found in "
            f"{REPORT_DIR}"
        )

    return files[-1]


# ============================================================
# WINDOW / DATASET START
# ============================================================

def window_number(value):

    match = re.fullmatch(
        r"W(\d+)",
        str(value)
    )

    if not match:
        return np.nan

    return int(
        match.group(1)
    )


def load_dataset_start():

    starts = []

    for filepath, label in [
        (PRE_SAMPLE_FILE, "pre-sample"),
        (DEVELOPMENT_FILE, "development"),
    ]:

        if not os.path.exists(filepath):
            raise RuntimeError(
                f"Missing {label} parquet: {filepath}"
            )

        df = pd.read_parquet(
            filepath,
            columns=["timestamp"]
        )

        ts = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        ).dropna()

        if ts.empty:
            raise RuntimeError(
                f"No valid timestamps in {filepath}"
            )

        starts.append(
            ts.min()
        )

    return min(starts)


def window_start(
    dataset_start,
    number
):

    return (
        dataset_start
        + pd.DateOffset(
            months=WINDOW_MONTHS * (number - 1)
        )
    )


def window_end(
    dataset_start,
    number
):

    return (
        dataset_start
        + pd.DateOffset(
            months=WINDOW_MONTHS * number
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filepath):

    print()
    print("=" * 118)
    print("DIRECTION SELECTOR V2 - PURGED WALK-FORWARD")
    print("=" * 118)
    print(f"Input: {filepath}")
    print()
    print("Frozen model:")
    print("  Logistic Regression")
    print("  Threshold = 0.50")
    print("  class_weight = balanced")
    print()
    print("Frozen features:")
    for feature in FEATURES:
        print(f"  - {feature}")
    print()
    print("PURGE RULE:")
    print(
        "A directional training label is allowed only if BOTH "
        "hypothetical outcomes were already resolved strictly "
        "before the test-window start."
    )
    print()
    print("No threshold optimization.")
    print("No feature changes.")
    print("No NO_TRADE model.")
    print("=" * 118)

    df = pd.read_csv(
        filepath
    )

    required = set(
        [
            "signal_number",
            "signal_timestamp",
            "window",
            "preferred_mode",
            "breakout_r_multiple",
            "inverse_r_multiple",
            "breakout_exit_timestamp",
            "inverse_exit_timestamp",
        ]
        + FEATURES
    )

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\n"
            "Use the full price_action_signal_features CSV generated "
            "from the paired-signal diagnostic."
        )

    datetime_columns = [
        "signal_timestamp",
        "breakout_exit_timestamp",
        "inverse_exit_timestamp",
    ]

    for column in datetime_columns:

        df[column] = pd.to_datetime(
            df[column],
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

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df["breakout_r_multiple"] = pd.to_numeric(
        df["breakout_r_multiple"],
        errors="coerce"
    )

    df["inverse_r_multiple"] = pd.to_numeric(
        df["inverse_r_multiple"],
        errors="coerce"
    )

    # Evaluation requires both outcomes to be resolved.
    df = df[
        df["breakout_r_multiple"]
        .notna()
        &
        df["inverse_r_multiple"]
        .notna()
        &
        df["breakout_exit_timestamp"]
        .notna()
        &
        df["inverse_exit_timestamp"]
        .notna()
    ].copy()

    # INVERSE = 1
    # BREAKOUT = 0
    #
    # NO_TRADE remains in TEST because V2 trades every signal,
    # but it is never a direction label for model fitting.
    df["direction_target"] = np.where(
        df["preferred_mode"] == "INVERSE",
        1,
        np.where(
            df["preferred_mode"] == "BREAKOUT",
            0,
            np.nan
        )
    )

    # Label becomes fully knowable only when BOTH paired outcomes
    # have resolved.
    df["label_known_at"] = pd.concat(
        [
            df["breakout_exit_timestamp"],
            df["inverse_exit_timestamp"],
        ],
        axis=1
    ).max(
        axis=1
    )

    df = (
        df
        .sort_values(
            [
                "window_number",
                "signal_timestamp",
                "signal_number",
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print(f"Signals available: {len(df)}")
    print(
        f"Windows: "
        f"W{df['window_number'].min()} "
        f"-> W{df['window_number'].max()}"
    )

    print()
    print("Preferred-mode distribution:")

    print(
        df["preferred_mode"]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# MODEL
# ============================================================

def make_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                )
            ),
        ]
    )


def fit_model(
    directional_train
):

    if directional_train.empty:
        raise RuntimeError(
            "No purged directional training labels."
        )

    target = (
        directional_train[
            "direction_target"
        ]
        .astype(int)
    )

    unique = target.unique()

    if len(unique) == 1:

        return {
            "kind": "constant",
            "value": int(unique[0]),
            "model": None,
        }

    model = make_model()

    model.fit(
        directional_train[
            FEATURES
        ],
        target
    )

    return {
        "kind": "model",
        "value": None,
        "model": model,
    }


def predict_model(
    fitted,
    test
):

    if fitted["kind"] == "constant":

        prediction = np.full(
            len(test),
            fitted["value"],
            dtype=int
        )

        probability = np.full(
            len(test),
            float(fitted["value"]),
            dtype=float
        )

        return prediction, probability

    model = fitted["model"]

    probability = model.predict_proba(
        test[
            FEATURES
        ]
    )[:, 1]

    prediction = (
        probability
        >= THRESHOLD
    ).astype(int)

    return prediction, probability


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def max_drawdown_r(
    r_values
):

    values = np.asarray(
        r_values,
        dtype=float
    )

    if len(values) == 0:
        return 0.0

    cumulative = np.cumsum(
        values
    )

    curve = np.concatenate(
        [
            np.array([0.0]),
            cumulative
        ]
    )

    peaks = np.maximum.accumulate(
        curve
    )

    drawdowns = (
        peaks
        - curve
    )

    return float(
        drawdowns.max()
    )


def max_consecutive_losses(
    r_values
):

    maximum = 0
    current = 0

    for value in r_values:

        if value < 0:

            current += 1

            maximum = max(
                maximum,
                current
            )

        elif value > 0:

            current = 0

    return int(
        maximum
    )


def profit_factor_r(
    r_values
):

    values = np.asarray(
        r_values,
        dtype=float
    )

    gross_profit = values[
        values > 0
    ].sum()

    gross_loss = -values[
        values < 0
    ].sum()

    if gross_loss == 0:
        return np.nan

    return float(
        gross_profit
        / gross_loss
    )


def metrics(
    strategy,
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    wins = int(
        (
            values > 0
        )
        .sum()
    )

    losses = int(
        (
            values < 0
        )
        .sum()
    )

    return {
        "strategy":
            strategy,

        "signals":
            int(
                len(values)
            ),

        "trades":
            int(
                len(values)
            ),

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            (
                wins
                / len(values)
                * 100
                if len(values)
                else 0.0
            ),

        "total_r":
            float(
                values.sum()
            ),

        "avg_r_per_trade":
            (
                float(
                    values.mean()
                )
                if len(values)
                else 0.0
            ),

        "profit_factor_r":
            profit_factor_r(
                values
            ),

        "max_drawdown_r":
            max_drawdown_r(
                values
            ),

        "max_consecutive_losses":
            max_consecutive_losses(
                values
            ),
    }


# ============================================================
# COEFFICIENTS
# ============================================================

def extract_coefficients(
    fitted,
    test_window,
    train_count
):

    rows = []

    if fitted["kind"] != "model":
        return rows

    lr = fitted[
        "model"
    ].named_steps[
        "model"
    ]

    for feature, coefficient in zip(
        FEATURES,
        lr.coef_[0]
    ):

        rows.append(
            {
                "test_window":
                    f"W{test_window}",

                "purged_train_directional_signals":
                    int(train_count),

                "feature":
                    feature,

                "coefficient":
                    float(
                        coefficient
                    ),
            }
        )

    rows.append(
        {
            "test_window":
                f"W{test_window}",

            "purged_train_directional_signals":
                int(train_count),

            "feature":
                "__INTERCEPT__",

            "coefficient":
                float(
                    lr.intercept_[0]
                ),
        }
    )

    return rows


# ============================================================
# ONE PURGED WALK-FORWARD WINDOW
# ============================================================

def run_window(
    df,
    dataset_start,
    test_window
):

    test_start = window_start(
        dataset_start,
        test_window
    )

    test_end = window_end(
        dataset_start,
        test_window
    )

    # Test membership uses the actual window boundaries.
    test = df[
        (
            df["signal_timestamp"]
            >= test_start
        )
        &
        (
            df["signal_timestamp"]
            < test_end
        )
    ].copy()

    if test.empty:
        return None

    # All earlier direction-labeled examples by signal time.
    raw_directional_train = df[
        (
            df["signal_timestamp"]
            < test_start
        )
        &
        (
            df["preferred_mode"]
            .isin(
                [
                    "BREAKOUT",
                    "INVERSE"
                ]
            )
        )
    ].copy()

    # PURGE:
    # the label is eligible only when BOTH paired outcomes
    # were already fully known strictly before test_start.
    purged_directional_train = raw_directional_train[
        raw_directional_train[
            "label_known_at"
        ]
        < test_start
    ].copy()

    excluded_for_leakage = (
        len(raw_directional_train)
        - len(purged_directional_train)
    )

    if purged_directional_train.empty:
        raise RuntimeError(
            f"W{test_window}: purge removed all directional training labels."
        )

    fitted = fit_model(
        purged_directional_train
    )

    pred, inverse_probability = predict_model(
        fitted,
        test
    )

    test[
        "inverse_probability"
    ] = inverse_probability

    test[
        "model_action"
    ] = np.where(
        pred == 1,
        "INVERSE",
        "BREAKOUT"
    )

    model_r = np.where(
        test[
            "model_action"
        ]
        == "INVERSE",
        test[
            "inverse_r_multiple"
        ],
        test[
            "breakout_r_multiple"
        ]
    ).astype(float)

    always_inverse_r = (
        test[
            "inverse_r_multiple"
        ]
        .to_numpy(
            dtype=float
        )
    )

    always_breakout_r = (
        test[
            "breakout_r_multiple"
        ]
        .to_numpy(
            dtype=float
        )
    )

    test[
        "model_r"
    ] = model_r

    test[
        "always_inverse_r"
    ] = always_inverse_r

    test[
        "always_breakout_r"
    ] = always_breakout_r

    # --------------------------------------------------------
    # Direction accuracy only where a true direction exists.
    # --------------------------------------------------------

    directional_mask = (
        test[
            "preferred_mode"
        ]
        .isin(
            [
                "BREAKOUT",
                "INVERSE"
            ]
        )
        .to_numpy()
    )

    directional_n = int(
        directional_mask.sum()
    )

    if directional_n > 0:

        truth = (
            test.loc[
                directional_mask,
                "direction_target"
            ]
            .astype(int)
            .to_numpy()
        )

        directional_accuracy = float(
            (
                pred[
                    directional_mask
                ]
                == truth
            )
            .mean()
        )

    else:

        directional_accuracy = np.nan

    inverse_predictions = int(
        (
            test[
                "model_action"
            ]
            == "INVERSE"
        )
        .sum()
    )

    breakout_predictions = int(
        (
            test[
                "model_action"
            ]
            == "BREAKOUT"
        )
        .sum()
    )

    # --------------------------------------------------------
    # Purge diagnostics
    # --------------------------------------------------------

    max_label_known_at = (
        purged_directional_train[
            "label_known_at"
        ]
        .max()
    )

    last_train_signal = (
        purged_directional_train[
            "signal_timestamp"
        ]
        .max()
    )

    label_gap_hours = (
        (
            test_start
            - max_label_known_at
        )
        .total_seconds()
        / 3600.0
        if pd.notna(
            max_label_known_at
        )
        else np.nan
    )

    metric_rows = []

    for strategy, values in [
        (
            "PURGED_DIRECTION_SELECTOR_V2",
            model_r
        ),
        (
            "ALWAYS_INVERSE",
            always_inverse_r
        ),
        (
            "ALWAYS_BREAKOUT",
            always_breakout_r
        ),
    ]:

        row = metrics(
            strategy,
            values
        )

        row.update(
            {
                "test_window":
                    f"W{test_window}",

                "test_start":
                    test_start.isoformat(),

                "test_end":
                    test_end.isoformat(),

                "raw_directional_train_signals":
                    int(
                        len(
                            raw_directional_train
                        )
                    ),

                "purged_train_directional_signals":
                    int(
                        len(
                            purged_directional_train
                        )
                    ),

                "excluded_for_label_leakage":
                    int(
                        excluded_for_leakage
                    ),

                "purged_train_retention_pct":
                    (
                        len(
                            purged_directional_train
                        )
                        / len(
                            raw_directional_train
                        )
                        * 100
                        if len(
                            raw_directional_train
                        )
                        else 0.0
                    ),

                "last_purged_train_signal":
                    (
                        last_train_signal.isoformat()
                        if pd.notna(
                            last_train_signal
                        )
                        else None
                    ),

                "last_known_training_label":
                    (
                        max_label_known_at.isoformat()
                        if pd.notna(
                            max_label_known_at
                        )
                        else None
                    ),

                "label_gap_before_test_hours":
                    label_gap_hours,

                "test_signals":
                    int(
                        len(test)
                    ),

                "test_directional_labels":
                    directional_n,

                "directional_accuracy_pct":
                    (
                        directional_accuracy
                        * 100
                        if not np.isnan(
                            directional_accuracy
                        )
                        else np.nan
                    ),

                "model_inverse_predictions":
                    inverse_predictions,

                "model_breakout_predictions":
                    breakout_predictions,
            }
        )

        metric_rows.append(
            row
        )

    predictions = test[
        [
            "signal_number",
            "signal_timestamp",
            "window",
            "preferred_mode",
            "pair_class",
            "model_action",
            "inverse_probability",
            "model_r",
            "always_inverse_r",
            "always_breakout_r",
            "breakout_r_multiple",
            "inverse_r_multiple",
            "breakout_exit_timestamp",
            "inverse_exit_timestamp",
        ]
        + FEATURES
    ].copy()

    predictions[
        "test_window"
    ] = f"W{test_window}"

    predictions[
        "test_start"
    ] = test_start

    predictions[
        "purged_train_directional_signals"
    ] = int(
        len(
            purged_directional_train
        )
    )

    predictions[
        "excluded_for_label_leakage"
    ] = int(
        excluded_for_leakage
    )

    coefficients = extract_coefficients(
        fitted=fitted,
        test_window=test_window,
        train_count=len(
            purged_directional_train
        )
    )

    purge_diagnostic = {
        "test_window":
            f"W{test_window}",

        "test_start":
            test_start.isoformat(),

        "test_end":
            test_end.isoformat(),

        "raw_directional_train_signals":
            int(
                len(
                    raw_directional_train
                )
            ),

        "purged_train_directional_signals":
            int(
                len(
                    purged_directional_train
                )
            ),

        "excluded_for_label_leakage":
            int(
                excluded_for_leakage
            ),

        "purged_train_retention_pct":
            (
                len(
                    purged_directional_train
                )
                / len(
                    raw_directional_train
                )
                * 100
                if len(
                    raw_directional_train
                )
                else 0.0
            ),

        "last_purged_train_signal":
            (
                last_train_signal.isoformat()
                if pd.notna(
                    last_train_signal
                )
                else None
            ),

        "last_known_training_label":
            (
                max_label_known_at.isoformat()
                if pd.notna(
                    max_label_known_at
                )
                else None
            ),

        "label_gap_before_test_hours":
            label_gap_hours,
    }

    return {
        "metrics":
            metric_rows,

        "predictions":
            predictions,

        "coefficients":
            coefficients,

        "purge_diagnostic":
            purge_diagnostic,
    }


# ============================================================
# OVERALL
# ============================================================

def build_overall(
    predictions
):

    rows = []

    for strategy, column in [
        (
            "PURGED_DIRECTION_SELECTOR_V2",
            "model_r"
        ),
        (
            "ALWAYS_INVERSE",
            "always_inverse_r"
        ),
        (
            "ALWAYS_BREAKOUT",
            "always_breakout_r"
        ),
    ]:

        rows.append(
            metrics(
                strategy,
                predictions[
                    column
                ].to_numpy(
                    dtype=float
                )
            )
        )

    directional = predictions[
        predictions[
            "preferred_mode"
        ]
        .isin(
            [
                "BREAKOUT",
                "INVERSE"
            ]
        )
    ].copy()

    if not directional.empty:

        accuracy = float(
            (
                directional[
                    "model_action"
                ]
                ==
                directional[
                    "preferred_mode"
                ]
            )
            .mean()
        )

    else:

        accuracy = np.nan

    overall = pd.DataFrame(
        rows
    )

    overall[
        "directional_accuracy_pct"
    ] = (
        accuracy
        * 100
        if not np.isnan(
            accuracy
        )
        else np.nan
    )

    overall[
        "directional_labeled_signals"
    ] = int(
        len(
            directional
        )
    )

    return overall


def build_coefficient_summary(
    coefficients
):

    if coefficients.empty:
        return pd.DataFrame()

    return (
        coefficients
        .groupby(
            "feature",
            as_index=False
        )
        .agg(
            windows=(
                "coefficient",
                "count"
            ),
            mean_coefficient=(
                "coefficient",
                "mean"
            ),
            median_coefficient=(
                "coefficient",
                "median"
            ),
            min_coefficient=(
                "coefficient",
                "min"
            ),
            max_coefficient=(
                "coefficient",
                "max"
            ),
            positive_windows=(
                "coefficient",
                lambda x: int(
                    (
                        x > 0
                    )
                    .sum()
                )
            ),
            negative_windows=(
                "coefficient",
                lambda x: int(
                    (
                        x < 0
                    )
                    .sum()
                )
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = find_input_file()

    dataset_start = load_dataset_start()

    print()
    print(
        f"Dataset window anchor: {dataset_start}"
    )

    df = load_dataset(
        input_file
    )

    max_window = int(
        df[
            "window_number"
        ]
        .max()
    )

    if max_window < FIRST_TEST_WINDOW:
        raise RuntimeError(
            "Not enough windows for purged walk-forward."
        )

    all_metrics = []
    prediction_frames = []
    all_coefficients = []
    purge_rows = []

    print()
    print("=" * 132)
    print("RUNNING PURGED WALK-FORWARD")
    print("=" * 132)

    for test_window in range(
        FIRST_TEST_WINDOW,
        max_window + 1
    ):

        result = run_window(
            df=df,
            dataset_start=dataset_start,
            test_window=test_window
        )

        if result is None:
            continue

        all_metrics.extend(
            result[
                "metrics"
            ]
        )

        prediction_frames.append(
            result[
                "predictions"
            ]
        )

        all_coefficients.extend(
            result[
                "coefficients"
            ]
        )

        purge_rows.append(
            result[
                "purge_diagnostic"
            ]
        )

        window_df = pd.DataFrame(
            result[
                "metrics"
            ]
        )

        model_row = (
            window_df[
                window_df[
                    "strategy"
                ]
                == "PURGED_DIRECTION_SELECTOR_V2"
            ]
            .iloc[0]
        )

        inverse_row = (
            window_df[
                window_df[
                    "strategy"
                ]
                == "ALWAYS_INVERSE"
            ]
            .iloc[0]
        )

        breakout_row = (
            window_df[
                window_df[
                    "strategy"
                ]
                == "ALWAYS_BREAKOUT"
            ]
            .iloc[0]
        )

        print(
            f"W{test_window:<2} | "
            f"Train raw/purged="
            f"{int(model_row['raw_directional_train_signals'])}/"
            f"{int(model_row['purged_train_directional_signals'])} | "
            f"Purged={int(model_row['excluded_for_label_leakage']):<2} | "
            f"N={int(model_row['test_signals']):<3} | "
            f"V2={model_row['total_r']:+6.2f}R | "
            f"INV={inverse_row['total_r']:+6.2f}R | "
            f"BRK={breakout_row['total_r']:+6.2f}R | "
            f"Acc={model_row['directional_accuracy_pct']:5.1f}%"
        )

    if not prediction_frames:
        raise RuntimeError(
            "No purged walk-forward predictions produced."
        )

    windows_df = pd.DataFrame(
        all_metrics
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    coefficients_df = pd.DataFrame(
        all_coefficients
    )

    purge_df = pd.DataFrame(
        purge_rows
    )

    overall_df = build_overall(
        predictions_df
    )

    coefficient_summary_df = (
        build_coefficient_summary(
            coefficients_df
        )
    )

    print()
    print("=" * 118)
    print("OVERALL PURGED WALK-FORWARD")
    print("=" * 118)

    print(
        overall_df[
            [
                "strategy",
                "signals",
                "wins",
                "losses",
                "win_rate_pct",
                "total_r",
                "avg_r_per_trade",
                "profit_factor_r",
                "max_drawdown_r",
                "max_consecutive_losses",
                "directional_accuracy_pct",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 132)
    print("PURGE DIAGNOSTICS")
    print("=" * 132)

    print(
        purge_df[
            [
                "test_window",
                "raw_directional_train_signals",
                "purged_train_directional_signals",
                "excluded_for_label_leakage",
                "purged_train_retention_pct",
                "last_known_training_label",
                "label_gap_before_test_hours",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    predictions_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_predictions_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    windows_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_windows_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    overall_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_overall_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficients_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_coefficients_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficient_summary_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_coefficient_summary_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    purge_file = os.path.join(
        REPORT_DIR,
        f"v2_purged_selector_purge_diagnostics_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    windows_df.to_csv(
        windows_file,
        index=False
    )

    overall_df.to_csv(
        overall_file,
        index=False
    )

    coefficients_df.to_csv(
        coefficients_file,
        index=False
    )

    coefficient_summary_df.to_csv(
        coefficient_summary_file,
        index=False
    )

    purge_df.to_csv(
        purge_file,
        index=False
    )

    print()
    print("=" * 118)
    print("PURGED V2 TEST FINISHED")
    print("=" * 118)
    print(f"Predictions:         {predictions_file}")
    print(f"Window metrics:      {windows_file}")
    print(f"Overall metrics:     {overall_file}")
    print(f"Coefficients:        {coefficients_file}")
    print(f"Coefficient summary: {coefficient_summary_file}")
    print(f"Purge diagnostics:   {purge_file}")
    print("=" * 118)

    print()
    print(
        "NOTE: paired-signal R still allows overlapping hypothetical trades. "
        "This test validates selector information flow, not executable "
        "portfolio returns."
    )


if __name__ == "__main__":
    main()
