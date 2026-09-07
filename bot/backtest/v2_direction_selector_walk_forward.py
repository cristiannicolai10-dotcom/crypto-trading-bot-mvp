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

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"

FIRST_TEST_WINDOW = 5

# Frozen minimal feature set selected from the prior
# stability diagnostic. No feature search in this script.
FEATURES = [
    "atr_vs_7d_median",
    "atr_change_3d_pct",
    "break_below_prior_3d_low_atr",
]

RANDOM_STATE = 42
MAX_ITER = 2000


# ============================================================
# INPUT
# ============================================================

def find_input_file():

    # Optional explicit input:
    # python bot/backtest/v2_direction_selector_walk_forward.py \
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
# WINDOW HELPERS
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


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filepath):

    print()
    print("=" * 110)
    print("DIRECTION SELECTOR V2 - WALK-FORWARD")
    print("=" * 110)
    print(f"Input: {filepath}")
    print()
    print("Selector decides ONLY direction:")
    print("  INVERSE  = LONG")
    print("  BREAKOUT = SHORT")
    print()
    print("It is NOT allowed to choose NO_TRADE.")
    print()
    print("Frozen features:")
    for feature in FEATURES:
        print(f"  - {feature}")
    print()
    print("Model: Logistic Regression")
    print("Threshold: fixed 0.50")
    print("Training: expanding chronological windows")
    print("First test: W5")
    print("No hyperparameter tuning")
    print("No threshold optimization")
    print("=" * 110)

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

    # R evaluation needs both hypothetical outcomes resolved.
    df = df[
        df["breakout_r_multiple"]
        .notna()
        &
        df["inverse_r_multiple"]
        .notna()
    ].copy()

    # Direction labels:
    # INVERSE  = 1
    # BREAKOUT = 0
    # NO_TRADE has no direction label and is excluded only from
    # model fitting / directional accuracy. It remains in the
    # test set because V2 must trade every candidate signal.
    df["direction_target"] = np.where(
        df["preferred_mode"] == "INVERSE",
        1,
        np.where(
            df["preferred_mode"] == "BREAKOUT",
            0,
            np.nan
        )
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


def fit_model(train):

    directional_train = train[
        train["preferred_mode"]
        .isin(
            [
                "BREAKOUT",
                "INVERSE"
            ]
        )
    ].copy()

    if directional_train.empty:
        raise RuntimeError(
            "No directional training labels."
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
            "train_directional_signals": len(
                directional_train
            ),
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
        "train_directional_signals": len(
            directional_train
        ),
    }


def predict_model(fitted, test):

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
        test[FEATURES]
    )[:, 1]

    prediction = (
        probability >= 0.50
    ).astype(int)

    return prediction, probability


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def max_drawdown_r(r_values):

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


def max_consecutive_losses(r_values):

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

    return int(maximum)


def profit_factor_r(r_values):

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


def metrics(strategy, values):

    values = np.asarray(
        values,
        dtype=float
    )

    wins = int(
        (values > 0)
        .sum()
    )

    losses = int(
        (values < 0)
        .sum()
    )

    return {
        "strategy":
            strategy,

        "signals":
            int(len(values)),

        "trades":
            int(len(values)),

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
    train_end_window
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

                "train_end_window":
                    f"W{train_end_window}",

                "feature":
                    feature,

                "coefficient":
                    float(coefficient),
            }
        )

    rows.append(
        {
            "test_window":
                f"W{test_window}",

            "train_end_window":
                f"W{train_end_window}",

            "feature":
                "__INTERCEPT__",

            "coefficient":
                float(lr.intercept_[0]),
        }
    )

    return rows


# ============================================================
# ONE WALK-FORWARD WINDOW
# ============================================================

def run_window(df, test_window):

    train = df[
        df["window_number"]
        < test_window
    ].copy()

    test = df[
        df["window_number"]
        == test_window
    ].copy()

    if train.empty or test.empty:
        return None

    train_end_window = int(
        train["window_number"]
        .max()
    )

    fitted = fit_model(
        train
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

    # Oracle diagnostic only:
    # pick the better direction for each signal.
    # This is NOT a tradable benchmark; it is an upper-bound reference.
    oracle_r = np.maximum(
        always_inverse_r,
        always_breakout_r
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

    test[
        "oracle_direction_r"
    ] = oracle_r

    # Directional accuracy only where a direction label exists.
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

    # Prediction composition
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

    metric_rows = []

    for strategy, values in [
        (
            "DIRECTION_SELECTOR_V2",
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

                "train_start_window":
                    f"W{int(train['window_number'].min())}",

                "train_end_window":
                    f"W{train_end_window}",

                "train_signals_all":
                    int(len(train)),

                "train_directional_signals":
                    int(
                        fitted[
                            "train_directional_signals"
                        ]
                    ),

                "test_signals":
                    int(len(test)),

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

    prediction_columns = [
        "signal_number",
        "signal_timestamp",
        "window",
        "sample_group",
        "preferred_mode",
        "pair_class",
        "model_action",
        "inverse_probability",
        "model_r",
        "always_inverse_r",
        "always_breakout_r",
        "oracle_direction_r",
        "breakout_r_multiple",
        "inverse_r_multiple",
    ] + FEATURES

    # Some older files may not contain sample_group/pair_class.
    prediction_columns = [
        column
        for column in prediction_columns
        if column in test.columns
    ]

    predictions = test[
        prediction_columns
    ].copy()

    predictions[
        "test_window"
    ] = f"W{test_window}"

    predictions[
        "train_end_window"
    ] = f"W{train_end_window}"

    coefficients = extract_coefficients(
        fitted=fitted,
        test_window=test_window,
        train_end_window=train_end_window
    )

    return {
        "metrics":
            metric_rows,

        "predictions":
            predictions,

        "coefficients":
            coefficients,
    }


# ============================================================
# OVERALL
# ============================================================

def build_overall(predictions):

    rows = []

    for strategy, column in [
        (
            "DIRECTION_SELECTOR_V2",
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
        len(directional)
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
                    (x > 0).sum()
                )
            ),
            negative_windows=(
                "coefficient",
                lambda x: int(
                    (x < 0).sum()
                )
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = find_input_file()

    df = load_dataset(
        input_file
    )

    max_window = int(
        df["window_number"]
        .max()
    )

    if max_window < FIRST_TEST_WINDOW:
        raise RuntimeError(
            "Not enough windows for walk-forward."
        )

    all_metrics = []
    prediction_frames = []
    all_coefficients = []

    print()
    print("=" * 120)
    print("RUNNING WALK-FORWARD")
    print("=" * 120)

    for test_window in range(
        FIRST_TEST_WINDOW,
        max_window + 1
    ):

        result = run_window(
            df,
            test_window
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
                == "DIRECTION_SELECTOR_V2"
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
            f"Train W1-W{test_window - 1:<2} | "
            f"N={int(model_row['signals']):<3} | "
            f"V2={model_row['total_r']:+6.2f}R | "
            f"INV={inverse_row['total_r']:+6.2f}R | "
            f"BRK={breakout_row['total_r']:+6.2f}R | "
            f"Acc={model_row['directional_accuracy_pct']:5.1f}% | "
            f"Pred I/B="
            f"{int(model_row['model_inverse_predictions'])}/"
            f"{int(model_row['model_breakout_predictions'])}"
        )

    if not prediction_frames:
        raise RuntimeError(
            "No walk-forward predictions produced."
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

    overall_df = build_overall(
        predictions_df
    )

    coefficient_summary_df = (
        build_coefficient_summary(
            coefficients_df
        )
    )

    # --------------------------------------------------------
    # PRINT OVERALL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("OVERALL WALK-FORWARD")
    print("=" * 110)

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
    print("=" * 120)
    print("V2 WINDOW STABILITY")
    print("=" * 120)

    v2_windows = windows_df[
        windows_df[
            "strategy"
        ]
        == "DIRECTION_SELECTOR_V2"
    ].copy()

    print(
        v2_windows[
            [
                "test_window",
                "test_signals",
                "test_directional_labels",
                "directional_accuracy_pct",
                "model_inverse_predictions",
                "model_breakout_predictions",
                "total_r",
                "profit_factor_r",
                "max_drawdown_r",
                "max_consecutive_losses",
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
        f"v2_direction_selector_predictions_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    windows_file = os.path.join(
        REPORT_DIR,
        f"v2_direction_selector_windows_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    overall_file = os.path.join(
        REPORT_DIR,
        f"v2_direction_selector_overall_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficients_file = os.path.join(
        REPORT_DIR,
        f"v2_direction_selector_coefficients_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficient_summary_file = os.path.join(
        REPORT_DIR,
        f"v2_direction_selector_coefficient_summary_"
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

    print()
    print("=" * 110)
    print("V2 DIRECTION SELECTOR TEST FINISHED")
    print("=" * 110)
    print(f"Predictions:         {predictions_file}")
    print(f"Window metrics:      {windows_file}")
    print(f"Overall metrics:     {overall_file}")
    print(f"Coefficients:        {coefficients_file}")
    print(f"Coefficient summary: {coefficient_summary_file}")
    print("=" * 110)

    print()
    print(
        "NOTE: paired-signal R allows overlapping hypothetical trades. "
        "Use this test for selector-vs-baseline comparison, not as "
        "portfolio return."
    )


if __name__ == "__main__":
    main()
