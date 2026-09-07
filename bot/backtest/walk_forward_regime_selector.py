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

# Fixed, pre-trade numeric features only.
# No year/window/sample_group and no outcome-derived columns.
FEATURES = [
    "atr_pct",
    "ema_spread_pct",
    "price_vs_ema200_pct",
    "ema200_slope_24h_pct",
    "ema200_slope_3d_pct",
    "volume_ratio",
    "alligator_spread_pct",
    "breakdown_depth_pct",
    "return_24h_pct",
    "return_3d_pct",
    "return_7d_pct",
]

# Fixed classifier settings. No hyperparameter search.
RANDOM_STATE = 42
MAX_ITER = 2000

VALID_LABELS = {
    "BREAKOUT",
    "INVERSE",
    "NO_TRADE",
}


# ============================================================
# INPUT
# ============================================================

def find_input_file():

    # Optional explicit file:
    # python bot/backtest/walk_forward_regime_selector.py reports/paired_signal_results_....csv
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
        f"paired_signal_results_{SYMBOL}_{TIMEFRAME}_*.csv"
    )

    files = sorted(
        glob.glob(pattern)
    )

    if not files:
        raise RuntimeError(
            "No paired_signal_results report found in "
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
# LOAD / VALIDATE DATA
# ============================================================

def load_dataset(filepath):

    print()
    print("=" * 110)
    print("WALK-FORWARD REGIME SELECTOR")
    print("=" * 110)
    print(f"Input: {filepath}")
    print()
    print("Stage 1: TRADE vs NO_TRADE")
    print("Stage 2: INVERSE vs BREAKOUT")
    print()
    print("Model: Logistic Regression")
    print("Training: expanding chronological windows")
    print("First test: W5")
    print("No hyperparameter tuning")
    print("No future-window information in model training")
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
        .apply(
            window_number
        )
    )

    df = df[
        df["window_number"]
        .notna()
    ].copy()

    df["window_number"] = (
        df["window_number"]
        .astype(int)
    )

    # Map paired-test labels into selector actions.
    #
    # preferred_mode currently contains:
    # BREAKOUT, INVERSE, NO_TRADE.
    #
    # Any AMBIGUOUS / UNRESOLVED observations are not
    # suitable supervised labels and are excluded.
    df = df[
        df["preferred_mode"]
        .isin(
            VALID_LABELS
        )
    ].copy()

    df["stage1_target"] = np.where(
        df["preferred_mode"]
        == "NO_TRADE",
        0,
        1
    )

    # Stage 2:
    # INVERSE  = 1
    # BREAKOUT = 0
    df["stage2_target"] = np.where(
        df["preferred_mode"]
        == "INVERSE",
        1,
        np.where(
            df["preferred_mode"]
            == "BREAKOUT",
            0,
            np.nan
        )
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

    # Evaluation requires both direction outcomes to be resolved.
    df = df[
        df["breakout_r_multiple"]
        .notna()
        &
        df["inverse_r_multiple"]
        .notna()
    ].copy()

    df = (
        df
        .sort_values(
            [
                "window_number",
                "signal_timestamp",
                "signal_number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(f"Signals available: {len(df)}")
    print(
        f"Windows: "
        f"W{df['window_number'].min()} "
        f"-> W{df['window_number'].max()}"
    )

    print()
    print("Label distribution:")

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

    # Median imputation is fit on TRAIN only because it is
    # inside the sklearn pipeline.
    #
    # Scaling is also fit on TRAIN only.
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


def fit_or_constant(
    train_x,
    train_y
):

    unique = pd.Series(
        train_y
    ).dropna().unique()

    if len(unique) == 0:
        raise RuntimeError(
            "Training target has no valid classes."
        )

    if len(unique) == 1:
        return {
            "kind": "constant",
            "value": int(
                unique[0]
            ),
            "model": None,
        }

    model = make_model()

    model.fit(
        train_x,
        train_y
    )

    return {
        "kind": "model",
        "value": None,
        "model": model,
    }


def predict_binary(
    fitted,
    x
):

    if fitted["kind"] == "constant":

        prediction = np.full(
            len(x),
            fitted["value"],
            dtype=int
        )

        probability = np.full(
            len(x),
            float(
                fitted["value"]
            ),
            dtype=float
        )

        return (
            prediction,
            probability
        )

    model = fitted["model"]

    prediction = model.predict(
        x
    ).astype(int)

    probability = model.predict_proba(
        x
    )[:, 1]

    return (
        prediction,
        probability
    )


# ============================================================
# METRICS
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

    equity = np.cumsum(
        values
    )

    equity_with_start = np.concatenate(
        [
            np.array([0.0]),
            equity
        ]
    )

    running_peak = np.maximum.accumulate(
        equity_with_start
    )

    drawdown = (
        running_peak
        - equity_with_start
    )

    return float(
        drawdown.max()
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

    positive = values[
        values > 0
    ].sum()

    negative = -values[
        values < 0
    ].sum()

    if negative == 0:
        return np.nan

    return float(
        positive
        / negative
    )


def strategy_metrics(
    action_name,
    r_values
):

    values = np.asarray(
        r_values,
        dtype=float
    )

    trades_mask = (
        values != 0
    )

    trade_values = values[
        trades_mask
    ]

    trades = int(
        len(
            trade_values
        )
    )

    wins = int(
        (
            trade_values > 0
        )
        .sum()
    )

    losses = int(
        (
            trade_values < 0
        )
        .sum()
    )

    return {
        "strategy":
            action_name,

        "signals":
            int(
                len(values)
            ),

        "trades":
            trades,

        "skipped":
            int(
                len(values)
                - trades
            ),

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            (
                wins
                / trades
                * 100
                if trades
                else 0.0
            ),

        "total_r":
            float(
                values.sum()
            ),

        "avg_r_per_trade":
            (
                float(
                    trade_values.mean()
                )
                if trades
                else 0.0
            ),

        "avg_r_per_signal":
            float(
                values.mean()
            )
            if len(values)
            else 0.0,

        "profit_factor_r":
            profit_factor_r(
                trade_values
            )
            if trades
            else np.nan,

        "max_drawdown_r":
            max_drawdown_r(
                values
            ),

        "max_consecutive_losses":
            max_consecutive_losses(
                trade_values
            ),
    }


# ============================================================
# COEFFICIENTS
# ============================================================

def extract_coefficients(
    fitted,
    test_window,
    stage,
    train_windows
):

    rows = []

    if fitted["kind"] != "model":
        return rows

    model = fitted[
        "model"
    ].named_steps[
        "model"
    ]

    coefficients = model.coef_[0]

    for feature, coefficient in zip(
        FEATURES,
        coefficients
    ):

        rows.append(
            {
                "test_window":
                    f"W{test_window}",

                "stage":
                    stage,

                "train_through_window":
                    f"W{max(train_windows)}",

                "feature":
                    feature,

                "coefficient":
                    float(
                        coefficient
                    ),
            }
        )

    return rows


# ============================================================
# ONE WALK-FORWARD WINDOW
# ============================================================

def run_window(
    df,
    test_window
):

    train = df[
        df[
            "window_number"
        ]
        < test_window
    ].copy()

    test = df[
        df[
            "window_number"
        ]
        == test_window
    ].copy()

    if train.empty or test.empty:
        return None

    train_windows = sorted(
        train[
            "window_number"
        ].unique()
        .tolist()
    )

    # --------------------------------------------------------
    # STAGE 1: TRADE vs NO_TRADE
    # --------------------------------------------------------

    stage1 = fit_or_constant(
        train[
            FEATURES
        ],
        train[
            "stage1_target"
        ]
    )

    stage1_pred, stage1_prob = predict_binary(
        stage1,
        test[
            FEATURES
        ]
    )

    # --------------------------------------------------------
    # STAGE 2: INVERSE vs BREAKOUT
    #
    # Train only on historical rows where trading was the
    # correct label.
    # --------------------------------------------------------

    stage2_train = train[
        train[
            "preferred_mode"
        ]
        .isin(
            [
                "BREAKOUT",
                "INVERSE"
            ]
        )
    ].copy()

    stage2 = fit_or_constant(
        stage2_train[
            FEATURES
        ],
        stage2_train[
            "stage2_target"
        ].astype(int)
    )

    stage2_pred, stage2_prob = predict_binary(
        stage2,
        test[
            FEATURES
        ]
    )

    # --------------------------------------------------------
    # MODEL ACTION
    # --------------------------------------------------------

    actions = []

    for trade_pred, direction_pred in zip(
        stage1_pred,
        stage2_pred
    ):

        if trade_pred == 0:
            actions.append(
                "NO_TRADE"
            )
        elif direction_pred == 1:
            actions.append(
                "INVERSE"
            )
        else:
            actions.append(
                "BREAKOUT"
            )

    test[
        "model_action"
    ] = actions

    test[
        "stage1_trade_probability"
    ] = stage1_prob

    test[
        "stage2_inverse_probability"
    ] = stage2_prob

    # --------------------------------------------------------
    # REALIZED R
    # --------------------------------------------------------

    model_r = np.where(
        test[
            "model_action"
        ]
        == "NO_TRADE",
        0.0,
        np.where(
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
        )
    )

    inverse_r = (
        test[
            "inverse_r_multiple"
        ]
        .to_numpy(
            dtype=float
        )
    )

    breakout_r = (
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
    ] = inverse_r

    test[
        "always_breakout_r"
    ] = breakout_r

    # --------------------------------------------------------
    # CLASSIFICATION DIAGNOSTICS
    # --------------------------------------------------------

    stage1_truth = (
        test[
            "stage1_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    stage1_accuracy = float(
        (
            stage1_pred
            == stage1_truth
        )
        .mean()
    )

    true_trade_mask = (
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

    if true_trade_mask.sum() > 0:

        stage2_truth = (
            test.loc[
                true_trade_mask,
                "stage2_target"
            ]
            .astype(int)
            .to_numpy()
        )

        stage2_accuracy = float(
            (
                stage2_pred[
                    true_trade_mask
                ]
                == stage2_truth
            )
            .mean()
        )

    else:
        stage2_accuracy = np.nan

    full_action_accuracy = float(
        (
            test[
                "model_action"
            ]
            == test[
                "preferred_mode"
            ]
        )
        .mean()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = []

    for strategy_name, values in [
        (
            "MODEL",
            model_r
        ),
        (
            "ALWAYS_INVERSE",
            inverse_r
        ),
        (
            "ALWAYS_BREAKOUT",
            breakout_r
        ),
    ]:

        row = strategy_metrics(
            strategy_name,
            values
        )

        row.update(
            {
                "test_window":
                    f"W{test_window}",

                "train_start_window":
                    f"W{min(train_windows)}",

                "train_end_window":
                    f"W{max(train_windows)}",

                "train_signals":
                    int(
                        len(train)
                    ),

                "test_signals":
                    int(
                        len(test)
                    ),

                "stage1_accuracy_pct":
                    stage1_accuracy
                    * 100,

                "stage2_accuracy_on_true_trade_pct":
                    (
                        stage2_accuracy
                        * 100
                        if not np.isnan(
                            stage2_accuracy
                        )
                        else np.nan
                    ),

                "full_action_accuracy_pct":
                    full_action_accuracy
                    * 100,
            }
        )

        metrics.append(
            row
        )

    coefficients = []

    coefficients.extend(
        extract_coefficients(
            fitted=stage1,
            test_window=test_window,
            stage="STAGE1_TRADE_VS_SKIP",
            train_windows=train_windows
        )
    )

    coefficients.extend(
        extract_coefficients(
            fitted=stage2,
            test_window=test_window,
            stage="STAGE2_INVERSE_VS_BREAKOUT",
            train_windows=train_windows
        )
    )

    output_columns = [
        "signal_number",
        "signal_timestamp",
        "window",
        "preferred_mode",
        "model_action",
        "stage1_trade_probability",
        "stage2_inverse_probability",
        "model_r",
        "always_inverse_r",
        "always_breakout_r",
        "breakout_r_multiple",
        "inverse_r_multiple",
    ] + FEATURES

    predictions = test[
        output_columns
    ].copy()

    predictions[
        "test_window"
    ] = f"W{test_window}"

    predictions[
        "train_end_window"
    ] = f"W{max(train_windows)}"

    return {
        "predictions":
            predictions,

        "metrics":
            metrics,

        "coefficients":
            coefficients,
    }


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_overall_summary(
    predictions_df
):

    rows = []

    for strategy_name, column in [
        (
            "MODEL",
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

        row = strategy_metrics(
            strategy_name,
            predictions_df[
                column
            ].to_numpy(
                dtype=float
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COEFFICIENT STABILITY SUMMARY
# ============================================================

def build_coefficient_summary(
    coefficients_df
):

    if coefficients_df.empty:
        return pd.DataFrame()

    return (
        coefficients_df
        .groupby(
            [
                "stage",
                "feature"
            ],
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
            "Not enough windows for walk-forward testing."
        )

    prediction_frames = []
    metric_rows = []
    coefficient_rows = []

    print()
    print("=" * 110)
    print("RUNNING EXPANDING WALK-FORWARD")
    print("=" * 110)

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

        prediction_frames.append(
            result[
                "predictions"
            ]
        )

        metric_rows.extend(
            result[
                "metrics"
            ]
        )

        coefficient_rows.extend(
            result[
                "coefficients"
            ]
        )

        window_metrics = pd.DataFrame(
            result[
                "metrics"
            ]
        )

        model_row = (
            window_metrics[
                window_metrics[
                    "strategy"
                ]
                == "MODEL"
            ]
            .iloc[0]
        )

        inverse_row = (
            window_metrics[
                window_metrics[
                    "strategy"
                ]
                == "ALWAYS_INVERSE"
            ]
            .iloc[0]
        )

        breakout_row = (
            window_metrics[
                window_metrics[
                    "strategy"
                ]
                == "ALWAYS_BREAKOUT"
            ]
            .iloc[0]
        )

        print(
            f"W{test_window:<2} | "
            f"Train W1-W{test_window - 1:<2} | "
            f"Signals={int(model_row['signals']):<3} | "
            f"MODEL R={model_row['total_r']:+6.2f} | "
            f"INV R={inverse_row['total_r']:+6.2f} | "
            f"BRK R={breakout_row['total_r']:+6.2f} | "
            f"ActionAcc={model_row['full_action_accuracy_pct']:5.1f}%"
        )

    if not prediction_frames:
        raise RuntimeError(
            "No walk-forward predictions were produced."
        )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    windows_df = pd.DataFrame(
        metric_rows
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
    )

    overall_df = build_overall_summary(
        predictions_df
    )

    coefficient_summary_df = (
        build_coefficient_summary(
            coefficients_df
        )
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("OVERALL WALK-FORWARD RESULT")
    print("=" * 110)

    print(
        overall_df[
            [
                "strategy",
                "signals",
                "trades",
                "skipped",
                "win_rate_pct",
                "total_r",
                "avg_r_per_trade",
                "avg_r_per_signal",
                "profit_factor_r",
                "max_drawdown_r",
                "max_consecutive_losses",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("MODEL BY TEST WINDOW")
    print("=" * 110)

    model_windows = (
        windows_df[
            windows_df[
                "strategy"
            ]
            == "MODEL"
        ]
        .copy()
    )

    print(
        model_windows[
            [
                "test_window",
                "train_signals",
                "test_signals",
                "trades",
                "skipped",
                "win_rate_pct",
                "total_r",
                "profit_factor_r",
                "max_drawdown_r",
                "stage1_accuracy_pct",
                "stage2_accuracy_on_true_trade_pct",
                "full_action_accuracy_pct",
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
        f"walk_forward_regime_selector_predictions_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    windows_file = os.path.join(
        REPORT_DIR,
        f"walk_forward_regime_selector_windows_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    overall_file = os.path.join(
        REPORT_DIR,
        f"walk_forward_regime_selector_overall_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficients_file = os.path.join(
        REPORT_DIR,
        f"walk_forward_regime_selector_coefficients_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficient_summary_file = os.path.join(
        REPORT_DIR,
        f"walk_forward_regime_selector_coefficient_summary_"
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
    print("WALK-FORWARD TEST FINISHED")
    print("=" * 110)
    print(f"Predictions:         {predictions_file}")
    print(f"Window metrics:      {windows_file}")
    print(f"Overall metrics:     {overall_file}")
    print(f"Coefficients:        {coefficients_file}")
    print(f"Coefficient summary: {coefficient_summary_file}")
    print("=" * 110)

    print()
    print(
        "NOTE: max_drawdown_r is a sequential signal-level "
        "diagnostic, not portfolio drawdown. The paired-signal "
        "dataset intentionally allows overlapping hypothetical trades."
    )


if __name__ == "__main__":
    main()
