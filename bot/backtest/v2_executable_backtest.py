import glob
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import backtest_inverse as bt

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
# CONFIG - FROZEN EXECUTABLE V2
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"

INITIAL_BALANCE = 1000.0
POSITION_FRACTION = 0.20
LEVERAGE = 1.0

SL_ATR = 1.25
TP_ATR = 3.00

TAKER_FEE_RATE = 0.00055

FIRST_TEST_WINDOW = 5
WINDOW_MONTHS = 6

MODEL_THRESHOLD = 0.50
RANDOM_STATE = 42
MAX_ITER = 2000

FEATURES = [
    "atr_vs_7d_median",
    "atr_change_3d_pct",
    "break_below_prior_3d_low_atr",
]

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

def find_signal_file():

    # Optional explicit input:
    #
    # python bot/backtest/v2_executable_backtest.py \
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
            "No price_action_signal_features CSV found in reports/."
        )

    return files[-1]


# ============================================================
# HISTORY
# ============================================================

def load_one_history(filepath, label):

    if not os.path.exists(filepath):
        raise RuntimeError(
            f"Missing {label}: {filepath}"
        )

    df = pd.read_parquet(filepath)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"{label}: "
        f"{len(df):,} candles | "
        f"{df.iloc[0]['timestamp']} -> "
        f"{df.iloc[-1]['timestamp']}"
    )

    return df


def load_combined_history():

    pre = load_one_history(
        PRE_SAMPLE_FILE,
        "Pre-sample"
    )

    dev = load_one_history(
        DEVELOPMENT_FILE,
        "Development"
    )

    df = pd.concat(
        [pre, dev],
        ignore_index=True
    )

    duplicates = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print()
    print(
        f"Combined history: {len(df):,} candles"
    )
    print(
        f"Duplicates removed: {duplicates}"
    )
    print(
        f"History start: {df.iloc[0]['timestamp']}"
    )
    print(
        f"History end:   {df.iloc[-1]['timestamp']}"
    )

    return df


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

    return int(match.group(1))


def window_start(dataset_start, number):

    return (
        dataset_start
        + pd.DateOffset(
            months=WINDOW_MONTHS * (number - 1)
        )
    )


def window_end(dataset_start, number):

    return (
        dataset_start
        + pd.DateOffset(
            months=WINDOW_MONTHS * number
        )
    )


# ============================================================
# SIGNAL DATA
# ============================================================

def load_signals(filepath, history):

    print()
    print("=" * 118)
    print("EXECUTABLE DIRECTION SELECTOR V2 BACKTEST")
    print("=" * 118)
    print(f"Signal source: {filepath}")
    print()
    print("Frozen V2:")
    print("  Logistic Regression")
    print("  threshold = 0.50")
    print("  class_weight = balanced")
    print()
    print("Features:")
    for feature in FEATURES:
        print(f"  - {feature}")
    print()
    print("Execution:")
    print("  initial balance = $1,000")
    print("  position size   = 20% of current balance")
    print("  leverage        = 1x")
    print("  one active BTC position at a time")
    print("  entry           = next 4h candle OPEN")
    print(f"  stop            = {SL_ATR} ATR")
    print(f"  target          = {TP_ATR} ATR")
    print(f"  taker fee       = {TAKER_FEE_RATE * 100:.3f}% / side")
    print("  same candle SL+TP = SL first")
    print("  funding         = NOT modeled")
    print("  slippage        = NOT modeled")
    print()
    print("Training labels are PURGED before every test window.")
    print("=" * 118)

    df = pd.read_csv(filepath)

    required = set(
        [
            "signal_number",
            "signal_timestamp",
            "window",
            "preferred_mode",
            "breakout_exit_timestamp",
            "inverse_exit_timestamp",
        ]
        + FEATURES
    )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Signal CSV missing required columns:\n"
            + "\n".join(missing)
        )

    for col in [
        "signal_timestamp",
        "breakout_exit_timestamp",
        "inverse_exit_timestamp",
    ]:
        df[col] = pd.to_datetime(
            df[col],
            utc=True,
            errors="coerce"
        )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # ATR RECOVERY
    #
    # Some price_action_signal_features exports do not contain
    # the raw ATR column. In that case, rebuild ATR from the
    # same historical candles using the existing backtest engine
    # and merge it by signal timestamp.
    # --------------------------------------------------------

    if "atr" not in df.columns:

        print()
        print(
            "ATR column not found in signal CSV. "
            "Rebuilding ATR from historical candles..."
        )

        indicator_history = history.copy()

        indicator_history = bt.calculate_indicators(
            indicator_history
        )

        if "atr" not in indicator_history.columns:
            raise RuntimeError(
                "backtest_inverse.calculate_indicators() "
                "did not produce an 'atr' column."
            )

        atr_source = indicator_history[
            [
                "timestamp",
                "atr",
            ]
        ].copy()

        atr_source["timestamp"] = pd.to_datetime(
            atr_source["timestamp"],
            utc=True,
            errors="coerce"
        )

        atr_source["atr"] = pd.to_numeric(
            atr_source["atr"],
            errors="coerce"
        )

        df = df.merge(
            atr_source,
            left_on="signal_timestamp",
            right_on="timestamp",
            how="left",
            validate="many_to_one"
        )

        df = df.drop(
            columns=["timestamp"]
        )

    else:

        df["atr"] = pd.to_numeric(
            df["atr"],
            errors="coerce"
        )

    missing_atr = int(
        df["atr"]
        .isna()
        .sum()
    )

    if missing_atr:

        missing_examples = (
            df.loc[
                df["atr"].isna(),
                "signal_timestamp"
            ]
            .head(5)
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            f"ATR could not be resolved for {missing_atr} signals. "
            f"Examples: {missing_examples}"
        )

    print(
        f"ATR available for all {len(df)} signal rows."
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

    df["direction_target"] = np.where(
        df["preferred_mode"] == "INVERSE",
        1,
        np.where(
            df["preferred_mode"] == "BREAKOUT",
            0,
            np.nan
        )
    )

    df["label_known_at"] = pd.concat(
        [
            df["breakout_exit_timestamp"],
            df["inverse_exit_timestamp"],
        ],
        axis=1
    ).max(axis=1)

    df = (
        df
        .sort_values(
            [
                "signal_timestamp",
                "signal_number",
            ]
        )
        .reset_index(drop=True)
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


def fit_purged_model(
    signals,
    test_start
):

    raw_train = signals[
        (
            signals["signal_timestamp"] < test_start
        )
        &
        (
            signals["preferred_mode"]
            .isin(
                [
                    "BREAKOUT",
                    "INVERSE",
                ]
            )
        )
    ].copy()

    purged_train = raw_train[
        (
            raw_train["label_known_at"]
            .notna()
        )
        &
        (
            raw_train["label_known_at"]
            < test_start
        )
    ].copy()

    if purged_train.empty:
        raise RuntimeError(
            f"No purged directional labels before {test_start}"
        )

    target = (
        purged_train[
            "direction_target"
        ]
        .astype(int)
    )

    unique = target.unique()

    if len(unique) == 1:

        fitted = {
            "kind": "constant",
            "value": int(unique[0]),
            "model": None,
        }

    else:

        model = make_model()

        model.fit(
            purged_train[
                FEATURES
            ],
            target
        )

        fitted = {
            "kind": "model",
            "value": None,
            "model": model,
        }

    return (
        fitted,
        raw_train,
        purged_train
    )


def predict_model(
    fitted,
    test_signals
):

    if fitted["kind"] == "constant":

        pred = np.full(
            len(test_signals),
            fitted["value"],
            dtype=int
        )

        prob = np.full(
            len(test_signals),
            float(fitted["value"]),
            dtype=float
        )

        return pred, prob

    prob = fitted[
        "model"
    ].predict_proba(
        test_signals[
            FEATURES
        ]
    )[:, 1]

    pred = (
        prob >= MODEL_THRESHOLD
    ).astype(int)

    return pred, prob


def precompute_model_actions(
    signals,
    dataset_start
):

    max_window = int(
        signals[
            "window_number"
        ].max()
    )

    model_rows = []
    coefficient_rows = []
    diagnostics = []

    for w in range(
        FIRST_TEST_WINDOW,
        max_window + 1
    ):

        start = window_start(
            dataset_start,
            w
        )

        end = window_end(
            dataset_start,
            w
        )

        test = signals[
            (
                signals["signal_timestamp"] >= start
            )
            &
            (
                signals["signal_timestamp"] < end
            )
        ].copy()

        if test.empty:
            continue

        (
            fitted,
            raw_train,
            purged_train
        ) = fit_purged_model(
            signals,
            start
        )

        pred, prob = predict_model(
            fitted,
            test
        )

        test["model_action"] = np.where(
            pred == 1,
            "INVERSE",
            "BREAKOUT"
        )

        test["inverse_probability"] = prob

        model_rows.append(
            test[
                [
                    "signal_number",
                    "model_action",
                    "inverse_probability",
                ]
            ]
        )

        diagnostics.append(
            {
                "test_window": f"W{w}",
                "test_start": start,
                "test_end": end,
                "raw_directional_train":
                    int(len(raw_train)),
                "purged_directional_train":
                    int(len(purged_train)),
                "excluded_for_label_leakage":
                    int(
                        len(raw_train)
                        - len(purged_train)
                    ),
                "test_signals":
                    int(len(test)),
                "inverse_predictions":
                    int(
                        (
                            test["model_action"]
                            == "INVERSE"
                        ).sum()
                    ),
                "breakout_predictions":
                    int(
                        (
                            test["model_action"]
                            == "BREAKOUT"
                        ).sum()
                    ),
            }
        )

        if fitted["kind"] == "model":

            lr = fitted[
                "model"
            ].named_steps[
                "model"
            ]

            for feature, coefficient in zip(
                FEATURES,
                lr.coef_[0]
            ):

                coefficient_rows.append(
                    {
                        "test_window": f"W{w}",
                        "feature": feature,
                        "coefficient":
                            float(coefficient),
                        "purged_directional_train":
                            int(len(purged_train)),
                    }
                )

            coefficient_rows.append(
                {
                    "test_window": f"W{w}",
                    "feature": "__INTERCEPT__",
                    "coefficient":
                        float(lr.intercept_[0]),
                    "purged_directional_train":
                        int(len(purged_train)),
                }
            )

    if not model_rows:
        raise RuntimeError(
            "No model predictions generated."
        )

    predictions = pd.concat(
        model_rows,
        ignore_index=True
    )

    return (
        predictions,
        pd.DataFrame(diagnostics),
        pd.DataFrame(coefficient_rows),
    )


# ============================================================
# PREPARE CANDIDATE ENTRIES
# ============================================================

def prepare_candidates(
    signals,
    history,
    model_predictions
):

    work = signals.merge(
        model_predictions,
        on="signal_number",
        how="left",
        validate="one_to_one"
    )

    work = work[
        work["window_number"]
        >= FIRST_TEST_WINDOW
    ].copy()

    history_index = {
        ts: idx
        for idx, ts
        in enumerate(
            history["timestamp"]
        )
    }

    entry_indices = []
    entry_timestamps = []

    missing_signal_bar = 0
    no_next_bar = 0

    for row in work.itertuples():

        idx = history_index.get(
            row.signal_timestamp
        )

        if idx is None:

            entry_indices.append(
                np.nan
            )

            entry_timestamps.append(
                pd.NaT
            )

            missing_signal_bar += 1
            continue

        if idx + 1 >= len(history):

            entry_indices.append(
                np.nan
            )

            entry_timestamps.append(
                pd.NaT
            )

            no_next_bar += 1
            continue

        entry_indices.append(
            idx + 1
        )

        entry_timestamps.append(
            history.iloc[
                idx + 1
            ]["timestamp"]
        )

    work["entry_index"] = entry_indices
    work["entry_timestamp"] = entry_timestamps

    work = work[
        work["entry_index"]
        .notna()
    ].copy()

    work["entry_index"] = (
        work["entry_index"]
        .astype(int)
    )

    print()
    print(
        f"Executable candidate signals: {len(work)}"
    )
    print(
        f"Missing signal candles:       {missing_signal_bar}"
    )
    print(
        f"No next candle:               {no_next_bar}"
    )

    return work


# ============================================================
# STRATEGY ACTION
# ============================================================

def action_for_strategy(
    strategy,
    candidate
):

    if strategy == "EXECUTABLE_V2":

        action = candidate[
            "model_action"
        ]

        if action == "INVERSE":
            return "LONG"

        if action == "BREAKOUT":
            return "SHORT"

        raise RuntimeError(
            "Missing model action for candidate "
            f"{candidate['signal_number']}"
        )

    if strategy == "ALWAYS_INVERSE":
        return "LONG"

    if strategy == "ALWAYS_BREAKOUT":
        return "SHORT"

    raise ValueError(strategy)


# ============================================================
# EXECUTION HELPERS
# ============================================================

def build_position(
    strategy,
    candidate,
    row,
    balance,
    direction
):

    entry_price = float(
        row["open"]
    )

    atr = float(
        candidate["atr"]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    # Derivatives-style notional exposure:
    # capital is not subtracted as spot principal.
    notional = (
        balance
        * POSITION_FRACTION
        * LEVERAGE
    )

    quantity = (
        notional
        / entry_price
    )

    entry_fee = (
        notional
        * TAKER_FEE_RATE
    )

    if direction == "LONG":

        stop = (
            entry_price
            - SL_ATR * atr
        )

        target = (
            entry_price
            + TP_ATR * atr
        )

    else:

        stop = (
            entry_price
            + SL_ATR * atr
        )

        target = (
            entry_price
            - TP_ATR * atr
        )

    return {
        "strategy":
            strategy,

        "signal_number":
            int(
                candidate[
                    "signal_number"
                ]
            ),

        "signal_timestamp":
            candidate[
                "signal_timestamp"
            ],

        "signal_window":
            candidate[
                "window"
            ],

        "direction":
            direction,

        "entry_timestamp":
            row[
                "timestamp"
            ],

        "entry_price":
            entry_price,

        "atr":
            atr,

        "stop_price":
            stop,

        "target_price":
            target,

        "notional":
            notional,

        "quantity":
            quantity,

        "entry_fee":
            entry_fee,

        "entry_balance_before_fee":
            balance,

        "model_action":
            candidate.get(
                "model_action",
                None
            ),

        "inverse_probability":
            candidate.get(
                "inverse_probability",
                np.nan
            ),

        "bars_held":
            0,
    }


def check_exit(
    position,
    candle,
    is_entry_candle=False
):

    direction = position[
        "direction"
    ]

    stop = position[
        "stop_price"
    ]

    target = position[
        "target_price"
    ]

    open_price = float(
        candle[
            "open"
        ]
    )

    high = float(
        candle[
            "high"
        ]
    )

    low = float(
        candle[
            "low"
        ]
    )

    # For positions carried from a previous candle,
    # handle a gap through stop/target at the current open.
    if not is_entry_candle:

        if direction == "LONG":

            if open_price <= stop:
                return (
                    True,
                    open_price,
                    "SL_GAP"
                )

            if open_price >= target:
                return (
                    True,
                    open_price,
                    "TP_GAP"
                )

        else:

            if open_price >= stop:
                return (
                    True,
                    open_price,
                    "SL_GAP"
                )

            if open_price <= target:
                return (
                    True,
                    open_price,
                    "TP_GAP"
                )

    if direction == "LONG":

        stop_hit = (
            low <= stop
        )

        target_hit = (
            high >= target
        )

    else:

        stop_hit = (
            high >= stop
        )

        target_hit = (
            low <= target
        )

    if stop_hit and target_hit:

        return (
            True,
            stop,
            "SL_FIRST_SAME_CANDLE"
        )

    if stop_hit:

        return (
            True,
            stop,
            "SL"
        )

    if target_hit:

        return (
            True,
            target,
            "TP"
        )

    return (
        False,
        np.nan,
        None
    )


def unrealized_pnl(
    position,
    mark_price
):

    if position is None:
        return 0.0

    if position[
        "direction"
    ] == "LONG":

        return (
            position[
                "quantity"
            ]
            * (
                mark_price
                - position[
                    "entry_price"
                ]
            )
        )

    return (
        position[
            "quantity"
        ]
        * (
            position[
                "entry_price"
            ]
            - mark_price
        )
    )


def finalize_trade(
    position,
    exit_row,
    exit_price,
    exit_reason,
    balance
):

    quantity = position[
        "quantity"
    ]

    if position[
        "direction"
    ] == "LONG":

        gross_pnl = (
            quantity
            * (
                exit_price
                - position[
                    "entry_price"
                ]
            )
        )

    else:

        gross_pnl = (
            quantity
            * (
                position[
                    "entry_price"
                ]
                - exit_price
            )
        )

    exit_notional = (
        quantity
        * exit_price
    )

    exit_fee = (
        exit_notional
        * TAKER_FEE_RATE
    )

    # entry fee was already deducted at entry.
    balance_after = (
        balance
        + gross_pnl
        - exit_fee
    )

    total_fees = (
        position[
            "entry_fee"
        ]
        + exit_fee
    )

    net_pnl = (
        gross_pnl
        - total_fees
    )

    risk_dollars = (
        quantity
        * SL_ATR
        * position[
            "atr"
        ]
    )

    r_multiple_net = (
        net_pnl
        / risk_dollars
        if risk_dollars > 0
        else np.nan
    )

    trade = {
        "strategy":
            position[
                "strategy"
            ],

        "signal_number":
            position[
                "signal_number"
            ],

        "signal_timestamp":
            position[
                "signal_timestamp"
            ],

        "signal_window":
            position[
                "signal_window"
            ],

        "direction":
            position[
                "direction"
            ],

        "model_action":
            position[
                "model_action"
            ],

        "inverse_probability":
            position[
                "inverse_probability"
            ],

        "entry_timestamp":
            position[
                "entry_timestamp"
            ],

        "exit_timestamp":
            exit_row[
                "timestamp"
            ],

        "entry_price":
            position[
                "entry_price"
            ],

        "exit_price":
            exit_price,

        "stop_price":
            position[
                "stop_price"
            ],

        "target_price":
            position[
                "target_price"
            ],

        "atr":
            position[
                "atr"
            ],

        "notional":
            position[
                "notional"
            ],

        "quantity":
            quantity,

        "gross_pnl":
            gross_pnl,

        "entry_fee":
            position[
                "entry_fee"
            ],

        "exit_fee":
            exit_fee,

        "fees_total":
            total_fees,

        "net_pnl":
            net_pnl,

        "r_multiple_net":
            r_multiple_net,

        "result":
            (
                "WIN"
                if net_pnl > 0
                else "LOSS"
            ),

        "exit_reason":
            exit_reason,

        "bars_held":
            position[
                "bars_held"
            ],

        "balance_after":
            balance_after,
    }

    return trade, balance_after


# ============================================================
# STRATEGY SIMULATION
# ============================================================

def simulate_strategy(
    strategy,
    history,
    candidates,
    dataset_start,
    max_test_window
):

    start_time = window_start(
        dataset_start,
        FIRST_TEST_WINDOW
    )

    theoretical_end = window_end(
        dataset_start,
        max_test_window
    )

    end_time = min(
        theoretical_end,
        history[
            "timestamp"
        ].max()
        + pd.Timedelta(
            hours=4
        )
    )

    start_indices = history.index[
        history["timestamp"]
        >= start_time
    ]

    end_indices = history.index[
        history["timestamp"]
        < end_time
    ]

    if len(start_indices) == 0:
        raise RuntimeError(
            "No history at executable test start."
        )

    start_idx = int(
        start_indices.min()
    )

    end_idx = int(
        end_indices.max()
    )

    scheduled = {}

    for _, candidate in candidates.iterrows():

        idx = int(
            candidate[
                "entry_index"
            ]
        )

        if (
            idx < start_idx
            or idx > end_idx
        ):
            continue

        scheduled.setdefault(
            idx,
            []
        ).append(
            candidate
        )

    balance = INITIAL_BALANCE
    position = None

    trades = []
    skip_rows = []
    equity_rows = []

    occupied_bars = 0

    for idx in range(
        start_idx,
        end_idx + 1
    ):

        candle = history.iloc[idx]

        candidates_here = scheduled.get(
            idx,
            []
        )

        # ----------------------------------------------------
        # OPEN DECISION HAPPENS AT CANDLE OPEN.
        #
        # If a prior position is active at this open, all
        # candidates scheduled for this open are unavailable.
        # Even if the old position exits later in this candle,
        # we cannot retroactively enter at the open.
        # ----------------------------------------------------

        opened_this_candle = False

        if position is not None:

            for candidate in candidates_here:

                skip_rows.append(
                    {
                        "strategy":
                            strategy,

                        "signal_number":
                            int(
                                candidate[
                                    "signal_number"
                                ]
                            ),

                        "signal_timestamp":
                            candidate[
                                "signal_timestamp"
                            ],

                        "entry_timestamp":
                            candle[
                                "timestamp"
                            ],

                        "window":
                            candidate[
                                "window"
                            ],

                        "reason":
                            "BUSY_AT_ENTRY_OPEN",
                    }
                )

        elif len(candidates_here) > 0:

            # If multiple signals share the same entry open,
            # use the earliest signal_number and reject extras.
            candidates_here = sorted(
                candidates_here,
                key=lambda x: int(
                    x[
                        "signal_number"
                    ]
                )
            )

            chosen = candidates_here[0]

            for extra in candidates_here[1:]:

                skip_rows.append(
                    {
                        "strategy":
                            strategy,

                        "signal_number":
                            int(
                                extra[
                                    "signal_number"
                                ]
                            ),

                        "signal_timestamp":
                            extra[
                                "signal_timestamp"
                            ],

                        "entry_timestamp":
                            candle[
                                "timestamp"
                            ],

                        "window":
                            extra[
                                "window"
                            ],

                        "reason":
                            "SAME_OPEN_COLLISION",
                    }
                )

            direction = action_for_strategy(
                strategy,
                chosen
            )

            position = build_position(
                strategy=strategy,
                candidate=chosen,
                row=candle,
                balance=balance,
                direction=direction
            )

            if position is None:

                skip_rows.append(
                    {
                        "strategy":
                            strategy,

                        "signal_number":
                            int(
                                chosen[
                                    "signal_number"
                                ]
                            ),

                        "signal_timestamp":
                            chosen[
                                "signal_timestamp"
                            ],

                        "entry_timestamp":
                            candle[
                                "timestamp"
                            ],

                        "window":
                            chosen[
                                "window"
                            ],

                        "reason":
                            "INVALID_ATR",
                    }
                )

            else:

                # Deduct entry fee immediately.
                balance -= position[
                    "entry_fee"
                ]

                opened_this_candle = True

        # ----------------------------------------------------
        # INTRACANDLE EXIT
        # ----------------------------------------------------

        if position is not None:

            occupied_bars += 1

            position[
                "bars_held"
            ] += 1

            (
                exited,
                exit_price,
                exit_reason
            ) = check_exit(
                position=position,
                candle=candle,
                is_entry_candle=opened_this_candle
            )

            if exited:

                trade, balance = finalize_trade(
                    position=position,
                    exit_row=candle,
                    exit_price=float(
                        exit_price
                    ),
                    exit_reason=exit_reason,
                    balance=balance
                )

                trades.append(
                    trade
                )

                position = None

        # ----------------------------------------------------
        # MARK-TO-MARKET EQUITY AT CANDLE CLOSE
        # ----------------------------------------------------

        equity = (
            balance
            + unrealized_pnl(
                position,
                float(
                    candle[
                        "close"
                    ]
                )
            )
        )

        equity_rows.append(
            {
                "strategy":
                    strategy,

                "timestamp":
                    candle[
                        "timestamp"
                    ],

                "balance":
                    balance,

                "equity":
                    equity,

                "position_open":
                    position is not None,

                "position_direction":
                    (
                        position[
                            "direction"
                        ]
                        if position is not None
                        else None
                    ),
            }
        )

    # --------------------------------------------------------
    # FORCE CLOSE ONLY IF STILL OPEN AT END OF AVAILABLE DATA.
    # This should usually be zero. It is explicitly labeled.
    # --------------------------------------------------------

    if position is not None:

        final_candle = history.iloc[
            end_idx
        ]

        exit_price = float(
            final_candle[
                "close"
            ]
        )

        trade, balance = finalize_trade(
            position=position,
            exit_row=final_candle,
            exit_price=exit_price,
            exit_reason="END_OF_DATA_FORCE_CLOSE",
            balance=balance
        )

        trades.append(
            trade
        )

        position = None

        equity_rows[-1][
            "balance"
        ] = balance

        equity_rows[-1][
            "equity"
        ] = balance

        equity_rows[-1][
            "position_open"
        ] = False

        equity_rows[-1][
            "position_direction"
        ] = None

    trades_df = pd.DataFrame(
        trades
    )

    skips_df = pd.DataFrame(
        skip_rows
    )

    equity_df = pd.DataFrame(
        equity_rows
    )

    return {
        "strategy":
            strategy,

        "trades":
            trades_df,

        "skips":
            skips_df,

        "equity":
            equity_df,

        "final_balance":
            balance,

        "occupied_bars":
            occupied_bars,

        "total_bars":
            len(
                equity_df
            ),
    }


# ============================================================
# METRICS
# ============================================================

def max_drawdown_from_equity(
    equity_values
):

    values = np.asarray(
        equity_values,
        dtype=float
    )

    if len(values) == 0:
        return 0.0, 0.0

    peaks = np.maximum.accumulate(
        values
    )

    dd_usd = (
        peaks
        - values
    )

    dd_pct = np.where(
        peaks > 0,
        dd_usd
        / peaks
        * 100,
        0.0
    )

    return (
        float(
            dd_usd.max()
        ),
        float(
            dd_pct.max()
        )
    )


def max_consecutive_losses(
    trades
):

    if trades.empty:
        return 0

    maximum = 0
    current = 0

    for result in trades[
        "result"
    ]:

        if result == "LOSS":

            current += 1

            maximum = max(
                maximum,
                current
            )

        else:

            current = 0

    return int(maximum)


def profit_factor_money(
    trades
):

    if trades.empty:
        return np.nan

    profits = trades.loc[
        trades["net_pnl"] > 0,
        "net_pnl"
    ].sum()

    losses = -trades.loc[
        trades["net_pnl"] < 0,
        "net_pnl"
    ].sum()

    if losses == 0:
        return np.nan

    return float(
        profits / losses
    )


def overall_metrics(
    result,
    eligible_signals
):

    trades = result[
        "trades"
    ]

    skips = result[
        "skips"
    ]

    equity = result[
        "equity"
    ]

    dd_usd, dd_pct = max_drawdown_from_equity(
        equity[
            "equity"
        ].to_numpy(
            dtype=float
        )
    )

    wins = (
        int(
            (
                trades["result"]
                == "WIN"
            ).sum()
        )
        if not trades.empty
        else 0
    )

    losses = (
        int(
            (
                trades["result"]
                == "LOSS"
            ).sum()
        )
        if not trades.empty
        else 0
    )

    fees = (
        float(
            trades[
                "fees_total"
            ].sum()
        )
        if not trades.empty
        else 0.0
    )

    gross_pnl = (
        float(
            trades[
                "gross_pnl"
            ].sum()
        )
        if not trades.empty
        else 0.0
    )

    net_pnl = (
        float(
            trades[
                "net_pnl"
            ].sum()
        )
        if not trades.empty
        else 0.0
    )

    avg_hold = (
        float(
            trades[
                "bars_held"
            ].mean()
        )
        if not trades.empty
        else 0.0
    )

    return {
        "strategy":
            result[
                "strategy"
            ],

        "initial_balance":
            INITIAL_BALANCE,

        "final_balance":
            float(
                result[
                    "final_balance"
                ]
            ),

        "return_pct":
            (
                result[
                    "final_balance"
                ]
                / INITIAL_BALANCE
                - 1
            ) * 100,

        "eligible_signals":
            int(
                eligible_signals
            ),

        "executed_trades":
            int(
                len(trades)
            ),

        "skipped_signals":
            int(
                len(skips)
            ),

        "busy_skips":
            (
                int(
                    (
                        skips["reason"]
                        == "BUSY_AT_ENTRY_OPEN"
                    ).sum()
                )
                if not skips.empty
                else 0
            ),

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            (
                wins
                / len(trades)
                * 100
                if len(trades)
                else 0.0
            ),

        "gross_pnl_usd":
            gross_pnl,

        "fees_usd":
            fees,

        "net_pnl_usd":
            net_pnl,

        "profit_factor":
            profit_factor_money(
                trades
            ),

        "max_drawdown_usd":
            dd_usd,

        "max_drawdown_pct":
            dd_pct,

        "max_consecutive_losses":
            max_consecutive_losses(
                trades
            ),

        "avg_hold_bars":
            avg_hold,

        "exposure_bars_pct":
            (
                result[
                    "occupied_bars"
                ]
                / result[
                    "total_bars"
                ]
                * 100
                if result[
                    "total_bars"
                ]
                else 0.0
            ),

        "end_of_data_force_closes":
            (
                int(
                    (
                        trades["exit_reason"]
                        == "END_OF_DATA_FORCE_CLOSE"
                    ).sum()
                )
                if not trades.empty
                else 0
            ),
    }


# ============================================================
# WINDOW SUMMARY FROM CONTINUOUS EQUITY
# ============================================================

def value_at_or_before(
    equity,
    timestamp
):

    subset = equity[
        equity["timestamp"]
        < timestamp
    ]

    if subset.empty:
        return INITIAL_BALANCE

    return float(
        subset.iloc[-1][
            "equity"
        ]
    )


def value_before_or_at_end(
    equity,
    timestamp
):

    subset = equity[
        equity["timestamp"]
        < timestamp
    ]

    if subset.empty:
        return INITIAL_BALANCE

    return float(
        subset.iloc[-1][
            "equity"
        ]
    )


def build_window_summary(
    result,
    dataset_start,
    max_test_window
):

    strategy = result[
        "strategy"
    ]

    trades = result[
        "trades"
    ]

    skips = result[
        "skips"
    ]

    equity = result[
        "equity"
    ]

    rows = []

    for w in range(
        FIRST_TEST_WINDOW,
        max_test_window + 1
    ):

        start = window_start(
            dataset_start,
            w
        )

        end = window_end(
            dataset_start,
            w
        )

        eq_start = value_at_or_before(
            equity,
            start
        )

        eq_end = value_before_or_at_end(
            equity,
            end
        )

        if trades.empty:

            window_trades = trades

        else:

            window_trades = trades[
                (
                    trades[
                        "signal_timestamp"
                    ]
                    >= start
                )
                &
                (
                    trades[
                        "signal_timestamp"
                    ]
                    < end
                )
            ]

        if skips.empty:

            window_skips = skips

        else:

            window_skips = skips[
                (
                    skips[
                        "signal_timestamp"
                    ]
                    >= start
                )
                &
                (
                    skips[
                        "signal_timestamp"
                    ]
                    < end
                )
            ]

        wins = (
            int(
                (
                    window_trades[
                        "result"
                    ]
                    == "WIN"
                ).sum()
            )
            if not window_trades.empty
            else 0
        )

        fees = (
            float(
                window_trades[
                    "fees_total"
                ].sum()
            )
            if not window_trades.empty
            else 0.0
        )

        rows.append(
            {
                "strategy":
                    strategy,

                "window":
                    f"W{w}",

                "window_start":
                    start,

                "window_end":
                    end,

                "start_equity":
                    eq_start,

                "end_equity":
                    eq_end,

                "window_return_pct":
                    (
                        eq_end
                        / eq_start
                        - 1
                    ) * 100
                    if eq_start != 0
                    else np.nan,

                "trades_entered":
                    int(
                        len(
                            window_trades
                        )
                    ),

                "wins":
                    wins,

                "losses":
                    int(
                        len(
                            window_trades
                        )
                        - wins
                    ),

                "win_rate_pct":
                    (
                        wins
                        / len(
                            window_trades
                        )
                        * 100
                        if len(
                            window_trades
                        )
                        else 0.0
                    ),

                "fees_usd":
                    fees,

                "net_pnl_of_trades_entered_usd":
                    (
                        float(
                            window_trades[
                                "net_pnl"
                            ].sum()
                        )
                        if not window_trades.empty
                        else 0.0
                    ),

                "skipped_signals":
                    int(
                        len(
                            window_skips
                        )
                    ),

                "busy_skips":
                    (
                        int(
                            (
                                window_skips[
                                    "reason"
                                ]
                                == "BUSY_AT_ENTRY_OPEN"
                            ).sum()
                        )
                        if not window_skips.empty
                        else 0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    signal_file = find_signal_file()

    history = load_combined_history()

    dataset_start = history[
        "timestamp"
    ].min()

    signals = load_signals(
        signal_file,
        history
    )

    max_test_window = int(
        signals[
            "window_number"
        ].max()
    )

    (
        model_predictions,
        model_diagnostics,
        model_coefficients,
    ) = precompute_model_actions(
        signals=signals,
        dataset_start=dataset_start
    )

    candidates = prepare_candidates(
        signals=signals,
        history=history,
        model_predictions=model_predictions
    )

    strategies = [
        "EXECUTABLE_V2",
        "ALWAYS_INVERSE",
        "ALWAYS_BREAKOUT",
    ]

    results = []

    print()
    print("=" * 118)
    print("SIMULATING CONTINUOUS PORTFOLIOS")
    print("=" * 118)

    for strategy in strategies:

        print(
            f"Running {strategy}..."
        )

        result = simulate_strategy(
            strategy=strategy,
            history=history,
            candidates=candidates,
            dataset_start=dataset_start,
            max_test_window=max_test_window
        )

        results.append(
            result
        )

    eligible_signals = len(
        candidates
    )

    overall_rows = [
        overall_metrics(
            result,
            eligible_signals
        )
        for result in results
    ]

    overall_df = pd.DataFrame(
        overall_rows
    )

    trade_frames = [
        result[
            "trades"
        ]
        for result in results
        if not result[
            "trades"
        ].empty
    ]

    all_trades_df = (
        pd.concat(
            trade_frames,
            ignore_index=True
        )
        if trade_frames
        else pd.DataFrame()
    )

    skip_frames = [
        result[
            "skips"
        ]
        for result in results
        if not result[
            "skips"
        ].empty
    ]

    all_skips_df = (
        pd.concat(
            skip_frames,
            ignore_index=True
        )
        if skip_frames
        else pd.DataFrame()
    )

    equity_frames = [
        result[
            "equity"
        ]
        for result in results
    ]

    all_equity_df = pd.concat(
        equity_frames,
        ignore_index=True
    )

    window_frames = [
        build_window_summary(
            result=result,
            dataset_start=dataset_start,
            max_test_window=max_test_window
        )
        for result in results
    ]

    windows_df = pd.concat(
        window_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 150)
    print("EXECUTABLE OVERALL RESULT")
    print("=" * 150)

    print(
        overall_df[
            [
                "strategy",
                "initial_balance",
                "final_balance",
                "return_pct",
                "eligible_signals",
                "executed_trades",
                "busy_skips",
                "win_rate_pct",
                "profit_factor",
                "max_drawdown_pct",
                "max_consecutive_losses",
                "fees_usd",
                "avg_hold_bars",
                "exposure_bars_pct",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 150)
    print("CONTINUOUS WINDOW RETURNS")
    print("=" * 150)

    pivot = windows_df.pivot(
        index="window",
        columns="strategy",
        values="window_return_pct"
    )

    print(
        pivot.to_string()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    overall_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_overall_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    windows_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_windows_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    trades_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_trades_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    skips_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_skipped_signals_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    equity_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_equity_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    diagnostics_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_model_diagnostics_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    coefficients_file = os.path.join(
        REPORT_DIR,
        f"executable_v2_model_coefficients_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    overall_df.to_csv(
        overall_file,
        index=False
    )

    windows_df.to_csv(
        windows_file,
        index=False
    )

    all_trades_df.to_csv(
        trades_file,
        index=False
    )

    all_skips_df.to_csv(
        skips_file,
        index=False
    )

    all_equity_df.to_csv(
        equity_file,
        index=False
    )

    model_diagnostics.to_csv(
        diagnostics_file,
        index=False
    )

    model_coefficients.to_csv(
        coefficients_file,
        index=False
    )

    print()
    print("=" * 118)
    print("EXECUTABLE V2 TEST FINISHED")
    print("=" * 118)
    print(f"Overall:          {overall_file}")
    print(f"Windows:          {windows_file}")
    print(f"Trades:           {trades_file}")
    print(f"Skipped signals:  {skips_file}")
    print(f"Equity curve:     {equity_file}")
    print(f"Model diagnostics:{diagnostics_file}")
    print(f"Coefficients:     {coefficients_file}")
    print("=" * 118)

    print()
    print(
        "Important: this is now one-position-at-a-time with "
        "fees and compounding, but funding and slippage are "
        "still not modeled."
    )


if __name__ == "__main__":
    main()
