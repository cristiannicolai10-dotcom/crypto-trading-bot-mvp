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
        "Install with:\n"
        "    pip install scikit-learn\n"
    ) from exc


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )

import backtest_inverse as bt


# ============================================================
# FROZEN STRATEGY / RISK RULES
# ============================================================

BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"

INITIAL_BALANCE = 1000.0

# User-defined capital allocation + leverage.
MARGIN_FRACTION = 0.20
LEVERAGE = 10.0

# Existing initial stop remains frozen.
INITIAL_SL_ATR = 1.25

# Fixed TP intentionally disabled:
# the profit-lock / trailing rules become the exit system.
FIXED_TP_ENABLED = False

MAX_HOLD_MINUTES = 180

# Profit rules are interpreted as NET ROE on the initial margin,
# after modeled entry + projected exit fee/slippage and accrued funding.
LOCK_TRIGGER_ROE = 0.10
LOCK_FLOOR_ROE = 0.05

TRAIL_TRIGGER_ROE = 0.20
TRAIL_GAP_ROE = 0.10

TAKER_FEE_RATE = 0.00055

# Cost stress only. This is not parameter optimization.
SLIPPAGE_BPS_GRID = [
    0,
    1,
    2,
    3,
    4,
    5,
]

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

# Approximation only, not an exact Bybit liquidation formula.
# With 10x, ~9.5% adverse price move is a conservative proxy.
LIQUIDATION_ADVERSE_MOVE = 0.095

PRE_SAMPLE_4H = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_presample_2019_2023.parquet"
)

DEVELOPMENT_4H = os.path.join(
    DATA_DIR,
    "BTCUSDT_4h_3y.parquet"
)

EXECUTION_1M_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_1m_signal_windows_W5plus.parquet"
)

FUNDING_FILE = os.path.join(
    DATA_DIR,
    "BTCUSDT_funding_W5plus.parquet"
)


# ============================================================
# INPUT
# ============================================================

def find_signal_file():

    if len(sys.argv) > 1:

        candidate = sys.argv[1]

        if not os.path.isabs(candidate):
            candidate = os.path.join(
                BASE_DIR,
                candidate
            )

        if not os.path.exists(candidate):
            raise RuntimeError(
                f"Signal file not found: {candidate}"
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
            "No price_action_signal_features CSV found."
        )

    return files[-1]


# ============================================================
# LOAD 4H HISTORY / ATR
# ============================================================

def load_4h_history():

    frames = []

    for filepath, label in [
        (PRE_SAMPLE_4H, "pre-sample"),
        (DEVELOPMENT_4H, "development"),
    ]:

        if not os.path.exists(filepath):
            raise RuntimeError(
                f"Missing {label}: {filepath}"
            )

        df = pd.read_parquet(filepath)

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

        frames.append(df)

    history = pd.concat(
        frames,
        ignore_index=True
    )

    history = (
        history
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

    history = bt.calculate_indicators(
        history
    )

    if "atr" not in history.columns:
        raise RuntimeError(
            "Indicator engine did not produce ATR."
        )

    history["atr"] = pd.to_numeric(
        history["atr"],
        errors="coerce"
    )

    return history


# ============================================================
# LOAD SIGNALS
# ============================================================

def window_number(value):

    m = re.fullmatch(
        r"W(\d+)",
        str(value)
    )

    if not m:
        return np.nan

    return int(m.group(1))


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


def load_signals(
    signal_file,
    history
):

    df = pd.read_csv(signal_file)

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
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Signal CSV missing columns:\n"
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

    atr_source = history[
        [
            "timestamp",
            "atr",
        ]
    ].copy()

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

    if df["atr"].isna().any():

        count = int(
            df["atr"].isna().sum()
        )

        raise RuntimeError(
            f"ATR missing for {count} signals."
        )

    # Next 4h candle open timestamp.
    df["entry_timestamp"] = (
        df["signal_timestamp"]
        + pd.Timedelta(hours=4)
    )

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
# LOAD 1M + FUNDING
# ============================================================

def load_execution_data():

    if not os.path.exists(
        EXECUTION_1M_FILE
    ):
        raise RuntimeError(
            f"Missing 1m execution file: "
            f"{EXECUTION_1M_FILE}\n"
            "Run download_v3_execution_data.py first."
        )

    one_min = pd.read_parquet(
        EXECUTION_1M_FILE
    )

    one_min["timestamp"] = pd.to_datetime(
        one_min["timestamp"],
        utc=True,
        errors="coerce"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:

        one_min[col] = pd.to_numeric(
            one_min[col],
            errors="coerce"
        )

    one_min = (
        one_min
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

    if os.path.exists(
        FUNDING_FILE
    ):

        funding = pd.read_parquet(
            FUNDING_FILE
        )

        if not funding.empty:

            funding["timestamp"] = pd.to_datetime(
                funding["timestamp"],
                utc=True,
                errors="coerce"
            )

            funding["funding_rate"] = pd.to_numeric(
                funding["funding_rate"],
                errors="coerce"
            )

            funding = (
                funding
                .dropna(
                    subset=[
                        "timestamp",
                        "funding_rate",
                    ]
                )
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

    else:

        funding = pd.DataFrame(
            columns=[
                "timestamp",
                "funding_rate",
            ]
        )

    return one_min, funding


# ============================================================
# V2 PURGED MODEL
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

    raw = signals[
        (
            signals["signal_timestamp"]
            < test_start
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

    purged = raw[
        (
            raw["label_known_at"]
            .notna()
        )
        &
        (
            raw["label_known_at"]
            < test_start
        )
    ].copy()

    if purged.empty:
        raise RuntimeError(
            f"No purged training data before {test_start}"
        )

    target = (
        purged["direction_target"]
        .astype(int)
    )

    if target.nunique() == 1:

        return {
            "kind": "constant",
            "value": int(target.iloc[0]),
            "model": None,
        }, raw, purged

    model = make_model()

    model.fit(
        purged[FEATURES],
        target
    )

    return {
        "kind": "model",
        "value": None,
        "model": model,
    }, raw, purged


def predict_direction(
    fitted,
    test
):

    if fitted["kind"] == "constant":

        pred = np.full(
            len(test),
            fitted["value"],
            dtype=int
        )

        prob = np.full(
            len(test),
            float(fitted["value"]),
            dtype=float
        )

        return pred, prob

    prob = fitted[
        "model"
    ].predict_proba(
        test[FEATURES]
    )[:, 1]

    pred = (
        prob >= MODEL_THRESHOLD
    ).astype(int)

    return pred, prob


def add_model_predictions(
    signals,
    dataset_start
):

    max_window = int(
        signals["window_number"]
        .max()
    )

    frames = []
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
                signals["signal_timestamp"]
                >= start
            )
            &
            (
                signals["signal_timestamp"]
                < end
            )
        ].copy()

        if test.empty:
            continue

        fitted, raw, purged = fit_purged_model(
            signals,
            start
        )

        pred, prob = predict_direction(
            fitted,
            test
        )

        test["model_action"] = np.where(
            pred == 1,
            "INVERSE",
            "BREAKOUT"
        )

        test["inverse_probability"] = prob

        frames.append(test)

        diagnostics.append(
            {
                "window": f"W{w}",
                "raw_train": int(len(raw)),
                "purged_train": int(len(purged)),
                "purged_count":
                    int(len(raw) - len(purged)),
                "test_signals": int(len(test)),
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

    out = pd.concat(
        frames,
        ignore_index=True
    )

    return out, pd.DataFrame(diagnostics)


# ============================================================
# PRICE / COST HELPERS
# ============================================================

def adverse_slippage_fill(
    raw_price,
    direction,
    is_entry,
    slip_rate
):

    # Market entry:
    # LONG buys higher, SHORT sells lower.
    if is_entry:

        if direction == "LONG":
            return raw_price * (
                1 + slip_rate
            )

        return raw_price * (
            1 - slip_rate
        )

    # Market exit:
    # LONG sells lower, SHORT buys higher.
    if direction == "LONG":
        return raw_price * (
            1 - slip_rate
        )

    return raw_price * (
        1 + slip_rate
    )


def net_pnl_if_exit(
    position,
    exit_fill,
    funding_pnl
):

    qty = position["quantity"]

    if position["direction"] == "LONG":

        gross = qty * (
            exit_fill
            - position["entry_fill"]
        )

    else:

        gross = qty * (
            position["entry_fill"]
            - exit_fill
        )

    exit_fee = (
        qty
        * exit_fill
        * TAKER_FEE_RATE
    )

    net = (
        gross
        - position["entry_fee"]
        - exit_fee
        + funding_pnl
    )

    return (
        gross,
        exit_fee,
        net
    )


def net_roe_if_exit_at_raw_price(
    position,
    raw_exit_price,
    funding_pnl,
    slip_rate
):

    exit_fill = adverse_slippage_fill(
        raw_price=raw_exit_price,
        direction=position["direction"],
        is_entry=False,
        slip_rate=slip_rate
    )

    _, _, net = net_pnl_if_exit(
        position,
        exit_fill,
        funding_pnl
    )

    return (
        net
        / position["margin"]
    )


def stop_trigger_for_target_net_roe(
    position,
    target_roe,
    funding_pnl,
    slip_rate
):

    qty = position["quantity"]
    entry = position["entry_fill"]
    margin = position["margin"]
    entry_fee = position["entry_fee"]

    target_net = (
        target_roe
        * margin
    )

    if position["direction"] == "LONG":

        # Solve for actual exit fill E:
        #
        # qty*(E-entry)
        # - entry_fee
        # - qty*E*fee
        # + funding
        # = target_net
        #
        numerator = (
            target_net
            + qty * entry
            + entry_fee
            - funding_pnl
        )

        denominator = (
            qty
            * (
                1 - TAKER_FEE_RATE
            )
        )

        exit_fill_needed = (
            numerator
            / denominator
        )

        # modeled stop execution:
        # exit_fill = trigger*(1-slip)
        trigger = (
            exit_fill_needed
            / (
                1 - slip_rate
            )
        )

    else:

        # qty*(entry-E)
        # - entry_fee
        # - qty*E*fee
        # + funding
        # = target_net
        numerator = (
            qty * entry
            - entry_fee
            + funding_pnl
            - target_net
        )

        denominator = (
            qty
            * (
                1 + TAKER_FEE_RATE
            )
        )

        exit_fill_needed = (
            numerator
            / denominator
        )

        # modeled buy-to-close:
        # exit_fill = trigger*(1+slip)
        trigger = (
            exit_fill_needed
            / (
                1 + slip_rate
            )
        )

    return float(trigger)


# ============================================================
# FUNDING
# ============================================================

def funding_events_for_trade(
    funding,
    start_ts,
    end_ts
):

    if funding.empty:
        return funding

    return funding[
        (
            funding["timestamp"]
            > start_ts
        )
        &
        (
            funding["timestamp"]
            <= end_ts
        )
    ].copy()


def funding_payment(
    position,
    price,
    rate
):

    position_value = (
        position["quantity"]
        * price
    )

    # Positive rate:
    # long pays, short receives.
    sign = (
        -1.0
        if position["direction"] == "LONG"
        else 1.0
    )

    return (
        sign
        * position_value
        * rate
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def strategy_direction(
    strategy,
    candidate
):

    if strategy == "V2":

        if candidate["model_action"] == "INVERSE":
            return "LONG"

        return "SHORT"

    if strategy == "ALWAYS_INVERSE":
        return "LONG"

    if strategy == "ALWAYS_BREAKOUT":
        return "SHORT"

    raise ValueError(strategy)


def simulate_one_trade(
    candidate,
    direction,
    one_min,
    funding,
    balance,
    slippage_bps
):

    slip_rate = (
        slippage_bps
        / 10000.0
    )

    planned_entry = candidate[
        "entry_timestamp"
    ]

    cutoff = (
        planned_entry
        + pd.Timedelta(
            minutes=MAX_HOLD_MINUTES
        )
    )

    bars = one_min[
        (
            one_min["timestamp"]
            >= planned_entry
        )
        &
        (
            one_min["timestamp"]
            <= cutoff
        )
    ].copy()

    if bars.empty:

        return None, "NO_1M_DATA"

    bars = bars.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    entry_bar = bars.iloc[0]

    if (
        entry_bar["timestamp"]
        > planned_entry
        + pd.Timedelta(minutes=1)
    ):

        return None, "ENTRY_1M_GAP"

    raw_entry = float(
        entry_bar["open"]
    )

    entry_fill = adverse_slippage_fill(
        raw_price=raw_entry,
        direction=direction,
        is_entry=True,
        slip_rate=slip_rate
    )

    margin = (
        balance
        * MARGIN_FRACTION
    )

    notional = (
        margin
        * LEVERAGE
    )

    quantity = (
        notional
        / entry_fill
    )

    entry_fee = (
        notional
        * TAKER_FEE_RATE
    )

    atr = float(
        candidate["atr"]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):

        return None, "INVALID_ATR"

    if direction == "LONG":

        initial_stop = (
            entry_fill
            - INITIAL_SL_ATR * atr
        )

        liquidation_proxy = (
            entry_fill
            * (
                1
                - LIQUIDATION_ADVERSE_MOVE
            )
        )

    else:

        initial_stop = (
            entry_fill
            + INITIAL_SL_ATR * atr
        )

        liquidation_proxy = (
            entry_fill
            * (
                1
                + LIQUIDATION_ADVERSE_MOVE
            )
        )

    position = {
        "direction": direction,
        "margin": margin,
        "notional": notional,
        "quantity": quantity,
        "raw_entry": raw_entry,
        "entry_fill": entry_fill,
        "entry_fee": entry_fee,
        "initial_stop": initial_stop,
        "stop_trigger": initial_stop,
        "liquidation_proxy":
            liquidation_proxy,
    }

    funding_rows = funding_events_for_trade(
        funding,
        planned_entry,
        cutoff
    )

    processed_funding = set()

    funding_pnl = 0.0

    peak_net_roe = -np.inf

    lock_5_activated = False
    trailing_activated = False

    current_floor_roe = None

    exit_raw_reference = None
    exit_fill = None
    exit_reason = None
    exit_timestamp = None

    bars_held = 0

    for _, bar in bars.iterrows():

        ts = bar["timestamp"]

        if ts < planned_entry:
            continue

        bars_held += 1

        # ----------------------------------------------------
        # Funding settlements up to this minute.
        # ----------------------------------------------------

        due = funding_rows[
            funding_rows["timestamp"]
            <= ts
        ]

        for _, frow in due.iterrows():

            fts = frow["timestamp"]

            if fts in processed_funding:
                continue

            funding_pnl += funding_payment(
                position=position,
                price=float(
                    bar["open"]
                ),
                rate=float(
                    frow["funding_rate"]
                )
            )

            processed_funding.add(
                fts
            )

        # ----------------------------------------------------
        # TIME EXIT at first 1m open at/after +180m.
        # ----------------------------------------------------

        if ts >= cutoff:

            exit_raw_reference = float(
                bar["open"]
            )

            exit_fill = adverse_slippage_fill(
                raw_price=exit_raw_reference,
                direction=direction,
                is_entry=False,
                slip_rate=slip_rate
            )

            exit_reason = "TIME_3H"
            exit_timestamp = ts
            break

        open_p = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        # ----------------------------------------------------
        # Gap liquidation proxy first.
        # ----------------------------------------------------

        if direction == "LONG":

            if open_p <= position[
                "liquidation_proxy"
            ]:

                exit_raw_reference = open_p
                exit_fill = adverse_slippage_fill(
                    raw_price=open_p,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )
                exit_reason = "LIQUIDATION_PROXY_GAP"
                exit_timestamp = ts
                break

        else:

            if open_p >= position[
                "liquidation_proxy"
            ]:

                exit_raw_reference = open_p
                exit_fill = adverse_slippage_fill(
                    raw_price=open_p,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )
                exit_reason = "LIQUIDATION_PROXY_GAP"
                exit_timestamp = ts
                break

        # ----------------------------------------------------
        # Existing stop applies throughout this minute.
        # Newly tightened stop below is effective from NEXT
        # minute. This avoids intraminute lookahead.
        # ----------------------------------------------------

        stop = position[
            "stop_trigger"
        ]

        if direction == "LONG":

            if open_p <= stop:

                exit_raw_reference = open_p
                exit_fill = adverse_slippage_fill(
                    raw_price=open_p,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )
                exit_reason = "STOP_GAP"
                exit_timestamp = ts
                break

            if low <= stop:

                exit_raw_reference = stop
                exit_fill = adverse_slippage_fill(
                    raw_price=stop,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )

                if trailing_activated:
                    exit_reason = "TRAILING_STOP"
                elif lock_5_activated:
                    exit_reason = "LOCK_5_STOP"
                else:
                    exit_reason = "INITIAL_SL"

                exit_timestamp = ts
                break

        else:

            if open_p >= stop:

                exit_raw_reference = open_p
                exit_fill = adverse_slippage_fill(
                    raw_price=open_p,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )
                exit_reason = "STOP_GAP"
                exit_timestamp = ts
                break

            if high >= stop:

                exit_raw_reference = stop
                exit_fill = adverse_slippage_fill(
                    raw_price=stop,
                    direction=direction,
                    is_entry=False,
                    slip_rate=slip_rate
                )

                if trailing_activated:
                    exit_reason = "TRAILING_STOP"
                elif lock_5_activated:
                    exit_reason = "LOCK_5_STOP"
                else:
                    exit_reason = "INITIAL_SL"

                exit_timestamp = ts
                break

        # ----------------------------------------------------
        # Profit management update for NEXT minute.
        # Use favorable extreme of current 1m candle.
        # ----------------------------------------------------

        favorable_raw = (
            high
            if direction == "LONG"
            else low
        )

        favorable_net_roe = (
            net_roe_if_exit_at_raw_price(
                position=position,
                raw_exit_price=favorable_raw,
                funding_pnl=funding_pnl,
                slip_rate=slip_rate
            )
        )

        peak_net_roe = max(
            peak_net_roe,
            favorable_net_roe
        )

        if (
            peak_net_roe
            >= LOCK_TRIGGER_ROE
        ):

            lock_5_activated = True

            lock_stop = (
                stop_trigger_for_target_net_roe(
                    position=position,
                    target_roe=
                        LOCK_FLOOR_ROE,
                    funding_pnl=
                        funding_pnl,
                    slip_rate=
                        slip_rate
                )
            )

            if direction == "LONG":

                position[
                    "stop_trigger"
                ] = max(
                    position[
                        "stop_trigger"
                    ],
                    lock_stop
                )

            else:

                position[
                    "stop_trigger"
                ] = min(
                    position[
                        "stop_trigger"
                    ],
                    lock_stop
                )

        if (
            peak_net_roe
            >= TRAIL_TRIGGER_ROE
        ):

            trailing_activated = True

            current_floor_roe = (
                peak_net_roe
                - TRAIL_GAP_ROE
            )

            trailing_stop = (
                stop_trigger_for_target_net_roe(
                    position=position,
                    target_roe=
                        current_floor_roe,
                    funding_pnl=
                        funding_pnl,
                    slip_rate=
                        slip_rate
                )
            )

            if direction == "LONG":

                position[
                    "stop_trigger"
                ] = max(
                    position[
                        "stop_trigger"
                    ],
                    trailing_stop
                )

            else:

                position[
                    "stop_trigger"
                ] = min(
                    position[
                        "stop_trigger"
                    ],
                    trailing_stop
                )

    # If +180m bar was unavailable, close at final available close.
    if exit_fill is None:

        final_bar = bars.iloc[-1]

        exit_raw_reference = float(
            final_bar["close"]
        )

        exit_fill = adverse_slippage_fill(
            raw_price=exit_raw_reference,
            direction=direction,
            is_entry=False,
            slip_rate=slip_rate
        )

        exit_reason = "DATA_END"
        exit_timestamp = final_bar[
            "timestamp"
        ]

    gross, exit_fee, net = net_pnl_if_exit(
        position,
        exit_fill,
        funding_pnl
    )

    realized_roe = (
        net
        / margin
    )

    entry_slippage_cost = (
        quantity
        * abs(
            entry_fill
            - raw_entry
        )
    )

    exit_slippage_cost = (
        quantity
        * abs(
            exit_fill
            - exit_raw_reference
        )
    )

    hold_minutes = (
        (
            exit_timestamp
            - planned_entry
        )
        .total_seconds()
        / 60.0
    )

    trade = {
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
        "window":
            candidate[
                "window"
            ],
        "direction":
            direction,
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
        "entry_timestamp":
            planned_entry,
        "exit_timestamp":
            exit_timestamp,
        "raw_entry":
            raw_entry,
        "entry_fill":
            entry_fill,
        "exit_raw_reference":
            exit_raw_reference,
        "exit_fill":
            exit_fill,
        "atr":
            atr,
        "margin":
            margin,
        "notional":
            notional,
        "quantity":
            quantity,
        "initial_stop":
            initial_stop,
        "final_stop":
            position[
                "stop_trigger"
            ],
        "peak_net_roe_pct":
            (
                peak_net_roe
                * 100
                if np.isfinite(
                    peak_net_roe
                )
                else np.nan
            ),
        "lock_5_activated":
            lock_5_activated,
        "trailing_activated":
            trailing_activated,
        "final_trailing_floor_roe_pct":
            (
                current_floor_roe
                * 100
                if current_floor_roe
                is not None
                else np.nan
            ),
        "gross_pnl":
            gross,
        "entry_fee":
            entry_fee,
        "exit_fee":
            exit_fee,
        "funding_pnl":
            funding_pnl,
        "entry_slippage_cost":
            entry_slippage_cost,
        "exit_slippage_cost":
            exit_slippage_cost,
        "net_pnl":
            net,
        "realized_net_roe_pct":
            realized_roe
            * 100,
        "result":
            (
                "WIN"
                if net > 0
                else "LOSS"
            ),
        "exit_reason":
            exit_reason,
        "bars_held_1m":
            bars_held,
        "hold_minutes":
            hold_minutes,
        "balance_before":
            balance,
        "balance_after":
            balance + net,
    }

    return trade, None


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(
    strategy,
    candidates,
    one_min,
    funding,
    slippage_bps
):

    balance = INITIAL_BALANCE

    trades = []
    skips = []

    busy_until = pd.Timestamp.min.tz_localize(
        "UTC"
    )

    peak_balance = INITIAL_BALANCE
    max_dd_pct = 0.0

    for _, candidate in (
        candidates
        .sort_values(
            [
                "entry_timestamp",
                "signal_number",
            ]
        )
        .iterrows()
    ):

        entry_ts = candidate[
            "entry_timestamp"
        ]

        if entry_ts < busy_until:

            skips.append(
                {
                    "strategy":
                        strategy,
                    "slippage_bps":
                        slippage_bps,
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
                        entry_ts,
                    "window":
                        candidate[
                            "window"
                        ],
                    "reason":
                        "BUSY_AT_ENTRY",
                }
            )

            continue

        direction = strategy_direction(
            strategy,
            candidate
        )

        trade, error = simulate_one_trade(
            candidate=candidate,
            direction=direction,
            one_min=one_min,
            funding=funding,
            balance=balance,
            slippage_bps=slippage_bps
        )

        if trade is None:

            skips.append(
                {
                    "strategy":
                        strategy,
                    "slippage_bps":
                        slippage_bps,
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
                        entry_ts,
                    "window":
                        candidate[
                            "window"
                        ],
                    "reason":
                        error,
                }
            )

            continue

        trade[
            "strategy"
        ] = strategy

        trade[
            "slippage_bps"
        ] = slippage_bps

        trades.append(trade)

        balance = trade[
            "balance_after"
        ]

        busy_until = trade[
            "exit_timestamp"
        ]

        peak_balance = max(
            peak_balance,
            balance
        )

        dd_pct = (
            (
                peak_balance
                - balance
            )
            / peak_balance
            * 100
        )

        max_dd_pct = max(
            max_dd_pct,
            dd_pct
        )

    trades_df = pd.DataFrame(trades)
    skips_df = pd.DataFrame(skips)

    return (
        trades_df,
        skips_df,
        balance,
        max_dd_pct
    )


# ============================================================
# METRICS
# ============================================================

def profit_factor(trades):

    if trades.empty:
        return np.nan

    gp = trades.loc[
        trades["net_pnl"] > 0,
        "net_pnl"
    ].sum()

    gl = -trades.loc[
        trades["net_pnl"] < 0,
        "net_pnl"
    ].sum()

    if gl == 0:
        return np.nan

    return float(gp / gl)


def max_consecutive_losses(trades):

    if trades.empty:
        return 0

    best = 0
    current = 0

    for result in trades[
        "result"
    ]:

        if result == "LOSS":
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def summarize(
    strategy,
    slippage_bps,
    trades,
    skips,
    final_balance,
    max_dd_pct,
    eligible_signals
):

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

    total_fees = (
        float(
            (
                trades[
                    "entry_fee"
                ]
                + trades[
                    "exit_fee"
                ]
            ).sum()
        )
        if not trades.empty
        else 0.0
    )

    funding_pnl = (
        float(
            trades[
                "funding_pnl"
            ].sum()
        )
        if not trades.empty
        else 0.0
    )

    slippage_cost = (
        float(
            (
                trades[
                    "entry_slippage_cost"
                ]
                + trades[
                    "exit_slippage_cost"
                ]
            ).sum()
        )
        if not trades.empty
        else 0.0
    )

    return {
        "strategy":
            strategy,
        "slippage_bps":
            slippage_bps,
        "initial_balance":
            INITIAL_BALANCE,
        "final_balance":
            final_balance,
        "return_pct":
            (
                final_balance
                / INITIAL_BALANCE
                - 1
            ) * 100,
        "eligible_signals":
            eligible_signals,
        "executed_trades":
            int(len(trades)),
        "skipped_signals":
            int(len(skips)),
        "wins":
            wins,
        "losses":
            int(
                len(trades)
                - wins
            ),
        "win_rate_pct":
            (
                wins
                / len(trades)
                * 100
                if len(trades)
                else 0.0
            ),
        "profit_factor":
            profit_factor(trades),
        "max_drawdown_pct_trade_close":
            max_dd_pct,
        "max_consecutive_losses":
            max_consecutive_losses(
                trades
            ),
        "fees_usd":
            total_fees,
        "funding_pnl_usd":
            funding_pnl,
        "slippage_cost_usd":
            slippage_cost,
        "net_pnl_usd":
            (
                float(
                    trades[
                        "net_pnl"
                    ].sum()
                )
                if not trades.empty
                else 0.0
            ),
        "avg_realized_net_roe_pct":
            (
                float(
                    trades[
                        "realized_net_roe_pct"
                    ].mean()
                )
                if not trades.empty
                else np.nan
            ),
        "avg_hold_minutes":
            (
                float(
                    trades[
                        "hold_minutes"
                    ].mean()
                )
                if not trades.empty
                else np.nan
            ),
        "max_hold_minutes":
            (
                float(
                    trades[
                        "hold_minutes"
                    ].max()
                )
                if not trades.empty
                else np.nan
            ),
        "lock5_activated_count":
            (
                int(
                    trades[
                        "lock_5_activated"
                    ].sum()
                )
                if not trades.empty
                else 0
            ),
        "trailing_activated_count":
            (
                int(
                    trades[
                        "trailing_activated"
                    ].sum()
                )
                if not trades.empty
                else 0
            ),
        "time_exit_count":
            (
                int(
                    (
                        trades[
                            "exit_reason"
                        ]
                        == "TIME_3H"
                    ).sum()
                )
                if not trades.empty
                else 0
            ),
        "initial_sl_count":
            (
                int(
                    (
                        trades[
                            "exit_reason"
                        ]
                        == "INITIAL_SL"
                    ).sum()
                )
                if not trades.empty
                else 0
            ),
        "lock5_exit_count":
            (
                int(
                    (
                        trades[
                            "exit_reason"
                        ]
                        == "LOCK_5_STOP"
                    ).sum()
                )
                if not trades.empty
                else 0
            ),
        "trailing_exit_count":
            (
                int(
                    (
                        trades[
                            "exit_reason"
                        ]
                        == "TRAILING_STOP"
                    ).sum()
                )
                if not trades.empty
                else 0
            ),
        "liquidation_proxy_count":
            (
                int(
                    trades[
                        "exit_reason"
                    ]
                    .str
                    .contains(
                        "LIQUIDATION",
                        na=False
                    )
                    .sum()
                )
                if not trades.empty
                else 0
            ),
    }


def build_window_summary(
    trades
):

    if trades.empty:
        return pd.DataFrame()

    rows = []

    for (
        strategy,
        slippage_bps,
        window
    ), group in trades.groupby(
        [
            "strategy",
            "slippage_bps",
            "window",
        ]
    ):

        start_balance = float(
            group.iloc[0][
                "balance_before"
            ]
        )

        end_balance = float(
            group.iloc[-1][
                "balance_after"
            ]
        )

        wins = int(
            (
                group[
                    "result"
                ]
                == "WIN"
            ).sum()
        )

        rows.append(
            {
                "strategy":
                    strategy,
                "slippage_bps":
                    slippage_bps,
                "window":
                    window,
                "start_balance":
                    start_balance,
                "end_balance":
                    end_balance,
                "return_pct":
                    (
                        end_balance
                        / start_balance
                        - 1
                    ) * 100,
                "trades":
                    int(len(group)),
                "wins":
                    wins,
                "losses":
                    int(
                        len(group)
                        - wins
                    ),
                "win_rate_pct":
                    wins
                    / len(group)
                    * 100,
                "net_pnl_usd":
                    float(
                        group[
                            "net_pnl"
                        ].sum()
                    ),
                "funding_pnl_usd":
                    float(
                        group[
                            "funding_pnl"
                        ].sum()
                    ),
                "avg_hold_minutes":
                    float(
                        group[
                            "hold_minutes"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    signal_file = find_signal_file()

    print()
    print("=" * 120)
    print("V3 - 10X / 3H / PROFIT-LOCK / COST-STRESS")
    print("=" * 120)
    print(
        f"Signal source: {signal_file}"
    )
    print()
    print(
        f"Margin fraction: {MARGIN_FRACTION * 100:.0f}%"
    )
    print(
        f"Leverage: {LEVERAGE:.0f}x"
    )
    print(
        f"Gross notional/account at entry: "
        f"{MARGIN_FRACTION * LEVERAGE:.1f}x"
    )
    print(
        f"Initial SL: {INITIAL_SL_ATR} ATR"
    )
    print(
        "Fixed TP: OFF"
    )
    print(
        f"Max hold: {MAX_HOLD_MINUTES} min"
    )
    print(
        "Profit rules use modeled NET ROE on initial margin:"
    )
    print(
        "  >=10% -> stop floor +5%"
    )
    print(
        "  >=20% -> trailing active, "
        "floor = peak NET ROE - 10 percentage points"
    )
    print(
        f"Taker fee: {TAKER_FEE_RATE * 100:.3f}% / side"
    )
    print(
        f"Slippage stress: {SLIPPAGE_BPS_GRID} bps / side"
    )
    print(
        "Funding: historical Bybit funding file"
    )
    print(
        "Execution candles: 1 minute"
    )
    print("=" * 120)

    history = load_4h_history()

    dataset_start = history[
        "timestamp"
    ].min()

    signals = load_signals(
        signal_file,
        history
    )

    candidates, model_diagnostics = (
        add_model_predictions(
            signals,
            dataset_start
        )
    )

    candidates = candidates[
        candidates[
            "window_number"
        ]
        >= FIRST_TEST_WINDOW
    ].copy()

    one_min, funding = (
        load_execution_data()
    )

    print()
    print(
        f"1m rows: {len(one_min):,}"
    )
    print(
        f"Funding rows: {len(funding):,}"
    )
    print(
        f"Executable candidate signals: "
        f"{len(candidates)}"
    )

    strategies = [
        "V2",
        "ALWAYS_INVERSE",
        "ALWAYS_BREAKOUT",
    ]

    all_trades = []
    all_skips = []
    summary_rows = []

    print()
    print("=" * 140)
    print("RUNNING COST STRESS")
    print("=" * 140)

    for slippage_bps in (
        SLIPPAGE_BPS_GRID
    ):

        for strategy in strategies:

            (
                trades,
                skips,
                final_balance,
                max_dd_pct,
            ) = simulate_portfolio(
                strategy=strategy,
                candidates=candidates,
                one_min=one_min,
                funding=funding,
                slippage_bps=slippage_bps
            )

            all_trades.append(trades)
            all_skips.append(skips)

            summary = summarize(
                strategy=strategy,
                slippage_bps=slippage_bps,
                trades=trades,
                skips=skips,
                final_balance=final_balance,
                max_dd_pct=max_dd_pct,
                eligible_signals=len(
                    candidates
                )
            )

            summary_rows.append(summary)

            print(
                f"slip={slippage_bps:>2} bps | "
                f"{strategy:<15} | "
                f"trades={summary['executed_trades']:<3} | "
                f"ret={summary['return_pct']:+8.2f}% | "
                f"PF={summary['profit_factor']:.3f} | "
                f"DD={summary['max_drawdown_pct_trade_close']:.2f}% | "
                f"funding={summary['funding_pnl_usd']:+.2f} | "
                f"maxhold={summary['max_hold_minutes']:.0f}m"
            )

    overall = pd.DataFrame(
        summary_rows
    )

    trades_df = pd.concat(
        all_trades,
        ignore_index=True
    )

    nonempty_skips = [
        df for df in all_skips
        if not df.empty
    ]

    skips_df = (
        pd.concat(
            nonempty_skips,
            ignore_index=True
        )
        if nonempty_skips
        else pd.DataFrame()
    )

    windows = build_window_summary(
        trades_df
    )

    print()
    print("=" * 140)
    print("V2 COST-STRESS SUMMARY")
    print("=" * 140)

    v2_summary = overall[
        overall["strategy"]
        == "V2"
    ][
        [
            "slippage_bps",
            "final_balance",
            "return_pct",
            "executed_trades",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct_trade_close",
            "fees_usd",
            "funding_pnl_usd",
            "slippage_cost_usd",
            "avg_hold_minutes",
            "max_hold_minutes",
            "lock5_activated_count",
            "trailing_activated_count",
            "time_exit_count",
            "liquidation_proxy_count",
        ]
    ]

    print(
        v2_summary.to_string(
            index=False
        )
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    overall_file = os.path.join(
        REPORT_DIR,
        f"v3_10x_cost_stress_overall_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    windows_file = os.path.join(
        REPORT_DIR,
        f"v3_10x_cost_stress_windows_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    trades_file = os.path.join(
        REPORT_DIR,
        f"v3_10x_cost_stress_trades_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    skips_file = os.path.join(
        REPORT_DIR,
        f"v3_10x_cost_stress_skips_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    diagnostics_file = os.path.join(
        REPORT_DIR,
        f"v3_10x_cost_stress_model_diagnostics_"
        f"{SYMBOL}_{TIMEFRAME}_{stamp}.csv"
    )

    overall.to_csv(
        overall_file,
        index=False
    )

    windows.to_csv(
        windows_file,
        index=False
    )

    trades_df.to_csv(
        trades_file,
        index=False
    )

    skips_df.to_csv(
        skips_file,
        index=False
    )

    model_diagnostics.to_csv(
        diagnostics_file,
        index=False
    )

    print()
    print("=" * 120)
    print("V3 TEST FINISHED")
    print("=" * 120)
    print(f"Overall:     {overall_file}")
    print(f"Windows:     {windows_file}")
    print(f"Trades:      {trades_file}")
    print(f"Skips:       {skips_file}")
    print(f"Diagnostics: {diagnostics_file}")
    print("=" * 120)

    print()
    print(
        "Important: liquidation is a proxy, not the exact "
        "Bybit maintenance-margin formula. Profit-lock/trailing "
        "levels are modeled net of fees/slippage/funding. After "
        "20% NET ROE, the trailing floor follows peak ROE minus "
        "10 percentage points; gaps can still realize below it."
    )


if __name__ == "__main__":
    main()
