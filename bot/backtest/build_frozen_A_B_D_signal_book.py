import glob
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")

LEVERAGE = 10.0
MAX_TOTAL_MARGIN_FRACTION = 0.20
TAKER_FEE_BPS_PER_SIDE = 5.5
SLIPPAGE_BPS_PER_SIDE = [0.0, 1.0, 2.0, 3.0, 5.0]

# FROZEN candidates. Do not optimize them in this script.
A_TOP_N = 6

B_FEATURE = "4h_trend_score"
B_ORIENTATION = "LOWER_BETTER"
B_TOP_N = 5

D_FEATURE = "5m_ret_1bar_pct"
D_ORIENTATION = "HIGHER_BETTER"
D_TOP_N = 1

PRIMARY_EVENT_MODE = "NEW_EPISODE"


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found for pattern:\n{pattern}")
    return files[-1]


def profit_factor(returns):
    r = pd.to_numeric(returns, errors="coerce").dropna()
    pos = r.loc[r > 0].sum()
    neg = -r.loc[r < 0].sum()
    if neg <= 0:
        return np.inf if pos > 0 else np.nan
    return float(pos / neg)


def load_A():
    path = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_top30_rank_rows_*.csv",
        )
    )

    df = pd.read_csv(path)

    df["available_at"] = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce",
    )

    required = {
        "universe",
        "event_mode",
        "event_id",
        "available_at",
        "split",
        "symbol",
        "coin_rank",
        "return_60m_pct",
        "return_60m_atr",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Trigger A rank rows missing columns:\n"
            + "\n".join(missing)
        )

    a = df.loc[
        (
            df["universe"]
            == "ALL_AVAILABLE_TOP30"
        )
        &
        (
            df["event_mode"]
            == PRIMARY_EVENT_MODE
        )
        &
        (
            pd.to_numeric(
                df["coin_rank"],
                errors="coerce",
            )
            <= A_TOP_N
        )
    ].copy()

    if a.empty:
        raise RuntimeError(
            "No Trigger A Top6 NEW_EPISODE rows found."
        )

    rows = []

    for (
        event_id,
        event_time,
        split,
    ), g in a.groupby(
        [
            "event_id",
            "available_at",
            "split",
        ],
        observed=True,
    ):
        rows.append(
            {
                "strategy_id":
                    "A_LONG_TOP6_5M_TREND",
                "trigger":
                    "A",
                "direction":
                    "LONG",
                "split":
                    split,
                "available_at":
                    event_time,
                "event_id":
                    f"A_{int(event_id)}",
                "selected_count":
                    int(
                        g["symbol"].nunique()
                    ),
                "selected_symbols":
                    ",".join(
                        g.sort_values(
                            "coin_rank"
                        )["symbol"]
                        .astype(str)
                        .tolist()
                    ),
                "gross_direction_return_pct":
                    float(
                        g[
                            "return_60m_pct"
                        ].mean()
                    ),
                "gross_direction_return_atr":
                    float(
                        g[
                            "return_60m_atr"
                        ].mean()
                    ),
                "source_file":
                    path,
            }
        )

    return pd.DataFrame(rows), path


def load_B_D():
    path = latest_file(
        os.path.join(
            REPORT_DIR,
            "trigger_B_D_selector_absolute_rank_events_*.csv",
        )
    )

    df = pd.read_csv(path)

    df["available_at"] = pd.to_datetime(
        df["available_at"],
        utc=True,
        errors="coerce",
    )

    required = {
        "trigger_name",
        "direction",
        "feature",
        "orientation",
        "split",
        "event_mode",
        "available_at",
        "top_n",
        "selected_symbols",
        "mean_direction_return_pct",
        "mean_direction_return_atr",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "B/D absolute-rank events missing columns:\n"
            + "\n".join(missing)
        )

    specs = [
        {
            "trigger_name":
                "TRIGGER_B_LONG",
            "strategy_id":
                "B_LONG_BOTTOM5_4H_TREND",
            "trigger":
                "B",
            "direction":
                "LONG",
            "feature":
                B_FEATURE,
            "orientation":
                B_ORIENTATION,
            "top_n":
                B_TOP_N,
        },
        {
            "trigger_name":
                "TRIGGER_D_SHORT",
            "strategy_id":
                "D_SHORT_TOP1_LAST5M_RETURN",
            "trigger":
                "D",
            "direction":
                "SHORT",
            "feature":
                D_FEATURE,
            "orientation":
                D_ORIENTATION,
            "top_n":
                D_TOP_N,
        },
    ]

    parts = []

    for spec in specs:
        x = df.loc[
            (
                df[
                    "trigger_name"
                ]
                == spec[
                    "trigger_name"
                ]
            )
            &
            (
                df[
                    "event_mode"
                ]
                == PRIMARY_EVENT_MODE
            )
            &
            (
                df[
                    "feature"
                ]
                == spec[
                    "feature"
                ]
            )
            &
            (
                df[
                    "orientation"
                ]
                == spec[
                    "orientation"
                ]
            )
            &
            (
                pd.to_numeric(
                    df["top_n"],
                    errors="coerce",
                )
                == spec[
                    "top_n"
                ]
            )
        ].copy()

        if x.empty:
            raise RuntimeError(
                f"No rows for {spec['strategy_id']}"
            )

        x["strategy_id"] = (
            spec["strategy_id"]
        )
        x["trigger"] = spec["trigger"]
        x["direction"] = (
            spec["direction"]
        )
        x["event_id"] = [
            f"{spec['trigger']}_{i+1}"
            for i in range(len(x))
        ]
        x["selected_count"] = (
            spec["top_n"]
        )
        x[
            "gross_direction_return_pct"
        ] = x[
            "mean_direction_return_pct"
        ]
        x[
            "gross_direction_return_atr"
        ] = x[
            "mean_direction_return_atr"
        ]
        x["source_file"] = path

        parts.append(
            x[
                [
                    "strategy_id",
                    "trigger",
                    "direction",
                    "split",
                    "available_at",
                    "event_id",
                    "selected_count",
                    "selected_symbols",
                    "gross_direction_return_pct",
                    "gross_direction_return_atr",
                    "source_file",
                ]
            ]
        )

    return pd.concat(
        parts,
        ignore_index=True,
    ), path


def add_cost_scenarios(signal_book):
    rows = []

    for _, row in signal_book.iterrows():
        gross_pct = float(
            row[
                "gross_direction_return_pct"
            ]
        )

        for slip_bps in (
            SLIPPAGE_BPS_PER_SIDE
        ):
            roundtrip_cost_pct = (
                2.0
                * (
                    TAKER_FEE_BPS_PER_SIDE
                    + slip_bps
                )
                * 0.01
            )

            net_pct = (
                gross_pct
                - roundtrip_cost_pct
            )

            margin_roe_pct = (
                net_pct
                * LEVERAGE
            )

            # If the total strategy uses 20% of current equity
            # as margin, 10x leverage means 2x equity notional.
            # This is per signal and ignores overlapping signals.
            equity_return_pct_if_full_20pct_margin = (
                net_pct
                * LEVERAGE
                * MAX_TOTAL_MARGIN_FRACTION
            )

            out = row.to_dict()

            out.update(
                {
                    "slippage_bps_per_side":
                        slip_bps,
                    "fee_bps_per_side":
                        TAKER_FEE_BPS_PER_SIDE,
                    "roundtrip_cost_pct":
                        roundtrip_cost_pct,
                    "net_direction_return_pct":
                        net_pct,
                    "net_margin_roe_pct_10x":
                        margin_roe_pct,
                    "equity_return_pct_if_full_20pct_margin":
                        equity_return_pct_if_full_20pct_margin,
                }
            )

            rows.append(out)

    return pd.DataFrame(rows)


def summarize_costs(cost_rows):
    rows = []

    for (
        strategy_id,
        trigger,
        direction,
        split,
        slip_bps,
    ), g in cost_rows.groupby(
        [
            "strategy_id",
            "trigger",
            "direction",
            "split",
            "slippage_bps_per_side",
        ],
        observed=True,
    ):
        net = g[
            "net_direction_return_pct"
        ]

        gross = g[
            "gross_direction_return_pct"
        ]

        rows.append(
            {
                "strategy_id":
                    strategy_id,
                "trigger":
                    trigger,
                "direction":
                    direction,
                "split":
                    split,
                "slippage_bps_per_side":
                    slip_bps,
                "events":
                    int(len(g)),
                "avg_selected_count":
                    float(
                        g[
                            "selected_count"
                        ].mean()
                    ),
                "mean_gross_direction_return_pct":
                    float(
                        gross.mean()
                    ),
                "median_gross_direction_return_pct":
                    float(
                        gross.median()
                    ),
                "gross_positive_rate":
                    float(
                        (gross > 0).mean()
                    ),
                "mean_net_direction_return_pct":
                    float(
                        net.mean()
                    ),
                "median_net_direction_return_pct":
                    float(
                        net.median()
                    ),
                "net_positive_rate":
                    float(
                        (net > 0).mean()
                    ),
                "net_profit_factor":
                    profit_factor(net),
                "mean_net_margin_roe_pct_10x":
                    float(
                        g[
                            "net_margin_roe_pct_10x"
                        ].mean()
                    ),
                "mean_equity_return_pct_if_full_20pct_margin":
                    float(
                        g[
                            "equity_return_pct_if_full_20pct_margin"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


def strategy_break_even_slippage(signal_book):
    rows = []

    for (
        strategy_id,
        trigger,
        direction,
        split,
    ), g in signal_book.groupby(
        [
            "strategy_id",
            "trigger",
            "direction",
            "split",
        ],
        observed=True,
    ):
        mean_gross_pct = float(
            g[
                "gross_direction_return_pct"
            ].mean()
        )

        gross_total_bps = (
            mean_gross_pct
            / 0.01
        )

        # gross = 2*fee + 2*slippage
        break_even_slippage = (
            gross_total_bps
            - 2.0
            * TAKER_FEE_BPS_PER_SIDE
        ) / 2.0

        rows.append(
            {
                "strategy_id":
                    strategy_id,
                "trigger":
                    trigger,
                "direction":
                    direction,
                "split":
                    split,
                "events":
                    len(g),
                "mean_gross_direction_return_pct":
                    mean_gross_pct,
                "break_even_slippage_bps_per_side_after_fees":
                    break_even_slippage,
            }
        )

    return pd.DataFrame(rows)


def overlap_report(signal_book):
    rows = []

    for event_time, g in signal_book.groupby(
        "available_at",
        observed=True,
    ):
        strategies = sorted(
            g[
                "strategy_id"
            ].unique()
        )

        directions = sorted(
            g[
                "direction"
            ].unique()
        )

        if len(strategies) <= 1:
            continue

        rows.append(
            {
                "available_at":
                    event_time,
                "strategies":
                    ",".join(
                        strategies
                    ),
                "triggers":
                    ",".join(
                        sorted(
                            g[
                                "trigger"
                            ].unique()
                        )
                    ),
                "directions":
                    ",".join(
                        directions
                    ),
                "strategy_count":
                    len(strategies),
                "direction_conflict":
                    len(directions) > 1,
            }
        )

    return pd.DataFrame(rows)


def main():
    print()
    print("=" * 120)
    print("FROZEN A / B / D SIGNAL BOOK + 60M COST STRESS")
    print("=" * 120)
    print("A: LONG, Trigger A, absolute Top6 by 5m trend_score")
    print("B: LONG, Trigger B NEW_EPISODE, absolute Top5 LOWEST 4h trend_score")
    print("D: SHORT, Trigger D NEW_EPISODE, absolute Top1 HIGHEST last 5m return")
    print()
    print("No selector fitting or threshold optimization occurs here.")
    print("=" * 120)

    a, a_source = load_A()
    bd, bd_source = load_B_D()

    signal_book = pd.concat(
        [
            a,
            bd,
        ],
        ignore_index=True,
    ).sort_values(
        [
            "available_at",
            "strategy_id",
        ]
    )

    cost_rows = add_cost_scenarios(
        signal_book
    )

    cost_summary = summarize_costs(
        cost_rows
    )

    break_even = (
        strategy_break_even_slippage(
            signal_book
        )
    )

    overlaps = overlap_report(
        signal_book
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    signal_file = os.path.join(
        REPORT_DIR,
        f"frozen_A_B_D_signal_book_{stamp}.csv",
    )

    costs_file = os.path.join(
        REPORT_DIR,
        f"frozen_A_B_D_cost_rows_{stamp}.csv",
    )

    summary_file = os.path.join(
        REPORT_DIR,
        f"frozen_A_B_D_cost_summary_{stamp}.csv",
    )

    break_even_file = os.path.join(
        REPORT_DIR,
        f"frozen_A_B_D_break_even_slippage_{stamp}.csv",
    )

    overlaps_file = os.path.join(
        REPORT_DIR,
        f"frozen_A_B_D_overlaps_{stamp}.csv",
    )

    signal_book.to_csv(
        signal_file,
        index=False,
    )

    cost_rows.to_csv(
        costs_file,
        index=False,
    )

    cost_summary.to_csv(
        summary_file,
        index=False,
    )

    break_even.to_csv(
        break_even_file,
        index=False,
    )

    overlaps.to_csv(
        overlaps_file,
        index=False,
    )

    print()
    print("=" * 160)
    print("COST SUMMARY")
    print("=" * 160)

    cols = [
        "strategy_id",
        "split",
        "slippage_bps_per_side",
        "events",
        "mean_gross_direction_return_pct",
        "mean_net_direction_return_pct",
        "net_positive_rate",
        "net_profit_factor",
        "mean_net_margin_roe_pct_10x",
    ]

    print(
        cost_summary[
            cols
        ]
        .sort_values(
            [
                "strategy_id",
                "split",
                "slippage_bps_per_side",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("BREAK-EVEN SLIPPAGE")
    print("=" * 120)

    print(
        break_even.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("OVERLAPS")
    print("=" * 120)

    if overlaps.empty:
        print(
            "No same-timestamp strategy overlaps."
        )
    else:
        print(
            overlaps.to_string(
                index=False
            )
        )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"A source:       {a_source}")
    print(f"B/D source:     {bd_source}")
    print(f"Signal book:    {signal_file}")
    print(f"Cost rows:      {costs_file}")
    print(f"Cost summary:   {summary_file}")
    print(f"Break-even:     {break_even_file}")
    print(f"Overlaps:       {overlaps_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
