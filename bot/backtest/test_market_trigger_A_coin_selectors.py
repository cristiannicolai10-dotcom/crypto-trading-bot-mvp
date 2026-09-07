import glob
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = "/home/botadmin/crypto-bot"
REPORT_DIR = os.path.join(BASE_DIR, "reports")

TOP_FRACTION = 0.20
MIN_ELIGIBLE_COINS = 10

HORIZONS = [5, 15, 30, 45, 60]

SELECTORS = {
    "ALL_COINS": [],
    "A1_TREND5M": [
        ("5m_trend_score", "HIGH"),
    ],
    "A2_TREND5M_LOW_1H_ALLIGATOR": [
        ("5m_trend_score", "HIGH"),
        ("1h_alligator_spread_pct", "LOW"),
    ],
    "A3_PLUS_4H_RETURN": [
        ("5m_trend_score", "HIGH"),
        ("1h_alligator_spread_pct", "LOW"),
        ("4h_ret_1bar_pct", "HIGH"),
    ],
}


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No files found for pattern:\n{pattern}"
        )
    return files[-1]


def load_inputs():
    outcomes_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_coin_outcomes_*.csv",
        )
    )

    events_file = latest_file(
        os.path.join(
            REPORT_DIR,
            "market_trigger_A_independent_events_*.csv",
        )
    )

    outcomes = pd.read_csv(outcomes_file)
    events = pd.read_csv(events_file)

    outcomes["available_at"] = pd.to_datetime(
        outcomes["available_at"],
        utc=True,
        errors="coerce",
    )

    events["available_at"] = pd.to_datetime(
        events["available_at"],
        utc=True,
        errors="coerce",
    )

    events["new_episode"] = (
        events["new_episode"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    required_outcome = {
        "event_id",
        "available_at",
        "split",
        "symbol",
        "full_history_universe",
        "return_60m_atr",
        "mfe_60m_atr",
        "mae_60m_atr",
        "trend_up_60m",
        "positive_60m",
        "5m_trend_score",
        "1h_alligator_spread_pct",
        "4h_ret_1bar_pct",
    }

    for h in HORIZONS:
        required_outcome.add(
            f"return_{h}m_pct"
        )

    missing = sorted(
        required_outcome
        - set(outcomes.columns)
    )

    if missing:
        raise RuntimeError(
            "Coin outcome CSV missing columns:\n"
            + "\n".join(missing)
        )

    required_events = {
        "event_id",
        "available_at",
        "split",
        "new_episode",
    }

    missing_events = sorted(
        required_events
        - set(events.columns)
    )

    if missing_events:
        raise RuntimeError(
            "Event CSV missing columns:\n"
            + "\n".join(missing_events)
        )

    outcomes["full_history_universe"] = (
        outcomes["full_history_universe"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return (
        outcomes,
        events,
        outcomes_file,
        events_file,
    )


def rank_component(group, column, direction):
    values = pd.to_numeric(
        group[column],
        errors="coerce",
    )

    valid = values.notna()

    ranks = pd.Series(
        np.nan,
        index=group.index,
        dtype=float,
    )

    if valid.sum() == 0:
        return ranks

    if direction == "HIGH":
        ranks.loc[valid] = (
            values.loc[valid]
            .rank(
                pct=True,
                method="average",
                ascending=True,
            )
        )

    elif direction == "LOW":
        ranks.loc[valid] = (
            values.loc[valid]
            .rank(
                pct=True,
                method="average",
                ascending=False,
            )
        )

    else:
        raise ValueError(
            f"Unknown rank direction: {direction}"
        )

    return ranks


def score_selector(group, selector_name):
    components = SELECTORS[selector_name]

    if selector_name == "ALL_COINS":
        out = group.copy()
        out["_selector_score"] = 1.0
        return out

    scored = group.copy()

    component_cols = []

    for idx, (feature, direction) in enumerate(
        components,
        start=1,
    ):
        col = f"_component_{idx}"

        scored[col] = rank_component(
            scored,
            feature,
            direction,
        )

        component_cols.append(col)

    scored["_selector_score"] = (
        scored[component_cols]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    scored = scored.loc[
        scored["_selector_score"].notna()
    ].copy()

    return scored


def select_top(group, selector_name):
    scored = score_selector(
        group,
        selector_name,
    )

    eligible = len(scored)

    if eligible < MIN_ELIGIBLE_COINS:
        return None, eligible

    if selector_name == "ALL_COINS":
        selected = scored.copy()

    else:
        top_n = max(
            1,
            int(
                math.ceil(
                    eligible
                    * TOP_FRACTION
                )
            ),
        )

        selected = (
            scored
            .sort_values(
                [
                    "_selector_score",
                    "symbol",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(top_n)
            .copy()
        )

    return selected, eligible


def summarize_selected_event(
    selected,
    eligible_coins,
    universe_name,
    event_mode,
    selector_name,
    event_id,
    split,
    event_time,
):
    row = {
        "universe": universe_name,
        "event_mode": event_mode,
        "selector": selector_name,
        "event_id": int(event_id),
        "split": split,
        "available_at": event_time,
        "eligible_coins": int(
            eligible_coins
        ),
        "selected_coins": int(
            selected["symbol"].nunique()
        ),
        "selected_symbols": ",".join(
            selected[
                "symbol"
            ]
            .astype(str)
            .sort_values()
            .tolist()
        ),
        "trend_up_rate_60m":
            float(
                selected[
                    "trend_up_60m"
                ].mean()
            ),
        "positive_rate_60m":
            float(
                selected[
                    "positive_60m"
                ].mean()
            ),
        "mean_return_60m_atr":
            float(
                selected[
                    "return_60m_atr"
                ].mean()
            ),
        "median_return_60m_atr":
            float(
                selected[
                    "return_60m_atr"
                ].median()
            ),
        "mean_mfe_60m_atr":
            float(
                selected[
                    "mfe_60m_atr"
                ].mean()
            ),
        "mean_mae_60m_atr":
            float(
                selected[
                    "mae_60m_atr"
                ].mean()
            ),
    }

    for h in HORIZONS:
        col = f"return_{h}m_pct"

        row[
            f"mean_return_{h}m_pct"
        ] = float(
            selected[col].mean()
        )

        row[
            f"median_return_{h}m_pct"
        ] = float(
            selected[col].median()
        )

        row[
            f"positive_rate_{h}m"
        ] = float(
            (
                selected[col]
                > 0
            ).mean()
        )

    return row


def build_event_selector_results(
    outcomes,
    events,
):
    rows = []

    event_modes = {
        "COOLDOWN_60M":
            set(
                events[
                    "event_id"
                ].astype(int)
            ),

        "NEW_EPISODE":
            set(
                events.loc[
                    events[
                        "new_episode"
                    ],
                    "event_id",
                ].astype(int)
            ),
    }

    universes = {
        "ALL_AVAILABLE":
            outcomes,

        "FULL_HISTORY_ONLY":
            outcomes.loc[
                outcomes[
                    "full_history_universe"
                ]
            ],
    }

    for universe_name, universe_frame in (
        universes.items()
    ):
        for event_mode, event_ids in (
            event_modes.items()
        ):
            mode_frame = universe_frame.loc[
                universe_frame[
                    "event_id"
                ].astype(int)
                .isin(
                    event_ids
                )
            ].copy()

            for (
                event_id,
                split,
                event_time,
            ), group in mode_frame.groupby(
                [
                    "event_id",
                    "split",
                    "available_at",
                ],
                observed=True,
            ):
                for selector_name in (
                    SELECTORS.keys()
                ):
                    selected, eligible = (
                        select_top(
                            group,
                            selector_name,
                        )
                    )

                    if selected is None:
                        continue

                    rows.append(
                        summarize_selected_event(
                            selected=selected,
                            eligible_coins=eligible,
                            universe_name=universe_name,
                            event_mode=event_mode,
                            selector_name=selector_name,
                            event_id=event_id,
                            split=split,
                            event_time=event_time,
                        )
                    )

    return pd.DataFrame(rows)


def add_uplift_vs_all(event_results):
    if event_results.empty:
        return event_results

    baseline_cols = [
        "universe",
        "event_mode",
        "event_id",
        "split",
        "available_at",
        "mean_return_60m_atr",
        "trend_up_rate_60m",
        "positive_rate_60m",
    ]

    for h in HORIZONS:
        baseline_cols.append(
            f"mean_return_{h}m_pct"
        )

    baseline = event_results.loc[
        event_results[
            "selector"
        ]
        == "ALL_COINS",
        baseline_cols,
    ].copy()

    rename = {
        "mean_return_60m_atr":
            "baseline_mean_return_60m_atr",
        "trend_up_rate_60m":
            "baseline_trend_up_rate_60m",
        "positive_rate_60m":
            "baseline_positive_rate_60m",
    }

    for h in HORIZONS:
        rename[
            f"mean_return_{h}m_pct"
        ] = (
            f"baseline_mean_return_{h}m_pct"
        )

    baseline = baseline.rename(
        columns=rename
    )

    merged = event_results.merge(
        baseline,
        on=[
            "universe",
            "event_mode",
            "event_id",
            "split",
            "available_at",
        ],
        how="left",
        validate="many_to_one",
    )

    merged[
        "uplift_return_60m_atr"
    ] = (
        merged[
            "mean_return_60m_atr"
        ]
        - merged[
            "baseline_mean_return_60m_atr"
        ]
    )

    merged[
        "uplift_trend_up_rate_60m"
    ] = (
        merged[
            "trend_up_rate_60m"
        ]
        - merged[
            "baseline_trend_up_rate_60m"
        ]
    )

    merged[
        "uplift_positive_rate_60m"
    ] = (
        merged[
            "positive_rate_60m"
        ]
        - merged[
            "baseline_positive_rate_60m"
        ]
    )

    for h in HORIZONS:
        merged[
            f"uplift_return_{h}m_pct"
        ] = (
            merged[
                f"mean_return_{h}m_pct"
            ]
            - merged[
                f"baseline_mean_return_{h}m_pct"
            ]
        )

    return merged


def summary_by_split(event_results):
    rows = []

    for (
        universe,
        event_mode,
        split,
        selector,
    ), group in event_results.groupby(
        [
            "universe",
            "event_mode",
            "split",
            "selector",
        ],
        observed=True,
    ):
        row = {
            "universe": universe,
            "event_mode": event_mode,
            "split": split,
            "selector": selector,
            "events": int(
                group[
                    "event_id"
                ].nunique()
            ),
            "avg_eligible_coins":
                float(
                    group[
                        "eligible_coins"
                    ].mean()
                ),
            "avg_selected_coins":
                float(
                    group[
                        "selected_coins"
                    ].mean()
                ),
            "mean_trend_up_rate_60m":
                float(
                    group[
                        "trend_up_rate_60m"
                    ].mean()
                ),
            "mean_positive_rate_60m":
                float(
                    group[
                        "positive_rate_60m"
                    ].mean()
                ),
            "mean_return_60m_atr":
                float(
                    group[
                        "mean_return_60m_atr"
                    ].mean()
                ),
            "median_event_return_60m_atr":
                float(
                    group[
                        "mean_return_60m_atr"
                    ].median()
                ),
            "mean_mfe_60m_atr":
                float(
                    group[
                        "mean_mfe_60m_atr"
                    ].mean()
                ),
            "mean_mae_60m_atr":
                float(
                    group[
                        "mean_mae_60m_atr"
                    ].mean()
                ),
            "mean_uplift_60m_atr":
                float(
                    group[
                        "uplift_return_60m_atr"
                    ].mean()
                ),
            "median_uplift_60m_atr":
                float(
                    group[
                        "uplift_return_60m_atr"
                    ].median()
                ),
            "selector_beats_all_rate":
                float(
                    (
                        group[
                            "uplift_return_60m_atr"
                        ]
                        > 0
                    ).mean()
                ),
        }

        for h in HORIZONS:
            row[
                f"mean_return_{h}m_pct"
            ] = float(
                group[
                    f"mean_return_{h}m_pct"
                ].mean()
            )

            row[
                f"mean_positive_rate_{h}m"
            ] = float(
                group[
                    f"positive_rate_{h}m"
                ].mean()
            )

            row[
                f"mean_uplift_{h}m_pct"
            ] = float(
                group[
                    f"uplift_return_{h}m_pct"
                ].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def compare_discovery_validation(summary):
    if summary.empty:
        return pd.DataFrame()

    metrics = [
        "events",
        "mean_trend_up_rate_60m",
        "mean_positive_rate_60m",
        "mean_return_60m_atr",
        "mean_uplift_60m_atr",
        "selector_beats_all_rate",
        "mean_mfe_60m_atr",
        "mean_mae_60m_atr",
        "mean_return_15m_pct",
        "mean_return_30m_pct",
        "mean_return_45m_pct",
        "mean_return_60m_pct",
    ]

    base = summary[
        [
            "universe",
            "event_mode",
            "split",
            "selector",
        ]
        + metrics
    ].copy()

    pivot = base.pivot(
        index=[
            "universe",
            "event_mode",
            "selector",
        ],
        columns="split",
        values=metrics,
    )

    pivot.columns = [
        f"{metric}_{split}"
        for metric, split
        in pivot.columns
    ]

    out = pivot.reset_index()

    if (
        "mean_uplift_60m_atr_DISCOVERY_60D"
        in out.columns
        and
        "mean_uplift_60m_atr_VALIDATION_30D"
        in out.columns
    ):
        out[
            "positive_uplift_both_periods"
        ] = (
            (
                out[
                    "mean_uplift_60m_atr_DISCOVERY_60D"
                ]
                > 0
            )
            &
            (
                out[
                    "mean_uplift_60m_atr_VALIDATION_30D"
                ]
                > 0
            )
        )

        out[
            "min_uplift_across_periods"
        ] = out[
            [
                "mean_uplift_60m_atr_DISCOVERY_60D",
                "mean_uplift_60m_atr_VALIDATION_30D",
            ]
        ].min(
            axis=1
        )

        out = out.sort_values(
            [
                "positive_uplift_both_periods",
                "min_uplift_across_periods",
            ],
            ascending=[
                False,
                False,
            ],
        )

    return out


def main():
    (
        outcomes,
        events,
        outcomes_file,
        events_file,
    ) = load_inputs()

    print()
    print("=" * 120)
    print("MARKET TRIGGER A - COIN SELECTOR A1/A2/A3 TEST")
    print("=" * 120)
    print(f"Outcomes source: {outcomes_file}")
    print(f"Events source:   {events_file}")
    print()
    print("Frozen selectors:")
    print("  A1 = rank(5m trend score)")
    print(
        "  A2 = A1 + inverse rank(1h Alligator spread)"
    )
    print(
        "  A3 = A2 + rank(4h return 1 bar)"
    )
    print(
        f"  Selection = top {TOP_FRACTION * 100:.0f}% "
        "per event, equal weights"
    )
    print()
    print("No thresholds are fitted.")
    print("No model is trained.")
    print("Discovery/validation split remains unchanged.")
    print("=" * 120)

    event_results = (
        build_event_selector_results(
            outcomes,
            events,
        )
    )

    event_results = (
        add_uplift_vs_all(
            event_results
        )
    )

    summary = summary_by_split(
        event_results
    )

    comparison = (
        compare_discovery_validation(
            summary
        )
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    event_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_selector_event_results_{stamp}.csv",
    )

    summary_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_selector_summary_{stamp}.csv",
    )

    comparison_file = os.path.join(
        REPORT_DIR,
        f"market_trigger_A_selector_comparison_{stamp}.csv",
    )

    event_results.to_csv(
        event_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    print()
    print("=" * 160)
    print("FULL-HISTORY COOLDOWN 60M")
    print("=" * 160)

    show = summary.loc[
        (
            summary["universe"]
            == "FULL_HISTORY_ONLY"
        )
        &
        (
            summary["event_mode"]
            == "COOLDOWN_60M"
        )
    ].copy()

    cols = [
        "split",
        "selector",
        "events",
        "avg_selected_coins",
        "mean_return_15m_pct",
        "mean_return_30m_pct",
        "mean_return_45m_pct",
        "mean_return_60m_pct",
        "mean_return_60m_atr",
        "mean_uplift_60m_atr",
        "selector_beats_all_rate",
        "mean_positive_rate_60m",
        "mean_trend_up_rate_60m",
        "mean_mfe_60m_atr",
        "mean_mae_60m_atr",
    ]

    print(
        show[cols]
        .sort_values(
            [
                "split",
                "mean_uplift_60m_atr",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    print()
    print("=" * 160)
    print("FULL-HISTORY STRICT NEW EPISODES")
    print("=" * 160)

    show_ep = summary.loc[
        (
            summary["universe"]
            == "FULL_HISTORY_ONLY"
        )
        &
        (
            summary["event_mode"]
            == "NEW_EPISODE"
        )
    ].copy()

    print(
        show_ep[cols]
        .sort_values(
            [
                "split",
                "mean_uplift_60m_atr",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    print()
    print("=" * 160)
    print("DISCOVERY -> VALIDATION COMPARISON")
    print("=" * 160)

    comp_show = comparison.loc[
        comparison[
            "universe"
        ]
        == "FULL_HISTORY_ONLY"
    ].copy()

    display_cols = [
        "event_mode",
        "selector",
        "positive_uplift_both_periods",
        "mean_uplift_60m_atr_DISCOVERY_60D",
        "mean_uplift_60m_atr_VALIDATION_30D",
        "selector_beats_all_rate_DISCOVERY_60D",
        "selector_beats_all_rate_VALIDATION_30D",
        "mean_return_60m_atr_DISCOVERY_60D",
        "mean_return_60m_atr_VALIDATION_30D",
    ]

    display_cols = [
        c for c in display_cols
        if c in comp_show.columns
    ]

    print(
        comp_show[
            display_cols
        ].to_string(index=False)
    )

    print()
    print("=" * 120)
    print("FILES")
    print("=" * 120)
    print(f"Event results: {event_file}")
    print(f"Summary:       {summary_file}")
    print(f"Comparison:    {comparison_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()
