from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_adaptive_voting_experiment import (
    RANDOM_SEED,
    SCENARIOS,
    STRATEGY_LABELS,
    calibrate_reliability,
    expert_probabilities,
    generate_robot_attributes,
    make_round,
    strategy_probabilities,
    true_objective,
)
from run_algorithm_experiment import (
    MAX_TRANSMISSION_ATTEMPTS,
    PACKET_LOSS_RATE,
    ROBOT_COUNTS,
    TRIALS,
    sample_active_robots,
)
from voting_mrta import apply_vote_retransmission


QUALITY_STRATEGIES = [
    "equal_fusion",
    "context_fusion",
    "adaptive_reliability",
    "oracle",
]
NEAR_THRESHOLDS = [0.01, 0.02, 0.05]
TOP_K_VALUES = [1, 3, 5]

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "adaptive_quality"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def objective_rank(objective: np.ndarray, active: np.ndarray, winner: int) -> int:
    active_indices = np.flatnonzero(active)
    ordered = active_indices[np.argsort(objective[active_indices], kind="stable")]
    return int(np.flatnonzero(ordered == winner)[0]) + 1


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for n in ROBOT_COUNTS:
            calibration_seed = RANDOM_SEED + scenario_index * 10000 + n
            reliability = calibrate_reliability(n, scenario, calibration_seed)
            rng = np.random.default_rng(
                RANDOM_SEED + 100000 + scenario_index * 10000 + n
            )

            for trial in range(TRIALS):
                active = sample_active_robots(n, rng)
                attributes = generate_robot_attributes(n, rng)
                objective = true_objective(attributes, scenario)
                active_indices = np.flatnonzero(active)
                optimal_robot = int(active_indices[np.argmin(objective[active])])
                optimal_score = float(objective[optimal_robot])

                expert_map = expert_probabilities(attributes, active)
                voter_uniforms = rng.random(n)
                tie_priority = rng.random(n)
                attempt_random = rng.random((MAX_TRANSMISSION_ATTEMPTS, n))

                for strategy in QUALITY_STRATEGIES:
                    probabilities, _ = strategy_probabilities(
                        strategy,
                        expert_map,
                        scenario,
                        reliability,
                        optimal_robot,
                    )
                    round_data = make_round(
                        probabilities,
                        active,
                        voter_uniforms,
                        tie_priority,
                    )
                    result = apply_vote_retransmission(
                        round_data,
                        attempt_random,
                        loss_rate=PACKET_LOSS_RATE,
                        max_attempts=MAX_TRANSMISSION_ATTEMPTS,
                    )

                    if result.winner is None:
                        winner_rank = np.nan
                        regret = np.nan
                        exact_optimal = False
                        near_flags = {threshold: False for threshold in NEAR_THRESHOLDS}
                        top_flags = {k: False for k in TOP_K_VALUES}
                        no_decision = True
                    else:
                        winner_rank = objective_rank(objective, active, result.winner)
                        regret = float(objective[result.winner] - optimal_score)
                        exact_optimal = winner_rank == 1
                        near_flags = {
                            threshold: regret <= threshold
                            for threshold in NEAR_THRESHOLDS
                        }
                        top_flags = {k: winner_rank <= k for k in TOP_K_VALUES}
                        no_decision = False

                    records.append(
                        {
                            "robots": n,
                            "trial": trial + 1,
                            "scenario": scenario.key,
                            "scenario_label": scenario.label,
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            "active_robots": int(active.sum()),
                            "winner_rank": winner_rank,
                            "regret": regret,
                            "exact_optimal": exact_optimal,
                            "near_0_01": near_flags[0.01],
                            "near_0_02": near_flags[0.02],
                            "near_0_05": near_flags[0.05],
                            "top_1": top_flags[1],
                            "top_3": top_flags[3],
                            "top_5": top_flags[5],
                            "no_decision": no_decision,
                        }
                    )

    raw = pd.DataFrame.from_records(records)

    summary = (
        raw.groupby(
            ["robots", "scenario", "scenario_label", "strategy", "strategy_label"],
            as_index=False,
        )
        .agg(
            exact_optimal_rate=("exact_optimal", "mean"),
            near_0_01_rate=("near_0_01", "mean"),
            near_0_02_rate=("near_0_02", "mean"),
            near_0_05_rate=("near_0_05", "mean"),
            top_1_rate=("top_1", "mean"),
            top_3_rate=("top_3", "mean"),
            top_5_rate=("top_5", "mean"),
            average_regret=("regret", "mean"),
            no_decision_rate=("no_decision", "mean"),
        )
        .reset_index(drop=True)
    )

    by_strategy = (
        summary.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            exact_optimal_rate=("exact_optimal_rate", "mean"),
            near_0_01_rate=("near_0_01_rate", "mean"),
            near_0_02_rate=("near_0_02_rate", "mean"),
            near_0_05_rate=("near_0_05_rate", "mean"),
            top_1_rate=("top_1_rate", "mean"),
            top_3_rate=("top_3_rate", "mean"),
            top_5_rate=("top_5_rate", "mean"),
            average_regret=("average_regret", "mean"),
        )
        .reset_index(drop=True)
    )

    return raw, summary, by_strategy


def adaptive_by_robot(summary: pd.DataFrame) -> pd.DataFrame:
    adaptive = summary[summary["strategy"] == "adaptive_reliability"]
    return (
        adaptive.groupby("robots", as_index=False)
        .agg(
            exact_optimal_rate=("exact_optimal_rate", "mean"),
            near_0_01_rate=("near_0_01_rate", "mean"),
            near_0_02_rate=("near_0_02_rate", "mean"),
            near_0_05_rate=("near_0_05_rate", "mean"),
            top_1_rate=("top_1_rate", "mean"),
            top_3_rate=("top_3_rate", "mean"),
            top_5_rate=("top_5_rate", "mean"),
            average_regret=("average_regret", "mean"),
        )
        .sort_values("robots")
        .reset_index(drop=True)
    )


def plot_near_optimal(summary: pd.DataFrame) -> None:
    data = adaptive_by_robot(summary)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    series = [
        ("Exact optimal", "exact_optimal_rate"),
        ("Near-optimal: regret <= 0.01", "near_0_01_rate"),
        ("Near-optimal: regret <= 0.02", "near_0_02_rate"),
        ("Near-optimal: regret <= 0.05", "near_0_05_rate"),
    ]
    for label, column in series:
        ax.plot(
            data["robots"],
            data[column],
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label,
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Selection Rate")
    ax.set_title("Adaptive Voting: Exact vs. Near-Optimal Selection")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_near_optimal_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_k(summary: pd.DataFrame) -> None:
    data = adaptive_by_robot(summary)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for label, column in [
        ("Top-1 (Exact optimal)", "top_1_rate"),
        ("Top-3", "top_3_rate"),
        ("Top-5", "top_5_rate"),
    ]:
        ax.plot(
            data["robots"],
            data[column],
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=label,
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Selection Rate")
    ax.set_title("Adaptive Voting: Top-k Candidate Selection")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_top_k_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_strategy = run_analysis()

    raw_path = DATA_DIR / "adaptive_quality_raw_results.csv"
    summary_path = DATA_DIR / "adaptive_quality_summary_results.csv"
    by_strategy_path = DATA_DIR / "adaptive_quality_by_strategy.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_strategy.to_csv(by_strategy_path, index=False)

    plot_near_optimal(summary)
    plot_top_k(summary)

    print("\nAdaptive quality metrics averaged across contexts and team sizes:")
    print(
        by_strategy[
            [
                "strategy_label",
                "exact_optimal_rate",
                "near_0_01_rate",
                "near_0_02_rate",
                "near_0_05_rate",
                "top_3_rate",
                "top_5_rate",
                "average_regret",
            ]
        ].to_string(index=False)
    )

    print("\nGenerated adaptive quality files:")
    for path in [raw_path, summary_path, by_strategy_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
