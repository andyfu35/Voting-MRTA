from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import (
    MAX_TRANSMISSION_ATTEMPTS,
    PACKET_LOSS_RATE,
    ROBOT_COUNTS,
    ROBOT_FAILURE_RATE,
    TRIALS,
    sample_active_robots,
)
from voting_mrta import FullRound, apply_vote_retransmission, generate_costs, get_winner


RANDOM_SEED = 20260811
CALIBRATION_TRIALS = 40
EXPERT_BETA = 20.0
COST_START = 10.0
COST_STEP = 5.0

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "adaptive"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    weights: tuple[float, float, float, float]


SCENARIOS = [
    Scenario("cost_dominant", "Cost-dominant", (0.70, 0.10, 0.10, 0.10)),
    Scenario("energy_critical", "Energy-critical", (0.20, 0.60, 0.10, 0.10)),
    Scenario("communication_critical", "Communication-critical", (0.20, 0.10, 0.60, 0.10)),
    Scenario("load_critical", "Load-critical", (0.20, 0.10, 0.10, 0.60)),
    Scenario("balanced", "Balanced", (0.25, 0.25, 0.25, 0.25)),
]

EXPERTS = [
    "cost",
    "energy",
    "communication",
    "load",
    "balanced",
]

EXPERT_LABELS = {
    "cost": "Cost Expert",
    "energy": "Energy-Aware Expert",
    "communication": "Communication-Aware Expert",
    "load": "Load-Aware Expert",
    "balanced": "Balanced Expert",
}

STATIC_STRATEGIES = [f"static_{expert}" for expert in EXPERTS]
STRATEGIES = STATIC_STRATEGIES + [
    "equal_fusion",
    "context_fusion",
    "adaptive_reliability",
    "oracle",
]

STRATEGY_LABELS = {
    **{f"static_{expert}": EXPERT_LABELS[expert] for expert in EXPERTS},
    "equal_fusion": "Equal Expert Fusion",
    "context_fusion": "Context-Aware Fusion",
    "adaptive_reliability": "Context + Reliability Fusion",
    "oracle": "Oracle Objective",
}


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def normalized_fixed_costs(n: int) -> np.ndarray:
    costs = generate_costs(n, cost_start=COST_START, cost_step=COST_STEP)
    if n == 1:
        return np.zeros(1, dtype=float)
    return (costs - costs.min()) / (costs.max() - costs.min())


def generate_robot_attributes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return [task cost, energy risk, communication risk, load risk]."""
    task_cost = normalized_fixed_costs(n)
    energy_risk = rng.random(n)
    communication_risk = rng.random(n)
    load_risk = rng.random(n)
    return np.column_stack(
        [task_cost, energy_risk, communication_risk, load_risk]
    )


def true_objective(attributes: np.ndarray, scenario: Scenario) -> np.ndarray:
    return attributes @ np.asarray(scenario.weights, dtype=float)


def expert_score(attributes: np.ndarray, expert: str) -> np.ndarray:
    cost = attributes[:, 0]
    energy = attributes[:, 1]
    communication = attributes[:, 2]
    load = attributes[:, 3]

    if expert == "cost":
        return cost
    if expert == "energy":
        return 0.40 * cost + 0.60 * energy
    if expert == "communication":
        return 0.40 * cost + 0.60 * communication
    if expert == "load":
        return 0.40 * cost + 0.60 * load
    if expert == "balanced":
        return 0.25 * (cost + energy + communication + load)
    raise ValueError(f"Unknown expert: {expert}")


def score_to_probability(scores: np.ndarray, active: np.ndarray) -> np.ndarray:
    probabilities = np.zeros(len(scores), dtype=float)
    active_scores = scores[active]
    shifted = active_scores - active_scores.min()
    weights = np.exp(-EXPERT_BETA * shifted)
    probabilities[active] = weights / weights.sum()
    return probabilities


def expert_probabilities(
    attributes: np.ndarray,
    active: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        expert: score_to_probability(expert_score(attributes, expert), active)
        for expert in EXPERTS
    }


def context_relevance(scenario: Scenario) -> np.ndarray:
    objective_weights = np.asarray(scenario.weights, dtype=float)
    max_weight = float(objective_weights.max())
    balanced_relevance = max(0.0, 1.0 - (max_weight - 0.25) / 0.45)
    raw = np.concatenate([objective_weights, [balanced_relevance]])
    return raw / raw.sum()


def sample_ballots(
    probabilities: np.ndarray,
    active: np.ndarray,
    voter_uniforms: np.ndarray,
) -> np.ndarray:
    ballots = np.full(len(probabilities), -1, dtype=int)
    cdf = np.cumsum(probabilities)
    sampled = np.searchsorted(cdf, voter_uniforms[active], side="right")
    sampled = np.minimum(sampled, len(probabilities) - 1)
    ballots[np.flatnonzero(active)] = sampled
    return ballots


def make_round(
    probabilities: np.ndarray,
    active: np.ndarray,
    voter_uniforms: np.ndarray,
    tie_priority: np.ndarray,
) -> FullRound:
    ballots = sample_ballots(probabilities, active, voter_uniforms)
    counts = np.bincount(ballots[active], minlength=len(probabilities))
    winner, tie = get_winner(counts, tie_priority)
    if winner is None:
        raise RuntimeError("A round with active voters must have a winner")

    return FullRound(
        costs=np.arange(len(probabilities), dtype=float),
        probabilities=probabilities,
        ballots=ballots,
        full_counts=counts,
        full_winner=winner,
        full_tie=tie,
        tie_priority=tie_priority,
        delivery_random=np.zeros(len(probabilities), dtype=float),
        active_robots=active,
    )


def fuse_probabilities(
    expert_probability_map: dict[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.vstack([expert_probability_map[expert] for expert in EXPERTS])
    fused = np.sum(matrix * weights[:, None], axis=0)
    total = fused.sum()
    if total <= 0:
        raise RuntimeError("Fused probability vector is empty")
    return fused / total


def calibrate_reliability(
    n: int,
    scenario: Scenario,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    correct = np.zeros(len(EXPERTS), dtype=float)

    for _ in range(CALIBRATION_TRIALS):
        active = sample_active_robots(n, rng)
        attributes = generate_robot_attributes(n, rng)
        objective = true_objective(attributes, scenario)
        active_indices = np.flatnonzero(active)
        optimal_robot = int(active_indices[np.argmin(objective[active])])
        probabilities = expert_probabilities(attributes, active)
        voter_uniforms = rng.random(n)
        tie_priority = rng.random(n)

        for index, expert in enumerate(EXPERTS):
            round_data = make_round(
                probabilities[expert],
                active,
                voter_uniforms,
                tie_priority,
            )
            correct[index] += round_data.full_winner == optimal_robot

    # Laplace smoothing prevents one short calibration run from eliminating an expert.
    return (correct + 1.0) / (CALIBRATION_TRIALS + 2.0)


def strategy_probabilities(
    strategy: str,
    expert_probability_map: dict[str, np.ndarray],
    scenario: Scenario,
    reliability: np.ndarray,
    optimal_robot: int,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy.startswith("static_"):
        expert = strategy.removeprefix("static_")
        weights = np.zeros(len(EXPERTS), dtype=float)
        weights[EXPERTS.index(expert)] = 1.0
        return expert_probability_map[expert], weights

    if strategy == "equal_fusion":
        weights = np.full(len(EXPERTS), 1.0 / len(EXPERTS))
        return fuse_probabilities(expert_probability_map, weights), weights

    base_context = context_relevance(scenario)
    if strategy == "context_fusion":
        return fuse_probabilities(expert_probability_map, base_context), base_context

    if strategy == "adaptive_reliability":
        # Context decides which experts are relevant; calibration suppresses experts
        # that historically performed poorly in that same context.
        raw = base_context * (0.20 + 0.80 * reliability)
        weights = raw / raw.sum()
        return fuse_probabilities(expert_probability_map, weights), weights

    if strategy == "oracle":
        probabilities = np.zeros_like(next(iter(expert_probability_map.values())))
        probabilities[optimal_robot] = 1.0
        return probabilities, np.zeros(len(EXPERTS), dtype=float)

    raise ValueError(f"Unknown strategy: {strategy}")


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    weight_records: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for n in ROBOT_COUNTS:
            calibration_seed = RANDOM_SEED + scenario_index * 10000 + n
            reliability = calibrate_reliability(n, scenario, calibration_seed)
            context = context_relevance(scenario)
            adaptive_raw = context * (0.20 + 0.80 * reliability)
            adaptive_weights = adaptive_raw / adaptive_raw.sum()

            for expert_index, expert in enumerate(EXPERTS):
                weight_records.append(
                    {
                        "robots": n,
                        "scenario": scenario.key,
                        "scenario_label": scenario.label,
                        "expert": expert,
                        "expert_label": EXPERT_LABELS[expert],
                        "context_relevance": context[expert_index],
                        "calibrated_reliability": reliability[expert_index],
                        "adaptive_weight": adaptive_weights[expert_index],
                    }
                )

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

                for strategy in STRATEGIES:
                    probabilities, strategy_weights = strategy_probabilities(
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

                    full_winner = round_data.full_winner
                    full_is_optimal = full_winner == optimal_robot
                    full_regret = float(objective[full_winner] - optimal_score)

                    if result.winner is None:
                        final_is_optimal = False
                        winner_preserved = False
                        regret = np.nan
                        no_decision = True
                    else:
                        final_is_optimal = result.winner == optimal_robot
                        winner_preserved = result.winner == full_winner
                        regret = float(objective[result.winner] - optimal_score)
                        no_decision = False

                    records.append(
                        {
                            "robots": n,
                            "trial": trial + 1,
                            "scenario": scenario.key,
                            "scenario_label": scenario.label,
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            "packet_loss_rate": PACKET_LOSS_RATE,
                            "robot_failure_rate": ROBOT_FAILURE_RATE,
                            "max_attempts": MAX_TRANSMISSION_ATTEMPTS,
                            "active_robots": int(active.sum()),
                            "full_is_optimal": full_is_optimal,
                            "final_is_optimal": final_is_optimal,
                            "full_regret": full_regret,
                            "regret": regret,
                            "winner_preserved": winner_preserved,
                            "tie": result.tie,
                            "no_decision": no_decision,
                            "weight_cost": strategy_weights[0],
                            "weight_energy": strategy_weights[1],
                            "weight_communication": strategy_weights[2],
                            "weight_load": strategy_weights[3],
                            "weight_balanced": strategy_weights[4],
                        }
                    )

    raw = pd.DataFrame.from_records(records)
    weights = pd.DataFrame.from_records(weight_records)

    summary = (
        raw.groupby(
            ["robots", "scenario", "scenario_label", "strategy", "strategy_label"],
            as_index=False,
        )
        .agg(
            full_vote_optimal_win_rate=("full_is_optimal", "mean"),
            optimal_win_rate=("final_is_optimal", "mean"),
            average_regret=("regret", "mean"),
            full_vote_average_regret=("full_regret", "mean"),
            winner_preservation_rate=("winner_preserved", "mean"),
            tie_rate=("tie", "mean"),
            no_decision_rate=("no_decision", "mean"),
        )
        .reset_index(drop=True)
    )

    by_strategy = (
        summary.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            optimal_win_rate=("optimal_win_rate", "mean"),
            average_regret=("average_regret", "mean"),
            winner_preservation_rate=("winner_preservation_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
        )
        .reset_index(drop=True)
    )

    return raw, summary, by_strategy, weights


def overall_by_robot(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["robots", "strategy", "strategy_label"], as_index=False)
        .agg(
            optimal_win_rate=("optimal_win_rate", "mean"),
            average_regret=("average_regret", "mean"),
            winner_preservation_rate=("winner_preservation_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
        )
        .reset_index(drop=True)
    )


def plot_overall_optimal(summary: pd.DataFrame) -> None:
    data = overall_by_robot(summary)
    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy in STRATEGIES:
        subset = data[data["strategy"] == strategy]
        ax.plot(
            subset["robots"],
            subset["optimal_win_rate"],
            marker="o",
            linewidth=2.2 if strategy in {"context_fusion", "adaptive_reliability"} else 1.4,
            linestyle="--" if strategy in {"equal_fusion", "context_fusion", "adaptive_reliability"} else "-",
            markersize=3.5,
            label=STRATEGY_LABELS[strategy],
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Adaptive Voting Performance Across Mixed Operating Contexts")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_overall_optimal_win_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regret(summary: pd.DataFrame) -> None:
    data = overall_by_robot(summary)
    focus = ["equal_fusion", "context_fusion", "adaptive_reliability", "oracle"]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for strategy in focus:
        subset = data[data["strategy"] == strategy]
        ax.plot(
            subset["robots"],
            subset["average_regret"],
            marker="o",
            linewidth=2.0,
            label=STRATEGY_LABELS[strategy],
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Objective Regret")
    ax.set_title("Adaptive Fusion Objective Regret")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_average_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_scenario(summary: pd.DataFrame) -> None:
    scenario_data = (
        summary.groupby(["scenario", "scenario_label", "strategy", "strategy_label"], as_index=False)
        .agg(optimal_win_rate=("optimal_win_rate", "mean"))
    )
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        subset = scenario_data[scenario_data["scenario"] == scenario.key]
        static = subset[subset["strategy"].isin(STATIC_STRATEGIES)]
        rows.append(
            {
                "scenario": scenario.label,
                "Best Fixed Expert": float(static["optimal_win_rate"].max()),
                "Equal Fusion": float(subset.loc[subset["strategy"] == "equal_fusion", "optimal_win_rate"].iloc[0]),
                "Context Fusion": float(subset.loc[subset["strategy"] == "context_fusion", "optimal_win_rate"].iloc[0]),
                "Context + Reliability": float(subset.loc[subset["strategy"] == "adaptive_reliability", "optimal_win_rate"].iloc[0]),
                "Oracle": float(subset.loc[subset["strategy"] == "oracle", "optimal_win_rate"].iloc[0]),
            }
        )

    plot_data = pd.DataFrame(rows)
    columns = ["Best Fixed Expert", "Equal Fusion", "Context Fusion", "Context + Reliability", "Oracle"]
    x = np.arange(len(plot_data))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for index, column in enumerate(columns):
        ax.bar(x + (index - 2) * width, plot_data[column], width, label=column)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["scenario"], rotation=20, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Can Adaptive Voting Match the Best Expert in Different Contexts?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_by_scenario.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_summary(by_strategy: pd.DataFrame) -> None:
    ordered = by_strategy.set_index("strategy").loc[STRATEGIES].reset_index()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["optimal_win_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["strategy_label"], rotation=30, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Average Strategy Quality Across All Contexts and Team Sizes")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_strategy_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights: pd.DataFrame) -> None:
    averaged = (
        weights.groupby(["scenario", "scenario_label", "expert", "expert_label"], as_index=False)
        .agg(adaptive_weight=("adaptive_weight", "mean"))
    )
    matrix = np.zeros((len(SCENARIOS), len(EXPERTS)), dtype=float)
    for i, scenario in enumerate(SCENARIOS):
        for j, expert in enumerate(EXPERTS):
            matrix[i, j] = float(
                averaged.loc[
                    (averaged["scenario"] == scenario.key)
                    & (averaged["expert"] == expert),
                    "adaptive_weight",
                ].iloc[0]
            )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.01, matrix.max()))
    ax.set_xticks(np.arange(len(EXPERTS)))
    ax.set_xticklabels([EXPERT_LABELS[e] for e in EXPERTS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENARIOS)))
    ax.set_yticklabels([scenario.label for scenario in SCENARIOS])
    ax.set_title("Learned Context + Reliability Expert Weights")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Average Expert Weight")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_expert_weights.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_static(by_strategy: pd.DataFrame) -> None:
    indexed = by_strategy.set_index("strategy")
    best_static = float(indexed.loc[STATIC_STRATEGIES, "optimal_win_rate"].max())
    labels = ["Best Fixed Expert", "Equal Fusion", "Context Fusion", "Context + Reliability", "Oracle"]
    values = [
        best_static,
        float(indexed.loc["equal_fusion", "optimal_win_rate"]),
        float(indexed.loc["context_fusion", "optimal_win_rate"]),
        float(indexed.loc["adaptive_reliability", "optimal_win_rate"]),
        float(indexed.loc["oracle", "optimal_win_rate"]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Does Adaptive Fusion Beat One Fixed Expert Across Contexts?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adaptive_gain_vs_fixed_expert.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    summary: pd.DataFrame,
    by_strategy: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    plot_overall_optimal(summary)
    plot_regret(summary)
    plot_by_scenario(summary)
    plot_strategy_summary(by_strategy)
    plot_weights(weights)
    plot_gain_vs_static(by_strategy)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_strategy, weights = run_experiment()

    raw_path = DATA_DIR / "adaptive_raw_results.csv"
    summary_path = DATA_DIR / "adaptive_summary_results.csv"
    by_strategy_path = DATA_DIR / "adaptive_by_strategy.csv"
    weights_path = DATA_DIR / "adaptive_expert_weights.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_strategy.to_csv(by_strategy_path, index=False)
    weights.to_csv(weights_path, index=False)
    generate_figures(summary, by_strategy, weights)

    print("\nAdaptive voting comparison across all contexts:")
    print(
        by_strategy[
            [
                "strategy_label",
                "optimal_win_rate",
                "average_regret",
                "winner_preservation_rate",
                "tie_rate",
            ]
        ].sort_values("optimal_win_rate", ascending=False).to_string(index=False)
    )
    print("\nGenerated adaptive voting files:")
    for path in [raw_path, summary_path, by_strategy_path, weights_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
