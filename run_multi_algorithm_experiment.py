from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import (
    COST_START,
    COST_STEP,
    MAX_TRANSMISSION_ATTEMPTS,
    PACKET_LOSS_RATE,
    ROBOT_COUNTS,
    ROBOT_FAILURE_RATE,
    TRIALS,
    VotingMethod,
    method_probabilities,
    sample_active_robots,
)
from voting_mrta import FullRound, apply_vote_retransmission, generate_costs, get_winner


RANDOM_SEED = 20260810
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multi_algorithm"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


RULES = {
    "inverse_alpha_1": VotingMethod("inverse_alpha_1", "Inverse alpha=1", "inverse", 1.0),
    "inverse_alpha_2": VotingMethod("inverse_alpha_2", "Inverse alpha=2", "inverse", 2.0),
    "inverse_alpha_3": VotingMethod("inverse_alpha_3", "Inverse alpha=3", "inverse", 3.0),
    "softmax_beta_025": VotingMethod("softmax_beta_025", "Softmax beta=0.25", "softmax", 0.25),
    "softmax_beta_05": VotingMethod("softmax_beta_05", "Softmax beta=0.5", "softmax", 0.5),
    "greedy": VotingMethod("greedy", "Greedy", "greedy", None),
}


@dataclass(frozen=True)
class VotingStrategy:
    key: str
    label: str
    components: tuple[str, ...]

    @property
    def is_multi(self) -> bool:
        return len(self.components) > 1


STRATEGIES = [
    VotingStrategy("single_inverse_1", "Single: Inverse alpha=1", ("inverse_alpha_1",)),
    VotingStrategy("single_inverse_2", "Single: Inverse alpha=2", ("inverse_alpha_2",)),
    VotingStrategy("single_inverse_3", "Single: Inverse alpha=3", ("inverse_alpha_3",)),
    VotingStrategy("single_softmax_025", "Single: Softmax beta=0.25", ("softmax_beta_025",)),
    VotingStrategy("single_softmax_05", "Single: Softmax beta=0.5", ("softmax_beta_05",)),
    VotingStrategy("single_greedy", "Single: Greedy", ("greedy",)),
    VotingStrategy(
        "multi_balanced",
        "Multi Balanced",
        ("inverse_alpha_1", "inverse_alpha_2", "softmax_beta_025"),
    ),
    VotingStrategy(
        "multi_strong",
        "Multi Strong",
        ("inverse_alpha_2", "inverse_alpha_3", "softmax_beta_05"),
    ),
    VotingStrategy(
        "multi_diverse",
        "Multi Diverse",
        ("inverse_alpha_1", "softmax_beta_025", "greedy"),
    ),
]

SINGLE_STRATEGY_FOR_RULE = {
    strategy.components[0]: strategy.key for strategy in STRATEGIES if not strategy.is_multi
}


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def sample_strategy_ballots(
    costs: np.ndarray,
    active: np.ndarray,
    strategy: VotingStrategy,
    voter_uniforms: np.ndarray,
    assignment_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ballots = np.full(len(costs), -1, dtype=int)
    ordered_active = assignment_order[active[assignment_order]]
    cache = {
        key: method_probabilities(costs, active, RULES[key]) for key in set(strategy.components)
    }
    aggregate_probabilities = np.zeros(len(costs), dtype=float)

    for position, voter in enumerate(ordered_active):
        key = strategy.components[position % len(strategy.components)]
        method = RULES[key]
        probabilities = cache[key]
        aggregate_probabilities += probabilities

        if method.family == "greedy":
            ballots[voter] = int(np.argmax(probabilities))
        else:
            candidate = int(
                np.searchsorted(
                    np.cumsum(probabilities),
                    voter_uniforms[voter],
                    side="right",
                )
            )
            ballots[voter] = min(candidate, len(costs) - 1)

    aggregate_probabilities /= int(active.sum())
    return ballots, aggregate_probabilities


def build_round(
    costs: np.ndarray,
    active: np.ndarray,
    strategy: VotingStrategy,
    voter_uniforms: np.ndarray,
    tie_priority: np.ndarray,
    assignment_order: np.ndarray,
) -> FullRound:
    ballots, probabilities = sample_strategy_ballots(
        costs, active, strategy, voter_uniforms, assignment_order
    )
    full_counts = np.bincount(ballots[active], minlength=len(costs))
    full_winner, full_tie = get_winner(full_counts, tie_priority)
    if full_winner is None:
        raise RuntimeError("A round with active voters must have a full-vote winner")

    return FullRound(
        costs=costs,
        probabilities=probabilities,
        ballots=ballots,
        full_counts=full_counts,
        full_winner=full_winner,
        full_tie=full_tie,
        tie_priority=tie_priority,
        delivery_random=np.zeros(len(costs), dtype=float),
        active_robots=active,
    )


def add_component_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    columns = [
        "component_mean_optimal_win_rate",
        "component_best_optimal_win_rate",
        "gain_vs_component_mean",
        "gain_vs_best_component",
    ]
    for column in columns:
        summary[column] = np.nan

    for strategy in STRATEGIES:
        if not strategy.is_multi:
            continue
        component_keys = [SINGLE_STRATEGY_FOR_RULE[key] for key in strategy.components]

        for n in ROBOT_COUNTS:
            target = (summary["strategy"] == strategy.key) & (summary["robots"] == n)
            components = summary[
                (summary["robots"] == n) & summary["strategy"].isin(component_keys)
            ]
            if len(components) != len(component_keys):
                raise RuntimeError(f"Missing component baseline for {strategy.label} at N={n}")

            ensemble_rate = float(summary.loc[target, "optimal_win_rate"].iloc[0])
            mean_rate = float(components["optimal_win_rate"].mean())
            best_rate = float(components["optimal_win_rate"].max())
            summary.loc[target, "component_mean_optimal_win_rate"] = mean_rate
            summary.loc[target, "component_best_optimal_win_rate"] = best_rate
            summary.loc[target, "gain_vs_component_mean"] = ensemble_rate - mean_rate
            summary.loc[target, "gain_vs_best_component"] = ensemble_rate - best_rate

    return summary


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []
    print(
        "Running multi-algorithm voting: "
        f"loss={PACKET_LOSS_RATE:.0%}, failure={ROBOT_FAILURE_RATE:.0%}, "
        f"attempts={MAX_TRANSMISSION_ATTEMPTS}, trials={TRIALS}"
    )

    for n in ROBOT_COUNTS:
        costs = generate_costs(n, cost_start=COST_START, cost_step=COST_STEP)

        for trial in range(TRIALS):
            # All strategies share failures, random draws, tie priority, packet
            # outcomes, and the randomized voter-to-algorithm assignment order.
            active = sample_active_robots(n, rng)
            active_indices = np.flatnonzero(active)
            optimal_robot = int(active_indices[np.argmin(costs[active])])
            optimal_cost = float(costs[optimal_robot])
            voter_uniforms = rng.random(n)
            tie_priority = rng.random(n)
            attempt_random = rng.random((MAX_TRANSMISSION_ATTEMPTS, n))
            assignment_order = rng.permutation(n)

            for strategy in STRATEGIES:
                round_data = build_round(
                    costs,
                    active,
                    strategy,
                    voter_uniforms,
                    tie_priority,
                    assignment_order,
                )
                result = apply_vote_retransmission(
                    round_data,
                    attempt_random,
                    loss_rate=PACKET_LOSS_RATE,
                    max_attempts=MAX_TRANSMISSION_ATTEMPTS,
                )

                full_winner = round_data.full_winner
                full_winner_cost = float(costs[full_winner])
                winner = result.winner
                no_decision = winner is None
                active_count = int(active.sum())
                delivered_votes = int(result.delivered.sum())

                records.append(
                    {
                        "robots": n,
                        "trial": trial + 1,
                        "strategy": strategy.key,
                        "strategy_label": strategy.label,
                        "strategy_type": "multi" if strategy.is_multi else "single",
                        "components": " + ".join(RULES[key].label for key in strategy.components),
                        "active_robots": active_count,
                        "optimal_robot": optimal_robot + 1,
                        "optimal_cost": optimal_cost,
                        "full_winner": full_winner + 1,
                        "full_is_optimal": full_winner == optimal_robot,
                        "full_regret": full_winner_cost - optimal_cost,
                        "lossy_winner": np.nan if no_decision else winner + 1,
                        "lossy_is_optimal": False if no_decision else winner == optimal_robot,
                        "winner_preserved": False if no_decision else winner == full_winner,
                        "lossy_tie": result.tie,
                        "no_decision": no_decision,
                        "regret": np.nan if no_decision else float(costs[winner]) - optimal_cost,
                        "effective_loss_rate_active": (active_count - delivered_votes) / active_count,
                    }
                )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            ["robots", "strategy", "strategy_label", "strategy_type", "components"],
            as_index=False,
        )
        .agg(
            full_vote_optimal_win_rate=("full_is_optimal", "mean"),
            optimal_win_rate=("lossy_is_optimal", "mean"),
            full_vote_average_regret=("full_regret", "mean"),
            average_regret=("regret", "mean"),
            winner_preservation_rate=("winner_preserved", "mean"),
            tie_rate=("lossy_tie", "mean"),
            no_decision_rate=("no_decision", "mean"),
            effective_loss_rate_active=("effective_loss_rate_active", "mean"),
        )
        .reset_index(drop=True)
    )

    order = {strategy.key: index for index, strategy in enumerate(STRATEGIES)}
    summary["strategy_order"] = summary["strategy"].map(order)
    summary = add_component_comparisons(summary)
    summary = summary.sort_values(["strategy_order", "robots"]).reset_index(drop=True)

    by_strategy = (
        summary.groupby(
            ["strategy", "strategy_label", "strategy_type", "components", "strategy_order"],
            as_index=False,
        )
        .agg(
            full_vote_optimal_win_rate=("full_vote_optimal_win_rate", "mean"),
            optimal_win_rate=("optimal_win_rate", "mean"),
            average_regret=("average_regret", "mean"),
            winner_preservation_rate=("winner_preservation_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
            component_mean_optimal_win_rate=("component_mean_optimal_win_rate", "mean"),
            component_best_optimal_win_rate=("component_best_optimal_win_rate", "mean"),
            gain_vs_component_mean=("gain_vs_component_mean", "mean"),
            gain_vs_best_component=("gain_vs_best_component", "mean"),
        )
        .sort_values("strategy_order")
        .reset_index(drop=True)
    )
    return raw, summary, by_strategy


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    percent_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy in STRATEGIES:
        data = summary[summary["strategy"] == strategy.key]
        ax.plot(
            data["robots"],
            data[metric],
            marker="o",
            linewidth=2.2 if strategy.is_multi else 1.5,
            linestyle="--" if strategy.is_multi else "-",
            markersize=3.5,
            label=strategy.label,
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n30% Packet Loss + 5% Permanent Failure + "
        f"{MAX_TRANSMISSION_ATTEMPTS} Transmission Attempts"
    )
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7.5)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    else:
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_summary(by_strategy: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(by_strategy))
    width = 0.36
    ax.bar(
        x - width / 2,
        by_strategy["full_vote_optimal_win_rate"],
        width,
        label="Complete-vote optimal rate",
    )
    ax.bar(
        x + width / 2,
        by_strategy["optimal_win_rate"],
        width,
        label="After communication loss",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(by_strategy["strategy_label"], rotation=30, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Single vs. Multi-Algorithm Voting Quality")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multi_algorithm_strategy_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_components(by_strategy: pd.DataFrame) -> None:
    data = by_strategy[by_strategy["strategy_type"] == "multi"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(data))
    width = 0.25
    ax.bar(x - width, data["component_mean_optimal_win_rate"], width, label="Mean component rate")
    ax.bar(x, data["component_best_optimal_win_rate"], width, label="Best component rate")
    ax.bar(x + width, data["optimal_win_rate"], width, label="Multi-algorithm final vote")
    ax.set_xticks(x)
    ax.set_xticklabels(data["strategy_label"], rotation=20, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Does Heterogeneous Voting Improve the Final Decision?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multi_algorithm_gain_vs_components.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_figures(summary: pd.DataFrame, by_strategy: pd.DataFrame) -> None:
    plot_metric(summary, "optimal_win_rate", "Optimal Win Rate", "Multi-Algorithm Voting Optimal Selection", "multi_algorithm_optimal_win_rate.png", True)
    plot_metric(summary, "average_regret", "Average Regret (Cost Units)", "Multi-Algorithm Voting Cost Regret", "multi_algorithm_average_regret.png")
    plot_metric(summary, "tie_rate", "Tie Rate", "Multi-Algorithm Voting Tie Rate", "multi_algorithm_tie_rate.png", True)
    plot_metric(summary, "winner_preservation_rate", "Winner Preservation Rate", "Multi-Algorithm Voting Winner Preservation", "multi_algorithm_winner_preservation_rate.png", True)
    plot_strategy_summary(by_strategy)
    plot_gain_vs_components(by_strategy)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_strategy = run_experiment()
    raw_path = DATA_DIR / "multi_algorithm_raw_results.csv"
    summary_path = DATA_DIR / "multi_algorithm_summary_results.csv"
    by_strategy_path = DATA_DIR / "multi_algorithm_by_strategy.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_strategy.to_csv(by_strategy_path, index=False)
    generate_figures(summary, by_strategy)

    columns = [
        "strategy_label",
        "strategy_type",
        "optimal_win_rate",
        "average_regret",
        "winner_preservation_rate",
        "tie_rate",
        "gain_vs_component_mean",
        "gain_vs_best_component",
    ]
    print("\nMulti-algorithm comparison averaged across robot team sizes:")
    print(by_strategy[columns].to_string(index=False))
    print("\nGenerated multi-algorithm files:")
    for path in [raw_path, summary_path, by_strategy_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
