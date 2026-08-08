from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from voting_mrta import (
    FullRound,
    apply_vote_retransmission,
    generate_costs,
    get_winner,
)


ROBOT_COUNTS = list(range(5, 101, 5))
TRIALS = 100
PACKET_LOSS_RATE = 0.30
ROBOT_FAILURE_RATE = 0.05
MAX_TRANSMISSION_ATTEMPTS = 3
RANDOM_SEED = 20260809

COST_START = 10.0
COST_STEP = 5.0

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "algorithms"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass(frozen=True)
class VotingMethod:
    key: str
    label: str
    family: str
    parameter: float | None = None


METHODS = [
    VotingMethod("inverse_alpha_1", "Inverse alpha=1", "inverse", 1.0),
    VotingMethod("inverse_alpha_2", "Inverse alpha=2", "inverse", 2.0),
    VotingMethod("inverse_alpha_3", "Inverse alpha=3", "inverse", 3.0),
    VotingMethod("inverse_alpha_4", "Inverse alpha=4", "inverse", 4.0),
    VotingMethod("softmax_beta_2", "Softmax beta=2", "softmax", 2.0),
    VotingMethod("softmax_beta_4", "Softmax beta=4", "softmax", 4.0),
    VotingMethod("greedy", "Greedy", "greedy", None),
]


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def sample_active_robots(n: int, rng: np.random.Generator) -> np.ndarray:
    active = rng.random(n) >= ROBOT_FAILURE_RATE
    if not active.any():
        # This is extraordinarily unlikely at 5%, but every trial needs at least
        # one available robot to define an allocation problem.
        active[int(rng.integers(0, n))] = True
    return active


def method_probabilities(
    costs: np.ndarray,
    active: np.ndarray,
    method: VotingMethod,
) -> np.ndarray:
    probabilities = np.zeros(len(costs), dtype=float)
    active_costs = costs[active]

    if method.family == "inverse":
        alpha = float(method.parameter)
        weights = np.power(1.0 / active_costs, alpha)
    elif method.family == "softmax":
        beta = float(method.parameter)
        c_min = float(active_costs.min())
        c_max = float(active_costs.max())
        if c_max == c_min:
            normalized_costs = np.zeros_like(active_costs)
        else:
            normalized_costs = (active_costs - c_min) / (c_max - c_min)
        weights = np.exp(-beta * normalized_costs)
    elif method.family == "greedy":
        best_local_index = int(np.argmin(active_costs))
        active_indices = np.flatnonzero(active)
        probabilities[active_indices[best_local_index]] = 1.0
        return probabilities
    else:
        raise ValueError(f"Unknown voting method family: {method.family}")

    probabilities[active] = weights / weights.sum()
    return probabilities


def sample_ballots(
    probabilities: np.ndarray,
    active: np.ndarray,
    voter_uniforms: np.ndarray,
    method: VotingMethod,
) -> np.ndarray:
    ballots = np.full(len(probabilities), -1, dtype=int)
    active_indices = np.flatnonzero(active)

    if method.family == "greedy":
        winner = int(np.argmax(probabilities))
        ballots[active] = winner
        return ballots

    cdf = np.cumsum(probabilities)
    sampled = np.searchsorted(cdf, voter_uniforms[active], side="right")
    sampled = np.minimum(sampled, len(probabilities) - 1)
    ballots[active_indices] = sampled
    return ballots


def build_round(
    costs: np.ndarray,
    active: np.ndarray,
    method: VotingMethod,
    voter_uniforms: np.ndarray,
    tie_priority: np.ndarray,
) -> FullRound:
    probabilities = method_probabilities(costs, active, method)
    ballots = sample_ballots(probabilities, active, voter_uniforms, method)
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


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []

    print(
        "Running voting algorithm comparison: "
        f"packet loss={PACKET_LOSS_RATE:.0%}, "
        f"permanent failure={ROBOT_FAILURE_RATE:.0%}, "
        f"max attempts={MAX_TRANSMISSION_ATTEMPTS}, "
        f"trials={TRIALS}"
    )

    for n in ROBOT_COUNTS:
        costs = generate_costs(n, cost_start=COST_START, cost_step=COST_STEP)

        for trial in range(TRIALS):
            # The failure mask, random ballot draws, tie priorities, and packet
            # outcomes are shared across methods to make the comparison paired.
            active = sample_active_robots(n, rng)
            active_indices = np.flatnonzero(active)
            optimal_robot = int(active_indices[np.argmin(costs[active])])
            optimal_cost = float(costs[optimal_robot])

            voter_uniforms = rng.random(n)
            tie_priority = rng.random(n)
            attempt_random = rng.random((MAX_TRANSMISSION_ATTEMPTS, n))

            for method in METHODS:
                round_data = build_round(
                    costs,
                    active,
                    method,
                    voter_uniforms,
                    tie_priority,
                )
                result = apply_vote_retransmission(
                    round_data,
                    attempt_random,
                    loss_rate=PACKET_LOSS_RATE,
                    max_attempts=MAX_TRANSMISSION_ATTEMPTS,
                )

                winner = result.winner
                no_decision = winner is None
                full_winner = round_data.full_winner

                full_winner_cost = float(costs[full_winner])
                full_is_optimal = full_winner == optimal_robot
                full_regret = full_winner_cost - optimal_cost

                if no_decision:
                    lossy_is_optimal = False
                    winner_preserved = False
                    regret = np.nan
                    winner_cost = np.nan
                else:
                    winner_cost = float(costs[winner])
                    lossy_is_optimal = winner == optimal_robot
                    winner_preserved = winner == full_winner
                    regret = winner_cost - optimal_cost

                active_count = int(active.sum())
                delivered_votes = int(result.delivered.sum())
                total_transmissions = int(result.attempts_used.sum())

                records.append(
                    {
                        "robots": n,
                        "trial": trial + 1,
                        "method": method.key,
                        "method_label": method.label,
                        "method_family": method.family,
                        "method_parameter": method.parameter,
                        "packet_loss_rate": PACKET_LOSS_RATE,
                        "robot_failure_rate": ROBOT_FAILURE_RATE,
                        "max_attempts": MAX_TRANSMISSION_ATTEMPTS,
                        "active_robots": active_count,
                        "observed_robot_failure_rate": 1.0 - active_count / n,
                        "optimal_robot": optimal_robot + 1,
                        "optimal_cost": optimal_cost,
                        "full_winner": full_winner + 1,
                        "full_winner_cost": full_winner_cost,
                        "full_is_optimal": full_is_optimal,
                        "full_regret": full_regret,
                        "lossy_winner": np.nan if winner is None else winner + 1,
                        "winner_cost": winner_cost,
                        "winner_preserved": winner_preserved,
                        "lossy_is_optimal": lossy_is_optimal,
                        "lossy_tie": result.tie,
                        "no_decision": no_decision,
                        "regret": regret,
                        "delivered_votes": delivered_votes,
                        "effective_loss_rate_active": (active_count - delivered_votes)
                        / active_count,
                        "average_attempts_per_active_vote": total_transmissions
                        / active_count,
                    }
                )

    raw = pd.DataFrame.from_records(records)

    summary = (
        raw.groupby(["robots", "method", "method_label"], as_index=False)
        .agg(
            full_vote_optimal_win_rate=("full_is_optimal", "mean"),
            optimal_win_rate=("lossy_is_optimal", "mean"),
            full_vote_average_regret=("full_regret", "mean"),
            average_regret=("regret", "mean"),
            winner_preservation_rate=("winner_preserved", "mean"),
            tie_rate=("lossy_tie", "mean"),
            no_decision_rate=("no_decision", "mean"),
            observed_robot_failure_rate=("observed_robot_failure_rate", "mean"),
            effective_loss_rate_active=("effective_loss_rate_active", "mean"),
            average_attempts_per_active_vote=(
                "average_attempts_per_active_vote",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )

    method_order = {method.key: index for index, method in enumerate(METHODS)}
    summary["method_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(["method_order", "robots"]).reset_index(drop=True)

    by_method = (
        summary.groupby(["method", "method_label", "method_order"], as_index=False)
        .agg(
            full_vote_optimal_win_rate=("full_vote_optimal_win_rate", "mean"),
            optimal_win_rate=("optimal_win_rate", "mean"),
            full_vote_average_regret=("full_vote_average_regret", "mean"),
            average_regret=("average_regret", "mean"),
            winner_preservation_rate=("winner_preservation_rate", "mean"),
            tie_rate=("tie_rate", "mean"),
            no_decision_rate=("no_decision_rate", "mean"),
            observed_robot_failure_rate=("observed_robot_failure_rate", "mean"),
            effective_loss_rate_active=("effective_loss_rate_active", "mean"),
            average_attempts_per_active_vote=(
                "average_attempts_per_active_vote",
                "mean",
            ),
        )
        .sort_values("method_order")
        .reset_index(drop=True)
    )

    return raw, summary, by_method


def plot_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    percent_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for method in METHODS:
        data = summary[summary["method"] == method.key]
        ax.plot(
            data["robots"],
            data[metric],
            marker="o",
            linewidth=1.7,
            markersize=3.8,
            label=method.label,
        )

    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n"
        f"30% Packet Loss + 5% Permanent Failure + "
        f"{MAX_TRANSMISSION_ATTEMPTS} Transmission Attempts"
    )
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    else:
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_summary(by_method: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(by_method))
    width = 0.36

    ax.bar(
        x - width / 2,
        by_method["full_vote_optimal_win_rate"],
        width,
        label="Complete-vote optimal rate",
    )
    ax.bar(
        x + width / 2,
        by_method["optimal_win_rate"],
        width,
        label="After communication loss",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(by_method["method_label"], rotation=25, ha="right")
    ax.set_ylabel("Optimal Win Rate")
    ax.set_title("Average Algorithm Quality Across Robot Team Sizes")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "algorithm_method_summary.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_figures(summary: pd.DataFrame, by_method: pd.DataFrame) -> None:
    plot_metric(
        summary,
        metric="optimal_win_rate",
        ylabel="Optimal Win Rate",
        title="Voting Algorithm Optimal Selection",
        filename="algorithm_optimal_win_rate.png",
        percent_y=True,
    )
    plot_metric(
        summary,
        metric="full_vote_optimal_win_rate",
        ylabel="Complete-Vote Optimal Win Rate",
        title="Intrinsic Voting Algorithm Quality Before Packet Loss",
        filename="algorithm_full_vote_optimal_win_rate.png",
        percent_y=True,
    )
    plot_metric(
        summary,
        metric="average_regret",
        ylabel="Average Regret (Cost Units)",
        title="Voting Algorithm Cost Regret",
        filename="algorithm_average_regret.png",
    )
    plot_metric(
        summary,
        metric="winner_preservation_rate",
        ylabel="Winner Preservation Rate",
        title="Voting Algorithm Winner Preservation",
        filename="algorithm_winner_preservation_rate.png",
        percent_y=True,
    )
    plot_metric(
        summary,
        metric="tie_rate",
        ylabel="Tie Rate",
        title="Voting Algorithm Tie Rate",
        filename="algorithm_tie_rate.png",
        percent_y=True,
    )
    plot_method_summary(by_method)


def print_summary(by_method: pd.DataFrame) -> None:
    columns = [
        "method_label",
        "full_vote_optimal_win_rate",
        "optimal_win_rate",
        "full_vote_average_regret",
        "average_regret",
        "winner_preservation_rate",
        "tie_rate",
    ]
    print("\nAlgorithm comparison averaged across robot team sizes:")
    print(by_method[columns].to_string(index=False))


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_method = run_experiment()

    raw_path = DATA_DIR / "algorithm_raw_results.csv"
    summary_path = DATA_DIR / "algorithm_summary_results.csv"
    by_method_path = DATA_DIR / "algorithm_by_method.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_method.to_csv(by_method_path, index=False)

    generate_figures(summary, by_method)
    print_summary(by_method)

    print("\nGenerated algorithm comparison files:")
    for path in [raw_path, summary_path, by_method_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
