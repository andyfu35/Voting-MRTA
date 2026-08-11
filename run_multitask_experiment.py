from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from run_algorithm_experiment import (
    MAX_TRANSMISSION_ATTEMPTS,
    PACKET_LOSS_RATE,
    ROBOT_COUNTS,
    ROBOT_FAILURE_RATE,
    TRIALS,
    sample_active_robots,
)


RANDOM_SEED = 20260811
TASK_LOAD_RATIOS = (0.25, 0.50, 0.75)
ROBOT_CAPACITY = 1
MIN_SPATIAL_COST = 0.05
SOFTMAX_COST_SCALE = 0.25
NEAR_OPTIMAL_GAP = 5.0

BASE_VOTING_METHODS = [
    "inverse_a1",
    "inverse_a2",
    "inverse_a3",
    "softmax_b1",
    "softmax_b2",
    "greedy_vote",
]
VOTING_METHODS = BASE_VOTING_METHODS + ["heterogeneous_vote"]
METHODS = ["centralized_optimal", "sequential_greedy"] + VOTING_METHODS

METHOD_LABELS = {
    "centralized_optimal": "Centralized Optimal",
    "sequential_greedy": "Sequential Greedy",
    "inverse_a1": "Voting: Inverse a=1",
    "inverse_a2": "Voting: Inverse a=2",
    "inverse_a3": "Voting: Inverse a=3",
    "softmax_b1": "Voting: Softmax b=1",
    "softmax_b2": "Voting: Softmax b=2",
    "greedy_vote": "Voting: Greedy",
    "heterogeneous_vote": "Voting: Heterogeneous",
}

HETEROGENEOUS_COMPONENTS = ["inverse_a1", "inverse_a2", "softmax_b1", "greedy_vote"]

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def generate_spatial_cost_matrix(
    n: int,
    task_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use one shared cost model for every allocation method.

    Robots and tasks are sampled in the same normalized 2-D workspace. The cost
    is Euclidean travel distance plus a small positive floor so inverse-cost
    voting remains numerically well-defined.
    """
    robot_positions = rng.random((n, 2))
    task_positions = rng.random((task_count, 2))
    delta = robot_positions[:, None, :] - task_positions[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return MIN_SPATIAL_COST + distance


def centralized_optimal_assignment(
    costs: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    active_indices = np.flatnonzero(active)
    row_ind, col_ind = linear_sum_assignment(costs[active_indices, :])
    assignment = np.full(costs.shape[1], -1, dtype=int)
    assignment[col_ind] = active_indices[row_ind]
    return assignment


def sequential_greedy_assignment(
    costs: np.ndarray,
    active: np.ndarray,
    task_order: np.ndarray,
) -> np.ndarray:
    available = set(np.flatnonzero(active).tolist())
    assignment = np.full(costs.shape[1], -1, dtype=int)

    for task in task_order:
        candidates = np.asarray(sorted(available), dtype=int)
        chosen = int(candidates[np.argmin(costs[candidates, task])])
        assignment[task] = chosen
        available.remove(chosen)

    return assignment


def per_task_probability(
    costs: np.ndarray,
    active: np.ndarray,
    method: str,
) -> np.ndarray:
    """Return candidate probabilities with shape [robots, tasks]."""
    n, task_count = costs.shape
    probabilities = np.zeros((n, task_count), dtype=float)
    active_indices = np.flatnonzero(active)

    for task in range(task_count):
        active_costs = costs[active_indices, task]

        if method.startswith("inverse_a"):
            alpha = float(method.removeprefix("inverse_a"))
            weights = np.power(1.0 / active_costs, alpha)
        elif method.startswith("softmax_b"):
            beta = float(method.removeprefix("softmax_b"))
            shifted = (active_costs - active_costs.min()) / SOFTMAX_COST_SCALE
            weights = np.exp(-beta * shifted)
        elif method == "greedy_vote":
            weights = np.zeros(len(active_indices), dtype=float)
            weights[int(np.argmin(active_costs))] = 1.0
        else:
            raise ValueError(f"Unknown base voting method: {method}")

        probabilities[active_indices, task] = weights / weights.sum()

    return probabilities


def sample_ballots(
    probabilities: np.ndarray,
    active: np.ndarray,
    voter_uniforms: np.ndarray,
) -> np.ndarray:
    """Return candidate robot index for each voter-task pair, or -1 if inactive."""
    n, task_count = probabilities.shape
    ballots = np.full((n, task_count), -1, dtype=int)
    active_indices = np.flatnonzero(active)

    for task in range(task_count):
        cdf = np.cumsum(probabilities[:, task])
        sampled = np.searchsorted(cdf, voter_uniforms[active_indices, task], side="right")
        sampled = np.minimum(sampled, n - 1)
        ballots[active_indices, task] = sampled

    return ballots


def assign_heterogeneous_policies(
    active: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, str]:
    active_indices = np.flatnonzero(active)
    ordered = rng.permutation(active_indices)
    return {
        int(robot): HETEROGENEOUS_COMPONENTS[index % len(HETEROGENEOUS_COMPONENTS)]
        for index, robot in enumerate(ordered)
    }


def heterogeneous_ballots(
    base_ballots: dict[str, np.ndarray],
    active: np.ndarray,
    voter_policy: dict[int, str],
) -> np.ndarray:
    any_ballots = next(iter(base_ballots.values()))
    ballots = np.full_like(any_ballots, -1)
    for robot in np.flatnonzero(active):
        ballots[robot, :] = base_ballots[voter_policy[int(robot)]][robot, :]
    return ballots


def delivery_mask(
    active: np.ndarray,
    attempt_random: np.ndarray,
) -> np.ndarray:
    """Stop-on-success delivery status for every voter-task vote."""
    delivered = np.any(attempt_random >= PACKET_LOSS_RATE, axis=0)
    return delivered & active[:, None]


def vote_support(
    ballots: np.ndarray,
    voters_included: np.ndarray,
    n: int,
) -> np.ndarray:
    task_count = ballots.shape[1]
    support = np.zeros((n, task_count), dtype=int)

    for task in range(task_count):
        voters = np.flatnonzero(voters_included[:, task])
        if len(voters) == 0:
            continue
        support[:, task] = np.bincount(ballots[voters, task], minlength=n)

    return support


def support_assignment(
    support: np.ndarray,
    active: np.ndarray,
    tie_priority: np.ndarray,
) -> np.ndarray:
    """Maximize team vote support subject to one task per active robot.

    A tiny random paired priority is used only to break equal-support global
    assignments. The terminal does not use task cost as a hidden secondary
    objective, so the experiment measures how informative the votes are.
    """
    active_indices = np.flatnonzero(active)
    objective = -support[active_indices, :].astype(float)
    objective += 1e-6 * tie_priority[active_indices, :]
    row_ind, col_ind = linear_sum_assignment(objective)
    assignment = np.full(support.shape[1], -1, dtype=int)
    assignment[col_ind] = active_indices[row_ind]
    return assignment


def assignment_total_cost(costs: np.ndarray, assignment: np.ndarray) -> float:
    tasks = np.arange(costs.shape[1])
    return float(costs[assignment, tasks].sum())


def independent_minimum_diagnostic(
    costs: np.ndarray,
    active: np.ndarray,
) -> tuple[bool, int]:
    """Show why independent per-task winners are not a feasible MRTA solution."""
    active_indices = np.flatnonzero(active)
    chosen = active_indices[np.argmin(costs[active_indices, :], axis=0)]
    _, counts = np.unique(chosen, return_counts=True)
    conflict = bool(np.any(counts > ROBOT_CAPACITY))
    max_load = int(counts.max()) if len(counts) else 0
    return conflict, max_load


def make_record(
    *,
    robots: int,
    tasks: int,
    load_ratio: float,
    trial: int,
    method: str,
    total_cost: float,
    optimal_cost: float,
    optimal_assignment: np.ndarray,
    assignment: np.ndarray,
    independent_conflict: bool,
    independent_max_load: int,
    received_vote_rate: float,
    full_assignment: np.ndarray | None = None,
) -> dict[str, object]:
    gap = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": robots,
        "tasks": tasks,
        "task_load_ratio": load_ratio,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap,
        "near_optimal_5pct": gap <= NEAR_OPTIMAL_GAP + 1e-12,
        "exact_assignment_match": bool(np.array_equal(assignment, optimal_assignment)),
        "assignment_preserved": (
            bool(np.array_equal(assignment, full_assignment))
            if full_assignment is not None
            else True
        ),
        "independent_conflict": independent_conflict,
        "independent_max_load": independent_max_load,
        "received_vote_rate": received_vote_rate,
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in ROBOT_COUNTS:
        for load_index, load_ratio in enumerate(TASK_LOAD_RATIOS):
            rng = np.random.default_rng(RANDOM_SEED + n * 1000 + load_index * 100000)

            for trial in range(TRIALS):
                active = sample_active_robots(n, rng)
                active_count = int(active.sum())
                task_count = max(1, int(np.floor(active_count * load_ratio)))
                task_count = min(task_count, active_count)

                costs = generate_spatial_cost_matrix(n, task_count, rng)
                optimal_assignment = centralized_optimal_assignment(costs, active)
                optimal_cost = assignment_total_cost(costs, optimal_assignment)
                independent_conflict, independent_max_load = independent_minimum_diagnostic(
                    costs, active
                )

                task_order = rng.permutation(task_count)
                tie_priority = rng.random((n, task_count))
                voter_uniforms = rng.random((n, task_count))
                attempt_random = rng.random(
                    (MAX_TRANSMISSION_ATTEMPTS, n, task_count)
                )
                delivered = delivery_mask(active, attempt_random)
                possible_votes = active_count * task_count
                received_vote_rate = float(delivered.sum() / possible_votes)

                records.append(
                    make_record(
                        robots=n,
                        tasks=task_count,
                        load_ratio=load_ratio,
                        trial=trial + 1,
                        method="centralized_optimal",
                        total_cost=optimal_cost,
                        optimal_cost=optimal_cost,
                        optimal_assignment=optimal_assignment,
                        assignment=optimal_assignment,
                        independent_conflict=independent_conflict,
                        independent_max_load=independent_max_load,
                        received_vote_rate=received_vote_rate,
                    )
                )

                greedy_assignment = sequential_greedy_assignment(
                    costs, active, task_order
                )
                records.append(
                    make_record(
                        robots=n,
                        tasks=task_count,
                        load_ratio=load_ratio,
                        trial=trial + 1,
                        method="sequential_greedy",
                        total_cost=assignment_total_cost(costs, greedy_assignment),
                        optimal_cost=optimal_cost,
                        optimal_assignment=optimal_assignment,
                        assignment=greedy_assignment,
                        independent_conflict=independent_conflict,
                        independent_max_load=independent_max_load,
                        received_vote_rate=received_vote_rate,
                    )
                )

                probability_map = {
                    method: per_task_probability(costs, active, method)
                    for method in BASE_VOTING_METHODS
                }
                base_ballots = {
                    method: sample_ballots(probability_map[method], active, voter_uniforms)
                    for method in BASE_VOTING_METHODS
                }
                voter_policy = assign_heterogeneous_policies(active, rng)
                ballots_map = {
                    **base_ballots,
                    "heterogeneous_vote": heterogeneous_ballots(
                        base_ballots, active, voter_policy
                    ),
                }

                full_voters = np.broadcast_to(active[:, None], (n, task_count))
                for method in VOTING_METHODS:
                    ballots = ballots_map[method]
                    full_support = vote_support(ballots, full_voters, n)
                    final_support = vote_support(ballots, delivered, n)
                    full_assignment = support_assignment(
                        full_support, active, tie_priority
                    )
                    final_assignment = support_assignment(
                        final_support, active, tie_priority
                    )
                    records.append(
                        make_record(
                            robots=n,
                            tasks=task_count,
                            load_ratio=load_ratio,
                            trial=trial + 1,
                            method=method,
                            total_cost=assignment_total_cost(costs, final_assignment),
                            optimal_cost=optimal_cost,
                            optimal_assignment=optimal_assignment,
                            assignment=final_assignment,
                            full_assignment=full_assignment,
                            independent_conflict=independent_conflict,
                            independent_max_load=independent_max_load,
                            received_vote_rate=received_vote_rate,
                        )
                    )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            ["robots", "task_load_ratio", "method", "method_label"],
            as_index=False,
        )
        .agg(
            average_tasks=("tasks", "mean"),
            average_total_cost=("total_cost", "mean"),
            average_optimality_gap_percent=("optimality_gap_percent", "mean"),
            near_optimal_5pct_rate=("near_optimal_5pct", "mean"),
            exact_assignment_match_rate=("exact_assignment_match", "mean"),
            assignment_preservation_rate=("assignment_preserved", "mean"),
            independent_conflict_rate=("independent_conflict", "mean"),
            average_independent_max_load=("independent_max_load", "mean"),
            received_vote_rate=("received_vote_rate", "mean"),
        )
        .reset_index(drop=True)
    )
    by_method = (
        summary.groupby(["method", "method_label"], as_index=False)
        .agg(
            average_optimality_gap_percent=("average_optimality_gap_percent", "mean"),
            near_optimal_5pct_rate=("near_optimal_5pct_rate", "mean"),
            exact_assignment_match_rate=("exact_assignment_match_rate", "mean"),
            assignment_preservation_rate=("assignment_preservation_rate", "mean"),
        )
        .reset_index(drop=True)
    )
    return raw, summary, by_method


def plot_optimality_gap(summary: pd.DataFrame) -> None:
    averaged = (
        summary.groupby(["robots", "method", "method_label"], as_index=False)
        .agg(gap=("average_optimality_gap_percent", "mean"))
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in METHODS:
        subset = averaged[averaged["method"] == method]
        ax.plot(
            subset["robots"],
            subset["gap"],
            marker="o",
            markersize=3.5,
            linewidth=2.2 if method == "heterogeneous_vote" else 1.5,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Optimality Gap (%)")
    ax.set_title("Multi-Task Allocation Quality with One Shared Cost Model")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multitask_optimality_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_near_optimal(summary: pd.DataFrame) -> None:
    averaged = (
        summary.groupby(["robots", "method", "method_label"], as_index=False)
        .agg(rate=("near_optimal_5pct_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in METHODS:
        subset = averaged[averaged["method"] == method]
        ax.plot(
            subset["robots"],
            subset["rate"],
            marker="o",
            markersize=3.5,
            linewidth=2.2 if method == "heterogeneous_vote" else 1.5,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Near-Optimal Allocation Rate (Gap <= 5%)")
    ax.set_title("How Often Does the Global Allocation Stay Near the Cost Optimum?")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multitask_near_optimal_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_load_ratio(summary: pd.DataFrame) -> None:
    data = (
        summary.groupby(["task_load_ratio", "method", "method_label"], as_index=False)
        .agg(gap=("average_optimality_gap_percent", "mean"))
    )
    x = np.arange(len(TASK_LOAD_RATIOS))
    width = 0.10
    fig, ax = plt.subplots(figsize=(13, 7))
    for index, method in enumerate(METHODS):
        subset = data[data["method"] == method].set_index("task_load_ratio")
        values = [float(subset.loc[ratio, "gap"]) for ratio in TASK_LOAD_RATIOS]
        offset = (index - (len(METHODS) - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(ratio * 100)}% task load" for ratio in TASK_LOAD_RATIOS])
    ax.set_ylabel("Average Optimality Gap (%)")
    ax.set_title("Voting Policies Under Increasing Simultaneous-Task Pressure")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multitask_by_load_ratio.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_independent_conflicts(summary: pd.DataFrame) -> None:
    diagnostic = (
        summary[summary["method"] == "centralized_optimal"]
        .groupby(["robots", "task_load_ratio"], as_index=False)
        .agg(conflict_rate=("independent_conflict_rate", "mean"))
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for ratio in TASK_LOAD_RATIOS:
        subset = diagnostic[diagnostic["task_load_ratio"] == ratio]
        ax.plot(
            subset["robots"],
            subset["conflict_rate"],
            marker="o",
            linewidth=2.0,
            label=f"Task load {int(ratio * 100)}%",
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Independent Min-Cost Conflict Rate")
    ax.set_title("Why Per-Task Winners Need a Global Assignment Layer")
    ax.set_xticks(ROBOT_COUNTS)
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multitask_independent_conflict_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_summary(by_method: pd.DataFrame) -> None:
    ordered = by_method.set_index("method").loc[METHODS].reset_index()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["average_optimality_gap_percent"])
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["method_label"], rotation=30, ha="right")
    ax.set_ylabel("Average Optimality Gap (%)")
    ax.set_title("Average Multi-Task Allocation Quality Across Team Sizes and Loads")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "multitask_method_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_figures(summary: pd.DataFrame, by_method: pd.DataFrame) -> None:
    plot_optimality_gap(summary)
    plot_near_optimal(summary)
    plot_by_load_ratio(summary)
    plot_independent_conflicts(summary)
    plot_method_summary(by_method)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_method = run_experiment()

    raw_path = DATA_DIR / "multitask_raw_results.csv"
    summary_path = DATA_DIR / "multitask_summary_results.csv"
    by_method_path = DATA_DIR / "multitask_by_method.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_method.to_csv(by_method_path, index=False)
    generate_figures(summary, by_method)

    print("\nMulti-task Voting-MRTA comparison:")
    print(
        by_method[
            [
                "method_label",
                "average_optimality_gap_percent",
                "near_optimal_5pct_rate",
                "assignment_preservation_rate",
            ]
        ]
        .sort_values("average_optimality_gap_percent")
        .to_string(index=False)
    )
    print("\nGenerated multi-task files:")
    for path in [raw_path, summary_path, by_method_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
