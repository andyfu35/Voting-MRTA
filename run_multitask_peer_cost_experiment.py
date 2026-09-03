from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RANDOM_SEED = 20260903
ROBOT_COUNT = 100
PACKET_LOSS_RATE = 0.30
DEFAULT_TRIALS = 100
TASK_COUNTS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
MIN_SPATIAL_COST = 0.05

INVERSE_ALPHA = 3.0
SOFTMAX_BETA = 5.0
RANK_GAMMA = 3.0
NEAR_OPTIMAL_GAP_PERCENT = 5.0

METHODS = (
    "hungarian",
    "sequential_greedy",
    "greedy",
    "inverse",
    "softmax",
    "rank",
)
METHOD_LABELS = {
    "hungarian": "Hungarian",
    "sequential_greedy": "Sequential Greedy",
    "greedy": "Greedy",
    "inverse": "Inverse",
    "softmax": "Softmax",
    "rank": "Rank",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_peer_cost"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def validate_experiment_config(
    robot_count: int,
    packet_loss_rate: float,
    task_counts: tuple[int, ...],
    trials: int,
) -> None:
    if robot_count < 2:
        raise ValueError("robot_count must be at least 2")
    if not 0.0 <= packet_loss_rate < 1.0:
        raise ValueError("packet_loss_rate must be in [0, 1)")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not task_counts:
        raise ValueError("task_counts must not be empty")
    if any(task_count <= 0 for task_count in task_counts):
        raise ValueError("every task_count must be positive")
    if any(task_count > robot_count for task_count in task_counts):
        raise ValueError(
            "task_count cannot exceed robot_count while robot capacity is one"
        )


def generate_spatial_cost_matrix(
    robot_count: int,
    task_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate the shared ground-truth robot-task cost matrix for one trial."""
    robot_positions = rng.random((robot_count, 2))
    task_positions = rng.random((task_count, 2))
    delta = robot_positions[:, None, :] - task_positions[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return MIN_SPATIAL_COST + distance


def sample_p2p_cost_visibility(
    robot_count: int,
    task_count: int,
    packet_loss_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return [task, receiver, sender] visibility for scalar task-cost messages.

    Each robot computes one cost per task. For every task, sender->receiver cost
    delivery is an independent directed Bernoulli event. Every robot always
    knows its own task cost.
    """
    visible = rng.random((task_count, robot_count, robot_count)) >= packet_loss_rate
    diagonal = np.arange(robot_count)
    visible[:, diagonal, diagonal] = True
    return visible


def sample_weighted_candidates(
    weights: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample exactly one candidate per receiver from normalized local weights."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square [receiver, candidate]")
    if uniforms.shape != (weights.shape[0],):
        raise ValueError("uniforms must have one value per receiver")
    row_sums = weights.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("every receiver must have positive candidate weight")

    probabilities = weights / row_sums[:, None]
    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.0
    return (cdf < uniforms[:, None]).sum(axis=1).astype(int)


def greedy_task_votes(
    task_costs: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """Vote for the cheapest candidate visible to each receiver."""
    visible_costs = np.where(visibility, task_costs[None, :], np.inf)
    return np.argmin(visible_costs, axis=1).astype(int)


def inverse_task_votes(
    task_costs: np.ndarray,
    visibility: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one vote from inverse-cost weights using the selected parameter."""
    weights = np.where(
        visibility,
        np.power(1.0 / task_costs[None, :], INVERSE_ALPHA),
        0.0,
    )
    return sample_weighted_candidates(weights, uniforms)


def softmax_task_votes(
    task_costs: np.ndarray,
    visibility: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one vote from receiver-local normalized softmax cost weights."""
    receiver_count = visibility.shape[0]
    weights = np.zeros_like(visibility, dtype=float)

    for receiver in range(receiver_count):
        candidates = np.flatnonzero(visibility[receiver])
        visible_costs = task_costs[candidates]
        c_min = float(visible_costs.min())
        c_max = float(visible_costs.max())
        if c_max > c_min:
            normalized = (visible_costs - c_min) / (c_max - c_min)
        else:
            normalized = np.zeros_like(visible_costs)
        weights[receiver, candidates] = np.exp(-SOFTMAX_BETA * normalized)

    return sample_weighted_candidates(weights, uniforms)


def rank_task_votes(
    task_costs: np.ndarray,
    visibility: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one vote from inverse-rank weights over actually visible costs."""
    receiver_count = visibility.shape[0]
    weights = np.zeros_like(visibility, dtype=float)

    for receiver in range(receiver_count):
        candidates = np.flatnonzero(visibility[receiver])
        ordered = candidates[np.argsort(task_costs[candidates], kind="stable")]
        ranks = np.arange(1, len(ordered) + 1, dtype=float)
        weights[receiver, ordered] = 1.0 / np.power(ranks, RANK_GAMMA)

    return sample_weighted_candidates(weights, uniforms)


def generate_task_votes(
    method: str,
    task_costs: np.ndarray,
    visibility: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Dispatch one task's local cost-to-vote policy."""
    if method == "greedy":
        return greedy_task_votes(task_costs, visibility)
    if method == "inverse":
        return inverse_task_votes(task_costs, visibility, uniforms)
    if method == "softmax":
        return softmax_task_votes(task_costs, visibility, uniforms)
    if method == "rank":
        return rank_task_votes(task_costs, visibility, uniforms)
    raise ValueError(f"unknown voting method: {method}")


def build_vote_support(
    costs: np.ndarray,
    visibility: np.ndarray,
    voter_uniforms: np.ndarray,
    method: str,
) -> np.ndarray:
    """Build candidate x task support from 100 receiver-local votes per task."""
    robot_count, task_count = costs.shape
    if visibility.shape != (task_count, robot_count, robot_count):
        raise ValueError("visibility shape must be [task, receiver, sender]")
    if voter_uniforms.shape != (task_count, robot_count):
        raise ValueError("voter_uniforms shape must be [task, receiver]")

    support = np.zeros((robot_count, task_count), dtype=np.int16)
    for task in range(task_count):
        votes = generate_task_votes(
            method,
            costs[:, task],
            visibility[task],
            voter_uniforms[task],
        )
        support[:, task] = np.bincount(votes, minlength=robot_count)
    return support


def solve_hungarian_optimal(costs: np.ndarray) -> np.ndarray:
    """Full-information minimum-total-cost assignment reference."""
    row_ind, col_ind = linear_sum_assignment(costs)
    assignment = np.full(costs.shape[1], -1, dtype=int)
    assignment[col_ind] = row_ind
    return assignment


def solve_sequential_greedy(costs: np.ndarray) -> np.ndarray:
    """Full-information heuristic baseline with one task per robot."""
    robot_count, task_count = costs.shape
    available = np.ones(robot_count, dtype=bool)
    assignment = np.full(task_count, -1, dtype=int)

    for task in range(task_count):
        candidates = np.flatnonzero(available)
        chosen = int(candidates[np.argmin(costs[candidates, task])])
        assignment[task] = chosen
        available[chosen] = False
    return assignment


def solve_support_assignment(
    support: np.ndarray,
    tie_priority: np.ndarray,
) -> np.ndarray:
    """Maximize received vote support subject to one task per robot.

    The tiny paired random priority only breaks equal-support assignments. True
    task cost is deliberately not used as a hidden secondary objective.
    """
    if support.shape != tie_priority.shape:
        raise ValueError("support and tie_priority must have identical shapes")

    objective = -support.astype(float) + 1e-6 * tie_priority
    row_ind, col_ind = linear_sum_assignment(objective)
    assignment = np.full(support.shape[1], -1, dtype=int)
    assignment[col_ind] = row_ind
    return assignment


def assignment_total_cost(costs: np.ndarray, assignment: np.ndarray) -> float:
    """Validate capacity-one feasibility and return total true assignment cost."""
    task_count = costs.shape[1]
    if assignment.shape != (task_count,):
        raise ValueError("assignment must contain one robot per task")
    if np.any(assignment < 0):
        raise ValueError("assignment contains an unassigned task")
    if len(np.unique(assignment)) != task_count:
        raise ValueError("assignment violates one-task-per-robot capacity")
    return float(costs[assignment, np.arange(task_count)].sum())


def evaluate_assignment(
    *,
    task_count: int,
    trial: int,
    method: str,
    costs: np.ndarray,
    assignment: np.ndarray,
    optimal_assignment: np.ndarray,
    optimal_cost: float,
) -> dict[str, object]:
    """Evaluate one method against the same full-information Hungarian optimum."""
    total_cost = assignment_total_cost(costs, assignment)
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": ROBOT_COUNT,
        "packet_loss_percent": int(PACKET_LOSS_RATE * 100),
        "tasks": task_count,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap_percent,
        "near_optimal_5pct": gap_percent <= NEAR_OPTIMAL_GAP_PERCENT + 1e-12,
        "exact_optimal_assignment": bool(
            np.array_equal(assignment, optimal_assignment)
        ),
    }


def run_trial(
    *,
    task_count: int,
    trial: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Run one paired multi-task trial for every compared method."""
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
    visibility = sample_p2p_cost_visibility(
        ROBOT_COUNT,
        task_count,
        PACKET_LOSS_RATE,
        rng,
    )
    voter_uniforms = rng.random((task_count, ROBOT_COUNT))
    tie_priority = rng.random((ROBOT_COUNT, task_count))

    optimal_assignment = solve_hungarian_optimal(costs)
    optimal_cost = assignment_total_cost(costs, optimal_assignment)
    records = [
        evaluate_assignment(
            task_count=task_count,
            trial=trial,
            method="hungarian",
            costs=costs,
            assignment=optimal_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
        )
    ]

    greedy_assignment = solve_sequential_greedy(costs)
    records.append(
        evaluate_assignment(
            task_count=task_count,
            trial=trial,
            method="sequential_greedy",
            costs=costs,
            assignment=greedy_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
        )
    )

    for method in ("greedy", "inverse", "softmax", "rank"):
        support = build_vote_support(
            costs,
            visibility,
            voter_uniforms,
            method,
        )
        assignment = solve_support_assignment(support, tie_priority)
        records.append(
            evaluate_assignment(
                task_count=task_count,
                trial=trial,
                method=method,
                costs=costs,
                assignment=assignment,
                optimal_assignment=optimal_assignment,
                optimal_cost=optimal_cost,
            )
        )

    return records


def run_experiment(
    *,
    task_counts: tuple[int, ...] = TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 100-robot, 30%-loss task-count sweep."""
    validate_experiment_config(
        ROBOT_COUNT,
        PACKET_LOSS_RATE,
        task_counts,
        trials,
    )
    records: list[dict[str, object]] = []

    for task_count in task_counts:
        rng = np.random.default_rng(seed + task_count * 100003)
        for trial in range(1, trials + 1):
            records.extend(run_trial(task_count=task_count, trial=trial, rng=rng))
        print(f"tasks={task_count:3d}/{max(task_counts)} complete")

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["tasks", "method", "method_label"], as_index=False)
        .agg(
            average_total_cost=("total_cost", "mean"),
            average_optimality_gap_percent=("optimality_gap_percent", "mean"),
            near_optimal_5pct_percent=(
                "near_optimal_5pct",
                lambda series: 100.0 * series.mean(),
            ),
            exact_optimal_assignment_percent=(
                "exact_optimal_assignment",
                lambda series: 100.0 * series.mean(),
            ),
        )
        .sort_values(["tasks", "method"])
        .reset_index(drop=True)
    )
    return raw, summary


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_report_tables(summary: pd.DataFrame) -> None:
    """Save report-ready tables without exposing selected tuning parameters."""
    gap = summary.pivot(
        index="tasks",
        columns="method_label",
        values="average_optimality_gap_percent",
    ).reset_index()
    near = summary.pivot(
        index="tasks",
        columns="method_label",
        values="near_optimal_5pct_percent",
    ).reset_index()
    exact = summary.pivot(
        index="tasks",
        columns="method_label",
        values="exact_optimal_assignment_percent",
    ).reset_index()

    gap.to_csv(DATA_DIR / "report_average_optimality_gap_percent.csv", index=False)
    near.to_csv(DATA_DIR / "report_near_optimal_5pct_percent.csv", index=False)
    exact.to_csv(DATA_DIR / "report_exact_optimal_assignment_percent.csv", index=False)


def save_gap_plot(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        part = summary[summary["method"] == method].sort_values("tasks")
        ax.plot(
            part["tasks"],
            part["average_optimality_gap_percent"],
            marker="o",
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Simultaneous task count")
    ax.set_ylabel("Average optimality gap (%)")
    ax.set_title("100 robots, 30% directed P2P cost loss, 100 trials/point")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "average_optimality_gap_vs_tasks.png", dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "multitask_peer_cost_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "multitask_peer_cost_summary.csv", index=False)
    save_report_tables(summary)
    save_gap_plot(summary)


def printable_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one pasteable report table with parameter-free method labels."""
    table = summary.pivot(
        index="tasks",
        columns="method_label",
        values=metric,
    ).reset_index()
    preferred = [
        "tasks",
        "Hungarian",
        "Sequential Greedy",
        "Greedy",
        "Inverse",
        "Softmax",
        "Rank",
    ]
    return table[[column for column in preferred if column in table.columns]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-task Voting-MRTA comparison with 100 robots, 30% directed "
            "P2P task-cost packet loss, and 100 trials per task-count/method point."
        )
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=list(TASK_COUNTS),
        help="Simultaneous task counts; each must be <= 100.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_counts = tuple(sorted(set(args.tasks)))
    raw, summary = run_experiment(
        task_counts=task_counts,
        trials=args.trials,
        seed=args.seed,
    )
    save_outputs(raw, summary)

    print("\nExperiment: 100 robots, 30% P2P cost loss")
    print(f"Trials per task-count/method point: {args.trials}")
    print("\nAverage optimality gap (%):")
    print(
        printable_table(
            summary,
            "average_optimality_gap_percent",
        ).round(2).to_string(index=False)
    )
    print("\nNear-optimal within 5% (% of trials):")
    print(
        printable_table(
            summary,
            "near_optimal_5pct_percent",
        ).round(1).to_string(index=False)
    )
    print("\nExact optimal assignment (% of trials):")
    print(
        printable_table(
            summary,
            "exact_optimal_assignment_percent",
        ).round(1).to_string(index=False)
    )
    print("\nSaved:")
    print(DATA_DIR)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
