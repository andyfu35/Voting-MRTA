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

NEAR_OPTIMAL_GAP_PERCENT = 5.0
OPTIMAL_COST_TOLERANCE_PERCENT = 1e-8
AUCTION_EPSILON_LEVELS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
AUCTION_MAX_STAGE_ITERATIONS = 50_000

METHODS = (
    "oracle",
    "p2p_greedy",
    "p2p_hungarian",
    "p2p_auction",
)
METHOD_LABELS = {
    "oracle": "Hungarian Oracle",
    "p2p_greedy": "P2P Greedy",
    "p2p_hungarian": "P2P Hungarian",
    "p2p_auction": "P2P Auction",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_peer_cost"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first owner/function boundary."""
    raise ValueError(
        "owner=run_multitask_peer_cost_experiment "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_experiment_config(
    robot_count: int,
    packet_loss_rate: float,
    task_counts: tuple[int, ...],
    trials: int,
) -> None:
    if robot_count < 2:
        fail("validate_experiment_config", "contract", "ROBOT_COUNT_TOO_SMALL", f"actual={robot_count}")
    if not 0.0 <= packet_loss_rate < 1.0:
        fail("validate_experiment_config", "contract", "INVALID_PACKET_LOSS", f"actual={packet_loss_rate}")
    if trials <= 0:
        fail("validate_experiment_config", "contract", "INVALID_TRIALS", f"actual={trials}")
    if not task_counts:
        fail("validate_experiment_config", "contract", "EMPTY_TASK_COUNTS", "task_counts is empty")
    if any(task_count <= 0 for task_count in task_counts):
        fail("validate_experiment_config", "contract", "INVALID_TASK_COUNT", f"actual={task_counts}")
    if any(task_count > robot_count for task_count in task_counts):
        fail(
            "validate_experiment_config",
            "state",
            "CAPACITY_EXCEEDED",
            f"robot_count={robot_count} task_counts={task_counts}",
        )


def generate_spatial_cost_matrix(robot_count: int, task_count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one ground-truth robot x task travel-cost matrix."""
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
    """Return [receiver, sender, task] directed P2P cost visibility."""
    visible = rng.random((robot_count, robot_count, task_count)) >= packet_loss_rate
    diagonal = np.arange(robot_count)
    visible[diagonal, diagonal, :] = True
    return visible


def build_receiver_cost_views(costs: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Materialize every receiver's incomplete robot x task cost matrix."""
    robot_count, task_count = costs.shape
    expected_shape = (robot_count, robot_count, task_count)
    if visibility.shape != expected_shape:
        fail(
            "build_receiver_cost_views",
            "contract",
            "VISIBILITY_SHAPE_MISMATCH",
            f"expected={expected_shape} actual={visibility.shape}",
        )
    return np.where(visibility, costs[None, :, :], np.inf)


def assignment_total_cost(costs: np.ndarray, assignment: np.ndarray) -> float:
    """Validate one-task-per-robot feasibility and return total true cost."""
    task_count = costs.shape[1]
    if assignment.shape != (task_count,):
        fail(
            "assignment_total_cost",
            "contract",
            "ASSIGNMENT_SHAPE_MISMATCH",
            f"expected={(task_count,)} actual={assignment.shape}",
        )
    if np.any(assignment < 0) or np.any(assignment >= costs.shape[0]):
        fail("assignment_total_cost", "state", "INVALID_ROBOT_INDEX", f"assignment={assignment.tolist()}")
    if len(np.unique(assignment)) != task_count:
        fail(
            "assignment_total_cost",
            "state",
            "CAPACITY_VIOLATION",
            "one robot was assigned to multiple simultaneous tasks",
        )
    selected = costs[assignment, np.arange(task_count)]
    if np.any(~np.isfinite(selected)):
        fail(
            "assignment_total_cost",
            "planning",
            "INFEASIBLE_EDGE_SELECTED",
            "assignment contains a cost edge unavailable to its optimizer",
        )
    return float(selected.sum())


def solve_hungarian_assignment(costs: np.ndarray) -> np.ndarray | None:
    """Solve the minimum-total-cost linear assignment exactly."""
    if costs.ndim != 2 or costs.shape[0] < costs.shape[1]:
        fail("solve_hungarian_assignment", "contract", "INVALID_COST_MATRIX", f"shape={costs.shape}")
    try:
        row_ind, col_ind = linear_sum_assignment(costs)
    except ValueError:
        return None
    assignment = np.full(costs.shape[1], -1, dtype=int)
    assignment[col_ind] = row_ind
    if np.any(assignment < 0):
        return None
    if np.any(~np.isfinite(costs[assignment, np.arange(costs.shape[1])])):
        return None
    return assignment


def solve_sequential_greedy(costs: np.ndarray, task_order: np.ndarray) -> np.ndarray | None:
    """Assign tasks sequentially to the cheapest still-available robot."""
    robot_count, task_count = costs.shape
    if task_order.shape != (task_count,):
        fail(
            "solve_sequential_greedy",
            "contract",
            "TASK_ORDER_SHAPE_MISMATCH",
            f"expected={(task_count,)} actual={task_order.shape}",
        )
    if set(task_order.tolist()) != set(range(task_count)):
        fail("solve_sequential_greedy", "contract", "INVALID_TASK_ORDER", f"task_order={task_order.tolist()}")
    available = np.ones(robot_count, dtype=bool)
    assignment = np.full(task_count, -1, dtype=int)
    for task in task_order:
        candidates = np.flatnonzero(available & np.isfinite(costs[:, task]))
        if len(candidates) == 0:
            return None
        chosen = int(candidates[np.argmin(costs[candidates, task])])
        assignment[task] = chosen
        available[chosen] = False
    return assignment


def run_batched_auction_stage(
    receiver_costs: np.ndarray,
    feasible: np.ndarray,
    prices: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one epsilon stage, resetting assignments while preserving prices."""
    receiver_count, robot_count, task_count = receiver_costs.shape
    owner = np.full((receiver_count, robot_count), -1, dtype=int)
    assignment = np.full((receiver_count, task_count), -1, dtype=int)
    unassigned = np.ones((receiver_count, task_count), dtype=bool)
    unassigned[~feasible, :] = False

    iterations = 0
    while True:
        active = feasible & unassigned.any(axis=1)
        if not np.any(active):
            return assignment, prices

        receiver_indices = np.flatnonzero(active)
        task_indices = np.argmax(unassigned[receiver_indices], axis=1)
        task_costs = receiver_costs[receiver_indices, :, task_indices]
        net_values = -task_costs - prices[receiver_indices]

        best_robot = np.argmax(net_values, axis=1)
        rows = np.arange(len(receiver_indices))
        best_value = net_values[rows, best_robot]

        second_values = net_values.copy()
        second_values[rows, best_robot] = -np.inf
        second_value = np.max(second_values, axis=1)
        only_one_candidate = np.isneginf(second_value)
        second_value = np.where(only_one_candidate, best_value - 1.0, second_value)

        previous_task = owner[receiver_indices, best_robot].copy()
        unassigned[receiver_indices, task_indices] = False

        has_previous = previous_task >= 0
        if np.any(has_previous):
            previous_receivers = receiver_indices[has_previous]
            previous_tasks = previous_task[has_previous]
            unassigned[previous_receivers, previous_tasks] = True
            assignment[previous_receivers, previous_tasks] = -1

        prices[receiver_indices, best_robot] += best_value - second_value + epsilon
        owner[receiver_indices, best_robot] = task_indices
        assignment[receiver_indices, task_indices] = best_robot

        iterations += 1
        if iterations > AUCTION_MAX_STAGE_ITERATIONS:
            fail(
                "run_batched_auction_stage",
                "planning",
                "AUCTION_STAGE_DID_NOT_CONVERGE",
                (
                    f"tasks={task_count} epsilon={epsilon:.1e} "
                    f"active_receivers={int(active.sum())} iterations={iterations}"
                ),
            )


def solve_batched_auction_assignments(receiver_costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve receiver-local assignments with epsilon-scaling Bertsekas auction."""
    if receiver_costs.ndim != 3:
        fail(
            "solve_batched_auction_assignments",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    receiver_count, robot_count, task_count = receiver_costs.shape
    if robot_count < task_count:
        fail(
            "solve_batched_auction_assignments",
            "state",
            "CAPACITY_EXCEEDED",
            f"robots={robot_count} tasks={task_count}",
        )

    if task_count < robot_count:
        dummy_costs = np.zeros(
            (receiver_count, robot_count, robot_count - task_count),
            dtype=float,
        )
        auction_costs = np.concatenate((receiver_costs, dummy_costs), axis=2)
    else:
        auction_costs = receiver_costs

    finite = np.isfinite(auction_costs)
    feasible = finite.any(axis=1).all(axis=1)
    prices = np.zeros((receiver_count, robot_count), dtype=float)
    assignment = np.full((receiver_count, auction_costs.shape[2]), -1, dtype=int)

    for epsilon in AUCTION_EPSILON_LEVELS:
        assignment, prices = run_batched_auction_stage(
            auction_costs,
            feasible,
            prices,
            epsilon,
        )

    real_assignment = assignment[:, :task_count]
    valid = feasible & np.all(real_assignment >= 0, axis=1)
    return real_assignment, valid


def solve_local_optimizer_proposals(
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one full assignment proposal per receiver for one optimizer."""
    receiver_count, _, task_count = receiver_costs.shape
    if method == "p2p_auction":
        return solve_batched_auction_assignments(receiver_costs)

    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        local_costs = receiver_costs[receiver]
        if method == "p2p_greedy":
            proposal = solve_sequential_greedy(local_costs, task_order)
        elif method == "p2p_hungarian":
            proposal = solve_hungarian_assignment(local_costs)
        else:
            fail("solve_local_optimizer_proposals", "contract", "UNKNOWN_METHOD", f"method={method}")
        if proposal is None:
            continue
        proposals[receiver] = proposal
        valid[receiver] = True
    return proposals, valid


def build_assignment_support(proposals: np.ndarray, valid: np.ndarray, robot_count: int) -> np.ndarray:
    """Count receiver assignment proposals into robot x task support."""
    if proposals.ndim != 2:
        fail("build_assignment_support", "contract", "INVALID_PROPOSAL_SHAPE", f"shape={proposals.shape}")
    if valid.shape != (proposals.shape[0],):
        fail(
            "build_assignment_support",
            "contract",
            "VALID_MASK_SHAPE_MISMATCH",
            f"valid={valid.shape} proposals={proposals.shape}",
        )
    task_count = proposals.shape[1]
    valid_proposals = proposals[valid]
    if len(valid_proposals) == 0:
        fail("build_assignment_support", "planning", "NO_VALID_PROPOSALS", f"tasks={task_count}")
    if np.any((valid_proposals < 0) | (valid_proposals >= robot_count)):
        fail(
            "build_assignment_support",
            "state",
            "INVALID_PROPOSAL_ROBOT",
            "valid proposal contains out-of-range robot index",
        )
    support = np.zeros((robot_count, task_count), dtype=np.int16)
    tasks = np.arange(task_count)
    for proposal in valid_proposals:
        support[proposal, tasks] += 1
    return support


def solve_support_consensus(support: np.ndarray, tie_priority: np.ndarray) -> np.ndarray:
    """Choose one feasible team assignment maximizing proposal support."""
    if support.shape != tie_priority.shape:
        fail(
            "solve_support_consensus",
            "contract",
            "TIE_PRIORITY_SHAPE_MISMATCH",
            f"support={support.shape} tie_priority={tie_priority.shape}",
        )
    objective = -support.astype(float) + 1e-9 * tie_priority
    assignment = solve_hungarian_assignment(objective)
    if assignment is None:
        fail("solve_support_consensus", "planning", "CONSENSUS_INFEASIBLE", f"shape={support.shape}")
    return assignment


def optimizer_consensus_assignment(
    *,
    method: str,
    costs: np.ndarray,
    visibility: np.ndarray,
    task_order: np.ndarray,
    tie_priority: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run one P2P optimizer through local proposals and shared consensus."""
    receiver_costs = build_receiver_cost_views(costs, visibility)
    proposals, valid = solve_local_optimizer_proposals(method, receiver_costs, task_order)
    support = build_assignment_support(proposals, valid, costs.shape[0])
    assignment = solve_support_consensus(support, tie_priority)
    return assignment, 100.0 * float(valid.mean())


def validate_zero_loss_optimizer_contract(seed: int) -> None:
    """Reject any exact optimizer that misses oracle cost with complete data."""
    rng = np.random.default_rng(seed + 99173)
    for task_count in (1, 5, 50, 100):
        costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
        visibility = np.ones((ROBOT_COUNT, ROBOT_COUNT, task_count), dtype=bool)
        task_order = rng.permutation(task_count)
        tie_priority = rng.random((ROBOT_COUNT, task_count))
        oracle = solve_hungarian_assignment(costs)
        if oracle is None:
            fail("validate_zero_loss_optimizer_contract", "planning", "ORACLE_INFEASIBLE", f"tasks={task_count}")
        oracle_cost = assignment_total_cost(costs, oracle)
        exact_methods = ("p2p_hungarian", "p2p_auction")
        if task_count == 1:
            exact_methods += ("p2p_greedy",)
        for method in exact_methods:
            assignment, valid_rate = optimizer_consensus_assignment(
                method=method,
                costs=costs,
                visibility=visibility,
                task_order=task_order,
                tie_priority=tie_priority,
            )
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
            if valid_rate != 100.0:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_PROPOSAL_FAILURE",
                    f"method={method} tasks={task_count} valid_rate={valid_rate}",
                )
            if abs(gap_percent) > OPTIMAL_COST_TOLERANCE_PERCENT:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
                    f"method={method} tasks={task_count} gap_percent={gap_percent}",
                )


def evaluate_assignment(
    *,
    task_count: int,
    trial: int,
    method: str,
    packet_loss_rate: float,
    costs: np.ndarray,
    assignment: np.ndarray,
    optimal_assignment: np.ndarray,
    optimal_cost: float,
    valid_proposal_rate_percent: float,
) -> dict[str, object]:
    """Evaluate one final assignment against the full-information oracle."""
    total_cost = assignment_total_cost(costs, assignment)
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": ROBOT_COUNT,
        "packet_loss_percent": 100.0 * packet_loss_rate,
        "tasks": task_count,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap_percent,
        "optimal_cost_match": abs(gap_percent) <= OPTIMAL_COST_TOLERANCE_PERCENT,
        "near_optimal_5pct": gap_percent <= NEAR_OPTIMAL_GAP_PERCENT + 1e-12,
        "exact_optimal_assignment": bool(np.array_equal(assignment, optimal_assignment)),
        "valid_proposal_rate_percent": valid_proposal_rate_percent,
    }


def run_trial(
    *,
    task_count: int,
    trial: int,
    packet_loss_rate: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Run one paired trial for every real assignment optimizer/baseline."""
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
    visibility = sample_p2p_cost_visibility(ROBOT_COUNT, task_count, packet_loss_rate, rng)
    task_order = rng.permutation(task_count)
    tie_priority = rng.random((ROBOT_COUNT, task_count))

    optimal_assignment = solve_hungarian_assignment(costs)
    if optimal_assignment is None:
        fail("run_trial", "planning", "ORACLE_INFEASIBLE", f"tasks={task_count} trial={trial}")
    optimal_cost = assignment_total_cost(costs, optimal_assignment)

    records = [
        evaluate_assignment(
            task_count=task_count,
            trial=trial,
            method="oracle",
            packet_loss_rate=packet_loss_rate,
            costs=costs,
            assignment=optimal_assignment,
            optimal_assignment=optimal_assignment,
            optimal_cost=optimal_cost,
            valid_proposal_rate_percent=100.0,
        )
    ]
    for method in ("p2p_greedy", "p2p_hungarian", "p2p_auction"):
        assignment, valid_rate = optimizer_consensus_assignment(
            method=method,
            costs=costs,
            visibility=visibility,
            task_order=task_order,
            tie_priority=tie_priority,
        )
        records.append(
            evaluate_assignment(
                task_count=task_count,
                trial=trial,
                method=method,
                packet_loss_rate=packet_loss_rate,
                costs=costs,
                assignment=assignment,
                optimal_assignment=optimal_assignment,
                optimal_cost=optimal_cost,
                valid_proposal_rate_percent=valid_rate,
            )
        )
    return records


def run_experiment(
    *,
    task_counts: tuple[int, ...] = TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the paired 100-robot task-count sweep."""
    validate_experiment_config(ROBOT_COUNT, packet_loss_rate, task_counts, trials)
    validate_zero_loss_optimizer_contract(seed)
    print(
        "Zero-loss optimizer contract: PASS "
        "(Hungarian/Auction match oracle; single-task Greedy matches oracle)"
    )

    records: list[dict[str, object]] = []
    for task_count in task_counts:
        rng = np.random.default_rng(seed + task_count * 100003)
        for trial in range(1, trials + 1):
            records.extend(
                run_trial(
                    task_count=task_count,
                    trial=trial,
                    packet_loss_rate=packet_loss_rate,
                    rng=rng,
                )
            )
        print(f"tasks={task_count:3d}/{max(task_counts)} complete")

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["tasks", "method", "method_label"], as_index=False)
        .agg(
            average_total_cost=("total_cost", "mean"),
            average_optimality_gap_percent=("optimality_gap_percent", "mean"),
            optimal_cost_match_percent=("optimal_cost_match", lambda series: 100.0 * series.mean()),
            near_optimal_5pct_percent=("near_optimal_5pct", lambda series: 100.0 * series.mean()),
            exact_optimal_assignment_percent=("exact_optimal_assignment", lambda series: 100.0 * series.mean()),
            average_valid_proposal_rate_percent=("valid_proposal_rate_percent", "mean"),
        )
        .sort_values(["tasks", "method"])
        .reset_index(drop=True)
    )
    return raw, summary


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = summary.pivot(index="tasks", columns="method_label", values=metric)
    ordered_labels = [METHOD_LABELS[method] for method in METHODS]
    return table.reindex(columns=ordered_labels).reset_index()


def save_report_tables(summary: pd.DataFrame) -> None:
    report_table(summary, "average_optimality_gap_percent").to_csv(
        DATA_DIR / "report_average_optimality_gap_percent.csv", index=False
    )
    report_table(summary, "optimal_cost_match_percent").to_csv(
        DATA_DIR / "report_optimal_cost_match_percent.csv", index=False
    )
    report_table(summary, "near_optimal_5pct_percent").to_csv(
        DATA_DIR / "report_near_optimal_5pct_percent.csv", index=False
    )
    report_table(summary, "exact_optimal_assignment_percent").to_csv(
        DATA_DIR / "report_exact_optimal_assignment_percent.csv", index=False
    )
    report_table(summary, "average_valid_proposal_rate_percent").to_csv(
        DATA_DIR / "report_valid_proposal_rate_percent.csv", index=False
    )


def save_metric_plot(summary: pd.DataFrame, *, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        part = summary[summary["method"] == method].sort_values("tasks")
        ax.plot(part["tasks"], part[metric], marker="o", label=METHOD_LABELS[method])
    ax.set_xlabel("Simultaneous task count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "optimizer_comparison_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "optimizer_comparison_summary.csv", index=False)
    save_report_tables(summary)
    save_metric_plot(
        summary,
        metric="average_optimality_gap_percent",
        ylabel="Average optimality gap (%)",
        filename="average_optimality_gap_percent.png",
    )
    save_metric_plot(
        summary,
        metric="optimal_cost_match_percent",
        ylabel="Optimal-cost match rate (%)",
        filename="optimal_cost_match_percent.png",
    )
    save_metric_plot(
        summary,
        metric="near_optimal_5pct_percent",
        ylabel="Near-optimal within 5% (%)",
        filename="near_optimal_5pct_percent.png",
    )


def parse_task_counts(values: list[int] | None) -> tuple[int, ...]:
    if values is None:
        return TASK_COUNTS
    parsed = tuple(values)
    if not parsed:
        raise argparse.ArgumentTypeError("task counts cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare real multi-task assignment optimizers with 100 robots under paired "
            "directed P2P cost loss. Hungarian is the full-information oracle; P2P "
            "Hungarian and Auction optimize each receiver's incomplete assignment matrix; "
            "P2P Greedy is the heuristic baseline."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=None,
        help="Task counts. Default: 5 10 20 30 40 50 60 70 80 90 100",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=PACKET_LOSS_RATE,
        help="Directed P2P loss probability (default: 0.30).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_counts = parse_task_counts(args.tasks)
    raw, summary = run_experiment(
        task_counts=task_counts,
        trials=args.trials,
        packet_loss_rate=args.packet_loss,
        seed=args.seed,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "optimizer_comparison_raw.csv")
    print(DATA_DIR / "optimizer_comparison_summary.csv")
    print(FIGURE_DIR)

    print("\nAverage optimality gap (%) - lower is better:")
    print(report_table(summary, "average_optimality_gap_percent").to_string(index=False))

    print("\nOptimal-cost match (% of trials) - higher is better:")
    print(report_table(summary, "optimal_cost_match_percent").to_string(index=False))

    print("\nNear-optimal within 5% (% of trials) - higher is better:")
    print(report_table(summary, "near_optimal_5pct_percent").to_string(index=False))

    print("\nValid local optimizer proposals (%) - diagnostic:")
    print(report_table(summary, "average_valid_proposal_rate_percent").to_string(index=False))


if __name__ == "__main__":
    main()
