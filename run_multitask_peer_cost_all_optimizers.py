from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_multitask_peer_cost_experiment import (
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    PACKET_LOSS_RATE,
    RANDOM_SEED,
    build_assignment_support,
    generate_spatial_cost_matrix,
    solve_hungarian_assignment,
    solve_local_optimizer_proposals,
    solve_support_consensus,
)
from run_multitask_workload_heuristics import solve_capacitated_heuristic_batch
from run_multitask_workload_optimizers import (
    MIN_COST_FLOW_METHOD,
    MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT,
    SINKHORN_METHOD,
    solve_min_cost_flow_batch,
    solve_sinkhorn_batch,
)

FIXED_ROBOT_COUNT = 100
WORKLOAD_TASK_COUNTS = tuple(range(100, 1001, 100))
MAX_TASK_COUNT = 1000
FORMAL_TRIALS = 20
GREEDY_METHOD = "p2p_sequential_greedy"
HUNGARIAN_METHOD = "p2p_hungarian"
DEFAULT_VOTING_METHODS = (
    GREEDY_METHOD,
    HUNGARIAN_METHOD,
    MIN_COST_FLOW_METHOD,
    SINKHORN_METHOD,
)
REPORT_METHOD_ORDER = ("oracle",) + DEFAULT_VOTING_METHODS
METHOD_LABELS = {
    "oracle": "Hungarian Oracle",
    GREEDY_METHOD: "Voting Greedy",
    HUNGARIAN_METHOD: "Voting Hungarian",
    MIN_COST_FLOW_METHOD: "Voting Min-Cost Flow",
    SINKHORN_METHOD: "Voting Sinkhorn + Rounding",
}
DEFAULT_VOTER_BATCH_SIZE = 4
DEFAULT_PROGRESS_EVERY_VOTERS = 25
DEFAULT_PARALLEL_WORKERS = max(1, min(4, os.cpu_count() or 1))
VOTER_SELECTION_SEED_OFFSET = 2_000_003
VISIBILITY_SEED_OFFSET = 4_000_007
GREEDY_ZERO_LOSS_CHECK_MAX_TASKS = 200
HUNGARIAN_ZERO_LOSS_CHECK_MAX_TASKS = 200
MIN_COST_FLOW_ZERO_LOSS_CHECK_MAX_TASKS = 200
SINKHORN_ZERO_LOSS_CHECK_MAX_TASKS = 200

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_peer_cost_fixed100_workload"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass(frozen=True)
class TrialJob:
    task_count: int
    trial: int
    packet_loss_rate: float
    seed: int
    max_voters: int | None
    voter_batch_size: int
    progress_every_voters: int


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first Experiment 2 owner/function boundary."""
    raise ValueError(
        "owner=run_multitask_peer_cost_all_optimizers "
        f"function={function} category={category} code={code} details={details}"
    )


def cost_match_tolerance_percent(method: str) -> float:
    """Return the objective-match tolerance owned by each report method."""
    if method == MIN_COST_FLOW_METHOD:
        return MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT
    if method in REPORT_METHOD_ORDER:
        return OPTIMAL_COST_TOLERANCE_PERCENT
    fail("cost_match_tolerance_percent", "contract", "UNKNOWN_METHOD", f"method={method}")


def resolve_voting_methods() -> tuple[str, ...]:
    """Return the canonical diverse fast Voting optimizer set."""
    return DEFAULT_VOTING_METHODS


def validate_workload_config(
    *,
    task_counts: tuple[int, ...],
    trials: int,
    packet_loss_rate: float,
    max_voters: int | None,
    voter_batch_size: int,
    progress_every_voters: int,
    workers: int,
) -> None:
    """Validate fixed-fleet workload, sampling, batching, and parallel runtime controls."""
    if not task_counts:
        fail("validate_workload_config", "contract", "EMPTY_TASK_COUNTS", "task_counts is empty")
    if any(task_count <= 0 for task_count in task_counts):
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_TASK_COUNT",
            f"expected>=1 actual={task_counts}",
        )
    if any(task_count > MAX_TASK_COUNT for task_count in task_counts):
        fail(
            "validate_workload_config",
            "contract",
            "TASK_COUNT_ABOVE_REPORT_BOUNDARY",
            f"expected<={MAX_TASK_COUNT} actual={task_counts}",
        )
    if trials <= 0:
        fail("validate_workload_config", "contract", "INVALID_TRIALS", f"actual={trials}")
    if not 0.0 <= packet_loss_rate < 1.0:
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_PACKET_LOSS",
            f"actual={packet_loss_rate}",
        )
    if max_voters is not None and max_voters <= 0:
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_MAX_VOTERS",
            f"expected>=1 actual={max_voters}",
        )
    if voter_batch_size <= 0:
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_VOTER_BATCH_SIZE",
            f"expected>=1 actual={voter_batch_size}",
        )
    if progress_every_voters <= 0:
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_PROGRESS_INTERVAL",
            f"expected>=1 actual={progress_every_voters}",
        )
    if workers <= 0:
        fail(
            "validate_workload_config",
            "contract",
            "INVALID_WORKER_COUNT",
            f"expected>=1 actual={workers}",
        )


def resolve_robot_capacity(task_count: int, robot_count: int = FIXED_ROBOT_COUNT) -> int:
    """Return the smallest uniform per-robot batch capacity that can hold all tasks."""
    if robot_count <= 0 or task_count <= 0:
        fail(
            "resolve_robot_capacity",
            "contract",
            "INVALID_CAPACITY_INPUT",
            f"robots={robot_count} tasks={task_count}",
        )
    return (task_count + robot_count - 1) // robot_count


def build_capacity_slot_cost_matrix(costs: np.ndarray, capacity_per_robot: int) -> np.ndarray:
    """Represent uniform physical-robot capacity as repeated capacity-one assignment slots."""
    if costs.ndim != 2:
        fail(
            "build_capacity_slot_cost_matrix",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "build_capacity_slot_cost_matrix",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    return np.repeat(costs, capacity_per_robot, axis=0)


def build_capacity_slot_cost_views(
    receiver_costs: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray:
    """Expand receiver-local physical robot rows for the capacity-one Hungarian owner."""
    if receiver_costs.ndim != 3:
        fail(
            "build_capacity_slot_cost_views",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "build_capacity_slot_cost_views",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    return np.repeat(receiver_costs, capacity_per_robot, axis=1)


def map_slot_assignments_to_robots(
    slot_assignments: np.ndarray,
    *,
    robot_count: int,
    capacity_per_robot: int,
) -> np.ndarray:
    """Map capacity-slot indices back to physical robot IDs without repairing invalid rows."""
    if slot_assignments.ndim not in (1, 2):
        fail(
            "map_slot_assignments_to_robots",
            "contract",
            "INVALID_SLOT_ASSIGNMENT_SHAPE",
            f"shape={slot_assignments.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "map_slot_assignments_to_robots",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    slot_count = robot_count * capacity_per_robot
    if np.any(slot_assignments < -1) or np.any(slot_assignments >= slot_count):
        fail(
            "map_slot_assignments_to_robots",
            "state",
            "SLOT_INDEX_OUT_OF_RANGE",
            f"slot_count={slot_count}",
        )
    return np.where(slot_assignments >= 0, slot_assignments // capacity_per_robot, -1).astype(int)


def validate_capacity_assignment(
    *,
    costs: np.ndarray,
    assignment: np.ndarray,
    capacity_per_robot: int,
) -> None:
    """Validate one physical assignment under the fixed-fleet batch-capacity contract."""
    if costs.ndim != 2:
        fail(
            "validate_capacity_assignment",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    task_count = costs.shape[1]
    if assignment.shape != (task_count,):
        fail(
            "validate_capacity_assignment",
            "contract",
            "ASSIGNMENT_SHAPE_MISMATCH",
            f"expected={(task_count,)} actual={assignment.shape}",
        )
    if np.any(assignment < 0) or np.any(assignment >= costs.shape[0]):
        fail(
            "validate_capacity_assignment",
            "state",
            "INVALID_ROBOT_INDEX",
            f"robots={costs.shape[0]}",
        )
    counts = np.bincount(assignment, minlength=costs.shape[0])
    if np.any(counts > capacity_per_robot):
        fail(
            "validate_capacity_assignment",
            "state",
            "CAPACITY_VIOLATION",
            (
                f"capacity_per_robot={capacity_per_robot} "
                f"actual_max={int(counts.max(initial=0))}"
            ),
        )
    selected = costs[assignment, np.arange(task_count)]
    if np.any(~np.isfinite(selected)):
        fail(
            "validate_capacity_assignment",
            "planning",
            "INFEASIBLE_EDGE_SELECTED",
            "assignment contains an unavailable cost edge",
        )


def assignment_total_cost_with_capacity(
    costs: np.ndarray,
    assignment: np.ndarray,
    capacity_per_robot: int,
) -> float:
    """Return true total cost after capacity validation."""
    validate_capacity_assignment(
        costs=costs,
        assignment=assignment,
        capacity_per_robot=capacity_per_robot,
    )
    return float(costs[assignment, np.arange(costs.shape[1])].sum())


def solve_capacity_oracle(costs: np.ndarray, capacity_per_robot: int) -> np.ndarray | None:
    """Solve the full-information capacitated minimum via the existing Hungarian owner."""
    slot_costs = build_capacity_slot_cost_matrix(costs, capacity_per_robot)
    slot_assignment = solve_hungarian_assignment(slot_costs)
    if slot_assignment is None:
        return None
    return map_slot_assignments_to_robots(
        slot_assignment,
        robot_count=costs.shape[0],
        capacity_per_robot=capacity_per_robot,
    )


def resolve_voter_count(robot_count: int, max_voters: int | None) -> int:
    """Use all physical robots as voters unless a preview explicitly caps receivers."""
    if max_voters is None:
        return robot_count
    return min(robot_count, max_voters)


def select_voter_indices(
    *,
    robot_count: int,
    voter_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select physical receiver IDs whose local proposals participate in Voting."""
    if voter_count <= 0 or voter_count > robot_count:
        fail(
            "select_voter_indices",
            "contract",
            "INVALID_VOTER_COUNT",
            f"robots={robot_count} voters={voter_count}",
        )
    if voter_count == robot_count:
        return np.arange(robot_count, dtype=int)
    return np.sort(rng.choice(robot_count, size=voter_count, replace=False).astype(int))


def sample_voter_batch_visibility(
    *,
    robot_count: int,
    task_count: int,
    packet_loss_rate: float,
    receiver_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample directed sender-robot to receiver-robot scalar-cost visibility."""
    if receiver_indices.ndim != 1:
        fail(
            "sample_voter_batch_visibility",
            "contract",
            "INVALID_RECEIVER_INDEX_SHAPE",
            f"shape={receiver_indices.shape}",
        )
    if np.any(receiver_indices < 0) or np.any(receiver_indices >= robot_count):
        fail(
            "sample_voter_batch_visibility",
            "state",
            "RECEIVER_INDEX_OUT_OF_RANGE",
            f"robots={robot_count} receivers={receiver_indices.tolist()}",
        )
    visible = rng.random((len(receiver_indices), robot_count, task_count)) >= packet_loss_rate
    local_rows = np.arange(len(receiver_indices))
    visible[local_rows, receiver_indices, :] = True
    return visible


def build_voter_batch_cost_views(costs: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Materialize only the current receiver batch's incomplete physical cost matrices."""
    if costs.ndim != 2:
        fail(
            "build_voter_batch_cost_views",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    if visibility.ndim != 3 or visibility.shape[1:] != costs.shape:
        fail(
            "build_voter_batch_cost_views",
            "contract",
            "VISIBILITY_SHAPE_MISMATCH",
            f"expected=(*,{costs.shape[0]},{costs.shape[1]}) actual={visibility.shape}",
        )
    return np.where(visibility, costs[None, :, :], np.inf)


def solve_voter_batch_proposals(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Route one receiver batch to the true owner of each canonical optimizer family."""
    if method == GREEDY_METHOD:
        return solve_capacitated_heuristic_batch(
            method=GREEDY_METHOD,
            receiver_costs=receiver_costs,
            task_order=task_order,
            capacity_per_robot=capacity_per_robot,
        )
    if method == HUNGARIAN_METHOD:
        slot_receiver_costs = build_capacity_slot_cost_views(
            receiver_costs,
            capacity_per_robot,
        )
        slot_proposals, valid = solve_local_optimizer_proposals(
            HUNGARIAN_METHOD,
            slot_receiver_costs,
            task_order,
        )
        proposals = map_slot_assignments_to_robots(
            slot_proposals,
            robot_count=receiver_costs.shape[1],
            capacity_per_robot=capacity_per_robot,
        )
        return proposals, valid
    if method == MIN_COST_FLOW_METHOD:
        return solve_min_cost_flow_batch(
            receiver_costs=receiver_costs,
            capacity_per_robot=capacity_per_robot,
        )
    if method == SINKHORN_METHOD:
        return solve_sinkhorn_batch(
            receiver_costs=receiver_costs,
            capacity_per_robot=capacity_per_robot,
        )
    fail("solve_voter_batch_proposals", "contract", "UNKNOWN_METHOD", f"method={method}")


def accumulate_proposal_support(
    *,
    support: np.ndarray,
    proposals: np.ndarray,
    valid: np.ndarray,
    robot_count: int,
) -> int:
    """Accumulate physical robot/task support for one voter batch."""
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0
    batch_support = build_assignment_support(proposals, valid, robot_count)
    support += batch_support.astype(support.dtype, copy=False)
    return valid_count


def report_voter_progress(
    *,
    task_count: int,
    trial: int,
    completed: int,
    total: int,
) -> None:
    """Report receiver progress only in single-worker diagnostic mode."""
    end = "\n" if completed >= total else "\r"
    print(
        f"robots={FIXED_ROBOT_COUNT:3d} tasks={task_count:4d} "
        f"trial={trial:3d} voters={completed:3d}/{total}",
        end=end,
        flush=True,
    )


def collect_voting_support(
    *,
    costs: np.ndarray,
    capacity_per_robot: int,
    packet_loss_rate: float,
    voter_indices: np.ndarray,
    task_order: np.ndarray,
    visibility_rng: np.random.Generator,
    voter_batch_size: int,
    progress_every_voters: int,
    task_count: int,
    trial: int,
    voting_methods: tuple[str, ...],
    emit_progress: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Stream physical receiver views and accumulate proposal support for each method."""
    robot_count = costs.shape[0]
    support_by_method = {
        method: np.zeros((robot_count, task_count), dtype=np.int32)
        for method in voting_methods
    }
    valid_counts = {method: 0 for method in voting_methods}

    next_progress = progress_every_voters
    total_voters = len(voter_indices)
    for start in range(0, total_voters, voter_batch_size):
        receiver_batch = voter_indices[start : start + voter_batch_size]
        visibility = sample_voter_batch_visibility(
            robot_count=robot_count,
            task_count=task_count,
            packet_loss_rate=packet_loss_rate,
            receiver_indices=receiver_batch,
            rng=visibility_rng,
        )
        receiver_costs = build_voter_batch_cost_views(costs, visibility)

        for method in voting_methods:
            proposals, valid = solve_voter_batch_proposals(
                method=method,
                receiver_costs=receiver_costs,
                task_order=task_order,
                capacity_per_robot=capacity_per_robot,
            )
            valid_counts[method] += accumulate_proposal_support(
                support=support_by_method[method],
                proposals=proposals,
                valid=valid,
                robot_count=robot_count,
            )

        completed = min(start + len(receiver_batch), total_voters)
        if emit_progress and (completed >= next_progress or completed == total_voters):
            report_voter_progress(
                task_count=task_count,
                trial=trial,
                completed=completed,
                total=total_voters,
            )
            while next_progress <= completed:
                next_progress += progress_every_voters

    return support_by_method, valid_counts


def build_capacity_slot_consensus_inputs(
    *,
    support: np.ndarray,
    tie_priority: np.ndarray,
    capacity_per_robot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand physical support/tie rows into capacity-one consensus slots."""
    if support.shape != tie_priority.shape:
        fail(
            "build_capacity_slot_consensus_inputs",
            "contract",
            "TIE_PRIORITY_SHAPE_MISMATCH",
            f"support={support.shape} tie_priority={tie_priority.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "build_capacity_slot_consensus_inputs",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    return (
        np.repeat(support, capacity_per_robot, axis=0),
        np.repeat(tie_priority, capacity_per_robot, axis=0),
    )


def solve_capacity_support_consensus(
    *,
    support: np.ndarray,
    tie_priority: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray:
    """Run the existing support-consensus owner on slots, then map back to physical robots."""
    slot_support, slot_tie_priority = build_capacity_slot_consensus_inputs(
        support=support,
        tie_priority=tie_priority,
        capacity_per_robot=capacity_per_robot,
    )
    slot_assignment = solve_support_consensus(slot_support, slot_tie_priority)
    return map_slot_assignments_to_robots(
        slot_assignment,
        robot_count=support.shape[0],
        capacity_per_robot=capacity_per_robot,
    )


def finalize_voting_assignments(
    *,
    support_by_method: dict[str, np.ndarray],
    valid_counts: dict[str, int],
    voter_count: int,
    tie_priority: np.ndarray,
    capacity_per_robot: int,
    task_count: int,
    trial: int,
    voting_methods: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, float]]:
    """Convert proposal support into one capacity-feasible physical assignment per method."""
    results: dict[str, tuple[np.ndarray, float]] = {}
    for method in voting_methods:
        valid_count = valid_counts[method]
        if valid_count == 0:
            fail(
                "finalize_voting_assignments",
                "planning",
                "NO_VALID_PROPOSALS",
                f"method={method} tasks={task_count} trial={trial} voters={voter_count}",
            )
        assignment = solve_capacity_support_consensus(
            support=support_by_method[method],
            tie_priority=tie_priority,
            capacity_per_robot=capacity_per_robot,
        )
        results[method] = (assignment, 100.0 * valid_count / voter_count)
    return results


def solve_zero_loss_consensus(
    *,
    method: str,
    costs: np.ndarray,
    capacity_per_robot: int,
    task_order: np.ndarray,
    tie_priority: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pass one complete-information proposal through the same Voting consensus boundary."""
    proposals, valid = solve_voter_batch_proposals(
        method=method,
        receiver_costs=costs[None, :, :],
        task_order=task_order,
        capacity_per_robot=capacity_per_robot,
    )
    if not bool(valid[0]):
        fail(
            "solve_zero_loss_consensus",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"method={method} shape={costs.shape} capacity={capacity_per_robot}",
        )
    support = build_assignment_support(proposals, valid, costs.shape[0])
    assignment = solve_capacity_support_consensus(
        support=support,
        tie_priority=tie_priority,
        capacity_per_robot=capacity_per_robot,
    )
    return assignment, 100.0


def build_zero_loss_case(seed: int, task_count: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Build one deterministic complete-information integration case shared by preflight owners."""
    rng = np.random.default_rng(seed + task_count * 100_003)
    costs = generate_spatial_cost_matrix(FIXED_ROBOT_COUNT, task_count, rng)
    capacity_per_robot = resolve_robot_capacity(task_count)
    task_order = rng.permutation(task_count)
    tie_priority = rng.random((FIXED_ROBOT_COUNT, task_count))
    return costs, capacity_per_robot, task_order, tie_priority


def validate_zero_loss_greedy_contract(seed: int, max_task_count: int) -> None:
    """Require the single Greedy baseline to return valid capacity-feasible proposals."""
    task_count = min(GREEDY_ZERO_LOSS_CHECK_MAX_TASKS, max_task_count)
    if task_count <= 0:
        return
    costs, capacity_per_robot, task_order, tie_priority = build_zero_loss_case(
        seed + 191_173,
        task_count,
    )
    assignment, valid_rate = solve_zero_loss_consensus(
        method=GREEDY_METHOD,
        costs=costs,
        capacity_per_robot=capacity_per_robot,
        task_order=task_order,
        tie_priority=tie_priority,
    )
    if valid_rate != 100.0:
        fail(
            "validate_zero_loss_greedy_contract",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"tasks={task_count} valid_rate={valid_rate}",
        )
    validate_capacity_assignment(
        costs=costs,
        assignment=assignment,
        capacity_per_robot=capacity_per_robot,
    )


def validate_zero_loss_hungarian_contract(seed: int, max_task_count: int) -> None:
    """Require Voting Hungarian to match the capacitated full-information Oracle."""
    task_count = min(HUNGARIAN_ZERO_LOSS_CHECK_MAX_TASKS, max_task_count)
    if task_count <= 0:
        return
    costs, capacity_per_robot, task_order, tie_priority = build_zero_loss_case(
        seed + 291_173,
        task_count,
    )
    oracle = solve_capacity_oracle(costs, capacity_per_robot)
    if oracle is None:
        fail(
            "validate_zero_loss_hungarian_contract",
            "planning",
            "ORACLE_INFEASIBLE",
            f"tasks={task_count} capacity={capacity_per_robot}",
        )
    oracle_cost = assignment_total_cost_with_capacity(costs, oracle, capacity_per_robot)
    assignment, valid_rate = solve_zero_loss_consensus(
        method=HUNGARIAN_METHOD,
        costs=costs,
        capacity_per_robot=capacity_per_robot,
        task_order=task_order,
        tie_priority=tie_priority,
    )
    if valid_rate != 100.0:
        fail(
            "validate_zero_loss_hungarian_contract",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"tasks={task_count} valid_rate={valid_rate}",
        )
    actual_cost = assignment_total_cost_with_capacity(costs, assignment, capacity_per_robot)
    gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
    if abs(gap_percent) > OPTIMAL_COST_TOLERANCE_PERCENT:
        fail(
            "validate_zero_loss_hungarian_contract",
            "planning",
            "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
            f"method={HUNGARIAN_METHOD} tasks={task_count} actual_gap_percent={gap_percent}",
        )


def validate_zero_loss_min_cost_flow_contract(seed: int, max_task_count: int) -> None:
    """Require scaled Min-Cost Flow to match the float Oracle within quantization tolerance."""
    task_count = min(MIN_COST_FLOW_ZERO_LOSS_CHECK_MAX_TASKS, max_task_count)
    if task_count <= 0:
        return
    costs, capacity_per_robot, task_order, tie_priority = build_zero_loss_case(
        seed + 391_173,
        task_count,
    )
    oracle = solve_capacity_oracle(costs, capacity_per_robot)
    if oracle is None:
        fail(
            "validate_zero_loss_min_cost_flow_contract",
            "planning",
            "ORACLE_INFEASIBLE",
            f"tasks={task_count} capacity={capacity_per_robot}",
        )
    oracle_cost = assignment_total_cost_with_capacity(costs, oracle, capacity_per_robot)
    assignment, valid_rate = solve_zero_loss_consensus(
        method=MIN_COST_FLOW_METHOD,
        costs=costs,
        capacity_per_robot=capacity_per_robot,
        task_order=task_order,
        tie_priority=tie_priority,
    )
    if valid_rate != 100.0:
        fail(
            "validate_zero_loss_min_cost_flow_contract",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"tasks={task_count} valid_rate={valid_rate}",
        )
    actual_cost = assignment_total_cost_with_capacity(costs, assignment, capacity_per_robot)
    gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
    if abs(gap_percent) > MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT:
        fail(
            "validate_zero_loss_min_cost_flow_contract",
            "planning",
            "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
            (
                f"method={MIN_COST_FLOW_METHOD} tasks={task_count} "
                f"expected_abs_gap_percent<={MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT} "
                f"actual={gap_percent}"
            ),
        )


def validate_zero_loss_sinkhorn_contract(seed: int, max_task_count: int) -> None:
    """Require Sinkhorn plus explicit rounding to return a valid capacity-feasible proposal."""
    task_count = min(SINKHORN_ZERO_LOSS_CHECK_MAX_TASKS, max_task_count)
    if task_count <= 0:
        return
    costs, capacity_per_robot, task_order, tie_priority = build_zero_loss_case(
        seed + 491_173,
        task_count,
    )
    assignment, valid_rate = solve_zero_loss_consensus(
        method=SINKHORN_METHOD,
        costs=costs,
        capacity_per_robot=capacity_per_robot,
        task_order=task_order,
        tie_priority=tie_priority,
    )
    if valid_rate != 100.0:
        fail(
            "validate_zero_loss_sinkhorn_contract",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"tasks={task_count} valid_rate={valid_rate}",
        )
    validate_capacity_assignment(
        costs=costs,
        assignment=assignment,
        capacity_per_robot=capacity_per_robot,
    )


def validate_zero_loss_optimizer_contract(seed: int, max_task_count: int) -> None:
    """Run bounded integration gates for each canonical optimizer family."""
    validate_zero_loss_greedy_contract(seed, max_task_count)
    validate_zero_loss_hungarian_contract(seed, max_task_count)
    validate_zero_loss_min_cost_flow_contract(seed, max_task_count)
    validate_zero_loss_sinkhorn_contract(seed, max_task_count)


def evaluate_assignment(
    *,
    robot_count: int,
    voter_count: int,
    capacity_per_robot: int,
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
    """Evaluate one fixed-fleet workload assignment against the capacitated Oracle."""
    total_cost = assignment_total_cost_with_capacity(
        costs,
        assignment,
        capacity_per_robot,
    )
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": robot_count,
        "voters": voter_count,
        "tasks": task_count,
        "capacity_per_robot": capacity_per_robot,
        "assignment_slots": robot_count * capacity_per_robot,
        "packet_loss_percent": 100.0 * packet_loss_rate,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "total_cost": total_cost,
        "optimal_total_cost": optimal_cost,
        "optimality_gap_percent": gap_percent,
        "optimal_cost_match": abs(gap_percent) <= cost_match_tolerance_percent(method),
        "near_optimal_5pct": gap_percent <= NEAR_OPTIMAL_GAP_PERCENT + 1e-12,
        "exact_optimal_assignment": bool(np.array_equal(assignment, optimal_assignment)),
        "valid_proposal_rate_percent": valid_proposal_rate_percent,
    }


def run_trial(
    *,
    task_count: int,
    trial: int,
    packet_loss_rate: float,
    seed: int,
    max_voters: int | None,
    voter_batch_size: int,
    progress_every_voters: int,
    voting_methods: tuple[str, ...],
    emit_voter_progress: bool,
) -> list[dict[str, object]]:
    """Run one paired workload trial with deterministic scenario and communication streams."""
    robot_count = FIXED_ROBOT_COUNT
    capacity_per_robot = resolve_robot_capacity(task_count, robot_count)
    voter_count = resolve_voter_count(robot_count, max_voters)
    trial_seed = seed + task_count * 100_003 + trial * 1_009

    scenario_rng = np.random.default_rng(trial_seed)
    costs = generate_spatial_cost_matrix(robot_count, task_count, scenario_rng)
    task_order = scenario_rng.permutation(task_count)
    tie_priority = scenario_rng.random((robot_count, task_count))

    optimal_assignment = solve_capacity_oracle(costs, capacity_per_robot)
    if optimal_assignment is None:
        fail(
            "run_trial",
            "planning",
            "ORACLE_INFEASIBLE",
            (
                f"robots={robot_count} tasks={task_count} "
                f"capacity={capacity_per_robot} trial={trial}"
            ),
        )
    optimal_cost = assignment_total_cost_with_capacity(
        costs,
        optimal_assignment,
        capacity_per_robot,
    )

    voter_rng = np.random.default_rng(trial_seed + VOTER_SELECTION_SEED_OFFSET)
    voter_indices = select_voter_indices(
        robot_count=robot_count,
        voter_count=voter_count,
        rng=voter_rng,
    )
    visibility_rng = np.random.default_rng(trial_seed + VISIBILITY_SEED_OFFSET)
    support_by_method, valid_counts = collect_voting_support(
        costs=costs,
        capacity_per_robot=capacity_per_robot,
        packet_loss_rate=packet_loss_rate,
        voter_indices=voter_indices,
        task_order=task_order,
        visibility_rng=visibility_rng,
        voter_batch_size=voter_batch_size,
        progress_every_voters=progress_every_voters,
        task_count=task_count,
        trial=trial,
        voting_methods=voting_methods,
        emit_progress=emit_voter_progress,
    )
    voting_results = finalize_voting_assignments(
        support_by_method=support_by_method,
        valid_counts=valid_counts,
        voter_count=voter_count,
        tie_priority=tie_priority,
        capacity_per_robot=capacity_per_robot,
        task_count=task_count,
        trial=trial,
        voting_methods=voting_methods,
    )

    records = [
        evaluate_assignment(
            robot_count=robot_count,
            voter_count=voter_count,
            capacity_per_robot=capacity_per_robot,
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
    for method in voting_methods:
        assignment, valid_rate = voting_results[method]
        records.append(
            evaluate_assignment(
                robot_count=robot_count,
                voter_count=voter_count,
                capacity_per_robot=capacity_per_robot,
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


def run_trial_job(job: TrialJob) -> list[dict[str, object]]:
    """Execute one independent trial inside a process worker without console contention."""
    return run_trial(
        task_count=job.task_count,
        trial=job.trial,
        packet_loss_rate=job.packet_loss_rate,
        seed=job.seed,
        max_voters=job.max_voters,
        voter_batch_size=job.voter_batch_size,
        progress_every_voters=job.progress_every_voters,
        voting_methods=resolve_voting_methods(),
        emit_voter_progress=False,
    )


def build_trial_jobs(
    *,
    task_counts: tuple[int, ...],
    trials: int,
    packet_loss_rate: float,
    seed: int,
    max_voters: int | None,
    voter_batch_size: int,
    progress_every_voters: int,
) -> list[TrialJob]:
    """Build deterministic independent trial jobs in report order."""
    return [
        TrialJob(
            task_count=task_count,
            trial=trial,
            packet_loss_rate=packet_loss_rate,
            seed=seed,
            max_voters=max_voters,
            voter_batch_size=voter_batch_size,
            progress_every_voters=progress_every_voters,
        )
        for task_count in task_counts
        for trial in range(1, trials + 1)
    ]


def configure_worker_thread_environment(workers: int) -> None:
    """Avoid nested BLAS/OpenMP oversubscription when process-level parallelism is enabled."""
    if workers <= 1:
        return
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


def report_trial_completion(index: int, total: int, job: TrialJob) -> None:
    """Report parent-owned trial completion without interleaving worker output."""
    print(
        f"completed={index:4d}/{total} robots={FIXED_ROBOT_COUNT:3d} "
        f"tasks={job.task_count:4d} trial={job.trial:3d}",
        flush=True,
    )


def execute_trial_jobs(
    jobs: list[TrialJob],
    *,
    workers: int,
) -> list[dict[str, object]]:
    """Execute independent trial jobs sequentially or through a bounded process pool."""
    records: list[dict[str, object]] = []
    total = len(jobs)
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            records.extend(
                run_trial(
                    task_count=job.task_count,
                    trial=job.trial,
                    packet_loss_rate=job.packet_loss_rate,
                    seed=job.seed,
                    max_voters=job.max_voters,
                    voter_batch_size=job.voter_batch_size,
                    progress_every_voters=job.progress_every_voters,
                    voting_methods=resolve_voting_methods(),
                    emit_voter_progress=True,
                )
            )
            report_trial_completion(index, total, job)
        return records

    configure_worker_thread_environment(workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(run_trial_job, jobs, chunksize=1)
        for index, (job, result) in enumerate(zip(jobs, results), start=1):
            records.extend(result)
            report_trial_completion(index, total, job)
    return records


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixed-fleet workload quality and proposal-validity metrics."""
    return (
        raw.groupby(
            [
                "robots",
                "voters",
                "tasks",
                "capacity_per_robot",
                "assignment_slots",
                "method",
                "method_label",
            ],
            as_index=False,
        )
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


def run_experiment(
    *,
    task_counts: tuple[int, ...] = WORKLOAD_TASK_COUNTS,
    trials: int = FORMAL_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
    max_voters: int | None = None,
    voter_batch_size: int = DEFAULT_VOTER_BATCH_SIZE,
    progress_every_voters: int = DEFAULT_PROGRESS_EVERY_VOTERS,
    workers: int = DEFAULT_PARALLEL_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the canonical fixed-100-robot workload sweep with bounded trial parallelism."""
    validate_workload_config(
        task_counts=task_counts,
        trials=trials,
        packet_loss_rate=packet_loss_rate,
        max_voters=max_voters,
        voter_batch_size=voter_batch_size,
        progress_every_voters=progress_every_voters,
        workers=workers,
    )
    voting_methods = resolve_voting_methods()
    validate_zero_loss_optimizer_contract(seed, max(task_counts))

    print(
        "Zero-loss optimizer contract: PASS "
        "(Greedy/Sinkhorn feasible; Hungarian exact; Min-Cost Flow within quantization tolerance)"
    )
    print(
        "Experiment 2 workload: robots=100 fixed; "
        "capacity_per_robot=ceil(tasks/100); tasks are batch workload"
    )
    print("Voting methods: " + ", ".join(METHOD_LABELS[method] for method in voting_methods))
    print(
        "Voting receivers: "
        + ("all 100 robots" if max_voters is None else f"up to {max_voters} sampled robots (preview mode)")
    )
    print(f"Trials per task point: {trials}; process workers: {workers}")

    jobs = build_trial_jobs(
        task_counts=task_counts,
        trials=trials,
        packet_loss_rate=packet_loss_rate,
        seed=seed,
        max_voters=max_voters,
        voter_batch_size=voter_batch_size,
        progress_every_voters=progress_every_voters,
    )
    records = execute_trial_jobs(jobs, workers=workers)
    raw = pd.DataFrame.from_records(records)
    return raw, summarize_results(raw)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one task-batch-by-method report table in stable display order."""
    table = summary.pivot(index="tasks", columns="method_label", values=metric)
    labels = [
        METHOD_LABELS[method]
        for method in REPORT_METHOD_ORDER
        if METHOD_LABELS[method] in table.columns
    ]
    return table.reindex(columns=labels).reset_index()


def save_report_tables(summary: pd.DataFrame) -> None:
    metrics = {
        "average_optimality_gap_percent": "report_average_optimality_gap_percent.csv",
        "optimal_cost_match_percent": "report_optimal_cost_match_percent.csv",
        "near_optimal_5pct_percent": "report_near_optimal_5pct_percent.csv",
        "exact_optimal_assignment_percent": "report_exact_optimal_assignment_percent.csv",
        "average_valid_proposal_rate_percent": "report_valid_proposal_rate_percent.csv",
    }
    for metric, filename in metrics.items():
        report_table(summary, metric).to_csv(DATA_DIR / filename, index=False)


def combine_plot_label(methods: tuple[str, ...]) -> str:
    """Combine exactly overlapping Voting series into one readable legend label."""
    names = [METHOD_LABELS[method].removeprefix("Voting ") for method in methods]
    return "Voting " + " / ".join(names)


def build_plot_series_groups(
    summary: pd.DataFrame,
    metric: str,
) -> list[tuple[tuple[str, ...], pd.DataFrame]]:
    """Group only numerically identical method curves; near-overlaps remain separate."""
    groups: list[tuple[tuple[str, ...], pd.DataFrame]] = []
    for method in REPORT_METHOD_ORDER[1:]:
        part = summary[summary["method"] == method].sort_values("tasks")
        if part.empty:
            continue
        matched_index: int | None = None
        for index, (_, reference) in enumerate(groups):
            same_tasks = np.array_equal(
                part["tasks"].to_numpy(),
                reference["tasks"].to_numpy(),
            )
            same_values = np.allclose(
                part[metric].to_numpy(dtype=float),
                reference[metric].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
            if same_tasks and same_values:
                matched_index = index
                break
        if matched_index is None:
            groups.append(((method,), part))
        else:
            methods, reference = groups[matched_index]
            groups[matched_index] = (methods + (method,), reference)
    return groups


def save_metric_plot(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Save a line-only fixed-fleet workload plot with exact-overlap legend merging."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for methods, part in build_plot_series_groups(summary, metric):
        ax.plot(part["tasks"], part[metric], label=combine_plot_label(methods))
    ax.set_xlabel("Task batch size (100 robots fixed)")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Persist fixed-100 workload data generated by the current canonical method set."""
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "workload_comparison_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "workload_comparison_summary.csv", index=False)
    save_report_tables(summary)
    save_metric_plot(
        summary,
        metric="average_optimality_gap_percent",
        ylabel="Cost error from minimum (%)",
        filename="average_optimality_gap_percent.png",
    )
    save_metric_plot(
        summary,
        metric="optimal_cost_match_percent",
        ylabel="Optimal-cost match rate (%)",
        filename="optimal_cost_match_percent.png",
        y_limits=(0.0, 100.0),
    )


def parse_task_counts(values: list[int] | None) -> tuple[int, ...]:
    if values is None:
        return WORKLOAD_TASK_COUNTS
    parsed = tuple(values)
    if not parsed:
        raise argparse.ArgumentTypeError("task counts cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the time-budgeted lossy P2P Voting workload experiment with 100 fixed robots, "
            "100..1000 tasks, Greedy/Hungarian/Min-Cost-Flow/Sinkhorn local optimizers, "
            "and parallel trials."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=None,
        help="Task batch sizes. Default: 100 200 300 ... 1000; robots remain fixed at 100.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=FORMAL_TRIALS,
        help=f"Trials per task point. Canonical time-budgeted default: {FORMAL_TRIALS}.",
    )
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=PACKET_LOSS_RATE,
        help="Directed P2P scalar cost loss probability (default: 0.30).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-voters",
        type=int,
        default=None,
        help="Optional preview cap on the 100 physical voters. Default: all 100 robots.",
    )
    parser.add_argument(
        "--voter-batch-size",
        type=int,
        default=DEFAULT_VOTER_BATCH_SIZE,
        help=(
            "Receiver-local views handled together inside one trial. "
            f"Default: {DEFAULT_VOTER_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--progress-every-voters",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_VOTERS,
        help=(
            "Single-worker diagnostic progress interval. Parallel mode reports completed trials "
            f"instead. Default: {DEFAULT_PROGRESS_EVERY_VOTERS}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=(
            "Independent trial worker processes. "
            f"Default: {DEFAULT_PARALLEL_WORKERS}; use 1 for serial diagnostics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_counts = parse_task_counts(args.tasks)
    raw, summary = run_experiment(
        task_counts=task_counts,
        trials=args.trials,
        packet_loss_rate=args.packet_loss,
        seed=args.seed,
        max_voters=args.max_voters,
        voter_batch_size=args.voter_batch_size,
        progress_every_voters=args.progress_every_voters,
        workers=args.workers,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "workload_comparison_raw.csv")
    print(DATA_DIR / "workload_comparison_summary.csv")
    print(FIGURE_DIR)

    print("\nCost error from minimum (%) - lower is better:")
    print(
        report_table(summary, "average_optimality_gap_percent").to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOptimal-cost match (% of trials) - higher is better:")
    print(
        report_table(summary, "optimal_cost_match_percent").to_string(
            index=False,
            float_format=lambda value: f"{value:.1f}",
        )
    )

    print("\nTrials within 5% of optimum (%) - supporting metric:")
    print(
        report_table(summary, "near_optimal_5pct_percent").to_string(
            index=False,
            float_format=lambda value: f"{value:.1f}",
        )
    )

    print("\nValid local optimizer proposals (%) - diagnostic:")
    print(
        report_table(summary, "average_valid_proposal_rate_percent").to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


if __name__ == "__main__":
    main()
