from __future__ import annotations

import numpy as np


CAPACITATED_HEURISTIC_METHODS = (
    "p2p_sequential_greedy",
    "p2p_global_greedy",
    "p2p_regret2_greedy",
)


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first workload-heuristic owner boundary."""
    raise ValueError(
        "owner=run_multitask_workload_heuristics "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_capacitated_problem(
    costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> None:
    """Validate one receiver-local capacitated heuristic problem."""
    if costs.ndim != 2:
        fail(
            "validate_capacitated_problem",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    robot_count, task_count = costs.shape
    if robot_count <= 0 or task_count <= 0:
        fail(
            "validate_capacitated_problem",
            "contract",
            "EMPTY_ASSIGNMENT_PROBLEM",
            f"shape={costs.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "validate_capacitated_problem",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    if task_count > robot_count * capacity_per_robot:
        fail(
            "validate_capacitated_problem",
            "state",
            "CAPACITY_EXCEEDED",
            (
                f"robots={robot_count} tasks={task_count} "
                f"capacity_per_robot={capacity_per_robot}"
            ),
        )
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        fail(
            "validate_capacitated_problem",
            "data",
            "INVALID_HEURISTIC_COST",
            "cost matrix may contain finite values or +inf unavailable edges only",
        )
    if task_order.shape != (task_count,):
        fail(
            "validate_capacitated_problem",
            "contract",
            "TASK_ORDER_SHAPE_MISMATCH",
            f"expected={(task_count,)} actual={task_order.shape}",
        )
    if set(task_order.tolist()) != set(range(task_count)):
        fail(
            "validate_capacitated_problem",
            "contract",
            "INVALID_TASK_ORDER",
            f"task_count={task_count}",
        )


def solve_sequential_greedy_capacitated(
    costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Assign tasks in paired task order to the cheapest robot with remaining capacity."""
    validate_capacitated_problem(costs, task_order, capacity_per_robot)
    robot_count, task_count = costs.shape
    remaining = np.full(robot_count, capacity_per_robot, dtype=int)
    assignment = np.full(task_count, -1, dtype=int)

    for task in task_order:
        candidates = np.flatnonzero((remaining > 0) & np.isfinite(costs[:, task]))
        if len(candidates) == 0:
            return None
        robot = int(candidates[np.argmin(costs[candidates, task])])
        assignment[task] = robot
        remaining[robot] -= 1
    return assignment


def solve_global_greedy_capacitated(
    costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Consume globally cheapest feasible robot-task edges until every task is assigned."""
    validate_capacitated_problem(costs, task_order, capacity_per_robot)
    robot_count, task_count = costs.shape
    remaining = np.full(robot_count, capacity_per_robot, dtype=int)
    assignment = np.full(task_count, -1, dtype=int)

    flat_costs = costs.reshape(-1)
    finite_edges = np.flatnonzero(np.isfinite(flat_costs))
    if len(finite_edges) == 0:
        return None
    ordered_edges = finite_edges[
        np.argsort(flat_costs[finite_edges], kind="stable")
    ]

    assigned_count = 0
    for edge in ordered_edges:
        robot = int(edge // task_count)
        task = int(edge % task_count)
        if assignment[task] >= 0 or remaining[robot] <= 0:
            continue
        assignment[task] = robot
        remaining[robot] -= 1
        assigned_count += 1
        if assigned_count == task_count:
            return assignment
    return None


def compute_static_regret2_priority(
    costs: np.ndarray,
    task_order: np.ndarray,
) -> np.ndarray | None:
    """Order tasks once by the gap between their two best receiver-visible robot costs."""
    finite = np.isfinite(costs)
    finite_count = finite.sum(axis=0)
    if np.any(finite_count == 0):
        return None

    masked = np.where(finite, costs, np.inf)
    if costs.shape[0] == 1:
        regret = np.full(costs.shape[1], np.inf, dtype=float)
    else:
        two_best = np.partition(masked, 1, axis=0)[:2]
        regret = two_best[1] - two_best[0]
        regret = np.where(finite_count == 1, np.inf, regret)

    task_rank = np.empty(costs.shape[1], dtype=int)
    task_rank[task_order] = np.arange(costs.shape[1])
    return np.lexsort((task_rank, -regret)).astype(int)


def solve_regret2_greedy_capacitated(
    costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Assign a static Regret-2 task priority to cheapest robots with remaining capacity."""
    validate_capacitated_problem(costs, task_order, capacity_per_robot)
    priority = compute_static_regret2_priority(costs, task_order)
    if priority is None:
        return None

    robot_count, task_count = costs.shape
    remaining = np.full(robot_count, capacity_per_robot, dtype=int)
    assignment = np.full(task_count, -1, dtype=int)
    for task in priority:
        candidates = np.flatnonzero((remaining > 0) & np.isfinite(costs[:, task]))
        if len(candidates) == 0:
            return None
        robot = int(candidates[np.argmin(costs[candidates, task])])
        assignment[task] = robot
        remaining[robot] -= 1
    return assignment


def solve_capacitated_heuristic(
    method: str,
    costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Route one receiver-local problem to its named fast heuristic owner."""
    if method == "p2p_sequential_greedy":
        return solve_sequential_greedy_capacitated(costs, task_order, capacity_per_robot)
    if method == "p2p_global_greedy":
        return solve_global_greedy_capacitated(costs, task_order, capacity_per_robot)
    if method == "p2p_regret2_greedy":
        return solve_regret2_greedy_capacitated(costs, task_order, capacity_per_robot)
    fail("solve_capacitated_heuristic", "contract", "UNKNOWN_METHOD", f"method={method}")


def solve_capacitated_heuristic_batch(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one fast capacitated heuristic proposal per receiver-local cost matrix."""
    if receiver_costs.ndim != 3:
        fail(
            "solve_capacitated_heuristic_batch",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    if method not in CAPACITATED_HEURISTIC_METHODS:
        fail(
            "solve_capacitated_heuristic_batch",
            "contract",
            "UNKNOWN_METHOD",
            f"method={method}",
        )

    receiver_count, _, task_count = receiver_costs.shape
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        proposal = solve_capacitated_heuristic(
            method,
            receiver_costs[receiver],
            task_order,
            capacity_per_robot,
        )
        if proposal is None:
            continue
        proposals[receiver] = proposal
        valid[receiver] = True
    return proposals, valid
