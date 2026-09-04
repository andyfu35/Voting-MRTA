from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ortools.graph.python import min_cost_flow
except ImportError:  # Dependency is checked at the named runtime boundary.
    min_cost_flow = None


MIN_COST_FLOW_METHOD = "p2p_min_cost_flow"
SINKHORN_METHOD = "p2p_sinkhorn"
FAST_WORKLOAD_OPTIMIZER_METHODS = (MIN_COST_FLOW_METHOD, SINKHORN_METHOD)
MIN_COST_FLOW_COST_SCALE = 1_000_000
MIN_COST_FLOW_ORACLE_TOLERANCE_PERCENT = 0.01


@dataclass(frozen=True)
class SinkhornConfig:
    epsilon: float = 0.08
    max_iterations: int = 30
    tolerance: float = 1e-5


DEFAULT_SINKHORN_CONFIG = SinkhornConfig()


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first fast workload-optimizer owner boundary."""
    raise ValueError(
        "owner=run_multitask_workload_optimizers "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_optimizer_problem(costs: np.ndarray, capacity_per_robot: int) -> None:
    """Validate one receiver-local capacitated optimizer problem."""
    if costs.ndim != 2:
        fail(
            "validate_optimizer_problem",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    robot_count, task_count = costs.shape
    if robot_count <= 0 or task_count <= 0:
        fail(
            "validate_optimizer_problem",
            "contract",
            "EMPTY_ASSIGNMENT_PROBLEM",
            f"shape={costs.shape}",
        )
    if capacity_per_robot <= 0:
        fail(
            "validate_optimizer_problem",
            "contract",
            "INVALID_ROBOT_CAPACITY",
            f"actual={capacity_per_robot}",
        )
    if task_count > robot_count * capacity_per_robot:
        fail(
            "validate_optimizer_problem",
            "state",
            "CAPACITY_EXCEEDED",
            (
                f"robots={robot_count} tasks={task_count} "
                f"capacity_per_robot={capacity_per_robot}"
            ),
        )
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        fail(
            "validate_optimizer_problem",
            "data",
            "INVALID_OPTIMIZER_COST",
            "cost matrix may contain finite values or +inf unavailable edges only",
        )
    if np.any(np.isfinite(costs).sum(axis=0) == 0):
        fail(
            "validate_optimizer_problem",
            "planning",
            "TASK_WITHOUT_VISIBLE_EDGE",
            "at least one task has no receiver-visible robot edge",
        )


def require_min_cost_flow_dependency():
    """Return the OR-Tools min-cost-flow module or fail at the dependency boundary."""
    if min_cost_flow is None:
        fail(
            "require_min_cost_flow_dependency",
            "dependency",
            "ORTOOLS_NOT_AVAILABLE",
            "install project requirements: pip install -r requirements.txt",
        )
    return min_cost_flow


def quantize_min_cost_flow_costs(costs: np.ndarray) -> np.ndarray:
    """Convert finite float costs to stable integer arc costs for OR-Tools."""
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        return np.empty(0, dtype=np.int64)
    scaled = np.rint(finite_costs * MIN_COST_FLOW_COST_SCALE)
    if np.any(np.abs(scaled) > np.iinfo(np.int64).max):
        fail(
            "quantize_min_cost_flow_costs",
            "data",
            "MIN_COST_FLOW_COST_OVERFLOW",
            f"scale={MIN_COST_FLOW_COST_SCALE}",
        )
    return scaled.astype(np.int64)


def solve_min_cost_flow_capacitated(
    costs: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Solve receiver-local capacitated assignment with OR-Tools SimpleMinCostFlow."""
    validate_optimizer_problem(costs, capacity_per_robot)
    ortools_mcf = require_min_cost_flow_dependency()
    robot_count, task_count = costs.shape
    finite = np.isfinite(costs)
    robot_indices, task_indices = np.nonzero(finite)
    if robot_indices.size == 0:
        return None

    source = 0
    robot_nodes = 1 + np.arange(robot_count, dtype=np.int64)
    task_nodes = 1 + robot_count + np.arange(task_count, dtype=np.int64)
    sink = 1 + robot_count + task_count

    source_starts = np.full(robot_count, source, dtype=np.int64)
    source_ends = robot_nodes
    source_caps = np.full(robot_count, capacity_per_robot, dtype=np.int64)
    source_costs = np.zeros(robot_count, dtype=np.int64)

    edge_starts = 1 + robot_indices.astype(np.int64)
    edge_ends = 1 + robot_count + task_indices.astype(np.int64)
    edge_caps = np.ones(robot_indices.size, dtype=np.int64)
    edge_costs = quantize_min_cost_flow_costs(costs)

    sink_starts = task_nodes
    sink_ends = np.full(task_count, sink, dtype=np.int64)
    sink_caps = np.ones(task_count, dtype=np.int64)
    sink_costs = np.zeros(task_count, dtype=np.int64)

    start_nodes = np.concatenate((source_starts, edge_starts, sink_starts))
    end_nodes = np.concatenate((source_ends, edge_ends, sink_ends))
    capacities = np.concatenate((source_caps, edge_caps, sink_caps))
    unit_costs = np.concatenate((source_costs, edge_costs, sink_costs))

    solver = ortools_mcf.SimpleMinCostFlow()
    arc_ids = solver.add_arcs_with_capacity_and_unit_cost(
        start_nodes,
        end_nodes,
        capacities,
        unit_costs,
    )
    supplies = np.zeros(sink + 1, dtype=np.int64)
    supplies[source] = task_count
    supplies[sink] = -task_count
    solver.set_nodes_supplies(np.arange(sink + 1, dtype=np.int64), supplies)
    status = solver.solve()
    if status != solver.OPTIMAL:
        return None

    edge_offset = robot_count
    edge_arc_ids = arc_ids[edge_offset : edge_offset + robot_indices.size]
    edge_flows = solver.flows(edge_arc_ids)
    chosen = np.flatnonzero(edge_flows > 0)
    if chosen.size != task_count:
        return None

    assignment = np.full(task_count, -1, dtype=int)
    assignment[task_indices[chosen]] = robot_indices[chosen]
    if np.any(assignment < 0):
        return None
    return assignment


def solve_min_cost_flow_batch(
    *,
    receiver_costs: np.ndarray,
    capacity_per_robot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one min-cost-flow proposal per receiver-local physical cost matrix."""
    if receiver_costs.ndim != 3:
        fail(
            "solve_min_cost_flow_batch",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    receiver_count, _, task_count = receiver_costs.shape
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        proposal = solve_min_cost_flow_capacitated(
            receiver_costs[receiver],
            capacity_per_robot,
        )
        if proposal is None:
            continue
        proposals[receiver] = proposal
        valid[receiver] = True
    return proposals, valid


def validate_sinkhorn_config(config: SinkhornConfig) -> None:
    """Validate the fixed Sinkhorn approximation contract."""
    if not np.isfinite(config.epsilon) or config.epsilon <= 0.0:
        fail(
            "validate_sinkhorn_config",
            "contract",
            "INVALID_SINKHORN_EPSILON",
            f"actual={config.epsilon}",
        )
    if config.max_iterations <= 0:
        fail(
            "validate_sinkhorn_config",
            "contract",
            "INVALID_SINKHORN_ITERATIONS",
            f"actual={config.max_iterations}",
        )
    if not np.isfinite(config.tolerance) or config.tolerance <= 0.0:
        fail(
            "validate_sinkhorn_config",
            "contract",
            "INVALID_SINKHORN_TOLERANCE",
            f"actual={config.tolerance}",
        )


def compute_sinkhorn_transport_plan(
    costs: np.ndarray,
    capacity_per_robot: int,
    config: SinkhornConfig = DEFAULT_SINKHORN_CONFIG,
) -> np.ndarray | None:
    """Compute a soft receiver-local transport plan under uniform row mass."""
    validate_optimizer_problem(costs, capacity_per_robot)
    validate_sinkhorn_config(config)
    robot_count, task_count = costs.shape
    finite = np.isfinite(costs)
    if np.any(finite.sum(axis=1) == 0):
        return None

    column_min = np.min(np.where(finite, costs, np.inf), axis=0)
    shifted = np.where(finite, costs - column_min[None, :], np.inf)
    kernel = np.where(finite, np.exp(-shifted / config.epsilon), 0.0)

    row_target = np.full(robot_count, task_count / robot_count, dtype=float)
    if np.any(row_target > capacity_per_robot + 1e-12):
        fail(
            "compute_sinkhorn_transport_plan",
            "state",
            "SINKHORN_ROW_MASS_EXCEEDS_CAPACITY",
            (
                f"target={float(row_target[0])} "
                f"capacity_per_robot={capacity_per_robot}"
            ),
        )
    column_target = np.ones(task_count, dtype=float)
    u = np.ones(robot_count, dtype=float)
    v = np.ones(task_count, dtype=float)

    for iteration in range(config.max_iterations):
        kv = kernel @ v
        if np.any(kv <= 0.0) or np.any(~np.isfinite(kv)):
            return None
        u = row_target / kv

        ktu = kernel.T @ u
        if np.any(ktu <= 0.0) or np.any(~np.isfinite(ktu)):
            return None
        v = column_target / ktu

        if iteration % 5 == 4 or iteration == config.max_iterations - 1:
            row_mass = u * (kernel @ v)
            column_mass = v * (kernel.T @ u)
            error = max(
                float(np.max(np.abs(row_mass - row_target))),
                float(np.max(np.abs(column_mass - column_target))),
            )
            if error <= config.tolerance:
                break

    plan = (u[:, None] * kernel) * v[None, :]
    if np.any(~np.isfinite(plan)):
        return None
    return plan


def round_sinkhorn_plan_to_capacity(
    plan: np.ndarray,
    costs: np.ndarray,
    capacity_per_robot: int,
) -> np.ndarray | None:
    """Round a soft plan greedily by scarcity and confidence without exceeding capacity."""
    if plan.shape != costs.shape:
        fail(
            "round_sinkhorn_plan_to_capacity",
            "contract",
            "SINKHORN_PLAN_SHAPE_MISMATCH",
            f"plan={plan.shape} costs={costs.shape}",
        )
    robot_count, task_count = costs.shape
    finite = np.isfinite(costs)
    finite_count = finite.sum(axis=0)
    if np.any(finite_count == 0):
        return None

    masked_plan = np.where(finite, plan, -np.inf)
    if robot_count == 1:
        confidence = np.full(task_count, np.inf, dtype=float)
    else:
        two_best = np.partition(masked_plan, -2, axis=0)[-2:]
        confidence = two_best[1] - two_best[0]
        confidence = np.where(finite_count == 1, np.inf, confidence)

    task_index = np.arange(task_count, dtype=int)
    task_priority = np.lexsort((task_index, -confidence, finite_count))
    remaining = np.full(robot_count, capacity_per_robot, dtype=int)
    assignment = np.full(task_count, -1, dtype=int)

    for task in task_priority:
        candidates = np.flatnonzero((remaining > 0) & finite[:, task])
        if candidates.size == 0:
            return None
        scores = plan[candidates, task]
        robot = int(candidates[np.argmax(scores)])
        assignment[task] = robot
        remaining[robot] -= 1
    return assignment


def solve_sinkhorn_capacitated(
    costs: np.ndarray,
    capacity_per_robot: int,
    config: SinkhornConfig = DEFAULT_SINKHORN_CONFIG,
) -> np.ndarray | None:
    """Compute a Sinkhorn soft plan and discretize it at the named rounding boundary."""
    plan = compute_sinkhorn_transport_plan(costs, capacity_per_robot, config)
    if plan is None:
        return None
    return round_sinkhorn_plan_to_capacity(plan, costs, capacity_per_robot)


def solve_sinkhorn_batch(
    *,
    receiver_costs: np.ndarray,
    capacity_per_robot: int,
    config: SinkhornConfig = DEFAULT_SINKHORN_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one Sinkhorn-plus-rounding proposal per receiver-local physical cost matrix."""
    if receiver_costs.ndim != 3:
        fail(
            "solve_sinkhorn_batch",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    validate_sinkhorn_config(config)
    receiver_count, _, task_count = receiver_costs.shape
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        proposal = solve_sinkhorn_capacitated(
            receiver_costs[receiver],
            capacity_per_robot,
            config,
        )
        if proposal is None:
            continue
        proposals[receiver] = proposal
        valid[receiver] = True
    return proposals, valid
