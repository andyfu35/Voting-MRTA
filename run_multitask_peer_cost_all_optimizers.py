from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multitask_optimizer_screening import (
    MILP_NUMERICAL_TOLERANCE_PERCENT,
    solve_milp_assignment,
)
from run_multitask_peer_cost_experiment import (
    DEFAULT_TRIALS,
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    PACKET_LOSS_RATE,
    RANDOM_SEED,
    assignment_total_cost,
    build_assignment_support,
    generate_spatial_cost_matrix,
    solve_hungarian_assignment,
    solve_local_optimizer_proposals,
    solve_support_consensus,
)

SCALING_TASK_COUNTS = (50, 100, 200, 400, 600, 800, 1000)
DEFAULT_VOTING_METHODS = ("p2p_greedy", "p2p_hungarian", "p2p_auction")
OPTIONAL_VOTING_METHODS = ("p2p_milp",)
REPORT_METHOD_ORDER = ("oracle",) + DEFAULT_VOTING_METHODS + OPTIONAL_VOTING_METHODS
METHOD_LABELS = {
    "oracle": "Hungarian Oracle",
    "p2p_greedy": "Voting Greedy",
    "p2p_hungarian": "Voting Hungarian",
    "p2p_auction": "Voting Auction",
    "p2p_milp": "Voting MILP",
}
DEFAULT_VOTER_BATCH_SIZE = 8
DEFAULT_PROGRESS_EVERY_VOTERS = 25
VOTER_SELECTION_SEED_OFFSET = 2_000_003
VISIBILITY_SEED_OFFSET = 4_000_007
MILP_ZERO_LOSS_CHECK_MAX_SIZE = 50

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_peer_cost_scaling"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first Experiment 2 owner/function boundary."""
    raise ValueError(
        "owner=run_multitask_peer_cost_all_optimizers "
        f"function={function} category={category} code={code} details={details}"
    )


def cost_match_tolerance_percent(method: str) -> float:
    """Return the objective-match tolerance owned by each active optimizer family."""
    if method == "p2p_milp":
        return MILP_NUMERICAL_TOLERANCE_PERCENT
    if method in REPORT_METHOD_ORDER:
        return OPTIMAL_COST_TOLERANCE_PERCENT
    fail("cost_match_tolerance_percent", "contract", "UNKNOWN_METHOD", f"method={method}")


def resolve_voting_methods(include_milp: bool) -> tuple[str, ...]:
    """Resolve the default fast methods plus an explicitly requested MILP probe."""
    if include_milp:
        return DEFAULT_VOTING_METHODS + OPTIONAL_VOTING_METHODS
    return DEFAULT_VOTING_METHODS


def validate_scaling_config(
    *,
    task_counts: tuple[int, ...],
    trials: int,
    packet_loss_rate: float,
    max_voters: int | None,
    voter_batch_size: int,
    progress_every_voters: int,
) -> None:
    """Validate matched robot/task scaling and runtime-only batching controls."""
    if not task_counts:
        fail("validate_scaling_config", "contract", "EMPTY_TASK_COUNTS", "task_counts is empty")
    if any(task_count < 2 for task_count in task_counts):
        fail(
            "validate_scaling_config",
            "contract",
            "TASK_COUNT_TOO_SMALL",
            f"expected>=2 actual={task_counts}",
        )
    if trials <= 0:
        fail("validate_scaling_config", "contract", "INVALID_TRIALS", f"actual={trials}")
    if not 0.0 <= packet_loss_rate < 1.0:
        fail(
            "validate_scaling_config",
            "contract",
            "INVALID_PACKET_LOSS",
            f"actual={packet_loss_rate}",
        )
    if max_voters is not None and max_voters <= 0:
        fail(
            "validate_scaling_config",
            "contract",
            "INVALID_MAX_VOTERS",
            f"expected>=1 actual={max_voters}",
        )
    if voter_batch_size <= 0:
        fail(
            "validate_scaling_config",
            "contract",
            "INVALID_VOTER_BATCH_SIZE",
            f"expected>=1 actual={voter_batch_size}",
        )
    if progress_every_voters <= 0:
        fail(
            "validate_scaling_config",
            "contract",
            "INVALID_PROGRESS_INTERVAL",
            f"expected>=1 actual={progress_every_voters}",
        )


def resolve_voter_count(robot_count: int, max_voters: int | None) -> int:
    """Use the whole fleet unless a preview explicitly caps the number of voting receivers."""
    if max_voters is None:
        return robot_count
    return min(robot_count, max_voters)


def select_voter_indices(
    *,
    robot_count: int,
    voter_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select the receiver identities whose local proposals participate in Voting."""
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
    """Sample directed sender->receiver task-cost visibility for one receiver batch."""
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
    """Materialize only the current receiver batch's incomplete cost matrices."""
    if costs.ndim != 2:
        fail(
            "build_voter_batch_cost_views",
            "contract",
            "INVALID_COST_MATRIX_SHAPE",
            f"shape={costs.shape}",
        )
    expected_tail = costs.shape
    if visibility.ndim != 3 or visibility.shape[1:] != expected_tail:
        fail(
            "build_voter_batch_cost_views",
            "contract",
            "VISIBILITY_SHAPE_MISMATCH",
            f"expected=(*,{expected_tail[0]},{expected_tail[1]}) actual={visibility.shape}",
        )
    return np.where(visibility, costs[None, :, :], np.inf)


def solve_milp_batch_proposals(receiver_costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the true MILP owner once per receiver-local incomplete matrix."""
    if receiver_costs.ndim != 3:
        fail(
            "solve_milp_batch_proposals",
            "contract",
            "INVALID_BATCH_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    receiver_count, _, task_count = receiver_costs.shape
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        proposal = solve_milp_assignment(receiver_costs[receiver])
        if proposal is None:
            continue
        proposals[receiver] = proposal
        valid[receiver] = True
    return proposals, valid


def solve_voter_batch_proposals(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route one voter batch to the existing optimizer owner without duplicating algorithms."""
    if method in DEFAULT_VOTING_METHODS:
        return solve_local_optimizer_proposals(method, receiver_costs, task_order)
    if method == "p2p_milp":
        return solve_milp_batch_proposals(receiver_costs)
    fail("solve_voter_batch_proposals", "contract", "UNKNOWN_METHOD", f"method={method}")


def accumulate_proposal_support(
    *,
    support: np.ndarray,
    proposals: np.ndarray,
    valid: np.ndarray,
    robot_count: int,
) -> int:
    """Accumulate one voter batch without failing when that batch has zero valid proposals."""
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
    """Report scalable receiver-batch progress without affecting experiment state."""
    end = "\n" if completed >= total else "\r"
    print(
        f"tasks=robots={task_count:4d} trial={trial:3d} voters={completed:4d}/{total}",
        end=end,
        flush=True,
    )


def collect_voting_support(
    *,
    costs: np.ndarray,
    packet_loss_rate: float,
    voter_indices: np.ndarray,
    task_order: np.ndarray,
    visibility_rng: np.random.Generator,
    voter_batch_size: int,
    progress_every_voters: int,
    task_count: int,
    trial: int,
    voting_methods: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Stream voter-local views in bounded batches and accumulate support per optimizer."""
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
            )
            valid_counts[method] += accumulate_proposal_support(
                support=support_by_method[method],
                proposals=proposals,
                valid=valid,
                robot_count=robot_count,
            )

        completed = min(start + len(receiver_batch), total_voters)
        if completed >= next_progress or completed == total_voters:
            report_voter_progress(
                task_count=task_count,
                trial=trial,
                completed=completed,
                total=total_voters,
            )
            while next_progress <= completed:
                next_progress += progress_every_voters

    return support_by_method, valid_counts


def finalize_voting_assignments(
    *,
    support_by_method: dict[str, np.ndarray],
    valid_counts: dict[str, int],
    voter_count: int,
    tie_priority: np.ndarray,
    task_count: int,
    trial: int,
    voting_methods: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, float]]:
    """Convert accumulated proposal support into one final assignment per Voting method."""
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
        assignment = solve_support_consensus(support_by_method[method], tie_priority)
        results[method] = (assignment, 100.0 * valid_count / voter_count)
    return results


def solve_zero_loss_consensus(
    *,
    method: str,
    costs: np.ndarray,
    task_order: np.ndarray,
    tie_priority: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run one complete-information local proposal through the same support consensus boundary."""
    receiver_costs = costs[None, :, :]
    proposals, valid = solve_voter_batch_proposals(
        method=method,
        receiver_costs=receiver_costs,
        task_order=task_order,
    )
    if not bool(valid[0]):
        fail(
            "solve_zero_loss_consensus",
            "planning",
            "ZERO_LOSS_PROPOSAL_FAILURE",
            f"method={method} shape={costs.shape}",
        )
    support = build_assignment_support(proposals, valid, costs.shape[0])
    return solve_support_consensus(support, tie_priority), 100.0


def validate_zero_loss_optimizer_contract(
    seed: int,
    max_task_count: int,
    voting_methods: tuple[str, ...],
) -> None:
    """Check exact optimizer integrations without turning preflight into a large MILP run."""
    single_rng = np.random.default_rng(seed + 91_001)
    single_costs = generate_spatial_cost_matrix(5, 1, single_rng)
    single_order = np.array([0], dtype=int)
    single_tie = single_rng.random((5, 1))
    single_oracle = solve_hungarian_assignment(single_costs)
    if single_oracle is None:
        fail("validate_zero_loss_optimizer_contract", "planning", "ORACLE_INFEASIBLE", "robots=5 tasks=1")
    single_oracle_cost = assignment_total_cost(single_costs, single_oracle)
    greedy, _ = solve_zero_loss_consensus(
        method="p2p_greedy",
        costs=single_costs,
        task_order=single_order,
        tie_priority=single_tie,
    )
    greedy_cost = assignment_total_cost(single_costs, greedy)
    greedy_gap = 100.0 * (greedy_cost - single_oracle_cost) / single_oracle_cost
    if abs(greedy_gap) > OPTIMAL_COST_TOLERANCE_PERCENT:
        fail(
            "validate_zero_loss_optimizer_contract",
            "planning",
            "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
            f"method=p2p_greedy robots=5 tasks=1 actual_gap_percent={greedy_gap}",
        )

    representative_sizes = sorted(
        {
            min(50, max_task_count),
            min(200, max_task_count),
            max_task_count,
        }
    )
    for size in representative_sizes:
        if size < 2:
            continue
        rng = np.random.default_rng(seed + 91_173 + size * 100_003)
        costs = generate_spatial_cost_matrix(size, size, rng)
        task_order = rng.permutation(size)
        tie_priority = rng.random((size, size))
        oracle = solve_hungarian_assignment(costs)
        if oracle is None:
            fail(
                "validate_zero_loss_optimizer_contract",
                "planning",
                "ORACLE_INFEASIBLE",
                f"robots={size} tasks={size}",
            )
        oracle_cost = assignment_total_cost(costs, oracle)
        for method in ("p2p_hungarian", "p2p_auction"):
            assignment, valid_rate = solve_zero_loss_consensus(
                method=method,
                costs=costs,
                task_order=task_order,
                tie_priority=tie_priority,
            )
            if valid_rate != 100.0:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_PROPOSAL_FAILURE",
                    f"method={method} robots={size} tasks={size} valid_rate={valid_rate}",
                )
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
            if abs(gap_percent) > OPTIMAL_COST_TOLERANCE_PERCENT:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
                    (
                        f"method={method} robots={size} tasks={size} "
                        f"expected_abs_gap_percent<={OPTIMAL_COST_TOLERANCE_PERCENT} "
                        f"actual={gap_percent}"
                    ),
                )

    if "p2p_milp" in voting_methods:
        size = min(MILP_ZERO_LOSS_CHECK_MAX_SIZE, max_task_count)
        if size >= 2:
            rng = np.random.default_rng(seed + 191_173 + size * 100_003)
            costs = generate_spatial_cost_matrix(size, size, rng)
            task_order = rng.permutation(size)
            tie_priority = rng.random((size, size))
            oracle = solve_hungarian_assignment(costs)
            if oracle is None:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ORACLE_INFEASIBLE",
                    f"method=p2p_milp robots={size} tasks={size}",
                )
            oracle_cost = assignment_total_cost(costs, oracle)
            assignment, valid_rate = solve_zero_loss_consensus(
                method="p2p_milp",
                costs=costs,
                task_order=task_order,
                tie_priority=tie_priority,
            )
            if valid_rate != 100.0:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_PROPOSAL_FAILURE",
                    f"method=p2p_milp robots={size} tasks={size} valid_rate={valid_rate}",
                )
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
            if abs(gap_percent) > MILP_NUMERICAL_TOLERANCE_PERCENT:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
                    (
                        f"method=p2p_milp robots={size} tasks={size} "
                        f"expected_abs_gap_percent<={MILP_NUMERICAL_TOLERANCE_PERCENT} "
                        f"actual={gap_percent}"
                    ),
                )


def evaluate_assignment(
    *,
    robot_count: int,
    voter_count: int,
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
    """Evaluate one matched-scale assignment against the full-information Oracle."""
    total_cost = assignment_total_cost(costs, assignment)
    gap_percent = 100.0 * (total_cost - optimal_cost) / optimal_cost
    return {
        "robots": robot_count,
        "voters": voter_count,
        "packet_loss_percent": 100.0 * packet_loss_rate,
        "tasks": task_count,
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
) -> list[dict[str, object]]:
    """Run one paired matched-scale trial with robot_count == task_count."""
    robot_count = task_count
    voter_count = resolve_voter_count(robot_count, max_voters)
    trial_seed = seed + task_count * 100_003 + trial * 1_009

    scenario_rng = np.random.default_rng(trial_seed)
    costs = generate_spatial_cost_matrix(robot_count, task_count, scenario_rng)
    task_order = scenario_rng.permutation(task_count)
    tie_priority = scenario_rng.random((robot_count, task_count))

    optimal_assignment = solve_hungarian_assignment(costs)
    if optimal_assignment is None:
        fail(
            "run_trial",
            "planning",
            "ORACLE_INFEASIBLE",
            f"robots={robot_count} tasks={task_count} trial={trial}",
        )
    optimal_cost = assignment_total_cost(costs, optimal_assignment)

    voter_rng = np.random.default_rng(trial_seed + VOTER_SELECTION_SEED_OFFSET)
    voter_indices = select_voter_indices(
        robot_count=robot_count,
        voter_count=voter_count,
        rng=voter_rng,
    )
    visibility_rng = np.random.default_rng(trial_seed + VISIBILITY_SEED_OFFSET)
    support_by_method, valid_counts = collect_voting_support(
        costs=costs,
        packet_loss_rate=packet_loss_rate,
        voter_indices=voter_indices,
        task_order=task_order,
        visibility_rng=visibility_rng,
        voter_batch_size=voter_batch_size,
        progress_every_voters=progress_every_voters,
        task_count=task_count,
        trial=trial,
        voting_methods=voting_methods,
    )
    voting_results = finalize_voting_assignments(
        support_by_method=support_by_method,
        valid_counts=valid_counts,
        voter_count=voter_count,
        tie_priority=tie_priority,
        task_count=task_count,
        trial=trial,
        voting_methods=voting_methods,
    )

    records = [
        evaluate_assignment(
            robot_count=robot_count,
            voter_count=voter_count,
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


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched-scale quality and local-proposal validity over paired trials."""
    return (
        raw.groupby(["robots", "voters", "tasks", "method", "method_label"], as_index=False)
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
    task_counts: tuple[int, ...] = SCALING_TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
    max_voters: int | None = None,
    voter_batch_size: int = DEFAULT_VOTER_BATCH_SIZE,
    progress_every_voters: int = DEFAULT_PROGRESS_EVERY_VOTERS,
    include_milp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the matched robot/task scaling sweep through 1000 tasks."""
    validate_scaling_config(
        task_counts=task_counts,
        trials=trials,
        packet_loss_rate=packet_loss_rate,
        max_voters=max_voters,
        voter_batch_size=voter_batch_size,
        progress_every_voters=progress_every_voters,
    )
    voting_methods = resolve_voting_methods(include_milp)
    validate_zero_loss_optimizer_contract(seed, max(task_counts), voting_methods)
    gate_methods = "Hungarian/Auction"
    if include_milp:
        gate_methods += "/MILP"
    print(
        "Zero-loss optimizer contract: PASS "
        f"({gate_methods} match Oracle under their numerical contracts; single-task Greedy matches Oracle)"
    )
    print("Experiment 2 scaling: robot_count == task_count")
    print("Voting methods: " + ", ".join(METHOD_LABELS[method] for method in voting_methods))
    print(
        "Voting receivers: "
        + ("all robots" if max_voters is None else f"up to {max_voters} sampled robots (preview mode)")
    )

    records: list[dict[str, object]] = []
    for task_count in task_counts:
        voter_count = resolve_voter_count(task_count, max_voters)
        for trial in range(1, trials + 1):
            print(
                f"tasks=robots={task_count:4d}/{max(task_counts)} "
                f"trial={trial:3d}/{trials} voters={voter_count} start",
                flush=True,
            )
            records.extend(
                run_trial(
                    task_count=task_count,
                    trial=trial,
                    packet_loss_rate=packet_loss_rate,
                    seed=seed,
                    max_voters=max_voters,
                    voter_batch_size=voter_batch_size,
                    progress_every_voters=progress_every_voters,
                    voting_methods=voting_methods,
                )
            )
        print(f"tasks=robots={task_count:4d}/{max(task_counts)} complete")

    raw = pd.DataFrame.from_records(records)
    return raw, summarize_results(raw)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one scale-by-method report table in stable display order."""
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
    """Save a line-only report plot; exactly overlapping curves share one legend entry."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for methods, part in build_plot_series_groups(summary, metric):
        ax.plot(part["tasks"], part[metric], label=combine_plot_label(methods))
    ax.set_xlabel("Matched robot / simultaneous task count")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Persist the scaling dataset separately from the old fixed-100-robot data."""
    ensure_output_dirs()
    raw.to_csv(DATA_DIR / "scaling_comparison_raw.csv", index=False)
    summary.to_csv(DATA_DIR / "scaling_comparison_summary.csv", index=False)
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
        return SCALING_TASK_COUNTS
    parsed = tuple(values)
    if not parsed:
        raise argparse.ArgumentTypeError("task counts cannot be empty")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale the lossy P2P Voting experiment to matched robot/task counts through 1000. "
            "Greedy, Hungarian, and Auction are enabled by default; MILP is an optional probe."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=None,
        help="Matched robot/task counts. Default: 50 100 200 400 600 800 1000",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
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
        help=(
            "Optional preview cap on voting receivers per trial. Default: all robots. "
            "For a fast trend preview use --max-voters 100."
        ),
    )
    parser.add_argument(
        "--voter-batch-size",
        type=int,
        default=DEFAULT_VOTER_BATCH_SIZE,
        help=(
            "Number of receiver-local cost views materialized at once. "
            f"Default: {DEFAULT_VOTER_BATCH_SIZE}. This changes memory/runtime only."
        ),
    )
    parser.add_argument(
        "--progress-every-voters",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_VOTERS,
        help="Print progress after roughly this many completed voting receivers (default: 25).",
    )
    parser.add_argument(
        "--include-milp",
        action="store_true",
        help=(
            "Also run Voting MILP on the same receiver views. This is intentionally optional "
            "because MILP is slower than the default three methods."
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
        include_milp=args.include_milp,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "scaling_comparison_raw.csv")
    print(DATA_DIR / "scaling_comparison_summary.csv")
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
