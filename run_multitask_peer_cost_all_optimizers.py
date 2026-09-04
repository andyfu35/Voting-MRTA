from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_multitask_optimizer_screening import (
    ACOConfig,
    MILP_NUMERICAL_TOLERANCE_PERCENT,
    solve_aco_assignment,
    solve_milp_assignment,
    validate_aco_config,
)
from run_multitask_peer_cost_experiment import (
    DEFAULT_TRIALS,
    NEAR_OPTIMAL_GAP_PERCENT,
    OPTIMAL_COST_TOLERANCE_PERCENT,
    PACKET_LOSS_RATE,
    RANDOM_SEED,
    ROBOT_COUNT,
    TASK_COUNTS,
    assignment_total_cost,
    build_assignment_support,
    build_receiver_cost_views,
    generate_spatial_cost_matrix,
    sample_p2p_cost_visibility,
    solve_hungarian_assignment,
    solve_local_optimizer_proposals,
    solve_support_consensus,
    validate_experiment_config,
)

METHODS = (
    "oracle",
    "p2p_greedy",
    "p2p_hungarian",
    "p2p_auction",
    "p2p_milp",
    "p2p_aco_ls",
)
METHOD_LABELS = {
    "oracle": "Hungarian Oracle",
    "p2p_greedy": "Voting Greedy",
    "p2p_hungarian": "Voting Hungarian",
    "p2p_auction": "Voting Auction",
    "p2p_milp": "Voting MILP",
    "p2p_aco_ls": "Voting ACO + Local Search",
}
EXISTING_P2P_METHODS = ("p2p_greedy", "p2p_hungarian", "p2p_auction")
PARALLEL_RECEIVER_METHODS = ("p2p_milp", "p2p_aco_ls")
EXACT_P2P_METHODS = ("p2p_hungarian", "p2p_auction", "p2p_milp")
ACO_SEED_OFFSET = 7_000_003
ACO_RECEIVER_SEED_STEP = 10_000_019
DEFAULT_PARALLEL_WORKERS = min(4, max(1, os.cpu_count() or 1))
DEFAULT_PROGRESS_EVERY_RECEIVERS = 25

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "multitask_peer_cost_all_optimizers"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first experiment-owner boundary."""
    raise ValueError(
        "owner=run_multitask_peer_cost_all_optimizers "
        f"function={function} category={category} code={code} details={details}"
    )


def cost_match_tolerance_percent(method: str) -> float:
    """Return the objective-match tolerance owned by each optimizer family."""
    if method == "p2p_milp":
        return MILP_NUMERICAL_TOLERANCE_PERCENT
    if method in METHODS:
        return OPTIMAL_COST_TOLERANCE_PERCENT
    fail("cost_match_tolerance_percent", "contract", "UNKNOWN_METHOD", f"method={method}")


def validate_parallel_config(parallel_workers: int, progress_every_receivers: int) -> None:
    """Validate receiver-parallel runtime controls without changing experiment semantics."""
    if parallel_workers <= 0:
        fail(
            "validate_parallel_config",
            "contract",
            "INVALID_PARALLEL_WORKERS",
            f"expected>=1 actual={parallel_workers}",
        )
    if progress_every_receivers <= 0:
        fail(
            "validate_parallel_config",
            "contract",
            "INVALID_PROGRESS_INTERVAL",
            f"expected>=1 actual={progress_every_receivers}",
        )


def aco_receiver_seed(*, seed: int, task_count: int, trial: int, receiver: int) -> int:
    """Derive a deterministic ACO search stream without consuming scenario RNG."""
    return (
        seed
        + ACO_SEED_OFFSET
        + task_count * 100_003
        + trial * 1_009
        + receiver * ACO_RECEIVER_SEED_STEP
    )


def validate_local_proposal(
    *,
    method: str,
    receiver: int,
    local_costs: np.ndarray,
    proposal: np.ndarray,
) -> None:
    """Reject a solver output that violates the receiver-local information boundary."""
    task_count = local_costs.shape[1]
    if proposal.shape != (task_count,):
        fail(
            "validate_local_proposal",
            "contract",
            "PROPOSAL_SHAPE_MISMATCH",
            f"method={method} receiver={receiver} expected={(task_count,)} actual={proposal.shape}",
        )
    if np.any(proposal < 0) or np.any(proposal >= local_costs.shape[0]):
        fail(
            "validate_local_proposal",
            "state",
            "INVALID_PROPOSAL_ROBOT",
            f"method={method} receiver={receiver}",
        )
    if len(np.unique(proposal)) != task_count:
        fail(
            "validate_local_proposal",
            "state",
            "PROPOSAL_CAPACITY_VIOLATION",
            f"method={method} receiver={receiver}",
        )
    selected = local_costs[proposal, np.arange(task_count)]
    if np.any(~np.isfinite(selected)):
        fail(
            "validate_local_proposal",
            "planning",
            "UNAVAILABLE_EDGE_SELECTED",
            f"method={method} receiver={receiver}",
        )


def solve_receiver_local_proposal(
    *,
    method: str,
    receiver: int,
    local_costs: np.ndarray,
    task_order: np.ndarray,
    seed: int,
    task_count: int,
    trial: int,
    aco_config: ACOConfig,
) -> tuple[int, np.ndarray | None]:
    """Solve one heavy receiver-local MILP or ACO proposal in an isolated worker."""
    if method == "p2p_milp":
        proposal = solve_milp_assignment(local_costs)
    elif method == "p2p_aco_ls":
        local_rng = np.random.default_rng(
            aco_receiver_seed(
                seed=seed,
                task_count=task_count,
                trial=trial,
                receiver=receiver,
            )
        )
        proposal = solve_aco_assignment(local_costs, task_order, local_rng, aco_config)
    else:
        fail(
            "solve_receiver_local_proposal",
            "contract",
            "UNKNOWN_PARALLEL_METHOD",
            f"method={method}",
        )
    return receiver, proposal


def store_receiver_proposal(
    *,
    method: str,
    receiver: int,
    receiver_costs: np.ndarray,
    proposal: np.ndarray | None,
    proposals: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Validate and store one receiver proposal without repairing invalid results."""
    if proposal is None:
        return
    validate_local_proposal(
        method=method,
        receiver=receiver,
        local_costs=receiver_costs[receiver],
        proposal=proposal,
    )
    proposals[receiver] = proposal
    valid[receiver] = True


def report_receiver_progress(
    *,
    method: str,
    task_count: int,
    trial: int,
    completed: int,
    total: int,
) -> None:
    """Show liveness for heavy receiver-local solves without affecting solver state."""
    end = "\n" if completed >= total else "\r"
    print(
        f"tasks={task_count:3d} trial={trial:3d} "
        f"{METHOD_LABELS[method]} receivers={completed:3d}/{total}",
        end=end,
        flush=True,
    )


def solve_serial_receiver_proposals(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    seed: int,
    task_count: int,
    trial: int,
    aco_config: ACOConfig,
    progress_every_receivers: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference serial execution for heavy receiver-local optimizer proposals."""
    receiver_count = receiver_costs.shape[0]
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    for receiver in range(receiver_count):
        resolved_receiver, proposal = solve_receiver_local_proposal(
            method=method,
            receiver=receiver,
            local_costs=receiver_costs[receiver],
            task_order=task_order,
            seed=seed,
            task_count=task_count,
            trial=trial,
            aco_config=aco_config,
        )
        store_receiver_proposal(
            method=method,
            receiver=resolved_receiver,
            receiver_costs=receiver_costs,
            proposal=proposal,
            proposals=proposals,
            valid=valid,
        )
        completed = receiver + 1
        if show_progress and (
            completed % progress_every_receivers == 0 or completed == receiver_count
        ):
            report_receiver_progress(
                method=method,
                task_count=task_count,
                trial=trial,
                completed=completed,
                total=receiver_count,
            )
    return proposals, valid


def solve_parallel_receiver_proposals(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    seed: int,
    task_count: int,
    trial: int,
    aco_config: ACOConfig,
    executor: ProcessPoolExecutor,
    progress_every_receivers: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute independent receiver-local MILP/ACO solves across persistent processes."""
    if method not in PARALLEL_RECEIVER_METHODS:
        fail(
            "solve_parallel_receiver_proposals",
            "contract",
            "UNKNOWN_PARALLEL_METHOD",
            f"method={method}",
        )

    receiver_count = receiver_costs.shape[0]
    proposals = np.full((receiver_count, task_count), -1, dtype=int)
    valid = np.zeros(receiver_count, dtype=bool)
    future_receivers = {}
    for receiver in range(receiver_count):
        future = executor.submit(
            solve_receiver_local_proposal,
            method=method,
            receiver=receiver,
            local_costs=receiver_costs[receiver],
            task_order=task_order,
            seed=seed,
            task_count=task_count,
            trial=trial,
            aco_config=aco_config,
        )
        future_receivers[future] = receiver

    completed = 0
    for future in as_completed(future_receivers):
        expected_receiver = future_receivers[future]
        try:
            receiver, proposal = future.result()
        except ValueError:
            raise
        except Exception as exc:
            fail(
                "solve_parallel_receiver_proposals",
                "runtime",
                "PARALLEL_RECEIVER_SOLVE_FAILED",
                (
                    f"method={method} tasks={task_count} trial={trial} "
                    f"receiver={expected_receiver} error={type(exc).__name__}:{exc}"
                ),
            )
        if receiver != expected_receiver:
            fail(
                "solve_parallel_receiver_proposals",
                "state",
                "WORKER_RECEIVER_MISMATCH",
                f"expected={expected_receiver} actual={receiver}",
            )
        store_receiver_proposal(
            method=method,
            receiver=receiver,
            receiver_costs=receiver_costs,
            proposal=proposal,
            proposals=proposals,
            valid=valid,
        )
        completed += 1
        if show_progress and (
            completed % progress_every_receivers == 0 or completed == receiver_count
        ):
            report_receiver_progress(
                method=method,
                task_count=task_count,
                trial=trial,
                completed=completed,
                total=receiver_count,
            )
    return proposals, valid


def solve_extended_local_optimizer_proposals(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    seed: int,
    task_count: int,
    trial: int,
    aco_config: ACOConfig,
    executor: ProcessPoolExecutor | None,
    progress_every_receivers: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Route each optimizer family to its existing owner and runtime boundary."""
    if method in EXISTING_P2P_METHODS:
        return solve_local_optimizer_proposals(method, receiver_costs, task_order)

    if receiver_costs.ndim != 3:
        fail(
            "solve_extended_local_optimizer_proposals",
            "contract",
            "INVALID_RECEIVER_COST_SHAPE",
            f"shape={receiver_costs.shape}",
        )
    _, _, actual_task_count = receiver_costs.shape
    if actual_task_count != task_count:
        fail(
            "solve_extended_local_optimizer_proposals",
            "contract",
            "TASK_COUNT_MISMATCH",
            f"expected={task_count} actual={actual_task_count}",
        )
    if method not in PARALLEL_RECEIVER_METHODS:
        fail(
            "solve_extended_local_optimizer_proposals",
            "contract",
            "UNKNOWN_METHOD",
            f"method={method}",
        )

    if executor is None:
        return solve_serial_receiver_proposals(
            method=method,
            receiver_costs=receiver_costs,
            task_order=task_order,
            seed=seed,
            task_count=task_count,
            trial=trial,
            aco_config=aco_config,
            progress_every_receivers=progress_every_receivers,
            show_progress=show_progress,
        )
    return solve_parallel_receiver_proposals(
        method=method,
        receiver_costs=receiver_costs,
        task_order=task_order,
        seed=seed,
        task_count=task_count,
        trial=trial,
        aco_config=aco_config,
        executor=executor,
        progress_every_receivers=progress_every_receivers,
        show_progress=show_progress,
    )


def optimizer_consensus_assignment(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    tie_priority: np.ndarray,
    seed: int,
    task_count: int,
    trial: int,
    aco_config: ACOConfig,
    executor: ProcessPoolExecutor | None,
    progress_every_receivers: int,
    show_progress: bool,
) -> tuple[np.ndarray, float]:
    """Run one optimizer proposal batch through the shared Voting consensus."""
    proposals, valid = solve_extended_local_optimizer_proposals(
        method=method,
        receiver_costs=receiver_costs,
        task_order=task_order,
        seed=seed,
        task_count=task_count,
        trial=trial,
        aco_config=aco_config,
        executor=executor,
        progress_every_receivers=progress_every_receivers,
        show_progress=show_progress,
    )
    support = build_assignment_support(proposals, valid, receiver_costs.shape[1])
    assignment = solve_support_consensus(support, tie_priority)
    return assignment, 100.0 * float(valid.mean())


def validate_zero_loss_optimizer_contract(
    seed: int,
    aco_config: ACOConfig,
    executor: ProcessPoolExecutor | None,
    progress_every_receivers: int,
) -> None:
    """Require exact optimizer families to recover the oracle with complete data."""
    rng = np.random.default_rng(seed + 99_173)
    for task_count in (1, 5, 50, 100):
        costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
        visibility = np.ones((ROBOT_COUNT, ROBOT_COUNT, task_count), dtype=bool)
        task_order = rng.permutation(task_count)
        tie_priority = rng.random((ROBOT_COUNT, task_count))
        receiver_costs = build_receiver_cost_views(costs, visibility)

        oracle = solve_hungarian_assignment(costs)
        if oracle is None:
            fail(
                "validate_zero_loss_optimizer_contract",
                "planning",
                "ORACLE_INFEASIBLE",
                f"tasks={task_count}",
            )
        oracle_cost = assignment_total_cost(costs, oracle)
        exact_methods = EXACT_P2P_METHODS
        if task_count == 1:
            exact_methods += ("p2p_greedy",)

        for method in exact_methods:
            assignment, valid_rate = optimizer_consensus_assignment(
                method=method,
                receiver_costs=receiver_costs,
                task_order=task_order,
                tie_priority=tie_priority,
                seed=seed,
                task_count=task_count,
                trial=0,
                aco_config=aco_config,
                executor=executor,
                progress_every_receivers=progress_every_receivers,
                show_progress=False,
            )
            if valid_rate != 100.0:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_PROPOSAL_FAILURE",
                    f"method={method} tasks={task_count} valid_rate={valid_rate}",
                )
            actual_cost = assignment_total_cost(costs, assignment)
            gap_percent = 100.0 * (actual_cost - oracle_cost) / oracle_cost
            tolerance = cost_match_tolerance_percent(method)
            if abs(gap_percent) > tolerance:
                fail(
                    "validate_zero_loss_optimizer_contract",
                    "planning",
                    "ZERO_LOSS_NOT_ORACLE_CONSISTENT",
                    (
                        f"method={method} tasks={task_count} "
                        f"expected_abs_gap_percent<={tolerance} actual={gap_percent}"
                    ),
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
    """Evaluate one Voting result against the full-information Hungarian oracle."""
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
    rng: np.random.Generator,
    seed: int,
    aco_config: ACOConfig,
    executor: ProcessPoolExecutor | None,
    progress_every_receivers: int,
) -> list[dict[str, object]]:
    """Generate one paired P2P scenario once and run every optimizer on it."""
    costs = generate_spatial_cost_matrix(ROBOT_COUNT, task_count, rng)
    visibility = sample_p2p_cost_visibility(ROBOT_COUNT, task_count, packet_loss_rate, rng)
    task_order = rng.permutation(task_count)
    tie_priority = rng.random((ROBOT_COUNT, task_count))
    receiver_costs = build_receiver_cost_views(costs, visibility)

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
    for method in METHODS[1:]:
        assignment, valid_rate = optimizer_consensus_assignment(
            method=method,
            receiver_costs=receiver_costs,
            task_order=task_order,
            tie_priority=tie_priority,
            seed=seed,
            task_count=task_count,
            trial=trial,
            aco_config=aco_config,
            executor=executor,
            progress_every_receivers=progress_every_receivers,
            show_progress=method in PARALLEL_RECEIVER_METHODS,
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


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quality and local-proposal validity over paired trials."""
    return (
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


def create_receiver_executor(parallel_workers: int) -> ProcessPoolExecutor | None:
    """Create the persistent process pool used only for independent heavy receiver solves."""
    if parallel_workers == 1:
        return None
    try:
        return ProcessPoolExecutor(max_workers=parallel_workers)
    except Exception as exc:
        fail(
            "create_receiver_executor",
            "runtime",
            "PARALLEL_EXECUTOR_START_FAILED",
            f"workers={parallel_workers} error={type(exc).__name__}:{exc}",
        )


def shutdown_receiver_executor(executor: ProcessPoolExecutor | None) -> None:
    """Shut down the persistent receiver process pool after the sweep or first failure."""
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def run_experiment(
    *,
    task_counts: tuple[int, ...] = TASK_COUNTS,
    trials: int = DEFAULT_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
    aco_config: ACOConfig = ACOConfig(),
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    progress_every_receivers: int = DEFAULT_PROGRESS_EVERY_RECEIVERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the paired 100-robot all-optimizer Voting sweep."""
    validate_experiment_config(ROBOT_COUNT, packet_loss_rate, task_counts, trials)
    validate_aco_config(aco_config)
    validate_parallel_config(parallel_workers, progress_every_receivers)

    executor = create_receiver_executor(parallel_workers)
    try:
        validate_zero_loss_optimizer_contract(
            seed,
            aco_config,
            executor,
            progress_every_receivers,
        )
        print(
            "Zero-loss optimizer contract: PASS "
            "(Hungarian/Auction/MILP match oracle; single-task Greedy matches oracle)"
        )
        print(
            f"Receiver-local runtime: workers={parallel_workers} "
            f"(MILP/ACO only; scenario RNG and solver parameters unchanged)"
        )

        records: list[dict[str, object]] = []
        for task_count in task_counts:
            # Keep the exact legacy P2P scenario RNG schedule so the existing
            # Greedy/Hungarian/Auction columns remain directly reproducible.
            rng = np.random.default_rng(seed + task_count * 100_003)
            for trial in range(1, trials + 1):
                print(
                    f"tasks={task_count:3d}/{max(task_counts)} trial={trial:3d}/{trials} start",
                    flush=True,
                )
                records.extend(
                    run_trial(
                        task_count=task_count,
                        trial=trial,
                        packet_loss_rate=packet_loss_rate,
                        rng=rng,
                        seed=seed,
                        aco_config=aco_config,
                        executor=executor,
                        progress_every_receivers=progress_every_receivers,
                    )
                )
            print(f"tasks={task_count:3d}/{max(task_counts)} complete")

        raw = pd.DataFrame.from_records(records)
        return raw, summarize_results(raw)
    finally:
        shutdown_receiver_executor(executor)


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def report_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one task-by-method table in canonical report order."""
    table = summary.pivot(index="tasks", columns="method_label", values=metric)
    labels = [METHOD_LABELS[method] for method in METHODS]
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
    """Persist raw data, report tables, and report figures."""
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
            "Compare Greedy, Hungarian, Auction, MILP, and ACO+Local Search inside the same "
            "lossy P2P Voting experiment using paired cost and packet-loss scenarios."
        )
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=None)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=PACKET_LOSS_RATE,
        help="Directed P2P loss probability (default: 0.30).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=(
            "Process workers for receiver-local MILP/ACO solves. "
            f"Default: {DEFAULT_PARALLEL_WORKERS}; use 1 for serial regression."
        ),
    )
    parser.add_argument(
        "--progress-every-receivers",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_RECEIVERS,
        help="Print MILP/ACO receiver progress every N completed solves (default: 25).",
    )
    parser.add_argument("--aco-ants", type=int, default=ACOConfig.ants)
    parser.add_argument("--aco-iterations", type=int, default=ACOConfig.iterations)
    parser.add_argument("--aco-alpha", type=float, default=ACOConfig.alpha)
    parser.add_argument("--aco-beta", type=float, default=ACOConfig.beta)
    parser.add_argument("--aco-evaporation", type=float, default=ACOConfig.evaporation)
    parser.add_argument("--aco-candidate-list", type=int, default=ACOConfig.candidate_list_size)
    parser.add_argument("--aco-elite-weight", type=float, default=ACOConfig.elite_weight)
    parser.add_argument("--aco-local-search-moves", type=int, default=ACOConfig.local_search_moves)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_counts = parse_task_counts(args.tasks)
    aco_config = ACOConfig(
        ants=args.aco_ants,
        iterations=args.aco_iterations,
        alpha=args.aco_alpha,
        beta=args.aco_beta,
        evaporation=args.aco_evaporation,
        candidate_list_size=args.aco_candidate_list,
        elite_weight=args.aco_elite_weight,
        local_search_moves=args.aco_local_search_moves,
    )
    raw, summary = run_experiment(
        task_counts=task_counts,
        trials=args.trials,
        packet_loss_rate=args.packet_loss,
        seed=args.seed,
        aco_config=aco_config,
        parallel_workers=args.workers,
        progress_every_receivers=args.progress_every_receivers,
    )
    save_outputs(raw, summary)

    print("\nSaved:")
    print(DATA_DIR / "optimizer_comparison_raw.csv")
    print(DATA_DIR / "optimizer_comparison_summary.csv")
    print(FIGURE_DIR)

    print("\nAverage optimality gap (%) - lower is better:")
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

    print("\nNear-optimal within 5% (% of trials) - higher is better:")
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
