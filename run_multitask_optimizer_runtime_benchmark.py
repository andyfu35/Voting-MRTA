from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_multitask_peer_cost_experiment import (
    PACKET_LOSS_RATE,
    RANDOM_SEED,
    generate_spatial_cost_matrix,
)
from run_multitask_peer_cost_all_optimizers import (
    DEFAULT_VOTING_METHODS,
    FIXED_ROBOT_COUNT,
    METHOD_LABELS,
    VOTER_SELECTION_SEED_OFFSET,
    VISIBILITY_SEED_OFFSET,
    WORKLOAD_TASK_COUNTS,
    build_voter_batch_cost_views,
    resolve_robot_capacity,
    resolve_voter_count,
    sample_voter_batch_visibility,
    select_voter_indices,
    solve_voter_batch_proposals,
)

TIMING_TRIALS = 5
DEFAULT_VOTER_BATCH_SIZE = 4
METHOD_ORDER_SEED_OFFSET = 8_000_021

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "multitask_peer_cost_fixed100_workload" / "data"
RAW_RUNTIME_CSV = DATA_DIR / "optimizer_runtime_raw.csv"
SUMMARY_RUNTIME_CSV = DATA_DIR / "optimizer_runtime_summary.csv"
OVERALL_RUNTIME_CSV = DATA_DIR / "optimizer_runtime_overall.csv"


def fail(function: str, category: str, code: str, details: str) -> None:
    """Raise one diagnostic at the first runtime-benchmark owner boundary."""
    raise ValueError(
        "owner=run_multitask_optimizer_runtime_benchmark "
        f"function={function} category={category} code={code} details={details}"
    )


def validate_timing_config(
    *,
    task_counts: tuple[int, ...],
    trials: int,
    packet_loss_rate: float,
    max_voters: int | None,
    voter_batch_size: int,
) -> None:
    """Validate timing-benchmark inputs without changing Experiment 2 semantics."""
    if not task_counts:
        fail("validate_timing_config", "contract", "EMPTY_TASK_COUNTS", "task_counts is empty")
    if any(task_count <= 0 or task_count > 1000 for task_count in task_counts):
        fail(
            "validate_timing_config",
            "contract",
            "INVALID_TASK_COUNT",
            f"expected=1..1000 actual={task_counts}",
        )
    if trials <= 0:
        fail("validate_timing_config", "contract", "INVALID_TRIALS", f"actual={trials}")
    if not 0.0 <= packet_loss_rate < 1.0:
        fail(
            "validate_timing_config",
            "contract",
            "INVALID_MESSAGE_LOSS",
            f"expected=[0,1) actual={packet_loss_rate}",
        )
    if max_voters is not None and max_voters <= 0:
        fail(
            "validate_timing_config",
            "contract",
            "INVALID_MAX_VOTERS",
            f"expected>=1 actual={max_voters}",
        )
    if voter_batch_size <= 0:
        fail(
            "validate_timing_config",
            "contract",
            "INVALID_VOTER_BATCH_SIZE",
            f"expected>=1 actual={voter_batch_size}",
        )


def rotate_method_order(
    methods: tuple[str, ...],
    *,
    trial_seed: int,
    batch_index: int,
) -> tuple[str, ...]:
    """Rotate the timed method order deterministically to reduce fixed-order timing bias."""
    if not methods:
        fail("rotate_method_order", "contract", "EMPTY_METHOD_SET", "methods is empty")
    offset = (trial_seed + METHOD_ORDER_SEED_OFFSET + batch_index) % len(methods)
    return methods[offset:] + methods[:offset]


def measure_optimizer_batch_runtime(
    *,
    method: str,
    receiver_costs: np.ndarray,
    task_order: np.ndarray,
    capacity_per_robot: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Measure only the receiver-local optimizer route with a monotonic high-resolution clock."""
    start = time.perf_counter()
    proposals, valid = solve_voter_batch_proposals(
        method=method,
        receiver_costs=receiver_costs,
        task_order=task_order,
        capacity_per_robot=capacity_per_robot,
    )
    elapsed = time.perf_counter() - start
    if not np.isfinite(elapsed) or elapsed < 0.0:
        fail(
            "measure_optimizer_batch_runtime",
            "time",
            "INVALID_MEASURED_RUNTIME",
            f"method={method} elapsed={elapsed}",
        )
    return proposals, valid, elapsed


def warm_up_optimizer_paths(seed: int) -> None:
    """Exercise every optimizer once before measurement so one-time setup is not reported."""
    task_count = 100
    rng = np.random.default_rng(seed + 771_019)
    costs = generate_spatial_cost_matrix(FIXED_ROBOT_COUNT, task_count, rng)
    task_order = rng.permutation(task_count)
    capacity_per_robot = resolve_robot_capacity(task_count)
    receiver_costs = costs[None, :, :]
    for method in DEFAULT_VOTING_METHODS:
        proposals, valid = solve_voter_batch_proposals(
            method=method,
            receiver_costs=receiver_costs,
            task_order=task_order,
            capacity_per_robot=capacity_per_robot,
        )
        if proposals.shape != (1, task_count) or valid.shape != (1,):
            fail(
                "warm_up_optimizer_paths",
                "contract",
                "WARMUP_OUTPUT_SHAPE_MISMATCH",
                f"method={method} proposals={proposals.shape} valid={valid.shape}",
            )


def run_timing_trial(
    *,
    task_count: int,
    trial: int,
    packet_loss_rate: float,
    seed: int,
    max_voters: int | None,
    voter_batch_size: int,
) -> list[dict[str, object]]:
    """Measure all four optimizer families on the same receiver-local views for one trial."""
    robot_count = FIXED_ROBOT_COUNT
    capacity_per_robot = resolve_robot_capacity(task_count, robot_count)
    voter_count = resolve_voter_count(robot_count, max_voters)
    trial_seed = seed + task_count * 100_003 + trial * 1_009

    scenario_rng = np.random.default_rng(trial_seed)
    costs = generate_spatial_cost_matrix(robot_count, task_count, scenario_rng)
    task_order = scenario_rng.permutation(task_count)

    voter_rng = np.random.default_rng(trial_seed + VOTER_SELECTION_SEED_OFFSET)
    voter_indices = select_voter_indices(
        robot_count=robot_count,
        voter_count=voter_count,
        rng=voter_rng,
    )
    visibility_rng = np.random.default_rng(trial_seed + VISIBILITY_SEED_OFFSET)

    runtime_seconds = {method: 0.0 for method in DEFAULT_VOTING_METHODS}
    valid_counts = {method: 0 for method in DEFAULT_VOTING_METHODS}

    for batch_index, start in enumerate(range(0, voter_count, voter_batch_size)):
        receiver_batch = voter_indices[start : start + voter_batch_size]
        visibility = sample_voter_batch_visibility(
            robot_count=robot_count,
            task_count=task_count,
            packet_loss_rate=packet_loss_rate,
            receiver_indices=receiver_batch,
            rng=visibility_rng,
        )
        receiver_costs = build_voter_batch_cost_views(costs, visibility)

        timed_methods = rotate_method_order(
            DEFAULT_VOTING_METHODS,
            trial_seed=trial_seed,
            batch_index=batch_index,
        )
        for method in timed_methods:
            _, valid, elapsed = measure_optimizer_batch_runtime(
                method=method,
                receiver_costs=receiver_costs,
                task_order=task_order,
                capacity_per_robot=capacity_per_robot,
            )
            runtime_seconds[method] += elapsed
            valid_counts[method] += int(valid.sum())

    records: list[dict[str, object]] = []
    for method in DEFAULT_VOTING_METHODS:
        total_seconds = runtime_seconds[method]
        records.append(
            {
                "robots": robot_count,
                "voters": voter_count,
                "tasks": task_count,
                "capacity_per_robot": capacity_per_robot,
                "message_loss_percent": 100.0 * packet_loss_rate,
                "trial": trial,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "local_optimizer_runtime_seconds": total_seconds,
                "local_optimizer_runtime_ms_per_voter": 1000.0 * total_seconds / voter_count,
                "valid_proposal_rate_percent": 100.0 * valid_counts[method] / voter_count,
            }
        )
    return records


def summarize_timings(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build task-specific and overall per-voter runtime summaries for the four optimizers."""
    summary = (
        raw.groupby(
            ["robots", "voters", "tasks", "capacity_per_robot", "method", "method_label"],
            as_index=False,
        )
        .agg(
            average_local_optimizer_runtime_seconds_per_trial=(
                "local_optimizer_runtime_seconds",
                "mean",
            ),
            average_local_optimizer_runtime_ms_per_voter=(
                "local_optimizer_runtime_ms_per_voter",
                "mean",
            ),
            average_valid_proposal_rate_percent=("valid_proposal_rate_percent", "mean"),
        )
        .sort_values(["tasks", "method"])
        .reset_index(drop=True)
    )
    overall = (
        raw.groupby(["method", "method_label"], as_index=False)
        .agg(
            average_local_optimizer_runtime_ms_per_voter=(
                "local_optimizer_runtime_ms_per_voter",
                "mean",
            ),
            median_local_optimizer_runtime_ms_per_voter=(
                "local_optimizer_runtime_ms_per_voter",
                "median",
            ),
            average_valid_proposal_rate_percent=("valid_proposal_rate_percent", "mean"),
        )
        .sort_values("average_local_optimizer_runtime_ms_per_voter")
        .reset_index(drop=True)
    )
    return summary, overall


def runtime_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a task-by-method table of mean local optimizer milliseconds per voter."""
    table = summary.pivot(
        index="tasks",
        columns="method_label",
        values="average_local_optimizer_runtime_ms_per_voter",
    )
    labels = [METHOD_LABELS[method] for method in DEFAULT_VOTING_METHODS]
    return table.reindex(columns=labels).reset_index()


def save_timing_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    """Persist runtime measurements separately from the formal assignment-quality outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_RUNTIME_CSV, index=False)
    summary.to_csv(SUMMARY_RUNTIME_CSV, index=False)
    overall.to_csv(OVERALL_RUNTIME_CSV, index=False)


def run_benchmark(
    *,
    task_counts: tuple[int, ...] = WORKLOAD_TASK_COUNTS,
    trials: int = TIMING_TRIALS,
    packet_loss_rate: float = PACKET_LOSS_RATE,
    seed: int = RANDOM_SEED,
    max_voters: int | None = None,
    voter_batch_size: int = DEFAULT_VOTER_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the separate paired runtime benchmark without changing formal quality results."""
    validate_timing_config(
        task_counts=task_counts,
        trials=trials,
        packet_loss_rate=packet_loss_rate,
        max_voters=max_voters,
        voter_batch_size=voter_batch_size,
    )
    warm_up_optimizer_paths(seed)

    voter_text = "all 100 robots" if max_voters is None else f"up to {max_voters} sampled robots"
    print("Optimizer runtime benchmark: local solve path only")
    print("Excludes message sampling, Voting support/consensus, Oracle, plotting, and file I/O")
    print(f"Task batches: {' '.join(str(value) for value in task_counts)}")
    print(f"Trials per task point: {trials}; voters: {voter_text}")
    print("Methods: " + ", ".join(METHOD_LABELS[method] for method in DEFAULT_VOTING_METHODS))

    records: list[dict[str, object]] = []
    total = len(task_counts) * trials
    completed = 0
    for task_count in task_counts:
        for trial in range(1, trials + 1):
            records.extend(
                run_timing_trial(
                    task_count=task_count,
                    trial=trial,
                    packet_loss_rate=packet_loss_rate,
                    seed=seed,
                    max_voters=max_voters,
                    voter_batch_size=voter_batch_size,
                )
            )
            completed += 1
            print(
                f"completed={completed:3d}/{total} tasks={task_count:4d} trial={trial:2d}",
                flush=True,
            )

    raw = pd.DataFrame.from_records(records)
    summary, overall = summarize_timings(raw)
    save_timing_outputs(raw, summary, overall)
    return raw, summary, overall


def parse_task_counts(values: list[int] | None) -> tuple[int, ...]:
    """Resolve optional CLI task counts to the canonical workload grid."""
    return WORKLOAD_TASK_COUNTS if values is None else tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-voter local optimizer compute time for the four canonical Experiment 2 "
            "methods on paired receiver-local views."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=None,
        help="Task batches. Default: 100 200 ... 1000.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TIMING_TRIALS,
        help=f"Paired timing trials per task point. Default: {TIMING_TRIALS}.",
    )
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=PACKET_LOSS_RATE,
        help="Directed peer-to-peer message-loss probability. Default: 0.30.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-voters",
        type=int,
        default=None,
        help="Optional quick-test cap. Default: all 100 voters.",
    )
    parser.add_argument(
        "--voter-batch-size",
        type=int,
        default=DEFAULT_VOTER_BATCH_SIZE,
        help=f"Receiver batch size used by the timed owner route. Default: {DEFAULT_VOTER_BATCH_SIZE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary, overall = run_benchmark(
        task_counts=parse_task_counts(args.tasks),
        trials=args.trials,
        packet_loss_rate=args.packet_loss,
        seed=args.seed,
        max_voters=args.max_voters,
        voter_batch_size=args.voter_batch_size,
    )

    print("\nAverage local optimizer compute time per voter (ms):")
    print(
        runtime_report_table(summary).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nOverall mean across the selected task points (ms per voter):")
    print(
        overall[["method_label", "average_local_optimizer_runtime_ms_per_voter"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nSaved:")
    print(RAW_RUNTIME_CSV)
    print(SUMMARY_RUNTIME_CSV)
    print(OVERALL_RUNTIME_CSV)


if __name__ == "__main__":
    main()
