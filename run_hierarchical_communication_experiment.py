from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import sample_active_robots
from run_decentralized_experiment import assign_sender_policies, required_quorum
from run_execution_simulation import (
    SERVICE_NOISE_SIGMA,
    TRAVEL_NOISE_SIGMA,
    build_base_cost,
    generate_scenario,
    lognormal_multipliers,
    simulate_execution,
)
from run_multitask_experiment import per_task_probability
from run_route_guardrail_experiment import guardrail_route_plan, route_dict


RANDOM_SEED = 20260814
ROBOT_COUNTS = (10, 20, 40, 60, 80, 100)
TASK_LOAD_RATIO = 1.00
TRIALS = 30
PACKET_LOSS_RATE = 0.30
MAX_ATTEMPTS = 3
ROBOT_GROUP_SIZE = 5
ROUTE_GUARDRAIL_TOLERANCE = 0.20
QUORUM_THRESHOLD = 2.0 / 3.0

# Synthetic communication-time model. This is intentionally reported as a
# proxy, not wall-clock runtime: all traffic is assumed to share one 100 Mbps
# channel and each sequential communication stage adds 5 ms of propagation /
# coordination delay.
NETWORK_BANDWIDTH_MBPS = 100.0
PER_STAGE_LATENCY_MS = 5.0
FLOAT_BYTES = 8
MESSAGE_METADATA_BYTES = 64

METHODS = (
    "flat_p2p",
    "hierarchical_5ary",
    "hierarchical_5ary_backup",
)
METHOD_LABELS = {
    "flat_p2p": "Flat P2P",
    "hierarchical_5ary": "5-ary Hierarchy",
    "hierarchical_5ary_backup": "5-ary + Backup Relay",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "hierarchical"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


@dataclass
class AggregateNode:
    leader: int
    counts: np.ndarray
    leaves: tuple[int, ...]


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def send_with_retries(
    rng: np.random.Generator,
    loss_rate: float = PACKET_LOSS_RATE,
) -> tuple[bool, int]:
    """One logical point-to-point message with stop-on-success retransmission."""
    attempts = 0
    for _ in range(MAX_ATTEMPTS):
        attempts += 1
        if rng.random() >= loss_rate:
            return True, attempts
    return False, attempts


def probability_support_from_counts(
    counts: np.ndarray,
    policies: list[str],
    probability_map: dict[str, np.ndarray],
) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        raise RuntimeError("At least one preference contribution is required")
    support = np.zeros_like(next(iter(probability_map.values())), dtype=float)
    for policy, count in zip(policies, counts, strict=True):
        if count > 0:
            support += (float(count) / total) * probability_map[policy]
    return support


def sender_policy_one_hot(
    active: np.ndarray,
    sender_policy: dict[int, str],
) -> tuple[list[str], np.ndarray]:
    policies = sorted(set(sender_policy.values()))
    policy_index = {policy: index for index, policy in enumerate(policies)}
    one_hot = np.zeros((len(active), len(policies)), dtype=float)
    for robot, policy in sender_policy.items():
        one_hot[int(robot), policy_index[policy]] = 1.0
    return policies, one_hot


def preference_payload_bytes(active_count: int, task_count: int) -> int:
    # Leaders forward an aggregated complete-preference matrix with the same
    # dimensions as one original active-robot preference matrix.
    return active_count * task_count * FLOAT_BYTES + MESSAGE_METADATA_BYTES


def communication_time_proxy_ms(total_bytes: float, stages: int) -> float:
    transfer_ms = total_bytes * 8.0 / (NETWORK_BANDWIDTH_MBPS * 1_000_000.0) * 1000.0
    return stages * PER_STAGE_LATENCY_MS + transfer_ms


def plan_statistics(
    plans: dict[int, tuple[tuple[int, ...], ...]],
    quorum_membership: int,
) -> tuple[tuple[tuple[int, ...], ...] | None, float, bool]:
    counts = Counter(plans.values())
    if not counts:
        return None, 0.0, False
    modal_plan, modal_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    modal_share = modal_count / quorum_membership
    safe_commit = modal_count >= required_quorum(quorum_membership, QUORUM_THRESHOLD)
    return modal_plan if safe_commit else None, modal_share, safe_commit


def plans_from_count_rows(
    count_rows: dict[int, np.ndarray],
    policies: list[str],
    probability_map: dict[str, np.ndarray],
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    plans: dict[int, tuple[tuple[int, ...], ...]] = {}
    cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    for receiver, counts in count_rows.items():
        key = tuple(int(value) for value in counts)
        if key not in cache:
            support = probability_support_from_counts(counts, policies, probability_map)
            cache[key] = guardrail_route_plan(
                scenario,
                base_cost,
                active,
                support,
                ROUTE_GUARDRAIL_TOLERANCE,
            )
        plans[int(receiver)] = cache[key]
    return plans


def flat_preference_exchange(
    active: np.ndarray,
    sender_one_hot: np.ndarray,
    rng: np.random.Generator,
    payload_bytes: int,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    active_indices = np.flatnonzero(active)
    active_count = len(active_indices)
    received: dict[int, np.ndarray] = {
        int(robot): sender_one_hot[int(robot)].copy() for robot in active_indices
    }
    logical_messages = 0
    attempts = 0

    for sender in active_indices:
        for receiver in active_indices:
            if sender == receiver:
                continue
            logical_messages += 1
            delivered, used = send_with_retries(rng)
            attempts += used
            if delivered:
                received[int(receiver)] += sender_one_hot[int(sender)]

    coverage = np.mean([float(row.sum()) / active_count for row in received.values()])
    bytes_sent = attempts * payload_bytes
    return received, {
        "logical_messages": float(logical_messages),
        "message_attempts": float(attempts),
        "bytes_transmitted": float(bytes_sent),
        "communication_stages": 1.0,
        "preference_coverage": float(coverage),
        "decision_makers": float(active_count),
    }


def aggregate_group(
    group: list[AggregateNode],
    *,
    trial: int,
    level: int,
    group_index: int,
    rng: np.random.Generator,
    payload_bytes: int,
    backup_enabled: bool,
) -> tuple[AggregateNode, int, int]:
    if len(group) == 1:
        return group[0], 0, 0

    shift = (trial + level + group_index) % len(group)
    primary = group[shift]
    backup = group[(shift + 1) % len(group)]
    aggregate = primary.counts.copy()
    all_leaves: list[int] = []
    for node in group:
        all_leaves.extend(node.leaves)

    logical_messages = 0
    attempts = 0

    for child in group:
        if child is primary:
            continue

        logical_messages += 1
        delivered, used = send_with_retries(rng)
        attempts += used
        if delivered:
            aggregate += child.counts
            continue

        if not backup_enabled:
            continue

        # Fallback path: if the child itself is the backup it already owns the
        # aggregate, otherwise first send it to the backup, then relay to primary.
        backup_has_child = child is backup
        if not backup_has_child:
            logical_messages += 1
            to_backup, used = send_with_retries(rng)
            attempts += used
            if not to_backup:
                continue

        logical_messages += 1
        relayed, used = send_with_retries(rng)
        attempts += used
        if relayed:
            aggregate += child.counts

    return (
        AggregateNode(
            leader=int(primary.leader),
            counts=aggregate,
            leaves=tuple(sorted(all_leaves)),
        ),
        logical_messages,
        attempts,
    )


def hierarchical_preference_exchange(
    active: np.ndarray,
    sender_one_hot: np.ndarray,
    *,
    trial: int,
    rng: np.random.Generator,
    payload_bytes: int,
    backup_enabled: bool,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    active_indices = np.flatnonzero(active)
    active_count = len(active_indices)
    nodes = [
        AggregateNode(
            leader=int(robot),
            counts=sender_one_hot[int(robot)].copy(),
            leaves=(int(robot),),
        )
        for robot in active_indices
    ]

    total_logical_messages = 0
    total_attempts = 0
    aggregation_levels = 0

    while len(nodes) > ROBOT_GROUP_SIZE:
        next_nodes: list[AggregateNode] = []
        for group_index, start in enumerate(range(0, len(nodes), ROBOT_GROUP_SIZE)):
            group = nodes[start : start + ROBOT_GROUP_SIZE]
            parent, logical, attempts = aggregate_group(
                group,
                trial=trial,
                level=aggregation_levels,
                group_index=group_index,
                rng=rng,
                payload_bytes=payload_bytes,
                backup_enabled=backup_enabled,
            )
            next_nodes.append(parent)
            total_logical_messages += logical
            total_attempts += attempts
        nodes = next_nodes
        aggregation_levels += 1

    # Keep the last <=5 leaders decentralized: they exchange their subtree
    # aggregates peer-to-peer rather than electing one permanent root.
    top_nodes = nodes
    top_counts: dict[int, np.ndarray] = {
        int(node.leader): node.counts.copy() for node in top_nodes
    }
    if len(top_nodes) > 1:
        for sender in top_nodes:
            for receiver in top_nodes:
                if sender is receiver:
                    continue
                total_logical_messages += 1
                delivered, used = send_with_retries(rng)
                total_attempts += used
                if delivered:
                    top_counts[int(receiver.leader)] += sender.counts

    coverage = np.mean(
        [float(counts.sum()) / active_count for counts in top_counts.values()]
    )
    stages = aggregation_levels + (1 if len(top_nodes) > 1 else 0)
    bytes_sent = total_attempts * payload_bytes
    return top_counts, {
        "logical_messages": float(total_logical_messages),
        "message_attempts": float(total_attempts),
        "bytes_transmitted": float(bytes_sent),
        "communication_stages": float(stages),
        "preference_coverage": float(coverage),
        "decision_makers": float(len(top_nodes)),
        "aggregation_levels": float(aggregation_levels),
        "top_leaders": float(len(top_nodes)),
    }


def execution_metrics(
    plan: tuple[tuple[int, ...], ...] | None,
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
    success_uniform: np.ndarray,
) -> dict[str, float]:
    if plan is None:
        return {
            "task_completion_rate": 0.0,
            "mission_success_rate": 0.0,
            "actual_makespan_sec": np.nan,
            "energy_per_completed_task": np.nan,
            "route_task_count_cv": np.nan,
            "duplicate_task_execution_rate": 0.0,
        }

    execution = simulate_execution(
        route_dict(plan, active),
        active,
        scenario,
        start_time_multiplier,
        transition_time_multiplier,
        service_time_multiplier,
        success_uniform,
    )
    task_count = len(scenario["service_times"])
    completed = float(execution["task_completion_rate"]) * task_count
    energy_per_completed = (
        float(execution["total_energy_units"]) / completed if completed > 1e-12 else np.nan
    )
    return {
        "task_completion_rate": float(execution["task_completion_rate"]),
        "mission_success_rate": float(bool(execution["mission_success"])),
        "actual_makespan_sec": float(execution["actual_makespan_sec"]),
        "energy_per_completed_task": energy_per_completed,
        "route_task_count_cv": float(execution["route_task_count_cv"]),
        "duplicate_task_execution_rate": float(
            execution["duplicate_task_execution_rate"]
        ),
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in ROBOT_COUNTS:
        for trial in range(TRIALS):
            rng = np.random.default_rng(RANDOM_SEED + n * 100000 + trial)
            active = sample_active_robots(n, rng)
            active_count = int(active.sum())
            task_count = max(1, int(round(active_count * TASK_LOAD_RATIO)))
            scenario = generate_scenario(n, task_count, rng)
            base_cost = build_base_cost(scenario, active)

            policy_rng = np.random.default_rng(RANDOM_SEED + 80000000 + n * 100000 + trial)
            sender_policy = assign_sender_policies(
                active,
                "heterogeneous_strong",
                policy_rng,
            )
            policies, one_hot = sender_policy_one_hot(active, sender_policy)
            probability_map = {
                policy: per_task_probability(base_cost, active, policy)
                for policy in policies
            }

            full_counts = one_hot.sum(axis=0)
            full_support = probability_support_from_counts(
                full_counts,
                policies,
                probability_map,
            )
            reference_plan = guardrail_route_plan(
                scenario,
                base_cost,
                active,
                full_support,
                ROUTE_GUARDRAIL_TOLERANCE,
            )

            start_time_multiplier = lognormal_multipliers(
                rng,
                TRAVEL_NOISE_SIGMA,
                (n, task_count),
            )
            transition_time_multiplier = lognormal_multipliers(
                rng,
                TRAVEL_NOISE_SIGMA,
                (task_count, task_count),
            )
            np.fill_diagonal(transition_time_multiplier, 1.0)
            service_time_multiplier = lognormal_multipliers(
                rng,
                SERVICE_NOISE_SIGMA,
                (task_count,),
            )
            success_uniform = rng.random((n, task_count))

            payload_bytes = preference_payload_bytes(active_count, task_count)

            for method in METHODS:
                method_rng = np.random.default_rng(
                    RANDOM_SEED
                    + 100000000
                    + n * 100000
                    + trial * 10
                    + METHODS.index(method)
                )

                if method == "flat_p2p":
                    count_rows, comm = flat_preference_exchange(
                        active,
                        one_hot,
                        method_rng,
                        payload_bytes,
                    )
                else:
                    count_rows, comm = hierarchical_preference_exchange(
                        active,
                        one_hot,
                        trial=trial,
                        rng=method_rng,
                        payload_bytes=payload_bytes,
                        backup_enabled=method == "hierarchical_5ary_backup",
                    )

                plans = plans_from_count_rows(
                    count_rows,
                    policies,
                    probability_map,
                    scenario,
                    base_cost,
                    active,
                )
                decision_makers = int(comm["decision_makers"])
                committed_plan, modal_share, safe_commit = plan_statistics(
                    plans,
                    decision_makers,
                )
                plan_match_reference = bool(
                    safe_commit and committed_plan == reference_plan
                )

                execution = execution_metrics(
                    committed_plan,
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )
                shared_channel_ms = communication_time_proxy_ms(
                    comm["bytes_transmitted"],
                    int(comm["communication_stages"]),
                )

                records.append(
                    {
                        "robots": n,
                        "active_robots": active_count,
                        "tasks": task_count,
                        "trial": trial + 1,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "packet_loss_rate": PACKET_LOSS_RATE,
                        "max_attempts": MAX_ATTEMPTS,
                        "group_size": ROBOT_GROUP_SIZE,
                        "route_guardrail_tolerance": ROUTE_GUARDRAIL_TOLERANCE,
                        "modal_plan_share": modal_share,
                        "safe_commit": safe_commit,
                        "plan_match_reference": plan_match_reference,
                        "shared_channel_latency_proxy_ms": shared_channel_ms,
                        **comm,
                        **execution,
                    }
                )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["robots", "method", "method_label"], as_index=False)
        .agg(
            safe_commit_rate=("safe_commit", "mean"),
            plan_match_reference_rate=("plan_match_reference", "mean"),
            modal_plan_share=("modal_plan_share", "mean"),
            preference_coverage=("preference_coverage", "mean"),
            logical_messages=("logical_messages", "mean"),
            message_attempts=("message_attempts", "mean"),
            bytes_transmitted=("bytes_transmitted", "mean"),
            communication_stages=("communication_stages", "mean"),
            shared_channel_latency_proxy_ms=("shared_channel_latency_proxy_ms", "mean"),
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success_rate", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            duplicate_task_execution_rate=("duplicate_task_execution_rate", "mean"),
        )
    )
    overall = (
        raw.groupby(["method", "method_label"], as_index=False)
        .agg(
            safe_commit_rate=("safe_commit", "mean"),
            plan_match_reference_rate=("plan_match_reference", "mean"),
            preference_coverage=("preference_coverage", "mean"),
            logical_messages=("logical_messages", "mean"),
            message_attempts=("message_attempts", "mean"),
            bytes_transmitted=("bytes_transmitted", "mean"),
            communication_stages=("communication_stages", "mean"),
            shared_channel_latency_proxy_ms=("shared_channel_latency_proxy_ms", "mean"),
            task_completion_rate=("task_completion_rate", "mean"),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
        )
    )
    return raw, summary, overall


def plot_lines(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    percent: bool = False,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for method in METHODS:
        subset = summary[summary["method"] == method].sort_values("robots")
        ax.plot(
            subset["robots"],
            subset[metric],
            marker="o",
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    raw.to_csv(DATA_DIR / "hierarchical_raw_results.csv", index=False)
    summary.to_csv(DATA_DIR / "hierarchical_summary_results.csv", index=False)
    overall.to_csv(DATA_DIR / "hierarchical_by_method.csv", index=False)

    plot_lines(
        summary,
        "message_attempts",
        "Average Transmission Attempts",
        "Complete-Preference Communication Attempts",
        "hierarchical_message_attempts.png",
        log_y=True,
    )
    plot_lines(
        summary,
        "bytes_transmitted",
        "Average Preference Bytes Transmitted",
        "Communication Volume: Flat P2P vs 5-ary Hierarchy",
        "hierarchical_bytes_transmitted.png",
        log_y=True,
    )
    plot_lines(
        summary,
        "shared_channel_latency_proxy_ms",
        "Shared-Channel Latency Proxy (ms)",
        "Bandwidth-Aware Communication Delay Proxy",
        "hierarchical_latency_proxy.png",
        log_y=True,
    )
    plot_lines(
        summary,
        "preference_coverage",
        "Average Fraction of Active Preferences Represented",
        "How Much Global Preference Information Reaches Decision Makers?",
        "hierarchical_preference_coverage.png",
        percent=True,
    )
    plot_lines(
        summary,
        "safe_commit_rate",
        "Safe Plan Commit Rate",
        "Can the Communication Architecture Still Form a 67% Plan Quorum?",
        "hierarchical_safe_commit_rate.png",
        percent=True,
    )
    plot_lines(
        summary,
        "plan_match_reference_rate",
        "Committed Plan Matches Full-Information Reference",
        "Does Hierarchical Aggregation Preserve the Full-Information Route Plan?",
        "hierarchical_plan_match_rate.png",
        percent=True,
    )
    plot_lines(
        summary,
        "actual_makespan_sec",
        "Actual Mission Makespan (s)",
        "Execution Quality of Safely Committed Plans",
        "hierarchical_makespan.png",
    )
    plot_lines(
        summary,
        "energy_per_completed_task",
        "Energy per Completed Task",
        "Energy Efficiency after Hierarchical Preference Aggregation",
        "hierarchical_energy_per_task.png",
    )


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, overall = run_experiment()
    save_outputs(raw, summary, overall)

    columns = [
        "method_label",
        "safe_commit_rate",
        "plan_match_reference_rate",
        "preference_coverage",
        "message_attempts",
        "bytes_transmitted",
        "communication_stages",
        "shared_channel_latency_proxy_ms",
        "task_completion_rate",
        "actual_makespan_sec",
        "energy_per_completed_task",
    ]
    print(overall[columns].to_string(index=False))


if __name__ == "__main__":
    main()
