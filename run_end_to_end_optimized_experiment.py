from __future__ import annotations

from collections import Counter
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
    full_information_support,
    generate_scenario,
    lognormal_multipliers,
    route_plan,
    simulate_execution,
)
from run_hierarchical_communication_experiment import (
    communication_time_proxy_ms,
    flat_preference_exchange,
    hierarchical_preference_exchange,
    preference_payload_bytes,
    probability_support_from_counts,
    send_with_retries,
    sender_policy_one_hot,
)
from run_multitask_experiment import per_task_probability
from run_route_guardrail_experiment import guardrail_route_plan, route_dict
from run_route_heuristic_optimization import optimized_route_plan


RANDOM_SEED = 20260814
ROBOT_COUNTS = (10, 20, 40, 60, 80, 100)
TASK_LOAD_RATIOS = (0.50, 1.00, 1.50)
TRIALS = 30
PACKET_LOSS_RATE = 0.30
MAX_ATTEMPTS = 3
ROUTE_GUARDRAIL_TOLERANCE = 0.20
QUORUM_THRESHOLD = 2.0 / 3.0

PLAN_PROPOSAL_BYTES = 128
QC_METADATA_BYTES = 128
ASSIGNMENT_ENTRY_BYTES = 8

METHODS = (
    "centralized_route_greedy",
    "full_info_optimized",
    "flat_current",
    "hierarchy_current",
    "hierarchy_backup_current",
    "hierarchy_backup_optimized",
)
METHOD_LABELS = {
    "centralized_route_greedy": "Centralized Route Greedy",
    "full_info_optimized": "Full-Info Optimized Reference",
    "flat_current": "Flat P2P + Current Route",
    "hierarchy_current": "5-ary + Current Route",
    "hierarchy_backup_current": "5-ary Backup + Current Route",
    "hierarchy_backup_optimized": "5-ary Backup + Optimized Route",
}
COMMUNICATION_METHODS = (
    "flat_current",
    "hierarchy_current",
    "hierarchy_backup_current",
    "hierarchy_backup_optimized",
)

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "end_to_end"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def qc_payload_bytes(task_count: int) -> int:
    return QC_METADATA_BYTES + task_count * ASSIGNMENT_ENTRY_BYTES


def plan_from_support(
    method: str,
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    support: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    if method == "hierarchy_backup_optimized" or method == "full_info_optimized":
        return optimized_route_plan(
            scenario,
            base_cost,
            active,
            support,
            order_mode="regret",
            insertion_mode="best_insertion",
        )
    return guardrail_route_plan(
        scenario,
        base_cost,
        active,
        support,
        ROUTE_GUARDRAIL_TOLERANCE,
    )


def plans_from_count_rows(
    method: str,
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
            cache[key] = plan_from_support(
                method,
                scenario,
                base_cost,
                active,
                support,
            )
        plans[int(receiver)] = cache[key]
    return plans


def proposal_quorum_exchange(
    plans: dict[int, tuple[tuple[int, ...], ...]],
    rng: np.random.Generator,
) -> tuple[
    tuple[tuple[int, ...], ...] | None,
    list[int],
    dict[str, float],
]:
    members = sorted(plans)
    if not members:
        return None, [], {
            "proposal_logical_messages": 0.0,
            "proposal_attempts": 0.0,
            "proposal_bytes": 0.0,
            "proposal_modal_share": 0.0,
            "proposal_strict_agreement": 0.0,
            "qc_witness_count": 0.0,
        }

    signatures = sorted(set(plans.values()))
    plan_to_label = {plan: index for index, plan in enumerate(signatures)}
    label_to_plan = {index: plan for plan, index in plan_to_label.items()}
    sender_label = {member: plan_to_label[plans[member]] for member in members}
    global_counts = Counter(sender_label.values())
    modal_count = max(global_counts.values())
    modal_share = modal_count / len(members)
    strict_agreement = float(modal_count == len(members))
    needed = required_quorum(len(members), QUORUM_THRESHOLD)

    visible: dict[int, list[int]] = {
        receiver: [sender_label[receiver]] for receiver in members
    }
    logical_messages = 0
    attempts = 0

    for sender in members:
        for receiver in members:
            if sender == receiver:
                continue
            logical_messages += 1
            delivered, used = send_with_retries(rng, PACKET_LOSS_RATE)
            attempts += used
            if delivered:
                visible[receiver].append(sender_label[sender])

    witnesses_by_label: dict[int, list[int]] = {}
    for receiver in members:
        counts = Counter(visible[receiver])
        eligible = [
            (count, label)
            for label, count in counts.items()
            if count >= needed
        ]
        if not eligible:
            continue
        _, chosen_label = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
        witnesses_by_label.setdefault(chosen_label, []).append(receiver)

    if not witnesses_by_label:
        qc_plan = None
        witnesses: list[int] = []
    else:
        chosen_label, witnesses = sorted(
            witnesses_by_label.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[0]
        qc_plan = label_to_plan[chosen_label]

    return qc_plan, sorted(witnesses), {
        "proposal_logical_messages": float(logical_messages),
        "proposal_attempts": float(attempts),
        "proposal_bytes": float(attempts * PLAN_PROPOSAL_BYTES),
        "proposal_modal_share": float(modal_share),
        "proposal_strict_agreement": strict_agreement,
        "qc_witness_count": float(len(witnesses)),
    }


def disseminate_qc(
    plan: tuple[tuple[int, ...], ...] | None,
    witnesses: list[int],
    active: np.ndarray,
    task_count: int,
    rng: np.random.Generator,
) -> tuple[dict[int, tuple[int, ...]], dict[str, float]]:
    if plan is None or not witnesses:
        return {}, {
            "qc_logical_messages": 0.0,
            "qc_attempts": 0.0,
            "qc_bytes": 0.0,
            "qc_delivery_rate": 0.0,
            "authorized_owner_rate": 0.0,
        }

    active_indices = [int(robot) for robot in np.flatnonzero(active)]
    payload = qc_payload_bytes(task_count)
    witness_set = set(witnesses)
    authorized: set[int] = set(witness_set)
    logical_messages = 0
    attempts = 0

    primary = witnesses[0]
    secondary = witnesses[1] if len(witnesses) > 1 else None

    for receiver in active_indices:
        if receiver in witness_set:
            continue
        logical_messages += 1
        delivered, used = send_with_retries(rng, PACKET_LOSS_RATE)
        attempts += used
        if delivered:
            authorized.add(receiver)
            continue
        if secondary is None or secondary == primary:
            continue
        logical_messages += 1
        delivered, used = send_with_retries(rng, PACKET_LOSS_RATE)
        attempts += used
        if delivered:
            authorized.add(receiver)

    routes = {
        robot: plan[robot]
        for robot in active_indices
        if robot in authorized
    }
    owners = [robot for robot in active_indices if len(plan[robot]) > 0]
    authorized_owners = sum(robot in authorized for robot in owners)
    owner_rate = authorized_owners / len(owners) if owners else 1.0

    return routes, {
        "qc_logical_messages": float(logical_messages),
        "qc_attempts": float(attempts),
        "qc_bytes": float(attempts * payload),
        "qc_delivery_rate": float(len(authorized) / len(active_indices)),
        "authorized_owner_rate": float(owner_rate),
    }


def actual_finish_time_cv(
    robot_routes: dict[int, tuple[int, ...]],
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
) -> float:
    finish_times: list[float] = []
    start_time = scenario["start_travel_time"]
    transition_time = scenario["transition_travel_time"]
    service_times = scenario["service_times"]

    for robot in np.flatnonzero(active):
        clock = 0.0
        previous: int | None = None
        for task in robot_routes.get(int(robot), ()):
            if previous is None:
                travel = (
                    start_time[robot, task]
                    * start_time_multiplier[robot, task]
                )
            else:
                travel = (
                    transition_time[robot, previous, task]
                    * transition_time_multiplier[previous, task]
                )
            service = service_times[task] * service_time_multiplier[task]
            clock += float(travel + service)
            previous = int(task)
        finish_times.append(clock)

    values = np.asarray(finish_times, dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std() / mean) if mean > 1e-12 else 0.0


def execution_record(
    robot_routes: dict[int, tuple[int, ...]],
    active: np.ndarray,
    scenario: dict[str, np.ndarray],
    start_time_multiplier: np.ndarray,
    transition_time_multiplier: np.ndarray,
    service_time_multiplier: np.ndarray,
    success_uniform: np.ndarray,
) -> dict[str, float]:
    execution = simulate_execution(
        robot_routes,
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
        float(execution["total_energy_units"]) / completed
        if completed > 1e-12
        else np.nan
    )
    return {
        "task_completion_rate": float(execution["task_completion_rate"]),
        "mission_success_rate": float(bool(execution["mission_success"])),
        "unexecuted_task_rate": float(execution["unexecuted_task_rate"]),
        "duplicate_task_execution_rate": float(
            execution["duplicate_task_execution_rate"]
        ),
        "actual_makespan_sec": float(execution["actual_makespan_sec"]),
        "energy_per_completed_task": float(energy_per_completed),
        "total_travel_distance_m": float(execution["total_travel_distance_m"]),
        "route_task_count_cv": float(execution["route_task_count_cv"]),
        "eta_mape": float(execution["mean_eta_absolute_percentage_error"]),
        "route_finish_time_cv": actual_finish_time_cv(
            robot_routes,
            active,
            scenario,
            start_time_multiplier,
            transition_time_multiplier,
            service_time_multiplier,
        ),
    }


def reference_plan_for_method(
    method: str,
    scenario: dict[str, np.ndarray],
    base_cost: np.ndarray,
    active: np.ndarray,
    full_support: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    if method == "hierarchy_backup_optimized":
        return optimized_route_plan(
            scenario,
            base_cost,
            active,
            full_support,
            order_mode="regret",
            insertion_mode="best_insertion",
        )
    return guardrail_route_plan(
        scenario,
        base_cost,
        active,
        full_support,
        ROUTE_GUARDRAIL_TOLERANCE,
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in ROBOT_COUNTS:
        for load_index, load_ratio in enumerate(TASK_LOAD_RATIOS):
            for trial in range(TRIALS):
                rng = np.random.default_rng(
                    RANDOM_SEED + n * 100000 + load_index * 10000 + trial
                )
                active = sample_active_robots(n, rng)
                active_count = int(active.sum())
                task_count = max(1, int(round(active_count * load_ratio)))
                scenario = generate_scenario(n, task_count, rng)
                base_cost = build_base_cost(scenario, active)

                policy_rng = np.random.default_rng(
                    RANDOM_SEED + 50000000 + n * 100000 + load_index * 10000 + trial
                )
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
                full_support = full_information_support(sender_policy, probability_map)

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

                centralized_plan = route_plan(
                    scenario,
                    base_cost,
                    active,
                    support=None,
                )
                centralized_routes = route_dict(centralized_plan, active)
                centralized_execution = execution_record(
                    centralized_routes,
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )
                records.append(
                    {
                        "robots": n,
                        "active_robots": active_count,
                        "tasks": task_count,
                        "task_load_ratio": load_ratio,
                        "trial": trial + 1,
                        "method": "centralized_route_greedy",
                        "method_label": METHOD_LABELS["centralized_route_greedy"],
                        "safe_commit_rate": 1.0,
                        "plan_match_reference_rate": 1.0,
                        "preference_coverage": 1.0,
                        "proposal_modal_share": 1.0,
                        "qc_delivery_rate": 1.0,
                        "authorized_owner_rate": 1.0,
                        "logical_messages": np.nan,
                        "message_attempts": np.nan,
                        "bytes_transmitted": np.nan,
                        "communication_stages": np.nan,
                        "shared_channel_latency_proxy_ms": np.nan,
                        **centralized_execution,
                    }
                )

                optimized_reference = optimized_route_plan(
                    scenario,
                    base_cost,
                    active,
                    full_support,
                    order_mode="regret",
                    insertion_mode="best_insertion",
                )
                optimized_reference_routes = route_dict(optimized_reference, active)
                optimized_reference_execution = execution_record(
                    optimized_reference_routes,
                    active,
                    scenario,
                    start_time_multiplier,
                    transition_time_multiplier,
                    service_time_multiplier,
                    success_uniform,
                )
                records.append(
                    {
                        "robots": n,
                        "active_robots": active_count,
                        "tasks": task_count,
                        "task_load_ratio": load_ratio,
                        "trial": trial + 1,
                        "method": "full_info_optimized",
                        "method_label": METHOD_LABELS["full_info_optimized"],
                        "safe_commit_rate": 1.0,
                        "plan_match_reference_rate": 1.0,
                        "preference_coverage": 1.0,
                        "proposal_modal_share": 1.0,
                        "qc_delivery_rate": 1.0,
                        "authorized_owner_rate": 1.0,
                        "logical_messages": np.nan,
                        "message_attempts": np.nan,
                        "bytes_transmitted": np.nan,
                        "communication_stages": np.nan,
                        "shared_channel_latency_proxy_ms": np.nan,
                        **optimized_reference_execution,
                    }
                )

                payload_bytes = preference_payload_bytes(active_count, task_count)

                for method in COMMUNICATION_METHODS:
                    method_index = COMMUNICATION_METHODS.index(method)
                    preference_rng = np.random.default_rng(
                        RANDOM_SEED
                        + 100000000
                        + n * 100000
                        + load_index * 10000
                        + trial * 10
                        + method_index
                    )
                    if method == "flat_current":
                        count_rows, preference_comm = flat_preference_exchange(
                            active,
                            one_hot,
                            preference_rng,
                            payload_bytes,
                        )
                    else:
                        count_rows, preference_comm = hierarchical_preference_exchange(
                            active,
                            one_hot,
                            trial=trial,
                            rng=preference_rng,
                            payload_bytes=payload_bytes,
                            backup_enabled=method in (
                                "hierarchy_backup_current",
                                "hierarchy_backup_optimized",
                            ),
                        )

                    plans = plans_from_count_rows(
                        method,
                        count_rows,
                        policies,
                        probability_map,
                        scenario,
                        base_cost,
                        active,
                    )
                    reference_plan = reference_plan_for_method(
                        method,
                        scenario,
                        base_cost,
                        active,
                        full_support,
                    )

                    proposal_rng = np.random.default_rng(
                        RANDOM_SEED
                        + 200000000
                        + n * 100000
                        + load_index * 10000
                        + trial * 10
                        + method_index
                    )
                    qc_plan, witnesses, proposal_comm = proposal_quorum_exchange(
                        plans,
                        proposal_rng,
                    )
                    safe_commit = float(qc_plan is not None)
                    plan_match = float(qc_plan == reference_plan) if qc_plan is not None else 0.0

                    qc_rng = np.random.default_rng(
                        RANDOM_SEED
                        + 300000000
                        + n * 100000
                        + load_index * 10000
                        + trial * 10
                        + method_index
                    )
                    authorized_routes, qc_comm = disseminate_qc(
                        qc_plan,
                        witnesses,
                        active,
                        task_count,
                        qc_rng,
                    )

                    pref_logical = float(preference_comm["logical_messages"])
                    pref_attempts = float(preference_comm["message_attempts"])
                    pref_bytes = float(preference_comm["bytes_transmitted"])
                    proposal_logical = proposal_comm["proposal_logical_messages"]
                    proposal_attempts = proposal_comm["proposal_attempts"]
                    proposal_bytes = proposal_comm["proposal_bytes"]
                    qc_logical = qc_comm["qc_logical_messages"]
                    qc_attempts = qc_comm["qc_attempts"]
                    qc_bytes = qc_comm["qc_bytes"]

                    total_logical = pref_logical + proposal_logical + qc_logical
                    total_attempts = pref_attempts + proposal_attempts + qc_attempts
                    total_bytes = pref_bytes + proposal_bytes + qc_bytes
                    stages = float(preference_comm["communication_stages"]) + 2.0
                    latency_proxy = communication_time_proxy_ms(total_bytes, int(np.ceil(stages)))

                    execution = execution_record(
                        authorized_routes,
                        active,
                        scenario,
                        start_time_multiplier,
                        transition_time_multiplier,
                        service_time_multiplier,
                        success_uniform,
                    )
                    records.append(
                        {
                            "robots": n,
                            "active_robots": active_count,
                            "tasks": task_count,
                            "task_load_ratio": load_ratio,
                            "trial": trial + 1,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "safe_commit_rate": safe_commit,
                            "plan_match_reference_rate": plan_match,
                            "preference_coverage": float(
                                preference_comm["preference_coverage"]
                            ),
                            "proposal_modal_share": proposal_comm[
                                "proposal_modal_share"
                            ],
                            "proposal_strict_agreement": proposal_comm[
                                "proposal_strict_agreement"
                            ],
                            "qc_witness_count": proposal_comm["qc_witness_count"],
                            "qc_delivery_rate": qc_comm["qc_delivery_rate"],
                            "authorized_owner_rate": qc_comm[
                                "authorized_owner_rate"
                            ],
                            "logical_messages": total_logical,
                            "message_attempts": total_attempts,
                            "bytes_transmitted": total_bytes,
                            "communication_stages": stages,
                            "shared_channel_latency_proxy_ms": latency_proxy,
                            **execution,
                        }
                    )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            ["robots", "task_load_ratio", "method", "method_label"],
            as_index=False,
        )
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success_rate", "mean"),
            unexecuted_task_rate=("unexecuted_task_rate", "mean"),
            duplicate_task_execution_rate=(
                "duplicate_task_execution_rate",
                "mean",
            ),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            route_finish_time_cv=("route_finish_time_cv", "mean"),
            eta_mape=("eta_mape", "mean"),
            safe_commit_rate=("safe_commit_rate", "mean"),
            plan_match_reference_rate=("plan_match_reference_rate", "mean"),
            preference_coverage=("preference_coverage", "mean"),
            proposal_modal_share=("proposal_modal_share", "mean"),
            qc_delivery_rate=("qc_delivery_rate", "mean"),
            authorized_owner_rate=("authorized_owner_rate", "mean"),
            logical_messages=("logical_messages", "mean"),
            message_attempts=("message_attempts", "mean"),
            bytes_transmitted=("bytes_transmitted", "mean"),
            communication_stages=("communication_stages", "mean"),
            shared_channel_latency_proxy_ms=(
                "shared_channel_latency_proxy_ms",
                "mean",
            ),
        )
    )
    overall = (
        raw.groupby(["method", "method_label"], as_index=False)
        .agg(
            task_completion_rate=("task_completion_rate", "mean"),
            mission_success_rate=("mission_success_rate", "mean"),
            unexecuted_task_rate=("unexecuted_task_rate", "mean"),
            duplicate_task_execution_rate=(
                "duplicate_task_execution_rate",
                "mean",
            ),
            actual_makespan_sec=("actual_makespan_sec", "mean"),
            energy_per_completed_task=("energy_per_completed_task", "mean"),
            total_travel_distance_m=("total_travel_distance_m", "mean"),
            route_task_count_cv=("route_task_count_cv", "mean"),
            route_finish_time_cv=("route_finish_time_cv", "mean"),
            eta_mape=("eta_mape", "mean"),
            safe_commit_rate=("safe_commit_rate", "mean"),
            plan_match_reference_rate=("plan_match_reference_rate", "mean"),
            preference_coverage=("preference_coverage", "mean"),
            proposal_modal_share=("proposal_modal_share", "mean"),
            qc_delivery_rate=("qc_delivery_rate", "mean"),
            authorized_owner_rate=("authorized_owner_rate", "mean"),
            logical_messages=("logical_messages", "mean"),
            message_attempts=("message_attempts", "mean"),
            bytes_transmitted=("bytes_transmitted", "mean"),
            communication_stages=("communication_stages", "mean"),
            shared_channel_latency_proxy_ms=(
                "shared_channel_latency_proxy_ms",
                "mean",
            ),
        )
    )
    return raw, summary, overall


def plot_by_load(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(METHODS))
    width = 0.24
    for index, load_ratio in enumerate(TASK_LOAD_RATIOS):
        values = []
        subset = summary[np.isclose(summary["task_load_ratio"], load_ratio)]
        for method in METHODS:
            values.append(float(subset[subset["method"] == method][metric].mean()))
        ax.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=f"Task load {load_ratio:.0%}",
        )
    ax.set_xticks(
        x,
        [METHOD_LABELS[method] for method in METHODS],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def plot_communication_vs_robots(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    log_y: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in COMMUNICATION_METHODS:
        subset = summary[summary["method"] == method]
        grouped = subset.groupby("robots", as_index=False)[metric].mean()
        ax.plot(
            grouped["robots"],
            grouped[metric],
            marker="o",
            label=METHOD_LABELS[method],
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def plot_tradeoff(overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in overall.iterrows():
        ax.scatter(
            row["actual_makespan_sec"],
            row["energy_per_completed_task"],
            s=90,
        )
        ax.annotate(
            str(row["method_label"]),
            (row["actual_makespan_sec"], row["energy_per_completed_task"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Average Actual Makespan (s)")
    ax.set_ylabel("Energy per Completed Task")
    ax.set_title("End-to-End Trade-off: Mission Time vs Energy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "end_to_end_tradeoff.png", dpi=180)
    plt.close(fig)


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    raw.to_csv(DATA_DIR / "end_to_end_raw_results.csv", index=False)
    summary.to_csv(DATA_DIR / "end_to_end_summary_results.csv", index=False)
    overall.to_csv(DATA_DIR / "end_to_end_by_method.csv", index=False)

    plot_by_load(
        summary,
        "task_completion_rate",
        "Task Completion Rate",
        "End-to-End Task Completion",
        "end_to_end_task_completion.png",
        percent=True,
    )
    plot_by_load(
        summary,
        "actual_makespan_sec",
        "Average Actual Makespan (s)",
        "End-to-End Mission Time",
        "end_to_end_makespan.png",
    )
    plot_by_load(
        summary,
        "energy_per_completed_task",
        "Energy per Completed Task",
        "End-to-End Energy Efficiency",
        "end_to_end_energy_per_task.png",
    )
    plot_by_load(
        summary,
        "total_travel_distance_m",
        "Total Travel Distance (m)",
        "End-to-End Route Distance",
        "end_to_end_travel_distance.png",
    )
    plot_by_load(
        summary,
        "route_finish_time_cv",
        "Actual Route Finish-Time CV",
        "End-to-End Workload Balance",
        "end_to_end_finish_time_cv.png",
    )
    plot_by_load(
        summary,
        "safe_commit_rate",
        "Safe Commit Rate",
        "Can the End-to-End System Safely Commit a Plan?",
        "end_to_end_safe_commit.png",
        percent=True,
    )
    plot_by_load(
        summary,
        "duplicate_task_execution_rate",
        "Duplicate Task Execution Rate",
        "Does End-to-End Decentralization Cause Duplicate Execution?",
        "end_to_end_duplicate_rate.png",
        percent=True,
    )
    plot_communication_vs_robots(
        summary,
        "message_attempts",
        "Average Transmission Attempts",
        "End-to-End Communication Attempts",
        "end_to_end_message_attempts.png",
    )
    plot_communication_vs_robots(
        summary,
        "bytes_transmitted",
        "Average Bytes Transmitted",
        "End-to-End Communication Volume",
        "end_to_end_bytes_transmitted.png",
    )
    plot_communication_vs_robots(
        summary,
        "shared_channel_latency_proxy_ms",
        "Shared-Channel Latency Proxy (ms)",
        "End-to-End Communication Delay Proxy",
        "end_to_end_latency_proxy.png",
    )
    plot_tradeoff(overall)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, overall = run_experiment()
    save_outputs(raw, summary, overall)

    columns = [
        "method_label",
        "task_completion_rate",
        "mission_success_rate",
        "actual_makespan_sec",
        "energy_per_completed_task",
        "total_travel_distance_m",
        "route_finish_time_cv",
        "safe_commit_rate",
        "plan_match_reference_rate",
        "qc_delivery_rate",
        "message_attempts",
        "bytes_transmitted",
        "shared_channel_latency_proxy_ms",
        "duplicate_task_execution_rate",
    ]
    print(overall[columns].to_string(index=False))


if __name__ == "__main__":
    main()
