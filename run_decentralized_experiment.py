from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_algorithm_experiment import ROBOT_FAILURE_RATE, sample_active_robots
from run_multitask_experiment import (
    assignment_total_cost,
    centralized_optimal_assignment,
    generate_spatial_cost_matrix,
    per_task_probability,
    support_assignment,
)


RANDOM_SEED = 20260811
DECENTRALIZED_ROBOT_COUNTS = (10, 20, 40, 60, 80, 100)
DECENTRALIZED_TRIALS = 50
TASK_LOAD_RATIO = 0.50
MAX_ATTEMPTS = 3
PACKET_LOSS_RATES = (0.30, 0.50, 0.70)
QUORUM_THRESHOLDS = (0.40, 0.50, 0.60, 2.0 / 3.0, 0.75)

HETEROGENEOUS_POLICIES = ("inverse_a2", "inverse_a3", "softmax_b2")
MODES = ("homogeneous_a3", "heterogeneous_strong")
MODE_LABELS = {
    "homogeneous_a3": "Homogeneous Inverse a=3",
    "heterogeneous_strong": "Heterogeneous Strong Policies",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "decentralized"
DATA_DIR = RESULT_DIR / "data"
FIGURE_DIR = RESULT_DIR / "figures"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def quorum_label(q: float) -> str:
    if np.isclose(q, 2.0 / 3.0):
        return "67%+"
    return f"{int(round(100 * q))}%+"


def required_quorum(active_count: int, q: float) -> int:
    """Require strictly more than q times the active membership."""
    return int(np.floor(q * active_count)) + 1


def assign_sender_policies(
    active: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> dict[int, str]:
    active_indices = np.flatnonzero(active)
    if mode == "homogeneous_a3":
        return {int(robot): "inverse_a3" for robot in active_indices}
    if mode != "heterogeneous_strong":
        raise ValueError(f"Unknown decentralized mode: {mode}")

    ordered = rng.permutation(active_indices)
    return {
        int(robot): HETEROGENEOUS_POLICIES[index % len(HETEROGENEOUS_POLICIES)]
        for index, robot in enumerate(ordered)
    }


def peer_delivery_matrix(
    active: np.ndarray,
    attempt_random: np.ndarray,
    loss_rate: float,
) -> tuple[np.ndarray, float, float]:
    """Return receiver x sender delivery for one peer-to-peer broadcast phase.

    A sender broadcasts one complete message to every active peer. The message is
    retried stop-on-success up to MAX_ATTEMPTS. A robot always has its own local
    message, so self-delivery is forced true and is excluded from communication
    delivery/overhead statistics.
    """
    n = len(active)
    active_pair = active[:, None] & active[None, :]
    peer_pair = active_pair & ~np.eye(n, dtype=bool)

    success_by_attempt = attempt_random >= loss_rate
    delivered = np.any(success_by_attempt, axis=0) & peer_pair
    np.fill_diagonal(delivered, active)

    if not np.any(peer_pair):
        return delivered, 1.0, 0.0

    peer_success = success_by_attempt[:, peer_pair]
    first_success = np.argmax(peer_success, axis=0) + 1
    ever_success = np.any(peer_success, axis=0)
    attempts_used = np.where(ever_success, first_success, MAX_ATTEMPTS)

    delivery_rate = float(delivered[peer_pair].mean())
    average_attempts = float(attempts_used.mean())
    return delivered, delivery_rate, average_attempts


def policy_counts_for_receivers(
    delivered: np.ndarray,
    active: np.ndarray,
    sender_policy: dict[int, str],
) -> tuple[list[str], np.ndarray]:
    policies = sorted(set(sender_policy.values()))
    policy_index = {policy: index for index, policy in enumerate(policies)}
    sender_one_hot = np.zeros((len(active), len(policies)), dtype=float)
    for sender, policy in sender_policy.items():
        sender_one_hot[sender, policy_index[policy]] = 1.0
    counts = delivered.astype(float) @ sender_one_hot
    return policies, counts


def assignment_from_policy_mix(
    counts: np.ndarray,
    policies: list[str],
    probability_map: dict[str, np.ndarray],
    active: np.ndarray,
    tie_priority: np.ndarray,
) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        raise RuntimeError("A receiver must always retain at least its own preference")

    support = np.zeros_like(next(iter(probability_map.values())), dtype=float)
    for policy, count in zip(policies, counts, strict=True):
        if count > 0:
            support += (count / total) * probability_map[policy]
    return support_assignment(support, active, tie_priority)


def local_assignments_from_received_preferences(
    delivered: np.ndarray,
    active: np.ndarray,
    sender_policy: dict[int, str],
    probability_map: dict[str, np.ndarray],
    tie_priority: np.ndarray,
) -> dict[int, np.ndarray]:
    policies, receiver_counts = policy_counts_for_receivers(
        delivered, active, sender_policy
    )
    active_indices = np.flatnonzero(active)
    cache: dict[tuple[int, ...], np.ndarray] = {}
    assignments: dict[int, np.ndarray] = {}

    for receiver in active_indices:
        key = tuple(int(value) for value in receiver_counts[receiver])
        if key not in cache:
            cache[key] = assignment_from_policy_mix(
                receiver_counts[receiver],
                policies,
                probability_map,
                active,
                tie_priority,
            )
        assignments[int(receiver)] = cache[key].copy()
    return assignments


def full_information_preference_assignment(
    active: np.ndarray,
    sender_policy: dict[int, str],
    probability_map: dict[str, np.ndarray],
    tie_priority: np.ndarray,
) -> np.ndarray:
    counts = Counter(sender_policy.values())
    policies = sorted(counts)
    values = np.asarray([counts[policy] for policy in policies], dtype=float)
    return assignment_from_policy_mix(
        values,
        policies,
        probability_map,
        active,
        tie_priority,
    )


def assignment_labels(
    assignments: dict[int, np.ndarray],
) -> tuple[dict[int, int], dict[int, np.ndarray], Counter[int]]:
    unique_tuples = sorted({tuple(value.tolist()) for value in assignments.values()})
    tuple_to_label = {value: index for index, value in enumerate(unique_tuples)}
    sender_labels: dict[int, int] = {}
    label_to_assignment: dict[int, np.ndarray] = {}

    for sender, assignment in assignments.items():
        key = tuple(assignment.tolist())
        label = tuple_to_label[key]
        sender_labels[sender] = label
        label_to_assignment[label] = assignment

    return sender_labels, label_to_assignment, Counter(sender_labels.values())


def local_quorum_commits(
    assignments: dict[int, np.ndarray],
    proposal_delivered: np.ndarray,
    active: np.ndarray,
    q: float,
) -> tuple[dict[int, int], dict[int, np.ndarray], Counter[int]]:
    sender_labels, label_to_assignment, global_counts = assignment_labels(assignments)
    active_indices = np.flatnonzero(active)
    needed = required_quorum(len(active_indices), q)
    commits: dict[int, int] = {}

    for receiver in active_indices:
        visible = [
            sender_labels[int(sender)]
            for sender in active_indices
            if proposal_delivered[receiver, sender]
        ]
        if not visible:
            continue
        counts = Counter(visible)
        best_count = max(counts.values())
        if best_count < needed:
            continue
        best_labels = [label for label, count in counts.items() if count == best_count]
        commits[int(receiver)] = min(best_labels)

    return commits, label_to_assignment, global_counts


def percent_gap(cost: float, optimal_cost: float) -> float:
    return 100.0 * (cost - optimal_cost) / optimal_cost


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []

    for n in DECENTRALIZED_ROBOT_COUNTS:
        for trial in range(DECENTRALIZED_TRIALS):
            rng = np.random.default_rng(RANDOM_SEED + n * 100000 + trial)
            active = sample_active_robots(n, rng)
            active_indices = np.flatnonzero(active)
            active_count = len(active_indices)
            task_count = max(1, int(np.floor(active_count * TASK_LOAD_RATIO)))
            task_count = min(task_count, active_count)

            costs = generate_spatial_cost_matrix(n, task_count, rng)
            optimal_assignment = centralized_optimal_assignment(costs, active)
            optimal_cost = assignment_total_cost(costs, optimal_assignment)
            tie_priority = rng.random((n, task_count))

            pref_attempt_random = rng.random((MAX_ATTEMPTS, n, n))
            proposal_attempt_random = rng.random((MAX_ATTEMPTS, n, n))

            for mode in MODES:
                policy_rng = np.random.default_rng(
                    RANDOM_SEED + 50000000 + n * 100000 + trial
                )
                sender_policy = assign_sender_policies(active, mode, policy_rng)
                used_policies = sorted(set(sender_policy.values()))
                probability_map = {
                    policy: per_task_probability(costs, active, policy)
                    for policy in used_policies
                }
                reference_assignment = full_information_preference_assignment(
                    active,
                    sender_policy,
                    probability_map,
                    tie_priority,
                )
                reference_cost = assignment_total_cost(costs, reference_assignment)
                reference_gap = percent_gap(reference_cost, optimal_cost)

                for loss_rate in PACKET_LOSS_RATES:
                    pref_delivered, pref_receive_rate, pref_attempts = peer_delivery_matrix(
                        active,
                        pref_attempt_random,
                        loss_rate,
                    )
                    local_assignments = local_assignments_from_received_preferences(
                        pref_delivered,
                        active,
                        sender_policy,
                        probability_map,
                        tie_priority,
                    )

                    _, _, proposal_counts = assignment_labels(local_assignments)
                    modal_count = max(proposal_counts.values())
                    modal_share = modal_count / active_count
                    strict_agreement = len(proposal_counts) == 1

                    reference_tuple = tuple(reference_assignment.tolist())
                    reference_match_fraction = float(
                        np.mean(
                            [
                                tuple(local_assignments[int(robot)].tolist())
                                == reference_tuple
                                for robot in active_indices
                            ]
                        )
                    )
                    local_gaps = [
                        percent_gap(
                            assignment_total_cost(
                                costs, local_assignments[int(robot)]
                            ),
                            optimal_cost,
                        )
                        for robot in active_indices
                    ]

                    proposal_delivered, proposal_receive_rate, proposal_attempts = (
                        peer_delivery_matrix(
                            active,
                            proposal_attempt_random,
                            loss_rate,
                        )
                    )

                    for q in QUORUM_THRESHOLDS:
                        commits, label_to_assignment, global_counts = local_quorum_commits(
                            local_assignments,
                            proposal_delivered,
                            active,
                            q,
                        )
                        committed_labels = list(commits.values())
                        distinct_commits = set(committed_labels)
                        any_commit = len(committed_labels) > 0
                        split_brain = len(distinct_commits) > 1
                        safe_commit = any_commit and not split_brain
                        committed_node_fraction = len(committed_labels) / active_count
                        needed = required_quorum(active_count, q)
                        global_quorum_exists = max(global_counts.values()) >= needed

                        if safe_commit:
                            committed_label = committed_labels[0]
                            committed_assignment = label_to_assignment[committed_label]
                            committed_gap = percent_gap(
                                assignment_total_cost(costs, committed_assignment),
                                optimal_cost,
                            )
                            committed_reference_match = bool(
                                np.array_equal(
                                    committed_assignment, reference_assignment
                                )
                            )
                        else:
                            committed_gap = np.nan
                            committed_reference_match = False

                        records.append(
                            {
                                "robots": n,
                                "active_robots": active_count,
                                "tasks": task_count,
                                "trial": trial + 1,
                                "mode": mode,
                                "mode_label": MODE_LABELS[mode],
                                "packet_loss_rate": loss_rate,
                                "effective_theoretical_loss": loss_rate**MAX_ATTEMPTS,
                                "quorum_threshold": q,
                                "quorum_label": quorum_label(q),
                                "required_quorum_votes": needed,
                                "preference_receive_rate": pref_receive_rate,
                                "proposal_receive_rate": proposal_receive_rate,
                                "preference_attempts_per_peer": pref_attempts,
                                "proposal_attempts_per_peer": proposal_attempts,
                                "modal_assignment_share": modal_share,
                                "strict_all_nodes_agree": strict_agreement,
                                "reference_match_fraction": reference_match_fraction,
                                "mean_local_optimality_gap_percent": float(
                                    np.mean(local_gaps)
                                ),
                                "reference_optimality_gap_percent": reference_gap,
                                "global_quorum_exists": global_quorum_exists,
                                "any_local_commit": any_commit,
                                "safe_commit": safe_commit,
                                "split_brain": split_brain,
                                "committed_node_fraction": committed_node_fraction,
                                "committed_reference_match": committed_reference_match,
                                "committed_optimality_gap_percent": committed_gap,
                            }
                        )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(
            [
                "robots",
                "mode",
                "mode_label",
                "packet_loss_rate",
                "quorum_threshold",
                "quorum_label",
            ],
            as_index=False,
        )
        .agg(
            average_active_robots=("active_robots", "mean"),
            average_tasks=("tasks", "mean"),
            preference_receive_rate=("preference_receive_rate", "mean"),
            proposal_receive_rate=("proposal_receive_rate", "mean"),
            preference_attempts_per_peer=("preference_attempts_per_peer", "mean"),
            modal_assignment_share=("modal_assignment_share", "mean"),
            strict_agreement_rate=("strict_all_nodes_agree", "mean"),
            reference_match_fraction=("reference_match_fraction", "mean"),
            mean_local_optimality_gap_percent=(
                "mean_local_optimality_gap_percent",
                "mean",
            ),
            reference_optimality_gap_percent=(
                "reference_optimality_gap_percent",
                "mean",
            ),
            global_quorum_exists_rate=("global_quorum_exists", "mean"),
            any_local_commit_rate=("any_local_commit", "mean"),
            safe_commit_rate=("safe_commit", "mean"),
            split_brain_rate=("split_brain", "mean"),
            committed_node_fraction=("committed_node_fraction", "mean"),
            committed_reference_match_rate=("committed_reference_match", "mean"),
            committed_optimality_gap_percent=(
                "committed_optimality_gap_percent",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )

    by_quorum = (
        summary.groupby(
            ["mode", "mode_label", "quorum_threshold", "quorum_label"],
            as_index=False,
        )
        .agg(
            safe_commit_rate=("safe_commit_rate", "mean"),
            split_brain_rate=("split_brain_rate", "mean"),
            committed_node_fraction=("committed_node_fraction", "mean"),
            committed_optimality_gap_percent=(
                "committed_optimality_gap_percent",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )
    return raw, summary, by_quorum


def plot_modal_agreement(summary: pd.DataFrame) -> None:
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["quorum_threshold"], 0.50)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(
            subset["robots"],
            subset["modal_assignment_share"],
            marker="o",
            linewidth=2.0,
            label=f"{int(loss_rate * 100)}% packet loss",
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Modal Assignment Share")
    ax.set_title("Does Group Size Stabilize Decentralized Assignment Proposals?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_modal_agreement_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_strict_vs_quorum(summary: pd.DataFrame) -> None:
    q = 2.0 / 3.0
    loss_rate = 0.70
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["packet_loss_rate"], loss_rate)
        & np.isclose(summary["quorum_threshold"], q)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(
        data["robots"],
        data["strict_agreement_rate"],
        marker="o",
        linewidth=2.0,
        label="Strict all-node agreement",
    )
    ax.plot(
        data["robots"],
        data["safe_commit_rate"],
        marker="o",
        linewidth=2.0,
        label="Safe commit with 67%+ quorum",
    )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Round Success Rate")
    ax.set_title("Strict Synchronization vs Quorum under 70% Per-Attempt Loss")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_strict_vs_quorum.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_safe_commit(summary: pd.DataFrame) -> None:
    loss_rate = 0.70
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["packet_loss_rate"], loss_rate)
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for q in QUORUM_THRESHOLDS:
        subset = data[np.isclose(data["quorum_threshold"], q)]
        ax.plot(
            subset["robots"],
            subset["safe_commit_rate"],
            marker="o",
            linewidth=2.0,
            label=quorum_label(q),
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Safe Commit Rate")
    ax.set_title("Quorum Commit Reliability under 70% Per-Attempt Packet Loss")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Quorum")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_safe_commit_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_split_brain(summary: pd.DataFrame) -> None:
    loss_rate = 0.70
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["packet_loss_rate"], loss_rate)
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for q in QUORUM_THRESHOLDS:
        subset = data[np.isclose(data["quorum_threshold"], q)]
        ax.plot(
            subset["robots"],
            subset["split_brain_rate"],
            marker="o",
            linewidth=2.0,
            label=quorum_label(q),
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Split-Brain Rate")
    ax.set_title("Can Two Conflicting Assignments Commit in the Same Round?")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Quorum")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_split_brain_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_optimality_gap(summary: pd.DataFrame) -> None:
    q = 2.0 / 3.0
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["quorum_threshold"], q)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(
            subset["robots"],
            subset["committed_optimality_gap_percent"],
            marker="o",
            linewidth=2.0,
            label=f"{int(loss_rate * 100)}% packet loss",
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Committed Assignment Optimality Gap (%)")
    ax.set_title("Quality of Safely Committed Decentralized Assignments (67%+ Quorum)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_optimality_gap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_message_delivery(summary: pd.DataFrame) -> None:
    data = summary[
        (summary["mode"] == "heterogeneous_strong")
        & np.isclose(summary["quorum_threshold"], 0.50)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for loss_rate in PACKET_LOSS_RATES:
        subset = data[np.isclose(data["packet_loss_rate"], loss_rate)]
        ax.plot(
            subset["robots"],
            subset["preference_receive_rate"],
            marker="o",
            linewidth=2.0,
            label=(
                f"{int(loss_rate * 100)}% loss; theory after retries "
                f"{100 * (1 - loss_rate**MAX_ATTEMPTS):.1f}% delivered"
            ),
        )
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Peer Preference Message Delivery Rate")
    ax.set_title("Peer-to-Peer Complete-Preference Delivery after Retransmission")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "decentralized_message_delivery_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_figures(summary: pd.DataFrame) -> None:
    plot_modal_agreement(summary)
    plot_strict_vs_quorum(summary)
    plot_safe_commit(summary)
    plot_split_brain(summary)
    plot_optimality_gap(summary)
    plot_message_delivery(summary)


def main() -> None:
    ensure_output_dirs()
    clear_old_outputs()
    raw, summary, by_quorum = run_experiment()

    raw_path = DATA_DIR / "decentralized_raw_results.csv"
    summary_path = DATA_DIR / "decentralized_summary_results.csv"
    by_quorum_path = DATA_DIR / "decentralized_by_quorum.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_quorum.to_csv(by_quorum_path, index=False)
    generate_figures(summary)

    focus = by_quorum[by_quorum["mode"] == "heterogeneous_strong"]
    print("\nDecentralized quorum summary (heterogeneous strong policies):")
    print(
        focus[
            [
                "quorum_label",
                "safe_commit_rate",
                "split_brain_rate",
                "committed_node_fraction",
                "committed_optimality_gap_percent",
            ]
        ].to_string(index=False)
    )
    print("\nGenerated decentralized voting files:")
    for path in [raw_path, summary_path, by_quorum_path]:
        print(f"  {path.relative_to(ROOT)}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
