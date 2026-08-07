from __future__ import annotations

import argparse

import numpy as np

from voting_mrta import apply_vote_loss, generate_full_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show one Voting-MRTA round vote-by-vote."
    )
    parser.add_argument("--robots", type=int, default=20, help="Number of robots")
    parser.add_argument(
        "--loss",
        type=float,
        default=0.20,
        help="Vote packet-loss probability, e.g. 0.20",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    round_data = generate_full_round(args.robots, rng)
    loss_result = apply_vote_loss(round_data, args.loss)

    print("=" * 78)
    print("ROBOT COSTS AND VOTING PROBABILITIES")
    print("=" * 78)
    print(f"{'Robot':>7} {'Cost':>10} {'Vote probability':>20}")

    for i, (cost, probability) in enumerate(
        zip(round_data.costs, round_data.probabilities),
        start=1,
    ):
        print(f"R{i:03d} {cost:10.2f} {probability:20.4%}")

    print("\n" + "=" * 78)
    print("VOTING AND COMMUNICATION")
    print("=" * 78)
    print(f"{'Voter':>7} {'Candidate':>12} {'Communication':>18}")

    for voter, (candidate, delivered) in enumerate(
        zip(round_data.ballots, loss_result.delivered),
        start=1,
    ):
        status = "RECEIVED" if delivered else "DROPPED"
        print(f"R{voter:03d} {f'R{candidate + 1:03d}':>12} {status:>18}")

    print("\n" + "=" * 78)
    print("FINAL COUNT")
    print("=" * 78)
    print(f"{'Candidate':>10} {'Full votes':>12} {'Received votes':>18}")

    candidates = np.flatnonzero(
        (round_data.full_counts > 0) | (loss_result.received_counts > 0)
    )

    for candidate in candidates:
        print(
            f"R{candidate + 1:03d}"
            f" {round_data.full_counts[candidate]:12d}"
            f" {loss_result.received_counts[candidate]:18d}"
        )

    full_winner = round_data.full_winner + 1
    lossy_winner = None if loss_result.winner is None else loss_result.winner + 1

    delivered_count = int(loss_result.delivered.sum())
    dropped_count = args.robots - delivered_count

    print("\n" + "-" * 78)
    print(f"Configured loss rate : {args.loss:.1%}")
    print(f"Delivered votes      : {delivered_count}/{args.robots}")
    print(f"Dropped votes        : {dropped_count}/{args.robots}")
    print(f"Observed loss rate   : {dropped_count / args.robots:.1%}")
    print(f"Full winner          : R{full_winner:03d}")

    if lossy_winner is None:
        print("Lossy winner         : NO DECISION")
        print("Winner preserved     : False")
    else:
        print(f"Lossy winner         : R{lossy_winner:03d}")
        print(f"Winner preserved     : {lossy_winner == full_winner}")
        print(f"Selected cost        : {round_data.costs[loss_result.winner]:.2f}")
        print(f"Minimum cost         : {round_data.costs.min():.2f}")
        print(
            "Regret               : "
            f"{round_data.costs[loss_result.winner] - round_data.costs.min():.2f}"
        )


if __name__ == "__main__":
    main()
