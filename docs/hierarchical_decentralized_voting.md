# Hierarchical Decentralized Voting-MRTA

This experiment studies whether complete preference voting can scale without flat all-to-all communication.

## Fixed decision model

The experiment deliberately keeps the allocation logic unchanged so communication is the only experimental variable:

- one shared pre-allocation base-cost model;
- heterogeneous strong voting policies: Inverse a=2, Inverse a=3, Softmax b=2;
- complete preference distributions, not single winners;
- one-shot route allocation;
- 20% route guardrail;
- 30% packet loss per attempt;
- three stop-on-success transmission attempts;
- the existing 5% permanent robot-failure sampling;
- 67%+ plan quorum.

Task load is fixed to 100% of the active robot count in this first communication-scaling experiment.

## Compared communication architectures

### Flat P2P

Every active robot sends its complete preference matrix directly to every other active robot.

The logical message count scales approximately as

```text
N * (N - 1)
```

before retransmission.

### 5-ary hierarchy

Active robots are recursively grouped in groups of at most five.

At each level one rotating leader sums the complete preference matrices received from its children. The aggregate has the same matrix dimensions as one original complete-preference message, so a leader forwards one aggregate instead of forwarding every child message separately.

The hierarchy stops when at most five top leaders remain. These final leaders exchange their subtree aggregates peer-to-peer; the experiment intentionally does not elect one permanent global root.

Because addition is associative,

```text
sum(group sums) == sum(all robot preferences)
```

when every aggregate is delivered. Therefore hierarchy alone does not mathematically discard second-, third-, or lower-ranked candidate information.

### 5-ary hierarchy with backup relay

The same hierarchy is used, but if a child aggregate cannot reach its primary group leader after all three attempts, the child tries a backup leader. If the backup receives the aggregate, it relays that aggregate to the primary leader.

This fallback is only used after direct-link failure, so its communication overhead should be small under the main 30% packet-loss setting while reducing the risk that one failed hierarchical edge removes an entire subtree from the global preference estimate.

Leaders rotate by trial/level/group index to avoid a permanent leader role in the simulation.

## Why hierarchy can help

Flat P2P requires O(N^2) logical preference messages. A fixed-branching aggregation tree requires O(N) upward messages plus a small top-leader exchange.

The trade-off is that the hierarchy introduces multiple sequential communication stages. It can therefore have:

- much lower message/byte volume;
- more communication hops;
- a larger consequence when an aggregate edge is lost, because one lost aggregate may represent several robots;
- a leader hot-spot that must later be handled with role rotation and failure recovery.

## Communication metrics

The experiment records:

- logical preference messages;
- actual transmission attempts after stop-on-success retransmission;
- bytes transmitted, treating a complete preference/aggregate matrix as active_robots * tasks float64 values plus small metadata;
- sequential communication stages;
- average fraction of active preferences represented at decision makers;
- a shared-channel latency proxy.

The latency proxy is not wall-clock runtime. It assumes a synthetic shared 100 Mbps channel and 5 ms of coordination/propagation delay per sequential communication stage:

```text
latency_proxy = stages * 5 ms + transmitted_bits / 100 Mbps
```

It is included to show the trade-off between extra hierarchy stages and sharply reduced total traffic.

## Decision-quality metrics

Every decision maker converts the preference aggregate it actually received into a route plan using the same 20% guardrail. The experiment then records:

- modal plan share;
- 67%+ safe commit rate;
- exact match to the full-information reference route plan;
- task completion rate;
- actual mission makespan;
- energy per completed task;
- route task-count CV;
- duplicate-task execution rate.

Execution metrics are evaluated only for a safely committed plan; when no quorum exists, the effective task-completion rate is zero for that trial.

## Important scope limits

This first hierarchy experiment isolates complete-preference aggregation. It does not yet model a mid-round leader crash, leader election timeout, or full quorum-certificate dissemination to every leaf robot. Permanent failures are sampled before grouping, so leaders are selected only from active robots.

The next fault-tolerance step should explicitly fail leaders during a round and compare leader rotation / backup takeover policies.

## Run

```bash
python3 run_hierarchical_communication_experiment.py
```

Outputs are written under:

```text
results/hierarchical/data/
results/hierarchical/figures/
```
