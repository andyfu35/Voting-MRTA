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

Every active robot sends its complete preference matrix directly to every other active robot. The logical message count scales approximately as `N * (N - 1)` before retransmission.

### 5-ary hierarchy

Active robots are recursively grouped in groups of at most five. At each level one rotating leader sums the complete preference matrices received from its children. The aggregate has the same matrix dimensions as one original complete-preference message, so a leader forwards one aggregate instead of forwarding every child message separately.

The hierarchy stops when at most five top leaders remain. These final leaders exchange their subtree aggregates peer-to-peer; the experiment intentionally does not elect one permanent global root.

Because addition is associative, `sum(group sums) == sum(all robot preferences)` when every aggregate is delivered. Therefore hierarchy alone does not mathematically discard second-, third-, or lower-ranked candidate information.

### 5-ary hierarchy with backup relay

The same hierarchy is used, but if a child aggregate cannot reach its primary group leader after all three attempts, the child tries a backup leader. If the backup receives the aggregate, it relays that aggregate to the primary leader.

This fallback is only used after direct-link failure, so its communication overhead should be small under the main 30% packet-loss setting while reducing the risk that one failed hierarchical edge removes an entire subtree from the global preference estimate.

Leaders rotate by trial/level/group index to avoid a permanent leader role in the simulation.

## Communication metrics

The experiment records logical preference messages, actual transmission attempts, bytes transmitted, sequential communication stages, preference coverage, and a shared-channel latency proxy.

The latency proxy is not wall-clock runtime. It assumes a synthetic shared 100 Mbps channel and 5 ms of coordination/propagation delay per sequential communication stage:

```text
latency_proxy = stages * 5 ms + transmitted_bits / 100 Mbps
```

## Decision-quality metrics

Every decision maker converts the preference aggregate it actually received into a route plan using the same 20% guardrail. The experiment records modal plan share, 67%+ safe commit rate, exact match to the full-information reference route plan, task completion, actual makespan, energy per completed task, route task-count CV, and duplicate-task execution rate.

## First validated results

Averaged across robot counts 10, 20, 40, 60, 80, 100 with 30 trials each:

```text
Flat P2P
  safe commit                 100.0%
  match full-info route       100.0%
  preference coverage          97.48%
  transmission attempts       4526.75
  preference bytes             230.70 MB
  communication stages           1.00
  latency proxy              18461.11 ms
  task completion               99.46%
  actual makespan              114.07 s
  energy/completed task         18.24

5-ary Hierarchy
  safe commit                 100.0%
  match full-info route       100.0%
  preference coverage          94.85%
  transmission attempts         74.20
  preference bytes               3.09 MB
  communication stages           2.67
  latency proxy                260.32 ms
  task completion               99.46%
  actual makespan              114.07 s
  energy/completed task         18.24

5-ary + Backup Relay
  safe commit                 100.0%
  match full-info route       100.0%
  preference coverage          97.66%
  transmission attempts         76.82
  preference bytes               3.19 MB
  communication stages           2.67
  latency proxy                268.20 ms
  task completion               99.46%
  actual makespan              114.07 s
  energy/completed task         18.24
```

Relative to Flat P2P, the plain 5-ary hierarchy reduces average transmission attempts by about 98.4% and preference-byte volume by about 98.7% in this experiment. Backup relay recovers preference coverage to approximately the flat-P2P level while adding only a small amount of traffic.

Despite the lower raw preference coverage of the plain hierarchy, all three communication architectures produced the same safely committed route plan as the full-information reference in every tested trial. Consequently execution-level completion, makespan, and energy were identical in this first experiment.

This supports the use of hierarchical aggregation as a communication optimization, but it does not prove that quality will remain identical under larger group sizes, higher loss, adversarial topology, or mid-round leader failures.

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
