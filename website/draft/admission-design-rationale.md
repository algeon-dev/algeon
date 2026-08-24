# Admission Design Rationale

> **Draft** — companion to [Admission & Memory Governance](../docs/design/admission.mdx).
> That page states the contract; this document records why the contract has
> this shape and how to configure it. It is not published on the website yet.

## The question this model answers

A GPU device is one shared, non-swappable memory pool. Admission answers one
question: which executable queries may enter without committing more memory
than the configured device capacity?

The model deliberately separates three concerns:

- capability validation decides whether the installed backend can execute a plan;
- a fixed query cap limits the attempt's allocation domain;
- the device ledger admits only committed integer charges.

Planner estimates may remain useful diagnostics, but they do not veto an
otherwise executable plan and do not size a grant.

## Fixed caps instead of predicted peaks

Relational and graph peaks depend on runtime data such as join fanout,
group-by cardinality, string widths, and native library workspaces. Treating a
predicted peak as an admission requirement either risks OOM or wastes the
device through padding.

Each device therefore has an immutable `default_query_limit_bytes`. An
embedding caller may request an explicit cap for a query; otherwise admission
uses the configured default, reduced only when frozen resident ownership makes
the device's resident-adjusted capacity smaller. The attempt receives exactly
that cap. Its query memory domain enforces the same value.

Exceeding the admitted cap is `QueryMemoryLimitExceeded`. A physical allocator
failure outside that cap is reported separately, so configuration pressure is
not confused with device or driver failure.

## The ledger equation

For device `d`:

```text
healthy(d)
active_attempts(d) < max_active_attempts(d)
R_d + sum(active_or_cleaning_query_caps) + B_q <= C_d
```

`C_d` is managed capacity, `R_d` is measured frozen-resident ownership, and
`B_q` is the query cap. Stream and worker slots are acquired as part of the
same admitted envelope. CUDA/NVML free-memory samples remain telemetry because
they cannot make a race-free admission promise.

If active queries temporarily consume the needed capacity, the query waits.
If its explicit cap is larger than `C_d - R_d` on every compatible device, it
fails with `QueryBudgetUnavailable`. Resident and health transitions revalidate
queued work.

## FIFO and concurrency

`max_active_attempts` bounds simultaneous attempt lifecycles and cleanup debt.
The queue is oldest-eligible FIFO per device: there is no overtaking budget,
aging rule, protected slot, or banked reserve. A multi-device waiter may run on
any compatible device that becomes available, without allowing younger work
to bypass it on a device they both need.

For independent small queries, choose a query cap that permits the intended
concurrency:

```text
max_active_attempts * default_query_limit_bytes <= managed_capacity
```

This is a configuration choice, not an automatic division. Keeping a device-
sized default cap with `max_active_attempts > 1` intentionally means only one
default-sized attempt can be admitted at a time. Explicit smaller caps can
still share the device when their committed sum fits.

## Frozen resident handoff

An ordinary runtime or GPU object is not an escaping owner. A successful
producer that returns a reusable GPU result must transfer it to a
`ResidentOwnerLease`:

1. stop new allocations in the producer domain;
2. drain host work and every registered stream;
3. verify every surviving allocation belongs to the published output owner;
4. seal the attempt report while the full query cap is still charged;
5. atomically replace that cap with measured aligned resident bytes `R_d`.

Capacity released by `B -> R` is never visible before the report. An unowned
residual allocation is a cleanup failure and quarantines the device instead of
being disguised as a resident result. Consumers borrow frozen inputs without
copying them into their own domain; any required graph orientation is built as
consumer-owned state and charged to the consumer cap.

## Operator-facing controls

| Knob | What it controls |
| --- | --- |
| `default_query_limit_bytes` | Fixed cap for queries without an explicit cap |
| `max_active_attempts` | Maximum concurrent attempt lifecycles on the device |
| `max_queued_attempts` | Bounded admission queue size |
| `max_queue_wait` | Maximum time a waiter may remain queued |
| `backend_reserve_bytes` | Capacity withheld from managed query ownership |
| `cache_cap_bytes` | Optional resident-cache ceiling; `0` keeps the cache disabled |

## Deliberate refusals

- No admission from planner estimates, SQL text, customer identity, or live
  CUDA/NVML free-memory readings.
- No proof-derived floor, grant growth, protected waiter, or overtake path.
- No automatic retry of a failed attempt; the application owns retry policy
  using the typed failure and sealed report evidence.

The ledger is an accounting authority, not a residency oracle. A reported
utilization value is arithmetic over committed query caps and measured frozen
resident owners.
