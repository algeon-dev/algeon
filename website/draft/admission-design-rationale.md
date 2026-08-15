# Admission Design Rationale

> **Draft** — companion to [Admission & Memory Governance](../docs/design/admission.mdx).
> That page states the contract; this document records *why* the contract has
> this shape, and how to configure it for different workload shapes. Not
> published on the website yet.

## The question this model answers

A GPU device is one shared, non-swappable pool of memory. The admission model
answers exactly one question: **how do concurrent queries share that pool
without ever oversubscribing it, given that nobody can predict a query's peak
memory precisely?** Every design decision below follows from refusing to
pretend that prediction is possible.

## Why a grant model, not a memory predictor

The obvious design — statically compute each query's peak VRAM, admit if it
fits — was built first and removed (2026-08-07). Peak memory of relational GPU
work depends on runtime data: join fanout, group-by cardinality, string
widths, libcudf's internal workspaces, cuGraph's graph workspaces. A static
predictor is either wrong (OOM under concurrency) or padded so far it wastes
the device. The serving plan therefore forbids the whole category: admission
*may not admit from an observation, multiplier, adapter guess, or soft
estimate*.

What replaces prediction is a split of responsibilities:

- **A provable lower bound.** The engine can prove a *floor* — the minimum
  bytes an attempt needs to start (source stage, per-phase minimums). This is
  evidence, not estimation.
- **An enforceable upper bound.** The grant is not a prediction of usage; it
  is a *ceiling the attempt cannot exceed*, enforced by giving each attempt a
  private allocation domain whose limit equals the grant. Execution adapts
  inside it: chunked source reads, retry with smaller chunks. Exceeding a
  grant on a supported plan is a contract defect, not a normal outcome.
- **Integer accounting for concurrency.** The ledger admits from committed
  integers only. Live CUDA free-memory samples are racy — another attempt can
  allocate between the sample and the use — so they are telemetry, never a
  placement input.

Bounded execution and the ledger are not redundant: bounded execution keeps
*one query* under its grant; the ledger keeps *the sum of all grants* under
the device.

## Why bounded attempt slots (`max_active_attempts`)

Bytes are not the only per-attempt resource. Every admitted attempt also
receives an envelope — an attempt lane, a root stream, an allocation domain,
a device worker scope — plus a source-read allowance. Those are slot-shaped
and finite. Two further reasons:

- **Failure blast radius.** Cleanup, fault isolation, and device quarantine
  operate per attempt. Unbounded concurrency means unbounded cleanup debt on
  a sick device.
- **GPUs do not multiplex like CPUs.** Beyond a small number of concurrent
  attempts, kernel-launch contention and allocator churn reduce total
  throughput. Where that knee sits depends on the workload, so the operator
  declares it; the engine refuses to guess.

The default is **1** (`DEFAULT_MAX_ACTIVE_QUERIES`): serial per device, safe
for the workload the project targets first (graph serving, where one query
can legitimately use most of a device). High-QPS deployments are expected to
raise it — see the scenarios below.

## Why the default floor is `managed_capacity / max(8, max_active_attempts)`

The attempt floor is the guaranteed minimum share: admission never grants
less, the ledger keeps one floor of headroom per open slot, and a protected
waiter accumulates bytes until it reaches its floor.

If the default were simply `capacity / max_active_attempts`, then at the
default `max_active_attempts = 1` the floor would equal the whole device:
any resident-cache byte would make every query unadmittable, and every grant
would be the full card. Cutting the capacity into **at least eight shares**
keeps the floor a meaningful minimum (12.5% of capacity) regardless of the
concurrency setting. When the slot count exceeds eight, the floor becomes one
share per slot, so all slots remain fillable. The `8` is a granularity
backstop, not a concurrency default.

## How a grant gets its size — the working-set hint, precisely

A common misreading is that callers pick their grant. They cannot: no
admission API accepts a byte amount, by design. The grant interval and the
selection inside it are both engine-owned:

1. **The interval.** `required_floor` is the profile floor raised by
   engine-issued capability evidence (phase floors, source stage). `max_new`
   is what the ledger's capacity equation leaves over. Admission requires
   `required_floor <= max_new`.
2. **The selection.** During service-side resolution the engine tries to
   derive a *complete phase-peak schedule* for the exact plan — the peak
   bytes across every execution phase, from owner-derived bounds. When the
   schedule is complete, that peak becomes `soft_working_set_hint_bytes`, and
   the grant is the hint clamped into `[required_floor, max_new]`: the query
   takes roughly what it will need and leaves the rest for others.
3. **The conservative default.** When any phase cannot be bounded, there is
   no hint and the ledger grants `max_new` — deliberately greedy, because the
   grant is a hard ceiling and an under-sized ceiling would turn an
   unboundable phase into a runtime failure.

So the hint is **not an operator knob and not a caller parameter**. It is the
engine right-sizing grants automatically wherever its own evidence is
complete (keyed graph algorithms with known frame sizes, bounded local-Parquet
stages), and falling back to whole-remaining-pool grants where it is not.

The operator-facing knobs are:

| Knob | Where | What it shapes |
| --- | --- | --- |
| `max_active_attempts` | device profile | Concurrent attempts per device; also the share count in the derived floor. |
| `attempt_floor_bytes` | device profile | The guaranteed minimum grant and the per-open-slot headroom unit. |
| `cache_cap_bytes` | device profile | How much of the pool resident caches may occupy (they reduce `max_new`). |
| `max_queued_attempts`, `max_queue_wait`, `max_overtakes_per_waiter` | queue policy | How long and how fairly work waits when the arithmetic says "not now". |

## Shaping throughput: two worked scenarios (96 GB device)

### Scenario A — high QPS, small queries (~2 GB each)

With the defaults (`max_active_attempts = 1`) execution is serial per device:
QPS is bounded by single-attempt latency. That is the conservative default,
not a model limit. Set `max_active_attempts = 32`:

- Derived floor = `96 / max(8, 32)` = 3 GB.
- First admission: 31 slots stay open, so the ledger reserves `31 × 3 = 93 GB`
  and `max_new = 3 GB`. The floor reservation automatically caps even a
  no-hint grant at about one share, so 32 attempts of ~3 GB run concurrently.
- With complete phase peaks (~2 GB), grants shrink further toward actual need.

Independent queries also spread across devices (one attempt, one GPU), so
fleet QPS scales as `devices × per-device slots`.

### Scenario B — one 50 GB query mixed with small queries

The floor reservation that protects concurrency in Scenario A is exactly what
can lock a big query out: admitting it must still leave one floor per
remaining open slot.

- `max_active_attempts = 32`, derived floor 3 GB: admitting the 50 GB query
  reserves `31 × 3 = 93 GB`, leaving `max_new = 3 GB < 50 GB`. It can never
  be admitted — rejected locally as a capacity-shape failure, not queued.
- Same slots, explicit `attempt_floor_bytes = 1 GB`: reserve `31 × 1 = 31 GB`,
  `max_new = 65 GB ≥ 50 GB`. Admitted with a grant in `[50, 65]` (50 with a
  complete phase peak).
- Afterward a small query sees `max_new = 96 − 50 − 30 = 16 GB`: ~2 GB with a
  hint, 16 GB without one.

If the big query merely cannot fit *right now* (rather than *ever*), it is
not rejected: it queues, younger small queries may overtake it at most
`max_overtakes_per_waiter` times (default 4), and then protection reserves a
slot and accumulates released bytes until its floor fits.

**Rule of thumb:** derive the floor only when the workload is uniform. Mixed
sizes want an explicit small `attempt_floor_bytes` (the headroom unit) and a
moderate `max_active_attempts`; the engine's phase-peak hints then keep small
queries from hoarding what the big ones need.

## What this model deliberately refuses to do

- No caller-supplied grant bytes, ever — the ledger would be accounting
  someone else's guess.
- No admission from CUDA/NVML readings — telemetry diagnoses, it never
  admits.
- No automatic retry of a failed attempt — capacity pressure surfaces as a
  typed outcome (`QueryAdmissionOutcome`); the application owns the policy.

These refusals are the contract's value: when the ledger says the fleet is at
80% reservation utilization, that number is arithmetic over commitments, not
a hopeful estimate.
