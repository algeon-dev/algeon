# Backend-service stress runbook

Every profile in this directory is a recorded experiment. Build before taking
the GPU lock, prepare fixtures separately, and use a new output directory for
each invocation. None of these commands downloads, creates, or overwrites a
fixture.

## Code axes (S / P / W / G / I)

`S`, `P`, `W`, `G`, and `I` are short handles, not self-explanatory names. Each
points at the place that owns the full meaning, which stays authoritative when
the two disagree. `S` cells are owned by a section of this runbook; the
`P`/`W`/`G`/`I` codes are owned by comments in
`crates/tools/src/single_node_multi_gpu_validate/`. Keep this table in
sync when a code is added or retired.

| Axis | What the codes number here | Owner |
| --- | --- | --- |
| `S` | stress scenario cells (recorded load experiments) | a section of this runbook |
| `P` | the validation program's own capability phases | `single_node_multi_gpu_validate` module docs |
| `W` | workload / observability artifact projections | `single_node_multi_gpu_validate` comments |
| `G` | the DBLP cuGraph workload priority set | `workload.rs` and `generate_s2_graph_profiles.py` |
| `I` | observability findings and gated probes | `single_node_multi_gpu_validate` comments |

### S: stress scenario cells

| Cell | What it records |
| --- | --- |
| `S1` | embedded-library mixed load; the feature-off cugraph cell; the OTLP (W8) collector cell |
| `S2` | concurrency/envelope/rate sweeps, knee-rate instantiation, claim-versus-VRAM calibration |
| `S3` | utilization, telemetry on/off A/B, heterogeneous ordinal profiles |
| `S4` | sustained-plateau QPS (optional spike/drain steps) |
| `S5` | storms: cancellation, deadline, sustained overload, and the I8 completion-sink cell |
| `S6` | quarantine (single, cycles, cache-resident) and close-under-load |
| `S7` | cache qualification: the paired P6.8 calibration and the cold-cache same-key probe |
| `S8` | Flight SQL server profiles (pooled / churn), bound to an in-process S2 baseline |

### P: program capability phases

- `P4`: the pull-only backend-metric snapshots (the audited metric store).
- `P5`: the cache opt-in control (`cache_cap_bytes`).
- `P6`: the measured production-validation run pipeline:
  - `P6.3` dry run; `P6.4` read-only fixture preflight; `P6.5` the measured public-API run (the published artifact schema).
  - `P6.6` the dedicated probes: deadline, cancellation, domain-cap exhaustion, finite-cohort overload, physical allocator pressure.
  - `P6.7` the bounded metrics audit over timestamped P4 snapshots.
  - `P6.8` the paired cache-disabled / cache-opt-in calibration.
- `P7` / `P8`: the contention profiles that opt into the sealed bounded-overload outcome.

### W: workload / observability projections

- `W1`: the compact per-device utilization/ledger/cache and process time series.
- `W3`: the multi-phase (multi-step) schedule and its per-step envelope aggregates.
- `W6`: the planned-attempt cap / offered-process bound every plan is checked against.
- `W8`: operability: per-attempt attribution completeness and the explicit reference OTLP exporter cell.
- `W9`: several DBLP graph fixtures (distinct source-hash slices) under shared registration names.

### G: DBLP cuGraph workload priority set

- `G1` `force_atlas2`, `G2` `bfs_deep`, `G3` `pagerank`, `G4` `betweenness` (`betweenness_centrality`).

### I: observability findings / gated probes

- `I1`: the histogram-saturation usability finding (non-blocking).
- `I5`: the allocation-fragmentation canary (first- vs second-half drift finding).
- `I8`: the serialized public completion-receiver (slow report sink) probe.
- `I11`: the graph-model claim/peak calibration gate (one typed row per measurement-phase graph attempt).
- `I13`: the public monitoring pull cost (with `W8`).

## Host and fixture requirements

The shipped S1-S7 profiles require one Linux host with CUDA, NVML, and four
independent NVIDIA L40S devices (ordinals `0,1,2,3`, 46,068 MiB each). They
record 44 GiB or 8 GiB per-device logical envelopes; aggregate VRAM is not a
pool. Run exactly one GPU workload on the host at a time with
`flock /tmp/cudf-gpu.lock`.

`sf1` profiles need TPC-H/TPC-DS SF1 and DBLP v12 where selected; `sf10`
profiles need their SF10 counterpart and DBLP v12 where selected. The W9
distinct-graph profiles additionally require the eight source-hash slices
derived from the same complete DBLP v12 input; they are not a second dataset.
Prepare them outside the measured window with the exact recorded command:

```bash
fixture/fixture.sh graph dblp-ingest \
  --source-json /path/to/dblp.v12.json \
  --out-root /opt/dlami/nvme/fixture/data/demo-v12-topology-r1 \
  --edge-slice-count 8 \
  --edge-slice-seed 20260804
```

The fixed BFS source `1091` belongs to `source-hash-07` under that partition,
so only that slice selects `bfs_deep`; PageRank, ForceAtlas2, and betweenness
are distributed across the full population. The S1 embedded-library profiles
additionally require their recorded ClickBench subset. The default paths and
preparatory commands are embedded in each in-process profile.
`--fixture-preflight` intentionally retains P6's exact one-TPC-H-plus-one-DBLP
contract, so it does not accept the broader S1/S3/S4 profiles. Every
`--measured-run` instead validates its complete selected fixture set before it
constructs a backend or output directory; prepare a missing fixture outside the
measured window using the recorded command. A template is not a runnable profile: first
substitute the named measured knee/plateau/overload integer rate into a copy
under `target/stress/profiles/` with the deterministic generator below; it
validates every emitted concrete profile before writing it.

```bash
cargo build --release -p tools --all-features
mkdir -p target/stress

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s1-embedded-sf1.json \
  --out-dir target/stress/s1-embedded-sf1 --measured-run
```

The base runner intentionally has `cugraph` disabled. Its S1 feature-off cell
keeps the relational cohort duration-paced for the full declared window and
adds a fixed schema-only `cugraph_pagerank` UDTF sample beside live relational
attempts; it needs no DBLP fixture and fails closed unless every graph attempt
seals `UnsupportedOperator` with `reason=required_feature_disabled`,
`required_feature=cugraph`, `host_fallback=disabled`, and operation
`algeon_datafusion.cugraph_sql.execute`. The artifact also requires zero native
fault evidence, a 250 ms per-rejection bound, and four GPU-reaching relational
measurement successes. Build this one without the cugraph suffix:

```bash
cargo build --release -p tools --all-features

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s1-feature-off-sf1.json \
  --out-dir target/stress/s1-feature-off-sf1 --measured-run
```

All other profiles that select ordinary cuGraph workloads require
`single-node-multi-gpu-validate-run-cugraph`; the base runner rejects them
before a backend is constructed instead of treating their graph errors as a
generic workload outcome.

## Targeted Nsight memory calibration

Use `stress-s2-scan-nsys-calibration-sf10.json` for the one isolated S2 scan
spot check. It retains the normal SF10 claim-versus-CUDA/NVML calibration but
has two 15-second steps (8 then 16 concurrency), rather than capturing the
whole six-minute sweep. GPU metrics need administrator profiling permission on
hosts with `RmProfilingAdminOnly=1`; `sudo -n` is intentional. Use
`--cuda-trace-scope=system-wide` under the GPU lock: process-tree tracing can
interfere with the runner's provenance subprocesses, while the lock prevents
another workload from entering the system-wide trace.

The current qualification host's Nsight Systems 2025.6 / NVIDIA 580.173.02
combination does not reliably publish a report after a system-wide metrics
capture finishes. The runner artifact from such a timeout remains useful for
its CUDA/NVML gate, but the missing trace is `not_evidenced`. Keep the outer
timeout so a failed profiler teardown cannot strand the GPU lock, and do not
substitute an older bound trace for current source.

```bash
RUN_DIR=target/stress/s2-scan-nsys-calibration-sf10
TRACE_PREFIX=target/stress/nsys/s2-scan-nsys-calibration-sf10
PROFILE=crates/tools/fixtures/stress/stress-s2-scan-nsys-calibration-sf10.json

flock /tmp/cudf-gpu.lock timeout --signal=INT --kill-after=10s 120s sudo -n \
  env "HOME=$HOME" "RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}" \
  "CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}" "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  "CUDF_INSTALL_DIR=${CUDF_INSTALL_DIR:-target/native/cudf-install}" \
  "CUVS_INSTALL_DIR=${CUVS_INSTALL_DIR:-target/native/cuvs-install}" \
  nsys profile --run-as=ubuntu --duration=90 --trace=cuda,nvtx,osrt \
  --cuda-trace-scope=system-wide --cuda-memory-usage=true \
  --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=10000 \
  --force-overwrite=false \
  --output="$TRACE_PREFIX" \
  target/release/algeon_tools single-node-multi-gpu-validate \
  --config="$PROFILE" --out-dir="$RUN_DIR" --measured-run

nsys stats --force-export=true --report cuda_api_sum --format csv "${TRACE_PREFIX}.nsys-rep" \
  > "${TRACE_PREFIX}.cuda_api_sum.csv"

jq -n \
  --arg run_profile "$PROFILE" \
  --arg run_profile_sha256 "$(sha256sum "$PROFILE" | awk '{print $1}')" \
  --arg release_runner target/release/algeon_tools \
  --arg release_runner_sha256 "$(sha256sum target/release/algeon_tools | awk '{print $1}')" \
  --arg run_artifact "$RUN_DIR/run-artifact.json" \
  --arg run_artifact_sha256 "$(sha256sum "$RUN_DIR/run-artifact.json" | awk '{print $1}')" \
  --arg nsys_trace "${TRACE_PREFIX}.nsys-rep" \
  --arg nsys_trace_sha256 "$(sha256sum "${TRACE_PREFIX}.nsys-rep" | awk '{print $1}')" \
  --arg nsys_sqlite "${TRACE_PREFIX}.sqlite" \
  --arg nsys_sqlite_sha256 "$(sha256sum "${TRACE_PREFIX}.sqlite" | awk '{print $1}')" \
  '{run_profile: $run_profile, run_profile_sha256: $run_profile_sha256,
    release_runner: $release_runner, release_runner_sha256: $release_runner_sha256,
    run_artifact: $run_artifact, run_artifact_sha256: $run_artifact_sha256,
    nsys_trace: $nsys_trace, nsys_trace_sha256: $nsys_trace_sha256,
    nsys_sqlite: $nsys_sqlite, nsys_sqlite_sha256: $nsys_sqlite_sha256}' \
  > "$RUN_DIR/nsys-memory-binding.json"
```

Only bind a trace after the outer wrapper exits successfully, the runner
artifact is qualifying, and the nonempty `.nsys-rep` and exported `.sqlite`
both exist. Bind the exact profile, release runner, artifact, trace, and SQLite
paths with their SHA-256 values in `nsys-memory-binding.json`; an interrupted,
discarded, non-qualifying, or unbound trace is diagnostic only and is not
qualification evidence. Nsight checks allocation/free lifetime, hidden copies,
and NVTX-to-CUDA correlation. The runner's same-run CUDA/NVML deltas remain
the physical VRAM byte authority.

## S1 OTLP collector cell

W8's export-path cell requires a live OTLP/gRPC collector at the exact
`host:port` supplied to the runner. Start the collector before the GPU run and
redirect its output to a retained log next to the run artifact; retain the
collector configuration and its SHA-256 with that log. The harness records a
successful exporter RPC, while the collector log is the receiver-side
diagnostic record if an export fails. This cell exercises the reference
histogram exporter; it does not claim a separately configured scalar-metric
SDK pipeline.

For a collector listening on `127.0.0.1:4317` with an OTLP/gRPC receiver:

```bash
mkdir -p target/stress/s1-otlp-sf1
target/stress/otelcol-v0.158.0/otelcol --config target/stress/otelcol-v0.158.0/collector.yaml \
  > target/stress/s1-otlp-sf1/collector.log 2>&1 &
sha256sum target/stress/otelcol-v0.158.0/collector.yaml \
  > target/stress/s1-otlp-sf1/collector-config.sha256

cargo build --release -p tools --all-features

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s1-embedded-sf1.json \
  --out-dir target/stress/s1-otlp-sf1/run \
  --measured-run \
  --otlp-metrics-endpoint http://127.0.0.1:4317 \
  --otlp-export-interval-ms 1000
```

Do not point this command at an OTLP/HTTP receiver (commonly port 4318): the
reference exporter uses OTLP/gRPC. A qualifying artifact has
`operability.otlp_export.status = "evidenced"`, at least one successful export,
and zero failed exports. A requested exporter that emits no successful export
is deliberately non-qualifying.

The listed duration is the configured warmup plus measurement time. After the
last submission, stress profiles permit at most their recorded
`completion_drain_grace_seconds`
`completion_drain_grace_seconds` for already-admitted work to seal its report;
this allowance never extends the measurement window or permits new work.
Budget a little longer for connection/setup, artifact writing, and fixture
preflight. A run that overruns or cannot meet its profile is evidence, not a
reason to edit the profile in place.

## In-process profiles

| Profiles | Required host/fixtures | Configured runtime |
| --- | --- | --- |
| `stress-s1-feature-off-sf1.json` | 4 L40S; TPC-H SF1 only (the graph probe uses an in-memory schema, not DBLP) | 2m10s |
| `stress-s1-embedded-sf1.json` | 4 L40S; SF1 TPC-H/TPC-DS + ClickBench + DBLP v12 | 2m10s |
| `stress-s1-embedded-sf10.json`, `stress-s1-embedded-cache-sf10.json` | 4 L40S; SF10 TPC-H/TPC-DS + ClickBench + DBLP v12 | 10m30s each |
| `stress-s1-embedded-{shared,distinct}-graph-interleave-sf10.json` | paired 4-L40S controls: SF10 TPC-H `q01`/`q06` plus the same G1-G4 mix over one DBLP v12 graph or eight source-hash slices | 10m30s each |
| `stress-s2-envelope-sweep-sf1.json` | 4 L40S; SF1 TPC-H/TPC-DS + DBLP v12 | 1m05s |
| `stress-s2-sweep-scan-sf1.json` | 4 L40S; SF1 TPC-H | 1m05s |
| `stress-s2-scan-nsys-calibration-sf10.json` | 4 L40S; SF10 TPC-H `q06`; targeted Nsight calibration | 1m05s |
| `stress-s2-sweep-aggregate-sf10.json`, `stress-s2-sweep-scan-sf10.json` | 4 L40S; SF10 TPC-H named by the profile | 6m10s each |
| `target/stress/profiles/stress-s2-sweep-g{1,2,3,4}-*-sf10.json` | 4 L40S; DBLP v12 named by the generated profile | 4m10s (G1) or 5m10s (G2-G4) |
| `target/stress/profiles/stress-s2-rate-{aggregate,scan,graph-io,g1-force-atlas2,g2-bfs-deep,g3-pagerank,g4-betweenness}-sf10.json` | 4 L40S; named SF10/DBLP fixtures; generated from the measured knee below | `10s + 13 × rate_phase_seconds` each |
| `stress-s3-utilization-sf1.json`, `stress-s3-utilization-telemetry-off-sf1.json` | 4 L40S; SF1 TPC-H/TPC-DS + DBLP v12 | 4m10s each |
| `stress-s3-heterogeneous-sf1.json` | 4 L40S; SF1 TPC-H/TPC-DS + DBLP v12 | 1m |
| `stress-s3-utilization-sf10.json` | 4 L40S; SF10 TPC-H/TPC-DS + DBLP v12 | 8m10s |
| `stress-s4-plateau-sf1.json` | 4 L40S; SF1 TPC-H/TPC-DS + DBLP v12 | 2m35s |
| `stress-s4-plateau-cache-off-sf10.json.template` | 4 L40S; SF10 TPC-H/TPC-DS + DBLP v12; substitute `__PLATEAU_QPS__`, `__SPIKE_QPS__`, `__DRAIN_QPS__` | 9m40s |
| `stress-s4-plateau-cache-on-sf10.json.template` | 4 L40S; SF10 TPC-H/TPC-DS + eight DBLP source-hash slices; substitute `__PLATEAU_QPS__`, `__SPIKE_QPS__`, `__DRAIN_QPS__` | 9m40s |
| `stress-s5-storms-sf1.json` | 4 L40S; SF1 TPC-H + DBLP v12 | 50s |
| `stress-s5-completion-sink-sf1.json` | 4 L40S; SF1 TPC-H; invoke with `--slow-report-sink-ms 20` | 1m05s plus at most one 20 ms drain slot per started attempt |
| `stress-s5-deadline-storm-sf10.json`, `stress-s5-cancellation-storm-sf10.json` | 4 L40S; SF10 TPC-H + DBLP v12 | 5m10s each |
| `stress-s5-overload-sustained-sf10.json.template` | 4 L40S; SF10 TPC-H + DBLP v12; substitute `__OVERLOAD_QPS__` | 5m10s |
| `stress-s6-close-under-load-sf1.json` | 4 L40S; SF1 TPC-H + DBLP v12 | 1m05s |
| `stress-s6-close-under-load.json` | 4 L40S; SF10 TPC-H + DBLP v12 | 2m10s |
| `stress-s6-quarantine-cycles-sf1.json` | 4 L40S; SF1 TPC-H + DBLP v12 | 1m35s |
| `stress-s6-quarantine-cycles.json` | 4 L40S; SF10 TPC-H + DBLP v12 | 5m10s |
| `stress-s6-quarantine-cache-sf10.json` | 4 L40S; SF10 TPC-H + resident-eligible DBLP BFS/PageRank; 6 GiB cache per ordinal | 5m40s plus cycle drain |
| `stress-s7-cache-sf1.json` | 4 L40S; SF1 TPC-H + DBLP v12 | 1m05s |
| `stress-s7-cache-thrash-sf10.json` | 4 L40S; SF10 TPC-H + DBLP v12 | 5m10s |
| `stress-s7-cache-stress-sf10.json.template` | 4 L40S; SF10 TPC-H + DBLP v12; substitute `__KNEE_QPS__` | 5m10s |
| `stress-s7-cache-distinct-graph-churn-sf10.json.template` | 4 L40S; eight DBLP source-hash slices with G2/G3 traffic; substitute `__KNEE_QPS__` | 5m10s |
| `stress-s7-cache-same-key-sf1.json` | 4 L40S; SF1 TPC-H + DBLP v12 | 1m10s |

The two S1 SF10 profiles deliberately use the same eight-way, zero-warmup
cohort: TPC-H `q01` and `q06`, TPC-DS `q10`, ClickBench `q28`, then G1-G4.
The runner repeats one closed-loop cohort at a time through the full
600-second measurement window, followed by the 30-second drain grace. This
bounds the historical ~492-second `q01` execution inside the cell while
preserving the required big/small interleave. The former broad catalog ordering
put `q01`/`q07` in a 300-second warmup cohort, leaving zero measurement rows;
retain that non-qualifying diagnostic. Cache A/B comparison requires matching
scheduling and workloads; only cache configuration may differ.

`stress-s1-embedded-shared-graph-interleave-sf10.json` and
`stress-s1-embedded-distinct-graph-interleave-sf10.json` are a paired
population control, not part of that cache A/B pair. They have identical
backend, scheduling, relational queries, ordered G1-G4 labels, and graph
claim requirements; only the DBLP source identity differs. Both use the eight
active device slots and repeat the ten-request stream through the 600-second
window. The distinct profile routes its G1-G4 requests across eight
independently authenticated slices, while the shared control routes the same
ordered requests through one DBLP v12 rowset. Their delta therefore prices
graph-population churn rather than a changed load shape.

## S3 telemetry and envelope cells

The telemetry A/B profiles have identical SF1 load, queue, device, and cache
settings. The ordinary profile omits `install_telemetry`, whose profile default
is `true`; the companion explicitly sets it to `false` for every ordinal.
Run both under the same host conditions, then compare their
`aggregate.latency.admission_wait_p50_ns`/`p95_ns`/`p99_ns` and
`time_series.observation_pull` fields. The resolved placement configuration
records `install_telemetry` per ordinal, so a stale or incorrectly built
profile cannot be mistaken for the telemetry-off half.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s3-utilization-sf1.json \
  --out-dir target/stress/s3-utilization-telemetry-on-sf1 --measured-run

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s3-utilization-telemetry-off-sf1.json \
  --out-dir target/stress/s3-utilization-telemetry-off-sf1 --measured-run

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s3-heterogeneous-sf1.json \
  --out-dir target/stress/s3-heterogeneous-sf1 --measured-run
```

The heterogeneous artifact must evidence `heterogeneous_ordinal_profiles` and
all-device progress from resolved startup envelopes. The telemetry pair records
the admission-latency A/B; it does not infer that telemetry was absent merely
because a snapshot happened to have no NVML values.

## S2 knee-rate profile instantiation

Derive a rate profile only from a qualifying final-source S2 concurrency
sweep. Record the selected sweep step, its observed `gpu_reaching_qps`, and
the knee rationale: the first step where GPU-reaching QPS has flattened while
queue-wait p95 turns up. The plan intentionally does not supply a numeric
flattening threshold, so the artifact summary must retain the full per-step
curve and explain that selection; do not infer it from an old artifact.

The rate profile accepts integer arrivals per second. Let `Q` be the selected
step's observed GPU-reaching QPS and derive `K` as the smallest positive even
integer no lower than `Q`:

```
K = max(2, 2 * ceil(Q / 2))
```

This rounds every odd or fractional knee upward once, so all six exact rate
multipliers are integers: `K/2`, `K`, `3K/2`, `2K`, `5K`, and `10K`. Record
both `Q` and `K`; do not independently round the individual rate phases.

The generator derives the same `rate_phase_seconds` duration for all thirteen
phases: a delay and a drop pass over the six rates, followed by a final
`0.5K` delay recovery phase. It uses:

```
W = min(60, floor((2 × (80_000 - 5K)) / (81K)))
```

The schedule's expected arrival budget is `5K + 81KW/2`: `5K` covers the
ten-second warmup at `0.5K`, `40KW` covers both delay and drop passes over the
six rate multipliers, and the final `KW/2` is recovery. The 80,000 budget
leaves 20,000 attempts below the 100,000 retention cap. `W` must be at least
one; the generator rejects a non-even `K` and any `K` above 1,758 (where
`W = 1`, expected 79,989 arrivals). A larger measured knee
needs a profile/scheduler design change before a qualifying rate run, not a
truncated artifact. Every artifact must still prove
`arrival_plan.truncated_by_attempt_cap = false`, because the seeded Poisson
schedule is the authoritative count.

Keep the generated JSON and the source in-process artifact beside the resulting
artifact. For example, `K=42` gives `W=46` and rates
`21,42,63,84,210,420`, followed by a final `21/s` recovery:

```bash
mkdir -p target/stress/profiles
python3 crates/tools/fixtures/stress/generate_s2_graph_profiles.py \
  --out-dir target/stress/profiles --knee-qps 42
python3 crates/tools/fixtures/stress/generate_s2_graph_profiles.py \
  --out-dir target/stress/profiles --knee-qps 42 --check
```

## S2 claim-versus-VRAM calibration

Every SF10 S2 sweep/rate profile has `vram_calibration`. Before using a knee,
inspect `execution.measurement.time_series.vram_calibration`: each configured
ordinal must have a synchronous pre-load CUDA-free baseline, a CUDA-peak row,
the contemporaneous admission-plus-cache-ledger claim, CUDA and (when
available) NVML deltas, and `status = "evidenced"`. The current profiles allow
only a fixed 64 MiB difference in either direction; they do not contain a
percentage multiplier. A larger actual-over-claim delta is an unsafe admission
contract. A larger claim-over-actual delta is stranded capacity. Both are
non-qualifying and must be repaired at the planner/allocation owner before
retaining the knee. The sampler brackets every CUDA/NVML pull with ledger
sequences: a row is eligible only when its admission-plus-cache claim did not
change across that bracket. The baseline is never a peak candidate; every
ordinal also needs a post-baseline stable row with both nonzero CUDA use and a
nonzero claim. A zero-versus-zero baseline, or an unstable ledger/telemetry
pair, is `failed`, not calibration evidence.

For any profile with `graph_claim_calibration` (G1-G4 and graph-algorithm
S5-S7 cells), also inspect
`execution.measurement.graph_claim_calibration`. Every measurement-phase graph
attempt must have a complete sealed typed execution classification. A physical
attempt must retain exactly one non-truncated engine row with
`execution_state=algorithm_started`, a measured direct peak, and
`reservation_high_water`, `cache_scope_high_water`, or `pool_delta`
provenance. A `cache_scope_high_water` is direct only after the cache scope's
fence and health proof completed, including an error path. A
`bfs_singleflight_follower` must retain zero graph rows plus its engine-minted
leader attempt id, which must reference an evidenced physical measurement-phase
leader; it never reuses the leader's peak. A
`graph_memory_admission_denied` row can replace a peak only with its exact
pre-executor engine state, `requested_bytes > budget_bytes`, and a sealed
native `MemoryReservationFailed` temporary execution failure. Missing,
conflicting, duplicate, unavailable, or classification/row-mismatched evidence
is `not_evidenced`; do not substitute device-wide telemetry for an algorithm
peak. DBLP `edge_scan` is relational GPU work, so it uses the device-level
calibration only.

Capture one representative final-source S2 class with the targeted procedure
above after the normal release build. This is a cross-check for CUDA allocation
lifetimes, hidden copies, and phase attribution; it does not replace the
artifact's CUDA/NVML accounting because RMM pool suballocations need not appear
as one CUDA allocation per operation.

The manifest is mandatory because the runner writes its artifact before the
`nsys` wrapper has finalized the `.nsys-rep`; passing that future output as
`--nsight-trace` would falsely hash a prior trace. Retain the profile, release
runner, run artifact, binding manifest, trace, and exported SQLite report as
one calibration set. Nsight must agree on allocation/free lifetime and phase
ownership before adjusting a memory model; it is not valid to replace a missing
runtime peak with NVML or a trace-derived total.

I8 uses a single deliberately slow completion receiver, not a synthetic
backend fault. Its rate (51/s) is strictly above the one-per-20-ms serial sink
capacity (50/s). Retain the artifact even when it is non-qualifying: its
`completion_sink` section records offered-versus-drain pressure, exact report
accounting, join failures, and quiescent close.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s5-completion-sink-sf1.json \
  --out-dir target/stress/s5-completion-sink-sf1 --measured-run \
  --slow-report-sink-ms 20
```

## S7 cache qualification cells

S7 compares a cache-disabled companion derived by the runner with the named
6-GiB-per-ordinal profile; do not substitute a separately edited cache-off
JSON. Both cells require the cugraph runner and execute the paired A/B run
with `--cache-calibration` under the GPU lock.

```bash
cargo build --release -p tools --all-features

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s7-cache-thrash-sf10.json \
  --out-dir target/stress/s7-cache-thrash-sf10 \
  --measured-run --cache-calibration

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s7-cache-same-key-sf1.json \
  --out-dir target/stress/s7-cache-same-key-sf1 \
  --measured-run --cache-calibration
```

Retain both `run-artifact.json` files. A qualifying artifact has status
`cache_calibration`, all cache clauses `evidenced`, and qualifying cache-off
and cache-on measurements. The thrash cell additionally records at least
48 GiB of cumulative source promotion across the four devices, an eviction
batch, direct physical-domain bytes never above the paired ledger charge, and
no unexplained cache-on-only admission failure. The same-key cell must retain
nonzero source-promotion coalescing and BFS follower evidence with fewer
physical BFS traversals than eligible BFS requests; it uses the unbounded
`bfs` catalog entry deliberately, because `bfs_deep` is depth-bounded and is
not eligible for exact-request BFS coalescing.

The distinct-graph churn template rotates PageRank over all eight authenticated
slice paths and adds `bfs_deep` only on `source-hash-07`. It exercises the
G2/G3 cache population under the same cache-calibration discipline, but does
not use the same-key probe: its point is eviction and rebuild across distinct
`ResidentGraphKey`s.

## S8 Flight server profiles

S8 needs the same host, a release `algeon_server` built with
`cugraph`, and the named TPC-H fixture. The Flight profile records the
driver, SQL label, warmup/measurement window, connection mode, server TOML
companion, and the exact in-process artifact/cell. `flight-sql-stress` refuses
to start unless that artifact is a measured single-node multi-GPU run artifact
(`algeon_tools.single_node_multi_gpu.run_artifact`), selects the same query,
and has the matching load cell. It writes only hashes and redacted projections;
raw SQL and server configuration are not copied to `summary.json`.

| Profiles | Baseline and server config | Configured runtime |
| --- | --- | --- |
| `stress-s8-flight-pooled-sf1.flight-profile`, `stress-s8-flight-churn-sf1.flight-profile` | `target/stress/s2-sweep-scan-sf1/run-artifact.json`, step 3; `server-s8-s2-scan-sf1.toml`; SF1 `lineitem.parquet` | 20s each |
| `stress-s8-flight-pooled-knee-sf10.flight-profile`, `stress-s8-flight-churn-knee-sf10.flight-profile` | `target/stress/s2-sweep-scan-sf10/run-artifact.json`, step 3; `server-s8-s2-knee-sf10.toml`; SF10 `lineitem.parquet` | 1m10s each |

Run the matching in-process cell first. Then build the server before the GPU
lock, start it in one terminal with the profile's companion TOML, register
`lineitem` in its mutable local-Parquet workspace, and run the Flight client in
another terminal. The server owns the GPU lock; do not lock the client as well.

```bash
cargo build --release -p tools --all-features
mkdir -p target/stress
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/stress-s2-sweep-scan-sf1.json \
  --out-dir target/stress/s2-sweep-scan-sf1 --measured-run

cargo build --release -p algeon-server --all-features --bin algeon_server
cargo build --release -p tools --all-features

ALGEON_SERVER_CONFIG_FILE=fixture/stress/server-s8-s2-scan-sf1.toml \
ALGEON_SERVER_BIND=127.0.0.1:50051 \
flock /tmp/cudf-gpu.lock target/release/algeon_server

# In another terminal after the server accepts connections. Use SF1 here;
# replace the location with the SF10 lineitem file for the knee profiles.
target/release/algeon_tools flight-sql-query http://127.0.0.1:50051 \
  "CREATE EXTERNAL TABLE lineitem STORED AS PARQUET LOCATION '/opt/dlami/nvme/fixture/data/tpch/sf1/parquet/lineitem.parquet'"

target/release/algeon_tools flight-sql-stress \
  --config crates/tools/fixtures/stress/stress-s8-flight-pooled-sf1.flight-profile \
  --out-dir target/stress/s8-flight-pooled-sf1
```

`summary.json` contains `matched_in_process` (artifact/profile hashes and the
selected aggregate) plus `delta_from_in_process`. Throughput and p50/p95/p99
are server-minus-in-process values. GPU-reaching QPS is deliberately marked
`not_observable_over_flight`, which is an S8 observability finding rather than
a fabricated server-side metric.
