# Algorithm scale measurements

[`merrec_benchmark.py`](../../../../../scripts/dev/merrec_benchmark.py) generates
SQL for the existing [`merrec_run.py`](../../../../../scripts/dev/merrec_run.py)
client. It preserves that client's result JSON, statistics and runtime artifacts.
The suite uses fixed inputs and nested cohorts. Runtime duration has no minimum
acceptance threshold. A case that completes below one second remains in the
comparison.

| Family | Scale | Timing boundary | CPU comparison |
| --- | --- | --- | --- |
| cuGraph item-user PageRank | May events, May-June, May-August, May-October | Complete query and the same graph from prepared weighted edges | Requires a separately measured, validated weighted undirected PageRank baseline |
| cuVS 64-dimensional exact kNN | 100, 1,000, 10,000, 100,000 shoppers queried against the complete taste table | Prepared-input scan, matrix packing, index handling, search and complete result receipt | FAISS `IndexFlatL2` on the identical sorted Float32 inputs |
| cuVS CAGRA | Identical dataset and query cohorts as exact kNN | Fresh index construction and search on each invocation | Exact distances provide the quality reference; build and search remain separate observations |
| cuML forest inference | Existing EMBER test cohorts and TPC-H lineitem cohorts | Existing `forest_predict` harness import, packing, prediction and assembly stages | Existing source-predictor and Treelite sample verification |

Keep `10_barbie_tie_ins.sql`, `31_world_bridges.sql` and
`30_audience_funnel.sql` in the complete-query evidence. Their input preparation
and result composition answer different cost questions from prepared-input
search. Preserve slower cases and failures when selecting a scale for a public
example. A scale crossover needs both CPU and GPU measurements at each reported
scale; generated SQL alone establishes no crossover.

## Generate and execute vector cases

Use a release server under the device lock as described in [README.md](README.md).
Record the binary checksum, server configuration, CPU partition count, device
inventory and native library identities with each measurement. Use separate
output directories for each binary or configuration. CPU timings need the same
host without concurrent compilation or other CPU benchmarks.

The prepared taste table comes from `14_taste64.sql`. Its Float32 `d0` through
`d63` features and IDs are shared by both implementations. The dataset remains
complete at every query scale. Query cohorts contain the first IDs in increasing
order and include self matches. Record the actual query count when a source has
fewer shoppers than the requested limit. CAGRA builds a fresh query-local index
for each statement.

```bash
MERREC_ROOT=/mnt/algeon-nvme/algeon/fixture/demo/merrec
MERREC_TASTES="$MERREC_ROOT/scale-results/full-demos/14_taste64.parquet"
MERREC_SUITE="$PWD/target/merrec-suite"
MERREC_PYTHON=/tmp/algeon-amazon-venv/bin/python

python3 scripts/dev/merrec_benchmark.py generate \
  --tastes "$MERREC_TASTES" --output "$MERREC_SUITE/sql" \
  --results-dir "$MERREC_SUITE/gpu-results"

"$MERREC_PYTHON" scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 --register \
  --data-dir "$MERREC_ROOT/parquet" --report-dir "$MERREC_SUITE/gpu-report" \
  --results-dir "$MERREC_SUITE/gpu-results" --timeout 1800 \
  "$MERREC_SUITE/sql/register_tastes.sql"

"$MERREC_PYTHON" scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 \
  --report-dir "$MERREC_SUITE/gpu-report" --results-dir "$MERREC_SUITE/gpu-results" \
  --timeout 1800 --repeats 5 \
  "$MERREC_SUITE/sql/knn_exact_100.sql" \
  "$MERREC_SUITE/sql/knn_exact_1000.sql" \
  "$MERREC_SUITE/sql/knn_exact_10000.sql" \
  "$MERREC_SUITE/sql/knn_exact_100000.sql"
```

Run the corresponding `knn_cagra_*.sql` files against the same server and
retain their results even when construction costs exceed exact search.
Registration fails when a suite table already exists. Reuse an existing
registration only for repetitions with unchanged files; start a fresh server
before changing the taste path or prepared graph snapshots.

The CPU baseline uses the provisioned environment's `faiss-cpu`, NumPy and
PyArrow. It records those library versions, thread count, source checksum and
all repetitions. It measures input load, sorting and packing, the flat-index
copy, exhaustive search and Arrow assembly separately. Parquet export and
correctness checks occur after those timers. Its prepared-input total excludes
Flight transport and source feature construction. GPU search observations
include output validation and can include build work still pending when the
search scope begins. Compare these recorded scopes with their boundaries
stated; exclusive GPU build and search durations require a CUDA timeline or
completion events at the corresponding boundaries. Compare complete-query
latency in its stated scope separately.

Inspect the loaded CPU backend before timing. A FAISS wheel and NumPy can load
different BLAS libraries. In a provisioned environment with `threadpoolctl`,
the following read-only inspection reports their paths, architectures and
thread counts:

```bash
"$MERREC_PYTHON" - <<'PY'
import faiss
import numpy
from threadpoolctl import threadpool_info
print(faiss.__version__, faiss.get_compile_options())
print(threadpool_info())
PY
```

Inspect the host's supported instruction sets before overriding a detected
BLAS target. OpenBLAS accepts a host-specific `OPENBLAS_CORETYPE` override for
dynamic builds ([runtime variables](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)).
The CPU report records that override, `OPENBLAS_NUM_THREADS` and
`OMP_NUM_THREADS`, together with FAISS compile options. The harness applies no
automatic target override. Preserve the loaded-library inspection alongside
the result when a target override is needed.

Select the CPU thread count with a separate tuning run while the host is idle.
For a 16-core host with 32 logical CPUs, 8, 16 and 32 threads provide a small
comparison. Include a small query batch and a larger batch because the
distance implementation can change with query count. FAISS 1.15.1 uses its
BLAS path when the query count times the dimension reaches `128000`;
the 64-dimensional suite crosses that boundary between 1,000 and 10,000
queries ([FAISS implementation](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/utils/distances.cpp#L553-L570)).
Retain tuning records, then use the selected configuration for the measured
matrix. Thread counts collected during compilation cannot establish that
selection.

Screen `cpu-knn --blas-threshold 0` for small batches when the default direct
path is slower. The override selects the BLAS distance path in FAISS while
preserving exhaustive squared-L2 search. Validate its complete results with
the same tolerance because the floating-point accumulation order changes.
An omitted flag preserves the library default; each CPU report records the
actual threshold as `blas_threshold`. Retain both path measurements and select
the validated CPU configuration before capturing the final repetitions.

```bash
for MERREC_BATCH in 100 1000 10000 100000; do
  "$MERREC_PYTHON" scripts/dev/merrec_benchmark.py cpu-knn \
    --tastes "$MERREC_TASTES" --output "$MERREC_SUITE/cpu" \
    --queries "$MERREC_BATCH" --threads 32 --repeats 5 --warmups 1 \
    --compare "$MERREC_SUITE/gpu-results/knn_exact_${MERREC_BATCH}_r5.parquet"
done
```

The complete GPU result must contain `k` distinct neighbors for each expected
query ID. The check recomputes CPU and GPU returned-candidate distances in
Float64 from the frozen Float32 input and compares their sorted distance sets
within `1e-6`. Equal-distance neighbors may have different IDs. This establishes
Float64 distance equivalence to FAISS returned candidates; it does not prove
Float64 top-k over the complete dataset.

Each implementation's reported distances are also checked against its own
recomputed candidate distances within the original `1e-6` tolerance.
`max_exact_distance_error` compares recomputed distance sets;
`max_reported_distance_error` measures reported accuracy. GPU `matches` requires
both checks. CPU reported-accuracy failures remain separate from the GPU
candidate reference and appear in `cpu_reported_accuracy` for each CPU run.

ANN quality includes ID overlap and the fraction within the recomputed kth
distance of the FAISS candidates. The candidate-equivalence check alone does
not validate an ANN speedup. Validate every GPU repetition used for a result
claim, using its own result file.

## Graph windows and preparation

Each generated graph uses `item_view` events from May 1 up to an exclusive
calendar endpoint. It sums repeat `(item_id, user_id)` interactions into
Float64 weights and maps users to `-(user_id + 1)`. PageRank uses an undirected
graph, damping `0.85`, epsilon `1e-5` and at most 100 iterations.

Run each `prepare_item_user_*.sql` once without a suffix or repeat argument.
Use the generator's `--results-dir` for that invocation; registration names
the resulting unsuffixed Parquet. Check its row count before starting graph
execution. A dataset without events in the selected window needs endpoints
appropriate for its source period, supplied by `--graph-window-ends`.

```bash
"$MERREC_PYTHON" scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 \
  --report-dir "$MERREC_SUITE/gpu-report" --results-dir "$MERREC_SUITE/gpu-results" \
  --timeout 1800 "$MERREC_SUITE/sql/prepare_item_user_20230601.sql"

"$MERREC_PYTHON" scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 \
  --report-dir "$MERREC_SUITE/gpu-report" --results-dir "$MERREC_SUITE/gpu-results" \
  --timeout 1800 "$MERREC_SUITE/sql/register_item_user_20230601.sql"

"$MERREC_PYTHON" scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 \
  --report-dir "$MERREC_SUITE/gpu-report" --results-dir "$MERREC_SUITE/gpu-results" \
  --timeout 1800 --repeats 5 \
  "$MERREC_SUITE/sql/pagerank_item_user_20230601_e2e.sql" \
  "$MERREC_SUITE/sql/pagerank_item_user_20230601_prepared_input.sql"
```

Repeat that sequence for `20230701`, `20230901` and `20231101` as device and
host capacity permit. Record input events, aggregated edge pairs, normalized
directed edges and graph vertices independently. Prepared edge export has its
own measured query; it is excluded from the prepared-input PageRank query.
Complete-query PageRank includes that aggregation. A prepared-input result
cannot replace complete-query latency in a comparison.

## Existing cuML scales

The [`forest_predict` harness](../../../../../crates/bench/src/forest_predict/mod.rs)
already provides EMBER test row caps and `lineitem_100k`, `lineitem_1m`,
`lineitem_10m` and `lineitem_full`. It records import, pack, predict and
assemble timings without changing workload environment variables. Use
[`check_forest_predict_samples.py`](../../../../../scripts/dev/check_forest_predict_samples.py)
for the source-model and Treelite correctness comparison.

```bash
target/release/forest_predict \
  --cases lineitem_100k,lineitem_1m,lineitem_10m,lineitem_full \
  --repeat 5 --output-dir target/forest-scale

~/.venvs/forest/bin/python scripts/dev/check_forest_predict_samples.py \
  --bench-dir target/forest-scale --threads 16
```

These existing correctness samples do not establish CPU latency for each
cohort. Publish a CPU/GPU inference crossover only after timing the source
predictor on the identical model, rows and Float32 feature order. Training
algorithms with different objectives or convergence policies require separate
quality and timing comparisons.
The forest executable owns its device lock; stop the Flight server before
invoking it.

## Inspect captured timing scopes

```bash
python3 scripts/dev/merrec_benchmark.py summarize \
  --report "$MERREC_SUITE/gpu-report" --warmups 1 > "$MERREC_SUITE/summary.json"

"$MERREC_PYTHON" -m unittest discover -s scripts/dev/tests -p test_merrec_benchmark.py
```

The summary retains failed runs and every raw repetition. Medians exclude the
first repetition by default. Operator stages are paired by plan execution and
span ID. Their sums are cumulative host time. The native span union accounts
for overlap, and its fraction is omitted when capture completeness or paired
events are missing. Vector `build_index` and `search` observations remain
separate. Forest import remains outside the native compute fraction.

Graph task spans include input consumption, graph construction and execution.
Vector `search` spans end after `search_into` validates its outputs. Validation
reduces device values to a host scalar and synchronizes the execution stream,
covering completed search plus output validation. CAGRA also normalizes missing
neighbor sentinels within this scope.

The `build_index` span ends without a completion fence. Brute-force index
construction can leave dataset norm computation pending; CAGRA can leave its
final graph copy pending on the root stream. The following completed `search`
scope can therefore include that remaining build work. Retain the separate
host-call observations, and use a CUDA timeline or completion events to
attribute exclusive GPU build, search or kernel durations. Existing
`task_runtime_rows` may merge task IDs across native fragments; the summary
reads runtime spans with their plan execution identity.
