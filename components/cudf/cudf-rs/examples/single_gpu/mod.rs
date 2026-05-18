#![allow(dead_code)]

//! Single-GPU multi-thread OOM isolation experiment.
//!
//! Tests whether a GPU OOM in one `spawn_blocking` thread cascades to other
//! threads sharing the same physical GPU.  Unlike the `multi_gpu` examples
//! (which test cross-device isolation), here all threads target **device 0**
//! and compete for the same VRAM.
//!
//! Three execution models are tested:
//!
//! 1. **Shared Runtime** — all threads clone a single `Runtime` (same stream,
//!    same memory resource).
//! 2. **Isolated Runtimes** — each thread builds its own `Runtime` (separate
//!    stream, separate `DeviceMemoryResource::cuda()`).
//! 3. **Concurrent heavy allocators** — all threads do allocation-heavy work
//!    (re-reading parquet, concat) simultaneously, forcing allocations to
//!    overlap in time with the OOM thread's growth.

use std::{
    path::Path,
    sync::{Arc, Barrier},
    time::Instant,
};

use cudf::{
    AggSpec, DataFrame, DeviceId, DropNaOptions, Error, ErrorKind, ExecutionContext, GroupByAgg, InfoDetail,
    ParquetReadOptions, Runtime, Series, SortOptions,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_PARQUET_PATH: &str = "/home/ubuntu/data/node.parquet";
const NODE_COLUMNS: [&str; 5] = ["entityName", "entityType", "risk_score", "influence", "importance"];

/// Rows read by healthy workflows — large enough to do real work.
const HEALTHY_NUM_ROWS: usize = 120_000;

/// Rows read by the OOM trigger workflow — the base before concat-doubling.
const OOM_BASE_NUM_ROWS: usize = 64_000;

/// Number of iterations for healthy workflows. Must be high enough that
/// healthy threads are still running when the OOM thread hits OOM.
const HEALTHY_ITERATIONS: usize = 8;

/// Maximum doublings before we give up on triggering OOM.
const MAX_CONCAT_DOUBLINGS: usize = 64;

/// Top-N rows to keep per sort-filter iteration.
const TOP_N: usize = 4_096;

/// Number of concurrent workflows.
const NUM_WORKFLOWS: usize = 4;

/// Iterations for allocation-heavy healthy workflows (concurrent-alloc stress test).
/// Each iteration re-reads parquet and concats, so these are heavy allocation ops.
const HEAVY_ALLOC_ITERATIONS: usize = 30;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    Completed,
    ExpectedOom,
    CascadeOom,
    UnexpectedError,
    UnexpectedSuccess,
}

impl WorkflowStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "Completed",
            Self::ExpectedOom => "ExpectedOom",
            Self::CascadeOom => "CascadeOom",
            Self::UnexpectedError => "UnexpectedError",
            Self::UnexpectedSuccess => "UnexpectedSuccess",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkflowReport {
    pub label: &'static str,
    pub status: WorkflowStatus,
    pub elapsed_ms: u128,
    pub summary: String,
}

#[derive(Clone)]
struct WorkflowContext {
    example_name: &'static str,
    label: &'static str,
    device: DeviceId,
    /// Optional barrier for coordinated start (used by concurrent-alloc test).
    barrier: Option<Arc<Barrier>>,
}

impl WorkflowContext {
    fn log(&self, phase: &str, detail: impl AsRef<str>) {
        println!(
            "[{}] device={} workflow={} phase={} {}",
            self.example_name,
            self.device.ordinal(),
            self.label,
            phase,
            detail.as_ref()
        );
    }

    fn wait_barrier(&self, msg: &str) {
        if let Some(ref barrier) = self.barrier {
            self.log("BARRIER", msg);
            barrier.wait();
        }
    }
}

#[derive(Clone)]
struct WorkflowSpec {
    label: &'static str,
    expect_oom: bool,
    run: fn(WorkflowContext, Runtime) -> cudf::Result<String>,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run the shared-runtime experiment: all 4 threads clone one `Runtime`.
pub async fn run_shared_runtime_oom() -> cudf::Result<()> {
    let device = DeviceId::new(0)?;
    let runtime = Runtime::builder().with_device(device).build()?;

    println!("[single_gpu_shared_runtime_oom] Runtime: {:?}", runtime);
    println!("[single_gpu_shared_runtime_oom] All 4 threads will SHARE this runtime (same stream, same MR)");
    println!();

    let specs = build_specs();
    run_example("single_gpu_shared_runtime_oom", device, specs, None, move |_spec| runtime.clone()).await
}

/// Run the isolated-runtime experiment: each thread builds its own `Runtime`.
pub async fn run_isolated_runtime_oom() -> cudf::Result<()> {
    let device = DeviceId::new(0)?;

    println!("[single_gpu_isolated_runtime_oom] Each thread will build its OWN Runtime (separate stream, separate MR)");
    println!();

    let specs = build_specs();
    run_example("single_gpu_isolated_runtime_oom", device, specs, None, move |_spec| {
        Runtime::builder().with_device(device).build().expect("failed to build per-thread Runtime")
    })
    .await
}

/// Run the concurrent heavy-allocation experiment: all threads do allocation-
/// heavy work simultaneously.  A barrier ensures all threads begin their
/// allocation-heavy phase at the same time.
///
/// The 3 healthy threads repeatedly re-read parquet and concat results,
/// generating continuous GPU allocations that overlap in time with the OOM
/// thread's concat-doubling growth.
pub async fn run_concurrent_alloc_oom() -> cudf::Result<()> {
    let device = DeviceId::new(0)?;
    let barrier = Arc::new(Barrier::new(NUM_WORKFLOWS));

    println!("[single_gpu_concurrent_alloc_oom] All threads use ISOLATED Runtimes on device 0");
    println!("[single_gpu_concurrent_alloc_oom] Healthy threads will continuously allocate (re-read parquet + concat)");
    println!("[single_gpu_concurrent_alloc_oom] A barrier synchronizes the start so allocations overlap");
    println!();

    let specs = build_concurrent_alloc_specs();
    run_example("single_gpu_concurrent_alloc_oom", device, specs, Some(barrier), move |_spec| {
        Runtime::builder().with_device(device).build().expect("failed to build per-thread Runtime")
    })
    .await
}

// ---------------------------------------------------------------------------
// Internal machinery
// ---------------------------------------------------------------------------

fn build_specs() -> [WorkflowSpec; NUM_WORKFLOWS] {
    [
        WorkflowSpec { label: "iterative-sort-filter", expect_oom: false, run: workflow_iterative_sort_filter },
        WorkflowSpec { label: "iterative-groupby", expect_oom: false, run: workflow_iterative_groupby },
        WorkflowSpec { label: "series-arithmetic", expect_oom: false, run: workflow_series_arithmetic },
        WorkflowSpec { label: "oom-concat-growth", expect_oom: true, run: workflow_oom_concat_growth },
    ]
}

async fn run_example<F>(
    example_name: &'static str,
    device: DeviceId,
    specs: [WorkflowSpec; NUM_WORKFLOWS],
    barrier: Option<Arc<Barrier>>,
    runtime_factory: F,
) -> cudf::Result<()>
where
    F: Fn(&WorkflowSpec) -> Runtime + Send + Sync + 'static,
{
    let factory = Arc::new(runtime_factory);
    let mut handles = Vec::with_capacity(specs.len());

    for spec in specs {
        let ctx = WorkflowContext { example_name, label: spec.label, device, barrier: barrier.clone() };
        let factory = Arc::clone(&factory);
        handles.push(tokio::task::spawn_blocking(move || execute_workflow(spec, ctx, factory.as_ref())));
    }

    let mut reports = Vec::with_capacity(handles.len());
    for handle in handles {
        reports.push(handle.await.map_err(|err| {
            Error::new(ErrorKind::InvalidOperation, "single-GPU workflow task panicked or was cancelled")
                .with_operation("examples::single_gpu::run_example")
                .set_source(err)
        })?);
    }

    print_summary(example_name, &reports);
    print_verdict(example_name, &reports);
    Ok(())
}

fn execute_workflow<F>(spec: WorkflowSpec, ctx: WorkflowContext, factory: &F) -> WorkflowReport
where F: Fn(&WorkflowSpec) -> Runtime {
    ctx.log(
        "START",
        if spec.expect_oom {
            "launching OOM-trigger workflow (no barriers, racing naturally)"
        } else {
            "launching healthy workflow (no barriers, racing naturally)"
        },
    );

    let started = Instant::now();
    let runtime = factory(&spec);
    let result = (spec.run)(ctx.clone(), runtime);

    match result {
        Ok(summary) if spec.expect_oom => {
            ctx.log("UNEXPECTED", "workflow completed without hitting OutOfMemory");
            WorkflowReport {
                label: spec.label,
                status: WorkflowStatus::UnexpectedSuccess,
                elapsed_ms: started.elapsed().as_millis(),
                summary,
            }
        },
        Ok(summary) => {
            ctx.log("DONE", &summary);
            WorkflowReport {
                label: spec.label,
                status: WorkflowStatus::Completed,
                elapsed_ms: started.elapsed().as_millis(),
                summary,
            }
        },
        Err(ref err) if spec.expect_oom && err.kind() == ErrorKind::OutOfMemory => {
            ctx.log("OOM", format!("expected OutOfMemory: {err}"));
            WorkflowReport {
                label: spec.label,
                status: WorkflowStatus::ExpectedOom,
                elapsed_ms: started.elapsed().as_millis(),
                summary: format!("OutOfMemory at {}: {}", err.operation(), err.message()),
            }
        },
        Err(ref err) if !spec.expect_oom && err.kind() == ErrorKind::OutOfMemory => {
            // A healthy workflow hit OOM — this is a cascade failure.
            ctx.log("CASCADE_OOM", format!("healthy workflow hit OOM: {err}"));
            println!("[{}] workflow={} CASCADE_OOM_DETAIL\n{err:?}", ctx.example_name, spec.label);
            WorkflowReport {
                label: spec.label,
                status: WorkflowStatus::CascadeOom,
                elapsed_ms: started.elapsed().as_millis(),
                summary: format!("CASCADE OutOfMemory at {}: {}", err.operation(), err.message()),
            }
        },
        Err(ref err) => {
            ctx.log("ERROR", format!("{err}"));
            println!("[{}] workflow={} FULL_ERROR\n{err:?}", ctx.example_name, spec.label);
            WorkflowReport {
                label: spec.label,
                status: WorkflowStatus::UnexpectedError,
                elapsed_ms: started.elapsed().as_millis(),
                summary: format!("{} at {}: {}", err.kind(), err.operation(), err.message()),
            }
        },
    }
}

fn print_summary(example_name: &str, reports: &[WorkflowReport]) {
    println!();
    println!("[{example_name}] ========== SUMMARY ==========");
    for report in reports {
        println!(
            "[{example_name}] workflow={:<25} status={:<20} elapsed_ms={:<8} result={}",
            report.label,
            report.status.as_str(),
            report.elapsed_ms,
            report.summary,
        );
    }
    println!("[{example_name}] =============================");
}

fn print_verdict(example_name: &str, reports: &[WorkflowReport]) {
    let completed = reports.iter().filter(|r| r.status == WorkflowStatus::Completed).count();
    let expected_oom = reports.iter().filter(|r| r.status == WorkflowStatus::ExpectedOom).count();
    let cascade_oom = reports.iter().filter(|r| r.status == WorkflowStatus::CascadeOom).count();
    let unexpected_err = reports.iter().filter(|r| r.status == WorkflowStatus::UnexpectedError).count();
    let unexpected_ok = reports.iter().filter(|r| r.status == WorkflowStatus::UnexpectedSuccess).count();

    println!();
    println!("[{example_name}] ========== VERDICT ==========");
    println!(
        "[{example_name}] completed={completed} expected_oom={expected_oom} cascade_oom={cascade_oom} unexpected_error={unexpected_err} unexpected_success={unexpected_ok}"
    );

    if completed == NUM_WORKFLOWS - 1 && expected_oom == 1 && cascade_oom == 0 {
        println!("[{example_name}] ISOLATION PRESERVED: OOM was contained to the trigger thread.");
        println!("[{example_name}] All {completed} healthy workflows completed successfully.");
    } else if expected_oom == 1 && cascade_oom > 0 {
        println!("[{example_name}] OOM CASCADE DETECTED: {cascade_oom} healthy workflow(s) also hit OutOfMemory.");
        println!("[{example_name}] This means GPU OOM on one thread CAN cascade to other threads on the same GPU.");
        println!("[{example_name}] Completed: {completed}, Cascade OOM: {cascade_oom}");
        if completed > 0 {
            println!("[{example_name}] Note: {completed} workflow(s) still completed — cascade is partial, not total.");
        }
    } else if expected_oom == 0 && unexpected_ok > 0 {
        println!("[{example_name}] OOM TRIGGER FAILED: the concat-growth workflow did not hit OutOfMemory.");
        println!("[{example_name}] This GPU may have too much VRAM for this test.");
    } else {
        println!("[{example_name}] UNEXPECTED OUTCOME: review individual workflow results above.");
    }
    println!("[{example_name}] =============================");
}

// ---------------------------------------------------------------------------
// Workflow implementations
// ---------------------------------------------------------------------------

/// Healthy workflow 1: repeatedly read, filter, sort.
fn workflow_iterative_sort_filter(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    let df = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;

    let mut current = df;
    for i in 0..HEALTHY_ITERATIONS {
        ctx.log("ITER", format!("sort-filter iteration {}/{HEALTHY_ITERATIONS}", i + 1));

        let clean = current.dropna(DropNaOptions::builder().subset(&["risk_score", "influence"]).build())?;

        let threshold = 30.0_f64 + (i as f64 * 2.0);
        let mask = clean.series("risk_score")?.gt_scalar(threshold)?;
        let filtered = clean.filter(&mask)?;

        let sorted = filtered.sort_values(
            ["risk_score"],
            SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]),
        )?;

        current = sorted.head(TOP_N)?;
        ctx.log("ITER", format!("iteration {} done, rows={}", i + 1, current.num_rows()));
    }

    let info = current.info_with(InfoDetail::ALL)?;
    let device_bytes = info.memory_accounting.as_ref().and_then(|m| m.device_allocated_bytes).unwrap_or(0);

    Ok(format!(
        "final_rows={} cols={} device_bytes={device_bytes} iterations={HEALTHY_ITERATIONS}",
        current.num_rows(),
        current.num_columns()
    ))
}

/// Healthy workflow 2: repeatedly groupby-aggregate.
fn workflow_iterative_groupby(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    let df = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;

    let mut last_groups = 0usize;
    for i in 0..HEALTHY_ITERATIONS {
        ctx.log("ITER", format!("groupby iteration {}/{HEALTHY_ITERATIONS}", i + 1));

        let clean = df.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;

        let grouped = clean.groupby(["entityType"]).agg(&[
            AggSpec::new("risk_score", GroupByAgg::Mean).with_alias("risk_mean"),
            AggSpec::new("risk_score", GroupByAgg::Max).with_alias("risk_max"),
            AggSpec::new("influence", GroupByAgg::Sum).with_alias("influence_sum"),
            AggSpec::new("entityName", GroupByAgg::Count { include_nulls: false }).with_alias("entity_count"),
        ])?;

        let sorted = grouped.sort_values(["entityType"], SortOptions::new())?;
        last_groups = sorted.num_rows();

        ctx.log("ITER", format!("iteration {} done, groups={last_groups}", i + 1));
    }

    Ok(format!("groups={last_groups} iterations={HEALTHY_ITERATIONS}"))
}

/// Healthy workflow 3: Series-level arithmetic in a loop.
fn workflow_series_arithmetic(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    let size = 200_000usize;
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();

    let base = runtime.series("values", &data)?;
    ctx.log("INIT", format!("created base Series with {} elements", base.len()));

    let mut accumulated_rows = base.len();
    for i in 0..HEALTHY_ITERATIONS {
        ctx.log("ITER", format!("arithmetic iteration {}/{HEALTHY_ITERATIONS}", i + 1));

        // Scalar operations: multiply, add
        let scaled = base.mul_scalar(2.0_f64 + i as f64)?;
        let shifted = scaled.add_scalar(100.0_f64)?;

        // Comparison: filter to values > threshold
        let threshold = 50.0_f64 + (i as f64 * 10.0);
        let mask = shifted.gt_scalar(threshold)?;

        // Use the mask to count how many passed (materialize to host)
        // We don't filter the series itself since that would shrink it,
        // instead just check the mask's non-null true count.
        let mask_len = mask.len();

        // Create a new series and concat
        let extra_data: Vec<f64> = (0..10_000).map(|j| (j as f64) * 0.01 + i as f64).collect();
        let extra = runtime.series("extra", &extra_data)?;
        let combined = Series::concat(&[&base, &extra])?;
        accumulated_rows += extra.len();

        ctx.log(
            "ITER",
            format!(
                "iteration {} done, mask_len={mask_len} combined_len={} accumulated={}",
                i + 1,
                combined.len(),
                accumulated_rows,
            ),
        );
    }

    Ok(format!("accumulated_rows={accumulated_rows} iterations={HEALTHY_ITERATIONS}"))
}

/// OOM trigger: repeatedly concat a DataFrame with itself until OOM.
fn workflow_oom_concat_growth(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    let df = read_node_subset(&runtime, &ctx, OOM_BASE_NUM_ROWS)?;

    // Widen with duplicated numeric columns for faster VRAM consumption.
    let numeric = inflate_numeric_frame(&df)?;
    ctx.log("OOM_TRIGGER", format!("inflated to {} cols, starting concat-doubling", numeric.num_columns()));

    let mut current = numeric;
    let operation_ctx = runtime_context(&runtime);
    for iteration in 0..MAX_CONCAT_DOUBLINGS {
        let before_rows = current.num_rows();
        ctx.log("CONCAT", format!("iteration={} rows_before={before_rows}", iteration + 1));

        let next = current.in_context(&operation_ctx).concat(&current).map_err(|err| {
            err.with_context("concat_iteration", (iteration + 1).to_string())
                .with_context("rows_before_concat", before_rows.to_string())
        })?;
        current = next;
    }

    // If we get here, OOM was never triggered.
    Ok(format!(
        "WARNING: concat growth completed {MAX_CONCAT_DOUBLINGS} iterations without OOM, final_rows={}",
        current.num_rows()
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn runtime_context(runtime: &Runtime) -> ExecutionContext<'_> {
    ExecutionContext::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}

fn read_node_subset(runtime: &Runtime, ctx: &WorkflowContext, num_rows: usize) -> cudf::Result<DataFrame> {
    ctx.log("READ", format!("path={NODE_PARQUET_PATH} num_rows={num_rows}"));
    let options = ParquetReadOptions::builder()
        .filepath(Path::new(NODE_PARQUET_PATH))
        .columns(&NODE_COLUMNS)
        .num_rows(num_rows)
        .with_runtime(runtime)
        .build();
    let df = DataFrame::read_parquet_with_options(options)?;
    ctx.log("READ", format!("loaded rows={} cols={}", df.num_rows(), df.num_columns()));
    Ok(df)
}

fn build_concurrent_alloc_specs() -> [WorkflowSpec; NUM_WORKFLOWS] {
    [
        WorkflowSpec { label: "heavy-alloc-read-concat-A", expect_oom: false, run: workflow_heavy_alloc_read_concat },
        WorkflowSpec { label: "heavy-alloc-read-concat-B", expect_oom: false, run: workflow_heavy_alloc_read_concat },
        WorkflowSpec { label: "heavy-alloc-read-concat-C", expect_oom: false, run: workflow_heavy_alloc_read_concat },
        WorkflowSpec { label: "oom-concat-growth", expect_oom: true, run: workflow_oom_concat_growth_with_barrier },
    ]
}

/// Healthy workflow for concurrent-alloc test: repeatedly re-read parquet and
/// concat the result onto an accumulator.  Each iteration allocates new GPU
/// memory via parquet decode + concat, keeping allocation pressure high.
fn workflow_heavy_alloc_read_concat(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    // Initial read
    let mut accumulator = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;
    let base_rows = accumulator.num_rows();
    let operation_ctx = runtime_context(&runtime);

    // Signal we're ready, wait for all threads
    ctx.wait_barrier("initial read done, waiting for all threads before heavy-alloc loop");

    for i in 0..HEAVY_ALLOC_ITERATIONS {
        ctx.log(
            "ITER",
            format!("heavy-alloc iteration {}/{HEAVY_ALLOC_ITERATIONS}, rows={}", i + 1, accumulator.num_rows()),
        );

        // Re-read a small chunk (forces new GPU allocation for parquet decode)
        let chunk = read_node_subset(&runtime, &ctx, 10_000)?;

        // Sort the chunk (forces temporary GPU allocations)
        let sorted_chunk = chunk.sort_values(
            ["risk_score"],
            SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]),
        )?;

        // Concat onto accumulator (forces new allocation for the combined table)
        accumulator = accumulator.in_context(&operation_ctx).concat(&sorted_chunk)?;

        // Groupby on the accumulator (forces more temporary allocations)
        let _grouped = accumulator.groupby(["entityType"]).agg(&[
            AggSpec::new("risk_score", GroupByAgg::Mean).with_alias("risk_mean"),
            AggSpec::new("influence", GroupByAgg::Sum).with_alias("influence_sum"),
        ])?;

        // Trim back to prevent the accumulator from growing unboundedly
        // (we want sustained allocation pressure, not self-inflicted OOM)
        if accumulator.num_rows() > base_rows * 2 {
            accumulator = accumulator.head(base_rows)?;
        }
    }

    Ok(format!("final_rows={} iterations={HEAVY_ALLOC_ITERATIONS}", accumulator.num_rows()))
}

/// OOM trigger variant that waits at the barrier before starting growth.
fn workflow_oom_concat_growth_with_barrier(ctx: WorkflowContext, runtime: Runtime) -> cudf::Result<String> {
    let df = read_node_subset(&runtime, &ctx, OOM_BASE_NUM_ROWS)?;
    let numeric = inflate_numeric_frame(&df)?;

    // Wait for all threads to be ready before starting the growth phase
    ctx.wait_barrier("inflated frame ready, waiting for all threads before concat-doubling");
    ctx.log("OOM_TRIGGER", format!("inflated to {} cols, starting concat-doubling", numeric.num_columns()));

    let mut current = numeric;
    let operation_ctx = runtime_context(&runtime);
    for iteration in 0..MAX_CONCAT_DOUBLINGS {
        let before_rows = current.num_rows();
        ctx.log("CONCAT", format!("iteration={} rows_before={before_rows}", iteration + 1));

        let next = current.in_context(&operation_ctx).concat(&current).map_err(|err| {
            err.with_context("concat_iteration", (iteration + 1).to_string())
                .with_context("rows_before_concat", before_rows.to_string())
        })?;
        current = next;
    }

    Ok(format!(
        "WARNING: concat growth completed {MAX_CONCAT_DOUBLINGS} iterations without OOM, final_rows={}",
        current.num_rows()
    ))
}

fn inflate_numeric_frame(df: &DataFrame) -> cudf::Result<DataFrame> {
    let risk = df.series("risk_score")?;
    let influence = df.series("influence")?;
    let importance = df.series("importance")?;

    df.select(["risk_score", "influence", "importance"])?
        .with_column("risk_score_dup_1", risk.clone())?
        .with_column("risk_score_dup_2", risk.clone())?
        .with_column("risk_score_dup_3", risk.clone())?
        .with_column("influence_dup_1", influence.clone())?
        .with_column("influence_dup_2", influence.clone())?
        .with_column("influence_dup_3", influence.clone())?
        .with_column("importance_dup_1", importance.clone())?
        .with_column("importance_dup_2", importance.clone())?
        .with_column("importance_dup_3", importance.clone())
}
