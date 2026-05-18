#![allow(dead_code)]

use std::{
    ffi::CStr,
    path::Path,
    sync::{Arc, Barrier},
    time::Instant,
};

use cudf::{
    AggSpec, DataFrame, DeviceId, DropNaOptions, Error, ErrorKind, ExecutionContext, GroupByAgg, InfoDetail,
    MergeOptions, ParquetReadOptions, Runtime, SortOptions,
};

pub const REQUIRED_VISIBLE_DEVICES: usize = 4;

const NODE_PARQUET_PATH: &str = "/home/ubuntu/data/node.parquet";
const NODE_COLUMNS: [&str; 5] = ["entityName", "entityType", "risk_score", "influence", "importance"];
const HEALTHY_NUM_ROWS: usize = 120_000;
const FAILING_NUM_ROWS: usize = 64_000;
const WORKFLOW_TOP_N: usize = 2_048;
const SLICE_ROWS: usize = 4_096;
const INIT_OOM_POOL_BYTES: usize = 1usize << 40;
const PRESSURE_CHUNK_BYTES: usize = 128 * 1024 * 1024;
const PRESSURE_RELEASE_CHUNKS: usize = 1;
const MAX_PRESSURE_CHUNKS: usize = 512;
const MAX_CONCAT_DOUBLINGS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    Completed,
    ExpectedOom,
    UnexpectedError,
    UnexpectedSuccess,
}

impl WorkflowStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "Completed",
            Self::ExpectedOom => "ExpectedOom",
            Self::UnexpectedError => "UnexpectedError",
            Self::UnexpectedSuccess => "UnexpectedSuccess",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowReport {
    pub label: &'static str,
    pub device_ordinal: usize,
    pub status: WorkflowStatus,
    pub elapsed_ms: u128,
    pub summary: String,
}

#[derive(Clone)]
struct WorkflowContext {
    example_name: &'static str,
    label: &'static str,
    device: DeviceId,
    barrier: Arc<Barrier>,
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

    fn wait_ready(&self, detail: &str) {
        self.log("READY", detail);
        self.barrier.wait();
    }
}

#[derive(Clone, Copy)]
struct WorkflowSpec {
    label: &'static str,
    device: DeviceId,
    expect_oom: bool,
    run: fn(WorkflowContext) -> cudf::Result<String>,
}

pub fn validate_exact_visible_devices(visible_devices: usize) -> cudf::Result<()> {
    if visible_devices == REQUIRED_VISIBLE_DEVICES {
        return Ok(());
    }

    Err(Error::new(
        ErrorKind::InvalidOperation,
        format!(
            "multi-GPU OOM examples require exactly {REQUIRED_VISIBLE_DEVICES} visible CUDA devices, found {visible_devices}"
        ),
    )
    .with_operation("examples::multi_gpu::validate_exact_visible_devices")
    .with_context("required_visible_devices", REQUIRED_VISIBLE_DEVICES.to_string())
    .with_context("visible_devices", visible_devices.to_string()))
}

pub fn validate_final_reports(reports: &[WorkflowReport]) -> cudf::Result<()> {
    if reports.len() != REQUIRED_VISIBLE_DEVICES {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "multi-GPU OOM example did not produce exactly four workflow reports",
        )
        .with_operation("examples::multi_gpu::validate_final_reports")
        .with_context("expected_workflow_reports", REQUIRED_VISIBLE_DEVICES.to_string())
        .with_context("actual_workflow_reports", reports.len().to_string()));
    }

    let completed = reports.iter().filter(|report| report.status == WorkflowStatus::Completed).count();
    let expected_oom = reports.iter().filter(|report| report.status == WorkflowStatus::ExpectedOom).count();
    let unexpected = reports
        .iter()
        .filter(|report| !matches!(report.status, WorkflowStatus::Completed | WorkflowStatus::ExpectedOom))
        .count();

    if completed == REQUIRED_VISIBLE_DEVICES - 1 && expected_oom == 1 && unexpected == 0 {
        return Ok(());
    }

    Err(Error::new(
        ErrorKind::InvalidOperation,
        "multi-GPU OOM example did not preserve the expected 3-success / 1-OOM isolation result",
    )
    .with_operation("examples::multi_gpu::validate_final_reports")
    .with_context("completed_workflows", completed.to_string())
    .with_context("expected_oom_workflows", expected_oom.to_string())
    .with_context("unexpected_failures", unexpected.to_string()))
}

pub async fn run_workflow_oom_example() -> cudf::Result<()> {
    let [gpu0, gpu1, gpu2, gpu3] = exact_four_devices()?;
    run_example(
        "multi_gpu_parallel_ops_oom_during_workflow",
        [
            WorkflowSpec {
                label: "gpu0-read-filter-sort",
                device: gpu0,
                expect_oom: false,
                run: gpu0_read_filter_sort_workflow,
            },
            WorkflowSpec { label: "gpu1-groupby-agg", device: gpu1, expect_oom: false, run: gpu1_groupby_workflow },
            WorkflowSpec {
                label: "gpu2-merge-concat",
                device: gpu2,
                expect_oom: false,
                run: gpu2_merge_concat_workflow,
            },
            WorkflowSpec {
                label: "gpu3-oom-during-workflow",
                device: gpu3,
                expect_oom: true,
                run: gpu3_workflow_oom_workflow,
            },
        ],
    )
    .await
}

pub async fn run_runtime_init_oom_example() -> cudf::Result<()> {
    let [gpu0, gpu1, gpu2, gpu3] = exact_four_devices()?;
    run_example(
        "multi_gpu_parallel_ops_oom_on_runtime_init",
        [
            WorkflowSpec {
                label: "gpu0-read-filter-sort",
                device: gpu0,
                expect_oom: false,
                run: gpu0_read_filter_sort_workflow,
            },
            WorkflowSpec { label: "gpu1-groupby-agg", device: gpu1, expect_oom: false, run: gpu1_groupby_workflow },
            WorkflowSpec {
                label: "gpu2-merge-concat",
                device: gpu2,
                expect_oom: false,
                run: gpu2_merge_concat_workflow,
            },
            WorkflowSpec {
                label: "gpu3-oom-on-runtime-init",
                device: gpu3,
                expect_oom: true,
                run: gpu3_runtime_init_oom_workflow,
            },
        ],
    )
    .await
}

pub async fn run_vram_pressure_oom_example() -> cudf::Result<()> {
    let [gpu0, gpu1, gpu2, gpu3] = exact_four_devices()?;
    run_example(
        "multi_gpu_parallel_ops_oom_after_vram_pressure",
        [
            WorkflowSpec {
                label: "gpu0-read-filter-sort",
                device: gpu0,
                expect_oom: false,
                run: gpu0_read_filter_sort_workflow,
            },
            WorkflowSpec { label: "gpu1-groupby-agg", device: gpu1, expect_oom: false, run: gpu1_groupby_workflow },
            WorkflowSpec {
                label: "gpu2-merge-concat",
                device: gpu2,
                expect_oom: false,
                run: gpu2_merge_concat_workflow,
            },
            WorkflowSpec {
                label: "gpu3-oom-after-pressure",
                device: gpu3,
                expect_oom: true,
                run: gpu3_pressure_oom_workflow,
            },
        ],
    )
    .await
}

async fn run_example(example_name: &'static str, specs: [WorkflowSpec; REQUIRED_VISIBLE_DEVICES]) -> cudf::Result<()> {
    let barrier = Arc::new(Barrier::new(specs.len()));
    let mut handles = Vec::with_capacity(specs.len());

    for spec in specs {
        let ctx =
            WorkflowContext { example_name, label: spec.label, device: spec.device, barrier: Arc::clone(&barrier) };
        handles.push(tokio::task::spawn_blocking(move || execute_workflow(spec, ctx)));
    }

    let mut reports = Vec::with_capacity(handles.len());
    for handle in handles {
        reports.push(handle.await.map_err(|err| {
            Error::new(ErrorKind::InvalidOperation, "multi-GPU workflow task panicked or was cancelled")
                .with_operation("examples::multi_gpu::run_example")
                .set_source(err)
        })?);
    }

    reports.sort_by_key(|report| report.device_ordinal);
    print_summary(example_name, &reports);
    validate_final_reports(&reports)?;
    println!("[{example_name}] VERDICT isolation preserved: 1 workflow hit OutOfMemory while the other 3 completed.");
    Ok(())
}

fn execute_workflow(spec: WorkflowSpec, ctx: WorkflowContext) -> WorkflowReport {
    ctx.log("START", if spec.expect_oom { "launching expected-OOM workflow" } else { "launching healthy workflow" });

    let started = Instant::now();
    let result = (|| {
        let _device_guard = ExampleDeviceGuard::new(spec.device)?;
        (spec.run)(ctx.clone())
    })();

    match result {
        Ok(summary) if spec.expect_oom => {
            ctx.log("ERR", "workflow unexpectedly completed without OutOfMemory");
            WorkflowReport {
                label: spec.label,
                device_ordinal: spec.device.ordinal(),
                status: WorkflowStatus::UnexpectedSuccess,
                elapsed_ms: started.elapsed().as_millis(),
                summary,
            }
        },
        Ok(summary) => {
            ctx.log("DONE", &summary);
            WorkflowReport {
                label: spec.label,
                device_ordinal: spec.device.ordinal(),
                status: WorkflowStatus::Completed,
                elapsed_ms: started.elapsed().as_millis(),
                summary,
            }
        },
        Err(err) if spec.expect_oom && err.kind() == ErrorKind::OutOfMemory => {
            ctx.log("ERR", format!("expected OutOfMemory: {err}"));
            println!(
                "[{}] device={} workflow={} FULL_ERROR\n{err:?}",
                ctx.example_name,
                spec.device.ordinal(),
                spec.label
            );
            WorkflowReport {
                label: spec.label,
                device_ordinal: spec.device.ordinal(),
                status: WorkflowStatus::ExpectedOom,
                elapsed_ms: started.elapsed().as_millis(),
                summary: format!("OutOfMemory at {}: {}", err.operation(), err.message()),
            }
        },
        Err(err) => {
            ctx.log("ERR", format!("{err}"));
            println!(
                "[{}] device={} workflow={} FULL_ERROR\n{err:?}",
                ctx.example_name,
                spec.device.ordinal(),
                spec.label
            );
            WorkflowReport {
                label: spec.label,
                device_ordinal: spec.device.ordinal(),
                status: WorkflowStatus::UnexpectedError,
                elapsed_ms: started.elapsed().as_millis(),
                summary: format!("{} at {}: {}", err.kind(), err.operation(), err.message()),
            }
        },
    }
}

fn exact_four_devices() -> cudf::Result<[DeviceId; REQUIRED_VISIBLE_DEVICES]> {
    let count = DeviceId::count()?;
    validate_exact_visible_devices(count)?;
    Ok([DeviceId::new(0)?, DeviceId::new(1)?, DeviceId::new(2)?, DeviceId::new(3)?])
}

fn build_runtime(ctx: &WorkflowContext) -> cudf::Result<Runtime> {
    Runtime::builder().with_device(ctx.device).build()
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

fn gpu0_read_filter_sort_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    let runtime = build_runtime(&ctx)?;
    ctx.wait_ready("runtime ready; waiting for all four workflows");

    let df = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;
    ctx.log("FILTER", "dropna subset=[risk_score,influence]");
    let dropna = df.dropna(DropNaOptions::builder().subset(&["risk_score", "influence"]).build())?;
    ctx.log("FILTER", format!("post-dropna rows={}", dropna.num_rows()));

    ctx.log("FILTER", "filtering risk_score > 50.0");
    let high_risk_mask = dropna.series("risk_score")?.gt_scalar(50.0_f64)?;
    let high_risk = dropna.filter(&high_risk_mask)?;
    ctx.log("FILTER", format!("post-filter rows={}", high_risk.num_rows()));

    ctx.log("SORT", "sorting by risk_score descending");
    let sorted = high_risk
        .sort_values(["risk_score"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]))?;
    let top = sorted.head(WORKFLOW_TOP_N)?;
    let info = top.info_with(InfoDetail::ALL)?;
    let device_allocated =
        info.memory_accounting.as_ref().and_then(|memory| memory.device_allocated_bytes).unwrap_or(0);

    Ok(format!("rows={} cols={} device_allocated_bytes={device_allocated}", top.num_rows(), top.num_columns()))
}

fn gpu1_groupby_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    let runtime = build_runtime(&ctx)?;
    ctx.wait_ready("runtime ready; waiting for all four workflows");

    let df = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;
    ctx.log("FILTER", "dropna subset=[risk_score]");
    let clean = df.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;

    ctx.log("GROUPBY", "aggregating entityType");
    let grouped = clean.groupby(["entityType"]).agg(&[
        AggSpec::new("risk_score", GroupByAgg::Mean).with_alias("risk_mean"),
        AggSpec::new("risk_score", GroupByAgg::Max).with_alias("risk_max"),
        AggSpec::new("influence", GroupByAgg::Sum).with_alias("influence_sum"),
        AggSpec::new("entityName", GroupByAgg::Count { include_nulls: false }).with_alias("entity_count"),
    ])?;
    let sorted = grouped.sort_values(["entityType"], SortOptions::new())?;

    Ok(format!("groups={} columns={}", sorted.num_rows(), sorted.num_columns()))
}

fn gpu2_merge_concat_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    let runtime = build_runtime(&ctx)?;
    let operation_ctx = runtime_context(&runtime);
    ctx.wait_ready("runtime ready; waiting for all four workflows");

    let df = read_node_subset(&runtime, &ctx, HEALTHY_NUM_ROWS)?;
    ctx.log("GROUPBY", "building lookup frame via entityType count(entityName)");
    let lookup = df.groupby(["entityType"]).count(["entityName"])?;

    ctx.log("MERGE", "joining lookup back into working frame");
    let merged = df.in_context(&operation_ctx).merge(&lookup, MergeOptions::new().how_inner().on(["entityType"]))?;
    let numeric = merged.select(["risk_score", "influence", "importance", "entityName_count"])?;
    let head = numeric.head(SLICE_ROWS)?;
    let tail = numeric.tail(SLICE_ROWS)?;

    ctx.log("CONCAT", "concatenating head/tail slices");
    let stitched = head.in_context(&operation_ctx).concat(&tail)?;
    Ok(format!("merged_rows={} stitched_rows={}", merged.num_rows(), stitched.num_rows()))
}

fn gpu3_workflow_oom_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    let runtime = build_runtime(&ctx)?;
    ctx.wait_ready("runtime ready; waiting for all four workflows");

    let df = read_node_subset(&runtime, &ctx, FAILING_NUM_ROWS)?;
    ctx.log("FILTER", "dropna subset=[risk_score]");
    let clean = df.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;
    ctx.log("SORT", "sorting by risk_score descending before growth");
    let sorted = clean
        .sort_values(["risk_score"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]))?;
    let numeric = inflate_numeric_frame_for_oom(&sorted)?;

    ctx.log("OOM_TRIGGER", "doubling the dataframe with in_context(...).concat until OutOfMemory");
    grow_until_oom(numeric, &runtime, &ctx)
}

fn gpu3_runtime_init_oom_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    ctx.wait_ready("waiting at barrier before oversized allocator initialization");
    ctx.log(
        "OOM_TRIGGER",
        format!(
            "requesting pool allocator with initial_pool_size={INIT_OOM_POOL_BYTES} max_pool_size={INIT_OOM_POOL_BYTES}"
        ),
    );

    let _runtime = Runtime::builder()
        .with_device(ctx.device)
        .with_pool_memory_resource(INIT_OOM_POOL_BYTES, Some(INIT_OOM_POOL_BYTES))
        .build()
        .map_err(|err| {
            err.with_context("requested_pool_bytes", INIT_OOM_POOL_BYTES.to_string())
                .with_context("device_ordinal", ctx.device.ordinal().to_string())
        })?;

    Ok(format!("unexpectedly initialized runtime with pool_bytes={INIT_OOM_POOL_BYTES}"))
}

fn gpu3_pressure_oom_workflow(ctx: WorkflowContext) -> cudf::Result<String> {
    let runtime = build_runtime(&ctx)?;
    ctx.wait_ready("runtime ready; waiting for all four workflows before applying VRAM pressure");

    ctx.log("PRESSURE", format!("allocating device buffers in {PRESSURE_CHUNK_BYTES}-byte chunks"));
    let mut pressure = VramPressure::apply(ctx.device, &ctx)?;
    pressure.release_tail(PRESSURE_RELEASE_CHUNKS, &ctx);

    let df = read_node_subset(&runtime, &ctx, FAILING_NUM_ROWS)?;
    ctx.log("FILTER", "dropna subset=[risk_score]");
    let clean = df.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;
    ctx.log("SORT", "sorting by risk_score descending under pressure");
    let sorted = clean
        .sort_values(["risk_score"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]))?;
    let numeric = inflate_numeric_frame_for_oom(&sorted)?;

    ctx.log("OOM_TRIGGER", "running repeated in_context(...).concat while pressure buffers remain live");
    let _keep_pressure_alive = &pressure;
    grow_until_oom(numeric, &runtime, &ctx)
}

fn inflate_numeric_frame_for_oom(df: &DataFrame) -> cudf::Result<DataFrame> {
    let risk = df.series("risk_score")?;
    let influence = df.series("influence")?;
    let importance = df.series("importance")?;

    let widened = df
        .select(["risk_score", "influence", "importance"])?
        .with_column("risk_score_dup_1", risk.clone())?
        .with_column("risk_score_dup_2", risk.clone())?
        .with_column("risk_score_dup_3", risk.clone())?
        .with_column("influence_dup_1", influence.clone())?
        .with_column("influence_dup_2", influence.clone())?
        .with_column("influence_dup_3", influence.clone())?
        .with_column("importance_dup_1", importance.clone())?
        .with_column("importance_dup_2", importance.clone())?
        .with_column("importance_dup_3", importance.clone())?;

    Ok(widened)
}

fn grow_until_oom(mut current: DataFrame, runtime: &Runtime, ctx: &WorkflowContext) -> cudf::Result<String> {
    let operation_ctx = runtime_context(runtime);
    for iteration in 0..MAX_CONCAT_DOUBLINGS {
        let before_rows = current.num_rows();
        ctx.log("CONCAT", format!("iteration={} rows_before={before_rows}", iteration + 1));
        let next = current.in_context(&operation_ctx).concat(&current).map_err(|err| {
            err.with_context("concat_iteration", (iteration + 1).to_string())
                .with_context("rows_before_concat", before_rows.to_string())
        })?;
        current = next;
    }

    Err(Error::new(
        ErrorKind::InvalidOperation,
        "concat growth loop reached the maximum number of iterations without hitting OutOfMemory",
    )
    .with_operation("examples::multi_gpu::grow_until_oom")
    .with_context("max_concat_doublings", MAX_CONCAT_DOUBLINGS.to_string())
    .with_context("final_rows", current.num_rows().to_string()))
}

fn runtime_context(runtime: &Runtime) -> ExecutionContext<'_> {
    ExecutionContext::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}

fn print_summary(example_name: &str, reports: &[WorkflowReport]) {
    println!("[{example_name}] SUMMARY");
    for report in reports {
        println!(
            "[{example_name}] device={} workflow={} status={} elapsed_ms={} result={}",
            report.device_ordinal,
            report.label,
            report.status.as_str(),
            report.elapsed_ms,
            report.summary
        );
    }
}

struct ExampleDeviceGuard {
    previous_device: i32,
}

impl ExampleDeviceGuard {
    fn new(device: DeviceId) -> cudf::Result<Self> {
        let mut previous_device = 0;
        check_status(
            unsafe { cudf_sys::cudf_cuda_get_device(&mut previous_device) },
            "examples::multi_gpu::device_guard::get_device",
        )?;

        if previous_device != device.ordinal() as i32 {
            check_status(
                unsafe { cudf_sys::cudf_cuda_set_device(device.ordinal() as i32) },
                "examples::multi_gpu::device_guard::set_device",
            )?;
        }

        Ok(Self { previous_device })
    }
}

impl Drop for ExampleDeviceGuard {
    fn drop(&mut self) {
        let _ = unsafe { cudf_sys::cudf_cuda_set_device(self.previous_device) };
    }
}

struct DeviceBuffer {
    raw: cudf_sys::cudf_device_buffer_handle,
    size_bytes: usize,
}

impl DeviceBuffer {
    fn new(size_bytes: usize) -> cudf::Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(
            unsafe { cudf_sys::cudf_device_buffer_create(size_bytes, &mut raw) },
            "examples::multi_gpu::device_buffer::create",
        )?;
        Ok(Self { raw, size_bytes })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_device_buffer_destroy(self.raw) };
    }
}

struct VramPressure {
    buffers: Vec<DeviceBuffer>,
    retained_bytes: usize,
}

impl VramPressure {
    fn apply(device: DeviceId, ctx: &WorkflowContext) -> cudf::Result<Self> {
        let _guard = ExampleDeviceGuard::new(device)?;
        let mut buffers = Vec::new();
        let mut retained_bytes = 0usize;

        loop {
            if buffers.len() >= MAX_PRESSURE_CHUNKS {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "VRAM pressure loop exceeded the maximum number of chunk allocations without hitting OutOfMemory",
                )
                .with_operation("examples::multi_gpu::VramPressure::apply")
                .with_context("max_pressure_chunks", MAX_PRESSURE_CHUNKS.to_string())
                .with_context("retained_bytes", retained_bytes.to_string()));
            }

            match DeviceBuffer::new(PRESSURE_CHUNK_BYTES) {
                Ok(buffer) => {
                    retained_bytes += buffer.size_bytes;
                    buffers.push(buffer);
                    ctx.log("PRESSURE", format!("allocated_chunks={} retained_bytes={retained_bytes}", buffers.len()));
                },
                Err(err) if err.kind() == ErrorKind::OutOfMemory => {
                    ctx.log(
                        "PRESSURE",
                        format!(
                            "stopped after allocated_chunks={} retained_bytes={retained_bytes} because of OutOfMemory",
                            buffers.len()
                        ),
                    );
                    break;
                },
                Err(err) => return Err(err),
            }
        }

        if buffers.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "VRAM pressure workflow could not allocate any device buffers before failure",
            )
            .with_operation("examples::multi_gpu::VramPressure::apply")
            .with_context("chunk_bytes", PRESSURE_CHUNK_BYTES.to_string()));
        }

        Ok(Self { buffers, retained_bytes })
    }

    fn release_tail(&mut self, chunks: usize, ctx: &WorkflowContext) {
        let release = chunks.min(self.buffers.len());
        for _ in 0..release {
            if let Some(buffer) = self.buffers.pop() {
                self.retained_bytes = self.retained_bytes.saturating_sub(buffer.size_bytes);
            }
        }
        ctx.log(
            "PRESSURE",
            format!(
                "released_chunks={} remaining_chunks={} retained_bytes={}",
                release,
                self.buffers.len(),
                self.retained_bytes
            ),
        );
    }
}

fn check_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> cudf::Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let kind = match status {
        cudf_sys::CUDF_ERROR_INVALID_OPERATION => ErrorKind::InvalidOperation,
        cudf_sys::CUDF_ERROR_CUDA => ErrorKind::CudaError,
        cudf_sys::CUDF_ERROR_FATAL_CUDA => ErrorKind::FatalCudaError,
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => ErrorKind::OutOfMemory,
        cudf_sys::CUDF_ERROR_IO => ErrorKind::IoError,
        cudf_sys::CUDF_ERROR_TYPE => ErrorKind::TypeError,
        cudf_sys::CUDF_ERROR_NULL => ErrorKind::NullError,
        cudf_sys::CUDF_ERROR_INDEX => ErrorKind::IndexError,
        _ => ErrorKind::Unknown,
    };

    let message = unsafe {
        let ptr = cudf_sys::cudf_last_error_message();
        if ptr.is_null() {
            String::from("(no error message)")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let err = match kind {
        ErrorKind::OutOfMemory | ErrorKind::CudaError => Error::new(kind, message).set_temporary(),
        ErrorKind::FatalCudaError => Error::new(kind, message).set_persistent(),
        _ => Error::new(kind, message).set_permanent(),
    };

    Err(err.with_operation(operation))
}
