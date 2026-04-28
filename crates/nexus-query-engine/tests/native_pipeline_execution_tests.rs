//! GPU execution ownership for compiled native pipelines.
//!
//! This target is intentionally separate from `native_contract_tests`: its
//! cases create a process-owned GPU backend, while the remaining contract
//! target is CPU-only.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    fs::{self, File},
    ops::Deref,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use nexus_query_engine::{
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation},
    admission::{
        CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
        strict_bounded_pipeline_admission,
    },
    capability::{ContractQuality, ContractStatus, ReservationRequirementScope},
    dynamic_filter::DynamicFilterProducer,
    error::Result,
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, execute_query_plan_with_gpu_inputs,
        materialize_gpu_result_to_host,
        output::{QueryExecutionResult, QueryOutput},
    },
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, GroupingSetsSpec,
        LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::{MemoryReservationEventAction, QueryMetrics},
    mode::NativePipelineExecutionMode,
    optimizer::{JoinBuildSideSelectionOptions, NativeOptimizerOptions, optimize_query_plan},
    pipeline::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
        compile_pipeline,
    },
    plan::{
        JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode, ProjectionOutput, QueryPlan,
        QueryPlanBuilder, WindowFunction, WindowSpec,
    },
    runtime::{
        DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore,
        GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendTenantDraft, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryHandle, QueryServiceConfig, ReservationKind, TenantId,
        TenantQueryContext, resident::GpuResidentInputMap,
    },
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, PartitionColumn, SourceNode},
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

const PIPELINE_TENANT: &str = "compiled-pipeline-tests";
const PIPELINE_MINIMUM_GRANT_BYTES: u64 = 64 * 1024 * 1024;

fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

#[path = "native_contract_tests/pipeline_compiled_execution/fixtures_parquet.rs"]
mod fixtures_parquet;
#[path = "native_contract_tests/pipeline_compiled_execution/fixtures_plans.rs"]
// Shared with the CPU contract target. Only a subset is exercised here, so
// unused fixture constructors are expected for this binary.
#[allow(dead_code)]
mod fixtures_plans;
#[path = "native_contract_tests/pipeline_compiled_execution/gpu_execution.rs"]
mod gpu_execution;
#[path = "native_contract_tests/bounded_join_admission/plans/remaining_topologies.rs"]
// Shared with the CPU contract target; this binary only calls three constructors.
#[allow(dead_code)]
mod remaining_topologies;

use fixtures_parquet::*;
use fixtures_plans::*;
use remaining_topologies::{
    aggregate_over_join_subplan_topk_plan, resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
    resident_grouped_aggregate_subquery_filter_plan,
};

struct SuiteBackend {
    mode: NativePipelineExecutionMode,
    core: GpuBackendCore,
}

static PIPELINE_BACKEND: Mutex<Option<SuiteBackend>> = Mutex::new(None);

fn build_suite_backend(mode: NativePipelineExecutionMode) -> Result<GpuBackendCore> {
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let policy = QueryServiceConfig {
        native_pipeline_execution_mode: mode,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
        ..QueryServiceConfig::default()
    };
    let controller = GpuBackendControllerDraft::from_legacy_policy(
        policy,
        vec![GpuBackendTenantDraft::new(
            PIPELINE_TENANT,
            GpuBackendMemoryShare::Bytes(PIPELINE_MINIMUM_GRANT_BYTES),
            GpuBackendMemoryShare::PolicyDefault,
        )],
    );
    // One admitted query owns the suite, while two service streams allow the
    // adaptive-history regression to compare two live contexts from that same
    // admitted handle.
    let device =
        GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive).with_installation_bounds(2, 1, 1);
    GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, deployment)).build()
}

fn ensure_suite_backend(backend: &mut Option<SuiteBackend>, mode: NativePipelineExecutionMode) -> Result<()> {
    if backend.as_ref().is_some_and(|backend| backend.mode == mode) {
        return Ok(());
    }
    if let Some(previous) = backend.take() {
        previous.core.close_and_wait(Duration::from_secs(30))?;
    }
    *backend = Some(SuiteBackend { mode, core: build_suite_backend(mode)? });
    Ok(())
}

// The shared execution cases historically spell their suite request as
// `cudf::Runtime::new()`. In this integration target that name intentionally
// resolves to an inert request token. It serializes backend ownership for the
// whole test and can only create contexts through the public backend/session
// route below; it never constructs or exposes a raw cuDF runtime.
mod cudf {
    use super::*;

    pub(super) struct Runtime {
        backend: RefCell<MutexGuard<'static, Option<SuiteBackend>>>,
        mode: Cell<Option<NativePipelineExecutionMode>>,
        handle: RefCell<Option<Arc<QueryHandle>>>,
    }

    impl Runtime {
        pub(super) fn new() -> std::result::Result<Self, String> {
            let backend = PIPELINE_BACKEND.lock().map_err(|_| "pipeline backend lock poisoned".to_string())?;
            Ok(Self { backend: RefCell::new(backend), mode: Cell::new(None), handle: RefCell::new(None) })
        }

        pub(super) fn acquire_context(
            self: &Arc<Self>,
            mode: NativePipelineExecutionMode,
            external_history: Option<Arc<AdaptiveMemoryHistory>>,
        ) -> Result<PipelineTestContext> {
            if let Some(installed_mode) = self.mode.get()
                && installed_mode != mode
            {
                panic!("one compiled-pipeline suite request cannot mix {installed_mode:?} and {mode:?}");
            }
            self.mode.set(Some(mode));

            let handle = if let Some(handle) = self.handle.borrow().as_ref() {
                Arc::clone(handle)
            } else {
                let mut backend = self.backend.borrow_mut();
                ensure_suite_backend(&mut backend, mode)?;
                let handle = Arc::new(
                    backend
                        .as_ref()
                        .expect("suite backend is installed")
                        .core
                        .session_capability()
                        .admit_query(TenantId::new(PIPELINE_TENANT))?,
                );
                *self.handle.borrow_mut() = Some(Arc::clone(&handle));
                handle
            };
            let context = handle.acquire_context()?;
            Ok(PipelineTestContext { context, _request: Arc::clone(self), external_history })
        }
    }

    impl Drop for Runtime {
        fn drop(&mut self) {
            if let Some(handle) = self.handle.get_mut().take() {
                handle.finish();
            }
        }
    }
}

struct PipelineTestContext {
    context: TenantQueryContext,
    _request: Arc<cudf::Runtime>,
    external_history: Option<Arc<AdaptiveMemoryHistory>>,
}

impl PipelineTestContext {
    fn install_query_history(&self, plan: &QueryPlan) {
        let Some(source) = &self.external_history else {
            return;
        };
        let Some(summary) = source.lookup_query_plan(plan) else {
            return;
        };
        let target = self.context.context().adaptive_memory_history().expect("adaptive service context owns history");
        target.record_observation(
            AdaptiveMemoryHistoryKey::for_query_plan(plan),
            AdaptiveMemoryObservation {
                resident_dataframe_peak_bytes: summary.max_resident_dataframe_peak_bytes,
                reservation_granted_bytes: summary.max_reservation_granted_bytes,
                source_chunks_committed: summary.max_source_chunks_committed,
            },
        );
    }
}

impl Deref for PipelineTestContext {
    type Target = ExecutionContext;

    fn deref(&self) -> &Self::Target {
        self.context.context()
    }
}

fn execute_query_plan(context: &PipelineTestContext, plan: &QueryPlan) -> Result<QueryExecutionResult> {
    context.install_query_history(plan);
    nexus_query_engine::exec::execute_query_plan(context.context.context(), plan)
}

type Int32Triple = (Option<i32>, Option<i32>, Option<i32>);
type Int32Quad = (Option<i32>, Option<i32>, Option<i32>, Option<i32>);

fn runtime_visible_gate_for_task_kind(
    pipeline: &CompiledPipeline,
    task_kind: PipelineTaskKind,
    reservation_kind: ReservationKind,
) -> Option<&ReservationGate> {
    let task = pipeline.tasks().iter().find(|task| task.kind() == task_kind)?;
    task.gates()
        .iter()
        .chain(
            pipeline
                .edges()
                .iter()
                .filter(|edge| edge.from() == task.id() || edge.to() == task.id())
                .flat_map(|edge| edge.gates()),
        )
        .chain(
            pipeline
                .barriers()
                .iter()
                .filter(|barrier| {
                    barrier.upstream_tasks().contains(&task.id()) || barrier.downstream_tasks().contains(&task.id())
                })
                .flat_map(|barrier| barrier.gates()),
        )
        .chain((pipeline.sink().input() == task.id()).then_some(pipeline.sink().gates()).into_iter().flatten())
        .find_map(|gate_id| {
            pipeline.gates().iter().find(|gate| gate.id() == *gate_id && gate.reservation_kind() == reservation_kind)
        })
}

// `fixtures_plans::adaptive` is shared with the CPU contract target. The
// compiled GPU target still type-checks that fixture but does not invoke it.
fn literal_expr_for_type(data_type: &DataType) -> Expr {
    let literal = match data_type {
        DataType::Decimal128(precision, scale) => {
            LiteralValue::Decimal128 { raw: 12_345, precision: *precision, scale: *scale }
        },
        _ => LiteralValue::Int32(12_345),
    };
    Expr::literal(literal).alias("secret_literal")
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int32_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<i32>, Option<i64>)> {
    let mut pairs =
        int32_values(batch, left_column).into_iter().zip(int64_values(batch, right_column)).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn string_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<String>, Option<i64>)> {
    let array = batch.column(left_column).as_any().downcast_ref::<StringArray>().unwrap();
    let left = (0..array.len())
        .map(|index| if array.is_null(index) { None } else { Some(array.value(index).to_string()) })
        .collect::<Vec<_>>();
    let mut pairs = left.into_iter().zip(int64_values(batch, right_column)).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn int32_triples_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
) -> Vec<Int32Triple> {
    let mut rows = int32_values(batch, first_column)
        .into_iter()
        .zip(int32_values(batch, second_column))
        .zip(int32_values(batch, third_column))
        .map(|((first, second), third)| (first, second, third))
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

fn int32_quads_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
    fourth_column: usize,
) -> Vec<Int32Quad> {
    let mut rows = int32_values(batch, first_column)
        .into_iter()
        .zip(int32_values(batch, second_column))
        .zip(int32_values(batch, third_column))
        .zip(int32_values(batch, fourth_column))
        .map(|(((first, second), third), fourth)| (first, second, third, fourth))
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(
        pipeline.tasks().iter().filter(|task| task.kind() == kind).count(),
        expected,
        "unexpected count for {kind:?} tasks"
    );
}

fn bounded_context(runtime: Arc<cudf::Runtime>) -> PipelineTestContext {
    bounded_context_with_mode(runtime, NativePipelineExecutionMode::StrictBounded)
}

fn bounded_context_with_mode(runtime: Arc<cudf::Runtime>, mode: NativePipelineExecutionMode) -> PipelineTestContext {
    runtime.acquire_context(mode, None).unwrap()
}

fn bounded_context_with_mode_and_memory_history(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
    memory_history: Arc<AdaptiveMemoryHistory>,
) -> PipelineTestContext {
    runtime.acquire_context(mode, Some(memory_history)).unwrap()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
