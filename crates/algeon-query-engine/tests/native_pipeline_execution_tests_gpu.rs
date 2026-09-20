//! GPU execution ownership for compiled native pipelines.
//!
//! This target is intentionally separate from `native_contract_tests`: its
//! cases create a process-owned GPU backend, while the remaining contract
//! target is CPU-only.

use std::{
    cell::RefCell,
    collections::BTreeSet,
    fs::{self, File},
    ops::Deref,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_engine::{
    admission::{
        AttemptCapabilityEvidence, BoundedPipelineAdmission, CompiledBoundedPipelineSupport, bounded_pipeline_decision,
        compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
    },
    dynamic_filter::DynamicFilterProducer,
    error::Result,
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, QueryExecutionRequest, materialize_gpu_result_to_host,
        output::{QueryExecutionResult, QueryOutput},
    },
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, GroupingSetsSpec,
        LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::QueryMetrics,
    mode::StatisticalAggregateExecutionMode,
    observability::{RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationOwner},
    optimizer::{JoinBuildSideSelectionOptions, NativeOptimizerOptions, optimize_query_plan},
    pipeline::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentLifetime, ResidentResource,
        ResidentResourceId, SinkBoundary, SinkBoundaryKind, compile_pipeline,
    },
    plan::{
        JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode, ProjectionOutput, QueryPlan,
        QueryPlanBuilder, WindowFunction, WindowSpec,
    },
    runtime::{
        AdmittedQueryContext, DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus,
        GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuMemoryOwnership, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle, QueueBounds,
        resident::GpuResidentInputMap,
    },
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource as NativeLocalParquetSource,
        PartitionColumn, SourceNode,
    },
};

fn bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<BoundedPipelineAdmission>> {
    bounded_pipeline_decision(
        plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )?
    .into_execution_admission()
}

fn bounded_pipeline_rejection_projection(
    plan: &QueryPlan,
) -> Result<Option<query_engine::capability::CapabilityRejectionProjection>> {
    Ok(bounded_pipeline_decision(
        plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )?
    .rejection_projection()
    .cloned())
}

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
#[expect(dead_code, reason = "shared fixture file partially exercised in this integration test binary")]
mod fixtures_plans;
#[path = "native_contract_tests/pipeline_compiled_execution/gpu_execution.rs"]
mod gpu_execution;
#[path = "native_contract_tests/pipeline_compiled_execution/lane_execution.rs"]
mod lane_execution;
#[path = "native_contract_tests/bounded_join_admission/plans/remaining_topologies.rs"]
// Shared with the CPU contract target; this binary only calls three constructors.
#[expect(dead_code, reason = "shared fixture file partially exercised in this integration test binary")]
mod remaining_topologies;

use fixtures_parquet::*;
use fixtures_plans::*;
use remaining_topologies::{
    AggregateOverJoinSubplanTopKVariant, ResidentGroupedAggregateInputPostJoinAggregateTopKVariant,
    aggregate_over_join_subplan_topk_plan_inner, resident_grouped_aggregate_input_post_join_aggregate_topk_plan_inner,
    resident_grouped_aggregate_subquery_filter_plan,
};

struct SuiteBackend {
    core: GpuBackendCore,
}

static PIPELINE_BACKEND: Mutex<Option<SuiteBackend>> = Mutex::new(None);

fn build_suite_backend() -> Result<GpuBackendCore> {
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let policy = QueryExecutionConfig {
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
        ..QueryExecutionConfig::default()
    };
    let controller = GpuBackendControllerDraft::new(policy, QueueBounds::default());
    // One backend owns the suite; each exact-plan context acquires its own handle.
    // The suite runs one query at a time, so it pins the explicit serial
    // admission the installation bounds were sized for.
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0)
            .with_max_active_attempts(1)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(3, 1, 1)),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, deployment)).build()
}

fn ensure_suite_backend(backend: &mut Option<SuiteBackend>) -> Result<()> {
    if backend.is_none() {
        *backend = Some(SuiteBackend { core: build_suite_backend()? });
    }
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
    }

    impl Runtime {
        pub(super) fn new() -> std::result::Result<Self, String> {
            let backend = PIPELINE_BACKEND.lock().map_err(|_| "pipeline backend lock poisoned".to_string())?;
            Ok(Self { backend: RefCell::new(backend) })
        }

        pub(super) fn acquire_context_for_plan(self: &Arc<Self>, plan: &QueryPlan) -> Result<PipelineTestContext> {
            query_engine::exec::preflight_bounded_query_plan(plan, StatisticalAggregateExecutionMode::default())?;
            let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
            self.acquire_context_with_evidence_items([evidence])
        }

        pub(super) fn acquire_context_for_plan_with_observation(
            self: &Arc<Self>,
            plan: &QueryPlan,
            owner: RuntimeObservationOwner,
        ) -> Result<PipelineTestContext> {
            query_engine::exec::preflight_bounded_query_plan(plan, StatisticalAggregateExecutionMode::default())?;
            let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
            let mut backend = self.backend.borrow_mut();
            ensure_suite_backend(&mut backend)?;
            let handle = backend
                .as_ref()
                .expect("suite backend is installed")
                .core
                .session_capability()
                .admit_query_with_request(
                    QueryAdmissionRequest::new()
                        .with_query_plan_capability_evidence(evidence)
                        .with_runtime_observation_owner(owner),
                )?;
            let context = handle.acquire_context()?;
            Ok(PipelineTestContext { context, _handle: handle, _request: Arc::clone(self) })
        }

        pub(super) fn acquire_context_for_compiled_executor_test(
            self: &Arc<Self>,
            plan: &QueryPlan,
        ) -> Result<PipelineTestContext> {
            let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
            self.acquire_context_with_evidence_items([evidence])
        }

        pub(super) fn acquire_context_with_evidence_items(
            self: &Arc<Self>,
            evidence_items: impl IntoIterator<Item = AttemptCapabilityEvidence>,
        ) -> Result<PipelineTestContext> {
            let evidence = AttemptCapabilityEvidence::aggregate(evidence_items)?;
            let mut backend = self.backend.borrow_mut();
            ensure_suite_backend(&mut backend)?;
            let handle = backend
                .as_ref()
                .expect("suite backend is installed")
                .core
                .session_capability()
                .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))?;
            let context = handle.acquire_context()?;
            Ok(PipelineTestContext { context, _handle: handle, _request: Arc::clone(self) })
        }
    }
}

struct PipelineTestContext {
    context: AdmittedQueryContext,
    _handle: QueryHandle,
    _request: Arc<cudf::Runtime>,
}

impl Deref for PipelineTestContext {
    type Target = ExecutionContext;

    fn deref(&self) -> &Self::Target {
        self.context.context()
    }
}

fn execute_query_plan(context: &PipelineTestContext, plan: &QueryPlan) -> Result<QueryExecutionResult> {
    let outcome = query_engine::exec::execute_query_plan(context.context.context(), QueryExecutionRequest::new(plan));
    let (result, metrics, _observation) = outcome.into_parts();
    result.map(|output| QueryExecutionResult::new(output, metrics))
}

fn execute_query_plan_with_gpu_inputs(
    context: &PipelineTestContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<QueryExecutionResult> {
    let outcome = query_engine::exec::execute_query_plan(
        context.context.context(),
        QueryExecutionRequest::new(plan).with_gpu_inputs(gpu_inputs),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    result.map(|output| QueryExecutionResult::new(output, metrics))
}

type Int32Triple = (Option<i32>, Option<i32>, Option<i32>);
type Int32Quad = (Option<i32>, Option<i32>, Option<i32>, Option<i32>);

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

fn bounded_context(runtime: Arc<cudf::Runtime>, plan: &QueryPlan) -> PipelineTestContext {
    runtime.acquire_context_for_plan(plan).unwrap()
}

fn bounded_observed_context(
    runtime: Arc<cudf::Runtime>,
    plan: &QueryPlan,
) -> (PipelineTestContext, RuntimeObservationHandle) {
    let owner =
        RuntimeObservationOwner::new("observed-compiled-pipeline", None, RuntimeObservationConfig::native_execution());
    let observation = owner.handle();
    let context = runtime.acquire_context_for_plan_with_observation(plan, owner).unwrap();
    (context, observation)
}

fn compiled_executor_context(runtime: Arc<cudf::Runtime>, plan: &QueryPlan) -> PipelineTestContext {
    runtime.acquire_context_for_compiled_executor_test(plan).unwrap()
}

fn bounded_context_for_evidence_items(
    runtime: Arc<cudf::Runtime>,
    evidence_items: impl IntoIterator<Item = AttemptCapabilityEvidence>,
) -> PipelineTestContext {
    runtime.acquire_context_with_evidence_items(evidence_items).unwrap()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
