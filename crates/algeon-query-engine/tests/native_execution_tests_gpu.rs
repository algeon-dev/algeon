//! Public-route GPU integration coverage for native execution.
//!
//! Every execution context in this target is admitted through the process-owned
//! backend core. Tests that specifically exercise raw context construction or
//! caller-owned runtime/cache/governor injection remain crate-private.

use std::{
    fs::{self, File},
    ops::Deref,
    path::Path,
    sync::{Arc, OnceLock},
};

use arrow::{
    array::{
        Array, ArrayRef, AsArray, BooleanArray, Date32Array, Date64Array, Decimal32Array, Decimal64Array,
        Decimal128Array, DurationNanosecondArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
        Int16Array, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray, StringViewArray,
        Time32MillisecondArray, Time64NanosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_engine::{
    chunk::plan_local_parquet_source_chunks,
    exec::{
        QueryExecutionRequest, admit_query_plan, execute_query_plan as execute_query_plan_direct,
        materialize_gpu_result_to_host,
        output::{QueryExecutionResult, QueryOutput},
        preflight_query_plan,
    },
};
use query_planning::{
    admission::AttemptCapabilityEvidence,
    chunk::SourceChunkPlanningPolicy,
    error::{QueryErrorCode, Result as QueryEngineResult},
    execution_contracts::MemoryControlPolicy,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, GroupingSetsSpec, LimitSpec, LiteralValue,
        SortExpr, SortSpec,
    },
    mode::StatisticalAggregateExecutionMode,
    plan::{
        JoinKey, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType, PlanNode, ProjectionOutput,
        QueryPlan, QueryPlanBuilder, WindowFunction, WindowSpec,
    },
    policy::PlanningPolicy,
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource as NativeLocalParquetSource,
        MissingColumn, PartitionColumn, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
    },
};
#[cfg(feature = "cugraph")]
use query_planning::{
    plan::GraphAlgorithm, plan::GraphAlgorithmParams, plan::GraphAlgorithmSpec, plan::GraphBetweennessCentralityParams,
    plan::GraphBetweennessSeeds, plan::GraphBfsEndpointPolicy, plan::GraphBfsParams, plan::GraphBfsPredicateInputs,
    plan::GraphConstructionOptions, plan::GraphConstructionPolicy, plan::GraphEdgeColumnBindings,
    plan::GraphEdgePredicateInput, plan::GraphPageRankParams, plan::GraphPageRankPersonalizationInput,
    plan::GraphPersonalizedPageRankParams, plan::GraphPredicateInput, plan::GraphResidencyPlanAnalysis,
    plan::GraphSccParams, plan::GraphSimilarityParams, plan::GraphVertexPairInput, plan::GraphWccParams,
    plan::graph_residency_descriptor_for_edge_plan,
};
use query_runtime::{
    attempt::{AttemptAdmissionBinding, AttemptOwner, BeginAttemptRequest},
    metrics::QueryMetrics,
    observability::{AttemptDisposition, AttemptQueryId, RuntimeObservationConfig, RuntimeTerminalOutcome},
    runtime::{
        AdmittedQueryContext, ExecutionContext, GpuBackendConfig, GpuBackendCore, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuMemoryOwnership, QueryAdmissionRequest,
        QueryExecutionConfig, QueryHandle, QueueBounds, SourceSchedulerPolicy,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
};

// Ordinary integration cases exercise real allocations under a bounded cap;
// failure cases request their smaller cap explicitly through the fixture.
const NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackendTestPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_work_per_attempt: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    planning_policy: PlanningPolicy,
}

impl Default for BackendTestPolicy {
    fn default() -> Self {
        Self {
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
            source_work_per_attempt: 1,
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            planning_policy: PlanningPolicy::default(),
        }
    }
}

impl BackendTestPolicy {
    fn from_memory_control_policy(policy: MemoryControlPolicy) -> (Self, u64) {
        let MemoryControlPolicy {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = policy;
        (
            Self {
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
                source_work_per_attempt: 1,
                source_scheduler_policy: SourceSchedulerPolicy::default(),
                statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
                planning_policy: PlanningPolicy::default(),
            },
            device_budget_bytes,
        )
    }

    fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    fn with_statistical_aggregate_execution_mode(
        mut self,
        statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    ) -> Self {
        self.statistical_aggregate_execution_mode = statistical_aggregate_execution_mode;
        self
    }

    fn with_planning_policy(mut self, planning_policy: PlanningPolicy) -> Self {
        self.planning_policy = planning_policy;
        self
    }
}

fn suite_backend_with_policy(policy: BackendTestPolicy) -> &'static GpuBackendCore {
    static BACKEND: OnceLock<(BackendTestPolicy, GpuBackendCore)> = OnceLock::new();
    let (installed_policy, backend) = BACKEND.get_or_init(|| {
        let device = GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_default_query_limit_bytes(NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES)
                .with_max_active_attempts(1)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                    4,
                    // One worker is reserved for lifecycle jobs.
                    policy.source_work_per_attempt + 1,
                    policy.source_work_per_attempt,
                ))
                .with_source_work_per_attempt(policy.source_work_per_attempt),
            GpuMemoryOwnership::WholeDeviceExclusive,
        );
        let backend = GpuBackendCore::new(
            GpuBackendConfig::new(
                vec![device],
                QueryExecutionConfig {
                    max_source_chunk_bytes: policy.max_source_chunk_bytes,
                    max_row_groups_per_chunk: policy.max_row_groups_per_chunk,
                    max_retry_attempts: policy.max_retry_attempts,
                    min_retry_chunk_row_groups: policy.min_retry_chunk_row_groups,
                    source_scheduler_policy: policy.source_scheduler_policy,
                    statistical_aggregate_execution_mode: policy.statistical_aggregate_execution_mode,
                    planning_policy: policy.planning_policy,
                    ..QueryExecutionConfig::default()
                },
                QueueBounds::default(),
            )
            .validate()
            .expect("valid backend config"),
        )
        .expect("native execution suite backend");
        (policy, backend)
    });
    assert_eq!(
        *installed_policy, policy,
        "one native-execution test process cannot install conflicting backend policies"
    );
    backend
}

struct BackendQueryContext {
    policy: BackendTestPolicy,
    requested_bytes: Option<u64>,
}

struct AdmittedPlanContext {
    context: AdmittedQueryContext,
    handle: QueryHandle,
}

impl Deref for AdmittedPlanContext {
    type Target = ExecutionContext;

    fn deref(&self) -> &Self::Target {
        self.context.context()
    }
}

impl AdmittedPlanContext {
    fn finish(&self) {
        self.handle.end_test_attempt();
    }
}

fn backend_context() -> BackendQueryContext {
    backend_context_with_grant(None)
}

fn backend_context_with_grant(requested_bytes: impl Into<Option<u64>>) -> BackendQueryContext {
    backend_context_with_policy_and_grant(BackendTestPolicy::default(), requested_bytes.into())
}

fn backend_context_with_statistical_aggregate_mode(
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> BackendQueryContext {
    backend_context_with_policy_and_grant(
        BackendTestPolicy::default().with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode),
        None,
    )
}

fn backend_context_with_grant_and_statistical_aggregate_mode(
    requested_bytes: impl Into<Option<u64>>,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> BackendQueryContext {
    backend_context_with_policy_and_grant(
        BackendTestPolicy::default().with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode),
        requested_bytes.into(),
    )
}

fn backend_context_with_memory_control_policy(policy: MemoryControlPolicy) -> BackendQueryContext {
    backend_context_with_memory_control_policy_and_statistical_aggregate_mode(
        policy,
        StatisticalAggregateExecutionMode::default(),
    )
}

fn backend_context_with_memory_control_policy_and_statistical_aggregate_mode(
    policy: MemoryControlPolicy,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> BackendQueryContext {
    let (policy, device_budget_bytes) = BackendTestPolicy::from_memory_control_policy(policy);
    backend_context_with_policy_and_grant(
        policy.with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode),
        Some(device_budget_bytes),
    )
}

fn backend_context_with_memory_and_scheduler_policy(
    memory_control_policy: MemoryControlPolicy,
    source_scheduler_policy: SourceSchedulerPolicy,
) -> BackendQueryContext {
    let (policy, device_budget_bytes) = BackendTestPolicy::from_memory_control_policy(memory_control_policy);
    backend_context_with_policy_and_grant(
        policy.with_source_scheduler_policy(source_scheduler_policy),
        Some(device_budget_bytes),
    )
}

fn backend_context_with_grant_and_planning_policy(
    requested_bytes: u64,
    planning_policy: PlanningPolicy,
) -> BackendQueryContext {
    backend_context_with_policy_and_grant(
        BackendTestPolicy::default().with_planning_policy(planning_policy),
        Some(requested_bytes),
    )
}

fn backend_context_with_memory_control_and_planning_policy(
    memory_control_policy: MemoryControlPolicy,
    planning_policy: PlanningPolicy,
) -> BackendQueryContext {
    let (policy, device_budget_bytes) = BackendTestPolicy::from_memory_control_policy(memory_control_policy);
    backend_context_with_policy_and_grant(policy.with_planning_policy(planning_policy), Some(device_budget_bytes))
}

fn backend_context_with_policy_and_grant(
    policy: BackendTestPolicy,
    requested_bytes: Option<u64>,
) -> BackendQueryContext {
    BackendQueryContext { policy, requested_bytes }
}

fn backend_runtime() -> Arc<cudf::Runtime> {
    backend_runtime_with_policy(BackendTestPolicy::default())
}

fn backend_runtime_with_policy(policy: BackendTestPolicy) -> Arc<cudf::Runtime> {
    // Fixture-only GPU work may use the backend's declared device after the
    // process owner is installed. Never inject this runtime or its GPU objects
    // into execution; admitted contexts provide the execution runtime.
    let _ = suite_backend_with_policy(policy);
    let device = cudf::DeviceId::new(0).expect("native execution suite fixture device");
    Arc::new(cudf::Runtime::builder().with_device(device).build().expect("native execution suite fixture runtime"))
}

impl BackendQueryContext {
    fn with_source_work_per_attempt(mut self, source_work_per_attempt: usize) -> Self {
        self.policy.source_work_per_attempt = source_work_per_attempt;
        self
    }

    fn runtime(&self) -> QueryEngineResult<cudf::Runtime> {
        Ok(backend_runtime_with_policy(self.policy).as_ref().clone())
    }

    fn admit_plan(&self, plan: &QueryPlan) -> QueryEngineResult<AdmittedPlanContext> {
        admit_query_plan(plan, self.policy.statistical_aggregate_execution_mode, self.policy.planning_policy)?;
        let evidence = capability_evidence_for_fixture_plan(plan)?;
        self.admit_evidence(evidence)
    }

    fn admit_evidence(&self, evidence: AttemptCapabilityEvidence) -> QueryEngineResult<AdmittedPlanContext> {
        self.admit_evidence_with_observation(evidence, None)
    }

    fn begin_native_attempt(&self, query_id: &str, observation: RuntimeObservationConfig) -> AttemptOwner {
        let attempt = suite_backend_with_policy(self.policy)
            .begin_attempt(BeginAttemptRequest {
                query_id: Some(AttemptQueryId::new(query_id).expect("valid attempt query id")),
                observation,
                ..BeginAttemptRequest::default()
            })
            .expect("suite backend begins an attempt")
            .owner;
        attempt.handle().set_disposition(AttemptDisposition::Native).expect("a new attempt accepts a disposition");
        attempt
    }

    fn admit_evidence_with_observation(
        &self,
        evidence: AttemptCapabilityEvidence,
        attempt: Option<AttemptAdmissionBinding>,
    ) -> QueryEngineResult<AdmittedPlanContext> {
        let request = attempt.map_or_else(QueryAdmissionRequest::test_attempt, QueryAdmissionRequest::new);
        let mut request = request.with_query_plan_capability_evidence(evidence);
        if let Some(bytes) = self.requested_bytes {
            request = request.with_query_cap_bytes(bytes);
        }
        let handle = suite_backend_with_policy(self.policy).session_capability().admit_query_with_request(request)?;
        let context = handle.acquire_context()?;
        Ok(AdmittedPlanContext { context, handle })
    }

    fn execute_admitted(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> QueryEngineResult<QueryExecutionResult> {
        let admitted_plan =
            admit_query_plan(plan, self.policy.statistical_aggregate_execution_mode, self.policy.planning_policy)?;
        let evidence = capability_evidence_for_fixture_plan(plan)?;
        let admitted = self.admit_evidence_with_observation(evidence, None)?;
        let outcome = execute_query_plan_direct(
            admitted.context.context(),
            QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(gpu_inputs),
        );
        let (result, metrics, _observation) = outcome.into_parts();
        // The attempt ends with the execution's outcome, as a frontend
        // commits it; a resident output then stays frozen past the attempt.
        admitted.handle.commit_test_attempt(match &result {
            Ok(_) => RuntimeTerminalOutcome::Succeeded,
            Err(error) => RuntimeTerminalOutcome::from_query_error(error),
        });
        result.map(|output| QueryExecutionResult::new(output, metrics.snapshot()))
    }
}

fn capability_evidence_for_fixture_plan(plan: &QueryPlan) -> QueryEngineResult<AttemptCapabilityEvidence> {
    preflight_query_plan(plan)?;
    let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
    #[cfg(feature = "cugraph")]
    let evidence = {
        let mut evidence = evidence;
        if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_))) {
            evidence.attest_cugraph_graph_execution_scope(plan);
        }
        evidence
    };
    Ok(evidence)
}

trait TestExecutionContext {
    fn execute_plan(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> QueryEngineResult<QueryExecutionResult>;
}

impl TestExecutionContext for AdmittedPlanContext {
    fn execute_plan(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> QueryEngineResult<QueryExecutionResult> {
        let execution_context = self.context.context();
        let admitted = admit_query_plan(
            plan,
            execution_context.statistical_aggregate_execution_mode(),
            execution_context.planning_policy(),
        )?;
        let outcome = execute_query_plan_direct(
            execution_context,
            QueryExecutionRequest::admitted(&admitted).with_gpu_inputs(gpu_inputs),
        );
        let (result, metrics, _observation) = outcome.into_parts();
        result.map(|output| QueryExecutionResult::new(output, metrics.snapshot()))
    }
}

impl TestExecutionContext for BackendQueryContext {
    fn execute_plan(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> QueryEngineResult<QueryExecutionResult> {
        self.execute_admitted(plan, gpu_inputs)
    }
}

fn execute_query_plan<C: TestExecutionContext + ?Sized>(
    context: &C,
    plan: &QueryPlan,
) -> QueryEngineResult<QueryExecutionResult> {
    context.execute_plan(plan, &GpuResidentInputMap::new())
}

fn execute_query_plan_with_gpu_inputs<C: TestExecutionContext + ?Sized>(
    context: &C,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
) -> QueryEngineResult<QueryExecutionResult> {
    context.execute_plan(plan, gpu_inputs)
}

#[path = "native_execution_tests/aggregate_execution.rs"]
mod aggregate_execution;
#[path = "native_execution_tests/aggregate_join_pushdown.rs"]
mod aggregate_join_pushdown;
#[path = "native_execution_tests/aggregate_test_support.rs"]
mod aggregate_test_support;
#[path = "native_execution_tests/bounded_budgeting.rs"]
mod bounded_budgeting;
#[path = "native_execution_tests/bounded_joins.rs"]
mod bounded_joins;
#[path = "native_execution_tests/core_execution.rs"]
mod core_execution;
#[path = "native_execution_tests/dimension_pruning.rs"]
mod dimension_pruning;
#[cfg(feature = "cugraph")]
#[path = "native_execution_tests/graph_algorithm_execution.rs"]
mod graph_algorithm_execution;
#[path = "native_execution_tests/join_test_support.rs"]
mod join_test_support;
#[path = "native_execution_tests/metrics_assertions.rs"]
mod metrics_assertions;
#[path = "native_execution_tests/ml_algorithm_execution.rs"]
mod ml_algorithm_execution;
#[path = "native_execution_tests/native_join_execution.rs"]
mod native_join_execution;
#[path = "native_execution_tests/parquet_fixtures.rs"]
mod parquet_fixtures;
#[path = "native_execution_tests/partition_execution.rs"]
mod partition_execution;
#[path = "native_execution_tests/projection_test_support.rs"]
mod projection_test_support;
#[path = "native_execution_tests/projection_validation.rs"]
mod projection_validation;
#[path = "native_execution_tests/semijoin_reduction.rs"]
mod semijoin_reduction;
#[path = "native_execution_tests/test_support.rs"]
mod test_support;
#[path = "native_execution_tests/unbounded_pruning_regressions.rs"]
mod unbounded_pruning_regressions;
#[cfg(feature = "cuvs")]
#[path = "native_execution_tests/vector_algorithm_execution.rs"]
mod vector_algorithm_execution;
#[path = "native_execution_tests/window_execution.rs"]
mod window_execution;

use aggregate_test_support::*;
use join_test_support::*;
use metrics_assertions::*;
use parquet_fixtures::{fixture_local_parquet_source as LocalParquetSource, *};
use projection_test_support::*;
use test_support::*;

#[test]
fn backend_session_executes_grouped_aggregate_from_gpu_resident_input() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2])), Arc::new(Int64Array::from(vec![10, 20, 7]))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "total")],
            ),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort =
        builder.add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();

    let plan = builder.finish_with_host_sink(sort).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "input".to_owned(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(output) = result.output() else { panic!("host sink must return Arrow output") };
    assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(output, 1).values().as_ref(), &[30, 7]);
    admitted.finish();
}
