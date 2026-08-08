//! Public-route GPU integration coverage for native execution.
//!
//! Every execution context in this target is admitted through the process-owned
//! backend core. Tests that specifically exercise raw context construction or
//! caller-owned runtime/cache/governor injection remain crate-private.

use std::{
    collections::HashMap,
    fs::{self, File},
    ops::Deref,
    path::Path,
    sync::{Arc, OnceLock},
};

use arrow::{
    array::{
        Array, AsArray, BooleanArray, Date32Array, Date64Array, Decimal128Array, DurationNanosecondArray, Float32Array,
        Float64Array, Int8Array, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray,
        Time32MillisecondArray, Time64NanosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray,
        UInt32Array, UInt64Array,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use nexus_query_engine::{
    chunk::{SourceChunkPlanningPolicy, plan_local_parquet_source_chunks},
    error::QueryErrorCode,
    exec::{
        execute_query_plan, execute_query_plan_with_gpu_inputs, execute_query_plan_with_gpu_inputs_observed,
        force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
        force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test, materialize_gpu_result_to_host,
        output::QueryOutput,
    },
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, GroupingSetsSpec, LimitSpec, LiteralValue,
        SortExpr, SortSpec,
    },
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{MemoryReservationEventAction, QueryMetrics},
    observability::{
        PlanExecutionBoundaryKind, RuntimeObservationConfig, RuntimeObservationPayload, RuntimeTerminalOutcome,
    },
    plan::{
        GraphEdgeBetweennessCentralityParams, JoinKey, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec,
        JoinType, ProjectionOutput, QueryPlanBuilder, WindowFunction, WindowSpec,
    },
    runtime::{
        AdmittedQueryContext, BoundedJoinReservationContract, DeploymentIoCapability, ExecutionBoundaryPolicy,
        ExecutionContext, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuMemoryOwnership, KvikioCompatModeStatus, MemoryControlPolicy,
        MemorySpaceId, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle, QueueBounds,
        ReservationKind, SourceSchedulerPolicy,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn,
        ParquetEqualityDeleteFile, ParquetPositionDeleteFile, PartitionColumn, ResolvedParquetSource, SourceNode,
        ZERO_COLUMN_COUNT_COLUMN,
    },
};
#[cfg(feature = "cugraph")]
use nexus_query_engine::{
    exec::force_next_graph_statistics_optional_resource_exhausted_for_test,
    graph_memory::GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
    graph_memory::GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES, graph_memory::GraphMemoryModel,
    graph_memory::GraphMemoryModelInput, graph_memory::GraphReservationOutcome, graph_memory::GraphSeedMode,
    graph_memory::GraphVertexEstimate, graph_memory::GraphVertexEstimateSource,
    graph_memory::graph_vertex_estimate_probe_bytes, plan::GraphAlgorithm, plan::GraphAlgorithmParams,
    plan::GraphAlgorithmSpec, plan::GraphBetweennessCentralityParams, plan::GraphBetweennessSeeds,
    plan::GraphBfsEndpointPolicy, plan::GraphBfsParams, plan::GraphBfsPredicateInputs, plan::GraphConstructionOptions,
    plan::GraphConstructionPolicy, plan::GraphEdgeColumnBindings, plan::GraphEdgePredicateInput,
    plan::GraphPageRankParams, plan::GraphPageRankPersonalizationInput, plan::GraphPersonalizedPageRankParams,
    plan::GraphPredicateInput, plan::GraphResidencyPlanAnalysis, plan::GraphSimilarityParams,
    plan::GraphTriangleCountAllParams, plan::GraphVertexPairInput, plan::graph_residency_descriptor_for_edge_plan,
    plan::stable_graph_result_schema,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

// Keep this integration fixture independent of the host inventory.  The
// native-execution cases derive both success and failure envelopes from their
// requested grants (up to 8 GiB), so a host-sized automatic profile would
// silently raise small grants through its derived attempt floor.
const NATIVE_EXECUTION_TEST_DEVICE_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const NATIVE_EXECUTION_TEST_ATTEMPT_FLOOR_BYTES: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackendTestPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
}

impl Default for BackendTestPolicy {
    fn default() -> Self {
        Self {
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
            source_scheduler_policy: SourceSchedulerPolicy::default(),
        }
    }
}

impl BackendTestPolicy {
    fn from_memory_control_policy(policy: MemoryControlPolicy) -> (Self, u64) {
        let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = policy
        else {
            panic!("the public backend test harness requires a bounded service policy");
        };
        (
            Self {
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
                source_scheduler_policy: SourceSchedulerPolicy::default(),
            },
            device_budget_bytes,
        )
    }

    fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }
}

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn suite_backend() -> &'static GpuBackendCore {
    suite_backend_with_policy(BackendTestPolicy::default())
}

fn suite_backend_with_policy(policy: BackendTestPolicy) -> &'static GpuBackendCore {
    static BACKEND: OnceLock<(BackendTestPolicy, GpuBackendCore)> = OnceLock::new();
    let (installed_policy, backend) = BACKEND.get_or_init(|| {
        let controller = GpuBackendControllerDraft::new(
            QueryExecutionConfig {
                max_source_chunk_bytes: policy.max_source_chunk_bytes,
                max_row_groups_per_chunk: policy.max_row_groups_per_chunk,
                max_retry_attempts: policy.max_retry_attempts,
                min_retry_chunk_row_groups: policy.min_retry_chunk_row_groups,
                source_scheduler_policy: policy.source_scheduler_policy,
                ..QueryExecutionConfig::default()
            },
            QueueBounds::default(),
        );
        let device = GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: NATIVE_EXECUTION_TEST_DEVICE_LIMIT_BYTES })
            .with_attempt_floor_bytes(NATIVE_EXECUTION_TEST_ATTEMPT_FLOOR_BYTES)
            .with_max_active_queries(1)
            .with_installation_bounds(4, 1, 1);
        let backend = GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, non_gds_deployment()))
            .build()
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
    context: AdmittedQueryContext,
    _handle: QueryHandle,
}

impl Deref for BackendQueryContext {
    type Target = ExecutionContext;

    fn deref(&self) -> &Self::Target {
        self.context.context()
    }
}

fn backend_context() -> BackendQueryContext {
    backend_context_with_grant(None)
}

fn backend_context_with_grant(requested_bytes: impl Into<Option<u64>>) -> BackendQueryContext {
    backend_context_with_policy_and_grant(BackendTestPolicy::default(), requested_bytes.into())
}

fn backend_context_with_memory_control_policy(policy: MemoryControlPolicy) -> BackendQueryContext {
    let (policy, device_budget_bytes) = BackendTestPolicy::from_memory_control_policy(policy);
    backend_context_with_policy_and_grant(policy, Some(device_budget_bytes))
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

fn backend_context_with_policy_and_grant(
    policy: BackendTestPolicy,
    requested_bytes: Option<u64>,
) -> BackendQueryContext {
    let request = requested_bytes.map_or_else(QueryAdmissionRequest::new, |bytes| {
        QueryAdmissionRequest::new().with_test_placement_requirement(fixture_placement_requirement(bytes))
    });
    let handle = suite_backend_with_policy(policy).session_capability().admit_query_with_request(request).unwrap();
    let context = handle.acquire_context().unwrap();
    BackendQueryContext { context, _handle: handle }
}

/// Creates the exact synthetic placement request used to bound one test
/// execution. It is not submitted-plan proof and cannot authorize an
/// arbitrary plan at execution. Production source-stage estimates deliberately
/// are not whole-attempt grant targets, so the fixture must state its test
/// grant directly rather than laundering it through a Parquet source model.
fn fixture_placement_requirement(bytes: u64) -> nexus_query_engine::admission::AttemptCapabilityEvidence {
    nexus_query_engine::admission::AttemptCapabilityEvidence::from_parts(
        0,
        0,
        Some(bytes),
        nexus_query_engine::admission::EstimateQuality::Measured,
        Vec::new(),
    )
}

fn backend_runtime() -> Arc<cudf::Runtime> {
    // Fixture-only GPU work may use the backend's declared device after the
    // process owner is installed. Never inject this runtime or its GPU objects
    // into execution; admitted contexts provide the execution runtime.
    let _ = suite_backend();
    let device = cudf::DeviceId::new(0).expect("native execution suite fixture device");
    Arc::new(cudf::Runtime::builder().with_device(device).build().expect("native execution suite fixture runtime"))
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
use parquet_fixtures::*;
use projection_test_support::*;
use test_support::*;

#[test]
fn backend_session_executes_grouped_aggregate_from_literal_batches() {
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
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("key")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "total")],
        ),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
        output_schema,
    );

    let result = execute_query_plan(&context, &builder.finish_with_host_sink(sort)).unwrap();
    let QueryOutput::HostArrow(output) = result.output() else { panic!("host sink must return Arrow output") };
    assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(output, 1).values().as_ref(), &[30, 7]);
}
