//! Native query execution.
//!
//! Entrypoints validate local source contracts, derive output mode from the
//! plan sink, and dispatch according to the bounded execution contract in the
//! [`ExecutionContext`].
//! GPU-resident input lookup is by plan-local names supplied in
//! [`GpuResidentInputMap`]. Metrics are owned by the returned
//! [`output::QueryExecutionResult`] so callers can record them after choosing host or
//! GPU-resident output handling.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use arrow::{
    array::Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec as CudfAggSpec, BudgetedExecutionOutcome, DataFrame, DuplicateKeepOption, ExecutionBudgetRequest,
    ExecutionBudgetToken, ExecutionOverrides, GroupByAgg, GroupByOptions, ReduceAgg, ShiftSpec as CudfShiftSpec,
    SortOptions as CudfSortOptions,
    core::Scalar as CudfScalar,
    frame::dataframe::{DropDuplicatesOptions, InfoDetail, ParquetReadOptions},
    io::source::SourceReadMode,
    ops::{col, lit, lit_null, reduction::ScanAgg, when},
};

mod bounded;
mod edge_normalize;
mod edge_null_filter;
mod execution_metrics;
mod graph_algorithm;
mod join;
mod materialize;
mod nvtx;
mod operators;
mod order;
/// Native execution output types.
pub mod output;
mod retry;
mod source;
mod util;
mod vector_algorithm;

#[cfg(test)]
pub(crate) use bounded::pipeline::YieldedFrameCounterTestHarness;
#[doc(hidden)]
pub use bounded::pipeline::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledSchedulingCycle, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
};
use edge_normalize::execute_edge_normalize;
#[doc(hidden)]
pub use execution_metrics::ExecutionMetrics;
use nvtx::attempt_execution_range;
use output::QueryOutput;
use source::{read_gpu_resident_source, read_literal_batch_source, read_local_parquet_source};

use self::{
    bounded::*,
    graph_algorithm::*,
    join::*,
    materialize::*,
    operators::*,
    order::*,
    output::{QueryExecutionOutcome, QueryOutput as Output},
    util::*,
    vector_algorithm::*,
};
#[cfg(feature = "cuvs")]
use crate::native_error::{cudf_interop_query_error, cuvs_query_error};
use crate::{
    chunk::{
        SourceChunk, SourceChunkBatch, SourceChunkPlan, SourceChunkPlanningPolicy, SourceTaskMemoryModel,
        annotate_source_chunk_batching_reasons, plan_local_parquet_source_chunks_with_execution_context,
        plan_source_chunk_batches, plan_source_chunk_batches_with_row_cap,
    },
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_implicit_cast, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, Expr, LimitSpec, LiteralValue},
    metrics::{QueryId, QueryMetrics, SourceReadRuntimeMetrics, TraceId},
    mode::StatisticalAggregateExecutionMode,
    native_error::{
        cudf_budget_exhaustion, cudf_query_error, cudf_query_error_with_budget_details,
        cudf_query_error_without_source, cudf_query_error_without_source_with_budget_details,
    },
    observability::{
        PlanEdgeRow, PlanExecutionId, PlanNodeRow, RuntimeExecutionObservationScope, RuntimePlanNodeId,
        RuntimeTerminalOutcome, RuntimeTopologyRecord,
    },
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, LimitNode, NodeId,
        PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode, WindowFrame, WindowNode,
    },
    prepared::*,
    runtime::{
        ExecutionContext, MemoryControlPolicy, ReservationKind, RetryFacts,
        resident::{
            GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildHandle, GpuResidentInputMap, GpuResultHandle,
        },
    },
    source::{
        GpuResidentSource, LocalParquetFile, LocalParquetSource, ZERO_COLUMN_COUNT_COLUMN,
        validate_local_parquet_source,
    },
    validation::*,
};

/// Validates a complete native plan without constructing a runtime or touching
/// GPU resources.
///
/// Frontends use this capability-first boundary before host work or native
/// admission. Execution entrypoints call the same validation internally.
pub fn preflight_query_plan(plan: &QueryPlan) -> Result<()> {
    nexus_query_planning::validate_query_plan(plan, crate::rapids_backend::rapids_analysis_context())
}

/// Decides the executable plan once and hands the admitted artifact to the
/// caller.
///
/// This boundary is used after frontend-owned plan resolution, before bounded
/// execution selects a compiled pipeline. Execution runs this same artifact
/// instead of re-deriving it; a missing artifact means the plan has no join
/// to decide. Unbounded plans continue to use [`preflight_query_plan`] alone.
/// A rejection carries the caller-visible sink as a diagnostic fact only.
pub fn preflight_bounded_query_plan(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<Option<crate::admission::BoundedPipelineAdmission>> {
    preflight_query_plan(plan)?;
    bounded::preflight_bounded_join_plan(plan, statistical_aggregate_execution_mode)
}

static NEXT_PLAN_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

/// Per-query execution state for unbounded native execution.
///
/// `node_results` memoizes every materialized node output for the duration of
/// one query so DAG-shaped plans reuse device-resident frames instead of
/// re-reading or recomputing shared subplans. Entries are released together when
/// `execute_query_plan` returns; there is intentionally no intra-query eviction
/// policy yet.
///
/// `hash_join_builds` memoizes reusable cuDF hash-join artifacts for repeated
/// probes against the same build node, build keys, and null-equality contract.
/// The repository retains the query runtime and device placement that produced
/// the artifacts, and is released with this scratch object at query end. It is
/// deliberately absent from bounded join execution until the lower layers expose
/// honest pre-build hash-table reservation bytes and the engine can reserve
/// build, probe, gather, and output memory before constructing join artifacts.
pub(super) struct ExecutionScratch {
    node_results: HashMap<NodeId, GpuDataFrame>,
    hash_join_builds: HashJoinBuildRepository,
}

impl ExecutionScratch {
    pub(super) fn new(context: &ExecutionContext) -> Result<Self> {
        Ok(Self {
            node_results: HashMap::new(),
            hash_join_builds: HashJoinBuildRepository::new(context.shared_runtime()?, context.device_placement()),
        })
    }
}

// Per-query cache of reusable cuDF hash-join builds for repeated probes.
pub(super) struct HashJoinBuildRepository {
    runtime: Arc<cudf::Runtime>,
    device_placement: crate::runtime::DevicePlacement,
    cache: HashMap<HashJoinCacheKey, GpuHashJoinBuildHandle>,
}

impl HashJoinBuildRepository {
    fn new(runtime: Arc<cudf::Runtime>, device_placement: crate::runtime::DevicePlacement) -> Self {
        Self { runtime, device_placement, cache: HashMap::new() }
    }

    pub(super) fn get(&self, key: &HashJoinCacheKey) -> Option<&GpuHashJoinBuildHandle> {
        debug_assert_eq!(
            self.device_placement,
            crate::runtime::DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
        );
        self.cache.get(key)
    }

    pub(super) fn insert(&mut self, key: HashJoinCacheKey, hash_join: GpuHashJoinBuildHandle) {
        debug_assert_eq!(
            self.device_placement,
            crate::runtime::DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
        );
        self.cache.insert(key, hash_join);
    }
}

// Hash-join build identity scoped to the build node and equality contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct HashJoinCacheKey {
    build_node: NodeId,
    build_keys: Vec<String>,
    null_equality: JoinNullEquality,
}

impl HashJoinCacheKey {
    pub(super) fn new(build_node: NodeId, build_keys: Vec<String>, null_equality: JoinNullEquality) -> Self {
        Self { build_node, build_keys, null_equality }
    }
}

/// Borrowed inputs for one native plan execution.
pub struct QueryExecutionRequest<'a> {
    plan: &'a QueryPlan,
    gpu_inputs: Option<&'a GpuResidentInputMap>,
    bounded_admission: Option<&'a crate::admission::BoundedPipelineAdmission>,
}

impl<'a> QueryExecutionRequest<'a> {
    /// Creates an execution request without external GPU-resident inputs.
    pub const fn new(plan: &'a QueryPlan) -> Self {
        Self { plan, gpu_inputs: None, bounded_admission: None }
    }

    /// Supplies GPU-resident inputs keyed by the source names in the plan.
    pub const fn with_gpu_inputs(mut self, gpu_inputs: &'a GpuResidentInputMap) -> Self {
        self.gpu_inputs = Some(gpu_inputs);
        self
    }

    /// Supplies the admission artifact the preflight decided for this plan.
    ///
    /// Execution checks the artifact against the plan instead of re-deriving
    /// it; a plan, mode, or sink the artifact does not bind to is a caller
    /// error. Callers without a preflight artifact omit this and keep the
    /// engine-local decision.
    pub const fn with_bounded_admission(mut self, admission: &'a crate::admission::BoundedPipelineAdmission) -> Self {
        self.bounded_admission = Some(admission);
        self
    }
}

/// Executes one native plan and retains metrics and observation evidence on
/// both success and structured failure.
///
/// A query-service context finishes only this plan execution. A standalone
/// observed context built by [`ExecutionContext::builder`] owns query
/// terminalization and returns its sealed snapshot in the outcome.
pub fn execute_query_plan(context: &ExecutionContext, request: QueryExecutionRequest<'_>) -> QueryExecutionOutcome {
    let empty_inputs = GpuResidentInputMap::new();
    let gpu_inputs = request.gpu_inputs.unwrap_or(&empty_inputs);
    execute_query_plan_with_observation(context, request.plan, request.bounded_admission, gpu_inputs)
}

fn execute_query_plan_with_observation(
    context: &ExecutionContext,
    plan: &QueryPlan,
    bounded_admission: Option<&crate::admission::BoundedPipelineAdmission>,
    gpu_inputs: &GpuResidentInputMap,
) -> QueryExecutionOutcome {
    let observation_handle = context.runtime_observation().clone();
    let manifest_query_id = observation_handle.query_id().map(|id| QueryId::new(id.as_str()));
    let manifest_trace_id = observation_handle.trace_id().map(|id| TraceId::new(id.as_str()));
    let manifest_is_authoritative = manifest_query_id.is_some();
    let query_id = manifest_query_id.clone().or_else(|| context.query_id().cloned()).unwrap_or_else(QueryId::mint);
    let trace_id = if manifest_is_authoritative { manifest_trace_id } else { context.trace_id().cloned() };
    let observed_context = context
        .with_runtime_observation_handle(observation_handle.clone())
        .and_then(
            |context| {
                if context.query_id().is_none() { context.with_query_id(query_id.clone()) } else { Ok(context) }
            },
        )
        .and_then(|context| match (&trace_id, context.trace_id()) {
            (Some(trace_id), None) => context.with_trace_id(trace_id.clone()),
            _ => Ok(context),
        });
    let plan_execution_id =
        PlanExecutionId::new(format!("native-plan-{}", NEXT_PLAN_EXECUTION_ID.fetch_add(1, Ordering::Relaxed)));
    let lifecycle = context.begin_runtime_execution_observation(plan_execution_id.clone());
    let execution_scope = lifecycle.execution_scope();
    let execution_metrics = ExecutionMetrics::bind(
        execution_scope.observation(),
        plan_execution_id,
        query_id.as_str(),
        trace_id.as_ref().map(TraceId::as_str),
    );
    let execution_scope = execution_scope.with_metrics(execution_metrics.recorder().clone());
    record_query_plan_topology(&execution_scope, plan);
    let result = observed_context
        .and_then(|context| context.with_runtime_execution_scope(execution_scope.clone()))
        .and_then(|observed_context| {
            execute_query_plan_output(
                &observed_context,
                plan,
                bounded_admission,
                gpu_inputs,
                &execution_metrics,
                Some(&execution_scope),
            )
        })
        .map_err(|error| with_optional_trace_id(error.with_query_id(query_id.clone()), trace_id.as_ref()));
    let terminal = match &result {
        Ok(_) => RuntimeTerminalOutcome::Succeeded,
        Err(error) => RuntimeTerminalOutcome::from_query_error(error),
    };
    let snapshot = lifecycle.finish(terminal);
    let mut metrics = query_metrics(context, query_id, trace_id);
    execution_metrics.project_onto(&mut metrics);
    QueryExecutionOutcome::new(result, metrics, snapshot)
}

fn record_query_plan_topology(scope: &RuntimeExecutionObservationScope, plan: &QueryPlan) {
    for node in plan.nodes() {
        let node_id = RuntimePlanNodeId::new(node.id.to_string());
        let operator_kind = plan_node_kind(&node.kind).to_string();
        scope.observation().record_topology(crate::observability::RuntimeObservationRecordClass::PlanNode, || {
            RuntimeTopologyRecord::PlanNode(PlanNodeRow {
                plan_execution_id: scope.plan_execution_id().clone(),
                node_id: node_id.clone(),
                operator_kind,
            })
        });
        for (input_ordinal, input) in node.input_ids().iter().enumerate() {
            scope.observation().record_topology(crate::observability::RuntimeObservationRecordClass::PlanEdge, || {
                RuntimeTopologyRecord::PlanEdge(PlanEdgeRow {
                    plan_execution_id: scope.plan_execution_id().clone(),
                    from_node_id: RuntimePlanNodeId::new(input.to_string()),
                    to_node_id: node_id.clone(),
                    input_ordinal: u32::try_from(input_ordinal).unwrap_or(u32::MAX),
                })
            });
        }
    }
}

fn plan_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
        PlanNode::EdgeNormalize(_) => "edge_normalize",
        PlanNode::GraphAlgorithm(_) => "graph_algorithm",
        PlanNode::VectorAlgorithm(_) => "vector_algorithm",
    }
}

fn query_metrics(context: &ExecutionContext, query_id: QueryId, trace_id: Option<TraceId>) -> QueryMetrics {
    QueryMetrics {
        query_id,
        trace_id,
        memory_control_policy: context.memory_control_policy(),
        execution_boundary_policy: context.boundary_policy(),
        ..Default::default()
    }
}

fn with_optional_trace_id(error: QueryError, trace_id: Option<&TraceId>) -> QueryError {
    match trace_id {
        Some(trace_id) => error.with_trace_id(trace_id.clone()),
        None => error,
    }
}

fn execute_query_plan_output(
    context: &ExecutionContext,
    plan: &QueryPlan,
    bounded_admission: Option<&crate::admission::BoundedPipelineAdmission>,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<Output> {
    context.check_cancellation("nexus_query_engine.exec.execute_query_plan")?;
    validate_query_plan(plan, crate::rapids_backend::rapids_analysis_context())?;
    let context = context.with_admitted_gpu_inputs(plan, gpu_inputs)?;
    // Derive the per-query exact-order scope here so every context clone,
    // including the source-read worker contexts on spawned threads, observes
    // the same scope.
    let datafusion_order_tracking = context.statistical_aggregate_execution_mode().is_datafusion_exact()
        && plan_requires_datafusion_exact_statistical_order(plan);
    let context = context.with_datafusion_order_tracking(datafusion_order_tracking);
    let context = &context;
    context.validate_admitted_plan_capabilities_with_gpu_inputs(plan, gpu_inputs)?;
    let _attempt_execution_range = attempt_execution_range(context);
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, bounded_admission, gpu_inputs, metrics, observation_scope)?
    } else {
        let mut scratch = ExecutionScratch::new(context)?;
        let frame = execute_node(context, plan, plan.sink().input(), gpu_inputs, &mut scratch, metrics)?;
        match plan.sink() {
            ResultSink::GpuResident { .. } => Output::GpuResident(gpu_result_handle(context, plan, frame)?),
            ResultSink::HostArrow { schema, .. } => {
                let executable_schema =
                    plan.node(plan.sink().input()).expect("validated sink input exists").output_schema().clone();
                Output::HostArrow(materialize_gpu_frame_to_host(context, frame, executable_schema, schema, metrics)?)
            },
        }
    };
    Ok(output)
}

/// Materialize a GPU-resident result handle to host Arrow and record boundary metrics.
///
/// The handle outlives its plan execution, so this reports into the caller's
/// detached metrics value rather than an execution's metrics owner.
pub fn materialize_gpu_result_to_host(handle: GpuResultHandle, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = std::time::Instant::now();
    let output = handle.to_host_record_batch()?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

fn gpu_result_handle(context: &ExecutionContext, plan: &QueryPlan, frame: GpuDataFrame) -> Result<GpuResultHandle> {
    GpuResultHandle::new(
        context,
        frame.with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(plan)),
        Arc::clone(plan.output_schema()),
    )
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    scratch: &mut ExecutionScratch,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    context.check_cancellation("nexus_query_engine.exec.execute_node")?;
    if let Some(frame) = scratch.node_results.get(&id) {
        return Ok(frame.clone());
    }

    let node = plan.node(id).ok_or_else(|| invalid_node_reference(id, "query plan references a missing node"))?;

    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, id, source, metrics),
            crate::source::SourceNode::GpuResident(source) => {
                read_gpu_resident_source(context, source, gpu_inputs, metrics)
            },
            crate::source::SourceNode::LiteralBatch(source) => read_literal_batch_source(context, source, metrics),
        },
        PlanNode::Filter(filter) => {
            let input = execute_node(context, plan, filter.input, gpu_inputs, scratch, metrics)?;
            execute_filter(context, filter, input, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_node(context, plan, projection.input, gpu_inputs, scratch, metrics)?;
            execute_projection(context, projection, input, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_node(context, plan, aggregate.input, gpu_inputs, scratch, metrics)?;
            let input_schema = plan
                .node(aggregate.input)
                .ok_or_else(|| invalid_node_reference(aggregate.input, "aggregate references a missing input"))?
                .output_schema();
            execute_aggregate(context, aggregate, input_schema, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Window(window) => {
            let input = execute_node(context, plan, window.input, gpu_inputs, scratch, metrics)?;
            let input_schema = plan
                .node(window.input)
                .ok_or_else(|| invalid_node_reference(window.input, "window references a missing input"))?
                .output_schema();
            execute_window(context, window, input_schema, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_node(context, plan, sort.input, gpu_inputs, scratch, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_node(context, plan, limit.input, gpu_inputs, scratch, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
        PlanNode::Join(join) => {
            let left = execute_node(context, plan, join.inputs[0], gpu_inputs, scratch, metrics)?;
            let right = execute_node(context, plan, join.inputs[1], gpu_inputs, scratch, metrics)?;
            execute_join(context, node.id, join, left, right, scratch, metrics)
        },
        PlanNode::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| execute_node(context, plan, *input, gpu_inputs, scratch, metrics))
                .collect::<Result<Vec<_>>>()?;
            execute_union(context, node.id, inputs, metrics)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let input = execute_node(context, plan, edge_normalize.input, gpu_inputs, scratch, metrics)?;
            execute_edge_normalize(context, node.id, edge_normalize, input, metrics)
        },
        PlanNode::GraphAlgorithm(graph) => {
            #[cfg(not(feature = "cugraph"))]
            return Err(graph_algorithm_feature_disabled(node.id, graph));
            #[cfg(feature = "cugraph")]
            {
                let input = execute_node(context, plan, graph.input, gpu_inputs, scratch, metrics)?;
                let side_inputs = graph
                    .inputs
                    .iter()
                    .skip(1)
                    .map(|input_id| {
                        execute_node(context, plan, *input_id, gpu_inputs, scratch, metrics)
                            .map(|frame| (*input_id, frame))
                    })
                    .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
                let result = execute_graph_algorithm(context, node.id, graph, input, side_inputs, metrics);
                Ok(result.map_err(|error| error.with_fact("node_id", node.id))?)
            }
        },
        PlanNode::VectorAlgorithm(vector) => {
            #[cfg(feature = "cuvs")]
            {
                validate_vector_algorithm_execution(node.id, vector)?;
                let mut inputs = std::collections::BTreeMap::new();
                for input_id in &vector.inputs {
                    if !inputs.contains_key(input_id) {
                        let input = execute_node(context, plan, *input_id, gpu_inputs, scratch, metrics)?;
                        inputs.insert(*input_id, input);
                    }
                }
                execute_vector_algorithm(context, plan, node.id, vector, &inputs)
            }
            #[cfg(not(feature = "cuvs"))]
            {
                execute_vector_algorithm(node.id, vector)
            }
        },
    }?;

    record_device_memory_observation(&frame, metrics)?;
    scratch.node_results.insert(id, frame.clone());
    Ok(frame)
}

#[cfg(feature = "cugraph")]
fn graph_preparation_bypass_input(plan: &QueryPlan, graph: &crate::plan::GraphAlgorithmNode) -> Option<NodeId> {
    if !graph.spec.edge_normalize_inserted {
        return None;
    }
    let PlanNode::EdgeNormalize(normalize) = &plan.node(graph.input)?.kind else {
        return None;
    };
    let raw_input = plan.node(normalize.input)?;
    let source = raw_input.output_schema().field_with_name(&graph.spec.edge_columns.source).ok()?;
    let destination = raw_input.output_schema().field_with_name(&graph.spec.edge_columns.destination).ok()?;
    (!source.is_nullable() && !destination.is_nullable()).then_some(normalize.input)
}

#[cfg(test)]
mod tests {
    use arrow::{array::Int32Array, datatypes::DataType};

    use super::*;
    use crate::{
        observability::{
            RuntimeObservationConfig, RuntimeObservationLossReason, RuntimeObservationSession,
            write_runtime_observation_jsonl,
        },
        plan::QueryPlanBuilder,
        runtime::ExecutionCancellation,
        source::{LiteralBatchSource, SourceNode},
    };

    const SENSITIVE_SOURCE_LOCATION: &str = "s3://alice:secret@private-bucket/orders.parquet?token=top-secret";

    fn literal_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))])
            .expect("literal batch is valid");
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        builder.finish_with_host_sink(source)
    }

    fn unavailable_key_session(query_id: &str) -> RuntimeObservationSession {
        let session = RuntimeObservationSession::with_unavailable_source_location_test_key(
            query_id,
            None,
            RuntimeObservationConfig::bounded(64, 1024 * 1024),
        );
        assert_eq!(session.handle().source_location_id(SENSITIVE_SOURCE_LOCATION), None);
        session
    }

    fn assert_key_unavailable_evidence(outcome: &QueryExecutionOutcome, expected_terminal: RuntimeTerminalOutcome) {
        let snapshot = outcome.observation().expect("key failure still returns sealed evidence");
        assert_eq!(snapshot.footer().terminal, Some(expected_terminal));
        assert!(snapshot.footer().degraded);
        assert!(!snapshot.footer().truncated);
        assert_eq!(
            snapshot
                .footer()
                .dropped_by_reason
                .iter()
                .find(|loss| loss.reason == RuntimeObservationLossReason::SourceLocationKeyUnavailable)
                .map(|loss| loss.count),
            Some(1)
        );

        let mut jsonl = Vec::new();
        write_runtime_observation_jsonl(snapshot, &mut jsonl).expect("degraded evidence exports");
        let jsonl = String::from_utf8(jsonl).expect("JSONL is UTF-8");
        assert!(jsonl.contains("source_location_key_unavailable"));
        assert!(!jsonl.contains(SENSITIVE_SOURCE_LOCATION));
        assert!(!jsonl.contains("source_location:blake3_keyed_256:"));
    }

    #[test]
    fn source_location_key_failure_preserves_successful_execution() {
        let plan = literal_plan();
        let observed = ExecutionContext::builder()
            .with_query_id("key-unavailable-success")
            .with_runtime_observation_session(unavailable_key_session("key-unavailable-success"))
            .build_with_runtime_observation()
            .expect("observed execution context builds");

        let outcome = execute_query_plan(observed.context(), QueryExecutionRequest::new(&plan));

        let output = outcome.result().expect("source-location key failure must not fail the query");
        assert_eq!(output.as_host_arrow().expect("host sink returns Arrow").num_rows(), 1);
        assert_key_unavailable_evidence(&outcome, RuntimeTerminalOutcome::Succeeded);
    }

    #[test]
    fn source_location_key_failure_preserves_cancellation() {
        let plan = literal_plan();
        let cancellation = ExecutionCancellation::new();
        cancellation.cancel();
        let observed = ExecutionContext::builder()
            .with_query_id("key-unavailable-cancelled")
            .with_cancellation(cancellation)
            .with_runtime_observation_session(unavailable_key_session("key-unavailable-cancelled"))
            .build_with_runtime_observation()
            .expect("cancelled observed execution context builds");
        let cancelled = execute_query_plan(observed.context(), QueryExecutionRequest::new(&plan));
        assert!(matches!(cancelled.result(), Err(error) if error.code() == QueryErrorCode::Cancelled));
        assert_key_unavailable_evidence(&cancelled, RuntimeTerminalOutcome::Cancelled);
    }
}
