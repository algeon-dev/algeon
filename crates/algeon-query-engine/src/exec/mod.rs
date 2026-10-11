//! Native query execution.
//!
//! Entrypoints validate local source contracts, derive output mode from the
//! plan sink, and dispatch according to the bounded execution contract in the
//! [`ExecutionContext`].
//! GPU-resident input lookup is by plan-local names supplied in
//! [`GpuResidentInputMap`]. Metrics are owned by the returned
//! [`output::QueryExecutionOutcome`] so callers can record them after choosing host or
//! GPU-resident output handling.
//!
//! [`execute_query_plan`] is synchronous and runs GPU work on the calling
//! thread. The query service calls it from inside the attempt's root
//! `QueryExecution` job on the device's GPU execution pool
//! (`website/docs/design/index.mdx#runtimes-and-threads`); a standalone
//! context calls it from wherever the caller chooses to block.

use std::{
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
    AggSpec as CudfAggSpec, BudgetedExecutionOutcome, DataFrame, DuplicateKeepOption, Exec, ExecutionBudgetRequest,
    ExecutionBudgetToken, GroupByAgg, GroupByOptions, ReduceAgg, ShiftSpec as CudfShiftSpec,
    SortOptions as CudfSortOptions,
    core::Scalar as CudfScalar,
    frame::dataframe::{DropDuplicatesOptions, InfoDetail, ParquetReadOptions, SourceRead},
    io::source::SourceReadMode,
    ops::{col, lit, lit_null, reduction::ScanAgg, when},
};

mod bounded;
#[cfg(any(feature = "cuvs", feature = "cuml"))]
mod dense_relation;
mod edge_normalize;
mod edge_null_filter;
mod execution_metrics;
mod graph_algorithm;
mod join;
mod materialize;
mod ml_algorithm;
mod nvtx;
mod operators;
mod order;
/// Native execution output types.
pub mod output;
mod retry;
mod source;
mod util;
mod vector_algorithm;

use cudf::interop::arrow_host::cudf_dtype_from_arrow_type;
use edge_normalize::execute_edge_normalize;
pub(crate) use execution_metrics::ExecutionMetrics;
#[cfg(not(feature = "cuml"))]
use ml_algorithm::ml_algorithm_execution_unavailable;
#[cfg(all(feature = "cuml", feature = "test-hooks"))]
#[doc(hidden)]
pub use ml_algorithm::{set_forest_after_import_hook_for_tests, set_forest_batch_element_limit_for_tests};
use nvtx::attempt_execution_range;
use output::QueryOutput;
#[cfg(any(not(feature = "cuml"), not(feature = "cuvs")))]
use query_planning::plan::PlanNode;
use query_planning::{
    admission::AdmittedPlan,
    chunk::{SourceChunkPlanningPolicy, SourceTaskMemoryModel},
    correlation::{QueryId, TraceId},
    datatypes::is_supported_arrow_implicit_cast,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::MemoryControlPolicy,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, Expr, LimitSpec, LiteralValue},
    mode::StatisticalAggregateExecutionMode,
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinOutputColumn, JoinSide, JoinType, LimitNode, NodeId, ProjectionNode,
        QueryPlan, SortNode, WindowFrame, WindowNode,
    },
    policy::PlanningPolicy,
    source::{
        GpuResidentSource, LocalParquetFile, LocalParquetSource, ZERO_COLUMN_COUNT_COLUMN,
        validate_local_parquet_source,
    },
    validate_query_plan,
    validation::*,
};
#[cfg(any(test, feature = "test-hooks"))]
use query_runtime::metrics::QueryMetrics;
use query_runtime::{
    metrics::SourceReadRuntimeMetrics,
    native_error::cudf_query_error,
    observability::{
        PlanEdgeRow, PlanExecutionId, PlanNodeRow, RuntimeExecutionObservationScope, RuntimeObservationRecordClass,
        RuntimePlanNodeId, RuntimeTerminalOutcome, RuntimeTopologyRecord,
    },
    runtime::{
        ExecutionContext, RetryFacts,
        resident::{
            GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildHandle, GpuResidentInputMap, GpuResultHandle,
        },
    },
};
#[doc(hidden)]
pub use source::sanitize_cache_component;
use source::{read_gpu_resident_source, read_literal_batch_source};
#[cfg(all(feature = "cuvs", feature = "test-hooks"))]
#[doc(hidden)]
pub use vector_algorithm::{set_vector_after_index_build_hook_for_tests, set_vector_pool_coverage_hook_for_tests};

#[cfg(feature = "cugraph")]
use self::graph_algorithm::*;
use self::{
    bounded::*,
    join::*,
    materialize::*,
    operators::*,
    order::*,
    output::{PlanMetrics, QueryExecutionOutcome, QueryOutput as Output},
    util::*,
    vector_algorithm::*,
};
#[cfg(feature = "cuvs")]
use crate::native_error::{cudf_interop_query_error, cuvs_query_error};
use crate::{
    chunk::{
        SourceChunk, SourceChunkBatch, SourceChunkPlan, annotate_source_chunk_batching_reasons,
        plan_local_parquet_source_chunks_with_execution_context, plan_source_chunk_batches,
        plan_source_chunk_batches_with_row_cap,
    },
    datatypes::timestamp_unit_to_cudf_resolution,
    native_error::{
        cudf_query_error_without_source, cudf_unsupported_type_error, memory_failure, with_attempt_domain_memory,
    },
    prepared::*,
    rapids_backend::rapids_analysis_context,
};

/// Validates a complete native plan without constructing a runtime or touching
/// GPU resources.
///
/// [`admit_query_plan`] and planning-time admission validate the plan they
/// decide; frontends call this only for a plan they check before it can be
/// admitted, such as the schema plan of a GPU relation function.
pub fn preflight_query_plan(plan: &QueryPlan) -> Result<()> {
    validate_query_plan(plan, rapids_analysis_context(PlanningPolicy::default()))
}

/// Admits the executable plan once and hands the artifact to the caller.
///
/// Callers that build a plan where it executes (graph, vector and ML relation
/// plans, tools, tests) admit it here; execution takes only the artifact. A
/// malformed plan, or one using a construct the backend does not support, is
/// the validator's error; a plan no compiled contract covers is a typed error
/// carrying the first contract it misses and the caller-visible sink as a
/// diagnostic fact.
pub fn admit_query_plan(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
) -> Result<Arc<AdmittedPlan>> {
    bounded::admit_or_execution_error(plan, statistical_aggregate_execution_mode, policy)
}

/// Process-wide counter behind plan execution ids. Ids are unique within a
/// process and restart at 1 with it; observation consumers scope them by
/// query id and do not compare them across processes.
static NEXT_PLAN_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

/// Inputs for one native plan execution.
///
/// Execution uses the supplied admission artifact without repeating admission.
/// Callers obtain that artifact from planning or [`admit_query_plan`].
pub struct QueryExecutionRequest<'a> {
    admitted: &'a AdmittedPlan,
    gpu_inputs: Option<&'a GpuResidentInputMap>,
    owned_gpu_inputs: Option<GpuResidentInputMap>,
}

impl<'a> QueryExecutionRequest<'a> {
    /// Creates a request that executes an admitted plan without external
    /// GPU-resident inputs.
    pub const fn admitted(admitted: &'a AdmittedPlan) -> Self {
        Self { admitted, gpu_inputs: None, owned_gpu_inputs: None }
    }

    /// Supplies GPU-resident inputs keyed by the source names in the plan.
    pub const fn with_gpu_inputs(mut self, gpu_inputs: &'a GpuResidentInputMap) -> Self {
        self.gpu_inputs = Some(gpu_inputs);
        self
    }

    /// Transfers GPU-resident inputs to execution. Each handle is released
    /// after its final scheduled source read; downstream frames retain their
    /// own aliases and reader lifetimes. Other caller-held handle clones remain
    /// reusable.
    pub fn with_owned_gpu_inputs(mut self, gpu_inputs: GpuResidentInputMap) -> Self {
        self.gpu_inputs = None;
        self.owned_gpu_inputs = Some(gpu_inputs);
        self
    }

    fn into_parts(self) -> (&'a AdmittedPlan, GpuResidentInputMap) {
        let gpu_inputs = match self.gpu_inputs {
            Some(inputs) => inputs.clone(),
            None => self.owned_gpu_inputs.unwrap_or_default(),
        };
        (self.admitted, gpu_inputs)
    }
}

/// Executes one native plan and retains metrics and observation evidence on
/// both success and structured failure.
///
/// A query-service context finishes only this plan execution. A standalone
/// context that owns its runtime observation owns query terminalization and
/// returns its sealed snapshot in the outcome.
pub fn execute_query_plan(context: &ExecutionContext, request: QueryExecutionRequest<'_>) -> QueryExecutionOutcome {
    let (admitted, gpu_inputs) = request.into_parts();
    execute_query_plan_with_observation(context, admitted, gpu_inputs)
}

/// Wraps execution in the plan's observation lifecycle.
///
/// The lifecycle begins before any context derivation can fail and finishes
/// on every path, so a plan that never reached its first task still seals a
/// snapshot with a terminal outcome. Errors from the derivation and from
/// execution alike pick up the attempt's memory facts and query identity.
fn execute_query_plan_with_observation(
    context: &ExecutionContext,
    admitted: &AdmittedPlan,
    gpu_inputs: GpuResidentInputMap,
) -> QueryExecutionOutcome {
    let plan = admitted.plan();
    let (query_id, trace_id) = plan_identity(context);
    let observed_context = context
        .with_runtime_observation_handle(context.runtime_observation().clone())
        .and_then(
            |context| {
                if context.query_id().is_none() { context.with_query_id(query_id.clone()) } else { Ok(context) }
            },
        )
        .and_then(|context| match (&trace_id, context.trace_id()) {
            (Some(trace_id), None) => context.with_trace_id(trace_id.clone()),
            _ => Ok(context),
        });
    let plan_execution_id = next_plan_execution_id("native-plan");
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
                admitted,
                gpu_inputs,
                &execution_metrics,
                Some(&execution_scope),
            )
        })
        .map_err(|error| with_attempt_domain_memory(context, error))
        .map_err(|error| with_optional_trace_id(error.with_query_id(query_id.clone()), trace_id.as_ref()));
    // The mark is attempt-wide and monotone, so the last plan to end carries
    // the attempt's peak, failed plans included.
    if let Ok(Some(usage)) = context.attempt_allocation_domain_usage() {
        execution_metrics.record_allocator_high_water_bytes(usage.peak_bytes);
    }
    let terminal = match &result {
        Ok(_) => RuntimeTerminalOutcome::Succeeded,
        Err(error) => RuntimeTerminalOutcome::from_query_error(error),
    };
    let snapshot = lifecycle.finish(terminal);
    let metrics = PlanMetrics::new(execution_metrics.recorder().clone(), query_id, trace_id);
    QueryExecutionOutcome::new(result, metrics, snapshot)
}

/// The query and trace identity a plan execution under `context` reports.
///
/// A query-bound observation's manifest is authoritative; otherwise the
/// context's own identity applies, and a query without one gets a fresh id.
fn plan_identity(context: &ExecutionContext) -> (QueryId, Option<TraceId>) {
    let observation = context.runtime_observation();
    match observation.query_id() {
        Some(query_id) => {
            (QueryId::new(query_id.as_str()), observation.trace_id().map(|trace_id| TraceId::new(trace_id.as_str())))
        },
        None => (context.query_id().cloned().unwrap_or_else(QueryId::mint), context.trace_id().cloned()),
    }
}

fn next_plan_execution_id(label: &str) -> PlanExecutionId {
    PlanExecutionId::new(format!("{label}-{}", NEXT_PLAN_EXECUTION_ID.fetch_add(1, Ordering::Relaxed)))
}

fn record_query_plan_topology(scope: &RuntimeExecutionObservationScope, plan: &QueryPlan) {
    for node in plan.nodes() {
        let node_id = RuntimePlanNodeId::new(node.id.to_string());
        let operator_kind = plan_node_kind(&node.kind).to_string();
        scope.observation().record_topology(RuntimeObservationRecordClass::PlanNode, || {
            RuntimeTopologyRecord::PlanNode(PlanNodeRow {
                plan_execution_id: scope.plan_execution_id().clone(),
                node_id: node_id.clone(),
                operator_kind,
            })
        });
        for (input_ordinal, input) in node.input_ids().iter().enumerate() {
            scope.observation().record_topology(RuntimeObservationRecordClass::PlanEdge, || {
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

fn with_optional_trace_id(error: QueryError, trace_id: Option<&TraceId>) -> QueryError {
    match trace_id {
        Some(trace_id) => error.with_trace_id(trace_id.clone()),
        None => error,
    }
}

/// Derives the executing context and runs the admitted plan.
///
/// Checks that cannot depend on GPU state run first, in this order: the
/// mode the artifact was admitted under, feature availability for ML and
/// vector operators, then the GPU-input bindings. Only then does the context
/// gain its inputs and order-tracking scope and pass control to bounded
/// execution.
fn execute_query_plan_output(
    context: &ExecutionContext,
    admitted: &AdmittedPlan,
    gpu_inputs: GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<Output> {
    context.check_cancellation("query_engine.exec.execute_query_plan")?;
    let plan = admitted.plan();
    // Operators read the mode from the context; the artifact was admitted
    // under its own copy, and the two must agree.
    if context.statistical_aggregate_execution_mode() != admitted.statistical_aggregate_execution_mode() {
        return Err(QueryError::new(
            QueryErrorCode::InternalInvariantViolation,
            "query_engine.exec.execute_query_plan",
            "the plan was admitted under a different statistical aggregate execution mode than the executing context",
        )
        .with_fact(
            "admitted_statistical_aggregate_execution_mode",
            admitted.statistical_aggregate_execution_mode().as_str(),
        )
        .with_fact(
            "context_statistical_aggregate_execution_mode",
            context.statistical_aggregate_execution_mode().as_str(),
        ));
    }
    // A build without the backend rejects before any source work.
    #[cfg(not(feature = "cuml"))]
    if plan.contains_ml_algorithm()
        && let Some(node) = plan.nodes().iter().find(|node| matches!(node.kind, PlanNode::MlAlgorithm(_)))
        && let PlanNode::MlAlgorithm(ml) = &node.kind
    {
        return Err(ml_algorithm_execution_unavailable(node.id, ml));
    }
    #[cfg(not(feature = "cuvs"))]
    if plan.contains_vector_algorithm()
        && let Some(node) = plan.nodes().iter().find(|node| matches!(node.kind, PlanNode::VectorAlgorithm(_)))
        && let PlanNode::VectorAlgorithm(vector) = &node.kind
    {
        return Err(vector_algorithm_execution_unavailable(node.id, vector));
    }
    let context = context.with_admitted_gpu_inputs(plan, &gpu_inputs)?;
    // Derive the per-query exact-order scope here so every context clone,
    // including the source-read worker contexts on spawned threads, observes
    // the same scope.
    let datafusion_order_tracking = context.statistical_aggregate_execution_mode().is_datafusion_exact()
        && plan_requires_datafusion_exact_statistical_order(plan);
    let context = context.with_datafusion_order_tracking(datafusion_order_tracking);
    let context = &context;
    context.validate_admitted_plan_capabilities_with_gpu_inputs(plan, &gpu_inputs)?;
    let _attempt_execution_range = attempt_execution_range(context);
    execute_bounded_query_plan(context, admitted, gpu_inputs, metrics, observation_scope)
}

/// Materialize a GPU-resident result handle to host Arrow and record boundary metrics.
///
/// The handle outlives its plan execution, so this reports into the caller's
/// detached metrics value rather than an execution's metrics owner.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn materialize_gpu_result_to_host(handle: GpuResultHandle, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = handle.to_host_record_batch()?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use arrow::{array::Int32Array, datatypes::DataType};
    use query_planning::{
        plan::QueryPlanBuilder,
        source::{LiteralBatchSource, SourceNode},
    };
    use query_runtime::{
        observability::{
            RuntimeObservationConfig, RuntimeObservationLossReason, RuntimeObservationSession,
            write_runtime_observation_jsonl,
        },
        runtime::{ExecutionCancellation, RuntimeObservedExecutionContext, StandaloneConfig},
    };

    use super::*;

    const SENSITIVE_SOURCE_LOCATION: &str = "s3://alice:secret@private-bucket/orders.parquet?token=top-secret";

    fn literal_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))])
            .expect("literal batch is valid");
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        builder.finish_with_host_sink(source).unwrap()
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
        let observed = RuntimeObservedExecutionContext::standalone(
            StandaloneConfig { query_id: Some("key-unavailable-success".into()), ..StandaloneConfig::default() },
            unavailable_key_session("key-unavailable-success"),
        )
        .expect("observed execution context builds");

        let admitted = admit_query_plan(
            &plan,
            observed.context().statistical_aggregate_execution_mode(),
            observed.context().planning_policy(),
        )
        .expect("literal plan admits");

        let outcome = execute_query_plan(observed.context(), QueryExecutionRequest::admitted(&admitted));

        let output = outcome.result().expect("source-location key failure must not fail the query");
        assert_eq!(output.as_host_arrow().expect("host sink returns Arrow").num_rows(), 1);
        assert_key_unavailable_evidence(&outcome, RuntimeTerminalOutcome::Succeeded);
    }

    #[test]
    fn source_location_key_failure_preserves_cancellation() {
        let plan = literal_plan();
        let cancellation = ExecutionCancellation::new();
        cancellation.cancel();
        let observed = RuntimeObservedExecutionContext::standalone(
            StandaloneConfig {
                query_id: Some("key-unavailable-cancelled".into()),
                cancellation,
                ..StandaloneConfig::default()
            },
            unavailable_key_session("key-unavailable-cancelled"),
        )
        .expect("cancelled observed execution context builds");
        let admitted = admit_query_plan(
            &plan,
            observed.context().statistical_aggregate_execution_mode(),
            observed.context().planning_policy(),
        )
        .expect("literal plan admits");
        let cancelled = execute_query_plan(observed.context(), QueryExecutionRequest::admitted(&admitted));
        assert!(matches!(cancelled.result(), Err(error) if error.code() == QueryErrorCode::Cancelled));
        assert_key_unavailable_evidence(&cancelled, RuntimeTerminalOutcome::Cancelled);
    }
}
