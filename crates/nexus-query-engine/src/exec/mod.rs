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
    cell::Cell,
    collections::HashMap,
    rc::Rc,
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
mod graph_algorithm;
mod join;
mod lowering;
mod lowering_policy;
mod materialize;
mod operators;
mod order;
/// Native execution output types.
pub mod output;
mod retry;
mod source;
mod util;
mod validate;
mod vector_algorithm;

#[doc(hidden)]
pub use bounded::pipeline::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
    CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
};
#[cfg(test)]
pub(crate) use bounded::pipeline::{ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness};
pub use bounded::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape,
    STRING_OFFSET_BYTES_CONSERVATIVE, join_output_upper_bound_bytes, partial_aggregate_upper_bound_bytes,
};
use edge_normalize::execute_edge_normalize;
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use graph_algorithm::force_next_graph_statistics_optional_resource_exhausted_for_test;
#[cfg(all(test, feature = "cugraph"))]
#[doc(hidden)]
pub(crate) use graph_algorithm::{
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
};
use output::QueryOutput;
pub use source::{
    ResolvedRemoteParquetDatasourceProbe, probe_resolved_remote_parquet_datasource, read_gpu_resident_source,
    read_literal_batch_source, read_local_parquet_source,
};
#[cfg(test)]
pub(crate) mod reservation_contract_testing {
    #![allow(missing_docs)]

    use super::bounded::{JoinMaterializationTxn, ReservationTracker};
    use crate::{
        error::{QueryError, Result},
        metrics::QueryMetrics,
        pipeline::PipelineTaskKind,
        runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
    };

    #[derive(Debug)]
    pub(crate) struct ReservationTrackerContractProbe {
        pub(crate) metrics: QueryMetrics,
        pub(crate) reserved_after_reserve: u64,
        pub(crate) reserved_after_alias_or_transfer: u64,
        pub(crate) reserved_after_intermediate_release: u64,
        pub(crate) reserved_after_final_release: u64,
    }

    #[derive(Debug)]
    pub(crate) struct ReservationTrackerDenialProbe {
        pub(crate) metrics: QueryMetrics,
        pub(crate) error: QueryError,
        pub(crate) reserved_after_denial: u64,
    }

    #[derive(Debug)]
    pub(crate) struct JoinMaterializationTxnRollbackProbe {
        pub(crate) metrics: QueryMetrics,
        pub(crate) reserved_after_rollback: u64,
    }

    pub(crate) fn probe_alias_contract() -> Result<ReservationTrackerContractProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let owned = tracker.try_reserve_owned_resident(
            request(ReservationKind::SourceRead, "source_read", 60, tracker.budget()),
            &mut metrics,
        )?;
        let reserved_after_reserve = tracker.reserved_bytes();
        let alias = tracker.alias_reservation(
            &owned,
            request(ReservationKind::OperatorOutput, "schema_only_projection", 60, tracker.budget()),
            &mut metrics,
        );
        let reserved_after_alias_or_transfer = tracker.reserved_bytes();
        tracker.release(alias, &mut metrics);
        let reserved_after_intermediate_release = tracker.reserved_bytes();
        tracker.release(owned, &mut metrics);
        let reserved_after_final_release = tracker.reserved_bytes();
        Ok(ReservationTrackerContractProbe {
            metrics,
            reserved_after_reserve,
            reserved_after_alias_or_transfer,
            reserved_after_intermediate_release,
            reserved_after_final_release,
        })
    }

    pub(crate) fn probe_transfer_contract() -> Result<ReservationTrackerContractProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let work = tracker.try_reserve_work(
            request(ReservationKind::OperatorOutput, "join_input_projection", 60, tracker.budget()),
            &mut metrics,
        )?;
        let reserved_after_reserve = tracker.reserved_bytes();
        let transferred = tracker.transfer(
            work,
            request(ReservationKind::ProbeSide, "join_probe_side", 60, tracker.budget()),
            &mut metrics,
        );
        let reserved_after_alias_or_transfer = tracker.reserved_bytes();
        let reserved_after_intermediate_release = tracker.reserved_bytes();
        tracker.release(transferred, &mut metrics);
        let reserved_after_final_release = tracker.reserved_bytes();
        Ok(ReservationTrackerContractProbe {
            metrics,
            reserved_after_reserve,
            reserved_after_alias_or_transfer,
            reserved_after_intermediate_release,
            reserved_after_final_release,
        })
    }

    pub(crate) fn probe_alias_outlives_source_release_contract() -> Result<ReservationTrackerContractProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let owned = tracker.try_reserve_owned_resident(
            request(ReservationKind::SourceRead, "join_source_read", 60, tracker.budget()),
            &mut metrics,
        )?;
        let reserved_after_reserve = tracker.reserved_bytes();
        let alias = tracker.alias_reservation(
            &owned,
            request(ReservationKind::ProbeSide, "join_probe_side", 25, tracker.budget()),
            &mut metrics,
        );
        let reserved_after_alias_or_transfer = tracker.reserved_bytes();
        tracker.release(owned, &mut metrics);
        let reserved_after_intermediate_release = tracker.reserved_bytes();
        tracker.release(alias, &mut metrics);
        let reserved_after_final_release = tracker.reserved_bytes();
        Ok(ReservationTrackerContractProbe {
            metrics,
            reserved_after_reserve,
            reserved_after_alias_or_transfer,
            reserved_after_intermediate_release,
            reserved_after_final_release,
        })
    }

    pub(crate) fn probe_denial_capacity_after_alias() -> Result<ReservationTrackerDenialProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let owned = tracker.try_reserve_owned_resident(
            request(ReservationKind::SourceRead, "source_read", 70, tracker.budget()),
            &mut metrics,
        )?;
        let alias = tracker.alias_reservation(
            &owned,
            request(ReservationKind::OperatorOutput, "schema_only_projection", 70, tracker.budget()),
            &mut metrics,
        );
        let error = match tracker.try_reserve_owned_resident(
            request(ReservationKind::ProbeSide, "join_probe_side", 40, tracker.budget()),
            &mut metrics,
        ) {
            Ok(reservation) => {
                tracker.release(reservation, &mut metrics);
                unreachable!("reservation denial probe unexpectedly admitted over-budget request")
            },
            Err(error) => error,
        };
        let reserved_after_denial = tracker.reserved_bytes();
        tracker.release(alias, &mut metrics);
        tracker.release(owned, &mut metrics);
        Ok(ReservationTrackerDenialProbe { metrics, error, reserved_after_denial })
    }

    pub(crate) fn probe_join_materialization_txn_rollback_contract() -> Result<JoinMaterializationTxnRollbackProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let task_id = "t4";
        let operator_kind = PipelineTaskKind::JoinMaterialize.as_str();
        metrics.record_compiled_pipeline_task_execution(task_id, operator_kind);
        let _task_observation_scope = tracker.enter_task_observation_scope(
            crate::observability::task_attempt_id(task_id, 1),
            operator_kind,
            crate::observability::RuntimeSpanId::new(2),
            None::<String>,
        );

        let gather_map = tracker.try_reserve_work(
            request(ReservationKind::GatherMap, "join_one_shot_indices", 30, tracker.budget()),
            &mut metrics,
        )?;
        let output = tracker.try_reserve_owned_resident(
            request(ReservationKind::JoinOutput, "join_output", 40, tracker.budget()),
            &mut metrics,
        )?;
        let mut txn = JoinMaterializationTxn::new();
        txn.set_gather_map_reservation(gather_map);
        txn.set_output_reservation(output);
        txn.rollback(&mut tracker, &mut metrics);
        let reserved_after_rollback = tracker.reserved_bytes();

        Ok(JoinMaterializationTxnRollbackProbe { metrics, reserved_after_rollback })
    }

    fn request(
        kind: ReservationKind,
        memory_boundary: &'static str,
        requested_bytes: u64,
        budget_bytes: u64,
    ) -> MemoryReservationRequest {
        MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
            .with_memory_boundary(memory_boundary)
            .with_stage("reservation_contract_probe")
    }
}

use self::{
    bounded::*,
    graph_algorithm::*,
    join::*,
    lowering::*,
    materialize::*,
    operators::*,
    order::*,
    output::{QueryExecutionOutcome, QueryExecutionResult as ExecutionResult, QueryOutput as Output},
    util::*,
    validate::*,
    vector_algorithm::*,
};
use crate::{
    chunk::{
        SourceChunk, SourceChunkBatch, SourceChunkPlan, SourceChunkPlanningPolicy, SourceTaskMemoryModel,
        annotate_source_chunk_batching_reasons, plan_local_parquet_source_chunks_with_execution_context,
        plan_source_chunk_batches,
    },
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_implicit_cast, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, Expr, LimitSpec, LiteralValue, SortSpec,
    },
    metrics::{QueryId, QueryMetrics, SourceReadRuntimeMetrics, TraceId},
    observability::{
        PlanEdgeRow, PlanExecutionId, PlanNodeRow, RuntimeExecutionObservationScope, RuntimeObservationConfig,
        RuntimeObservationOwner, RuntimeObservationSession, RuntimePlanExecutionLifecycle, RuntimePlanNodeId,
        RuntimeQueryLifecycle, RuntimeTerminalOutcome, RuntimeTopologyRecord, RuntimeTraceId,
    },
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, LimitNode, NodeId,
        PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode, WindowFrame, WindowNode,
    },
    runtime::{
        BoundedJoinReservationContract, ExecutionContext, MemoryControlPolicy, MemoryReservationRequest,
        ReservationKind, RetryFacts,
        resident::{
            GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildHandle, GpuResidentInputMap, GpuResultHandle,
        },
    },
    source::{
        GpuResidentSource, LocalParquetFile, LocalParquetSource, ZERO_COLUMN_COUNT_COLUMN,
        validate_local_parquet_source,
    },
};

/// Validates a complete native plan without constructing a runtime or touching
/// GPU resources.
///
/// Frontends use this capability-first boundary before host work or native
/// admission. Execution entrypoints call the same validation internally.
pub fn preflight_query_plan(plan: &QueryPlan) -> Result<()> {
    validate::validate_query_plan(plan)
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

/// Execute a native plan with no external GPU-resident inputs.
///
/// The plan is validated before execution. Output residency follows
/// [`ResultSink`]: host sinks materialize Arrow output, while GPU-resident
/// sinks return a runtime/device-affine handle.
pub fn execute_query_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<ExecutionResult> {
    execute_query_plan_with_gpu_inputs(context, plan, &GpuResidentInputMap::new())
}

/// Execute a native plan with caller-supplied GPU-resident inputs.
///
/// `gpu_inputs` is keyed by the names referenced by GPU-resident source nodes
/// in `plan`. Bounded plans dispatch through the context's bounded pipeline
/// contract.
pub fn execute_query_plan_with_gpu_inputs(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ExecutionResult> {
    let query_id = context.query_id().cloned().unwrap_or_else(QueryId::mint);
    let trace_id = context.trace_id().cloned();

    execute_query_plan_with_correlation(context, plan, gpu_inputs, query_id.clone(), trace_id.clone())
        .map_err(|error| error.with_query_id(query_id).with_optional_trace_id(trace_id.as_ref()))
}

/// Executes a native plan while retaining metrics and sealed observations on
/// both success and structured failure.
pub fn execute_query_plan_observed(
    context: &ExecutionContext,
    plan: &QueryPlan,
    config: RuntimeObservationConfig,
) -> QueryExecutionOutcome {
    execute_query_plan_with_gpu_inputs_observed(context, plan, &GpuResidentInputMap::new(), config)
}

/// Executes a native plan with GPU-resident inputs while retaining metrics and
/// sealed observations on every structured outcome.
pub fn execute_query_plan_with_gpu_inputs_observed(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    config: RuntimeObservationConfig,
) -> QueryExecutionOutcome {
    let query_id = context.query_id().cloned().unwrap_or_else(QueryId::mint);
    let trace_id = context.trace_id().cloned();
    let observation = match context.runtime_observation_owner() {
        Some(owner) => ExecutionObservationAuthority::PlanOwner(owner),
        None if context.runtime_observation().is_enabled() => {
            ExecutionObservationAuthority::PlanHandle(context.runtime_observation().clone())
        },
        None => ExecutionObservationAuthority::QueryOwner(RuntimeObservationOwner::new(
            query_id.clone(),
            trace_id.as_ref().map(|trace_id| RuntimeTraceId::new(trace_id.to_string())),
            config,
        )),
    };
    execute_query_plan_with_gpu_inputs_and_runtime_observation_authority(context, plan, gpu_inputs, observation)
}

/// Executes using a caller-created owner. This injection point supports
/// deterministic clocks while the lifecycle still consumes seal authority.
pub fn execute_query_plan_with_runtime_observation(
    context: &ExecutionContext,
    plan: &QueryPlan,
    observation: RuntimeObservationSession,
) -> QueryExecutionOutcome {
    execute_query_plan_with_gpu_inputs_and_runtime_observation(context, plan, &GpuResidentInputMap::new(), observation)
}

/// Evidence-preserving execution with explicit GPU inputs and owner lifetime.
pub fn execute_query_plan_with_gpu_inputs_and_runtime_observation(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    observation: RuntimeObservationSession,
) -> QueryExecutionOutcome {
    execute_query_plan_with_gpu_inputs_and_runtime_observation_authority(
        context,
        plan,
        gpu_inputs,
        ExecutionObservationAuthority::QueryOwner(RuntimeObservationOwner::from_session(observation)),
    )
}

enum ExecutionObservationAuthority {
    QueryOwner(RuntimeObservationOwner),
    PlanOwner(RuntimeObservationOwner),
    PlanHandle(crate::observability::RuntimeObservationHandle),
}

impl ExecutionObservationAuthority {
    fn handle(&self) -> crate::observability::RuntimeObservationHandle {
        match self {
            Self::QueryOwner(owner) | Self::PlanOwner(owner) => owner.handle(),
            Self::PlanHandle(handle) => handle.clone(),
        }
    }

    fn owner(&self) -> Option<RuntimeObservationOwner> {
        match self {
            Self::QueryOwner(owner) | Self::PlanOwner(owner) => Some(owner.clone()),
            Self::PlanHandle(_) => None,
        }
    }
}

enum ExecutionObservationLifecycle {
    Query(RuntimeQueryLifecycle),
    Plan(RuntimePlanExecutionLifecycle),
}

impl ExecutionObservationLifecycle {
    fn execution_scope(&self) -> RuntimeExecutionObservationScope {
        match self {
            Self::Query(lifecycle) => lifecycle.execution_scope(),
            Self::Plan(lifecycle) => lifecycle.execution_scope(),
        }
    }

    fn finish(self, outcome: RuntimeTerminalOutcome) -> Option<crate::observability::SealedRuntimeObservationSnapshot> {
        match self {
            Self::Query(lifecycle) => lifecycle.finish(outcome),
            Self::Plan(lifecycle) => {
                lifecycle.finish(outcome);
                None
            },
        }
    }
}

fn execute_query_plan_with_gpu_inputs_and_runtime_observation_authority(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    observation: ExecutionObservationAuthority,
) -> QueryExecutionOutcome {
    let observation_handle = observation.handle();
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
    let lifecycle = match &observation {
        ExecutionObservationAuthority::QueryOwner(owner) => {
            ExecutionObservationLifecycle::Query(RuntimeQueryLifecycle::from_owner(owner.clone(), plan_execution_id))
        },
        ExecutionObservationAuthority::PlanOwner(owner) => ExecutionObservationLifecycle::Plan(
            RuntimePlanExecutionLifecycle::from_owner(owner.clone(), plan_execution_id),
        ),
        ExecutionObservationAuthority::PlanHandle(handle) => ExecutionObservationLifecycle::Plan(
            RuntimePlanExecutionLifecycle::from_handle(handle.clone(), plan_execution_id),
        ),
    };
    let execution_scope = lifecycle.execution_scope();
    record_query_plan_topology(&execution_scope, plan);
    let mut metrics = query_metrics(context, query_id.clone(), trace_id.clone());
    let observation_owner = observation.owner();
    let result = observed_context
        .and_then(|context| match observation_owner {
            Some(owner) => context.with_runtime_observation_owner(owner),
            None => Ok(context),
        })
        .and_then(|context| context.with_runtime_execution_scope(execution_scope.clone()))
        .and_then(|observed_context| {
            execute_query_plan_output(&observed_context, plan, gpu_inputs, &mut metrics, Some(&execution_scope))
        })
        .map_err(|error| error.with_query_id(query_id).with_optional_trace_id(trace_id.as_ref()));
    let terminal = match &result {
        Ok(_) => RuntimeTerminalOutcome::Succeeded,
        Err(error) => RuntimeTerminalOutcome::from_query_error(error),
    };
    let snapshot = lifecycle.finish(terminal);
    metrics.finalize_reservation_projection();
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

fn execute_query_plan_with_correlation(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    query_id: QueryId,
    trace_id: Option<TraceId>,
) -> Result<ExecutionResult> {
    let mut metrics = query_metrics(context, query_id, trace_id);
    let output = execute_query_plan_output(context, plan, gpu_inputs, &mut metrics, None)?;
    metrics.finalize_reservation_projection();
    Ok(ExecutionResult::new(output, metrics))
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

fn execute_query_plan_output(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<Output> {
    context.check_cancellation("nexus_query_engine.exec.execute_query_plan")?;
    validate_query_plan(plan)?;
    context.validate_admitted_plan_capabilities(plan)?;
    let _order_tracking = DataFusionOrderTrackingGuard::new(
        context.stddev_samp_execution_mode().is_datafusion_exact()
            && plan_requires_datafusion_exact_statistical_order(plan),
    );
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, gpu_inputs, metrics, observation_scope)?
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

trait QueryErrorCorrelationExt {
    fn with_optional_trace_id(self, trace_id: Option<&TraceId>) -> Self;
}

impl QueryErrorCorrelationExt for QueryError {
    fn with_optional_trace_id(self, trace_id: Option<&TraceId>) -> Self {
        match trace_id {
            Some(trace_id) => self.with_trace_id(trace_id.clone()),
            None => self,
        }
    }
}

/// Materialize a GPU-resident result handle to host Arrow and record boundary metrics.
pub fn materialize_gpu_result_to_host(handle: GpuResultHandle, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    materialize_to_host(handle, metrics)
}

fn gpu_result_handle(context: &ExecutionContext, plan: &QueryPlan, frame: GpuDataFrame) -> Result<GpuResultHandle> {
    GpuResultHandle::new(context, frame, Arc::clone(plan.output_schema()))
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
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
        observability::{RuntimeObservationLossReason, write_runtime_observation_jsonl},
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
        let context = ExecutionContext::builder()
            .with_query_id("key-unavailable-success")
            .build()
            .expect("execution context builds");

        let outcome = execute_query_plan_with_runtime_observation(
            &context,
            &plan,
            unavailable_key_session("key-unavailable-success"),
        );

        let output = outcome.result().expect("source-location key failure must not fail the query");
        assert_eq!(output.as_host_arrow().expect("host sink returns Arrow").num_rows(), 1);
        assert_key_unavailable_evidence(&outcome, RuntimeTerminalOutcome::Succeeded);
    }

    #[test]
    fn source_location_key_failure_preserves_failure_and_cancellation() {
        let plan = literal_plan();
        let mismatched_context =
            ExecutionContext::builder().with_query_id("different-query").build().expect("execution context builds");
        let failed = execute_query_plan_with_runtime_observation(
            &mismatched_context,
            &plan,
            unavailable_key_session("key-unavailable-failure"),
        );
        let failure = match failed.result() {
            Err(error) => error,
            Ok(_) => panic!("query identity mismatch remains a failure"),
        };
        assert_eq!(failure.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(failure.fact_value("reason"), Some("runtime_observation_query_id_mismatch"));
        assert_key_unavailable_evidence(&failed, RuntimeTerminalOutcome::from_query_error(&failure));

        let cancellation = ExecutionCancellation::new();
        cancellation.cancel();
        let cancelled_context = ExecutionContext::builder()
            .with_query_id("key-unavailable-cancelled")
            .with_cancellation(cancellation)
            .build()
            .expect("cancelled execution context builds");
        let cancelled = execute_query_plan_with_runtime_observation(
            &cancelled_context,
            &plan,
            unavailable_key_session("key-unavailable-cancelled"),
        );
        assert!(matches!(cancelled.result(), Err(error) if error.code() == QueryErrorCode::Cancelled));
        assert_key_unavailable_evidence(&cancelled, RuntimeTerminalOutcome::Cancelled);
    }
}
