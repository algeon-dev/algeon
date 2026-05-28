//! Native query execution.
//!
//! Entrypoints validate local source contracts, derive output mode from the
//! plan sink, and dispatch to unbounded, strict bounded, or adaptive bounded
//! execution according to the [`ExecutionContext`].
//! GPU-resident input lookup is by plan-local names supplied in
//! [`GpuResidentInputMap`]. Metrics are owned by the returned
//! [`QueryExecutionResult`] so callers can record them after choosing host or
//! GPU-resident output handling.

#![deny(missing_docs)]

use std::{cell::Cell, collections::HashMap, rc::Rc, sync::Arc, time::Instant};

use arrow::{
    array::Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec as CudfAggSpec, BudgetedExecutionOutcome, DataFrame, DuplicateKeepOption, ExecutionBudgetEnforcement,
    ExecutionBudgetRequest, ExecutionBudgetToken, ExecutionOverrides, GroupByAgg, GroupByOptions, ReduceAgg,
    ShiftSpec as CudfShiftSpec, SortOptions as CudfSortOptions,
    core::Scalar as CudfScalar,
    frame::dataframe::{DropDuplicatesOptions, InfoDetail, ParquetReadOptions},
    io::source::SourceReadMode,
    ops::{col, lit, lit_null, reduction::ScanAgg, when},
};

mod bounded;
mod edge_normalize;
mod graph_algorithm;
mod join;
mod lowering;
mod materialize;
mod operators;
mod order;
mod source;
mod types;
mod util;
mod validate;

#[doc(hidden)]
pub use bounded::strict::{
    CompiledBoundedPipelineSupport, CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler,
    CompiledPipelineState, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    ResidentFrameStreamProducerTestHarness, StrictBoundedPipelineAdmission, YieldedFrameCounterTestHarness,
    compiled_bounded_pipeline_support, compiled_bounded_query_plan_support, strict_bounded_pipeline_admission,
};
pub use bounded::{
    AdaptivePipelineAdmission, JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES,
    PartialAggregateShape, STRING_OFFSET_BYTES_CONSERVATIVE, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history, join_output_upper_bound_bytes,
    partial_aggregate_upper_bound_bytes,
};
use edge_normalize::execute_edge_normalize;
pub use source::{
    read_gpu_resident_source, read_literal_batch_source, read_local_parquet_source, validate_local_parquet_source,
};
pub use types::{
    GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildCache, GpuHashJoinBuildHandle,
    GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap,
    GpuResultHandle, QueryExecutionResult, QueryOutput, QueryOutputMode,
};

#[doc(hidden)]
pub mod reservation_contract_testing {
    #![allow(missing_docs)]

    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::{bounded::ReservationTracker, types::GpuLocalParquetSourceCacheKey};
    use crate::{
        error::{QueryError, Result},
        metrics::QueryMetrics,
        runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
        source::LocalParquetSource,
    };

    #[derive(Debug)]
    pub struct ReservationTrackerContractProbe {
        pub metrics: QueryMetrics,
        pub reserved_after_reserve: u64,
        pub reserved_after_alias_or_transfer: u64,
        pub reserved_after_intermediate_release: u64,
        pub reserved_after_final_release: u64,
    }

    #[derive(Debug)]
    pub struct ReservationTrackerDenialProbe {
        pub metrics: QueryMetrics,
        pub error: QueryError,
        pub reserved_after_denial: u64,
    }

    #[derive(Debug)]
    pub struct SourceCacheAliasRelabelProbe {
        pub metrics: QueryMetrics,
        pub reserved_after_cache_insert: u64,
        pub reserved_after_borrow: u64,
        pub reserved_after_relabel: u64,
        pub reserved_after_alias_release: u64,
        pub reserved_after_cache_release: u64,
        pub protected_after_borrow: usize,
        pub protected_after_relabel: usize,
        pub protected_after_alias_release: usize,
    }

    pub fn probe_alias_contract() -> Result<ReservationTrackerContractProbe> {
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

    pub fn probe_transfer_contract() -> Result<ReservationTrackerContractProbe> {
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

    pub fn probe_alias_outlives_source_release_contract() -> Result<ReservationTrackerContractProbe> {
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

    pub fn probe_denial_capacity_after_alias() -> Result<ReservationTrackerDenialProbe> {
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

    pub fn probe_source_cache_alias_relabel_contract() -> Result<SourceCacheAliasRelabelProbe> {
        let mut metrics = QueryMetrics::default();
        let mut tracker = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let key = source_cache_key();
        let mut source = tracker.try_reserve_owned_resident(
            request(ReservationKind::SourceRead, "join_source_read", 60, tracker.budget()),
            &mut metrics,
        )?;
        tracker.promote_source_cache_insert(
            &mut source,
            60,
            request(ReservationKind::SourceCacheRetained, "source_cache_retained", 60, tracker.budget()),
            &mut metrics,
            &key,
        )?;
        let reserved_after_cache_insert = tracker.reserved_bytes();
        tracker.release(source, &mut metrics);

        let borrowed = tracker.borrow_accounted_source_cache(key.clone(), 60, &mut metrics);
        let reserved_after_borrow = tracker.reserved_bytes();
        let protected_after_borrow = tracker.source_cache_protected_key_count(&key);
        let relabeled = tracker.relabel_alias_reservation(
            borrowed,
            request(ReservationKind::ProbeSide, "join_probe_side", 60, tracker.budget()),
            &mut metrics,
        );
        let reserved_after_relabel = tracker.reserved_bytes();
        let protected_after_relabel = tracker.source_cache_protected_key_count(&key);
        tracker.release(relabeled, &mut metrics);
        let reserved_after_alias_release = tracker.reserved_bytes();
        let protected_after_alias_release = tracker.source_cache_protected_key_count(&key);
        tracker.release_source_cache_baseline_bytes(60, &mut metrics);
        let reserved_after_cache_release = tracker.reserved_bytes();

        Ok(SourceCacheAliasRelabelProbe {
            metrics,
            reserved_after_cache_insert,
            reserved_after_borrow,
            reserved_after_relabel,
            reserved_after_alias_release,
            reserved_after_cache_release,
            protected_after_borrow,
            protected_after_relabel,
            protected_after_alias_release,
        })
    }

    fn source_cache_key() -> GpuLocalParquetSourceCacheKey {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", ArrowDataType::Int64, true)]));
        let source =
            LocalParquetSource::new("/tmp/source-cache-alias-contract.parquet", schema, vec!["c0".to_string()]);
        GpuLocalParquetSourceCacheKey::from_source(&source)
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

pub(crate) use self::validate::validate_literal_batch_source;
use self::{
    bounded::*,
    graph_algorithm::*,
    join::*,
    lowering::*,
    materialize::*,
    operators::*,
    order::*,
    types::{QueryExecutionResult as ExecutionResult, QueryOutput as Output},
    util::*,
    validate::*,
};
use crate::{
    chunk::{
        SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceTaskMemoryModel,
        plan_local_parquet_source_chunks,
    },
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_implicit_cast, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, Expr, LimitSpec, LiteralValue, SortSpec,
    },
    metrics::QueryMetrics,
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, LimitNode, NodeId,
        PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode, WindowFrame, WindowNode,
    },
    runtime::{
        BoundedJoinReservationContract, ExecutionContext, MemoryControlPolicy, MemoryReservationRequest,
        ReservationKind, RetryFacts,
    },
    source::{
        ByteRangeDisposition, GpuResidentSource, LocalParquetFile, LocalParquetSource, PartitionColumn,
        PartitionColumnDisposition, ScanLimitDisposition, SchemaMappingDisposition, ZERO_COLUMN_COUNT_COLUMN,
        ZeroColumnCountDisposition,
    },
};

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
    durable_hash_join_builds: Option<Arc<GpuHashJoinBuildCache>>,
}

impl ExecutionScratch {
    pub(super) fn new(context: &ExecutionContext) -> Self {
        Self {
            node_results: HashMap::new(),
            hash_join_builds: HashJoinBuildRepository::new(context.shared_runtime(), context.device_placement()),
            durable_hash_join_builds: context.hash_join_build_cache(),
        }
    }
}

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
/// in `plan`. Bounded plans dispatch through the context's strict/adaptive
/// pipeline mode; unbounded plans execute the operator tree directly.
pub fn execute_query_plan_with_gpu_inputs(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ExecutionResult> {
    context.check_cancellation("cudf_query_engine.exec.execute_query_plan")?;
    validate_query_plan(plan)?;
    let _order_tracking = DataFusionOrderTrackingGuard::new(
        context.stddev_samp_execution_mode().is_datafusion_exact() && plan_requires_datafusion_exact_stddev_order(plan),
    );
    let mut metrics = QueryMetrics {
        phase_c_memory_control_policy: context.memory_control_policy(),
        execution_boundary_policy: context.boundary_policy(),
        ..Default::default()
    };
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, gpu_inputs, &mut metrics)?
    } else {
        let mut scratch = ExecutionScratch::new(context);
        let frame = execute_node(context, plan, plan.sink().input(), gpu_inputs, &mut scratch, &mut metrics)?;
        match plan.sink() {
            ResultSink::GpuResident { .. } => Output::GpuResident(gpu_result_handle(context, plan, frame)?),
            ResultSink::HostArrow { schema, .. } => {
                let executable_schema =
                    plan.node(plan.sink().input()).expect("validated sink input exists").output_schema().clone();
                Output::HostArrow(materialize_gpu_frame_to_host(
                    context,
                    frame,
                    executable_schema,
                    schema,
                    &mut metrics,
                )?)
            },
        }
    };
    Ok(ExecutionResult::new(output, metrics))
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
) -> Result<types::GpuDataFrame> {
    context.check_cancellation("cudf_query_engine.exec.execute_node")?;
    if let Some(frame) = scratch.node_results.get(&id) {
        return Ok(frame.clone());
    }

    let node = plan.node(id).ok_or_else(|| invalid_node_reference(id, "query plan references a missing node"))?;

    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, source, metrics),
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
            let input = execute_node(context, plan, graph.input, gpu_inputs, scratch, metrics)?;
            Ok(execute_graph_algorithm(context, node.id, graph, input, metrics)
                .map_err(|error| error.with_fact("node_id", node.id))?)
        },
    }?;

    record_device_memory_observation(&frame, metrics)?;
    scratch.node_results.insert(id, frame.clone());
    Ok(frame)
}
