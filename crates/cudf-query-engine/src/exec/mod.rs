//! Native query execution.

use std::{cell::Cell, collections::HashMap, rc::Rc, sync::Arc, time::Instant};

use arrow::{
    array::Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec as CudfAggSpec, BudgetedExecutionOutcome, DataFrame, DuplicateKeepOption, ExecutionBudgetEnforcement,
    ExecutionBudgetRequest, ExecutionBudgetToken, ExecutionOverrides, GroupByAgg, GroupByOptions, ReduceAgg,
    SortOptions as CudfSortOptions,
    core::Scalar as CudfScalar,
    frame::dataframe::{DropDuplicatesOptions, InfoDetail, ParquetReadOptions},
    io::source::SourceReadMode,
    ops::{col, lit},
};

mod bounded;
mod join;
mod lowering;
mod materialize;
mod operators;
mod source;
mod types;
mod util;
mod validate;

#[doc(hidden)]
pub use bounded::strict::{
    CompiledBoundedPipelineSupport, CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler,
    CompiledPipelineState, CompiledTaskExecutor, CompiledTaskOutput, ReleasedCompiledResource,
    StrictBoundedPipelineAdmission, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
    strict_bounded_pipeline_admission,
};
pub use bounded::{
    AdaptivePipelineAdmission, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history,
};
pub use source::{read_gpu_resident_source, read_local_parquet_source, validate_local_parquet_source};
pub use types::{
    GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildCache, GpuHashJoinBuildHandle,
    GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap,
    GpuResultHandle, QueryExecutionResult, QueryOutput, QueryOutputMode,
};

use self::{
    bounded::*,
    join::*,
    lowering::*,
    materialize::*,
    operators::*,
    types::{QueryExecutionResult as ExecutionResult, QueryOutput as Output},
    util::*,
    validate::*,
};
use crate::{
    chunk::{SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, plan_local_parquet_source_chunks},
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_implicit_cast, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortSpec},
    metrics::QueryMetrics,
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, LimitNode, NodeId,
        PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode,
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

pub fn execute_query_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<ExecutionResult> {
    execute_query_plan_with_gpu_inputs(context, plan, &GpuResidentInputMap::new())
}

pub fn execute_query_plan_with_gpu_inputs(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ExecutionResult> {
    validate_query_plan(plan)?;
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
            execute_aggregate(context, aggregate, input, &node.facts.output_schema, metrics)
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
    }?;

    record_device_memory_observation(&frame, metrics)?;
    scratch.node_results.insert(id, frame.clone());
    Ok(frame)
}
