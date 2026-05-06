//! Native query execution.

use std::{cell::Cell, collections::HashMap, rc::Rc, sync::Arc, time::Instant};

use arrow::{
    array::Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec as CudfAggSpec, DataFrame, DataFrameHashJoin, DuplicateKeepOption, ExecutionOverrides, GroupByAgg,
    GroupByOptions, SortOptions as CudfSortOptions,
    frame::dataframe::{DropDuplicatesOptions, InfoDetail, ParquetReadOptions},
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

pub use source::{read_local_parquet_source, validate_local_parquet_source};
pub use types::{ExecutionMode, GpuDataFrame, QueryExecutionResult, QueryOutput};

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
    datatypes::{cudf_dtype_from_arrow_type, timestamp_unit_to_cudf_resolution},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortSpec},
    metrics::QueryMetrics,
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinNullEquality, LimitNode, NodeId, PlanNode, ProjectionNode, QueryPlan,
        ResultSink, SortNode,
    },
    runtime::{ExecutionContext, MemoryControlPolicy, MemoryReservationRequest, ReservationKind, RetryFacts},
    source::{
        ByteRangeDisposition, LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionColumnDisposition,
        SchemaMappingDisposition, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
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
/// `hash_join_cache` memoizes reusable cuDF hash-join artifacts for repeated
/// probes against the same build node, build keys, and null-equality contract.
/// The cache is scoped to this scratch object and is released at query end. It
/// is deliberately absent from bounded join execution until cudf-rs exposes
/// honest hash-table byte observability and the engine can reserve build,
/// probe, gather, and output memory before constructing join artifacts.
#[derive(Default)]
pub(super) struct ExecutionScratch {
    node_results: HashMap<NodeId, GpuDataFrame>,
    hash_join_cache: HashMap<HashJoinCacheKey, DataFrameHashJoin>,
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
    validate_query_plan(plan)?;
    let mut metrics =
        QueryMetrics { phase_c_memory_control_policy: context.memory_control_policy(), ..Default::default() };
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, &mut metrics)?
    } else {
        let mut scratch = ExecutionScratch::default();
        let frame = execute_node(context, plan, plan.sink().input(), &mut scratch, &mut metrics)?;
        match plan.sink() {
            ResultSink::GpuResident { .. } => Output::GpuResident(frame),
            ResultSink::HostArrow { schema, .. } => {
                Output::HostArrow(materialize_to_host(frame, schema, &mut metrics)?)
            },
        }
    };
    Ok(ExecutionResult::new(output, metrics))
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
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
        },
        PlanNode::Filter(filter) => {
            let input = execute_node(context, plan, filter.input, scratch, metrics)?;
            execute_filter(context, filter, input, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_node(context, plan, projection.input, scratch, metrics)?;
            execute_projection(context, projection, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_node(context, plan, aggregate.input, scratch, metrics)?;
            execute_aggregate(context, aggregate, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_node(context, plan, sort.input, scratch, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_node(context, plan, limit.input, scratch, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
        PlanNode::Join(join) => {
            let left = execute_node(context, plan, join.inputs[0], scratch, metrics)?;
            let right = execute_node(context, plan, join.inputs[1], scratch, metrics)?;
            execute_join(context, node.id, join, left, right, scratch, metrics)
        },
    }?;

    record_device_memory_observation(&frame, metrics)?;
    scratch.node_results.insert(id, frame.clone());
    Ok(frame)
}
