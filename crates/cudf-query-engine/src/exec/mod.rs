//! Native query execution.

use std::{cell::Cell, rc::Rc, sync::Arc, time::Instant};

use arrow::{
    array::Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    AggSpec as CudfAggSpec, DataFrame, ExecutionOverrides, GroupByAgg, SortOptions as CudfSortOptions, TypeId,
    frame::dataframe::{InfoDetail, ParquetReadOptions},
    ops::{col, lit},
};

mod bounded;
mod lowering;
mod materialize;
mod operators;
mod source;
mod types;
mod util;

pub use source::{read_local_parquet_source, validate_local_parquet_source};
pub use types::{ExecutionMode, GpuDataFrame, QueryExecutionResult, QueryOutput};

use self::{
    bounded::*,
    lowering::*,
    materialize::*,
    operators::*,
    types::{QueryExecutionResult as ExecutionResult, QueryOutput as Output},
    util::*,
};
use crate::{
    chunk::{SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, plan_local_parquet_source_chunks},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortSpec},
    metrics::QueryMetrics,
    plan::{AggregateNode, FilterNode, LimitNode, NodeId, PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode},
    runtime::{ExecutionContext, MemoryControlPolicy, MemoryReservationRequest, ReservationKind, RetryFacts},
    source::{
        ByteRangeDisposition, LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionColumnDisposition,
        SchemaMappingDisposition, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
    },
};

pub fn execute_query_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<ExecutionResult> {
    let mut metrics =
        QueryMetrics { phase_c_memory_control_policy: context.memory_control_policy(), ..Default::default() };
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, &mut metrics)?
    } else {
        let frame = execute_node(context, plan, plan.sink().input(), &mut metrics)?;
        match plan.sink() {
            ResultSink::GpuResident { .. } => Output::GpuResident(frame),
            ResultSink::HostArrow { .. } => Output::HostArrow(materialize_to_host(frame, &mut metrics)?),
        }
    };
    Ok(ExecutionResult::new(output, metrics))
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    metrics: &mut QueryMetrics,
) -> Result<types::GpuDataFrame> {
    let node = plan.node(id).ok_or_else(|| invalid_node_reference(id, "query plan references a missing node"))?;

    let frame = match &node.kind {
        PlanNode::Source(source) => match source {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, source, metrics),
        },
        PlanNode::Filter(filter) => {
            let input = execute_node(context, plan, filter.input, metrics)?;
            execute_filter(context, filter, input, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_node(context, plan, projection.input, metrics)?;
            execute_projection(context, projection, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_node(context, plan, aggregate.input, metrics)?;
            execute_aggregate(context, aggregate, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_node(context, plan, sort.input, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_node(context, plan, limit.input, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
    }?;

    record_device_memory_observation(&frame, metrics)?;
    Ok(frame)
}
