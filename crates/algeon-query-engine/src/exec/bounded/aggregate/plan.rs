use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{AggregateNode, FilterNode, LimitNode, LocalParquetSource, NodeId, ProjectionNode, SortNode};

pub(in crate::exec::bounded) type ProjectionStage<'a> = (&'a ProjectionNode, &'a ArrowSchemaRef);

/// A stage applied to the aggregate's output rows. Only projections are supported.
#[derive(Clone, Copy)]
pub(in crate::exec::bounded) enum BoundedUnaryStage<'a> {
    Projection { node: &'a ProjectionNode },
}

/// The linear plan shape the bounded aggregate paths recognize, borrowed from the native plan: a
/// local Parquet source, optional projections and a filter before the aggregate (the
/// source-adjacent stages), one aggregate, then optional projections, a sort and a limit.
///
/// `aggregate_input_schema` is the schema the aggregate reads and `aggregate.1` its output schema.
#[derive(Clone)]
pub(in crate::exec::bounded) struct BoundedPlan<'a> {
    pub(in crate::exec::bounded) source_id: NodeId,
    pub(in crate::exec::bounded) source: &'a LocalParquetSource,
    pub(in crate::exec::bounded) pre_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) filter: Option<&'a FilterNode>,
    pub(in crate::exec::bounded) post_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) aggregate_id: NodeId,
    pub(in crate::exec::bounded) aggregate_input_schema: &'a ArrowSchemaRef,
    pub(in crate::exec::bounded) aggregate: (&'a AggregateNode, &'a ArrowSchemaRef),
    pub(in crate::exec::bounded) post_aggregate_stages: Vec<BoundedUnaryStage<'a>>,
    pub(in crate::exec::bounded) sort: Option<&'a SortNode>,
    pub(in crate::exec::bounded) limit: Option<(NodeId, &'a LimitNode)>,
}

/// A single-consumer aggregate output path whose compiled sort contract
/// permits partition-local pruning before the global top-k task.
pub(in crate::exec::bounded) struct SourceAggregateTopK<'a> {
    pub(in crate::exec::bounded) post_aggregate_stages: Vec<BoundedUnaryStage<'a>>,
    pub(in crate::exec::bounded) sort: &'a SortNode,
    pub(in crate::exec::bounded) limit: (NodeId, &'a LimitNode),
}

impl<'a> BoundedPlan<'a> {
    /// Whether a filter or projection sits between the source and the aggregate. Each source chunk
    /// then needs those stages applied before its partial aggregate.
    pub(in crate::exec::bounded) fn has_source_adjacent_stages(&self) -> bool {
        !self.pre_filter_projections.is_empty() || self.filter.is_some() || !self.post_filter_projections.is_empty()
    }
}
