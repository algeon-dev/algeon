use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::window::WindowExpr,
    physical_plan::{empty::EmptyExec, placeholder_row::PlaceholderRowExec},
};
use query_planning::expr::LimitSpec;

use super::{
    AggregateExec, BoundedWindowAggExec, CapabilityReason, CrossJoinExec, ExecutionPlan, FilterExec, GpuCandidate,
    HashJoinExec, LexOrdering, NestedLoopJoinExec, ProjectionExec, SortExec, UnionExec, WindowAggExec,
};
use crate::{
    native::DECLARED_OUTPUT_ORDERING_SORT_OPERATOR,
    report::{CandidateShape, PlanBlocker, SinglePartitionOutcome},
};

/// Borrowed, normalized representation of a DataFusion physical subtree that
/// the native path can lower. It is one recursive input tree whose root is the
/// candidate's output.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativePlan<'a> {
    /// Candidate metadata emitted in planning reports, derived from `input`.
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) input: NormalizedNativeInput<'a>,
}

impl<'a> NormalizedNativePlan<'a> {
    pub(in crate::native) fn new(input: NormalizedNativeInput<'a>, output_schema_fields: usize) -> Self {
        let candidate = GpuCandidate::new(input.shape(), output_schema_fields);
        Self { candidate, input }
    }
}

/// Recursive native input tree. These variants are the boundary between shape
/// recognition and lowering; adding one requires matching lowering support.
#[derive(Debug, Clone)]
pub(in crate::native) enum NormalizedNativeInput<'a> {
    Source(NormalizedNativeSource<'a>),
    GpuResidentSource(NormalizedNativeGpuResidentSource<'a>),
    LiteralSource(NormalizedNativeLiteralSource<'a>),
    Filter(NormalizedNativeFilter<'a>),
    Projection(NormalizedNativeProjection<'a>),
    Limit(NormalizedNativeLimit<'a>),
    Aggregate(NormalizedNativeAggregate<'a>),
    Sort(NormalizedNativeSort<'a>),
    Window(NormalizedNativeWindow<'a>),
    HashJoin(NormalizedNativeHashJoin<'a>),
    CrossJoin(NormalizedNativeCrossJoin<'a>),
    NestedLoopJoin(NormalizedNativeNestedLoopJoin<'a>),
    Union(NormalizedNativeUnion<'a>),
}

impl NormalizedNativeInput<'_> {
    // Shapes are consumed by reports and tests, so they describe the accepted
    // physical operators. They do not describe implementation details.
    pub(in crate::native) fn shape(&self) -> CandidateShape {
        match self {
            Self::Source(source) => source.shape(),
            Self::GpuResidentSource(source) => source.shape(),
            Self::LiteralSource(source) => source.shape(),
            Self::Filter(filter) => filter.shape(),
            Self::Projection(projection) => projection.shape(),
            Self::Limit(limit) => limit.shape(),
            Self::Aggregate(aggregate) => aggregate.shape(),
            Self::Sort(sort) => sort.shape(),
            Self::Window(window) => window.shape(),
            Self::HashJoin(join) => join.shape(),
            Self::CrossJoin(join) => join.shape(),
            Self::NestedLoopJoin(join) => join.shape(),
            Self::Union(union) => union.shape(),
        }
    }
}

/// In-memory literal source recognized by native normalization.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NormalizedNativeLiteralSource<'a> {
    /// DataFusion placeholder one-row, zero-column input.
    PlaceholderRow(&'a PlaceholderRowExec),
    /// DataFusion typed empty input.
    Empty(&'a EmptyExec),
    /// DataFusion memory source, including SQL `VALUES`.
    Memory(&'a DataSourceExec),
}

impl NormalizedNativeLiteralSource<'_> {
    fn shape(&self) -> CandidateShape {
        match self {
            Self::PlaceholderRow(_) => CandidateShape::leaf("PlaceholderRowExec"),
            Self::Empty(_) => CandidateShape::leaf("EmptyExec"),
            Self::Memory(_) => CandidateShape::leaf("MemorySource"),
        }
    }
}

/// GPU-resident output supplied by another DataFusion execution node, such as
/// a cuGraph UDTF producer.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeGpuResidentSource<'a> {
    pub(in crate::native) source_exec: &'a dyn ExecutionPlan,
    pub(in crate::native) input_name: String,
    pub(in crate::native) row_count_upper_bound: Option<u64>,
    pub(in crate::native) unique_output_key: Option<BTreeSet<String>>,
}

impl NormalizedNativeGpuResidentSource<'_> {
    fn shape(&self) -> CandidateShape {
        CandidateShape::leaf("GpuResidentSource")
    }
}

/// Local Parquet source recognized by the native path. A directly attached
/// `FilterExec` is stored here so Parquet pruning and projection analyze scan,
/// filter, and output columns together.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSource<'a> {
    pub(in crate::native) scan: &'a DataSourceExec,
    pub(in crate::native) filter: Option<&'a FilterExec>,
}

impl NormalizedNativeSource<'_> {
    fn shape(&self) -> CandidateShape {
        let source_shape = CandidateShape::leaf("LocalParquetScan");
        if self.filter.is_some() { CandidateShape::new("FilterExec", vec![source_shape]) } else { source_shape }
    }
}

/// Non-source filter in a recursive native input tree.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeFilter<'a> {
    pub(in crate::native) filter: &'a FilterExec,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeFilter<'_> {
    fn shape(&self) -> CandidateShape {
        unary_shape("FilterExec", self.input.shape())
    }
}

/// Projection inside a recursive native input tree.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeProjection<'a> {
    pub(in crate::native) projection: &'a ProjectionExec,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeProjection<'_> {
    fn shape(&self) -> CandidateShape {
        unary_shape("ProjectionExec", self.input.shape())
    }
}

/// A limit in a recursive native input tree: a global limit, a local limit
/// over one partition, or a fetch-limited coalesce.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeLimit<'a> {
    /// The DataFusion operator the limit came from, for the candidate shape.
    pub(in crate::native) operator: &'static str,
    pub(in crate::native) spec: LimitSpec,
    /// DataFusion reports an ordering for the limit's input, so the limit
    /// keeps the first rows in that order.
    pub(in crate::native) input_ordered: bool,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeLimit<'_> {
    fn shape(&self) -> CandidateShape {
        unary_shape(self.operator, self.input.shape())
    }
}

/// Aggregate in a recursive native input tree, with the projection DataFusion
/// placed directly below it.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeAggregate<'a> {
    pub(in crate::native) aggregate: &'a AggregateExec,
    /// Projection directly below the executable aggregate; its expressions
    /// are the aggregate's inputs.
    pub(in crate::native) pre_aggregate_projection: Option<&'a ProjectionExec>,
    /// The exchange between a distributed final aggregate and its partial,
    /// when `aggregate` is the partial side of such a pair.
    pub(in crate::native) distributed_exchange: Option<&'a str>,
    /// Whole final output statistics, before peeling a distributed partial/final pair.
    pub(in crate::native) statistics_output: &'a AggregateExec,
    /// Mirrors root aggregate semantics. True only for a validated partial
    /// aggregate from a distributed final/partial pair.
    pub(in crate::native) aggregate_allows_partial: bool,
    /// Native aggregate output schema. For distributed pairs this preserves the
    /// final schema while lowering the partial aggregate state producer.
    pub(in crate::native) aggregate_output_schema: SchemaRef,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeAggregate<'_> {
    fn shape(&self) -> CandidateShape {
        let mut input = self.input.shape();
        if self.pre_aggregate_projection.is_some() {
            input = unary_shape("ProjectionExec", input);
        }
        if !self.aggregate_allows_partial {
            return unary_shape("AggregateExec", input);
        }
        let partial = unary_shape("PartialAggregateExec", input);
        let exchanged = match self.distributed_exchange {
            Some(exchange) => unary_shape(exchange, partial),
            None => partial,
        };
        unary_shape("FinalAggregateExec", exchanged)
    }
}

/// Sort inside a recursive native input tree, with the fetch of a limit or
/// merge folded into it.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSort<'a> {
    /// The DataFusion operator the sort came from, for the candidate shape.
    pub(in crate::native) origin: NormalizedNativeSortOrigin,
    pub(in crate::native) sort: NormalizedNativeSortExprs<'a>,
    pub(in crate::native) fetch: Option<LimitSpec>,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

/// Which DataFusion operator a native sort stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum NormalizedNativeSortOrigin {
    Sort,
    /// A merge of sorted partitions, which one native sort reproduces.
    SortPreservingMerge,
    /// A merge over the per-partition sort that produced its runs; one native
    /// sort replaces both. `sort_fetch` records whether that sort had a fetch.
    SortPreservingMergeOverSort {
        sort_fetch: bool,
    },
    /// The plan's declared output ordering, made explicit above a fetch.
    DeclaredOutputOrdering,
}

impl NormalizedNativeSort<'_> {
    fn shape(&self) -> CandidateShape {
        let name = match (self.origin, self.fetch.is_some()) {
            (NormalizedNativeSortOrigin::Sort, false) => "SortExec",
            (NormalizedNativeSortOrigin::Sort, true) => "SortExec(fetch)",
            (NormalizedNativeSortOrigin::SortPreservingMerge, false) => "SortPreservingMergeExec",
            (NormalizedNativeSortOrigin::SortPreservingMerge, true) => "SortPreservingMergeExec(fetch)",
            (NormalizedNativeSortOrigin::SortPreservingMergeOverSort { sort_fetch }, merge_fetch) => {
                let merge = if merge_fetch { "SortPreservingMergeExec(fetch)" } else { "SortPreservingMergeExec" };
                let sort = if sort_fetch { "SortExec(fetch)" } else { "SortExec" };
                return unary_shape(merge, unary_shape(sort, self.input.shape()));
            },
            (NormalizedNativeSortOrigin::DeclaredOutputOrdering, _) => DECLARED_OUTPUT_ORDERING_SORT_OPERATOR,
        };
        unary_shape(name, self.input.shape())
    }
}

/// Sort ordering carried by a normalized native sort node.
///
/// A sort borrows the original DataFusion `SortExec`. A sort-preserving merge
/// or a declared output ordering provides only an ordering; the normalizer
/// stores it with the schemas it resolves against.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSortExprs<'a> {
    ordering: &'a LexOrdering,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl<'a> NormalizedNativeSortExprs<'a> {
    pub(in crate::native) fn from_sort_exec(sort: &'a SortExec) -> Self {
        Self { ordering: sort.expr(), input_schema: sort.input().schema(), output_schema: sort.schema() }
    }

    pub(in crate::native) fn from_ordering(
        ordering: &'a LexOrdering,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
    ) -> Self {
        Self { ordering, input_schema, output_schema }
    }

    pub(in crate::native) fn ordering(&self) -> &LexOrdering {
        self.ordering
    }

    pub(in crate::native) fn input_schema(&self) -> &SchemaRef {
        &self.input_schema
    }

    pub(in crate::native) fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }
}

/// DataFusion window node with a recursively normalized input.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeWindow<'a> {
    pub(in crate::native) window: NormalizedNativeWindowExec<'a>,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NormalizedNativeWindowExec<'a> {
    Unbounded(&'a WindowAggExec),
    Bounded(&'a BoundedWindowAggExec),
}

impl NormalizedNativeWindowExec<'_> {
    pub(in crate::native) fn window_expr(&self) -> &[Arc<dyn WindowExpr>] {
        match self {
            Self::Unbounded(window) => window.window_expr(),
            Self::Bounded(window) => window.window_expr(),
        }
    }

    pub(in crate::native) fn input_schema(&self) -> SchemaRef {
        match self {
            Self::Unbounded(window) => window.input().schema(),
            Self::Bounded(window) => window.input().schema(),
        }
    }

    pub(in crate::native) fn schema(&self) -> SchemaRef {
        match self {
            Self::Unbounded(window) => window.schema(),
            Self::Bounded(window) => window.schema(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Unbounded(_) => "WindowAggExec",
            Self::Bounded(_) => "BoundedWindowAggExec",
        }
    }
}

impl NormalizedNativeWindow<'_> {
    fn shape(&self) -> CandidateShape {
        unary_shape(self.window.name(), self.input.shape())
    }
}

/// Hash join with both inputs recursively normalized before join lowering
/// canonicalizes DataFusion join sides.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeHashJoin<'a> {
    pub(in crate::native) join: &'a HashJoinExec,
    pub(in crate::native) fetch: Option<LimitSpec>,
    pub(in crate::native) left: Box<NormalizedNativeInput<'a>>,
    pub(in crate::native) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeHashJoin<'_> {
    fn shape(&self) -> CandidateShape {
        let name = if self.fetch.is_some() { "HashJoinExec(fetch)" } else { "HashJoinExec" };
        CandidateShape::new(name, vec![self.left.shape(), self.right.shape()])
    }
}

/// DataFusion physical cross join with normalized children.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeCrossJoin<'a> {
    pub(in crate::native) join: &'a CrossJoinExec,
    pub(in crate::native) left: Box<NormalizedNativeInput<'a>>,
    pub(in crate::native) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeCrossJoin<'_> {
    fn shape(&self) -> CandidateShape {
        CandidateShape::new("CrossJoinExec", vec![self.left.shape(), self.right.shape()])
    }
}

/// Nested-loop or cross join with normalized children.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeNestedLoopJoin<'a> {
    pub(in crate::native) join: &'a NestedLoopJoinExec,
    pub(in crate::native) left: Box<NormalizedNativeInput<'a>>,
    pub(in crate::native) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeNestedLoopJoin<'_> {
    fn shape(&self) -> CandidateShape {
        CandidateShape::new("NestedLoopJoinExec", vec![self.left.shape(), self.right.shape()])
    }
}

/// DataFusion physical union with recursively normalized children.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeUnion<'a> {
    pub(in crate::native) union: &'a UnionExec,
    pub(in crate::native) inputs: Vec<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeUnion<'_> {
    fn shape(&self) -> CandidateShape {
        CandidateShape::new("UnionExec", self.inputs.iter().map(NormalizedNativeInput::shape).collect())
    }
}

/// Rejection paired with the candidate shape that was being tested and the
/// plan element that blocked it. Reports rely on all three pieces to explain
/// native fallback pressure. `single_partition` says whether planning for one
/// partition lowers the shape; only a site that checked it sets
/// [`SinglePartitionOutcome::StillRejected`].
#[derive(Debug, Clone)]
pub(in crate::native) struct NativePlanRejection {
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) reason: CapabilityReason,
    pub(in crate::native) blocker: Box<PlanBlocker>,
    pub(in crate::native) single_partition: SinglePartitionOutcome,
}

fn unary_shape(operator: &str, input: CandidateShape) -> CandidateShape {
    CandidateShape::new(operator, vec![input])
}
