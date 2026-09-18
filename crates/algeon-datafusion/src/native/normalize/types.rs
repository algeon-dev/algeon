use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::window::WindowExpr,
    physical_plan::{empty::EmptyExec, placeholder_row::PlaceholderRowExec},
};
use query_engine::expr::LimitSpec;

use super::{
    AggregateExec, AlgeonIcebergScanExec, BoundedWindowAggExec, CapabilityReason, CrossJoinExec, ExecutionPlan,
    FilterExec, GpuCandidate, HashJoinExec, LexOrdering, NestedLoopJoinExec, ProjectionExec, SortExec, UnionExec,
    WindowAggExec, wrap_shape,
};

/// Borrowed, normalized representation of a DataFusion physical subtree that
/// the native path knows how to lower.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativePlan<'a> {
    /// Candidate metadata emitted in planning reports. The shape and operator
    /// count must stay aligned with the normalized tree below.
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) root: NormalizedNativeRoot<'a>,
    pub(in crate::native) sort: Option<NormalizedNativeSortExprs<'a>>,
    pub(in crate::native) limit: Option<LimitSpec>,
}

impl<'a> NormalizedNativePlan<'a> {
    /// Returns the source-shaped aggregate input for the slice1 path.
    ///
    /// This only proves that the aggregate root input is a source variant,
    /// optionally with a filter attached to that source. Parquet/source
    /// validity is checked later during source lowering. Unary wrappers,
    /// aggregate inputs, or join subtrees inside the normalized input subtree
    /// use the general input lowering path instead; root-level pre/post
    /// aggregate projection, sort, and limit may still belong to the slice1
    /// path.
    pub(in crate::native) fn single_source_input(&self) -> Option<&NormalizedNativeSource<'a>> {
        let root = self.aggregate_root()?;
        match &root.input {
            NormalizedNativeInput::Source(source) => Some(source),
            NormalizedNativeInput::GpuResidentSource(_) => None,
            NormalizedNativeInput::LiteralSource(_)
            | NormalizedNativeInput::Filter(_)
            | NormalizedNativeInput::Projection(_)
            | NormalizedNativeInput::Limit(_)
            | NormalizedNativeInput::Aggregate(_)
            | NormalizedNativeInput::Sort(_)
            | NormalizedNativeInput::Window(_)
            | NormalizedNativeInput::HashJoin(_)
            | NormalizedNativeInput::CrossJoin(_)
            | NormalizedNativeInput::NestedLoopJoin(_)
            | NormalizedNativeInput::Union(_) => None,
        }
    }

    pub(in crate::native) fn root_input(&self) -> &NormalizedNativeInput<'a> {
        match &self.root {
            NormalizedNativeRoot::Aggregate(root) => &root.input,
            NormalizedNativeRoot::Input(root) => &root.input,
        }
    }

    pub(in crate::native) fn aggregate_root(&self) -> Option<&NormalizedNativeAggregateRoot<'a>> {
        match &self.root {
            NormalizedNativeRoot::Aggregate(root) => Some(root),
            NormalizedNativeRoot::Input(_) => None,
        }
    }
}

/// Native roots are either aggregate stacks or input stacks. Aggregate roots
/// keep pre/post aggregate operators separate because distributed aggregate
/// provenance and output schemas are semantic contracts.
#[derive(Debug, Clone)]
pub(in crate::native) enum NormalizedNativeRoot<'a> {
    Aggregate(NormalizedNativeAggregateRoot<'a>),
    Input(NormalizedNativeInputRoot<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(in crate::native) enum PostAggregateProjectionOrder {
    #[default]
    BelowSort,
    AboveSort,
}

/// Root aggregate plus the operators that DataFusion may place immediately
/// before or after it in the supported native shapes.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeAggregateRoot<'a> {
    pub(in crate::native) input: NormalizedNativeInput<'a>,
    pub(in crate::native) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(in crate::native) aggregate: &'a AggregateExec,
    /// Whole final output statistics, before peeling a distributed partial/final pair.
    pub(in crate::native) statistics_output: &'a AggregateExec,
    /// True when `aggregate` is a safe partial side of a DataFusion
    /// partial/final pair. Lowering may then accept partial aggregate state
    /// layouts, but only after the safety predicates below have matched them.
    pub(in crate::native) aggregate_allows_partial: bool,
    /// Output schema for the native aggregate node. For distributed pairs this
    /// is the final aggregate schema, even though `aggregate` points at the
    /// partial `AggregateExec` that computes the state.
    pub(in crate::native) aggregate_output_schema: SchemaRef,
    pub(in crate::native) post_aggregate_projection: Option<&'a ProjectionExec>,
    pub(in crate::native) projection_order: PostAggregateProjectionOrder,
}

/// Root whose result is the input subtree itself, optionally followed by a
/// projection, sort, and fetch.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeInputRoot<'a> {
    pub(in crate::native) input: NormalizedNativeInput<'a>,
    pub(in crate::native) projection: Option<&'a ProjectionExec>,
    pub(in crate::native) projection_order: PostAggregateProjectionOrder,
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
    // Shape strings are consumed by reports/tests, so these helpers describe
    // the accepted physical tree rather than debug-printing implementation
    // details.
    pub(in crate::native) fn shape(&self) -> String {
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

    // Operator counts are planning diagnostics. They should count the matched
    // physical operators represented by the normalized tree and the root
    // wrappers that candidate helpers add.
    pub(in crate::native) fn operator_count(&self) -> usize {
        match self {
            Self::Source(source) => source.operator_count(),
            Self::GpuResidentSource(source) => source.operator_count(),
            Self::LiteralSource(source) => source.operator_count(),
            Self::Filter(filter) => filter.operator_count(),
            Self::Projection(projection) => projection.operator_count(),
            Self::Limit(limit) => limit.operator_count(),
            Self::Aggregate(aggregate) => aggregate.operator_count(),
            Self::Sort(sort) => sort.operator_count(),
            Self::Window(window) => window.operator_count(),
            Self::HashJoin(join) => join.operator_count(),
            Self::CrossJoin(join) => join.operator_count(),
            Self::NestedLoopJoin(join) => join.operator_count(),
            Self::Union(union) => union.operator_count(),
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
    fn shape(&self) -> String {
        match self {
            Self::PlaceholderRow(_) => "PlaceholderRowExec".to_string(),
            Self::Empty(_) => "EmptyExec".to_string(),
            Self::Memory(_) => "MemorySource".to_string(),
        }
    }

    fn operator_count(&self) -> usize {
        1
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
    fn shape(&self) -> String {
        self.source_exec.name().to_string()
    }

    fn operator_count(&self) -> usize {
        1
    }
}

/// Local Parquet source recognized by the native path. A directly attached
/// `FilterExec` is stored here so Parquet pruning/projection can reason about
/// scan, filter, and output columns together.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSource<'a> {
    pub(in crate::native) scan: NormalizedNativeScan<'a>,
    pub(in crate::native) filter: Option<&'a FilterExec>,
}

impl NormalizedNativeSource<'_> {
    pub(in crate::native) fn local_parquet_source_exec(&self) -> Option<&DataSourceExec> {
        match self.scan {
            NormalizedNativeScan::LocalParquet(source_exec) => Some(source_exec),
            NormalizedNativeScan::TableFormat(_) => None,
        }
    }

    pub(in crate::native) fn table_format_scan_exec(&self) -> Option<&AlgeonIcebergScanExec> {
        match self.scan {
            NormalizedNativeScan::LocalParquet(_) => None,
            NormalizedNativeScan::TableFormat(scan_exec) => Some(scan_exec),
        }
    }

    fn shape(&self) -> String {
        let source_shape = match self.scan {
            NormalizedNativeScan::LocalParquet(_) => "LocalParquetScan".to_string(),
            NormalizedNativeScan::TableFormat(scan_exec) => {
                format!("TableFormatScan({})", scan_exec.table_format().code())
            },
        };
        if self.filter.is_some() { wrap_shape("FilterExec", source_shape) } else { source_shape }
    }

    fn operator_count(&self) -> usize {
        1 + usize::from(self.filter.is_some())
    }
}

/// Source scan leaf recognized by native normalization.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NormalizedNativeScan<'a> {
    LocalParquet(&'a DataSourceExec),
    TableFormat(&'a AlgeonIcebergScanExec),
}

/// Non-source filter in a recursive native input tree.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeFilter<'a> {
    pub(in crate::native) filter: &'a FilterExec,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeFilter<'_> {
    fn shape(&self) -> String {
        wrap_shape("FilterExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

/// Projection inside a recursive native input tree.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeProjection<'a> {
    pub(in crate::native) projection: &'a ProjectionExec,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeProjection<'_> {
    fn shape(&self) -> String {
        wrap_shape("ProjectionExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

/// Global limit inside a recursive native input tree.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeLimit<'a> {
    pub(in crate::native) spec: LimitSpec,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeLimit<'_> {
    fn shape(&self) -> String {
        wrap_shape("GlobalLimitExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

/// Aggregate inside a recursive native input tree, for example above a join.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeAggregate<'a> {
    pub(in crate::native) aggregate: &'a AggregateExec,
    /// Whole final output statistics, before peeling a distributed partial/final pair.
    pub(in crate::native) statistics_output: &'a AggregateExec,
    /// Mirrors root aggregate semantics: true only for a validated partial
    /// aggregate from a distributed final/partial pair.
    pub(in crate::native) aggregate_allows_partial: bool,
    /// Native aggregate output schema. For distributed pairs this preserves the
    /// final schema while lowering the partial aggregate state producer.
    pub(in crate::native) aggregate_output_schema: SchemaRef,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeAggregate<'_> {
    fn shape(&self) -> String {
        wrap_shape("AggregateExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

/// Sort inside a recursive native input tree, for example as the ordering
/// contract immediately below a native window.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSort<'a> {
    pub(in crate::native) sort: NormalizedNativeSortExprs<'a>,
    pub(in crate::native) fetch: Option<LimitSpec>,
    pub(in crate::native) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeSort<'_> {
    fn shape(&self) -> String {
        let name = if self.fetch.is_some() { "SortExec(fetch)" } else { "SortExec" };
        wrap_shape(name, self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

/// Sort ordering carried by a normalized native sort node.
///
/// Most physical sort roots borrow the original DataFusion `SortExec`.
/// Sort-preserving merge roots provide only an ordering contract, so the
/// normalizer stores the ordering plus the schema it should be resolved
/// against.
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
    fn shape(&self) -> String {
        wrap_shape(self.window.name(), self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
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
    fn shape(&self) -> String {
        let name = if self.fetch.is_some() { "HashJoinExec(fetch)" } else { "HashJoinExec" };
        wrap_shape(name, format!("{},{}", self.left.shape(), self.right.shape()))
    }

    fn operator_count(&self) -> usize {
        1 + self.left.operator_count() + self.right.operator_count()
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
    fn shape(&self) -> String {
        wrap_shape("CrossJoinExec", format!("{},{}", self.left.shape(), self.right.shape()))
    }

    fn operator_count(&self) -> usize {
        1 + self.left.operator_count() + self.right.operator_count()
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
    fn shape(&self) -> String {
        wrap_shape("NestedLoopJoinExec", format!("{},{}", self.left.shape(), self.right.shape()))
    }

    fn operator_count(&self) -> usize {
        1 + self.left.operator_count() + self.right.operator_count()
    }
}

/// DataFusion physical union with recursively normalized children.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeUnion<'a> {
    pub(in crate::native) union: &'a UnionExec,
    pub(in crate::native) inputs: Vec<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeUnion<'_> {
    fn shape(&self) -> String {
        wrap_shape("UnionExec", self.inputs.iter().map(NormalizedNativeInput::shape).collect::<Vec<_>>().join(","))
    }

    fn operator_count(&self) -> usize {
        1 + self.inputs.iter().map(NormalizedNativeInput::operator_count).sum::<usize>()
    }
}

/// Rejection paired with the candidate shape that was being tested. Reports
/// rely on both pieces to explain native fallback pressure.
#[derive(Debug, Clone)]
pub(in crate::native) struct NativePlanRejection {
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) reason: CapabilityReason,
}

/// Supported root wrappers outside the aggregate/input subtree. These variants
/// encode ordering semantics, not just display names.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NativeStackRoot {
    LocalLimit,
    Sort,
    SortFetch,
    SortPreservingMerge,
    SortPreservingMergeFetch,
    SortPreservingMergeSort,
    SortPreservingMergeFetchSort,
}

/// Aggregate shape family used by diagnostics string builders.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NativeAggregateShapeKind {
    Global,
    SortStack(NativeStackRoot),
}
