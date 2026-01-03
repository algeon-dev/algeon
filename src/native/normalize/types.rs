use arrow::datatypes::SchemaRef;

use super::*;

/// Borrowed, normalized representation of a DataFusion physical subtree that
/// the native path knows how to lower.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativePlan<'a> {
    /// Candidate metadata emitted in planning reports. The shape and operator
    /// count must stay aligned with the normalized tree below.
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) root: NormalizedNativeRoot<'a>,
    pub(in crate::native) sort: Option<&'a SortExec>,
    pub(in crate::native) limit_fetch: Option<usize>,
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
            NormalizedNativeInput::Filter(_)
            | NormalizedNativeInput::Projection(_)
            | NormalizedNativeInput::Aggregate(_)
            | NormalizedNativeInput::HashJoin(_)
            | NormalizedNativeInput::NestedLoopJoin(_) => None,
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

/// Root aggregate plus the operators that DataFusion may place immediately
/// before or after it in the supported native shapes.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeAggregateRoot<'a> {
    pub(in crate::native) input: NormalizedNativeInput<'a>,
    pub(in crate::native) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(in crate::native) aggregate: &'a AggregateExec,
    /// True when `aggregate` is a safe partial side of a DataFusion
    /// partial/final pair. Lowering may then accept partial aggregate state
    /// layouts, but only after the safety predicates below have matched them.
    pub(in crate::native) aggregate_allows_partial: bool,
    /// Output schema for the native aggregate node. For distributed pairs this
    /// is the final aggregate schema, even though `aggregate` points at the
    /// partial `AggregateExec` that computes the state.
    pub(in crate::native) aggregate_output_schema: SchemaRef,
    pub(in crate::native) post_aggregate_projection: Option<&'a ProjectionExec>,
}

/// Root whose result is the input subtree itself, optionally followed by a
/// projection, sort, and fetch.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeInputRoot<'a> {
    pub(in crate::native) input: NormalizedNativeInput<'a>,
    pub(in crate::native) projection: Option<&'a ProjectionExec>,
}

/// Recursive native input tree. These variants are the boundary between shape
/// recognition and lowering; adding one requires matching lowering support.
#[derive(Debug, Clone)]
pub(in crate::native) enum NormalizedNativeInput<'a> {
    Source(NormalizedNativeSource<'a>),
    Filter(NormalizedNativeFilter<'a>),
    Projection(NormalizedNativeProjection<'a>),
    Aggregate(NormalizedNativeAggregate<'a>),
    HashJoin(NormalizedNativeHashJoin<'a>),
    NestedLoopJoin(NormalizedNativeNestedLoopJoin<'a>),
}

impl NormalizedNativeInput<'_> {
    // Shape strings are consumed by reports/tests, so these helpers describe
    // the accepted physical tree rather than debug-printing implementation
    // details.
    pub(in crate::native) fn shape(&self) -> String {
        match self {
            Self::Source(source) => source.shape(),
            Self::Filter(filter) => filter.shape(),
            Self::Projection(projection) => projection.shape(),
            Self::Aggregate(aggregate) => aggregate.shape(),
            Self::HashJoin(join) => join.shape(),
            Self::NestedLoopJoin(join) => join.shape(),
        }
    }

    // Operator counts are planning diagnostics. They should count the matched
    // physical operators represented by the normalized tree and the root
    // wrappers that candidate helpers add.
    pub(in crate::native) fn operator_count(&self) -> usize {
        match self {
            Self::Source(source) => source.operator_count(),
            Self::Filter(filter) => filter.operator_count(),
            Self::Projection(projection) => projection.operator_count(),
            Self::Aggregate(aggregate) => aggregate.operator_count(),
            Self::HashJoin(join) => join.operator_count(),
            Self::NestedLoopJoin(join) => join.operator_count(),
        }
    }
}

/// Local Parquet source recognized by the native path. A directly attached
/// `FilterExec` is stored here so Parquet pruning/projection can reason about
/// scan, filter, and output columns together.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeSource<'a> {
    pub(in crate::native) source_exec: &'a DataSourceExec,
    pub(in crate::native) filter: Option<&'a FilterExec>,
}

impl NormalizedNativeSource<'_> {
    fn shape(&self) -> String {
        if self.filter.is_some() {
            wrap_shape("FilterExec", "LocalParquetScan".to_string())
        } else {
            "LocalParquetScan".to_string()
        }
    }

    fn operator_count(&self) -> usize {
        1 + usize::from(self.filter.is_some())
    }
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

/// Aggregate inside a recursive native input tree, for example above a join.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeAggregate<'a> {
    pub(in crate::native) aggregate: &'a AggregateExec,
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

/// Hash join with both inputs recursively normalized before join lowering
/// canonicalizes DataFusion join sides.
#[derive(Debug, Clone)]
pub(in crate::native) struct NormalizedNativeHashJoin<'a> {
    pub(in crate::native) join: &'a HashJoinExec,
    pub(in crate::native) left: Box<NormalizedNativeInput<'a>>,
    pub(in crate::native) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeHashJoin<'_> {
    fn shape(&self) -> String {
        wrap_shape("HashJoinExec", format!("{},{}", self.left.shape(), self.right.shape()))
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

/// Rejection paired with the candidate shape that was being tested. Reports
/// rely on both pieces to explain native fallback pressure.
#[derive(Debug, Clone)]
pub(in crate::native) struct NativePlanRejection {
    pub(in crate::native) candidate: GpuCandidate,
    pub(in crate::native) reason: FallbackReason,
}

/// Supported root wrappers outside the aggregate/input subtree. These variants
/// encode ordering semantics, not just display names.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NativeStackRoot {
    LocalLimit,
    Sort,
    SortFetch,
    SortPreservingMergeFetch,
}

/// Aggregate shape family used by diagnostics string builders.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) enum NativeAggregateShapeKind {
    Global,
    SortStack(NativeStackRoot),
}
