use super::*;

/// Boolean filtering operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    /// Input whose rows are tested.
    pub input: NodeId,
    /// Predicate evaluated against the input schema.
    pub predicate: Expr,
}

/// Expression projection over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionNode {
    /// Input whose columns and computed expressions are projected.
    pub input: NodeId,
    /// Output expressions and fields in result schema order.
    pub outputs: Vec<ProjectionOutput>,
    output_schema: SchemaRef,
}

impl ProjectionNode {
    pub fn new(input: NodeId, outputs: Vec<ProjectionOutput>) -> Self {
        let output_schema = projection_output_schema(&outputs);
        Self { input, outputs, output_schema }
    }

    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    pub fn expressions(&self) -> impl Iterator<Item = &Expr> {
        self.outputs.iter().map(|output| &output.expr)
    }

    pub fn output_fields(&self) -> impl Iterator<Item = &Field> {
        self.outputs.iter().map(|output| &output.field)
    }
}

/// One typed output of a projection node.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionOutput {
    /// Expression evaluated for this output column.
    pub expr: Expr,
    /// Arrow field that names and types the executable output column.
    pub field: Field,
}

impl ProjectionOutput {
    pub fn new(expr: Expr, field: Field) -> Self {
        Self { expr, field }
    }
}

fn projection_output_schema(outputs: &[ProjectionOutput]) -> SchemaRef {
    Arc::new(Schema::new(outputs.iter().map(|output| output.field.clone()).collect::<Vec<_>>()))
}

/// Aggregate operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateNode {
    /// Input whose rows are grouped and aggregated.
    pub input: NodeId,
    /// Group and aggregate expressions. Distributed or partial/final aggregate
    /// states must be represented explicitly in this spec.
    pub spec: AggregateSpec,
}

/// Window operation over one pre-ordered input.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowNode {
    /// Input whose rows receive appended window outputs.
    pub input: NodeId,
    /// Window partition/order/function contract.
    pub spec: WindowSpec,
}

/// Native window contract.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    /// Partition expressions. Current executable forms require column
    /// expressions and treat null partition keys as equal.
    pub partition_by: Vec<Expr>,
    /// Window ordering. Ranking functions require at least one ordering key.
    pub order_by: Vec<SortExpr>,
    /// Window functions appended to the input schema in order.
    pub functions: Vec<WindowFunction>,
}

impl WindowSpec {
    pub fn new(partition_by: Vec<Expr>, order_by: Vec<SortExpr>, functions: Vec<WindowFunction>) -> Self {
        Self { partition_by, order_by, functions }
    }
}

/// Supported native window functions.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFunction {
    /// Full-partition aggregate window. The frame must be constant for every
    /// row in the partition.
    Aggregate { function: AggregateFunction, expr: Expr, alias: String, frame: WindowFrame },
    /// SQL `rank()` over the window peer groups.
    Rank { alias: String },
}

/// Supported native aggregate window frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrame {
    /// `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.
    WholePartition,
    /// `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
    RowsUnboundedPrecedingToCurrent,
}

impl WindowFunction {
    pub fn aggregate(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self::Aggregate { function, expr, alias: alias.into(), frame: WindowFrame::WholePartition }
    }

    pub fn cumulative_aggregate(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self::Aggregate { function, expr, alias: alias.into(), frame: WindowFrame::RowsUnboundedPrecedingToCurrent }
    }

    pub fn rank(alias: impl Into<String>) -> Self {
        Self::Rank { alias: alias.into() }
    }

    pub fn alias(&self) -> &str {
        match self {
            Self::Aggregate { alias, .. } | Self::Rank { alias } => alias,
        }
    }
}

/// Sort operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct SortNode {
    /// Input whose rows are sorted.
    pub input: NodeId,
    /// Sort expressions and optional fetch.
    pub spec: SortSpec,
}

/// Limit operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitNode {
    /// Input whose rows are limited.
    pub input: NodeId,
    /// Skip/fetch contract.
    pub spec: LimitSpec,
}

/// Row-wise concatenation of schema-compatible inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct UnionNode {
    /// Inputs concatenated in order.
    pub inputs: Vec<NodeId>,
}
