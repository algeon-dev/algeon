//! Native query plan node payloads.
//!
//! Node structs carry operator-local contracts. Graph connectivity and id
//! validity are owned by [`QueryPlan`](super::QueryPlan) and validated by
//! execution/planning entrypoints.

#![deny(missing_docs)]

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
    /// Create a projection and derive its output schema from `outputs`.
    pub fn new(input: NodeId, outputs: Vec<ProjectionOutput>) -> Self {
        let output_schema = projection_output_schema(&outputs);
        Self { input, outputs, output_schema }
    }

    /// Projection output schema in output expression order.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    /// Projection expressions in output order.
    pub fn expressions(&self) -> impl Iterator<Item = &Expr> {
        self.outputs.iter().map(|output| &output.expr)
    }

    /// Projection fields in output order.
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
    /// Create one typed projection output.
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
    /// Create a native window specification.
    pub fn new(partition_by: Vec<Expr>, order_by: Vec<SortExpr>, functions: Vec<WindowFunction>) -> Self {
        Self { partition_by, order_by, functions }
    }
}

/// Supported native window functions.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFunction {
    /// Full-partition aggregate window. The frame must be constant for every
    /// row in the partition.
    Aggregate {
        /// Aggregate function evaluated over the frame.
        function: AggregateFunction,
        /// Input expression to aggregate.
        expr: Expr,
        /// Output column name.
        alias: String,
        /// Native frame contract.
        frame: WindowFrame,
    },
    /// SQL `rank()` over the window peer groups.
    Rank {
        /// Output column name.
        alias: String,
    },
    /// SQL `row_number()` over the ordered window partition.
    RowNumber {
        /// Output column name.
        alias: String,
    },
    /// SQL `LAG(expr, offset, default)` or `LEAD(expr, offset, default)` over
    /// the ordered window partition.
    ///
    /// Within each partition, the value column is shifted by `offset` rows in
    /// `direction`; rows that fall outside the partition take `default` (or
    /// `NULL` when `default` is `None`). Execution requires at least one
    /// `ORDER BY` key so the per-partition row order is well-defined.
    ///
    /// `default` and the input expression must share the output Arrow data
    /// type. Type coercion is performed by the planner before lowering
    /// reaches the native IR.
    LagLead {
        /// Input expression supplying the shifted values.
        expr: Expr,
        /// Non-negative shift magnitude. `offset == 0` is a no-op (returns
        /// the input value for every row) and is supported for symmetry.
        offset: u32,
        /// Fill value when the shift overflows the partition. `None` means
        /// `NULL` of the output type.
        default: Option<LiteralValue>,
        /// Output column name.
        alias: String,
        /// Whether the shift moves the value forward (`Lag`) or backward
        /// (`Lead`) within each partition's ordered rows.
        direction: ShiftDirection,
    },
}

/// Direction of a per-partition row shift for [`WindowFunction::LagLead`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDirection {
    /// SQL `LAG(...)`: output row `i` carries the value from row `i - offset`
    /// within the same partition, in window order.
    Lag,
    /// SQL `LEAD(...)`: output row `i` carries the value from row
    /// `i + offset` within the same partition, in window order.
    Lead,
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
    /// Create a full-partition aggregate window function.
    pub fn aggregate(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self::Aggregate { function, expr, alias: alias.into(), frame: WindowFrame::WholePartition }
    }

    /// Create a cumulative aggregate window function from partition start to current row.
    pub fn cumulative_aggregate(function: AggregateFunction, expr: Expr, alias: impl Into<String>) -> Self {
        Self::Aggregate { function, expr, alias: alias.into(), frame: WindowFrame::RowsUnboundedPrecedingToCurrent }
    }

    /// Create a `rank()` window function.
    pub fn rank(alias: impl Into<String>) -> Self {
        Self::Rank { alias: alias.into() }
    }

    /// Create a `row_number()` window function.
    pub fn row_number(alias: impl Into<String>) -> Self {
        Self::RowNumber { alias: alias.into() }
    }

    /// Create a `LAG(expr, offset, default)` window function.
    pub fn lag(expr: Expr, offset: u32, default: Option<LiteralValue>, alias: impl Into<String>) -> Self {
        Self::LagLead { expr, offset, default, alias: alias.into(), direction: ShiftDirection::Lag }
    }

    /// Create a `LEAD(expr, offset, default)` window function.
    pub fn lead(expr: Expr, offset: u32, default: Option<LiteralValue>, alias: impl Into<String>) -> Self {
        Self::LagLead { expr, offset, default, alias: alias.into(), direction: ShiftDirection::Lead }
    }

    /// Output column name for this window function.
    pub fn alias(&self) -> &str {
        match self {
            Self::Aggregate { alias, .. }
            | Self::Rank { alias }
            | Self::RowNumber { alias }
            | Self::LagLead { alias, .. } => alias,
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

/// Plan-visible edge-list canonicalization before a cuGraph algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeNormalizeNode {
    /// Edge-list input.
    pub input: NodeId,
    /// Normalization operations applied to the edge list.
    pub spec: EdgeNormalizeSpec,
}
