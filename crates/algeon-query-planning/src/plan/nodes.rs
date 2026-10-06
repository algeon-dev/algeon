//! Native query plan node payloads.
//!
//! Node structs carry operator-local contracts. Graph connectivity and id
//! validity are owned by [`QueryPlan`](super::QueryPlan) and validated by
//! execution/planning entrypoints.
use std::sync::Arc;

use super::{
    AggregateFunction, AggregateSpec, DataType, EdgeNormalizeSpec, Expr, Field, LimitSpec, LiteralValue, NodeId,
    Schema, SchemaRef, SortExpr, SortSpec,
};

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
    pub(crate) fn new(input: NodeId, outputs: Vec<ProjectionOutput>) -> Self {
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

    /// Whether a function numbers rows by their position in the input: rank
    /// under an ORDER BY, and row-number under an ORDER BY or a PARTITION BY.
    /// Positional numbering needs each partition's rows adjacent, in the
    /// window's order. Cumulative aggregates and LAG/LEAD sort by the window
    /// keys themselves.
    pub(crate) fn counts_rows_in_input_order(&self) -> bool {
        self.functions.iter().any(|function| match function {
            WindowFunction::Rank { .. } => !self.order_by.is_empty(),
            WindowFunction::RowNumber { .. } => !self.order_by.is_empty() || !self.partition_by.is_empty(),
            _ => false,
        })
    }
}

/// Supported native window functions.
///
/// Each function owns the output field it appends to the window input.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFunction {
    /// Full-partition aggregate window. The frame must be constant for every
    /// row in the partition.
    Aggregate {
        /// Aggregate function evaluated over the frame.
        function: AggregateFunction,
        /// Input expression to aggregate.
        expr: Expr,
        /// Output field. The frontend declares the aggregate's result type and
        /// nullability.
        output: Field,
        /// Native frame contract.
        frame: WindowFrame,
    },
    /// SQL `rank()` over the window peer groups. Its output is a non-nullable
    /// UInt64.
    Rank {
        /// Output field.
        output: Field,
    },
    /// SQL `row_number()` over the ordered window partition. Its output is a
    /// non-nullable UInt64; internal dense-id mapping may request Int32 or
    /// Int64 for downstream native consumers.
    RowNumber {
        /// Output field.
        output: Field,
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
        /// Output field, typed like the input expression.
        output: Field,
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
    pub fn aggregate(function: AggregateFunction, expr: Expr, output: Field) -> Self {
        Self::Aggregate { function, expr, output, frame: WindowFrame::WholePartition }
    }

    /// Create a cumulative aggregate window function from partition start to current row.
    pub fn cumulative_aggregate(function: AggregateFunction, expr: Expr, output: Field) -> Self {
        Self::Aggregate { function, expr, output, frame: WindowFrame::RowsUnboundedPrecedingToCurrent }
    }

    /// Create a `rank()` window function.
    pub fn rank(alias: impl Into<String>) -> Self {
        Self::Rank { output: Field::new(alias, DataType::UInt64, false) }
    }

    /// Create a `row_number()` window function.
    pub fn row_number(alias: impl Into<String>) -> Self {
        Self::RowNumber { output: Field::new(alias, DataType::UInt64, false) }
    }

    /// Create a one-based `row_number()` with an Int64 output.
    pub fn row_number_i64(alias: impl Into<String>) -> Self {
        Self::RowNumber { output: Field::new(alias, DataType::Int64, false) }
    }

    /// Create a one-based `row_number()` with an Int32 output. Execution fails
    /// with a size error when the input has more than `i32::MAX` rows.
    pub fn row_number_i32(alias: impl Into<String>) -> Self {
        Self::RowNumber { output: Field::new(alias, DataType::Int32, false) }
    }

    /// Create a `LAG(expr, offset, default)` window function.
    pub fn lag(expr: Expr, offset: u32, default: Option<LiteralValue>, output: Field) -> Self {
        Self::LagLead { expr, offset, default, output, direction: ShiftDirection::Lag }
    }

    /// Create a `LEAD(expr, offset, default)` window function.
    pub fn lead(expr: Expr, offset: u32, default: Option<LiteralValue>, output: Field) -> Self {
        Self::LagLead { expr, offset, default, output, direction: ShiftDirection::Lead }
    }

    /// The field this function appends to the window input.
    pub fn output(&self) -> &Field {
        match self {
            Self::Aggregate { output, .. }
            | Self::Rank { output }
            | Self::RowNumber { output }
            | Self::LagLead { output, .. } => output,
        }
    }

    /// Output column name for this window function.
    pub fn alias(&self) -> &str {
        self.output().name()
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
