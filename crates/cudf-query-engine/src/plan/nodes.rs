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
