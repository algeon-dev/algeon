//! Output facts attached to native plan nodes.
//!
//! Facts describe already-established contracts: the output schema and the
//! output ordering. They are not hidden optimizer hints.

use super::{SchemaRef, SortExpr};

/// Output contract of one plan node.
///
/// Construction derives `output_schema` from the node's inputs and payload,
/// except for aggregates, joins and unions, whose frontend-declared schema it
/// checks instead. Changing a fact never requests hidden executor behavior; a
/// behavior change needs an explicit IR field or operator.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanFacts {
    /// Schema produced by the node.
    pub output_schema: SchemaRef,
    /// Known ordering of the node output.
    ///
    /// Empty means no ordering contract is known. Sort nodes populate this
    /// directly, and order-preserving operators may carry it forward.
    pub output_ordering: Vec<SortExpr>,
}

impl PlanFacts {
    pub(crate) fn new(output_schema: SchemaRef) -> Self {
        Self { output_schema, output_ordering: Vec::new() }
    }

    pub(crate) fn with_output_ordering(mut self, output_ordering: Vec<SortExpr>) -> Self {
        self.output_ordering = output_ordering;
        self
    }
}
