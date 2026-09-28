//! Execution-time source refinement of an admitted plan.

use super::{PlanNode, QueryNode, QueryPlan};
use crate::{
    expr::{BinaryOp, Expr},
    source::{LocalParquetSource, SourceNode},
};

impl QueryPlan {
    /// Names the first part of `refined` that differs from this plan by more
    /// than footer pruning may change, or `None` when `refined` only narrows
    /// Parquet row-group selections and appends reader-predicate conjuncts.
    ///
    /// Advisory output bindings are not compared: they carry frontend volume
    /// signals, and a refined source invalidates the bindings that depend on it.
    pub(crate) fn first_change_beyond_source_refinement(&self, refined: &Self) -> Option<String> {
        if self.sink != refined.sink {
            return Some("sink".to_string());
        }
        if self.annotations.dynamic_filter_routes != refined.annotations.dynamic_filter_routes {
            return Some("dynamic_filter_routes".to_string());
        }
        if self.nodes.len() != refined.nodes.len() {
            return Some("node_count".to_string());
        }
        self.nodes
            .iter()
            .zip(&refined.nodes)
            .find(|(admitted, refined)| !node_is_source_refinement(admitted, refined))
            .map(|(admitted, _)| format!("node:{}", admitted.id))
    }
}

fn node_is_source_refinement(admitted: &QueryNode, refined: &QueryNode) -> bool {
    if let (PlanNode::Source(before), PlanNode::Source(after)) = (&admitted.kind, &refined.kind)
        && let (SourceNode::LocalParquet(before), SourceNode::LocalParquet(after)) = (before.as_ref(), after.as_ref())
    {
        return admitted.id == refined.id && admitted.facts == refined.facts && source_is_refinement(before, after);
    }
    admitted == refined
}

fn source_is_refinement(before: &LocalParquetSource, after: &LocalParquetSource) -> bool {
    if before.files.len() != after.files.len()
        || !reader_predicate_extends(before.reader_predicate.as_ref(), after.reader_predicate.as_ref())
    {
        return false;
    }
    let mut expected = before.clone();
    for (file, refined) in expected.files.iter_mut().zip(&after.files) {
        if !row_groups_narrow(file.row_groups.as_deref(), refined.row_groups.as_deref()) {
            return false;
        }
        file.row_groups.clone_from(&refined.row_groups);
        file.row_group_rows.clone_from(&refined.row_group_rows);
    }
    expected.reader_predicate.clone_from(&after.reader_predicate);
    expected.refresh_diagnostics();
    expected == *after
}

/// `after` keeps a subsequence of `before`; `None` selects every row group.
fn row_groups_narrow(before: Option<&[usize]>, after: Option<&[usize]>) -> bool {
    match (before, after) {
        (_, None) => before.is_none(),
        (None, Some(_)) => true,
        (Some(before), Some(after)) => {
            let mut remaining = before.iter();
            after.iter().all(|row_group| remaining.any(|candidate| candidate == row_group))
        },
    }
}

/// Seeding appends each derived conjunct as `existing AND derived`, so a
/// refined predicate keeps the admitted one as its leftmost conjunct.
fn reader_predicate_extends(before: Option<&Expr>, after: Option<&Expr>) -> bool {
    let Some(before) = before else {
        return true;
    };
    let mut current = after;
    while let Some(expr) = current {
        if expr == before {
            return true;
        }
        current = match expr {
            Expr::Binary { left, op: BinaryOp::And, .. } => Some(left),
            _ => None,
        };
    }
    false
}
