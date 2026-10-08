//! Plan-shape accessors shared by optimizer rules.
//!
//! Each helper reads one node or one expression and answers a structural
//! question: which local Parquet source sits at a node, which nodes consume it,
//! which input column a projection output renames, which file columns a
//! predicate reads. They hold no rule state and perform no rewrite.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    expr::{ColumnRef, Expr},
    plan::{JoinNode, JoinOutputColumn, NodeId, PlanNode, QueryNode, QueryPlan},
    source::{LocalParquetSource, SourceColumnRef, SourceNode},
};

/// Returns the local Parquet source at a plan node, if the node has that shape.
pub fn local_parquet_source(plan: &QueryPlan, node_id: NodeId) -> Option<&LocalParquetSource> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else {
        return None;
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        return None;
    };
    Some(source)
}

/// The local Parquet source at `node_id` in a node list under in-place edit.
/// The id check rejects a list whose ids no longer match positions, so a rule
/// cannot narrow the wrong source after a rebuild.
pub(super) fn mutable_local_parquet_source(
    nodes: &mut [QueryNode],
    node_id: NodeId,
) -> Option<&mut LocalParquetSource> {
    let node = nodes.get_mut(node_id.index())?;
    if node.id != node_id {
        return None;
    }
    let PlanNode::Source(source) = &mut node.kind else {
        return None;
    };
    let SourceNode::LocalParquet(source) = source.as_mut() else {
        return None;
    };
    Some(source)
}

/// Consumers of each node in plan order. A node that reads one input twice
/// appears twice, and the sink is not a consumer.
pub(super) fn consumers_by_node(plan: &QueryPlan) -> BTreeMap<NodeId, Vec<NodeId>> {
    let mut consumers = BTreeMap::<NodeId, Vec<NodeId>>::new();
    for node in plan.nodes() {
        for input in node.input_ids() {
            consumers.entry(*input).or_default().push(node.id);
        }
    }
    consumers
}

/// The input column a projection output passes through under any number of
/// renames, or `None` for a computed output.
pub(super) fn simple_projection_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => simple_projection_column(expr),
        _ => None,
    }
}

/// [`simple_projection_column`] restricted to at most one rename. Key lineage
/// uses it so a nested alias is reported as an expression-derived key instead
/// of being traced.
pub(super) fn simple_projection_column_single_alias(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => match expr.as_ref() {
            Expr::Column(column) => Some(column),
            _ => None,
        },
        _ => None,
    }
}

/// The join output with `output_name`, or `None` when two outputs share the
/// name and lineage by name would be ambiguous.
pub(super) fn unique_join_output_for_name<'a>(join: &'a JoinNode, output_name: &str) -> Option<&'a JoinOutputColumn> {
    let mut matches = join.spec.output_projection.iter().filter(|output| output.output_name == output_name);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// Returns file-column projection roots needed for a key column and predicate.
pub(crate) fn reader_projection_indices(
    source: &LocalParquetSource,
    key_column: &str,
    predicate: &Expr,
) -> Option<BTreeSet<usize>> {
    let mut columns = BTreeSet::new();
    let SourceColumnRef::File(key_index) = source.source_column_ref(key_column)? else {
        return None;
    };
    columns.insert(key_index);
    for column in predicate_columns(predicate) {
        let SourceColumnRef::File(column_index) = source.source_column_ref(&column)? else {
            return None;
        };
        columns.insert(column_index);
    }
    Some(columns)
}

/// File-column indices that `expr` and `predicate` read, or `None` when any
/// referenced column is a partition or missing-null column, which a reader
/// predicate cannot evaluate, or when nothing is referenced.
pub(super) fn reader_projection_indices_for_exprs(
    source: &LocalParquetSource,
    expr: &Expr,
    predicate: Option<&Expr>,
) -> Option<BTreeSet<usize>> {
    let mut columns = BTreeSet::new();
    for column in predicate_columns(expr) {
        let SourceColumnRef::File(column_index) = source.source_column_ref(&column)? else {
            return None;
        };
        columns.insert(column_index);
    }
    if let Some(predicate) = predicate {
        for column in predicate_columns(predicate) {
            let SourceColumnRef::File(column_index) = source.source_column_ref(&column)? else {
                return None;
            };
            columns.insert(column_index);
        }
    }
    (!columns.is_empty()).then_some(columns)
}

/// Returns sorted, deduplicated column names referenced by an expression.
pub(crate) fn predicate_columns(expr: &Expr) -> Vec<String> {
    let mut columns = Vec::new();
    collect_expr_columns(expr, &mut columns);
    columns.sort();
    columns.dedup();
    columns
}

fn collect_expr_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Column(column) => columns.push(column.name().to_string()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Alias { expr: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Trim { expr: inner, .. }
        | Expr::LTrim { expr: inner, .. }
        | Expr::RTrim { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::Abs(inner) => collect_expr_columns(inner, columns),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_expr_columns(left, columns);
            collect_expr_columns(right, columns);
        },
        Expr::InList { expr, list, .. } => {
            collect_expr_columns(expr, columns);
            for item in list {
                collect_expr_columns(item, columns);
            }
        },
        Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::UnaryMath { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. } => collect_expr_columns(expr, columns),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_expr_columns(expr, columns);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_expr_columns(&branch.when, columns);
                collect_expr_columns(&branch.then, columns);
            }
            collect_expr_columns(else_expr, columns);
        },
        Expr::Literal(_) => {},
    }
}
