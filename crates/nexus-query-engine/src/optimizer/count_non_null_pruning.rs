use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::{Field, Schema, SchemaRef};

use super::rewrite;
use crate::{
    expr::{AggregateExpr, AggregateFunction, Expr},
    plan::{AggregateNode, FilterNode, NodeId, PlanFacts, PlanNode, ProjectionNode, QueryNode, QueryPlan},
    source::{LocalParquetSource, SourceColumnRef, SourceNode},
};

/// Stable rule name used in optimizer diagnostics.
pub const COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME: &str = "count_non_null_column_pruning";

/// Configuration for rewriting footer-proven non-null `COUNT(column)` inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountNonNullColumnPruningOptions {
    /// Whether the pass may rewrite eligible count arguments and prune now-unused source columns.
    pub enabled: bool,
}

impl CountNonNullColumnPruningOptions {
    /// Returns options with count non-null pruning disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for CountNonNullColumnPruningOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for footer-proven count argument rewrites.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CountNonNullColumnPruningDiagnostics {
    /// Number of `COUNT(column)` candidates considered.
    pub considered_count: usize,
    /// Number of aggregate expressions rewritten to count-star semantics.
    pub applied_count: usize,
    /// Per-candidate evidence in traversal order.
    pub candidates: Vec<CountNonNullColumnPruningCandidateDiagnostics>,
}

impl CountNonNullColumnPruningDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate evidence for one count argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountNonNullColumnPruningCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Aggregate node containing the candidate.
    pub aggregate_node_id: NodeId,
    /// Count argument column at the aggregate input boundary.
    pub column: String,
    /// Source node that proves the column non-null, when discovered.
    pub source_node_id: Option<NodeId>,
    /// Physical file-column index used for the null-count proof, when discovered.
    pub file_column_index: Option<usize>,
    /// True when this candidate was rewritten.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<CountNonNullColumnPruningRejectedReason>,
}

/// Reasons a count argument was not rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountNonNullColumnPruningRejectedReason {
    /// The count argument does not trace to a supported source column.
    UnsupportedInputLineage,
    /// Footer metadata did not prove zero nulls in every selected row group.
    NullCountProofUnavailable,
    /// No already-retained aggregate input column proved safe as a row-presence carrier.
    NonNullCarrierUnavailable,
}

pub(crate) struct CountNonNullColumnPruningResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: CountNonNullColumnPruningDiagnostics,
}

pub(crate) fn optimize_count_non_null_column_pruning(
    plan: QueryPlan,
    options: &CountNonNullColumnPruningOptions,
) -> CountNonNullColumnPruningResult {
    if !options.enabled {
        return CountNonNullColumnPruningResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: CountNonNullColumnPruningDiagnostics::default(),
        };
    }
    if !plan.dynamic_filter_routes().is_empty() {
        // This pass compacts node ids. Dynamic-filter route discovery currently
        // runs later, so routed plans should only appear if a caller runs this
        // pass directly or reorders the optimizer. Keep those routes valid by
        // declining the compacting rewrite.
        return CountNonNullColumnPruningResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: CountNonNullColumnPruningDiagnostics::default(),
        };
    }

    let mut rewriter = CountNonNullColumnPruningRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id(), RequiredColumns::All);
    if !rewriter.changed || rewriter.diagnostics.applied_count == 0 {
        let diagnostics = rewriter.diagnostics.clone();
        return CountNonNullColumnPruningResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }

    CountNonNullColumnPruningResult {
        plan: rewrite::finish_compacted_rewritten_plan(rewriter.nodes, new_root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan),
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum RequiredColumns {
    All,
    Only(BTreeSet<String>),
}

impl RequiredColumns {
    fn contains(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(columns) => columns.contains(name),
        }
    }

    fn with_expr_columns(self, expr: &Expr) -> Self {
        match self {
            Self::All => Self::All,
            Self::Only(mut columns) => {
                collect_expr_columns(expr, &mut columns);
                Self::Only(columns)
            },
        }
    }
}

struct CountNonNullColumnPruningRewriter<'a> {
    original: &'a QueryPlan,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<(NodeId, RequiredColumns), NodeId>,
    diagnostics: CountNonNullColumnPruningDiagnostics,
    changed: bool,
}

impl<'a> CountNonNullColumnPruningRewriter<'a> {
    fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics: CountNonNullColumnPruningDiagnostics::default(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId, required: RequiredColumns) -> NodeId {
        if let Some(new_id) = self.memo.get(&(old_id, required.clone())).copied() {
            return new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let (kind, facts) = match &old_node.kind {
            PlanNode::Aggregate(aggregate) => self.rewrite_aggregate(old_id, aggregate, old_node.facts.clone()),
            PlanNode::Projection(projection) => self.rewrite_projection(projection, old_node.facts.clone(), &required),
            PlanNode::Filter(filter) => self.rewrite_filter(filter, old_node.facts.clone(), required.clone()),
            PlanNode::Source(source) => self.rewrite_source(source, old_node.facts.clone(), &required),
            _ => (self.clone_kind_with_all_inputs(&old_node.kind), old_node.facts.clone()),
        };
        let new_id = self.push_node(kind, facts);
        self.memo.insert((old_id, required), new_id);
        new_id
    }

    fn rewrite_aggregate(
        &mut self,
        old_id: NodeId,
        aggregate: &AggregateNode,
        facts: PlanFacts,
    ) -> (PlanNode, PlanFacts) {
        let mut spec = aggregate.spec.clone();
        let row_presence_carrier_columns = row_presence_carrier_columns(&spec.group_exprs);
        for expr in &mut spec.aggr_exprs {
            self.rewrite_count_expr(old_id, aggregate.input, &row_presence_carrier_columns, expr);
        }
        let mut input_columns = BTreeSet::new();
        for expr in &spec.group_exprs {
            collect_expr_columns(expr, &mut input_columns);
        }
        for expr in &spec.aggr_exprs {
            collect_expr_columns(&expr.expr, &mut input_columns);
            if let Some(filter) = &expr.filter {
                collect_expr_columns(filter, &mut input_columns);
            }
        }
        let input = self.rewrite_subtree(aggregate.input, RequiredColumns::Only(input_columns));
        (PlanNode::Aggregate(AggregateNode { input, spec }), facts)
    }

    fn rewrite_count_expr(
        &mut self,
        aggregate_node_id: NodeId,
        aggregate_input: NodeId,
        row_presence_carrier_columns: &[String],
        expr: &mut AggregateExpr,
    ) {
        if expr.function != AggregateFunction::Count || expr.distinct {
            return;
        }
        let Some(column) = plain_column_name(&expr.expr) else {
            return;
        };

        self.diagnostics.considered_count += 1;
        let proof = resolve_non_null_source_file_column(self.original, aggregate_input, column);
        let mut candidate = CountNonNullColumnPruningCandidateDiagnostics {
            rule_name: COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
            aggregate_node_id,
            column: column.to_string(),
            source_node_id: proof.as_ref().map(|proof| proof.source_node_id),
            file_column_index: proof.as_ref().map(|proof| proof.file_column_index),
            applied: false,
            rejected_reason: None,
        };

        match proof {
            Some(proof) if proof.proven_non_null => {
                match count_star_replacement_expr(self.original, aggregate_input, column, row_presence_carrier_columns)
                {
                    Some(replacement) => {
                        expr.expr = replacement;
                        candidate.applied = true;
                        self.diagnostics.applied_count += 1;
                        self.changed = true;
                    },
                    None => {
                        candidate.rejected_reason =
                            Some(CountNonNullColumnPruningRejectedReason::NonNullCarrierUnavailable);
                    },
                }
            },
            Some(_) => {
                candidate.rejected_reason = Some(CountNonNullColumnPruningRejectedReason::NullCountProofUnavailable);
            },
            None => {
                candidate.rejected_reason = Some(CountNonNullColumnPruningRejectedReason::UnsupportedInputLineage);
            },
        }
        self.diagnostics.candidates.push(candidate);
    }

    fn rewrite_projection(
        &mut self,
        projection: &ProjectionNode,
        mut facts: PlanFacts,
        required: &RequiredColumns,
    ) -> (PlanNode, PlanFacts) {
        let outputs = projection
            .outputs
            .iter()
            .filter(|output| required.contains(output.field.name()))
            .cloned()
            .collect::<Vec<_>>();
        if outputs.len() != projection.outputs.len() {
            self.changed = true;
        }
        let mut input_columns = BTreeSet::new();
        for output in &outputs {
            collect_expr_columns(&output.expr, &mut input_columns);
        }
        let input = self.rewrite_subtree(projection.input, RequiredColumns::Only(input_columns));
        let projection = ProjectionNode::new(input, outputs);
        facts.output_schema = projection.output_schema().clone();
        (PlanNode::Projection(projection), facts)
    }

    fn rewrite_filter(
        &mut self,
        filter: &FilterNode,
        mut facts: PlanFacts,
        required: RequiredColumns,
    ) -> (PlanNode, PlanFacts) {
        let input_required = required.with_expr_columns(&filter.predicate);
        let input = self.rewrite_subtree(filter.input, input_required);
        let input_schema = self.nodes[input.index()].output_schema().clone();
        facts.output_schema = input_schema;
        (PlanNode::Filter(FilterNode { input, predicate: filter.predicate.clone() }), facts)
    }

    fn rewrite_source(
        &mut self,
        source: &SourceNode,
        mut facts: PlanFacts,
        required: &RequiredColumns,
    ) -> (PlanNode, PlanFacts) {
        let source = match source {
            SourceNode::LocalParquet(source) => {
                let rewritten = prune_local_parquet_source(source, required);
                if &rewritten != source {
                    self.changed = true;
                }
                SourceNode::LocalParquet(rewritten)
            },
            SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => source.clone(),
        };
        facts.output_schema = source.output_schema().clone();
        (PlanNode::Source(Box::new(source)), facts)
    }

    fn clone_kind_with_all_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input, RequiredColumns::All))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

fn row_presence_carrier_columns(group_exprs: &[Expr]) -> Vec<String> {
    let mut columns = Vec::new();
    let mut seen = BTreeSet::new();
    for expr in group_exprs {
        if let Some(column) = plain_column_name(expr)
            && seen.insert(column.to_string())
        {
            columns.push(column.to_string());
        }
    }
    columns
}

fn count_star_replacement_expr(
    plan: &QueryPlan,
    aggregate_input: NodeId,
    count_column: &str,
    row_presence_carrier_columns: &[String],
) -> Option<Expr> {
    if row_presence_carrier_columns.is_empty() {
        return Some(Expr::literal(crate::expr::LiteralValue::Int64(1)));
    }
    row_presence_carrier_columns.iter().find_map(|carrier| {
        if carrier == count_column {
            return None;
        }
        let proof = resolve_non_null_source_file_column(plan, aggregate_input, carrier)?;
        proof.proven_non_null.then(|| Expr::column(carrier.clone()))
    })
}

#[derive(Debug, Clone)]
struct NonNullSourceProof {
    source_node_id: NodeId,
    file_column_index: usize,
    proven_non_null: bool,
}

fn resolve_non_null_source_file_column(
    plan: &QueryPlan,
    node_id: NodeId,
    column_name: &str,
) -> Option<NonNullSourceProof> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => {
            let SourceNode::LocalParquet(source) = source.as_ref() else {
                return None;
            };
            let SourceColumnRef::File(file_column_index) = source.source_column_ref(column_name)? else {
                return None;
            };
            Some(NonNullSourceProof {
                source_node_id: node_id,
                file_column_index,
                proven_non_null: source.selected_file_column_has_no_nulls(file_column_index),
            })
        },
        PlanNode::Filter(filter) => resolve_non_null_source_file_column(plan, filter.input, column_name),
        PlanNode::Projection(projection) => {
            let output = projection.outputs.iter().find(|output| output.field.name() == column_name)?;
            let input_column = plain_column_name(&output.expr)?;
            resolve_non_null_source_file_column(plan, projection.input, input_column)
        },
        _ => None,
    }
}

fn prune_local_parquet_source(source: &LocalParquetSource, required: &RequiredColumns) -> LocalParquetSource {
    let RequiredColumns::Only(required) = required else {
        return source.clone();
    };
    if source.is_zero_column_count_source() || source.is_count_star_probe_source() {
        return source.clone();
    }

    let mut required = required.clone();
    if let Some(predicate) = &source.reader_predicate {
        collect_expr_columns(predicate, &mut required);
    }

    let required_file_indices = required
        .iter()
        .filter_map(|column| match source.source_column_ref(column) {
            Some(SourceColumnRef::File(index)) => Some(index),
            Some(SourceColumnRef::Partition(_)) | Some(SourceColumnRef::MissingNull(_)) | None => None,
        })
        .collect::<BTreeSet<_>>();

    let mut read_columns = Vec::new();
    let mut read_column_file_indices = Vec::new();
    for (column, file_index) in source.read_columns.iter().zip(source.read_column_file_indices.iter()) {
        if required.contains(column) || required_file_indices.contains(file_index) {
            read_columns.push(column.clone());
            read_column_file_indices.push(*file_index);
        }
    }
    let projected_output_columns =
        source.projected_output_columns.iter().filter(|column| required.contains(*column)).cloned().collect::<Vec<_>>();
    let pure_filter_columns =
        source.pure_filter_columns.iter().filter(|column| required.contains(*column)).cloned().collect::<Vec<_>>();
    let retained_partition_indices = source
        .partition_columns
        .iter()
        .enumerate()
        .filter(|(_, column)| required.contains(column.name.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let retained_missing_indices = source
        .missing_columns
        .iter()
        .enumerate()
        .filter(|(_, column)| required.contains(column.name.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let partition_index_map = retained_index_map(&retained_partition_indices);
    let missing_index_map = retained_index_map(&retained_missing_indices);
    let Some(projected_output_column_source_refs) =
        remapped_source_refs(source, &projected_output_columns, &partition_index_map, &missing_index_map)
    else {
        return source.clone();
    };
    let Some(pure_filter_column_source_refs) =
        remapped_source_refs(source, &pure_filter_columns, &partition_index_map, &missing_index_map)
    else {
        return source.clone();
    };
    let partition_columns =
        retained_partition_indices.iter().map(|index| source.partition_columns[*index].clone()).collect::<Vec<_>>();
    let missing_columns =
        retained_missing_indices.iter().map(|index| source.missing_columns[*index].clone()).collect::<Vec<_>>();
    let Some(files) = pruned_partition_values(source, &retained_partition_indices) else {
        return source.clone();
    };

    if read_columns == source.read_columns
        && projected_output_columns == source.projected_output_columns
        && pure_filter_columns == source.pure_filter_columns
        && partition_columns == source.partition_columns
        && missing_columns == source.missing_columns
    {
        return source.clone();
    }
    if read_columns.is_empty() && partition_columns.is_empty() && missing_columns.is_empty() {
        return source.clone();
    }

    let Some(schema) = pruned_source_schema(
        source,
        &read_columns,
        &projected_output_columns,
        &pure_filter_columns,
        &partition_columns,
        &missing_columns,
    ) else {
        return source.clone();
    };
    let Some(output_schema_override) = pruned_output_schema_override(
        source,
        &projected_output_columns,
        &pure_filter_columns,
        &partition_columns,
        &missing_columns,
    ) else {
        return source.clone();
    };
    let mut pruned = source.clone();
    pruned.files = files;
    pruned.schema = schema;
    pruned.output_schema_override = output_schema_override;
    pruned.read_columns = read_columns;
    pruned.read_column_file_indices = read_column_file_indices;
    pruned.projected_output_columns = projected_output_columns;
    pruned.projected_output_column_source_refs = projected_output_column_source_refs;
    pruned.pure_filter_columns = pure_filter_columns;
    pruned.pure_filter_column_source_refs = pure_filter_column_source_refs;
    pruned.partition_columns = partition_columns;
    pruned.missing_columns = missing_columns;
    pruned.refresh_diagnostics();
    pruned
}

fn retained_index_map(retained_indices: &[usize]) -> BTreeMap<usize, usize> {
    retained_indices.iter().enumerate().map(|(new_index, old_index)| (*old_index, new_index)).collect()
}

fn remapped_source_refs(
    source: &LocalParquetSource,
    columns: &[String],
    partition_index_map: &BTreeMap<usize, usize>,
    missing_index_map: &BTreeMap<usize, usize>,
) -> Option<Vec<SourceColumnRef>> {
    columns
        .iter()
        .map(|column| remap_source_ref(source.source_column_ref(column)?, partition_index_map, missing_index_map))
        .collect()
}

fn remap_source_ref(
    source_ref: SourceColumnRef,
    partition_index_map: &BTreeMap<usize, usize>,
    missing_index_map: &BTreeMap<usize, usize>,
) -> Option<SourceColumnRef> {
    match source_ref {
        SourceColumnRef::File(index) => Some(SourceColumnRef::File(index)),
        SourceColumnRef::Partition(index) => partition_index_map.get(&index).copied().map(SourceColumnRef::Partition),
        SourceColumnRef::MissingNull(index) => missing_index_map.get(&index).copied().map(SourceColumnRef::MissingNull),
    }
}

fn pruned_partition_values(
    source: &LocalParquetSource,
    retained_partition_indices: &[usize],
) -> Option<Vec<crate::source::LocalParquetFile>> {
    source
        .files
        .iter()
        .map(|file| {
            let mut file = file.clone();
            if !source.partition_columns.is_empty() {
                file.partition_values = retained_partition_indices
                    .iter()
                    .map(|index| file.partition_values.get(*index).cloned())
                    .collect::<Option<Vec<_>>>()?;
            }
            Some(file)
        })
        .collect()
}

fn pruned_source_schema(
    source: &LocalParquetSource,
    read_columns: &[String],
    projected_output_columns: &[String],
    pure_filter_columns: &[String],
    partition_columns: &[crate::source::PartitionColumn],
    missing_columns: &[crate::source::MissingColumn],
) -> Option<SchemaRef> {
    let mut fields = Vec::with_capacity(
        read_columns.len()
            + projected_output_columns.len()
            + pure_filter_columns.len()
            + partition_columns.len()
            + missing_columns.len(),
    );
    let mut seen = BTreeSet::new();
    for column in read_columns {
        let field = source.schema.field_with_name(column).ok()?.as_ref().clone();
        if seen.insert(field.name().clone()) {
            fields.push(field);
        }
    }
    if let Some(output_schema) = source.output_schema_override.as_ref() {
        for column in projected_output_columns.iter().chain(pure_filter_columns.iter()) {
            let field = output_schema.field_with_name(column).ok()?.as_ref().clone();
            if seen.insert(field.name().clone()) {
                fields.push(field);
            }
        }
    }
    for column in partition_columns {
        if seen.insert(column.name.clone()) {
            fields.push(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable));
        }
    }
    for column in missing_columns {
        if seen.insert(column.name.clone()) {
            fields.push(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable));
        }
    }
    Some(std::sync::Arc::new(Schema::new(fields)))
}

fn pruned_output_schema_override(
    source: &LocalParquetSource,
    projected_output_columns: &[String],
    pure_filter_columns: &[String],
    partition_columns: &[crate::source::PartitionColumn],
    missing_columns: &[crate::source::MissingColumn],
) -> Option<Option<SchemaRef>> {
    let Some(output_schema) = source.output_schema_override.as_ref() else {
        return Some(None);
    };
    let mut fields = Vec::with_capacity(
        projected_output_columns.len() + pure_filter_columns.len() + partition_columns.len() + missing_columns.len(),
    );
    let mut seen = BTreeSet::new();
    for column in projected_output_columns.iter().chain(pure_filter_columns.iter()) {
        let field = output_schema.field_with_name(column).ok()?.as_ref().clone();
        if seen.insert(field.name().clone()) {
            fields.push(field);
        }
    }
    for column in partition_columns {
        if seen.insert(column.name.clone()) {
            fields.push(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable));
        }
    }
    for column in missing_columns {
        if seen.insert(column.name.clone()) {
            fields.push(Field::new(column.name.as_str(), column.data_type.clone(), column.nullable));
        }
    }
    Some(Some(std::sync::Arc::new(Schema::new(fields))))
}

fn plain_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) if column.qualifier().is_none() => Some(column.name()),
        Expr::Alias { expr, .. } => plain_column_name(expr),
        _ => None,
    }
}

fn collect_expr_columns(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => {
            if column.qualifier().is_none() {
                out.insert(column.name().to_string());
            }
        },
        Expr::Literal(_) => {},
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Trim { expr: inner, .. }
        | Expr::LTrim { expr: inner, .. }
        | Expr::RTrim { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::Abs(inner) => collect_expr_columns(inner, out),
        Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => collect_expr_columns(expr, out),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_expr_columns(left, out);
            collect_expr_columns(right, out);
        },
        Expr::Concat(parts) => {
            for part in parts {
                collect_expr_columns(part, out);
            }
        },
        Expr::InList { expr, list, .. } => {
            collect_expr_columns(expr, out);
            for item in list {
                collect_expr_columns(item, out);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_expr_columns(&branch.when, out);
                collect_expr_columns(&branch.then, out);
            }
            collect_expr_columns(else_expr, out);
        },
    }
}
