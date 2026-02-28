use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::rewrite;
use crate::{
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr},
    plan::{
        AggregateNode, FilterNode, JoinOutputColumn, JoinSide, NodeId, PlanFacts, PlanNode, ProjectionNode,
        ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
    },
    source::{LocalParquetSource, ParquetSourceLocation, SourceColumnRef, SourceNode},
};

/// Stable rule name used in optimizer diagnostics.
pub const SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME: &str = "shared_subplan_materialization";

/// Configuration for shared-subplan materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedSubplanMaterializationOptions {
    /// Whether the pass may rewrite repeated native subplans.
    pub enabled: bool,
}

impl SharedSubplanMaterializationOptions {
    /// Returns options with shared-subplan materialization disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for SharedSubplanMaterializationOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for shared-subplan materialization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedSubplanMaterializationDiagnostics {
    /// Number of grouped aggregate nodes considered as possible materialization roots.
    pub considered_count: usize,
    /// Number of repeated aggregate families discovered.
    pub candidate_count: usize,
    /// Number of repeated aggregate families rewritten into shared materializations.
    pub applied_count: usize,
    /// Per-candidate evidence in optimizer order.
    pub candidates: Vec<SharedSubplanMaterializationCandidateDiagnostics>,
}

impl SharedSubplanMaterializationDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.candidate_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.candidate_count += other.candidate_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate repeated-subplan materialization evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedSubplanMaterializationCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Stable fingerprint for the repeated aggregate family.
    pub family_key: String,
    /// Aggregate nodes that belong to this repeated family.
    pub aggregate_node_ids: Vec<NodeId>,
    /// Number of aggregate nodes in the repeated family.
    pub reuse_count: usize,
    /// Grouping expressions used by the aggregate family.
    pub group_exprs: Vec<String>,
    /// Aggregate expressions used by the aggregate family.
    pub aggregate_exprs: Vec<String>,
    /// Abstract local-source fingerprints used for family grouping.
    pub source_fingerprints: Vec<String>,
    /// Number of exact source variants after retaining reader-predicate text.
    pub exact_source_variant_count: usize,
    /// Sum of selected source bytes across all repeated branches, if known.
    pub repeated_selected_source_bytes: Option<u64>,
    /// Conservative source-byte duplication estimate: total repeated bytes minus the largest branch.
    pub duplicate_selected_source_bytes: Option<u64>,
    /// True when the returned plan carries the shared materialization rewrite.
    pub applied: bool,
    /// Rejection reason when the candidate was discovered but not rewritten.
    pub rejected_reason: Option<SharedSubplanMaterializationRejectedReason>,
}

/// Reasons a shared-subplan materialization candidate was not rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedSubplanMaterializationRejectedReason {
    /// Branch source predicates could not be reconstructed after the aggregate.
    PredicateNotPostAggregateFilterable,
    /// The repeated scalar aggregate shape is not safe to combine into one filtered aggregate.
    UnsupportedScalarAggregateShape,
    /// Branch predicate columns could not be carried to the shared aggregate input.
    PredicateColumnPropagationUnsupported,
    /// The candidate was not reachable from the query sink.
    NotReachableFromSink,
}

pub(crate) struct SharedSubplanMaterializationResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: SharedSubplanMaterializationDiagnostics,
}

pub(crate) fn optimize_shared_subplan_materialization(
    plan: QueryPlan,
    options: &SharedSubplanMaterializationOptions,
) -> SharedSubplanMaterializationResult {
    if !options.enabled {
        return SharedSubplanMaterializationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: SharedSubplanMaterializationDiagnostics::default(),
        };
    }

    let scalar_derivation = optimize_scalar_aggregate_from_grouped_aggregate(plan);
    let discovery = discover_shared_subplan_materialization_candidates(&scalar_derivation.plan);
    if discovery.families.is_empty() {
        if scalar_derivation.change.changed() || !scalar_derivation.diagnostics.is_empty() {
            return scalar_derivation;
        }
        return SharedSubplanMaterializationResult {
            plan: scalar_derivation.plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let root_id = scalar_derivation.plan.root_id();
    let sink = scalar_derivation.plan.sink().clone();
    let mut rewriter =
        SharedSubplanMaterializationRewriter::new(&scalar_derivation.plan, discovery.families, discovery.diagnostics);
    let new_root = rewriter.rewrite_subtree(root_id);
    let mut diagnostics = scalar_derivation.diagnostics;
    diagnostics.merge(rewriter.diagnostics);
    let changed = rewriter.changed;
    let new_sink = rewrite::sink_with_input(&sink, new_root);
    let rewritten_plan = QueryPlan::from_rewritten_parts(rewriter.nodes, new_root, new_sink);
    let change = if changed || scalar_derivation.change.changed() {
        super::NativeOptimizerChange::Changed
    } else {
        super::NativeOptimizerChange::Unchanged
    };
    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
    SharedSubplanMaterializationResult { plan: rewritten_plan, change, diagnostics }
}

#[derive(Debug, Clone)]
struct ScalarAggregateDerivationRewrite {
    scalar_aggregate_node_id: NodeId,
    grouped_aggregate_node_id: NodeId,
    derived_spec: AggregateSpec,
}

fn optimize_scalar_aggregate_from_grouped_aggregate(plan: QueryPlan) -> SharedSubplanMaterializationResult {
    let discovery = discover_scalar_aggregate_from_grouped_aggregate_candidates(&plan);
    if discovery.rewrites.is_empty() {
        return SharedSubplanMaterializationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let root = plan.root_id();
    let sink = plan.sink().clone();
    let mut rewriter = ScalarAggregateDerivationRewriter::new(&plan, discovery.rewrites);
    let new_root = rewriter.rewrite_subtree(root);
    let new_sink = rewrite::sink_with_input(&sink, new_root);
    let rewritten_plan = QueryPlan::from_rewritten_parts(rewriter.nodes, new_root, new_sink);
    SharedSubplanMaterializationResult {
        plan: rewritten_plan,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: discovery.diagnostics,
    }
}

struct ScalarAggregateDerivationDiscovery {
    diagnostics: SharedSubplanMaterializationDiagnostics,
    rewrites: Vec<ScalarAggregateDerivationRewrite>,
}

fn discover_scalar_aggregate_from_grouped_aggregate_candidates(plan: &QueryPlan) -> ScalarAggregateDerivationDiscovery {
    let reachable = rewrite::sink_reachable_nodes(plan);
    let aggregates = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if reachable.contains(&node.id) => Some((node.id, aggregate)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let grouped =
        aggregates.iter().filter(|(_, aggregate)| derivable_grouped_aggregate(aggregate)).copied().collect::<Vec<_>>();
    let mut rowset_signatures = BTreeMap::new();
    let mut diagnostics = SharedSubplanMaterializationDiagnostics::default();
    let mut rewrites = Vec::new();

    for (scalar_id, scalar_aggregate) in aggregates {
        if !derivable_scalar_aggregate(scalar_aggregate) {
            continue;
        }
        diagnostics.considered_count += 1;
        let Some(scalar_rowset_signature) =
            rowset_signature(plan, scalar_aggregate.input, &mut rowset_signatures, &mut BTreeSet::new())
        else {
            continue;
        };
        for (grouped_id, grouped_aggregate) in &grouped {
            if *grouped_id == scalar_id {
                continue;
            }
            let Some(grouped_rowset_signature) =
                rowset_signature(plan, grouped_aggregate.input, &mut rowset_signatures, &mut BTreeSet::new())
            else {
                continue;
            };
            if grouped_rowset_signature != scalar_rowset_signature {
                continue;
            }
            let Some(derived_spec) =
                derived_scalar_aggregate_spec(plan, *grouped_id, grouped_aggregate, scalar_id, scalar_aggregate)
            else {
                continue;
            };

            let repeated_selected_source_bytes = sum_optional_u64([
                selected_source_bytes_below(plan, grouped_aggregate.input),
                selected_source_bytes_below(plan, scalar_aggregate.input),
            ]);
            let duplicate_selected_source_bytes = repeated_selected_source_bytes
                .zip(
                    selected_source_bytes_below(plan, grouped_aggregate.input)
                        .max(selected_source_bytes_below(plan, scalar_aggregate.input)),
                )
                .map(|(total, largest)| total.saturating_sub(largest));
            let exact_source_variant_count =
                exact_source_variant_count_for_inputs(plan, grouped_aggregate.input, scalar_aggregate.input);

            diagnostics.candidate_count += 1;
            diagnostics.applied_count += 1;
            diagnostics.candidates.push(SharedSubplanMaterializationCandidateDiagnostics {
                rule_name: SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
                family_key: format!(
                    "shared-grouped-scalar-v1-fnv64-{:016x}",
                    stable_fnv1a64(scalar_rowset_signature.as_bytes())
                ),
                aggregate_node_ids: vec![*grouped_id, scalar_id],
                reuse_count: 2,
                group_exprs: grouped_aggregate.spec.group_exprs.iter().map(ToString::to_string).collect(),
                aggregate_exprs: scalar_aggregate.spec.aggr_exprs.iter().map(aggregate_expr_signature).collect(),
                source_fingerprints: vec![scalar_rowset_signature],
                exact_source_variant_count,
                repeated_selected_source_bytes,
                duplicate_selected_source_bytes,
                applied: true,
                rejected_reason: None,
            });
            rewrites.push(ScalarAggregateDerivationRewrite {
                scalar_aggregate_node_id: scalar_id,
                grouped_aggregate_node_id: *grouped_id,
                derived_spec,
            });
            break;
        }
    }

    ScalarAggregateDerivationDiscovery { diagnostics, rewrites }
}

fn derivable_grouped_aggregate(aggregate: &AggregateNode) -> bool {
    !aggregate.spec.group_exprs.is_empty()
        && aggregate.spec.grouping_sets.is_none()
        && aggregate.spec.aggr_exprs.iter().any(derivable_sum_expr)
}

fn derivable_scalar_aggregate(aggregate: &AggregateNode) -> bool {
    aggregate.spec.group_exprs.is_empty()
        && aggregate.spec.grouping_sets.is_none()
        && !aggregate.spec.aggr_exprs.is_empty()
        && aggregate.spec.aggr_exprs.iter().all(derivable_sum_expr)
}

fn derivable_sum_expr(expr: &AggregateExpr) -> bool {
    expr.function == AggregateFunction::Sum && !expr.distinct && expr.filter.is_none()
}

fn derived_scalar_aggregate_spec(
    plan: &QueryPlan,
    grouped_aggregate_node_id: NodeId,
    grouped_aggregate: &AggregateNode,
    scalar_aggregate_node_id: NodeId,
    scalar_aggregate: &AggregateNode,
) -> Option<AggregateSpec> {
    let mut used_group_aliases = BTreeSet::new();
    let mut aggr_exprs = Vec::with_capacity(scalar_aggregate.spec.aggr_exprs.len());
    for scalar_expr in &scalar_aggregate.spec.aggr_exprs {
        if !sum_output_is_exact(plan, scalar_aggregate_node_id, &scalar_expr.alias) {
            return None;
        }
        let scalar_arg_signature =
            normalized_aggregate_argument_signature(plan, scalar_aggregate.input, &scalar_expr.expr);
        let grouped_expr = grouped_aggregate.spec.aggr_exprs.iter().find(|grouped_expr| {
            derivable_sum_expr(grouped_expr)
                && grouped_expr.function == scalar_expr.function
                && sum_output_is_exact(plan, grouped_aggregate_node_id, &grouped_expr.alias)
                && normalized_aggregate_argument_signature(plan, grouped_aggregate.input, &grouped_expr.expr)
                    == scalar_arg_signature
                && used_group_aliases.insert(grouped_expr.alias.clone())
        })?;
        aggr_exprs.push(AggregateExpr::new(
            AggregateFunction::Sum,
            Expr::column(grouped_expr.alias.clone()),
            scalar_expr.alias.clone(),
        ));
    }
    Some(AggregateSpec::new(Vec::new(), aggr_exprs))
}

fn sum_output_is_exact(plan: &QueryPlan, aggregate_node_id: NodeId, alias: &str) -> bool {
    let Some(data_type) = plan.node(aggregate_node_id).and_then(|node| {
        node.output_schema().fields().iter().find(|field| field.name() == alias).map(|field| field.data_type())
    }) else {
        return false;
    };
    !matches!(data_type, DataType::Float16 | DataType::Float32 | DataType::Float64)
}

fn normalized_aggregate_argument_signature(plan: &QueryPlan, input: NodeId, expr: &Expr) -> String {
    if let Expr::Column(column) = expr
        && let Some(node) = plan.node(input)
        && let PlanNode::Projection(projection) = &node.kind
        && let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column.name())
    {
        return output.expr.to_string();
    }
    expr.to_string()
}

fn rowset_signature(
    plan: &QueryPlan,
    node_id: NodeId,
    memo: &mut BTreeMap<NodeId, Option<String>>,
    visiting: &mut BTreeSet<NodeId>,
) -> Option<String> {
    if let Some(signature) = memo.get(&node_id) {
        return signature.clone();
    }
    if !visiting.insert(node_id) {
        memo.insert(node_id, None);
        return None;
    }
    let signature = rowset_signature_uncached(plan, node_id, memo, visiting);
    visiting.remove(&node_id);
    memo.insert(node_id, signature.clone());
    signature
}

fn rowset_signature_uncached(
    plan: &QueryPlan,
    node_id: NodeId,
    memo: &mut BTreeMap<NodeId, Option<String>>,
    visiting: &mut BTreeSet<NodeId>,
) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => Some(local_parquet_rowset_signature(source)),
            SourceNode::GpuResident(source) => Some(format!("gpu_resident:{}", source.input_name)),
            SourceNode::LiteralBatch(source) => Some(format!("literal_batch:{:?}", source.batches)),
        },
        PlanNode::Filter(filter) => rowset_signature(plan, filter.input, memo, visiting)
            .map(|input| format!("filter({input};predicate={})", filter.predicate)),
        PlanNode::Projection(projection) => rowset_signature(plan, projection.input, memo, visiting),
        PlanNode::Join(join) => {
            if !matches!(join.spec.join_type, crate::plan::JoinType::Inner | crate::plan::JoinType::Cross) {
                return None;
            }
            let left = rowset_signature(plan, join.inputs[0], memo, visiting)?;
            let right = rowset_signature(plan, join.inputs[1], memo, visiting)?;
            let keys =
                join.spec.keys.iter().map(|key| format!("{}={}", key.left, key.right)).collect::<Vec<_>>().join(",");
            let residual =
                join.spec.residual_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
            Some(format!(
                "join(type={},keys={keys},residual={residual},nulls={:?},left={left},right={right})",
                join.spec.join_type, join.spec.null_equality
            ))
        },
        PlanNode::Sort(sort) => rowset_signature(plan, sort.input, memo, visiting),
        _ => None,
    }
}

fn local_parquet_rowset_signature(source: &LocalParquetSource) -> String {
    let mut file_signatures = source.files.iter().map(local_parquet_file_rowset_signature).collect::<Vec<_>>();
    file_signatures.sort();
    let files = file_signatures.join("+");
    let predicate = source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
    format!(
        "local_parquet(files={files};predicate={predicate};scan_limit={};max_rows={};zero_column_count={})",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        source.max_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string()),
        source.is_zero_column_count_source(),
    )
}

fn local_parquet_file_rowset_signature(file: &crate::source::LocalParquetFile) -> String {
    let row_groups = file
        .row_groups
        .as_ref()
        .map(|row_groups| row_groups.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        .unwrap_or_else(|| "all".to_string());
    format!(
        "stable_id={};location={};row_groups={};partition_values={:?}",
        file.stable_id,
        file.location.reader_path(),
        row_groups,
        file.partition_values
    )
}

fn selected_source_bytes_below(plan: &QueryPlan, input: NodeId) -> Option<u64> {
    let mut sources = Vec::new();
    collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
    sum_optional_u64(sources.iter().map(|source| source.selected_source_bytes))
}

fn exact_source_variant_count_for_inputs(plan: &QueryPlan, left: NodeId, right: NodeId) -> usize {
    let mut variants = BTreeSet::new();
    for input in [left, right] {
        let mut sources = Vec::new();
        collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
        variants.insert(sources.iter().map(|source| source.exact_fingerprint.as_str()).collect::<Vec<_>>().join("|"));
    }
    variants.len()
}

struct ScalarAggregateDerivationRewriter<'a> {
    original: &'a QueryPlan,
    replacements: BTreeMap<NodeId, ScalarAggregateDerivationRewrite>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
}

impl<'a> ScalarAggregateDerivationRewriter<'a> {
    fn new(original: &'a QueryPlan, rewrites: Vec<ScalarAggregateDerivationRewrite>) -> Self {
        Self {
            original,
            replacements: rewrites.into_iter().map(|rewrite| (rewrite.scalar_aggregate_node_id, rewrite)).collect(),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = if let Some(replacement) = self.replacements.get(&old_id).cloned() {
            let input = self.rewrite_subtree(replacement.grouped_aggregate_node_id);
            PlanNode::Aggregate(AggregateNode { input, spec: replacement.derived_spec })
        } else {
            self.clone_kind_with_rewritten_inputs(&old_node.kind)
        };
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

struct SharedSubplanMaterializationDiscovery {
    diagnostics: SharedSubplanMaterializationDiagnostics,
    families: Vec<SharedAggregateFamilyRewrite>,
}

fn discover_shared_subplan_materialization_candidates(plan: &QueryPlan) -> SharedSubplanMaterializationDiscovery {
    let mut diagnostics = SharedSubplanMaterializationDiagnostics::default();
    let mut families = BTreeMap::<String, Vec<AggregateMaterializationSignature>>::new();

    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if aggregate.spec.grouping_sets.is_some() {
            continue;
        }
        if aggregate.spec.group_exprs.is_empty() && !scalar_filtered_aggregate_supported(&aggregate.spec) {
            continue;
        }
        let Some(signature) = aggregate_materialization_signature(plan, node.id, aggregate.input, &aggregate.spec)
        else {
            continue;
        };
        diagnostics.considered_count += 1;
        families.entry(signature.family_key.clone()).or_default().push(signature);
    }

    let mut rewrite_families = Vec::new();
    for mut family in families.into_values() {
        if family.len() < 2 {
            continue;
        }
        family.sort_by_key(|signature| signature.aggregate_node_id);
        let representative = &family[0];
        let repeated_selected_source_bytes = sum_optional_u64(family.iter().map(|signature| signature.source_bytes));
        let duplicate_selected_source_bytes = repeated_selected_source_bytes.map(|total| {
            let largest = family.iter().filter_map(|signature| signature.source_bytes).max().unwrap_or(0);
            total.saturating_sub(largest)
        });
        let exact_source_variant_count =
            family.iter().map(|signature| signature.exact_source_signature.as_str()).collect::<BTreeSet<_>>().len();

        let rewrite_plan = shared_aggregate_family_rewrite_plan(plan, &family);
        let applied = rewrite_plan.is_ok();
        let rejected_reason = rewrite_plan.as_ref().err().copied();
        diagnostics.candidate_count += 1;
        diagnostics.candidates.push(SharedSubplanMaterializationCandidateDiagnostics {
            rule_name: SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
            family_key: representative.family_key.clone(),
            aggregate_node_ids: family.iter().map(|signature| signature.aggregate_node_id).collect(),
            reuse_count: family.len(),
            group_exprs: representative.group_exprs.clone(),
            aggregate_exprs: materialization_family_aggregate_exprs(&family),
            source_fingerprints: representative.source_fingerprints.clone(),
            exact_source_variant_count,
            repeated_selected_source_bytes,
            duplicate_selected_source_bytes,
            applied,
            rejected_reason,
        });
        if let Ok(rewrite_plan) = rewrite_plan {
            rewrite_families.push(rewrite_plan);
        }
    }

    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
    SharedSubplanMaterializationDiscovery { diagnostics, families: rewrite_families }
}

#[derive(Debug, Clone)]
struct AggregateMaterializationSignature {
    aggregate_node_id: NodeId,
    aggregate_output_columns: BTreeSet<String>,
    family_key: String,
    group_exprs: Vec<String>,
    aggregate_exprs: Vec<String>,
    source_fingerprints: Vec<String>,
    source_occurrences_unique: bool,
    source_predicates: Vec<SourcePredicateSignature>,
    source_columns: BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
    exact_source_signature: String,
    source_bytes: Option<u64>,
}

fn aggregate_materialization_signature(
    plan: &QueryPlan,
    aggregate_node_id: NodeId,
    input: NodeId,
    spec: &AggregateSpec,
) -> Option<AggregateMaterializationSignature> {
    let mut sources = Vec::new();
    collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
    if sources.is_empty() {
        return None;
    }
    sources.sort_by(|left, right| left.abstract_fingerprint.cmp(&right.abstract_fingerprint));

    let scalar_filtered_family = spec.group_exprs.is_empty() && scalar_filtered_aggregate_supported(spec);
    let source_occurrences_unique =
        if scalar_filtered_family { scalar_source_occurrences_are_unique(plan, input)? } else { true };
    let source_fingerprints = sources
        .iter()
        .map(|source| {
            if scalar_filtered_family { source.rowset_fingerprint.clone() } else { source.abstract_fingerprint.clone() }
        })
        .collect::<Vec<_>>();
    let exact_source_signature = sources
        .iter()
        .map(|source| {
            if scalar_filtered_family {
                source.exact_rowset_fingerprint.as_str()
            } else {
                source.exact_fingerprint.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join("|");
    let source_bytes = sum_optional_u64(sources.iter().map(|source| source.selected_source_bytes));
    let group_exprs = spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>();
    let aggregate_exprs = spec.aggr_exprs.iter().map(aggregate_expr_signature).collect::<Vec<_>>();
    let aggregate_output_columns = plan
        .node(aggregate_node_id)?
        .output_schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<BTreeSet<_>>();
    let source_predicates = sources
        .iter()
        .map(|source| SourcePredicateSignature {
            abstract_fingerprint: if scalar_filtered_family {
                source.rowset_fingerprint.clone()
            } else {
                source.abstract_fingerprint.clone()
            },
            predicate: source.predicate.clone(),
        })
        .collect::<Vec<_>>();
    let mut source_columns = BTreeMap::<String, BTreeMap<String, SourceColumnMaterialization>>::new();
    for source in &sources {
        let fingerprint = if scalar_filtered_family {
            source.rowset_fingerprint.clone()
        } else {
            source.abstract_fingerprint.clone()
        };
        source_columns.entry(fingerprint).or_default().extend(source.columns.clone());
    }
    let scalar_input_shape =
        if scalar_filtered_family { Some(scalar_materialization_input_shape(plan, input)?) } else { None };
    let family_signature = if scalar_filtered_family {
        format!(
            "scalar_filtered;input_shape={};sources={}",
            scalar_input_shape.as_deref().unwrap_or(""),
            source_fingerprints.join("|")
        )
    } else {
        format!(
            "groups={};aggr={};sources={}",
            group_exprs.join(","),
            aggregate_exprs.join(","),
            source_fingerprints.join("|")
        )
    };
    let family_key = format!("shared-subplan-v1-fnv64-{:016x}", stable_fnv1a64(family_signature.as_bytes()));

    Some(AggregateMaterializationSignature {
        aggregate_node_id,
        aggregate_output_columns,
        family_key,
        group_exprs,
        aggregate_exprs,
        source_fingerprints,
        source_occurrences_unique,
        source_predicates,
        source_columns,
        exact_source_signature,
        source_bytes,
    })
}

#[derive(Debug, Clone)]
struct LocalSourceMaterializationSignature {
    rowset_fingerprint: String,
    exact_rowset_fingerprint: String,
    abstract_fingerprint: String,
    exact_fingerprint: String,
    predicate: Option<Expr>,
    selected_source_bytes: Option<u64>,
    columns: BTreeMap<String, SourceColumnMaterialization>,
}

#[derive(Debug, Clone)]
struct SourcePredicateSignature {
    abstract_fingerprint: String,
    predicate: Option<Expr>,
}

#[derive(Debug, Clone)]
struct SourceColumnMaterialization {
    name: String,
    file_index: usize,
    source_ref: SourceColumnRef,
    field: Field,
}

fn collect_local_parquet_sources(
    plan: &QueryPlan,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    sources: &mut Vec<LocalSourceMaterializationSignature>,
) {
    if !visited.insert(node_id) {
        return;
    }
    let Some(node) = plan.node(node_id) else {
        return;
    };
    match &node.kind {
        PlanNode::Source(source) => {
            if let SourceNode::LocalParquet(source) = source.as_ref() {
                sources.push(local_source_materialization_signature(source));
            }
        },
        _ => {
            for input in node.input_ids() {
                collect_local_parquet_sources(plan, *input, visited, sources);
            }
        },
    }
}

fn local_source_materialization_signature(source: &LocalParquetSource) -> LocalSourceMaterializationSignature {
    let mut file_signatures = source.files.iter().map(file_signature).collect::<Vec<_>>();
    file_signatures.sort();
    let files = file_signatures.join("+");
    let read_columns = source.read_columns.join(",");
    let projected_output_columns = source.projected_output_columns.join(",");
    let pure_filter_columns = source.pure_filter_columns.join(",");
    let partition_columns = source
        .partition_columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",");
    let missing_columns = source
        .missing_columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",");
    let rowset_common = format!(
        "files={files};partition={partition_columns};missing={missing_columns};scan_limit={};max_rows={}",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        source.max_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string())
    );
    let common = format!(
        "{rowset_common};read={read_columns};projected={projected_output_columns};pure_filter={pure_filter_columns}"
    );
    let predicate = source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
    let selected_source_bytes =
        source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());
    let columns = source
        .read_columns
        .iter()
        .zip(source.read_column_file_indices.iter())
        .filter_map(|(column, file_index)| {
            let field = source.schema.field_with_name(column).ok()?.as_ref().clone();
            let source_ref = source.source_column_ref(column)?;
            Some((
                column.clone(),
                SourceColumnMaterialization { name: column.clone(), file_index: *file_index, source_ref, field },
            ))
        })
        .collect();

    LocalSourceMaterializationSignature {
        rowset_fingerprint: format!("{rowset_common};predicate=*"),
        exact_rowset_fingerprint: format!("{rowset_common};predicate={predicate}"),
        abstract_fingerprint: format!("{common};predicate=*"),
        exact_fingerprint: format!("{common};predicate={predicate}"),
        predicate: source.reader_predicate.clone(),
        selected_source_bytes,
        columns,
    }
}

fn scalar_materialization_input_shape(plan: &QueryPlan, node_id: NodeId) -> Option<String> {
    scalar_materialization_input_shape_inner(plan, node_id, true)
}

fn scalar_materialization_input_shape_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    allow_count_star_projection_elision: bool,
) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let signature = local_source_materialization_signature(source);
                Some(format!("local_parquet({})", signature.rowset_fingerprint))
            },
            _ => None,
        },
        PlanNode::Filter(filter) => Some(format!(
            "filter(predicate={};input={})",
            filter.predicate,
            scalar_materialization_input_shape_inner(plan, filter.input, false)?
        )),
        PlanNode::Projection(projection)
            if allow_count_star_projection_elision && count_star_projection_is_elidable(projection) =>
        {
            scalar_materialization_input_shape_inner(plan, projection.input, true)
        },
        PlanNode::Projection(projection) => {
            let outputs = projection
                .outputs
                .iter()
                .map(|output| {
                    format!(
                        "{}:{:?}:nullable={}={}",
                        output.field.name(),
                        output.field.data_type(),
                        output.field.is_nullable(),
                        output.expr
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            Some(format!(
                "projection(outputs=[{outputs}];input={})",
                scalar_materialization_input_shape_inner(plan, projection.input, false)?
            ))
        },
        PlanNode::Join(join) => Some(format!(
            "join(spec={:?};input0={};input1={})",
            join.spec,
            scalar_materialization_input_shape_inner(plan, join.inputs[0], false)?,
            scalar_materialization_input_shape_inner(plan, join.inputs[1], false)?
        )),
        PlanNode::Sort(sort) => Some(format!(
            "sort(spec={:?};input={})",
            sort.spec,
            scalar_materialization_input_shape_inner(plan, sort.input, false)?
        )),
        PlanNode::Limit(_) => None,
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn count_star_projection_is_elidable(projection: &ProjectionNode) -> bool {
    projection.outputs.is_empty()
        || projection.outputs.iter().all(|output| {
            scalar_count_star_expr(&output.expr) || projection_output_preserves_column(output, output.field.name())
        })
}

#[derive(Debug, Clone)]
struct SharedAggregateFamilyRewrite {
    family_key: String,
    representative_aggregate_node_id: NodeId,
    widened_source_predicates: BTreeMap<String, Expr>,
    widened_source_columns: BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
    kind: SharedAggregateFamilyRewriteKind,
}

#[derive(Debug, Clone)]
enum SharedAggregateFamilyRewriteKind {
    Grouped { branch_filters: BTreeMap<NodeId, Option<Expr>> },
    ScalarFiltered { required_aggregate_input_columns: BTreeSet<String>, branches: Vec<ScalarFilteredAggregateBranch> },
}

#[derive(Debug, Clone)]
struct ScalarFilteredAggregateBranch {
    aggregate_node_id: NodeId,
    outputs: Vec<ScalarFilteredAggregateOutput>,
}

#[derive(Debug, Clone)]
struct ScalarFilteredAggregateOutput {
    original_output_field: Field,
    materialized_alias: String,
    aggregate_expr: AggregateExpr,
}

fn shared_aggregate_family_rewrite_plan(
    plan: &QueryPlan,
    family: &[AggregateMaterializationSignature],
) -> Result<SharedAggregateFamilyRewrite, SharedSubplanMaterializationRejectedReason> {
    let representative =
        family.first().ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
    if representative.group_exprs.is_empty() {
        return shared_scalar_filtered_aggregate_rewrite_plan(plan, family);
    }
    let mut predicates_by_source = BTreeMap::<String, BTreeSet<String>>::new();
    for signature in family {
        for source in &signature.source_predicates {
            predicates_by_source
                .entry(source.abstract_fingerprint.clone())
                .or_default()
                .insert(source.predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string));
        }
    }
    let widened_source_fingerprints = predicates_by_source
        .into_iter()
        .filter_map(|(fingerprint, predicates)| (predicates.len() > 1).then_some(fingerprint))
        .collect::<BTreeSet<_>>();
    if widened_source_fingerprints.is_empty() {
        return Ok(SharedAggregateFamilyRewrite {
            family_key: representative.family_key.clone(),
            representative_aggregate_node_id: representative.aggregate_node_id,
            widened_source_predicates: BTreeMap::new(),
            widened_source_columns: BTreeMap::new(),
            kind: SharedAggregateFamilyRewriteKind::Grouped {
                branch_filters: family.iter().map(|signature| (signature.aggregate_node_id, None)).collect(),
            },
        });
    }

    let mut widened_source_predicates = BTreeMap::new();
    let mut branch_filters = BTreeMap::new();
    for fingerprint in &widened_source_fingerprints {
        let mut predicates = Vec::new();
        let mut seen = BTreeSet::new();
        for signature in family {
            let source = signature
                .source_predicates
                .iter()
                .find(|source| &source.abstract_fingerprint == fingerprint)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            if !predicate_references_only_columns(&predicate, &signature.aggregate_output_columns) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
            }
            if seen.insert(predicate.to_string()) {
                predicates.push(predicate);
            }
        }
        widened_source_predicates.insert(
            fingerprint.clone(),
            disjunction(predicates)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
        );
    }
    for signature in family {
        let mut predicates = Vec::new();
        for source in &signature.source_predicates {
            if !widened_source_fingerprints.contains(&source.abstract_fingerprint) {
                continue;
            }
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            if !predicate_references_only_columns(&predicate, &signature.aggregate_output_columns) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
            }
            predicates.push(predicate);
        }
        if predicates.is_empty() {
            return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
        }
        branch_filters.insert(
            signature.aggregate_node_id,
            Some(
                conjunction(predicates)
                    .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
            ),
        );
    }

    // Reject disconnected candidates early so diagnostics do not claim an applied
    // rewrite that the sink-reachable clone cannot observe.
    let reachable = rewrite::sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
        widened_source_columns: BTreeMap::new(),
        kind: SharedAggregateFamilyRewriteKind::Grouped { branch_filters },
    })
}

fn shared_scalar_filtered_aggregate_rewrite_plan(
    plan: &QueryPlan,
    family: &[AggregateMaterializationSignature],
) -> Result<SharedAggregateFamilyRewrite, SharedSubplanMaterializationRejectedReason> {
    let representative =
        family.first().ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
    let representative_node = plan
        .node(representative.aggregate_node_id)
        .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
    let PlanNode::Aggregate(representative_aggregate) = &representative_node.kind else {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    };
    if !scalar_filtered_aggregate_supported(&representative_aggregate.spec) {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }
    if family.iter().any(|signature| !source_materialization_sources_are_unique(signature)) {
        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
    }

    let mut predicates_by_source = BTreeMap::<String, BTreeSet<String>>::new();
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        if !scalar_filtered_aggregate_supported(&aggregate.spec) {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        }
        for aggregate_expr in &aggregate.spec.aggr_exprs {
            if scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr) {
                continue;
            }
            let Some(branch_signature) =
                scalar_input_expr_lineage_signature(plan, aggregate.input, &aggregate_expr.expr)
            else {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            };
            match scalar_input_expr_lineage_signature(plan, representative_aggregate.input, &aggregate_expr.expr) {
                Some(representative_signature) if branch_signature != representative_signature => {
                    return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
                },
                Some(_) => {},
                None => {
                    if scalar_input_expr_name_collides_with_unresolved_output(
                        plan,
                        representative_aggregate.input,
                        &aggregate_expr.expr,
                    ) {
                        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
                    }
                },
            }
        }
        for source in &signature.source_predicates {
            predicates_by_source
                .entry(source.abstract_fingerprint.clone())
                .or_default()
                .insert(source.predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string));
        }
    }

    let widened_source_fingerprints = predicates_by_source
        .into_iter()
        .filter_map(|(fingerprint, predicates)| (predicates.len() > 1).then_some(fingerprint))
        .collect::<BTreeSet<_>>();
    if widened_source_fingerprints.is_empty() {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }

    let mut widened_source_predicates = BTreeMap::new();
    let mut required_source_columns = BTreeMap::<String, BTreeSet<String>>::new();
    let mut widened_source_columns = BTreeMap::<String, BTreeMap<String, SourceColumnMaterialization>>::new();
    for fingerprint in &widened_source_fingerprints {
        let mut predicates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut required_columns = BTreeSet::new();
        for signature in family {
            if let Some(columns) = signature.source_columns.get(fingerprint) {
                merge_source_columns(widened_source_columns.entry(fingerprint.clone()).or_default(), columns)?;
            }
            let source = signature
                .source_predicates
                .iter()
                .find(|source| &source.abstract_fingerprint == fingerprint)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            collect_expr_column_names(&predicate, &mut required_columns);
            if seen.insert(predicate.to_string()) {
                predicates.push(predicate);
            }
        }
        required_source_columns.insert(fingerprint.clone(), required_columns);
        widened_source_predicates.insert(
            fingerprint.clone(),
            disjunction(predicates)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
        );
    }

    let mut required_aggregate_input_columns = BTreeSet::new();
    for columns in required_source_columns.values() {
        for column in columns {
            if !subtree_can_propagate_materialized_column(
                plan,
                representative_aggregate.input,
                column,
                &widened_source_columns,
            ) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            }
            required_aggregate_input_columns.insert(column.clone());
        }
    }
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        for aggregate_expr in &aggregate.spec.aggr_exprs {
            if scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr) {
                continue;
            }
            collect_expr_column_names(&aggregate_expr.expr, &mut required_aggregate_input_columns);
        }
        for source in &signature.source_predicates {
            if let Some(columns) = signature.source_columns.get(&source.abstract_fingerprint) {
                merge_source_columns(
                    widened_source_columns.entry(source.abstract_fingerprint.clone()).or_default(),
                    columns,
                )?;
            }
        }
    }
    if !source_column_names_are_unique_across_sources(&widened_source_columns) {
        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
    }
    for column in &required_aggregate_input_columns {
        if !subtree_can_propagate_materialized_column(
            plan,
            representative_aggregate.input,
            column,
            &widened_source_columns,
        ) {
            return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
        }
    }

    let mut branches = Vec::with_capacity(family.len());
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        if aggregate.spec.aggr_exprs.len() != node.output_schema().fields().len() {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        }
        let mut predicates = Vec::new();
        for source in &signature.source_predicates {
            if !widened_source_fingerprints.contains(&source.abstract_fingerprint) {
                continue;
            }
            predicates.push(
                source
                    .predicate
                    .clone()
                    .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
            );
        }
        let filter = conjunction(predicates)
            .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
        let outputs = aggregate
            .spec
            .aggr_exprs
            .iter()
            .zip(node.output_schema().fields())
            .enumerate()
            .map(|(index, (aggregate_expr, field))| {
                let count_star = scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr);
                let mut aggregate_expr = aggregate_expr.clone();
                aggregate_expr.alias =
                    format!("__cudf_shared_scalar_{}_{}", signature.aggregate_node_id.index(), index);
                aggregate_expr.filter = Some(filter.clone());
                if count_star {
                    aggregate_expr.expr = Expr::literal(crate::expr::LiteralValue::Int64(1));
                }
                ScalarFilteredAggregateOutput {
                    original_output_field: field.as_ref().clone(),
                    materialized_alias: aggregate_expr.alias.clone(),
                    aggregate_expr,
                }
            })
            .collect();
        branches.push(ScalarFilteredAggregateBranch { aggregate_node_id: signature.aggregate_node_id, outputs });
    }

    let reachable = rewrite::sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
        widened_source_columns,
        kind: SharedAggregateFamilyRewriteKind::ScalarFiltered { required_aggregate_input_columns, branches },
    })
}

struct SharedSubplanMaterializationRewriter<'a> {
    original: &'a QueryPlan,
    families_by_key: BTreeMap<String, SharedAggregateFamilyRewrite>,
    replacements: BTreeMap<NodeId, String>,
    materialized_nodes: BTreeMap<String, NodeId>,
    branch_nodes: BTreeMap<NodeId, NodeId>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: SharedSubplanMaterializationDiagnostics,
    changed: bool,
}

impl<'a> SharedSubplanMaterializationRewriter<'a> {
    fn new(
        original: &'a QueryPlan,
        families: Vec<SharedAggregateFamilyRewrite>,
        diagnostics: SharedSubplanMaterializationDiagnostics,
    ) -> Self {
        let mut replacements = BTreeMap::new();
        let mut families_by_key = BTreeMap::new();
        for family in families {
            match &family.kind {
                SharedAggregateFamilyRewriteKind::Grouped { branch_filters } => {
                    for aggregate_node_id in branch_filters.keys() {
                        replacements.insert(*aggregate_node_id, family.family_key.clone());
                    }
                },
                SharedAggregateFamilyRewriteKind::ScalarFiltered { branches, .. } => {
                    for branch in branches {
                        replacements.insert(branch.aggregate_node_id, family.family_key.clone());
                    }
                },
            }
            families_by_key.insert(family.family_key.clone(), family);
        }
        Self {
            original,
            families_by_key,
            replacements,
            materialized_nodes: BTreeMap::new(),
            branch_nodes: BTreeMap::new(),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics,
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if self.replacements.contains_key(&old_id) {
            self.changed = true;
            return self.branch_node(old_id);
        }
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn branch_node(&mut self, old_aggregate_id: NodeId) -> NodeId {
        if let Some(new_id) = self.branch_nodes.get(&old_aggregate_id) {
            return *new_id;
        }
        let family_key = self.replacements.get(&old_aggregate_id).expect("replacement family must exist").clone();
        let materialized_id = self.materialized_node(&family_key);
        let family = self.families_by_key.get(&family_key).expect("replacement family must exist").clone();
        let new_id = match &family.kind {
            SharedAggregateFamilyRewriteKind::Grouped { branch_filters } => {
                let predicate = branch_filters.get(&old_aggregate_id).cloned().flatten();
                if let Some(predicate) = predicate {
                    let mut facts =
                        self.nodes.get(materialized_id.index()).expect("materialized node must exist").facts.clone();
                    facts.schema_behavior = SchemaBehavior::PreservesInput;
                    self.push_node(PlanNode::Filter(FilterNode { input: materialized_id, predicate }), facts)
                } else {
                    materialized_id
                }
            },
            SharedAggregateFamilyRewriteKind::ScalarFiltered { branches, .. } => {
                let branch = branches
                    .iter()
                    .find(|branch| branch.aggregate_node_id == old_aggregate_id)
                    .expect("scalar filtered branch must exist");
                let outputs = branch
                    .outputs
                    .iter()
                    .map(|output| {
                        ProjectionOutput::new(
                            Expr::column(output.materialized_alias.clone())
                                .alias(output.original_output_field.name().clone()),
                            output.original_output_field.clone(),
                        )
                    })
                    .collect();
                let projection = ProjectionNode::new(materialized_id, outputs);
                let facts = PlanFacts::device(Arc::clone(projection.output_schema()), SchemaBehavior::Projection);
                self.push_node(PlanNode::Projection(projection), facts)
            },
        };
        self.branch_nodes.insert(old_aggregate_id, new_id);
        new_id
    }

    fn materialized_node(&mut self, family_key: &str) -> NodeId {
        if let Some(new_id) = self.materialized_nodes.get(family_key) {
            return *new_id;
        }
        let family = self.families_by_key.get(family_key).expect("materialization family must exist").clone();
        let new_id = self.clone_materialized_subtree(family.representative_aggregate_node_id, &family);
        self.materialized_nodes.insert(family_key.to_string(), new_id);
        new_id
    }

    fn clone_materialized_subtree(&mut self, old_id: NodeId, family: &SharedAggregateFamilyRewrite) -> NodeId {
        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        if old_id == family.representative_aggregate_node_id
            && let SharedAggregateFamilyRewriteKind::ScalarFiltered {
                required_aggregate_input_columns, branches, ..
            } = &family.kind
        {
            let PlanNode::Aggregate(aggregate) = &old_node.kind else {
                panic!("scalar filtered materialization root must be an aggregate");
            };
            let input = self.clone_materialized_subtree_with_required_columns(
                aggregate.input,
                family,
                required_aggregate_input_columns,
            );
            let spec = AggregateSpec::new(
                Vec::new(),
                branches
                    .iter()
                    .flat_map(|branch| branch.outputs.iter().map(|output| output.aggregate_expr.clone()))
                    .collect(),
            );
            let output_schema = Arc::new(Schema::new(
                branches
                    .iter()
                    .flat_map(|branch| {
                        branch.outputs.iter().map(|output| {
                            Field::new(
                                output.materialized_alias.clone(),
                                output.original_output_field.data_type().clone(),
                                output.original_output_field.is_nullable(),
                            )
                        })
                    })
                    .collect::<Vec<_>>(),
            ));
            return self.push_node(
                PlanNode::Aggregate(crate::plan::AggregateNode { input, spec }),
                PlanFacts::device(output_schema, SchemaBehavior::Aggregate),
            );
        }
        let new_facts = old_node.facts.clone();
        let new_kind = match &old_node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    let mut source = source.clone();
                    let signature = local_source_materialization_signature(&source);
                    let mut source_facts = None;
                    if let Some(columns) = matching_source_entry(&family.widened_source_columns, &signature) {
                        if widen_local_parquet_source(&mut source, columns) {
                            source_facts =
                                Some(PlanFacts::device(Arc::clone(&source.schema), SchemaBehavior::SourceOutput));
                        }
                    }
                    if let Some(predicate) = matching_source_entry(&family.widened_source_predicates, &signature) {
                        source.reader_predicate = Some(predicate.clone());
                        source.refresh_diagnostics();
                    }
                    let new_kind = PlanNode::Source(Box::new(SourceNode::LocalParquet(source)));
                    return self.push_node(new_kind, source_facts.unwrap_or_else(|| old_node.facts.clone()));
                },
                _ => PlanNode::Source(source.clone()),
            },
            _ => self.clone_kind_with_input_mapper(&old_node.kind, |this, input| {
                this.clone_materialized_subtree(input, family)
            }),
        };
        self.push_node(new_kind, new_facts)
    }

    fn clone_materialized_subtree_with_required_columns(
        &mut self,
        old_id: NodeId,
        family: &SharedAggregateFamilyRewrite,
        required_output_columns: &BTreeSet<String>,
    ) -> NodeId {
        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        match &old_node.kind {
            PlanNode::Source(source) => {
                let (new_kind, facts) = match source.as_ref() {
                    SourceNode::LocalParquet(source) => {
                        let mut source = source.clone();
                        let signature = local_source_materialization_signature(&source);
                        let mut facts = old_node.facts.clone();
                        if let Some(columns) = matching_source_entry(&family.widened_source_columns, &signature) {
                            if widen_local_parquet_source(&mut source, columns) {
                                facts = PlanFacts::device(Arc::clone(&source.schema), SchemaBehavior::SourceOutput);
                            }
                        }
                        if let Some(predicate) = matching_source_entry(&family.widened_source_predicates, &signature) {
                            source.reader_predicate = Some(predicate.clone());
                            source.refresh_diagnostics();
                        }
                        (PlanNode::Source(Box::new(SourceNode::LocalParquet(source))), facts)
                    },
                    _ => (PlanNode::Source(source.clone()), old_node.facts.clone()),
                };
                self.push_node(new_kind, facts)
            },
            PlanNode::Filter(filter) => {
                let mut input_required = required_output_columns.clone();
                collect_expr_column_names(&filter.predicate, &mut input_required);
                let input =
                    self.clone_materialized_subtree_with_required_columns(filter.input, family, &input_required);
                let mut filter = filter.clone();
                filter.input = input;
                let input_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(
                    PlanNode::Filter(filter),
                    PlanFacts::device(input_schema, SchemaBehavior::PreservesInput),
                )
            },
            PlanNode::Projection(projection) => {
                let mut outputs = projection.outputs.clone();
                let mut output_names =
                    outputs.iter().map(|output| output.field.name().clone()).collect::<BTreeSet<_>>();
                for column in required_output_columns {
                    if output_names.insert(column.clone()) {
                        let input_field = field_with_name(
                            self.original.node(projection.input).expect("projection input must exist").output_schema(),
                            column,
                        )
                        .cloned()
                        .or_else(|| {
                            propagatable_materialized_field(
                                self.original,
                                projection.input,
                                column,
                                &family.widened_source_columns,
                            )
                        })
                        .expect("required projection column must exist on input");
                        outputs.push(ProjectionOutput::new(Expr::column(column.clone()), input_field));
                    }
                }
                let mut input_required = BTreeSet::new();
                for output in &outputs {
                    collect_expr_column_names(&output.expr, &mut input_required);
                }
                let input =
                    self.clone_materialized_subtree_with_required_columns(projection.input, family, &input_required);
                let projection = ProjectionNode::new(input, outputs);
                let facts = PlanFacts::device(Arc::clone(projection.output_schema()), SchemaBehavior::Projection);
                self.push_node(PlanNode::Projection(projection), facts)
            },
            PlanNode::Join(join) => {
                let mut output_projection = join.spec.output_projection.clone();
                let mut left_required = BTreeSet::new();
                let mut right_required = BTreeSet::new();
                for key in &join.spec.keys {
                    left_required.insert(key.left.name().to_string());
                    right_required.insert(key.right.name().to_string());
                }
                if let Some(predicate) = &join.spec.residual_predicate {
                    collect_join_expr_columns(predicate, &mut left_required, &mut right_required);
                }
                for output in &output_projection {
                    match output.side {
                        JoinSide::Left => {
                            left_required.insert(output.column.name().to_string());
                        },
                        JoinSide::Right => {
                            right_required.insert(output.column.name().to_string());
                        },
                    }
                }

                for column in required_output_columns {
                    if let Some(output) = output_projection.iter().find(|output| output.output_name == *column) {
                        match output.side {
                            JoinSide::Left => {
                                left_required.insert(output.column.name().to_string());
                            },
                            JoinSide::Right => {
                                right_required.insert(output.column.name().to_string());
                            },
                        }
                        continue;
                    }
                    let side = unique_join_input_that_can_propagate_column(
                        self.original,
                        join.inputs[0],
                        join.inputs[1],
                        column,
                    )
                    .or_else(|| {
                        unique_join_input_that_can_propagate_materialized_column(
                            self.original,
                            join.inputs[0],
                            join.inputs[1],
                            column,
                            &family.widened_source_columns,
                        )
                    })
                    .expect("required join column must have one input owner");
                    match side {
                        JoinSide::Left => {
                            left_required.insert(column.clone());
                            output_projection.push(JoinOutputColumn::left(
                                crate::expr::ColumnRef::input(0, column.clone(), column.clone()),
                                column.clone(),
                            ));
                        },
                        JoinSide::Right => {
                            right_required.insert(column.clone());
                            output_projection.push(JoinOutputColumn::right(
                                crate::expr::ColumnRef::input(1, column.clone(), column.clone()),
                                column.clone(),
                            ));
                        },
                    }
                }

                let left =
                    self.clone_materialized_subtree_with_required_columns(join.inputs[0], family, &left_required);
                let right =
                    self.clone_materialized_subtree_with_required_columns(join.inputs[1], family, &right_required);
                let mut join = join.clone();
                join.inputs = [left, right];
                join.spec.output_projection = output_projection;
                let output_schema = join_output_schema(
                    self.nodes[left.index()].output_schema(),
                    self.nodes[right.index()].output_schema(),
                    &join.spec.output_projection,
                );
                self.push_node(PlanNode::Join(join), PlanFacts::device(output_schema, SchemaBehavior::Join))
            },
            PlanNode::Sort(sort) => {
                let mut input_required = required_output_columns.clone();
                for sort_expr in &sort.spec.expressions {
                    collect_expr_column_names(&sort_expr.expr, &mut input_required);
                }
                let input = self.clone_materialized_subtree_with_required_columns(sort.input, family, &input_required);
                let mut sort = sort.clone();
                sort.input = input;
                let mut facts = old_node.facts.clone();
                facts.output_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(PlanNode::Sort(sort), facts)
            },
            PlanNode::Limit(limit) => {
                let input =
                    self.clone_materialized_subtree_with_required_columns(limit.input, family, required_output_columns);
                let mut limit = limit.clone();
                limit.input = input;
                let mut facts = old_node.facts.clone();
                facts.output_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(PlanNode::Limit(limit), facts)
            },
            PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {
                panic!("unsupported scalar filtered materialization subtree")
            },
        }
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        self.clone_kind_with_input_mapper(kind, |this, input| this.rewrite_subtree(input))
    }

    fn clone_kind_with_input_mapper(
        &mut self,
        kind: &PlanNode,
        mut rewrite_input: impl FnMut(&mut Self, NodeId) -> NodeId,
    ) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| rewrite_input(self, input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

fn matching_source_entry<'a, T>(
    entries: &'a BTreeMap<String, T>,
    signature: &LocalSourceMaterializationSignature,
) -> Option<&'a T> {
    entries.get(&signature.abstract_fingerprint).or_else(|| entries.get(&signature.rowset_fingerprint))
}

fn merge_source_columns(
    target: &mut BTreeMap<String, SourceColumnMaterialization>,
    source: &BTreeMap<String, SourceColumnMaterialization>,
) -> Result<(), SharedSubplanMaterializationRejectedReason> {
    for (name, column) in source {
        if let Some(existing) = target.get(name) {
            if existing.file_index != column.file_index
                || existing.source_ref != column.source_ref
                || existing.field.name() != column.field.name()
                || existing.field.data_type() != column.field.data_type()
                || existing.field.is_nullable() != column.field.is_nullable()
            {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            }
        } else {
            target.insert(name.clone(), column.clone());
        }
    }
    Ok(())
}

fn source_column_names_are_unique_across_sources(
    sources: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> bool {
    let mut owners = BTreeMap::<&str, &str>::new();
    for (source, columns) in sources {
        for name in columns.keys() {
            if let Some(existing) = owners.insert(name.as_str(), source.as_str())
                && existing != source
            {
                return false;
            }
        }
    }
    true
}

fn source_materialization_sources_are_unique(signature: &AggregateMaterializationSignature) -> bool {
    signature.source_occurrences_unique && source_fingerprints_are_unique(signature)
}

fn source_fingerprints_are_unique(signature: &AggregateMaterializationSignature) -> bool {
    let mut seen = BTreeSet::new();
    signature.source_fingerprints.iter().all(|fingerprint| seen.insert(fingerprint))
}

fn scalar_source_occurrences_are_unique(plan: &QueryPlan, node_id: NodeId) -> Option<bool> {
    let mut source_nodes = BTreeSet::new();
    let mut rowsets = BTreeSet::new();
    let mut active = BTreeSet::new();
    scalar_source_occurrences_are_unique_inner(plan, node_id, &mut active, &mut source_nodes, &mut rowsets)
}

fn scalar_source_occurrences_are_unique_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    active: &mut BTreeSet<NodeId>,
    source_nodes: &mut BTreeSet<NodeId>,
    rowsets: &mut BTreeSet<String>,
) -> Option<bool> {
    if !active.insert(node_id) {
        return Some(false);
    }
    let node = plan.node(node_id)?;
    let unique = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                source_nodes.insert(node_id)
                    && rowsets.insert(local_source_materialization_signature(source).rowset_fingerprint)
            },
            _ => false,
        },
        _ => {
            let mut unique = true;
            for input in node.input_ids() {
                unique &= scalar_source_occurrences_are_unique_inner(plan, *input, active, source_nodes, rowsets)?;
            }
            unique
        },
    };
    active.remove(&node_id);
    Some(unique)
}

fn widen_local_parquet_source(
    source: &mut LocalParquetSource,
    columns: &BTreeMap<String, SourceColumnMaterialization>,
) -> bool {
    let mut changed = false;
    for column in columns.values() {
        if source.read_columns.iter().any(|existing| existing == &column.name) {
            continue;
        }
        source.read_columns.push(column.name.clone());
        source.read_column_file_indices.push(column.file_index);
        changed = true;
    }
    if !changed {
        return false;
    }

    let mut fields = source.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut field_names = fields.iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    for column in columns.values() {
        if field_names.insert(column.name.clone()) {
            fields.push(column.field.clone());
        }
    }
    source.schema = Arc::new(Schema::new(fields));
    source.projected_output_columns = source.read_columns.clone();
    source.projected_output_column_source_refs =
        source.read_column_file_indices.iter().copied().map(crate::source::SourceColumnRef::File).collect();
    source.pure_filter_columns.clear();
    source.pure_filter_column_source_refs.clear();
    source.refresh_diagnostics();
    true
}

fn scalar_filtered_aggregate_supported(spec: &AggregateSpec) -> bool {
    spec.group_exprs.is_empty()
        && spec.grouping_sets.is_none()
        && !spec.aggr_exprs.is_empty()
        && spec.aggr_exprs.iter().all(scalar_filtered_aggregate_expr_supported)
}

fn scalar_filtered_aggregate_expr_supported(expr: &AggregateExpr) -> bool {
    if expr.filter.is_some() {
        return false;
    }
    if expr.distinct && expr.function != AggregateFunction::Count {
        return false;
    }
    match expr.function {
        AggregateFunction::Count if !expr.distinct && scalar_count_star_expr(&expr.expr) => true,
        AggregateFunction::Count
        | AggregateFunction::NUnique
        | AggregateFunction::Sum
        | AggregateFunction::Min
        | AggregateFunction::Max
        | AggregateFunction::Avg => scalar_aggregate_value_expr_supported(&expr.expr),
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => false,
    }
}

fn scalar_count_star_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value) if !matches!(value, crate::expr::LiteralValue::Null(_)))
        || matches!(expr, Expr::Alias { expr, .. } if scalar_count_star_expr(expr))
}

fn scalar_count_star_aggregate_expr(plan: &QueryPlan, input: NodeId, expr: &Expr) -> bool {
    if scalar_count_star_expr(expr) {
        return true;
    }
    match expr {
        Expr::Column(column) => projected_non_null_literal_column(plan, input, column.name()),
        Expr::Alias { expr, .. } => scalar_count_star_aggregate_expr(plan, input, expr),
        _ => false,
    }
}

fn projected_non_null_literal_column(plan: &QueryPlan, node_id: NodeId, column: &str) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .find(|output| output.field.name() == column)
            .is_some_and(|output| scalar_count_star_expr(&output.expr)),
        _ => false,
    }
}

fn scalar_aggregate_value_expr_supported(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::Alias { expr, .. } => scalar_aggregate_value_expr_supported(expr),
        _ => false,
    }
}

fn materialization_family_aggregate_exprs(family: &[AggregateMaterializationSignature]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut aggregate_exprs = Vec::new();
    for signature in family {
        for aggregate_expr in &signature.aggregate_exprs {
            if seen.insert(aggregate_expr.clone()) {
                aggregate_exprs.push(aggregate_expr.clone());
            }
        }
    }
    aggregate_exprs
}

fn conjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::And, right)))
}

fn disjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::Or, right)))
}

fn predicate_references_only_columns(predicate: &Expr, allowed_columns: &BTreeSet<String>) -> bool {
    let mut columns = BTreeSet::new();
    collect_expr_columns(predicate, &mut columns);
    columns.iter().all(|column| allowed_columns.contains(column))
}

fn collect_expr_columns(expr: &Expr, columns: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => {
            columns.insert(column.name().to_string());
        },
        Expr::Literal(_) => {},
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => collect_expr_columns(expr, columns),
        Expr::RegexpReplace { expr, .. } => collect_expr_columns(expr, columns),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_expr_columns(expr, columns);
            }
        },
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_expr_columns(left, columns);
            collect_expr_columns(right, columns);
        },
        Expr::InList { expr, list, .. } => {
            collect_expr_columns(expr, columns);
            for expr in list {
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
    }
}

fn collect_expr_column_names(expr: &Expr, columns: &mut BTreeSet<String>) {
    collect_expr_columns(expr, columns);
}

fn collect_join_expr_columns(expr: &Expr, left: &mut BTreeSet<String>, right: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(crate::expr::ColumnQualifier::Input { index: 0 }) => {
                left.insert(column.name().to_string());
            },
            Some(crate::expr::ColumnQualifier::Input { index: 1 }) => {
                right.insert(column.name().to_string());
            },
            _ => {},
        },
        Expr::Literal(_) => {},
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. }
        | Expr::RegexpReplace { expr, .. } => collect_join_expr_columns(expr, left, right),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_join_expr_columns(expr, left, right);
            }
        },
        Expr::Binary { left: lhs, right: rhs, .. }
        | Expr::IsDistinctFrom { left: lhs, right: rhs }
        | Expr::IsNotDistinctFrom { left: lhs, right: rhs } => {
            collect_join_expr_columns(lhs, left, right);
            collect_join_expr_columns(rhs, left, right);
        },
        Expr::InList { expr, list, .. } => {
            collect_join_expr_columns(expr, left, right);
            for expr in list {
                collect_join_expr_columns(expr, left, right);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_join_expr_columns(&branch.when, left, right);
                collect_join_expr_columns(&branch.then, left, right);
            }
            collect_join_expr_columns(else_expr, left, right);
        },
    }
}

fn field_with_name<'a>(schema: &'a Arc<Schema>, name: &str) -> Option<&'a Field> {
    schema.fields().iter().find(|field| field.name() == name).map(|field| field.as_ref())
}

fn unique_join_input_that_can_propagate_column(
    plan: &QueryPlan,
    left: NodeId,
    right: NodeId,
    column: &str,
) -> Option<JoinSide> {
    match (subtree_can_propagate_column(plan, left, column), subtree_can_propagate_column(plan, right, column)) {
        (true, false) => Some(JoinSide::Left),
        (false, true) => Some(JoinSide::Right),
        _ => None,
    }
}

fn unique_join_input_that_can_propagate_materialized_column(
    plan: &QueryPlan,
    left: NodeId,
    right: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> Option<JoinSide> {
    match (
        subtree_can_propagate_materialized_column(plan, left, column, widened_source_columns),
        subtree_can_propagate_materialized_column(plan, right, column, widened_source_columns),
    ) {
        (true, false) => Some(JoinSide::Left),
        (false, true) => Some(JoinSide::Right),
        _ => None,
    }
}

fn subtree_can_propagate_column(plan: &QueryPlan, node_id: NodeId, column: &str) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if field_with_name(node.output_schema(), column).is_some() {
        return true;
    }
    match &node.kind {
        PlanNode::Source(_) => false,
        PlanNode::Filter(filter) => subtree_can_propagate_column(plan, filter.input, column),
        PlanNode::Projection(projection) => subtree_can_propagate_column(plan, projection.input, column),
        PlanNode::Join(join) => {
            unique_join_input_that_can_propagate_column(plan, join.inputs[0], join.inputs[1], column).is_some()
        },
        PlanNode::Sort(sort) => subtree_can_propagate_column(plan, sort.input, column),
        PlanNode::Limit(limit) => subtree_can_propagate_column(plan, limit.input, column),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn subtree_can_propagate_materialized_column(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                if field_with_name(&source.schema, column).is_some() {
                    return true;
                }
                let signature = local_source_materialization_signature(source);
                matching_source_entry(widened_source_columns, &signature)
                    .is_some_and(|columns| columns.contains_key(column))
            },
            _ => false,
        },
        PlanNode::Filter(filter) => {
            subtree_can_propagate_materialized_column(plan, filter.input, column, widened_source_columns)
        },
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return projection_output_preserves_column(output, column);
            }
            subtree_can_propagate_materialized_column(plan, projection.input, column, widened_source_columns)
        },
        PlanNode::Join(join) => unique_join_input_that_can_propagate_materialized_column(
            plan,
            join.inputs[0],
            join.inputs[1],
            column,
            widened_source_columns,
        )
        .is_some(),
        PlanNode::Sort(sort) => {
            subtree_can_propagate_materialized_column(plan, sort.input, column, widened_source_columns)
        },
        PlanNode::Limit(limit) => {
            subtree_can_propagate_materialized_column(plan, limit.input, column, widened_source_columns)
        },
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn scalar_input_expr_lineage_signature(plan: &QueryPlan, node_id: NodeId, expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => scalar_input_column_lineage_signature(plan, node_id, column.name()),
        Expr::Alias { expr, .. } => scalar_input_expr_lineage_signature(plan, node_id, expr),
        _ => None,
    }
}

fn scalar_input_expr_name_collides_with_unresolved_output(plan: &QueryPlan, node_id: NodeId, expr: &Expr) -> bool {
    match expr {
        Expr::Column(column) => {
            let Some(node) = plan.node(node_id) else {
                return true;
            };
            field_with_name(node.output_schema(), column.name()).is_some()
                && scalar_input_column_lineage_signature(plan, node_id, column.name()).is_none()
        },
        Expr::Alias { expr, .. } => scalar_input_expr_name_collides_with_unresolved_output(plan, node_id, expr),
        _ => false,
    }
}

fn scalar_input_column_lineage_signature(plan: &QueryPlan, node_id: NodeId, column: &str) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                field_with_name(&source.schema, column)?;
                let signature = local_source_materialization_signature(source);
                let source_ref = source.source_column_ref(column)?;
                Some(format!(
                    "{};column={column};source_ref={}",
                    signature.rowset_fingerprint,
                    source_column_ref_signature(source_ref)
                ))
            },
            _ => None,
        },
        PlanNode::Filter(filter) => scalar_input_column_lineage_signature(plan, filter.input, column),
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return scalar_input_expr_lineage_signature(plan, projection.input, &output.expr);
            }
            scalar_input_column_lineage_signature(plan, projection.input, column)
        },
        PlanNode::Sort(sort) => scalar_input_column_lineage_signature(plan, sort.input, column),
        PlanNode::Limit(limit) => scalar_input_column_lineage_signature(plan, limit.input, column),
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn propagatable_materialized_field(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> Option<Field> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                if let Some(field) = field_with_name(&source.schema, column) {
                    return Some(field.clone());
                }
                let signature = local_source_materialization_signature(source);
                matching_source_entry(widened_source_columns, &signature)
                    .and_then(|columns| columns.get(column))
                    .map(|column| column.field.clone())
            },
            _ => None,
        },
        PlanNode::Filter(filter) => propagatable_materialized_field(plan, filter.input, column, widened_source_columns),
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return projection_output_preserves_column(output, column).then(|| output.field.clone());
            }
            propagatable_materialized_field(plan, projection.input, column, widened_source_columns)
        },
        PlanNode::Join(join) => {
            if let Some(output) = join.spec.output_projection.iter().find(|output| output.output_name == column) {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                return propagatable_materialized_field(plan, input, output.column.name(), widened_source_columns);
            }
            match unique_join_input_that_can_propagate_materialized_column(
                plan,
                join.inputs[0],
                join.inputs[1],
                column,
                widened_source_columns,
            )? {
                JoinSide::Left => propagatable_materialized_field(plan, join.inputs[0], column, widened_source_columns),
                JoinSide::Right => {
                    propagatable_materialized_field(plan, join.inputs[1], column, widened_source_columns)
                },
            }
        },
        PlanNode::Sort(sort) => propagatable_materialized_field(plan, sort.input, column, widened_source_columns),
        PlanNode::Limit(limit) => propagatable_materialized_field(plan, limit.input, column, widened_source_columns),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn projection_output_preserves_column(output: &ProjectionOutput, column: &str) -> bool {
    output.field.name() == column && expr_is_identity_column(&output.expr, column)
}

fn expr_is_identity_column(expr: &Expr, column: &str) -> bool {
    match expr {
        Expr::Column(expr_column) => expr_column.name() == column,
        Expr::Alias { expr, alias } => alias == column && expr_is_identity_column(expr, column),
        _ => false,
    }
}

fn join_output_schema(
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    outputs: &[JoinOutputColumn],
) -> Arc<Schema> {
    Arc::new(Schema::new(
        outputs
            .iter()
            .map(|output| {
                let input_schema = match output.side {
                    JoinSide::Left => left_schema,
                    JoinSide::Right => right_schema,
                };
                let input_field = field_with_name(input_schema, output.column.name())
                    .expect("join output column must exist on input schema");
                Field::new(output.output_name.clone(), input_field.data_type().clone(), input_field.is_nullable())
            })
            .collect::<Vec<_>>(),
    ))
}

fn source_column_ref_signature(source_ref: SourceColumnRef) -> String {
    match source_ref {
        SourceColumnRef::File(index) => format!("file:{index}"),
        SourceColumnRef::Partition(index) => format!("partition:{index}"),
        SourceColumnRef::MissingNull(index) => format!("missing_null:{index}"),
    }
}

fn file_signature(file: &crate::source::LocalParquetFile) -> String {
    let partition_values = file.partition_values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>().join(",");
    format!(
        "stable_id={};location={};row_groups={};partition_values={}",
        file.stable_id,
        location_signature(&file.location),
        file.row_groups
            .as_ref()
            .map(|row_groups| row_groups.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "all".to_string()),
        partition_values
    )
}

fn location_signature(location: &ParquetSourceLocation) -> String {
    match location {
        ParquetSourceLocation::LocalPath(path) => format!("local:{}", path.display()),
        ParquetSourceLocation::ObjectStoreUri { scheme, uri } => format!("{scheme}:{uri}"),
    }
}

fn aggregate_expr_signature(expr: &AggregateExpr) -> String {
    format!(
        "fn={};distinct={};arg={};filter={};alias={}",
        aggregate_function_signature(expr.function),
        expr.distinct,
        expr.expr,
        expr.filter.as_ref().map_or_else(|| "none".to_string(), ToString::to_string),
        expr.alias
    )
}

fn aggregate_function_signature(function: AggregateFunction) -> &'static str {
    function.as_str()
}

fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    let mut sum = 0_u64;
    for value in values {
        sum = sum.checked_add(value?)?;
    }
    Some(sum)
}

fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
