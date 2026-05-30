use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema};

use crate::{
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr},
    plan::{
        AggregateNode, FilterNode, JoinOutputColumn, JoinSide, NodeId, PlanFacts, PlanNode, ProjectionNode,
        ProjectionOutput, QueryNode, QueryPlan, ResultSink, SchemaBehavior, UnionNode,
    },
    source::{LocalParquetSource, ParquetSourceLocation, SourceNode},
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
    let new_sink = sink_with_input(&sink, new_root);
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
    let new_sink = sink_with_input(&sink, new_root);
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
    let reachable = sink_reachable_nodes(plan);
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
        "local_parquet(files={files};predicate={predicate};scan_limit={};zero_column_count={})",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
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
        match kind {
            PlanNode::Source(source) => PlanNode::Source(source.clone()),
            PlanNode::Filter(filter) => {
                let mut filter = filter.clone();
                filter.input = self.rewrite_subtree(filter.input);
                PlanNode::Filter(filter)
            },
            PlanNode::Projection(projection) => {
                let mut projection = projection.clone();
                projection.input = self.rewrite_subtree(projection.input);
                PlanNode::Projection(projection)
            },
            PlanNode::Aggregate(aggregate) => {
                let mut aggregate = aggregate.clone();
                aggregate.input = self.rewrite_subtree(aggregate.input);
                PlanNode::Aggregate(aggregate)
            },
            PlanNode::Window(window) => {
                let mut window = window.clone();
                window.input = self.rewrite_subtree(window.input);
                PlanNode::Window(window)
            },
            PlanNode::Sort(sort) => {
                let mut sort = sort.clone();
                sort.input = self.rewrite_subtree(sort.input);
                PlanNode::Sort(sort)
            },
            PlanNode::Limit(limit) => {
                let mut limit = limit.clone();
                limit.input = self.rewrite_subtree(limit.input);
                PlanNode::Limit(limit)
            },
            PlanNode::Join(join) => {
                let mut join = join.clone();
                join.inputs = [self.rewrite_subtree(join.inputs[0]), self.rewrite_subtree(join.inputs[1])];
                PlanNode::Join(join)
            },
            PlanNode::Union(union) => {
                let mut union = union.clone();
                union.inputs = union.inputs.iter().map(|input| self.rewrite_subtree(*input)).collect();
                PlanNode::Union(union)
            },
            PlanNode::EdgeNormalize(edge_normalize) => {
                let mut edge_normalize = edge_normalize.clone();
                edge_normalize.input = self.rewrite_subtree(edge_normalize.input);
                PlanNode::EdgeNormalize(edge_normalize)
            },
            PlanNode::GraphAlgorithm(graph) => {
                let mut graph = graph.clone();
                graph.input = self.rewrite_subtree(graph.input);
                graph.inputs = graph.inputs.iter().map(|input| self.rewrite_subtree(*input)).collect();
                PlanNode::GraphAlgorithm(graph)
            },
        }
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
        if aggregate.spec.group_exprs.is_empty() && !scalar_count_aggregate_supported(&aggregate.spec) {
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
            aggregate_exprs: representative.aggregate_exprs.clone(),
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
    source_predicates: Vec<SourcePredicateSignature>,
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

    let source_fingerprints = sources.iter().map(|source| source.abstract_fingerprint.clone()).collect::<Vec<_>>();
    let exact_source_signature =
        sources.iter().map(|source| source.exact_fingerprint.as_str()).collect::<Vec<_>>().join("|");
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
            abstract_fingerprint: source.abstract_fingerprint.clone(),
            predicate: source.predicate.clone(),
        })
        .collect::<Vec<_>>();
    let family_signature = format!(
        "groups={};aggr={};sources={}",
        group_exprs.join(","),
        aggregate_exprs.join(","),
        source_fingerprints.join("|")
    );
    let family_key = format!("shared-subplan-v1-fnv64-{:016x}", stable_fnv1a64(family_signature.as_bytes()));

    Some(AggregateMaterializationSignature {
        aggregate_node_id,
        aggregate_output_columns,
        family_key,
        group_exprs,
        aggregate_exprs,
        source_fingerprints,
        source_predicates,
        exact_source_signature,
        source_bytes,
    })
}

#[derive(Debug, Clone)]
struct LocalSourceMaterializationSignature {
    abstract_fingerprint: String,
    exact_fingerprint: String,
    predicate: Option<Expr>,
    selected_source_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
struct SourcePredicateSignature {
    abstract_fingerprint: String,
    predicate: Option<Expr>,
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
    let common = format!(
        "files={files};read={read_columns};projected={projected_output_columns};pure_filter={pure_filter_columns};partition={partition_columns};missing={missing_columns};scan_limit={}",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string())
    );
    let predicate = source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
    let selected_source_bytes =
        source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());

    LocalSourceMaterializationSignature {
        abstract_fingerprint: format!("{common};predicate=*"),
        exact_fingerprint: format!("{common};predicate={predicate}"),
        predicate: source.reader_predicate.clone(),
        selected_source_bytes,
    }
}

#[derive(Debug, Clone)]
struct SharedAggregateFamilyRewrite {
    family_key: String,
    representative_aggregate_node_id: NodeId,
    widened_source_predicates: BTreeMap<String, Expr>,
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
    let reachable = sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
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
    if !scalar_count_aggregate_supported(&representative_aggregate.spec) {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }

    let mut predicates_by_source = BTreeMap::<String, BTreeSet<String>>::new();
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        if !scalar_count_aggregate_supported(&aggregate.spec) {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
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
    for fingerprint in &widened_source_fingerprints {
        let mut predicates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut required_columns = BTreeSet::new();
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
            if !subtree_can_propagate_column(plan, representative_aggregate.input, column) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            }
            required_aggregate_input_columns.insert(column.clone());
        }
    }
    for aggregate_expr in &representative_aggregate.spec.aggr_exprs {
        collect_expr_column_names(&aggregate_expr.expr, &mut required_aggregate_input_columns);
    }
    for column in &required_aggregate_input_columns {
        if !subtree_can_propagate_column(plan, representative_aggregate.input, column) {
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
        let original_output_field = node
            .output_schema()
            .fields()
            .first()
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?
            .as_ref()
            .clone();
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
        let mut aggregate_expr = aggregate.spec.aggr_exprs[0].clone();
        aggregate_expr.alias = format!("__cudf_shared_scalar_count_{}", signature.aggregate_node_id.index());
        aggregate_expr.filter = Some(filter.clone());
        branches.push(ScalarFilteredAggregateBranch {
            aggregate_node_id: signature.aggregate_node_id,
            original_output_field,
            materialized_alias: aggregate_expr.alias.clone(),
            aggregate_expr,
        });
    }

    let reachable = sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
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
                let output = ProjectionOutput::new(
                    Expr::column(branch.materialized_alias.clone()).alias(branch.original_output_field.name().clone()),
                    branch.original_output_field.clone(),
                );
                let projection = ProjectionNode::new(materialized_id, vec![output]);
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
            let spec =
                AggregateSpec::new(Vec::new(), branches.iter().map(|branch| branch.aggregate_expr.clone()).collect());
            let output_schema = Arc::new(Schema::new(
                branches
                    .iter()
                    .map(|branch| {
                        Field::new(
                            branch.materialized_alias.clone(),
                            branch.original_output_field.data_type().clone(),
                            branch.original_output_field.is_nullable(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ));
            return self.push_node(
                PlanNode::Aggregate(crate::plan::AggregateNode { input, spec }),
                PlanFacts::device(output_schema, SchemaBehavior::Aggregate),
            );
        }
        let new_kind = match &old_node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    let mut source = source.clone();
                    let signature = local_source_materialization_signature(&source);
                    if let Some(predicate) = family.widened_source_predicates.get(&signature.abstract_fingerprint) {
                        source.reader_predicate = Some(predicate.clone());
                        source.refresh_diagnostics();
                    }
                    PlanNode::Source(Box::new(SourceNode::LocalParquet(source)))
                },
                _ => PlanNode::Source(source.clone()),
            },
            _ => self.clone_kind_with_input_mapper(&old_node.kind, |this, input| {
                this.clone_materialized_subtree(input, family)
            }),
        };
        self.push_node(new_kind, old_node.facts.clone())
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
                let new_kind = match source.as_ref() {
                    SourceNode::LocalParquet(source) => {
                        let mut source = source.clone();
                        let signature = local_source_materialization_signature(&source);
                        if let Some(predicate) = family.widened_source_predicates.get(&signature.abstract_fingerprint) {
                            source.reader_predicate = Some(predicate.clone());
                            source.refresh_diagnostics();
                        }
                        PlanNode::Source(Box::new(SourceNode::LocalParquet(source)))
                    },
                    _ => PlanNode::Source(source.clone()),
                };
                self.push_node(new_kind, old_node.facts.clone())
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
                        .or_else(|| propagatable_field(self.original, projection.input, column))
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
        match kind {
            PlanNode::Source(source) => PlanNode::Source(source.clone()),
            PlanNode::Filter(filter) => {
                let mut filter = filter.clone();
                filter.input = rewrite_input(self, filter.input);
                PlanNode::Filter(filter)
            },
            PlanNode::Projection(projection) => {
                let mut projection = projection.clone();
                projection.input = rewrite_input(self, projection.input);
                PlanNode::Projection(projection)
            },
            PlanNode::Aggregate(aggregate) => {
                let mut aggregate = aggregate.clone();
                aggregate.input = rewrite_input(self, aggregate.input);
                PlanNode::Aggregate(aggregate)
            },
            PlanNode::Window(window) => {
                let mut window = window.clone();
                window.input = rewrite_input(self, window.input);
                PlanNode::Window(window)
            },
            PlanNode::Sort(sort) => {
                let mut sort = sort.clone();
                sort.input = rewrite_input(self, sort.input);
                PlanNode::Sort(sort)
            },
            PlanNode::Limit(limit) => {
                let mut limit = limit.clone();
                limit.input = rewrite_input(self, limit.input);
                PlanNode::Limit(limit)
            },
            PlanNode::Join(join) => {
                let mut join = join.clone();
                join.inputs = [rewrite_input(self, join.inputs[0]), rewrite_input(self, join.inputs[1])];
                PlanNode::Join(join)
            },
            PlanNode::Union(union) => {
                let inputs = union.inputs.iter().map(|input| rewrite_input(self, *input)).collect();
                PlanNode::Union(UnionNode { inputs })
            },
            PlanNode::EdgeNormalize(edge_normalize) => {
                let mut edge_normalize = edge_normalize.clone();
                edge_normalize.input = rewrite_input(self, edge_normalize.input);
                PlanNode::EdgeNormalize(edge_normalize)
            },
            PlanNode::GraphAlgorithm(graph) => {
                let mut graph = graph.clone();
                graph.input = rewrite_input(self, graph.input);
                PlanNode::GraphAlgorithm(graph)
            },
        }
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

fn sink_with_input(sink: &ResultSink, input: NodeId) -> ResultSink {
    match sink {
        ResultSink::GpuResident { .. } => ResultSink::GpuResident { input },
        ResultSink::HostArrow { schema, .. } => ResultSink::HostArrow { input, schema: schema.clone() },
    }
}

fn sink_reachable_nodes(plan: &QueryPlan) -> BTreeSet<NodeId> {
    let mut reachable = BTreeSet::new();
    let mut stack = vec![plan.sink().input()];
    while let Some(node_id) = stack.pop() {
        if !reachable.insert(node_id) {
            continue;
        }
        if let Some(node) = plan.node(node_id) {
            stack.extend(node.input_ids().iter().copied());
        }
    }
    reachable
}

fn scalar_count_aggregate_supported(spec: &AggregateSpec) -> bool {
    spec.group_exprs.is_empty()
        && spec.grouping_sets.is_none()
        && spec.aggr_exprs.len() == 1
        && spec
            .aggr_exprs
            .iter()
            .all(|expr| expr.function == AggregateFunction::Count && !expr.distinct && expr.filter.is_none())
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

fn propagatable_field(plan: &QueryPlan, node_id: NodeId, column: &str) -> Option<Field> {
    let node = plan.node(node_id)?;
    if let Some(field) = field_with_name(node.output_schema(), column) {
        return Some(field.clone());
    }
    match &node.kind {
        PlanNode::Source(_) => None,
        PlanNode::Filter(filter) => propagatable_field(plan, filter.input, column),
        PlanNode::Projection(projection) => propagatable_field(plan, projection.input, column),
        PlanNode::Join(join) => {
            match unique_join_input_that_can_propagate_column(plan, join.inputs[0], join.inputs[1], column)? {
                JoinSide::Left => propagatable_field(plan, join.inputs[0], column),
                JoinSide::Right => propagatable_field(plan, join.inputs[1], column),
            }
        },
        PlanNode::Sort(sort) => propagatable_field(plan, sort.input, column),
        PlanNode::Limit(limit) => propagatable_field(plan, limit.input, column),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
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
