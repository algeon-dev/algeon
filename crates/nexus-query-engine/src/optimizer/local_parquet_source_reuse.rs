use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{Field, Schema};

use super::rewrite;
use crate::{
    expr::Expr,
    plan::{
        FilterNode, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
    },
    source::{LocalParquetFile, LocalParquetSource, SourceColumnRef, SourceNode},
};

/// Stable rule name used in optimizer diagnostics.
pub const LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME: &str = "local_parquet_source_reuse";

const DEFAULT_MAX_VARIANT_SELECTED_BYTES: u64 = 64 * 1024 * 1024;

/// Configuration for sharing compatible repeated local Parquet sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParquetSourceReuseOptions {
    /// Whether the pass may replace repeated source variants with one superset scan.
    pub enabled: bool,
    /// Maximum selected source bytes for each source variant; variants with unknown selected bytes are not reused.
    pub max_variant_selected_bytes: u64,
}

impl LocalParquetSourceReuseOptions {
    /// Returns options with source reuse disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, max_variant_selected_bytes: DEFAULT_MAX_VARIANT_SELECTED_BYTES }
    }
}

impl Default for LocalParquetSourceReuseOptions {
    fn default() -> Self {
        Self { enabled: true, max_variant_selected_bytes: DEFAULT_MAX_VARIANT_SELECTED_BYTES }
    }
}

/// Diagnostics for compatible local Parquet source reuse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalParquetSourceReuseDiagnostics {
    /// Number of repeated source groups considered.
    pub considered_count: usize,
    /// Number of repeated source groups rewritten.
    pub applied_count: usize,
    /// Per-candidate evidence in deterministic order.
    pub candidates: Vec<LocalParquetSourceReuseCandidateDiagnostics>,
}

impl LocalParquetSourceReuseDiagnostics {
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

/// Per-candidate source reuse evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParquetSourceReuseCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Source nodes that can share one superset source.
    pub source_node_ids: Vec<NodeId>,
    /// Number of source variants in the group.
    pub reuse_count: usize,
    /// Stable file identifiers for the shared source group.
    pub stable_file_ids: Vec<String>,
    /// Superset source read columns.
    pub superset_read_columns: Vec<String>,
    /// Whether any branch predicate is replayed as a native filter.
    pub branch_filter_count: usize,
    /// Whether the group was rewritten.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<LocalParquetSourceReuseRejectedReason>,
}

/// Reasons a repeated local Parquet source group was not reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalParquetSourceReuseRejectedReason {
    /// All source variants have the exact same scan contract.
    ExactDuplicateSkipped,
    /// The source group uses unsupported source metadata for this conservative pass.
    UnsupportedSourceContract,
    /// Reusing the group would require widening a large source variant.
    VariantTooLarge,
}

pub(crate) struct LocalParquetSourceReuseResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: LocalParquetSourceReuseDiagnostics,
}

pub(crate) fn optimize_local_parquet_source_reuse(
    plan: QueryPlan,
    options: &LocalParquetSourceReuseOptions,
) -> LocalParquetSourceReuseResult {
    if !options.enabled {
        return LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: LocalParquetSourceReuseDiagnostics::default(),
        };
    }

    let discovery = discover_source_reuse_candidates(&plan, options);
    if discovery.rewrites.is_empty() {
        return LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let mut rewriter = LocalParquetSourceReuseRewriter::new(&plan, discovery.rewrites);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let new_sink = rewrite::sink_with_input(plan.sink(), new_root);
    let rewritten_plan = QueryPlan::from_rewritten_parts(rewriter.nodes, new_root, new_sink);
    LocalParquetSourceReuseResult {
        plan: rewritten_plan,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: discovery.diagnostics,
    }
}

struct SourceReuseDiscovery {
    diagnostics: LocalParquetSourceReuseDiagnostics,
    rewrites: Vec<SourceReuseRewrite>,
}

fn discover_source_reuse_candidates(
    plan: &QueryPlan,
    options: &LocalParquetSourceReuseOptions,
) -> SourceReuseDiscovery {
    let reachable = rewrite::sink_reachable_nodes(plan);
    let mut groups = BTreeMap::<SourceGroupKey, Vec<SourceVariant>>::new();
    for node in plan.nodes() {
        if !reachable.contains(&node.id) {
            continue;
        }
        let PlanNode::Source(source) = &node.kind else {
            continue;
        };
        let SourceNode::LocalParquet(source) = source.as_ref() else {
            continue;
        };
        if let Some(variant) = source_variant(node.id, source) {
            groups.entry(variant.group_key.clone()).or_default().push(variant);
        }
    }

    let mut diagnostics = LocalParquetSourceReuseDiagnostics::default();
    let mut rewrites = Vec::new();
    for mut variants in groups.into_values() {
        if variants.len() < 2 {
            continue;
        }
        variants.sort_by_key(|variant| variant.source_node_id);
        diagnostics.considered_count += 1;

        let exact_variant_count =
            variants.iter().map(|variant| variant.exact_key.as_str()).collect::<BTreeSet<_>>().len();
        let stable_file_ids = variants[0].stable_file_ids.clone();
        let source_node_ids = variants.iter().map(|variant| variant.source_node_id).collect::<Vec<_>>();
        let exact_duplicate = exact_variant_count == 1;
        let branch_filter_count = if exact_duplicate {
            0
        } else {
            variants.iter().filter(|variant| variant.reader_predicate.is_some()).count()
        };

        let rejected_reason = if variants.iter().any(|variant| !variant.supported_contract) {
            Some(LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
        } else if variants
            .iter()
            .any(|variant| variant.selected_source_bytes.is_none_or(|bytes| bytes > options.max_variant_selected_bytes))
        {
            Some(LocalParquetSourceReuseRejectedReason::VariantTooLarge)
        } else {
            None
        };

        let superset_read_columns = superset_read_columns(&variants);
        let applied = rejected_reason.is_none();
        diagnostics.candidates.push(LocalParquetSourceReuseCandidateDiagnostics {
            rule_name: LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
            source_node_ids: source_node_ids.clone(),
            reuse_count: variants.len(),
            stable_file_ids,
            superset_read_columns: superset_read_columns.iter().map(|column| column.name.clone()).collect(),
            branch_filter_count,
            applied,
            rejected_reason,
        });
        if applied {
            diagnostics.applied_count += 1;
            rewrites.push(SourceReuseRewrite::new(variants, superset_read_columns, exact_duplicate));
        }
    }

    SourceReuseDiscovery { diagnostics, rewrites }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SourceGroupKey {
    file_keys: Vec<FileGroupKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FileGroupKey {
    location: String,
    stable_id: String,
    file_size_bytes: Option<u64>,
    row_groups: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct SourceVariant {
    source_node_id: NodeId,
    source: LocalParquetSource,
    group_key: SourceGroupKey,
    exact_key: String,
    stable_file_ids: Vec<String>,
    selected_source_bytes: Option<u64>,
    reader_predicate: Option<Expr>,
    supported_contract: bool,
}

fn source_variant(source_node_id: NodeId, source: &LocalParquetSource) -> Option<SourceVariant> {
    if source.files.is_empty() {
        return None;
    }
    let group_key = SourceGroupKey {
        file_keys: source
            .files
            .iter()
            .map(|file| FileGroupKey {
                location: file.location.reader_path(),
                stable_id: file.stable_id.clone(),
                file_size_bytes: file.file_size_bytes,
                row_groups: file.row_groups.clone(),
            })
            .collect(),
    };
    let stable_file_ids = source.files.iter().map(|file| file.stable_id.clone()).collect();
    let selected_source_bytes =
        source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());
    Some(SourceVariant {
        source_node_id,
        source: source.clone(),
        group_key,
        exact_key: exact_source_key(source),
        stable_file_ids,
        selected_source_bytes,
        reader_predicate: source.reader_predicate.clone(),
        supported_contract: source_contract_supported(source),
    })
}

fn source_contract_supported(source: &LocalParquetSource) -> bool {
    source.scan_limit.is_none()
        && !source.is_zero_column_count_source()
        && source.partition_columns.is_empty()
        && source.missing_columns.is_empty()
        && source
            .projected_output_column_source_refs
            .iter()
            .chain(source.pure_filter_column_source_refs.iter())
            .all(|source_ref| matches!(source_ref, SourceColumnRef::File(_)))
        && source.files.iter().all(file_contract_supported)
}

fn file_contract_supported(file: &LocalParquetFile) -> bool {
    file.partition_values.is_empty()
}

fn exact_source_key(source: &LocalParquetSource) -> String {
    format!(
        "schema={:?};read={:?};read_idx={:?};projected={:?};projected_refs={:?};pure={:?};pure_refs={:?};predicate={}",
        source.schema,
        source.read_columns,
        source.read_column_file_indices,
        source.projected_output_columns,
        source.projected_output_column_source_refs,
        source.pure_filter_columns,
        source.pure_filter_column_source_refs,
        source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string)
    )
}

#[derive(Debug, Clone)]
struct SupersetColumn {
    name: String,
    file_index: usize,
    field: Field,
}

fn superset_read_columns(variants: &[SourceVariant]) -> Vec<SupersetColumn> {
    let mut columns = BTreeMap::<usize, SupersetColumn>::new();
    for variant in variants {
        for (column, file_index) in
            variant.source.read_columns.iter().zip(variant.source.read_column_file_indices.iter())
        {
            columns.entry(*file_index).or_insert_with(|| {
                let field = variant
                    .source
                    .schema
                    .field_with_name(column)
                    .expect("source read column must exist in source schema")
                    .as_ref()
                    .clone();
                SupersetColumn { name: column.clone(), file_index: *file_index, field }
            });
        }
    }
    columns.into_values().collect()
}

#[derive(Debug, Clone)]
struct SourceReuseRewrite {
    variants: BTreeMap<NodeId, SourceVariant>,
    shared_source: LocalParquetSource,
    replay_branch_predicates: bool,
}

impl SourceReuseRewrite {
    fn new(variants: Vec<SourceVariant>, superset_read_columns: Vec<SupersetColumn>, exact_duplicate: bool) -> Self {
        let representative = variants.first().expect("source reuse rewrite requires at least one variant");
        if exact_duplicate {
            let shared_source = representative.source.clone();
            let variants = variants.into_iter().map(|variant| (variant.source_node_id, variant)).collect();
            return Self { variants, shared_source, replay_branch_predicates: false };
        }

        let fields = superset_read_columns.iter().map(|column| column.field.clone()).collect::<Vec<_>>();
        let read_columns = superset_read_columns.iter().map(|column| column.name.clone()).collect::<Vec<_>>();
        let read_column_file_indices = superset_read_columns.iter().map(|column| column.file_index).collect::<Vec<_>>();
        let projected_output_columns = projected_columns_for_shared_source(&variants, &read_columns);
        let pure_filter_columns = read_columns
            .iter()
            .filter(|column| !projected_output_columns.contains(*column))
            .cloned()
            .collect::<Vec<_>>();
        let mut files = representative.source.files.clone();
        for file in &mut files {
            file.projected_row_group_compressed_bytes = None;
            file.projected_row_group_uncompressed_bytes = None;
            file.pure_filter_row_group_compressed_bytes = None;
            file.pure_filter_row_group_uncompressed_bytes = None;
        }

        let schema = Arc::new(Schema::new(fields));
        let shared_source = LocalParquetSource::from_files(files, schema, read_columns)
            .with_column_file_indices(read_column_file_indices, Vec::new(), Vec::new())
            .with_column_roles(projected_output_columns, pure_filter_columns);
        let variants = variants.into_iter().map(|variant| (variant.source_node_id, variant)).collect();
        Self { variants, shared_source, replay_branch_predicates: true }
    }
}

fn projected_columns_for_shared_source(variants: &[SourceVariant], read_columns: &[String]) -> Vec<String> {
    let mut projected = BTreeSet::new();
    for variant in variants {
        for column in &variant.source.projected_output_columns {
            projected.insert(column.clone());
        }
    }
    read_columns.iter().filter(|column| projected.contains(*column)).cloned().collect()
}

struct LocalParquetSourceReuseRewriter<'a> {
    original: &'a QueryPlan,
    rewrites_by_source: BTreeMap<NodeId, usize>,
    rewrites: Vec<SourceReuseRewrite>,
    shared_nodes: BTreeMap<usize, NodeId>,
    branch_nodes: BTreeMap<NodeId, NodeId>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
}

impl<'a> LocalParquetSourceReuseRewriter<'a> {
    fn new(original: &'a QueryPlan, rewrites: Vec<SourceReuseRewrite>) -> Self {
        let mut rewrites_by_source = BTreeMap::new();
        for (index, rewrite) in rewrites.iter().enumerate() {
            for source_node_id in rewrite.variants.keys() {
                rewrites_by_source.insert(*source_node_id, index);
            }
        }
        Self {
            original,
            rewrites_by_source,
            rewrites,
            shared_nodes: BTreeMap::new(),
            branch_nodes: BTreeMap::new(),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if self.rewrites_by_source.contains_key(&old_id) {
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

    fn shared_node(&mut self, rewrite_index: usize) -> NodeId {
        if let Some(new_id) = self.shared_nodes.get(&rewrite_index) {
            return *new_id;
        }
        let source = self.rewrites[rewrite_index].shared_source.clone();
        let output_schema = Arc::clone(&source.schema);
        let new_id = self.push_node(
            PlanNode::Source(Box::new(SourceNode::LocalParquet(source))),
            PlanFacts::device(output_schema, SchemaBehavior::SourceOutput),
        );
        self.shared_nodes.insert(rewrite_index, new_id);
        new_id
    }

    fn branch_node(&mut self, old_source_id: NodeId) -> NodeId {
        if let Some(new_id) = self.branch_nodes.get(&old_source_id) {
            return *new_id;
        }
        let rewrite_index = *self.rewrites_by_source.get(&old_source_id).expect("source rewrite must exist");
        let variant =
            self.rewrites[rewrite_index].variants.get(&old_source_id).expect("source variant must exist").clone();
        let mut current = self.shared_node(rewrite_index);
        if self.rewrites[rewrite_index].replay_branch_predicates
            && let Some(predicate) = variant.reader_predicate.clone()
        {
            let output_schema = self.nodes[current.index()].output_schema().clone();
            current = self.push_node(
                PlanNode::Filter(FilterNode { input: current, predicate }),
                PlanFacts::device(output_schema, SchemaBehavior::PreservesInput),
            );
        }
        if self.nodes[current.index()].output_schema() != &variant.source.schema {
            let outputs = variant
                .source
                .schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(Expr::column(field.name().clone()), field.as_ref().clone()))
                .collect::<Vec<_>>();
            let projection = ProjectionNode::new(current, outputs);
            let output_schema = Arc::clone(projection.output_schema());
            current = self.push_node(
                PlanNode::Projection(projection),
                PlanFacts::device(output_schema, SchemaBehavior::Projection),
            );
        }
        self.branch_nodes.insert(old_source_id, current);
        current
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}
