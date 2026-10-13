//! Local Parquet source reuse: one superset scan for repeated reads of the
//! same files.
//!
//! The rule matches two or more `LocalParquetSource` nodes over the same files,
//! row groups and reader contracts that differ in read columns or reader
//! predicate. In the rewrite, one shared source reads the union of the columns
//! with no reader predicate; each variant becomes a filter that replays its
//! predicate, when it had one, and a projection back to the variant's own schema,
//! so every consumer sees the columns it saw before. Exact duplicates share the
//! representative source as it is. As a precondition, the variants list files in
//! the same order and agree on each column's file index and field; partition
//! columns, missing columns, scan limits and count-star probe sources are outside
//! the rule. A merged scan that widens a variant is taken only when a known upper
//! bound on its selected bytes fits `max_variant_selected_bytes`, since widening
//! reads bytes the variant did not. Facts: the shared source's facts are its
//! widened schema, a replay filter keeps its input schema, and the projection
//! derives the variant schema; selected row-group byte diagnostics are cleared
//! because they described the narrower reads. Dynamic-filter routes are predicted
//! by running route discovery over a copy, because variants that would receive
//! different consumer routes cannot share one source.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::{dynamic_filter as dynamic_filter_optimizer, rewrite};
use crate::{
    backend::AnalysisContext,
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterRoute},
    error::{QueryError, QueryErrorCode, Result},
    expr::{BinaryOp, CaseWhen, Expr},
    plan::{
        FilterNode, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, RewrittenNodes,
    },
    source::{
        LocalParquetFile, LocalParquetSource, ParquetReaderDecimalType, ResolvedParquetSource, SourceColumnRef,
        SourceNode,
    },
    util::escape_kv_delimited,
};

/// Stable rule name used in optimizer diagnostics.
pub const LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME: &str = "local_parquet_source_reuse";

/// Configuration for sharing compatible repeated local Parquet sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParquetSourceReuseOptions {
    /// Whether the pass may replace repeated source variants with one superset scan.
    pub enabled: bool,
    /// Maximum selected source bytes of a merged scan that widens a variant.
    ///
    /// A merged scan widens a variant when it reads columns the variant does
    /// not read or drops the variant's reader predicate. Such a group is
    /// reused only when an upper bound on the merged scan's selected bytes is
    /// known and at most this value, whatever consumes the variants. A group
    /// whose merged scan is identical to what every variant reads is reused at
    /// any size.
    pub max_variant_selected_bytes: u64,
}

impl LocalParquetSourceReuseOptions {
    /// Default selected byte cap for a merged scan that widens a variant.
    pub const DEFAULT_MAX_VARIANT_SELECTED_BYTES: u64 = 500 * 1024 * 1024;

    /// Returns options with source reuse disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, max_variant_selected_bytes: Self::DEFAULT_MAX_VARIANT_SELECTED_BYTES }
    }

    /// Rejects a zero `max_variant_selected_bytes`; no other field is checked.
    pub fn validate(&self) -> Result<()> {
        if self.max_variant_selected_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidConfiguration,
                "query_engine.local_parquet_source_reuse_options.validate",
                "local Parquet source-reuse option must be greater than zero",
            )
            .with_fact("field", "max_variant_selected_bytes")
            .with_fact("value", self.max_variant_selected_bytes)
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for LocalParquetSourceReuseOptions {
    fn default() -> Self {
        Self { enabled: true, max_variant_selected_bytes: Self::DEFAULT_MAX_VARIANT_SELECTED_BYTES }
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
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    pub(crate) fn merge(&mut self, other: Self) {
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
    /// Configured maximum selected bytes of a merged scan that widens a variant.
    pub max_variant_selected_bytes: u64,
    /// Whether the group was rewritten.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<LocalParquetSourceReuseRejectedReason>,
}

/// Reasons a repeated local Parquet source group was not reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalParquetSourceReuseRejectedReason {
    /// Not produced by the current rule: exact duplicates are reused as one
    /// source without replay filters and report `applied`. Retained as a
    /// report value.
    ExactDuplicateSkipped,
    /// The source group uses unsupported source metadata for this conservative pass.
    UnsupportedSourceContract,
    /// The merged scan would widen a variant, and no known upper bound on its
    /// selected bytes is within the configured cap.
    WidenedReadTooLarge,
    /// Reusing the group would merge source variants that need different
    /// dynamic-filter consumer routes.
    DynamicFilterConsumerConflict,
}

pub(crate) struct LocalParquetSourceReuseResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: super::NativeOptimizerChange,
    pub(crate) diagnostics: LocalParquetSourceReuseDiagnostics,
}

/// Runs the rule once over the plan. Each group of two or more variants gets
/// one diagnostics row; an `Err` means the rewriter built a node list that
/// fails the plan invariants.
pub(crate) fn optimize_local_parquet_source_reuse(
    plan: QueryPlan,
    options: &LocalParquetSourceReuseOptions,
    dynamic_filter_options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
    ctx: AnalysisContext<'_>,
) -> Result<LocalParquetSourceReuseResult> {
    if !options.enabled {
        return Ok(LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: LocalParquetSourceReuseDiagnostics::default(),
        });
    }

    let discovery = discover_source_reuse_candidates(&plan, options, dynamic_filter_options, ctx);
    if discovery.rewrites.is_empty() {
        return Ok(LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        });
    }

    let mut rewriter = LocalParquetSourceReuseRewriter::new(&plan, discovery.rewrites);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let rewritten_plan = plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?;
    Ok(LocalParquetSourceReuseResult {
        plan: rewritten_plan,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: discovery.diagnostics,
    })
}

struct SourceReuseDiscovery {
    diagnostics: LocalParquetSourceReuseDiagnostics,
    rewrites: Vec<SourceReuseRewrite>,
}

fn discover_source_reuse_candidates(
    plan: &QueryPlan,
    options: &LocalParquetSourceReuseOptions,
    dynamic_filter_options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
    ctx: AnalysisContext<'_>,
) -> SourceReuseDiscovery {
    let mut groups = BTreeMap::<SourceGroupKey, Vec<SourceVariant>>::new();
    for node in plan.nodes() {
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

    let dynamic_filter_routes_by_source = if groups.values().any(|variants| variants.len() >= 2) {
        dynamic_filter_source_route_signatures(plan, dynamic_filter_options, ctx)
    } else {
        BTreeMap::new()
    };
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

        let superset_read_columns = superset_read_columns(&variants);
        let widened_read_too_large = !exact_duplicate
            && variants.iter().any(|variant| source_reuse_would_widen_variant(variant, &superset_read_columns))
            && superset_selected_bytes_upper_bound(&variants, &superset_read_columns)
                .is_none_or(|bytes| bytes > options.max_variant_selected_bytes);
        let dynamic_filter_consumer_conflict =
            source_group_has_dynamic_filter_consumer_conflict(&variants, &dynamic_filter_routes_by_source);

        // Check order fixes which reason a group with several defects
        // reports; the reason strings are report contracts.
        let rejected_reason = if variants.iter().any(|variant| !variant.supported_contract)
            || source_group_has_conflicting_file_order(&variants)
        {
            Some(LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
        } else if dynamic_filter_consumer_conflict {
            Some(LocalParquetSourceReuseRejectedReason::DynamicFilterConsumerConflict)
        } else if widened_read_too_large {
            Some(LocalParquetSourceReuseRejectedReason::WidenedReadTooLarge)
        } else if source_group_has_conflicting_file_columns(&variants) {
            Some(LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
        } else {
            None
        };

        let applied = rejected_reason.is_none();
        diagnostics.candidates.push(LocalParquetSourceReuseCandidateDiagnostics {
            rule_name: LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
            source_node_ids: source_node_ids.clone(),
            reuse_count: variants.len(),
            stable_file_ids,
            superset_read_columns: superset_read_columns.iter().map(|column| column.name.clone()).collect(),
            branch_filter_count,
            max_variant_selected_bytes: options.max_variant_selected_bytes,
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

fn source_reuse_would_widen_variant(variant: &SourceVariant, superset_read_columns: &[SupersetColumn]) -> bool {
    variant.reader_predicate.is_some() || !variant_read_columns_match_superset(variant, superset_read_columns)
}

/// Upper bound on the selected bytes of the merged superset scan.
///
/// Variants group only when they read the same files with the same row-group
/// selection, so the merged scan reads those row groups for the superset
/// columns. A variant that already reads every superset column bounds it by
/// its own selected bytes. Otherwise column bytes add up, and the sum over the
/// variants covers the union of their columns.
fn superset_selected_bytes_upper_bound(
    variants: &[SourceVariant],
    superset_read_columns: &[SupersetColumn],
) -> Option<u64> {
    variants
        .iter()
        .filter(|variant| variant_read_columns_match_superset(variant, superset_read_columns))
        .filter_map(|variant| variant.selected_source_bytes)
        .min()
        .or_else(|| variants.iter().try_fold(0_u64, |total, variant| total.checked_add(variant.selected_source_bytes?)))
}

/// Route signatures each source would receive, by source node. Route
/// discovery runs last in the pass sequence, so the routes are predicted by
/// running it over a clone of the current plan; the clone is discarded.
fn dynamic_filter_source_route_signatures(
    plan: &QueryPlan,
    options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
    ctx: AnalysisContext<'_>,
) -> BTreeMap<NodeId, BTreeSet<String>> {
    if !options.enabled {
        return BTreeMap::new();
    }
    let result = dynamic_filter_optimizer::optimize_dynamic_filter_route_discovery(plan.clone(), options, ctx);
    let mut signatures = BTreeMap::<NodeId, BTreeSet<String>>::new();
    for route in result.plan.dynamic_filter_routes() {
        let DynamicFilterConsumer::LocalParquetSource { source_node_id } = &route.consumer else {
            continue;
        };
        signatures.entry(*source_node_id).or_default().insert(dynamic_filter_source_route_signature(route));
    }
    signatures
}

fn dynamic_filter_source_route_signature(route: &DynamicFilterRoute) -> String {
    format!(
        "producer_kind={};producer_node={};producer_columns={};consumer_columns={};lowering={}",
        route.producer.kind(),
        route.producer.node_id(),
        route.producer_columns.join("+"),
        route.consumer_columns.join("+"),
        route.consumer_lowering,
    )
}

fn source_group_has_dynamic_filter_consumer_conflict(
    variants: &[SourceVariant],
    dynamic_filter_routes_by_source: &BTreeMap<NodeId, BTreeSet<String>>,
) -> bool {
    let mut route_sets = variants
        .iter()
        .map(|variant| dynamic_filter_routes_by_source.get(&variant.source_node_id).cloned().unwrap_or_default());
    let Some(first) = route_sets.next() else {
        return false;
    };
    route_sets.any(|route_set| route_set != first)
}

/// What variants must share to be one group: the same files, each with the
/// same stable id, resolved contract, decimal reader contract, size and
/// row-group selection, compared as a sorted set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SourceGroupKey {
    file_keys: Vec<FileGroupKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FileGroupKey {
    location: String,
    stable_id: String,
    resolved_contract: String,
    reader_decimal_contract: String,
    file_size_bytes: Option<u64>,
    row_groups: Option<Vec<usize>>,
}

/// One source node of a group.
#[derive(Debug, Clone)]
struct SourceVariant {
    source_node_id: NodeId,
    source: LocalParquetSource,
    group_key: SourceGroupKey,
    /// Full scan contract including columns and predicate; variants with one
    /// exact key are duplicates.
    exact_key: String,
    stable_file_ids: Vec<String>,
    /// File keys in the source's own order. Variants must agree on it because
    /// the shared source keeps the representative's file list and the file
    /// indexes of every variant's columns refer to that order.
    ordered_file_keys: Vec<FileGroupKey>,
    selected_source_bytes: Option<u64>,
    reader_predicate: Option<Expr>,
    supported_contract: bool,
}

fn source_variant(source_node_id: NodeId, source: &LocalParquetSource) -> Option<SourceVariant> {
    if source.files.is_empty() {
        return None;
    }
    let ordered_file_keys = source
        .files
        .iter()
        .map(|file| FileGroupKey {
            location: file.location.reader_path(),
            stable_id: file.stable_id.clone(),
            resolved_contract: resolved_source_contract_key(&file.resolved_source),
            reader_decimal_contract: reader_decimal_contract_key(&file.reader_decimal_types),
            file_size_bytes: file.file_size_bytes,
            row_groups: file.row_groups.clone(),
        })
        .collect::<Vec<_>>();
    let mut file_keys = ordered_file_keys.clone();
    file_keys.sort();
    let group_key = SourceGroupKey { file_keys };
    let stable_file_ids =
        source.files.iter().map(|file| file.location.diagnostic_stable_id(file.stable_id.as_str())).collect();
    let selected_source_bytes = conservative_selected_source_bytes(source);
    Some(SourceVariant {
        source_node_id,
        source: source.clone(),
        group_key,
        exact_key: exact_source_key(source),
        stable_file_ids,
        ordered_file_keys,
        selected_source_bytes,
        reader_predicate: source.reader_predicate.clone(),
        supported_contract: source_contract_supported(source),
    })
}

/// Selected bytes from footer statistics, otherwise the whole file sizes when
/// no file narrows its row groups; `None` when a row-group selection has no
/// statistics, since the file size would then overstate the read.
fn conservative_selected_source_bytes(source: &LocalParquetSource) -> Option<u64> {
    if let Some(bytes) =
        source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes())
    {
        return Some(bytes);
    }
    source.files.iter().try_fold(0_u64, |total, file| {
        if file.row_groups.is_some() {
            return None;
        }
        total.checked_add(file.file_size_bytes?)
    })
}

/// Whether the rule understands every part of the source's read contract.
/// Partition and missing columns, scan limits, zero-column and count-star
/// probe sources, and non-file column references each change what a widened
/// read would mean and are left to their own sources.
fn source_contract_supported(source: &LocalParquetSource) -> bool {
    source.scan_limit.is_none()
        && source_output_schema_override_supported(source)
        && !source.is_zero_column_count_source()
        && !source.is_count_star_probe_source()
        && source.partition_columns.is_empty()
        && source.missing_columns.is_empty()
        && source
            .projected_output_column_source_refs
            .iter()
            .chain(source.pure_filter_column_source_refs.iter())
            .all(|source_ref| matches!(source_ref, SourceColumnRef::File(_)))
        && source.files.iter().all(file_contract_supported)
}

fn source_output_schema_override_supported(source: &LocalParquetSource) -> bool {
    match source.output_schema_override.as_ref() {
        Some(output_schema) => schemas_have_same_visible_fields(&source.schema, output_schema),
        None => true,
    }
}

fn schemas_have_same_visible_fields(left: &Schema, right: &Schema) -> bool {
    left.fields().len() == right.fields().len()
        && left
            .fields()
            .iter()
            .zip(right.fields())
            .all(|(left, right)| left.name() == right.name() && left.data_type() == right.data_type())
}

fn source_group_has_conflicting_file_columns(variants: &[SourceVariant]) -> bool {
    let mut columns_by_file_index = BTreeMap::<usize, (String, Field)>::new();
    let mut columns_by_name = BTreeMap::<String, (usize, Field)>::new();
    for variant in variants {
        for (column, file_index) in
            variant.source.read_columns.iter().zip(variant.source.read_column_file_indices.iter())
        {
            let Ok(field) = variant.source.schema.field_with_name(column) else {
                return true;
            };
            let entry = (column.clone(), field.as_ref().clone());
            match columns_by_file_index.get(file_index) {
                Some(existing) if existing != &entry => return true,
                Some(_) => {},
                None => {
                    columns_by_file_index.insert(*file_index, entry);
                },
            }
            let entry = (*file_index, field.as_ref().clone());
            match columns_by_name.get(column) {
                Some(existing) if existing != &entry => return true,
                Some(_) => {},
                None => {
                    columns_by_name.insert(column.clone(), entry);
                },
            }
        }
    }
    false
}

fn source_group_has_conflicting_file_order(variants: &[SourceVariant]) -> bool {
    let Some(first) = variants.first() else { return false };
    variants.iter().any(|variant| variant.ordered_file_keys != first.ordered_file_keys)
}

fn file_contract_supported(file: &LocalParquetFile) -> bool {
    file.partition_values.is_empty()
}

fn resolved_source_contract_key(resolved_source: &ResolvedParquetSource) -> String {
    let mut key = String::new();
    push_contract_value(&mut key, "original_location", resolved_source.original_location.as_str());
    push_contract_value(&mut key, "reader_location", resolved_source.reader_location());
    push_optional_contract_value(&mut key, "storage_domain", resolved_source.storage_domain.as_deref());
    push_optional_contract_value(&mut key, "content_identity", resolved_source.content_identity.as_deref());
    push_optional_contract_value(&mut key, "storage_profile_id", resolved_source.storage_profile_id());
    push_contract_value(&mut key, "read_strategy", resolved_source.read_strategy().as_str());
    push_contract_value(&mut key, "credential_mode", resolved_source.credential_mode().as_str());
    push_contract_value(&mut key, "lakehouse_cache_status", resolved_source.cache_status.as_str());
    push_contract_value(&mut key, "lakehouse_cache_read_mode", resolved_source.lakehouse_cache_read_mode.as_str());
    push_optional_contract_value(&mut key, "lakehouse_cache_root", resolved_source.lakehouse_cache_root.as_deref());
    push_optional_contract_value(&mut key, "lakehouse_cache_path", resolved_source.lakehouse_cache_path.as_deref());
    push_contract_value(
        &mut key,
        "lakehouse_cache_max_bytes",
        resolved_source
            .lakehouse_cache_max_bytes
            .map_or_else(|| "none".to_string(), |bytes| bytes.to_string())
            .as_str(),
    );
    push_contract_value(&mut key, "lakehouse_cache_fill_mode", resolved_source.lakehouse_cache_fill_mode.as_str());
    key
}

fn reader_decimal_contract_key(reader_decimal_types: &[ParquetReaderDecimalType]) -> String {
    let mut key = String::new();
    push_contract_value(&mut key, "reader_decimal_types.len", reader_decimal_types.len().to_string().as_str());
    for (index, decimal_type) in reader_decimal_types.iter().enumerate() {
        push_contract_value(
            &mut key,
            format!("reader_decimal_types.{index}.column_name").as_str(),
            decimal_type.column_name.as_str(),
        );
        push_contract_value(
            &mut key,
            format!("reader_decimal_types.{index}.type_id").as_str(),
            decimal_type.type_id.as_str(),
        );
        push_contract_value(
            &mut key,
            format!("reader_decimal_types.{index}.cudf_scale").as_str(),
            decimal_type.cudf_scale.to_string().as_str(),
        );
    }
    key
}

fn push_optional_contract_value(output: &mut String, key: &str, value: Option<&str>) {
    match value {
        Some(value) => push_contract_value(output, key, format!("some:{value}").as_str()),
        None => push_contract_value(output, key, "none"),
    }
}

fn push_contract_value(output: &mut String, key: &str, value: &str) {
    output.push_str(escape_kv_delimited(key).as_str());
    output.push('=');
    output.push_str(escape_kv_delimited(value).as_str());
    output.push(';');
}

fn exact_source_key(source: &LocalParquetSource) -> String {
    format!(
        "schema={:?};read={:?};read_idx={:?};projected={:?};projected_refs={:?};pure={:?};pure_refs={:?};predicate={};unique_keys={:?}",
        source.schema,
        source.read_columns,
        source.read_column_file_indices,
        source.projected_output_columns,
        source.projected_output_column_source_refs,
        source.pure_filter_columns,
        source.pure_filter_column_source_refs,
        source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string),
        source.unique_key_columns,
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

fn variant_read_columns_match_superset(variant: &SourceVariant, superset_read_columns: &[SupersetColumn]) -> bool {
    if variant.source.read_columns.len() != superset_read_columns.len() {
        return false;
    }
    let variant_columns = variant.source.read_columns.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let superset_columns = superset_read_columns.iter().map(|column| column.name.as_str()).collect::<BTreeSet<_>>();
    variant_columns == superset_columns
}

/// One applied group: the shared source and how each variant reads it.
#[derive(Debug, Clone)]
struct SourceReuseRewrite {
    variants: BTreeMap<NodeId, SourceVariant>,
    shared_source: LocalParquetSource,
    /// False for exact duplicates, whose predicate the shared source still
    /// applies itself.
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
        let unique_key_columns = representative
            .source
            .unique_key_columns
            .iter()
            .filter(|key| {
                variants.iter().all(|variant| variant.source.unique_key_columns.contains(key))
                    && key.iter().all(|column| projected_output_columns.contains(column))
            })
            .cloned()
            .collect::<Vec<_>>();
        let file_indices_by_column = superset_read_columns
            .iter()
            .map(|column| (column.name.as_str(), column.file_index))
            .collect::<BTreeMap<_, _>>();
        let projected_output_column_file_indices =
            source_file_indices_for_columns(&file_indices_by_column, &projected_output_columns);
        let pure_filter_column_file_indices =
            source_file_indices_for_columns(&file_indices_by_column, &pure_filter_columns);
        let mut files = representative.source.files.clone();
        // These byte counts described the representative's column roles; the
        // shared source has different roles, so they are cleared. Keeping them
        // would report wrong evidence.
        for file in &mut files {
            file.projected_row_group_compressed_bytes = None;
            file.projected_row_group_uncompressed_bytes = None;
            file.pure_filter_row_group_compressed_bytes = None;
            file.pure_filter_row_group_uncompressed_bytes = None;
        }

        let schema = Arc::new(Schema::new(fields));
        let shared_source = LocalParquetSource::from_files(files, schema, read_columns)
            .with_column_roles(projected_output_columns, pure_filter_columns)
            .with_column_file_indices(
                read_column_file_indices,
                projected_output_column_file_indices,
                pure_filter_column_file_indices,
            )
            .with_unique_key_columns(unique_key_columns);
        let variants = variants.into_iter().map(|variant| (variant.source_node_id, variant)).collect();
        Self { variants, shared_source, replay_branch_predicates: true }
    }
}

fn source_file_indices_for_columns(file_indices_by_column: &BTreeMap<&str, usize>, columns: &[String]) -> Vec<usize> {
    columns
        .iter()
        .map(|column| {
            *file_indices_by_column
                .get(column.as_str())
                .expect("source reuse role columns must be read by the shared source")
        })
        .collect()
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
        let new_id =
            self.push_node(PlanNode::Source(Box::new(SourceNode::LocalParquet(source))), PlanFacts::new(output_schema));
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
            let output_schema = Arc::clone(self.nodes[current.index()].output_schema());
            current = self.push_node(
                PlanNode::Filter(FilterNode { input: current, predicate: branch_replay_predicate(predicate) }),
                PlanFacts::new(output_schema),
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
            current = self.push_node(PlanNode::Projection(projection), PlanFacts::new(output_schema));
        }
        self.branch_nodes.insert(old_source_id, current);
        current
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

/// Rebuilds a reader predicate as a filter predicate. Plan validation admits a
/// declared `output_type` on a binary expression only for a decimal division,
/// so every other declared type is dropped and the backend infers it.
fn branch_replay_predicate(predicate: Expr) -> Expr {
    match predicate {
        Expr::Binary { left, op, right, output_type } => {
            let keep_output_type = matches!(op, BinaryOp::Divide)
                && output_type.as_ref().is_some_and(|data_type| matches!(data_type, DataType::Decimal128(_, _)));
            Expr::Binary {
                left: Box::new(branch_replay_predicate(*left)),
                op,
                right: Box::new(branch_replay_predicate(*right)),
                output_type: keep_output_type.then_some(output_type).flatten(),
            }
        },
        Expr::Not(expr) => Expr::logical_not(branch_replay_predicate(*expr)),
        Expr::IsNull(expr) => Expr::is_null(branch_replay_predicate(*expr)),
        Expr::IsNotNull(expr) => Expr::is_not_null(branch_replay_predicate(*expr)),
        Expr::Cast { expr, target_type } => Expr::cast(branch_replay_predicate(*expr), target_type),
        Expr::TryCast { expr, target_type } => Expr::try_cast(branch_replay_predicate(*expr), target_type),
        Expr::Like { expr, pattern, escape, negated } => {
            Expr::Like { expr: Box::new(branch_replay_predicate(*expr)), pattern, escape, negated }
        },
        Expr::Substring { expr, start, length } => Expr::substring(branch_replay_predicate(*expr), start, length),
        Expr::Lower(expr) => Expr::lower(branch_replay_predicate(*expr)),
        Expr::Upper(expr) => Expr::upper(branch_replay_predicate(*expr)),
        Expr::Trim { expr, trim_chars } => Expr::trim(branch_replay_predicate(*expr), trim_chars),
        Expr::LTrim { expr, trim_chars } => Expr::ltrim(branch_replay_predicate(*expr), trim_chars),
        Expr::RTrim { expr, trim_chars } => Expr::rtrim(branch_replay_predicate(*expr), trim_chars),
        Expr::StringContains { expr, pattern } => Expr::string_contains(branch_replay_predicate(*expr), pattern),
        Expr::StartsWith { expr, prefix } => Expr::starts_with(branch_replay_predicate(*expr), prefix),
        Expr::EndsWith { expr, suffix } => Expr::ends_with(branch_replay_predicate(*expr), suffix),
        Expr::StringLength { expr, unit, output_type } => {
            Expr::StringLength { expr: Box::new(branch_replay_predicate(*expr)), unit, output_type }
        },
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => {
            Expr::regexp_replace(branch_replay_predicate(*expr), pattern, replacement, output_type)
        },
        Expr::Concat(exprs) => Expr::concat(exprs.into_iter().map(branch_replay_predicate).collect()),
        Expr::Abs(expr) => Expr::abs(branch_replay_predicate(*expr)),
        Expr::UnaryMath { op, expr, output_type } => Expr::unary_math(op, branch_replay_predicate(*expr), output_type),
        Expr::Round { expr, decimal_places, output_type } => {
            Expr::round(branch_replay_predicate(*expr), decimal_places, output_type)
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: Box::new(branch_replay_predicate(*expr)),
            list: list.into_iter().map(branch_replay_predicate).collect(),
            negated,
        },
        Expr::IsDistinctFrom { left, right } => {
            Expr::is_distinct_from(branch_replay_predicate(*left), branch_replay_predicate(*right))
        },
        Expr::IsNotDistinctFrom { left, right } => {
            Expr::is_not_distinct_from(branch_replay_predicate(*left), branch_replay_predicate(*right))
        },
        Expr::Case { branches, else_expr } => Expr::case(
            branches
                .into_iter()
                .map(|branch| CaseWhen::new(branch_replay_predicate(branch.when), branch_replay_predicate(branch.then)))
                .collect(),
            branch_replay_predicate(*else_expr),
        ),
        Expr::Extract { field, expr } => Expr::extract(field, branch_replay_predicate(*expr)),
        Expr::DateTrunc { field, expr, output_type } => {
            Expr::date_trunc(field, branch_replay_predicate(*expr), output_type)
        },
        Expr::Alias { expr, alias } => branch_replay_predicate(*expr).alias(alias),
        Expr::Column(_) | Expr::Literal(_) => predicate,
    }
}
