use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::{consumers_by_node, dynamic_filter as dynamic_filter_optimizer, rewrite};
use crate::{
    dynamic_filter::DynamicFilterConsumer,
    error::{QueryError, QueryErrorCode},
    expr::{BinaryOp, CaseWhen, Expr},
    plan::{
        FilterNode, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
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
    /// Maximum selected source bytes for each source variant; variants with unknown selected bytes are not reused.
    pub max_variant_selected_bytes: u64,
}

impl LocalParquetSourceReuseOptions {
    /// Default per-variant selected byte cap for source reuse.
    pub const DEFAULT_MAX_VARIANT_SELECTED_BYTES: u64 = 500 * 1024 * 1024;

    /// Returns options with source reuse disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, max_variant_selected_bytes: Self::DEFAULT_MAX_VARIANT_SELECTED_BYTES }
    }

    /// Validates internal consistency for local Parquet source-reuse limits.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.max_variant_selected_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.local_parquet_source_reuse_options.validate",
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
    /// Configured maximum selected bytes accepted for each source variant.
    pub max_variant_selected_bytes: u64,
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
    /// Reusing the group would merge source variants that need different
    /// dynamic-filter consumer routes.
    DynamicFilterConsumerConflict,
}

#[doc(hidden)]
pub struct LocalParquetSourceReuseResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: LocalParquetSourceReuseDiagnostics,
}

#[doc(hidden)]
pub fn optimize_local_parquet_source_reuse(
    plan: QueryPlan,
    options: &LocalParquetSourceReuseOptions,
    dynamic_filter_options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
) -> LocalParquetSourceReuseResult {
    if !options.enabled {
        return LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: LocalParquetSourceReuseDiagnostics::default(),
        };
    }

    let discovery = discover_source_reuse_candidates(&plan, options, dynamic_filter_options);
    if discovery.rewrites.is_empty() {
        return LocalParquetSourceReuseResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let mut rewriter = LocalParquetSourceReuseRewriter::new(&plan, discovery.rewrites);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let rewritten_plan = rewrite::finish_rewritten_plan(rewriter.nodes, new_root, plan.sink())
        .with_keyed_hash_join_strategy_options_from(&plan);
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
    dynamic_filter_options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
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

    let dynamic_filter_routes_by_source = if groups.values().any(|variants| variants.len() >= 2) {
        dynamic_filter_source_route_signatures(plan, dynamic_filter_options)
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
        let has_unknown_variant_size = variants.iter().any(|variant| variant.selected_source_bytes.is_none());
        let any_variant_too_large = variants.iter().any(|variant| {
            variant.selected_source_bytes.is_some_and(|bytes| bytes > options.max_variant_selected_bytes)
        });
        let large_variant_blocks_source_adjacent_grouped_aggregate = (has_unknown_variant_size
            || any_variant_too_large)
            && variants.iter().any(|variant| {
                source_variant_feeds_source_adjacent_grouped_aggregate(plan, variant.source_node_id)
                    && source_reuse_would_widen_variant(variant, &superset_read_columns)
            });
        let dynamic_filter_consumer_conflict =
            source_group_has_dynamic_filter_consumer_conflict(&variants, &dynamic_filter_routes_by_source);

        let rejected_reason = if variants.iter().any(|variant| !variant.supported_contract)
            || source_group_has_conflicting_file_order(&variants)
        {
            Some(LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
        } else if dynamic_filter_consumer_conflict {
            Some(LocalParquetSourceReuseRejectedReason::DynamicFilterConsumerConflict)
        } else if large_variant_blocks_source_adjacent_grouped_aggregate {
            Some(LocalParquetSourceReuseRejectedReason::VariantTooLarge)
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

fn source_variant_feeds_source_adjacent_grouped_aggregate(plan: &QueryPlan, source_node_id: NodeId) -> bool {
    let parents = consumers_by_node(plan);
    let mut stack = parents.get(&source_node_id).cloned().unwrap_or_default();
    let mut visited = BTreeSet::new();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }
        let Some(node) = plan.node(node_id) else {
            continue;
        };
        match &node.kind {
            PlanNode::Aggregate(aggregate)
                if !aggregate.spec.group_exprs.is_empty() && aggregate.spec.grouping_sets.is_none() =>
            {
                return true;
            },
            PlanNode::Filter(_) | PlanNode::Projection(_) => {
                stack.extend(parents.get(&node_id).into_iter().flatten().copied());
            },
            _ => {},
        }
    }
    false
}

fn source_reuse_would_widen_variant(variant: &SourceVariant, superset_read_columns: &[SupersetColumn]) -> bool {
    variant.reader_predicate.is_some() || !variant_read_columns_match_superset(variant, superset_read_columns)
}

fn dynamic_filter_source_route_signatures(
    plan: &QueryPlan,
    options: &dynamic_filter_optimizer::DynamicFilterRouteDiscoveryOptions,
) -> BTreeMap<NodeId, BTreeSet<String>> {
    if !options.enabled {
        return BTreeMap::new();
    }
    let result = dynamic_filter_optimizer::optimize_dynamic_filter_route_discovery(plan.clone(), options);
    let mut signatures = BTreeMap::<NodeId, BTreeSet<String>>::new();
    for route in result.plan.dynamic_filter_routes() {
        let DynamicFilterConsumer::LocalParquetSource { source_node_id } = &route.consumer else {
            continue;
        };
        signatures.entry(*source_node_id).or_default().insert(dynamic_filter_source_route_signature(route));
    }
    signatures
}

fn dynamic_filter_source_route_signature(route: &crate::dynamic_filter::DynamicFilterRoute) -> String {
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

#[derive(Debug, Clone)]
struct SourceVariant {
    source_node_id: NodeId,
    source: LocalParquetSource,
    group_key: SourceGroupKey,
    exact_key: String,
    stable_file_ids: Vec<String>,
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

fn source_contract_supported(source: &LocalParquetSource) -> bool {
    source.scan_limit.is_none()
        && source_output_schema_override_supported(source)
        && !source.is_zero_column_count_source()
        && !source.is_count_star_probe_source()
        && !source.has_row_delete_files()
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
    push_contract_value(&mut key, "storage_profile_id", resolved_source.storage_profile_id().unwrap_or("none"));
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
    push_contract_value(output, key, value.unwrap_or("none"));
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
                PlanNode::Filter(FilterNode { input: current, predicate: branch_replay_predicate(predicate) }),
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
        Expr::StringLength { expr, output_type } => Expr::string_length(branch_replay_predicate(*expr), output_type),
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => {
            Expr::regexp_replace(branch_replay_predicate(*expr), pattern, replacement, output_type)
        },
        Expr::Concat(exprs) => Expr::concat(exprs.into_iter().map(branch_replay_predicate).collect()),
        Expr::Abs(expr) => Expr::abs(branch_replay_predicate(*expr)),
        Expr::Sqrt { expr, output_type } => Expr::sqrt(branch_replay_predicate(*expr), output_type),
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
