use std::collections::BTreeSet;
#[cfg(feature = "cugraph")]
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
#[cfg(feature = "cugraph")]
use datafusion::{datasource::source::DataSourceExec, physical_plan::ExecutionPlan};
use datafusion::{
    datasource::{
        listing::PartitionedFile,
        physical_plan::{FileScanConfig, FileSource, ParquetSource},
    },
    physical_plan::filter::FilterExec,
};
use parquet::file::metadata::{KeyValue, ParquetMetaData};
use query_engine::{
    expr::LiteralValue as NativeLiteralValue,
    parquet_stats::{ColumnByteKind, row_group_column_bytes, row_group_null_counts_for_root_column},
    source::{
        GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY, GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY,
        GraphSimpleDirectedEdgeTopology, LocalParquetFile as NativeLocalParquetFile,
        PartitionColumnDisposition as NativePartitionColumnDisposition, SourceColumnRef as NativeSourceColumnRef,
    },
};
#[cfg(feature = "cugraph")]
use query_engine::{
    plan::{GraphAlgorithmSpec, GraphResidencyDescriptor, graph_residency_descriptor_for_local_parquet_source},
    source::LocalParquetSource as NativeLocalParquetSource,
};
use serde_json::Value;

#[cfg(feature = "cugraph")]
use super::footers::FooterScope;
use super::{
    footers::parquet_footer,
    literal::{NativeLiteralContext, native_literal_value},
    pruning::{
        coalesced_local_parquet_scan_files, explicit_row_group_selection, local_object_path_to_filesystem,
        metadata_row_group_selections, scan_predicate_is_dynamic_filter_hint, validate_pruning_only_parquet_predicate,
    },
};
use crate::capability::CapabilityReason;

mod columns;
mod diagnostics;
mod s3_parquet;

pub(super) use columns::{
    NativeParquetColumnRequirements, native_source_output_schema, required_parquet_column_requirements,
    required_parquet_column_requirements_for_scan_columns,
};
pub(super) use diagnostics::{local_parquet_source_diagnostics, source_diagnostics_for_normalized_plan};
pub(crate) use s3_parquet::{ParquetReadStrategyConfig, S3_STORAGE_PROFILE_ID, ambient_s3_endpoint};
use s3_parquet::{ParquetSourceMetadata, resolve_parquet_source_with_metadata};

pub(super) fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<(), CapabilityReason> {
    if !native_parquet_object_store_scan_is_supported(file_scan) {
        return Err(CapabilityReason::NativeSourceNonLocalObjectStore);
    }
    validate_partition_columns(file_scan, &column_requirements.partition_column_indices)?;
    if file_scan.expr_adapter_factory.is_some() {
        return Err(CapabilityReason::NativeSourceExprAdapter);
    }
    if !native_parquet_file_reader_factory_is_supported(parquet_source) {
        return Err(CapabilityReason::NativeSourceParquetFileReaderFactory);
    }
    if !native_declared_output_ordering_scan_is_supported(file_scan) {
        return Err(CapabilityReason::NativeSourceOutputOrdering);
    }
    if !native_preserve_order_scan_is_supported(file_scan) {
        return Err(CapabilityReason::NativeSourcePreserveOrder);
    }
    if parquet_source
        .filter()
        .is_some_and(|predicate| filter.is_none() && scan_predicate_is_dynamic_filter_hint(predicate.as_ref()))
    {
        validate_pruning_only_parquet_predicate(parquet_source, filter, &source_schema)?;
    }
    local_parquet_files(file_scan, parquet_source, filter, source_schema, column_requirements, source_reads)?;
    Ok(())
}

pub(super) fn native_preserve_order_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    !file_scan.preserve_order || native_scan_read_order_matches_file_groups(file_scan)
}

pub(super) fn native_declared_output_ordering_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    file_scan.output_ordering.is_empty() || native_scan_read_order_matches_file_groups(file_scan)
}

/// Whether the native reader reproduces the row order of the scan's file
/// groups.
///
/// Native lowering reads the scan as one sequential file list. One file group
/// is read in its own order. Groups that are byte ranges of a single file are
/// coalesced back into that file (`coalesced_local_parquet_scan_files`) and
/// read in row-group order, which is the file's order. Groups over several
/// files are parallel sorted runs that a sequential read does not merge.
fn native_scan_read_order_matches_file_groups(file_scan: &FileScanConfig) -> bool {
    if file_scan.file_groups.len() == 1 {
        return true;
    }
    let mut locations =
        file_scan.file_groups.iter().flat_map(|group| group.files()).map(|file| &file.object_meta.location);
    locations.next().is_some_and(|first| locations.all(|location| location == first))
}

pub(super) fn native_parquet_object_store_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    let object_store_url = file_scan.object_store_url.as_str();
    object_store_url.starts_with("file://") || object_store_url.starts_with("s3://")
}

const STOCK_PARQUET_FILE_READER_FACTORY_DEBUG_PREFIXES: [&str; 2] =
    ["DefaultParquetFileReaderFactory { store: ", "CachedParquetFileReaderFactory { store: "];

pub(crate) fn native_parquet_file_reader_factory_is_supported(parquet_source: &ParquetSource) -> bool {
    // DataFusion's normal local-file scans can carry its default or cached
    // object-store reader factory. Native local reads represent that contract
    // directly through filesystem paths; caller-provided factories remain
    // rejected because they may change bytes, metadata, caching, credentials,
    // or IO semantics in ways the native source IR cannot represent.
    //
    // `ParquetFileReaderFactory` in DataFusion 55 has no `Any` supertrait, so
    // the concrete type is unreachable. The derived `Debug` of the two stock
    // factories opens with `Name { store: `; requiring the full opening, not
    // only the type name, rejects wrappers such as
    // `DefaultParquetFileReaderFactoryProxy` and same-named types with other
    // fields. A caller type that reproduces both name and first field is
    // indistinguishable until the trait exposes a type identity.
    parquet_source.parquet_file_reader_factory().is_none_or(|factory| {
        let debug = format!("{factory:?}");
        STOCK_PARQUET_FILE_READER_FACTORY_DEBUG_PREFIXES.iter().any(|prefix| debug.starts_with(prefix))
    })
}

/// Derives a stable graph-residency descriptor directly from a DataFusion
/// local-Parquet scan.
///
/// This is intentionally narrower than native execution lowering. Graph cache
/// identity is a complete edge *rowset* contract, so the scan's parallel file
/// grouping and declared output order do not affect it. They do matter to a
/// host-visible native execution plan and therefore remain rejected by
/// [`validate_slice1_parquet_scan`].
///
/// Pushed reader predicates and scan limits can change the effective rowset in
/// ways this direct adapter does not yet model, so they continue to bypass.
#[cfg(feature = "cugraph")]
pub(crate) fn graph_residency_descriptor_for_datafusion_local_parquet_scan(
    source_exec: &DataSourceExec,
    graph_spec: &GraphAlgorithmSpec,
    source_reads: &ParquetReadStrategyConfig,
) -> Option<GraphResidencyDescriptor> {
    let source = graph_local_parquet_source_for_datafusion_local_parquet_scan(source_exec, source_reads)?;
    graph_residency_descriptor_for_local_parquet_source(&source, graph_spec).descriptor().cloned()
}

/// Builds the engine-owned source model for the exact DataFusion local-Parquet
/// rowset accepted by the direct graph path.
///
/// Callers can use this only to carry source-owned descriptive evidence into
/// an engine capability decision; filters, limits, and unsupported scan semantics
/// remain fail-closed in the shared bridge below.
#[cfg(feature = "cugraph")]
pub(crate) fn graph_local_parquet_source_for_datafusion_local_parquet_scan(
    source_exec: &DataSourceExec,
    source_reads: &ParquetReadStrategyConfig,
) -> Option<NativeLocalParquetSource> {
    let _footers = FooterScope::enter();
    let (file_scan, parquet_source) = source_exec.downcast_to_file_source::<ParquetSource>()?;
    graph_residency_local_parquet_scan_source(source_exec, file_scan, parquet_source, source_reads).ok()
}

/// Returns the exact row count for the same narrow local-Parquet rowset that
/// may form a graph-residency key.
///
/// The conversion deliberately rejects pushed predicates and scan limits, so
/// the total is footer-proven for the exact rows consumed by a graph build.
/// This lets graph admission use a real construction model when DataFusion's
/// generic scan statistics are merely inexact.
#[cfg(feature = "cugraph")]
pub(crate) fn exact_row_count_for_datafusion_local_parquet_scan(
    source_exec: &DataSourceExec,
    source_reads: &ParquetReadStrategyConfig,
) -> Option<u64> {
    let source = graph_local_parquet_source_for_datafusion_local_parquet_scan(source_exec, source_reads)?;
    source.exact_selected_row_count()
}

/// Returns the footer-proven graph vertex upper bound for the same exact
/// local-Parquet rowset accepted by the graph-residency bridge.
///
/// This deliberately shares the rowset checks with
/// [`exact_row_count_for_datafusion_local_parquet_scan`]: a detached fixture
/// manifest or a query/session setting must never price a different scan.
#[cfg(feature = "cugraph")]
pub(crate) fn graph_vertex_upper_bound_for_datafusion_local_parquet_scan(
    source_exec: &DataSourceExec,
    source_reads: &ParquetReadStrategyConfig,
) -> Option<u64> {
    let source = graph_local_parquet_source_for_datafusion_local_parquet_scan(source_exec, source_reads)?;
    source.graph_vertex_upper_bound()
}

/// Returns whether an exact DataFusion local-Parquet edge rowset proves that
/// Python-compatible null filtering and directed-pair deduplication are
/// semantic no-ops for these endpoint columns.
///
/// The engine-owned source contract rejects projections that stop denoting the
/// complete physical file rowset and refuses multi-file inputs because a
/// per-file certificate cannot establish cross-file uniqueness.
#[cfg(feature = "cugraph")]
pub(crate) fn datafusion_local_parquet_scan_has_simple_directed_edge_topology(
    source_exec: &DataSourceExec,
    source_column: &str,
    destination_column: &str,
    source_reads: &ParquetReadStrategyConfig,
) -> bool {
    graph_local_parquet_source_for_datafusion_local_parquet_scan(source_exec, source_reads)
        .is_some_and(|source| source.has_simple_directed_edge_topology(source_column, destination_column))
}

#[cfg(feature = "cugraph")]
fn graph_residency_local_parquet_scan_source(
    source_exec: &DataSourceExec,
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeLocalParquetSource, CapabilityReason> {
    // Cache identity must describe the exact graph rowset. The general native
    // lowerer can model these features for execution, but this source-only
    // bridge deliberately leaves them unkeyable until it also fingerprints
    // their complete semantics.
    if file_scan.limit.is_some() || parquet_source.filter().is_some() {
        return Err(CapabilityReason::NativeSourcePushedDownParquetPredicate);
    }
    if !native_parquet_object_store_scan_is_supported(file_scan) {
        return Err(CapabilityReason::NativeSourceNonLocalObjectStore);
    }
    if file_scan.expr_adapter_factory.is_some() {
        return Err(CapabilityReason::NativeSourceExprAdapter);
    }
    if !native_parquet_file_reader_factory_is_supported(parquet_source) {
        return Err(CapabilityReason::NativeSourceParquetFileReaderFactory);
    }

    let required_scan_output_indices = (0..source_exec.schema().fields().len()).collect::<Vec<_>>();
    let requirements = required_parquet_column_requirements_for_scan_columns(
        file_scan,
        parquet_source,
        None,
        &required_scan_output_indices,
    )?;
    validate_partition_columns(file_scan, &requirements.partition_column_indices)?;
    let source_schema = native_source_output_schema(file_scan, &requirements)?;
    let files =
        local_parquet_files(file_scan, parquet_source, None, Arc::clone(&source_schema), &requirements, source_reads)?;

    Ok(NativeLocalParquetSource::from_files(files, source_schema, requirements.read_columns)
        .with_column_roles(requirements.projected_output_columns, requirements.pure_filter_columns)
        .with_column_source_refs(
            requirements.read_column_file_indices,
            requirements.projected_output_column_source_refs,
            requirements.pure_filter_column_source_refs,
        )
        .with_file_column_count(file_scan.file_schema().fields().len())
        .with_partition_columns(requirements.partition_columns)
        .with_missing_columns(requirements.missing_columns))
}

/// Converts DataFusion `PartitionedFile` entries into native local Parquet files.
///
/// The bridge carries over stable file identity, filesystem paths, selected row
/// groups, and required partition values while rejecting scan metadata that the
/// native source contract cannot represent.
pub(super) fn local_parquet_files(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<Vec<NativeLocalParquetFile>, CapabilityReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if local_parquet_scan_is_empty(file_scan) {
        return Err(CapabilityReason::NativeSourceEmpty);
    }
    let scan_files = coalesced_local_parquet_scan_files(&files)?;
    let metadata_row_group_selections =
        metadata_row_group_selections(&files, file_scan, parquet_source, filter, &source_schema);
    let mut native_files = Vec::with_capacity(scan_files.len());
    for (file_index, scan_file) in scan_files.iter().enumerate() {
        let file = scan_file.file;
        let mut native_file = native_parquet_file_for_scan(file_scan, file, source_reads)?;
        if !column_requirements.partition_column_indices.is_empty() {
            native_file = native_file
                .with_partition_values(native_partition_values(file, &column_requirements.partition_column_indices)?);
        }
        if let Some(row_groups) = scan_file.byte_range_row_groups.clone() {
            native_file = native_file.with_row_groups(row_groups);
        } else if let Some(row_groups) = explicit_row_group_selection(file)? {
            native_file = native_file.with_row_groups(row_groups);
        } else if let Some(row_groups) =
            metadata_row_group_selections.as_ref().and_then(|selections| selections.get(file_index))
        {
            native_file = native_file.with_row_groups(row_groups.clone());
        }
        native_file = attach_row_group_byte_metadata(native_file, column_requirements);
        native_files.push(native_file);
    }
    Ok(native_files)
}

pub(super) fn local_parquet_scan_is_empty(file_scan: &FileScanConfig) -> bool {
    file_scan.file_groups.iter().all(|group| group.files().is_empty())
}

fn native_parquet_file_for_scan(
    file_scan: &FileScanConfig,
    file: &PartitionedFile,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeLocalParquetFile, CapabilityReason> {
    if file_scan.object_store_url.as_str().starts_with("file://") {
        let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
        let stable_id = format!(
            "local:{}:{}:mtime={}",
            path.display(),
            file.object_meta.size,
            file.object_meta.last_modified.to_rfc3339()
        );
        return Ok(NativeLocalParquetFile::new(path)
            .with_stable_id(stable_id)
            .with_file_size_bytes(file.object_meta.size));
    }

    if !file_scan.object_store_url.as_str().starts_with("s3://") {
        return Err(CapabilityReason::NativeSourceNonLocalObjectStore);
    }

    let uri = object_store_file_uri(file_scan, file)?;
    let metadata = ParquetSourceMetadata {
        size_bytes: Some(file.object_meta.size),
        content_identity: file.object_meta.e_tag.as_ref().or(file.object_meta.version.as_ref()).cloned(),
    };
    let resolved = resolve_parquet_source_with_metadata(uri.as_str(), source_reads, &metadata)
        .map_err(|_error| CapabilityReason::NativeSourceNonLocalObjectStore)?;
    NativeLocalParquetFile::from_resolved_parquet_source(resolved)
        .map(|native_file| native_file.with_file_size_bytes(file.object_meta.size))
        .map_err(|_error| CapabilityReason::NativeSourceNonLocalObjectStore)
}

fn object_store_file_uri(file_scan: &FileScanConfig, file: &PartitionedFile) -> Result<String, CapabilityReason> {
    let location = file.object_meta.location.as_ref();
    if location.starts_with("s3://") {
        return Ok(location.to_string());
    }
    if location.contains("://") {
        return Err(CapabilityReason::NativeSourceNonLocalObjectStore);
    }
    let base = file_scan.object_store_url.as_str().trim_end_matches('/');
    let key = location.trim_start_matches('/');
    Ok(format!("{base}/{key}"))
}

pub(super) fn attach_row_group_byte_metadata(
    mut native_file: NativeLocalParquetFile,
    column_requirements: &NativeParquetColumnRequirements,
) -> NativeLocalParquetFile {
    let Ok(footer) = parquet_footer(native_file.path.as_path()) else {
        return native_file;
    };
    let metadata = footer.as_ref();
    let read_column_indexes = column_requirements.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
    let projected_column_indexes = column_requirements
        .projected_output_column_source_refs
        .iter()
        .filter_map(file_source_ref_index)
        .collect::<BTreeSet<_>>();
    let pure_filter_column_indexes = column_requirements
        .pure_filter_column_source_refs
        .iter()
        .filter_map(file_source_ref_index)
        .filter(|index| !projected_column_indexes.contains(index))
        .collect::<BTreeSet<_>>();

    if let Some(row_group_bytes) = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &read_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>()
    {
        native_file = native_file.with_row_group_bytes(row_group_bytes);
    }
    let projected_compressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>();
    let projected_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Uncompressed))
        .collect::<Option<Vec<_>>>();
    if let (Some(compressed), Some(uncompressed)) = (projected_compressed_bytes, projected_uncompressed_bytes) {
        native_file = native_file.with_projected_row_group_bytes(compressed, uncompressed);
    }
    let pure_filter_compressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>();
    let pure_filter_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Uncompressed))
        .collect::<Option<Vec<_>>>();
    if let (Some(compressed), Some(uncompressed)) = (pure_filter_compressed_bytes, pure_filter_uncompressed_bytes) {
        native_file = native_file.with_pure_filter_row_group_bytes(compressed, uncompressed);
    }
    if let Some(row_group_rows) = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows()).ok())
        .collect::<Option<Vec<_>>>()
    {
        native_file = native_file.with_row_group_rows(row_group_rows);
    }
    if let Some(vertex_upper_bound) = graph_vertex_upper_bound_from_footer(metadata) {
        native_file = native_file.with_graph_vertex_upper_bound(vertex_upper_bound);
    }
    if let Some(topology) = graph_simple_directed_edge_topology_from_footer(metadata) {
        native_file = native_file.with_graph_simple_directed_edge_topology(topology);
    }
    for column_index in read_column_indexes {
        let Some(null_counts) = row_group_null_counts_for_root_column(metadata, column_index) else {
            continue;
        };
        native_file = native_file.with_row_group_null_counts(column_index, null_counts);
    }

    native_file
}

/// Reads exactly one parseable graph-vertex upper bound from the Parquet
/// footer.  Duplicate, empty, or malformed metadata is deliberately ignored
/// rather than selecting an arbitrary producer value.
fn graph_vertex_upper_bound_from_footer(metadata: &ParquetMetaData) -> Option<u64> {
    graph_vertex_upper_bound_from_key_value_metadata(metadata.file_metadata().key_value_metadata()?)
}

fn graph_vertex_upper_bound_from_key_value_metadata(metadata: &[KeyValue]) -> Option<u64> {
    let mut entries = metadata.iter().filter(|entry| entry.key == GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY);
    let value = entries.next()?.value.as_deref()?.parse::<u64>().ok()?;
    entries.next().is_none().then_some(value)
}

/// JSON schema carried by the v1 simple-directed topology footer key.
///
/// Keep the payload intentionally small and index-based: the native source
/// contract compares the declared physical column roles to the resolved scan
/// bindings, so aliases cannot accidentally claim a proof for another column.
/// Reads exactly one valid simple-directed topology certificate from a Parquet
/// footer. A malformed, duplicate, or overbroad entry remains unavailable;
/// callers then retain the normal Python-compatible edge-normalize path.
fn graph_simple_directed_edge_topology_from_footer(
    metadata: &ParquetMetaData,
) -> Option<GraphSimpleDirectedEdgeTopology> {
    graph_simple_directed_edge_topology_from_key_value_metadata(metadata.file_metadata().key_value_metadata()?)
}

fn graph_simple_directed_edge_topology_from_key_value_metadata(
    metadata: &[KeyValue],
) -> Option<GraphSimpleDirectedEdgeTopology> {
    let mut entries =
        metadata.iter().filter(|entry| entry.key == GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY);
    let value = entries.next()?.value.as_deref()?;
    if entries.next().is_some() {
        return None;
    }
    let Value::Object(footer) = serde_json::from_str(value).ok()? else {
        return None;
    };
    if footer.len() != 2 {
        return None;
    }
    let source_column_index = footer.get("source_column_index")?.as_u64()?.try_into().ok()?;
    let destination_column_index = footer.get("destination_column_index")?.as_u64()?.try_into().ok()?;
    GraphSimpleDirectedEdgeTopology::new(source_column_index, destination_column_index)
}

fn file_source_ref_index(source_ref: &NativeSourceColumnRef) -> Option<usize> {
    match source_ref {
        NativeSourceColumnRef::File(index) => Some(*index),
        NativeSourceColumnRef::Partition(_) | NativeSourceColumnRef::MissingNull(_) => None,
    }
}

fn validate_partition_columns(
    file_scan: &FileScanConfig,
    required_partition_indices: &[usize],
) -> Result<(), CapabilityReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if required_partition_indices.is_empty() {
        return Ok(());
    }

    let partition_column_count = file_scan.table_partition_cols().len();
    if required_partition_indices.iter().any(|index| *index >= partition_column_count) {
        return Err(CapabilityReason::NativeSourcePartitionValues);
    }
    for file in &files {
        if required_partition_indices.iter().any(|index| file.partition_values.get(*index).is_none()) {
            return Err(CapabilityReason::NativeSourcePartitionValues);
        }
        native_partition_values(file, required_partition_indices)?;
    }
    Ok(())
}

fn partition_column_disposition(
    file_scan: &FileScanConfig,
    files: &[&PartitionedFile],
    required_partition_indices: &[usize],
) -> NativePartitionColumnDisposition {
    if required_partition_indices.is_empty() {
        return NativePartitionColumnDisposition::Absent;
    }
    let partition_column_count = file_scan.table_partition_cols().len();
    if required_partition_indices.iter().all(|index| *index < partition_column_count)
        && files.iter().all(|file| native_partition_values(file, required_partition_indices).is_ok())
    {
        NativePartitionColumnDisposition::Represented
    } else {
        NativePartitionColumnDisposition::Rejected
    }
}

fn native_partition_values(
    file: &PartitionedFile,
    required_partition_indices: &[usize],
) -> Result<Vec<NativeLiteralValue>, CapabilityReason> {
    required_partition_indices
        .iter()
        .map(|index| {
            file.partition_values
                .get(*index)
                .ok_or(CapabilityReason::NativeSourcePartitionValues)
                .and_then(|value| native_literal_value(value, NativeLiteralContext::SourceValue))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::{
        file::metadata::KeyValue,
        schema::{parser::parse_message_type, types::SchemaDescriptor},
    };
    use query_engine::parquet_stats::single_leaf_primitive_root_column_index;

    use super::{
        GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY, GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY,
        GraphSimpleDirectedEdgeTopology, graph_simple_directed_edge_topology_from_key_value_metadata,
        graph_vertex_upper_bound_from_key_value_metadata,
    };

    #[test]
    fn null_count_leaf_mapping_refuses_nested_roots_and_maps_later_primitives() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional group payload {
                optional int32 x;
                optional int32 y;
              }
              optional int32 fact_key;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), None);
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(2));
    }

    #[test]
    fn graph_vertex_footer_metadata_accepts_one_valid_value_and_fails_closed_otherwise() {
        let entry = |value: Option<&str>| {
            KeyValue::new(GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY.to_string(), value.map(ToString::to_string))
        };

        assert_eq!(graph_vertex_upper_bound_from_key_value_metadata(&[entry(Some("4894081"))]), Some(4_894_081));
        assert_eq!(graph_vertex_upper_bound_from_key_value_metadata(&[entry(Some("bad"))]), None);
        assert_eq!(graph_vertex_upper_bound_from_key_value_metadata(&[entry(None)]), None);
        assert_eq!(
            graph_vertex_upper_bound_from_key_value_metadata(&[entry(Some("7")), entry(Some("8"))]),
            None,
            "duplicate footer entries must not let admission choose an arbitrary producer value"
        );
    }

    #[test]
    fn simple_directed_edge_topology_footer_requires_one_exact_schema() {
        let entry = |value: Option<&str>| {
            KeyValue::new(
                GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY.to_string(),
                value.map(ToString::to_string),
            )
        };

        assert_eq!(
            graph_simple_directed_edge_topology_from_key_value_metadata(&[entry(Some(
                r#"{"destination_column_index":2,"source_column_index":1}"#,
            ))]),
            GraphSimpleDirectedEdgeTopology::new(1, 2),
        );
        assert_eq!(graph_simple_directed_edge_topology_from_key_value_metadata(&[entry(Some("not-json"))]), None);
        assert_eq!(
            graph_simple_directed_edge_topology_from_key_value_metadata(&[entry(Some(
                r#"{"source_column_index":1,"destination_column_index":1}"#,
            ))]),
            None,
            "a topology certificate must name two distinct endpoint columns"
        );
        assert_eq!(
            graph_simple_directed_edge_topology_from_key_value_metadata(&[entry(Some(
                r#"{"source_column_index":1,"destination_column_index":2,"extra":true}"#,
            ))]),
            None,
            "unknown footer fields must not silently broaden a certificate"
        );
        assert_eq!(
            graph_simple_directed_edge_topology_from_key_value_metadata(&[
                entry(Some(r#"{"source_column_index":1,"destination_column_index":2}"#)),
                entry(Some(r#"{"source_column_index":1,"destination_column_index":2}"#)),
            ]),
            None,
            "duplicate footer entries must not let source lowering choose an arbitrary certificate"
        );
    }
}
