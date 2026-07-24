use std::collections::BTreeSet;

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    physical_plan::filter::FilterExec,
};
use nexus_query_engine::{
    expr::LiteralValue as NativeLiteralValue,
    parquet_stats::{ColumnByteKind, row_group_column_bytes, row_group_null_counts_for_root_column},
    source::{
        LocalParquetFile as NativeLocalParquetFile, PartitionColumnDisposition as NativePartitionColumnDisposition,
        SourceColumnRef as NativeSourceColumnRef,
    },
};
use parquet::file::reader::{FileReader, SerializedFileReader};

use super::{
    literal::{NativeLiteralContext, native_literal_value},
    pruning::{
        coalesced_local_parquet_scan_files, explicit_row_group_selection, local_object_path_to_filesystem,
        metadata_row_group_selections, scan_predicate_is_dynamic_filter_hint, validate_pruning_only_parquet_predicate,
    },
};
use crate::{
    capability::CapabilityReason,
    table_format::{
        TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatReadCapability,
        resolve_table_format_parquet_source_with_metadata,
    },
};

mod columns;
mod diagnostics;

pub(super) use columns::{
    NativeParquetColumnRequirements, native_source_output_schema, required_parquet_column_requirements,
    required_parquet_column_requirements_for_scan_columns,
};
pub(super) use diagnostics::{local_parquet_source_diagnostics, source_diagnostics_for_normalized_plan};

pub(super) fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
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
    local_parquet_files(file_scan, parquet_source, filter, source_schema, column_requirements)?;
    Ok(())
}

pub(super) fn native_preserve_order_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    // DataFusion may use multiple ordered file groups as parallel sorted runs.
    // Native lowering flattens groups into one sequential file list, which only
    // preserves the order contract when there is a single group.
    !file_scan.preserve_order || file_scan.file_groups.len() == 1
}

pub(super) fn native_declared_output_ordering_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    // See `native_preserve_order_scan_is_supported`: flattening multiple file
    // groups does not merge parallel sorted runs, so it cannot reproduce a
    // declared scan ordering.
    file_scan.output_ordering.is_empty() || file_scan.file_groups.len() == 1
}

pub(super) fn native_parquet_object_store_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    let object_store_url = file_scan.object_store_url.as_str();
    object_store_url.starts_with("file://") || object_store_url.starts_with("s3://")
}

pub(super) fn native_parquet_file_reader_factory_is_supported(parquet_source: &ParquetSource) -> bool {
    let Some(factory) = parquet_source.parquet_file_reader_factory() else {
        return true;
    };

    // DataFusion's normal local-file scans can carry its default or cached
    // object-store reader factory. Native local reads represent that contract
    // directly through filesystem paths; caller-provided factories remain
    // rejected because they may change bytes, metadata, caching, credentials,
    // or IO semantics in ways the native source IR cannot represent.
    let debug = format!("{factory:?}");
    debug.starts_with("DefaultParquetFileReaderFactory") || debug.starts_with("CachedParquetFileReaderFactory")
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
        let mut native_file = native_parquet_file_for_scan(file_scan, file)?;
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
    file: &datafusion::datasource::listing::PartitionedFile,
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
    let mut metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(file.object_meta.size), None);
    if let Some(content_identity) = file.object_meta.e_tag.as_ref().or(file.object_meta.version.as_ref()) {
        metadata = metadata.with_content_identity(content_identity.clone());
    }
    let resolved = resolve_table_format_parquet_source_with_metadata(
        uri.as_str(),
        &TableFormatReadCapability::object_store("s3"),
        &TableFormatParquetReadStrategyConfig::from_env(),
        &metadata,
    )
    .map_err(|_error| CapabilityReason::NativeSourceNonLocalObjectStore)?;
    NativeLocalParquetFile::from_resolved_parquet_source(resolved)
        .map(|native_file| native_file.with_file_size_bytes(file.object_meta.size))
        .map_err(|_error| CapabilityReason::NativeSourceNonLocalObjectStore)
}

fn object_store_file_uri(
    file_scan: &FileScanConfig,
    file: &datafusion::datasource::listing::PartitionedFile,
) -> Result<String, CapabilityReason> {
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
    let Ok(reader) = SerializedFileReader::try_from(native_file.path.as_path()) else {
        return native_file;
    };
    let metadata = reader.metadata();
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
    for column_index in read_column_indexes {
        let Some(null_counts) = row_group_null_counts_for_root_column(metadata, column_index) else {
            continue;
        };
        native_file = native_file.with_row_group_null_counts(column_index, null_counts);
    }

    native_file
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
    files: &[&datafusion::datasource::listing::PartitionedFile],
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
    file: &datafusion::datasource::listing::PartitionedFile,
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

    use nexus_query_engine::parquet_stats::single_leaf_primitive_root_column_index;
    use parquet::schema::{parser::parse_message_type, types::SchemaDescriptor};

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
}
