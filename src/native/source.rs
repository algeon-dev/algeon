use std::collections::BTreeSet;

use arrow::datatypes::SchemaRef;
use cudf_query_engine::{
    ByteRangeDisposition as NativeByteRangeDisposition, LocalParquetFile as NativeLocalParquetFile,
    PartitionColumnDisposition as NativePartitionColumnDisposition, PredicateDisposition as NativePredicateDisposition,
    RejectedSourceFeature as NativeRejectedSourceFeature, SchemaMappingDisposition as NativeSchemaMappingDisposition,
    SourceDiagnostics as NativeSourceDiagnostics, ZeroColumnCountDisposition as NativeZeroColumnCountDisposition,
};
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    physical_plan::{ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, projection::ProjectionExec},
};

use super::{
    normalize::NormalizedNativePlan,
    pruning::{
        coalesced_local_parquet_scan_files, explicit_row_group_selection, has_row_level_selection,
        local_object_path_to_filesystem, metadata_row_group_selections, parquet_predicate_disposition,
        selected_row_groups_from_access_plans, validate_pruning_only_parquet_predicate,
    },
    validate::{collect_required_columns, native_zero_column_count_source_is_safe},
};
use crate::policy::FallbackReason;

pub(super) fn source_diagnostics_for_normalized_plan(
    normalized: &NormalizedNativePlan<'_>,
) -> Vec<NativeSourceDiagnostics> {
    let Some((file_scan, parquet_source)) = normalized.source_exec.downcast_to_file_source::<ParquetSource>() else {
        return Vec::new();
    };
    let column_requirements = required_parquet_column_requirements(
        file_scan,
        normalized.filter,
        normalized.pre_aggregate_projection,
        normalized.aggregate,
    )
    .ok();
    vec![local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        normalized.filter,
        normalized.source_exec.schema(),
        column_requirements.as_ref(),
        native_zero_column_count_source_is_safe(
            normalized.filter,
            normalized.pre_aggregate_projection,
            normalized.aggregate,
        ),
    )]
}

pub(super) fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
) -> Result<(), FallbackReason> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(FallbackReason::NativeSourceNonLocalObjectStore);
    }
    if !file_scan.table_partition_cols().is_empty() {
        return Err(FallbackReason::NativeSourcePartitionColumns);
    }
    if file_scan.expr_adapter_factory.is_some() {
        return Err(FallbackReason::NativeSourceExprAdapter);
    }
    if file_scan.limit.is_some() {
        return Err(FallbackReason::NativeSourceScanLimit);
    }
    if file_scan.preserve_order {
        return Err(FallbackReason::NativeSourcePreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        return Err(FallbackReason::NativeSourceOutputOrdering);
    }
    if parquet_source.filter().is_some() {
        validate_pruning_only_parquet_predicate(parquet_source, filter, &source_schema)?;
    }
    local_parquet_files(file_scan, parquet_source, filter, source_schema)?;
    Ok(())
}

pub(super) fn local_parquet_source_diagnostics(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: Option<&NativeParquetColumnRequirements>,
    zero_column_count_safe: bool,
) -> NativeSourceDiagnostics {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    let coalesced_files = coalesced_local_parquet_scan_files(&files);
    let diagnostic_files = coalesced_files
        .as_ref()
        .map_or_else(|_| files.clone(), |coalesced| coalesced.iter().map(|file| file.file).collect::<Vec<_>>());
    let known_bytes = diagnostic_files
        .iter()
        .map(|file| file.object_meta.size)
        .try_fold(0u64, |total, bytes| Some(total.saturating_add(bytes)));
    let ranged_file_count = files.iter().filter(|file| file.range.is_some()).count();
    let byte_range_disposition = if ranged_file_count == 0 {
        NativeByteRangeDisposition::FullFiles
    } else if coalesced_files.is_ok() {
        NativeByteRangeDisposition::CoalescedFullFiles
    } else {
        NativeByteRangeDisposition::Rejected
    };
    let selected_row_groups = selected_row_groups_from_access_plans(&files).or_else(|| {
        metadata_row_group_selections(&files, file_scan, parquet_source, filter, &source_schema)
            .map(|selections| selections.iter().map(Vec::len).sum())
    });
    let read_column_count = column_requirements.map_or(0, |requirements| requirements.read_columns.len());
    let projected_output_column_count =
        column_requirements.map_or(0, |requirements| requirements.projected_output_columns.len());
    let pure_filter_column_count = column_requirements.map_or(0, |requirements| requirements.pure_filter_columns.len());
    let zero_column_count_disposition =
        column_requirements.map_or(NativeZeroColumnCountDisposition::NotRequired, |requirements| {
            if requirements.read_columns.is_empty() {
                if zero_column_count_safe {
                    NativeZeroColumnCountDisposition::Represented
                } else {
                    NativeZeroColumnCountDisposition::Rejected
                }
            } else {
                NativeZeroColumnCountDisposition::NotRequired
            }
        });

    let mut rejected_features = Vec::new();
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        rejected_features.push(NativeRejectedSourceFeature::NonLocalObjectStore);
    }
    if !file_scan.table_partition_cols().is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::PartitionColumns);
    }
    if file_scan.expr_adapter_factory.is_some() {
        rejected_features.push(NativeRejectedSourceFeature::ExprAdapter);
    }
    if file_scan.limit.is_some() {
        rejected_features.push(NativeRejectedSourceFeature::ScanLimit);
    }
    if file_scan.preserve_order {
        rejected_features.push(NativeRejectedSourceFeature::PreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::OutputOrdering);
    }
    let predicate_disposition = parquet_predicate_disposition(parquet_source, filter, &source_schema);
    if predicate_disposition == NativePredicateDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::PushedDownParquetPredicate);
    }
    if files.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::EmptySource);
    }
    if zero_column_count_disposition == NativeZeroColumnCountDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::ZeroColumnCount);
    }
    let ranges_are_safely_coalesced = ranged_file_count > 0
        && file_scan.object_store_url.as_str().starts_with("file://")
        && file_scan.table_partition_cols().is_empty()
        && file_scan.expr_adapter_factory.is_none()
        && file_scan.limit.is_none()
        && !file_scan.preserve_order
        && file_scan.output_ordering.is_empty()
        && coalesced_files.is_ok();
    for file in &files {
        if !file.partition_values.is_empty() {
            rejected_features.push(NativeRejectedSourceFeature::PartitionValues);
        }
        if file.range.is_some() && !ranges_are_safely_coalesced {
            rejected_features.push(NativeRejectedSourceFeature::ByteRange);
        }
        if file.ordering.is_some() {
            rejected_features.push(NativeRejectedSourceFeature::FileOrdering);
        }
        if has_row_level_selection(file) {
            rejected_features.push(NativeRejectedSourceFeature::RowSelection);
        }
    }
    rejected_features.sort_by_key(|feature| feature.as_str());
    rejected_features.dedup();

    NativeSourceDiagnostics {
        format: cudf_query_engine::SourceFormat::Parquet,
        direct_device_read: rejected_features.is_empty(),
        file_count: diagnostic_files.len(),
        ranged_file_count,
        known_bytes,
        selected_row_groups,
        predicate_disposition,
        byte_range_disposition,
        partition_column_disposition: if !file_scan.table_partition_cols().is_empty()
            || files.iter().any(|file| !file.partition_values.is_empty())
        {
            NativePartitionColumnDisposition::Rejected
        } else {
            NativePartitionColumnDisposition::Absent
        },
        zero_column_count_disposition,
        schema_mapping_disposition: NativeSchemaMappingDisposition::DirectByName,
        read_column_count,
        projected_output_column_count,
        pure_filter_column_count,
        rejected_features,
    }
}

pub(super) fn local_parquet_files(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    let scan_files = coalesced_local_parquet_scan_files(&files)?;
    let metadata_row_group_selections =
        metadata_row_group_selections(&files, file_scan, parquet_source, filter, &source_schema);
    let mut native_files = Vec::with_capacity(scan_files.len());
    for (file_index, scan_file) in scan_files.iter().enumerate() {
        let file = scan_file.file;
        if !file.partition_values.is_empty() {
            return Err(FallbackReason::NativeSourcePartitionValues);
        }
        if file.ordering.is_some() {
            return Err(FallbackReason::NativeSourceFileOrdering);
        }
        let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
        let stable_id = format!("local:{}:{}", path.display(), file.object_meta.size);
        let mut native_file =
            NativeLocalParquetFile::new(path).with_stable_id(stable_id).with_file_size_bytes(file.object_meta.size);
        if let Some(row_groups) = explicit_row_group_selection(file)? {
            native_file = native_file.with_row_groups(row_groups);
        } else if let Some(row_groups) =
            metadata_row_group_selections.as_ref().and_then(|selections| selections.get(file_index))
        {
            native_file = native_file.with_row_groups(row_groups.clone());
        }
        native_files.push(native_file);
    }
    Ok(native_files)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeParquetColumnRequirements {
    pub(super) read_columns: Vec<String>,
    pub(super) read_column_file_indices: Vec<usize>,
    pub(super) projected_output_columns: Vec<String>,
    pub(super) projected_output_column_file_indices: Vec<usize>,
    pub(super) pure_filter_columns: Vec<String>,
    pub(super) pure_filter_column_file_indices: Vec<usize>,
}

pub(super) fn required_parquet_column_requirements(
    file_scan: &FileScanConfig,
    filter: Option<&FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = file_scan.file_schema();
    if native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate) {
        return Ok(NativeParquetColumnRequirements {
            read_columns: Vec::new(),
            read_column_file_indices: Vec::new(),
            projected_output_columns: Vec::new(),
            projected_output_column_file_indices: Vec::new(),
            pure_filter_columns: Vec::new(),
            pure_filter_column_file_indices: Vec::new(),
        });
    }

    let mut scan_projected_columns = BTreeSet::new();
    let mut filter_columns = BTreeSet::new();
    let mut output_columns = BTreeSet::new();

    if let Some(projection) = file_scan.file_source().projection() {
        for index in projection.column_indices() {
            if index >= file_schema.fields().len() {
                return Err(FallbackReason::NativeSourcePartitionColumns);
            }
            scan_projected_columns.insert(file_schema.field(index).name().clone());
        }
    }

    if let Some(filter) = filter {
        collect_required_columns(filter.predicate(), &mut filter_columns)?;
    }
    if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            collect_required_columns(&projection_expr.expr, &mut output_columns)?;
        }
    } else {
        for aggregate_expr in aggregate.aggr_expr() {
            for expr in aggregate_expr.expressions() {
                collect_required_columns(&expr, &mut output_columns)?;
            }
        }
    }
    for (group_expr, _alias) in aggregate.group_expr().expr() {
        collect_required_columns(group_expr, &mut output_columns)?;
    }

    let mut read_columns = scan_projected_columns;
    read_columns.extend(filter_columns.iter().cloned());
    read_columns.extend(output_columns.iter().cloned());

    for column in &read_columns {
        if file_schema.index_of(column).is_err() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }

    let pure_filter_columns = filter_columns.difference(&output_columns).cloned().collect::<Vec<_>>();
    let read_columns = read_columns.into_iter().collect::<Vec<_>>();
    let projected_output_columns = output_columns.into_iter().collect::<Vec<_>>();
    let read_column_file_indices = column_file_indices(&file_schema, &read_columns)?;
    let projected_output_column_file_indices = column_file_indices(&file_schema, &projected_output_columns)?;
    let pure_filter_column_file_indices = column_file_indices(&file_schema, &pure_filter_columns)?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_file_indices,
        pure_filter_columns,
        pure_filter_column_file_indices,
    })
}

fn column_file_indices(file_schema: &SchemaRef, columns: &[String]) -> Result<Vec<usize>, FallbackReason> {
    columns
        .iter()
        .map(|column| file_schema.index_of(column).map_err(|_error| FallbackReason::UnsupportedExpression))
        .collect()
}
