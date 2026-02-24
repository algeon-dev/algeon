use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use cudf_query_engine::{
    ByteRangeDisposition as NativeByteRangeDisposition, PartitionColumnDisposition as NativePartitionColumnDisposition,
    PredicateDisposition as NativePredicateDisposition, RejectedSourceFeature as NativeRejectedSourceFeature,
    ScanLimitDisposition as NativeScanLimitDisposition, SchemaMappingDisposition as NativeSchemaMappingDisposition,
    SourceDiagnostics as NativeSourceDiagnostics, ZeroColumnCountDisposition as NativeZeroColumnCountDisposition,
};
use datafusion::{
    datasource::physical_plan::{FileScanConfig, ParquetSource},
    physical_plan::filter::FilterExec,
};

use super::{
    NativeParquetColumnRequirements, native_parquet_file_reader_factory_is_supported,
    native_preserve_order_scan_is_supported, partition_column_disposition, required_parquet_column_requirements,
};
use crate::native::{
    normalize::{NormalizedNativeInput, NormalizedNativePlan, NormalizedNativeSource},
    pruning::{
        LocalParquetByteRangeRepresentation, coalesced_local_parquet_scan_files, has_row_level_selection,
        metadata_row_group_selections, parquet_predicate_disposition, selected_row_groups_from_access_plans,
    },
    validate::native_zero_column_count_source_is_safe,
};

/// Collects native source diagnostics for every source in a normalized plan.
///
/// The diagnostics are produced from DataFusion scan metadata but use native IR
/// enums and counters so they can be reported without exposing DataFusion types.
pub(in crate::native) fn source_diagnostics_for_normalized_plan(
    normalized: &NormalizedNativePlan<'_>,
) -> Vec<NativeSourceDiagnostics> {
    let mut diagnostics = Vec::new();
    collect_source_diagnostics_for_input(normalized.root_input(), normalized, &mut diagnostics);
    diagnostics
}

fn collect_source_diagnostics_for_input(
    input: &NormalizedNativeInput<'_>,
    normalized: &NormalizedNativePlan<'_>,
    diagnostics: &mut Vec<NativeSourceDiagnostics>,
) {
    match input {
        NormalizedNativeInput::Source(source) => {
            if let Some(diagnostic) = source_diagnostics_for_source(source, normalized) {
                diagnostics.push(diagnostic);
            }
        },
        NormalizedNativeInput::GpuResidentSource(_) => {},
        NormalizedNativeInput::LiteralSource(_) => {},
        NormalizedNativeInput::Filter(filter) => {
            collect_source_diagnostics_for_input(&filter.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Projection(projection) => {
            collect_source_diagnostics_for_input(&projection.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Limit(limit) => {
            collect_source_diagnostics_for_input(&limit.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            collect_source_diagnostics_for_input(&aggregate.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Sort(sort) => {
            collect_source_diagnostics_for_input(&sort.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Window(window) => {
            collect_source_diagnostics_for_input(&window.input, normalized, diagnostics);
        },
        NormalizedNativeInput::HashJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, normalized, diagnostics);
            collect_source_diagnostics_for_input(&join.right, normalized, diagnostics);
        },
        NormalizedNativeInput::CrossJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, normalized, diagnostics);
            collect_source_diagnostics_for_input(&join.right, normalized, diagnostics);
        },
        NormalizedNativeInput::NestedLoopJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, normalized, diagnostics);
            collect_source_diagnostics_for_input(&join.right, normalized, diagnostics);
        },
        NormalizedNativeInput::Union(union) => {
            for input in &union.inputs {
                collect_source_diagnostics_for_input(input, normalized, diagnostics);
            }
        },
    }
}

fn source_diagnostics_for_source(
    source: &NormalizedNativeSource<'_>,
    normalized: &NormalizedNativePlan<'_>,
) -> Option<NativeSourceDiagnostics> {
    let source_exec = source.local_parquet_source_exec()?;
    let (file_scan, parquet_source) = source_exec.downcast_to_file_source::<ParquetSource>()?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let aggregate_root = normalized.aggregate_root();
    let column_requirements = aggregate_root.and_then(|root| {
        required_parquet_column_requirements(
            file_scan,
            parquet_source,
            source.filter,
            normalized.single_source_input().and(root.pre_aggregate_projection),
            root.aggregate,
        )
        .ok()
    });
    Some(local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        source.filter,
        table_schema,
        column_requirements.as_ref(),
        false,
        aggregate_root.is_some_and(|root| {
            native_zero_column_count_source_is_safe(
                source.filter,
                normalized.single_source_input().and(root.pre_aggregate_projection),
                root.aggregate,
            ) && column_requirements.as_ref().is_some_and(|requirements| requirements.read_columns.is_empty())
        }),
    ))
}

/// Builds native diagnostics for a DataFusion local Parquet scan.
///
/// This mirrors the validation path but records dispositions and rejection
/// feature codes instead of returning the first fallback reason.
pub(in crate::native) fn local_parquet_source_diagnostics(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: Option<&NativeParquetColumnRequirements>,
    reader_pushdown: bool,
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
        let coalesced_files = coalesced_files.as_ref().expect("coalesced file result already checked");
        if coalesced_files
            .iter()
            .any(|file| file.byte_range_representation == LocalParquetByteRangeRepresentation::RowGroupSelection)
        {
            NativeByteRangeDisposition::RowGroupSelection
        } else {
            NativeByteRangeDisposition::CoalescedFullFiles
        }
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
    let schema_mapping_disposition =
        column_requirements.map_or(NativeSchemaMappingDisposition::DirectByName, |requirements| {
            if !requirements.missing_columns.is_empty() {
                NativeSchemaMappingDisposition::MissingNullableColumns
            } else {
                NativeSchemaMappingDisposition::DirectByName
            }
        });

    let mut rejected_features = Vec::new();
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        rejected_features.push(NativeRejectedSourceFeature::NonLocalObjectStore);
    }
    let partition_column_indices =
        column_requirements.map_or(&[][..], |requirements| requirements.partition_column_indices.as_slice());
    let partition_column_disposition = partition_column_disposition(file_scan, &files, partition_column_indices);
    if partition_column_disposition == NativePartitionColumnDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::PartitionColumns);
    }
    if file_scan.expr_adapter_factory.is_some() {
        rejected_features.push(NativeRejectedSourceFeature::ExprAdapter);
    }
    if !native_parquet_file_reader_factory_is_supported(parquet_source) {
        rejected_features.push(NativeRejectedSourceFeature::ParquetFileReaderFactory);
    }
    let scan_limit_disposition = scan_limit_disposition(file_scan, &files);
    if scan_limit_disposition == NativeScanLimitDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::ScanLimit);
    }
    if !native_preserve_order_scan_is_supported(file_scan) {
        rejected_features.push(NativeRejectedSourceFeature::PreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::OutputOrdering);
    }
    let predicate_disposition = parquet_predicate_disposition(parquet_source, filter, &source_schema, reader_pushdown);
    if predicate_disposition == NativePredicateDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::PushedDownParquetPredicate);
    }
    if files.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::EmptySource);
    }
    if zero_column_count_disposition == NativeZeroColumnCountDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::ZeroColumnCount);
    }
    if matches!(schema_mapping_disposition, NativeSchemaMappingDisposition::Rejected) {
        rejected_features.push(NativeRejectedSourceFeature::SchemaMapping);
    }
    let ranges_are_safely_coalesced = ranged_file_count > 0
        && file_scan.object_store_url.as_str().starts_with("file://")
        && partition_column_disposition != NativePartitionColumnDisposition::Rejected
        && file_scan.expr_adapter_factory.is_none()
        && scan_limit_disposition != NativeScanLimitDisposition::Rejected
        && native_preserve_order_scan_is_supported(file_scan)
        && file_scan.output_ordering.is_empty()
        && coalesced_files.is_ok();
    for file in &files {
        if partition_column_disposition == NativePartitionColumnDisposition::Rejected
            && !file.partition_values.is_empty()
        {
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
        selected_row_group_stats: None,
        predicate_disposition,
        byte_range_disposition,
        partition_column_disposition,
        zero_column_count_disposition,
        scan_limit_disposition,
        schema_mapping_disposition,
        read_column_count,
        projected_output_column_count,
        pure_filter_column_count,
        rejected_features,
    }
}

fn scan_limit_disposition(
    file_scan: &FileScanConfig,
    files: &[&datafusion::datasource::listing::PartitionedFile],
) -> NativeScanLimitDisposition {
    if file_scan.limit.is_none() {
        return NativeScanLimitDisposition::Absent;
    }
    if !native_preserve_order_scan_is_supported(file_scan)
        || !file_scan.output_ordering.is_empty()
        || files.iter().any(|file| file.ordering.is_some())
    {
        NativeScanLimitDisposition::Rejected
    } else {
        NativeScanLimitDisposition::Represented
    }
}
