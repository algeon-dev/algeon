use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::physical_plan::{FileScanConfig, ParquetSource, parquet::ParquetAccessPlan},
    physical_plan::filter::FilterExec,
};
use query_planning::source::{
    ByteRangeDisposition as NativeByteRangeDisposition, PartitionColumnDisposition as NativePartitionColumnDisposition,
    PredicateDisposition as NativePredicateDisposition, RejectedSourceFeature as NativeRejectedSourceFeature,
    ScanLimitDisposition as NativeScanLimitDisposition, SchemaMappingDisposition as NativeSchemaMappingDisposition,
    SourceDiagnostics as NativeSourceDiagnostics, ZeroColumnCountDisposition as NativeZeroColumnCountDisposition,
};

use super::{
    NativeParquetColumnRequirements, native_declared_output_ordering_scan_is_supported,
    native_parquet_file_reader_factory_is_supported, native_parquet_object_store_scan_is_supported,
    native_preserve_order_scan_is_supported, partition_column_disposition,
};
use crate::native::{
    footers::parquet_footer,
    lowering::{aggregate_input_demand, parquet_column_requirements, parquet_reads_row_count_only},
    normalize::{NormalizedNativeAggregate, NormalizedNativeInput, NormalizedNativePlan, NormalizedNativeSource},
    pruning::{
        CoalescedLocalParquetFile, LocalParquetByteRangeRepresentation, coalesced_local_parquet_scan_files,
        explicit_row_group_selection, has_row_level_selection, local_object_path_to_filesystem,
        metadata_row_group_selections, parquet_predicate_disposition, selected_row_groups_from_access_plans,
    },
};

/// Collects native source diagnostics for every source in a normalized plan.
///
/// The diagnostics are produced from DataFusion scan metadata but use native IR
/// enums and counters so they can be reported without exposing DataFusion types.
pub(in crate::native) fn source_diagnostics_for_normalized_plan(
    normalized: &NormalizedNativePlan<'_>,
) -> Vec<NativeSourceDiagnostics> {
    let mut diagnostics = Vec::new();
    collect_source_diagnostics_for_input(&normalized.input, None, &mut diagnostics);
    diagnostics
}

/// `aggregate` is the aggregate directly above `input`, whose demand decides
/// which columns a source below it reads.
fn collect_source_diagnostics_for_input(
    input: &NormalizedNativeInput<'_>,
    aggregate: Option<&NormalizedNativeAggregate<'_>>,
    diagnostics: &mut Vec<NativeSourceDiagnostics>,
) {
    match input {
        NormalizedNativeInput::Source(source) => {
            if let Some(diagnostic) = source_diagnostics_for_source(source, aggregate) {
                diagnostics.push(diagnostic);
            }
        },
        NormalizedNativeInput::GpuResidentSource(_) => {},
        NormalizedNativeInput::LiteralSource(_) => {},
        NormalizedNativeInput::Filter(filter) => {
            collect_source_diagnostics_for_input(&filter.input, None, diagnostics);
        },
        NormalizedNativeInput::Projection(projection) => {
            collect_source_diagnostics_for_input(&projection.input, None, diagnostics);
        },
        NormalizedNativeInput::Limit(limit) => {
            collect_source_diagnostics_for_input(&limit.input, None, diagnostics);
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            collect_source_diagnostics_for_input(&aggregate.input, Some(aggregate), diagnostics);
        },
        NormalizedNativeInput::Sort(sort) => {
            collect_source_diagnostics_for_input(&sort.input, None, diagnostics);
        },
        NormalizedNativeInput::Window(window) => {
            collect_source_diagnostics_for_input(&window.input, None, diagnostics);
        },
        NormalizedNativeInput::HashJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, None, diagnostics);
            collect_source_diagnostics_for_input(&join.right, None, diagnostics);
        },
        NormalizedNativeInput::CrossJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, None, diagnostics);
            collect_source_diagnostics_for_input(&join.right, None, diagnostics);
        },
        NormalizedNativeInput::NestedLoopJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, None, diagnostics);
            collect_source_diagnostics_for_input(&join.right, None, diagnostics);
        },
        NormalizedNativeInput::Union(union) => {
            for input in &union.inputs {
                collect_source_diagnostics_for_input(input, None, diagnostics);
            }
        },
    }
}

fn source_diagnostics_for_source(
    source: &NormalizedNativeSource<'_>,
    aggregate: Option<&NormalizedNativeAggregate<'_>>,
) -> Option<NativeSourceDiagnostics> {
    let source_exec = source.scan;
    let (file_scan, parquet_source) = source_exec.downcast_to_file_source::<ParquetSource>()?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let demand = aggregate.and_then(|aggregate| aggregate_input_demand(aggregate).ok());
    let column_requirements =
        demand.as_ref().and_then(|demand| parquet_column_requirements(source_exec, source.filter, demand).ok());
    Some(local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        source.filter,
        table_schema,
        column_requirements.as_ref(),
        SourceRead {
            reader_pushdown: false,
            row_count_only: demand
                .as_ref()
                .is_some_and(|demand| parquet_reads_row_count_only(parquet_source, source.filter, demand)),
            // Outside an aggregate this walk does not know what the parent reads.
            ordered: demand.as_ref().is_none_or(|demand| demand.ordered),
        },
    ))
}

/// How lowering reads a scan, which decides the dispositions its diagnostics
/// report.
#[derive(Debug, Clone, Copy)]
pub(in crate::native) struct SourceRead {
    /// The source evaluates a reader predicate.
    pub(in crate::native) reader_pushdown: bool,
    /// The source counts rows without reading a column.
    pub(in crate::native) row_count_only: bool,
    /// The scan's parent reads its rows in their declared order.
    pub(in crate::native) ordered: bool,
}

/// Builds native diagnostics for a DataFusion local Parquet scan.
///
/// This mirrors the validation path and records dispositions plus every rejection
/// feature code. Validation returns only the first fallback reason.
pub(in crate::native) fn local_parquet_source_diagnostics(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: Option<&NativeParquetColumnRequirements>,
    read: SourceRead,
) -> NativeSourceDiagnostics {
    let SourceRead { reader_pushdown, row_count_only: zero_column_count_safe, ordered } = read;
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
    } else if let Ok(coalesced_files) = coalesced_files.as_ref() {
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
    let coalesced_selected_row_groups =
        coalesced_files.as_ref().ok().and_then(|files| selected_row_groups_from_coalesced_files(files));
    let selected_row_groups =
        coalesced_selected_row_groups.or_else(|| selected_row_groups_from_access_plans(&files)).or_else(|| {
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
    if !native_parquet_object_store_scan_is_supported(file_scan) {
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
    let scan_limit_disposition = scan_limit_disposition(file_scan, ordered);
    if scan_limit_disposition == NativeScanLimitDisposition::Rejected {
        rejected_features.push(NativeRejectedSourceFeature::ScanLimit);
    }
    if !native_preserve_order_scan_is_supported(file_scan, ordered) {
        rejected_features.push(NativeRejectedSourceFeature::PreserveOrder);
    }
    if !native_declared_output_ordering_scan_is_supported(file_scan, ordered) {
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
        && native_preserve_order_scan_is_supported(file_scan, ordered)
        && native_declared_output_ordering_scan_is_supported(file_scan, ordered)
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
        if has_row_level_selection(file) && !ranges_are_safely_coalesced {
            rejected_features.push(NativeRejectedSourceFeature::RowSelection);
        }
    }
    rejected_features.sort_by_key(|feature| feature.as_str());
    rejected_features.dedup();

    NativeSourceDiagnostics {
        source_read_strategy: "local_kvikio".to_string(),
        credential_mode: "none".to_string(),
        lakehouse_cache_status: "disabled".to_string(),
        lakehouse_cache_read_mode: "disabled".to_string(),
        lakehouse_cache_fill_mode: "disabled".to_string(),
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

fn scan_limit_disposition(file_scan: &FileScanConfig, ordered: bool) -> NativeScanLimitDisposition {
    if file_scan.limit.is_none() {
        return NativeScanLimitDisposition::Absent;
    }
    if !native_preserve_order_scan_is_supported(file_scan, ordered)
        || !native_declared_output_ordering_scan_is_supported(file_scan, ordered)
    {
        NativeScanLimitDisposition::Rejected
    } else {
        NativeScanLimitDisposition::Represented
    }
}

fn selected_row_groups_from_coalesced_files(files: &[CoalescedLocalParquetFile<'_>]) -> Option<usize> {
    let has_row_group_selection = files
        .iter()
        .any(|file| file.byte_range_row_groups.is_some() || file.file.extensions.get::<ParquetAccessPlan>().is_some());
    has_row_group_selection.then(|| {
        files.iter().try_fold(0usize, |total, file| {
            selected_row_group_count_for_coalesced_file(file).map(|count| total.saturating_add(count))
        })
    })?
}

fn selected_row_group_count_for_coalesced_file(file: &CoalescedLocalParquetFile<'_>) -> Option<usize> {
    if let Some(row_groups) = &file.byte_range_row_groups {
        return Some(row_groups.len());
    }
    if let Some(row_groups) = explicit_row_group_selection(file.file).ok()? {
        return Some(row_groups.len());
    }
    let path = local_object_path_to_filesystem(file.file.object_meta.location.as_ref());
    Some(parquet_footer(path.as_path()).ok()?.num_row_groups())
}
