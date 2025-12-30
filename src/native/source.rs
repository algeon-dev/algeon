use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use cudf_query_engine::{
    ByteRangeDisposition as NativeByteRangeDisposition, LiteralValue as NativeLiteralValue,
    LocalParquetFile as NativeLocalParquetFile, PartitionColumn as NativePartitionColumn,
    PartitionColumnDisposition as NativePartitionColumnDisposition, PredicateDisposition as NativePredicateDisposition,
    RejectedSourceFeature as NativeRejectedSourceFeature, SchemaMappingDisposition as NativeSchemaMappingDisposition,
    SourceColumnRef as NativeSourceColumnRef, SourceDiagnostics as NativeSourceDiagnostics,
    ZeroColumnCountDisposition as NativeZeroColumnCountDisposition,
};
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::{aggregates::AggregateExec, filter::FilterExec, projection::ProjectionExec},
};

use super::{
    literal::{NativeLiteralContext, native_literal_value},
    normalize::{NormalizedNativeInput, NormalizedNativePlan, NormalizedNativeSource},
    pruning::{
        coalesced_local_parquet_scan_files, explicit_row_group_selection, has_row_level_selection,
        local_object_path_to_filesystem, metadata_row_group_selections, parquet_predicate_disposition,
        selected_row_groups_from_access_plans, validate_pruning_only_parquet_predicate,
    },
    validate::{collect_required_columns, is_supported_native_type, native_zero_column_count_source_is_safe},
};
use crate::policy::FallbackReason;

/// Collects native source diagnostics for every source in a normalized plan.
///
/// The diagnostics are produced from DataFusion scan metadata but use native IR
/// enums and counters so they can be reported without exposing DataFusion types.
pub(super) fn source_diagnostics_for_normalized_plan(
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
        NormalizedNativeInput::Filter(filter) => {
            collect_source_diagnostics_for_input(&filter.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Projection(projection) => {
            collect_source_diagnostics_for_input(&projection.input, normalized, diagnostics);
        },
        NormalizedNativeInput::Aggregate(aggregate) => {
            collect_source_diagnostics_for_input(&aggregate.input, normalized, diagnostics);
        },
        NormalizedNativeInput::HashJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, normalized, diagnostics);
            collect_source_diagnostics_for_input(&join.right, normalized, diagnostics);
        },
        NormalizedNativeInput::NestedLoopJoin(join) => {
            collect_source_diagnostics_for_input(&join.left, normalized, diagnostics);
            collect_source_diagnostics_for_input(&join.right, normalized, diagnostics);
        },
    }
}

fn source_diagnostics_for_source(
    source: &NormalizedNativeSource<'_>,
    normalized: &NormalizedNativePlan<'_>,
) -> Option<NativeSourceDiagnostics> {
    let (file_scan, parquet_source) = source.source_exec.downcast_to_file_source::<ParquetSource>()?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let aggregate_root = normalized.aggregate_root();
    let column_requirements = aggregate_root.and_then(|root| {
        required_parquet_column_requirements(
            file_scan,
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
        aggregate_root.is_some_and(|root| {
            native_zero_column_count_source_is_safe(
                source.filter,
                normalized.single_source_input().and(root.pre_aggregate_projection),
                root.aggregate,
            )
        }),
    ))
}

pub(super) fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
) -> Result<(), FallbackReason> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(FallbackReason::NativeSourceNonLocalObjectStore);
    }
    validate_partition_columns(file_scan, &column_requirements.partition_column_indices)?;
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
    local_parquet_files(file_scan, parquet_source, filter, source_schema, column_requirements)?;
    Ok(())
}

/// Builds native diagnostics for a DataFusion local Parquet scan.
///
/// This mirrors the validation path but records dispositions and rejection
/// feature codes instead of returning the first fallback reason.
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
    let partition_column_indices =
        column_requirements.map_or(&[][..], |requirements| requirements.partition_column_indices.as_slice());
    let partition_column_disposition = partition_column_disposition(file_scan, &files, partition_column_indices);
    if partition_column_disposition == NativePartitionColumnDisposition::Rejected {
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
        && partition_column_disposition != NativePartitionColumnDisposition::Rejected
        && file_scan.expr_adapter_factory.is_none()
        && file_scan.limit.is_none()
        && !file_scan.preserve_order
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
        predicate_disposition,
        byte_range_disposition,
        partition_column_disposition,
        zero_column_count_disposition,
        schema_mapping_disposition: NativeSchemaMappingDisposition::DirectByName,
        read_column_count,
        projected_output_column_count,
        pure_filter_column_count,
        rejected_features,
    }
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
        if file.ordering.is_some() {
            return Err(FallbackReason::NativeSourceFileOrdering);
        }
        let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
        let stable_id = format!("local:{}:{}", path.display(), file.object_meta.size);
        let mut native_file =
            NativeLocalParquetFile::new(path).with_stable_id(stable_id).with_file_size_bytes(file.object_meta.size);
        if !column_requirements.partition_column_indices.is_empty() {
            native_file = native_file
                .with_partition_values(native_partition_values(file, &column_requirements.partition_column_indices)?);
        }
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

/// Column-role bridge from a DataFusion Parquet scan to native source metadata.
///
/// DataFusion exposes columns through multiple coordinate systems: table schema,
/// file schema, pushed scan projection, filter projection, and partition values.
/// The native engine needs a frontend-neutral description instead: which
/// physical file columns the source must read and emit, which partition columns
/// it must synthesize, which emitted source columns are expected to remain after
/// source-adjacent native filtering/projection, and which emitted columns exist
/// only to evaluate filters. This struct keeps those roles separate so lowering
/// can remove filter-only columns after filters while still emitting them from
/// the source when needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeParquetColumnRequirements {
    /// Physical Parquet columns that the native source must read and emit.
    ///
    /// This excludes partition columns because they are supplied from
    /// DataFusion `PartitionedFile` metadata, not from file pages. Columns used
    /// only by native filters are included so those filters can consume them.
    pub(super) read_columns: Vec<String>,
    /// File-schema indices for `read_columns`.
    pub(super) read_column_file_indices: Vec<usize>,
    /// Source columns expected to remain after source-adjacent native projection.
    ///
    /// These may include file columns and synthetic partition columns. This is
    /// role/diagnostic metadata and does not define the source node's emitted
    /// schema; `native_source_output_schema` emits all `read_columns` plus all
    /// represented partition columns.
    pub(super) projected_output_columns: Vec<String>,
    /// File or partition source references for `projected_output_columns`.
    pub(super) projected_output_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Source columns emitted only so source-adjacent predicates can evaluate.
    ///
    /// These should be removed by a later native projection when they are not
    /// needed downstream.
    pub(super) pure_filter_columns: Vec<String>,
    /// File or partition source references for `pure_filter_columns`.
    pub(super) pure_filter_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Required partition columns represented as synthetic native source columns.
    pub(super) partition_columns: Vec<NativePartitionColumn>,
    /// DataFusion table partition-column indices required by this source.
    pub(super) partition_column_indices: Vec<usize>,
    /// DataFusion scan-output column names consumed by the lowered subtree.
    ///
    /// This preserves the adapter-side view of the scan projection so later
    /// lowering can connect DataFusion expressions to native source columns.
    pub(super) scan_output_columns: Vec<String>,
}

/// Computes source column roles for aggregate-root native lowering.
///
/// Required/projected roles are derived from the aggregate, optional
/// pre-aggregate projection, and optional source filter. Filter-only columns are
/// read and emitted by the native source so filters can evaluate them, then
/// omitted from the columns expected to survive the source-adjacent native
/// projection. When the aggregate can be answered as a zero-column count, no
/// file or partition columns are required.
pub(super) fn required_parquet_column_requirements(
    file_scan: &FileScanConfig,
    filter: Option<&FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    if native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate) {
        return Ok(NativeParquetColumnRequirements {
            read_columns: Vec::new(),
            read_column_file_indices: Vec::new(),
            projected_output_columns: Vec::new(),
            projected_output_column_source_refs: Vec::new(),
            pure_filter_columns: Vec::new(),
            pure_filter_column_source_refs: Vec::new(),
            partition_columns: Vec::new(),
            partition_column_indices: Vec::new(),
            scan_output_columns: Vec::new(),
        });
    }

    let mut filter_columns: BTreeSet<String> = BTreeSet::new();
    let mut output_columns: BTreeSet<String> = BTreeSet::new();
    let mut scan_output_columns: BTreeSet<String> = BTreeSet::new();

    let scan_projection = file_scan.file_source().projection();
    let scan_output_schema = file_scan.projected_schema().map_err(|_error| FallbackReason::UnsupportedExpression)?;

    if let Some(filter) = filter {
        collect_source_required_columns(filter.predicate(), scan_projection, &table_schema, &mut filter_columns)?;
        collect_required_columns(filter.predicate(), &scan_output_schema, &mut scan_output_columns)?;
    }
    if let Some(filter_projection) = filter.and_then(|filter| filter.projection().as_ref()) {
        collect_source_projection_indices(
            filter_projection,
            scan_projection,
            &scan_output_schema,
            &table_schema,
            &mut output_columns,
        )?;
        collect_scan_projection_indices(filter_projection, &scan_output_schema, &mut scan_output_columns)?;
    } else if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            collect_source_required_columns(
                &projection_expr.expr,
                scan_projection,
                &table_schema,
                &mut output_columns,
            )?;
            collect_required_columns(&projection_expr.expr, &projection.input().schema(), &mut scan_output_columns)?;
        }
    } else {
        for aggregate_expr in aggregate.aggr_expr() {
            for expr in aggregate_expr.expressions() {
                collect_source_required_columns(&expr, scan_projection, &table_schema, &mut output_columns)?;
                collect_required_columns(&expr, &aggregate.input().schema(), &mut scan_output_columns)?;
            }
        }
        for (group_expr, _alias) in aggregate.group_expr().expr() {
            collect_source_required_columns(group_expr, scan_projection, &table_schema, &mut output_columns)?;
            collect_required_columns(group_expr, &aggregate.input().schema(), &mut scan_output_columns)?;
        }
    }

    let mut read_columns: BTreeSet<String> = BTreeSet::new();
    read_columns.extend(filter_columns.iter().cloned());
    read_columns.extend(output_columns.iter().cloned());

    let read_columns =
        read_columns.into_iter().filter(|column| file_schema.index_of(column).is_ok()).collect::<BTreeSet<_>>();

    let pure_filter_columns = filter_columns.difference(&output_columns).cloned().collect::<Vec<_>>();
    let read_columns = read_columns.into_iter().collect::<Vec<_>>();
    let required_source_columns = filter_columns.union(&output_columns).cloned().collect::<BTreeSet<_>>();
    let projected_output_columns = output_columns.into_iter().collect::<Vec<_>>();
    let (partition_columns, partition_column_indices) =
        native_required_partition_columns(file_scan, &required_source_columns)?;
    let read_column_file_indices = column_file_indices(file_schema, &read_columns)?;
    let projected_output_column_source_refs =
        column_source_refs(file_schema, &partition_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs = column_source_refs(file_schema, &partition_columns, &pure_filter_columns)?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        partition_columns,
        partition_column_indices,
        scan_output_columns: scan_output_columns.into_iter().collect(),
    })
}

/// Computes source column roles when lowering starts from required scan-output columns.
///
/// This path is used by native shapes that already know which DataFusion scan
/// output columns are needed. The function maps those names back through any
/// scan projection to physical file columns or synthetic partition columns.
/// File columns used only by source-adjacent filters remain part of the emitted
/// source schema and are tracked as `pure_filter_columns` so later native
/// projection can remove them.
pub(super) fn required_parquet_column_requirements_for_scan_columns(
    file_scan: &FileScanConfig,
    filter: Option<&FilterExec>,
    required_scan_output_indices: &[usize],
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let scan_output_schema = file_scan.projected_schema().map_err(|_error| FallbackReason::UnsupportedExpression)?;

    let mut filter_columns: BTreeSet<String> = BTreeSet::new();
    let mut output_columns: BTreeSet<String> = BTreeSet::new();
    let mut scan_output_columns: BTreeSet<String> = BTreeSet::new();

    if let Some(filter) = filter {
        collect_source_required_columns(filter.predicate(), scan_projection, &table_schema, &mut filter_columns)?;
        collect_required_columns(filter.predicate(), &scan_output_schema, &mut scan_output_columns)?;
    }

    for index in required_scan_output_indices {
        let field = scan_output_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(field.name(), *index));
        collect_source_required_columns(&expr, scan_projection, &table_schema, &mut output_columns)?;
        scan_output_columns.insert(field.name().clone());
    }

    let mut read_columns: BTreeSet<String> = BTreeSet::new();
    read_columns.extend(filter_columns.iter().cloned());
    read_columns.extend(output_columns.iter().cloned());

    let read_columns =
        read_columns.into_iter().filter(|column| file_schema.index_of(column).is_ok()).collect::<BTreeSet<_>>();

    let pure_filter_columns = filter_columns.difference(&output_columns).cloned().collect::<Vec<_>>();
    let read_columns = read_columns.into_iter().collect::<Vec<_>>();
    let required_source_columns = filter_columns.union(&output_columns).cloned().collect::<BTreeSet<_>>();
    let projected_output_columns = output_columns.into_iter().collect::<Vec<_>>();
    let (partition_columns, partition_column_indices) =
        native_required_partition_columns(file_scan, &required_source_columns)?;
    let read_column_file_indices = column_file_indices(file_schema, &read_columns)?;
    let projected_output_column_source_refs =
        column_source_refs(file_schema, &partition_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs = column_source_refs(file_schema, &partition_columns, &pure_filter_columns)?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        partition_columns,
        partition_column_indices,
        scan_output_columns: scan_output_columns.into_iter().collect(),
    })
}

fn collect_scan_projection_indices(
    projection: &[usize],
    scan_output_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    for index in projection {
        let field = scan_output_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
        required.insert(field.name().clone());
    }
    Ok(())
}

fn collect_source_projection_indices(
    projection: &[usize],
    scan_projection: Option<&ProjectionExprs>,
    scan_output_schema: &SchemaRef,
    table_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    for index in projection {
        let field = scan_output_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(field.name(), *index));
        collect_source_required_columns(&expr, scan_projection, table_schema, required)?;
    }
    Ok(())
}

fn collect_source_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    scan_projection: Option<&ProjectionExprs>,
    table_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    let source_expr = match scan_projection {
        Some(projection) => projection.unproject_expr(expr).map_err(|_error| FallbackReason::UnsupportedExpression)?,
        None => Arc::clone(expr),
    };
    collect_required_columns(&source_expr, table_schema, required)
}

fn column_file_indices(file_schema: &SchemaRef, columns: &[String]) -> Result<Vec<usize>, FallbackReason> {
    columns
        .iter()
        .map(|column| file_schema.index_of(column).map_err(|_error| FallbackReason::UnsupportedExpression))
        .collect()
}

/// Builds the native source node's emitted schema from file and partition columns.
///
/// The schema follows all read file columns first, then synthetic partition
/// columns, so native expressions can address both kinds of source data
/// uniformly. `projected_output_columns` is role/diagnostic metadata about what
/// should remain after source-adjacent native filtering/projection; it does not
/// define this schema. Pure filter columns are still emitted here and should be
/// removed by a later native projection when they are not needed downstream.
pub(super) fn native_source_output_schema(
    file_scan: &FileScanConfig,
    requirements: &NativeParquetColumnRequirements,
) -> Result<SchemaRef, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let mut fields = Vec::with_capacity(requirements.read_columns.len() + requirements.partition_columns.len());
    for column in &requirements.read_columns {
        fields
            .push(file_schema.field_with_name(column).map_err(|_error| FallbackReason::UnsupportedExpression)?.clone());
    }
    for partition_column in &requirements.partition_columns {
        fields.push(arrow::datatypes::Field::new(
            partition_column.name.as_str(),
            partition_column.data_type.clone(),
            partition_column.nullable,
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Returns partition columns required by native expressions.
///
/// DataFusion keeps partition values on each `PartitionedFile`. The native IR
/// models required partition fields as synthetic source columns with per-file
/// literal values, making the metadata reusable outside the DataFusion adapter.
fn native_required_partition_columns(
    file_scan: &FileScanConfig,
    required_columns: &BTreeSet<String>,
) -> Result<(Vec<NativePartitionColumn>, Vec<usize>), FallbackReason> {
    let mut columns = Vec::new();
    let mut indices = Vec::new();
    for (index, field) in file_scan.table_partition_cols().iter().enumerate() {
        if required_columns.contains(field.name()) {
            if !is_supported_native_type(field.data_type()) {
                return Err(FallbackReason::UnsupportedType);
            }
            columns.push(NativePartitionColumn::new(field.name(), field.data_type().clone(), field.is_nullable()));
            indices.push(index);
        }
    }
    Ok((columns, indices))
}

/// Resolves logical source column names to file-schema or partition-column refs.
fn column_source_refs(
    file_schema: &SchemaRef,
    partition_columns: &[NativePartitionColumn],
    columns: &[String],
) -> Result<Vec<NativeSourceColumnRef>, FallbackReason> {
    columns
        .iter()
        .map(|column| {
            if let Ok(index) = file_schema.index_of(column) {
                return Ok(NativeSourceColumnRef::File(index));
            }
            partition_columns
                .iter()
                .position(|field| &field.name == column)
                .map(NativeSourceColumnRef::Partition)
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect()
}

fn validate_partition_columns(
    file_scan: &FileScanConfig,
    required_partition_indices: &[usize],
) -> Result<(), FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if required_partition_indices.is_empty() {
        return Ok(());
    }

    let partition_column_count = file_scan.table_partition_cols().len();
    if required_partition_indices.iter().any(|index| *index >= partition_column_count) {
        return Err(FallbackReason::NativeSourcePartitionValues);
    }
    for file in &files {
        if required_partition_indices.iter().any(|index| file.partition_values.get(*index).is_none()) {
            return Err(FallbackReason::NativeSourcePartitionValues);
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
) -> Result<Vec<NativeLiteralValue>, FallbackReason> {
    required_partition_indices
        .iter()
        .map(|index| {
            file.partition_values
                .get(*index)
                .ok_or(FallbackReason::NativeSourcePartitionValues)
                .and_then(|value| native_literal_value(value, NativeLiteralContext::SourceValue))
        })
        .collect()
}
