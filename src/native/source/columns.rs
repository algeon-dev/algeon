use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::{aggregates::AggregateExec, filter::FilterExec, projection::ProjectionExec},
};
use nexus_query_engine::{
    MissingColumn as NativeMissingColumn, PartitionColumn as NativePartitionColumn,
    SourceColumnRef as NativeSourceColumnRef,
};
use parquet::{
    arrow::parquet_to_arrow_schema,
    file::reader::{FileReader, SerializedFileReader},
};

use crate::{
    native::{
        pruning::{
            local_object_path_to_filesystem, scan_predicate_is_dynamic_filter_hint,
            scan_predicate_without_dynamic_filter_hints,
        },
        validate::{collect_required_columns, is_supported_native_type, native_zero_column_count_source_is_safe},
    },
    policy::FallbackReason,
};

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
#[derive(Debug, Clone, PartialEq)]
pub(in crate::native) struct NativeParquetColumnRequirements {
    /// Physical Parquet columns that the native source must read and emit.
    ///
    /// This excludes partition columns because they are supplied from
    /// DataFusion `PartitionedFile` metadata, not from file pages. Columns used
    /// only by native filters are included so those filters can consume them.
    pub(in crate::native) read_columns: Vec<String>,
    /// File-schema indices for `read_columns`.
    pub(in crate::native) read_column_file_indices: Vec<usize>,
    /// Source columns expected to remain after source-adjacent native projection.
    ///
    /// These may include file columns and synthetic partition columns. This is
    /// role/diagnostic metadata and does not define the source node's emitted
    /// schema; `native_source_output_schema` emits all `read_columns` plus all
    /// represented partition columns.
    pub(in crate::native) projected_output_columns: Vec<String>,
    /// File or partition source references for `projected_output_columns`.
    pub(in crate::native) projected_output_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Source columns emitted only so source-adjacent predicates can evaluate.
    ///
    /// These should be removed by a later native projection when they are not
    /// needed downstream.
    pub(in crate::native) pure_filter_columns: Vec<String>,
    /// File or partition source references for `pure_filter_columns`.
    pub(in crate::native) pure_filter_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Source columns required by source-adjacent predicates.
    pub(in crate::native) predicate_columns: Vec<String>,
    /// Required partition columns represented as synthetic native source columns.
    pub(in crate::native) partition_columns: Vec<NativePartitionColumn>,
    /// Optional literal values for synthetic partition-style columns.
    ///
    /// `None` means the value must come from file partition metadata. `Some`
    /// represents a table-format default value materialized for every row of
    /// the file.
    pub(in crate::native) partition_column_values: Vec<Option<nexus_query_engine::LiteralValue>>,
    /// DataFusion table partition-column indices required by this source.
    pub(in crate::native) partition_column_indices: Vec<usize>,
    /// Required nullable table columns missing from the physical file schema.
    pub(in crate::native) missing_columns: Vec<NativeMissingColumn>,
    /// DataFusion scan-output column names consumed by the lowered subtree.
    ///
    /// This preserves the adapter-side view of the scan projection so later
    /// lowering can connect DataFusion expressions to native source columns.
    pub(in crate::native) scan_output_columns: Vec<String>,
    /// Whether this requirement exists only to preserve row presence.
    ///
    /// Table-format field-id rebinding must keep such carriers backed by a
    /// physical file read; a missing/default synthetic value does not prove row
    /// positions for source execution.
    pub(in crate::native) requires_physical_row_presence: bool,
}

/// Computes source column roles for aggregate-root native lowering.
///
/// Required/projected roles are derived from the aggregate, optional
/// pre-aggregate projection, and optional source filter. Filter-only columns are
/// read and emitted by the native source so filters can evaluate them, then
/// omitted from the columns expected to survive the source-adjacent native
/// projection. When the aggregate can be answered as a zero-column count, no
/// file or partition columns are required.
pub(in crate::native) fn required_parquet_column_requirements(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = native_physical_file_schema(file_scan)?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    if native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate)
        && !parquet_source_has_row_level_predicate(parquet_source)
    {
        return Ok(NativeParquetColumnRequirements {
            read_columns: Vec::new(),
            read_column_file_indices: Vec::new(),
            projected_output_columns: Vec::new(),
            projected_output_column_source_refs: Vec::new(),
            pure_filter_columns: Vec::new(),
            pure_filter_column_source_refs: Vec::new(),
            predicate_columns: Vec::new(),
            partition_columns: Vec::new(),
            partition_column_values: Vec::new(),
            partition_column_indices: Vec::new(),
            missing_columns: Vec::new(),
            scan_output_columns: Vec::new(),
            requires_physical_row_presence: false,
        });
    }

    let mut filter_columns: BTreeSet<String> = BTreeSet::new();
    let mut output_columns: BTreeSet<String> = BTreeSet::new();
    let mut scan_output_columns: BTreeSet<String> = BTreeSet::new();

    let scan_projection = file_scan.file_source().projection();
    let scan_output_schema = file_scan.projected_schema().map_err(|_error| FallbackReason::UnsupportedExpression)?;

    if filter.is_none() {
        collect_scan_predicate_source_columns(parquet_source, scan_projection, &table_schema, &mut filter_columns)?;
    }
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
        for filter in aggregate.filter_expr().iter().flatten() {
            collect_source_required_columns(filter, scan_projection, &table_schema, &mut output_columns)?;
            collect_required_columns(filter, &aggregate.input().schema(), &mut scan_output_columns)?;
        }
        for (group_expr, _alias) in aggregate.group_expr().expr() {
            collect_source_required_columns(group_expr, scan_projection, &table_schema, &mut output_columns)?;
            collect_required_columns(group_expr, &aggregate.input().schema(), &mut scan_output_columns)?;
        }
    }

    let mut read_columns: BTreeSet<String> = BTreeSet::new();
    read_columns.extend(filter_columns.iter().cloned());
    read_columns.extend(output_columns.iter().cloned());

    let mut read_columns =
        read_columns.into_iter().filter(|column| file_schema.index_of(column).is_ok()).collect::<BTreeSet<_>>();
    let mut pure_filter_columns = filter_columns.difference(&output_columns).cloned().collect::<Vec<_>>();
    let required_source_columns = filter_columns.union(&output_columns).cloned().collect::<BTreeSet<_>>();

    let mut requires_physical_row_presence = false;
    if read_columns.is_empty() && required_source_columns.is_empty() {
        let carrier = first_supported_physical_file_column(&file_schema)?;
        read_columns.insert(carrier.clone());
        pure_filter_columns.push(carrier);
        requires_physical_row_presence = true;
    }

    let read_columns = read_columns.into_iter().collect::<Vec<_>>();
    let projected_output_columns = output_columns.into_iter().collect::<Vec<_>>();
    let (partition_columns, partition_column_indices) =
        native_required_partition_columns(file_scan, &required_source_columns)?;
    let missing_columns =
        native_required_missing_columns(file_scan, &file_schema, &required_source_columns, &partition_columns)?;
    let read_column_file_indices = column_file_indices(&file_schema, &read_columns)?;
    let projected_output_column_source_refs =
        column_source_refs(&file_schema, &partition_columns, &missing_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs =
        column_source_refs(&file_schema, &partition_columns, &missing_columns, &pure_filter_columns)?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        predicate_columns: filter_columns.into_iter().collect(),
        partition_columns,
        partition_column_values: vec![None; partition_column_indices.len()],
        partition_column_indices,
        missing_columns,
        scan_output_columns: scan_output_columns.into_iter().collect(),
        requires_physical_row_presence,
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
pub(in crate::native) fn required_parquet_column_requirements_for_scan_columns(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    required_scan_output_indices: &[usize],
) -> Result<NativeParquetColumnRequirements, FallbackReason> {
    let file_schema = native_physical_file_schema(file_scan)?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let scan_output_schema = file_scan.projected_schema().map_err(|_error| FallbackReason::UnsupportedExpression)?;

    let mut filter_columns: BTreeSet<String> = BTreeSet::new();
    let mut output_columns: BTreeSet<String> = BTreeSet::new();
    let mut ordered_output_columns = Vec::new();
    let mut scan_output_columns: BTreeSet<String> = BTreeSet::new();

    if filter.is_none() {
        collect_scan_predicate_source_columns(parquet_source, scan_projection, &table_schema, &mut filter_columns)?;
    }
    if let Some(filter) = filter {
        collect_source_required_columns(filter.predicate(), scan_projection, &table_schema, &mut filter_columns)?;
        collect_required_columns(filter.predicate(), &scan_output_schema, &mut scan_output_columns)?;
    }

    for index in required_scan_output_indices {
        let field = scan_output_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(field.name(), *index));
        let mut expr_output_columns = BTreeSet::new();
        collect_source_required_columns(&expr, scan_projection, &table_schema, &mut expr_output_columns)?;
        for column in expr_output_columns {
            if output_columns.insert(column.clone()) {
                ordered_output_columns.push(column);
            }
        }
        scan_output_columns.insert(field.name().clone());
    }

    let mut read_columns: BTreeSet<String> = BTreeSet::new();
    read_columns.extend(filter_columns.iter().cloned());
    read_columns.extend(output_columns.iter().cloned());
    let mut read_columns =
        read_columns.into_iter().filter(|column| file_schema.index_of(column).is_ok()).collect::<Vec<_>>();
    let mut pure_filter_columns = filter_columns.difference(&output_columns).cloned().collect::<Vec<_>>();
    let mut requires_physical_row_presence = false;
    if read_columns.is_empty() && filter_columns.is_empty() && output_columns.is_empty() {
        let carrier = first_supported_physical_file_column(&file_schema)?;
        read_columns.push(carrier.clone());
        pure_filter_columns.push(carrier);
        requires_physical_row_presence = true;
    }
    let required_source_columns = filter_columns.union(&output_columns).cloned().collect::<BTreeSet<_>>();
    let projected_output_columns = ordered_output_columns;
    let (partition_columns, partition_column_indices) =
        native_required_partition_columns(file_scan, &required_source_columns)?;
    let missing_columns =
        native_required_missing_columns(file_scan, &file_schema, &required_source_columns, &partition_columns)?;
    let read_column_file_indices = column_file_indices(&file_schema, &read_columns)?;
    let projected_output_column_source_refs =
        column_source_refs(&file_schema, &partition_columns, &missing_columns, &projected_output_columns)?;
    let pure_filter_column_source_refs =
        column_source_refs(&file_schema, &partition_columns, &missing_columns, &pure_filter_columns)?;

    Ok(NativeParquetColumnRequirements {
        read_columns,
        read_column_file_indices,
        projected_output_columns,
        projected_output_column_source_refs,
        pure_filter_columns,
        pure_filter_column_source_refs,
        predicate_columns: filter_columns.into_iter().collect(),
        partition_columns,
        partition_column_values: vec![None; partition_column_indices.len()],
        partition_column_indices,
        missing_columns,
        scan_output_columns: scan_output_columns.into_iter().collect(),
        requires_physical_row_presence,
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

pub(in crate::native) fn parquet_source_has_row_level_predicate(parquet_source: &ParquetSource) -> bool {
    parquet_source.filter().is_some_and(|predicate| !scan_predicate_is_dynamic_filter_hint(predicate.as_ref()))
}

fn collect_scan_predicate_source_columns(
    parquet_source: &ParquetSource,
    scan_projection: Option<&ProjectionExprs>,
    table_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    let Some(predicate) = parquet_source.filter() else {
        return Ok(());
    };
    let Some(predicate) = scan_predicate_without_dynamic_filter_hints(&predicate) else {
        return Ok(());
    };
    collect_required_columns(&predicate, table_schema, required)
        .or_else(|_| collect_source_required_columns(&predicate, scan_projection, table_schema, required))
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

fn native_physical_file_schema(file_scan: &FileScanConfig) -> Result<SchemaRef, FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        if files.len() > 1 {
            return Err(FallbackReason::NativeSourceSchemaMapping);
        }
        return Ok(Arc::clone(file_scan.file_schema()));
    }
    let Some(first_file) = files.first() else {
        return Ok(Arc::clone(file_scan.file_schema()));
    };
    let first_schema = local_parquet_footer_schema(first_file.object_meta.location.as_ref())?;
    for file in files.iter().skip(1) {
        let schema = local_parquet_footer_schema(file.object_meta.location.as_ref())?;
        if schema.as_ref() != first_schema.as_ref() {
            return Err(FallbackReason::NativeSourceSchemaMapping);
        }
    }
    Ok(first_schema)
}

fn local_parquet_footer_schema(location: &str) -> Result<SchemaRef, FallbackReason> {
    let path = local_object_path_to_filesystem(location);
    let reader =
        SerializedFileReader::try_from(path.as_path()).map_err(|_error| FallbackReason::NativeSourceSchemaMapping)?;
    let metadata = reader.metadata().file_metadata();
    let schema = parquet_to_arrow_schema(metadata.schema_descr(), metadata.key_value_metadata())
        .map_err(|_error| FallbackReason::NativeSourceSchemaMapping)?;
    Ok(Arc::new(schema))
}

fn column_file_indices(file_schema: &SchemaRef, columns: &[String]) -> Result<Vec<usize>, FallbackReason> {
    columns
        .iter()
        .map(|column| file_schema.index_of(column).map_err(|_error| FallbackReason::UnsupportedExpression))
        .collect()
}

fn first_supported_physical_file_column(file_schema: &SchemaRef) -> Result<String, FallbackReason> {
    file_schema
        .fields()
        .iter()
        .find(|field| is_supported_native_type(field.data_type()))
        .map(|field| field.name().clone())
        .ok_or(FallbackReason::UnsupportedType)
}

/// Builds the native source node's emitted schema from file and partition columns.
///
/// The schema follows all read file columns first, then synthetic partition
/// columns, so native expressions can address both kinds of source data
/// uniformly. `projected_output_columns` is role/diagnostic metadata about what
/// should remain after source-adjacent native filtering/projection; it does not
/// define this schema. Pure filter columns are still emitted here and should be
/// removed by a later native projection when they are not needed downstream.
pub(in crate::native) fn native_source_output_schema(
    file_scan: &FileScanConfig,
    requirements: &NativeParquetColumnRequirements,
) -> Result<SchemaRef, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let mut fields = Vec::with_capacity(
        requirements.read_columns.len() + requirements.partition_columns.len() + requirements.missing_columns.len(),
    );
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
    for missing_column in &requirements.missing_columns {
        fields.push(arrow::datatypes::Field::new(
            missing_column.name.as_str(),
            missing_column.data_type.clone(),
            missing_column.nullable,
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

fn native_required_missing_columns(
    file_scan: &FileScanConfig,
    file_schema: &SchemaRef,
    required_columns: &BTreeSet<String>,
    partition_columns: &[NativePartitionColumn],
) -> Result<Vec<NativeMissingColumn>, FallbackReason> {
    let table_schema = file_scan.file_source().table_schema().table_schema();
    let mut columns = Vec::new();
    for column in required_columns {
        if file_schema.index_of(column).is_ok() || partition_columns.iter().any(|field| &field.name == column) {
            continue;
        }
        let field = table_schema.field_with_name(column).map_err(|_error| FallbackReason::NativeSourceSchemaMapping)?;
        if !field.is_nullable() {
            return Err(FallbackReason::NativeSourceSchemaMapping);
        }
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        columns.push(NativeMissingColumn::new(column.as_str(), field.data_type().clone(), field.is_nullable()));
    }
    Ok(columns)
}

/// Resolves logical source column names to file-schema or partition-column refs.
fn column_source_refs(
    file_schema: &SchemaRef,
    partition_columns: &[NativePartitionColumn],
    missing_columns: &[NativeMissingColumn],
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
                .or_else(|| {
                    missing_columns
                        .iter()
                        .position(|field| &field.name == column)
                        .map(NativeSourceColumnRef::MissingNull)
                })
                .ok_or(FallbackReason::UnsupportedExpression)
        })
        .collect()
}
