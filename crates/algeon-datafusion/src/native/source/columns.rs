use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    execution::object_store::ObjectStoreUrl,
    object_store::ObjectMeta,
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::filter::FilterExec,
};
use parquet::arrow::parquet_to_arrow_schema;
use query_planning::source::{
    MissingColumn as NativeMissingColumn, PartitionColumn as NativePartitionColumn,
    SourceColumnRef as NativeSourceColumnRef,
};

use crate::{
    capability::CapabilityReason,
    native::{
        footers::{FooterUnavailable, object_parquet_footer},
        pruning::{scan_predicate_is_dynamic_filter_hint, scan_predicate_without_dynamic_filter_hints},
        validate::collect_required_columns,
    },
};

/// Column-role bridge from a DataFusion Parquet scan to native source metadata.
///
/// DataFusion exposes columns through several coordinate systems (table schema,
/// file schema, pushed scan projection, filter projection, and partition values).
/// The native engine needs a frontend-neutral description of which physical file
/// columns the source must read and emit, which partition columns it must
/// synthesize, which emitted source columns are expected to remain after
/// source-adjacent native filtering/projection, and which emitted columns exist
/// only to evaluate filters. This struct keeps those roles separate so lowering
/// can remove filter-only columns after filters while still emitting them from
/// the source when needed.
#[derive(Debug, Clone, PartialEq, Default)]
pub(in crate::native) struct NativeParquetColumnRequirements {
    /// Physical Parquet columns that the native source must read and emit.
    ///
    /// This excludes partition columns, which are supplied from DataFusion
    /// `PartitionedFile` metadata and are absent from file pages. Columns used
    /// only by native filters are included so those filters can consume them.
    pub(in crate::native) read_columns: Vec<String>,
    /// File-schema indices for `read_columns`.
    pub(in crate::native) read_column_file_indices: Vec<usize>,
    /// Source columns expected to remain after source-adjacent native projection.
    ///
    /// These may include file columns and synthetic partition columns. This is
    /// role/diagnostic metadata and does not define the source node's emitted
    /// schema; `native_source_output_schema` emits all `read_columns` plus all
    /// partition and missing columns.
    pub(in crate::native) projected_output_columns: Vec<String>,
    /// File, partition, or missing-column source references for `projected_output_columns`.
    pub(in crate::native) projected_output_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Source columns emitted only so source-adjacent predicates can evaluate.
    ///
    /// These should be removed by a later native projection when they are not
    /// needed downstream.
    pub(in crate::native) pure_filter_columns: Vec<String>,
    /// File, partition, or missing-column source references for `pure_filter_columns`.
    pub(in crate::native) pure_filter_column_source_refs: Vec<NativeSourceColumnRef>,
    /// Source columns required by source-adjacent predicates.
    pub(in crate::native) predicate_columns: Vec<String>,
    /// Required partition columns represented as synthetic native source columns.
    pub(in crate::native) partition_columns: Vec<NativePartitionColumn>,
    /// DataFusion table partition-column indices required by this source.
    pub(in crate::native) partition_column_indices: Vec<usize>,
    /// Required nullable table columns missing from the physical file schema.
    pub(in crate::native) missing_columns: Vec<NativeMissingColumn>,
    /// DataFusion scan-output column names consumed by the lowered subtree.
    ///
    /// This preserves the adapter-side view of the scan projection so later
    /// lowering can connect DataFusion expressions to native source columns.
    pub(in crate::native) scan_output_columns: Vec<String>,
}

impl NativeParquetColumnRequirements {
    /// Requirements of a scan that counts rows without reading a column.
    pub(in crate::native) fn row_count_only() -> Self {
        Self::default()
    }
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
) -> Result<NativeParquetColumnRequirements, CapabilityReason> {
    let file_schema = native_physical_file_schema(file_scan)?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let scan_output_schema = file_scan.projected_schema().map_err(|_error| CapabilityReason::UnsupportedExpression)?;

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
        let field = scan_output_schema.fields().get(*index).ok_or(CapabilityReason::UnsupportedExpression)?;
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
    if read_columns.is_empty() && filter_columns.is_empty() && output_columns.is_empty() {
        let carrier = first_supported_physical_file_column(&file_schema)?;
        read_columns.push(carrier.clone());
        pure_filter_columns.push(carrier);
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
        partition_column_indices,
        missing_columns,
        scan_output_columns: scan_output_columns.into_iter().collect(),
    })
}

pub(in crate::native) fn parquet_source_has_row_level_predicate(parquet_source: &ParquetSource) -> bool {
    parquet_source.filter().is_some_and(|predicate| !scan_predicate_is_dynamic_filter_hint(predicate.as_ref()))
}

fn collect_scan_predicate_source_columns(
    parquet_source: &ParquetSource,
    scan_projection: Option<&ProjectionExprs>,
    table_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), CapabilityReason> {
    let Some(predicate) = parquet_source.filter() else {
        return Ok(());
    };
    let Some(predicate) = scan_predicate_without_dynamic_filter_hints(&predicate) else {
        return Ok(());
    };
    collect_required_columns(&predicate, table_schema, required)
        .or_else(|_| collect_source_required_columns(&predicate, scan_projection, table_schema, required))
}

fn collect_source_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    scan_projection: Option<&ProjectionExprs>,
    table_schema: &SchemaRef,
    required: &mut BTreeSet<String>,
) -> Result<(), CapabilityReason> {
    let source_expr = match scan_projection {
        Some(projection) => {
            projection.unproject_expr(expr).map_err(|_error| CapabilityReason::UnsupportedExpression)?
        },
        None => Arc::clone(expr),
    };
    collect_required_columns(&source_expr, table_schema, required)
}

/// The physical schema of the scan's Parquet objects, which column indices
/// address.
///
/// Every distinct object's footer must hold the same columns in the same
/// order with the same Arrow types. A column may be required in one file and
/// optional in another when the table schema declares it nullable; DataFusion
/// widens nullability the same way when it merges file schemas. A remote scan
/// of one object keeps the DataFusion file schema without reading its footer.
pub(in crate::native) fn native_physical_file_schema(
    file_scan: &FileScanConfig,
) -> Result<SchemaRef, CapabilityReason> {
    let mut checked_locations = HashSet::new();
    // Byte-range splits of one object share its footer, so each distinct object is checked once.
    let objects = file_scan
        .file_groups
        .iter()
        .flat_map(|group| group.files())
        .map(|file| &file.object_meta)
        .filter(|object| checked_locations.insert(&object.location))
        .collect::<Vec<_>>();
    let local = file_scan.object_store_url.as_str().starts_with("file://");
    if objects.is_empty() || !local && objects.len() == 1 {
        return Ok(Arc::clone(file_scan.file_schema()));
    }
    let table_schema = file_scan.file_schema();
    let mut physical_schema: Option<SchemaRef> = None;
    for object in objects {
        let schema = parquet_footer_schema(&file_scan.object_store_url, object)?;
        let nullable_in_table = |field: &Field| {
            table_schema.field_with_name(field.name()).map_or(true, |table_field| table_field.is_nullable())
        };
        if schema.fields().iter().any(|field| field.is_nullable() && !nullable_in_table(field)) {
            return Err(CapabilityReason::NativeSourceFileSchemaMismatch);
        }
        match &physical_schema {
            None => physical_schema = Some(schema),
            Some(first) => {
                if first.fields().len() != schema.fields().len()
                    || first.fields().iter().zip(schema.fields()).any(|(first, field)| {
                        first.name() != field.name()
                            || data_type_without_metadata(first.data_type())
                                != data_type_without_metadata(field.data_type())
                    })
                {
                    return Err(CapabilityReason::NativeSourceFileSchemaMismatch);
                }
            },
        }
    }
    Ok(physical_schema.unwrap_or_else(|| Arc::clone(file_scan.file_schema())))
}

/// `data_type` with the metadata of every nested field removed: writers
/// attach field ids and other annotations to nested fields, which do not
/// change how the column decodes.
fn data_type_without_metadata(data_type: &DataType) -> DataType {
    let field_without_metadata = |field: &FieldRef| -> FieldRef {
        Arc::new(
            Field::new(field.name(), data_type_without_metadata(field.data_type()), field.is_nullable())
                .with_metadata(HashMap::new()),
        )
    };
    match data_type {
        DataType::List(field) => DataType::List(field_without_metadata(field)),
        DataType::LargeList(field) => DataType::LargeList(field_without_metadata(field)),
        DataType::ListView(field) => DataType::ListView(field_without_metadata(field)),
        DataType::LargeListView(field) => DataType::LargeListView(field_without_metadata(field)),
        DataType::FixedSizeList(field, size) => DataType::FixedSizeList(field_without_metadata(field), *size),
        DataType::Map(field, sorted) => DataType::Map(field_without_metadata(field), *sorted),
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(field_without_metadata).collect()),
        other => other.clone(),
    }
}

fn parquet_footer_schema(
    object_store_url: &ObjectStoreUrl,
    object: &ObjectMeta,
) -> Result<SchemaRef, CapabilityReason> {
    let footer = object_parquet_footer(object_store_url, object).map_err(|unavailable| match unavailable {
        FooterUnavailable::Local => CapabilityReason::NativeSourceSchemaMapping,
        FooterUnavailable::RemoteNotCached => CapabilityReason::NativeSourceRemoteFooterUnavailable,
    })?;
    let metadata = footer.file_metadata();
    let schema = parquet_to_arrow_schema(metadata.schema_descr(), metadata.key_value_metadata())
        .map_err(|_error| CapabilityReason::NativeSourceSchemaMapping)?;
    Ok(Arc::new(schema))
}

fn column_file_indices(file_schema: &SchemaRef, columns: &[String]) -> Result<Vec<usize>, CapabilityReason> {
    columns
        .iter()
        .map(|column| file_schema.index_of(column).map_err(|_error| CapabilityReason::UnsupportedExpression))
        .collect()
}

fn first_supported_physical_file_column(file_schema: &SchemaRef) -> Result<String, CapabilityReason> {
    file_schema
        .fields()
        .iter()
        .find(|field| is_supported_arrow_type(field.data_type()))
        .map(|field| field.name().clone())
        .ok_or(CapabilityReason::UnsupportedType)
}

/// Builds the native source node's emitted schema from file, partition, and missing columns.
///
/// The schema lists all read file columns first, then synthetic partition
/// columns, then missing columns. Native expressions can address all three kinds
/// of source data uniformly. `projected_output_columns` is role/diagnostic metadata about what
/// should remain after source-adjacent native filtering/projection; it does not
/// define this schema. Pure filter columns are still emitted here and should be
/// removed by a later native projection when they are not needed downstream.
pub(in crate::native) fn native_source_output_schema(
    file_scan: &FileScanConfig,
    requirements: &NativeParquetColumnRequirements,
) -> Result<SchemaRef, CapabilityReason> {
    let file_schema = file_scan.file_schema();
    let mut fields = Vec::with_capacity(
        requirements.read_columns.len() + requirements.partition_columns.len() + requirements.missing_columns.len(),
    );
    for column in &requirements.read_columns {
        fields.push(
            file_schema.field_with_name(column).map_err(|_error| CapabilityReason::UnsupportedExpression)?.clone(),
        );
    }
    for partition_column in &requirements.partition_columns {
        fields.push(Field::new(
            partition_column.name.as_str(),
            partition_column.data_type.clone(),
            partition_column.nullable,
        ));
    }
    for missing_column in &requirements.missing_columns {
        fields.push(Field::new(
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
) -> Result<(Vec<NativePartitionColumn>, Vec<usize>), CapabilityReason> {
    let mut columns = Vec::new();
    let mut indices = Vec::new();
    for (index, field) in file_scan.table_partition_cols().iter().enumerate() {
        if required_columns.contains(field.name()) {
            if !is_supported_arrow_type(field.data_type()) {
                return Err(CapabilityReason::UnsupportedType);
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
) -> Result<Vec<NativeMissingColumn>, CapabilityReason> {
    let table_schema = file_scan.file_source().table_schema().table_schema();
    let mut columns = Vec::new();
    for column in required_columns {
        if file_schema.index_of(column).is_ok() || partition_columns.iter().any(|field| &field.name == column) {
            continue;
        }
        let field =
            table_schema.field_with_name(column).map_err(|_error| CapabilityReason::NativeSourceSchemaMapping)?;
        if !field.is_nullable() {
            return Err(CapabilityReason::NativeSourceSchemaMapping);
        }
        if !is_supported_arrow_type(field.data_type()) {
            return Err(CapabilityReason::UnsupportedType);
        }
        columns.push(NativeMissingColumn::new(column.as_str(), field.data_type().clone(), field.is_nullable()));
    }
    Ok(columns)
}

/// Resolves logical source column names to file-schema, partition-column, or missing-column refs.
fn column_source_refs(
    file_schema: &SchemaRef,
    partition_columns: &[NativePartitionColumn],
    missing_columns: &[NativeMissingColumn],
    columns: &[String],
) -> Result<Vec<NativeSourceColumnRef>, CapabilityReason> {
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
                .ok_or(CapabilityReason::UnsupportedExpression)
        })
        .collect()
}
