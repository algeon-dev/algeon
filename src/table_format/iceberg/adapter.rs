//! Root-owned adapters for the DataFusion-free Iceberg leaf contracts.
//!
//! The leaf owns catalog/snapshot/task semantics. This module deliberately
//! performs only the one-way adaptation into root table-format facts, where
//! native admission, KvikIO/source strategy, cache policy, and DataFusion
//! execution remain owned by `datafusion-nexus`.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef, UnionFields};
use nexus_iceberg::{
    IcebergColumnBinding, IcebergColumnRole, IcebergColumnSource, IcebergDefaultValue, IcebergDeleteContent,
    IcebergDeleteFile, IcebergError, IcebergErrorCode, IcebergErrorStatus, IcebergFileFormat, IcebergNameMapping,
    IcebergNameMappingField, IcebergPartitionTransform, IcebergScalarValue, IcebergScanCandidate, IcebergScanRequest,
    IcebergSchema, IcebergSchemaField,
};

use crate::{
    error::{Error, ErrorCode, Result},
    table_format::{
        TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatPartitionTransform,
        TableFormatPartitionValue, TableFormatReadCapability, TableFormatScalarValue, TableFormatScanPlan,
        TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
    },
};

const CANDIDATE_CONVERSION_OPERATION: &str = "datafusion_nexus.table_format.iceberg.convert_scan_candidate";

/// Converts a frozen leaf candidate into root-owned scan facts.
///
/// The root facts are intentionally sufficient for existing native admission,
/// while the exact leaf candidate is retained on the resulting plan. This is
/// what prevents the conversion from erasing Iceberg-only semantics that are
/// not yet executable natively (for example residuals, pruning evidence,
/// sequence values, file ranges, or nested name mappings).
pub fn iceberg_scan_candidate_to_table_format_scan_plan(
    candidate: IcebergScanCandidate,
    catalog_type: Option<&str>,
) -> Result<TableFormatScanPlan> {
    let candidate = Arc::new(candidate);
    let logical_schema = logical_schema_for_request(&candidate.resolved_table.schema, &candidate.request)?;
    let native_source_schema = canonical_schema(&candidate.resolved_table.schema.arrow_schema);
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_logical_schema(logical_schema)
        .with_native_source_schema(native_source_schema)
        .with_iceberg_candidate(Arc::clone(&candidate));

    if let Some(snapshot_id) = candidate.resolved_table.snapshot.snapshot_id {
        plan = plan.with_snapshot_id(snapshot_id.to_string());
    }
    if let Some(catalog_type) = catalog_type {
        plan = plan.with_catalog_type(catalog_type);
    }
    match candidate.request.source_limit.row_limit.map(usize::try_from).transpose() {
        Ok(limit) => plan = plan.with_scan_limit(limit),
        Err(_) => {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::TableFormatPlanningFailed,
                "Iceberg source row limit exceeds the root adapter platform range",
            ));
        },
    }

    append_request_columns(
        &mut plan,
        &candidate.resolved_table.schema,
        &candidate.request,
        candidate.name_mapping.as_ref(),
    )?;
    append_candidate_semantic_rejections(&mut plan, &candidate);
    append_candidate_files(&mut plan, &candidate)?;
    Ok(plan)
}

fn append_request_columns(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
    request: &IcebergScanRequest,
    name_mapping: Option<&IcebergNameMapping>,
) -> Result<()> {
    match &request.projection {
        nexus_iceberg::IcebergProjection::AllColumns => {
            for (index, field) in schema.fields.iter().enumerate() {
                let arrow = schema.arrow_schema.field(index);
                let binding = IcebergColumnBinding {
                    field_id: field.field_id,
                    output_name: field.name.clone(),
                    source: IcebergColumnSource::PhysicalFile,
                    role: IcebergColumnRole::Projected,
                    missing_default: field.initial_default.clone(),
                };
                append_binding(plan, schema, &binding, name_mapping, Some(arrow))?;
            }
        },
        nexus_iceberg::IcebergProjection::Columns(bindings) => {
            for binding in bindings {
                append_binding(plan, schema, binding, name_mapping, None)?;
            }
        },
    }
    for binding in &request.filter_bindings {
        append_binding(plan, schema, binding, name_mapping, None)?;
    }
    Ok(())
}

fn append_binding(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
    binding: &IcebergColumnBinding,
    name_mapping: Option<&IcebergNameMapping>,
    top_level_arrow: Option<&Field>,
) -> Result<()> {
    let (schema_field, arrow, is_top_level) = schema_field(schema, binding.field_id).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidExecutionPlan,
            CANDIDATE_CONVERSION_OPERATION,
            "Iceberg candidate references a field absent from its frozen schema",
        )
        .with_fact("field_id", binding.field_id)
    })?;
    let arrow = canonical_field(top_level_arrow.unwrap_or(arrow));
    let source = match binding.source {
        IcebergColumnSource::PhysicalFile
            if is_top_level && candidate_has_identity_partition_field(plan, binding.field_id) =>
        {
            TableFormatColumnSource::IdentityPartitionValue
        },
        IcebergColumnSource::PhysicalFile if is_top_level => TableFormatColumnSource::DirectTopLevelFileColumn,
        IcebergColumnSource::PartitionConstant => TableFormatColumnSource::IdentityPartitionValue,
        IcebergColumnSource::MissingWithDefault => TableFormatColumnSource::MissingNullableColumn,
        // Root native scan facts have no metadata-column source. Retain the
        // leaf candidate unchanged and reject only the optional native path.
        IcebergColumnSource::MetadataColumn | IcebergColumnSource::PhysicalFile => {
            TableFormatColumnSource::UnsupportedNestedProjection
        },
    };
    let source_name =
        matches!(source, TableFormatColumnSource::DirectTopLevelFileColumn).then(|| schema_field.name.clone());
    let mut mapping = TableFormatColumnMapping::new(
        Some(binding.field_id),
        binding.output_name.clone(),
        source_name,
        arrow.data_type().clone(),
        arrow.is_nullable(),
        source,
        TableFormatColumnRole::Projected,
    );
    if let Some(default) = binding.missing_default.as_ref() {
        match default {
            IcebergDefaultValue::Scalar(default) if iceberg_scalar_is_not_native_materializable(default) => {
                plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                    TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported,
                    "Iceberg schema-evolution default cannot be materialized by the native source",
                ));
            },
            IcebergDefaultValue::Scalar(default) => {
                mapping = mapping.with_missing_default_value(root_scalar(default));
            },
            IcebergDefaultValue::Unsupported => {
                plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                    TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported,
                    "Iceberg schema-evolution default is unsupported by the native source",
                ));
            },
        }
    }
    if let Some(names) = name_mapping_names_for_field_id(name_mapping, binding.field_id) {
        mapping = mapping.with_field_id_source_names(names);
    }

    match binding.role {
        IcebergColumnRole::Projected => plan.projected_columns.push(mapping),
        IcebergColumnRole::FilterOnly => {
            mapping.role = TableFormatColumnRole::PureFilter;
            plan.pure_filter_columns.push(mapping);
        },
        IcebergColumnRole::ProjectedAndFilter => {
            plan.projected_columns.push(mapping.clone());
            mapping.role = TableFormatColumnRole::PureFilter;
            plan.pure_filter_columns.push(mapping);
        },
    }
    Ok(())
}

fn candidate_has_identity_partition_field(plan: &TableFormatScanPlan, field_id: i32) -> bool {
    let Some(candidate) = plan.iceberg_candidate() else {
        return false;
    };
    !candidate.data_files.is_empty()
        && candidate.data_files.iter().all(|file| {
            file.partition_values.iter().any(|partition| {
                partition.source_field_id == field_id && partition.transform == IcebergPartitionTransform::Identity
            })
        })
}

fn append_candidate_semantic_rejections(plan: &mut TableFormatScanPlan, candidate: &IcebergScanCandidate) {
    use nexus_iceberg::IcebergPredicateEnforcement;

    if matches!(candidate.request.predicate.enforcement, IcebergPredicateEnforcement::PartiallyEnforced) {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported,
            "Iceberg candidate retains a residual predicate for its eventual consumer",
        ));
    }

    let projection_bindings = match &candidate.request.projection {
        nexus_iceberg::IcebergProjection::AllColumns => &[][..],
        nexus_iceberg::IcebergProjection::Columns(bindings) => bindings,
    };
    for binding in projection_bindings.iter().chain(&candidate.request.filter_bindings) {
        let identity_partition_file_count = candidate
            .data_files
            .iter()
            .filter(|file| {
                file.partition_values.iter().any(|partition| {
                    partition.source_field_id == binding.field_id
                        && partition.transform == IcebergPartitionTransform::Identity
                })
            })
            .count();
        if identity_partition_file_count > 0 && identity_partition_file_count != candidate.data_files.len() {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported,
                "Iceberg field is an identity partition constant for only some selected files",
            ));
        }
    }
    if candidate.name_mapping.as_ref().is_some_and(iceberg_name_mapping_has_nested_fields) {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported,
            "Iceberg nested name mapping is retained but cannot be materialized by the native source",
        ));
    }
}

fn iceberg_name_mapping_has_nested_fields(mapping: &IcebergNameMapping) -> bool {
    mapping.fields.iter().any(|field| !field.fields.is_empty() || iceberg_name_mapping_field_has_nested_fields(field))
}

fn iceberg_name_mapping_field_has_nested_fields(field: &IcebergNameMappingField) -> bool {
    !field.fields.is_empty() || field.fields.iter().any(iceberg_name_mapping_field_has_nested_fields)
}

fn append_candidate_files(plan: &mut TableFormatScanPlan, candidate: &IcebergScanCandidate) -> Result<()> {
    for (data_index, data_file) in candidate.data_files.iter().enumerate() {
        let id = format!("iceberg-data-{data_index}");
        if data_file.range.start != 0 || data_file.range.length != data_file.file_size_bytes {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                TableFormatUnsupportedFeatureCode::IcebergFileRangeUnsupported,
                id.clone(),
                "Iceberg file range is not a whole-file native Parquet read and cannot be materialized by the native source",
            ));
        }
        if data_file.encryption_key_metadata_present {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                TableFormatUnsupportedFeatureCode::IcebergEncryptedFileUnsupported,
                id.clone(),
                "Iceberg data-file encryption metadata is retained but native reads do not support encrypted files",
            ));
        }
        for partition in &data_file.partition_values {
            if !matches!(partition.transform, IcebergPartitionTransform::Identity) {
                plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                    TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
                    id.clone(),
                    "Iceberg partition transform is retained but cannot be materialized by the native source",
                ));
            }
            if iceberg_scalar_is_not_native_materializable(&partition.value) {
                plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                    TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported,
                    id.clone(),
                    "Iceberg partition scalar is retained but cannot be materialized by the native source",
                ));
            }
        }
        let mut root = TableFormatDataFile::new(
            id.clone(),
            data_file.location.clone(),
            root_file_format(data_file.format),
            read_capability(&data_file.location),
        )
        .with_size_bytes(data_file.file_size_bytes);
        if let Some(row_count) = data_file.record_count {
            root = root.with_row_count(row_count);
        }
        let partition_values = data_file
            .partition_values
            .iter()
            .map(|value| {
                root_partition_value(value, &candidate.resolved_table.schema).ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        CANDIDATE_CONVERSION_OPERATION,
                        "Iceberg candidate partition references a field absent from its frozen schema",
                    )
                    .with_fact("data_file_id", &id)
                    .with_fact("field_id", value.source_field_id)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        root = root.with_partition_values(partition_values);

        for (delete_index, delete_file) in data_file.delete_files.iter().enumerate() {
            if delete_file.encryption_key_metadata_present {
                plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
                    TableFormatUnsupportedFeatureCode::IcebergEncryptedFileUnsupported,
                    id.clone(),
                    "Iceberg delete-file encryption metadata is retained but native reads do not support encrypted files",
                ));
            }
            let delete = root_delete_file(&id, data_index, delete_index, delete_file);
            match delete_file.content {
                IcebergDeleteContent::Position => root.push_position_delete_file(delete.clone()),
                IcebergDeleteContent::Equality => root.push_equality_delete_file(delete.clone()),
            }
            plan.delete_files.push(delete);
        }
        plan.data_files.push(root);
    }
    Ok(())
}

fn root_delete_file(
    data_file_id: &str,
    data_index: usize,
    delete_index: usize,
    delete_file: &IcebergDeleteFile,
) -> TableFormatDeleteFile {
    let content = match delete_file.content {
        IcebergDeleteContent::Position => TableFormatDeleteFileContent::PositionDeletes,
        IcebergDeleteContent::Equality => TableFormatDeleteFileContent::EqualityDeletes,
    };
    let mut root = TableFormatDeleteFile::for_data_file(
        format!("iceberg-delete-{data_index}-{delete_index}"),
        data_file_id,
        delete_file.location.clone(),
        root_file_format(delete_file.format),
        content,
    )
    .with_size_bytes(delete_file.file_size_bytes)
    .with_read_capability(read_capability(&delete_file.location));
    if let Some(referenced) = delete_file.referenced_data_file.as_ref() {
        root = root.with_referenced_data_file(referenced.clone());
    }
    if !delete_file.equality_field_ids.is_empty() {
        root = root.with_equality_ids(delete_file.equality_field_ids.iter().copied());
    }
    root
}

fn root_partition_value(
    value: &nexus_iceberg::IcebergPartitionValue,
    schema: &IcebergSchema,
) -> Option<TableFormatPartitionValue> {
    let (_, arrow, _) = schema_field(schema, value.source_field_id)?;
    let (_, name, nullable) = schema_field(schema, value.source_field_id)
        .map(|(field, arrow, _)| (field.field_id, field.name.clone(), arrow.is_nullable()))?;
    Some(TableFormatPartitionValue::new(
        name,
        canonical_data_type(arrow.data_type()),
        nullable,
        root_partition_transform(value.transform),
        root_scalar(&value.value),
    ))
}

fn root_partition_transform(transform: IcebergPartitionTransform) -> TableFormatPartitionTransform {
    match transform {
        IcebergPartitionTransform::Identity => TableFormatPartitionTransform::Identity,
        IcebergPartitionTransform::Bucket(buckets) => TableFormatPartitionTransform::Bucket { buckets },
        IcebergPartitionTransform::Truncate(width) => TableFormatPartitionTransform::Truncate { width },
        IcebergPartitionTransform::Year => TableFormatPartitionTransform::Year,
        IcebergPartitionTransform::Month => TableFormatPartitionTransform::Month,
        IcebergPartitionTransform::Day => TableFormatPartitionTransform::Day,
        IcebergPartitionTransform::Hour => TableFormatPartitionTransform::Hour,
        IcebergPartitionTransform::Void => TableFormatPartitionTransform::Unknown("void".to_string()),
        IcebergPartitionTransform::Unknown => TableFormatPartitionTransform::Unknown("unknown".to_string()),
    }
}

fn root_scalar(value: &IcebergScalarValue) -> TableFormatScalarValue {
    match value {
        IcebergScalarValue::Null => TableFormatScalarValue::Null,
        IcebergScalarValue::Boolean(value) => TableFormatScalarValue::Boolean(*value),
        IcebergScalarValue::Int(value) => TableFormatScalarValue::Int32(*value),
        IcebergScalarValue::Long(value) => TableFormatScalarValue::Int64(*value),
        IcebergScalarValue::Float { bits } => TableFormatScalarValue::Float32(f32::from_bits(*bits)),
        IcebergScalarValue::Double { bits } => TableFormatScalarValue::Float64(f64::from_bits(*bits)),
        IcebergScalarValue::Decimal { unscaled, .. } => TableFormatScalarValue::Int128(*unscaled),
        IcebergScalarValue::String(value) => TableFormatScalarValue::Utf8(value.clone()),
        IcebergScalarValue::Date(value) => TableFormatScalarValue::Date32(*value),
        IcebergScalarValue::Time(value) => TableFormatScalarValue::Time64Micros(*value),
        IcebergScalarValue::TimestampMicros(value) => TableFormatScalarValue::TimestampMicros(*value),
        IcebergScalarValue::TimestampNanos(value) => TableFormatScalarValue::TimestampNanos(*value),
        IcebergScalarValue::Uuid(value) => TableFormatScalarValue::Uuid(*value),
        IcebergScalarValue::Binary(value) => TableFormatScalarValue::Binary(value.clone()),
    }
}

fn iceberg_scalar_is_not_native_materializable(value: &IcebergScalarValue) -> bool {
    matches!(
        value,
        IcebergScalarValue::Time(_)
            | IcebergScalarValue::TimestampMicros(_)
            | IcebergScalarValue::TimestampNanos(_)
            | IcebergScalarValue::Uuid(_)
            | IcebergScalarValue::Binary(_)
    )
}

fn logical_schema_for_request(schema: &IcebergSchema, request: &IcebergScanRequest) -> Result<SchemaRef> {
    let fields = match &request.projection {
        nexus_iceberg::IcebergProjection::AllColumns => {
            schema.arrow_schema.fields().iter().map(|field| canonical_field(field.as_ref())).collect::<Vec<_>>()
        },
        nexus_iceberg::IcebergProjection::Columns(bindings) => bindings
            .iter()
            .map(|binding| {
                let (_, arrow, _) = schema_field(schema, binding.field_id).ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        CANDIDATE_CONVERSION_OPERATION,
                        "Iceberg projection references a field absent from its frozen schema",
                    )
                    .with_fact("field_id", binding.field_id)
                })?;
                Ok(Arc::new(Field::new(
                    binding.output_name.clone(),
                    canonical_data_type(arrow.data_type()),
                    arrow.is_nullable(),
                )))
            })
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(Arc::new(Schema::new(fields)))
}

fn canonical_schema(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(schema.fields().iter().map(|field| canonical_field(field.as_ref())).collect::<Vec<_>>()))
}

fn canonical_field(field: &Field) -> Arc<Field> {
    Arc::new(Field::new(field.name(), canonical_data_type(field.data_type()), field.is_nullable()))
}

/// Removes connector-specific Arrow field metadata from a DataFusion-facing
/// Iceberg schema, including every nested field held inside its data types.
///
/// The same transformation is used for logical schemas, native source schemas,
/// and column bindings so the DataFusion wrapper and lowered native plan expose
/// exactly one output schema.
fn canonical_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::List(field) => DataType::List(canonical_field(field)),
        DataType::ListView(field) => DataType::ListView(canonical_field(field)),
        DataType::FixedSizeList(field, length) => DataType::FixedSizeList(canonical_field(field), *length),
        DataType::LargeList(field) => DataType::LargeList(canonical_field(field)),
        DataType::LargeListView(field) => DataType::LargeListView(canonical_field(field)),
        DataType::Struct(fields) => {
            DataType::Struct(Fields::from(fields.iter().map(|field| canonical_field(field)).collect::<Vec<_>>()))
        },
        DataType::Union(fields, mode) => {
            let (type_ids, fields): (Vec<_>, Vec<_>) =
                fields.iter().map(|(type_id, field)| (type_id, canonical_field(field))).unzip();
            let fields = UnionFields::try_new(type_ids, fields)
                .expect("a valid Arrow union remains valid after canonicalizing its child fields");
            DataType::Union(fields, *mode)
        },
        DataType::Dictionary(key, value) => {
            DataType::Dictionary(Box::new(canonical_data_type(key)), Box::new(canonical_data_type(value)))
        },
        DataType::Map(field, keys_sorted) => DataType::Map(canonical_field(field), *keys_sorted),
        DataType::RunEndEncoded(run_ends, values) => {
            DataType::RunEndEncoded(canonical_field(run_ends), canonical_field(values))
        },
        _ => data_type.clone(),
    }
}

fn schema_field(schema: &IcebergSchema, field_id: i32) -> Option<(&IcebergSchemaField, &Field, bool)> {
    schema.fields.iter().zip(schema.arrow_schema.fields()).find_map(|(field, arrow)| {
        (field.field_id == field_id).then_some((field, arrow.as_ref(), true)).or_else(|| {
            find_nested_schema_field(field, arrow.as_ref(), field_id).map(|(field, arrow)| (field, arrow, false))
        })
    })
}

fn find_nested_schema_field<'a>(
    field: &'a IcebergSchemaField,
    arrow: &'a Field,
    field_id: i32,
) -> Option<(&'a IcebergSchemaField, &'a Field)> {
    let children = arrow_children(arrow.data_type());
    field.children.iter().zip(children).find_map(|(child, arrow)| {
        (child.field_id == field_id)
            .then_some((child, arrow))
            .or_else(|| find_nested_schema_field(child, arrow, field_id))
    })
}

fn arrow_children(data_type: &DataType) -> Vec<&Field> {
    match data_type {
        DataType::Struct(fields) => fields.iter().map(|field| field.as_ref()).collect(),
        DataType::List(field)
        | DataType::ListView(field)
        | DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field],
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.as_ref()).collect(),
        _ => Vec::new(),
    }
}

fn name_mapping_names_for_field_id(mapping: Option<&IcebergNameMapping>, field_id: i32) -> Option<Vec<String>> {
    let mut names = Vec::new();
    for field in mapping.into_iter().flat_map(|mapping| &mapping.fields) {
        collect_name_mapping_names(field, field_id, &mut names);
    }
    (!names.is_empty()).then_some(names)
}

fn collect_name_mapping_names(field: &IcebergNameMappingField, field_id: i32, names: &mut Vec<String>) {
    if field.field_id == Some(field_id) {
        for name in &field.names {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    for child in &field.fields {
        collect_name_mapping_names(child, field_id, names);
    }
}

fn root_file_format(format: IcebergFileFormat) -> TableFormatDataFileFormat {
    match format {
        IcebergFileFormat::Parquet => TableFormatDataFileFormat::Parquet,
        IcebergFileFormat::Orc => TableFormatDataFileFormat::Orc,
        IcebergFileFormat::Avro => TableFormatDataFileFormat::Avro,
        IcebergFileFormat::Puffin => TableFormatDataFileFormat::Unknown("puffin".to_string()),
    }
}

fn read_capability(location: &str) -> TableFormatReadCapability {
    let scheme = location.split_once(':').map(|(scheme, _)| scheme);
    match scheme {
        None | Some("file") => TableFormatReadCapability::local_file_system(),
        Some(scheme) if !scheme.is_empty() => TableFormatReadCapability::object_store(scheme.to_ascii_lowercase()),
        Some(_) => TableFormatReadCapability::unsupported("empty_uri_scheme"),
    }
}

/// Converts one normalized leaf error exactly once at the root boundary.
///
/// The leaf deliberately exposes no vendor source chain. We preserve its
/// machine classification and typed safe facts without attaching any raw
/// upstream error source to the root error.
pub(crate) fn root_error_from_iceberg(
    source: IcebergError,
    operation: &'static str,
    message: impl Into<String>,
) -> Error {
    let mut error = Error::new(root_error_code(&source), operation, message)
        .with_log_safe_fact("iceberg_error_code", source.code.as_str())
        .with_log_safe_fact("iceberg_error_status", iceberg_error_status_code(source.status))
        .with_log_safe_fact("iceberg_operation", source.operation.as_str())
        .with_fact("iceberg_message", source.sanitized_message);
    let context = source.typed_context.as_ref();
    if let Some(kind) = context.catalog_kind {
        error = error.with_log_safe_fact(
            "iceberg_catalog_kind",
            match kind {
                nexus_iceberg::IcebergCatalogErrorKind::Glue => "glue",
                nexus_iceberg::IcebergCatalogErrorKind::Rest => "rest",
            },
        );
    }
    if let Some(catalog_handle) = context.catalog_handle.as_ref() {
        error = error.with_fact("iceberg_catalog_handle", catalog_handle);
    }
    if let Some(table_identifier) = context.table_identifier.as_ref() {
        error = error.with_fact("iceberg_table_identifier", table_identifier);
    }
    if let Some(snapshot_id) = context.snapshot_id {
        error = error.with_fact("snapshot_id", snapshot_id);
    }
    if let Some(schema_id) = context.schema_id {
        error = error.with_fact("iceberg_schema_id", schema_id);
    }
    if let Some(field_id) = context.field_id {
        error = error.with_fact("field_id", field_id);
    }
    if let Some(file_count) = context.file_count {
        error = error.with_fact("iceberg_file_count", file_count);
    }
    error
}

fn root_error_code(error: &IcebergError) -> ErrorCode {
    if error.status == IcebergErrorStatus::Temporary {
        return ErrorCode::SourceTemporaryFailure;
    }
    match error.code {
        IcebergErrorCode::Authentication => ErrorCode::SourceUnauthenticated,
        IcebergErrorCode::Access => ErrorCode::SourceAccessDenied,
        IcebergErrorCode::NotFound => ErrorCode::SourceObjectNotFound,
        IcebergErrorCode::TemporaryDependency | IcebergErrorCode::Timeout => ErrorCode::SourceTemporaryFailure,
        IcebergErrorCode::Cancelled => ErrorCode::Cancelled,
        IcebergErrorCode::InvalidMetadata => ErrorCode::InvalidExecutionPlan,
        IcebergErrorCode::UnsupportedSchema => ErrorCode::UnsupportedType,
        IcebergErrorCode::Internal => ErrorCode::SourceDependencyFailure,
    }
}

const fn iceberg_error_status_code(status: IcebergErrorStatus) -> &'static str {
    match status {
        IcebergErrorStatus::Permanent => "permanent",
        IcebergErrorStatus::Temporary => "temporary",
        IcebergErrorStatus::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use nexus_iceberg::{
        IcebergDataFile, IcebergErrorContext, IcebergFileRange, IcebergFileSequence, IcebergOperation,
        IcebergPartitionValue, IcebergPredicateEnforcement, IcebergPredicateIntent, IcebergProjection,
        IcebergResidualPredicate, IcebergScanCandidate, IcebergSchemaField, IcebergSourceLimit, IcebergTableIdentifier,
        ResolvedIcebergSnapshot, ResolvedIcebergTableFacts,
    };

    use super::*;

    #[test]
    fn leaf_error_conversion_preserves_classification_and_redacts_at_the_root_boundary() {
        let leaf = IcebergError::new(
            IcebergErrorCode::Authentication,
            IcebergErrorStatus::Permanent,
            IcebergOperation::CatalogConnect,
            "catalog https://user:password@example.invalid token=secret",
        )
        .with_context(IcebergErrorContext { snapshot_id: Some(42), ..Default::default() });

        let error = root_error_from_iceberg(leaf, "datafusion_nexus.test.iceberg", "Iceberg catalog connection failed");

        assert_eq!(error.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(error.status(), crate::error::ErrorStatus::Permanent);
        assert_eq!(error.fact_value("iceberg_error_code"), Some("authentication"));
        assert_eq!(error.fact_value("snapshot_id"), Some("42"));
        assert!(error.fact_value("iceberg_message").is_some_and(|message| !message.contains("password")));
        assert!(error.fact_value("iceberg_message").is_some_and(|message| !message.contains("secret")));
        assert!(error.source_error().is_none(), "the root boundary must not retain a leaf/vendor source chain");
    }

    #[test]
    fn wrapped_vendor_source_auth_and_access_errors_keep_root_codes() {
        use iceberg::{Error as VendorIcebergError, ErrorKind};

        let auth = VendorIcebergError::new(ErrorKind::Unexpected, "catalog request failed")
            .with_source(std::io::Error::other("ExpiredTokenException: token=not-for-logs"));
        let access = VendorIcebergError::new(ErrorKind::Unexpected, "catalog request failed")
            .with_source(std::io::Error::other("PermissionDenied: forbidden"));

        let auth = root_error_from_iceberg(
            IcebergError::from_iceberg_error(IcebergOperation::CatalogConnect, auth),
            "datafusion_nexus.test.iceberg",
            "Iceberg catalog connection failed",
        );
        let access = root_error_from_iceberg(
            IcebergError::from_iceberg_error(IcebergOperation::CatalogConnect, access),
            "datafusion_nexus.test.iceberg",
            "Iceberg catalog connection failed",
        );

        assert_eq!(auth.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(access.code(), ErrorCode::SourceAccessDenied);
        assert!(auth.fact_value("iceberg_message").is_some_and(|message| !message.contains("not-for-logs")));
        assert!(auth.source_error().is_none());
        assert!(access.source_error().is_none());
    }

    #[test]
    fn candidate_conversion_retains_exact_leaf_semantics_and_rejects_only_native_acceleration() {
        let schema = IcebergSchema::new(
            7,
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("part", DataType::FixedSizeBinary(16), true),
            ])),
            vec![IcebergSchemaField::new(1, "id", true, "long"), IcebergSchemaField::new(2, "part", false, "uuid")],
        )
        .unwrap();
        let candidate = IcebergScanCandidate::try_new(
            ResolvedIcebergTableFacts {
                table: IcebergTableIdentifier::new(vec!["analytics".to_string()], "events").unwrap(),
                snapshot: ResolvedIcebergSnapshot { snapshot_id: Some(42), sequence_number: Some(11), schema_id: 7 },
                schema,
            },
            IcebergScanRequest {
                projection: IcebergProjection::Columns(vec![IcebergColumnBinding {
                    field_id: 1,
                    output_name: "id".to_string(),
                    source: IcebergColumnSource::MissingWithDefault,
                    role: IcebergColumnRole::Projected,
                    missing_default: Some(IcebergDefaultValue::Scalar(IcebergScalarValue::Uuid([8; 16]))),
                }]),
                filter_bindings: Vec::new(),
                predicate: IcebergPredicateIntent {
                    enforcement: IcebergPredicateEnforcement::ConsumerMustEnforce,
                    residual: Some(IcebergResidualPredicate { canonical: "field(1) > 0".to_string() }),
                },
                iceberg_predicate: None,
                source_limit: IcebergSourceLimit { row_limit: Some(10), file_limit: Some(1) },
                pruning: Default::default(),
            },
            vec![IcebergDataFile {
                location: "s3://warehouse/events/data.parquet".to_string(),
                format: IcebergFileFormat::Parquet,
                file_size_bytes: 1024,
                range: IcebergFileRange { start: 128, length: 896 },
                record_count: Some(99),
                sequence: IcebergFileSequence { data_sequence_number: Some(9), file_sequence_number: Some(10) },
                encryption_key_metadata_present: false,
                partition_values: vec![IcebergPartitionValue {
                    source_field_id: 2,
                    partition_field_id: 1002,
                    transform: IcebergPartitionTransform::Identity,
                    value: IcebergScalarValue::Uuid([7; 16]),
                }],
                delete_files: Vec::new(),
            }],
            None,
        )
        .unwrap();

        let plan = iceberg_scan_candidate_to_table_format_scan_plan(candidate, Some("rest")).unwrap();
        let retained = plan.iceberg_candidate().expect("the exact leaf candidate is retained");

        assert_eq!(retained.resolved_table.snapshot.snapshot_id, Some(42));
        assert_eq!(retained.request.predicate.enforcement, IcebergPredicateEnforcement::ConsumerMustEnforce);
        assert_eq!(
            retained.request.predicate.residual.as_ref().map(|value| value.canonical.as_str()),
            Some("field(1) > 0")
        );
        assert_eq!(retained.data_files[0].range, IcebergFileRange { start: 128, length: 896 });
        assert_eq!(plan.snapshot_id.as_deref(), Some("42"));
        assert!(
            !plan.unsupported_features.iter().any(|feature| {
                feature.code == TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported
            })
        );
        assert!(
            plan.unsupported_features
                .iter()
                .any(|feature| { feature.code == TableFormatUnsupportedFeatureCode::IcebergFileRangeUnsupported })
        );
        assert!(
            plan.unsupported_features
                .iter()
                .any(|feature| { feature.code == TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported })
        );
        assert!(plan.unsupported_features.iter().any(|feature| {
            feature.code == TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported
                && feature.message.contains("schema-evolution default")
        }));
        assert_eq!(plan.data_files[0].partition_values[0].value, TableFormatScalarValue::Uuid([7; 16]));
    }

    #[test]
    fn candidate_conversion_retains_materializable_schema_defaults_for_missing_old_file_columns() {
        let schema =
            IcebergSchema::new(
                7,
                Arc::new(Schema::new(vec![Field::new("region", DataType::Utf8, true)])),
                vec![IcebergSchemaField::new(1, "region", false, "string").with_initial_default(
                    IcebergDefaultValue::Scalar(IcebergScalarValue::String("unknown".to_string())),
                )],
            )
            .unwrap();
        let candidate = IcebergScanCandidate::try_new(
            ResolvedIcebergTableFacts {
                table: IcebergTableIdentifier::new(vec!["analytics".to_string()], "events").unwrap(),
                snapshot: ResolvedIcebergSnapshot { snapshot_id: Some(42), sequence_number: Some(11), schema_id: 7 },
                schema,
            },
            IcebergScanRequest::default(),
            Vec::new(),
            None,
        )
        .unwrap();

        let plan = iceberg_scan_candidate_to_table_format_scan_plan(candidate, Some("rest")).unwrap();

        assert_eq!(
            plan.projected_columns[0].missing_default_value,
            Some(TableFormatScalarValue::Utf8("unknown".to_string()))
        );
        assert!(
            !plan
                .unsupported_features
                .iter()
                .any(|feature| { feature.code == TableFormatUnsupportedFeatureCode::IcebergScalarValueUnsupported })
        );
    }
}
