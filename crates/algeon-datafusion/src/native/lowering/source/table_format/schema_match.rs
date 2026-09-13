use std::{collections::HashMap, path::Path, sync::Arc};

use parquet::{arrow::parquet_to_arrow_schema, file::reader::SerializedFileReader};
use query_engine::datatypes::arrow_field_metadata_matches_except_parquet_field_id;

use super::{
    DataType, Field, FileReader, Schema, SchemaRef, TableFormatColumnMapping, TableFormatDataFile, TableFormatScanPlan,
    table_format_local_path, table_format_native_parquet_file,
};

pub(super) fn table_format_file_schema(scan_plan: &TableFormatScanPlan) -> SchemaRef {
    table_format_local_footer_schema(scan_plan)
        .unwrap_or_else(|| strip_table_format_schema_metadata(&scan_plan.native_source_schema))
}

pub(super) fn table_format_local_footer_schema(scan_plan: &TableFormatScanPlan) -> Option<SchemaRef> {
    scan_plan
        .data_files
        .iter()
        .filter(|file| file.read_capability.is_native_local_parquet_capable())
        .map(|file| table_format_local_path(&file.location))
        .find_map(|path| table_format_local_footer_arrow_schema(path.as_path()))
        .map(strip_table_format_file_schema_metadata)
}

pub(super) fn table_format_local_footer_arrow_schema(path: &Path) -> Option<Schema> {
    SerializedFileReader::try_from(path).ok().and_then(|reader| {
        let metadata = reader.metadata().file_metadata();
        parquet_to_arrow_schema(metadata.schema_descr(), metadata.key_value_metadata()).ok()
    })
}

pub(super) fn table_format_data_file_footer_arrow_schema(file: &TableFormatDataFile) -> Option<Schema> {
    if let Some(schema) = file.footer_schema.as_ref() {
        return Some(schema.as_ref().clone());
    }
    let native_file = table_format_native_parquet_file(file).ok()?;
    table_format_local_footer_arrow_schema(native_file.path.as_path())
}

pub(super) fn strip_table_format_file_schema_metadata(schema: Schema) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), field.is_nullable()))
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn strip_table_format_schema_metadata(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), field.is_nullable()))
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn table_format_arrow_fields_match_except_name(left: &Field, right: &Field) -> bool {
    left.is_nullable() == right.is_nullable()
        && table_format_field_metadata_matches_except_parquet_field_id(left.metadata(), right.metadata())
        && table_format_arrow_data_types_match_except_list_child_names(left.data_type(), right.data_type())
}

pub(super) fn table_format_footer_field_matches_mapping(field: &Field, mapping: &TableFormatColumnMapping) -> bool {
    table_format_footer_nullability_matches_mapping(field.is_nullable(), mapping.nullable)
        && table_format_footer_data_type_matches_mapping(field.data_type(), &mapping.data_type)
}

pub(super) fn table_format_footer_nullability_matches_mapping(footer_nullable: bool, logical_nullable: bool) -> bool {
    !footer_nullable || logical_nullable
}

pub(super) fn table_format_footer_data_type_matches_mapping(footer_type: &DataType, logical_type: &DataType) -> bool {
    match (footer_type, logical_type) {
        (DataType::Timestamp(footer_unit, _), DataType::Timestamp(logical_unit, Some(_)))
            if footer_unit == logical_unit =>
        {
            true
        },
        _ => {
            table_format_arrow_data_types_match_except_list_child_names(footer_type, logical_type)
                || table_format_signed_integer_widening(footer_type, logical_type)
        },
    }
}

pub(super) fn table_format_footer_data_type_needs_logical_source_contract(
    footer_type: &DataType,
    logical_type: &DataType,
) -> bool {
    table_format_signed_integer_widening(footer_type, logical_type)
}

fn table_format_signed_integer_widening(footer_type: &DataType, logical_type: &DataType) -> bool {
    let Some(footer_width) = signed_integer_width(footer_type) else {
        return false;
    };
    let Some(logical_width) = signed_integer_width(logical_type) else {
        return false;
    };
    footer_width < logical_width
}

fn signed_integer_width(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int8 => Some(8),
        DataType::Int16 => Some(16),
        DataType::Int32 => Some(32),
        DataType::Int64 => Some(64),
        _ => None,
    }
}

pub(super) fn table_format_arrow_data_types_match_except_list_child_names(left: &DataType, right: &DataType) -> bool {
    match (left, right) {
        (DataType::List(left_field), DataType::List(right_field))
        | (DataType::LargeList(left_field), DataType::LargeList(right_field)) => {
            table_format_arrow_fields_match_except_name(left_field.as_ref(), right_field.as_ref())
        },
        (
            DataType::FixedSizeList(left_field, left_dimension),
            DataType::FixedSizeList(right_field, right_dimension),
        ) => {
            left_dimension == right_dimension
                && table_format_arrow_fields_match_except_name(left_field.as_ref(), right_field.as_ref())
        },
        _ => left == right,
    }
}

pub(super) fn table_format_field_metadata_matches_except_parquet_field_id(
    left: &HashMap<String, String>,
    right: &HashMap<String, String>,
) -> bool {
    arrow_field_metadata_matches_except_parquet_field_id(left, right)
}
