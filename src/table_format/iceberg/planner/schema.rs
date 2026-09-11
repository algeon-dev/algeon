use std::sync::Arc;

use super::{
    DataType, Error, ErrorCode, Field, FieldRef, Fields, NestedField, Result, Schema, SchemaRef,
    TASK_CONVERSION_OPERATION, TableFormat, Type, UnionFields, VendorSchema,
};

/// Converts an Iceberg schema into the metadata-free Arrow schema exposed to DataFusion.
///
/// `iceberg-rust` includes Parquet/Iceberg field-id metadata on Arrow fields.
/// DataFusion logical projections and aliases often drop that metadata, and
/// its physical/logical schema check treats the metadata difference as a hard
/// mismatch. Keep Iceberg field ids in
/// [`crate::table_format::TableFormatColumnMapping::field_id`] and expose
/// metadata-free Arrow schemas at DataFusion boundaries.
pub fn iceberg_schema_to_datafusion_arrow_schema(schema: &VendorSchema) -> Result<SchemaRef> {
    let arrow_schema = iceberg::arrow::schema_to_arrow_schema(schema).map_err(|error| {
        Error::new(
            ErrorCode::UnsupportedType,
            TASK_CONVERSION_OPERATION,
            "failed to convert Iceberg schema to Arrow schema",
        )
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(error)
    })?;
    Ok(Arc::new(strip_arrow_schema_metadata(arrow_schema)))
}

pub(super) fn strip_arrow_schema_metadata(schema: Schema) -> Schema {
    Schema::new(schema.fields().iter().map(strip_arrow_field_metadata).collect::<Vec<_>>())
}

pub(super) fn strip_arrow_field_metadata(field: &FieldRef) -> Field {
    let stripped_type = strip_arrow_data_type_metadata(field.data_type());
    let mut stripped = Field::new(field.name(), stripped_type, field.is_nullable());
    if let Some(dict_is_ordered) = field.dict_is_ordered() {
        stripped = stripped.with_dict_is_ordered(dict_is_ordered);
    }
    stripped
}

pub(super) fn strip_arrow_data_type_metadata(data_type: &DataType) -> DataType {
    match data_type {
        DataType::List(field) => DataType::List(Arc::new(strip_arrow_field_metadata(field))),
        DataType::ListView(field) => DataType::ListView(Arc::new(strip_arrow_field_metadata(field))),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(Arc::new(strip_arrow_field_metadata(field)), *size)
        },
        DataType::LargeList(field) => DataType::LargeList(Arc::new(strip_arrow_field_metadata(field))),
        DataType::LargeListView(field) => DataType::LargeListView(Arc::new(strip_arrow_field_metadata(field))),
        DataType::Struct(fields) => {
            DataType::Struct(Fields::from(fields.iter().map(strip_arrow_field_metadata).collect::<Vec<_>>()))
        },
        DataType::Union(fields, mode) => {
            let stripped_fields = fields.iter().map(|(_, field)| strip_arrow_field_metadata(field)).collect::<Vec<_>>();
            let type_ids = fields.iter().map(|(type_id, _)| type_id).collect::<Vec<_>>();
            DataType::Union(
                UnionFields::try_new(type_ids, stripped_fields)
                    .expect("stripping Arrow metadata preserves valid union type ids"),
                *mode,
            )
        },
        DataType::Dictionary(key_type, value_type) => DataType::Dictionary(
            Box::new(strip_arrow_data_type_metadata(key_type)),
            Box::new(strip_arrow_data_type_metadata(value_type)),
        ),
        DataType::Map(field, sorted) => DataType::Map(Arc::new(strip_arrow_field_metadata(field)), *sorted),
        DataType::RunEndEncoded(run_ends, values) => DataType::RunEndEncoded(
            Arc::new(strip_arrow_field_metadata(run_ends)),
            Arc::new(strip_arrow_field_metadata(values)),
        ),
        other => other.clone(),
    }
}

pub(super) fn iceberg_field_to_arrow_data_type(field: &NestedField) -> Result<DataType> {
    iceberg_type_to_arrow_data_type(field.field_type.as_ref())
}

pub(super) fn iceberg_type_to_arrow_data_type(field_type: &Type) -> Result<DataType> {
    let data_type = iceberg::arrow::type_to_arrow_type(field_type).map_err(|error| {
        Error::new(
            ErrorCode::UnsupportedType,
            TASK_CONVERSION_OPERATION,
            "failed to convert Iceberg field type to Arrow data type",
        )
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(error)
    })?;
    Ok(strip_arrow_data_type_metadata(&data_type))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn strip_arrow_schema_metadata_recurses_and_preserves_dictionary_ordering() {
        let dictionary = Arc::new(
            Field::new("dictionary", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), true)
                .with_dict_is_ordered(true)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "3".to_string())])),
        );
        let item = Arc::new(
            Field::new("item", DataType::Struct(Fields::from(vec![dictionary])), true)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "2".to_string())])),
        );
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("items", DataType::List(item), false)
                    .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "1".to_string())])),
            ],
            HashMap::from([("iceberg.schema".to_string(), "metadata".to_string())]),
        );

        let stripped = strip_arrow_schema_metadata(schema);
        assert!(stripped.metadata().is_empty());
        let items = stripped.field(0);
        assert!(items.metadata().is_empty());
        let DataType::List(item) = items.data_type() else {
            panic!("items field must remain a list");
        };
        assert!(item.metadata().is_empty());
        let DataType::Struct(fields) = item.data_type() else {
            panic!("list item must remain a struct");
        };
        let (_, dictionary) = fields.find("dictionary").expect("nested dictionary field must remain present");
        assert!(dictionary.metadata().is_empty());
        assert_eq!(dictionary.dict_is_ordered(), Some(true));
    }
}
