use arrow::datatypes::{DataType, SchemaRef};
use serde_json::{Value, json};

use crate::cugraph_sql::{CugraphAlgorithm, stable_result_schema};

pub(super) fn result_schemas_json(algorithm: CugraphAlgorithm) -> String {
    let (default_schema, schemas) = if algorithm == CugraphAlgorithm::Bfs {
        (
            "raw",
            json!({
                "raw": bfs_raw_mode_schema(false),
                "normalized": bfs_normalized_mode_schema(false),
                "raw_with_target_info": bfs_raw_mode_schema(true),
                "normalized_with_target_info": bfs_normalized_mode_schema(true),
                "path": bfs_path_mode_schema()
            }),
        )
    } else {
        let mut schema = schema_json_value(&stable_result_schema(algorithm));
        if let Some(output_columns) = algorithm.definition().logical_string_output_columns()
            && let Some(fields) = schema.get_mut("fields").and_then(Value::as_array_mut)
        {
            for field in fields {
                if output_columns.iter().any(|column| field["name"] == column.name) {
                    field["data_type"] = json!("Int64|Utf8");
                }
            }
        }
        ("default", json!({"default": schema}))
    };
    json!({
        "default_schema": default_schema,
        "schemas": schemas,
        "call_specific_authority": "gpu_validate_call",
        "note": "Generic registry schemas; run gpu_validate_call for the concrete table-specific output schema."
    })
    .to_string()
}

fn schema_json_value(schema: &SchemaRef) -> Value {
    json!({
        "fields": schema
            .fields()
            .iter()
            .map(|field| {
                json!({
                    "name": field.name(),
                    "data_type": data_type_label(field.data_type()),
                    "nullable": field.is_nullable(),
                })
            })
            .collect::<Vec<_>>()
    })
}

fn bfs_raw_mode_schema(with_target_info: bool) -> Value {
    json!({"fields": bfs_raw_mode_fields(with_target_info)})
}

fn bfs_raw_mode_fields(with_target_info: bool) -> Vec<Value> {
    let mut fields = vec![
        metadata_field("vertex", "Int64|Utf8", false),
        metadata_field("distance", "Int64", false),
        metadata_field("predecessor", "Int64|Utf8", true),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    fields
}

fn bfs_normalized_mode_schema(with_target_info: bool) -> Value {
    let mut fields = vec![
        metadata_field("vertex", "Int64|Utf8", false),
        metadata_field("distance", "Int64", true),
        metadata_field("predecessor", "Int64|Utf8", true),
        metadata_field("reachable", "Boolean", false),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    json!({"fields": fields})
}

fn append_bfs_target_info_fields(fields: &mut Vec<Value>, with_target_info: bool) {
    if with_target_info {
        fields.push(metadata_field("target_found", "Boolean", false));
        fields.push(metadata_field("target_distance", "Int64", true));
    }
}

fn bfs_path_mode_schema() -> Value {
    json!({
        "fields": [
            metadata_field("path_index", "Int64", false),
            metadata_field("source", "Int64|Utf8", false),
            metadata_field("target", "Int64|Utf8", false),
            metadata_field("vertex", "Int64|Utf8", false),
            metadata_field("distance", "Int64", false)
        ]
    })
}

fn metadata_field(name: &str, data_type: &str, nullable: bool) -> Value {
    json!({"name": name, "data_type": data_type, "nullable": nullable})
}

fn data_type_label(data_type: &DataType) -> String {
    match data_type {
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Utf8 => "Utf8".to_string(),
        DataType::LargeUtf8 => "LargeUtf8".to_string(),
        DataType::Utf8View => "Utf8View".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        other => format!("{other:?}"),
    }
}
