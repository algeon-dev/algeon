use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// Stable classifier schema: the most probable class and its probability. The
/// ID type belongs to the predict relation.
pub fn ml_classification_output_schema(id_type: DataType) -> SchemaRef {
    prediction_schema(
        id_type,
        [Field::new("prediction", DataType::Int32, false), Field::new("probability", DataType::Float32, false)],
    )
}

/// Stable regression schema. The ID type belongs to the predict relation.
pub fn ml_regression_output_schema(id_type: DataType) -> SchemaRef {
    prediction_schema(id_type, [Field::new("prediction", DataType::Float32, false)])
}

fn prediction_schema<const N: usize>(id_type: DataType, outputs: [Field; N]) -> SchemaRef {
    let mut fields = vec![Field::new("row_ordinal", DataType::UInt64, false), Field::new("id", id_type, false)];
    fields.extend(outputs);
    Arc::new(Schema::new(fields))
}

/// Stable clustering result schema for the input ID type.
pub fn ml_hdbscan_output_schema(id_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("row_ordinal", DataType::UInt64, false),
        Field::new("id", id_type, false),
        Field::new("cluster_id", DataType::Int64, false),
        Field::new("probability", DataType::Float32, false),
    ]))
}

/// Stable manifold embedding schema for the input ID type and dimension count.
pub fn ml_embedding_output_schema(id_type: DataType, n_components: usize) -> SchemaRef {
    let mut fields = vec![Field::new("row_ordinal", DataType::UInt64, false), Field::new("id", id_type, false)];
    fields.extend((0..n_components).map(|index| Field::new(format!("dim_{index}"), DataType::Float32, false)));
    Arc::new(Schema::new(fields))
}
