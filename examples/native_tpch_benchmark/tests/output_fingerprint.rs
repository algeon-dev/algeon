use std::sync::Arc;

use arrow::{
    array::{Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use super::*;

#[test]
fn output_fingerprint_is_independent_of_record_batch_boundaries() {
    let single_batch = vec![int_batch(&[1, 2, 3, 4])];
    let split_batches = vec![int_batch(&[1, 2]), int_batch(&[3, 4])];
    let different_rows = vec![int_batch(&[1, 2, 4, 3])];
    let schema = single_batch[0].schema();

    let single_fingerprint = validation::fingerprint_batches(&schema, &single_batch).unwrap();
    let split_fingerprint = validation::fingerprint_batches(&schema, &split_batches).unwrap();
    let different_fingerprint = validation::fingerprint_batches(&schema, &different_rows).unwrap();

    assert_eq!(single_fingerprint, split_fingerprint);
    assert_ne!(single_fingerprint, different_fingerprint);
    assert_eq!(
        validation::parity_status(
            Some(&execution::QueryMeasurement {
                output_batches: single_batch.len(),
                output_rows: 4,
                output_bytes: 0,
                output_fingerprint: single_fingerprint,
            }),
            Some(&execution::QueryMeasurement {
                output_batches: split_batches.len(),
                output_rows: 4,
                output_bytes: 0,
                output_fingerprint: split_fingerprint,
            }),
        ),
        validation::ParityStatus::Matched
    );
}

#[test]
fn output_fingerprint_includes_schema_nullability_and_metadata() {
    let nullable = vec![int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[1, 2])];
    let non_nullable = vec![int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[1, 2])];
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("logical_source".to_string(), "cpu".to_string());
    let schema_metadata =
        vec![int_batch_with_field(Field::new("value", DataType::Int32, false), Some(metadata), &[1, 2])];

    let nullable_fingerprint = validation::fingerprint_batches(&nullable[0].schema(), &nullable).unwrap();
    let non_nullable_fingerprint = validation::fingerprint_batches(&non_nullable[0].schema(), &non_nullable).unwrap();
    let schema_metadata_fingerprint =
        validation::fingerprint_batches(&schema_metadata[0].schema(), &schema_metadata).unwrap();

    assert_ne!(nullable_fingerprint, non_nullable_fingerprint);
    assert_ne!(non_nullable_fingerprint, schema_metadata_fingerprint);
}

#[test]
fn output_fingerprint_includes_schema_for_empty_outputs() {
    let nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[]).schema();
    let non_nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[]).schema();

    assert_ne!(
        validation::fingerprint_batches(&nullable_schema, &[]).unwrap(),
        validation::fingerprint_batches(&non_nullable_schema, &[]).unwrap()
    );
}

#[test]
fn output_fingerprint_distinguishes_null_from_empty_string() {
    let null_string = vec![string_batch(&[None])];
    let empty_string = vec![string_batch(&[Some("")])];

    assert_ne!(
        validation::fingerprint_batches(&null_string[0].schema(), &null_string).unwrap(),
        validation::fingerprint_batches(&empty_string[0].schema(), &empty_string).unwrap()
    );
}

fn int_batch(values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

fn int_batch_with_field(
    field: Field,
    metadata: Option<std::collections::HashMap<String, String>>,
    values: &[i32],
) -> RecordBatch {
    let schema = match metadata {
        Some(metadata) => Schema::new_with_metadata(vec![field], metadata),
        None => Schema::new(vec![field]),
    };
    let schema = Arc::new(schema);
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

fn string_batch(values: &[Option<&str>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(values.to_vec()))]).unwrap()
}
