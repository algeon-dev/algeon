use std::sync::Arc;

use arrow::{
    array::{Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::SessionContext;

use super::*;

#[test]
fn output_fingerprint_is_independent_of_record_batch_boundaries() {
    let single_batch = vec![int_batch(&[1, 2, 3, 4])];
    let split_batches = vec![int_batch(&[1, 2]), int_batch(&[3, 4])];
    let different_rows = vec![int_batch(&[1, 2, 4, 3])];
    let schema = single_batch[0].schema();

    let single_fingerprint = validation::fingerprint_batches(&schema, &[0], &single_batch).unwrap();
    let split_fingerprint = validation::fingerprint_batches(&schema, &[0], &split_batches).unwrap();
    let different_fingerprint = validation::fingerprint_batches(&schema, &[0], &different_rows).unwrap();

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

    let nullable_fingerprint = validation::fingerprint_batches(&nullable[0].schema(), &[], &nullable).unwrap();
    let non_nullable_fingerprint =
        validation::fingerprint_batches(&non_nullable[0].schema(), &[], &non_nullable).unwrap();
    let schema_metadata_fingerprint =
        validation::fingerprint_batches(&schema_metadata[0].schema(), &[], &schema_metadata).unwrap();

    assert_ne!(nullable_fingerprint, non_nullable_fingerprint);
    assert_ne!(non_nullable_fingerprint, schema_metadata_fingerprint);
}

#[test]
fn output_fingerprint_includes_schema_for_empty_outputs() {
    let nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[]).schema();
    let non_nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[]).schema();

    assert_ne!(
        validation::fingerprint_batches(&nullable_schema, &[], &[]).unwrap(),
        validation::fingerprint_batches(&non_nullable_schema, &[], &[]).unwrap()
    );
}

#[test]
fn output_fingerprint_distinguishes_null_from_empty_string() {
    let null_string = vec![string_batch(&[None])];
    let empty_string = vec![string_batch(&[Some("")])];

    assert_ne!(
        validation::fingerprint_batches(&null_string[0].schema(), &[], &null_string).unwrap(),
        validation::fingerprint_batches(&empty_string[0].schema(), &[], &empty_string).unwrap()
    );
}

#[test]
fn output_fingerprint_allows_reordering_within_order_by_ties() {
    let cpu = key_value_fingerprint(&[0], &[(3, "a"), (3, "b"), (1, "c")]);

    assert_eq!(cpu, key_value_fingerprint(&[0], &[(3, "b"), (3, "a"), (1, "c")]));
    assert_eq!(
        validation::parity_status(
            Some(&measurement(cpu.clone())),
            Some(&measurement(key_value_fingerprint(&[0], &[(3, "b"), (3, "a"), (1, "c")]))),
        ),
        validation::ParityStatus::Matched
    );
    // Moving a row across a key change breaks the ORDER BY sequence.
    assert_ne!(cpu, key_value_fingerprint(&[0], &[(1, "c"), (3, "a"), (3, "b")]));
    // Same key sequence, different rows.
    assert_ne!(cpu, key_value_fingerprint(&[0], &[(3, "a"), (3, "a"), (1, "c")]));
    assert_ne!(cpu, key_value_fingerprint(&[0], &[(3, "c"), (3, "b"), (1, "a")]));
    // Keys covering every column fix the whole row sequence.
    assert_ne!(
        key_value_fingerprint(&[0, 1], &[(3, "a"), (3, "b"), (1, "c")]),
        key_value_fingerprint(&[0, 1], &[(3, "b"), (3, "a"), (1, "c")])
    );
}

#[test]
fn output_fingerprint_without_order_by_compares_row_multisets() {
    let cpu = key_value_fingerprint(&[], &[(3, "a"), (3, "b"), (1, "c")]);

    assert_eq!(cpu, key_value_fingerprint(&[], &[(1, "c"), (3, "b"), (3, "a")]));
    assert_ne!(cpu, key_value_fingerprint(&[], &[(1, "c"), (3, "b"), (3, "b")]));
    assert_ne!(cpu, key_value_fingerprint(&[], &[(1, "c"), (3, "b")]));
}

#[tokio::test]
async fn order_key_columns_follow_the_top_level_order_by() {
    let ctx = SessionContext::new();
    ctx.register_batch("t", key_value_batch(&[(1, "a")])).unwrap();

    // TPC-H Q11 shape: ORDER BY an aggregate alias.
    assert_eq!(order_keys(&ctx, "select k, count(v) as value from t group by k order by value desc").await, [1]);
    assert_eq!(order_keys(&ctx, "select k, v from t order by v desc, k limit 3").await, [1, 0]);
    assert_eq!(order_keys(&ctx, "select k, v from t").await, [0_usize; 0]);
    assert_eq!(order_keys(&ctx, "select * from (select k, v from t order by v) s").await, [0_usize; 0]);
    // A hidden key is projected away, so every output column becomes a key.
    assert_eq!(order_keys(&ctx, "select v from t order by k").await, [0]);
    assert_eq!(order_keys(&ctx, "select k, v from t order by k + 1").await, [0, 1]);
}

async fn order_keys(ctx: &SessionContext, sql: &str) -> Vec<usize> {
    validation::order_key_columns(ctx.sql(sql).await.unwrap().logical_plan())
}

fn key_value_fingerprint(order_key_columns: &[usize], rows: &[(i32, &str)]) -> String {
    let batch = key_value_batch(rows);
    validation::fingerprint_batches(&batch.schema(), order_key_columns, &[batch]).unwrap()
}

fn key_value_batch(rows: &[(i32, &str)]) -> RecordBatch {
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Utf8, false)]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(rows.iter().map(|(key, _)| *key).collect::<Vec<_>>())),
            Arc::new(StringArray::from(rows.iter().map(|(_, value)| *value).collect::<Vec<_>>())),
        ],
    )
    .unwrap()
}

fn measurement(output_fingerprint: String) -> execution::QueryMeasurement {
    execution::QueryMeasurement { output_batches: 1, output_rows: 3, output_bytes: 0, output_fingerprint }
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
