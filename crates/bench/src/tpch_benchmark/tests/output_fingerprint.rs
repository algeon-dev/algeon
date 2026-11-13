use std::sync::Arc;

use algeon_datafusion::native_report::bench_support::{ParityRowOrdering, ParitySortKey};
use arrow::{
    array::{ArrayRef, Decimal128Array, Float64Array, Int32Array, StringArray, Time64MicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::prelude::SessionContext;

use super::*;
use crate::tpch_benchmark::config::DEFAULT_PARITY_FLOAT_TOLERANCE;

#[test]
fn output_fingerprint_is_independent_of_record_batch_boundaries() {
    let single_batch = vec![int_batch(&[1, 2, 3, 4])];
    let split_batches = vec![int_batch(&[1, 2]), int_batch(&[3, 4])];
    let different_rows = vec![int_batch(&[1, 2, 4, 3])];
    let schema = single_batch[0].schema();

    let ordering = order_by(&[0]);

    let single_fingerprint = validation::fingerprint_batches(&schema, &ordering, &single_batch).unwrap();
    let split_fingerprint = validation::fingerprint_batches(&schema, &ordering, &split_batches).unwrap();
    let different_fingerprint = validation::fingerprint_batches(&schema, &ordering, &different_rows).unwrap();

    assert_eq!(single_fingerprint, split_fingerprint);
    assert_ne!(single_fingerprint, different_fingerprint);
    assert_eq!(
        parity(measurement(ordering.clone(), single_batch), measurement(ordering, split_batches)),
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

    let nullable_fingerprint =
        validation::fingerprint_batches(&nullable[0].schema(), &ParityRowOrdering::Unordered, &nullable).unwrap();
    let non_nullable_fingerprint =
        validation::fingerprint_batches(&non_nullable[0].schema(), &ParityRowOrdering::Unordered, &non_nullable)
            .unwrap();
    let schema_metadata_fingerprint =
        validation::fingerprint_batches(&schema_metadata[0].schema(), &ParityRowOrdering::Unordered, &schema_metadata)
            .unwrap();

    assert_ne!(nullable_fingerprint, non_nullable_fingerprint);
    assert_ne!(non_nullable_fingerprint, schema_metadata_fingerprint);
}

#[test]
fn output_fingerprint_includes_schema_for_empty_outputs() {
    let nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[]).schema();
    let non_nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[]).schema();

    assert_ne!(
        validation::fingerprint_batches(&nullable_schema, &ParityRowOrdering::Unordered, &[]).unwrap(),
        validation::fingerprint_batches(&non_nullable_schema, &ParityRowOrdering::Unordered, &[]).unwrap()
    );
}

#[test]
fn output_fingerprint_distinguishes_null_from_empty_string() {
    let null_string = vec![string_batch(&[None])];
    let empty_string = vec![string_batch(&[Some("")])];

    assert_ne!(
        validation::fingerprint_batches(&null_string[0].schema(), &ParityRowOrdering::Unordered, &null_string).unwrap(),
        validation::fingerprint_batches(&empty_string[0].schema(), &ParityRowOrdering::Unordered, &empty_string)
            .unwrap()
    );
}

#[test]
fn output_fingerprint_allows_reordering_within_order_by_ties() {
    let cpu = key_value_fingerprint(&[0], &[(3, "a"), (3, "b"), (1, "c")]);
    let cpu_batches = vec![key_value_batch(&[(3, "a"), (3, "b"), (1, "c")])];
    let tied_batches = vec![key_value_batch(&[(3, "b"), (3, "a"), (1, "c")])];
    let crossed_batches = vec![key_value_batch(&[(1, "c"), (3, "a"), (3, "b")])];

    assert_eq!(cpu, key_value_fingerprint(&[0], &[(3, "b"), (3, "a"), (1, "c")]));
    assert_eq!(
        parity(measurement(order_by(&[0]), cpu_batches.clone()), measurement(order_by(&[0]), tied_batches.clone())),
        validation::ParityStatus::Matched
    );
    // Moving a row across a key change breaks the ORDER BY sequence.
    assert_ne!(cpu, key_value_fingerprint(&[0], &[(1, "c"), (3, "a"), (3, "b")]));
    assert_eq!(
        parity(measurement(order_by(&[0]), cpu_batches.clone()), measurement(order_by(&[0]), crossed_batches)),
        validation::ParityStatus::Mismatched
    );
    // Unknown ORDER BY keys fix the emitted row sequence.
    assert_eq!(
        parity(
            measurement(ParityRowOrdering::Ordered, cpu_batches),
            measurement(ParityRowOrdering::Ordered, tied_batches)
        ),
        validation::ParityStatus::Mismatched
    );
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

#[test]
fn parity_applies_float_tolerance_only_to_float_values() {
    // q14 at SF500: GPU and DuckDB against DataFusion CPU, one ulp apart.
    let cpu = float_batch(&[16.64917006252482]);
    let one_ulp = float_batch(&[16.649170062524817]);

    assert_ne!(
        measurement(ParityRowOrdering::Unordered, vec![cpu.clone()]).output_fingerprint,
        measurement(ParityRowOrdering::Unordered, vec![one_ulp.clone()]).output_fingerprint
    );
    assert_eq!(single_batch_parity(&cpu, &one_ulp), validation::ParityStatus::Matched);
    assert_eq!(single_batch_parity(&float_batch(&[-0.0]), &float_batch(&[0.0])), validation::ParityStatus::Matched);
    assert_eq!(
        single_batch_parity(&float_batch(&[f64::NAN]), &float_batch(&[f64::NAN])),
        validation::ParityStatus::Matched
    );
    assert_eq!(single_batch_parity(&cpu, &float_batch(&[16.6491705])), validation::ParityStatus::Mismatched);
    // One unit in the last decimal place is far below the float bounds and must still mismatch.
    assert_eq!(
        single_batch_parity(&decimal_batch(12_345_678_901_234), &decimal_batch(12_345_678_901_235)),
        validation::ParityStatus::Mismatched
    );
}

#[test]
fn parity_does_not_match_values_of_a_type_it_cannot_compare() {
    // Different Time64 values must not count as matched just because the comparator has no
    // canonical form for the type.
    let time_batch = |value: i64| {
        let schema = Arc::new(Schema::new(vec![Field::new("t", DataType::Time64(TimeUnit::Microsecond), false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Time64MicrosecondArray::from(vec![value])) as ArrayRef]).unwrap()
    };
    assert_eq!(single_batch_parity(&time_batch(1), &time_batch(2)), validation::ParityStatus::Mismatched);
}

#[tokio::test]
async fn row_ordering_follows_the_top_level_order_by() {
    let ctx = SessionContext::new();
    ctx.register_batch("t", key_value_batch(&[(1, "a")])).unwrap();

    // TPC-H Q11 shape: ORDER BY an aggregate alias.
    assert_eq!(
        row_ordering(&ctx, "select k, count(v) as value from t group by k order by value desc").await,
        ParityRowOrdering::OrderBy(vec![ParitySortKey { column: 1, descending: true }])
    );
    assert_eq!(
        row_ordering(&ctx, "select k, v from t order by v desc, k limit 3").await,
        ParityRowOrdering::OrderBy(vec![
            ParitySortKey { column: 1, descending: true },
            ParitySortKey { column: 0, descending: false },
        ])
    );
    assert_eq!(row_ordering(&ctx, "select k, v from t").await, ParityRowOrdering::Unordered);
    assert_eq!(
        row_ordering(&ctx, "select * from (select k, v from t order by v) s").await,
        ParityRowOrdering::Unordered
    );
    // A hidden key is projected away, so the emitted row sequence must match.
    assert_eq!(row_ordering(&ctx, "select v from t order by k").await, ParityRowOrdering::Ordered);
    assert_eq!(row_ordering(&ctx, "select k, v from t order by k + 1").await, ParityRowOrdering::Ordered);
}

async fn row_ordering(ctx: &SessionContext, sql: &str) -> ParityRowOrdering {
    validation::row_ordering(ctx.sql(sql).await.unwrap().logical_plan())
}

fn order_by(columns: &[usize]) -> ParityRowOrdering {
    ParityRowOrdering::OrderBy(columns.iter().map(|&column| ParitySortKey { column, descending: false }).collect())
}

fn key_value_fingerprint(order_key_columns: &[usize], rows: &[(i32, &str)]) -> String {
    let batch = key_value_batch(rows);
    validation::fingerprint_batches(&batch.schema(), &order_by(order_key_columns), &[batch]).unwrap()
}

fn parity(cpu: execution::QueryMeasurement, native: execution::QueryMeasurement) -> validation::ParityStatus {
    validation::parity_status(Some(&cpu), Some(&native), DEFAULT_PARITY_FLOAT_TOLERANCE)
}

fn single_batch_parity(cpu: &RecordBatch, native: &RecordBatch) -> validation::ParityStatus {
    parity(
        measurement(ParityRowOrdering::Unordered, vec![cpu.clone()]),
        measurement(ParityRowOrdering::Unordered, vec![native.clone()]),
    )
}

fn float_batch(values: &[f64]) -> RecordBatch {
    single_column_batch(DataType::Float64, Arc::new(Float64Array::from(values.to_vec())))
}

fn decimal_batch(value: i128) -> RecordBatch {
    let array = Decimal128Array::from(vec![value]).with_precision_and_scale(15, 2).unwrap();
    single_column_batch(DataType::Decimal128(15, 2), Arc::new(array))
}

fn single_column_batch(data_type: DataType, array: ArrayRef) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, false)]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
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

fn measurement(row_ordering: ParityRowOrdering, batches: Vec<RecordBatch>) -> execution::QueryMeasurement {
    let schema = batches[0].schema();
    execution::QueryMeasurement {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: 0,
        output_fingerprint: validation::fingerprint_batches(&schema, &row_ordering, &batches).unwrap(),
        output: execution::QueryOutput { schema, row_ordering, batches },
    }
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
