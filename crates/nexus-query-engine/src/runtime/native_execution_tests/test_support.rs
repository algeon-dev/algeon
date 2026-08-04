use super::*;

pub(super) fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

pub(super) fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array")
}

pub(super) fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array")
}

pub(super) fn float64_array(batch: &RecordBatch, index: usize) -> &Float64Array {
    batch.column(index).as_any().downcast_ref::<Float64Array>().expect("expected Float64Array")
}

pub(super) fn decimal128_array(batch: &RecordBatch, index: usize) -> &Decimal128Array {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array")
}

pub(super) fn string_array(batch: &RecordBatch, index: usize) -> &StringArray {
    batch.column(index).as_any().downcast_ref::<StringArray>().expect("expected StringArray")
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
