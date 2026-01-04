use std::{
    fs::{self, File},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Decimal128Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

pub(crate) static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("value", DataType::Int32, false)]))
}

pub(crate) fn input_batch() -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap()
}

pub(crate) fn sort_input_batches() -> Vec<RecordBatch> {
    vec![
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(vec![4, 2])), Arc::new(Int32Array::from(vec![40, 20]))],
        )
        .unwrap(),
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(vec![5, 3])), Arc::new(Int32Array::from(vec![50, 30]))],
        )
        .unwrap(),
    ]
}

pub(crate) fn int32_values(batch: &RecordBatch, index: usize) -> Vec<i32> {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array").values().to_vec()
}

pub(crate) fn int64_values(batch: &RecordBatch, index: usize) -> Vec<i64> {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array").values().to_vec()
}

pub(crate) fn decimal128_values(batch: &RecordBatch, index: usize) -> Vec<i128> {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array").values().to_vec()
}

pub(crate) fn write_parquet_file(path: &std::path::Path, batch: &RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = datafusion::parquet::arrow::ArrowWriter::try_new(file, Arc::clone(&batch.schema()), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

pub(crate) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
