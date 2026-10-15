//! Builders shared by the integration exec suite and the in-crate adapter tests.
//!
//! Unit and integration targets share these fixtures through the public adapter
//! API. Keep backend construction in the target that owns each test.

use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::error::Error as AlgeonError;
use arrow::{
    array::{Int32Array, Int64Array, StringViewArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, ScalarValue},
    parquet::arrow::ArrowWriter,
    physical_expr::expressions::{LikeExpr, col, lit},
    physical_plan::{ExecutionPlan, filter::FilterExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::executor::block_on;

pub(super) fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().unwrap()
}

pub(super) fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().unwrap()
}

pub(super) fn adapter_error(error: &DataFusionError) -> &AlgeonError {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source.downcast_ref::<AlgeonError>().expect("DataFusion external error should preserve adapter Error")
}

pub(super) fn parquet_file_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]))
}

pub(super) fn write_parquet_file(path: &std::path::Path) {
    let schema = parquet_file_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

pub(super) fn write_ilike_parquet_file(path: &std::path::Path) {
    write_ilike_parquet_file_with_labels(
        path,
        &[
            Some("Alpha".to_string()),
            Some("beta".to_string()),
            Some("ALMANAC".to_string()),
            None,
            Some("zeta".to_string()),
        ],
    );
}

pub(super) fn write_ilike_parquet_file_with_labels(path: &std::path::Path, labels: &[Option<String>]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8View, true),
    ]));
    let ids =
        (1..=i32::try_from(labels.len()).expect("mixed-boundary fixture row count must fit i32")).collect::<Vec<_>>();
    let labels = labels.iter().map(|label| label.as_deref()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringViewArray::from(labels))],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_target_partitions(path, 1)
}

pub(super) fn parquet_scan_with_target_partitions(
    path: &std::path::Path,
    target_partitions: usize,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(target_partitions),
    );
    let df = block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn ilike_filter_plan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    ilike_filter_plan_with_pattern(path, "al%".to_string())
}

pub(super) fn ilike_filter_plan_with_pattern(path: &std::path::Path, pattern: String) -> Arc<dyn ExecutionPlan> {
    ilike_filter_over_scan_with_pattern(parquet_scan(path), pattern)
}

pub(super) fn ilike_filter_over_scan_with_pattern(
    scan: Arc<dyn ExecutionPlan>,
    pattern: String,
) -> Arc<dyn ExecutionPlan> {
    let schema = scan.schema();
    let predicate =
        Arc::new(LikeExpr::new(false, true, col("label", &schema).unwrap(), lit(ScalarValue::Utf8View(Some(pattern)))));
    Arc::new(FilterExec::try_new(predicate, scan).unwrap())
}
