use std::{fs::File, sync::Arc, time::Duration};

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
};
use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionContext},
};
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;

#[test]
fn backend_session_does_not_register_session_config_extension() {
    let _serial = super::root_gpu_fixture::lock();
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
}

/// A default backend requires native plans. Over a table of several files,
/// DataFusion plans several partitions as a limit per partition and as partial
/// and final `MEDIAN` stages, which native lowering rejects, so the installed
/// session must plan one partition for these queries to run.
#[tokio::test]
async fn default_backend_session_runs_multi_file_limits_and_medians_natively() {
    let _serial = super::root_gpu_fixture::lock();
    let dir = tempfile::tempdir().unwrap();
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("a", DataType::Int64, false)]));
    for part in 0..3_i64 {
        let values = (part * 100..(part + 1) * 100).collect::<Vec<_>>();
        let keys = values.iter().map(|value| value % 7).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys)), Arc::new(Int64Array::from(values))],
        )
        .unwrap();
        let file = File::create(dir.path().join(format!("part{part}.parquet"))).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let ctx = SessionContext::new_with_state(
        backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build(),
    );
    ctx.register_parquet("t", dir.path().to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    for (sql, expected_rows) in
        [("SELECT k, COUNT(*) AS c FROM t GROUP BY k LIMIT 2", 2), ("SELECT k, MEDIAN(a) AS m FROM t GROUP BY k", 7)]
    {
        let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
        let batches = attempt
            .sql(&ctx, sql)
            .await
            .unwrap_or_else(|error| panic!("{sql}: {error}"))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), expected_rows, "{sql}");
    }
    backend.close_and_wait_async(Duration::from_secs(10)).await.unwrap();
}
