//! KvikIO process lifecycle: Algeon's defaults are applied once before I/O,
//! survive libcudf's own lazy setup, and the bounce pools drain after I/O.

use cudf::{
    DataFrame, ErrorCode, ErrorKind, ErrorStatus, Exec, ParquetReadOptions, Runtime,
    io::{
        KvikioCompatMode, KvikioProcessConfig, configure_kvikio_process, drain_kvikio_bounce_buffer_pools,
        kvikio_process_config,
    },
};
use cudf_sys::test_support;
use tempfile::TempDir;

struct ActiveIoGuard {
    active: bool,
}

impl ActiveIoGuard {
    fn begin() -> Self {
        // SAFETY: nextest runs this test alone in its own process, so nothing
        // else races this process-global hook; the guard balances the
        // increment on every exit path.
        let status = unsafe { test_support::cudf_test_io_kvikio_active_io_begin() };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
        Self { active: true }
    }

    fn end(mut self) {
        // SAFETY: this is the matching decrement for `begin` above.
        let status = unsafe { test_support::cudf_test_io_kvikio_active_io_end() };
        self.active = false;
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    }
}

impl Drop for ActiveIoGuard {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: best-effort unwind cleanup for the increment owned by
            // this guard. The normal path asserts the returned status.
            let _ = unsafe { test_support::cudf_test_io_kvikio_active_io_end() };
        }
    }
}

/// The configuration `configure_kvikio_process` must report: Algeon's default
/// for each field, unless the KvikIO environment pins that field.
fn expected_process_config() -> KvikioProcessConfig {
    let mut expected = KvikioProcessConfig::DEFAULT;
    if let Ok(mode) = std::env::var("KVIKIO_COMPAT_MODE") {
        expected.compat_mode = match mode.trim().to_ascii_lowercase().as_str() {
            "on" => KvikioCompatMode::On,
            "off" => KvikioCompatMode::Off,
            "auto" => KvikioCompatMode::Auto,
            other => panic!("unexpected KVIKIO_COMPAT_MODE={other}"),
        };
    }
    if let Ok(threads) = std::env::var("KVIKIO_NTHREADS").or_else(|_| std::env::var("KVIKIO_NUM_THREADS")) {
        expected.thread_count = threads.trim().parse().expect("KVIKIO_NTHREADS is a positive integer");
    }
    if let Ok(task_size) = std::env::var("KVIKIO_TASK_SIZE") {
        expected.task_size_bytes = task_size.trim().parse().expect("KVIKIO_TASK_SIZE is a positive integer");
    }
    expected
}

fn write_and_read_local_parquet(path: &std::path::Path) {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let frame = DataFrame::from_series([
        ("id", runtime.series(&[1_i32, 2, 3, 4]).unwrap()),
        ("value", runtime.series(&[10_i64, 20, 30, 40]).unwrap()),
    ])
    .unwrap();
    frame.write_parquet(path).expect("compat-mode Parquet write");

    let decoded = DataFrame::read_parquet(
        ParquetReadOptions::builder().filepath(path).with_runtime(&runtime).build(),
        Exec::RETAINED,
    )
    .expect("compat-mode Parquet read");
    assert_eq!(decoded.column_names(), vec!["id", "value"]);
    assert_eq!(decoded.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(decoded.series("value").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn process_defaults_survive_the_first_local_file_open() {
    kvikio_process_config().expect("test requires a libcudf build with KvikIO support");
    let expected = expected_process_config();

    let configured = configure_kvikio_process(0, &KvikioProcessConfig::DEFAULT).expect("configure before any I/O");
    assert_eq!(configured, expected);
    assert_eq!(kvikio_process_config(), Some(expected));

    // libcudf's own once-only setup runs on the first local file open and
    // resets the thread count unless it already ran inside configure.
    let dir = TempDir::new().unwrap();
    write_and_read_local_parquet(&dir.path().join("kvikio-process-contract.parquet"));
    assert_eq!(kvikio_process_config(), Some(expected), "the first local file open must not change the pool");

    let again = configure_kvikio_process(0, &KvikioProcessConfig::DEFAULT).expect("idempotent reconfigure");
    assert_eq!(again, expected);
}

#[test]
fn active_io_fences_process_configuration_and_pool_drain() {
    kvikio_process_config().expect("test requires a libcudf build with KvikIO support");
    let before = configure_kvikio_process(0, &KvikioProcessConfig::DEFAULT).expect("configure before any I/O");

    let active_io = ActiveIoGuard::begin();
    let error =
        configure_kvikio_process(0, &KvikioProcessConfig::DEFAULT).expect_err("active I/O must fence configure");
    assert_eq!(error.code(), ErrorCode::IoBusy);
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(error.status(), ErrorStatus::Temporary);
    assert_eq!(error.detail_value("reason"), Some("active_io"));
    assert_eq!(kvikio_process_config(), Some(before), "a fenced configure changes nothing");

    let error = drain_kvikio_bounce_buffer_pools().expect_err("active I/O must fence pool destruction");
    assert_eq!(error.code(), ErrorCode::IoBusy);
    assert_eq!(error.detail_value("reason"), Some("active_io"));
    active_io.end();
}

#[test]
fn compat_parquet_roundtrip_drains_kvikio_bounce_pools() {
    let configured = configure_kvikio_process(0, &KvikioProcessConfig::DEFAULT).expect("configure before any I/O");
    assert_eq!(configured.compat_mode, expected_process_config().compat_mode);

    // Start from an idle, drained pool so the final pair tests the lifecycle
    // generated by this real local Parquet round trip alone.
    drain_kvikio_bounce_buffer_pools().expect("initial idle drain");

    let dir = TempDir::new().unwrap();
    write_and_read_local_parquet(&dir.path().join("kvikio-shutdown-contract.parquet"));

    // The pinned bounce buffer that libcudf's compatibility-mode read borrowed
    // stays in KvikIO's pool until this drain, so the first drain clears bytes.
    // Zero here means the shim drained a KvikIO instance other than the one
    // libcudf reads through.
    let cleared = drain_kvikio_bounce_buffer_pools().expect("first idle drain after real I/O");
    assert!(cleared > 0, "the drain after a local read must clear libcudf's bounce buffers");
    assert_eq!(
        drain_kvikio_bounce_buffer_pools().expect("second idle drain"),
        0,
        "draining an already empty KvikIO pool must be idempotent"
    );
}
