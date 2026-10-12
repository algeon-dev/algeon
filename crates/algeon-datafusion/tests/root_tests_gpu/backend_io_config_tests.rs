use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    observability::{KvikioCompatMode, KvikioProcessConfig},
};
use cudf::io::kvikio_process_config;

/// Algeon's default for each field, unless the KvikIO environment pins it.
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

#[test]
fn backend_build_applies_kvikio_process_defaults() {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();

    assert_eq!(backend.kvikio_process_config(), Some(expected_process_config()));
    assert_eq!(backend.kvikio_process_config(), kvikio_process_config());
}
