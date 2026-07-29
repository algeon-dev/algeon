use cudf::{DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions, ErrorCode};

#[test]
fn pool_request_after_default_async_install_is_rejected_without_mutation() {
    let device = DeviceId::current().unwrap();
    let async_options = DeviceMemoryManagerOptions::default();
    DeviceMemoryManager::install_for_device(device).unwrap();
    DeviceMemoryManager::install_for_device_with_options(
        device,
        DeviceMemoryManagerOptions {
            base: DeviceMemoryManagerBase::CudaAsync {
                initial_pool_size: Some(0),
                release_threshold: Some(usize::MAX),
            },
            allocation_limit_bytes: None,
        },
    )
    .unwrap();

    let pool_options = DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 64 * 1024 * 1024,
            maximum_pool_size: Some(64 * 1024 * 1024),
        },
        allocation_limit_bytes: None,
    };
    let mismatch = DeviceMemoryManager::install_for_device_with_options(device, pool_options).unwrap_err();

    assert_eq!(mismatch.code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(async_options),
        "a conflicting explicit request must not replace the process-global async allocator",
    );
}
