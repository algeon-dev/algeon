use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions, ErrorCode, ErrorKind,
    QueryMemoryDomainOptions, Runtime,
};

#[test]
fn query_memory_domain_rejects_pool_backed_manager() {
    let device = DeviceId::current().unwrap();
    let pool_options = DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 64 * 1024 * 1024,
            maximum_pool_size: Some(64 * 1024 * 1024),
        },
        allocation_limit_bytes: None,
    };
    let manager = DeviceMemoryManager::install_for_device_with_options(device, pool_options).unwrap();

    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), Some(pool_options));
    DeviceMemoryManager::install_for_device_with_options(device, pool_options).unwrap();
    Runtime::builder().with_device(device).build().unwrap();

    let mismatch = DeviceMemoryManager::install_for_device_with_options(device, DeviceMemoryManagerOptions::default())
        .unwrap_err();
    assert_eq!(mismatch.code(), ErrorCode::RuntimeBindingMismatch);

    let native_async_options = cudf_sys::cudf_memory_manager_options_t::default();
    let mut raw_manager = std::ptr::null_mut();
    let native_status = unsafe {
        cudf_sys::cudf_memory_manager_for_device_with_options(
            i32::try_from(device.ordinal()).unwrap(),
            &native_async_options,
            &mut raw_manager,
        )
    };
    assert_eq!(native_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(raw_manager.is_null());
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(pool_options),
        "a conflicting explicit request must not replace the process-global pool",
    );

    let err = manager
        .query_memory_domain(50_001, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
