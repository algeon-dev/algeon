use std::{ptr, sync::Arc};

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerInstallationCause,
    DeviceMemoryManagerInstallationIdentity, DeviceMemoryManagerOptions, ErrorCode, ErrorKind, InstallSpec,
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
    let manager = DeviceMemoryManager::install(device, InstallSpec { options: pool_options, ..InstallSpec::default() })
        .unwrap()
        .0;

    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), Some(pool_options));
    DeviceMemoryManager::install(device, InstallSpec { options: pool_options, ..InstallSpec::default() }).unwrap();
    Runtime::builder().with_device(device).build().unwrap();

    let mismatch = DeviceMemoryManager::install(
        device,
        InstallSpec { options: DeviceMemoryManagerOptions::default(), ..InstallSpec::default() },
    )
    .unwrap_err();
    assert_eq!(mismatch.error().code(), ErrorCode::RuntimeBindingMismatch);

    let native_async_options = cudf_sys::cudf_memory_manager_options_t::default();
    let mut raw_manager = ptr::null_mut();
    let mut installation = cudf_sys::cudf_memory_manager_installation_record_t::default();
    let native_status = unsafe {
        cudf_sys::cudf_memory_manager_install(
            i32::try_from(device.ordinal()).unwrap(),
            &native_async_options,
            None,
            ptr::null_mut(),
            ptr::null(),
            &mut raw_manager,
            &mut installation,
        )
    };
    assert_eq!(native_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(raw_manager.is_null());
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(pool_options),
        "a conflicting explicit request must not replace the process-global pool",
    );

    let owned_conflict = DeviceMemoryManager::install(
        device,
        InstallSpec {
            options: DeviceMemoryManagerOptions::default(),
            owner: Some(DeviceMemoryManagerInstallationIdentity::new(*b"algeon-test-id\0\0", 1, [0x42; 16])),
            on_failure: Some(Arc::new(|_| {})),
        },
    )
    .unwrap_err();
    assert_eq!(owned_conflict.error().code(), ErrorCode::RuntimeBindingMismatch);
    let state = owned_conflict
        .installation_state()
        .expect("an owned-install conflict must preserve the authoritative native installation record");
    assert_eq!(state.cause, DeviceMemoryManagerInstallationCause::OptionsMismatch);
    assert_eq!(state.effective_options, Some(pool_options));

    let err = manager
        .query_memory_domain(50_001, QueryMemoryDomainOptions::new(8 * 1024 * 1024).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
