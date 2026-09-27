use std::sync::Arc;

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerInstallationIdentity, DeviceMemoryManagerInstallationRetry,
    DeviceMemoryManagerInstallationStatus, DeviceMemoryManagerOptions, InstallSpec,
};
use cudf_sys::test_support;

#[test]
fn callback_panic_before_rust_receives_generation_does_not_block_installation() {
    let device = DeviceId::current().unwrap();
    let identity = DeviceMemoryManagerInstallationIdentity::new([0x61; 16], 1, [0x62; 16]);
    let options = DeviceMemoryManagerOptions {
        allocation_limit_bytes: Some(1024 * 1024),
        ..DeviceMemoryManagerOptions::default()
    };

    unsafe { test_support::cudf_test_memory_manager_invoke_failure_callback_before_install_returns() };
    let (manager, outcome) = DeviceMemoryManager::install(
        device,
        InstallSpec {
            options,
            owner: Some(identity),
            on_failure: Some(Arc::new(|_| panic!("injected callback panic before install returns"))),
        },
    )
    .expect("an observation callback failure must not decide manager installation");

    let state = outcome.state;
    assert_eq!(state.status, DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(state.retry, DeviceMemoryManagerInstallationRetry::Reusable);
    assert!(state.failure_callback_installed);
    assert!(state.current_resource_installed);
    assert!(!state.failure_callback_panicked);
    assert_eq!(state.identity, Some(identity));
    assert!(state.installation_generation.is_some());
    assert!(!DeviceMemoryManager::failure_callback_state_for_device(device).unwrap().failure_callback_panicked);
    assert_eq!(
        DeviceMemoryManager::installation_state_for_device(manager.device()).unwrap().status,
        DeviceMemoryManagerInstallationStatus::Owned
    );
}
