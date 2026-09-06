use std::sync::Arc;

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerInstallationCause, DeviceMemoryManagerInstallationIdentity,
    DeviceMemoryManagerInstallationOutcomeKind, DeviceMemoryManagerInstallationRetry,
    DeviceMemoryManagerInstallationStatus, DeviceMemoryManagerOptions,
};

#[test]
fn failed_current_resource_switch_is_retryable_but_never_reusable() {
    let device = DeviceId::current().unwrap();
    let identity = DeviceMemoryManagerInstallationIdentity::new([0x41; 16], 1, [0x42; 16]);
    let options = DeviceMemoryManagerOptions::default();

    unsafe { cudf_sys::test_support::cudf_test_memory_manager_fail_next_current_resource_install() };
    let failure = DeviceMemoryManager::try_install_for_device_with_owned_failure_callback_details_and_options(
        device,
        identity,
        Arc::new(|_| {}),
        options,
    )
    .unwrap_err();

    let incomplete = failure
        .installation_state()
        .expect("the rejected locked install transaction must return its exact native record");
    assert_eq!(incomplete.status, DeviceMemoryManagerInstallationStatus::Incomplete);
    assert_eq!(incomplete.retry, DeviceMemoryManagerInstallationRetry::Retryable);
    assert_eq!(incomplete.cause, DeviceMemoryManagerInstallationCause::CurrentResourceInstallFailed);
    assert!(incomplete.failure_callback_installed);
    assert!(!incomplete.current_resource_installed);
    assert_eq!(incomplete.identity, Some(identity));
    let generation = incomplete.installation_generation.expect("registered callback has a generation");

    let (_manager, recovered) =
        DeviceMemoryManager::install_for_device_with_owned_failure_callback_details_and_options(
            device,
            identity,
            Arc::new(|_| {}),
            options,
        )
        .unwrap();
    assert_eq!(recovered.kind, DeviceMemoryManagerInstallationOutcomeKind::Reused);
    assert_eq!(recovered.state.status, DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(recovered.state.retry, DeviceMemoryManagerInstallationRetry::Reusable);
    assert_eq!(recovered.state.cause, DeviceMemoryManagerInstallationCause::None);
    assert!(recovered.state.current_resource_installed);
    assert_eq!(recovered.state.installation_generation, Some(generation));
}
