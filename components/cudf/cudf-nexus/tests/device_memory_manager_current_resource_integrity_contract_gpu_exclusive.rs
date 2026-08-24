use std::{ptr, sync::Arc};

use cudf::{
    DeviceId, DeviceMemoryFailureAction, DeviceMemoryManager, DeviceMemoryManagerInstallationCause,
    DeviceMemoryManagerInstallationIdentity, DeviceMemoryManagerInstallationRetry,
    DeviceMemoryManagerInstallationStatus, DeviceMemoryManagerOptions, testing::RawHandleTestingExt,
};

#[test]
fn direct_rmm_replacement_is_detected_and_integrity_loss_is_sticky() {
    let device = DeviceId::current().unwrap();
    let identity = DeviceMemoryManagerInstallationIdentity::new([0x51; 16], 1, [0x52; 16]);
    let (manager, installed) = DeviceMemoryManager::install_for_device_with_owned_failure_callback_details_and_options(
        device,
        identity,
        Arc::new(|_| DeviceMemoryFailureAction::Decline),
        DeviceMemoryManagerOptions::default(),
    )
    .unwrap();
    let generation = installed.state.installation_generation.unwrap();

    let mut foreign = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut foreign) }, cudf_sys::CUDF_SUCCESS);
    assert!(!foreign.is_null());
    assert_eq!(
        unsafe { cudf_sys::test_support::cudf_test_memory_resource_set_current_untracked(foreign) },
        cudf_sys::CUDF_SUCCESS,
    );

    let replaced = DeviceMemoryManager::installation_state_for_device(device).unwrap();
    assert_eq!(replaced.status, DeviceMemoryManagerInstallationStatus::Incomplete);
    assert_eq!(replaced.retry, DeviceMemoryManagerInstallationRetry::PermanentlyUnusable);
    assert_eq!(replaced.cause, DeviceMemoryManagerInstallationCause::CurrentResourceReplaced);
    assert!(!replaced.current_resource_installed);
    assert_eq!(replaced.identity, Some(identity));
    assert_eq!(replaced.installation_generation, Some(generation));

    assert_eq!(
        unsafe {
            cudf_sys::test_support::cudf_test_memory_resource_set_current_raw(
                manager.resource().testing_raw_handle_value() as *mut core::ffi::c_void,
            )
        },
        cudf_sys::CUDF_SUCCESS,
    );
    let restored_but_unusable = DeviceMemoryManager::installation_state_for_device(device).unwrap();
    assert!(restored_but_unusable.current_resource_installed);
    assert_eq!(restored_but_unusable.retry, DeviceMemoryManagerInstallationRetry::PermanentlyUnusable);
    assert_eq!(restored_but_unusable.cause, DeviceMemoryManagerInstallationCause::CurrentResourceReplaced);

    DeviceMemoryManager::install_for_device_with_owned_failure_callback_details_and_options(
        device,
        identity,
        Arc::new(|_| DeviceMemoryFailureAction::Decline),
        DeviceMemoryManagerOptions::default(),
    )
    .unwrap_err();
    assert_eq!(
        unsafe { cudf_sys::cudf_memory_resource_destroy(foreign, device.ordinal() as i32) },
        cudf_sys::CUDF_SUCCESS,
    );
}
