use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerInstallationCause, DeviceMemoryManagerInstallationIdentity,
    DeviceMemoryManagerInstallationOutcomeKind, DeviceMemoryManagerInstallationRetry,
    DeviceMemoryManagerInstallationStatus, DeviceMemoryManagerOptions, ErrorCode, ErrorKind, InstallSpec,
    QueryMemoryDomainOptions, Runtime,
};

#[test]
fn identified_install_is_reusable_but_cannot_be_rebound() {
    let device = DeviceId::current().unwrap();
    let before = DeviceMemoryManager::installation_state_for_device(device).unwrap();
    assert_eq!(before.status, DeviceMemoryManagerInstallationStatus::Uninstalled);
    assert_eq!(before.retry, DeviceMemoryManagerInstallationRetry::Installable);
    assert_eq!(before.cause, DeviceMemoryManagerInstallationCause::None);

    let identity = DeviceMemoryManagerInstallationIdentity::new([0x11; 16], 1, [0x22; 16]);
    let missing_callback =
        DeviceMemoryManager::install(device, InstallSpec { owner: Some(identity), ..InstallSpec::default() })
            .unwrap_err();
    assert_eq!(missing_callback.error().code(), ErrorCode::RuntimeBindingMismatch);
    assert!(missing_callback.installation_state().is_none(), "validation must precede native installation");
    assert_eq!(DeviceMemoryManager::installation_state_for_device(device).unwrap(), before);

    let callback_calls = Arc::new(AtomicUsize::new(0));
    let callback_calls_for_install = Arc::clone(&callback_calls);
    let callback = Arc::new(move |_| {
        callback_calls_for_install.fetch_add(1, Ordering::SeqCst);
    });
    let options = DeviceMemoryManagerOptions {
        allocation_limit_bytes: Some(8 * 1024 * 1024),
        ..DeviceMemoryManagerOptions::default()
    };
    let (_manager, installed) = DeviceMemoryManager::install(
        device,
        InstallSpec { options, owner: Some(identity), on_failure: Some(callback) },
    )
    .unwrap();
    assert_eq!(installed.kind, DeviceMemoryManagerInstallationOutcomeKind::Installed);
    assert_eq!(installed.state.status, DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(installed.state.retry, DeviceMemoryManagerInstallationRetry::Reusable);
    assert_eq!(installed.state.cause, DeviceMemoryManagerInstallationCause::None);
    assert_eq!(installed.state.identity, Some(identity));
    let generation = installed.state.installation_generation.unwrap();

    let replacement_callback = Arc::new(|_| {});
    let (_manager, reused) = DeviceMemoryManager::install(
        device,
        InstallSpec { options, owner: Some(identity), on_failure: Some(replacement_callback) },
    )
    .unwrap();
    assert_eq!(reused.kind, DeviceMemoryManagerInstallationOutcomeKind::Reused);
    assert_eq!(reused.state.installation_generation, Some(generation));

    let other_identity = DeviceMemoryManagerInstallationIdentity::new([0x33; 16], 1, [0x22; 16]);
    let error = DeviceMemoryManager::install(
        device,
        InstallSpec { options, owner: Some(other_identity), on_failure: Some(Arc::new(|_| {})) },
    )
    .unwrap_err();
    assert_eq!(error.error().code(), ErrorCode::RuntimeBindingMismatch);
    let rejected = error.installation_state().expect("identity rejection must preserve the locked native record");
    assert_eq!(rejected.status, DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(rejected.retry, DeviceMemoryManagerInstallationRetry::Reusable);
    assert_eq!(rejected.cause, DeviceMemoryManagerInstallationCause::IdentityMismatch);
    assert_eq!(rejected.installation_generation, Some(generation));
    assert_eq!(rejected.identity, Some(identity));
    assert_eq!(rejected.effective_options, Some(options));

    let after = DeviceMemoryManager::installation_state_for_device(device).unwrap();
    assert_eq!(after.identity, Some(identity));
    assert_eq!(after.installation_generation, Some(generation));
    assert_eq!(after.effective_options, Some(options));

    let conflicting_options = DeviceMemoryManagerOptions {
        allocation_limit_bytes: Some(16 * 1024 * 1024),
        ..DeviceMemoryManagerOptions::default()
    };
    let conflict = DeviceMemoryManager::install(
        device,
        InstallSpec { options: conflicting_options, owner: Some(identity), on_failure: Some(Arc::new(|_| {})) },
    )
    .unwrap_err();
    assert_eq!(conflict.error().code(), ErrorCode::RuntimeBindingMismatch);
    assert_eq!(conflict.installation_state().unwrap().cause, DeviceMemoryManagerInstallationCause::OptionsMismatch);
    assert_eq!(DeviceMemoryManager::installation_state_for_device(device).unwrap(), after);

    let manager = DeviceMemoryManager::for_device(device).unwrap();
    let domain = manager
        .query_memory_domain(7, QueryMemoryDomainOptions::new(1024 * 1024).with_ceiling_bytes(1024 * 1024))
        .expect("a static-partition manager supports a bounded private query domain");
    let identity = domain.identity().expect("query domain exposes its immutable identity");
    assert_eq!(identity.device, device);
    assert_eq!(identity.domain_id, 7);
    drop(domain);
    let runtime =
        Runtime::builder().with_device(device).with_memory_resource(manager.resource().clone()).build().unwrap();
    let error = match runtime.device_buffer(9 * 1024 * 1024) {
        Ok(_) => panic!("manager-wide allocation limit must reject an oversized buffer"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn allocation_limit_validation_is_cuda_free_and_rejects_inconsistent_pool_bounds() {
    let zero = DeviceMemoryManagerOptions { allocation_limit_bytes: Some(0), ..DeviceMemoryManagerOptions::default() };
    assert_eq!(zero.validate().unwrap_err().code(), ErrorCode::ValueOutOfRange);

    let oversized_initial = DeviceMemoryManagerOptions {
        base: cudf::DeviceMemoryManagerBase::Pool {
            initial_pool_size: 16 * 1024 * 1024,
            maximum_pool_size: Some(16 * 1024 * 1024),
        },
        allocation_limit_bytes: Some(8 * 1024 * 1024),
    };
    assert_eq!(oversized_initial.validate().unwrap_err().code(), ErrorCode::ValueOutOfRange);

    let oversized_maximum = DeviceMemoryManagerOptions {
        base: cudf::DeviceMemoryManagerBase::Pool {
            initial_pool_size: 4 * 1024 * 1024,
            maximum_pool_size: Some(16 * 1024 * 1024),
        },
        allocation_limit_bytes: Some(8 * 1024 * 1024),
    };
    assert_eq!(oversized_maximum.validate().unwrap_err().code(), ErrorCode::ValueOutOfRange);

    let inverted_pool = DeviceMemoryManagerOptions {
        base: cudf::DeviceMemoryManagerBase::Pool {
            initial_pool_size: 8 * 1024 * 1024,
            maximum_pool_size: Some(4 * 1024 * 1024),
        },
        allocation_limit_bytes: None,
    };
    assert_eq!(inverted_pool.validate().unwrap_err().code(), ErrorCode::ValueOutOfRange);

    let zero_maximum_is_the_existing_unbounded_sentinel = DeviceMemoryManagerOptions {
        base: cudf::DeviceMemoryManagerBase::Pool { initial_pool_size: 8 * 1024 * 1024, maximum_pool_size: Some(0) },
        allocation_limit_bytes: Some(8 * 1024 * 1024),
    };
    zero_maximum_is_the_existing_unbounded_sentinel.validate().unwrap();
}
