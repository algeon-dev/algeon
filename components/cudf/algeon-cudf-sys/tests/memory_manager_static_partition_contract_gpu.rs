use std::{
    ffi::CStr,
    ptr,
    sync::{Arc, Barrier},
};

#[test]
fn native_manager_limit_is_enforced_inspected_and_immutable() {
    const LIMIT: usize = 1024 * 1024;
    const FIRST_ALLOCATION: usize = 3 * LIMIT / 4;
    const SECOND_ALLOCATION: usize = LIMIT / 2;

    let mut device = -1;
    assert_eq!(unsafe { cudf_sys::cudf_cuda_get_device(&mut device) }, cudf_sys::CUDF_SUCCESS);

    let invalid = cudf_sys::cudf_memory_manager_options_t {
        base_kind: cudf_sys::CUDF_MEMORY_MANAGER_BASE_POOL,
        initial_pool_size_bytes: (2 * LIMIT) as u64,
        maximum_pool_size_bytes: (2 * LIMIT) as u64,
        allocation_limit_bytes: LIMIT as u64,
        ..Default::default()
    };
    let mut manager = ptr::null_mut();
    let mut installation = cudf_sys::cudf_memory_manager_installation_record_t::default();
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_install(
                device,
                ptr::null(),
                None,
                ptr::null_mut(),
                ptr::null(),
                &mut manager,
                &mut installation,
            )
        },
        cudf_sys::CUDF_ERROR_NULL,
    );
    assert_eq!(installation.struct_size, 0, "a null-options request must clear the supplied record");
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_install(
                device,
                &invalid,
                None,
                ptr::null_mut(),
                ptr::null(),
                &mut manager,
                &mut installation,
            )
        },
        cudf_sys::CUDF_ERROR_INVALID_ARGUMENT,
    );
    assert!(manager.is_null());
    assert_eq!(installation.struct_size, 0, "validation before registry access has no installation record");

    let options =
        cudf_sys::cudf_memory_manager_options_t { allocation_limit_bytes: LIMIT as u64, ..Default::default() };
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_install(
                device,
                &options,
                None,
                ptr::null_mut(),
                ptr::null(),
                &mut manager,
                &mut installation,
            )
        },
        cudf_sys::CUDF_SUCCESS,
    );
    assert!(!manager.is_null());

    let mut exists = 0;
    let mut inspected = cudf_sys::cudf_memory_manager_options_t::default();
    assert_eq!(
        unsafe { cudf_sys::cudf_memory_manager_options_for_device(device, &mut exists, &mut inspected) },
        cudf_sys::CUDF_SUCCESS,
    );
    assert_eq!(exists, 1);
    assert_eq!(inspected.allocation_limit_bytes, LIMIT as u64);

    let mut conflicting = options;
    conflicting.allocation_limit_bytes = (2 * LIMIT) as u64;
    let mut conflicting_manager = ptr::null_mut();
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_install(
                device,
                &conflicting,
                None,
                ptr::null_mut(),
                ptr::null(),
                &mut conflicting_manager,
                &mut installation,
            )
        },
        cudf_sys::CUDF_ERROR_RUNTIME_BINDING_MISMATCH,
    );
    assert!(conflicting_manager.is_null());
    assert_eq!(installation.cause, cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_OPTIONS_MISMATCH);
    assert_eq!(installation.effective_options.allocation_limit_bytes, LIMIT as u64);

    let mut resource = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_memory_manager_resource(manager, &mut resource) }, cudf_sys::CUDF_SUCCESS,);
    assert!(!resource.is_null());

    let mut installation_exists = 0;
    let mut _failure_callback_installed = 0;
    let mut current_resource_installed = 0;
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_installation_state(
                device,
                &mut installation_exists,
                &mut _failure_callback_installed,
                &mut current_resource_installed,
            )
        },
        cudf_sys::CUDF_SUCCESS,
    );
    assert_eq!(installation_exists, 1, "the manager must have installed a device-domain resource");
    assert_eq!(current_resource_installed, 1, "the manager limit must be installed as the device current resource");

    let mut stream = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut stream) }, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let mut manager_route = ptr::null_mut();
    assert_eq!(
        unsafe { cudf_sys::cudf_memory_manager_route_current_resource(manager, stream, resource, &mut manager_route) },
        cudf_sys::CUDF_SUCCESS,
    );
    assert!(
        !manager_route.is_null(),
        "ordinary allocations must exercise the installed manager resource through its explicit stream route"
    );

    let exec = cudf_sys::cudf_exec_options_t {
        device,
        flags: cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
        stream,
        mr: resource,
        ..Default::default()
    };

    let mut domain = ptr::null_mut();
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_create_query_memory_domain(
                manager,
                7_001,
                FIRST_ALLOCATION,
                FIRST_ALLOCATION,
                &mut domain,
            )
        },
        cudf_sys::CUDF_SUCCESS,
    );
    assert!(!domain.is_null());

    let mut domain_stream = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut domain_stream) }, cudf_sys::CUDF_SUCCESS);
    assert!(!domain_stream.is_null());

    let mut reservation = ptr::null_mut();
    let mut domain_resource = ptr::null_mut();
    assert_eq!(
        unsafe {
            cudf_sys::cudf_query_memory_domain_attach_stream(
                domain,
                domain_stream,
                7_002,
                FIRST_ALLOCATION,
                &mut reservation,
                &mut domain_resource,
            )
        },
        cudf_sys::CUDF_SUCCESS,
    );
    assert!(!reservation.is_null());
    assert!(!domain_resource.is_null());
    let domain_exec = cudf_sys::cudf_exec_options_t { stream: domain_stream, mr: domain_resource, ..exec };

    // A private query-domain allocation consumes the same manager partition
    // before an ordinary allocation observes it. This is the direction the
    // former direct-manager-only regression did not cover.
    let mut first = ptr::null_mut();
    let first_status = unsafe { cudf_sys::cudf_device_buffer_create_ex(FIRST_ALLOCATION, &domain_exec, &mut first) };
    let first_error = unsafe {
        let message = cudf_sys::cudf_last_error_message();
        (!message.is_null()).then(|| CStr::from_ptr(message).to_string_lossy().into_owned())
    };
    assert_eq!(first_status, cudf_sys::CUDF_SUCCESS, "{first_error:?}");
    assert!(!first.is_null());

    let mut second = ptr::null_mut();
    assert_eq!(
        unsafe { cudf_sys::cudf_device_buffer_create_ex(SECOND_ALLOCATION, &exec, &mut second) },
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY,
    );
    assert!(second.is_null());

    assert_eq!(unsafe { cudf_sys::cudf_device_buffer_destroy(first) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(
        unsafe { cudf_sys::cudf_device_buffer_create_ex(SECOND_ALLOCATION, &exec, &mut second) },
        cudf_sys::CUDF_SUCCESS,
    );
    assert!(!second.is_null());
    assert_eq!(unsafe { cudf_sys::cudf_device_buffer_destroy(second) }, cudf_sys::CUDF_SUCCESS);

    // Two private pools must race through one manager-wide live-byte gate.
    // Each request fits its own domain but their combined live bytes exceed
    // the static partition, so exactly one may win and the full partition
    // must be reusable once that allocation is released.
    let mut left_domain = ptr::null_mut();
    let mut right_domain = ptr::null_mut();
    for (domain_id, domain) in [(7_003, &mut left_domain), (7_004, &mut right_domain)] {
        assert_eq!(
            unsafe {
                cudf_sys::cudf_memory_manager_create_query_memory_domain(
                    manager,
                    domain_id,
                    FIRST_ALLOCATION,
                    FIRST_ALLOCATION,
                    domain,
                )
            },
            cudf_sys::CUDF_SUCCESS,
        );
        assert!(!domain.is_null());
    }

    let mut left_stream = ptr::null_mut();
    let mut right_stream = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut left_stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut right_stream) }, cudf_sys::CUDF_SUCCESS);
    let mut left_reservation = ptr::null_mut();
    let mut right_reservation = ptr::null_mut();
    let mut left_resource = ptr::null_mut();
    let mut right_resource = ptr::null_mut();
    for (domain, stream, reservation_id, reservation, resource) in [
        (left_domain, left_stream, 7_005, &mut left_reservation, &mut left_resource),
        (right_domain, right_stream, 7_006, &mut right_reservation, &mut right_resource),
    ] {
        assert_eq!(
            unsafe {
                cudf_sys::cudf_query_memory_domain_attach_stream(
                    domain,
                    stream,
                    reservation_id,
                    FIRST_ALLOCATION,
                    reservation,
                    resource,
                )
            },
            cudf_sys::CUDF_SUCCESS,
        );
    }

    let barrier = Arc::new(Barrier::new(3));
    let spawn_racer = |stream: usize, resource: usize| {
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            let exec = cudf_sys::cudf_exec_options_t {
                device,
                flags: cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
                stream: stream as cudf_sys::cudf_stream_handle,
                mr: resource as cudf_sys::cudf_memory_resource_handle,
                ..Default::default()
            };
            barrier.wait();
            let mut buffer = ptr::null_mut();
            let status = unsafe { cudf_sys::cudf_device_buffer_create_ex(FIRST_ALLOCATION, &exec, &mut buffer) };
            (status, buffer as usize)
        })
    };
    let left_racer = spawn_racer(left_stream as usize, left_resource as usize);
    let right_racer = spawn_racer(right_stream as usize, right_resource as usize);
    barrier.wait();
    let race_results = [left_racer.join().unwrap(), right_racer.join().unwrap()];
    assert_eq!(
        race_results.iter().filter(|(status, _)| *status == cudf_sys::CUDF_SUCCESS).count(),
        1,
        "only one private-domain allocation may fit within the shared static partition: {race_results:?}"
    );
    assert_eq!(
        race_results.iter().filter(|(status, _)| *status == cudf_sys::CUDF_ERROR_OUT_OF_MEMORY).count(),
        1,
        "the losing private domain must receive an ordinary OOM: {race_results:?}"
    );
    for (_, buffer) in race_results {
        if buffer != 0 {
            assert_eq!(
                unsafe { cudf_sys::cudf_device_buffer_destroy(buffer as cudf_sys::cudf_device_buffer_handle) },
                cudf_sys::CUDF_SUCCESS,
            );
        }
    }

    let mut full_recovery = ptr::null_mut();
    assert_eq!(
        unsafe { cudf_sys::cudf_device_buffer_create_ex(LIMIT, &exec, &mut full_recovery) },
        cudf_sys::CUDF_SUCCESS,
        "releasing both domains must restore the full manager partition"
    );
    assert_eq!(unsafe { cudf_sys::cudf_device_buffer_destroy(full_recovery) }, cudf_sys::CUDF_SUCCESS);

    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_destroy(left_resource, device) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_destroy(right_resource, device) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_reservation_detach(left_reservation) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_reservation_detach(right_reservation) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_query_memory_domain_destroy(left_domain) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_query_memory_domain_destroy(right_domain) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(left_stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(right_stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_destroy(domain_resource, device) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_reservation_detach(reservation) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_query_memory_domain_destroy(domain) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(domain_stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_current_resource_route_destroy(manager_route) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_destroy(resource, device) }, cudf_sys::CUDF_SUCCESS);
}
