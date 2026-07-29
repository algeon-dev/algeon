use std::ptr;

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
    assert_eq!(
        unsafe { cudf_sys::cudf_memory_manager_for_device_with_options(device, &invalid, &mut manager) },
        cudf_sys::CUDF_ERROR_INVALID_ARGUMENT,
    );
    assert!(manager.is_null());

    let options =
        cudf_sys::cudf_memory_manager_options_t { allocation_limit_bytes: LIMIT as u64, ..Default::default() };
    assert_eq!(
        unsafe { cudf_sys::cudf_memory_manager_for_device_with_options(device, &options, &mut manager) },
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
            cudf_sys::cudf_memory_manager_for_device_with_options(device, &conflicting, &mut conflicting_manager)
        },
        cudf_sys::CUDF_ERROR_INVALID_OPERATION,
    );
    assert!(conflicting_manager.is_null());

    let mut resource = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_memory_manager_resource(manager, &mut resource) }, cudf_sys::CUDF_SUCCESS,);
    assert!(!resource.is_null());
    let mut stream = ptr::null_mut();
    assert_eq!(unsafe { cudf_sys::cudf_stream_create(&mut stream) }, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let exec = cudf_sys::cudf_exec_options_t {
        device,
        flags: cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
        stream,
        mr: resource,
        ..Default::default()
    };

    let mut first = ptr::null_mut();
    assert_eq!(
        unsafe { cudf_sys::cudf_device_buffer_create_ex(FIRST_ALLOCATION, &exec, &mut first) },
        cudf_sys::CUDF_SUCCESS,
    );
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
    assert_eq!(unsafe { cudf_sys::cudf_stream_destroy(stream) }, cudf_sys::CUDF_SUCCESS);
    assert_eq!(unsafe { cudf_sys::cudf_memory_resource_destroy(resource, device) }, cudf_sys::CUDF_SUCCESS);
}
