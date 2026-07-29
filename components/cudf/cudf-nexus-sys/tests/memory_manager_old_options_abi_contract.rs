use std::{mem, ptr};

#[repr(C)]
struct OriginalMemoryManagerOptions {
    struct_size: usize,
    base_kind: cudf_sys::cudf_memory_manager_base_kind_t,
    flags: u32,
    initial_pool_size_bytes: u64,
    maximum_pool_size_bytes: u64,
    release_threshold_bytes: u64,
    reserved: [u64; 4],
}

#[test]
fn zero_initialized_original_options_layout_remains_unbounded() {
    assert_eq!(mem::size_of::<OriginalMemoryManagerOptions>(), 72);
    assert_eq!(mem::size_of::<cudf_sys::cudf_memory_manager_options_t>(), 72);

    let mut device = -1;
    assert_eq!(unsafe { cudf_sys::cudf_cuda_get_device(&mut device) }, cudf_sys::CUDF_SUCCESS);
    let original = OriginalMemoryManagerOptions {
        struct_size: mem::size_of::<OriginalMemoryManagerOptions>(),
        base_kind: cudf_sys::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC,
        flags: 0,
        initial_pool_size_bytes: 0,
        maximum_pool_size_bytes: 0,
        release_threshold_bytes: u64::MAX,
        reserved: [0; 4],
    };
    let mut manager = ptr::null_mut();
    assert_eq!(
        unsafe {
            cudf_sys::cudf_memory_manager_for_device_with_options(device, ptr::from_ref(&original).cast(), &mut manager)
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
    assert_eq!(inspected.allocation_limit_bytes, 0);
}
