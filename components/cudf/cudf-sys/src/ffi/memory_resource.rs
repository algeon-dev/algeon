use std::os::raw::c_void;

use crate::{
    cudf_memory_failure_callback_t, cudf_memory_failure_callback_v2_t, cudf_memory_manager_handle,
    cudf_memory_manager_options_t, cudf_memory_pool_usage_t, cudf_memory_resource_handle, cudf_memory_resource_usage_t,
    cudf_query_memory_domain_handle, cudf_reservation_arena_handle, cudf_reservation_handle, cudf_status_t,
    cudf_stream_handle,
};

extern "C" {
    // -- Memory Resource -----------------------------------------------------

    /// Create an owned CUDA-backed RMM memory resource handle.
    ///
    /// Attach this handle to [`crate::ExecutionContext`] or
    /// [`crate::cudf_exec_options_t`] when calling `_ex` APIs to direct
    /// allocations through it.
    pub fn cudf_memory_resource_create_cuda(out: *mut cudf_memory_resource_handle) -> cudf_status_t;

    /// Create an owned pool memory resource handle.
    ///
    /// Like [`cudf_memory_resource_create_cuda`], this creates an owned
    /// resource object without changing the process-global allocator.
    pub fn cudf_memory_resource_create_pool(
        initial_pool_size: usize,
        maximum_pool_size: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Create an owned cudaMallocAsync-backed memory resource handle.
    pub fn cudf_memory_resource_create_cuda_async(
        initial_pool_size: usize,
        release_threshold: u64,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Create an owned limiting memory resource handle around `upstream`.
    pub fn cudf_memory_resource_create_limiting(
        upstream: cudf_memory_resource_handle,
        allocation_limit: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Create an owned failure-callback memory resource handle around `upstream`.
    pub fn cudf_memory_resource_create_failure_callback(
        upstream: cudf_memory_resource_handle,
        callback: cudf_memory_failure_callback_t,
        callback_arg: *mut c_void,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Create an owned reclaim-capable failure-callback memory resource handle around `upstream`.
    pub fn cudf_memory_resource_create_failure_callback_v2(
        upstream: cudf_memory_resource_handle,
        callback: cudf_memory_failure_callback_v2_t,
        callback_arg: *mut c_void,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Create an owned budget-tracking memory resource handle around `upstream`.
    pub fn cudf_memory_resource_create_budget_tracking(
        upstream: cudf_memory_resource_handle,
        allocation_limit: usize,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Read usage counters from a budget-tracking memory resource.
    pub fn cudf_memory_resource_usage(
        handle: cudf_memory_resource_handle,
        out: *mut cudf_memory_resource_usage_t,
    ) -> cudf_status_t;

    /// Read physical CUDA async-pool counters from an async-pool-backed resource.
    pub fn cudf_memory_resource_pool_usage(
        handle: cudf_memory_resource_handle,
        out: *mut cudf_memory_pool_usage_t,
    ) -> cudf_status_t;

    /// Explicitly trim idle bytes from an async-pool-backed resource.
    pub fn cudf_memory_resource_trim_to(handle: cudf_memory_resource_handle, keep_bytes: u64) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA device.
    pub fn cudf_memory_manager_for_device(device: i32, out: *mut cudf_memory_manager_handle) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA device with options.
    pub fn cudf_memory_manager_for_device_with_options(
        device: i32,
        options: *const cudf_memory_manager_options_t,
        out: *mut cudf_memory_manager_handle,
    ) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA
    /// device, installing a process-lifetime failure callback if this call
    /// creates the manager.
    pub fn cudf_memory_manager_for_device_with_failure_callback(
        device: i32,
        callback: cudf_memory_failure_callback_t,
        callback_arg: *mut c_void,
        out: *mut cudf_memory_manager_handle,
    ) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA
    /// device, installing a reclaim-capable process-lifetime failure callback
    /// if this call creates the manager.
    pub fn cudf_memory_manager_for_device_with_failure_callback_v2(
        device: i32,
        callback: cudf_memory_failure_callback_v2_t,
        callback_arg: *mut c_void,
        out: *mut cudf_memory_manager_handle,
    ) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA
    /// device with options and a process-lifetime failure callback.
    pub fn cudf_memory_manager_for_device_with_failure_callback_and_options(
        device: i32,
        callback: cudf_memory_failure_callback_t,
        callback_arg: *mut c_void,
        options: *const cudf_memory_manager_options_t,
        out: *mut cudf_memory_manager_handle,
    ) -> cudf_status_t;

    /// Resolve the process-local reservation-aware memory manager for one CUDA
    /// device with options and a reclaim-capable process-lifetime failure callback.
    pub fn cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
        device: i32,
        callback: cudf_memory_failure_callback_v2_t,
        callback_arg: *mut c_void,
        options: *const cudf_memory_manager_options_t,
        out: *mut cudf_memory_manager_handle,
    ) -> cudf_status_t;

    /// Query whether a device memory manager exists and whether it has a failure callback.
    pub fn cudf_memory_manager_failure_callback_state(
        device: i32,
        exists: *mut i32,
        failure_callback_installed: *mut i32,
    ) -> cudf_status_t;

    /// Create a memory-resource handle alias for a device memory manager.
    pub fn cudf_memory_manager_resource(
        manager: cudf_memory_manager_handle,
        out: *mut cudf_memory_resource_handle,
    ) -> cudf_status_t;

    /// Read physical CUDA async-pool counters for a device memory manager.
    pub fn cudf_memory_manager_pool_usage(
        manager: cudf_memory_manager_handle,
        out: *mut cudf_memory_pool_usage_t,
    ) -> cudf_status_t;

    /// Explicitly trim idle bytes from a device memory manager's async pool.
    pub fn cudf_memory_manager_trim_to(manager: cudf_memory_manager_handle, keep_bytes: u64) -> cudf_status_t;

    /// Attach a hard live-byte reservation to a CUDA stream.
    pub fn cudf_memory_manager_attach_reservation(
        manager: cudf_memory_manager_handle,
        stream: cudf_stream_handle,
        reservation_id: u64,
        limit_bytes: usize,
        out: *mut cudf_reservation_handle,
    ) -> cudf_status_t;

    /// Create a query-scoped reservation arena for a memory manager.
    pub fn cudf_memory_manager_create_reservation_arena(
        manager: cudf_memory_manager_handle,
        arena_id: u64,
        limit_bytes: usize,
        out: *mut cudf_reservation_arena_handle,
    ) -> cudf_status_t;

    /// Attach a stream-bound reservation as an arena member.
    pub fn cudf_memory_manager_attach_reservation_to_arena(
        manager: cudf_memory_manager_handle,
        stream: cudf_stream_handle,
        arena: cudf_reservation_arena_handle,
        reservation_id: u64,
        limit_bytes: usize,
        out: *mut cudf_reservation_handle,
    ) -> cudf_status_t;

    /// Read usage counters for a live stream-bound reservation.
    pub fn cudf_reservation_usage(
        reservation: cudf_reservation_handle,
        out: *mut cudf_memory_resource_usage_t,
    ) -> cudf_status_t;

    /// Resize the live-byte limit for an attached reservation.
    pub fn cudf_reservation_resize(reservation: cudf_reservation_handle, limit_bytes: usize) -> cudf_status_t;

    /// Detach a reservation from its stream and destroy the reservation handle.
    pub fn cudf_reservation_detach(reservation: cudf_reservation_handle) -> cudf_status_t;

    /// Read usage counters for a live reservation arena.
    pub fn cudf_reservation_arena_usage(
        arena: cudf_reservation_arena_handle,
        out: *mut cudf_memory_resource_usage_t,
    ) -> cudf_status_t;

    /// Resize the live-byte limit for a reservation arena.
    pub fn cudf_reservation_arena_resize(arena: cudf_reservation_arena_handle, limit_bytes: usize) -> cudf_status_t;

    /// Destroy a reservation arena handle.
    pub fn cudf_reservation_arena_destroy(arena: cudf_reservation_arena_handle) -> cudf_status_t;

    /// Create an opt-in query memory domain for a memory manager.
    pub fn cudf_memory_manager_create_query_memory_domain(
        manager: cudf_memory_manager_handle,
        domain_id: u64,
        grant_bytes: usize,
        ceiling_bytes: usize,
        release_threshold_bytes: u64,
        out: *mut cudf_query_memory_domain_handle,
    ) -> cudf_status_t;

    /// Attach a stream to a query memory domain.
    pub fn cudf_query_memory_domain_attach_stream(
        domain: cudf_query_memory_domain_handle,
        stream: cudf_stream_handle,
        reservation_id: u64,
        out: *mut cudf_reservation_handle,
    ) -> cudf_status_t;

    /// Read usage counters for a query memory domain.
    pub fn cudf_query_memory_domain_usage(
        domain: cudf_query_memory_domain_handle,
        out: *mut cudf_memory_resource_usage_t,
    ) -> cudf_status_t;

    /// Read physical CUDA async-pool counters for a query memory domain.
    pub fn cudf_query_memory_domain_pool_usage(
        domain: cudf_query_memory_domain_handle,
        out: *mut cudf_memory_pool_usage_t,
    ) -> cudf_status_t;

    /// Resize a query memory domain's live-byte grant.
    pub fn cudf_query_memory_domain_resize_grant(
        domain: cudf_query_memory_domain_handle,
        grant_bytes: usize,
    ) -> cudf_status_t;

    /// Trim a query memory domain's idle pool pages.
    pub fn cudf_query_memory_domain_trim(domain: cudf_query_memory_domain_handle, keep_bytes: u64) -> cudf_status_t;

    /// Destroy a query memory domain handle.
    pub fn cudf_query_memory_domain_destroy(domain: cudf_query_memory_domain_handle) -> cudf_status_t;

    /// Destroy a memory resource created by this shim on its owning device.
    pub fn cudf_memory_resource_destroy(handle: cudf_memory_resource_handle, device: i32) -> cudf_status_t;

    /// Borrow the underlying raw `device_memory_resource*` from `handle`.
    ///
    /// This does not transfer ownership. The returned pointer is valid only
    /// while `handle` remains alive.
    pub fn cudf_memory_resource_raw(handle: cudf_memory_resource_handle, out: *mut *mut c_void) -> cudf_status_t;
}
