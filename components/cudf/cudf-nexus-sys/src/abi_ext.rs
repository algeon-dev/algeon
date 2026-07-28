//! Rust-only initialization helpers for size-tagged C ABI structs.

#![deny(unsafe_op_in_unsafe_fn)]

use crate::{
    CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS, CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL, CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC,
    CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS, CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS,
    CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS, cudf_exec_options_t, cudf_memory_failure_info_t,
    cudf_memory_manager_options_t, cudf_memory_pool_usage_t, cudf_memory_resource_usage_t,
};

/// Number of reserved execution-option slots in the current C ABI.
pub const CUDF_EXEC_OPTIONS_RESERVED_SLOTS: usize = 4;

impl cudf_memory_resource_usage_t {
    /// Return a zeroed, size-tagged resource-usage result buffer.
    pub const fn defaults() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>(),
            allocation_limit_bytes: 0,
            current_bytes: 0,
            peak_bytes: 0,
            total_bytes: 0,
            current_aligned_bytes: 0,
            peak_aligned_bytes: 0,
            total_allocations: 0,
            current_allocations: 0,
            peak_allocations: 0,
            failed_allocation_count: 0,
            largest_failed_allocation_bytes: 0,
            reserved: [0; CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS as usize],
        }
    }
}

impl Default for cudf_memory_resource_usage_t {
    fn default() -> Self {
        Self::defaults()
    }
}

impl cudf_memory_pool_usage_t {
    /// Return a zeroed, size-tagged pool-usage result buffer.
    pub const fn defaults() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>(),
            reserved_current_bytes: 0,
            reserved_high_bytes: 0,
            used_current_bytes: 0,
            used_high_bytes: 0,
            reserved: [0; CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS as usize],
        }
    }
}

impl Default for cudf_memory_pool_usage_t {
    fn default() -> Self {
        Self::defaults()
    }
}

impl cudf_memory_failure_info_t {
    /// Return a zeroed, size-tagged allocation-failure payload.
    pub const fn defaults() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>(),
            device_ordinal: -1,
            flags: 0,
            callback_attempt: 0,
            trim_retry_count: 0,
            requested_bytes: 0,
            pool_reserved_current_bytes: 0,
            pool_reserved_high_bytes: 0,
            pool_used_current_bytes: 0,
            pool_used_high_bytes: 0,
            failure_source: CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL,
            reserved_failure_info_padding: 0,
            reserved: [0; CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS as usize - 1],
        }
    }
}

impl Default for cudf_memory_failure_info_t {
    fn default() -> Self {
        Self::defaults()
    }
}

impl cudf_memory_manager_options_t {
    /// Return the ABI-default memory-manager configuration.
    pub const fn defaults() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>(),
            base_kind: CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC,
            flags: 0,
            initial_pool_size_bytes: 0,
            maximum_pool_size_bytes: 0,
            release_threshold_bytes: u64::MAX,
            reserved: [0; CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS as usize],
        }
    }
}

impl Default for cudf_memory_manager_options_t {
    fn default() -> Self {
        Self::defaults()
    }
}

impl cudf_exec_options_t {
    /// Return a zeroed, size-tagged execution-options payload.
    pub const fn defaults() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>(),
            device: -1,
            flags: 0,
            stream: core::ptr::null_mut(),
            mr: core::ptr::null_mut(),
            reserved: [0; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
        }
    }
}

impl Default for cudf_exec_options_t {
    fn default() -> Self {
        Self::defaults()
    }
}
