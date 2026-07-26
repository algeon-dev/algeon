use std::{ptr::NonNull, slice};

use crate::{
    core::{Error, ErrorCode, Result, error},
    testing::ColumnView,
};

/// Host allocation kind used by the explicit D2H materialization benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D2hBenchmarkHostMemory {
    /// Ordinary pageable Rust heap memory.
    Pageable,
    /// CUDA page-locked host memory.
    Pinned,
}

enum D2hBenchmarkStorage {
    Pageable(Vec<u8>),
    Pinned { ptr: NonNull<u8>, capacity: usize },
}

/// Reusable host destination for explicit D2H measurement.
///
/// This unstable testing type exists for examples and external measurement
/// tools. It is not a production materialization API.
pub struct D2hBenchmarkBuffer {
    storage: D2hBenchmarkStorage,
    initialized_len: usize,
}

impl D2hBenchmarkBuffer {
    /// Allocate one reusable benchmark destination.
    pub fn new(kind: D2hBenchmarkHostMemory, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                "testing::D2hBenchmarkBuffer::new",
                "D2H benchmark buffer capacity must be greater than zero",
            ));
        }
        let storage = match kind {
            D2hBenchmarkHostMemory::Pageable => D2hBenchmarkStorage::Pageable(vec![0u8; capacity]),
            D2hBenchmarkHostMemory::Pinned => {
                let mut raw = std::ptr::null_mut();
                error::check_status_in(
                    unsafe { cudf_sys::cudf_pinned_host_alloc(capacity, &mut raw) },
                    "testing::D2hBenchmarkBuffer::new",
                )?;
                let ptr = NonNull::new(raw.cast::<u8>()).ok_or_else(|| {
                    Error::new_with(
                        ErrorCode::ForeignInvariantViolation,
                        "testing::D2hBenchmarkBuffer::new",
                        "pinned host allocation returned null",
                    )
                })?;
                D2hBenchmarkStorage::Pinned { ptr, capacity }
            },
        };
        Ok(Self { storage, initialized_len: 0 })
    }

    /// Allocation kind of this destination.
    pub fn kind(&self) -> D2hBenchmarkHostMemory {
        match self.storage {
            D2hBenchmarkStorage::Pageable(_) => D2hBenchmarkHostMemory::Pageable,
            D2hBenchmarkStorage::Pinned { .. } => D2hBenchmarkHostMemory::Pinned,
        }
    }

    /// Destination capacity in bytes.
    pub fn capacity(&self) -> usize {
        match &self.storage {
            D2hBenchmarkStorage::Pageable(bytes) => bytes.len(),
            D2hBenchmarkStorage::Pinned { capacity, .. } => *capacity,
        }
    }

    /// Copy one fixed-width column into this destination using production's
    /// synchronization and FFI-copy semantics.
    pub fn copy_from_fixed_width(&mut self, view: &ColumnView<'_>) -> Result<usize> {
        let capacity = self.capacity();
        let host_ptr = match &mut self.storage {
            D2hBenchmarkStorage::Pageable(bytes) => bytes.as_mut_ptr(),
            D2hBenchmarkStorage::Pinned { ptr, .. } => ptr.as_ptr(),
        };
        let copied = crate::core::column::column_view_copy_fixed_width_to_host_for_benchmark(view, host_ptr, capacity)?;
        self.initialized_len = copied;
        Ok(copied)
    }

    /// Bytes initialized by the most recent successful copy.
    pub fn initialized_bytes(&self) -> &[u8] {
        match &self.storage {
            D2hBenchmarkStorage::Pageable(bytes) => &bytes[..self.initialized_len],
            D2hBenchmarkStorage::Pinned { ptr, .. } => {
                // SAFETY: the allocation remains owned by self, and
                // initialized_len is set only from a successful bounded copy.
                unsafe { slice::from_raw_parts(ptr.as_ptr(), self.initialized_len) }
            },
        }
    }
}

impl Drop for D2hBenchmarkBuffer {
    fn drop(&mut self) {
        if let D2hBenchmarkStorage::Pinned { ptr, .. } = &self.storage {
            // SAFETY: ptr came from cudf_pinned_host_alloc and this object is
            // its sole owner.
            let _ = unsafe { cudf_sys::cudf_pinned_host_free(ptr.as_ptr().cast()) };
        }
    }
}

/// Construct a malformed non-empty STRING column view without an offsets
/// child so integration tests can lock down foreign-contract validation.
pub fn synthetic_string_view_without_children(size: usize) -> Result<ColumnView<'static>> {
    crate::core::column::synthetic_string_view_without_children(size)
}

/// Trigger the internal raw-byte materialization helper on a borrowed column
/// view so integration tests can lock down its downstream error contract.
pub fn column_view_to_raw_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_raw_bytes_for_test(view)
}

/// Trigger the internal fixed-width host-byte materialization helper on a
/// borrowed column view so integration tests can lock down its downstream
/// error contract.
pub fn column_view_to_fixed_width_host_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_fixed_width_host_bytes_for_test(view)
}

/// Trigger the fixed-width host-byte materialization overflow contract used by
/// integration tests.
pub fn trigger_column_view_fixed_width_host_byte_range_overflow_for_test() -> Result<()> {
    crate::core::column::trigger_fixed_width_host_byte_range_overflow_for_test()
}
