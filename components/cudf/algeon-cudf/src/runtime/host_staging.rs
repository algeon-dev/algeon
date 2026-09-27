//! Reusable page-locked (pinned) host staging slabs for batched
//! host-to-device uploads.
//!
//! Pinned host memory makes `cudaMemcpyAsync` a true asynchronous DMA:
//! pageable sources are staged through a driver-internal locked ring that
//! serializes concurrent copies process-wide, which measurably collapses
//! multi-query throughput. Pinning is expensive (page-table work scales with
//! bytes), so slabs are allocated lazily, retained for the process lifetime,
//! and recycled through a bounded pool instead of being allocated per upload.
//!
//! Sizing knobs (read once at first use):
//! - `CUDF_HOST_STAGING_SLAB_BYTES`: bytes per slab (default 16 MiB).
//! - `CUDF_HOST_STAGING_SLAB_COUNT`: maximum retained slabs (default 8).

use std::{
    ffi::c_void,
    mem,
    ptr::NonNull,
    sync::{Condvar, Mutex, OnceLock},
};

use crate::core::error::{self, ErrorCode, Result};

const DEFAULT_SLAB_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_SLAB_COUNT: usize = 8;
const MIN_SLAB_BYTES: usize = 1024 * 1024;
const MIN_SLAB_COUNT: usize = 2;

/// One owned pinned host allocation.
pub(crate) struct PinnedSlab {
    ptr: NonNull<u8>,
    capacity: usize,
}

// SAFETY: the slab exclusively owns a process-wide pinned host allocation
// with no thread affinity; moving ownership between threads is safe.
unsafe impl Send for PinnedSlab {}

impl PinnedSlab {
    fn alloc(capacity: usize) -> Result<Self> {
        let mut out: *mut c_void = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_pinned_host_alloc(capacity, &mut out) },
            "host_staging::pinned_slab_alloc",
        )?;
        Ok(Self {
            ptr: error::require_non_null(out, "host_staging::pinned_slab_alloc", "pinned host slab")?.cast(),
            capacity,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Copy `src` into the slab at byte `offset`.
    ///
    /// Callers must ensure no device copy issued from the same region is
    /// still pending (the pool lease plus a CUDA event provides that).
    pub(crate) fn write_at(&mut self, offset: usize, src: &[u8]) {
        assert!(
            offset.checked_add(src.len()).is_some_and(|end| end <= self.capacity),
            "pinned slab write out of bounds"
        );
        // SAFETY: bounds asserted above; the slab is exclusively leased, so
        // no concurrent host access aliases this region.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr().add(offset), src.len());
        }
    }
}

impl Drop for PinnedSlab {
    fn drop(&mut self) {
        // SAFETY: the pointer came from cudf_pinned_host_alloc and is
        // released exactly once.
        unsafe {
            let _ = cudf_sys::cudf_pinned_host_free(self.ptr.as_ptr().cast());
        }
    }
}

struct PoolState {
    free: Vec<PinnedSlab>,
    created: usize,
}

/// Bounded, lazily-filled pool of pinned staging slabs.
pub(crate) struct PinnedSlabPool {
    state: Mutex<PoolState>,
    available: Condvar,
    slab_bytes: usize,
    max_slabs: usize,
}

impl PinnedSlabPool {
    fn new(slab_bytes: usize, max_slabs: usize) -> Self {
        Self {
            state: Mutex::new(PoolState { free: Vec::new(), created: 0 }),
            available: Condvar::new(),
            slab_bytes,
            max_slabs,
        }
    }

    pub(crate) fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }

    /// Lease one slab, allocating lazily up to the pool cap and blocking
    /// while every slab is leased out.
    pub(crate) fn acquire(&'static self) -> Result<PinnedSlabLease> {
        let mut state = self.state.lock().map_err(|_poisoned| {
            error::Error::new_with(
                ErrorCode::InvariantViolation,
                "host_staging::acquire",
                "pinned staging pool mutex was poisoned",
            )
        })?;
        loop {
            if let Some(slab) = state.free.pop() {
                return Ok(PinnedSlabLease { pool: self, slab: Some(slab) });
            }
            if state.created < self.max_slabs {
                state.created += 1;
                drop(state);
                match PinnedSlab::alloc(self.slab_bytes) {
                    Ok(slab) => return Ok(PinnedSlabLease { pool: self, slab: Some(slab) }),
                    Err(error) => {
                        let mut state = self.state.lock().map_err(|_poisoned| {
                            error::Error::new_with(
                                ErrorCode::InvariantViolation,
                                "host_staging::acquire",
                                "pinned staging pool mutex was poisoned",
                            )
                        })?;
                        state.created -= 1;
                        drop(state);
                        self.available.notify_one();
                        return Err(error.reanchor("host_staging::acquire"));
                    },
                }
            }
            state = self.available.wait(state).map_err(|_poisoned| {
                error::Error::new_with(
                    ErrorCode::InvariantViolation,
                    "host_staging::acquire",
                    "pinned staging pool mutex was poisoned",
                )
            })?;
        }
    }

    #[cfg(test)]
    pub(crate) fn leased_count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.created - state.free.len()
    }

    fn release(&self, slab: PinnedSlab) {
        {
            let mut state = self.state.lock().expect("pinned staging pool mutex poisoned");
            state.free.push(slab);
        }
        self.available.notify_one();
    }
}

/// RAII lease over one pool slab.
pub(crate) struct PinnedSlabLease {
    pool: &'static PinnedSlabPool,
    slab: Option<PinnedSlab>,
}

impl PinnedSlabLease {
    /// Retain a source whose pending CUDA access could not be fenced, and
    /// restore pool capacity for uploads on unaffected devices.
    pub(crate) fn quarantine(&mut self) {
        if let Some(slab) = self.slab.take() {
            mem::forget(slab);
            if let Ok(mut state) = self.pool.state.lock() {
                state.created -= 1;
            }
            self.pool.available.notify_one();
        }
    }

    pub(crate) fn slab(&mut self) -> &mut PinnedSlab {
        self.slab.as_mut().expect("leased slab is present until drop")
    }
}

impl Drop for PinnedSlabLease {
    fn drop(&mut self) {
        if let Some(slab) = self.slab.take() {
            self.pool.release(slab);
        }
    }
}

fn env_usize(name: &str, default: usize, minimum: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map_or(default, |value| value.max(minimum))
}

/// Process-wide pinned staging pool.
pub(crate) fn pinned_staging_pool() -> &'static PinnedSlabPool {
    static POOL: OnceLock<PinnedSlabPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let slab_bytes = env_usize("CUDF_HOST_STAGING_SLAB_BYTES", DEFAULT_SLAB_BYTES, MIN_SLAB_BYTES);
        let max_slabs = env_usize("CUDF_HOST_STAGING_SLAB_COUNT", DEFAULT_SLAB_COUNT, MIN_SLAB_COUNT);
        PinnedSlabPool::new(slab_bytes, max_slabs)
    })
}

#[cfg(test)]
mod tests {
    use super::PinnedSlabPool;

    #[test]
    fn quarantined_source_does_not_consume_pool_capacity() {
        let pool = Box::leak(Box::new(PinnedSlabPool::new(1024, 2)));
        let mut first = pool.acquire().unwrap();
        let second = pool.acquire().unwrap();
        first.quarantine();
        let replacement = pool.acquire().unwrap();
        assert_eq!(pool.state.lock().unwrap().created, 2);
        drop(replacement);
        drop(second);
        drop(first);
        assert_eq!(pool.state.lock().unwrap().free.len(), 2);
    }
}
