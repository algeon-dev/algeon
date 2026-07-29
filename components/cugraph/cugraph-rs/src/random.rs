//! Random number generator state for cuGraph algorithms.
//!
//! Some cuGraph algorithms (e.g., Force Atlas 2, Leiden, random walks) require
//! a random number generator state. This module provides a safe RAII wrapper
//! around `cugraph_rng_state_t`.
//!
//! # Example
//!
//! ```no_run
//! use cudf::Runtime;
//! use cugraph::{ResourceHandle, random::RngState};
//!
//! let runtime = std::sync::Arc::new(Runtime::new().unwrap());
//! let handle =
//!     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
//! let entered = handle.enter("RngState docs").unwrap();
//! let rng = RngState::new(&entered, 42).unwrap();
//! ```

use std::{marker::PhantomData, ptr};

use cugraph_sys::{cugraph_error_t, cugraph_rng_state_create, cugraph_rng_state_free, cugraph_rng_state_t};

use crate::{
    cuda::DeviceGuard,
    error::{Result, check_error, require_non_null},
    handle::EnteredContext,
};

/// RAII wrapper around `cugraph_rng_state_t`.
///
/// The RNG state is used by stochastic algorithms and is updated in-place
/// with each call. Not thread-safe — must be used from the thread that
/// created it.
pub struct RngState {
    raw: *mut cugraph_rng_state_t,
    device: i32,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for RngState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RngState").finish_non_exhaustive()
    }
}

impl RngState {
    /// Create a new RNG state with the given seed.
    ///
    /// Different seeds produce different random sequences. In multi-GPU
    /// settings, each GPU should use a different seed.
    pub fn new(handle: &EnteredContext<'_>, seed: u64) -> Result<Self> {
        let mut raw: *mut cugraph_rng_state_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `handle` outlives the call and `raw`/`error` are valid
        // out-pointers for cuGraph to initialize.
        unsafe {
            let code = cugraph_rng_state_create(handle.handle_ptr(), seed, &mut raw, &mut error);
            check_error(code, error, "RngState::new")?;
        }

        let raw = require_non_null(raw, "RngState::new", "rng_state")?;

        Ok(Self { raw, device: handle.device(), _nosend_nosync: PhantomData })
    }

    /// Get the raw C pointer. For internal use by other modules.
    pub(crate) fn as_ptr(&mut self) -> *mut cugraph_rng_state_t {
        self.raw
    }
}

impl Drop for RngState {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "RngState::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
            unsafe { cugraph_rng_state_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::*;
    use crate::ResourceHandle;

    #[test]
    fn test_rng_state_creation() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_rng_state_creation").expect("entered");
        let rng = RngState::new(&entered, 42).expect("rng state");
        assert!(!rng.raw.is_null());
    }

    #[test]
    fn test_rng_state_different_seeds() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_rng_state_different_seeds").expect("entered");
        let _rng1 = RngState::new(&entered, 0).expect("rng state 0");
        let _rng2 = RngState::new(&entered, u64::MAX).expect("rng state max");
    }

    #[test]
    fn test_rng_state_not_send_sync() {
        assert_not_impl_any!(RngState: Send, Sync);
    }
}
