//! CUDA device identification.
//!
//! [`DeviceId`] is a validated GPU ordinal. Use it with
//! [`RuntimeBuilder::with_device`](super::RuntimeBuilder::with_device) to pin
//! a [`Runtime`](super::Runtime) — and all resources it creates — to a
//! specific GPU.
//!
//! The device is a first-class part of the execution contract: when a
//! `Runtime` pinned to device *N* is used for an operation, the FFI layer
//! re-enters device *N* before any GPU work is enqueued. See [`super::Runtime`]
//! for details.

use std::sync::OnceLock;

use crate::{
    core::error::{self, Error, ErrorCode, Result},
    device::inventory::{self, DeviceMemory},
};

/// A validated CUDA device ordinal.
///
/// Create one with [`DeviceId::new`] (validates against the device count) or
/// [`DeviceId::current`] (queries the runtime's active device).
///
/// # Examples
///
/// ```no_run
/// use cudf::DeviceId;
///
/// let gpu0 = DeviceId::new(0).unwrap();
/// assert_eq!(gpu0.ordinal(), 0);
///
/// let current = DeviceId::current().unwrap();
/// assert_eq!(current, DeviceId::DEFAULT);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(i32);

/// Cached device count — queried once, valid for process lifetime.
static DEVICE_COUNT_CACHE: OnceLock<usize> = OnceLock::new();

impl DeviceId {
    /// The default GPU (device ordinal 0).
    pub const DEFAULT: Self = Self(0);

    /// Create a `DeviceId` from a zero-based ordinal, validating that the
    /// device exists.
    ///
    /// Returns an error if `ordinal >= DeviceId::count()`.
    pub fn new(ordinal: usize) -> Result<Self> {
        let count = Self::count()?;
        if ordinal >= count {
            return Err(Error::new_with(ErrorCode::IndexOutOfBounds, "DeviceId::new", "device ordinal out of range")
                .with_detail("resource", "cuda_device")
                .with_detail("ordinal", ordinal.to_string())
                .with_detail("device_count", count.to_string())
                .with_detail("reason", "device_ordinal_out_of_range")
                .with_detail("remedy", "choose a device ordinal less than DeviceId::count()"));
        }
        Ok(Self(ordinal as i32))
    }

    /// Query the calling thread's current CUDA device.
    pub fn current() -> Result<Self> {
        let mut device: i32 = 0;
        error::check_status_in(unsafe { cudf_sys::cudf_cuda_get_device(&mut device) }, "DeviceId::current")?;
        Ok(Self(device))
    }

    /// Create a device id from a native callback payload that already came
    /// from cudf's device-bound resource path.
    pub(crate) const fn from_validated_ordinal_raw(ordinal: i32) -> Self {
        Self(ordinal)
    }

    /// Return the number of CUDA-capable devices visible to the process.
    ///
    /// The result is cached after the first call (device count does not change
    /// during a process's lifetime).
    pub fn count() -> Result<usize> {
        if let Some(&cached) = DEVICE_COUNT_CACHE.get() {
            return Ok(cached);
        }
        let mut count: i32 = 0;
        error::check_status_in(unsafe { cudf_sys::cudf_cuda_get_device_count(&mut count) }, "DeviceId::count")?;
        let n = count as usize;
        // If another thread raced us, that's fine — same value.
        let _ = DEVICE_COUNT_CACHE.set(n);
        Ok(n)
    }

    /// Query the device's currently available and total memory.
    pub fn available_memory(self) -> Result<DeviceMemory> {
        inventory::available_memory(self)
    }

    /// Query the device's CUDA streaming multiprocessor count.
    pub fn multiprocessor_count(self) -> Result<u32> {
        inventory::multiprocessor_count(self)
    }

    /// Query the device's CUDA PCI bus id.
    pub fn pci_bus_id(self) -> Result<String> {
        inventory::pci_bus_id(self)
    }

    /// The zero-based device ordinal.
    pub fn ordinal(self) -> usize {
        self.0 as usize
    }

    /// Raw i32 ordinal for FFI calls.
    pub(crate) fn ordinal_raw(self) -> i32 {
        self.0
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "device:{}", self.0)
    }
}
