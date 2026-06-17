//! Per-device CUDA inventory facts.

use std::{ffi::CStr, num::TryFromIntError};

use crate::{
    core::error::{self, Error, ErrorCode, Result},
    runtime::DeviceId,
};

/// Snapshot of CUDA memory currently reported for one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Bytes available to the CUDA context at the time of the query.
    pub free_bytes: u64,
    /// Total bytes reported by the CUDA device.
    pub total_bytes: u64,
}

const CUDA_PCI_BUS_ID_BUFFER_LEN: usize = 32;

pub(crate) fn available_memory(device: DeviceId) -> Result<DeviceMemory> {
    let mut free_bytes = 0usize;
    let mut total_bytes = 0usize;
    error::check_status_in(
        unsafe { cudf_sys::cudf_device_memory_info(device.ordinal_raw(), &mut free_bytes, &mut total_bytes) },
        "DeviceId::available_memory",
    )?;
    Ok(DeviceMemory { free_bytes: free_bytes as u64, total_bytes: total_bytes as u64 })
}

pub(crate) fn pci_bus_id(device: DeviceId) -> Result<String> {
    let mut out = [0; CUDA_PCI_BUS_ID_BUFFER_LEN];
    error::check_status_in(
        unsafe { cudf_sys::cudf_cuda_device_pci_bus_id(device.ordinal_raw(), out.as_mut_ptr(), out.len()) },
        "DeviceId::pci_bus_id",
    )?;
    let bus_id = unsafe { CStr::from_ptr(out.as_ptr()) };
    bus_id.to_str().map(str::to_owned).map_err(|source| {
        error::foreign_invariant_violation(
            "DeviceId::pci_bus_id",
            "CUDA PCI bus id is not valid UTF-8",
            "cuda_pci_bus_id",
            "invalid_utf8",
            "ensure the CUDA runtime returns a valid PCI bus id string",
        )
        .with_source(source)
    })
}

pub(crate) fn multiprocessor_count(device: DeviceId) -> Result<u32> {
    let value = device_attribute(
        device,
        cudf_sys::CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        "DeviceId::multiprocessor_count",
    )?;
    u32::try_from(value).map_err(|error| device_attribute_range_error("DeviceId::multiprocessor_count", value, error))
}

pub(crate) fn device_attribute(device: DeviceId, attr: i32, operation: &'static str) -> Result<i32> {
    let mut out = 0i32;
    error::check_status_in(
        unsafe { cudf_sys::cudf_device_attribute(device.ordinal_raw(), attr, &mut out) },
        operation,
    )?;
    Ok(out)
}

fn device_attribute_range_error(operation: &'static str, value: i32, source: TryFromIntError) -> Error {
    Error::new_with(
        ErrorCode::ForeignInvariantViolation,
        operation,
        "CUDA device attribute returned a value outside the expected range",
    )
    .with_detail("resource", "cuda_device_attribute")
    .with_detail("value", value.to_string())
    .with_detail("reason", "negative_unsigned_attribute")
    .with_detail("remedy", "report the CUDA runtime attribute contract violation")
    .with_source(source)
}
