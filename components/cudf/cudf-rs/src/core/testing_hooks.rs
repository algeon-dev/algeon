use super::{error::Result, memory::DeviceMemoryResource, types};

pub(crate) fn stats_memory_resource() -> Result<DeviceMemoryResource> {
    DeviceMemoryResource::__test_stats()
}

pub(crate) fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    types::validity_to_ffi_bitmask(validity)
}
