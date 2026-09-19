//! Generated raw FFI bindings to the pinned cuVS 26.08 C API.

#![allow(clippy::allow_attributes)] // reason = "generated FFI bindings use allow attributes"

use std::{
    ffi::{c_char, c_int, c_uint, c_void},
    fmt,
};

pub const EXPECTED_CUVS_VERSION: (u16, u16, u16) = (26, 8, 1);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut CUstream_st;

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct cudaDataType_t(c_uint);

impl cudaDataType_t {
    pub const CUDA_R_32F: Self = Self(0);
    pub const CUDA_R_16F: Self = Self(2);
    pub const CUDA_R_8I: Self = Self(3);
    pub const CUDA_R_8U: Self = Self(8);

    pub const fn from_raw(value: c_uint) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> c_uint {
        self.0
    }
}

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    unused_attributes,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
mod bindings;

impl Copy for bindings::cuvsIvfPqSearchParams {}

impl Clone for bindings::cuvsIvfPqSearchParams {
    fn clone(&self) -> Self {
        *self
    }
}

impl fmt::Debug for bindings::cuvsIvfPqSearchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("cuvsIvfPqSearchParams")
            .field("n_probes", &self.n_probes)
            .field("lut_dtype", &self.lut_dtype)
            .field("internal_distance_dtype", &self.internal_distance_dtype)
            .field("coarse_search_dtype", &self.coarse_search_dtype)
            .field("max_internal_batch_size", &self.max_internal_batch_size)
            .field("preferred_shmem_carveout", &self.preferred_shmem_carveout)
            .finish()
    }
}

pub use bindings::*;

#[allow(non_camel_case_types)]
pub type cudaError_t = c_int;
#[allow(non_camel_case_types)]
pub type cudaMemcpyKind = c_uint;

#[allow(non_upper_case_globals)]
pub const cudaSuccess: cudaError_t = 0;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeVersionError {
    QueryFailed,
    Incompatible { expected: (u16, u16, u16), actual: (u16, u16, u16) },
}

impl fmt::Display for RuntimeVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueryFailed => write!(f, "cuvsVersionGet failed"),
            Self::Incompatible { expected, actual } => write!(
                f,
                "cuVS runtime version {}.{}.{} does not match required {}.{}.{}",
                actual.0, actual.1, actual.2, expected.0, expected.1, expected.2
            ),
        }
    }
}

impl std::error::Error for RuntimeVersionError {}

/// Query and validate the exact native cuVS version required by this binding.
pub fn validate_runtime_version() -> Result<(u16, u16, u16), RuntimeVersionError> {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;
    let status = unsafe { cuvsVersionGet(&mut major, &mut minor, &mut patch) };
    if status != cuvsError_t::CUVS_SUCCESS {
        return Err(RuntimeVersionError::QueryFailed);
    }
    let actual = (major, minor, patch);
    if actual != EXPECTED_CUVS_VERSION {
        return Err(RuntimeVersionError::Incompatible { expected: EXPECTED_CUVS_VERSION, actual });
    }
    Ok(actual)
}
