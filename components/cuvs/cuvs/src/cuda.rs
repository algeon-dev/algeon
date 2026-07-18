use std::{ffi::c_void, marker::PhantomData, ptr};

use crate::{
    error::{Result, check_cuda},
    resources::Resources,
};

pub fn device_count() -> Result<i32> {
    let mut count = 0;
    unsafe { check_cuda(cuvs_sys::cudaGetDeviceCount(&mut count), "cuda::device_count")? };
    Ok(count)
}

pub fn synchronize_device() -> Result<()> {
    unsafe { check_cuda(cuvs_sys::cudaDeviceSynchronize(), "cuda::synchronize_device") }
}

#[derive(Debug)]
pub struct CudaStream {
    raw: cuvs_sys::cudaStream_t,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe { check_cuda(cuvs_sys::cudaStreamCreate(&mut raw), "CudaStream::new")? };
        Ok(Self { raw, _not_send_sync: PhantomData })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check_cuda(cuvs_sys::cudaStreamSynchronize(self.raw), "CudaStream::synchronize") }
    }

    pub fn set_on(self, resources: &mut Resources) -> Result<()> {
        resources.set_stream(self)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = cuvs_sys::cudaStreamDestroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}
