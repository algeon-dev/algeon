use std::{ffi::c_void, marker::PhantomData};

use crate::{
    cuda::CudaStream,
    error::{Result, check_cuvs, invalid_state},
};

#[derive(Debug)]
pub struct Resources {
    raw: cuvs_sys::cuvsResources_t,
    owned_stream: Option<CudaStream>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Resources {
    pub fn new() -> Result<Self> {
        let mut raw = 0;
        unsafe { check_cuvs(cuvs_sys::cuvsResourcesCreate(&mut raw), "Resources::new")? };
        if raw == 0 {
            return Err(invalid_state("Resources::new", "cuVS returned a null resources handle"));
        }
        Ok(Self { raw, owned_stream: None, _not_send_sync: PhantomData })
    }

    pub fn set_stream(&mut self, stream: CudaStream) -> Result<()> {
        let raw = stream.as_raw();
        unsafe {
            check_cuvs(cuvs_sys::cuvsStreamSet(self.raw, raw), "Resources::set_stream")?;
        }
        self.owned_stream = Some(stream);
        Ok(())
    }

    /// Set a caller-owned CUDA stream on this cuVS resources handle.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid CUDA stream for the active device. The caller
    /// must keep it alive and must not destroy it until this `Resources` value
    /// has either been dropped or had its stream replaced.
    pub unsafe fn set_stream_raw(&mut self, stream: *mut c_void) -> Result<()> {
        check_cuvs(unsafe { cuvs_sys::cuvsStreamSet(self.raw, stream) }, "Resources::set_stream_raw")
    }

    pub(crate) fn raw(&self) -> cuvs_sys::cuvsResources_t {
        self.raw
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        if self.raw != 0 {
            unsafe {
                let _ = cuvs_sys::cuvsResourcesDestroy(self.raw);
            }
            self.raw = 0;
        }
    }
}
