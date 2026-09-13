use std::{ptr, ptr::NonNull};

use crate::error::{Result, check_cuvs, invalid_state};

pub(crate) struct ParamsHandle<T> {
    raw: Option<NonNull<T>>,
    destroy: unsafe extern "C" fn(*mut T) -> cuvs_sys::cuvsError_t,
}

impl<T> ParamsHandle<T> {
    pub(crate) fn new(
        create: unsafe extern "C" fn(*mut *mut T) -> cuvs_sys::cuvsError_t,
        destroy: unsafe extern "C" fn(*mut T) -> cuvs_sys::cuvsError_t,
        operation: &'static str,
        name: &'static str,
    ) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(create(&mut raw), operation)? };
        Ok(Self {
            raw: Some(NonNull::new(raw).ok_or_else(|| invalid_state(operation, format!("cuVS returned null {name}")))?),
            destroy,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut T {
        self.raw.expect("params remain live during native call").as_ptr()
    }

    pub(crate) fn close(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else { return Ok(()) };
        let result = unsafe { check_cuvs((self.destroy)(raw.as_ptr()), operation) };
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl<T> Drop for ParamsHandle<T> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe {
                let _ = (self.destroy)(raw.as_ptr());
            }
        }
    }
}
