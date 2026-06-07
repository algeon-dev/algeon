//! Scalar value types.
//!
//! For scalar fill values, prefer [`Series::fillna()`](crate::Series::fillna)
//! which accepts plain Rust types.

use std::{
    ffi::CString,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    error::{self, Error, ErrorKind, Result},
    types::{CudfType, DataType},
};
use crate::runtime::execution_domain::ExecutionDomain;

/// An owned scalar value on the GPU.
///
/// Scalars can hold a single value of any cudf type, plus a validity flag.
pub struct Scalar {
    handle: NonNull<c_void>,
    pub(crate) domain: Arc<ExecutionDomain>,
}

// SAFETY: Scalar owns an immutable GPU value via an opaque C++ handle.
// The scalar value cannot be mutated after creation, making concurrent reads safe.
unsafe impl Send for Scalar {}
unsafe impl Sync for Scalar {}

impl Scalar {
    fn from_handle_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Self {
        Self { handle: NonNull::new(handle).expect("null scalar handle"), domain }
    }

    pub(crate) fn from_value_in_domain<T: CudfType>(value: T, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.explicit_exec_options_raw();
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe {
                    cudf_sys::cudf_scalar_create_numeric_ex(
                        T::TYPE_ID.to_raw(),
                        &value as *const T as *const c_void,
                        1,
                        options,
                        &mut handle,
                    )
                },
                None => unsafe {
                    cudf_sys::cudf_scalar_create_numeric(
                        T::TYPE_ID.to_raw(),
                        &value as *const T as *const c_void,
                        1,
                        &mut handle,
                    )
                },
            },
            "scalar::from_value_in_domain",
        )?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(domain)))
    }

    pub(crate) fn from_string_in_domain(value: &str, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let c_str = CString::new(value).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "string contains interior null byte")
                .with_operation("scalar::from_string_in_domain")
        })?;
        let mut handle = ptr::null_mut();
        let exec_options = domain.explicit_exec_options_raw();
        error::check_status_in(
            match exec_options.as_ref() {
                Some(options) => unsafe {
                    cudf_sys::cudf_scalar_create_string_ex(c_str.as_ptr(), 1, options, &mut handle)
                },
                None => unsafe { cudf_sys::cudf_scalar_create_string(c_str.as_ptr(), 1, &mut handle) },
            },
            "scalar::from_string_in_domain",
        )?;
        Ok(Self::from_handle_with_domain(handle, Arc::clone(domain)))
    }

    /// Create a valid numeric scalar from a Rust value.
    pub fn from_value<T: CudfType>(value: T) -> Result<Self> {
        let domain = ExecutionDomain::ambient_default();
        Self::from_value_in_domain(value, &domain).map_err(|e| e.with_operation("scalar::from_value"))
    }

    /// Create a null scalar of the given numeric type.
    pub fn null_numeric<T: CudfType>() -> Result<Self> {
        // SAFETY: All CudfType implementors (i8, i16, i32, i64, u8, u16, u32, u64,
        // f32, f64, bool) have valid all-zero bit patterns. The trait is sealed by
        // its pub-in-private location, so no external types can implement it.
        let zero = unsafe { std::mem::zeroed::<T>() };
        let mut handle = ptr::null_mut();
        // SAFETY: &zero is a valid pointer. is_valid=0 marks this as null.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric(
                    T::TYPE_ID.to_raw(),
                    &zero as *const T as *const c_void,
                    0, // is_valid = false
                    &mut handle,
                )
            },
            "scalar::null_numeric",
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
    }

    /// Create a valid string scalar.
    pub fn from_string(value: &str) -> Result<Self> {
        let domain = ExecutionDomain::ambient_default();
        Self::from_string_in_domain(value, &domain).map_err(|e| e.with_operation("scalar::from_string"))
    }

    /// Create a null string scalar.
    pub fn null_string() -> Result<Self> {
        // SAFETY (unwrap): empty string literal "" contains no interior null bytes.
        let empty = CString::new("").unwrap();
        let mut handle = ptr::null_mut();
        // SAFETY: empty is a valid CString pointer. is_valid=0 marks null.
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string(empty.as_ptr(), 0, &mut handle) },
            "scalar::null_string",
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
    }

    /// Returns the raw FFI handle. For internal use by operations.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    /// Construct a `Scalar` from a raw FFI handle, taking ownership.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_scalar_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw(handle: *mut c_void) -> Self {
        Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default())
    }

    pub(crate) fn with_execution_domain(mut self, domain: Arc<ExecutionDomain>) -> Self {
        self.domain = domain;
        self
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }

    #[doc(hidden)]
    pub fn __debug_rebind_execution_domain_explicit(mut self, identity: u64) -> Result<Self> {
        self.domain = ExecutionDomain::debug_explicit(identity)?;
        Ok(self)
    }

    /// Returns the data type of this scalar.
    pub fn data_type(&self) -> Result<DataType> {
        let mut raw_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
        let mut scale = 0;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_data_type(self.handle.as_ptr(), &mut raw_id, &mut scale) },
            "scalar::data_type",
        )?;
        DataType::from_raw_parts(raw_id, scale)
    }

    /// Returns `true` if this scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> Result<bool> {
        let mut out: i32 = 0;
        if let Some(exec_options) = self.domain.explicit_exec_options_raw() {
            // SAFETY: self.handle is valid (NonNull invariant from construction).
            error::check_status_in(
                unsafe { cudf_sys::cudf_scalar_is_valid_ex(self.handle.as_ptr(), &mut out, &exec_options) },
                "scalar::is_valid",
            )?;
        } else {
            // SAFETY: self.handle is valid (NonNull invariant from construction).
            error::check_status_in(
                unsafe { cudf_sys::cudf_scalar_is_valid(self.handle.as_ptr(), &mut out) },
                "scalar::is_valid",
            )?;
        }
        self.domain.synchronize_for_host_visibility("scalar::is_valid")?;
        Ok(out != 0)
    }

    /// Extract the value as type `T`.
    ///
    /// Returns an error if the scalar's type doesn't match `T::TYPE_ID`.
    pub fn value<T: CudfType>(&self) -> Result<T> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new(ErrorKind::TypeError, "scalar type does not match requested type")
                .with_operation("scalar::value")
                .with_context("actual_type_id", format!("{:?}", dt.id()).to_lowercase())
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase()));
        }
        // SAFETY: All CudfType implementors have valid all-zero bit patterns.
        let mut out = unsafe { std::mem::zeroed::<T>() };
        if let Some(exec_options) = self.domain.explicit_exec_options_raw() {
            // SAFETY: self.handle is valid. Type was checked to match T::TYPE_ID above.
            // &mut out is a valid pointer to receive the scalar value.
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_scalar_value_ex(
                        self.handle.as_ptr(),
                        &mut out as *mut T as *mut c_void,
                        &exec_options,
                    )
                },
                "scalar::value",
            )?;
        } else {
            // SAFETY: self.handle is valid. Type was checked to match T::TYPE_ID above.
            // &mut out is a valid pointer to receive the scalar value.
            error::check_status_in(
                unsafe { cudf_sys::cudf_scalar_value(self.handle.as_ptr(), &mut out as *mut T as *mut c_void) },
                "scalar::value",
            )?;
        }
        self.domain.synchronize_for_host_visibility("scalar::value")?;
        Ok(out)
    }

    /// Extract the value as a string.
    ///
    /// Returns an error if the scalar is not a string type.
    pub fn string_value(&self) -> Result<String> {
        let mut ptr: *mut c_char = ptr::null_mut();
        let mut len: i32 = 0;
        if let Some(exec_options) = self.domain.explicit_exec_options_raw() {
            // SAFETY: self.handle is valid (NonNull invariant from construction).
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_scalar_string_value_ex(self.handle.as_ptr(), &mut ptr, &mut len, &exec_options)
                },
                "scalar::string_value",
            )?;
        } else {
            // SAFETY: self.handle is valid (NonNull invariant from construction).
            error::check_status_in(
                unsafe { cudf_sys::cudf_scalar_string_value(self.handle.as_ptr(), &mut ptr, &mut len) },
                "scalar::string_value",
            )?;
        }
        // SAFETY: ptr and len are returned by a successful FFI call.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
        let value = String::from_utf8_lossy(bytes).into_owned();
        // SAFETY: ptr was allocated by the C shim and must be freed with cudf_string_free.
        unsafe { cudf_sys::cudf_string_free(ptr) };
        self.domain.synchronize_for_host_visibility("scalar::string_value")?;
        Ok(value)
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        // SAFETY: self.handle is valid (NonNull from construction). Drop runs
        // exactly once, so the handle is destroyed exactly once.
        unsafe {
            cudf_sys::cudf_scalar_destroy(self.handle.as_ptr());
        }
    }
}
