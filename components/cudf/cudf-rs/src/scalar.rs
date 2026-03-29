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

use crate::{
    error::{self, Error, ErrorKind, Result},
    execution_domain::ExecutionDomain,
    types::{CudfType, DataType},
};

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

    /// Create a valid numeric scalar from a Rust value.
    pub fn from_value<T: CudfType>(value: T) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: &value is a valid pointer to a T. T::TYPE_ID correctly
        // describes the type. is_valid=1 marks this as a valid scalar.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric(
                    T::TYPE_ID.to_raw(),
                    &value as *const T as *const c_void,
                    1, // is_valid
                    &mut handle,
                )
            },
            "scalar::from_value",
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
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
        let c_str = CString::new(value).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "string contains interior null byte")
                .with_operation("scalar::from_string")
        })?;
        let mut handle = ptr::null_mut();
        // SAFETY: c_str is a valid CString pointer. is_valid=1.
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string(c_str.as_ptr(), 1, &mut handle) },
            "scalar::from_string",
        )?;
        Ok(Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default()))
    }

    /// Create a null string scalar.
    pub fn null_string() -> Result<Self> {
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

    /// Construct a `Scalar` from a raw FFI handle, taking ownership.
    ///
    /// # Safety
    /// The handle must be a valid `cudf_scalar_handle` and the caller must
    /// transfer ownership.
    pub(crate) unsafe fn from_raw(handle: *mut c_void) -> Self {
        Self::from_handle_with_domain(handle, ExecutionDomain::ambient_default())
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
        self.domain.synchronize_for_host_visibility("scalar::is_valid")?;
        let mut out: i32 = 0;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_is_valid(self.handle.as_ptr(), &mut out) },
            "scalar::is_valid",
        )?;
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
        self.domain.synchronize_for_host_visibility("scalar::value")?;
        // SAFETY: All CudfType implementors have valid all-zero bit patterns.
        let mut out = unsafe { std::mem::zeroed::<T>() };
        // SAFETY: self.handle is valid. Type was checked to match T::TYPE_ID above.
        // &mut out is a valid pointer to receive the scalar value.
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_value(self.handle.as_ptr(), &mut out as *mut T as *mut c_void) },
            "scalar::value",
        )?;
        Ok(out)
    }

    /// Extract the value as a string.
    ///
    /// Returns an error if the scalar is not a string type.
    pub fn string_value(&self) -> Result<String> {
        self.domain.synchronize_for_host_visibility("scalar::string_value")?;
        let mut ptr: *mut c_char = ptr::null_mut();
        let mut len: i32 = 0;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_string_value(self.handle.as_ptr(), &mut ptr, &mut len) },
            "scalar::string_value",
        )?;
        // SAFETY: ptr and len are returned by a successful FFI call.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
        let value = String::from_utf8_lossy(bytes).into_owned();
        // SAFETY: ptr was allocated by the C shim and must be freed with cudf_string_free.
        unsafe { cudf_sys::cudf_string_free(ptr) };
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
