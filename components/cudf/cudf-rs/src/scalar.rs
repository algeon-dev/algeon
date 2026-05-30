//! Scalar value types.

use std::{
    ffi::{CStr, CString},
    os::raw::c_void,
    ptr::{self, NonNull},
};

use crate::{
    error::{self, Error, ErrorKind, Result},
    types::{CudfType, DataType, TypeId},
};

/// An owned scalar value on the GPU.
///
/// Scalars can hold a single value of any cudf type, plus a validity flag.
pub struct Scalar {
    handle: NonNull<c_void>,
}

unsafe impl Send for Scalar {}
unsafe impl Sync for Scalar {}

impl Scalar {
    /// Create a valid numeric scalar from a Rust value.
    pub fn from_value<T: CudfType>(value: T) -> Result<Self> {
        let mut handle = ptr::null_mut();
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
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a null scalar of the given numeric type.
    pub fn null_numeric<T: CudfType>() -> Result<Self> {
        let zero = unsafe { std::mem::zeroed::<T>() };
        let mut handle = ptr::null_mut();
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
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a valid string scalar.
    pub fn from_string(value: &str) -> Result<Self> {
        let c_str = CString::new(value).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "string contains interior null byte")
                .with_operation("scalar::from_string")
        })?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string(c_str.as_ptr(), 1, &mut handle) },
            "scalar::from_string",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
    }

    /// Create a null string scalar.
    pub fn null_string() -> Result<Self> {
        let empty = CString::new("").unwrap();
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string(empty.as_ptr(), 0, &mut handle) },
            "scalar::null_string",
        )?;
        Ok(Self { handle: NonNull::new(handle).expect("null handle on success") })
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
        Self { handle: NonNull::new(handle).expect("null scalar handle") }
    }

    /// Returns the data type of this scalar.
    pub fn data_type(&self) -> Result<DataType> {
        let mut raw_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_type(self.handle.as_ptr(), &mut raw_id) },
            "scalar::data_type",
        )?;
        Ok(DataType::new(TypeId::from_raw(raw_id)))
    }

    /// Returns `true` if this scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> Result<bool> {
        let mut out: i32 = 0;
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
        let mut out = unsafe { std::mem::zeroed::<T>() };
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
        let mut ptr: *const std::os::raw::c_char = ptr::null();
        let mut len: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_string_value(self.handle.as_ptr(), &mut ptr, &mut len) },
            "scalar::string_value",
        )?;
        let c_str = unsafe { CStr::from_ptr(ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_scalar_destroy(self.handle.as_ptr());
        }
    }
}
