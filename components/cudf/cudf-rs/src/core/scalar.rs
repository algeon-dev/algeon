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
    error::{self, Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
    types::{CudfType, DataType, TypeId},
};
use crate::runtime::execution_domain::ExecutionDomain;

fn negative_string_length_error(operation: &'static str, raw_len: i32) -> Error {
    error::foreign_invariant_violation(
        operation,
        "FFI returned negative string length",
        "string_scalar_value",
        "negative_string_length",
        "ensure the foreign scalar backend returns a non-negative string length",
    )
    .with_detail("raw_len", raw_len.to_string())
}

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
    fn from_handle_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Ok(Self { handle: error::require_non_null(handle, "scalar::from_handle_with_domain", "scalar")?, domain })
    }

    pub(crate) fn from_value_in_domain<T: CudfType>(value: T, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    &value as *const T as *const c_void,
                    1,
                    &exec_options,
                    &mut handle,
                )
            },
            "scalar::from_value_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    pub(crate) fn from_str_in_domain(value: &str, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let c_str = CString::new(value).map_err(|_| {
            error::interior_null_byte_error(
                "scalar::from_str_in_domain",
                "string_scalar",
                "string scalar contains interior null byte",
            )
        })?;
        let mut handle = ptr::null_mut();
        let exec_options = domain.exec_options_raw();
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string_ex(c_str.as_ptr(), 1, &exec_options, &mut handle) },
            "scalar::from_str_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    pub(crate) fn null_numeric_in_domain<T: CudfType>(domain: &Arc<ExecutionDomain>) -> Result<Self> {
        // SAFETY: `CudfType` is sealed to the built-in fixed-width cudf Rust
        // primitives, all of which have valid all-zero bit patterns.
        let zero = unsafe { std::mem::zeroed::<T>() };
        let mut handle = ptr::null_mut();
        let exec_options = domain.exec_options_raw();
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    &zero as *const T as *const c_void,
                    0,
                    &exec_options,
                    &mut handle,
                )
            },
            "scalar::null_numeric_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    pub(crate) fn null_string_in_domain(domain: &Arc<ExecutionDomain>) -> Result<Self> {
        // SAFETY (unwrap): empty string literal "" contains no interior null bytes.
        let empty = CString::default();
        let mut handle = ptr::null_mut();
        let exec_options = domain.exec_options_raw();
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string_ex(empty.as_ptr(), 0, &exec_options, &mut handle) },
            "scalar::null_string_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    /// Returns the raw FFI handle. For internal use by operations.
    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    pub(crate) unsafe fn from_raw_with_domain(handle: *mut c_void, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn copy_into_domain(&self, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        self.copy_into_domain_with_operation(domain, "scalar::copy_into_domain")
    }

    fn copy_into_domain_with_operation(&self, domain: &Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        if self.data_type()?.id() == TypeId::Dictionary32 {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "scalar copy does not support Dictionary32 (no libcudf scalar subclass exists)",
            )
            .with_fact(FACT_TYPE_ID, TypeId::Dictionary32));
        }

        let mut handle = ptr::null_mut();
        let exec_options = domain.exec_options_raw();
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_copy_ex(self.handle.as_ptr(), &exec_options, &mut handle) },
            operation,
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }
    pub(crate) fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
    pub(crate) fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }
    pub(crate) fn __debug_rebind_execution_domain_explicit(mut self, identity: u64) -> Result<Self> {
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
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning a host-visible boolean result.
    pub fn is_valid(&self) -> Result<bool> {
        let mut out: i32 = 0;
        let exec_options = self.domain.exec_options_raw();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_is_valid_ex(self.handle.as_ptr(), &mut out, &exec_options) },
            "scalar::is_valid",
        )?;
        self.domain.synchronize_for_host_visibility("scalar::is_valid")?;
        Ok(out != 0)
    }

    /// Extract the value as type `T`.
    ///
    /// Returns an error if the scalar's type doesn't match `T::TYPE_ID`.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn value<T: CudfType>(&self) -> Result<T> {
        let dt = self.data_type()?;
        if dt.id() != T::TYPE_ID {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "scalar::value",
                "scalar type does not match requested type",
            )
            .with_fact(FACT_LEFT_TYPE_ID, dt.id())
            .with_fact(FACT_RIGHT_TYPE_ID, T::TYPE_ID));
        }
        // SAFETY: `CudfType` is sealed to built-in primitives with valid
        // all-zero bit patterns.
        let mut out = unsafe { std::mem::zeroed::<T>() };
        let exec_options = self.domain.exec_options_raw();
        // SAFETY: self.handle is valid. Type was checked to match T::TYPE_ID above.
        // &mut out is a valid pointer to receive the scalar value.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_value_ex(self.handle.as_ptr(), &mut out as *mut T as *mut c_void, &exec_options)
            },
            "scalar::value",
        )?;
        self.domain.synchronize_for_host_visibility("scalar::value")?;
        Ok(out)
    }

    /// Extract the value as a string.
    ///
    /// Returns an error if the scalar is not a string type.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible string.
    pub fn string_value(&self) -> Result<String> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::String {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "scalar::string_value",
                "scalar type does not match requested type",
            )
            .with_fact(FACT_LEFT_TYPE_ID, dt.id())
            .with_fact(FACT_RIGHT_TYPE_ID, TypeId::String));
        }

        let mut ptr: *mut c_char = ptr::null_mut();
        let mut len: i32 = 0;
        let exec_options = self.domain.exec_options_raw();
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_string_value_ex(self.handle.as_ptr(), &mut ptr, &mut len, &exec_options) },
            "scalar::string_value",
        )?;
        // SAFETY: ptr and len are returned by a successful FFI call.
        let byte_len = usize::try_from(len).map_err(|_| negative_string_length_error("scalar::string_value", len))?;
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_len) };
        let value = String::from_utf8_lossy(bytes).into_owned();
        // SAFETY: ptr was allocated by the C shim and must be freed with cudf_string_free.
        unsafe { cudf_sys::cudf_string_free(ptr) };
        self.domain.synchronize_for_host_visibility("scalar::string_value")?;
        Ok(value)
    }
}

pub(crate) fn trigger_negative_string_length_error_for_test() -> Result<()> {
    Err(negative_string_length_error("scalar::string_value", -1))
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

/// Conversion from a host value into a device-resident [`Scalar`] that adopts
/// the execution provenance of an existing GPU object.
///
/// This trait unifies the construction path for fixed-width primitives (via
/// [`CudfType`]) and string slices, so high-level Series/DataFrame APIs can
/// expose a single generic `*_scalar` method that accepts both numeric and
/// string operands.
///
/// Users typically do not implement this trait themselves; the blanket impls
/// for `T: CudfType`, `&str`, `String`, and `&String` cover the supported
/// host-side scalar inputs. The trait is sealed via the private
/// [`Scalar`] constructors it delegates to.
///
/// The `domain` argument is an opaque execution-provenance handle exposed only
/// internally; this trait is invoked through generic bounds on safe wrappers
/// (e.g. [`Series::eq_scalar`](crate::Series::eq_scalar)) rather than called
/// directly by end users.
pub trait IntoRuntimeScalar {
    /// Build an owned device-resident [`Scalar`] bound to the same execution
    /// provenance as `domain`.
    #[allow(private_interfaces)]
    fn into_runtime_scalar(self, domain: &Arc<ExecutionDomain>) -> Result<Scalar>;
}

impl<T: CudfType> IntoRuntimeScalar for T {
    #[allow(private_interfaces)]
    fn into_runtime_scalar(self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        Scalar::from_value_in_domain(self, domain)
    }
}

impl IntoRuntimeScalar for &str {
    #[allow(private_interfaces)]
    fn into_runtime_scalar(self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        Scalar::from_str_in_domain(self, domain)
    }
}

impl IntoRuntimeScalar for &String {
    #[allow(private_interfaces)]
    fn into_runtime_scalar(self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        Scalar::from_str_in_domain(self.as_str(), domain)
    }
}

impl IntoRuntimeScalar for String {
    #[allow(private_interfaces)]
    fn into_runtime_scalar(self, domain: &Arc<ExecutionDomain>) -> Result<Scalar> {
        Scalar::from_str_in_domain(self.as_str(), domain)
    }
}
