//! Scalar value types.
//!
//! For scalar fill values, prefer [`SeriesOps::fillna()`](crate::SeriesOps::fillna)
//! which accepts plain Rust types.

use std::{
    ffi::CString,
    os::raw::{c_char, c_void},
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    error::{self, Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
    types::{CudfType, DataType, TimestampResolution, TypeId, materialize_bool},
};
use crate::{core::native_teardown::drop_device_owned_handle, runtime::execution_domain::ExecutionDomain};

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

fn unsupported_scalar_dtype_error(
    operation: &'static str,
    message: &'static str,
    type_id: TypeId,
    expected_family: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedDtype, operation, message)
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("reason", "unsupported_scalar_dtype")
        .with_detail("expected_family", expected_family)
}

fn scalar_type_mismatch_error(
    operation: &'static str,
    message: &'static str,
    actual: TypeId,
    requested: TypeId,
) -> Error {
    Error::new_with(ErrorCode::TypeMismatch, operation, message)
        .with_fact(FACT_LEFT_TYPE_ID, actual)
        .with_fact(FACT_RIGHT_TYPE_ID, requested)
}

fn scalar_family_type_mismatch_error(
    operation: &'static str,
    message: &'static str,
    actual: TypeId,
    expected_family: &'static str,
) -> Error {
    Error::new_with(ErrorCode::TypeMismatch, operation, message)
        .with_fact(FACT_LEFT_TYPE_ID, actual)
        .with_detail("expected_family", expected_family)
}

fn decimal_dtype_required(dtype: DataType, operation: &'static str) -> Result<DataType> {
    if dtype.id().is_decimal() {
        Ok(dtype)
    } else {
        Err(unsupported_scalar_dtype_error(
            operation,
            "decimal scalar construction requires a decimal dtype",
            dtype.id(),
            "decimal",
        ))
    }
}

fn timestamp_type_required(type_id: TypeId, operation: &'static str) -> Result<TypeId> {
    if type_id.is_timestamp() {
        Ok(type_id)
    } else {
        Err(unsupported_scalar_dtype_error(
            operation,
            "timestamp scalar construction requires a timestamp type_id",
            type_id,
            "timestamp",
        ))
    }
}

fn duration_type_required(type_id: TypeId, operation: &'static str) -> Result<TypeId> {
    if type_id.is_duration() {
        Ok(type_id)
    } else {
        Err(unsupported_scalar_dtype_error(
            operation,
            "duration scalar construction requires a duration type_id",
            type_id,
            "duration",
        ))
    }
}

fn decimal_raw_range_error(operation: &'static str, dtype: DataType, min: i128, max: i128, raw: i128) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "decimal raw value is outside the requested dtype range",
    )
    .with_fact(FACT_TYPE_ID, dtype.id())
    .with_detail("resource", "decimal_raw")
    .with_detail("reason", "raw_value_out_of_range")
    .with_detail("min_raw", min.to_string())
    .with_detail("max_raw", max.to_string())
    .with_detail("raw", raw.to_string())
}

fn temporal_days_raw_range_error(operation: &'static str, type_id: TypeId, raw: i64) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "day-resolution temporal raw value is outside the i32 representation range",
    )
    .with_fact(FACT_TYPE_ID, type_id)
    .with_detail("resource", "temporal_raw")
    .with_detail("reason", "raw_value_out_of_range")
    .with_detail("min_raw", i32::MIN.to_string())
    .with_detail("max_raw", i32::MAX.to_string())
    .with_detail("raw", raw.to_string())
}

fn validate_temporal_raw_range(raw: i64, type_id: TypeId, operation: &'static str) -> Result<()> {
    if matches!(type_id, TypeId::TimestampDays | TypeId::DurationDays)
        && (raw < i64::from(i32::MIN) || raw > i64::from(i32::MAX))
    {
        Err(temporal_days_raw_range_error(operation, type_id, raw))
    } else {
        Ok(())
    }
}

/// Host value of a device [`Scalar`], tagged with the cudf type it is
/// created as.
///
/// [`Runtime::scalar`](crate::Runtime::scalar), [`lit`](crate::ops::expr::lit),
/// and the `*_scalar` comparisons on [`SeriesOps`](crate::SeriesOps) create
/// device scalars from it. Every [`CudfType`] primitive, `&str`, `&String`,
/// and `String` convert into the matching variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    /// Fixed-point `raw * 10^scale` stored as `DECIMAL32`.
    Decimal32 {
        raw: i32,
        scale: i32,
    },
    /// Fixed-point `raw * 10^scale` stored as `DECIMAL64`.
    Decimal64 {
        raw: i64,
        scale: i32,
    },
    /// Fixed-point `raw * 10^scale` stored as `DECIMAL128`.
    Decimal128 {
        raw: i128,
        scale: i32,
    },
    /// `raw` units of `resolution` since the Unix epoch, without parsing or
    /// timezone conversion.
    Timestamp {
        raw: i64,
        resolution: TimestampResolution,
    },
    /// `raw` units of `resolution`.
    Duration {
        raw: i64,
        resolution: TimestampResolution,
    },
    /// A null of the given type.
    Null(DataType),
}

impl ScalarValue {
    /// Fixed-point value of a decimal `dtype` known only at runtime.
    ///
    /// Fails with `UnsupportedDtype` unless `dtype` is Decimal32, Decimal64,
    /// or Decimal128, and with `UnsupportedOperation` when `raw` does not fit
    /// that type's storage width.
    pub fn decimal(raw: i128, dtype: DataType) -> Result<Self> {
        const OPERATION: &str = "scalar_value::decimal";
        let scale = dtype.scale();
        match dtype.id() {
            TypeId::Decimal32 => Ok(Self::Decimal32 {
                raw: i32::try_from(raw).map_err(|_| {
                    decimal_raw_range_error(OPERATION, dtype, i128::from(i32::MIN), i128::from(i32::MAX), raw)
                })?,
                scale,
            }),
            TypeId::Decimal64 => Ok(Self::Decimal64 {
                raw: i64::try_from(raw).map_err(|_| {
                    decimal_raw_range_error(OPERATION, dtype, i128::from(i64::MIN), i128::from(i64::MAX), raw)
                })?,
                scale,
            }),
            TypeId::Decimal128 => Ok(Self::Decimal128 { raw, scale }),
            other => Err(unsupported_scalar_dtype_error(
                OPERATION,
                "decimal scalar construction requires a decimal dtype",
                other,
                "decimal",
            )),
        }
    }
}

impl From<&str> for ScalarValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<&String> for ScalarValue {
    fn from(value: &String) -> Self {
        Self::String(value.clone())
    }
}

impl From<String> for ScalarValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

/// An owned scalar value on the GPU.
///
/// Scalars can hold a single value of any cudf type, plus a validity flag.
pub struct Scalar {
    handle: NonNull<cudf_sys::cudf_scalar_handle_state>,
    pub(crate) domain: Arc<ExecutionDomain>,
}

// SAFETY: Scalar owns an immutable GPU value via an opaque C++ handle.
// The scalar value cannot be mutated after creation, making concurrent reads safe.
unsafe impl Send for Scalar {}
unsafe impl Sync for Scalar {}

impl Scalar {
    fn from_handle_with_domain(handle: cudf_sys::cudf_scalar_handle, domain: Arc<ExecutionDomain>) -> Result<Self> {
        Ok(Self { handle: error::require_non_null(handle, "scalar::from_handle_with_domain", "scalar")?, domain })
    }

    pub(crate) fn from_value_in_domain<T: CudfType>(value: T, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    &value as *const T as *const c_void,
                    1,
                    exec_options.options(),
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
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string_ex(c_str.as_ptr(), 1, exec_options.options(), &mut handle) },
            "scalar::from_str_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    pub(crate) fn from_scalar_value_in_domain(value: &ScalarValue, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        const OPERATION: &str = "scalar::from_scalar_value_in_domain";
        match value {
            ScalarValue::Bool(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Int8(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Int16(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Int32(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Int64(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::UInt8(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::UInt16(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::UInt32(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::UInt64(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Float32(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::Float64(value) => Self::from_value_in_domain(*value, domain),
            ScalarValue::String(value) => Self::from_str_in_domain(value, domain),
            ScalarValue::Decimal32 { raw, scale } => Self::decimal_from_typed_raw_in_domain(
                raw,
                DataType::fixed_point(TypeId::Decimal32, *scale)?,
                1,
                domain,
                OPERATION,
            ),
            ScalarValue::Decimal64 { raw, scale } => Self::decimal_from_typed_raw_in_domain(
                raw,
                DataType::fixed_point(TypeId::Decimal64, *scale)?,
                1,
                domain,
                OPERATION,
            ),
            ScalarValue::Decimal128 { raw, scale } => Self::decimal_from_typed_raw_in_domain(
                raw,
                DataType::fixed_point(TypeId::Decimal128, *scale)?,
                1,
                domain,
                OPERATION,
            ),
            ScalarValue::Timestamp { raw, resolution } => {
                Self::timestamp_from_raw_in_domain(*raw, resolution.type_id(), domain)
            },
            ScalarValue::Duration { raw, resolution } => {
                Self::duration_from_raw_in_domain(*raw, resolution.duration_type_id(), domain)
            },
            ScalarValue::Null(dtype) => Self::null_in_domain(*dtype, domain),
        }
    }

    pub(crate) fn null_numeric_in_domain<T: CudfType>(domain: &Arc<ExecutionDomain>) -> Result<Self> {
        // SAFETY: `CudfType` is sealed to the built-in fixed-width cudf Rust
        // primitives, all of which have valid all-zero bit patterns.
        let zero = unsafe { std::mem::zeroed::<T>() };
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_numeric_ex(
                    T::TYPE_ID.to_raw(),
                    &zero as *const T as *const c_void,
                    0,
                    exec_options.options(),
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
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_create_string_ex(empty.as_ptr(), 0, exec_options.options(), &mut handle) },
            "scalar::null_string_in_domain",
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    pub(crate) fn timestamp_from_raw_in_domain(
        raw: i64,
        type_id: TypeId,
        domain: &Arc<ExecutionDomain>,
    ) -> Result<Self> {
        let type_id = timestamp_type_required(type_id, "scalar::timestamp_from_raw_in_domain")?;
        validate_temporal_raw_range(raw, type_id, "scalar::timestamp_from_raw_in_domain")?;
        Self::temporal_from_raw_in_domain(
            raw,
            type_id,
            1,
            domain,
            "scalar::timestamp_from_raw_in_domain",
            cudf_sys::cudf_scalar_create_timestamp_raw_ex,
        )
    }

    pub(crate) fn duration_from_raw_in_domain(
        raw: i64,
        type_id: TypeId,
        domain: &Arc<ExecutionDomain>,
    ) -> Result<Self> {
        let type_id = duration_type_required(type_id, "scalar::duration_from_raw_in_domain")?;
        validate_temporal_raw_range(raw, type_id, "scalar::duration_from_raw_in_domain")?;
        Self::temporal_from_raw_in_domain(
            raw,
            type_id,
            1,
            domain,
            "scalar::duration_from_raw_in_domain",
            cudf_sys::cudf_scalar_create_duration_raw_ex,
        )
    }

    pub(crate) fn null_decimal_in_domain(dtype: DataType, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let dtype = decimal_dtype_required(dtype, "scalar::null_decimal_in_domain")?;
        Self::decimal_from_typed_raw_in_domain::<i32>(ptr::null(), dtype, 0, domain, "scalar::null_decimal_in_domain")
    }

    pub(crate) fn null_timestamp_in_domain(type_id: TypeId, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let type_id = timestamp_type_required(type_id, "scalar::null_timestamp_in_domain")?;
        Self::temporal_from_raw_in_domain(
            0,
            type_id,
            0,
            domain,
            "scalar::null_timestamp_in_domain",
            cudf_sys::cudf_scalar_create_timestamp_raw_ex,
        )
    }

    pub(crate) fn null_duration_in_domain(type_id: TypeId, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        let type_id = duration_type_required(type_id, "scalar::null_duration_in_domain")?;
        Self::temporal_from_raw_in_domain(
            0,
            type_id,
            0,
            domain,
            "scalar::null_duration_in_domain",
            cudf_sys::cudf_scalar_create_duration_raw_ex,
        )
    }

    pub(crate) fn null_in_domain(dtype: DataType, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        match dtype.id() {
            TypeId::Int8 => Self::null_numeric_in_domain::<i8>(domain),
            TypeId::Int16 => Self::null_numeric_in_domain::<i16>(domain),
            TypeId::Int32 => Self::null_numeric_in_domain::<i32>(domain),
            TypeId::Int64 => Self::null_numeric_in_domain::<i64>(domain),
            TypeId::UInt8 => Self::null_numeric_in_domain::<u8>(domain),
            TypeId::UInt16 => Self::null_numeric_in_domain::<u16>(domain),
            TypeId::UInt32 => Self::null_numeric_in_domain::<u32>(domain),
            TypeId::UInt64 => Self::null_numeric_in_domain::<u64>(domain),
            TypeId::Float32 => Self::null_numeric_in_domain::<f32>(domain),
            TypeId::Float64 => Self::null_numeric_in_domain::<f64>(domain),
            TypeId::Bool8 => Self::null_numeric_in_domain::<bool>(domain),
            TypeId::String => Self::null_string_in_domain(domain),
            TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => Self::null_decimal_in_domain(dtype, domain),
            TypeId::TimestampDays
            | TypeId::TimestampSeconds
            | TypeId::TimestampMilliseconds
            | TypeId::TimestampMicroseconds
            | TypeId::TimestampNanoseconds => Self::null_timestamp_in_domain(dtype.id(), domain),
            TypeId::DurationDays
            | TypeId::DurationSeconds
            | TypeId::DurationMilliseconds
            | TypeId::DurationMicroseconds
            | TypeId::DurationNanoseconds => Self::null_duration_in_domain(dtype.id(), domain),
            other => Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                "scalar::null_in_domain",
                "null scalar construction does not support this dtype",
            )
            .with_fact(FACT_TYPE_ID, other)
            .with_detail("reason", "unsupported_null_scalar_dtype")),
        }
    }

    fn decimal_from_typed_raw_in_domain<T>(
        value: *const T,
        dtype: DataType,
        is_valid: i32,
        domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_create_decimal_raw_ex(
                    dtype.id().to_raw(),
                    value as *const c_void,
                    dtype.scale(),
                    is_valid,
                    exec_options.options(),
                    &mut handle,
                )
            },
            operation,
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    fn temporal_from_raw_in_domain(
        raw: i64,
        type_id: TypeId,
        is_valid: i32,
        domain: &Arc<ExecutionDomain>,
        operation: &'static str,
        create: unsafe extern "C" fn(
            cudf_sys::cudf_type_id_t,
            i64,
            i32,
            *const cudf_sys::cudf_exec_options_t,
            *mut cudf_sys::cudf_scalar_handle,
        ) -> cudf_sys::cudf_status_t,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { create(type_id.to_raw(), raw, is_valid, exec_options.options(), &mut handle) },
            operation,
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    /// Returns the raw FFI handle. For internal use by operations.
    pub(crate) fn raw_handle(&self) -> cudf_sys::cudf_scalar_handle {
        self.handle.as_ptr()
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    pub(crate) unsafe fn from_raw_with_domain(
        handle: cudf_sys::cudf_scalar_handle,
        domain: Arc<ExecutionDomain>,
    ) -> Result<Self> {
        Self::from_handle_with_domain(handle, domain)
    }

    pub(crate) fn copy_into_domain(&self, domain: &Arc<ExecutionDomain>) -> Result<Self> {
        self.copy_into_domain_with_operation(domain, "scalar::copy_into_domain")
    }

    fn copy_into_domain_with_operation(&self, domain: &Arc<ExecutionDomain>, operation: &'static str) -> Result<Self> {
        let dtype = self.data_type()?;
        if dtype.id() == TypeId::Dictionary32 {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "scalar copy does not support Dictionary32 (no libcudf scalar subclass exists)",
            )
            .with_fact(FACT_TYPE_ID, TypeId::Dictionary32));
        }

        if self.domain.device_id() != domain.device_id() {
            return self.recreate_in_domain_via_host(dtype, domain, operation);
        }

        // Hold the fence across the copy below: it keeps the source scalar's
        // stream from reusing its device bytes while the copy is in flight.
        let _consumer_fence = ExecutionDomain::insert_stream_dependency(&self.domain, domain, operation)?;

        let mut handle = ptr::null_mut();
        let exec_options = domain.enter_execution()?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_copy_ex(self.handle.as_ptr(), exec_options.options(), &mut handle) },
            operation,
        )?;
        Self::from_handle_with_domain(handle, Arc::clone(domain))
    }

    fn recreate_in_domain_via_host(
        &self,
        dtype: DataType,
        domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        self.host_value(dtype, operation)
            .and_then(|value| Self::from_scalar_value_in_domain(&value, domain))
            .map_err(|e| e.reanchor_if_needed(operation))
    }

    /// Host copy of this `dtype` scalar as the value that recreates it.
    fn host_value(&self, dtype: DataType, operation: &'static str) -> Result<ScalarValue> {
        if !self.is_valid()? {
            return Ok(ScalarValue::Null(dtype));
        }
        let id = dtype.id();
        if let Some(resolution) = id.timestamp_resolution() {
            return Ok(ScalarValue::Timestamp { raw: self.timestamp_raw_value()?, resolution });
        }
        if let Some(resolution) = id.duration_resolution() {
            return Ok(ScalarValue::Duration { raw: self.duration_raw_value()?, resolution });
        }
        Ok(match id {
            TypeId::Int8 => ScalarValue::Int8(self.value()?),
            TypeId::Int16 => ScalarValue::Int16(self.value()?),
            TypeId::Int32 => ScalarValue::Int32(self.value()?),
            TypeId::Int64 => ScalarValue::Int64(self.value()?),
            TypeId::UInt8 => ScalarValue::UInt8(self.value()?),
            TypeId::UInt16 => ScalarValue::UInt16(self.value()?),
            TypeId::UInt32 => ScalarValue::UInt32(self.value()?),
            TypeId::UInt64 => ScalarValue::UInt64(self.value()?),
            TypeId::Float32 => ScalarValue::Float32(self.value()?),
            TypeId::Float64 => ScalarValue::Float64(self.value()?),
            TypeId::Bool8 => ScalarValue::Bool(self.value()?),
            TypeId::String => ScalarValue::String(self.string_value()?),
            TypeId::Decimal32 => ScalarValue::Decimal32 { raw: self.decimal32_raw_value()?, scale: dtype.scale() },
            TypeId::Decimal64 => ScalarValue::Decimal64 { raw: self.decimal64_raw_value()?, scale: dtype.scale() },
            TypeId::Decimal128 => ScalarValue::Decimal128 { raw: self.decimal128_raw_value()?, scale: dtype.scale() },
            other => {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    "cross-device scalar relocation does not support this scalar dtype",
                )
                .with_fact(FACT_TYPE_ID, other)
                .with_detail("reason", "unsupported_cross_device_scalar_relocation"));
            },
        })
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
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_rebind_execution_domain_explicit(mut self, identity: u64) -> Result<Self> {
        let target = ExecutionDomain::debug_explicit(identity)?;
        let _consumer_fence = ExecutionDomain::insert_stream_dependency(
            &self.domain,
            &target,
            "scalar::__debug_rebind_execution_domain_explicit",
        )?;
        self.domain = target;
        Ok(self)
    }

    /// Returns the data type of this scalar.
    pub fn data_type(&self) -> Result<DataType> {
        // Raw discriminant: validated by `DataType::from_raw_parts` so an
        // unmirrored libcudf type id stays a recoverable error, not UB.
        let mut raw_id: cudf_sys::cudf_type_id_t = 0;
        let mut scale = 0;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_data_type(self.handle.as_ptr(), &mut raw_id, &mut scale) },
            "scalar::data_type",
        )?;
        DataType::from_raw_parts(raw_id as i32, scale)
    }

    /// Returns `true` if this scalar holds a valid (non-null) value.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning a host-visible boolean result.
    pub fn is_valid(&self) -> Result<bool> {
        let mut out: i32 = 0;
        let exec_options = self.domain.enter_execution()?;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe { cudf_sys::cudf_scalar_is_valid_ex(self.handle.as_ptr(), &mut out, exec_options.options()) },
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
        if T::TYPE_ID == TypeId::Bool8 {
            return materialize_bool(self.raw_value_unchecked::<u8>("scalar::value")?, "scalar::value");
        }

        // SAFETY: `CudfType` is sealed to built-in primitives with valid
        // all-zero bit patterns.
        let mut out = unsafe { std::mem::zeroed::<T>() };
        let exec_options = self.domain.enter_execution()?;
        // SAFETY: self.handle is valid. Type was checked to match T::TYPE_ID above.
        // &mut out is a valid pointer to receive the scalar value.
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_value_ex(
                    self.handle.as_ptr(),
                    &mut out as *mut T as *mut c_void,
                    exec_options.options(),
                )
            },
            "scalar::value",
        )?;
        self.domain.synchronize_for_host_visibility("scalar::value")?;
        Ok(out)
    }

    fn raw_value_unchecked<T: Copy>(&self, operation: &'static str) -> Result<T> {
        // SAFETY: raw scalar value types used by this module are plain integer
        // representations with valid all-zero bit patterns.
        let mut out = unsafe { std::mem::zeroed::<T>() };
        let exec_options = self.domain.enter_execution()?;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_value_ex(
                    self.handle.as_ptr(),
                    &mut out as *mut T as *mut c_void,
                    exec_options.options(),
                )
            },
            operation,
        )?;
        self.domain.synchronize_for_host_visibility(operation)?;
        Ok(out)
    }

    /// Extract a Decimal32 scalar's raw fixed-point representation.
    ///
    /// This returns the stored integer representation only. It does not apply
    /// decimal scale formatting or host parsing.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn decimal32_raw_value(&self) -> Result<i32> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::Decimal32 {
            return Err(scalar_type_mismatch_error(
                "scalar::decimal32_raw_value",
                "decimal32 raw extraction requires a Decimal32 scalar",
                dt.id(),
                TypeId::Decimal32,
            ));
        }
        self.raw_value_unchecked("scalar::decimal32_raw_value")
    }

    /// Extract a Decimal64 scalar's raw fixed-point representation.
    ///
    /// This returns the stored integer representation only. It does not apply
    /// decimal scale formatting or host parsing.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn decimal64_raw_value(&self) -> Result<i64> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::Decimal64 {
            return Err(scalar_type_mismatch_error(
                "scalar::decimal64_raw_value",
                "decimal64 raw extraction requires a Decimal64 scalar",
                dt.id(),
                TypeId::Decimal64,
            ));
        }
        self.raw_value_unchecked("scalar::decimal64_raw_value")
    }

    /// Extract a Decimal128 scalar's raw fixed-point representation.
    ///
    /// This returns the stored integer representation only. It does not apply
    /// decimal scale formatting or host parsing.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn decimal128_raw_value(&self) -> Result<i128> {
        let dt = self.data_type()?;
        if dt.id() != TypeId::Decimal128 {
            return Err(scalar_type_mismatch_error(
                "scalar::decimal128_raw_value",
                "decimal128 raw extraction requires a Decimal128 scalar",
                dt.id(),
                TypeId::Decimal128,
            ));
        }
        self.raw_value_unchecked("scalar::decimal128_raw_value")
    }

    /// Extract a timestamp scalar's raw count in its declared resolution.
    ///
    /// This returns the stored integer representation only. It does not parse,
    /// format, or apply timezone behavior.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn timestamp_raw_value(&self) -> Result<i64> {
        let dt = self.data_type()?;
        if !dt.id().is_timestamp() {
            return Err(scalar_family_type_mismatch_error(
                "scalar::timestamp_raw_value",
                "timestamp raw extraction requires a timestamp scalar",
                dt.id(),
                "timestamp",
            ));
        }
        if dt.id() == TypeId::TimestampDays {
            Ok(i64::from(self.raw_value_unchecked::<i32>("scalar::timestamp_raw_value")?))
        } else {
            self.raw_value_unchecked("scalar::timestamp_raw_value")
        }
    }

    /// Extract a duration scalar's raw count in its declared resolution.
    ///
    /// This returns the stored integer representation only. It does not parse
    /// or format the duration.
    ///
    /// # Synchronization
    /// This may synchronize the scalar's bound execution domain before
    /// returning the host-visible value.
    pub fn duration_raw_value(&self) -> Result<i64> {
        let dt = self.data_type()?;
        if !dt.id().is_duration() {
            return Err(scalar_family_type_mismatch_error(
                "scalar::duration_raw_value",
                "duration raw extraction requires a duration scalar",
                dt.id(),
                "duration",
            ));
        }
        if dt.id() == TypeId::DurationDays {
            Ok(i64::from(self.raw_value_unchecked::<i32>("scalar::duration_raw_value")?))
        } else {
            self.raw_value_unchecked("scalar::duration_raw_value")
        }
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
        let exec_options = self.domain.enter_execution()?;
        // SAFETY: self.handle is valid (NonNull invariant from construction).
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_scalar_string_value_ex(self.handle.as_ptr(), &mut ptr, &mut len, exec_options.options())
            },
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

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_negative_string_length_error_for_test() -> Result<()> {
    Err(negative_string_length_error("scalar::string_value", -1))
}

impl Drop for Scalar {
    fn drop(&mut self) {
        drop_device_owned_handle(self.handle.as_ptr(), self.domain.device_id(), "Scalar::drop", |owned| unsafe {
            cudf_sys::cudf_scalar_destroy(owned)
        });
    }
}
