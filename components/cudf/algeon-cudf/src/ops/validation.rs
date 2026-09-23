//! GPU-resident input validation with bounded host-visible evidence.

use std::{mem, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView, DeviceBuffer},
        error::{
            self, DeviceIdFact, Error, ErrorCode, FACT_CONTROL_PLANE_SCALAR_BYTES, FACT_EXECUTION_BINDING_ID,
            FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_VALIDATION_DEVICE, FACT_VALIDATION_REASON,
            FACT_VALIDATION_STREAM, Result,
        },
        types::TypeId,
    },
    runtime::{
        BoundaryKind, BoundaryObservation, ExecutionBindingId, ExecutionPlacement, QueryMemoryObservation,
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution, record_boundary_observation_in_domain,
    },
};

const FINITE_FLOAT32_OPERATION: &str = "validation::validate_finite_float32";
const FINITE_FLOAT32_BUFFER_OPERATION: &str = "validation::validate_finite_float32_buffer";
const FINITE_FLOAT64_OPERATION: &str = "validation::validate_finite_float64";
const INT64_BUFFER_RANGE_OPERATION: &str = "validation::validate_int64_buffer_range";

/// Successful proof that one Float32 or Float64 payload contains only finite values.
///
/// The proof is bound to the exact retained execution binding that performed
/// the device reduction. `observation()` exposes the bounded control-scalar
/// boundary separately from bulk Arrow host materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FiniteValidationProof {
    validated_values: usize,
    binding_id: ExecutionBindingId,
    placement: ExecutionPlacement,
    observation: BoundaryObservation,
}

impl FiniteValidationProof {
    /// Number of floating values covered by this proof.
    pub(crate) fn validated_values(self) -> usize {
        self.validated_values
    }

    /// Exact stream/memory-resource binding used for validation.
    pub(crate) fn binding_id(self) -> ExecutionBindingId {
        self.binding_id
    }

    /// Execution placement associated with the validated payload.
    pub(crate) fn placement(self) -> ExecutionPlacement {
        self.placement
    }

    /// Bounded control-scalar transfer observation.
    pub(crate) fn observation(self) -> BoundaryObservation {
        self.observation
    }
}

/// Successful proof that one exact raw Int64 buffer is within a half-open
/// range on its retained execution binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Int64RangeValidationProof {
    validated_values: usize,
    min_inclusive: i64,
    max_exclusive: i64,
    binding_id: ExecutionBindingId,
    placement: ExecutionPlacement,
    observation: BoundaryObservation,
}

impl Int64RangeValidationProof {
    pub(crate) fn validated_values(self) -> usize {
        self.validated_values
    }

    pub(crate) fn min_inclusive(self) -> i64 {
        self.min_inclusive
    }

    pub(crate) fn max_exclusive(self) -> i64 {
        self.max_exclusive
    }

    pub(crate) fn binding_id(self) -> ExecutionBindingId {
        self.binding_id
    }

    pub(crate) fn placement(self) -> ExecutionPlacement {
        self.placement
    }

    pub(crate) fn observation(self) -> BoundaryObservation {
        self.observation
    }
}

fn exact_buffer_element_count(buffer: &DeviceBuffer, element_bytes: usize, operation: &'static str) -> Result<usize> {
    let size_bytes = buffer.size_bytes().map_err(|error| error.reanchor(operation))?;
    if size_bytes % element_bytes != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "device buffer size is not aligned to the requested validation element type",
        )
        .with_detail("size_bytes", size_bytes.to_string())
        .with_detail("element_bytes", element_bytes.to_string())
        .with_detail("reason", "validation_buffer_size_misaligned"));
    }
    let element_count = size_bytes / element_bytes;
    if element_count > i32::MAX as usize {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "device buffer validation element count exceeds cudf size_type",
        )
        .with_detail("element_count", element_count.to_string())
        .with_detail("limit", i32::MAX.to_string())
        .with_detail("reason", "validation_element_count_exceeds_cudf_size_type"));
    }
    Ok(element_count)
}

fn buffer_validation_observation(
    buffer: &DeviceBuffer,
    validated_values: usize,
) -> (ExecutionPlacement, BoundaryObservation) {
    let placement = buffer.placement();
    let observation =
        BoundaryObservation::new(BoundaryKind::BoundedControlScalar, placement, validated_values, 1, true)
            .with_memory(QueryMemoryObservation::empty().with_downloaded_bytes(mem::size_of::<i32>()));
    record_boundary_observation_in_domain(buffer.execution_domain(), observation);
    (placement, observation)
}

pub(crate) fn validate_finite_float32_buffer(buffer: &DeviceBuffer) -> Result<FiniteValidationProof> {
    let validated_values = exact_buffer_element_count(buffer, mem::size_of::<f32>(), FINITE_FLOAT32_BUFFER_OPERATION)?;
    let input = buffer.as_ptr().map_err(|error| error.reanchor(FINITE_FLOAT32_BUFFER_OPERATION))? as *const f32;
    let execution = buffer.execution_domain().enter_execution()?;
    let mut non_finite_count = 0i32;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_float32_buffer_non_finite_count_ex(
                input,
                validated_values,
                &mut non_finite_count,
                execution.options(),
            )
        },
        FINITE_FLOAT32_BUFFER_OPERATION,
        [
            ("validated_values", validated_values.to_string()),
            ("validation_evidence", "bounded_control_scalar".to_string()),
        ],
    )?;

    let (placement, observation) = buffer_validation_observation(buffer, validated_values);
    let non_finite_count = usize::try_from(non_finite_count).map_err(|_| {
        Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            FINITE_FLOAT32_BUFFER_OPERATION,
            "finite-value validation returned a negative invalid-value count",
        )
        .with_detail("raw_non_finite_count", non_finite_count.to_string())
    })?;
    if non_finite_count != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            FINITE_FLOAT32_BUFFER_OPERATION,
            "Float32 device buffer contains values that are not finite",
        )
        .with_fact(FACT_VALIDATION_REASON, "non_finite_vector_value")
        .with_fact(FACT_EXECUTION_BINDING_ID, buffer.binding_id().get())
        .with_fact(FACT_VALIDATION_DEVICE, DeviceIdFact(placement.device.ordinal_raw()))
        .with_fact(FACT_VALIDATION_STREAM, placement.stream_debug)
        .with_fact(FACT_CONTROL_PLANE_SCALAR_BYTES, mem::size_of::<i32>())
        .with_detail("non_finite_count", non_finite_count.to_string())
        .with_detail("validated_values", validated_values.to_string())
        .with_detail("validation_evidence", "bounded_control_scalar")
        .with_detail("remedy", "replace or reject NaN and infinite values before GPU consumer execution"));
    }

    Ok(FiniteValidationProof { validated_values, binding_id: buffer.binding_id(), placement, observation })
}

pub(crate) fn validate_int64_buffer_range(
    buffer: &DeviceBuffer,
    min_inclusive: i64,
    max_exclusive: i64,
) -> Result<Int64RangeValidationProof> {
    if min_inclusive >= max_exclusive {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            INT64_BUFFER_RANGE_OPERATION,
            "Int64 validation range must be non-empty",
        )
        .with_detail("min_inclusive", min_inclusive.to_string())
        .with_detail("max_exclusive", max_exclusive.to_string())
        .with_detail("reason", "invalid_validation_range"));
    }
    let validated_values = exact_buffer_element_count(buffer, mem::size_of::<i64>(), INT64_BUFFER_RANGE_OPERATION)?;
    let input = buffer.as_ptr().map_err(|error| error.reanchor(INT64_BUFFER_RANGE_OPERATION))? as *const i64;
    let execution = buffer.execution_domain().enter_execution()?;
    let mut out_of_range_count = 0i32;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_int64_buffer_out_of_range_count_ex(
                input,
                validated_values,
                min_inclusive,
                max_exclusive,
                &mut out_of_range_count,
                execution.options(),
            )
        },
        INT64_BUFFER_RANGE_OPERATION,
        [
            ("validated_values", validated_values.to_string()),
            ("min_inclusive", min_inclusive.to_string()),
            ("max_exclusive", max_exclusive.to_string()),
            ("validation_evidence", "bounded_control_scalar".to_string()),
        ],
    )?;

    let (placement, observation) = buffer_validation_observation(buffer, validated_values);
    let out_of_range_count = usize::try_from(out_of_range_count).map_err(|_| {
        Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            INT64_BUFFER_RANGE_OPERATION,
            "Int64 range validation returned a negative invalid-value count",
        )
        .with_detail("raw_out_of_range_count", out_of_range_count.to_string())
    })?;
    if out_of_range_count != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            INT64_BUFFER_RANGE_OPERATION,
            "Int64 device buffer contains values outside the required range",
        )
        .with_fact(FACT_VALIDATION_REASON, "integer_value_out_of_range")
        .with_fact(FACT_EXECUTION_BINDING_ID, buffer.binding_id().get())
        .with_fact(FACT_VALIDATION_DEVICE, DeviceIdFact(placement.device.ordinal_raw()))
        .with_fact(FACT_VALIDATION_STREAM, placement.stream_debug)
        .with_fact(FACT_CONTROL_PLANE_SCALAR_BYTES, mem::size_of::<i32>())
        .with_detail("out_of_range_count", out_of_range_count.to_string())
        .with_detail("validated_values", validated_values.to_string())
        .with_detail("min_inclusive", min_inclusive.to_string())
        .with_detail("max_exclusive", max_exclusive.to_string())
        .with_detail("validation_evidence", "bounded_control_scalar")
        .with_detail("remedy", "reject or repair out-of-range integer values before downstream indexed access"));
    }

    Ok(Int64RangeValidationProof {
        validated_values,
        min_inclusive,
        max_exclusive,
        binding_id: buffer.binding_id(),
        placement,
        observation,
    })
}

pub(crate) fn validate_int64_buffer_range_or_sentinel(
    buffer: &DeviceBuffer,
    min_inclusive: i64,
    max_exclusive: i64,
    sentinel: i64,
) -> Result<()> {
    if min_inclusive >= max_exclusive {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            INT64_BUFFER_RANGE_OPERATION,
            "Int64 validation range must be non-empty",
        ));
    }
    let validated_values = exact_buffer_element_count(buffer, mem::size_of::<i64>(), INT64_BUFFER_RANGE_OPERATION)?;
    let input = buffer.as_ptr().map_err(|error| error.reanchor(INT64_BUFFER_RANGE_OPERATION))? as *const i64;
    let execution = buffer.execution_domain().enter_execution()?;
    let mut invalid_count = 0i32;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_int64_buffer_out_of_range_or_sentinel_count_ex(
                input,
                validated_values,
                min_inclusive,
                max_exclusive,
                sentinel,
                &mut invalid_count,
                execution.options(),
            )
        },
        INT64_BUFFER_RANGE_OPERATION,
        [
            ("validated_values", validated_values.to_string()),
            ("min_inclusive", min_inclusive.to_string()),
            ("max_exclusive", max_exclusive.to_string()),
            ("sentinel", sentinel.to_string()),
        ],
    )?;
    if invalid_count != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            INT64_BUFFER_RANGE_OPERATION,
            "Int64 device buffer contains values outside the range and sentinel",
        )
        .with_detail("invalid_count", invalid_count.to_string())
        .with_detail("sentinel", sentinel.to_string())
        .with_detail("validated_values", validated_values.to_string()));
    }
    Ok(())
}

pub(crate) fn add_scalar_to_float32_buffer(values: &mut DeviceBuffer, scalar: f32) -> Result<()> {
    let operation = "device_buffer::add_scalar_to_float32";
    let count = exact_buffer_element_count(values, mem::size_of::<f32>(), operation)?;
    if !scalar.is_finite() {
        return Err(Error::new_with(ErrorCode::ValueOutOfRange, operation, "Float32 scalar must be finite"));
    }
    let value_ptr = values.as_mut_ptr().map_err(|error| error.reanchor(operation))? as *mut f32;
    let execution = values.execution_domain().enter_execution()?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_float32_buffer_add_scalar_ex(value_ptr, count, scalar, execution.options()) },
        operation,
        [("element_count", count.to_string()), ("scalar", scalar.to_string())],
    )
}

pub(crate) fn fill_int64_where_float32_infinite(
    values: &mut DeviceBuffer,
    paired_values: &DeviceBuffer,
    fill_value: i64,
) -> Result<()> {
    let operation = INT64_BUFFER_RANGE_OPERATION;
    let count = exact_buffer_element_count(values, mem::size_of::<i64>(), operation)?;
    let paired_count = exact_buffer_element_count(paired_values, mem::size_of::<f32>(), operation)?;
    if paired_count != count
        || paired_values.binding_id() != values.binding_id()
        || paired_values.device() != values.device()
    {
        return Err(Error::new_with(
            ErrorCode::RuntimeBindingMismatch,
            operation,
            "paired buffers must share element count and exact execution binding",
        ));
    }
    let value_ptr = values.as_mut_ptr().map_err(|error| error.reanchor(operation))? as *mut i64;
    let paired_ptr = paired_values.as_ptr().map_err(|error| error.reanchor(operation))? as *const f32;
    let execution = values.execution_domain().enter_execution()?;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_int64_buffer_fill_where_float32_infinite_ex(
                value_ptr,
                paired_ptr,
                count,
                fill_value,
                execution.options(),
            )
        },
        operation,
        [("validated_values", count.to_string()), ("fill_value", fill_value.to_string())],
    )
}

pub(crate) fn replace_int64_value_and_set_paired_float32_infinite(
    values: &mut DeviceBuffer,
    paired_values: &mut DeviceBuffer,
    from: i64,
    to: i64,
) -> Result<()> {
    let operation = "device_buffer::replace_int64_value_and_set_paired_float32_infinite";
    let count = exact_buffer_element_count(values, mem::size_of::<i64>(), operation)?;
    if exact_buffer_element_count(paired_values, mem::size_of::<f32>(), operation)? != count
        || values.binding_id() != paired_values.binding_id()
        || values.device() != paired_values.device()
    {
        return Err(Error::new_with(
            ErrorCode::RuntimeBindingMismatch,
            operation,
            "paired buffers must share element count and exact execution binding",
        ));
    }
    let value_ptr = values.as_mut_ptr().map_err(|error| error.reanchor(operation))? as *mut i64;
    let paired_ptr = paired_values.as_mut_ptr().map_err(|error| error.reanchor(operation))? as *mut f32;
    let execution = values.execution_domain().enter_execution()?;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_int64_float32_buffer_replace_value_and_set_paired_infinite_ex(
                value_ptr,
                paired_ptr,
                count,
                from,
                to,
                execution.options(),
            )
        },
        operation,
        [("element_count", count.to_string())],
    )
}

pub(crate) fn validate_finite_float32_unless_int64_equals(
    values: &DeviceBuffer,
    paired_values: &DeviceBuffer,
    excluded_value: i64,
) -> Result<()> {
    let count = exact_buffer_element_count(values, mem::size_of::<f32>(), FINITE_FLOAT32_OPERATION)?;
    let paired_count = exact_buffer_element_count(paired_values, mem::size_of::<i64>(), FINITE_FLOAT32_OPERATION)?;
    if count != paired_count
        || values.binding_id() != paired_values.binding_id()
        || values.device() != paired_values.device()
    {
        return Err(Error::new_with(
            ErrorCode::RuntimeBindingMismatch,
            FINITE_FLOAT32_OPERATION,
            "paired buffers must share element count and exact execution binding",
        ));
    }
    let value_ptr = values.as_ptr().map_err(|error| error.reanchor(FINITE_FLOAT32_OPERATION))? as *const f32;
    let paired_ptr = paired_values.as_ptr().map_err(|error| error.reanchor(FINITE_FLOAT32_OPERATION))? as *const i64;
    let execution = values.execution_domain().enter_execution()?;
    let mut invalid_count = 0i32;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_float32_buffer_non_finite_unless_int64_equals_count_ex(
                value_ptr,
                paired_ptr,
                count,
                excluded_value,
                &mut invalid_count,
                execution.options(),
            )
        },
        FINITE_FLOAT32_OPERATION,
        [("validated_values", count.to_string()), ("excluded_value", excluded_value.to_string())],
    )?;
    if invalid_count != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            FINITE_FLOAT32_OPERATION,
            "Float32 buffer contains non-finite values outside excluded positions",
        )
        .with_detail("non_finite_count", invalid_count.to_string())
        .with_detail("validated_values", count.to_string()));
    }
    Ok(())
}

fn validate_float32_type(input: &ColumnView<'_>) -> Result<()> {
    let actual = input.data_type().map_err(|error| error.reanchor(FINITE_FLOAT32_OPERATION))?.id();
    if actual != TypeId::Float32 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            FINITE_FLOAT32_OPERATION,
            "finite-value validation requires a Float32 column",
        )
        .with_fact(FACT_LEFT_TYPE_ID, actual)
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Float32));
    }
    Ok(())
}

/// Validate a Float32 column using a device-side finite predicate and
/// reduction on its retained execution stream.
///
/// Null, NaN, positive infinity, and negative infinity are rejected. The
/// input payload remains device-resident; only one `i32` invalid-count scalar
/// is made host-visible before this function returns.
pub(crate) fn validate_finite_float32(input: &ColumnView<'_>) -> Result<FiniteValidationProof> {
    validate_float32_type(input)?;
    let execution =
        ResolvedExecution::unary(input.execution_domain(), &ExecutionOverrides::new(), FINITE_FLOAT32_OPERATION)?;
    validate_finite_with_exec(
        input,
        execution.output_domain(),
        execution.exec_options(),
        FINITE_FLOAT32_OPERATION,
        TypeId::Float32,
    )
}

fn validate_finite_with_exec(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
    element: TypeId,
) -> Result<FiniteValidationProof> {
    let validated_values = input.len().map_err(|error| error.reanchor(operation))?;
    let mut non_finite_count = 0i32;
    error::check_status_in_with_details(
        unsafe {
            match element {
                TypeId::Float32 => {
                    cudf_sys::cudf_float32_non_finite_count_ex(input.raw_handle(), &mut non_finite_count, exec_options)
                },
                TypeId::Float64 => {
                    cudf_sys::cudf_float64_non_finite_count_ex(input.raw_handle(), &mut non_finite_count, exec_options)
                },
                _ => unreachable!("finite validation is restricted to floating elements"),
            }
        },
        operation,
        [
            ("validated_values", validated_values.to_string()),
            ("validation_evidence", "bounded_control_scalar".to_string()),
        ],
    )?;

    let control_scalar_bytes = mem::size_of::<i32>();
    let placement = output_domain.placement();
    let observation =
        BoundaryObservation::new(BoundaryKind::BoundedControlScalar, placement, validated_values, 1, true)
            .with_memory(QueryMemoryObservation::empty().with_downloaded_bytes(control_scalar_bytes));
    record_boundary_observation_in_domain(output_domain, observation);

    let non_finite_count = usize::try_from(non_finite_count).map_err(|_| {
        Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            operation,
            "finite-value validation returned a negative invalid-value count",
        )
        .with_detail("raw_non_finite_count", non_finite_count.to_string())
    })?;
    if non_finite_count != 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            format!("{element} input contains values that are not finite"),
        )
        .with_fact(FACT_VALIDATION_REASON, "non_finite_vector_value")
        .with_fact(FACT_EXECUTION_BINDING_ID, output_domain.binding_id().get())
        .with_fact(FACT_VALIDATION_DEVICE, DeviceIdFact(placement.device.ordinal_raw()))
        .with_fact(FACT_VALIDATION_STREAM, placement.stream_debug)
        .with_fact(FACT_CONTROL_PLANE_SCALAR_BYTES, control_scalar_bytes)
        .with_detail("non_finite_count", non_finite_count.to_string())
        .with_detail("validated_values", validated_values.to_string())
        .with_detail("validation_evidence", "bounded_control_scalar")
        .with_detail("remedy", "replace or reject null, NaN, and infinite values before GPU consumer execution"));
    }

    Ok(FiniteValidationProof { validated_values, binding_id: output_domain.binding_id(), placement, observation })
}

/// Validate a Float64 column with the same bounded control-scalar evidence as Float32.
pub(crate) fn validate_finite_float64(input: &ColumnView<'_>) -> Result<FiniteValidationProof> {
    let actual = input.data_type().map_err(|error| error.reanchor(FINITE_FLOAT64_OPERATION))?.id();
    if actual != TypeId::Float64 {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            FINITE_FLOAT64_OPERATION,
            "finite-value validation requires a Float64 column",
        )
        .with_fact(FACT_LEFT_TYPE_ID, actual)
        .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Float64));
    }
    let execution =
        ResolvedExecution::unary(input.execution_domain(), &ExecutionOverrides::new(), FINITE_FLOAT64_OPERATION)?;
    validate_finite_with_exec(
        input,
        execution.output_domain(),
        execution.exec_options(),
        FINITE_FLOAT64_OPERATION,
        TypeId::Float64,
    )
}

impl Column {
    /// Validate that every value in this Float32 column is finite.
    pub(crate) fn validate_finite_float32(&self) -> Result<FiniteValidationProof> {
        let view = self.view().map_err(|error| error.reanchor(FINITE_FLOAT32_OPERATION))?;
        validate_finite_float32(&view)
    }
}
