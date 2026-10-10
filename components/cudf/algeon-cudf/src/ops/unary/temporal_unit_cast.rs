//! Casts between two timestamp types, or two duration types, of different
//! resolution, with the semantics of a DataFusion `CAST`.
//!
//! DataFusion casts without the safe flag, for explicit `CAST` and for the
//! casts that type coercion inserts, such as the one that takes a duration to
//! the unit of the timestamp it is added to. libcudf differs from Arrow in both
//! directions, and this module follows Arrow:
//!
//! - A coarser source multiplies the stored count and fails when a valid row's
//!   product leaves `Int64`. libcudf wraps.
//! - A finer source divides the stored count and truncates toward zero, so
//!   -1500 ms is -1 s. libcudf floors and gives -2 s. A timestamp cast to day
//!   resolution keeps libcudf's floor, because Arrow takes the calendar date
//!   there, which floors.

use std::sync::Arc;

use super::{cast_out_of_range_error, libcudf_cast, stored_counts};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::Result,
        scalar::Scalar,
        types::{DataType, TimestampResolution, TypeId},
    },
    ops::{
        binary::{self, BinaryOperator},
        reduction,
    },
    runtime::execution_domain::ExecutionDomain,
};

/// The resolutions of a cast between two timestamp types or between two
/// duration types.
pub(super) fn resolutions(from: TypeId, to: TypeId) -> Option<(TimestampResolution, TimestampResolution)> {
    from.timestamp_resolution()
        .zip(to.timestamp_resolution())
        .or_else(|| from.duration_resolution().zip(to.duration_resolution()))
}

pub(super) fn cast(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    (from, to): (TimestampResolution, TimestampResolution),
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let (from_ticks, to_ticks) = (from.nanoseconds_per_tick(), to.nanoseconds_per_tick());
    if from_ticks > to_ticks {
        let factor = from_ticks / to_ticks;
        validate_product_fits(input, input_type, output_type, factor, output_domain, exec_options, operation)?;
        return libcudf_cast(input, input_type, output_type, output_domain, exec_options, operation);
    }
    if to == TimestampResolution::Days {
        return libcudf_cast(input, input_type, output_type, output_domain, exec_options, operation);
    }

    let int64 = DataType::new(TypeId::Int64);
    let counts = stored_counts(input, input_type, output_domain, exec_options, operation)?;
    let divisor = Scalar::from_value_in_domain(to_ticks / from_ticks, output_domain)?;
    let quotients = binary::binary_operation_vs_impl(
        &counts.view()?,
        &divisor,
        BinaryOperator::Div,
        &int64,
        output_domain,
        exec_options,
        operation,
    )?;
    let duration_type = DataType::new(to.duration_type_id());
    let durations = libcudf_cast(&quotients.view()?, &int64, &duration_type, output_domain, exec_options, operation)?;
    if output_type.id().is_duration() {
        return Ok(durations);
    }
    libcudf_cast(&durations.view()?, &duration_type, output_type, output_domain, exec_options, operation)
}

/// Fails when a valid row's stored count times `factor` does not fit `Int64`.
///
/// The count fits exactly when it lies in `[i64::MIN / factor, i64::MAX / factor]`
/// with both bounds truncated toward zero, so one min/max reduction decides.
fn validate_product_fits(
    input: &ColumnView<'_>,
    input_type: &DataType,
    output_type: &DataType,
    factor: i64,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<()> {
    let counts = stored_counts(input, input_type, output_domain, exec_options, operation)?;
    let (min, max) = reduction::minmax_impl(&counts.view()?, output_domain, exec_options, operation)?;
    if !min.is_valid()? {
        return Ok(());
    }
    let out_of_range = |reason: &'static str| {
        cast_out_of_range_error(
            operation,
            input_type,
            output_type,
            reason,
            "cast to a finer resolution overflows Int64 in a valid row",
        )
    };
    if min.value::<i64>()? < i64::MIN / factor {
        return Err(out_of_range("below_target_range"));
    }
    if max.value::<i64>()? > i64::MAX / factor {
        return Err(out_of_range("above_target_range"));
    }
    Ok(())
}
