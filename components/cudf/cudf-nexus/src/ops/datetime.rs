//! GPU-accelerated datetime extraction operations.
//!
//! These functions operate on timestamp-typed columns and return newly
//! allocated `INT16` component columns. Null input elements produce null
//! output elements.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_TYPE_ID, Result},
        types::TypeId,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Datetime components supported by cudf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DatetimeComponent {
    /// Calendar year.
    Year = 0,
    /// Calendar month, 1 through 12.
    Month = 1,
    /// Day of month, 1 through 31.
    Day = 2,
}

/// Datetime floor/ceil/round frequencies supported by cudf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DatetimeRoundingFrequency {
    Day = 0,
    Hour = 1,
    Minute = 2,
    Second = 3,
    Millisecond = 4,
    Microsecond = 5,
    Nanosecond = 6,
}

impl DatetimeRoundingFrequency {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            DatetimeRoundingFrequency::Day => "day",
            DatetimeRoundingFrequency::Hour => "hour",
            DatetimeRoundingFrequency::Minute => "minute",
            DatetimeRoundingFrequency::Second => "second",
            DatetimeRoundingFrequency::Millisecond => "millisecond",
            DatetimeRoundingFrequency::Microsecond => "microsecond",
            DatetimeRoundingFrequency::Nanosecond => "nanosecond",
        }
    }

    pub(crate) fn to_raw(self) -> cudf_sys::cudf_datetime_rounding_frequency_t {
        match self {
            DatetimeRoundingFrequency::Day => cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_DAY,
            DatetimeRoundingFrequency::Hour => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_HOUR
            },
            DatetimeRoundingFrequency::Minute => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_MINUTE
            },
            DatetimeRoundingFrequency::Second => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_SECOND
            },
            DatetimeRoundingFrequency::Millisecond => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_MILLISECOND
            },
            DatetimeRoundingFrequency::Microsecond => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_MICROSECOND
            },
            DatetimeRoundingFrequency::Nanosecond => {
                cudf_sys::cudf_datetime_rounding_frequency_t::CUDF_DATETIME_ROUNDING_NANOSECOND
            },
        }
    }
}

impl DatetimeComponent {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            DatetimeComponent::Year => "year",
            DatetimeComponent::Month => "month",
            DatetimeComponent::Day => "day",
        }
    }

    pub(crate) fn to_raw(self) -> cudf_sys::cudf_datetime_component_t {
        match self {
            DatetimeComponent::Year => cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_YEAR,
            DatetimeComponent::Month => cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_MONTH,
            DatetimeComponent::Day => cudf_sys::cudf_datetime_component_t::CUDF_DATETIME_COMPONENT_DAY,
        }
    }
}

fn require_timestamp_input(
    input: &ColumnView<'_>,
    component: DatetimeComponent,
    operation: &'static str,
) -> Result<TypeId> {
    let type_id = input.data_type().map_err(|e| e.reanchor(operation))?.id();
    if type_id.is_timestamp() {
        Ok(type_id)
    } else {
        Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "datetime component extraction requires timestamp input",
        )
        .with_fact(FACT_TYPE_ID, type_id)
        .with_detail("expected_family", "timestamp")
        .with_detail("component", component.name()))
    }
}

fn require_timestamp_offset_input(input: &ColumnView<'_>, operation: &'static str) -> Result<TypeId> {
    let type_id = input.data_type().map_err(|e| e.reanchor(operation))?.id();
    if type_id.is_timestamp() {
        Ok(type_id)
    } else {
        Err(Error::new_with(ErrorCode::TypeMismatch, operation, "datetime offset requires timestamp input")
            .with_fact(FACT_TYPE_ID, type_id)
            .with_detail("expected_family", "timestamp"))
    }
}

/// Floor timestamp values to the requested frequency.
pub fn floor_datetimes(input: &ColumnView<'_>, frequency: DatetimeRoundingFrequency) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "datetime::floor")?;
    floor_datetimes_impl(input, frequency, execution.output_domain(), execution.exec_options(), "datetime::floor")
}

pub(crate) fn floor_datetimes_impl(
    input: &ColumnView<'_>,
    frequency: DatetimeRoundingFrequency,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let input_type_id = require_timestamp_offset_input(input, operation)?;
    let mut out = ptr::null_mut();
    let frequency_ffi = frequency.to_raw();
    let status = unsafe { cudf_sys::cudf_datetime_floor_ex(input.raw_handle(), frequency_ffi, &mut out, exec_options) };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("frequency", frequency.name().to_string()),
            ("input_type_id", input_type_id.to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Extract a datetime component as a new `INT16` column.
///
/// The input must be timestamp typed. This allocates a new device column and
/// does not materialize values to host memory; later host export or value
/// extraction may synchronize.
pub fn extract_component(input: &ColumnView<'_>, component: DatetimeComponent) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "datetime::extract_component")?;
    extract_component_impl(
        input,
        component,
        execution.output_domain(),
        execution.exec_options(),
        "datetime::extract_component",
    )
}

pub(crate) fn extract_component_impl(
    input: &ColumnView<'_>,
    component: DatetimeComponent,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let input_type_id = require_timestamp_input(input, component, operation)?;
    let mut out = ptr::null_mut();
    let component_ffi = component.to_raw();
    let status = unsafe {
        cudf_sys::cudf_datetime_extract_component_ex(input.raw_handle(), component_ffi, &mut out, exec_options)
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("component", component.name().to_string()),
            ("input_type_id", input_type_id.to_string()),
            ("output_type_id", TypeId::Int16.to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Add calendrical months to a timestamp column using libcudf month-end
/// semantics.
///
/// The input must be timestamp typed. The result has the same timestamp
/// resolution as the input.
pub fn add_calendrical_months(input: &ColumnView<'_>, months: i32) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "datetime::add_calendrical_months")?;
    add_calendrical_months_impl(
        input,
        months,
        execution.output_domain(),
        execution.exec_options(),
        "datetime::add_calendrical_months",
    )
}

pub(crate) fn add_calendrical_months_impl(
    input: &ColumnView<'_>,
    months: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    let input_type_id = require_timestamp_offset_input(input, operation)?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_datetime_add_calendrical_months_scalar_ex(input.raw_handle(), months, &mut out, exec_options)
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("months", months.to_string()),
            ("input_type_id", input_type_id.to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
