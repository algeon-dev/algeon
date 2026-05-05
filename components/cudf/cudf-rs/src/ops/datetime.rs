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

/// Datetime components supported by cudf-rs.
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
        Some(execution.exec_options()),
        "datetime::extract_component",
    )
}

pub(crate) fn extract_component_impl(
    input: &ColumnView<'_>,
    component: DatetimeComponent,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let input_type_id = require_timestamp_input(input, component, operation)?;
    let mut out = ptr::null_mut();
    let component_ffi = component.to_raw();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_datetime_extract_component_ex(input.raw_handle(), component_ffi, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_datetime_extract_component(input.raw_handle(), component_ffi, &mut out) },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("component", component.name().to_string()),
            ("input_type_id", input_type_id.to_string()),
            ("output_type_id", TypeId::Int16.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
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
        Some(execution.exec_options()),
        "datetime::add_calendrical_months",
    )
}

pub(crate) fn add_calendrical_months_impl(
    input: &ColumnView<'_>,
    months: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    let input_type_id = require_timestamp_offset_input(input, operation)?;
    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_datetime_add_calendrical_months_scalar_ex(input.raw_handle(), months, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_datetime_add_calendrical_months_scalar(input.raw_handle(), months, &mut out) },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("months", months.to_string()),
            ("input_type_id", input_type_id.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}
