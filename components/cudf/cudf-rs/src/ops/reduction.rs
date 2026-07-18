//! GPU-accelerated reduction and scan operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::Runtime;
//!
//! let rt = Runtime::new().unwrap();
//! let series = rt.series(&[10i32, 20, 30]).unwrap();
//!
//! let total = series.sum_value::<i32>().unwrap();
//! assert_eq!(total, 60);
//!
//! let cumsum = series.cumsum().unwrap();
//! assert_eq!(cumsum.to_vec_strict::<i32>().unwrap(), vec![10, 30, 60]);
//! ```

use std::{os::raw::c_void, ptr, sync::Arc};

pub use crate::core::types::NullPolicy;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, Result},
        scalar::Scalar,
        types::DataType,
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Aggregation type for reduce operations.
#[derive(Debug, Clone)]
pub enum ReduceAgg {
    Sum,
    Min,
    Max,
    Product,
    Mean,
    Median,
    Any,
    All,
    /// Count distinct elements, with configurable null handling.
    NUnique {
        null_policy: NullPolicy,
    },
    /// Variance with degrees-of-freedom correction.
    Variance {
        ddof: u32,
    },
    /// Standard deviation with degrees-of-freedom correction.
    Std {
        ddof: u32,
    },
}

/// Aggregation type for scan (prefix) operations.
///
/// Only a subset of aggregations are supported for scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanAgg {
    /// Inclusive or exclusive cumulative sum.
    Sum,
    /// Inclusive or exclusive cumulative minimum.
    Min,
    /// Inclusive or exclusive cumulative maximum.
    Max,
    /// Inclusive or exclusive cumulative product.
    Product,
}

/// Scan type: inclusive or exclusive prefix scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ScanType {
    Inclusive = 0,
    Exclusive = 1,
}

impl ScanType {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_scan_type_t {
        match self {
            ScanType::Inclusive => cudf_sys::cudf_scan_type_t::CUDF_SCAN_INCLUSIVE,
            ScanType::Exclusive => cudf_sys::cudf_scan_type_t::CUDF_SCAN_EXCLUSIVE,
        }
    }
}

fn reduce_agg_name(agg: &ReduceAgg) -> &'static str {
    match agg {
        ReduceAgg::Sum => "sum",
        ReduceAgg::Min => "min",
        ReduceAgg::Max => "max",
        ReduceAgg::Product => "product",
        ReduceAgg::Mean => "mean",
        ReduceAgg::Median => "median",
        ReduceAgg::Any => "any",
        ReduceAgg::All => "all",
        ReduceAgg::NUnique { .. } => "nunique",
        ReduceAgg::Variance { .. } => "variance",
        ReduceAgg::Std { .. } => "std",
    }
}

pub(crate) fn ddof_range_error(operation: &'static str, ddof: u32) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, "ddof exceeds libcudf i32 range")
        .with_detail("reason", "ddof_exceeds_i32_range")
        .with_detail("remedy", "use a ddof value within libcudf's i32 range")
        .with_detail("ddof", ddof.to_string())
}

pub(crate) fn checked_ddof_to_i32(ddof: u32, operation: &'static str) -> Result<i32> {
    i32::try_from(ddof).map_err(|_| ddof_range_error(operation, ddof))
}

/// Create a C-side reduce aggregation handle.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_reduce_aggregation_destroy`.
unsafe fn create_reduce_agg_handle(agg: &ReduceAgg) -> Result<*mut c_void> {
    use cudf_sys::cudf_reduce_aggregation_kind_t::*;

    let mut handle = ptr::null_mut();
    match agg {
        ReduceAgg::Sum => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_SUM, &mut handle) },
                "reduction::create_reduce_agg_handle(sum)",
            )?;
        },
        ReduceAgg::Min => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MIN, &mut handle) },
                "reduction::create_reduce_agg_handle(min)",
            )?;
        },
        ReduceAgg::Max => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MAX, &mut handle) },
                "reduction::create_reduce_agg_handle(max)",
            )?;
        },
        ReduceAgg::Product => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_PRODUCT, &mut handle) },
                "reduction::create_reduce_agg_handle(product)",
            )?;
        },
        ReduceAgg::Mean => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MEAN, &mut handle) },
                "reduction::create_reduce_agg_handle(mean)",
            )?;
        },
        ReduceAgg::Median => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MEDIAN, &mut handle) },
                "reduction::create_reduce_agg_handle(median)",
            )?;
        },
        ReduceAgg::Any => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_ANY, &mut handle) },
                "reduction::create_reduce_agg_handle(any)",
            )?;
        },
        ReduceAgg::All => {
            error::check_status_in(
                // SAFETY: the aggregation kind and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_ALL, &mut handle) },
                "reduction::create_reduce_agg_handle(all)",
            )?;
        },
        ReduceAgg::NUnique { null_policy } => {
            error::check_status_in(
                // SAFETY: the null policy and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create_nunique(*null_policy as i32, &mut handle) },
                "reduction::create_reduce_agg_handle(nunique)",
            )?;
        },
        ReduceAgg::Variance { ddof } => {
            let ddof = checked_ddof_to_i32(*ddof, "reduction::create_reduce_agg_handle(variance)")?;
            error::check_status_in(
                // SAFETY: the checked ddof and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create_variance(ddof, &mut handle) },
                "reduction::create_reduce_agg_handle(variance)",
            )?;
        },
        ReduceAgg::Std { ddof } => {
            let ddof = checked_ddof_to_i32(*ddof, "reduction::create_reduce_agg_handle(std)")?;
            error::check_status_in(
                // SAFETY: the checked ddof and out-pointer are valid.
                unsafe { cudf_sys::cudf_reduce_aggregation_create_std(ddof, &mut handle) },
                "reduction::create_reduce_agg_handle(std)",
            )?;
        },
    }
    Ok(handle)
}

/// Create a C-side scan aggregation handle.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_scan_aggregation_destroy`.
unsafe fn create_scan_agg_handle(agg: ScanAgg) -> Result<*mut c_void> {
    use cudf_sys::cudf_reduce_aggregation_kind_t::*;

    let mut handle = ptr::null_mut();
    let kind = match agg {
        ScanAgg::Sum => CUDF_REDUCE_AGG_SUM,
        ScanAgg::Min => CUDF_REDUCE_AGG_MIN,
        ScanAgg::Max => CUDF_REDUCE_AGG_MAX,
        ScanAgg::Product => CUDF_REDUCE_AGG_PRODUCT,
    };
    // SAFETY: `kind` is one of the supported scan aggregations and `handle`
    // is a live out-pointer whose status is checked before use.
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_scan_aggregation_create(kind, &mut handle) },
        "reduction::create_scan_agg_handle",
        [("scan_aggregation", format!("{:?}", agg).to_lowercase())],
    )?;
    Ok(handle)
}

/// Reduce a column to a single scalar value.
pub fn reduce(input: &ColumnView<'_>, agg: &ReduceAgg, output_type: &DataType) -> Result<Scalar> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "reduction::reduce")?;
    reduce_impl(input, agg, output_type, execution.output_domain(), execution.exec_options(), "reduction::reduce")
}

pub(crate) fn reduce_impl(
    input: &ColumnView<'_>,
    agg: &ReduceAgg,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Scalar> {
    reduce_impl_with_status_error_mapper(input, agg, output_type, output_domain, exec_options, operation, |error| error)
}

pub(crate) fn reduce_impl_with_status_error_mapper<F>(
    input: &ColumnView<'_>,
    agg: &ReduceAgg,
    output_type: &DataType,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
    map_status_error: F,
) -> Result<Scalar>
where
    F: FnOnce(Error) -> Error,
{
    // SAFETY: create_reduce_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_reduce_agg_handle(agg).map_err(|error| error.reanchor(operation))? };
    let dtype_handle = output_type.to_raw_handle_for(operation)?;

    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). agg_handle and dtype_handle
    // were returned by successful FFI calls.
    let status =
        unsafe { cudf_sys::cudf_reduce_ex(input.raw_handle(), agg_handle, dtype_handle, &mut out, exec_options) };

    // SAFETY: agg_handle and dtype_handle are valid handles that must be
    // destroyed regardless of success/failure.
    unsafe {
        cudf_sys::cudf_reduce_aggregation_destroy(agg_handle);
        cudf_sys::cudf_data_type_destroy(dtype_handle);
    }

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("aggregation", reduce_agg_name(agg).to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )
    .map_err(map_status_error)?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Scalar::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Compute a prefix scan (cumulative operation) on a column.
pub fn scan(input: &ColumnView<'_>, agg: ScanAgg, scan_type: ScanType, null_policy: NullPolicy) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "reduction::scan")?;
    scan_impl(
        input,
        agg,
        scan_type,
        null_policy,
        execution.output_domain(),
        execution.exec_options(),
        "reduction::scan",
    )
}

pub(crate) fn scan_impl(
    input: &ColumnView<'_>,
    agg: ScanAgg,
    scan_type: ScanType,
    null_policy: NullPolicy,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<Column> {
    // SAFETY: create_scan_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_scan_agg_handle(agg).map_err(|error| error.reanchor(operation))? };

    let mut out = ptr::null_mut();
    let scan_type_ffi = scan_type.to_raw();
    // SAFETY: input handle is valid (NonNull). agg_handle was returned by a
    // successful FFI call.
    let status = unsafe {
        cudf_sys::cudf_scan_ex(
            input.raw_handle(),
            agg_handle,
            scan_type_ffi,
            null_policy.to_raw() as i32,
            &mut out,
            exec_options,
        )
    };

    // SAFETY: agg_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_scan_aggregation_destroy(agg_handle) };

    error::check_status_in_with_details(
        status,
        operation,
        [
            ("aggregation", format!("{:?}", agg)),
            ("scan_type", format!("{:?}", scan_type)),
            ("null_policy", format!("{:?}", null_policy)),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Compute the min and max of a column simultaneously.
pub fn minmax(input: &ColumnView<'_>) -> Result<(Scalar, Scalar)> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "reduction::minmax")?;
    minmax_impl(input, execution.output_domain(), execution.exec_options(), "reduction::minmax")
}

pub(crate) fn minmax_impl(
    input: &ColumnView<'_>,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<(Scalar, Scalar)> {
    let mut out_min = ptr::null_mut();
    let mut out_max = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_minmax_ex(input.raw_handle(), &mut out_min, &mut out_max, exec_options) };
    error::check_status_in_with_details(status, operation, [("execution_authority", "retained_domain".to_string())])?;
    // SAFETY: out_min/out_max were returned by a successful FFI call.
    unsafe {
        Ok((
            Scalar::from_raw_with_domain(out_min, Arc::clone(output_domain))?,
            Scalar::from_raw_with_domain(out_max, Arc::clone(output_domain))?,
        ))
    }
}
