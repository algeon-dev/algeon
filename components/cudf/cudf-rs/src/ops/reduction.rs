//! GPU-accelerated reduction and scan operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DataType, TypeId, core::Column, ops::reduction};
//!
//! let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let view = col.view().unwrap();
//!
//! // Reduce to a single scalar.
//! let total = reduction::reduce(
//!     &view,
//!     &reduction::ReduceAgg::Sum,
//!     &DataType::new(TypeId::Int32),
//! )
//! .unwrap();
//! assert_eq!(total.value::<i32>().unwrap(), 60);
//!
//! // Cumulative (prefix) sum.
//! let cumsum = reduction::scan(
//!     &view,
//!     reduction::ScanAgg::Sum,
//!     reduction::ScanType::Inclusive,
//!     reduction::NullPolicy::Exclude,
//! )
//! .unwrap();
//! // cumsum: [10, 30, 60]
//! ```

use std::{os::raw::c_void, ptr};

use crate::core::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
    scalar::Scalar,
    types::DataType,
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

pub use crate::core::types::NullPolicy;

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
        ReduceAgg::Variance { .. } => "variance",
        ReduceAgg::Std { .. } => "std",
    }
}

pub(crate) fn checked_ddof_to_i32(ddof: u32, operation: &'static str) -> Result<i32> {
    i32::try_from(ddof).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "ddof exceeds libcudf i32 range")
            .with_operation(operation)
            .with_context("ddof", ddof.to_string())
    })
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
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_SUM, &mut handle),
                "reduction::create_reduce_agg_handle(sum)",
            )?;
        },
        ReduceAgg::Min => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MIN, &mut handle),
                "reduction::create_reduce_agg_handle(min)",
            )?;
        },
        ReduceAgg::Max => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MAX, &mut handle),
                "reduction::create_reduce_agg_handle(max)",
            )?;
        },
        ReduceAgg::Product => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_PRODUCT, &mut handle),
                "reduction::create_reduce_agg_handle(product)",
            )?;
        },
        ReduceAgg::Mean => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MEAN, &mut handle),
                "reduction::create_reduce_agg_handle(mean)",
            )?;
        },
        ReduceAgg::Median => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_MEDIAN, &mut handle),
                "reduction::create_reduce_agg_handle(median)",
            )?;
        },
        ReduceAgg::Any => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_ANY, &mut handle),
                "reduction::create_reduce_agg_handle(any)",
            )?;
        },
        ReduceAgg::All => {
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(CUDF_REDUCE_AGG_ALL, &mut handle),
                "reduction::create_reduce_agg_handle(all)",
            )?;
        },
        ReduceAgg::Variance { ddof } => {
            let ddof = checked_ddof_to_i32(*ddof, "reduction::create_reduce_agg_handle(variance)")?;
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create_variance(ddof, &mut handle),
                "reduction::create_reduce_agg_handle(variance)",
            )?;
        },
        ReduceAgg::Std { ddof } => {
            let ddof = checked_ddof_to_i32(*ddof, "reduction::create_reduce_agg_handle(std)")?;
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create_std(ddof, &mut handle),
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
    // SAFETY: handle is initialized to null and set by the C shim on success.
    error::check_status_in_with(
        cudf_sys::cudf_scan_aggregation_create(kind, &mut handle),
        "reduction::create_scan_agg_handle",
        [("scan_aggregation", format!("{:?}", agg).to_lowercase())],
    )?;
    Ok(handle)
}

/// Reduce a column to a single scalar value.
pub fn reduce(input: &ColumnView<'_>, agg: &ReduceAgg, output_type: &DataType) -> Result<Scalar> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    reduce_impl(input, agg, output_type, exec_options.as_ref(), "reduction::reduce")
        .map(|scalar| scalar.with_execution_domain(output_domain))
}

pub(crate) fn reduce_impl(
    input: &ColumnView<'_>,
    agg: &ReduceAgg,
    output_type: &DataType,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Scalar> {
    // SAFETY: create_reduce_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_reduce_agg_handle(agg)? };
    let dtype_handle = output_type.to_raw_handle_for("reduction::reduce")?;

    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). agg_handle and dtype_handle
    // were returned by successful FFI calls.
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_reduce_ex(input.raw_handle(), agg_handle, dtype_handle, &mut out, options)
        },
        None => unsafe { cudf_sys::cudf_reduce(input.raw_handle(), agg_handle, dtype_handle, &mut out) },
    };

    // SAFETY: agg_handle and dtype_handle are valid handles that must be
    // destroyed regardless of success/failure.
    unsafe {
        cudf_sys::cudf_reduce_aggregation_destroy(agg_handle);
        cudf_sys::cudf_data_type_destroy(dtype_handle);
    }

    error::check_status_in_with(
        status,
        operation,
        [
            ("aggregation", reduce_agg_name(agg).to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Scalar::from_raw(out) })
}

/// Compute a prefix scan (cumulative operation) on a column.
pub fn scan(input: &ColumnView<'_>, agg: ScanAgg, scan_type: ScanType, null_policy: NullPolicy) -> Result<Column> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    scan_impl(input, agg, scan_type, null_policy, exec_options.as_ref(), "reduction::scan")
        .map(|column| column.with_execution_domain(output_domain))
}

pub(crate) fn scan_impl(
    input: &ColumnView<'_>,
    agg: ScanAgg,
    scan_type: ScanType,
    null_policy: NullPolicy,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Column> {
    // SAFETY: create_scan_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_scan_agg_handle(agg)? };

    let mut out = ptr::null_mut();
    let scan_type_ffi = scan_type.to_raw();
    // SAFETY: input handle is valid (NonNull). agg_handle was returned by a
    // successful FFI call.
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_scan_ex(input.raw_handle(), agg_handle, scan_type_ffi, null_policy as i32, &mut out, options)
        },
        None => unsafe {
            cudf_sys::cudf_scan(input.raw_handle(), agg_handle, scan_type_ffi, null_policy as i32, &mut out)
        },
    };

    // SAFETY: agg_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_scan_aggregation_destroy(agg_handle) };

    error::check_status_in_with(
        status,
        operation,
        [
            ("aggregation", format!("{:?}", agg)),
            ("scan_type", format!("{:?}", scan_type)),
            ("null_policy", format!("{:?}", null_policy)),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Compute the min and max of a column simultaneously.
pub fn minmax(input: &ColumnView<'_>) -> Result<(Scalar, Scalar)> {
    let output_domain = std::sync::Arc::clone(input.execution_domain());
    let exec_options = output_domain.explicit_exec_options_raw();
    minmax_impl(input, exec_options.as_ref(), "reduction::minmax").map(|(min, max)| {
        (min.with_execution_domain(std::sync::Arc::clone(&output_domain)), max.with_execution_domain(output_domain))
    })
}

pub(crate) fn minmax_impl(
    input: &ColumnView<'_>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<(Scalar, Scalar)> {
    let mut out_min = ptr::null_mut();
    let mut out_max = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe { cudf_sys::cudf_minmax_ex(input.raw_handle(), &mut out_min, &mut out_max, options) },
        None => unsafe { cudf_sys::cudf_minmax(input.raw_handle(), &mut out_min, &mut out_max) },
    };
    error::check_status_in_with(
        status,
        operation,
        [("execution_context_provided", exec_options.is_some().to_string())],
    )?;
    // SAFETY: out_min/out_max were returned by a successful FFI call.
    Ok(unsafe { (Scalar::from_raw(out_min), Scalar::from_raw(out_max)) })
}
