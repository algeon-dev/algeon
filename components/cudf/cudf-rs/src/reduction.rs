//! GPU-accelerated reduction and scan operations.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     column::Column,
//!     reduction,
//!     types::{DataType, TypeId},
//! };
//!
//! let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let view = col.view().unwrap();
//!
//! // Reduce to a single scalar.
//! let total = reduction::reduce(
//!     &view,
//!     &reduction::ReduceAggregation::Sum,
//!     &DataType::new(TypeId::Int32),
//! )
//! .unwrap();
//! assert_eq!(total.value::<i32>().unwrap(), 60);
//!
//! // Cumulative (prefix) sum.
//! let cumsum = reduction::scan(
//!     &view,
//!     reduction::ScanAggregation::Sum,
//!     reduction::ScanType::Inclusive,
//!     reduction::NullPolicy::Exclude,
//! )
//! .unwrap();
//! // cumsum: [10, 30, 60]
//! ```

use std::{os::raw::c_void, ptr};

use crate::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
    scalar::Scalar,
    types::DataType,
};

/// Aggregation type for reduce operations.
#[derive(Debug, Clone)]
pub enum ReduceAggregation {
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
        ddof: i32,
    },
    /// Standard deviation with degrees-of-freedom correction.
    Std {
        ddof: i32,
    },
}

/// Aggregation type for scan (prefix) operations.
///
/// Only a subset of aggregations are supported for scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanAggregation {
    Sum,
    Min,
    Max,
    Product,
}

/// Scan type: inclusive or exclusive prefix scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ScanType {
    Inclusive = 0,
    Exclusive = 1,
}

pub use crate::types::NullPolicy;

fn reduce_agg_name(agg: &ReduceAggregation) -> &'static str {
    match agg {
        ReduceAggregation::Sum => "sum",
        ReduceAggregation::Min => "min",
        ReduceAggregation::Max => "max",
        ReduceAggregation::Product => "product",
        ReduceAggregation::Mean => "mean",
        ReduceAggregation::Median => "median",
        ReduceAggregation::Any => "any",
        ReduceAggregation::All => "all",
        ReduceAggregation::Variance { .. } => "variance",
        ReduceAggregation::Std { .. } => "std",
    }
}

/// Create a C-side reduce aggregation handle.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_reduce_aggregation_destroy`.
unsafe fn create_reduce_agg_handle(agg: &ReduceAggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    match agg {
        ReduceAggregation::Sum => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(0, &mut handle),
                "reduction::create_reduce_agg_handle(sum)",
            )?;
        },
        ReduceAggregation::Min => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(1, &mut handle),
                "reduction::create_reduce_agg_handle(min)",
            )?;
        },
        ReduceAggregation::Max => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(2, &mut handle),
                "reduction::create_reduce_agg_handle(max)",
            )?;
        },
        ReduceAggregation::Product => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(3, &mut handle),
                "reduction::create_reduce_agg_handle(product)",
            )?;
        },
        ReduceAggregation::Mean => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(4, &mut handle),
                "reduction::create_reduce_agg_handle(mean)",
            )?;
        },
        ReduceAggregation::Median => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(5, &mut handle),
                "reduction::create_reduce_agg_handle(median)",
            )?;
        },
        ReduceAggregation::Any => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(6, &mut handle),
                "reduction::create_reduce_agg_handle(any)",
            )?;
        },
        ReduceAggregation::All => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create(7, &mut handle),
                "reduction::create_reduce_agg_handle(all)",
            )?;
        },
        ReduceAggregation::Variance { ddof } => {
            if *ddof < 0 {
                return Err(Error::new(ErrorKind::InvalidOperation, "ddof must be >= 0")
                    .with_operation("reduction::create_reduce_agg_handle(variance)")
                    .with_context("ddof", ddof.to_string()));
            }
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create_variance(*ddof, &mut handle),
                "reduction::create_reduce_agg_handle(variance)",
            )?;
        },
        ReduceAggregation::Std { ddof } => {
            if *ddof < 0 {
                return Err(Error::new(ErrorKind::InvalidOperation, "ddof must be >= 0")
                    .with_operation("reduction::create_reduce_agg_handle(std)")
                    .with_context("ddof", ddof.to_string()));
            }
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_reduce_aggregation_create_std(*ddof, &mut handle),
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
unsafe fn create_scan_agg_handle(agg: ScanAggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    let kind = match agg {
        ScanAggregation::Sum => 0,
        ScanAggregation::Min => 1,
        ScanAggregation::Max => 2,
        ScanAggregation::Product => 3,
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
pub fn reduce(input: &ColumnView<'_>, agg: &ReduceAggregation, output_type: &DataType) -> Result<Scalar> {
    // SAFETY: create_reduce_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_reduce_agg_handle(agg)? };
    let dtype_handle = output_type.to_raw_handle_for("reduction::reduce")?;

    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). agg_handle and dtype_handle
    // were returned by successful FFI calls.
    let status = unsafe { cudf_sys::cudf_reduce(input.raw_handle(), agg_handle, dtype_handle, &mut out) };

    // SAFETY: agg_handle and dtype_handle are valid handles that must be
    // destroyed regardless of success/failure.
    unsafe {
        cudf_sys::cudf_reduce_aggregation_destroy(agg_handle);
        cudf_sys::cudf_data_type_destroy(dtype_handle);
    }

    error::check_status_in_with(
        status,
        "reduction::reduce",
        [
            ("aggregation", reduce_agg_name(agg).to_string()),
            ("output_type_id", format!("{:?}", output_type.id()).to_lowercase()),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Scalar::from_raw(out) })
}

/// Compute a prefix scan (cumulative operation) on a column.
pub fn scan(
    input: &ColumnView<'_>,
    agg: ScanAggregation,
    scan_type: ScanType,
    null_policy: NullPolicy,
) -> Result<Column> {
    // SAFETY: create_scan_agg_handle calls FFI with valid parameters.
    let agg_handle = unsafe { create_scan_agg_handle(agg)? };

    let mut out = ptr::null_mut();
    // SAFETY: input handle is valid (NonNull). agg_handle was returned by a
    // successful FFI call.
    let status =
        unsafe { cudf_sys::cudf_scan(input.raw_handle(), agg_handle, scan_type as i32, null_policy as i32, &mut out) };

    // SAFETY: agg_handle is a valid handle that must be destroyed.
    unsafe { cudf_sys::cudf_scan_aggregation_destroy(agg_handle) };

    error::check_status_in_with(
        status,
        "reduction::scan",
        [
            ("aggregation", format!("{:?}", agg)),
            ("scan_type", format!("{:?}", scan_type)),
            ("null_policy", format!("{:?}", null_policy)),
        ],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    Ok(unsafe { Column::from_raw(out) })
}

/// Compute the min and max of a column simultaneously.
pub fn minmax(input: &ColumnView<'_>) -> Result<(Scalar, Scalar)> {
    let mut out_min = ptr::null_mut();
    let mut out_max = ptr::null_mut();
    error::check_status_in(
        // SAFETY: input handle is valid (NonNull maintained by ColumnView).
        unsafe { cudf_sys::cudf_minmax(input.raw_handle(), &mut out_min, &mut out_max) },
        "reduction::minmax",
    )?;
    // SAFETY: out_min/out_max were returned by a successful FFI call.
    Ok(unsafe { (Scalar::from_raw(out_min), Scalar::from_raw(out_max)) })
}
