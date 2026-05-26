//! GPU-accelerated reduction and scan operations.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::scalar::Scalar;
use crate::types::DataType;
use std::os::raw::c_void;
use std::ptr;

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
    Variance { ddof: i32 },
    /// Standard deviation with degrees-of-freedom correction.
    Std { ddof: i32 },
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

/// Null handling policy for scan operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullPolicy {
    Exclude = 0,
    Include = 1,
}

/// Create a C-side reduce aggregation handle.
unsafe fn create_reduce_agg_handle(agg: &ReduceAggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    match agg {
        ReduceAggregation::Sum => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                0, &mut handle,
            ))?;
        }
        ReduceAggregation::Min => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                1, &mut handle,
            ))?;
        }
        ReduceAggregation::Max => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                2, &mut handle,
            ))?;
        }
        ReduceAggregation::Product => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                3, &mut handle,
            ))?;
        }
        ReduceAggregation::Mean => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                4, &mut handle,
            ))?;
        }
        ReduceAggregation::Median => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                5, &mut handle,
            ))?;
        }
        ReduceAggregation::Any => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                6, &mut handle,
            ))?;
        }
        ReduceAggregation::All => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create(
                7, &mut handle,
            ))?;
        }
        ReduceAggregation::Variance { ddof } => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create_variance(
                *ddof, &mut handle,
            ))?;
        }
        ReduceAggregation::Std { ddof } => {
            error::check_status(cudf_sys::cudf_reduce_aggregation_create_std(
                *ddof, &mut handle,
            ))?;
        }
    }
    Ok(handle)
}

/// Create a C-side scan aggregation handle.
unsafe fn create_scan_agg_handle(agg: ScanAggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    let kind = match agg {
        ScanAggregation::Sum => 0,
        ScanAggregation::Min => 1,
        ScanAggregation::Max => 2,
        ScanAggregation::Product => 3,
    };
    error::check_status(cudf_sys::cudf_scan_aggregation_create(
        kind, &mut handle,
    ))?;
    Ok(handle)
}

/// Reduce a column to a single scalar value.
pub fn reduce(
    input: &ColumnView<'_>,
    agg: &ReduceAggregation,
    output_type: &DataType,
) -> Result<Scalar> {
    let agg_handle = unsafe { create_reduce_agg_handle(agg)? };
    let dtype_handle = output_type.to_raw_handle()?;

    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_reduce(input.raw_handle(), agg_handle, dtype_handle, &mut out)
    };

    unsafe {
        cudf_sys::cudf_reduce_aggregation_destroy(agg_handle);
        cudf_sys::cudf_data_type_destroy(dtype_handle);
    }

    error::check_status(status)?;
    Ok(unsafe { Scalar::from_raw(out) })
}

/// Compute a prefix scan (cumulative operation) on a column.
pub fn scan(
    input: &ColumnView<'_>,
    agg: ScanAggregation,
    scan_type: ScanType,
    null_policy: NullPolicy,
) -> Result<Column> {
    let agg_handle = unsafe { create_scan_agg_handle(agg)? };

    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_scan(
            input.raw_handle(),
            agg_handle,
            scan_type as i32,
            null_policy as i32,
            &mut out,
        )
    };

    unsafe { cudf_sys::cudf_scan_aggregation_destroy(agg_handle) };

    error::check_status(status)?;
    Ok(unsafe { Column::from_raw(out) })
}

/// Compute the min and max of a column simultaneously.
pub fn minmax(input: &ColumnView<'_>) -> Result<(Scalar, Scalar)> {
    let mut out_min = ptr::null_mut();
    let mut out_max = ptr::null_mut();
    error::check_status(unsafe {
        cudf_sys::cudf_minmax(input.raw_handle(), &mut out_min, &mut out_max)
    })?;
    Ok(unsafe { (Scalar::from_raw(out_min), Scalar::from_raw(out_max)) })
}
