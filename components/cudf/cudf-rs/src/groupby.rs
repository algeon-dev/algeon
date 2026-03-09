//! GPU-accelerated groupby aggregation.

use crate::column::{Column, ColumnView};
use crate::error::{self, Result};
use crate::table::{Table, TableView};
use std::os::raw::c_void;
use std::ptr;

/// Aggregation type for groupby operations.
#[derive(Debug, Clone)]
pub enum Aggregation {
    Sum,
    Min,
    Max,
    /// Count non-null values (if `include_nulls` is false) or all values.
    Count { include_nulls: bool },
    Mean,
    Median,
    NUnique,
    /// Standard deviation with degrees-of-freedom correction.
    Std { ddof: i32 },
    /// Variance with degrees-of-freedom correction.
    Variance { ddof: i32 },
}

/// A single aggregation request: which column to aggregate and how.
pub struct AggregationRequest<'a> {
    /// The values column to aggregate.
    pub values: &'a ColumnView<'a>,
    /// One or more aggregations to apply to this column.
    pub aggregations: &'a [Aggregation],
}

/// Null handling policy for groupby keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullPolicy {
    Exclude = 0,
    Include = 1,
}

/// Result of a groupby aggregation.
pub struct GroupByResult {
    /// The unique key groups.
    pub keys: Table,
    /// Aggregation results, one `Vec<Column>` per request. Each inner vec
    /// has one column per aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

/// Create a C-side aggregation handle from an `Aggregation` enum.
unsafe fn create_agg_handle(agg: &Aggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    match agg {
        Aggregation::Sum => {
            error::check_status(cudf_sys::cudf_aggregation_create(0, &mut handle))?;
        }
        Aggregation::Min => {
            error::check_status(cudf_sys::cudf_aggregation_create(1, &mut handle))?;
        }
        Aggregation::Max => {
            error::check_status(cudf_sys::cudf_aggregation_create(2, &mut handle))?;
        }
        Aggregation::Count { include_nulls } => {
            let policy = if *include_nulls { 1 } else { 0 };
            error::check_status(cudf_sys::cudf_aggregation_create_count(
                policy,
                &mut handle,
            ))?;
        }
        Aggregation::Mean => {
            error::check_status(cudf_sys::cudf_aggregation_create(5, &mut handle))?;
        }
        Aggregation::Median => {
            error::check_status(cudf_sys::cudf_aggregation_create(6, &mut handle))?;
        }
        Aggregation::NUnique => {
            error::check_status(cudf_sys::cudf_aggregation_create(7, &mut handle))?;
        }
        Aggregation::Std { ddof } => {
            error::check_status(cudf_sys::cudf_aggregation_create_std(
                *ddof,
                &mut handle,
            ))?;
        }
        Aggregation::Variance { ddof } => {
            error::check_status(cudf_sys::cudf_aggregation_create_variance(
                *ddof,
                &mut handle,
            ))?;
        }
    }
    Ok(handle)
}

/// Perform a groupby aggregation.
///
/// Groups rows by the `keys` table, then applies aggregations from each
/// request to the corresponding values column.
pub fn groupby_aggregate(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
) -> Result<GroupByResult> {
    let num_requests = requests.len() as i32;

    // Collect values column view handles.
    let value_handles: Vec<*mut c_void> = requests
        .iter()
        .map(|r| r.values.raw_handle())
        .collect();

    // Collect agg_counts and create aggregation handles.
    let mut agg_counts: Vec<i32> = Vec::with_capacity(requests.len());
    let mut all_agg_handles: Vec<*mut c_void> = Vec::new();

    for req in requests {
        agg_counts.push(req.aggregations.len() as i32);
        for agg in req.aggregations {
            let handle = unsafe { create_agg_handle(agg)? };
            all_agg_handles.push(handle);
        }
    }

    let mut out_keys = ptr::null_mut();
    // This receives a pointer to a heap-allocated array of column handles
    // (cudf_column_handle* = void**).
    let mut out_values: *mut *mut c_void = ptr::null_mut();
    let mut out_total_columns: i32 = 0;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys.raw_handle(),
            value_handles.as_ptr(),
            all_agg_handles.as_ptr(),
            agg_counts.as_ptr(),
            num_requests,
            null_policy as i32,
            &mut out_keys,
            &mut out_values,
            &mut out_total_columns,
        )
    };

    // Destroy aggregation handles regardless of success/failure.
    for handle in &all_agg_handles {
        unsafe { cudf_sys::cudf_aggregation_destroy(*handle) };
    }

    error::check_status(status)?;

    let keys_table = unsafe { Table::from_raw(out_keys) };

    // Collect result columns from the flat array.
    let total = out_total_columns as usize;
    let col_array = unsafe { std::slice::from_raw_parts(out_values, total) };

    let mut results: Vec<Vec<Column>> = Vec::with_capacity(requests.len());
    let mut offset = 0;
    for count in &agg_counts {
        let n = *count as usize;
        let mut cols = Vec::with_capacity(n);
        for i in 0..n {
            cols.push(unsafe { Column::from_raw(col_array[offset + i]) });
        }
        offset += n;
        results.push(cols);
    }

    // Free the array itself (not the columns — we took ownership above).
    unsafe { cudf_sys::cudf_free_column_array(out_values) };

    Ok(GroupByResult {
        keys: keys_table,
        results,
    })
}
