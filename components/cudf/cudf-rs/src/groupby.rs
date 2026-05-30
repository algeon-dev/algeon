//! GPU-accelerated groupby aggregation.

use std::{os::raw::c_void, ptr};

use crate::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
    table::{Table, TableView},
};

/// Aggregation type for groupby operations.
#[derive(Debug, Clone)]
pub enum Aggregation {
    Sum,
    Min,
    Max,
    /// Count non-null values (if `include_nulls` is false) or all values.
    Count {
        include_nulls: bool,
    },
    Mean,
    Median,
    NUnique,
    /// Standard deviation with degrees-of-freedom correction.
    Std {
        ddof: i32,
    },
    /// Variance with degrees-of-freedom correction.
    Variance {
        ddof: i32,
    },
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
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(0, &mut handle),
                "groupby::create_agg_handle(sum)",
            )?;
        },
        Aggregation::Min => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(1, &mut handle),
                "groupby::create_agg_handle(min)",
            )?;
        },
        Aggregation::Max => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(2, &mut handle),
                "groupby::create_agg_handle(max)",
            )?;
        },
        Aggregation::Count { include_nulls } => {
            let policy = if *include_nulls { 1 } else { 0 };
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_count(policy, &mut handle),
                "groupby::create_agg_handle(count)",
            )?;
        },
        Aggregation::Mean => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(5, &mut handle),
                "groupby::create_agg_handle(mean)",
            )?;
        },
        Aggregation::Median => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(6, &mut handle),
                "groupby::create_agg_handle(median)",
            )?;
        },
        Aggregation::NUnique => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(7, &mut handle),
                "groupby::create_agg_handle(nunique)",
            )?;
        },
        Aggregation::Std { ddof } => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_std(*ddof, &mut handle),
                "groupby::create_agg_handle(std)",
            )?;
        },
        Aggregation::Variance { ddof } => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_variance(*ddof, &mut handle),
                "groupby::create_agg_handle(variance)",
            )?;
        },
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
    let num_requests = i32::try_from(requests.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "number of aggregation requests exceeds i32::MAX")
            .with_operation("groupby::groupby_aggregate")
            .with_context("num_requests", requests.len().to_string())
    })?;

    // Collect values column view handles.
    let value_handles: Vec<*mut c_void> = requests.iter().map(|r| r.values.raw_handle()).collect();

    // Collect agg_counts and create aggregation handles.
    let mut agg_counts: Vec<i32> = Vec::with_capacity(requests.len());
    let mut all_agg_handles: Vec<*mut c_void> = Vec::new();

    for req in requests {
        if req.aggregations.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "each aggregation request must contain at least one aggregation",
            )
            .with_operation("groupby::groupby_aggregate")
            .with_context("request_index", agg_counts.len().to_string())
            .with_context("num_requests", num_requests.to_string()));
        }

        let agg_count = i32::try_from(req.aggregations.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "aggregation count per request exceeds i32::MAX")
                .with_operation("groupby::groupby_aggregate")
                .with_context("request_index", agg_counts.len().to_string())
                .with_context("agg_count", req.aggregations.len().to_string())
        })?;
        agg_counts.push(agg_count);

        for agg in req.aggregations {
            match agg {
                Aggregation::Std { ddof } | Aggregation::Variance { ddof } if *ddof < 0 => {
                    return Err(Error::new(ErrorKind::InvalidOperation, "ddof must be >= 0")
                        .with_operation("groupby::groupby_aggregate")
                        .with_context("request_index", agg_counts.len().to_string())
                        .with_context("ddof", ddof.to_string()));
                },
                _ => {},
            }
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

    let total_requested_aggs: i32 = agg_counts.iter().sum();
    error::check_status_in_with(
        status,
        "groupby::groupby_aggregate",
        [
            ("num_requests", num_requests.to_string()),
            ("total_requested_aggs", total_requested_aggs.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
        ],
    )?;

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

    Ok(GroupByResult { keys: keys_table, results })
}
