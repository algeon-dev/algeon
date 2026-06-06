//! Low-level groupby aggregation execution.

use std::{os::raw::c_void, ptr};

use smallvec::SmallVec;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

// GroupBy callers usually touch a handful of value columns and aggregations per
// request; larger plans spill to heap-backed storage automatically.
const GROUPBY_VALUE_HANDLE_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_COUNT_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_HANDLE_INLINE_CAPACITY: usize = 16;

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

pub use crate::core::types::NullPolicy;

/// Result of a groupby aggregation.
pub struct GroupByResult {
    /// The unique key groups.
    pub keys: Table,
    /// Aggregation results, one `Vec<Column>` per request. Each inner vec
    /// has one column per aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

fn groupby_aggregate_impl(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<GroupByResult> {
    let num_requests = i32::try_from(requests.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "number of aggregation requests exceeds i32::MAX")
            .with_operation(operation)
            .with_context("num_requests", requests.len().to_string())
    })?;

    let value_handles: SmallVec<[*mut c_void; GROUPBY_VALUE_HANDLE_INLINE_CAPACITY]> =
        requests.iter().map(|r| r.values.raw_handle()).collect();

    let mut agg_counts: SmallVec<[i32; GROUPBY_AGG_COUNT_INLINE_CAPACITY]> = SmallVec::with_capacity(requests.len());
    let mut all_agg_handles: SmallVec<[*mut c_void; GROUPBY_AGG_HANDLE_INLINE_CAPACITY]> = SmallVec::new();

    for req in requests {
        if req.aggregations.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "each aggregation request must contain at least one aggregation",
            )
            .with_operation(operation)
            .with_context("request_index", agg_counts.len().to_string())
            .with_context("num_requests", num_requests.to_string()));
        }

        let agg_count = i32::try_from(req.aggregations.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "aggregation count per request exceeds i32::MAX")
                .with_operation(operation)
                .with_context("request_index", agg_counts.len().to_string())
                .with_context("agg_count", req.aggregations.len().to_string())
        })?;
        agg_counts.push(agg_count);

        for agg in req.aggregations {
            match agg {
                Aggregation::Std { ddof } | Aggregation::Variance { ddof } if *ddof < 0 => {
                    return Err(Error::new(ErrorKind::InvalidOperation, "ddof must be >= 0")
                        .with_operation(operation)
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
    let mut out_values: *mut *mut c_void = ptr::null_mut();
    let mut out_total_columns: i32 = 0;

    let status = if let Some(options) = exec_options {
        unsafe {
            cudf_sys::cudf_groupby_aggregate_ex(
                keys.raw_handle(),
                value_handles.as_ptr(),
                all_agg_handles.as_ptr(),
                agg_counts.as_ptr(),
                num_requests,
                i32::try_from(all_agg_handles.len()).expect("aggregation count exceeds i32::MAX"),
                null_policy as i32,
                &mut out_keys,
                &mut out_values,
                &mut out_total_columns,
                options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_groupby_aggregate(
                keys.raw_handle(),
                value_handles.as_ptr(),
                all_agg_handles.as_ptr(),
                agg_counts.as_ptr(),
                num_requests,
                i32::try_from(all_agg_handles.len()).expect("aggregation count exceeds i32::MAX"),
                null_policy as i32,
                &mut out_keys,
                &mut out_values,
                &mut out_total_columns,
            )
        }
    };

    for handle in &all_agg_handles {
        unsafe { cudf_sys::cudf_aggregation_destroy(*handle) };
    }

    let total_requested_aggs: i32 = agg_counts.iter().sum();
    error::check_status_in_with(
        status,
        operation,
        [
            ("num_requests", num_requests.to_string()),
            ("total_requested_aggs", total_requested_aggs.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    let keys_table = unsafe { Table::from_raw(out_keys) };
    let total = out_total_columns as usize;
    let col_array = if total == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(out_values, total) } };

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

    unsafe { cudf_sys::cudf_free_column_array(out_values) };

    Ok(GroupByResult { keys: keys_table, results })
}

fn compatible_groupby_domain(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
) -> Result<std::sync::Arc<ExecutionDomain>> {
    let _ = requests;
    Ok(std::sync::Arc::clone(keys.execution_domain()))
}

fn bind_groupby_result_domain(result: GroupByResult, output_domain: std::sync::Arc<ExecutionDomain>) -> GroupByResult {
    GroupByResult {
        keys: result.keys.with_execution_domain(std::sync::Arc::clone(&output_domain)),
        results: result
            .results
            .into_iter()
            .map(|columns| {
                columns
                    .into_iter()
                    .map(|column| column.with_execution_domain(std::sync::Arc::clone(&output_domain)))
                    .collect()
            })
            .collect(),
    }
}

/// Create a C-side aggregation handle from an `Aggregation` enum.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_aggregation_destroy`.
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
    let output_domain = compatible_groupby_domain(keys, requests)?;
    let exec_options = output_domain.explicit_exec_options_raw();
    groupby_aggregate_impl(keys, requests, null_policy, exec_options.as_ref(), "groupby::groupby_aggregate")
        .map(|result| bind_groupby_result_domain(result, output_domain))
}

pub(crate) fn groupby_aggregate_with_exec_options(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<GroupByResult> {
    groupby_aggregate_impl(keys, requests, null_policy, Some(exec_options), operation)
}
