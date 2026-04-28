//! Low-level groupby aggregation execution.

use std::{os::raw::c_void, ptr, sync::Arc};

use smallvec::SmallVec;

pub use crate::core::types::NullPolicy;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Result},
        table::{Table, TableView},
    },
    ops::{
        groupby::errors::{
            foreign_groupby_contract_error, groupby_request_agg_count_limit_error, groupby_request_count_limit_error,
            groupby_total_agg_handle_count_limit_error, unsupported_groupby_validation_error,
        },
        reduction::{checked_ddof_to_i32, ddof_range_error},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

// GroupBy callers usually touch a handful of value columns and aggregations per
// request; larger plans spill to heap-backed storage automatically.
const GROUPBY_VALUE_HANDLE_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_COUNT_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_HANDLE_INLINE_CAPACITY: usize = 16;

/// Shared aggregation vocabulary for high-level and low-level groupby APIs.
#[derive(Debug, Clone)]
pub enum GroupByAgg {
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
        ddof: u32,
    },
    /// Variance with degrees-of-freedom correction.
    Variance {
        ddof: u32,
    },
}

/// A single aggregation request: which column to aggregate and how.
pub struct AggregationRequest<'a> {
    /// The values column to aggregate.
    pub values: &'a ColumnView<'a>,
    /// One or more aggregations to apply to this column.
    pub aggregations: &'a [GroupByAgg],
}

/// Output payload of a low-level groupby aggregation.
pub struct GroupByAggregationOutput {
    /// The unique key groups.
    pub keys: Table,
    /// Aggregation results, one `Vec<Column>` per request. Each inner vec
    /// has one column per aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

pub(crate) fn groupby_aggregate_impl(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<GroupByAggregationOutput> {
    let request_count_usize = requests.len();
    let request_count = i32::try_from(request_count_usize)
        .map_err(|_| groupby_request_count_limit_error(operation, request_count_usize))?;

    let value_handles: SmallVec<[*mut c_void; GROUPBY_VALUE_HANDLE_INLINE_CAPACITY]> =
        requests.iter().map(|r| r.values.raw_handle()).collect();

    let mut agg_counts: SmallVec<[i32; GROUPBY_AGG_COUNT_INLINE_CAPACITY]> = SmallVec::with_capacity(requests.len());
    let mut all_agg_handles: SmallVec<[*mut c_void; GROUPBY_AGG_HANDLE_INLINE_CAPACITY]> = SmallVec::new();

    for (request_index, req) in requests.iter().enumerate() {
        if req.aggregations.is_empty() {
            return Err(unsupported_groupby_validation_error(
                operation,
                "each aggregation request must contain at least one aggregation",
                "empty_request_aggregations",
                "provide at least one GroupByAgg per AggregationRequest",
            )
            .with_detail("request_index", request_index.to_string())
            .with_detail("request_count", request_count.to_string()));
        }

        let agg_count = i32::try_from(req.aggregations.len()).map_err(|_| {
            groupby_request_agg_count_limit_error(operation, request_index, req.aggregations.len(), request_count_usize)
        })?;
        agg_counts.push(agg_count);

        for agg in req.aggregations {
            match agg {
                GroupByAgg::Std { ddof } | GroupByAgg::Variance { ddof } if *ddof > i32::MAX as u32 => {
                    return Err(
                        ddof_range_error(operation, *ddof).with_detail("request_index", request_index.to_string())
                    );
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

    let null_policy_ffi = null_policy.to_raw();
    let status = if let Some(options) = exec_options {
        unsafe {
            cudf_sys::cudf_groupby_aggregate_ex(
                keys.raw_handle(),
                value_handles.as_ptr(),
                all_agg_handles.as_ptr(),
                agg_counts.as_ptr(),
                request_count,
                i32::try_from(all_agg_handles.len()).map_err(|_| {
                    groupby_total_agg_handle_count_limit_error(operation, all_agg_handles.len(), request_count_usize)
                })?,
                null_policy_ffi,
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
                request_count,
                i32::try_from(all_agg_handles.len()).map_err(|_| {
                    groupby_total_agg_handle_count_limit_error(operation, all_agg_handles.len(), request_count_usize)
                })?,
                null_policy_ffi,
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
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("request_count", request_count.to_string()),
            ("total_requested_aggs", total_requested_aggs.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;

    let keys_table = unsafe { Table::from_raw_with_domain(out_keys, Arc::clone(output_domain)) }?;
    let total = usize::try_from(out_total_columns).map_err(|_| {
        foreign_groupby_contract_error(
            operation,
            "FFI returned negative total column count",
            "result_columns",
            "negative_column_count",
            "ensure the foreign groupby backend returns a non-negative result column count",
        )
        .with_detail("reported_table_column_count", out_total_columns.to_string())
    })?;
    let col_array = if total == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(out_values, total) } };

    let mut results: Vec<Vec<Column>> = Vec::with_capacity(requests.len());
    let mut offset = 0;
    for count in &agg_counts {
        let n = *count as usize;
        let mut cols = Vec::with_capacity(n);
        for i in 0..n {
            cols.push(unsafe { Column::from_raw_with_domain(col_array[offset + i], Arc::clone(output_domain)) }?);
        }
        offset += n;
        results.push(cols);
    }

    unsafe { cudf_sys::cudf_free_column_array(out_values) };

    Ok(GroupByAggregationOutput { keys: keys_table, results })
}

fn compatible_groupby_domain(keys: &TableView<'_>, requests: &[AggregationRequest<'_>]) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        keys.execution_domain(),
        requests.iter().map(|request| request.values.execution_domain()),
        &ctx,
        "groupby::groupby_aggregate",
    )
}
/// Create a C-side aggregation handle from an [`GroupByAgg`] enum.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_aggregation_destroy`.
unsafe fn create_agg_handle(agg: &GroupByAgg) -> Result<*mut c_void> {
    use cudf_sys::{cudf_aggregation_kind_t::*, cudf_null_policy_t};

    let mut handle = ptr::null_mut();
    match agg {
        GroupByAgg::Sum => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_SUM, &mut handle),
                "groupby::create_agg_handle(sum)",
            )?;
        },
        GroupByAgg::Min => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_MIN, &mut handle),
                "groupby::create_agg_handle(min)",
            )?;
        },
        GroupByAgg::Max => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_MAX, &mut handle),
                "groupby::create_agg_handle(max)",
            )?;
        },
        GroupByAgg::Count { include_nulls } => {
            let policy = if *include_nulls {
                cudf_null_policy_t::CUDF_NULL_POLICY_INCLUDE
            } else {
                cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE
            };
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_count(policy, &mut handle),
                "groupby::create_agg_handle(count)",
            )?;
        },
        GroupByAgg::Mean => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_MEAN, &mut handle),
                "groupby::create_agg_handle(mean)",
            )?;
        },
        GroupByAgg::Median => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_MEDIAN, &mut handle),
                "groupby::create_agg_handle(median)",
            )?;
        },
        GroupByAgg::NUnique => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(CUDF_AGG_NUNIQUE, &mut handle),
                "groupby::create_agg_handle(nunique)",
            )?;
        },
        GroupByAgg::Std { ddof } => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_std(
                    checked_ddof_to_i32(*ddof, "groupby::create_agg_handle(std)")?,
                    &mut handle,
                ),
                "groupby::create_agg_handle(std)",
            )?;
        },
        GroupByAgg::Variance { ddof } => {
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_variance(
                    checked_ddof_to_i32(*ddof, "groupby::create_agg_handle(variance)")?,
                    &mut handle,
                ),
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
) -> Result<GroupByAggregationOutput> {
    let execution = compatible_groupby_domain(keys, requests)?;
    groupby_aggregate_impl(
        keys,
        requests,
        null_policy,
        execution.output_domain(),
        Some(execution.exec_options()),
        "groupby::groupby_aggregate",
    )
}
