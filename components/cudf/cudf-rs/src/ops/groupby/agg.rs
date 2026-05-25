//! Low-level groupby aggregation execution.

use std::{marker::PhantomData, os::raw::c_void, ptr, ptr::NonNull, sync::Arc};

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
        reduction::{ScanAgg, checked_ddof_to_i32, ddof_range_error},
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

/// A single grouped scan request: which column to scan and how.
pub struct GroupByScanRequest<'a> {
    /// The values column to scan.
    pub values: &'a ColumnView<'a>,
    /// One or more cumulative aggregations to apply to this column.
    pub aggregations: &'a [ScanAgg],
}

/// Output payload of a low-level groupby aggregation.
pub struct GroupByAggregationOutput {
    /// The unique key groups.
    pub keys: Table,
    /// Aggregation results, one `Vec<Column>` per request. Each inner vec
    /// has one column per aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

/// Output payload of a low-level grouped scan.
pub struct GroupByScanOutput {
    /// Group labels for each output row.
    pub keys: Table,
    /// Scan results, one `Vec<Column>` per request. Each inner vec has one
    /// column per scan aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

/// Stable facts observed after preparing reusable groupby state.
///
/// These are direct facts from cuDF's `groupby::get_groups({})` call. The
/// grouped-key byte count describes the materialized grouped-key output that
/// is released before `prepare_groups` returns; it is not a retained internal
/// prepared-handle memory measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedGroupByFacts {
    /// Number of key groups after the prepared handle's null-key policy is
    /// applied.
    pub num_groups: usize,
    /// Device allocation footprint of the grouped-key table materialized
    /// during preparation.
    pub grouped_key_output_bytes: usize,
    /// Maximum input rows represented by any one prepared group.
    pub max_group_size: usize,
}

/// Reusable groupby state for one key table.
///
/// This is a low-level contract around a retained `cudf::groupby::groupby`
/// object. The key columns borrowed by the input [`TableView`] must outlive the
/// prepared handle. Each aggregate call may use different value columns and
/// aggregation requests, but they must be compatible with the original key row
/// count.
pub struct PreparedGroupBy<'keys> {
    handle: NonNull<c_void>,
    keys_domain: Arc<ExecutionDomain>,
    keys_are_sorted: bool,
    null_policy: NullPolicy,
    _keys: PhantomData<&'keys TableView<'keys>>,
}

impl<'keys> PreparedGroupBy<'keys> {
    /// Create reusable groupby state for `keys`.
    pub fn new(keys: &TableView<'keys>, null_policy: NullPolicy) -> Result<Self> {
        let ctx = ExecutionOverrides::new();
        Self::new_impl(keys, null_policy, false, &ctx, "groupby::PreparedGroupBy::new")
    }

    /// Create reusable groupby state for `keys` using a borrowed execution
    /// override.
    pub fn new_in(keys: &TableView<'keys>, null_policy: NullPolicy, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        Self::new_impl(keys, null_policy, false, ctx, "groupby::PreparedGroupBy::new_in")
    }

    /// Create reusable groupby state for keys already sorted ascending with
    /// nulls after.
    ///
    /// Passing unsorted keys violates cuDF's sorted-key groupby contract.
    pub fn new_sorted(keys: &TableView<'keys>, null_policy: NullPolicy) -> Result<Self> {
        let ctx = ExecutionOverrides::new();
        Self::new_impl(keys, null_policy, true, &ctx, "groupby::PreparedGroupBy::new_sorted")
    }

    /// Create reusable groupby state for sorted keys using a borrowed
    /// execution override.
    ///
    /// Passing unsorted keys violates cuDF's sorted-key groupby contract.
    pub fn new_sorted_in(
        keys: &TableView<'keys>,
        null_policy: NullPolicy,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        Self::new_impl(keys, null_policy, true, ctx, "groupby::PreparedGroupBy::new_sorted_in")
    }

    /// Build reusable grouping state and return cuDF-reported cardinality.
    ///
    /// This may allocate temporary grouped-key output internally. The returned
    /// facts include the exact grouped-key output footprint, but no retained
    /// internal groupby-state memory estimate is inferred.
    pub fn prepare_groups(&self) -> Result<PreparedGroupByFacts> {
        prepared_groupby_facts_raw(self.handle.as_ptr(), &self.keys_domain, "groupby::PreparedGroupBy::prepare_groups")
    }

    fn new_impl(
        keys: &TableView<'keys>,
        null_policy: NullPolicy,
        keys_are_sorted: bool,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::unary(keys.execution_domain(), ctx, operation)?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_prepared_groupby_create_ex(
                    keys.raw_handle(),
                    null_policy.to_raw(),
                    i32::from(keys_are_sorted),
                    &mut handle,
                    execution.exec_options(),
                )
            },
            operation,
            [("null_policy", (null_policy as i32).to_string()), ("keys_are_sorted", keys_are_sorted.to_string())],
        )?;
        Ok(Self {
            handle: error::require_non_null(handle, operation, "prepared groupby")?,
            keys_domain: Arc::clone(execution.output_domain()),
            keys_are_sorted,
            null_policy,
            _keys: PhantomData,
        })
    }

    /// Execute aggregation requests using the retained groupby state.
    pub fn aggregate(&self, requests: &[AggregationRequest<'_>]) -> Result<GroupByAggregationOutput> {
        let operation = "groupby::PreparedGroupBy::aggregate";
        let execution = compatible_prepared_groupby_domain(&self.keys_domain, requests, operation)?;
        groupby_aggregate_impl_with_prepared(
            self.handle.as_ptr(),
            requests,
            self.null_policy,
            self.keys_are_sorted,
            execution.output_domain(),
            execution.exec_options(),
            operation,
        )
    }
}

pub(crate) fn prepared_groupby_facts_raw(
    handle: *mut c_void,
    keys_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<PreparedGroupByFacts> {
    let exec_options = keys_domain.exec_options_raw();
    let mut num_groups: i64 = 0;
    let mut grouped_key_output_bytes: usize = 0;
    let mut max_group_size: i64 = 0;
    error::check_status_in_with_details(
        unsafe {
            cudf_sys::cudf_prepared_groupby_prepare_groups_ex(
                handle,
                &mut num_groups,
                &mut grouped_key_output_bytes,
                &mut max_group_size,
                &exec_options,
            )
        },
        operation,
        [("execution_authority", "retained_groupby".to_string())],
    )?;
    let num_groups = usize::try_from(num_groups).map_err(|_| {
        foreign_groupby_contract_error(
            operation,
            "FFI returned negative group count",
            "num_groups",
            "negative_group_count",
            "ensure the foreign prepared groupby backend returns a non-negative group count",
        )
        .with_detail("reported_num_groups", num_groups.to_string())
    })?;
    let max_group_size = usize::try_from(max_group_size).map_err(|_| {
        foreign_groupby_contract_error(
            operation,
            "FFI returned negative maximum group size",
            "max_group_size",
            "negative_max_group_size",
            "ensure the foreign prepared groupby backend returns a non-negative maximum group size",
        )
        .with_detail("reported_max_group_size", max_group_size.to_string())
    })?;
    Ok(PreparedGroupByFacts { num_groups, grouped_key_output_bytes, max_group_size })
}

impl Drop for PreparedGroupBy<'_> {
    fn drop(&mut self) {
        unsafe { cudf_sys::cudf_prepared_groupby_destroy(self.handle.as_ptr()) };
    }
}

pub(crate) fn groupby_aggregate_impl(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
    keys_are_sorted: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
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
    let keys_are_sorted_ffi = i32::from(keys_are_sorted);
    let total_agg_handle_count = i32::try_from(all_agg_handles.len()).map_err(|_| {
        groupby_total_agg_handle_count_limit_error(operation, all_agg_handles.len(), request_count_usize)
    })?;
    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate_with_keys_are_sorted_ex(
            keys.raw_handle(),
            value_handles.as_ptr(),
            all_agg_handles.as_ptr(),
            agg_counts.as_ptr(),
            request_count,
            total_agg_handle_count,
            null_policy_ffi,
            keys_are_sorted_ffi,
            &mut out_keys,
            &mut out_values,
            &mut out_total_columns,
            exec_options,
        )
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
            ("execution_authority", "retained_domain".to_string()),
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

pub(crate) fn groupby_aggregate_impl_with_prepared(
    prepared: *mut c_void,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
    keys_are_sorted: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
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

    let total_agg_handle_count = i32::try_from(all_agg_handles.len()).map_err(|_| {
        groupby_total_agg_handle_count_limit_error(operation, all_agg_handles.len(), request_count_usize)
    })?;
    let status = unsafe {
        cudf_sys::cudf_prepared_groupby_aggregate_ex(
            prepared,
            value_handles.as_ptr(),
            all_agg_handles.as_ptr(),
            agg_counts.as_ptr(),
            request_count,
            total_agg_handle_count,
            &mut out_keys,
            &mut out_values,
            &mut out_total_columns,
            exec_options,
        )
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
            ("keys_are_sorted", keys_are_sorted.to_string()),
            ("execution_authority", "retained_groupby".to_string()),
        ],
    )?;

    let keys_table = unsafe { Table::from_raw_with_domain(out_keys, Arc::clone(output_domain)) }?;
    let total = usize::try_from(out_total_columns).map_err(|_| {
        foreign_groupby_contract_error(
            operation,
            "FFI returned negative total column count",
            "result_columns",
            "negative_column_count",
            "ensure the foreign prepared groupby backend returns a non-negative result column count",
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

pub(crate) fn groupby_scan_impl(
    keys: &TableView<'_>,
    requests: &[GroupByScanRequest<'_>],
    null_policy: NullPolicy,
    keys_are_sorted: bool,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<GroupByScanOutput> {
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
                "each grouped scan request must contain at least one aggregation",
                "empty_request_aggregations",
                "provide at least one ScanAgg per GroupByScanRequest",
            )
            .with_detail("request_index", request_index.to_string())
            .with_detail("request_count", request_count.to_string()));
        }

        let agg_count = i32::try_from(req.aggregations.len()).map_err(|_| {
            groupby_request_agg_count_limit_error(operation, request_index, req.aggregations.len(), request_count_usize)
        })?;
        agg_counts.push(agg_count);

        for agg in req.aggregations {
            let handle = unsafe { create_groupby_scan_agg_handle(*agg)? };
            all_agg_handles.push(handle);
        }
    }

    let mut out_keys = ptr::null_mut();
    let mut out_values: *mut *mut c_void = ptr::null_mut();
    let mut out_total_columns: i32 = 0;

    let null_policy_ffi = null_policy.to_raw();
    let keys_are_sorted_ffi = i32::from(keys_are_sorted);
    let total_agg_count = i32::try_from(all_agg_handles.len()).map_err(|_| {
        groupby_total_agg_handle_count_limit_error(operation, all_agg_handles.len(), request_count_usize)
    })?;
    let status = unsafe {
        cudf_sys::cudf_groupby_scan_ex(
            keys.raw_handle(),
            value_handles.as_ptr(),
            all_agg_handles.as_ptr(),
            agg_counts.as_ptr(),
            request_count,
            total_agg_count,
            null_policy_ffi,
            keys_are_sorted_ffi,
            &mut out_keys,
            &mut out_values,
            &mut out_total_columns,
            exec_options,
        )
    };

    for handle in &all_agg_handles {
        unsafe { cudf_sys::cudf_groupby_scan_aggregation_destroy(*handle) };
    }

    let total_requested_aggs: i32 = agg_counts.iter().sum();
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("request_count", request_count.to_string()),
            ("total_requested_aggs", total_requested_aggs.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
            ("keys_are_sorted", keys_are_sorted.to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;

    let keys_table = unsafe { Table::from_raw_with_domain(out_keys, Arc::clone(output_domain)) }?;
    let total = usize::try_from(out_total_columns).map_err(|_| {
        foreign_groupby_contract_error(
            operation,
            "FFI returned negative total column count",
            "result_columns",
            "negative_column_count",
            "ensure the foreign groupby scan backend returns a non-negative result column count",
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

    Ok(GroupByScanOutput { keys: keys_table, results })
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

fn compatible_prepared_groupby_domain(
    keys_domain: &Arc<ExecutionDomain>,
    requests: &[AggregationRequest<'_>],
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        keys_domain,
        requests.iter().map(|request| request.values.execution_domain()),
        &ctx,
        operation,
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

/// Create a C-side grouped scan aggregation handle.
///
/// # Safety
/// Caller must destroy the returned handle via
/// `cudf_groupby_scan_aggregation_destroy`.
unsafe fn create_groupby_scan_agg_handle(agg: ScanAgg) -> Result<*mut c_void> {
    use cudf_sys::cudf_reduce_aggregation_kind_t::*;

    let mut handle = ptr::null_mut();
    let kind = match agg {
        ScanAgg::Sum => CUDF_REDUCE_AGG_SUM,
        ScanAgg::Min => CUDF_REDUCE_AGG_MIN,
        ScanAgg::Max => CUDF_REDUCE_AGG_MAX,
        ScanAgg::Product => CUDF_REDUCE_AGG_PRODUCT,
    };
    error::check_status_in_with_details(
        cudf_sys::cudf_groupby_scan_aggregation_create(kind, &mut handle),
        "groupby::create_groupby_scan_agg_handle",
        [("scan_aggregation", format!("{:?}", agg).to_lowercase())],
    )?;
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
        false,
        execution.output_domain(),
        execution.exec_options(),
        "groupby::groupby_aggregate",
    )
}

/// Perform a groupby aggregation over keys already sorted in ascending order
/// with nulls after.
///
/// This exposes cuDF's pre-sorted key contract. Callers must only use it when
/// every key column is sorted ascending with nulls ordered after non-null
/// values; otherwise results are undefined by the lower-level groupby contract.
pub fn groupby_aggregate_sorted(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
) -> Result<GroupByAggregationOutput> {
    let execution = compatible_groupby_domain(keys, requests)?;
    groupby_aggregate_impl(
        keys,
        requests,
        null_policy,
        true,
        execution.output_domain(),
        execution.exec_options(),
        "groupby::groupby_aggregate_sorted",
    )
}

/// Perform a grouped prefix scan over rows already sorted by group keys.
///
/// The output rows are aligned to the sorted input rows. Callers that need a
/// different final order should add their own row-position column before
/// sorting and restore that order after consuming the scan output.
pub fn groupby_scan_sorted(
    keys: &TableView<'_>,
    requests: &[GroupByScanRequest<'_>],
    null_policy: NullPolicy,
) -> Result<GroupByScanOutput> {
    let execution = compatible_groupby_scan_domain(keys, requests)?;
    groupby_scan_impl(
        keys,
        requests,
        null_policy,
        true,
        execution.output_domain(),
        execution.exec_options(),
        "groupby::groupby_scan_sorted",
    )
}

fn compatible_groupby_scan_domain(
    keys: &TableView<'_>,
    requests: &[GroupByScanRequest<'_>],
) -> Result<ResolvedExecution> {
    let ctx = ExecutionOverrides::new();
    ResolvedExecution::many(
        keys.execution_domain(),
        requests.iter().map(|request| request.values.execution_domain()),
        &ctx,
        "groupby::groupby_scan_sorted",
    )
}
