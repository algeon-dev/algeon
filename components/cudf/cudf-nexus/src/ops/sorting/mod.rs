//! GPU-accelerated sorting operations.
//!
//! **Advanced/Core.** Most users should prefer
//! [`DataFrame::sort_values`](crate::DataFrame::sort_values) with
//! [`SortOptions`].
//!
//! # Example
//!
//! ```no_run
//! use cudf::{Runtime, SortOptions};
//!
//! let rt = Runtime::new().unwrap();
//! let df = cudf::dataframe! {
//!     "value" => rt.series(&[30i32, 10, 20]).unwrap(),
//! }
//! .unwrap();
//! let sorted = df.sort_values(["value"], SortOptions::new()).unwrap();
//! assert_eq!(
//!     sorted.series("value").unwrap().to_vec_strict::<i32>().unwrap(),
//!     vec![10, 20, 30]
//! );
//! ```

mod errors;
pub(crate) mod testing_hooks;

use std::{ptr, sync::Arc};

use self::errors::{sort_key_count_limit_error, sort_option_count_mismatch_error};
use crate::{
    core::{
        column::Column,
        error::{self, FACT_KEYS_COUNT, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

const SORTED_ORDER_INDEX_SIZE_BYTES: usize = std::mem::size_of::<i32>();
const SORTED_ORDER_MAX_ROWS: usize = i32::MAX as usize;

/// Column sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Order {
    Ascending = 0,
    Descending = 1,
}

/// Where nulls appear in the sorted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullOrder {
    After = 0,
    Before = 1,
}

fn validate_sort_params(
    operation: &'static str,
    num_keys: usize,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<()> {
    if let Some(orders) = column_order
        && orders.len() != num_keys
    {
        return Err(sort_option_count_mismatch_error(
            operation,
            format!("column_order length ({}) must equal number of keys ({})", orders.len(), num_keys),
            "column_order_length_mismatch",
            "provide either no column_order values or exactly one value per sort key",
            num_keys,
            orders.len(),
        ));
    }

    if let Some(nulls) = null_precedence
        && nulls.len() != num_keys
    {
        return Err(sort_option_count_mismatch_error(
            operation,
            format!("null_precedence length ({}) must equal number of keys ({})", nulls.len(), num_keys),
            "null_precedence_length_mismatch",
            "provide either no null_precedence values or exactly one value per sort key",
            num_keys,
            nulls.len(),
        ));
    }

    Ok(())
}

fn checked_num_keys(operation: &'static str, num_keys: usize) -> Result<i32> {
    i32::try_from(num_keys).map_err(|_| sort_key_count_limit_error(operation, num_keys))
}

/// Helper: convert Option slices to raw pointers.
struct SortParams {
    column_order: *const cudf_sys::cudf_order_t,
    null_precedence: *const cudf_sys::cudf_null_order_t,
    _resolved_null_precedence: Option<Vec<NullOrder>>,
}

#[derive(Clone, Copy)]
pub(crate) struct SortKeyConfig<'a> {
    column_order: Option<&'a [Order]>,
    null_precedence: Option<&'a [NullOrder]>,
}

impl<'a> SortKeyConfig<'a> {
    pub(crate) fn new(column_order: Option<&'a [Order]>, null_precedence: Option<&'a [NullOrder]>) -> Self {
        Self { column_order, null_precedence }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SortExecution<'a> {
    output_domain: &'a Arc<ExecutionDomain>,
    exec_options: &'a cudf_sys::cudf_exec_options_t,
    operation: &'static str,
}

impl<'a> SortExecution<'a> {
    pub(crate) fn new(
        output_domain: &'a Arc<ExecutionDomain>,
        exec_options: &'a cudf_sys::cudf_exec_options_t,
        operation: &'static str,
    ) -> Self {
        Self { output_domain, exec_options, operation }
    }
}

fn sort_params(column_order: Option<&[Order]>, null_precedence: Option<&[NullOrder]>) -> SortParams {
    let co = match column_order {
        Some(s) => s.as_ptr() as *const cudf_sys::cudf_order_t,
        None => ptr::null(),
    };
    let resolved_null_precedence = null_precedence.map(|nulls| {
        nulls
            .iter()
            .enumerate()
            .map(|(index, &null_order)| {
                let order = column_order.map_or(Order::Ascending, |orders| orders[index]);
                cudf_null_order_for_output_placement(order, null_order)
            })
            .collect::<Vec<_>>()
    });
    let np = match &resolved_null_precedence {
        Some(s) => s.as_ptr() as *const cudf_sys::cudf_null_order_t,
        None => ptr::null(),
    };
    SortParams { column_order: co, null_precedence: np, _resolved_null_precedence: resolved_null_precedence }
}

fn cudf_null_order_for_output_placement(order: Order, null_order: NullOrder) -> NullOrder {
    match (order, null_order) {
        (Order::Ascending, placement) => placement,
        (Order::Descending, NullOrder::Before) => NullOrder::After,
        (Order::Descending, NullOrder::After) => NullOrder::Before,
    }
}

pub(crate) fn sort_table_impl(
    input: &TableView<'_>,
    ordering: SortKeyConfig<'_>,
    stable: bool,
    execution: SortExecution<'_>,
) -> Result<Table> {
    let operation = execution.operation;
    let num_keys = input.num_columns()?;
    validate_sort_params(operation, num_keys, ordering.column_order, ordering.null_precedence)?;
    let num_keys_i32 = checked_num_keys(operation, num_keys)?;
    let params = sort_params(ordering.column_order, ordering.null_precedence);
    let mut out = ptr::null_mut();
    let status = if stable {
        unsafe {
            cudf_sys::cudf_stable_sort_ex(
                input.raw_handle(),
                params.column_order,
                params.null_precedence,
                num_keys_i32,
                &mut out,
                execution.exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_sort_ex(
                input.raw_handle(),
                params.column_order,
                params.null_precedence,
                num_keys_i32,
                &mut out,
                execution.exec_options,
            )
        }
    };

    error::check_status_in(status, operation).map_err(|err| {
        err.with_fact(FACT_KEYS_COUNT, num_keys).with_details([
            ("column_order_provided", ordering.column_order.is_some().to_string()),
            ("null_precedence_provided", ordering.null_precedence.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ])
    })?;

    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

pub(crate) fn sort_by_key_impl(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    ordering: SortKeyConfig<'_>,
    stable: bool,
    execution: SortExecution<'_>,
) -> Result<Table> {
    let operation = execution.operation;
    let num_key_columns = keys.num_columns()?;
    validate_sort_params(operation, num_key_columns, ordering.column_order, ordering.null_precedence)?;
    let num_key_columns_i32 = checked_num_keys(operation, num_key_columns)?;
    let params = sort_params(ordering.column_order, ordering.null_precedence);
    let mut out = ptr::null_mut();
    let status = if stable {
        unsafe {
            cudf_sys::cudf_stable_sort_by_key_ex(
                values.raw_handle(),
                keys.raw_handle(),
                params.column_order,
                params.null_precedence,
                num_key_columns_i32,
                &mut out,
                execution.exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_sort_by_key_ex(
                values.raw_handle(),
                keys.raw_handle(),
                params.column_order,
                params.null_precedence,
                num_key_columns_i32,
                &mut out,
                execution.exec_options,
            )
        }
    };

    error::check_status_in(status, operation).map_err(|err| {
        err.with_fact(FACT_KEYS_COUNT, num_key_columns).with_details([
            ("column_order_provided", ordering.column_order.is_some().to_string()),
            ("null_precedence_provided", ordering.null_precedence.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ])
    })?;

    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

pub(crate) fn sorted_order_impl(
    input: &TableView<'_>,
    ordering: SortKeyConfig<'_>,
    execution: SortExecution<'_>,
) -> Result<Column> {
    let operation = execution.operation;
    let num_keys = input.num_columns()?;
    validate_sort_params(operation, num_keys, ordering.column_order, ordering.null_precedence)?;
    let num_keys_i32 = checked_num_keys(operation, num_keys)?;
    let params = sort_params(ordering.column_order, ordering.null_precedence);
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_sorted_order_ex(
            input.raw_handle(),
            params.column_order,
            params.null_precedence,
            num_keys_i32,
            &mut out,
            execution.exec_options,
        )
    };

    error::check_status_in(status, operation).map_err(|err| {
        err.with_fact(FACT_KEYS_COUNT, num_keys).with_details([
            ("column_order_provided", ordering.column_order.is_some().to_string()),
            ("null_precedence_provided", ordering.null_precedence.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ])
    })?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

/// Sort the rows of a table by all its columns.
///
/// If `column_order` or `null_precedence` are `None`, cuDF defaults are
/// used (ascending, nulls after).
pub fn sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "sorting::sort")?;
    sort_table_impl(
        input,
        SortKeyConfig::new(column_order, null_precedence),
        false,
        SortExecution::new(execution.output_domain(), execution.exec_options(), "sorting::sort"),
    )
}

/// Stable sort (preserves relative order of equal elements).
pub fn stable_sort(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "sorting::stable_sort")?;
    sort_table_impl(
        input,
        SortKeyConfig::new(column_order, null_precedence),
        true,
        SortExecution::new(execution.output_domain(), execution.exec_options(), "sorting::stable_sort"),
    )
}

/// Sort the `values` table by a separate `keys` table.
pub fn sort_by_key(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution =
        ResolvedExecution::binary(values.execution_domain(), keys.execution_domain(), &ctx, "sorting::sort_by_key")?;
    sort_by_key_impl(
        values,
        keys,
        SortKeyConfig::new(column_order, null_precedence),
        false,
        SortExecution::new(execution.output_domain(), execution.exec_options(), "sorting::sort_by_key"),
    )
}

/// Stable sort the `values` table by a separate `keys` table.
///
/// Preserves the relative order of rows with equal keys.
pub fn stable_sort_by_key(
    values: &TableView<'_>,
    keys: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::binary(
        values.execution_domain(),
        keys.execution_domain(),
        &ctx,
        "sorting::stable_sort_by_key",
    )?;
    sort_by_key_impl(
        values,
        keys,
        SortKeyConfig::new(column_order, null_precedence),
        true,
        SortExecution::new(execution.output_domain(), execution.exec_options(), "sorting::stable_sort_by_key"),
    )
}

/// Return the row indices that would sort the table.
///
/// The result is a single INT32 column.
pub fn sorted_order(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<Column> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "sorting::sorted_order")?;
    sorted_order_impl(
        input,
        SortKeyConfig::new(column_order, null_precedence),
        SortExecution::new(execution.output_domain(), execution.exec_options(), "sorting::sorted_order"),
    )
}

/// Return exact bytes required for a sorted-order INT32 row-index column.
///
/// This is a pre-allocation planning helper for bounded callers. It mirrors
/// the row-index reservation shape used by DataFrame join gather maps:
/// libcudf sorted-order output uses `cudf::size_type` / `INT32` row indices,
/// so `output_rows` is rejected above `INT32_MAX`. This helper performs no
/// allocation and does not estimate libcudf sort temporary workspace.
pub fn sorted_order_reservation_size_bytes_for_len(output_rows: usize) -> Result<usize> {
    if output_rows > SORTED_ORDER_MAX_ROWS {
        return Err(error::Error::new_with(
            error::ErrorCode::ValueOutOfRange,
            "sorting::sorted_order_reservation_size_bytes_for_len",
            "sorted-order row count exceeds INT32_MAX",
        )
        .with_detail("reason", "row_count_exceeds_cudf_size_type")
        .with_detail("output_rows", output_rows.to_string())
        .with_detail("max_output_rows", SORTED_ORDER_MAX_ROWS.to_string()));
    }

    output_rows.checked_mul(SORTED_ORDER_INDEX_SIZE_BYTES).ok_or_else(|| {
        error::Error::new_with(
            error::ErrorCode::ValueOutOfRange,
            "sorting::sorted_order_reservation_size_bytes_for_len",
            "sorted-order reservation size overflowed usize",
        )
        .with_detail("reason", "size_overflowed_usize")
        .with_detail("output_rows", output_rows.to_string())
        .with_detail("index_size_bytes", SORTED_ORDER_INDEX_SIZE_BYTES.to_string())
    })
}

/// Check whether the table is already sorted according to the given order.
pub fn is_sorted(
    input: &TableView<'_>,
    column_order: Option<&[Order]>,
    null_precedence: Option<&[NullOrder]>,
) -> Result<bool> {
    let operation = "sorting::is_sorted";
    let num_keys = input.num_columns()?;
    validate_sort_params(operation, num_keys, column_order, null_precedence)?;
    let num_keys_i32 = checked_num_keys(operation, num_keys)?;
    let params = sort_params(column_order, null_precedence);
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, operation)?;
    let mut out: i32 = 0;
    error::check_status_in(
        // SAFETY: input handle is valid. co/np are either null or valid slices of num_keys elements.
        unsafe {
            cudf_sys::cudf_is_sorted_ex(
                input.raw_handle(),
                params.column_order,
                params.null_precedence,
                num_keys_i32,
                &mut out,
                execution.exec_options(),
            )
        },
        operation,
    )
    .map_err(|err| {
        err.with_fact(FACT_KEYS_COUNT, num_keys).with_details([
            ("column_order_provided", column_order.is_some().to_string()),
            ("null_precedence_provided", null_precedence.is_some().to_string()),
            ("execution_authority", "retained_domain".to_string()),
            ("host_boundary", "sortedness_boolean".to_string()),
        ])
    })?;
    Ok(out != 0)
}

/// High-level sort configuration for [`DataFrame::sort_values`](crate::DataFrame::sort_values).
///
/// Provides a cuDF-Python-like interface over the low-level `Order`/`NullOrder`
/// slices used by the core sorting functions.
///
/// # Defaults
///
/// - `descending`: empty (all columns ascending)
/// - `nulls_first`: empty (all columns nulls-last)
/// - `stable`: `false` (unstable sort)
///
/// # Length rules
///
/// `descending` and `nulls_first` vectors must be either:
/// - **empty** — meaning "use defaults for all sort keys", or
/// - **exactly equal** in length to the number of sort-key columns.
///
/// Any other length is rejected at sort time with
/// `ErrorKind::Unsupported` plus `ErrorCode::UnsupportedOperation`.
///
/// # Example
///
/// ```no_run
/// use cudf::SortOptions;
///
/// let opts = SortOptions::new()
///     .with_descending(vec![true, false])
///     .with_nulls_first(vec![false, true])
///     .with_stable(true);
/// ```
#[derive(Debug, Clone)]
pub struct SortOptions {
    /// Per-column descending flags. Empty means all ascending.
    pub descending: Vec<bool>,
    /// Per-column nulls-first flags. Empty means all nulls-last.
    pub nulls_first: Vec<bool>,
    /// If `true`, use a stable sort that preserves the relative order of
    /// equal elements. Default is `false` (unstable).
    pub stable: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SortOptions {
    /// Create a new `SortOptions` with all defaults (ascending, nulls-last,
    /// unstable).
    pub fn new() -> Self {
        Self { descending: Vec::new(), nulls_first: Vec::new(), stable: false }
    }

    /// Set per-column descending flags.
    pub fn with_descending(mut self, descending: Vec<bool>) -> Self {
        self.descending = descending;
        self
    }

    /// Set per-column nulls-first flags.
    pub fn with_nulls_first(mut self, nulls_first: Vec<bool>) -> Self {
        self.nulls_first = nulls_first;
        self
    }

    /// Set whether to use a stable sort.
    pub fn with_stable(mut self, stable: bool) -> Self {
        self.stable = stable;
        self
    }

    /// Validate and expand option vectors against the number of sort keys,
    /// returning `(Vec<Order>, Option<Vec<NullOrder>>)`.
    ///
    /// - Empty vectors are expanded to defaults for all keys.
    /// - Non-empty vectors must match `num_keys` exactly.
    pub(crate) fn resolve(
        &self,
        num_keys: usize,
        operation: &'static str,
    ) -> Result<(Vec<Order>, Option<Vec<NullOrder>>)> {
        let orders = self.resolve_orders(num_keys, operation)?;
        let null_orders = self.resolve_null_orders(num_keys, operation)?;
        Ok((orders, null_orders))
    }

    fn resolve_orders(&self, num_keys: usize, operation: &'static str) -> Result<Vec<Order>> {
        if self.descending.is_empty() {
            Ok(vec![Order::Ascending; num_keys])
        } else if self.descending.len() == num_keys {
            Ok(self.descending.iter().map(|&d| if d { Order::Descending } else { Order::Ascending }).collect())
        } else {
            Err(sort_option_count_mismatch_error(
                operation,
                format!(
                    "descending length ({}) must be 0 or equal to number of sort keys ({})",
                    self.descending.len(),
                    num_keys
                ),
                "descending_length_mismatch",
                "provide either zero descending flags or exactly one flag per sort key",
                num_keys,
                self.descending.len(),
            ))
        }
    }

    fn resolve_null_orders(&self, num_keys: usize, operation: &'static str) -> Result<Option<Vec<NullOrder>>> {
        if self.nulls_first.is_empty() {
            // Expand the high-level default explicitly so behavior stays stable
            // even if cuDF's implicit default changes across versions.
            Ok(Some(vec![NullOrder::After; num_keys]))
        } else if self.nulls_first.len() == num_keys {
            Ok(Some(self.nulls_first.iter().map(|&nf| if nf { NullOrder::Before } else { NullOrder::After }).collect()))
        } else {
            Err(sort_option_count_mismatch_error(
                operation,
                format!(
                    "nulls_first length ({}) must be 0 or equal to number of sort keys ({})",
                    self.nulls_first.len(),
                    num_keys
                ),
                "nulls_first_length_mismatch",
                "provide either zero nulls_first flags or exactly one flag per sort key",
                num_keys,
                self.nulls_first.len(),
            ))
        }
    }
}
