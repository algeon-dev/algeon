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

use self::errors::{sort_key_count_limit_error, sort_option_count_mismatch_error, unsupported_sort_validation_error};
use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorCode, FACT_KEYS_COUNT, Result},
        table::{Table, TableView},
        types::{DataType, TypeId},
    },
    runtime::{
        DeviceId, execution_context::ExecutionOverrides, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution,
    },
};

const SORTED_ORDER_INDEX_SIZE_BYTES: usize = std::mem::size_of::<i32>();
const SORTED_ORDER_MAX_ROWS: usize = i32::MAX as usize;

/// Stable contract label for the libcudf-owned sorted-order radix preflight.
pub const SORTED_ORDER_RADIX_PREFLIGHT_CONTRACT: &str = "sorted_order_radix_owner_preflight";

/// Allocation components returned by the owner-level unstable one-key radix
/// sorted-order preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedOrderRadixPreflight {
    key_input_bytes: usize,
    key_output_bytes: usize,
    sequence_bytes: usize,
    temporary_workspace_bytes: usize,
    retained_order_bytes: usize,
    active_phase_peak_bytes: usize,
}

impl SortedOrderRadixPreflight {
    /// Requested bytes for the temporary transformed-key input buffer.
    pub fn key_input_bytes(self) -> usize {
        self.key_input_bytes
    }

    /// Requested bytes for the temporary key-output buffer.
    pub fn key_output_bytes(self) -> usize {
        self.key_output_bytes
    }

    /// Requested bytes for the INT32 sequence/value buffer.
    pub fn sequence_bytes(self) -> usize {
        self.sequence_bytes
    }

    /// Exact CUB temporary-storage query result for the selected device/order.
    pub fn temporary_workspace_bytes(self) -> usize {
        self.temporary_workspace_bytes
    }

    /// Requested bytes for the retained INT32 sorted-order output.
    pub fn retained_order_bytes(self) -> usize {
        self.retained_order_bytes
    }

    /// Checked sum of all five co-resident allocations during radix execution.
    pub fn active_phase_peak_bytes(self) -> usize {
        self.active_phase_peak_bytes
    }
}

pub(crate) fn sorted_order_radix_key_supported(type_id: TypeId) -> bool {
    type_id.is_signed_int()
        || type_id.is_unsigned_int()
        || type_id.is_timestamp()
        || type_id.is_duration()
        || type_id.is_float()
}

fn sorted_order_radix_preflight_unsupported(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    unsupported_sort_validation_error(operation, message, reason, remedy)
        .with_detail("resource", "sorted_order_radix_preflight")
}

fn checked_native_preflight_bytes(raw: u64, field: &'static str, operation: &'static str) -> Result<usize> {
    usize::try_from(raw).map_err(|_| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "native preflight byte count exceeds usize")
            .with_detail("resource", field)
            .with_detail("reason", "native_byte_count_exceeds_usize")
            .with_detail("native_bytes", raw.to_string())
    })
}

fn checked_preflight_sum(lhs: usize, rhs: usize, operation: &'static str) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "sorted-order radix preflight byte sum overflowed usize")
            .with_detail("resource", "active_phase_peak_bytes")
            .with_detail("reason", "component_sum_overflowed_usize")
    })
}

fn sorted_order_radix_preflight_impl(
    num_rows: usize,
    key_type: DataType,
    null_count: usize,
    stable: bool,
    key_order: Order,
    device: DeviceId,
    operation: &'static str,
) -> Result<SortedOrderRadixPreflight> {
    if num_rows > SORTED_ORDER_MAX_ROWS {
        return Err(sorted_order_radix_preflight_unsupported(
            operation,
            "sorted-order radix preflight rows exceed cudf::size_type",
            "row_count_exceeds_cudf_size_type",
            "reduce row_count to fit within cudf::size_type",
        )
        .with_detail("row_count", num_rows.to_string())
        .with_detail("max_row_count", SORTED_ORDER_MAX_ROWS.to_string()));
    }
    if stable {
        return Err(sorted_order_radix_preflight_unsupported(
            operation,
            "sorted-order radix preflight supports only unstable sorting",
            "stable_sort_unsupported",
            "use unstable sorting or retain the unresolved sort workspace contract",
        ));
    }
    if null_count != 0 {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            "sorted-order radix preflight requires a non-null key",
        )
        .with_detail("resource", "sort_key")
        .with_detail("reason", "nullable_key_unsupported")
        .with_detail("null_count", null_count.to_string())
        .with_detail("remedy", "drop nulls or use the unresolved general sorted-order path"));
    }
    if !sorted_order_radix_key_supported(key_type.id()) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "sorted-order radix preflight key type is unsupported",
        )
        .with_fact(crate::core::error::FACT_TYPE_ID, key_type.id())
        .with_detail("resource", "sort_key")
        .with_detail("reason", "key_type_not_radix_dispatchable")
        .with_detail(
            "remedy",
            "use an integral, chrono, or floating-point key or retain the unresolved general sorted-order path",
        ));
    }

    let num_rows = error::usize_to_i64(num_rows, operation, "num_rows")?;
    let null_count = error::usize_to_i64(null_count, operation, "null_count")?;
    let mut raw = cudf_sys::cudf_sorted_order_radix_preflight_result_t {
        key_input_bytes: 0,
        key_output_bytes: 0,
        sequence_bytes: 0,
        temporary_workspace_bytes: 0,
        retained_order_bytes: 0,
        active_phase_peak_bytes: 0,
    };
    let status = unsafe {
        cudf_sys::cudf_sorted_order_radix_preflight(
            num_rows,
            key_type.id().to_raw(),
            null_count,
            1,
            0,
            match key_order {
                Order::Ascending => cudf_sys::CUDF_ORDER_ASCENDING,
                Order::Descending => cudf_sys::CUDF_ORDER_DESCENDING,
            },
            device.ordinal_raw(),
            &mut raw,
        )
    };
    error::check_status_in(status, operation)?;

    let key_input_bytes = checked_native_preflight_bytes(raw.key_input_bytes, "key_input_bytes", operation)?;
    let key_output_bytes = checked_native_preflight_bytes(raw.key_output_bytes, "key_output_bytes", operation)?;
    let sequence_bytes = checked_native_preflight_bytes(raw.sequence_bytes, "sequence_bytes", operation)?;
    let temporary_workspace_bytes =
        checked_native_preflight_bytes(raw.temporary_workspace_bytes, "temporary_workspace_bytes", operation)?;
    let retained_order_bytes =
        checked_native_preflight_bytes(raw.retained_order_bytes, "retained_order_bytes", operation)?;
    let active_phase_peak_bytes =
        checked_native_preflight_bytes(raw.active_phase_peak_bytes, "active_phase_peak_bytes", operation)?;
    if sequence_bytes != retained_order_bytes {
        return Err(Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            operation,
            "native sorted-order radix preflight sequence and retained-order bytes differ",
        )
        .with_detail("resource", "sorted_order_radix_preflight_result")
        .with_detail("reason", "sequence_retained_component_mismatch")
        .with_detail("sequence_bytes", sequence_bytes.to_string())
        .with_detail("retained_order_bytes", retained_order_bytes.to_string()));
    }
    let expected_peak = checked_preflight_sum(key_input_bytes, key_output_bytes, operation)?;
    let expected_peak = checked_preflight_sum(expected_peak, sequence_bytes, operation)?;
    let expected_peak = checked_preflight_sum(expected_peak, temporary_workspace_bytes, operation)?;
    let expected_peak = checked_preflight_sum(expected_peak, retained_order_bytes, operation)?;
    if expected_peak != active_phase_peak_bytes {
        return Err(Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            operation,
            "native sorted-order radix preflight active phase peak is not the checked component sum",
        )
        .with_detail("resource", "sorted_order_radix_preflight_result")
        .with_detail("reason", "active_phase_peak_component_sum_mismatch")
        .with_detail("expected_peak_bytes", expected_peak.to_string())
        .with_detail("reported_peak_bytes", active_phase_peak_bytes.to_string()));
    }

    Ok(SortedOrderRadixPreflight {
        key_input_bytes,
        key_output_bytes,
        sequence_bytes,
        temporary_workspace_bytes,
        retained_order_bytes,
        active_phase_peak_bytes,
    })
}

/// Query the libcudf-owned unstable one-key radix sorted-order allocation
/// contract for a candidate device without allocating or requiring a stream.
pub fn sorted_order_radix_preflight(
    num_rows: usize,
    key_type: DataType,
    null_count: usize,
    stable: bool,
    key_order: Order,
    device: DeviceId,
) -> Result<SortedOrderRadixPreflight> {
    sorted_order_radix_preflight_impl(
        num_rows,
        key_type,
        null_count,
        stable,
        key_order,
        device,
        "sorting::sorted_order_radix_preflight",
    )
}

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
