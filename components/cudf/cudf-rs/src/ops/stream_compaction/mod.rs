//! GPU-accelerated stream compaction: drop_nulls, distinct (deduplicate).

mod errors;

use std::{ptr, sync::Arc};

use self::errors::{key_index_out_of_bounds_error, unsupported_stream_compaction_validation_error};
pub use crate::core::types::NullEquality;
use crate::{
    core::{
        error::{self, FACT_KEYS_COUNT, Result},
        table::{Table, TableView},
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

/// Policy for which duplicate rows to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DuplicateKeepOption {
    /// Keep an unspecified occurrence of each duplicate group.
    KeepAny = 0,
    /// Keep the first occurrence.
    KeepFirst = 1,
    /// Keep the last occurrence.
    KeepLast = 2,
    /// Remove all duplicates (keep none).
    KeepNone = 3,
}

impl DuplicateKeepOption {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_duplicate_keep_option_t {
        match self {
            DuplicateKeepOption::KeepAny => cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            DuplicateKeepOption::KeepFirst => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST
            },
            DuplicateKeepOption::KeepLast => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST
            },
            DuplicateKeepOption::KeepNone => {
                cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE
            },
        }
    }
}

/// Whether NaN elements should be considered equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NanEquality {
    /// All NaN values compare equal.
    AllEqual = 0,
    /// NaN values compare unequal (IEEE 754 behavior).
    Unequal = 1,
}

impl NanEquality {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_stream_compaction_nan_equality_t {
        match self {
            NanEquality::AllEqual => {
                cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL
            },
            NanEquality::Unequal => {
                cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL
            },
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DistinctConfig<'a> {
    keys: &'a [usize],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    nan_equality: NanEquality,
}

impl<'a> DistinctConfig<'a> {
    pub(crate) fn new(
        keys: &'a [usize],
        keep: DuplicateKeepOption,
        null_equality: NullEquality,
        nan_equality: NanEquality,
    ) -> Self {
        Self { keys, keep, null_equality, nan_equality }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StreamCompactionExecution<'a> {
    output_domain: &'a Arc<ExecutionDomain>,
    exec_options: Option<&'a cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
}

impl<'a> StreamCompactionExecution<'a> {
    pub(crate) fn new(
        output_domain: &'a Arc<ExecutionDomain>,
        exec_options: Option<&'a cudf_sys::cudf_exec_options_t>,
        operation: &'static str,
    ) -> Self {
        Self { output_domain, exec_options, operation }
    }
}

fn checked_key_indices_i32(input: &TableView<'_>, keys: &[usize], operation: &'static str) -> Result<Vec<i32>> {
    let num_columns = input.num_columns()?;

    keys.iter()
        .map(|&key| {
            if key >= num_columns {
                Err(key_index_out_of_bounds_error(operation, key, num_columns))
            } else {
                i32::try_from(key).map_err(|_| error::range_limit_error(operation, "column_index", "i32::MAX", key))
            }
        })
        .collect()
}

pub(crate) fn drop_nulls_impl(
    input: &TableView<'_>,
    keys: &[usize],
    keep_threshold: i32,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: Option<&cudf_sys::cudf_exec_options_t>,
    operation: &'static str,
) -> Result<Table> {
    let keys_i32 = checked_key_indices_i32(input, keys, operation)?;
    let keys_count = error::usize_to_i32(keys_i32.len(), operation, "keys_count")?;

    if keep_threshold < 0 {
        return Err(unsupported_stream_compaction_validation_error(
            operation,
            "keep_threshold must be >= 0",
            "negative_keep_threshold",
            "use keep_threshold >= 0",
        )
        .with_fact(FACT_KEYS_COUNT, keys_i32.len())
        .with_detail("keep_threshold", keep_threshold.to_string()));
    }

    let mut out = ptr::null_mut();
    let status = match exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_drop_nulls_ex(
                input.raw_handle(),
                keys_i32.as_ptr(),
                keys_count,
                keep_threshold,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_drop_nulls(input.raw_handle(), keys_i32.as_ptr(), keys_count, keep_threshold, &mut out)
        },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("keys_count", keys_count.to_string()),
            ("keep_threshold", keep_threshold.to_string()),
            ("execution_context_provided", exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

pub(crate) fn distinct_impl(
    input: &TableView<'_>,
    config: DistinctConfig<'_>,
    execution: StreamCompactionExecution<'_>,
) -> Result<Table> {
    let operation = execution.operation;
    let keys_i32 = checked_key_indices_i32(input, config.keys, operation)?;
    let keys_count = error::usize_to_i32(keys_i32.len(), operation, "keys_count")?;

    let mut out = ptr::null_mut();
    let keep_ffi = config.keep.to_raw();
    let null_eq_ffi = config.null_equality.to_stream_compaction_raw();
    let nan_eq_ffi = config.nan_equality.to_raw();
    let status = match execution.exec_options {
        Some(options) => unsafe {
            cudf_sys::cudf_distinct_ex(
                input.raw_handle(),
                keys_i32.as_ptr(),
                keys_count,
                keep_ffi,
                null_eq_ffi,
                nan_eq_ffi,
                &mut out,
                options,
            )
        },
        None => unsafe {
            cudf_sys::cudf_distinct(
                input.raw_handle(),
                keys_i32.as_ptr(),
                keys_count,
                keep_ffi,
                null_eq_ffi,
                nan_eq_ffi,
                &mut out,
            )
        },
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("keys_count", keys_count.to_string()),
            ("keep", format!("{:?}", config.keep).to_lowercase()),
            ("null_equality", format!("{:?}", config.null_equality).to_lowercase()),
            ("nan_equality", format!("{:?}", config.nan_equality).to_lowercase()),
            ("execution_context_provided", execution.exec_options.is_some().to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

/// Drop rows from a table where key columns contain nulls.
///
/// Rows are kept only if at least `keep_threshold` of the specified `keys`
/// columns are non-null. For the common "drop any null" behavior, set
/// `keep_threshold` equal to the number of key columns.
///
/// # Example
/// ```no_run
/// use cudf::{Runtime, frame::dataframe::DropNaOptions};
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap(),
/// }
/// .unwrap();
/// let result = df.dropna(DropNaOptions::new()).unwrap();
/// assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3]);
/// ```
pub fn drop_nulls(input: &TableView<'_>, keys: &[usize], keep_threshold: i32) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "stream_compaction::drop_nulls")?;
    drop_nulls_impl(
        input,
        keys,
        keep_threshold,
        execution.output_domain(),
        Some(execution.exec_options()),
        "stream_compaction::drop_nulls",
    )
}

/// Create a new table with distinct rows based on key columns.
///
/// This is the GPU equivalent of `drop_duplicates(subset=[...])`.
/// The order of rows in the output is **not** specified.
///
/// # Example
/// ```no_run
/// use cudf::{Runtime, frame::dataframe::DropDuplicatesOptions};
///
/// let rt = Runtime::new().unwrap();
/// let df = cudf::dataframe! {
///     "value" => rt.series(&[1i32, 2, 1, 3]).unwrap(),
/// }
/// .unwrap();
/// let result = df.drop_duplicates(DropDuplicatesOptions::new()).unwrap();
/// assert_eq!(result.num_rows(), 3);
/// ```
pub fn distinct(
    input: &TableView<'_>,
    keys: &[usize],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    nan_equality: NanEquality,
) -> Result<Table> {
    let ctx = ExecutionOverrides::new();
    let execution = ResolvedExecution::unary(input.execution_domain(), &ctx, "stream_compaction::distinct")?;
    distinct_impl(
        input,
        DistinctConfig::new(keys, keep, null_equality, nan_equality),
        StreamCompactionExecution::new(
            execution.output_domain(),
            Some(execution.exec_options()),
            "stream_compaction::distinct",
        ),
    )
}
