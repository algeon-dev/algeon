//! GPU-accelerated stream compaction: drop_nulls, distinct, and adjacent unique.

mod errors;

use std::{ptr, sync::Arc};

use self::errors::{key_index_out_of_bounds_error, unsupported_stream_compaction_validation_error};
pub use crate::core::types::NullEquality;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, FACT_KEYS_COUNT, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

fn check_status_with_keys_count<I>(
    status: cudf_sys::cudf_status_t,
    operation: &'static str,
    keys_count: usize,
    details: I,
) -> Result<()>
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    error::check_status_in(status, operation)
        .map_err(|err| err.with_fact(FACT_KEYS_COUNT, keys_count).with_details(details))
}

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
            DuplicateKeepOption::KeepAny => cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            DuplicateKeepOption::KeepFirst => cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST,
            DuplicateKeepOption::KeepLast => cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST,
            DuplicateKeepOption::KeepNone => cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE,
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
            NanEquality::AllEqual => cudf_sys::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            NanEquality::Unequal => cudf_sys::CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL,
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
    exec_options: &'a cudf_sys::cudf_exec_options_t,
    operation: &'static str,
}

impl<'a> StreamCompactionExecution<'a> {
    pub(crate) fn new(
        output_domain: &'a Arc<ExecutionDomain>,
        exec_options: &'a cudf_sys::cudf_exec_options_t,
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
    exec_options: &cudf_sys::cudf_exec_options_t,
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
    let status = unsafe {
        cudf_sys::cudf_drop_nulls_ex(
            input.raw_handle(),
            keys_i32.as_ptr(),
            keys_count,
            keep_threshold,
            &mut out,
            exec_options,
        )
    };
    check_status_with_keys_count(
        status,
        operation,
        keys_i32.len(),
        [("keep_threshold", keep_threshold.to_string()), ("execution_authority", "retained_domain".to_string())],
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
    let status = unsafe {
        cudf_sys::cudf_distinct_ex(
            input.raw_handle(),
            keys_i32.as_ptr(),
            keys_count,
            keep_ffi,
            null_eq_ffi,
            nan_eq_ffi,
            &mut out,
            execution.exec_options,
        )
    };
    check_status_with_keys_count(
        status,
        operation,
        keys_i32.len(),
        [
            ("keep", format!("{:?}", config.keep).to_lowercase()),
            ("null_equality", format!("{:?}", config.null_equality).to_lowercase()),
            ("nan_equality", format!("{:?}", config.nan_equality).to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

pub(crate) fn unique_impl(
    input: &TableView<'_>,
    keys: &[usize],
    keep: DuplicateKeepOption,
    null_equality: NullEquality,
    execution: StreamCompactionExecution<'_>,
) -> Result<Table> {
    let operation = execution.operation;
    let keys_i32 = checked_key_indices_i32(input, keys, operation)?;
    let keys_count = error::usize_to_i32(keys_i32.len(), operation, "keys_count")?;

    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_unique_ex(
            input.raw_handle(),
            keys_i32.as_ptr(),
            keys_count,
            keep.to_raw(),
            null_equality.to_stream_compaction_raw(),
            &mut out,
            execution.exec_options,
        )
    };
    check_status_with_keys_count(
        status,
        operation,
        keys_i32.len(),
        [
            ("keep", format!("{keep:?}").to_lowercase()),
            ("null_equality", format!("{null_equality:?}").to_lowercase()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    unsafe { Table::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

pub(crate) fn ordered_unique_indices_impl(
    input: &TableView<'_>,
    row_order: &ColumnView<'_>,
    keys: &[usize],
    keep: DuplicateKeepOption,
    execution: StreamCompactionExecution<'_>,
) -> Result<Column> {
    let operation = execution.operation;
    let keys_i32 = checked_key_indices_i32(input, keys, operation)?;
    let keys_count = error::usize_to_i32(keys_i32.len(), operation, "keys_count")?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_ordered_unique_indices_ex(
            input.raw_handle(),
            row_order.raw_handle(),
            keys_i32.as_ptr(),
            keys_count,
            keep.to_raw(),
            &mut out,
            execution.exec_options,
        )
    };
    check_status_with_keys_count(
        status,
        operation,
        keys_i32.len(),
        [("keep", format!("{keep:?}").to_lowercase()), ("execution_authority", "retained_domain".to_string())],
    )?;
    unsafe { Column::from_raw_with_domain(out, Arc::clone(execution.output_domain)) }
}

#[cfg(test)]
mod tests {
    use super::check_status_with_keys_count;
    use crate::core::error::{FactKeyName, FactValue};

    #[test]
    fn ffi_failure_records_keys_count_as_a_typed_fact() {
        let err = check_status_with_keys_count(
            cudf_sys::CUDF_ERROR_INVALID_ARGUMENT,
            "stream_compaction::test",
            2,
            [("execution_authority", "retained_domain".to_string())],
        )
        .expect_err("injected FFI failure should remain an error");

        assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(2)));
        assert_eq!(err.detail_value("keys_count"), None);
        assert_eq!(err.detail_value("execution_authority"), Some("retained_domain"));
    }
}
