//! GPU row partitioning operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        error::{self, Error, ErrorCode, Result},
        table::{Table, TableView},
    },
    runtime::execution_domain::ExecutionDomain,
};

#[derive(Debug)]
pub(crate) struct HashPartitionedTable {
    table: Table,
    offsets: Vec<usize>,
}

impl HashPartitionedTable {
    pub fn into_parts(self) -> (Table, Vec<usize>) {
        (self.table, self.offsets)
    }
}

pub(crate) fn hash_partition_impl(
    input: &TableView<'_>,
    column_indices: &[usize],
    num_partitions: usize,
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
    operation: &'static str,
) -> Result<HashPartitionedTable> {
    if column_indices.is_empty() {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            operation,
            "hash partition requires at least one key column",
        )
        .with_detail("reason", "empty_partition_keys"));
    }
    if num_partitions == 0 || num_partitions > i32::MAX as usize {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "hash partition requires 1..=i32::MAX partitions",
        )
        .with_detail("reason", "invalid_partition_count")
        .with_detail("num_partitions", num_partitions.to_string()));
    }
    let key_indices = column_indices
        .iter()
        .map(|index| {
            i32::try_from(*index).map_err(|_| {
                Error::new_with(ErrorCode::IndexOutOfBounds, operation, "hash partition key index exceeds i32::MAX")
                    .with_detail("reason", "partition_key_index_overflow")
                    .with_detail("column_index", index.to_string())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut offsets = vec![0i32; num_partitions + 1];
    let mut out = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_partition_ex(
            input.raw_handle(),
            key_indices.as_ptr(),
            key_indices.len() as i32,
            num_partitions as i32,
            &mut out,
            offsets.as_mut_ptr(),
            offsets.len(),
            exec_options,
        )
    };
    error::check_status_in_with_details(
        status,
        operation,
        [
            ("num_partitions", num_partitions.to_string()),
            ("key_count", column_indices.len().to_string()),
            ("execution_authority", "retained_domain".to_string()),
        ],
    )?;
    let offsets = offsets
        .into_iter()
        .map(|offset| {
            usize::try_from(offset).map_err(|_| {
                Error::new_with(ErrorCode::ValueOutOfRange, operation, "libcudf returned a negative partition offset")
                    .with_detail("reason", "negative_partition_offset")
                    .with_detail("partition_offset", offset.to_string())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let table = unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }?;
    Ok(HashPartitionedTable { table, offsets })
}
