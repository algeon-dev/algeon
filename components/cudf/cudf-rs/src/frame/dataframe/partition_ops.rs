use std::sync::Arc;

use smallvec::SmallVec;

use super::owned::DataFrame;
use crate::{
    core::error::Result,
    internal::resolve::resolve_column_names,
    ops::partitioning,
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

const PARTITION_KEY_INLINE_CAPACITY: usize = 4;

/// Hash-partitioned DataFrame rows plus partition row offsets.
#[derive(Debug, Clone)]
pub struct HashPartitionedDataFrame {
    dataframe: DataFrame,
    offsets: Vec<usize>,
}

impl HashPartitionedDataFrame {
    pub fn dataframe(&self) -> &DataFrame {
        &self.dataframe
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn into_parts(self) -> (DataFrame, Vec<usize>) {
        (self.dataframe, self.offsets)
    }
}

impl DataFrame {
    /// Hash-partition rows by the supplied key columns.
    pub fn hash_partition_by<I, S>(&self, by: I, num_partitions: usize) -> Result<HashPartitionedDataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.hash_partition_by_impl(by, num_partitions, Arc::clone(&self.domain), "dataframe::hash_partition_by")
    }

    pub(crate) fn hash_partition_by_in_context<I, S>(
        &self,
        by: I,
        num_partitions: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<HashPartitionedDataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.hash_partition_by_impl(by, num_partitions, output_domain, "dataframe::with_execution::hash_partition_by")
    }

    fn hash_partition_by_impl<I, S>(
        &self,
        by: I,
        num_partitions: usize,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<HashPartitionedDataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;
        let by: SmallVec<[String; PARTITION_KEY_INLINE_CAPACITY]> =
            by.into_iter().map(|name| name.as_ref().to_string()).collect();
        let by_refs: SmallVec<[&str; PARTITION_KEY_INLINE_CAPACITY]> = by.iter().map(String::as_str).collect();
        let key_indices = resolve_column_names(&self.columns, &by_refs, operation)?;
        let tv = self.build_table_view()?;
        let partitioned = partitioning::hash_partition_impl(
            &tv,
            &key_indices,
            num_partitions,
            &output_domain,
            execution.exec_options(),
            operation,
        )?;
        let (table, offsets) = partitioned.into_parts();
        let dataframe =
            Self::from_table_with_names(table.with_execution_domain(output_domain), self.column_names_owned())?;
        Ok(HashPartitionedDataFrame { dataframe, offsets })
    }
}
