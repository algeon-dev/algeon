use super::{DataFrame, DataFrameJoinResult};
use crate::{
    Result,
    frame::series::Series,
    ops::{join::MergeOptions, sorting::SortOptions},
    runtime::ExecutionContext,
};

/// Scoped borrowed execution override for one `DataFrame` operation chain.
#[must_use]
pub struct DataFrameExecutionScope<'df, 'ctx, 'handles> {
    df: &'df DataFrame,
    ctx: &'ctx ExecutionContext<'handles>,
}

impl DataFrame {
    /// Borrow a per-operation execution override for subsequent high-level calls.
    pub fn in_context<'df, 'ctx, 'handles>(
        &'df self,
        ctx: &'ctx ExecutionContext<'handles>,
    ) -> DataFrameExecutionScope<'df, 'ctx, 'handles> {
        DataFrameExecutionScope { df: self, ctx }
    }
}

impl<'df, 'ctx, 'handles> DataFrameExecutionScope<'df, 'ctx, 'handles> {
    /// Sort using the borrowed execution override.
    pub fn sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.sort_values_in_context(by, options, self.ctx)
    }

    /// Gather rows using the borrowed execution override.
    pub fn gather(&self, indices: &Series) -> Result<DataFrame> {
        self.df.gather_in_context(indices, self.ctx)
    }

    /// Filter rows using the borrowed execution override.
    pub fn filter(&self, mask: &Series) -> Result<DataFrame> {
        self.df.filter_in_context(mask, self.ctx)
    }

    /// Concatenate with another frame using the borrowed execution override.
    pub fn concat(&self, other: &DataFrame) -> Result<DataFrame> {
        self.df.concat_in_context(other, self.ctx)
    }

    /// Merge with another frame using the borrowed execution override.
    pub fn merge(&self, right: &DataFrame, options: MergeOptions) -> Result<DataFrame> {
        self.df.merge_in_context(right, options, self.ctx)
    }

    /// Merge-split with another frame using the borrowed execution override.
    pub fn merge_split(&self, right: &DataFrame, options: MergeOptions) -> Result<DataFrameJoinResult> {
        self.df.merge_split_in_context(right, options, self.ctx)
    }
}
