use super::{DataFrame, DataFrameJoinResult, DropDuplicatesOptions, DropNaOptions};
use crate::{
    Expr, Result,
    frame::series::Series,
    ops::{join::MergeOptions, sorting::SortOptions},
    runtime::ExecutionContext,
};

macro_rules! dataframe_scope_forward {
    ($(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident;) => {
        $(#[$meta])*
        pub fn $method(&self, $($param: $pty),*) -> $ret {
            self.df.$target($($param,)* self.ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident; )*) => {
        $(
            dataframe_scope_forward!(
                $(#[$meta])*
                fn $method($($param : $pty),*) -> $ret => $target;
            );
        )*
    };
}

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

    /// Filter rows with a fused expression using the borrowed execution override.
    pub fn filter_expr(&self, predicate: Expr) -> Result<DataFrame> {
        self.df.filter_expr_in_context(predicate, self.ctx)
    }

    /// Filter rows where all predicates are true using the borrowed execution override.
    pub fn filter_all<I>(&self, predicates: I) -> Result<DataFrame>
    where I: IntoIterator<Item = Expr> {
        self.df.filter_all_in_context(predicates, self.ctx)
    }

    /// Filter rows where any predicate is true using the borrowed execution override.
    pub fn filter_any<I>(&self, predicates: I) -> Result<DataFrame>
    where I: IntoIterator<Item = Expr> {
        self.df.filter_any_in_context(predicates, self.ctx)
    }

    dataframe_scope_forward!(
        /// Return the first `n` rows using the borrowed execution override.
        fn head(n: usize) -> Result<DataFrame> => head_in_context;

        /// Return the last `n` rows using the borrowed execution override.
        fn tail(n: usize) -> Result<DataFrame> => tail_in_context;

        /// Return `length` rows starting at `offset` using the borrowed execution override.
        fn slice(offset: usize, length: usize) -> Result<DataFrame> => slice_in_context;

        /// Gather rows using the borrowed execution override.
        fn gather(indices: &Series) -> Result<DataFrame> => gather_in_context;

        /// Filter rows using the borrowed execution override.
        fn filter(mask: &Series) -> Result<DataFrame> => filter_in_context;

        /// Concatenate with another frame using the borrowed execution override.
        fn concat(other: &DataFrame) -> Result<DataFrame> => concat_in_context;

        /// Concatenate with multiple frames using the borrowed execution override.
        fn concat_many(others: &[&DataFrame]) -> Result<DataFrame> => concat_many_in_context;

        /// Merge with another frame using the borrowed execution override.
        fn merge(right: &DataFrame, options: MergeOptions) -> Result<DataFrame> => merge_in_context;

        /// Merge-split with another frame using the borrowed execution override.
        fn merge_split(right: &DataFrame, options: MergeOptions) -> Result<DataFrameJoinResult> => merge_split_in_context;

        /// Drop rows containing null values using the borrowed execution override.
        fn dropna(opts: DropNaOptions) -> Result<DataFrame> => dropna_in_context;

        /// Remove duplicate rows using the borrowed execution override.
        fn drop_duplicates(opts: DropDuplicatesOptions) -> Result<DataFrame> => drop_duplicates_in_context;
    );
}
