use super::{
    DataFrame, DataFrameCrossJoinOptions, DataFrameFilteredJoin, DataFrameFilteredJoinOptions, DataFrameHashJoin,
    DataFrameHashJoinOptions, DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameJoinSides,
    DataFrameMixedJoinOptions, DataFrameMixedJoinSizeData, DataFramePredicateJoinOptions, DropDuplicatesOptions,
    DropNaOptions, WithColumnExpr,
};
use crate::{
    Expr, JoinType, Result,
    frame::series::Series,
    ops::{
        ast::AstExpression,
        join::{MergeOptions, SizedConditionalPairJoinType},
        sorting::SortOptions,
    },
    runtime::ExecutionOverrides,
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
    ctx: &'ctx ExecutionOverrides<'handles>,
}

impl DataFrame {
    /// Borrow a per-operation execution override for subsequent high-level calls.
    pub fn with_execution<'df, 'ctx, 'handles>(
        &'df self,
        ctx: &'ctx ExecutionOverrides<'handles>,
    ) -> DataFrameExecutionScope<'df, 'ctx, 'handles> {
        DataFrameExecutionScope { df: self, ctx }
    }
}

impl<'df, 'ctx, 'handles> DataFrameExecutionScope<'df, 'ctx, 'handles> {
    /// Evaluate expressions and add or replace columns using the borrowed execution override.
    ///
    /// Expression intermediates and outputs use the resolved stream and memory
    /// resource. Unchanged columns are shared when the resolved domain matches
    /// the source frame, and device-copied when the override changes retained
    /// provenance so the resulting frame still has one provenance bundle.
    pub fn with_columns<I, T>(&self, exprs: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        self.df.with_columns_in_context(exprs, self.ctx, "dataframe::with_execution::with_columns")
    }

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

    /// Materialize selected columns by gathering rows with join indices using
    /// the borrowed execution override.
    pub fn materialize_join_indices<I, S>(&self, indices: &DataFrameJoinIndices, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.materialize_join_indices_in_context(
            indices,
            columns,
            self.ctx,
            "dataframe::with_execution::materialize_join_indices",
        )
    }

    /// Return exact reservation bytes for materializing selected columns by
    /// gathering rows with join indices using the borrowed execution override.
    ///
    /// For non-empty gather maps, this measures by executing the gather with
    /// the borrowed override, reading the output allocation footprint, and
    /// discarding the temporary result. Its cost is comparable to
    /// `materialize_join_indices` itself; reserve-then-materialize roughly
    /// doubles the gather work.
    pub fn materialize_join_indices_reservation_size_bytes<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.materialize_join_indices_reservation_size_bytes_in_context(
            indices,
            columns,
            self.ctx,
            "dataframe::with_execution::materialize_join_indices_reservation_size_bytes",
        )
    }

    /// Build a reusable hash-join artifact using the borrowed execution override.
    pub fn build_hash_join<I, S>(&self, build_on: I, options: DataFrameHashJoinOptions) -> Result<DataFrameHashJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.build_hash_join_in_context(build_on, options, self.ctx)
    }

    /// Return exact pre-build reservation bytes for a reusable hash-join
    /// artifact using the borrowed execution override.
    pub fn hash_join_pre_build_reservation_size_bytes<I, S>(
        &self,
        build_on: I,
        options: DataFrameHashJoinOptions,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.hash_join_pre_build_reservation_size_bytes_in_context(build_on, options, self.ctx)
    }

    /// Build a reusable filtered-join artifact using the borrowed execution override.
    pub fn build_filtered_join<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
    ) -> Result<DataFrameFilteredJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.build_filtered_join_in_context(build_on, options, self.ctx)
    }

    /// Return exact pre-build reservation bytes for a reusable filtered-join
    /// artifact using the borrowed execution override.
    pub fn filtered_join_pre_build_reservation_size_bytes<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.filtered_join_pre_build_reservation_size_bytes_in_context(build_on, options, self.ctx)
    }

    /// Produce conditional inner join row-index pairs using the borrowed execution override.
    pub fn conditional_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Inner, self.ctx)
    }

    /// Return the exact conditional inner join row count using the borrowed execution override.
    pub fn conditional_inner_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        self.df.conditional_join_size_in_context(
            right,
            predicate,
            options,
            SizedConditionalPairJoinType::Inner,
            self.ctx,
        )
    }

    /// Produce conditional inner join row-index pairs using an exact caller-supplied output row count.
    pub fn conditional_inner_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            SizedConditionalPairJoinType::Inner,
            output_size,
            self.ctx,
        )
    }

    /// Produce conditional left join row-index pairs using the borrowed execution override.
    pub fn conditional_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Left, self.ctx)
    }

    /// Return the exact conditional left join row count using the borrowed execution override.
    pub fn conditional_left_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        self.df.conditional_join_size_in_context(
            right,
            predicate,
            options,
            SizedConditionalPairJoinType::Left,
            self.ctx,
        )
    }

    /// Produce conditional left join row-index pairs using an exact caller-supplied output row count.
    pub fn conditional_left_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            SizedConditionalPairJoinType::Left,
            output_size,
            self.ctx,
        )
    }

    /// Produce conditional full join row-index pairs using the borrowed execution override.
    pub fn conditional_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Full, self.ctx)
    }

    /// Produce conditional left semi join indices using the borrowed execution override.
    pub fn conditional_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_in_context(right, predicate, options, true, self.ctx)
    }

    /// Return the exact conditional left semi join row count using the borrowed execution override.
    pub fn conditional_left_semi_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        self.df.conditional_single_side_join_size_in_context(right, predicate, options, true, self.ctx)
    }

    /// Produce conditional left semi join indices using an exact caller-supplied output row count.
    pub fn conditional_left_semi_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            true,
            output_size,
            self.ctx,
        )
    }

    /// Produce conditional left anti join indices using the borrowed execution override.
    pub fn conditional_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_in_context(right, predicate, options, false, self.ctx)
    }

    /// Return the exact conditional left anti join row count using the borrowed execution override.
    pub fn conditional_left_anti_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        self.df.conditional_single_side_join_size_in_context(right, predicate, options, false, self.ctx)
    }

    /// Produce conditional left anti join indices using an exact caller-supplied output row count.
    pub fn conditional_left_anti_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            false,
            output_size,
            self.ctx,
        )
    }

    /// Produce mixed inner join row-index pairs using the borrowed execution override.
    pub fn mixed_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Inner, self.ctx)
    }

    /// Produce mixed left join row-index pairs using the borrowed execution override.
    pub fn mixed_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Left, self.ctx)
    }

    /// Begin retained size-data computation for a mixed inner join using the borrowed execution override.
    pub fn mixed_inner_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        self.df.mixed_join_size_data_in_context(
            right,
            predicate,
            options,
            crate::ops::join::SizedMixedJoinType::Inner,
            self.ctx,
        )
    }

    /// Begin retained size-data computation for a mixed left join using the borrowed execution override.
    pub fn mixed_left_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        self.df.mixed_join_size_data_in_context(
            right,
            predicate,
            options,
            crate::ops::join::SizedMixedJoinType::Left,
            self.ctx,
        )
    }

    /// Produce mixed full join row-index pairs using the borrowed execution override.
    pub fn mixed_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Full, self.ctx)
    }

    /// Produce mixed left semi join indices using the borrowed execution override.
    pub fn mixed_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.mixed_single_side_join_indices_in_context(right, predicate, options, true, self.ctx)
    }

    /// Produce mixed left anti join indices using the borrowed execution override.
    pub fn mixed_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.mixed_single_side_join_indices_in_context(right, predicate, options, false, self.ctx)
    }

    /// Filter existing join-index pairs with a predicate using the borrowed execution override.
    pub fn filter_join_indices(
        &self,
        right: &DataFrame,
        left_indices: &DataFrameJoinIndices,
        right_indices: &DataFrameJoinIndices,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: JoinType,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.filter_join_indices_in_context(
            right,
            left_indices,
            right_indices,
            predicate,
            options,
            join_type,
            self.ctx,
        )
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
        fn merge_split(right: &DataFrame, options: MergeOptions) -> Result<DataFrameJoinSides> => merge_split_in_context;

        /// Produce inner join row-index pairs using the borrowed execution override.
        fn inner_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => inner_join_indices_in_context;

        /// Produce left join row-index pairs using the borrowed execution override.
        fn left_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => left_join_indices_in_context;

        /// Produce full join row-index pairs using the borrowed execution override.
        fn full_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => full_join_indices_in_context;

        /// Cross join and materialize selected columns using the borrowed execution override.
        fn cross_join_projected(right: &DataFrame, options: DataFrameCrossJoinOptions) -> Result<DataFrame> => cross_join_projected_in_context;

        /// Produce left semi join indices using the borrowed execution override.
        fn left_semi_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinIndices> => left_semi_join_indices_in_context;

        /// Produce left anti join indices using the borrowed execution override.
        fn left_anti_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinIndices> => left_anti_join_indices_in_context;

        /// Left semi join using the borrowed execution override.
        fn left_semi_join(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrame> => left_semi_join_in_context;

        /// Left anti join using the borrowed execution override.
        fn left_anti_join(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrame> => left_anti_join_in_context;

        /// Drop rows containing null values using the borrowed execution override.
        fn dropna(opts: DropNaOptions) -> Result<DataFrame> => dropna_in_context;

        /// Remove duplicate rows using the borrowed execution override.
        fn drop_duplicates(opts: DropDuplicatesOptions) -> Result<DataFrame> => drop_duplicates_in_context;
    );
}
