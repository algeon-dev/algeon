use std::sync::Arc;

use super::{
    ConcatByNameOptions, DataFrame, DataFrameConditionalFullJoinSizeData, DataFrameCrossJoinOptions,
    DataFrameFilteredJoin, DataFrameFilteredJoinOptions, DataFrameHashJoin, DataFrameHashJoinOptions,
    DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameJoinSides,
    DataFrameMixedJoinOptions, DataFrameMixedJoinSizeData, DataFrameMixedSingleSideJoinSizeData,
    DataFramePredicateJoinOptions, DataFramePreparedSort, DropDuplicatesOptions, DropNaOptions,
    HashPartitionedDataFrame, WithColumnExpr,
};
use crate::{
    CudfType, DataType, Expr, JoinType, Result,
    core::{
        column::{Column, position_sequence_in_domain},
        table::Table,
    },
    frame::{
        factorization::{Codebook, Factorized},
        series::Series,
    },
    ops::{
        ast::AstExpression,
        join::{MergeOptions, SizedConditionalPairJoinType, SizedMixedJoinType},
        sorting::SortOptions,
    },
    runtime::Exec,
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

/// The GPU operations of one `DataFrame`, run where an [`Exec`] selects.
#[must_use]
pub struct FrameOps<'a> {
    pub(super) df: &'a DataFrame,
    pub(super) ctx: Exec<'a>,
}

impl DataFrame {
    /// The GPU operations of this frame, run where `exec` selects.
    pub fn on<'a>(&'a self, exec: Exec<'a>) -> FrameOps<'a> {
        FrameOps { df: self, ctx: exec }
    }
}

impl FrameOps<'_> {
    /// Concatenate `other` row-wise after aligning columns by name.
    pub fn concat_by_name(&self, other: &DataFrame, options: ConcatByNameOptions) -> Result<DataFrame> {
        self.concat_many_by_name(&[other], options)
    }

    /// Concatenate `others` row-wise after aligning columns by name;
    /// [`ConcatByNameOptions`] selects how columns missing from a frame are
    /// handled.
    pub fn concat_many_by_name(&self, others: &[&DataFrame], options: ConcatByNameOptions) -> Result<DataFrame> {
        self.df.concat_many_by_name_in_context(others, options, self.ctx)
    }

    /// Build a reusable codebook from a named column's distinct values.
    ///
    /// The returned [`Codebook`] owns both the distinct key series and an
    /// explicit positional code series, so it can be reused safely across
    /// later `encode` / `decode` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn codebook(&self, column: &str) -> Result<Codebook> {
        let operation = "dataframe::codebook";
        let series = self.df.series(column).map_err(|e| e.reanchor(operation))?;
        series.on(self.ctx).codebook().map_err(|e| e.reanchor(operation).with_column(column.to_string()))
    }

    /// Factorize a named column into row-aligned encoded indices plus a reusable codebook.
    ///
    /// This is equivalent to `pandas.factorize()` but returns a typed
    /// [`Factorized`] result instead of a raw `(codes, uniques)` tuple.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn factorize(&self, column: &str) -> Result<Factorized> {
        let operation = "dataframe::factorize";
        let series = self.df.series(column).map_err(|e| e.reanchor(operation))?;
        series.on(self.ctx).factorize().map_err(|e| e.reanchor(operation).with_column(column.to_string()))
    }

    /// Build an owned [`Table`] by deep-copying each column.
    ///
    /// This is useful when calling APIs that operate on `Table` directly.
    ///
    /// # Cost
    ///
    /// GPU-allocating: deep-copies every column to a new `Table`. Each column
    /// gets its own fresh device allocation.
    pub fn to_table_copy(&self) -> Result<Table> {
        let operation = "dataframe::to_table_copy";
        let output_domain = self.ctx.resolve_output_domain(&self.df.domain)?;
        let columns = self
            .df
            .columns
            .iter()
            .map(|(name, column)| {
                column.copy_into_domain(Arc::clone(&output_domain), operation).map_err(|e| e.with_column(name.clone()))
            })
            .collect::<Result<Vec<Column>>>()?;
        if columns.is_empty() {
            Table::from_columns_in_domain_with_num_rows(columns, self.df.num_rows(), output_domain, operation)
        } else {
            Table::from_columns(columns).map_err(|e| e.reanchor(operation))
        }
    }

    /// Evaluate expressions against this DataFrame and add/replace the
    /// resulting columns.
    ///
    /// Use this for declarative computed columns. For pre-computed Series,
    /// use [`with_column`](Self::with_column) (single) or
    /// [`DataFrame::assign`] (batch).
    ///
    /// All expressions are evaluated against the **original** state of `self`,
    /// not an incrementally mutated copy. This ensures that expressions can
    /// reference columns that other expressions will overwrite, and the order
    /// of evaluation does not matter.
    ///
    /// Each item must resolve to an output name. You can provide either:
    ///
    /// - a bare expression, which uses [`Expr::output_name()`](crate::ops::expr::Expr::output_name)
    /// - an explicit `(name, expr)` pair, which always uses the provided name
    ///
    /// Bare expressions can still get their names via
    /// [`.alias()`](crate::ops::expr::Expr::alias), by being a bare
    /// [`col()`](crate::ops::expr::col) reference, or by being a single-input
    /// transform that inherits its input name (for example `cast`, `is_null`,
    /// `fillna`, cumulative scans, and string unary operations). Broadcast
    /// reductions do **not** inherit an output name; give them an alias or an
    /// explicit `(name, expr)` pair. Items without a resolvable name produce
    /// an error **before** any GPU work.
    ///
    /// # Execution and Cost
    ///
    /// Expression intermediates and outputs use the selected stream and
    /// memory resource. Unchanged columns are shared with `Arc::clone` when
    /// the output provenance matches the frame's, as with [`Exec::RETAINED`],
    /// and device-copied otherwise so the result keeps one provenance.
    ///
    /// # Errors
    ///
    /// Returns an error if any item lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I, T>(&self, exprs: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        self.df.with_columns_in_context(exprs, self.ctx, "dataframe::with_columns")
    }

    /// Add or replace one precomputed column.
    ///
    /// `series` must already have the output provenance, for example the
    /// output of another operation on the same `Exec`. Unchanged columns of
    /// another provenance are copied into it. [`DataFrame::with_column`]
    /// adds a column of the frame's own provenance without GPU work.
    pub fn with_column(&self, name: &str, series: Series) -> Result<DataFrame> {
        let operation = "dataframe::with_column";
        self.ctx
            .resolve_output_domain(self.df.execution_domain())
            .and_then(|output_domain| self.df.with_column_in_domain(name, series, output_domain, operation))
            .map_err(|error| error.reanchor_if_needed(operation))
    }

    /// Return a new DataFrame with the named column cast to a different type.
    ///
    /// All other columns are preserved unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the cast
    /// is unsupported.
    pub fn cast_column(&self, column: &str, target_type: impl Into<DataType>) -> Result<DataFrame> {
        self.df.cast_column_in_context(column, target_type.into(), self.ctx, "dataframe::cast_column")
    }

    /// Add a zero-based Int32 row-position column.
    pub fn with_position_column(&self, output_name: &str) -> Result<DataFrame> {
        let operation = "dataframe::with_position_column";
        let output_domain = self.ctx.resolve_output_domain(self.df.execution_domain())?;
        let row_ids = Series::from_column(position_sequence_in_domain(self.df.num_rows(), &output_domain, operation)?)
            .map_err(|e| e.reanchor(operation).with_column(output_name.to_string()))?;
        self.df.with_column_in_domain(output_name, row_ids, output_domain, operation).map_err(|e| e.reanchor(operation))
    }

    /// Add a fixed-width arithmetic sequence column.
    pub fn with_sequence_column<T: CudfType>(&self, output_name: &str, start: T, step: T) -> Result<DataFrame> {
        let operation = "dataframe::with_sequence_column";
        let output_domain = self.ctx.resolve_output_domain(self.df.execution_domain())?;
        let values = Series::from_column(Column::from_sequence_in_domain(
            start,
            step,
            self.df.num_rows(),
            Arc::clone(&output_domain),
            operation,
        )?)
        .map_err(|error| error.reanchor(operation).with_column(output_name.to_string()))?;
        self.df
            .with_column_in_domain(output_name, values, output_domain, operation)
            .map_err(|error| error.reanchor(operation))
    }

    /// Add an encoded integer-code column using a reusable codebook.
    pub fn with_encoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<DataFrame> {
        let encoded = codebook
            .encode(&self.df.series(input_column).map_err(|e| e.reanchor("dataframe::with_encoded_column"))?, self.ctx)
            .map_err(|e| e.reanchor("dataframe::with_encoded_column").with_column(input_column.to_string()))?;
        let output_domain = Arc::clone(encoded.as_series().as_column().execution_domain());
        self.df
            .with_column_in_domain(output_name, encoded, output_domain, "dataframe::with_encoded_column")
            .map_err(|e| e.reanchor("dataframe::with_encoded_column"))
    }

    /// Add a decoded value column from integer codes.
    pub fn with_decoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<DataFrame> {
        let decoded = codebook
            .decode_series(
                &self.df.series(input_column).map_err(|e| e.reanchor("dataframe::with_decoded_column"))?,
                self.ctx,
            )
            .map_err(|e| e.reanchor("dataframe::with_decoded_column").with_column(input_column.to_string()))?;
        let output_domain = Arc::clone(decoded.as_column().execution_domain());
        self.df
            .with_column_in_domain(output_name, decoded, output_domain, "dataframe::with_decoded_column")
            .map_err(|e| e.reanchor("dataframe::with_decoded_column"))
    }

    /// Sort.
    pub fn sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.sort_values_in_context(by, options, self.ctx)
    }

    /// Prepare a retained sorted-order gather map for a later DataFrame sort.
    ///
    /// This resolves and validates keys/options with the same user-facing
    /// rules as [`sort_values`](Self::sort_values), then retains only the INT32
    /// sorted-order row-index column. Use
    /// [`DataFramePreparedSort::materialized_output_reservation_size_bytes`]
    /// to reserve the exact gathered output bytes before calling
    /// [`DataFramePreparedSort::materialize`].
    ///
    /// This contract covers the retained sorted-order bytes and the
    /// materialized gather output sizing, and retains the unresolved
    /// libcudf sort temporary-workspace contract.
    pub fn prepare_sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<DataFramePreparedSort>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.prepare_sort_values_in_context(by, options, self.ctx)
    }

    /// Hash-partition rows.
    pub fn hash_partition_by<I, S>(&self, by: I, num_partitions: usize) -> Result<HashPartitionedDataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.hash_partition_by_in_context(by, num_partitions, self.ctx)
    }

    /// Filter rows using a validated fused predicate expression.
    ///
    /// This is the preferred high-level API for multi-condition row
    /// filtering. It avoids chained intermediate `DataFrame`
    /// materializations by lowering a supported predicate subset to a
    /// dedicated row-predicate execution path and applying one final table
    /// filter.
    ///
    /// V1 supported predicate syntax is intentionally narrower than the full
    /// [`Expr`] DSL:
    ///
    /// - column references and literals
    /// - comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
    /// - null checks: `is_null`, `is_not_null`
    /// - boolean composition: `!`, `&`, `|`
    /// - alias wrappers around a supported inner predicate
    ///
    /// Unsupported predicate nodes are evaluated through the general
    /// expression engine and applied as a boolean mask until the fused
    /// row-predicate path supports those expressions directly.
    ///
    /// The predicate must evaluate to a row-aligned `Bool8`. Rows where the
    /// predicate is `false` or null are dropped.
    pub fn filter_expr(&self, predicate: Expr) -> Result<DataFrame> {
        self.df.filter_expr_in_context(predicate, self.ctx)
    }

    /// Filter rows where all predicates are true.
    ///
    /// This is the preferred ergonomic wrapper for common multi-condition
    /// conjunctions. It is equivalent to combining the predicates with `&`
    /// and passing the result to [`FrameOps::filter_expr`](crate::FrameOps::filter_expr).
    ///
    /// Empty input is the logical identity and returns `self.clone()`.
    pub fn filter_all<I>(&self, predicates: I) -> Result<DataFrame>
    where I: IntoIterator<Item = Expr> {
        self.df.filter_all_in_context(predicates, self.ctx)
    }

    /// Filter rows where any predicate is true.
    ///
    /// This is the preferred ergonomic wrapper for common multi-condition
    /// disjunctions. It is equivalent to combining the predicates with `|`
    /// and passing the result to [`FrameOps::filter_expr`](crate::FrameOps::filter_expr).
    ///
    /// Empty input returns an empty DataFrame with the same schema.
    pub fn filter_any<I>(&self, predicates: I) -> Result<DataFrame>
    where I: IntoIterator<Item = Expr> {
        self.df.filter_any_in_context(predicates, self.ctx)
    }

    /// Materialize selected columns by gathering rows with join indices.
    ///
    /// This performs a GPU gather of the selected columns and therefore copies
    /// device buffers for those output columns. It is not a zero-copy schema
    /// projection. The row count of the output is the length of `indices`.
    pub fn materialize_join_indices<I, S>(&self, indices: &DataFrameJoinIndices, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.materialize_join_indices_in_context(indices, columns, self.ctx, "dataframe::materialize_join_indices")
    }

    /// Return the exact retained device allocation footprint of
    /// [`materialize_join_indices`](Self::materialize_join_indices) for the
    /// selected payload columns.
    ///
    /// This is a lower-level reservation contract for bounded engines. It is
    /// implemented in the cuDF layer by materializing non-empty projected
    /// gather results, observing the table allocation footprint, and releasing
    /// the temporary result before returning. Empty gather maps return zero
    /// without executing the gather. Callers should reserve the returned byte
    /// count before performing the durable materialization.
    ///
    /// Cost note: unlike
    /// [`hash_join_pre_build_reservation_size_bytes`](Self::hash_join_pre_build_reservation_size_bytes),
    /// this contract executes the non-empty gather end-to-end and discards the
    /// result. Its cost is comparable to
    /// [`materialize_join_indices`](Self::materialize_join_indices) itself, so
    /// reserve-then-materialize roughly doubles the gather work.
    ///
    /// Doc-only deprecation: this helper is retained for legacy callers that
    /// need an exact post-gather allocation measurement. It is an executable
    /// preflight, not a closed-form bound, and can allocate the full gather
    /// working set before the caller has reserved the returned bytes. New
    /// bounded execution paths should prefer an explicit closed-form
    /// reservation contract and then reconcile the observed retained size after
    /// materialization.
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
            "dataframe::materialize_join_indices_reservation_size_bytes",
        )
    }

    /// Build a reusable hash-join artifact.
    pub fn build_hash_join<I, S>(&self, build_on: I, options: DataFrameHashJoinOptions) -> Result<DataFrameHashJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.df.build_hash_join_in_context(build_on, options, self.ctx)
    }

    /// Return exact device bytes that must be reserved before building a
    /// reusable hash-join artifact from this DataFrame's named key columns.
    ///
    /// This is a planning contract for bounded engines. It returns the exact
    /// retained bytes for the reusable hash table and build-key device views,
    /// without building or allocating retained payload. libcudf includes retained
    /// null-payload sanitization for flat nullable string keys; nested build
    /// keys remain unsupported.
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

    /// Build a reusable filtered-join artifact.
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
    /// artifact.
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

    /// Produce conditional inner join row-index pairs.
    pub fn conditional_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Inner, self.ctx)
    }

    /// Return the exact conditional inner join row count.
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

    /// Produce conditional left join row-index pairs.
    pub fn conditional_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Left, self.ctx)
    }

    /// Return the exact conditional left join row count.
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

    /// Produce conditional full join row-index pairs.
    pub fn conditional_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.conditional_join_indices_in_context(right, predicate, options, JoinType::Full, self.ctx)
    }

    /// Begin retained size-data computation for a conditional full join.
    pub fn conditional_full_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameConditionalFullJoinSizeData<'a>> {
        self.df.conditional_full_join_size_data_in_context(right, predicate, options, self.ctx)
    }

    /// Produce conditional left semi join indices.
    pub fn conditional_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_in_context(right, predicate, options, true, self.ctx)
    }

    /// Return the exact conditional left semi join row count.
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

    /// Produce conditional left anti join indices.
    pub fn conditional_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.conditional_single_side_join_indices_in_context(right, predicate, options, false, self.ctx)
    }

    /// Return the exact conditional left anti join row count.
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

    /// Produce mixed inner join row-index pairs.
    pub fn mixed_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Inner, self.ctx)
    }

    /// Produce mixed left join row-index pairs.
    pub fn mixed_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Left, self.ctx)
    }

    /// Begin retained size-data computation for a mixed inner join.
    pub fn mixed_inner_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        self.df.mixed_join_size_data_in_context(right, predicate, options, SizedMixedJoinType::Inner, self.ctx)
    }

    /// Begin retained size-data computation for a mixed left join.
    pub fn mixed_left_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        self.df.mixed_join_size_data_in_context(right, predicate, options, SizedMixedJoinType::Left, self.ctx)
    }

    /// Begin retained size-data computation for a mixed full join.
    pub fn mixed_full_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        self.df.mixed_join_size_data_in_context(right, predicate, options, SizedMixedJoinType::Full, self.ctx)
    }

    /// Produce mixed full join row-index pairs.
    pub fn mixed_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.df.mixed_join_indices_in_context(right, predicate, options, JoinType::Full, self.ctx)
    }

    /// Produce mixed left semi join indices.
    pub fn mixed_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.mixed_single_side_join_indices_in_context(right, predicate, options, true, self.ctx)
    }

    /// Begin retained size-data computation for a mixed left semi join.
    pub fn mixed_left_semi_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        self.df.mixed_single_side_join_size_data_in_context(right, predicate, options, true, self.ctx)
    }

    /// Produce mixed left anti join indices.
    pub fn mixed_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.df.mixed_single_side_join_indices_in_context(right, predicate, options, false, self.ctx)
    }

    /// Begin retained size-data computation for a mixed left anti join.
    pub fn mixed_left_anti_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        self.df.mixed_single_side_join_size_data_in_context(right, predicate, options, false, self.ctx)
    }

    /// Filter existing join-index pairs with a predicate.
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
        /// Return the first `n` rows.
        fn head(n: usize) -> Result<DataFrame> => head_in_context;

        /// Return the last `n` rows.
        fn tail(n: usize) -> Result<DataFrame> => tail_in_context;

        /// Return `length` rows starting at `offset`.
        fn slice(offset: usize, length: usize) -> Result<DataFrame> => slice_in_context;

        /// Gather rows.
        fn gather(indices: &Series) -> Result<DataFrame> => gather_in_context;

        /// Filter rows.
        fn filter(mask: &Series) -> Result<DataFrame> => filter_in_context;

        /// Filter rows to those where `column` values appear in a `Series`.
        fn filter_isin(column: &str, values: &Series) -> Result<DataFrame> => filter_isin_in_context;

        /// Concatenate two DataFrames row-wise.
        ///
        /// Both DataFrames must have the same column names in the same order
        /// and matching column types.
        ///
        /// # Errors
        ///
        /// Returns an error if the schemas do not match (different names, count,
        /// order, or types).
        fn concat(other: &DataFrame) -> Result<DataFrame> => concat_in_context;

        /// Concatenate multiple DataFrames row-wise.
        ///
        /// All DataFrames must have the same column names in the same order
        /// and matching column types. An empty `others` slice returns a clone
        /// of `self`.
        ///
        /// This is more efficient than chaining multiple [`concat`](Self::concat)
        /// calls because it passes all table views to a single FFI call.
        ///
        /// # Errors
        ///
        /// Returns an error if any DataFrame's schema does not match `self`
        /// (different names, count, order, or types).
        fn concat_many(others: &[&DataFrame]) -> Result<DataFrame> => concat_many_in_context;

        /// Merge (join) two DataFrames with cuDF-style options.
        ///
        /// This is the unified high-level entry point for joins. Configure the merge
        /// via [`MergeOptions`](crate::MergeOptions).
        ///
        /// # Key modes
        ///
        /// - `on(keys)`: shared key column names present in both sides. Output
        ///   contains one shared key column per key.
        /// - `left_on(keys)` + `right_on(keys)`: asymmetric key names. Both key
        ///   columns are preserved in the output.
        ///
        /// # Suffix handling
        ///
        /// When `on()` is used: if a left payload and right payload column share
        /// the same name, both are suffixed with the configured suffixes (default
        /// `"_left"` / `"_right"`).
        ///
        /// When `left_on()` / `right_on()` is used: both key columns are preserved,
        /// and conflicting payload names get the same suffix treatment.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - No keys are specified (`on` or `left_on`+`right_on`)
        /// - `left_on` is set without `right_on` or vice versa
        /// - `left_on` and `right_on` have different lengths
        /// - A key column is not found in its respective DataFrame
        fn merge(right: &DataFrame, options: MergeOptions) -> Result<DataFrame> => merge_in_context;

        /// Merge (join) two DataFrames, returning split left/right results.
        ///
        /// Like [`merge`](Self::merge) but returns a [`DataFrameJoinSides`] with
        /// separate left and right `DataFrame`s instead of merging columns into one.
        /// Each side contains `[key_columns..., payload_columns...]`.
        ///
        /// # Errors
        ///
        /// Same error conditions as [`merge`](Self::merge).
        fn merge_split(right: &DataFrame, options: MergeOptions) -> Result<DataFrameJoinSides> => merge_split_in_context;

        /// Return device-side gather index pairs for an inner equality join.
        ///
        /// This is an index-producing query-planning helper. It allocates only the
        /// left and right `INT32` device gather maps and does not copy payload
        /// columns. Duplicate key matches produce one row pair per match.
        fn inner_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => inner_join_indices_in_context;

        /// Return device-side gather index pairs for a left equality join.
        ///
        /// Misses on the right side contain libcudf's no-match sentinel. Use
        /// [`materialize_join_indices`](Self::materialize_join_indices) to gather
        /// selected right-side columns with nullification for those misses.
        fn left_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => left_join_indices_in_context;

        /// Return device-side gather index pairs for a full equality join.
        ///
        /// Misses on either side contain libcudf's no-match sentinel. Use
        /// [`materialize_join_indices`](Self::materialize_join_indices) to gather
        /// selected columns with nullification for unmatched rows.
        fn full_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinPairIndices> => full_join_indices_in_context;

        /// Cross join two DataFrames and materialize only the requested output columns.
        ///
        /// Source columns are selected before calling libcudf. Output names are
        /// taken from [`DataFrameCrossJoinOptions`], allowing same-named source
        /// columns from each side to be projected with distinct result names.
        fn cross_join_projected(right: &DataFrame, options: DataFrameCrossJoinOptions) -> Result<DataFrame> => cross_join_projected_in_context;

        /// Return device-side gather indices for the left rows that have a match in `right`.
        ///
        /// This is an index-producing query-planning helper. It allocates only an
        /// `INT32` device gather map and does not copy payload columns. Duplicate
        /// matches on the right side do not duplicate left rows.
        fn left_semi_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinIndices> => left_semi_join_indices_in_context;

        /// Return device-side gather indices for the left rows that have no match in `right`.
        ///
        /// This is an index-producing query-planning helper. It allocates only an
        /// `INT32` device gather map and does not copy payload columns.
        fn left_anti_join_indices(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrameJoinIndices> => left_anti_join_indices_in_context;

        /// Left semi join, returning only left-side columns.
        ///
        /// This helper first computes left-side join indices and then gathers the
        /// left frame's columns. The gather copies selected device rows into new
        /// columns. Right-side payload columns are never included, and duplicate
        /// right matches do not duplicate left rows.
        fn left_semi_join(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrame> => left_semi_join_in_context;

        /// Left anti join, returning only left-side columns.
        ///
        /// This helper first computes left-side join indices and then gathers the
        /// left frame's columns. The gather copies selected device rows into new
        /// columns. Right-side payload columns are never included.
        fn left_anti_join(right: &DataFrame, options: DataFrameJoinOptions) -> Result<DataFrame> => left_anti_join_in_context;

        /// Drop rows containing null values.
        ///
        /// Equivalent to cuDF Python's `DataFrame.dropna()`.
        fn dropna(opts: DropNaOptions) -> Result<DataFrame> => dropna_in_context;

        /// Remove duplicate rows.
        ///
        /// Equivalent to cuDF Python's `DataFrame.drop_duplicates()`.
        fn drop_duplicates(opts: DropDuplicatesOptions) -> Result<DataFrame> => drop_duplicates_in_context;

        /// Remove consecutive duplicate rows.
        ///
        /// Rows that compare equal on the selected columns must already be
        /// adjacent. Use [`drop_duplicates`](Self::drop_duplicates) otherwise.
        fn drop_consecutive_duplicates(opts: DropDuplicatesOptions) -> Result<DataFrame> => drop_consecutive_duplicates_in_context;
    );
}
