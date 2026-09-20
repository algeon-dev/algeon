use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::{
        error::{Error, ErrorCode, ErrorKind, FACT_ACTUAL_ROWS, FACT_EXPECTED_ROWS, Result},
        scalar::Scalar,
        types::DataType,
    },
    frame::series::{IntoSeries, Series},
    hash::{FastIndexMap, fast_index_map_with_capacity},
    internal::resolve::missing_column_error,
    ops::{self, replace::replace_nulls_with_exec_options, unary},
    runtime::{
        BudgetedExecutionOutcome, ExecutionBudgetToken, ExecutionOverrides, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution,
    },
};

/// One item accepted by [`DataFrame::with_columns`](super::DataFrame::with_columns).
///
/// This models the two supported ways to provide computed columns:
///
/// - a bare expression, which must resolve its own output name
/// - an explicit `(name, expr)` pair, where the provided name wins
pub enum WithColumnExpr {
    /// Use the expression's own output-name rules.
    Expr(ops::expr::Expr),
    /// Force the output name explicitly.
    Named { name: String, expr: ops::expr::Expr },
}

impl WithColumnExpr {
    fn into_parts(self) -> (Option<String>, ops::expr::Expr) {
        match self {
            Self::Expr(expr) => (None, expr),
            Self::Named { name, expr } => (Some(name), expr),
        }
    }
}

impl From<ops::expr::Expr> for WithColumnExpr {
    fn from(expr: ops::expr::Expr) -> Self {
        Self::Expr(expr)
    }
}

impl<S> From<(S, ops::expr::Expr)> for WithColumnExpr
where S: Into<String>
{
    fn from((name, expr): (S, ops::expr::Expr)) -> Self {
        Self::Named { name: name.into(), expr }
    }
}

fn resolve_with_column_exprs<I, T>(exprs: I, operation: &'static str) -> Result<Vec<(String, ops::expr::Expr)>>
where
    I: IntoIterator<Item = T>,
    T: Into<WithColumnExpr>,
{
    exprs
        .into_iter()
        .map(|expr| {
            let (explicit_name, expr) = expr.into().into_parts();
            let name = explicit_name.as_deref().or_else(|| expr.output_name()).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    operation,
                    "expression has no output name; use .alias() to name it",
                )
                .with_detail("resource", "output_name")
                .with_detail("reason", "missing_output_name")
                .with_detail("remedy", "use .alias(...) or provide an explicit (name, expr) pair")
            })?;
            Ok((name.to_string(), expr))
        })
        .collect()
}

impl DataFrame {
    /// Return a new DataFrame with a single column added or replaced.
    ///
    /// The `name` parameter controls the column name in the resulting
    /// DataFrame. `DataFrame` is the schema authority for column naming.
    ///
    /// Use [`assign`](Self::assign) for batch addition of multiple columns,
    /// or [`with_columns`](Self::with_columns) for expression-based computed
    /// columns.
    ///
    /// # Cost
    /// Unchanged columns are shared via `Arc::clone`; no GPU copy.
    ///
    /// # Errors
    /// Returns an error if the new column's row count does not match the
    /// DataFrame's existing row count.
    pub fn with_column(&self, name: &str, col: impl IntoSeries) -> Result<Self> {
        let series: Series = col.into_series()?;
        self.validate_series_insertion_domain(&series, "dataframe::with_column", name)?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows
            && col_rows != self_rows
        {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                "dataframe::with_column",
                format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
            )
            .with_column(name.to_string())
            .with_fact(FACT_ACTUAL_ROWS, col_rows)
            .with_fact(FACT_EXPECTED_ROWS, self_rows));
        }

        let mut new_columns = self.columns.clone();
        new_columns.insert(name.to_string(), Arc::clone(series.column_arc()));

        let row_count = expected_rows.unwrap_or(col_rows);
        let domain = Arc::clone(&self.domain);
        Ok(Self { columns: new_columns, row_count, domain })
    }

    pub(crate) fn with_column_in_domain(
        &self,
        name: &str,
        col: impl IntoSeries,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let series: Series = col.into_series()?;
        self.assemble_with_columns_in_domain(vec![name.to_string()], vec![series], output_domain, operation)
    }

    /// Return a new DataFrame with multiple columns added or replaced from
    /// `(name, Series)` pairs.
    ///
    /// The pair key controls the column name in the resulting DataFrame.
    /// `DataFrame` is the schema authority for column naming.
    ///
    /// For each pair, if `name` already exists the column is replaced;
    /// otherwise it is appended. All unchanged columns are shared via
    /// `Arc::clone` without copying GPU data.
    ///
    /// # Duplicate name handling
    ///
    /// If the same output name appears more than once in `pairs`, the **last
    /// value wins**; earlier entries for that name are silently overwritten.
    /// This matches the semantics of Python's `DataFrame.assign(**kwargs)` and
    /// is intentional: `assign` is a batch "set these columns" operation, not a
    /// schema-definition operation. Methods that define schema shape
    /// ([`select`](Self::select), [`rename_columns`](Self::rename_columns))
    /// reject duplicates with an error.
    ///
    /// If `self` has zero columns but a non-zero logical row count (for
    /// example after [`without_columns`](Self::without_columns) or [`select`](Self::select) removes
    /// every column), inserted Series must still match that row count. Only a
    /// truly empty `0 x 0` DataFrame lets the first inserted Series establish
    /// the row count.
    ///
    /// # Errors
    ///
    /// Returns an error if any Series has a different row count than the
    /// DataFrame.
    pub fn assign<I, S>(&self, pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: AsRef<str>,
    {
        let mut new_columns = self.columns.clone();
        let mut expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        for (name, series) in pairs {
            let name = name.as_ref();
            let series_rows = series.len();

            if let Some(self_rows) = expected_rows {
                if series_rows != self_rows {
                    return Err(Error::new_with(
                        ErrorCode::ColumnLengthMismatch,
                        "dataframe::assign",
                        format!("series '{}' has {} rows but DataFrame has {} rows", name, series_rows, self_rows),
                    )
                    .with_column(name.to_string())
                    .with_fact(FACT_ACTUAL_ROWS, series_rows)
                    .with_fact(FACT_EXPECTED_ROWS, self_rows));
                }
            } else {
                expected_rows = Some(series_rows);
            }

            self.validate_series_insertion_domain(&series, "dataframe::assign", name)?;
            let column = Arc::clone(series.column_arc());
            new_columns.insert(name.to_string(), column);
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = Arc::clone(&self.domain);
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Evaluate expressions against this DataFrame and add/replace the
    /// resulting columns.
    ///
    /// Use this for declarative computed columns. For pre-computed Series,
    /// use [`with_column`](Self::with_column) (single) or
    /// [`assign`](Self::assign) (batch).
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
    /// This convenience path evaluates through the DataFrame's retained
    /// provenance. Unchanged columns are shared with `Arc::clone` and do not
    /// copy GPU data. Use [`DataFrame::with_execution`](Self::with_execution)
    /// and call `.with_columns(...)` on the returned scope to evaluate
    /// expressions with borrowed stream / memory-resource overrides; that
    /// scoped path may device-copy unchanged columns when the override changes
    /// the result frame's retained provenance.
    ///
    /// # Errors
    ///
    /// Returns an error if any item lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I, T>(&self, exprs: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        self.with_columns_in_context(exprs, &ExecutionOverrides::new(), "dataframe::with_columns")
    }

    /// Evaluate expressions and add/replace columns under a hard execution-budget token.
    ///
    /// This has the same expression semantics and validation as
    /// [`with_columns`](Self::with_columns), but expression evaluation and any
    /// required unchanged-column relocation are routed through the token's
    /// execution overrides. The returned report records the resulting
    /// DataFrame device footprint and row count.
    pub fn with_columns_with_budget<I, T>(
        &self,
        exprs: I,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        let operation = "dataframe::with_columns_with_budget";
        let ctx = token.execution_overrides();
        let exprs = resolve_with_column_exprs(exprs, operation)?;
        let output = self
            .with_resolved_columns_in_context(&exprs, &ctx, operation)
            .map_err(|error| annotate_dataframe_budget_execution_error(error, token, operation))?;
        let output_bytes = output.device_allocated_size_bytes().map_err(|error| error.reanchor_if_needed(operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output.num_rows());

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    /// Add or replace one precomputed column under a hard execution-budget
    /// token.
    ///
    /// The result takes the token's output domain, which `series` must
    /// already carry, for example the output of a budgeted operation under
    /// the same token. As in
    /// [`with_columns_with_budget`](Self::with_columns_with_budget), unchanged
    /// columns of another provenance are copied into that domain through the
    /// token. The report records the resulting DataFrame device footprint and
    /// row count.
    pub fn with_column_with_budget(
        &self,
        name: &str,
        series: Series,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let operation = "dataframe::with_column_with_budget";
        let output = token
            .execution_overrides()
            .resolve_output_domain(&self.domain)
            .and_then(|output_domain| self.with_column_in_domain(name, series, output_domain, operation))
            .map_err(|error| annotate_dataframe_budget_execution_error(error, token, operation))?;
        let output_bytes = output.device_allocated_size_bytes().map_err(|error| error.reanchor_if_needed(operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output.num_rows());

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    pub(crate) fn with_columns_in_context<I, T>(
        &self,
        exprs: I,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        let exprs = resolve_with_column_exprs(exprs, operation)?;
        self.with_resolved_columns_in_context(&exprs, ctx, operation)
    }

    fn with_resolved_columns_in_context(
        &self,
        exprs: &[(String, ops::expr::Expr)],
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let (output_domain, results) =
            ops::expr::eval_many_in_context(self, ctx, operation, exprs.iter().map(|(_, expr)| expr))?;
        self.assemble_with_columns_in_domain(
            exprs.iter().map(|(name, _)| name.clone()).collect(),
            results,
            output_domain,
            operation,
        )
    }

    fn assemble_with_columns_in_domain(
        &self,
        names: Vec<String>,
        results: Vec<Series>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let mut replacements: FastIndexMap<String, Series> = fast_index_map_with_capacity(results.len());
        for (name, series) in names.into_iter().zip(results) {
            let series_rows = series.len();
            if series_rows != self.row_count {
                return Err(Error::new_with(
                    ErrorCode::ColumnLengthMismatch,
                    operation,
                    format!("series '{}' has {} rows but DataFrame has {} rows", name, series_rows, self.row_count),
                )
                .with_column(name)
                .with_fact(FACT_ACTUAL_ROWS, series_rows)
                .with_fact(FACT_EXPECTED_ROWS, self.row_count));
            }
            self.validate_series_against_output_domain(&series, &output_domain, operation, &name)?;
            replacements.insert(name, series);
        }

        let mut new_columns = fast_index_map_with_capacity(self.columns.len() + replacements.len());
        for (name, column) in &self.columns {
            if let Some(series) = replacements.get(name) {
                new_columns.insert(name.clone(), series.arc());
            } else if ExecutionDomain::same_provenance(column.execution_domain(), &output_domain) {
                new_columns.insert(name.clone(), Arc::clone(column));
            } else {
                // copy_into_domain installs both fence directions internally.
                let copied = column
                    .copy_into_domain(Arc::clone(&output_domain), operation)
                    .map_err(|e| e.with_column(name.clone()))?;
                new_columns.insert(name.clone(), Arc::new(copied));
            }
        }

        for (name, series) in replacements {
            if !self.columns.contains_key(&name) {
                new_columns.insert(name, series.arc());
            }
        }

        let domain = output_domain;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    fn validate_series_against_output_domain(
        &self,
        series: &Series,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
        column_name: &str,
    ) -> Result<()> {
        ExecutionDomain::require_same_provenance(
            output_domain,
            series.as_column().execution_domain(),
            operation,
            "with_columns expression output must resolve into the frame output provenance",
            "evaluate expressions with the DataFrame execution context instead of calling default Series operations inside expression evaluation",
        )
        .map_err(|error| error.with_column(column_name.to_string()))
    }

    /// Return a new DataFrame with the named column cast to a different type.
    ///
    /// All other columns are preserved unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the cast
    /// is unsupported.
    pub fn cast_column(&self, column: &str, target_type: impl Into<DataType>) -> Result<Self> {
        self.cast_column_in_context(column, target_type.into(), &ExecutionOverrides::new(), "dataframe::cast_column")
    }

    /// Return a new DataFrame with one column cast under a hard execution-budget token.
    ///
    /// This is the budgeted counterpart to [`cast_column`](Self::cast_column).
    /// The direct cast allocation and any required unchanged-column relocation
    /// into the budgeted output provenance are routed through the token's
    /// stream-bound reservation.
    pub fn cast_column_with_budget(
        &self,
        column: &str,
        target_type: impl Into<DataType>,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Self>> {
        let operation = "dataframe::cast_column_with_budget";
        let context = token.execution_overrides();
        let output = self
            .cast_column_in_context(column, target_type.into(), &context, operation)
            .map_err(|error| annotate_dataframe_budget_execution_error(error, token, operation))?;
        let output_bytes = output.device_allocated_size_bytes().map_err(|error| error.reanchor_if_needed(operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output.num_rows());
        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn cast_column_in_context(
        &self,
        column: &str,
        target_type: DataType,
        context: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let input = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;
        let input_view = input.view().map_err(|error| error.reanchor(operation))?;
        let execution = ResolvedExecution::unary(input.execution_domain(), context, operation)?;
        let output_column = unary::cast_impl(
            &input_view,
            &target_type,
            execution.output_domain(),
            execution.exec_options(),
            operation,
        )?;
        let output_series = Series::from_column(output_column).map_err(|error| error.reanchor(operation))?;
        self.assemble_with_columns_in_domain(
            vec![column.to_string()],
            vec![output_series],
            Arc::clone(execution.output_domain()),
            operation,
        )
    }

    /// Return a new DataFrame with null values in the named column replaced
    /// by a scalar value.
    ///
    /// All other columns are preserved unchanged. The scalar type must be
    /// compatible with the column type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub(crate) fn replace_nulls(&self, column: &str, value: &Scalar) -> Result<Self> {
        let operation = "dataframe::replace_nulls";
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;

        let _consumer_fence =
            ExecutionDomain::insert_stream_dependency(value.execution_domain(), col.execution_domain(), operation)?;
        let cv = col.view()?;
        let exec_options = col.execution_domain().enter_execution()?;
        let new_col =
            replace_nulls_with_exec_options(&cv, value, col.execution_domain(), exec_options.options(), operation)?;

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = Arc::clone(&self.domain);
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }
}

fn annotate_dataframe_budget_execution_error(
    error: Error,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Error {
    match error.kind() {
        ErrorKind::ResourceExhausted | ErrorKind::DependencyFailure => token.annotate_error(error, operation),
        _ => error.reanchor_if_needed(operation),
    }
}
