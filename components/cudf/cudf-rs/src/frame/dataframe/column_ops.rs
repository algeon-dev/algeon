use std::sync::Arc;

use super::owned::{DataFrame, derive_frame_domain_from_columns};
use crate::{
    core::{
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::DataType,
    },
    frame::series::{IntoSeries, Series},
    internal::resolve::missing_column_error,
    ops::{self, unary},
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

impl DataFrame {
    /// Return a new DataFrame with a single column added or replaced.
    ///
    /// The `name` parameter controls the column name in the resulting
    /// DataFrame. The input's own [`Series::name()`] is **ignored** —
    /// DataFrame is the schema authority for column naming.
    ///
    /// Use [`assign`](Self::assign) for batch addition of multiple columns,
    /// or [`with_columns`](Self::with_columns) for expression-based computed
    /// columns.
    ///
    /// # Cost
    /// Unchanged columns are shared via `Arc::clone` — no GPU copy.
    ///
    /// # Errors
    /// Returns an error if the new column's row count does not match the
    /// DataFrame's existing row count.
    pub fn with_column(&self, name: &str, col: impl IntoSeries) -> Result<Self> {
        let series: Series = col.into_series()?;
        self.validate_series_insertion_domain(&series, "dataframe::with_column", name)?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows {
            if col_rows != self_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
                )
                .with_operation("dataframe::with_column")
                .with_context("column_name", name.to_string())
                .with_context("column_rows", col_rows.to_string())
                .with_context("dataframe_rows", self_rows.to_string()));
            }
        }

        let mut new_columns = self.columns.clone();
        new_columns.insert(name.to_string(), Arc::clone(series.column_arc()));

        let row_count = expected_rows.unwrap_or(col_rows);
        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::with_column")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Return a new DataFrame with multiple columns added or replaced from
    /// `(name, Series)` pairs.
    ///
    /// The pair key controls the column name in the resulting DataFrame. The
    /// input Series' own [`Series::name()`] is **ignored** — DataFrame is the
    /// schema authority for column naming.
    ///
    /// For each pair, if `name` already exists the column is replaced;
    /// otherwise it is appended. All unchanged columns are shared via
    /// `Arc::clone` without copying GPU data.
    ///
    /// # Duplicate name handling
    ///
    /// If the same output name appears more than once in `pairs`, the **last
    /// value wins** — earlier entries for that name are silently overwritten.
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
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("series '{}' has {} rows but DataFrame has {} rows", name, series_rows, self_rows),
                    )
                    .with_operation("dataframe::assign")
                    .with_context("column_name", name.to_string())
                    .with_context("series_rows", series_rows.to_string())
                    .with_context("dataframe_rows", self_rows.to_string()));
                }
            } else {
                expected_rows = Some(series_rows);
            }

            self.validate_series_insertion_domain(&series, "dataframe::assign", name)?;
            let column = Arc::clone(series.column_arc());
            new_columns.insert(name.to_string(), column);
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::assign")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Evaluate expressions against this DataFrame and add/replace the
    /// resulting columns.
    ///
    /// Use this for declarative computed columns. For pre-computed Series,
    /// use [`with_column`](Self::with_column) (single) or
    /// [`assign`](Self::assign) (batch).
    ///
    /// All expressions are evaluated against the **original** state of `self`
    /// — not an incrementally mutated copy. This ensures that expressions can
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
    /// # Errors
    ///
    /// Returns an error if any item lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I, T>(&self, exprs: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<WithColumnExpr>,
    {
        let exprs: Vec<(Option<String>, ops::expr::Expr)> =
            exprs.into_iter().map(|expr| expr.into().into_parts()).collect();

        let names: Vec<String> = exprs
            .iter()
            .map(|(explicit_name, expr)| {
                explicit_name.as_deref().or_else(|| expr.output_name()).map(str::to_string).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "expression has no output name; use .alias() to name it")
                        .with_operation("dataframe::with_columns")
                })
            })
            .collect::<Result<_>>()?;

        let results: Vec<Series> = exprs.iter().map(|(_, expr)| expr.eval(self)).collect::<Result<_>>()?;
        let pairs: Vec<(&str, Series)> = names.iter().map(String::as_str).zip(results).collect();
        self.assign(pairs)
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
        let target_type = target_type.into();
        let col = self.columns.get(column).ok_or_else(|| missing_column_error("dataframe::cast_column", column))?;

        let cv = col.view()?;
        let new_col = unary::cast(&cv, &target_type)?.with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::cast_column")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
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

        crate::runtime::execution_domain::ExecutionDomain::require_same_device_and_fence_secondary(
            col.execution_domain(),
            value.execution_domain(),
            operation,
        )?;
        let cv = col.view()?;
        let exec_options = col.execution_domain().explicit_exec_options_raw();
        let new_col = crate::ops::replace::replace_nulls_with_exec_options(
            &cv,
            value,
            col.execution_domain(),
            exec_options.as_ref(),
            operation,
        )?;

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, operation)?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }
}
