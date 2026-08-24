//! Expression DSL for declarative column operations.
//!
//! Use the DSL entry points [`col`], [`lit`], and [`when`] to start
//! expressions, then extend them with arithmetic, comparisons, casts,
//! naming, aggregation expression methods, and string expression methods
//! before evaluating them against a [`DataFrame`].
//!
//! # Reduction Shape
//!
//! Expr reductions accessed through [`Expr::agg`] are **broadcast
//! aggregates**. They reduce their input to a single scalar internally, then
//! broadcast that scalar back to the DataFrame row count so the result remains
//! row-aligned and can participate in larger expressions.
//!
//! That means `col("x").agg().sum()` is appropriate in expressions like
//! `col("x") / col("x").agg().sum()`, but broadcast reductions do not
//! inherit an output name for
//! [`DataFrame::with_columns`](crate::frame::dataframe::DataFrame::with_columns).
//! Use [`.alias()`](Expr::alias) or an explicit `(name, expr)` pair when you
//! want to materialize the broadcast result as a named column.
//!
//! If you want an actual scalar reduction, use the [`Series`] reduction APIs
//! such as [`Series::sum`] or [`Series::sum_value`] instead.
//!
//! # Literal Evaluation and Memory Behavior
//!
//! Literal constructors such as [`lit`], [`lit_decimal32_raw`],
//! [`lit_decimal64_raw`], [`lit_decimal128_raw`], [`lit_timestamp_raw`], and
//! [`lit_duration_raw`] are lazy expression builders. Creating the [`Expr`]
//! does not allocate GPU memory and does not borrow or transfer host memory.
//!
//! During [`Expr::eval`] or [`Expr::eval_in`], each literal is materialized as
//! a device-resident [`Scalar`] in the resolved output execution domain, then
//! broadcast into a newly allocated device [`Column`] with
//! the input [`DataFrame`] row count. The returned [`Series`] owns that new
//! column and retains the resolved provenance: by default the input frame's
//! stream and memory resource, or the same-device borrowed stream and memory
//! resource supplied through [`ExecutionOverrides`].
//!
//! Literal evaluation enqueues libcudf work and does not intentionally
//! synchronize the host before returning. Later host-visible reads of the
//! result, such as value extraction or host export, may synchronize the
//! result's execution domain.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{
//!     DataFrame, Expr, Runtime,
//!     ops::expr::{col, lit},
//! };
//!
//! let rt = Runtime::new().unwrap();
//! let x = rt.series(&[1i32, 2, 3]).unwrap();
//! let y = rt.series(&[10i32, 20, 30]).unwrap();
//! let df = DataFrame::from_series([("x", x), ("y", y)]).unwrap();
//!
//! let df2 = df
//!     .with_columns([
//!         (col("x") + col("y")).alias("sum"),
//!         (col("x") * 2i32).alias("doubled"),
//!         (col("x") / col("x").agg().sum()).alias("share_of_total"),
//!     ])
//!     .unwrap();
//! ```

use std::{cell::RefCell, sync::Arc};

use super::{
    binary::{self, BinaryOperator},
    concatenate, copying,
    datetime::{self as datetime_ops, DatetimeComponent, DatetimeRoundingFrequency},
    reduction::{self, ReduceAgg, ScanAgg, ScanType},
    replace::{self, FillPolicy},
    search,
    strings::{self as string_ops, StringSliceOptions, StripSide},
    unary::{self, UnaryOperator},
};
use crate::{
    core::{
        column::Column,
        error::{Error, ErrorCode, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::{
        hash::{FastHashMap, fast_hash_map_with_capacity},
        resolve::missing_column_error,
    },
    runtime::{
        ExecutionOverrides,
        execution_domain::{EnteredExecution, ExecutionDomain},
    },
};

struct ExprEvalContext<'df> {
    df: &'df DataFrame,
    output_domain: Arc<ExecutionDomain>,
    exec_options: EnteredExecution,
    operation: &'static str,
    normalized_columns: RefCell<FastHashMap<String, Series>>,
}

impl<'df> ExprEvalContext<'df> {
    fn new(df: &'df DataFrame, overrides: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Self> {
        let output_domain = overrides.resolve_output_domain(df.execution_domain())?;
        let exec_options = output_domain.enter_execution()?;
        Ok(Self {
            df,
            output_domain,
            exec_options,
            operation,
            normalized_columns: RefCell::new(fast_hash_map_with_capacity(df.num_columns())),
        })
    }

    fn with_output_domain(
        df: &'df DataFrame,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let exec_options = output_domain.enter_execution()?;
        Ok(Self {
            df,
            output_domain,
            exec_options,
            operation,
            normalized_columns: RefCell::new(fast_hash_map_with_capacity(df.num_columns())),
        })
    }

    fn required_exec_options(&self) -> &cudf_sys::cudf_exec_options_t {
        self.exec_options.options()
    }

    fn output_domain(&self) -> Arc<ExecutionDomain> {
        Arc::clone(&self.output_domain)
    }

    fn column(&self, name: &str) -> Result<Series> {
        if let Some(series) = self.normalized_columns.borrow().get(name) {
            return Ok(series.clone());
        }

        let column =
            self.df.columns_map().get(name).cloned().ok_or_else(|| missing_column_error(self.operation, name))?;
        let normalized = if ExecutionDomain::same_provenance(column.execution_domain(), &self.output_domain) {
            column
        } else {
            // copy_into_domain installs both fence directions internally.
            Arc::new(
                column
                    .copy_into_domain(Arc::clone(&self.output_domain), self.operation)
                    .map_err(|error| error.with_column(name.to_string()))?,
            )
        };
        let normalized = Series::from_column_arc(normalized).map_err(|error| error.with_column(name.to_string()))?;
        self.normalized_columns.borrow_mut().insert(name.to_string(), normalized.clone());
        Ok(normalized)
    }

    fn literal(&self, lit_val: &LitValue) -> Result<Series> {
        let scalar = lit_val.to_scalar_in_domain(&self.output_domain)?;
        let col = Column::from_scalar(&scalar, self.df.num_rows())?;
        Series::from_column(col)
    }

    fn wrap_column(&self, column: Column) -> Result<Series> {
        Series::from_column(column)
    }
}

pub(crate) fn eval_many_in_context<'a, I>(
    df: &DataFrame,
    overrides: &ExecutionOverrides<'_>,
    operation: &'static str,
    exprs: I,
) -> Result<(Arc<ExecutionDomain>, Vec<Series>)>
where
    I: IntoIterator<Item = &'a Expr>,
{
    let ctx = ExprEvalContext::new(df, overrides, operation)?;
    let output_domain = ctx.output_domain();
    let results = exprs.into_iter().map(|expr| expr.eval_with_context(&ctx)).collect::<Result<_>>()?;
    Ok((output_domain, results))
}

pub(crate) fn eval_in_output_domain(
    df: &DataFrame,
    output_domain: Arc<ExecutionDomain>,
    operation: &'static str,
    expr: &Expr,
) -> Result<Series> {
    let ctx = ExprEvalContext::with_output_domain(df, output_domain, operation)?;
    expr.eval_with_context(&ctx)
}
// ===========================================================================
// LitValue — type-erased scalar factory (crate-private)
// ===========================================================================

/// A type-erased factory for creating a [`Scalar`] from a literal value.
///
/// Not part of the public API — use [`lit()`] to create literals.
mod agg;
mod core;
mod datetime;
mod lit;
mod preflight;
mod strings;
mod when;

pub use core::{Expr, IntoExpr, col, concat};

pub use agg::ExprAggNamespace;
pub use datetime::ExprDatetimeNamespace;
pub use lit::{
    IntoLit, LitValue, lit, lit_decimal32_raw, lit_decimal64_raw, lit_decimal128_raw, lit_duration_raw, lit_null,
    lit_timestamp_raw,
};
pub use preflight::{
    DATAFRAME_EXPRESSION_ALLOCATION_PREFLIGHT_CONTRACT, DataFrameExpressionAllocationPreflight,
    DataFrameExpressionInputColumn, DataFrameExpressionPeakGroup,
    dataframe_expression_allocation_preflight_for_metadata,
};
pub use strings::{ExprStringNamespace, StringOperator};
pub use when::{ThenBuilder, WhenBuilder, when};
