//! Expression DSL, operation option types and host-side sizing helpers.
//!
//! GPU operations run through [`crate::DataFrame::on`] and
//! [`crate::Series::on`]; the kernels behind them are crate-internal. This
//! module keeps what those methods take and return, such as
//! [`sorting::SortOptions`], [`join::MergeOptions`],
//! [`binary::BinaryOperator`] and [`ast::AstExpression`], plus host-side
//! helpers like [`binary::infer_binary_output_dtype`] and
//! [`materialization::table_upper_bound_preflight`].
//!
//! The root re-exports the expression DSL surface ([`Expr`], [`col`], [`lit`],
//! [`lit_null`] and [`when`]). [`Expr::agg`] exposes broadcast aggregate
//! expressions for row-aligned use, and [`Expr::str`] exposes string-specific
//! expression methods.
//!
//! The expression DSL is eager when evaluated against a
//! [`crate::DataFrame`]; it builds a declarative expression tree, but it is not
//! a lazy query optimizer. See [`expr`] for the row-aligned broadcast-reduction
//! semantics and naming rules around [`Expr::agg`].

pub mod ast;
pub mod binary;
pub mod concatenate;
pub mod copying;
pub mod datetime;
pub mod dictionary;
pub mod expr;
pub(crate) mod factorization;
pub mod filter;
pub mod groupby;
pub mod join;
pub mod lookup;
pub mod materialization;
pub mod partitioning;
pub mod reduction;
pub mod replace;
pub mod reshape;
pub mod search;
pub mod sorting;
pub mod stream_compaction;
pub mod strings;
pub mod unary;
pub(crate) mod validation;

pub use expr::{Expr, col, lit, lit_null, when};
