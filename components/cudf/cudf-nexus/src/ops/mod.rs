//! Low-level operation modules plus the expression DSL entry points.
//!
//! Most users should start from high-level [`crate::DataFrame`] and
//! [`crate::Series`] methods. `cudf::ops` is the lower-level operation layer:
//! it groups domain-specific kernels under namespaced modules such as
//! [`join`], [`groupby`], [`sorting`], [`filter`], [`strings`], and
//! [`reduction`].
//!
//! The root keeps only the high-frequency expression DSL surface
//! ([`Expr`], [`col`], [`lit`], and [`when`]) plus the standard arithmetic
//! trait implementations for borrowed
//! [`Column`] values.
//!
//! Within that expression DSL surface, [`Expr::agg`] exposes broadcast aggregate expressions for
//! row-aligned use, and [`Expr::str`] exposes string-specific expression methods.
//!
//! # Ownership and Evaluation
//!
//! Most low-level functions in this module family take borrowed
//! [`ColumnView`](crate::core::column::ColumnView) or
//! [`TableView`](crate::core::table::TableView) inputs and return newly owned
//! [`Column`] or
//! [`Table`](crate::core::table::Table) outputs. This is the advanced API
//! boundary where callers opt into explicit low-level data layout and
//! operation selection.
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
pub mod hashing;
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

use std::ops::{Add, Div, Mul, Sub};

pub use expr::{Expr, col, lit, lit_null, when};

use crate::core::{column::Column, error::Result, scalar::Scalar};

macro_rules! impl_column_scalar_ops {
    ($($rust_type:ty),+) => {
        $(
            impl Add<$rust_type> for &Column {
                type Output = Result<Column>;
                fn add(self, rhs: $rust_type) -> Self::Output {
                    self.add_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Sub<$rust_type> for &Column {
                type Output = Result<Column>;
                fn sub(self, rhs: $rust_type) -> Self::Output {
                    self.sub_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Mul<$rust_type> for &Column {
                type Output = Result<Column>;
                fn mul(self, rhs: $rust_type) -> Self::Output {
                    self.mul_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
            impl Div<$rust_type> for &Column {
                type Output = Result<Column>;
                fn div(self, rhs: $rust_type) -> Self::Output {
                    self.div_scalar(&Scalar::from_value_in_domain(rhs, self.execution_domain())?)
                }
            }
        )+
    };
}

impl_column_scalar_ops!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl Add<&Column> for &Column {
    type Output = Result<Column>;
    fn add(self, rhs: &Column) -> Self::Output {
        Column::add(self, rhs)
    }
}

impl Sub<&Column> for &Column {
    type Output = Result<Column>;
    fn sub(self, rhs: &Column) -> Self::Output {
        Column::sub(self, rhs)
    }
}

impl Mul<&Column> for &Column {
    type Output = Result<Column>;
    fn mul(self, rhs: &Column) -> Self::Output {
        Column::mul(self, rhs)
    }
}

impl Div<&Column> for &Column {
    type Output = Result<Column>;
    fn div(self, rhs: &Column) -> Self::Output {
        Column::div(self, rhs)
    }
}
