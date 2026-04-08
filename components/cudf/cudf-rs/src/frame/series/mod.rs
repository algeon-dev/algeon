//! A named, `Arc`-shared GPU column — the primary user-facing column type.
//!
//! `Series` wraps an `Arc<Column>` with an optional name. It is the
//! high-level counterpart to the low-level [`Column`] / [`ColumnView`] types.
//!
//! Clone is cheap: it bumps the `Arc` reference count — no GPU data is copied.

mod constructors;
mod context;
mod display;
mod naming;
mod ops;
mod owned;
pub(crate) mod reduction;
mod strings;

pub use owned::{IntoSeries, Series};
