//! A named, `Arc`-shared GPU column — the primary user-facing column type.
//!
//! `Series` wraps an `Arc<Column>` with an optional name. It is the
//! high-level counterpart to the low-level [`Column`] / [`ColumnView`] types.
//!
//! Clone is cheap: it bumps the `Arc` reference count — no GPU data is copied.

mod constructors;
mod naming;
mod owned;
mod view;

pub use owned::{IntoSeries, Series};
pub use view::SeriesView;
