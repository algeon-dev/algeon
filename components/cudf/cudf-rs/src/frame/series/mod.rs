//! An `Arc`-shared GPU column vector — the primary user-facing 1D type.
//!
//! `Series` wraps an `Arc<Column>` without schema metadata. It is the
//! high-level counterpart to the low-level owned [`Column`](crate::core::Column)
//! type.
//!
//! Clone is cheap: it bumps the `Arc` reference count — no GPU data is copied.

mod constructors;
mod context;
mod display;
mod host_column_data;
mod ops;
mod owned;
pub(crate) mod reduction;
mod strings;

pub use host_column_data::HostColumnData;
pub use owned::{IntoSeries, Series};
