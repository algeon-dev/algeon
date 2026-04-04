//! GPU column types: owned `Column` and borrowed `ColumnView`.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).
//! [`DeviceSlice`] is an even lower-level raw helper and is not the preferred
//! cross-crate interop contract; see [`crate::interop`] for borrowed GPU-view
//! interop.

mod convert;
mod device_slice;
mod owned;
mod view;

pub use device_slice::DeviceSlice;
pub use owned::Column;
pub use view::ColumnView;
