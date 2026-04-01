//! GPU column types: owned `Column` and borrowed `ColumnView`.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).

mod convert;
mod device_slice;
mod owned;
mod view;

pub use device_slice::DeviceSlice;
pub use owned::Column;
pub use view::ColumnView;
