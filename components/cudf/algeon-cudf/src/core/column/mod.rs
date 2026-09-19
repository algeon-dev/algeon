//! GPU column types and raw device-memory helpers.
//!
//! **Advanced/Core.** Most users should use [`Series`](crate::Series).
//! [`DeviceSlice`] is an even lower-level raw helper and is not the preferred
//! cross-crate interop contract; see [`crate::interop`] for borrowed GPU-view
//! interop.

mod convert;
mod device_buffer;
mod device_slice;
mod owned;
mod sequence;
mod testing_hooks;
mod view;

pub(crate) use convert::copy_view_to_owned;
pub use device_buffer::{DeviceBuffer, ValidatedFloat32DeviceBuffer, ValidatedInt64DeviceBufferRange};
pub use device_slice::DeviceSlice;
pub use owned::Column;
pub(crate) use sequence::position_sequence_in_domain;
pub(crate) use testing_hooks::slice_view_for_test;
pub use view::ColumnView;
pub(crate) use view::{
    adopt_owned_column_view_for_test, column_view_copy_fixed_width_to_host_for_benchmark,
    column_view_to_fixed_width_host_bytes_for_test, column_view_to_raw_bytes_for_test,
    trigger_fixed_width_host_byte_range_overflow_for_test,
};
