//! GPU table types: owned `Table` and borrowed `TableView`.
//!
//! **Advanced/Core.** Most users should use [`DataFrame`](crate::DataFrame).

mod owned;
mod view;

pub(crate) const TABLE_HANDLE_INLINE_CAPACITY: usize = 8;

pub use owned::Table;
pub use view::TableView;
