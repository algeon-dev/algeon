//! GPU table types.
//!
//! **Advanced/Core.** Most users should use [`DataFrame`](crate::DataFrame).

mod errors;
mod owned;
pub(crate) mod testing_hooks;
mod view;

pub(crate) const TABLE_HANDLE_INLINE_CAPACITY: usize = 8;
pub(crate) use errors::table_column_count_limit_error;
pub use owned::Table;
pub use view::TableView;
