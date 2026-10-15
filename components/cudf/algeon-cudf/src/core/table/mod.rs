//! Core table definitions for collections of equal-length GPU columns.
//!
//! Named-column tabular workflows use [`DataFrame`](crate::DataFrame).

mod errors;
mod owned;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;
mod view;

pub(crate) const TABLE_HANDLE_INLINE_CAPACITY: usize = 8;
pub(crate) use errors::table_column_count_limit_error;
pub use owned::Table;
pub use view::TableView;
