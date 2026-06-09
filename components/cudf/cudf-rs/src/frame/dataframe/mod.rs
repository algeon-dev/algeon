//! A named-column wrapper around GPU columns for ergonomic DataFrame operations.
//!
//! [`DataFrame`] stores columns individually as `Arc<Column>`, enabling
//! schema-only operations like rename, select, without_columns, and clone to
//! share GPU memory via reference counting instead of deep-copying column
//! data.

mod context;
mod display;
mod info;
mod io;
mod names;
mod owned;
mod reshape;
mod select;
mod table_view;

pub use context::DataFrameExecutionScope;
pub use info::{
    ColumnMemoryAccounting, DataFrameColumnInfo, DataFrameInfo, DataFrameMemoryAccounting, DataTypeCount, InfoDetail,
};
pub use io::{
    CsvReadOptions, CsvWriteOptions, JsonReadOptions, JsonWriteOptions, ParquetReadOptions, ParquetWriteOptions,
};
pub use owned::{DataFrame, DataFrameJoinResult};
pub(crate) use owned::{derive_frame_domain_from_columns, resolve_frame_domain_from_authoritative_source};
#[doc(hidden)]
pub use reshape::IntoWithColumnExpr;
pub use reshape::{DropDuplicatesOptions, DropNaHow, DropNaOptions, MeltOptions};
pub use table_view::DataFrameTableView;
