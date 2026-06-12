//! A named-column wrapper around GPU columns for ergonomic DataFrame operations.
//!
//! [`DataFrame`] stores columns individually as `Arc<Column>`, enabling
//! schema-only operations like rename, select, without_columns, and clone to
//! share GPU memory via reference counting instead of deep-copying column
//! data.

mod column_ops;
mod concat_ops;
mod context;
mod display;
mod fillna_ops;
mod info;
mod io;
mod names;
mod owned;
mod reshape;
mod row_ops;
mod row_predicate;
mod select;
mod sort_ops;

pub use column_ops::WithColumnExpr;
pub use concat_ops::{ColumnOrder, ConcatByNameOptions, SchemaAlignment};
pub use context::DataFrameExecutionScope;
pub use fillna_ops::{DropDuplicatesOptions, DropNaHow, DropNaOptions};
pub use info::{
    ColumnMemoryAccounting, DataFrameColumnInfo, DataFrameInfo, DataFrameMemoryAccounting, DataTypeCount, InfoDetail,
};
pub use io::{
    CsvReadOptions, CsvWriteOptions, JsonReadOptions, JsonWriteOptions, ParquetReadOptions, ParquetWriteOptions,
};
pub use owned::{DataFrame, DataFrameJoinSides};
pub(crate) use owned::{derive_frame_domain_from_columns, resolve_frame_domain_from_authoritative_source};
pub use reshape::MeltOptions;
