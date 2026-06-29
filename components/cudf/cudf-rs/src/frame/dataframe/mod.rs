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
mod gpu_record_batch;
pub(crate) mod group_keys;
mod info;
mod io;
mod join;
mod names;
mod owned;
mod reshape;
mod row_ops;
mod row_predicate;
mod select;
mod sort_ops;

pub use column_ops::WithColumnExpr;
pub(crate) use concat_ops::trigger_concat_by_name_missing_planned_dtype_for_test;
pub use concat_ops::{ConcatByNameOptions, SchemaAlignment};
pub use context::DataFrameExecutionScope;
pub(crate) use fillna_ops::trigger_dataframe_dropna_subset_count_limit_error_for_test;
pub use fillna_ops::{DropDuplicatesOptions, DropNaHow, DropNaOptions};
pub use gpu_record_batch::GpuRecordBatch;
pub use info::{
    ColumnMemoryAccounting, DataFrameColumnInfo, DataFrameInfo, DataFrameMemoryAccounting, DataTypeCount, InfoDetail,
};
pub use io::{
    CsvReadOptions, CsvWriteOptions, JsonReadOptions, JsonWriteOptions, MultiParquetReadOptions, ParquetReadOptions,
    ParquetWriteOptions,
};
pub use join::{
    DataFrameCrossJoinOptions, DataFrameFilteredJoin, DataFrameFilteredJoinOptions, DataFrameFilteredJoinProbeState,
    DataFrameHashJoin, DataFrameHashJoinMaterialized, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices,
    DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameMixedJoinOptions,
    DataFrameMixedJoinSizeData, DataFramePredicateJoinOptions,
};
pub use owned::{DataFrame, DataFrameJoinSides};
pub(crate) use owned::{derive_frame_domain_from_columns, resolve_frame_domain_from_authoritative_source};
pub use reshape::MeltOptions;
pub(crate) use row_ops::{
    trigger_dataframe_row_range_end_limit_error_for_test, trigger_dataframe_row_range_overflow_error_for_test,
    trigger_dataframe_row_range_start_limit_error_for_test,
};
