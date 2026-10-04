//! Arrow `RecordBatch` interop: GPU `DataFrame` ↔ Arrow `RecordBatch`
//! conversion.
//!
//! This module is specifically the Rust `arrow` crate / `RecordBatch` path.
//! It provides a **host-copy** conversion: data is transferred GPU → CPU →
//! Arrow (or Arrow → CPU → GPU).
//!
//! Device-resident handoff to other RAPIDS libraries uses the prepared exports
//! in [`crate::interop::contract`], not this module.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DataFrame, Runtime};
//!
//! let rt = Runtime::new().unwrap();
//! let ids = rt.series(&[1i32, 2, 3]).unwrap();
//! let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
//! let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();
//!
//! // GPU → Arrow RecordBatch (host-copy)
//! let batch = df.to_record_batch().unwrap();
//! assert_eq!(batch.num_rows(), 3);
//!
//! // Arrow RecordBatch → GPU DataFrame (host-copy) inside the explicit runtime
//! let df2 = DataFrame::from_record_batch(&rt, &batch).unwrap();
//! assert_eq!(df2.num_columns(), 2);
//! ```
mod batch;
mod estimate;
mod export;
mod import;
mod schema;

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) mod testing_hooks;

pub use batch::{dataframe_from_record_batch, dataframe_from_record_batches};
pub(crate) use estimate::estimate_arrow_host_record_batch_memory_bytes;
pub use estimate::estimate_list_import_device_memory_bytes;
#[doc(hidden)]
pub use estimate::estimate_record_batches_import_required_budget_bytes;
#[cfg(any(test, feature = "test-hooks"))]
use export::{map_numeric_to_arrow_error, map_record_batch_error};
pub use import::all_null_list_series;
#[cfg(any(test, feature = "test-hooks"))]
use import::{large_string_array_to_column, string_array_to_column};
pub use schema::{
    ArrowHostField, ArrowHostSchema, arrow_type_from_cudf_dtype, cudf_dtype_can_export_to_arrow_type,
    cudf_dtype_from_arrow_type, is_supported_arrow_numeric_type, is_supported_arrow_type, validate_arrow_schema,
    validate_record_batch_schema,
};

fn record_host_export_call() {
    #[cfg(any(test, feature = "test-hooks"))]
    testing_hooks::record_host_export_call();
}

fn record_host_import_call() {
    #[cfg(any(test, feature = "test-hooks"))]
    testing_hooks::record_host_import_call();
}
