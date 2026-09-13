//! Arrow `RecordBatch` interop: GPU `DataFrame` ↔ Arrow `RecordBatch`
//! conversion.
//!
//! This module is specifically the Rust `arrow` crate / `RecordBatch` path.
//! It provides a **host-copy** conversion: data is transferred GPU → CPU →
//! Arrow (or Arrow → CPU → GPU).
//!
//! This is deliberately distinct from [`crate::interop::arrow_device`], which
//! implements Arrow C Device ABI handoff for existing device-backed buffers.
//! That device-interop path can preserve device pointers for many supported
//! layouts, but it is a different contract surface from `RecordBatch` and is
//! not the API implemented here.
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
use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::runtime::{BoundaryObservation, execution_domain::ExecutionDomain, record_boundary_observation_in_domain};

mod batch;
mod estimate;
mod export;
mod import;
mod schema;

pub(crate) mod testing_hooks;

pub use batch::{dataframe_from_record_batch, dataframe_from_record_batches};
pub(crate) use estimate::estimate_arrow_host_record_batch_memory_bytes;
pub use estimate::{estimate_list_import_device_memory_bytes, estimate_record_batches_import_required_budget_bytes};
use export::{map_numeric_to_arrow_error, map_record_batch_error};
pub use import::all_null_list_series;
use import::{large_string_array_to_column, string_array_to_column};
pub use schema::{
    ArrowHostField, ArrowHostSchema, arrow_type_from_cudf_dtype, cudf_dtype_can_export_to_arrow_type,
    cudf_dtype_from_arrow_type, is_supported_arrow_numeric_type, is_supported_arrow_type, validate_arrow_schema,
    validate_record_batch_schema,
};

thread_local! {
    static HOST_ARROW_LAST_BOUNDARY_OBSERVATION: Cell<Option<BoundaryObservation>> = const { Cell::new(None) };
    static HOST_ARROW_EXPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
    static HOST_ARROW_IMPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
}

static HOST_ARROW_EXPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_ARROW_IMPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
pub(super) fn reset_host_interop_observation() {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_EXPORT_CALL_COUNT, baseline);
    });
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_IMPORT_CALL_COUNT, baseline);
    });
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(|last| last.set(None));
}

pub(super) fn host_export_call_count() -> usize {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_EXPORT_CALL_COUNT, baseline))
}

pub(super) fn host_import_call_count() -> usize {
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_IMPORT_CALL_COUNT, baseline))
}

fn reset_call_count_baseline(counter: &AtomicUsize, baseline: &Cell<usize>) {
    baseline.set(counter.load(Ordering::Acquire));
}

fn call_count_since(counter: &AtomicUsize, baseline: &Cell<usize>) -> usize {
    counter.load(Ordering::Acquire).saturating_sub(baseline.get())
}

pub(super) fn last_host_boundary_observation() -> Option<BoundaryObservation> {
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(Cell::get)
}

fn record_host_export_call() {
    HOST_ARROW_EXPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

fn record_host_import_call() {
    HOST_ARROW_IMPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

fn record_host_boundary_observation_in_domain(domain: &Arc<ExecutionDomain>, observation: BoundaryObservation) {
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(|last| last.set(Some(observation)));
    record_boundary_observation_in_domain(domain, observation);
}
