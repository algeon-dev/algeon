#[path = "../../support/closing_backend.rs"]
mod closing_backend;

use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, GpuMemoryOwnership},
    cugraph_sql::{
        CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphGraphOptions, CugraphSqlConfig, stable_result_schema,
    },
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::AlgeonNativeExec,
    native_report::NativeDataFusionExecutionMetrics,
    planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray, StringViewArray,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::DataFusionError,
    datasource::MemTable,
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::{ExecutionPlan, displayable},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

fn typed_algeon_error(error: &DataFusionError) -> &Error {
    error.find_algeon_error().expect("cuGraph DataFusion error must preserve the typed Algeon error")
}

mod binder;
mod gpu;
mod metadata;
mod planning;

mod support;
