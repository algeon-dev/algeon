use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, GpuDeviceMemoryLimit, GpuMemoryOwnership},
    cugraph_sql::{
        CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphAlgorithmParams, CugraphBfsOutputMode, CugraphBfsSource,
        CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphSqlConfig,
        parse_cugraph_algorithm_spec, stable_result_schema,
    },
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::{AlgeonNativeComposedExec, AlgeonNativeExec},
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
    common::{DataFusionError, ScalarValue},
    datasource::MemTable,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::{ExecutionPlan, displayable},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

fn typed_algeon_error(error: &DataFusionError) -> &Error {
    error.find_algeon_error().expect("cuGraph DataFusion error must preserve the typed Algeon error")
}

#[path = "cugraph_sql_tests/gpu.rs"]
pub mod gpu;
#[cfg(test)]
#[path = "cugraph_sql_tests/metadata.rs"]
mod metadata;
#[cfg(test)]
#[path = "cugraph_sql_tests/parser.rs"]
mod parser;
#[cfg(test)]
#[path = "cugraph_sql_tests/planning.rs"]
mod planning;
#[path = "cugraph_sql_tests/support.rs"]
mod support;
