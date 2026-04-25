use std::{
    fs,
    path::Path,
    process::Command,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::ScalarValue,
    datasource::MemTable,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    backend::{NexusAdmissionPolicy, NexusGpuBackend},
    graph_sql::{
        CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
        CUGRAPH_VALIDATE_CALL_FUNCTION, CugraphAlgorithm, CugraphAlgorithmParams, CugraphBfsOutputMode,
        CugraphBfsSource, CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphSqlConfig,
        parse_cugraph_algorithm_spec, stable_result_schema,
    },
    native_exec::{NativeCudfComposedExec, NativeCudfExec},
    native_report::NativeDataFusionExecutionMetrics,
    planner::CudfOptimizerConfig,
    policy::GpuFallbackPolicy,
    session::CudfSessionStateBuilderExt,
};
use nexus_query_engine::runtime::SlotGuarantee;

#[path = "cugraph_sql_tests/gpu.rs"]
pub mod gpu;
#[path = "cugraph_sql_tests/metadata.rs"]
mod metadata;
#[path = "cugraph_sql_tests/parser.rs"]
mod parser;
#[path = "cugraph_sql_tests/planning.rs"]
mod planning;
#[path = "cugraph_sql_tests/support.rs"]
mod support;
