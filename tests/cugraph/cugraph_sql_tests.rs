use std::{
    fs,
    path::Path,
    process::Command,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, ScalarValue},
    datasource::MemTable,
    execution::SessionStateBuilder,
    logical_expr::{col, lit},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    backend::{NexusAdmissionPolicy, NexusGpuBackend},
    error::{DataFusionErrorExt, Error, ErrorCode},
    graph_sql::{
        CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, CugraphAlgorithmParams, CugraphBfsOutputMode, CugraphBfsSource,
        CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphSqlConfig, parse_cugraph_algorithm_spec,
        stable_result_schema,
    },
    native_exec::{NexusNativeComposedExec, NexusNativeExec},
    native_report::NativeDataFusionExecutionMetrics,
    planner::NexusNativeOptimizerConfig,
    session::NexusSessionStateBuilderExt,
};
use nexus_query_engine::runtime::SlotGuarantee;

fn typed_nexus_error(error: &DataFusionError) -> &Error {
    error.find_nexus_error().expect("cuGraph DataFusion error must preserve the typed Nexus error")
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
