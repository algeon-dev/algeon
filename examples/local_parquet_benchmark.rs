use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use cudf_query_engine::{
    ExecutionContext as NativeExecutionContext, MemoryControlPolicy as NativeMemoryControlPolicy,
    PlanNode as NativePlanNode, QueryMetrics as NativeQueryMetrics, QueryOutput as NativeQueryOutput,
    QueryPlan as NativeQueryPlan, SourceNode as NativeSourceNode, execute_query_plan as execute_native_query_plan,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue, utils::memory::get_record_batch_memory_size},
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSourceExec,
    },
    execution::{SessionStateBuilder, context::TaskContext},
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::{
        arrow::ArrowWriter,
        basic::Compression,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    },
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect, displayable,
        filter::FilterExec,
        limit::LocalLimitExec,
        metrics::MetricsSet,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CandidateOutcome, CudfGpuIslandExec, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
    CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt, Error as AdapterError, GpuFallbackPolicy,
    LargerIslandBenchmarkMetrics, LargerIslandBenchmarkPlanning, NativeCudfExec, NativeCudfExecutionConfig,
    NativeDataFusionPlanningReport, TimedBenchmarkRun, lower_physical_plan_to_native, optimize_plan,
};

#[path = "local_parquet_benchmark/case_runner.rs"]
mod case_runner;
#[path = "local_parquet_benchmark/config.rs"]
mod config;
#[path = "local_parquet_benchmark/dataset.rs"]
mod dataset;
#[path = "local_parquet_benchmark/errors.rs"]
mod errors;
#[path = "local_parquet_benchmark/execution.rs"]
mod execution;
#[path = "local_parquet_benchmark/matrix.rs"]
mod matrix;
#[path = "local_parquet_benchmark/metrics.rs"]
mod metrics;
#[path = "local_parquet_benchmark/probes.rs"]
mod probes;
#[path = "local_parquet_benchmark/reports.rs"]
mod reports;
#[path = "local_parquet_benchmark/validation.rs"]
mod validation;
#[path = "local_parquet_benchmark/workload.rs"]
mod workload;

use case_runner::*;
use config::*;
use dataset::*;
use errors::*;
use execution::*;
use matrix::*;
use metrics::*;
use probes::*;
use reports::*;
use validation::*;
use workload::*;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let run_mode = BenchmarkRunMode::from_env("CUDF_PARQUET_BENCH_RUN")?;
    let config = LocalParquetBenchmarkConfig::from_env()?;

    match run_mode {
        BenchmarkRunMode::SingleCase => {
            let report = run_configured_benchmark_case(&config).await?;
            print!("{}", benchmark_case_prologue(&report, &config));
            println!();
            print!("{}", report.report.to_text());
        },
        BenchmarkRunMode::PhaseCMatrix => {
            let report = run_phase_c_matrix(&config).await?;
            print!("{}", report.to_text());
            if !report.decision_gate_pass() {
                return Err(plan_error(format!(
                    "Phase C DataFusion native exec matrix decision gate failed: unexpected_outcomes={}",
                    report.unexpected_outcome_count()
                )));
            }
        },
    }

    Ok(())
}

#[cfg(test)]
#[path = "local_parquet_benchmark/tests.rs"]
mod tests;
