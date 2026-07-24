use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{record_batch::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, utils::memory::get_record_batch_memory_size},
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect, metrics::MetricsSet},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    error::{DataFusionErrorExt, Error as AdapterError},
    native_exec::NexusNativeExecutionConfig,
    native_report::{
        NativeAggregateStrategyMetricRow, NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome,
        NativeDataFusionPlanningReport, NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath,
        NativeDataFusionRuntimeOutcome, NativeDynamicFilterRuntimeMetricRow, NativeGraphReservationMetricRow,
        NativeGraphRuntimeMetricRow, NativeJoinStrategyMetricRow, NativeSourceDiagnosticSummary,
        NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow,
    },
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};
use datafusion_nexus_workloads::tpch::{
    NEXUS_PARQUET_DIR_ENV, NEXUS_TPCH_GENERATION_COMMAND_ENV, NEXUS_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES, TPCH_TABLES,
    TpchDatasetMetadata, TpchQuery, TpchTableLayout, TpchWorkloadError, default_tpch_dataset_with_fingerprint,
    describe_tpch_dataset_with_content_fingerprint, prepare_tpch_report_dataset, register_tpch_table_layout,
};
use nexus_query_engine::mode::NativePipelineExecutionMode;

use crate::report_workload::tpch::tpch_report_session_config;

mod case_runner;
mod config;
mod dataset;
mod errors;
mod execution;
mod reports;
mod strategy_gate;
mod validation;

use case_runner::*;
use config::*;
use errors::*;
use reports::*;

/// Run the native TPC-H benchmark end to end from environment configuration,
/// print the TSV report to stdout, and enforce the strategy gates.
pub async fn run_main() -> DataFusionResult<()> {
    let config = NativeTpchBenchmarkConfig::from_env()?;
    let work_dir = unique_temp_dir("datafusion-nexus-native-tpch-benchmark");
    let report = run_native_tpch_benchmark(&config, &work_dir).await?;
    let text = report.to_tsv();
    print!("{text}");
    let large_keyed_gate_result = strategy_gate::validate_large_keyed_join_strategy_gate_from_env(&text);
    let sf100_reservation_gate_result =
        strategy_gate::validate_sf100_bounded_reservation_gate_for_config(&text, &config);
    let sf100_full_reservation_health_gate_result =
        strategy_gate::validate_sf100_full_reservation_health_gate_for_config(&text, &config);
    fs::remove_dir_all(work_dir).ok();
    match large_keyed_gate_result {
        Ok(Some(summary)) => eprintln!("{}", summary.progress_line()),
        Ok(None) => {},
        Err(error) => {
            eprintln!("[native_tpch_benchmark] stage=large_keyed_join_strategy_gate outcome=failed error={error}");
            return Err(error);
        },
    }
    match sf100_reservation_gate_result {
        Ok(Some(summary)) => eprintln!("{}", summary.progress_line()),
        Ok(None) => {},
        Err(error) => {
            eprintln!("[native_tpch_benchmark] stage=sf100_bounded_reservation_gate outcome=failed error={error}");
            return Err(error);
        },
    }
    match sf100_full_reservation_health_gate_result {
        Ok(Some(summary)) => eprintln!("{}", summary.progress_line()),
        Ok(None) => {},
        Err(error) => {
            eprintln!("[native_tpch_benchmark] stage=sf100_full_reservation_health_gate outcome=failed error={error}");
            return Err(error);
        },
    }
    Ok(())
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests;
