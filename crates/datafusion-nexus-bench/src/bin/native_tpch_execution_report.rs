use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::native_report::NativeDataFusionExecutionReport;
use datafusion_nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        RelationalExecutionSelection,
        tpch::{
            tpch_datafusion_execution_workload_items_with_optimizer_config, tpch_native_optimizer_config,
            tpch_report_session_config,
        },
    },
};
use datafusion_nexus_workloads::tpch::{TpchDatasetMetadata, TpchWorkloadResult, prepare_tpch_report_dataset};
use nexus_query_engine::mode::NativePipelineExecutionMode;

const NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_NATIVE_PIPELINE_EXECUTION_MODE";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpch-execution");
    let TpchExecutionReport { report, metadata } = tpch_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpch_report(dir: &std::path::Path) -> TpchWorkloadResult<TpchExecutionReport> {
    let relational_execution = RelationalExecutionSelection::native();
    let native_pipeline_execution_mode = native_pipeline_execution_mode_from_env();
    eprintln!(
        "[native_tpch_execution_report] stage=run_config relational_execution={} native_pipeline_execution_mode={}",
        relational_execution.as_str(),
        native_pipeline_execution_mode.as_str(),
    );
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let optimizer_config = relational_execution
        .configure_native_optimizer(
            tpch_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode),
        )
        .expect("native execution report installs the native rule");
    let items = tpch_datafusion_execution_workload_items_with_optimizer_config(
        &root,
        tpch_report_session_config(),
        optimizer_config,
    )
    .await?;
    let report = NativeDataFusionExecutionReport::collect(items);
    Ok(TpchExecutionReport { report, metadata })
}

struct TpchExecutionReport {
    report: NativeDataFusionExecutionReport,
    metadata: TpchDatasetMetadata,
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn native_pipeline_execution_mode_from_env() -> NativePipelineExecutionMode {
    let mode = std::env::var(NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV)
        .unwrap_or_else(|_| NativePipelineExecutionMode::StrictBounded.as_str().to_string());
    mode.parse::<NativePipelineExecutionMode>().unwrap_or_else(|error| {
        panic!(
            "unsupported {NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV}={}; expected strict_bounded or adaptive",
            error.value()
        )
    })
}
