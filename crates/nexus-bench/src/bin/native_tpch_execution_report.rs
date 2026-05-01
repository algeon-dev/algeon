use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::native_report::NativeDataFusionExecutionReport;
use nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        RelationalExecutionSelection,
        tpch::{
            tpch_datafusion_execution_workload_items_with_optimizer_config, tpch_native_optimizer_config,
            tpch_report_session_config,
        },
    },
};
use nexus_workloads::tpch::{TpchDatasetMetadata, TpchWorkloadResult, prepare_tpch_report_dataset};

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpch-execution");
    let TpchExecutionReport { report, metadata } = tpch_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpch_report(dir: &std::path::Path) -> TpchWorkloadResult<TpchExecutionReport> {
    let relational_execution = RelationalExecutionSelection::native();
    eprintln!("[native_tpch_execution_report] stage=run_config relational_execution={}", relational_execution.as_str(),);
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let optimizer_config = relational_execution
        .configure_native_optimizer(tpch_native_optimizer_config())
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
