use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_cudf::{
    NativeDataFusionExecutionReport, TpchDatasetMetadata, prepare_tpch_report_dataset,
    tpch_datafusion_execution_workload_items, tpch_report_session_config,
};

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-cudf-native-tpch-execution");
    let TpchExecutionReport { report, metadata } = tpch_report(&dir).await.unwrap();
    print!("{}", report.with_tpch_dataset_metadata(metadata).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpch_report(dir: &std::path::Path) -> datafusion_cudf::TpchWorkloadResult<TpchExecutionReport> {
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let items = tpch_datafusion_execution_workload_items(&root, tpch_report_session_config()).await?;
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
