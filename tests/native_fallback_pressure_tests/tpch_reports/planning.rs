use super::{super::*, tpch_expectations::*};

#[path = "planning/assertions.rs"]
mod assertions;

use assertions::assert_tpch_datafusion_planning_report_contract;

#[tokio::test]
async fn tpch_datafusion_planning_report_emits_one_stable_row_per_query() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-datafusion-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeDataFusionPlanningReport::collect(items).with_tpch_dataset_metadata(metadata.clone());

    assert_tpch_datafusion_planning_report_contract(&report, metadata.metadata_fingerprint.as_str());

    fs::remove_dir_all(dir).unwrap();
}
