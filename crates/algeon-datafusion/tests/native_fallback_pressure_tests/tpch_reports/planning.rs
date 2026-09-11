use super::{super::*, tpch_expectations::*};

#[path = "planning/assertions.rs"]
mod assertions;

use assertions::assert_tpch_datafusion_planning_report_contract;

#[tokio::test]
async fn tpch_datafusion_planning_report_tracks_primary_rows_and_mixed_plan_fragments() {
    let dir = unique_temp_dir("algeon-datafusion-tpch-datafusion-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeDataFusionPlanningReport::collect(items)
        .with_workload_dataset_metadata(metadata.to_workload_dataset_metadata());

    assert_tpch_datafusion_planning_report_contract(&report, metadata.metadata_fingerprint.as_str());

    fs::remove_dir_all(dir).unwrap();
}
