//! An Iceberg table executed natively on the GPU returns the rows a
//! DataFusion listing table returns over the same data files.

#[path = "support/warehouse.rs"]
mod support;

use std::{collections::HashMap, sync::Arc, time::Duration};

use algeon_datafusion::{
    AttemptOptions,
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    planner::AlgeonGpuExecutionMode,
};
use datafusion::{
    arrow::{array::RecordBatch, util::pretty::pretty_format_batches},
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::SessionStateBuilder,
    physical_plan::displayable,
    prelude::{SessionConfig, SessionContext},
};
use futures::TryStreamExt;

use crate::support::{Warehouse, people_schema, write_people};

const SQL: &str = "SELECT name, year, SUM(id) AS total FROM {table} WHERE id > 1 GROUP BY name, year ORDER BY name";

fn sorted_text(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).unwrap().to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_warehouse_table_executes_natively_with_listing_table_results() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let mut files = write_people(&table, &[1, 2, 3, 250]).await;
    files.extend(write_people(&table, &[101, 102, 103]).await);
    let data_dir = files[0].file_path().rsplit_once('/').unwrap().0.to_string();
    warehouse.append(&table, files).await;

    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .devices([device])
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .build()
        .expect("the Iceberg GPU test requires CUDA device 0");
    let state = backend
        .install_on(
            SessionStateBuilder::new_with_default_features()
                .with_config(SessionConfig::new().with_target_partitions(1)),
        )
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    warehouse.register(&ctx).await;

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let plan = attempt.plan_sql(&ctx, &SQL.replace("{table}", "iceberg.ns.people")).await.unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    let operators = rendered.lines().map(|line| line.trim().split(':').next().unwrap()).collect::<Vec<_>>();
    assert_eq!(operators, ["NativeAdmissionExec", "AlgeonNativeExec"], "{rendered}");
    let native = attempt.execute_physical_plan(plan, &ctx).unwrap().try_collect::<Vec<_>>().await.unwrap();
    attempt.completion().await;

    let cpu = SessionContext::new();
    let options = ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
    let config = ListingTableConfig::new(ListingTableUrl::parse(format!("{data_dir}/")).unwrap())
        .with_listing_options(options)
        .infer_schema(&cpu.state())
        .await
        .unwrap();
    cpu.register_table("listing", Arc::new(ListingTable::try_new(config).unwrap())).unwrap();
    let expected = cpu.sql(&SQL.replace("{table}", "listing")).await.unwrap().collect().await.unwrap();

    assert_eq!(native.iter().map(RecordBatch::num_rows).sum::<usize>(), 6);
    assert_eq!(sorted_text(&native), sorted_text(&expected));
    backend.close_and_wait(Duration::from_secs(10)).unwrap();
}
