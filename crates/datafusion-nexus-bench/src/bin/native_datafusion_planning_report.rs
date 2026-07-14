use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::NativeCudfExecutionConfig,
    native_report::NativeDataFusionPlanningReport,
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    policy::GpuFallbackPolicy,
    session::CudfSessionStateBuilderExt,
};
use datafusion_nexus_bench::{
    ClickbenchDatasetMetadata, TpcdsDatasetMetadata, TpchDatasetMetadata,
    clickbench_datafusion_planning_workload_items_with_optimizer_config, clickbench_native_optimizer_config,
    clickbench_report_session_config, prepare_clickbench_report_dataset, prepare_tpcds_report_dataset,
    prepare_tpch_report_dataset, tpcds_datafusion_planning_workload_items_with_optimizer_config,
    tpcds_native_optimizer_config, tpcds_report_session_config,
    tpch_datafusion_planning_workload_items_with_optimizer_config, tpch_native_optimizer_config,
    tpch_report_session_config,
};
use nexus_query_engine::mode::NativePipelineExecutionMode;

const CUDF_NATIVE_DATAFUSION_PLANNING_WORKLOAD_ENV: &str = "CUDF_NATIVE_DATAFUSION_PLANNING_WORKLOAD";
const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";
const CUDF_TPCH_PLANNING_PARQUET_PUSHDOWN_FILTERS_ENV: &str = "CUDF_TPCH_PLANNING_PARQUET_PUSHDOWN_FILTERS";
const CUDF_TPCDS_PLANNING_PARQUET_PUSHDOWN_FILTERS_ENV: &str = "CUDF_TPCDS_PLANNING_PARQUET_PUSHDOWN_FILTERS";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-datafusion-planning");
    let workload =
        std::env::var(CUDF_NATIVE_DATAFUSION_PLANNING_WORKLOAD_ENV).unwrap_or_else(|_| "synthetic".to_string());
    let report = match workload.as_str() {
        "synthetic" => synthetic_report(&dir).await,
        "tpch" => {
            let TpchPlanningReport { report, metadata } = tpch_report(&dir).await.unwrap();
            report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata())
        },
        "tpcds" => {
            let TpcdsPlanningReport { report, metadata } = tpcds_report(&dir).await.unwrap();
            report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata())
        },
        "clickbench" => {
            let ClickbenchPlanningReport { report, metadata } = clickbench_report().await.unwrap();
            report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata())
        },
        value => {
            panic!(
                "unsupported {CUDF_NATIVE_DATAFUSION_PLANNING_WORKLOAD_ENV}={value}; expected synthetic, tpch, tpcds, or clickbench"
            )
        },
    };
    print!("{}", report.to_tsv());

    fs::remove_dir_all(dir).unwrap();
}

async fn synthetic_report(dir: &std::path::Path) -> NativeDataFusionPlanningReport {
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let workload = [
        ("sum_session", SQL_SUM_QUERY),
        ("not_session", SQL_NOT_QUERY),
        ("cast_session", SQL_CAST_QUERY),
        ("avg_session", SQL_AVG_QUERY),
        ("filtered_session", SQL_FILTERED_QUERY),
        ("modulo_session", SQL_MODULO_QUERY),
    ];
    let mut reports = Vec::with_capacity(workload.len());
    for (query_id, sql) in workload {
        reports.push(native_datafusion_planning_report(query_id, &path, sql).await);
    }
    for (query_id, sql) in [
        ("distributed_avg_session", SQL_AVG_QUERY),
        ("distributed_filtered_avg_session", SQL_FILTERED_AVG_QUERY),
        ("distributed_distinct_avg_session", SQL_DISTINCT_AVG_QUERY),
    ] {
        reports.push(native_datafusion_planning_report_with_session(query_id, &path, sql, SessionConfig::new()).await);
    }

    let rows = reports.into_iter().flat_map(|report| report.rows).collect();
    NativeDataFusionPlanningReport { rows, workload_dataset: None }
}

async fn tpch_report(dir: &std::path::Path) -> datafusion_nexus_bench::TpchWorkloadResult<TpchPlanningReport> {
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let optimizer_config =
        tpch_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
    let mut session_config = tpch_report_session_config();
    if let Some(pushdown_filters) = optional_env_bool(CUDF_TPCH_PLANNING_PARQUET_PUSHDOWN_FILTERS_ENV) {
        session_config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    }
    let items =
        tpch_datafusion_planning_workload_items_with_optimizer_config(&root, session_config, optimizer_config).await?;
    let report = NativeDataFusionPlanningReport::collect(items);
    Ok(TpchPlanningReport { report, metadata })
}

struct TpchPlanningReport {
    report: NativeDataFusionPlanningReport,
    metadata: TpchDatasetMetadata,
}

async fn tpcds_report(dir: &std::path::Path) -> datafusion_nexus_bench::TpcdsWorkloadResult<TpcdsPlanningReport> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let optimizer_config =
        tpcds_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
    let mut session_config = tpcds_report_session_config();
    if let Some(pushdown_filters) = optional_env_bool(CUDF_TPCDS_PLANNING_PARQUET_PUSHDOWN_FILTERS_ENV) {
        session_config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    }
    let items =
        tpcds_datafusion_planning_workload_items_with_optimizer_config(&root, session_config, optimizer_config).await?;
    let report = NativeDataFusionPlanningReport::collect(items);
    Ok(TpcdsPlanningReport { report, metadata })
}

struct TpcdsPlanningReport {
    report: NativeDataFusionPlanningReport,
    metadata: TpcdsDatasetMetadata,
}

async fn clickbench_report() -> datafusion_nexus_bench::ClickbenchWorkloadResult<ClickbenchPlanningReport> {
    let (root, metadata) = prepare_clickbench_report_dataset()?.ok_or_else(|| {
        datafusion_nexus_bench::ClickbenchWorkloadError::InvalidDataset(
            "no ClickBench fixture found; set CUDF_CLICKBENCH_PARQUET_DIR or stage \
             fixture/data/clickbench/{hits.parquet, partitioned/} before running this report"
                .to_string(),
        )
    })?;
    let optimizer_config = clickbench_native_optimizer_config()
        .with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
    let items = clickbench_datafusion_planning_workload_items_with_optimizer_config(
        &root,
        clickbench_report_session_config(),
        optimizer_config,
    )
    .await?;
    let report = NativeDataFusionPlanningReport::collect(items);
    Ok(ClickbenchPlanningReport { report, metadata })
}

struct ClickbenchPlanningReport {
    report: NativeDataFusionPlanningReport,
    metadata: ClickbenchDatasetMetadata,
}

async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    native_datafusion_planning_report_with_session(query_id, path, sql, direct_friendly_sql_session_config()).await
}

async fn native_datafusion_planning_report_with_session(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
            .with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env()),
    );
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    NativeDataFusionPlanningReport::from_retained_rule(query_id, plan, native_rule.as_ref())
}

fn native_pipeline_execution_mode_from_env() -> NativePipelineExecutionMode {
    let mode = std::env::var(CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV)
        .unwrap_or_else(|_| NativePipelineExecutionMode::StrictBounded.as_str().to_string());
    mode.parse::<NativePipelineExecutionMode>().unwrap_or_else(|error| {
        panic!(
            "unsupported {CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV}={}; expected strict_bounded or adaptive",
            error.value()
        )
    })
}

fn optional_env_bool(name: &'static str) -> Option<bool> {
    std::env::var_os(name).map(|value| match value.to_string_lossy().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        value => panic!("unsupported {name}={value}; expected true/false"),
    })
}

fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 15, 20, 25, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

const SQL_SUM_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_NOT_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE NOT (payload < 10)
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_CAST_QUERY: &str = "
SELECT key, SUM(CAST(key AS BIGINT)) AS sum_cast
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_cast DESC, key ASC
LIMIT 2
";

const SQL_AVG_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_FILTERED_QUERY: &str = "
SELECT key, SUM(m00) FILTER (WHERE payload >= 10) AS sum_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_m00 DESC, key ASC
LIMIT 2
";

const SQL_FILTERED_AVG_QUERY: &str = "
SELECT key, AVG(m00) FILTER (WHERE payload >= 10) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_DISTINCT_AVG_QUERY: &str = "
SELECT key, AVG(DISTINCT m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_MODULO_QUERY: &str = "
SELECT key, SUM(payload % 2) AS modulo_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY modulo_payload DESC, key ASC
LIMIT 2
";
