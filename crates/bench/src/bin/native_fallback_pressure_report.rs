use std::{
    error::Error,
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    native_exec::AlgeonNativeExecutionConfig,
    native_report::{NativeFallbackPressureReport, NativeFallbackPressureWorkloadItem},
};
use arrow::{
    array::{Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        tpcds::{tpcds_fallback_pressure_workload_items, tpcds_report_session_config},
        tpch::{
            tpch_fallback_pressure_workload_items, tpch_report_session_config,
            tpch_scalar_fallback_pressure_workload_items,
        },
    },
};
use datafusion::{
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use workloads::{
    tpcds::{TpcdsDatasetMetadata, WorkloadResult, prepare_tpcds_report_dataset},
    tpch::{TpchDatasetMetadata, prepare_tpch_report_dataset},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = unique_temp_dir("algeon-datafusion-native-fallback-pressure");
    let workload =
        std::env::var("ALGEON_NATIVE_FALLBACK_PRESSURE_WORKLOAD").unwrap_or_else(|_| "synthetic".to_string());

    let (items, workload_dataset) = match workload.as_str() {
        "synthetic" => (synthetic_workload_items(&dir).await, None),
        "tpch" => {
            let TpchWorkload { items, metadata } = tpch_workload_items(&dir).await?;
            (items, Some(metadata.to_workload_dataset_metadata()))
        },
        "tpch_scalar" => {
            let TpchWorkload { items, metadata } = tpch_scalar_workload_items(&dir).await?;
            (items, Some(metadata.to_workload_dataset_metadata()))
        },
        "tpcds" => {
            let TpcdsWorkload { items, metadata } = tpcds_workload_items(&dir).await?;
            (items, Some(metadata.to_workload_dataset_metadata()))
        },
        value => {
            return Err(format!(
                "unsupported ALGEON_NATIVE_FALLBACK_PRESSURE_WORKLOAD={value}; expected synthetic, tpch, tpch_scalar, or tpcds"
            )
            .into());
        },
    };

    let execution_config = AlgeonNativeExecutionConfig::default();
    let mut report = NativeFallbackPressureReport::collect_with_native_execution_config(items, &execution_config);
    if let Some(metadata) = workload_dataset {
        report = report.with_workload_dataset_metadata(metadata);
    }
    print!("{}", report.to_tsv());

    fs::remove_dir_all(dir)?;
    Ok(())
}

async fn synthetic_workload_items(dir: &std::path::Path) -> Vec<NativeFallbackPressureWorkloadItem> {
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let workload = [
        ("sum_rewrite", SQL_SUM_QUERY),
        ("not_rewrite", SQL_NOT_QUERY),
        ("cast_rewrite", SQL_CAST_QUERY),
        ("avg_rewrite", SQL_AVG_QUERY),
        ("filtered_reject", SQL_FILTERED_QUERY),
        ("modulo_rewrite", SQL_MODULO_QUERY),
    ];
    let mut items = Vec::with_capacity(workload.len());
    for (query_id, sql) in workload {
        items.push(NativeFallbackPressureWorkloadItem::new(
            query_id,
            sql_plan(&path, direct_friendly_sql_session_config(), sql).await,
        ));
    }
    for (query_id, sql) in [
        ("distributed_avg_rewrite", SQL_AVG_QUERY),
        ("distributed_filtered_avg_reject", SQL_FILTERED_AVG_QUERY),
        ("distributed_distinct_avg_rewrite", SQL_DISTINCT_AVG_QUERY),
    ] {
        items.push(NativeFallbackPressureWorkloadItem::new(query_id, sql_plan(&path, SessionConfig::new(), sql).await));
    }

    items
}

async fn tpch_workload_items(dir: &std::path::Path) -> WorkloadResult<TpchWorkload> {
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let items = tpch_fallback_pressure_workload_items(&root, tpch_report_session_config()).await?;
    Ok(TpchWorkload { items, metadata })
}

async fn tpch_scalar_workload_items(dir: &std::path::Path) -> WorkloadResult<TpchWorkload> {
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let items = tpch_scalar_fallback_pressure_workload_items(&root, tpch_report_session_config()).await?;
    Ok(TpchWorkload { items, metadata })
}

struct TpchWorkload {
    items: Vec<NativeFallbackPressureWorkloadItem>,
    metadata: TpchDatasetMetadata,
}

async fn tpcds_workload_items(dir: &std::path::Path) -> WorkloadResult<TpcdsWorkload> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let items = tpcds_fallback_pressure_workload_items(&root, tpcds_report_session_config()).await?;
    Ok(TpcdsWorkload { items, metadata })
}

struct TpcdsWorkload {
    items: Vec<NativeFallbackPressureWorkloadItem>,
    metadata: TpcdsDatasetMetadata,
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
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
