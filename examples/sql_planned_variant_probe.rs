use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::SourceDiagnostics;
use datafusion::{
    common::Result as DataFusionResult,
    parquet::arrow::ArrowWriter,
    physical_plan::{ExecutionPlan, displayable},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{CandidateOutcome, lower_physical_plan_to_native};

const SQL_BENCHMARK_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let dir = unique_temp_dir("datafusion-cudf-sql-planned-probe")?;
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path)?;

    println!("sql_planned_variant_probe");
    println!("parquet_path={}", path.display());
    println!("sql={}", SQL_BENCHMARK_QUERY.split_whitespace().collect::<Vec<_>>().join(" "));

    run_variant("default_sql_session", SessionConfig::new(), &path).await?;
    run_variant("direct_friendly_sql_session", direct_friendly_sql_session_config(), &path).await?;

    fs::remove_dir_all(dir)?;
    Ok(())
}

async fn run_variant(name: &str, config: SessionConfig, path: &Path) -> DataFusionResult<()> {
    let plan = sql_benchmark_plan(path, config).await?;
    let physical_plan = displayable(plan.as_ref()).one_line().to_string();
    let physical_plan_tree = displayable(plan.as_ref()).indent(true).to_string();
    let lowering = lower_physical_plan_to_native(plan);
    let rejected_reasons = lowering
        .report
        .outcomes()
        .iter()
        .filter_map(|outcome| match outcome {
            CandidateOutcome::Rejected { reason, .. } => Some(reason.code()),
            CandidateOutcome::Rewritten { .. } => None,
        })
        .collect::<Vec<_>>();

    println!();
    println!("variant={name}");
    println!("physical_plan={physical_plan}");
    println!("physical_plan_tree=\n{physical_plan_tree}");
    println!("candidate_count={}", lowering.candidate_count());
    println!("rewrite_count={}", lowering.rewritten_count());
    println!("reject_count={}", lowering.rejected_count());
    println!("rejected_reasons={}", empty_as_none(&rejected_reasons));
    println!("source_diagnostic_count={}", lowering.source_diagnostics.len());
    if lowering.source_diagnostics.is_empty() {
        println!("source_diagnostic=none");
    }
    for (index, diagnostics) in lowering.source_diagnostics.iter().enumerate() {
        println!("source_diagnostic[{index}]={}", format_source_diagnostics(diagnostics));
    }

    Ok(())
}

async fn sql_benchmark_plan(path: &Path, config: SessionConfig) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet(
        "bench",
        path.to_str().expect("probe path should be valid UTF-8"),
        ParquetReadOptions::default(),
    )
    .await?;
    let df = ctx.sql(SQL_BENCHMARK_QUERY).await?;
    df.create_physical_plan().await
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

fn format_source_diagnostics(diagnostics: &SourceDiagnostics) -> String {
    let rejected_features = diagnostics.rejected_features.iter().map(|feature| feature.as_str()).collect::<Vec<_>>();
    format!(
        "format={:?}, direct_device_read={}, file_count={}, ranged_file_count={}, known_bytes={}, selected_row_groups={}, predicate={}, rejected_features={}",
        diagnostics.format,
        diagnostics.direct_device_read,
        diagnostics.file_count,
        diagnostics.ranged_file_count,
        diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
        diagnostics.selected_row_groups.map_or_else(|| "unknown".to_string(), |count| count.to_string()),
        diagnostics.predicate_disposition.as_str(),
        empty_as_none(&rejected_features),
    )
}

fn empty_as_none(values: &[&str]) -> String {
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

fn write_numeric_parquet_file(path: &Path) -> DataFusionResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn unique_temp_dir(prefix: &str) -> DataFusionResult<std::path::PathBuf> {
    let nonce =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock should be after UNIX_EPOCH").as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path)?;
    Ok(path)
}
