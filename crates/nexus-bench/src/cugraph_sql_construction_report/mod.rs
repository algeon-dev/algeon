use std::{
    error::Error,
    fs::{self, File},
    io,
    path::Path,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::{ExecutionPlan, collect, displayable, metrics::MetricsSet},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    cugraph_sql::CugraphSqlConfig,
    native_exec::NexusNativeExecutionConfig,
    native_report::NativeDataFusionExecutionMetrics,
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};

const ROWS_ENV: &str = "NEXUS_CUGRAPH_CONSTRUCTION_ROWS";
const REMOVED_MEMORY_POLICY_ENV: &str = "NEXUS_CUGRAPH_CONSTRUCTION_MEMORY_POLICY";
const DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_CUGRAPH_CONSTRUCTION_DEVICE_BUDGET_BYTES";
const MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_CUGRAPH_CONSTRUCTION_MAX_SOURCE_CHUNK_BYTES";

const COLUMNS: &[&str] = &[
    "variant",
    "sql_construction_policy",
    "edge_normalize_inserted",
    "runtime_construction_policy",
    "runtime_outcome",
    "input_rows",
    "graph_input_rows",
    "graph_output_rows",
    "output_rows",
    "output_bytes",
    "wall_ms",
    "graph_elapsed_ms",
    "graph_output_device_bytes",
    "gpu_resident_input_handoffs",
    "host_to_device_uploads",
    "device_to_host_materializations",
    "failure",
];

pub async fn run() -> Result<(), Box<dyn Error>> {
    reject_removed_memory_policy_env()?;
    let rows = env_usize(ROWS_ENV, 4096);
    let dir = unique_temp_dir("datafusion-nexus-cugraph-construction-report");
    let result = run_report(&dir, rows).await;
    fs::remove_dir_all(&dir).ok();
    result
}

async fn run_report(dir: &Path, rows: usize) -> Result<(), Box<dyn Error>> {
    let path = dir.join("edges.parquet");
    write_synthetic_edges(&path, rows)?;

    let config = direct_friendly_session_config();
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(
        NexusNativeOptimizerConfig::default().with_native_execution_config(native_execution_config()),
    ));
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_nexus_native_rule(native_rule)
        .try_with_cugraph_sql(CugraphSqlConfig::default())?
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("synthetic_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await?;
    register_views(&ctx).await?;

    println!(
        "# rows={rows}\tmemory_policy=bounded\tdevice_budget_bytes={}\tmax_source_chunk_bytes={}",
        env_u64(DEVICE_BUDGET_BYTES_ENV, NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,),
        env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,),
    );
    println!("{}", COLUMNS.join("\t"));

    for variant in variants() {
        println!("{}", collect_variant(&ctx, rows, variant).await.to_tsv());
    }
    Ok(())
}

async fn register_views(ctx: &SessionContext) -> datafusion::common::Result<()> {
    ctx.sql("CREATE VIEW graph_edges AS SELECT src, dst FROM synthetic_edges").await?.collect().await?;
    ctx.sql(
        "CREATE VIEW canonical_undirected_edges AS \
         SELECT src, dst \
         FROM ( \
           SELECT src, dst FROM graph_edges \
           UNION ALL \
           SELECT dst AS src, src AS dst FROM graph_edges \
         ) e \
         GROUP BY src, dst",
    )
    .await?
    .collect()
    .await?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Variant {
    label: &'static str,
    sql_construction_policy: &'static str,
    expected_edge_normalize_inserted: bool,
    sql: &'static str,
}

fn variants() -> [Variant; 3] {
    [
        Variant {
            label: "default_edge_normalize",
            sql_construction_policy: "python_cugraph",
            expected_edge_normalize_inserted: true,
            sql: "SELECT vertex, partition \
                  FROM cugraph_louvain('graph_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
                  ORDER BY vertex",
        },
        Variant {
            label: "precanonical_default",
            sql_construction_policy: "python_cugraph",
            expected_edge_normalize_inserted: false,
            sql: "SELECT vertex, partition \
                  FROM cugraph_louvain('canonical_undirected_edges', 'src', 'dst', NULL, '{\"directed\":false}') \
                  ORDER BY vertex",
        },
        Variant {
            label: "precanonical_raw",
            sql_construction_policy: "raw_libcugraph",
            expected_edge_normalize_inserted: false,
            sql: "SELECT vertex, partition \
                  FROM cugraph_louvain('canonical_undirected_edges', 'src', 'dst', NULL, \
                                       '{\"directed\":false,\"construction_policy\":\"raw_libcugraph\"}') \
                  ORDER BY vertex",
        },
    ]
}

async fn collect_variant(ctx: &SessionContext, input_rows: usize, variant: Variant) -> ReportRow {
    let planned = match ctx.sql(variant.sql).await {
        Ok(df) => df.create_physical_plan().await,
        Err(error) => {
            return ReportRow::failed(variant, input_rows, format!("sql_planning_failed: {error}"));
        },
    };
    let plan = match planned {
        Ok(plan) => plan,
        Err(error) => {
            return ReportRow::failed(variant, input_rows, format!("physical_planning_failed: {error}"));
        },
    };
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    let policy_marker = format!("construction_policy={}", variant.sql_construction_policy);
    let normalize_marker = format!("edge_normalize_inserted={}", variant.expected_edge_normalize_inserted);
    if !plan_display.contains(&policy_marker) || !plan_display.contains(&normalize_marker) {
        return ReportRow::failed(
            variant,
            input_rows,
            format!(
                "plan_marker_mismatch: expected {policy_marker} and {normalize_marker}; plan={}",
                compact_plan_display(&plan_display)
            ),
        );
    }

    let started = Instant::now();
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let wall = started.elapsed();
            let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
            let output_bytes = batches.iter().map(get_record_batch_memory_size).sum();
            let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
            ReportRow::succeeded(variant, input_rows, output_rows, output_bytes, wall.as_nanos(), metrics)
        },
        Err(error) => ReportRow::failed(variant, input_rows, error.to_string()),
    }
}

fn plan_metrics_recursive(plan: &dyn ExecutionPlan) -> MetricsSet {
    let mut metrics = MetricsSet::new();
    append_plan_metrics(&mut metrics, plan);
    metrics
}

fn append_plan_metrics(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
    if let Some(plan_metrics) = plan.metrics() {
        for metric in plan_metrics.iter() {
            metrics.push(Arc::clone(metric));
        }
    }
    for child in plan.children() {
        append_plan_metrics(metrics, child.as_ref());
    }
}

#[derive(Debug)]
struct ReportRow {
    variant: &'static str,
    sql_construction_policy: &'static str,
    edge_normalize_inserted: bool,
    runtime_construction_policy: String,
    runtime_outcome: &'static str,
    input_rows: usize,
    graph_input_rows: Option<usize>,
    graph_output_rows: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    wall_ns: Option<u128>,
    graph_elapsed_ns: Option<usize>,
    graph_output_device_bytes: Option<usize>,
    gpu_resident_input_handoffs: usize,
    host_to_device_uploads: usize,
    device_to_host_materializations: usize,
    failure: Option<String>,
}

impl ReportRow {
    fn succeeded(
        variant: Variant,
        input_rows: usize,
        output_rows: usize,
        output_bytes: usize,
        wall_ns: u128,
        metrics: NativeDataFusionExecutionMetrics,
    ) -> Self {
        let graph = metrics.graph_runtime_rows.first();
        Self {
            variant: variant.label,
            sql_construction_policy: variant.sql_construction_policy,
            edge_normalize_inserted: variant.expected_edge_normalize_inserted,
            runtime_construction_policy: graph
                .map(|row| row.construction_policy.clone())
                .unwrap_or_else(|| "none".to_string()),
            runtime_outcome: "succeeded",
            input_rows,
            graph_input_rows: graph.map(|row| row.input_rows),
            graph_output_rows: graph.map(|row| row.output_rows),
            output_rows: Some(output_rows),
            output_bytes: Some(output_bytes),
            wall_ns: Some(wall_ns),
            graph_elapsed_ns: graph.map(|row| row.elapsed_ns),
            graph_output_device_bytes: graph.and_then(|row| row.output_device_bytes),
            gpu_resident_input_handoffs: metrics.gpu_resident_input_handoffs,
            host_to_device_uploads: metrics.host_to_device_uploads,
            device_to_host_materializations: metrics.device_to_host_materializations,
            failure: None,
        }
    }

    fn failed(variant: Variant, input_rows: usize, failure: String) -> Self {
        Self {
            variant: variant.label,
            sql_construction_policy: variant.sql_construction_policy,
            edge_normalize_inserted: variant.expected_edge_normalize_inserted,
            runtime_construction_policy: "none".to_string(),
            runtime_outcome: "failed",
            input_rows,
            graph_input_rows: None,
            graph_output_rows: None,
            output_rows: None,
            output_bytes: None,
            wall_ns: None,
            graph_elapsed_ns: None,
            graph_output_device_bytes: None,
            gpu_resident_input_handoffs: 0,
            host_to_device_uploads: 0,
            device_to_host_materializations: 0,
            failure: Some(failure),
        }
    }

    fn to_tsv(&self) -> String {
        let fields = [
            self.variant.to_string(),
            self.sql_construction_policy.to_string(),
            self.edge_normalize_inserted.to_string(),
            self.runtime_construction_policy.clone(),
            self.runtime_outcome.to_string(),
            self.input_rows.to_string(),
            format_optional_usize(self.graph_input_rows),
            format_optional_usize(self.graph_output_rows),
            format_optional_usize(self.output_rows),
            format_optional_usize(self.output_bytes),
            format_optional_ms(self.wall_ns),
            format_optional_ms(self.graph_elapsed_ns.map(|value| value as u128)),
            format_optional_usize(self.graph_output_device_bytes),
            self.gpu_resident_input_handoffs.to_string(),
            self.host_to_device_uploads.to_string(),
            self.device_to_host_materializations.to_string(),
            self.failure.as_deref().map(tsv_field).unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), COLUMNS.len());
        fields.join("\t")
    }
}

fn direct_friendly_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn native_execution_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(
        env_u64(DEVICE_BUDGET_BYTES_ENV, NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES),
        env_u64(MAX_SOURCE_CHUNK_BYTES_ENV, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES),
        1024,
        3,
        1,
    )
}

fn reject_removed_memory_policy_env() -> Result<(), Box<dyn Error>> {
    if std::env::var_os(REMOVED_MEMORY_POLICY_ENV).is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{REMOVED_MEMORY_POLICY_ENV} is unavailable; benchmark execution is always bounded"),
        )
        .into());
    }
    Ok(())
}

fn write_synthetic_edges(path: &Path, rows: usize) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let mut src = Vec::with_capacity(rows);
    let mut dst = Vec::with_capacity(rows);
    let mut weight = Vec::with_capacity(rows);
    for index in 0..rows {
        let community = (index % 8) as i64;
        let local = ((index / 8) % 16) as i64;
        let base = community * 100;
        src.push(base + local);
        dst.push(base + ((local + 1) % 16));
        weight.push(1.0 + f64::from((index % 7) as u32) * 0.1);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(src)) as ArrayRef,
            Arc::new(Int64Array::from(dst)) as ArrayRef,
            Arc::new(Float64Array::from(weight)) as ArrayRef,
        ],
    )?;
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn format_optional_ms(value: Option<u128>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{:.3}", value as f64 / 1_000_000.0))
}

fn tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn compact_plan_display(value: &str) -> String {
    let compact = value.replace(['\t', '\n', '\r'], " ");
    const MAX: usize = 512;
    if compact.len() <= MAX { compact } else { format!("{}...", &compact[..MAX]) }
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_tsv_row_contract_matches_columns() {
        let row = ReportRow::failed(variants()[0], 10, "bad\tfailure".to_string());
        assert_eq!(COLUMNS.first(), Some(&"variant"));
        assert_eq!(COLUMNS.last(), Some(&"failure"));
        assert_eq!(row.to_tsv().split('\t').count(), COLUMNS.len());
    }
}
