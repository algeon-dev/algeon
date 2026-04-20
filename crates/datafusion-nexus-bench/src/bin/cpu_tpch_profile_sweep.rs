use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{record_batch::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    common::{Result as DataFusionResult, utils::memory::get_record_batch_memory_size},
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus_workloads::tpch::{
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES, TPCH_TABLES,
    TpchDatasetMetadata, TpchQuery, TpchWorkloadError, discover_tpch_table_layout, prepare_tpch_report_dataset,
    register_tpch_table_layout,
};

const TSV_COLUMNS: &[&str] =
    &["query_id", "cpu_wall_ms", "output_batches", "output_rows", "output_bytes", "output_fingerprint", "failure"];

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = CpuTpchConfig::from_env()?;
    let work_dir = unique_temp_dir("datafusion-nexus-cpu-tpch-profile-sweep");
    let (_root, metadata) = prepare_tpch_report_dataset(&work_dir)
        .map_err(|error| datafusion::common::DataFusionError::External(Box::new(error)))?;

    print_metadata(&metadata, &config);
    println!("{}", TSV_COLUMNS.join("\t"));
    for query in config.selected_queries() {
        match run_cpu_query(&metadata, query, &config).await {
            Ok(row) => println!("{}", row.to_tsv(query.id)),
            Err(error) => {
                println!("{}\tnone\tnone\tnone\tnone\tnone\t{}", query.id, stable_tsv_field(&error.to_string()))
            },
        }
    }

    fs::remove_dir_all(work_dir).ok();
    Ok(())
}

#[derive(Debug, Clone)]
struct CpuTpchConfig {
    query_ids: Vec<String>,
    target_partitions: usize,
    batch_size: usize,
    repartition_file_scans: bool,
    repartition_aggregations: bool,
    repartition_joins: bool,
    repartition_sorts: bool,
    dynamic_filter_pushdown: bool,
    parquet_pushdown_filters: bool,
}

impl CpuTpchConfig {
    fn from_env() -> DataFusionResult<Self> {
        Ok(Self {
            query_ids: query_ids_from_env("CUDF_TPCH_CPU_QUERY_IDS")?,
            target_partitions: env_usize("CUDF_TPCH_CPU_TARGET_PARTITIONS", default_parallelism())?,
            batch_size: env_usize("CUDF_TPCH_CPU_BATCH_SIZE", 8192)?,
            repartition_file_scans: env_bool("CUDF_TPCH_CPU_REPARTITION_FILE_SCANS", true)?,
            repartition_aggregations: env_bool("CUDF_TPCH_CPU_REPARTITION_AGGREGATIONS", true)?,
            repartition_joins: env_bool("CUDF_TPCH_CPU_REPARTITION_JOINS", true)?,
            repartition_sorts: env_bool("CUDF_TPCH_CPU_REPARTITION_SORTS", true)?,
            dynamic_filter_pushdown: env_bool("CUDF_TPCH_CPU_DYNAMIC_FILTER_PUSHDOWN", true)?,
            parquet_pushdown_filters: env_bool("CUDF_TPCH_CPU_PARQUET_PUSHDOWN_FILTERS", false)?,
        })
    }

    fn session_config(&self) -> SessionConfig {
        let mut config = SessionConfig::new()
            .with_target_partitions(self.target_partitions)
            .with_batch_size(self.batch_size)
            .with_repartition_file_scans(self.repartition_file_scans)
            .with_repartition_aggregations(self.repartition_aggregations)
            .with_repartition_joins(self.repartition_joins)
            .with_repartition_sorts(self.repartition_sorts);
        config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = self.dynamic_filter_pushdown;
        config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = self.dynamic_filter_pushdown;
        config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = self.dynamic_filter_pushdown;
        config.options_mut().optimizer.enable_dynamic_filter_pushdown = self.dynamic_filter_pushdown;
        config.options_mut().execution.parquet.pushdown_filters = self.parquet_pushdown_filters;
        config
    }

    fn selected_queries(&self) -> Vec<&'static TpchQuery> {
        self.query_ids
            .iter()
            .map(|query_id| {
                TPCH_QUERIES
                    .iter()
                    .find(|query| query.id == query_id)
                    .expect("validated CPU benchmark config references only known TPC-H query ids")
            })
            .collect()
    }
}

#[derive(Debug)]
struct CpuQueryRow {
    wall_ms: f64,
    output_batches: usize,
    output_rows: usize,
    output_bytes: usize,
    output_fingerprint: String,
}

impl CpuQueryRow {
    fn to_tsv(&self, query_id: &str) -> String {
        let fields = [
            query_id.to_string(),
            format!("{:.3}", self.wall_ms),
            self.output_batches.to_string(),
            self.output_rows.to_string(),
            self.output_bytes.to_string(),
            stable_tsv_field(&self.output_fingerprint),
            "none".to_string(),
        ];
        debug_assert_eq!(fields.len(), TSV_COLUMNS.len());
        fields.join("\t")
    }
}

async fn run_cpu_query(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &CpuTpchConfig,
) -> DataFusionResult<CpuQueryRow> {
    let ctx = SessionContext::new_with_config(config.session_config());
    register_tpch_tables(&ctx, &metadata.root).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let schema = plan.schema();
    let started = Instant::now();
    let batches = collect_cpu_plan_once(plan).await?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output_fingerprint = fingerprint_batches(schema.as_ref(), &batches)?;
    Ok(CpuQueryRow {
        wall_ms,
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
        output_fingerprint,
    })
}

async fn collect_cpu_plan_once(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<Vec<RecordBatch>> {
    collect(plan, Arc::new(TaskContext::default())).await
}

async fn register_tpch_tables(ctx: &SessionContext, root: &Path) -> DataFusionResult<()> {
    for table in TPCH_TABLES {
        let layout = discover_tpch_table_layout(root, table).map_err(workload_error)?;
        register_tpch_table_layout(ctx, table, &layout).await.map_err(workload_error)?;
    }
    Ok(())
}

fn workload_error(error: TpchWorkloadError) -> datafusion::common::DataFusionError {
    datafusion::common::DataFusionError::External(Box::new(error))
}

fn print_metadata(metadata: &TpchDatasetMetadata, config: &CpuTpchConfig) {
    println!("# tpch_dataset_root={}", metadata.root.display());
    println!("# tpch_dataset_scale_factor={}", metadata.scale_factor);
    println!("# tpch_dataset_provenance={}", metadata.provenance);
    println!("# tpch_dataset_generation_command={}", metadata.generation_command);
    println!("# tpch_dataset_table_count={}", metadata.table_count);
    println!("# tpch_dataset_file_count={}", metadata.file_count);
    println!("# tpch_dataset_total_rows={}", metadata.total_rows);
    println!("# tpch_dataset_total_bytes={}", metadata.total_bytes);
    println!("# tpch_dataset_metadata_fingerprint={}", metadata.metadata_fingerprint);
    println!("# tpch_dataset_content_fingerprint={}", metadata.content_fingerprint.as_deref().unwrap_or("none"));
    println!("# tpch_dataset_table_row_counts={}", metadata.table_row_counts());
    println!("# tpch_dataset_table_byte_sizes={}", metadata.table_byte_sizes());
    println!("# tpch_dataset_table_row_group_counts={}", metadata.table_row_group_counts());
    println!("# cpu_profile_query_ids={}", config.query_ids.join(","));
    println!("# cpu_profile_target_partitions={}", config.target_partitions);
    println!("# cpu_profile_batch_size={}", config.batch_size);
    println!("# cpu_profile_repartition_file_scans={}", config.repartition_file_scans);
    println!("# cpu_profile_repartition_aggregations={}", config.repartition_aggregations);
    println!("# cpu_profile_repartition_joins={}", config.repartition_joins);
    println!("# cpu_profile_repartition_sorts={}", config.repartition_sorts);
    println!("# cpu_profile_dynamic_filter_pushdown={}", config.dynamic_filter_pushdown);
    println!("# cpu_profile_parquet_pushdown_filters={}", config.parquet_pushdown_filters);
    println!("# cpu_profile_cargo_profile={}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("# cpu_profile_wall_time_fields=single_measured_iteration");
    println!(
        "# env_{}={}",
        CUDF_TPCH_PARQUET_DIR_ENV,
        std::env::var(CUDF_TPCH_PARQUET_DIR_ENV).unwrap_or_else(|_| "unset".to_string())
    );
    println!(
        "# env_{}={}",
        CUDF_TPCH_SCALE_FACTOR_ENV,
        std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unset".to_string())
    );
    println!(
        "# env_{}={}",
        CUDF_TPCH_GENERATION_COMMAND_ENV,
        std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unset".to_string())
    );
}

fn fingerprint_batches(schema: &arrow::datatypes::Schema, batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hash = Fnv1a64::new();
    for field in schema.fields() {
        hash.update(field.name().as_bytes());
        hash.update(&[0xff]);
    }
    for batch in batches {
        for row in 0..batch.num_rows() {
            for column in 0..batch.num_columns() {
                let value = array_value_to_string(batch.column(column).as_ref(), row)?;
                hash.update(value.as_bytes());
                hash.update(&[0x1f]);
            }
            hash.update(&[0x1e]);
        }
    }
    Ok(format!("output-fnv1a64:{:016x}", hash.finish()))
}

struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

fn query_ids_from_env(name: &'static str) -> DataFusionResult<Vec<String>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect());
    };
    let query_ids = value
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|query_id| !query_id.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if query_ids.is_empty() {
        return Err(datafusion::common::DataFusionError::Plan(format!("{name} must name at least one query id")));
    }
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in &query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(datafusion::common::DataFusionError::Plan(format!("unknown TPC-H query id {query_id}")));
        }
    }
    Ok(query_ids)
}

fn env_usize(name: &'static str, default: usize) -> DataFusionResult<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| datafusion::common::DataFusionError::Plan(format!("invalid {name}: {error}")))
}

fn env_bool(name: &'static str, default: bool) -> DataFusionResult<bool> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    match value.to_string_lossy().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => {
            Err(datafusion::common::DataFusionError::Plan(format!("invalid {name}: {other}; expected true or false")))
        },
    }
}

fn default_parallelism() -> usize {
    std::thread::available_parallelism().map(usize::from).unwrap_or(1)
}

fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_profile_tsv_row_contract_matches_columns() {
        let row = CpuQueryRow {
            wall_ms: 1.25,
            output_batches: 2,
            output_rows: 3,
            output_bytes: 4,
            output_fingerprint: "finger\tprint".to_string(),
        };
        assert_eq!(TSV_COLUMNS.first(), Some(&"query_id"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"failure"));
        assert_eq!(row.to_tsv("q01").split('\t').count(), TSV_COLUMNS.len());
    }

    #[tokio::test]
    async fn register_tpch_tables_supports_sibling_split_layout() {
        let dir = unique_temp_dir("cpu-tpch-profile-sibling-split-test");
        datafusion_nexus_workloads::tpch::write_tiny_tpch_parquet_dataset(&dir).unwrap();
        fs::rename(dir.join("lineitem.parquet"), dir.join("lineitem_0000.parquet")).unwrap();

        let ctx = SessionContext::new();
        register_tpch_tables(&ctx, &dir).await.unwrap();
        let batches = ctx.sql("SELECT COUNT(*) AS c FROM lineitem").await.unwrap().collect().await.unwrap();

        assert_eq!(batches[0].num_rows(), 1);
        fs::remove_dir_all(dir).ok();
    }
}
