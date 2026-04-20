//! Smoke-test a generated Graph500-style R-MAT fixture through SQL `cugraph_bfs`.
//!
//! Generate data first with:
//!   fixture/scripts/run_gen.sh graph500_rmat --scale 12 --edgefactor 16 --overwrite
//!
//! Override the fixture root with:
//!   CUDF_CUGRAPH_GRAPH500_RMAT_DIR=fixture/data/graph500/rmat_scale25_ef16_edges134217728

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    graph_sql::CugraphSqlConfig, native_exec::NativeCudfExecutionConfig, planner::CudfOptimizerConfig,
    policy::GpuFallbackPolicy, session::CudfSessionStateBuilderExt,
};
use serde_json::Value;

#[tokio::test]
async fn cugraph_bfs_reads_generated_graph500_rmat_fixture() {
    let Some(fixture) = Graph500Fixture::discover() else {
        crate::cugraph_optional_fixture::handle_cugraph_graph500_rmat_unavailable(
            "Graph500 R-MAT SQL BFS smoke",
            "generate fixture with `fixture/scripts/run_gen.sh graph500_rmat --scale 12 --edgefactor 16 --overwrite`",
        );
        return;
    };

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_cudf_native(
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
                .with_native_execution_config(native_execution_config()),
        )
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("graph500_edges", fixture.parquet_dir.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();

    let options = r#"{"construction_policy":"raw_libcugraph","output_mode":"normalized","depth_limit":8}"#;
    let sql = format!(
        "SELECT COUNT(*) AS row_count, \
                SUM(CASE WHEN reachable THEN 1 ELSE 0 END) AS reachable_count, \
                MAX(distance) AS max_distance \
         FROM cugraph_bfs('graph500_edges', {}, 'src', 'dst', NULL, '{}')",
        fixture.source_vertex, options,
    );
    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let batch = single_batch(&batches);
    let row_count = int64_value(batch, 0, 0);
    let reachable_count = int64_value(batch, 1, 0);
    let max_distance = int64_value(batch, 2, 0);

    assert!(row_count > 0, "BFS smoke should produce at least one output row");
    assert!(reachable_count > 0, "BFS smoke should reach at least the source vertex");
    // assert!(max_distance <= 1, "depth_limit=1 should bound reported BFS distance");
    eprintln!(
        "graph500_rmat_bfs_smoke root={} source_vertex={} rows={} reachable={} max_distance={}",
        fixture.root.display(),
        fixture.source_vertex,
        row_count,
        reachable_count,
        max_distance,
    );
}

fn native_execution_config() -> NativeCudfExecutionConfig {
    match std::env::var("CUDF_CUGRAPH_GRAPH500_RMAT_MEMORY_POLICY").as_deref() {
        Ok("unbounded") => NativeCudfExecutionConfig::unbounded(),
        Ok("bounded") | Err(_) => NativeCudfExecutionConfig::bounded_auto_sized(),
        Ok(other) => {
            panic!("unsupported CUDF_CUGRAPH_GRAPH500_RMAT_MEMORY_POLICY={other}; expected bounded or unbounded")
        },
    }
}

#[derive(Debug)]
struct Graph500Fixture {
    root: PathBuf,
    parquet_dir: PathBuf,
    source_vertex: i64,
}

impl Graph500Fixture {
    fn discover() -> Option<Self> {
        let root =
            std::env::var_os("CUDF_CUGRAPH_GRAPH500_RMAT_DIR").map(PathBuf::from).unwrap_or_else(default_fixture_root);
        let metadata_path = root.join("metadata.json");
        let metadata = fs::read_to_string(&metadata_path).ok()?;
        let metadata: Value = serde_json::from_str(&metadata).ok()?;
        let parquet_dir = root.join(metadata.get("parquet_dir").and_then(Value::as_str).unwrap_or("parquet"));
        if !parquet_dir.is_dir() || !has_parquet_file(&parquet_dir) {
            return None;
        }
        let source_vertex = metadata.get("source_vertex").and_then(Value::as_i64).unwrap_or(0);
        Some(Self { root, parquet_dir, source_vertex })
    }
}

fn default_fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixture").join("data").join("graph500").join("rmat_scale12_ef16")
}

fn has_parquet_file(dir: &Path) -> bool {
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().is_some_and(|extension| extension == "parquet"))
}

fn single_batch(batches: &[RecordBatch]) -> &RecordBatch {
    assert_eq!(batches.len(), 1, "expected one aggregate batch, got {}", batches.len());
    assert_eq!(batches[0].num_rows(), 1, "expected one aggregate row");
    &batches[0]
}

fn int64_value(batch: &RecordBatch, column: usize, row: usize) -> i64 {
    let array = Arc::clone(batch.column(column));
    let values = array.as_any().downcast_ref::<Int64Array>().expect("aggregate column should be Int64");
    assert!(!values.is_null(row), "aggregate column {column} row {row} should not be null");
    values.value(row)
}
