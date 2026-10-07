//! Shared cuGraph SQL fixtures and explicit Python-oracle acceptance entry.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::{collect, displayable},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use serde_json::Value;

use crate::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    cugraph_sql::CugraphSqlConfig,
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};

/// Builds a device-0 backend with native-preferred SQL execution.
pub fn backend_cugraph_sql_context(builder: SessionStateBuilder) -> (AlgeonGpuBackend, SessionContext) {
    backend_cugraph_sql_context_with_mode(builder, AlgeonGpuExecutionMode::NativePreferred)
}

/// Builds a device-0 backend with the selected SQL execution mode.
pub fn backend_cugraph_sql_context_with_mode(
    builder: SessionStateBuilder,
    mode: AlgeonGpuExecutionMode,
) -> (AlgeonGpuBackend, SessionContext) {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().with_execution_mode(mode).devices([device]).build().unwrap();
    let state = backend.install_on(builder).unwrap().try_with_cugraph_sql(CugraphSqlConfig::default()).unwrap().build();
    (backend, SessionContext::new_with_state(state))
}

/// Reads vertex and value columns from PageRank result batches.
pub fn pagerank_rows(batches: &[RecordBatch]) -> Vec<PageRankRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let values = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for index in 0..batch.num_rows() {
            rows.push(PageRankRow { vertex: vertices.value(index), value: values.value(index) });
        }
    }
    rows
}

fn python_pagerank_oracle(path: &Path) -> Option<PythonPageRankOracle> {
    let script = r#"
import json
import sys

try:
    import cudf
    import cugraph
except Exception as exc:
    print(f"Python cuGraph oracle unavailable: {exc}", file=sys.stderr)
    sys.exit(77)

df = cudf.read_parquet(sys.argv[1])
df = df[df["active"]][["src", "dst", "weight"]]
graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight", renumber=True)
result = cugraph.pagerank(graph, alpha=0.85, tol=1.0e-5, max_iter=100)
value_col = "pagerank" if "pagerank" in result.columns else [name for name in result.columns if name != "vertex"][0]
result = result.rename(columns={value_col: "value"})
result["vertex"] = result["vertex"].astype("int64")
result["value"] = result["value"].astype("float64")
result = result.sort_values("vertex")
rows = [{"vertex": int(row.vertex), "value": float(row.value)} for row in result.to_pandas().itertuples()]
print(json.dumps({
    "versions": {
        "cudf": getattr(cudf, "__version__", "unknown"),
        "cugraph": getattr(cugraph, "__version__", "unknown"),
    },
    "rows": rows,
}))
"#;
    let output = match Command::new("python3").arg("-c").arg(script).arg(path).output() {
        Ok(output) => output,
        Err(error) => {
            handle_python_oracle_unavailable(format!("Python cuGraph oracle unavailable: {error}"));
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
    }
    assert!(
        output.status.success(),
        "Python cuGraph oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| PageRankRow { vertex: row["vertex"].as_i64().unwrap(), value: row["value"].as_f64().unwrap() })
        .collect();
    Some(PythonPageRankOracle { versions, rows })
}

fn handle_python_oracle_unavailable(message: impl AsRef<str>) -> ! {
    handle_cugraph_python_oracle_unavailable("cuGraph Python oracle", message);
}

/// Writes the weighted active-edge fixture used by PageRank acceptance.
pub fn write_tiny_directed_edges(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1, 2, 3, 0, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 3, 4, 1, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, true, true, true, true, false])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[derive(Debug, Clone)]
pub struct PageRankRow {
    pub vertex: i64,
    pub value: f64,
}

#[derive(Debug)]
struct PythonPageRankOracle {
    versions: String,
    rows: Vec<PageRankRow>,
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Fails a selected acceptance run when its required Python oracle is unavailable.
pub fn handle_cugraph_python_oracle_unavailable(label: &str, message: impl AsRef<str>) -> ! {
    panic!("required {label} is unavailable: {}", message.as_ref().trim());
}

/// Executes the provisioned Python-cuGraph PageRank comparison.
///
/// Calls `before_collect` after physical planning and `after_collect` after
/// collection, allowing the tools caller to check host Arrow transfer counts.
pub async fn run_pagerank_filter_sort_limit_acceptance(before_collect: impl FnOnce(), after_collect: impl FnOnce()) {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-pagerank-compose");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let oracle = python_pagerank_oracle(&path)
        .expect("Python cuGraph oracle is required for the selected PageRank composition acceptance test");
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let (_backend, ctx) =
        backend_cugraph_sql_context(SessionStateBuilder::new().with_config(config).with_default_features());
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM graph_edges)) \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("AlgeonNativeComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );

    before_collect();
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    after_collect();

    let actual = pagerank_rows(&batches);
    let mut expected = oracle.rows.clone();
    expected.retain(|row| row.value > 0.0);
    expected.sort_by(|left, right| {
        right.value.partial_cmp(&left.value).unwrap().then_with(|| left.vertex.cmp(&right.vertex))
    });
    expected.truncate(3);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} composed PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}
