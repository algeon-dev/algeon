use super::*;

pub(super) async fn assert_sql_schema(ctx: &SessionContext, sql: &str, algorithm: CugraphAlgorithm) {
    let df = ctx.sql(sql).await.unwrap();
    let actual = df.schema().inner();
    let expected = stable_result_schema(algorithm);
    assert_eq!(actual.fields().len(), expected.fields().len());
    for (actual, expected) in actual.fields().iter().zip(expected.fields()) {
        assert_eq!(actual.name(), expected.name());
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(actual.is_nullable(), expected.is_nullable());
    }
}

pub(super) fn assert_error_contains(
    result: datafusion::common::Result<datafusion_nexus::cugraph_sql::CugraphAlgorithmSpec>,
    expected: &str,
) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "expected error containing {expected:?}, got {error:?}",);
}

pub(super) fn schema_fields(algorithm: CugraphAlgorithm) -> Vec<Field> {
    stable_result_schema(algorithm).fields().iter().map(|field| field.as_ref().clone()).collect()
}

// Used by the explicit Python-cuGraph PageRank tools runner.
#[cfg_attr(test, allow(dead_code))]
pub(super) fn find_composed_exec(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) -> Option<&NexusNativeComposedExec> {
    if let Some(composed) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(composed);
    }
    plan.children().into_iter().find_map(|child| find_composed_exec(child.as_ref()))
}

pub(super) fn find_cugraph_algo_exec(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) -> Option<&dyn datafusion::physical_plan::ExecutionPlan> {
    if plan.name() == "CugraphAlgoExec" {
        return Some(plan);
    }
    plan.children().into_iter().find_map(|child| find_cugraph_algo_exec(child.as_ref()))
}

pub(super) fn plan_contains_native_exec(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.children().into_iter().any(|child| plan_contains_native_exec(child.as_ref()))
}

pub(super) fn metric_usize(metrics: &datafusion::physical_plan::metrics::MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

pub(super) fn plan_metric_usize(plan: &dyn datafusion::physical_plan::ExecutionPlan, name: &str) -> usize {
    let own = plan.metrics().as_ref().map(|metrics| metric_usize(metrics, name)).unwrap_or(0);
    own + plan.children().into_iter().map(|child| plan_metric_usize(child.as_ref(), name)).sum::<usize>()
}

pub(super) fn plan_metrics_recursive(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) -> datafusion::physical_plan::metrics::MetricsSet {
    let mut metrics = datafusion::physical_plan::metrics::MetricsSet::new();
    append_plan_metrics(&mut metrics, plan);
    metrics
}

fn append_plan_metrics(
    metrics: &mut datafusion::physical_plan::metrics::MetricsSet,
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) {
    if let Some(plan_metrics) = plan.metrics() {
        for metric in plan_metrics.iter() {
            metrics.push(Arc::clone(metric));
        }
    }
    for child in plan.children() {
        append_plan_metrics(metrics, child.as_ref());
    }
}

pub(super) fn backend_cugraph_sql_context(builder: SessionStateBuilder) -> (NexusGpuBackend, SessionContext) {
    let backend = NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .device_profiles([datafusion_nexus::backend::NexusGpuDeviceProfile::new(0)])
        .build()
        .unwrap();
    let state = backend.install_on_with_cugraph_sql(builder, CugraphSqlConfig::default()).unwrap().build();
    (backend, SessionContext::new_with_state(state))
}

pub(super) fn default_backend_cugraph_sql_context() -> (NexusGpuBackend, SessionContext) {
    backend_cugraph_sql_context(SessionStateBuilder::new_with_default_features())
}

pub(super) async fn drop_session_context_on_blocking_thread(ctx: SessionContext) {
    tokio::task::spawn_blocking(move || drop(ctx)).await.unwrap();
}

#[derive(Debug, Clone)]
pub(super) struct PageRankRow {
    pub(super) vertex: i64,
    pub(super) value: f64,
}

// Used by the explicit Python-cuGraph PageRank tools runner.
#[cfg_attr(test, allow(dead_code))]
#[derive(Debug)]
pub(super) struct PythonPageRankOracle {
    pub(super) versions: String,
    pub(super) rows: Vec<PageRankRow>,
}

pub(super) fn pagerank_rows(batches: &[RecordBatch]) -> Vec<PageRankRow> {
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

pub(super) fn pagerank_values(batches: &[RecordBatch]) -> Vec<f64> {
    batches
        .iter()
        .flat_map(|batch| batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap().values().iter().copied())
        .collect()
}

// Used by the explicit Python-cuGraph PageRank tools runner.
#[cfg_attr(test, allow(dead_code))]
pub(super) fn python_pagerank_oracle(path: &Path) -> Option<PythonPageRankOracle> {
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
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let versions = value["versions"].to_string();
    let rows = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| PageRankRow { vertex: row["vertex"].as_i64().unwrap(), value: row["value"].as_f64().unwrap() })
        .collect();
    Some(PythonPageRankOracle { versions, rows })
}

// Used by the explicit Python-cuGraph PageRank tools runner.
#[cfg_attr(test, allow(dead_code))]
pub(super) fn handle_python_oracle_unavailable(message: impl AsRef<str>) -> ! {
    crate::cugraph_optional_fixture::handle_cugraph_python_oracle_unavailable("cuGraph Python oracle", message);
}

pub(super) fn write_tiny_directed_edges(path: &Path) {
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

pub(super) const FILTERED_GRAPH_BASE_ROWS: usize = 1_000_000;
pub(super) const FILTERED_GRAPH_ACTIVE_ROWS: usize = 4;

pub(super) fn write_large_filtered_directed_edges(path: &Path) {
    write_large_filtered_directed_edges_with_active_rows(path, FILTERED_GRAPH_ACTIVE_ROWS);
}

pub(super) fn write_large_filtered_directed_edges_with_active_rows(path: &Path, active_rows: usize) {
    assert!(active_rows <= FILTERED_GRAPH_BASE_ROWS);
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let mut sources = Vec::with_capacity(FILTERED_GRAPH_BASE_ROWS);
    let mut destinations = Vec::with_capacity(FILTERED_GRAPH_BASE_ROWS);
    let mut active = Vec::with_capacity(FILTERED_GRAPH_BASE_ROWS);
    for row in 0..FILTERED_GRAPH_BASE_ROWS {
        let (src, dst) = match row {
            0 => (0, 1),
            1 => (0, 2),
            2 => (1, 3),
            3 => (2, 3),
            row => ((row % 1024) as i64, ((row + 1) % 1024) as i64),
        };
        sources.push(src);
        destinations.push(dst);
        active.push(row < active_rows);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(sources)) as ArrayRef,
            Arc::new(Int64Array::from(destinations)) as ArrayRef,
            Arc::new(BooleanArray::from(active)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(16_384)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_tiny_communities(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 0, 4, 5, 3, 3, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 4.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn register_edge_memtable(
    ctx: &SessionContext,
    table_name: &str,
    src_type: DataType,
    dst_type: DataType,
    weight_type: DataType,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", src_type.clone(), false),
        Field::new("dst", dst_type.clone(), false),
        Field::new("weight", weight_type.clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![edge_column(&src_type), edge_column(&dst_type), edge_column(&weight_type)],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_vertex_memtable(
    ctx: &SessionContext,
    table_name: &str,
    column_name: &str,
    data_type: DataType,
    nullable: bool,
) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, data_type.clone(), nullable)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![edge_column(&data_type)]).unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_vertex_pair_memtable(
    ctx: &SessionContext,
    table_name: &str,
    first_type: DataType,
    second_type: DataType,
    nullable: bool,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("first", first_type.clone(), nullable),
        Field::new("second", second_type.clone(), nullable),
    ]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![edge_column(&first_type), edge_column(&second_type)]).unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_personalization_memtable(ctx: &SessionContext, table_name: &str, vertex_type: DataType) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", vertex_type.clone(), false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![edge_column(&vertex_type), edge_column(&DataType::Float64)])
            .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

/// Registers the small undirected graph used to prove weighted similarity.
///
/// Its pair `(0, 1)` has weighted neighborhoods `{1: 1, 2: 2}` and
/// `{0: 1, 2: 3}`.  Its weighted Jaccard numerator is therefore `min(2, 3) =
/// 2`, while the denominator is `max(1, 0) + max(0, 1) + max(2, 3) = 5`.
pub(super) fn register_weighted_similarity_edges_i64(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 2, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 3.0, 5.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_i64_similarity_pairs(
    ctx: &SessionContext,
    table_name: &str,
    nullable: bool,
    first: Vec<Option<i64>>,
    second: Vec<Option<i64>>,
) {
    assert_eq!(first.len(), second.len(), "similarity pair columns must have the same length");
    let schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Int64, nullable),
        Field::new("second", DataType::Int64, nullable),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(first)) as ArrayRef, Arc::new(Int64Array::from(second)) as ArrayRef],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_directed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_empty_directed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_disconnected_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 2, 10])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 3, 11])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_i64_vertex_values(ctx: &SessionContext, table_name: &str, column_name: &str, values: Vec<i64>) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(values)) as ArrayRef]).unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, false),
        Field::new("allowed_dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_nullable_allowed_bfs_edges_with_nulls(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, true),
        Field::new("allowed_dst", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(0_i64), Some(2), None])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2_i64), Some(3), Some(3)])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_all_null_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Int64, true),
        Field::new("allowed_dst", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![None, None] as Vec<Option<i64>>)) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2_i64), Some(3)])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_nullable_vertex_memtable_with_nulls(ctx: &SessionContext, table_name: &str, column_name: &str) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(3_i64), None])) as ArrayRef],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_string_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_string_pagerank_parity_edges(ctx: &SessionContext, string_table: &str, numeric_table: &str) {
    let string_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true), Field::new("dst", DataType::Utf8, true)]));
    let string_batch = RecordBatch::try_new(
        Arc::clone(&string_schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("a"), Some("a"), Some("b"), Some("c"), None, Some("d")]))
                as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b"), Some("b"), Some("c"), Some("d"), Some("d"), Some("a"), None]))
                as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(string_table, Arc::new(MemTable::try_new(string_schema, vec![vec![string_batch]]).unwrap()))
        .unwrap();

    let numeric_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, true), Field::new("dst", DataType::Int64, true)]));
    let numeric_batch = RecordBatch::try_new(
        Arc::clone(&numeric_schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(1), Some(2), Some(3), None, Some(4)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2), Some(2), Some(3), Some(4), Some(4), Some(1), None])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(numeric_table, Arc::new(MemTable::try_new(numeric_schema, vec![vec![numeric_batch]]).unwrap()))
        .unwrap();
}

pub(super) fn register_mixed_string_algorithm_parity_edges(
    ctx: &SessionContext,
    string_table: &str,
    numeric_table: &str,
) {
    let string_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Utf8View, false),
        Field::new("dst", DataType::LargeUtf8, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let string_batch = RecordBatch::try_new(
        Arc::clone(&string_schema),
        vec![
            Arc::new(StringViewArray::from(vec!["a", "a", "b", "c", "c", "d"])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec!["b", "c", "c", "a", "d", "b"])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(string_table, Arc::new(MemTable::try_new(string_schema, vec![vec![string_batch]]).unwrap()))
        .unwrap();

    let numeric_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let numeric_batch = RecordBatch::try_new(
        Arc::clone(&numeric_schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 1, 2, 3, 3, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 3, 3, 1, 4, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(numeric_table, Arc::new(MemTable::try_new(numeric_schema, vec![vec![numeric_batch]]).unwrap()))
        .unwrap();
}

pub(super) fn register_string_algorithm_parity_side_inputs(ctx: &SessionContext) {
    let string_seed_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let string_seed_batch = RecordBatch::try_new(
        Arc::clone(&string_seed_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "c"])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.75_f64, 0.25])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(
        "string_ppr_seeds",
        Arc::new(MemTable::try_new(string_seed_schema, vec![vec![string_seed_batch]]).unwrap()),
    )
    .unwrap();

    let numeric_seed_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let numeric_seed_batch = RecordBatch::try_new(
        Arc::clone(&numeric_seed_schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.75_f64, 0.25])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(
        "numeric_ppr_seeds",
        Arc::new(MemTable::try_new(numeric_seed_schema, vec![vec![numeric_seed_batch]]).unwrap()),
    )
    .unwrap();

    let string_pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::LargeUtf8, false),
        Field::new("second", DataType::Utf8View, false),
    ]));
    let string_pair_batch = RecordBatch::try_new(
        Arc::clone(&string_pair_schema),
        vec![
            Arc::new(LargeStringArray::from(vec!["a", "b", "a"])) as ArrayRef,
            Arc::new(StringViewArray::from(vec!["b", "d", "a"])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(
        "string_candidate_pairs",
        Arc::new(MemTable::try_new(string_pair_schema, vec![vec![string_pair_batch]]).unwrap()),
    )
    .unwrap();

    let numeric_pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Int64, false),
        Field::new("second", DataType::Int64, false),
    ]));
    let numeric_pair_batch = RecordBatch::try_new(
        Arc::clone(&numeric_pair_schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2_i64, 4, 1])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(
        "numeric_candidate_pairs",
        Arc::new(MemTable::try_new(numeric_pair_schema, vec![vec![numeric_pair_batch]]).unwrap()),
    )
    .unwrap();
}

pub(super) fn register_large_string_pagerank_edges(
    ctx: &SessionContext,
    string_table: &str,
    numeric_table: &str,
    vertex_count: usize,
) {
    assert!(vertex_count > 1);
    let edge_count = vertex_count + (vertex_count.saturating_sub(1) / 2);
    let mut string_sources = Vec::with_capacity(edge_count);
    let mut string_destinations = Vec::with_capacity(edge_count);
    let mut numeric_sources = Vec::with_capacity(edge_count);
    let mut numeric_destinations = Vec::with_capacity(edge_count);
    for source in 0..vertex_count {
        let source_id = i64::try_from(source + 1).unwrap();
        let next_id = i64::try_from((source + 1) % vertex_count + 1).unwrap();
        string_sources.push(format!("vertex-{source:05}"));
        string_destinations.push(format!("vertex-{next:05}", next = (source + 1) % vertex_count));
        numeric_sources.push(source_id);
        numeric_destinations.push(next_id);
        if source > 0 && source % 2 == 0 {
            string_sources.push(format!("vertex-{source:05}"));
            string_destinations.push("vertex-00000".to_string());
            numeric_sources.push(source_id);
            numeric_destinations.push(1);
        }
    }

    let string_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let string_batch = RecordBatch::try_new(
        Arc::clone(&string_schema),
        vec![
            Arc::new(StringArray::from(string_sources)) as ArrayRef,
            Arc::new(StringArray::from(string_destinations)) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(string_table, Arc::new(MemTable::try_new(string_schema, vec![vec![string_batch]]).unwrap()))
        .unwrap();

    let numeric_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let numeric_batch = RecordBatch::try_new(
        Arc::clone(&numeric_schema),
        vec![
            Arc::new(Int64Array::from(numeric_sources)) as ArrayRef,
            Arc::new(Int64Array::from(numeric_destinations)) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(numeric_table, Arc::new(MemTable::try_new(numeric_schema, vec![vec![numeric_batch]]).unwrap()))
        .unwrap();
}

pub(super) fn register_disconnected_string_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "x"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "c", "y"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_nullable_string_bfs_edges_with_nulls(ctx: &SessionContext, table_name: &str) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true), Field::new("dst", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), Some("c"), Some("a")])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b"), Some("x"), Some("c"), Some("d"), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_string_bfs_edges_with_edge_id(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Utf8, false),
        Field::new("src", DataType::Utf8, false),
        Field::new("dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["e0", "e1", "e2", "e3"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_string_allowed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("allowed_src", DataType::Utf8, false),
        Field::new("allowed_dst", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["c", "d"])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_string_vertex_memtable(
    ctx: &SessionContext,
    table_name: &str,
    column_name: &str,
    values: &[&str],
) {
    let schema = Arc::new(Schema::new(vec![Field::new(column_name, DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(values.to_vec())) as ArrayRef])
            .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn bfs_distance_for(batch: &RecordBatch, vertex: i64) -> Option<i64> {
    let vertices = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows()).find_map(|row| (vertices.value(row) == vertex).then(|| distances.value(row)))
}

pub(super) fn string_bfs_distance_for(batch: &RecordBatch, vertex: &str) -> Option<i64> {
    let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let distances = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows())
        .find_map(|row| (vertices.value(row) == vertex && !distances.is_null(row)).then(|| distances.value(row)))
}

pub(super) fn string_bfs_predecessor_for(batch: &RecordBatch, vertex: &str) -> Option<String> {
    let vertices = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let predecessors = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    (0..batch.num_rows()).find_map(|row| {
        (vertices.value(row) == vertex && !predecessors.is_null(row)).then(|| predecessors.value(row).to_string())
    })
}

pub(super) fn int64_column(batch: &RecordBatch, index: usize) -> Vec<i64> {
    let values = batch.column(index).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows()).map(|row| values.value(row)).collect()
}

pub(super) fn assert_empty_i64_path_batch(batch: &RecordBatch) {
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(
        schema_field_names(batch.schema().as_ref()),
        vec!["path_index", "source", "target", "vertex", "distance"]
    );
    for index in 0..batch.num_columns() {
        assert_eq!(batch.schema().field(index).data_type(), &DataType::Int64);
    }
}

pub(super) fn edge_column(data_type: &DataType) -> ArrayRef {
    match data_type {
        DataType::Int32 => Arc::new(Int32Array::from(vec![0_i32, 1])) as ArrayRef,
        DataType::Int64 => Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
        DataType::Float32 => Arc::new(Float32Array::from(vec![1.0_f32, 2.0])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
        DataType::Utf8 => Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec!["a", "b"])) as ArrayRef,
        DataType::Utf8View => Arc::new(StringViewArray::from(vec!["a", "b"])) as ArrayRef,
        other => panic!("test helper does not support {other:?}"),
    }
}

pub(super) fn single_batch(batches: &[RecordBatch]) -> &RecordBatch {
    assert_eq!(batches.len(), 1, "expected one RecordBatch");
    &batches[0]
}

pub(super) fn schema_field_names(schema: &Schema) -> Vec<&str> {
    schema.fields().iter().map(|field| field.name().as_str()).collect()
}

pub(super) fn json_value(value: &str) -> serde_json::Value {
    serde_json::from_str(value).unwrap()
}

pub(super) fn json_array_item_by_name<'a>(array: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    array
        .as_array()
        .unwrap_or_else(|| panic!("expected JSON array while looking for item named {name}: {array}"))
        .iter()
        .find(|item| item["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("expected JSON array item named {name}: {array}"))
}

pub(super) fn string_value(batch: &RecordBatch, column: usize, row: usize) -> &str {
    batch.column(column).as_any().downcast_ref::<StringArray>().unwrap().value(row)
}

pub(super) fn bool_value(batch: &RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).as_any().downcast_ref::<BooleanArray>().unwrap().value(row)
}

pub(super) fn is_null_value(batch: &RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).is_null(row)
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
