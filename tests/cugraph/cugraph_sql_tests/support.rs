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
    result: datafusion::common::Result<datafusion_nexus::graph_sql::CugraphAlgorithmSpec>,
    expected: &str,
) {
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "expected error containing {expected:?}, got {error:?}",);
}

pub(super) fn schema_fields(algorithm: CugraphAlgorithm) -> Vec<Field> {
    stable_result_schema(algorithm).fields().iter().map(|field| field.as_ref().clone()).collect()
}

pub(super) fn find_composed_exec(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
) -> Option<&NativeCudfComposedExec> {
    if let Some(composed) = plan.downcast_ref::<NativeCudfComposedExec>() {
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
    plan.downcast_ref::<NativeCudfExec>().is_some()
        || plan.downcast_ref::<NativeCudfComposedExec>().is_some()
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

pub(super) fn backend_cugraph_sql_state(builder: SessionStateBuilder) -> datafusion::execution::SessionState {
    let backend = NexusGpuBackend::builder()
        .fallback_policy(GpuFallbackPolicy::GpuPreferred)
        .admission(NexusAdmissionPolicy::default().with_slot_guarantee(SlotGuarantee::Optimistic))
        .build()
        .unwrap();
    backend.install_on_with_cugraph_sql(builder, CugraphSqlConfig::default()).unwrap().build()
}

pub(super) fn default_backend_cugraph_sql_state() -> datafusion::execution::SessionState {
    backend_cugraph_sql_state(SessionStateBuilder::new_with_default_features())
}

pub(super) async fn drop_session_context_on_blocking_thread(ctx: SessionContext) {
    tokio::task::spawn_blocking(move || drop(ctx)).await.unwrap();
}

#[derive(Debug, Clone)]
pub(super) struct PageRankRow {
    pub(super) vertex: i64,
    pub(super) value: f64,
}

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
            return None;
        },
    };
    if output.status.code() == Some(77) {
        handle_python_oracle_unavailable(String::from_utf8_lossy(&output.stderr));
        return None;
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

pub(super) fn handle_python_oracle_unavailable(message: impl AsRef<str>) {
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

pub(super) fn register_nullable_edge_memtable(
    ctx: &SessionContext,
    table_name: &str,
    src_nullable: bool,
    dst_nullable: bool,
    weight_nullable: bool,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, src_nullable),
        Field::new("dst", DataType::Int64, dst_nullable),
        Field::new("weight", DataType::Float64, weight_nullable),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

pub(super) fn register_nullable_edge_id_memtable(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, true),
        Field::new("dst", DataType::Int64, true),
        Field::new("weight", DataType::Float64, true),
        Field::new("edge_id", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10_i64, 12])) as ArrayRef,
        ],
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

pub(super) fn register_personalization_memtable(
    ctx: &SessionContext,
    table_name: &str,
    vertex_type: DataType,
    value_type: DataType,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", vertex_type.clone(), false),
        Field::new("value", value_type.clone(), false),
    ]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![edge_column(&vertex_type), edge_column(&value_type)]).unwrap();
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

pub(super) fn register_large_string_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::LargeUtf8, false),
        Field::new("dst", DataType::LargeUtf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(LargeStringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec!["b", "c", "d", "d"])) as ArrayRef,
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

pub(super) fn string_path_column(batch: &RecordBatch, index: usize) -> Vec<&str> {
    let values = batch.column(index).as_any().downcast_ref::<StringArray>().unwrap();
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
        other => panic!("test helper does not support {other:?}"),
    }
}

pub(super) async fn assert_validate_error(
    ctx: &SessionContext,
    function_name: &str,
    table_name: &str,
    call_json: &str,
    expected_code: &str,
) -> RecordBatch {
    let sql = format!(
        "SELECT valid, error_code, src_dtype, dst_dtype, weight_dtype \
         FROM cugraph_validate_call('{function_name}', '{table_name}', '{call_json}')"
    );
    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let batch = single_batch(&batches).clone();
    assert!(!bool_value(&batch, 0, 0), "validation should fail for {sql}");
    assert_eq!(string_value(&batch, 1, 0), expected_code, "unexpected validation error for {sql}");
    batch
}

pub(super) async fn validate_call_batch(ctx: &SessionContext, function_name: &str, call_json: &str) -> RecordBatch {
    let escaped = call_json.replace('\'', "''");
    let sql = format!(
        "SELECT valid, error_code, message \
         FROM cugraph_validate_call('{function_name}', 'edges', '{escaped}')"
    );
    single_batch(&ctx.sql(&sql).await.unwrap().collect().await.unwrap()).clone()
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

pub(super) fn test_data_type_label(data_type: &DataType) -> &'static str {
    match data_type {
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Boolean => "Boolean",
        DataType::Utf8 => "Utf8",
        DataType::LargeUtf8 => "LargeUtf8",
        _ => "Other",
    }
}

pub(super) fn json_array_item_by_name<'a>(array: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    array
        .as_array()
        .unwrap_or_else(|| panic!("expected JSON array while looking for item named {name}: {array}"))
        .iter()
        .find(|item| item["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("expected JSON array item named {name}: {array}"))
}

pub(super) fn assert_json_array_contains_string(array: &serde_json::Value, expected: &str) {
    assert!(
        array.as_array().unwrap().iter().any(|item| item.as_str() == Some(expected)),
        "expected JSON string array to contain {expected:?}: {array}",
    );
}

pub(super) fn assert_json_columns_parse_and_have_no_control_chars(batch: &RecordBatch) {
    for (column_index, field) in batch.schema().fields().iter().enumerate() {
        if !field.name().ends_with("_json") {
            continue;
        }
        let values = batch.column(column_index).as_any().downcast_ref::<StringArray>().unwrap();
        for row_index in 0..batch.num_rows() {
            let value = json_value(values.value(row_index));
            assert_json_strings_have_no_control_chars(&value, &format!("{}[{row_index}]", field.name()));
        }
    }
}

pub(super) fn assert_json_strings_have_no_control_chars(value: &serde_json::Value, path: &str) {
    match value {
        serde_json::Value::String(text) => {
            if let Some(control) = text.chars().find(|ch| ch.is_control()) {
                panic!("JSON string at {path} contains control character U+{:04X}", control as u32);
            }
        },
        serde_json::Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                assert_json_strings_have_no_control_chars(item, &format!("{path}[{index}]"));
            }
        },
        serde_json::Value::Object(entries) => {
            for (key, item) in entries {
                assert_json_strings_have_no_control_chars(item, &format!("{path}.{key}"));
            }
        },
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {},
    }
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
