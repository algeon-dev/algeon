use std::{fs::File, sync::Arc, time::Duration};

#[cfg(feature = "cuvs")]
use algeon_datafusion::native_report::NativeDataFusionExecutionMetrics;
use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    },
    cuml_sql::CumlAlgoExec,
    native_exec::AlgeonNativeExecutionConfig,
    planner::{AlgeonGpuExecutionMode, FinalPlanDisposition, classify_final_plan},
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
#[cfg(feature = "cuvs")]
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::{
    datasource::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect, execute_stream},
    prelude::{ParquetReadOptions, SessionContext},
};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use tokio::{task::yield_now, time::timeout};

fn backend() -> AlgeonGpuBackend {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 * 1024 * 1024 });
    AlgeonGpuBackend::builder()
        .with_execution_mode(AlgeonGpuExecutionMode::NativePreferred)
        .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            512 * 1024 * 1024,
            256 * 1024 * 1024,
            1,
            3,
            1,
        ))
        .devices([AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)])
        .build()
        .expect("cuML GPU backend")
}

fn context() -> (AlgeonGpuBackend, SessionContext) {
    let backend = backend();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .unwrap()
        .try_with_cuvs_sql()
        .unwrap()
        .try_with_cuml_sql()
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
        Field::new("d2", DataType::Float32, false),
    ]));
    let x = (0..8).map(|i| i as f32 - 3.0).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..8)) as ArrayRef,
            Arc::new(Float32Array::from(x.clone())),
            Arc::new(Float32Array::from(x.iter().map(|v| 2.0 * v).collect::<Vec<_>>())),
            Arc::new(Float32Array::from(x.iter().map(|v| -v).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    ctx.register_table("t", Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![batch]]).unwrap())).unwrap();
    (backend, ctx)
}

fn close(backend: AlgeonGpuBackend, ctx: SessionContext) {
    drop(ctx);
    let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent && close.observation_quiescent);
}

async fn plan(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
    ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
}

#[cfg(feature = "cuvs")]
fn recursive_plan_metrics(plan: &dyn ExecutionPlan) -> MetricsSet {
    fn append(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
        if let Some(plan_metrics) = plan.metrics() {
            for metric in plan_metrics.iter() {
                metrics.push(Arc::clone(metric));
            }
        }
        for child in plan.children() {
            append(metrics, child.as_ref());
        }
    }
    let mut metrics = MetricsSet::new();
    append(&mut metrics, plan);
    metrics
}

fn find_exec(plan: &dyn ExecutionPlan) -> Option<&CumlAlgoExec> {
    plan.downcast_ref::<CumlAlgoExec>()
        .or_else(|| plan.children().into_iter().find_map(|child| find_exec(child.as_ref())))
}

fn check_oracle(batches: &[RecordBatch]) {
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
    let ids = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let score = batches[0].column(1).as_any().downcast_ref::<Float32Array>().unwrap();
    let second = batches[0].column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    for i in 0..8 {
        assert_eq!(ids.value(i), i as i64);
        let x = i as f32 - 3.0;
        let expected = x.abs() * 6.0_f32.sqrt();
        assert!((score.value(i).abs() - expected).abs() < 1e-2, "score {} expected {expected}", score.value(i));
        assert!(second.value(i).abs() < 1e-2);
    }
}

#[tokio::test]
async fn tsvd_wide_sql_matches_host_oracle_with_host_input_boundary() {
    let (backend, ctx) = context();
    let sql = "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM t), n_components => 2)";
    let physical = plan(&ctx, sql).await;
    assert!(find_exec(physical.as_ref()).is_some());
    assert_eq!(classify_final_plan(physical.as_ref()), FinalPlanDisposition::Mixed);
    check_oracle(&collect(physical, ctx.task_ctx()).await.unwrap());
    close(backend, ctx);
}

#[tokio::test]
async fn hdbscan_and_umap_wide_and_fixed_list_sql_return_stable_shapes() {
    let (backend, ctx) = context();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("vector", DataType::FixedSizeList(Arc::clone(&item), 3), false),
    ]));
    let values = (0..8)
        .flat_map(|i| {
            let x = i as f32;
            [x, 2.0 * x, -x]
        })
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..8)),
            Arc::new(FixedSizeListArray::new(item, 3, Arc::new(Float32Array::from(values)), None)),
        ],
    )
    .unwrap();
    ctx.register_table("lists", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    for (name, options, outputs) in [
        ("cuml_hdbscan", "min_samples => 2, min_cluster_size => 2", ["cluster_id", "probability"]),
        ("cuml_umap", "n_components => 2, n_neighbors => 3, init => 'random'", ["dim_0", "dim_1"]),
    ] {
        for (table, features) in [("t", "d0, d1, d2"), ("lists", "vector")] {
            let sql = format!("SELECT * FROM {name}(input => (SELECT id, {features} FROM {table}), {options})");
            let batches = collect(plan(&ctx, &sql).await, ctx.task_ctx()).await.unwrap();
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
            let schema = batches[0].schema();
            assert_eq!(schema.field(0).name(), "row_ordinal");
            assert_eq!(schema.field(2).name(), outputs[0]);
            assert_eq!(schema.field(3).name(), outputs[1]);
            assert_eq!(schema.field(3).data_type(), &DataType::Float32);
            let ids = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(ids.values().as_ref(), &(0..8_i64).collect::<Vec<_>>());
        }
    }
    for (name, options, columns) in [
        ("cuml_tsvd", "n_components => 2", ["pc_0", "pc_1"]),
        ("cuml_hdbscan", "min_samples => 2, min_cluster_size => 2", ["cluster_id", "probability"]),
        ("cuml_umap", "n_neighbors => 3, init => 'random'", ["dim_0", "dim_1"]),
    ] {
        let sql = format!("SELECT * FROM {name}(input => (SELECT id, d0, d1, d2 FROM t WHERE false), {options})");
        let batches = collect(plan(&ctx, &sql).await, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0, "{name}");
        let schema = batches[0].schema();
        assert_eq!(schema.field(2).name(), columns[0]);
        assert_eq!(schema.field(3).name(), columns[1]);
    }
    close(backend, ctx);
}

#[tokio::test]
async fn hdbscan_three_blob_sql_preserves_noise_and_row_identity() {
    let (backend, ctx) = context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let mut x = Vec::new();
    let mut y = Vec::new();
    for index in 0..30 {
        let center = (index / 10) as f32 * 50.0;
        x.push(center + (index % 3) as f32 * 0.03);
        y.push(center + (index % 10 / 3) as f32 * 0.03);
    }
    x.push(500.0);
    y.push(-500.0);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(100..131)) as ArrayRef,
            Arc::new(Float32Array::from(x)),
            Arc::new(Float32Array::from(y)),
        ],
    )
    .unwrap();
    ctx.register_table("blobs", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    let sql =
        "SELECT * FROM cuml_hdbscan(input => (SELECT id, d0, d1 FROM blobs), min_samples => 5, min_cluster_size => 5)";
    let output = collect(plan(&ctx, sql).await, ctx.task_ctx()).await.unwrap();
    assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 31);
    let ordinals = output[0].column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
    let ids = output[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let labels = output[0].column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ordinals.values().as_ref(), &(0..31_u64).collect::<Vec<_>>());
    assert_eq!(ids.values().as_ref(), &(100..131_i64).collect::<Vec<_>>());
    assert_eq!(labels.value(30), -1);
    assert_eq!(
        (0..30).map(|i| labels.value(i)).filter(|id| *id >= 0).collect::<std::collections::BTreeSet<_>>().len(),
        3
    );
    close(backend, ctx);
}

#[tokio::test]
async fn logistic_sql_scores_independent_predict_relation_for_binary_and_three_classes() {
    let (backend, ctx) = context();
    for classes in [2, 3] {
        let centers = if classes == 2 { vec![-8.0_f32, 8.0] } else { vec![-8.0_f32, 0.0, 8.0] };
        let training_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("d0", DataType::Float32, false),
            Field::new("d1", DataType::Float32, false),
            Field::new("label", DataType::Int32, false),
        ]));
        let training_x =
            centers.iter().flat_map(|center| (0..6).map(move |i| center + (i as f32 - 2.5) * 0.1)).collect::<Vec<_>>();
        let training = RecordBatch::try_new(
            Arc::clone(&training_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..training_x.len() as i64)) as ArrayRef,
                Arc::new(Float32Array::from(training_x.clone())),
                Arc::new(Float32Array::from(training_x.iter().map(|value| value * 0.25).collect::<Vec<_>>())),
                Arc::new(Int32Array::from((0..classes).flat_map(|class| [class as i32; 6]).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let training_name = format!("train_{classes}");
        ctx.register_table(&training_name, Arc::new(MemTable::try_new(training_schema, vec![vec![training]]).unwrap()))
            .unwrap();
        let predict_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("d0", DataType::Float32, false),
            Field::new("d1", DataType::Float32, false),
        ]));
        let predict = RecordBatch::try_new(
            Arc::clone(&predict_schema),
            vec![
                Arc::new(StringArray::from((0..classes).map(|index| format!("p{index}")).collect::<Vec<_>>()))
                    as ArrayRef,
                Arc::new(Float32Array::from(centers.clone())),
                Arc::new(Float32Array::from(centers.iter().map(|value| value * 0.25).collect::<Vec<_>>())),
            ],
        )
        .unwrap();
        let predict_name = format!("predict_{classes}");
        ctx.register_table(&predict_name, Arc::new(MemTable::try_new(predict_schema, vec![vec![predict]]).unwrap()))
            .unwrap();
        for empty in [false, true] {
            let predicate = if empty { " WHERE false" } else { "" };
            let sql = format!(
                "SELECT * FROM cuml_logistic_regression(\
                training => (SELECT id, d0, d1, label FROM {training_name}), \
                predict => (SELECT id, d0, d1 FROM {predict_name}{predicate}))"
            );
            let output = collect(plan(&ctx, &sql).await, ctx.task_ctx()).await.unwrap();
            assert_eq!(output[0].schema().field(1).data_type(), &DataType::Utf8);
            assert_eq!(output[0].schema().field(2).data_type(), &DataType::Int32);
            assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), if empty { 0 } else { classes });
            if !empty {
                let ids = output[0].column(1).as_any().downcast_ref::<StringArray>().unwrap();
                let labels = output[0].column(2).as_any().downcast_ref::<Int32Array>().unwrap();
                for index in 0..classes {
                    assert_eq!(ids.value(index), format!("p{index}"));
                    assert_eq!(labels.value(index), index as i32);
                }
            }
        }
    }
    close(backend, ctx);
}

#[tokio::test]
async fn random_forest_sql_classifies_and_regresses_independent_relations() {
    let (backend, ctx) = context();
    let x = (0..48).map(|i| i as f32 - 24.0).collect::<Vec<_>>();
    let predict_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
    ]));
    let predicted = RecordBatch::try_new(
        Arc::clone(&predict_schema),
        vec![Arc::new(Int64Array::from_iter_values(100..148)) as ArrayRef, Arc::new(Float32Array::from(x.clone()))],
    )
    .unwrap();
    ctx.register_table("rf_predict", Arc::new(MemTable::try_new(predict_schema, vec![vec![predicted]]).unwrap()))
        .unwrap();
    for (function, target_type) in
        [("cuml_random_forest_classifier", DataType::Int32), ("cuml_random_forest_regressor", DataType::Float32)]
    {
        let training_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("d0", DataType::Float32, false),
            Field::new("label", target_type.clone(), false),
        ]));
        let labels: ArrayRef = if target_type == DataType::Int32 {
            Arc::new(Int32Array::from(x.iter().map(|v| i32::from(*v >= 0.0)).collect::<Vec<_>>()))
        } else {
            Arc::new(Float32Array::from(x.iter().map(|v| 2.0 * v + 1.0).collect::<Vec<_>>()))
        };
        let batch = RecordBatch::try_new(
            Arc::clone(&training_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..48)) as ArrayRef,
                Arc::new(Float32Array::from(x.clone())),
                labels,
            ],
        )
        .unwrap();
        let table = format!("rf_train_{function}");
        ctx.register_table(&table, Arc::new(MemTable::try_new(training_schema, vec![vec![batch]]).unwrap())).unwrap();
        for empty in [false, true] {
            let condition = if empty { " WHERE false" } else { "" };
            let sql = format!(
                "SELECT * FROM {function}(training => (SELECT id, d0, label FROM {table}), predict => (SELECT id, d0 FROM rf_predict{condition}), n_trees => 64)"
            );
            let batches = collect(plan(&ctx, &sql).await, ctx.task_ctx()).await.unwrap();
            let total = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            assert_eq!(total, if empty { 0 } else { 48 });
            assert_eq!(batches[0].schema().field(2).data_type(), &target_type);
            if !empty {
                let ids = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(ids.value(0), 100);
                if target_type == DataType::Int32 {
                    let predictions = batches[0].column(2).as_any().downcast_ref::<Int32Array>().unwrap();
                    let accuracy =
                        (0..48).filter(|&i| predictions.value(i) == i32::from(i >= 24)).count() as f32 / 48.0;
                    assert!(accuracy >= 0.95, "RF SQL accuracy {accuracy}");
                } else {
                    let predictions = batches[0].column(2).as_any().downcast_ref::<Float32Array>().unwrap();
                    let mse = (0..48).map(|i| (predictions.value(i) - (2.0 * x[i] + 1.0)).powi(2)).sum::<f32>() / 48.0;
                    assert!(mse < 12.0, "RF SQL MSE {mse}");
                }
            }
        }
    }
    close(backend, ctx);
}

#[tokio::test]
async fn fixed_size_list_and_parquet_sources_execute() {
    let (backend, ctx) = context();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("vector", DataType::FixedSizeList(Arc::clone(&item), 3), false),
    ]));
    let values = (0..8)
        .flat_map(|i| {
            let x = i as f32 - 3.0;
            [x, 2.0 * x, -x]
        })
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..8)),
            Arc::new(FixedSizeListArray::new(item, 3, Arc::new(Float32Array::from(values)), None)),
        ],
    )
    .unwrap();
    ctx.register_table("lists", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    let sql = "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, vector FROM lists), n_components => 2)";
    check_oracle(&collect(plan(&ctx, sql).await, ctx.task_ctx()).await.unwrap());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.parquet");
    let batches = ctx.table("t").await.unwrap().collect().await.unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), batches[0].schema(), None).unwrap();
    writer.write(&batches[0]).unwrap();
    writer.close().unwrap();
    ctx.register_parquet("parquet_vectors", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let sql = "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM parquet_vectors), n_components => 2)";
    check_oracle(&collect(plan(&ctx, sql).await, ctx.task_ctx()).await.unwrap());
    close(backend, ctx);
}

#[cfg(feature = "cuvs")]
#[tokio::test]
async fn relational_composition_and_cuvs_share_one_admission() {
    let (backend, ctx) = context();
    let sql = "WITH components AS (
        SELECT id, pc_0 FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM t), n_components => 1)),
        neighbors AS (
        SELECT neighbor_id FROM cuvs_brute_force_knn(
            dataset => (SELECT id, d0, d1, d2 FROM t),
            queries => (SELECT id, d0, d1, d2 FROM t WHERE id = 0),
            k => 1, metric => 'l2_expanded'))
        SELECT c.id, c.pc_0 FROM components c JOIN neighbors n ON c.id = n.neighbor_id
        WHERE c.id >= 0 ORDER BY c.id";
    let physical = plan(&ctx, sql).await;
    assert!(find_exec(physical.as_ref()).is_some());
    let result = collect(Arc::clone(&physical), ctx.task_ctx()).await.unwrap();
    assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&recursive_plan_metrics(physical.as_ref()));
    assert!(metrics.admission_sequence.is_some(), "both GPU function occurrences must share admission");
    assert_eq!(result[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 0);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .expect("combined GPU function admission must be released");
    close(backend, ctx);
}

#[tokio::test]
async fn dropping_result_stream_releases_admission_and_plan_can_run_again() {
    let (backend, ctx) = context();
    let sql = "SELECT * FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM t), n_components => 1)";
    let physical = plan(&ctx, sql).await;
    let mut stream = execute_stream(Arc::clone(&physical), ctx.task_ctx()).unwrap();
    assert!(stream.next().await.transpose().unwrap().is_some());
    drop(stream);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(collect(physical, ctx.task_ctx()).await.unwrap().iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .unwrap();
    close(backend, ctx);
}
