use std::{
    fs::{self, File},
    iter,
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::SessionContext;
use parquet::arrow::ArrowWriter;
use workloads::{
    forest::{
        LINEITEM_FOREST_FEATURES, describe_ember2024_dataset, describe_lineitem_forest_workload, ember2024_table_name,
        register_ember2024_tables,
    },
    tpch::{TpchTableLayout, register_tpch_table_layout, write_tiny_tpch_parquet_dataset},
};

fn write_split(path: &Path, rows: i64, features: &[DataType]) {
    let fields = iter::once(Field::new("id", DataType::Int64, false))
        .chain(
            features
                .iter()
                .enumerate()
                .map(|(index, data_type)| Field::new(format!("f{index}"), data_type.clone(), false)),
        )
        .chain(iter::once(Field::new("label", DataType::Int32, false)))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let columns = iter::once(Arc::new(Int64Array::from_iter_values(0..rows)) as ArrayRef)
        .chain(features.iter().map(|data_type| match data_type {
            DataType::Float64 => Arc::new(Float64Array::from_iter_values((0..rows).map(|row| row as f64))) as ArrayRef,
            _ => Arc::new(Float32Array::from_iter_values((0..rows).map(|row| row as f32))) as ArrayRef,
        }))
        .chain(iter::once(Arc::new(Int32Array::from_iter_values((0..rows).map(|row| (row % 2) as i32))) as ArrayRef))
        .collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_ember_fixture(root: &Path) {
    fs::create_dir_all(root.join("parquet")).unwrap();
    fs::create_dir_all(root.join("models")).unwrap();
    write_split(&root.join("parquet/train.parquet"), 5, &vec![DataType::Float32; 3]);
    write_split(&root.join("parquet/test.parquet"), 4, &vec![DataType::Float32; 3]);
    fs::write(root.join("models/EMBER2024_all.model"), "tree\nversion=v4\n").unwrap();
    // thrember also ships pickled multi-booster models, which no forest loader reads.
    fs::write(root.join("models/EMBER2024_behavior.model"), b"\x80\x04\x95tree\n").unwrap();
}

#[tokio::test]
async fn ember_discovery_reads_feature_count_orders_splits_and_skips_pickled_models() {
    let temporary = tempfile::tempdir().unwrap();
    write_ember_fixture(temporary.path());

    let metadata = describe_ember2024_dataset(temporary.path()).unwrap();
    assert_eq!(metadata.feature_count, 3);
    assert_eq!(metadata.feature_columns(), ["f0", "f1", "f2"]);
    assert_eq!(
        metadata.splits.iter().map(|split| (split.split.as_str(), split.row_count)).collect::<Vec<_>>(),
        [("test", 4), ("train", 5)]
    );
    assert_eq!(metadata.models.iter().map(|model| model.name.as_str()).collect::<Vec<_>>(), ["EMBER2024_all"]);
    let projected = metadata.to_neutral_metadata();
    assert_eq!(projected.total_rows, 9);
    assert_eq!(projected.table_row_counts, "test=4,train=5");

    let ctx = SessionContext::new();
    register_ember2024_tables(&ctx, &metadata).await.unwrap();
    let batches = ctx
        .sql(&format!("SELECT count(*) FROM {}", ember2024_table_name("train")))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0), 5);
}

#[test]
fn ember_discovery_rejects_non_float32_features_and_disagreeing_splits() {
    let temporary = tempfile::tempdir().unwrap();
    write_ember_fixture(temporary.path());
    write_split(&temporary.path().join("parquet/challenge.parquet"), 2, &vec![DataType::Float32; 2]);
    let error = describe_ember2024_dataset(temporary.path()).unwrap_err().to_string();
    assert!(error.contains("split challenge has 2 features"), "{error}");

    write_split(
        &temporary.path().join("parquet/challenge.parquet"),
        2,
        &[DataType::Float32, DataType::Float64, DataType::Float32],
    );
    let error = describe_ember2024_dataset(temporary.path()).unwrap_err().to_string();
    assert!(error.contains("expected f1 Float32"), "{error}");
}

/// Writes `lineitem` the way the SF10 fixture declares it: every column
/// nullable, decimals for the money columns.
fn write_nullable_lineitem(path: &Path) {
    let decimal = DataType::Decimal128(15, 2);
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, true),
        Field::new("l_partkey", DataType::Int64, true),
        Field::new("l_suppkey", DataType::Int64, true),
        Field::new("l_linenumber", DataType::Int32, true),
        Field::new("l_quantity", decimal.clone(), true),
        Field::new("l_discount", decimal.clone(), true),
        Field::new("l_tax", decimal, true),
    ]));
    let money = |values: [i128; 4]| {
        Arc::new(Decimal128Array::from_iter_values(values).with_precision_and_scale(15, 2).unwrap()) as ArrayRef
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 11, 12, 13])),
            Arc::new(Int64Array::from(vec![5, 6, 7, 8])),
            Arc::new(Int32Array::from(vec![1, 2, 1, 7])),
            money([1700, 3600, 800, 2800]),
            money([4, 9, 10, 6]),
            money([2, 6, 2, 0]),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn lineitem_forest_input_is_non_nullable_float32_with_unique_ids() {
    let temporary = tempfile::tempdir().unwrap();
    let tpch = temporary.path().join("tpch");
    write_tiny_tpch_parquet_dataset(&tpch).unwrap();
    write_nullable_lineitem(&tpch.join("lineitem.parquet"));
    let model = temporary.path().join("lineitem.json");
    fs::write(&model, "{}").unwrap();

    let workload = describe_lineitem_forest_workload(&tpch, &model).unwrap();
    let ctx = SessionContext::new();
    register_tpch_table_layout(&ctx, "lineitem", &TpchTableLayout::from_metadata(&workload.lineitem)).await.unwrap();
    let batches = ctx.sql(&workload.input_sql("lineitem")).await.unwrap().collect().await.unwrap();
    let result = concat_batches(&batches[0].schema(), &batches).unwrap();

    let schema = result.schema();
    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    assert!(!schema.field(0).is_nullable());
    for (index, column) in LINEITEM_FOREST_FEATURES.iter().enumerate() {
        let field = schema.field(index + 1);
        assert_eq!(field.name(), column);
        assert_eq!(field.data_type(), &DataType::Float32, "{column}");
        assert!(!field.is_nullable(), "{column}");
    }
    let ids = result.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids.values().as_ref(), [9, 10, 17, 31]);
    let quantity = result.column(5).as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(quantity.values().as_ref(), [17.0, 36.0, 8.0, 28.0]);
}
