//! Incremental partitioned final-merge plan shared by the black-box budgeting
//! cases and the crate's compiled-scheduler GPU suite. The includer provides
//! `LocalParquetSource`, its metadata-enriching fixture source constructor.

use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use query_planning::{
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, ColumnRef, Expr},
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, QueryPlanBuilder},
    source::{LocalParquetFile, SourceNode},
};

use super::LocalParquetSource;

pub(crate) fn incremental_partitioned_merge_builder(
    dir: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
) -> (QueryPlanBuilder, NodeId, Arc<Schema>) {
    let path = dir.join(format!("input-{row_groups}.parquet"));
    let input_schema = write_unique_key_row_grouped_parquet(&path, row_groups, rows_per_row_group);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("sum_value", DataType::Int64, true),
        Field::new("count_value", DataType::Int64, false),
        Field::new("avg_value", DataType::Float64, true),
    ]));
    let dimension_path = dir.join("dimension.parquet");
    let dimension_schema = write_single_row_dimension_parquet(&dimension_path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string(), "d".to_string()],
    );
    let dimension = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dimension_path).with_file_size_bytes(fs::metadata(&dimension_path).unwrap().len())],
        Arc::clone(&dimension_schema),
        vec!["d".to_string()],
    );
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    // COUNT and AVG carry partial state whose column shape differs from the
    // merged state, so merging raw slices with merged states in one set would
    // fail on schema mismatch.
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "count_value"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("value"), "avg_value"),
        ],
    );
    // The aggregate consumes the streamed probe output of a compiled join, the
    // shape whose partials arrive one chunk at a time.
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let dimension = builder.add_source(SourceNode::LocalParquet(dimension));
    let join = builder
        .add_join(
            source,
            dimension,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
                JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
        .unwrap();
    let aggregate = builder.add_aggregate(join, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    (builder, aggregate, aggregated_schema)
}

/// Uncompressed row groups whose `key` is unique across the whole file, so
/// every partial aggregate keeps all of its input rows and merging never
/// shrinks the retained state: the shape of a high-cardinality group by.
/// `d` is a constant join key so a one-row dimension keeps every row.
fn write_unique_key_row_grouped_parquet(path: &Path, row_groups: usize, rows_per_row_group: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
    ]));
    let rows = row_groups * rows_per_row_group;
    let keys = (0..rows).map(|row| row as i64).collect::<Vec<_>>();
    let values = (0..rows).map(|row| (row % 7) as i64).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(values)),
            Arc::new(Int64Array::from(vec![0i64; rows])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(rows_per_row_group))
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

/// One-row dimension whose `d` matches every row of
/// [`write_unique_key_row_grouped_parquet`].
fn write_single_row_dimension_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0i64]))]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}
