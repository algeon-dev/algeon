//! Hand-built engine query plans for adapter exec-node tests.
//!
//! These plans exercise `AlgeonNativeExec` / `AlgeonNativeComposedExec` /
//! `AlgeonNativeGpuProducerExec` directly, without going through SQL or the
//! optimizer rule. Only tests that need access to the hidden API
//! repair use these builders.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::file::reader::{FileReader, SerializedFileReader};
use query_engine::{
    expr::ColumnRef,
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlan, QueryPlanBuilder},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

use super::support_core::{join_left_schema, join_right_schema};

pub(super) fn host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_with_host_sink(source)
}

pub(super) fn gpu_resident_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_gpu_resident(source)
}

pub(super) fn bounded_left_build_left_semi_join_host_sink_plan_with_paths(
    left_path: &std::path::Path,
    right_path: &std::path::Path,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source_with_exact_metadata(
        left_path,
        join_left_schema(),
        vec!["key".to_string(), "l_payload".to_string(), "l_m00".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source_with_exact_metadata(
        right_path,
        join_right_schema(),
        vec!["key".to_string(), "r_payload".to_string(), "r_m01".to_string()],
    )));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
        .with_build_side(JoinSide::Left);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

pub(super) fn local_parquet_gpu_resident_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = local_parquet_source_with_exact_metadata(
        path,
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_gpu_resident(source)
}

pub(super) fn local_parquet_host_sink_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = local_parquet_source_with_exact_metadata(
        path,
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_with_host_sink(source)
}

pub(super) fn local_parquet_source_with_exact_metadata(
    path: &std::path::Path,
    schema: Arc<Schema>,
    read_columns: Vec<String>,
) -> LocalParquetSource {
    LocalParquetSource::from_files(vec![exact_local_parquet_file(path)], schema, read_columns)
}

fn exact_local_parquet_file(path: &std::path::Path) -> LocalParquetFile {
    let reader = SerializedFileReader::try_from(path).unwrap_or_else(|error| {
        panic!("written native fixture must be readable as Parquet: {}: {error}", path.display())
    });
    let metadata = reader.metadata();
    let file_size = std::fs::metadata(path)
        .unwrap_or_else(|error| panic!("written native fixture must be stat-able: {}: {error}", path.display()))
        .len();
    let row_group_rows = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows()).expect("fixture row count fits u64"))
        .collect::<Vec<_>>();
    let row_group_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.compressed_size()).expect("fixture compressed bytes fit u64"))
        .collect::<Vec<_>>();
    let projected_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.total_byte_size()).expect("fixture uncompressed bytes fit u64"))
        .collect::<Vec<_>>();
    LocalParquetFile::new(path)
        .with_file_size_bytes(file_size)
        .with_row_groups((0..metadata.num_row_groups()).collect())
        .with_row_group_rows(row_group_rows)
        .with_row_group_bytes(row_group_bytes.clone())
        .with_projected_row_group_bytes(row_group_bytes, projected_uncompressed_bytes)
}

pub(super) fn gpu_resident_host_sink_plan(input_name: &str, schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

pub(super) fn bounded_gpu_resident_left_build_left_semi_join_host_sink_plan(
    left_input_name: &str,
    right_input_name: &str,
    input_schema: Arc<Schema>,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(left_input_name, Arc::clone(&input_schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(right_input_name, input_schema)));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
        .with_build_side(JoinSide::Left);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

fn local_parquet_source() -> LocalParquetSource {
    LocalParquetSource::new("/tmp/algeon-datafusion-native-exec-test.parquet", schema(), vec!["key".to_string()])
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]))
}
