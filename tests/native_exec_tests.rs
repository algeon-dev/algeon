use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{LocalParquetSource, MemoryControlPolicy, QueryPlan, QueryPlanBuilder, SourceNode};
use datafusion::physical_plan::{ExecutionPlan, displayable, execution_plan::Boundedness};
use datafusion_cudf::{ErrorCode, NativeCudfExec, NativeCudfExecutionConfig};

#[test]
fn native_exec_constructs_host_arrow_leaf_with_conservative_properties() {
    let native_plan = host_sink_plan();
    let schema = Arc::clone(native_plan.output_schema());
    let exec = NativeCudfExec::try_new(native_plan, bounded_config()).unwrap();

    assert_eq!(exec.name(), "NativeCudfExec");
    assert!(exec.children().is_empty());
    assert_eq!(exec.schema(), schema);
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.properties().boundedness, Boundedness::Bounded);
    assert!(exec.properties().output_ordering().is_none());

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NativeCudfExec"));
    assert!(display.contains("sink=host_arrow"));
    assert!(display.contains("memory_control_policy=bounded"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("output_boundary=device_to_host"));
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = NativeCudfExec::try_new(gpu_resident_plan(), bounded_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn native_exec_rejects_unbounded_memory_control_policy() {
    let config = bounded_config().with_memory_control_policy(MemoryControlPolicy::Unbounded);
    let error = NativeCudfExec::try_new(host_sink_plan(), config).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("memory_control_policy"), Some("unbounded"));
}

fn bounded_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

fn host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_with_host_sink(source)
}

fn gpu_resident_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_gpu_resident(source)
}

fn local_parquet_source() -> LocalParquetSource {
    LocalParquetSource::new("/tmp/datafusion-cudf-native-exec-test.parquet", schema(), vec!["key".to_string()])
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]))
}
