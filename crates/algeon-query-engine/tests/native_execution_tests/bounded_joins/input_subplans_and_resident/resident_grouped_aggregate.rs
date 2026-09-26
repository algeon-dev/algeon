use std::sync::Arc;

use query_engine::{
    admission::{admit, bounded_join_unsupported_shape_facts},
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn bounded_native_query_executes_mixed_keyed_residual_join_with_grouped_aggregate_input() {
    let context = mixed_grouped_aggregate_execution_context(8 * 1024 * 1024 * 1024);
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-bounded-mixed-aggregate-input");
    let fact_path = dir.join("fact.parquet");
    let grouped_fact_path = dir.join("grouped_fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = write_mixed_aggregate_input_fact_parquet(&runtime, &fact_path);
    let grouped_fact_schema = write_mixed_aggregate_input_fact_row_grouped_parquet(&grouped_fact_path);
    let dim_schema = write_mixed_aggregate_input_dim_parquet(&runtime, &dim_path);
    let plan = mixed_keyed_residual_join_grouped_aggregate_input_plan(
        &fact_path,
        &grouped_fact_path,
        &dim_path,
        &fact_schema,
        &grouped_fact_schema,
        &dim_schema,
    );
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .unwrap();
    assert!(bounded_join_unsupported_shape_facts(&plan, &decision, rapids_analysis_context()).is_none());

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_reusable_hash_join_single_pass_probe");
    assert_eq!(join_metrics[0].join_type, "inner");
    assert_eq!(join_metrics[1].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[1].join_type, "inner");
    assert!(join_metrics[1].residual_predicate);
    assert_eq!(join_metrics[1].output_rows, 2);
    assert_budgeted_success_event_with_boundary(metrics, "query_engine.aggregate.execute_partial", 2, "chunk_operator");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(float64_array(batch, 0).values().as_ref(), &[40.0]);

    fs::remove_dir_all(dir).unwrap();
}

fn mixed_grouped_aggregate_execution_context(device_budget_bytes: u64) -> BackendQueryContext {
    backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    })
}
