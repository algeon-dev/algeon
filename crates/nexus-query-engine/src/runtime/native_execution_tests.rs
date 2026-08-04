use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{Array, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
#[cfg(feature = "cugraph")]
use nexus_query_engine::exec::execute_query_plan_with_gpu_inputs_observed;
#[cfg(feature = "cugraph")]
use nexus_query_engine::observability::{
    PlanExecutionBoundaryKind, RuntimeObservationConfig, RuntimeObservationPayload, RuntimeTerminalOutcome,
};
use nexus_query_engine::{
    error::QueryErrorCode,
    exec::{execute_query_plan, execute_query_plan_with_gpu_inputs, output::QueryOutput},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::{MemoryReservationEventAction, QueryMetrics},
    mode::StatisticalAggregateExecutionMode,
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, ProjectionOutput, QueryPlanBuilder},
    runtime::{
        ExecutionContext, MemoryControlPolicy, ReservationKind,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, SourceNode,
        ZERO_COLUMN_COUNT_COLUMN,
    },
};
#[cfg(feature = "cugraph")]
use nexus_query_engine::{
    exec::force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    exec::force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
    graph_memory::GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, graph_memory::GraphMemoryCalibrationOverride,
    graph_memory::GraphReservationOutcome, plan::GraphAlgorithm, plan::GraphAlgorithmParams, plan::GraphAlgorithmSpec,
    plan::GraphBetweennessCentralityParams, plan::GraphBetweennessSeeds, plan::GraphBfsEndpointPolicy,
    plan::GraphBfsParams, plan::GraphBfsPredicateInputs, plan::GraphConstructionOptions, plan::GraphConstructionPolicy,
    plan::GraphEdgeBetweennessCentralityParams, plan::GraphEdgeColumnBindings, plan::GraphPageRankParams,
    plan::GraphPredicateInput, plan::GraphSimilarityParams, plan::GraphVertexPairInput,
    plan::stable_graph_result_schema,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[test]
fn execute_query_plan_stamps_correlation_on_early_errors() {
    use std::time::{Duration, Instant};

    use nexus_query_engine::{
        metrics::{QueryId, TraceId},
        runtime::ExecutionCancellation,
    };

    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let plan = builder.finish_with_host_sink(source);

    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let context = ExecutionContext::builder()
        .with_cancellation(cancellation)
        .with_query_id(QueryId::new("query-exec"))
        .with_trace_id(TraceId::new("trace-exec"))
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("cancelled context should fail before native execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.query_id().map(QueryId::as_str), Some("query-exec"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("trace-exec"));
    assert_eq!(error.fact_value("query_id"), Some("query-exec"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-exec"));

    let expired_context = ExecutionContext::builder()
        .with_cancellation(ExecutionCancellation::with_deadline(Some(Instant::now() - Duration::from_millis(1))))
        .with_query_id(QueryId::new("query-deadline"))
        .with_trace_id(TraceId::new("trace-deadline"))
        .build()
        .unwrap();
    let deadline_error = match execute_query_plan(&expired_context, &plan) {
        Ok(_) => panic!("an elapsed deadline must stop the native execution safe boundary"),
        Err(error) => error,
    };
    assert_eq!(deadline_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(deadline_error.fact_value("reason"), Some("deadline_expired"));
    assert_eq!(deadline_error.query_id().map(QueryId::as_str), Some("query-deadline"));
    assert_eq!(deadline_error.trace_id().map(TraceId::as_str), Some("trace-deadline"));
}

#[path = "native_execution_tests/join_test_support.rs"]
mod join_test_support;
#[path = "native_execution_tests/parquet_fixtures.rs"]
mod parquet_fixtures;
#[path = "native_execution_tests/test_support.rs"]
mod test_support;

#[path = "native_execution_tests/aggregate_execution.rs"]
mod aggregate_execution;
#[path = "native_execution_tests/bounded_budgeting.rs"]
mod bounded_budgeting;
#[path = "native_execution_tests/bounded_joins.rs"]
mod bounded_joins;
#[path = "native_execution_tests/core_execution.rs"]
mod core_execution;
#[cfg(feature = "cugraph")]
#[path = "native_execution_tests/graph_algorithm_execution.rs"]
mod graph_algorithm_execution;
#[path = "native_execution_tests/native_join_execution.rs"]
mod native_join_execution;
#[path = "native_execution_tests/semijoin_reduction.rs"]
mod semijoin_reduction;
#[path = "native_execution_tests/unbounded_pruning_regressions.rs"]
mod unbounded_pruning_regressions;
#[cfg(feature = "cuvs")]
#[path = "native_execution_tests/vector_algorithm_execution.rs"]
mod vector_algorithm_execution;
// This uncached execution suite runs with durable source caching disabled.

use join_test_support::*;
use parquet_fixtures::*;
use test_support::*;
