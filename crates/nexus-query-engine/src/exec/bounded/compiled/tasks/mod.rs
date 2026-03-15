use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    rc::Rc,
    time::Instant,
};

use arrow::record_batch::RecordBatch;

use super::{
    join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection},
    reservations::validate_task_runtime_gates,
    scheduler::{CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput},
    state::CompiledPipelineState,
};
use crate::{
    capability::{
        KeyedHashJoinStrategyInputFacts, keyed_hash_join_strategy_input_facts_for_plan,
        keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness,
    },
    dynamic_filter::{
        DEFAULT_SMALL_IN_LIST_DYNAMIC_FILTER_LIMIT, DIMENSION_SIZED_DYNAMIC_FILTER_CONSUMER_BYTES, DynamicFilter,
        DynamicFilterConsumer, DynamicFilterProducer, DynamicFilterRoute, DynamicFilterSet,
        LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON, LARGE_LOCAL_PARQUET_RANGE_DYNAMIC_FILTER_PRODUCER_ROWS,
        MAX_DYNAMIC_FILTER_BUILD_ROWS, TINY_LOCAL_PARQUET_DYNAMIC_FILTER_CONSUMER_ROW_GROUPS,
        dynamic_filter_consumer_source_feeds_downstream_work_fanout, dynamic_filter_i64_values,
        dynamic_filter_predicate_for_schema,
    },
    error::{QueryError, QueryErrorCode, Result},
    exec::{
        bounded::{
            ActiveReservation, BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput,
            BoundedCompiledJoinProbeState, BoundedJoinLeafLocalParquetSourcePrefetch,
            BoundedJoinLeafLocalParquetSourceStream, ReservationTracker, ResidentGpuFrame,
            execute_bounded_compiled_join_build_from_frame, execute_bounded_compiled_join_build_from_resident_frame,
            execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
            execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
            execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
            execute_bounded_generic_gpu_resident_source, execute_bounded_generic_literal_batch_source,
            execute_bounded_generic_local_parquet_source, execute_bounded_generic_unary,
            execute_bounded_generic_unary_from_frame, execute_bounded_generic_unary_from_frame_with_reservation_kind,
            execute_bounded_generic_union, execute_bounded_join_leaf_source,
            execute_compiled_bounded_aggregate_from_frame, execute_compiled_bounded_aggregate_from_stream,
            execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
            execute_final_sort_node_from_stream_with_retry, materialize_bounded_join_leaf_local_parquet_source_stream,
            plan_bounded_join_leaf_local_parquet_source_stream, release_bounded_compiled_join_build_state,
            release_bounded_compiled_join_probe_state, release_reservations_reverse,
            reusable_hash_join_stream_target_output_bytes, try_start_bounded_join_leaf_local_parquet_source_prefetch,
        },
        materialize_bounded_frame_to_host,
        types::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    metrics::QueryMetrics,
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId,
        SinkBoundaryKind,
    },
    plan::{
        FilterNode, JoinNode, JoinNullEquality, JoinSide, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode,
        QueryPlan, ResultSink, SortNode,
    },
    runtime::{ExecutionContext, MemoryControlPolicy, MemoryReservationRequest, MemorySpaceId, ReservationKind},
    source::SourceNode,
};

const OPERATION: &str = "nexus_query_engine.exec.bounded.compiled.tasks";

mod executor;
mod frame_stream;
mod helpers;
mod resource;

// Sibling modules of `tasks/` rely on `use super::*;` to inherit these names.
pub use executor::CompiledGpuTaskExecutor;
pub(in crate::exec::bounded) use frame_stream::ResidentFrameStreamPayload;
#[allow(unused_imports)]
pub(in crate::exec::bounded) use frame_stream::ResidentFrameStreamSource;
pub use frame_stream::{ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness};
use frame_stream::{
    map_dynamic_filter_resident_frame_stream, map_filter_resident_frame_stream, map_projection_resident_frame_stream,
};
use helpers::{
    attach_task_facts, invalid, join_node, join_strategy, join_strategy_needs_build_key_uniqueness_probe, limit_node,
    measure_join_build_key_unique, plan_node, single_input_resource, single_output_resource, sort_node,
    task_diagnostic_resident_resource,
};
pub use resource::CompiledGpuResource;
use resource::CompiledGpuResourcePayload;
