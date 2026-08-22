use super::{source::record_bounded_parquet_source_metrics, *};
use crate::{mode::StatisticalAggregateExecutionMode, runtime::ExecutionBoundaryPolicy};

// Bounded execution ownership boundary:
// - pipeline: the bounded execution facade.
// - compiled: the shared compiled pipeline runtime.
// - remaining modules: bounded runtime/operator support; none owns a separate
//   admission or reporting mode.
mod aggregate;
mod attempt;
pub(in crate::exec) mod chunks;
mod compiled;
mod facts;
mod final_sort;
mod generic;
pub(in crate::exec) mod join;
mod partition_row_policy;
mod partition_target_policy;
pub(in crate::exec) mod pipeline;
mod reservation;
mod source_batching_policy;

use aggregate::*;
pub use aggregate::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
pub(in crate::exec::bounded) use aggregate::{
    SourceAdjacentAggregateChunkCapHints, apply_source_adjacent_aggregate_chunk_caps,
};
pub(in crate::exec) use attempt::BoundedExecutionAttempt;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
use final_sort::*;
#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) use generic::execute_bounded_graph_algorithm_from_inputs;
pub(in crate::exec::bounded) use generic::{
    execute_bounded_edge_normalize_from_input, execute_bounded_generic_projection,
    execute_bounded_generic_projection_from_frame, execute_bounded_generic_unary,
    execute_bounded_generic_unary_from_frame, execute_bounded_generic_unary_from_frame_with_reservation_kind,
};
use generic::{
    execute_bounded_generic_gpu_resident_source, execute_bounded_generic_literal_batch_source,
    execute_bounded_generic_local_parquet_source, execute_bounded_generic_query_plan,
    execute_bounded_generic_unary_consuming_input_with_reservation_kind, execute_bounded_generic_union,
    release_reservations_reverse,
};
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream,
    execute_bounded_compiled_join_build_from_frame, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
    execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
    execute_bounded_join_leaf_source, materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream, release_bounded_compiled_join_build_state,
    release_bounded_compiled_join_probe_state, reusable_hash_join_stream_target_output_bytes,
    stream_bounded_join_leaf_local_parquet_source_chunks, try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
pub use join::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE,
    join_output_upper_bound_bytes,
};
pub(in crate::exec::bounded) use partition_row_policy::BoundedPartitionRowPolicy;
pub(in crate::exec::bounded) use partition_target_policy::BoundedPartitionTargetPolicy;
use pipeline::try_execute_bounded_pipeline;
pub(in crate::exec::bounded) use reservation::reconcile_work_reservation_to_output;
pub(in crate::exec) use reservation::{ActiveReservation, CommittedPartial, ReservationTracker, ResidentGpuFrame};
pub(in crate::exec::bounded) use source_batching_policy::BoundedSourceBatchingPolicy;

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.execute_query_plan")?;
    if let Some(output) = try_execute_bounded_pipeline(context, plan, gpu_inputs, metrics, observation_scope)? {
        return Ok(output);
    }

    if plan.contains_vector_algorithm() {
        return Err(bounded_vector_algorithm_error(plan));
    }

    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics, observation_scope);
    }

    let bounded = match BoundedPlan::analyze(plan) {
        Ok(bounded) => bounded,
        Err(_error) => {
            return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics, observation_scope);
        },
    };
    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics, observation_scope)?;
    let frame = execute_bounded_aggregate_to_frame(&mut execution, &bounded)?;
    let frame = execute_post_aggregate_stages(&mut execution, &bounded, frame)?;
    let frame = execute_final_sort_limit_with_retry(&mut execution, &bounded, frame)?;
    let frame = execute_post_sort_projections(&mut execution, &bounded, frame)?;
    let mut frame = execute_standalone_limit(&mut execution, &bounded, frame)?;

    record_device_memory_observation(&frame.frame, execution.metrics_mut())?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let live_reservation = frame.reservation.transfer_grant_live_reservation();
            let output_frame = frame
                .frame
                .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(plan));
            let handle = match GpuResultHandle::new_with_live_reservation(
                execution.context(),
                output_frame,
                Arc::clone(plan.output_schema()),
                live_reservation,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    execution.release(frame.reservation);
                    return Err(error);
                },
            };
            execution.release(frame.reservation);
            debug_assert_eq!(execution.reservations().reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                &mut execution,
                frame,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
            )?;
            debug_assert_eq!(execution.reservations().reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn bounded_vector_algorithm_error(plan: &QueryPlan) -> QueryError {
    let (node_id, algorithm) = first_vector_algorithm(plan);
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.bounded.vector_algorithm",
        "bounded vector execution requires a lower-level cuVS peak-memory preflight contract",
    )
    .with_fact("execution_mode", "bounded")
    .with_fact("node_id", node_id)
    .with_fact("operator", "vector_algorithm")
    .with_fact("algorithm", algorithm)
    .with_fact("rejection_reason", "blocked_lower_level_contract")
    .with_fact("contract_status", "blocked_by_lower_level")
    .with_fact("memory_boundary", crate::plan::VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY)
    .with_fact("required_contract", crate::plan::VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT)
}

fn first_vector_algorithm(plan: &QueryPlan) -> (NodeId, &'static str) {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::VectorAlgorithm(vector) => Some((node.id, vector.spec.algorithm.as_str())),
            _ => None,
        })
        .expect("contains_vector_algorithm guarantees a vector node")
}

pub(super) fn preflight_bounded_join_plan(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<()> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        // A frontend-owned incremental D2H cursor executes through a physical
        // GPU-resident sink, but runtime rejection facts remain a contract with
        // that frontend. Project them from the retained caller-visible host
        // sink so the explicit materialization reservation is not erased.
        let rejection_contract_plan = plan.clone_with_caller_visible_sink_contract();
        let plan = &rejection_contract_plan;
        let contract = BoundedJoinReservationContract;
        let mut error = unsupported_bounded_shape(
            bounded_join_unsupported_message(contract),
            "bounded_join_plan_shape_unsupported",
        );
        let mut projection_error_source = None;
        let rejection_projection =
            match crate::admission::bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
                plan,
                statistical_aggregate_execution_mode,
            ) {
                Ok(Some(projection)) => Some(projection),
                Ok(None) => return Ok(()),
                Err(projection_error) => {
                    projection_error_source = Some(projection_error);
                    None
                },
            };
        let mut facts = crate::admission::bounded_join_unsupported_shape_facts_with_projection(
            plan,
            boundary_policy,
            rejection_projection.as_ref(),
        );
        if let Some(projection_error) = projection_error_source.as_ref() {
            crate::admission::insert_bounded_join_rejection_projection_error_facts(&mut facts, projection_error);
        }
        for (key, value) in facts {
            error = error.with_fact(key, value);
        }
        if let Some(projection_error) = projection_error_source {
            error = error.with_source(projection_error);
        }
        return Err(error);
    }

    Ok(())
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    preflight_bounded_join_plan(plan, context.boundary_policy(), context.statistical_aggregate_execution_mode())
}

fn bounded_join_unsupported_message(contract: BoundedJoinReservationContract) -> String {
    format!(
        "bounded native join execution requires bounded capability admission: {}",
        contract.supported_shape_requirement()
    )
}
