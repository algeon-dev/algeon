use super::*;

mod common;
mod compiled;
mod filtered;
mod hash;
mod keyless;
mod leaf;
mod mixed;
mod projected_cross;

pub(in crate::exec::bounded) use compiled::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinProbeState, execute_bounded_compiled_join_build_from_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_probe_from_frame,
};
use filtered::execute_bounded_root_filtered_join;
use hash::{
    BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState, execute_bounded_root_hash_join,
    execute_bounded_root_hash_join_build_from_frame, execute_bounded_root_hash_join_materialize,
    execute_bounded_root_hash_join_probe_from_frame,
};
use keyless::execute_bounded_root_keyless_conditional_join;
pub(super) use leaf::{
    execute_bounded_join_leaf_input, execute_bounded_join_leaf_source, validate_bounded_join_leaf_input_capabilities,
};
use mixed::execute_bounded_root_mixed_join;
use projected_cross::execute_bounded_root_projected_cross_join;

pub(super) fn execute_bounded_root_join_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_root_join_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let join_node_id = plan.sink().input();
    let node = plan
        .node(join_node_id)
        .ok_or_else(|| invalid_node_reference(join_node_id, "bounded root join plan references a missing join node"))?;
    let PlanNode::Join(join) = &node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded root join execution requires the sink input to be a native join",
            "missing_root_join",
        ));
    };
    validate_bounded_join_leaf_input_capabilities(plan, join.inputs[0])?;
    validate_bounded_join_leaf_input_capabilities(plan, join.inputs[1])?;

    let left = execute_bounded_join_leaf_input(context, plan, join.inputs[0], gpu_inputs, &mut reservations, metrics)?;
    let right = execute_bounded_join_leaf_input(context, plan, join.inputs[1], gpu_inputs, &mut reservations, metrics)?;
    let committed = execute_bounded_root_join(context, node.id, join, left, right, &mut reservations, metrics)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

pub(super) fn execute_bounded_root_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if node.is_bounded_root_hash_join_compatible() {
        execute_bounded_root_hash_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_filtered_join_compatible() {
        execute_bounded_root_filtered_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_keyless_conditional_join_compatible() {
        execute_bounded_root_keyless_conditional_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_projected_cross_join_compatible() {
        execute_bounded_root_projected_cross_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_mixed_join_compatible() {
        execute_bounded_root_mixed_join(context, node_id, node, left, right, reservations, metrics)
    } else {
        Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash equi joins, filtered equality semi/anti joins, keyless conditional joins, projected cross joins, and mixed keyed residual joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()))
    }
}
