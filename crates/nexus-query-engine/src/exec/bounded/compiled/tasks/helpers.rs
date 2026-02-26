use super::*;

pub(super) fn plan_node(task: &PipelineTask) -> Result<NodeId> {
    task.plan_node().ok_or_else(|| {
        invalid("compiled executable task is missing its plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
    })
}

pub(super) fn single_output_resource(task: &PipelineTask) -> Result<ResidentResourceId> {
    if task.output_resources().len() != 1 {
        return Err(invalid("compiled executable task must declare exactly one output resource")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
            .with_fact("output_resource_count", task.output_resources().len()));
    }
    Ok(task.output_resources()[0])
}

pub(super) fn single_input_resource(task: &PipelineTask) -> Result<ResidentResourceId> {
    if task.input_resources().len() != 1 {
        return Err(invalid("compiled executable task must declare exactly one input resource")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
            .with_fact("input_resource_count", task.input_resources().len()));
    }
    Ok(task.input_resources()[0])
}

pub(super) fn task_diagnostic_resident_resource(task: &PipelineTask) -> Option<ResidentResourceId> {
    if task.output_resources().len() == 1 {
        Some(task.output_resources()[0])
    } else if task.input_resources().len() == 1 {
        Some(task.input_resources()[0])
    } else {
        None
    }
}

pub(super) fn join_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a JoinNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled join task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Join(join) = &node.kind else {
        return Err(invalid("compiled join task does not reference a join plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(join)
}

pub(super) fn join_strategy(
    plan: &QueryPlan,
    node: &JoinNode,
    task: &PipelineTask,
    node_id: NodeId,
    observed_build_rows: u64,
    build_side_bytes: u64,
) -> Result<CompiledJoinStrategySelection> {
    let plan_facts = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, node);
    let strategy_facts = KeyedHashJoinStrategyInputFacts {
        estimated_build_rows: plan_facts.estimated_build_rows,
        observed_build_rows: Some(observed_build_rows),
        build_side_bytes: Some(build_side_bytes),
        probe_side_bytes: plan_facts.probe_side_bytes,
        probe_reuse_count: plan_facts.probe_reuse_count,
        reusable_prebuild_supported: plan_facts.reusable_prebuild_supported,
    };
    CompiledJoinExecutionStrategy::select_from_join_node_with_strategy_facts(node, strategy_facts)
        .map_err(|decision| {
            let (reason, facts) = match decision {
                crate::capability::KeyedHashJoinStrategyDecision::Rejected { reason, facts } => (reason, facts),
                crate::capability::KeyedHashJoinStrategyDecision::Selected { .. } => {
                    unreachable!("selected strategy decisions do not map to strategy-selection errors")
                },
            };
            let mut error = invalid("compiled join task rejected its keyed hash join execution strategy")
                .with_fact("reason", reason)
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str());
            for (key, value) in facts {
                error = error.with_fact(key, value);
            }
            error
        })?
        .ok_or_else(|| {
            invalid("compiled join task references an unsupported join execution strategy")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", node.spec.build_side)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
                .with_fact("output_projection_len", node.spec.output_projection.len())
        })
}

pub(super) fn sort_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a SortNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled final sort task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Sort(sort) = &node.kind else {
        return Err(invalid("compiled final sort task does not reference a sort plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(sort)
}

pub(super) fn limit_node<'a>(plan: &'a QueryPlan, task: &PipelineTask, node_id: NodeId) -> Result<&'a LimitNode> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid("compiled limit task references a missing plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
    })?;
    let PlanNode::Limit(limit) = &node.kind else {
        return Err(invalid("compiled limit task does not reference a limit plan node")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str()));
    };
    Ok(limit)
}

pub(super) fn attach_task_facts(error: QueryError, task: &PipelineTask, node_id: NodeId) -> QueryError {
    error
        .with_fact_if_absent("compiled_task_id", task.id())
        .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
        .with_fact_if_absent("plan_node_id", node_id)
}

pub(super) fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
