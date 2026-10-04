use std::time::Instant;

use cudf::{DuplicateKeepOption, frame::dataframe::DropDuplicatesOptions};

use crate::{
    admission::JoinStrategyChoice,
    capability::{
        JoinForm, KeyedHashJoinStrategyInputFacts, KeyedHashJoinStrategyOptions,
        keyed_hash_join_strategy_input_facts_for_plan,
        keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness,
    },
    error::{QueryError, QueryErrorCode, Result},
    exec::{attach_budget_token_failure_facts, bounded::BoundedExecutionAttempt, execution_budget_token_for_boundary},
    native_error::cudf_query_error,
    pipeline::{JoinPlan, PipelineTask, ResidentResourceId},
    plan::{JoinNode, JoinSide, LimitNode, NodeId, PlanNode, QueryPlan, SortNode},
    runtime::resident::GpuDataFrame,
};

const OPERATION: &str = "query_engine.exec.bounded.compiled.tasks";

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

pub(super) fn task_join_plan(task: &PipelineTask, node_id: NodeId) -> Result<&JoinPlan> {
    task.join_plan().ok_or_else(|| {
        invalid("compiled join task carries no join plan")
            .with_fact("compiled_task_id", task.id())
            .with_fact("plan_node_id", node_id)
            .with_fact("pipeline_task_kind", task.kind().as_str())
    })
}

/// Chooses the join strategy for observed facts within the task's compiled
/// join plan.
pub(super) fn refine_join_strategy(
    options: &KeyedHashJoinStrategyOptions,
    node: &JoinNode,
    task: &PipelineTask,
    node_id: NodeId,
    strategy_facts: KeyedHashJoinStrategyInputFacts,
) -> Result<JoinStrategyChoice> {
    task_join_plan(task, node_id)?.refine(node, strategy_facts, options).map_err(|error| {
        error
            .with_fact_if_absent("compiled_task_id", task.id())
            .with_fact_if_absent("plan_node_id", node_id)
            .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
    })
}

pub(super) fn join_strategy_needs_build_key_uniqueness_probe(
    plan: &QueryPlan,
    options: &KeyedHashJoinStrategyOptions,
    form: JoinForm,
    node: &JoinNode,
    node_id: NodeId,
    observed_build_rows: u64,
    build_side_bytes: u64,
) -> bool {
    let plan_facts = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, node);
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
        form,
        node,
        KeyedHashJoinStrategyInputFacts {
            observed_build_rows: Some(observed_build_rows),
            build_side_bytes: Some(build_side_bytes),
            build_key_unique: None,
            ..plan_facts
        },
        options,
    )
}

pub(super) fn measure_join_build_key_unique(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build_frame: &GpuDataFrame,
) -> Result<bool> {
    let key_columns = node
        .spec
        .keys
        .iter()
        .map(|key| match node.spec.build_side {
            JoinSide::Left => key.left.name().to_string(),
            JoinSide::Right => key.right.name().to_string(),
        })
        .collect::<Vec<_>>();
    let key_refs = key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let token = execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "join_build_key_uniqueness_probe",
        "query_engine.bounded.join.build_key_uniqueness_probe.budget_token",
        "failed to create join build-key uniqueness probe budget token",
    )?;
    let overrides = token.exec();
    let timer = Instant::now();
    let keys = match build_frame.frame().select(key_refs.clone()) {
        Ok(keys) => keys,
        Err(error) => {
            execution.metrics().add_operator_time(timer.elapsed());
            return Err(cudf_query_error(
                "query_engine.bounded.join.build_key_uniqueness_probe.input",
                "failed to project join build-key uniqueness probe columns",
                error,
            ));
        },
    };
    let unique_keys = match keys.on(overrides).drop_duplicates(
        DropDuplicatesOptions::builder().subset(key_refs.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
    ) {
        Ok(unique_keys) => unique_keys,
        Err(error) => {
            execution.metrics().add_operator_time(timer.elapsed());
            return Err(attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.join.build_key_uniqueness_probe.distinct_keys",
                    "failed to compute join build-key uniqueness",
                    error,
                ),
                &token,
            ));
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());
    Ok(unique_keys.num_rows() == build_frame.num_rows())
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
