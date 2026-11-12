use query_planning::{
    capability::{
        JoinStrategy, JoinStrategyChoice, JoinStrategyEvidence, KeyedHashJoinStrategyInputFacts,
        KeyedHashJoinStrategyOptions, ObservedBuildRows, keyed_hash_join_build_is_large,
        keyed_hash_join_observed_build_is_large, keyed_hash_join_strategy_input_facts_for_plan,
        keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness,
    },
    error::Result,
    pipeline::PipelineTask,
    plan::{JoinNode, JoinSide, NodeId, QueryPlan, estimated_output_bytes},
};
use query_runtime::runtime::resident::GpuDataFrame;

use super::{CompiledGpuTaskExecutor, join_build_key_uniqueness_probe_can_fallback};
use crate::exec::bounded::{
    BoundedExecutionAttempt, BoundedJoinBuildStreamObservation, BoundedPartitionTargetPolicy,
    PartitionedJoinBuildCommitment,
    compiled::helpers::{
        join_node, join_strategy_needs_build_key_uniqueness_probe, measure_join_build_key_unique, refine_join_strategy,
        task_join_plan,
    },
    partitioned_join_build_commitment, stateless_streaming_probe_supported,
};

// Size hints the build input stream carries about itself (Parquet footer rows
// and encoded bytes, or an upstream producer's byte hints). Strategy and
// partition count also incorporate observed rows and device bytes.
#[derive(Clone, Copy)]
pub(super) struct JoinBuildStreamEstimate {
    pub(super) rows: Option<u64>,
    pub(super) bytes: Option<u64>,
}

/// Decides, after each build chunk, whether the rest of a streaming build input
/// is consumed partition by partition before the whole side is resident.
///
/// A large plan estimate alone does not commit, because estimates bound the
/// build from above. The build commits when its size evidence allows a large build
/// and either the observed rows or bytes reach the large-build thresholds, or
/// the held chunks reach
/// `BoundedPartitionTargetPolicy::undecided_join_build_hold_bytes`. In the
/// second case the build is taken as large because holding more whole chunks
/// would pass that share of the grant. A build that never commits is
/// concatenated and selected on its complete observation.
///
/// The commitment is final once made: large-build evidence only grows with more
/// observed rows and bytes, and under large-build evidence a stateless
/// partitioned equi join is the only strategy this node can take unless a
/// build-key uniqueness probe could still flip a left join to one-shot.
pub(super) fn settle_partitioned_join_build(
    plan: &QueryPlan,
    options: KeyedHashJoinStrategyOptions,
    task: &PipelineTask,
    node_id: NodeId,
    estimate: JoinBuildStreamEstimate,
    execution: &mut BoundedExecutionAttempt<'_>,
    observation: BoundedJoinBuildStreamObservation,
) -> Result<Option<PartitionedJoinBuildCommitment>> {
    let node = join_node(plan, task, node_id)?;
    let join_plan = task_join_plan(task, node_id)?;
    if !join_plan.allowed().contains(&JoinStrategy::PartitionedHashEqui) || !stateless_streaming_probe_supported(node) {
        return Ok(None);
    }
    let plan_facts = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, node);
    let estimated_build_bytes = plan_facts.build_side_bytes.or(estimate.bytes);
    let facts = KeyedHashJoinStrategyInputFacts {
        estimated_build_rows: plan_facts.estimated_build_rows.or(estimate.rows),
        observed_build_rows: ObservedBuildRows::Partial(observation.rows),
        build_side_bytes: Some(estimated_build_bytes.map_or(observation.bytes, |bytes| bytes.max(observation.bytes))),
        build_key_unique: None,
        ..plan_facts
    };
    let observed_large = keyed_hash_join_observed_build_is_large(observation.rows, observation.bytes, &options);
    let hold_exhausted = observation.bytes
        >= BoundedPartitionTargetPolicy::default().undecided_join_build_hold_bytes(execution.device_budget_bytes());
    if !keyed_hash_join_build_is_large(&facts, &options)
        || !(observed_large || hold_exhausted)
        || keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
            join_plan.form(),
            node,
            facts,
            &options,
        )
    {
        return Ok(None);
    }
    let choice = refine_join_strategy(&options, node, task, node_id, facts)?;
    if choice.strategy() != JoinStrategy::PartitionedHashEqui {
        return Ok(None);
    }
    let build_input = match node.spec.build_side {
        JoinSide::Left => node.inputs[0],
        JoinSide::Right => node.inputs[1],
    };
    let build_basis_bytes = [estimated_output_bytes(plan, build_input), estimate.bytes, Some(observation.bytes)]
        .into_iter()
        .flatten()
        .max();
    let commitment =
        partitioned_join_build_commitment(execution, options, build_basis_bytes, plan_facts.probe_side_bytes);
    let mut facts = choice.evidence().map(JoinStrategyEvidence::facts).unwrap_or_default();
    facts.insert("build_consumption", "partitioned_input_stream".to_string());
    facts.insert("partition_count", commitment.partition_count.to_string());
    facts.insert("partition_target_bytes", commitment.partition_target_bytes.to_string());
    facts.insert(
        "build_partition_basis_bytes",
        build_basis_bytes.map_or_else(|| "none".to_string(), |bytes| bytes.to_string()),
    );
    facts.insert("build_chunks_before_commitment", observation.chunks.to_string());
    execution.metrics().record_join_strategy_decision(node_id, &facts);
    Ok(Some(commitment))
}

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn select_join_build_strategy(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        build_frame: &GpuDataFrame,
        build_side_bytes: u64,
    ) -> Result<JoinStrategyChoice> {
        let observed_build_rows = build_frame.num_rows() as u64;
        let mut build_key_uniqueness_probe_status = None;
        let mut build_key_uniqueness_probe_error_code = None;
        let build_key_unique = if join_strategy_needs_build_key_uniqueness_probe(
            self.plan,
            &self.policy.keyed_hash_join,
            task_join_plan(task, node_id)?.form(),
            node,
            node_id,
            observed_build_rows,
            build_side_bytes,
        ) {
            match measure_join_build_key_unique(&mut self.execution, node, build_frame) {
                Ok(unique) => {
                    build_key_uniqueness_probe_status = Some("measured");
                    Some(unique)
                },
                Err(error) if join_build_key_uniqueness_probe_can_fallback(&error) => {
                    build_key_uniqueness_probe_status = Some("fallback_temporary_failure");
                    build_key_uniqueness_probe_error_code = Some(error.code().as_str().to_string());
                    None
                },
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let strategy_facts = KeyedHashJoinStrategyInputFacts {
            observed_build_rows: ObservedBuildRows::Complete(observed_build_rows),
            build_side_bytes: Some(build_side_bytes),
            build_key_unique,
            ..keyed_hash_join_strategy_input_facts_for_plan(self.plan, node_id, node)
        };
        let choice = refine_join_strategy(&self.policy.keyed_hash_join, node, task, node_id, strategy_facts)?;
        if let Some(evidence) = choice.evidence() {
            let mut facts = evidence.facts();
            facts.insert(
                "build_key_unique",
                build_key_unique.map_or_else(|| "none".to_string(), |unique| unique.to_string()),
            );
            facts.insert(
                "build_key_uniqueness_probe_status",
                build_key_uniqueness_probe_status.unwrap_or("none").to_string(),
            );
            facts.insert(
                "build_key_uniqueness_probe_error_code",
                build_key_uniqueness_probe_error_code.unwrap_or_else(|| "none".to_string()),
            );
            self.execution.metrics().record_join_strategy_decision(node_id, &facts);
        }
        Ok(choice)
    }
}
