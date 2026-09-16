use super::{CompiledGpuTaskExecutor, join_build_key_uniqueness_probe_can_fallback};
use crate::{
    admission::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection},
    capability::{
        KeyedHashJoinStrategyInputFacts, keyed_hash_join_build_is_large, keyed_hash_join_strategy_input_facts_for_plan,
        keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options,
        partitioned_equi_join_supported,
    },
    error::Result,
    exec::bounded::{
        BoundedExecutionAttempt, BoundedJoinBuildStreamObservation, PartitionedJoinBuildCommitment,
        compiled::helpers::{
            join_strategy, join_strategy_from_facts, join_strategy_needs_build_key_uniqueness_probe,
            measure_join_build_key_unique,
        },
        partitioned_join_build_commitment, stateless_streaming_probe_supported,
    },
    pipeline::PipelineTask,
    plan::{JoinNode, JoinSide, NodeId, QueryPlan, estimated_output_bytes},
    runtime::resident::GpuDataFrame,
};

// Size hints the build input stream carries about itself (Parquet footer rows
// and encoded bytes, or an upstream producer's reservation totals). They only
// help pick a strategy and partition count; observed facts still decide.
#[derive(Clone, Copy)]
pub(super) struct JoinBuildStreamEstimate {
    pub(super) rows: Option<u64>,
    pub(super) bytes: Option<u64>,
}

/// Decides whether the rest of a streaming build input can be consumed
/// partition by partition before the whole side is resident.
///
/// The commitment is final once made: large-build evidence only grows with more
/// observed rows and bytes, and under large-build evidence a stateless
/// partitioned equi join is the only strategy this node can take unless a
/// build-key uniqueness probe could still flip a left join to one-shot. Without
/// such evidence the caller keeps accumulating and selects on observed facts.
pub(super) fn settle_partitioned_join_build(
    plan: &QueryPlan,
    task: &PipelineTask,
    node_id: NodeId,
    node: &JoinNode,
    estimate: JoinBuildStreamEstimate,
    execution: &mut BoundedExecutionAttempt<'_>,
    observation: Option<BoundedJoinBuildStreamObservation>,
) -> Result<Option<PartitionedJoinBuildCommitment>> {
    if node.spec.residual_predicate.is_some()
        || !partitioned_equi_join_supported(node)
        || !stateless_streaming_probe_supported(node)
    {
        return Ok(None);
    }
    let plan_facts = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, node);
    let facts = KeyedHashJoinStrategyInputFacts {
        estimated_build_rows: plan_facts.estimated_build_rows.or(estimate.rows),
        observed_build_rows: observation.map(|observation| observation.rows),
        build_side_bytes: observation
            .map(|observation| observation.bytes)
            .or(plan_facts.build_side_bytes)
            .or(estimate.bytes),
        build_key_unique: None,
        ..plan_facts
    };
    let options = plan.keyed_hash_join_strategy_options();
    if !keyed_hash_join_build_is_large(&facts, &options)
        || keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options(node, facts, &options)
    {
        return Ok(None);
    }
    let selection = join_strategy_from_facts(plan, node, task, node_id, facts)?;
    if selection.strategy != CompiledJoinExecutionStrategy::PartitionedHashEqui {
        return Ok(None);
    }
    let build_input = match node.spec.build_side {
        JoinSide::Left => node.inputs[0],
        JoinSide::Right => node.inputs[1],
    };
    let build_basis_bytes =
        [estimated_output_bytes(plan, build_input), estimate.bytes, observation.map(|observation| observation.bytes)]
            .into_iter()
            .flatten()
            .max();
    let commitment =
        partitioned_join_build_commitment(execution, options, build_basis_bytes, plan_facts.probe_side_bytes);
    let mut facts = selection.facts.unwrap_or_default();
    facts.insert("build_consumption", "partitioned_input_stream".to_string());
    facts.insert("partition_count", commitment.partition_count.to_string());
    facts.insert("partition_target_bytes", commitment.partition_target_bytes.to_string());
    facts.insert(
        "build_partition_basis_bytes",
        build_basis_bytes.map_or_else(|| "none".to_string(), |bytes| bytes.to_string()),
    );
    facts.insert("build_chunks_before_commitment", observation.map_or(0, |observation| observation.chunks).to_string());
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
    ) -> Result<CompiledJoinStrategySelection> {
        let observed_build_rows = build_frame.num_rows() as u64;
        let mut build_key_uniqueness_probe_status = None;
        let mut build_key_uniqueness_probe_error_code = None;
        let build_key_unique = if join_strategy_needs_build_key_uniqueness_probe(
            self.plan,
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
        let mut strategy =
            join_strategy(self.plan, node, task, node_id, observed_build_rows, build_side_bytes, build_key_unique)?;
        if let Some(facts) = strategy.facts.as_mut() {
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
        }
        if let Some(facts) = strategy.facts.as_ref() {
            self.execution.metrics().record_join_strategy_decision(node_id, facts);
        }
        Ok(strategy)
    }
}
