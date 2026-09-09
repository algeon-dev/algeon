use super::{CompiledGpuTaskExecutor, join_build_key_uniqueness_probe_can_fallback};
use crate::{
    admission::CompiledJoinStrategySelection,
    error::Result,
    exec::bounded::compiled::helpers::{
        join_strategy, join_strategy_needs_build_key_uniqueness_probe, measure_join_build_key_unique,
    },
    pipeline::PipelineTask,
    plan::{JoinNode, NodeId},
    runtime::resident::GpuDataFrame,
};

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
            self.execution.metrics_mut().record_join_strategy_decision(node_id, facts);
        }
        Ok(strategy)
    }
}
