use std::collections::BTreeMap;

use crate::{
    capability::{
        JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
        KeylessConditionalJoinKind, join_node_execution_capability, keyed_hash_join_strategy_decision,
        keyed_hash_join_strategy_decision_for_plan, one_shot_equi_join_supported,
    },
    plan::{JoinNode, NodeId, QueryPlan},
};

/// DataFusion-free strategy metadata for compiled join tasks.
///
/// This is intentionally node-local capability metadata, not a bounded
/// family/path selector. The scheduler still executes task edges and barriers;
/// the strategy only tells the task interpreter which retained state shape a
/// join node produces.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompiledJoinExecutionStrategy {
    HashEqui,
    OneShotEqui,
    PartitionedHashEqui,
    FilteredSemiAnti,
    KeylessConditional { join_type: KeylessConditionalJoinKind },
    ProjectedCross,
    MixedKeyedResidual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledJoinStrategySelection {
    pub(crate) strategy: CompiledJoinExecutionStrategy,
    pub(crate) facts: Option<BTreeMap<&'static str, String>>,
}

impl CompiledJoinExecutionStrategy {
    pub(crate) fn from_plan_join_node(
        plan: &QueryPlan,
        node_id: NodeId,
        join: &JoinNode,
    ) -> std::result::Result<Option<Self>, KeyedHashJoinStrategyDecision> {
        match keyed_hash_join_strategy_decision_for_plan(plan, node_id, join) {
            Some(decision) => Self::from_keyed_hash_strategy_decision(decision).map(Some),
            None => Ok(Self::from_non_keyed_hash_join_node(join)),
        }
    }

    pub(crate) fn select_from_join_node_with_strategy_facts(
        join: &JoinNode,
        facts: KeyedHashJoinStrategyInputFacts,
    ) -> std::result::Result<Option<CompiledJoinStrategySelection>, KeyedHashJoinStrategyDecision> {
        if join_node_execution_capability(join) != Some(JoinExecutionCapability::KeyedHash) {
            if one_shot_equi_join_supported(join) {
                match keyed_hash_join_strategy_decision(join, facts) {
                    decision @ KeyedHashJoinStrategyDecision::Selected {
                        strategy: KeyedHashJoinStrategy::OneShotEqui,
                        ..
                    } => return Self::select_from_keyed_hash_strategy_decision(decision).map(Some),
                    KeyedHashJoinStrategyDecision::Selected {
                        strategy: KeyedHashJoinStrategy::ReusableHashEqui,
                        ..
                    } => {},
                    KeyedHashJoinStrategyDecision::Selected {
                        strategy: KeyedHashJoinStrategy::PartitionedHashEqui,
                        ..
                    } => {},
                    rejected @ KeyedHashJoinStrategyDecision::Rejected { .. } => return Err(rejected),
                }
            }
            return Ok(Self::from_non_keyed_hash_join_node(join)
                .map(|strategy| CompiledJoinStrategySelection { strategy, facts: None }));
        }
        Self::select_from_keyed_hash_strategy_decision(keyed_hash_join_strategy_decision(join, facts)).map(Some)
    }

    fn from_keyed_hash_strategy_decision(
        decision: KeyedHashJoinStrategyDecision,
    ) -> std::result::Result<Self, KeyedHashJoinStrategyDecision> {
        Self::select_from_keyed_hash_strategy_decision(decision).map(|selection| selection.strategy)
    }

    fn select_from_keyed_hash_strategy_decision(
        decision: KeyedHashJoinStrategyDecision,
    ) -> std::result::Result<CompiledJoinStrategySelection, KeyedHashJoinStrategyDecision> {
        match decision {
            KeyedHashJoinStrategyDecision::Selected { strategy, facts } => match strategy {
                KeyedHashJoinStrategy::ReusableHashEqui => {
                    Ok(CompiledJoinStrategySelection { strategy: Self::HashEqui, facts: Some(facts) })
                },
                KeyedHashJoinStrategy::OneShotEqui => {
                    Ok(CompiledJoinStrategySelection { strategy: Self::OneShotEqui, facts: Some(facts) })
                },
                KeyedHashJoinStrategy::PartitionedHashEqui => {
                    Ok(CompiledJoinStrategySelection { strategy: Self::PartitionedHashEqui, facts: Some(facts) })
                },
            },
            rejected @ KeyedHashJoinStrategyDecision::Rejected { .. } => Err(rejected),
        }
    }

    fn from_non_keyed_hash_join_node(join: &JoinNode) -> Option<Self> {
        match join_node_execution_capability(join)? {
            JoinExecutionCapability::KeyedHash => None,
            JoinExecutionCapability::KeyedFilteredSemiAnti => Some(Self::FilteredSemiAnti),
            JoinExecutionCapability::KeylessConditional(join_type) => Some(Self::KeylessConditional { join_type }),
            JoinExecutionCapability::ProjectedCross => Some(Self::ProjectedCross),
            JoinExecutionCapability::KeyedResidualMixed => Some(Self::MixedKeyedResidual),
        }
    }
}
