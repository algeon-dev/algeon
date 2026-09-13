use std::collections::BTreeMap;

use crate::{
    capability::{
        JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
        KeyedHashJoinStrategyOptions, KeylessConditionalJoinKind, join_node_execution_capability,
        keyed_hash_join_strategy_decision_for_plan, keyed_hash_join_strategy_decision_with_options,
        one_shot_equi_join_supported, partitioned_keyed_residual_join_supported,
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
#[doc(hidden)]
pub enum CompiledJoinExecutionStrategy {
    HashEqui,
    OneShotEqui,
    PartitionedHashEqui,
    PartitionedMixedKeyedResidual,
    FilteredSemiAnti,
    KeylessConditional { join_type: KeylessConditionalJoinKind },
    KeylessOuterCross,
    ProjectedCross,
    MixedKeyedResidual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct CompiledJoinStrategySelection {
    #[doc(hidden)]
    pub strategy: CompiledJoinExecutionStrategy,
    #[doc(hidden)]
    pub facts: Option<BTreeMap<&'static str, String>>,
}

impl CompiledJoinExecutionStrategy {
    #[doc(hidden)]
    pub fn from_plan_join_node(
        plan: &QueryPlan,
        node_id: NodeId,
        join: &JoinNode,
    ) -> std::result::Result<Option<Self>, KeyedHashJoinStrategyDecision> {
        match keyed_hash_join_strategy_decision_for_plan(plan, node_id, join) {
            Some(decision) => Self::from_keyed_hash_strategy_decision(decision).map(Some),
            None => Ok(Self::from_non_keyed_hash_join_node(join)),
        }
    }

    #[doc(hidden)]
    pub fn select_from_join_node_with_strategy_facts(
        join: &JoinNode,
        facts: KeyedHashJoinStrategyInputFacts,
        options: &KeyedHashJoinStrategyOptions,
    ) -> std::result::Result<Option<CompiledJoinStrategySelection>, KeyedHashJoinStrategyDecision> {
        if join_node_execution_capability(join) != Some(JoinExecutionCapability::KeyedHash) {
            if one_shot_equi_join_supported(join) || partitioned_keyed_residual_join_supported(join) {
                match keyed_hash_join_strategy_decision_with_options(join, facts, options) {
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
                    decision @ KeyedHashJoinStrategyDecision::Selected {
                        strategy: KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
                        ..
                    } => return Self::select_from_keyed_hash_strategy_decision(decision).map(Some),
                    rejected @ KeyedHashJoinStrategyDecision::Rejected { .. } => return Err(rejected),
                }
            }
            return Ok(Self::from_non_keyed_hash_join_node(join)
                .map(|strategy| CompiledJoinStrategySelection { strategy, facts: None }));
        }
        Self::select_from_keyed_hash_strategy_decision(keyed_hash_join_strategy_decision_with_options(
            join, facts, options,
        ))
        .map(Some)
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
                KeyedHashJoinStrategy::PartitionedMixedKeyedResidual => Ok(CompiledJoinStrategySelection {
                    strategy: Self::PartitionedMixedKeyedResidual,
                    facts: Some(facts),
                }),
            },
            rejected @ KeyedHashJoinStrategyDecision::Rejected { .. } => Err(rejected),
        }
    }

    fn from_non_keyed_hash_join_node(join: &JoinNode) -> Option<Self> {
        match join_node_execution_capability(join)? {
            JoinExecutionCapability::KeyedHash => None,
            JoinExecutionCapability::KeyedFilteredSemiAnti => Some(Self::FilteredSemiAnti),
            JoinExecutionCapability::KeylessConditional(join_type) => Some(Self::KeylessConditional { join_type }),
            JoinExecutionCapability::KeylessOuterCross => Some(Self::KeylessOuterCross),
            JoinExecutionCapability::ProjectedCross => Some(Self::ProjectedCross),
            JoinExecutionCapability::KeyedResidualMixed => Some(Self::MixedKeyedResidual),
        }
    }
}
