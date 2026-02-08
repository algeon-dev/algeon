use crate::{
    capability::{
        JoinExecutionCapability, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeyedHashJoinStrategyInputFacts,
        KeylessConditionalJoinKind, join_node_execution_capability, keyed_hash_join_strategy_decision,
        keyed_hash_join_strategy_decision_for_plan,
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
    FilteredSemiAnti,
    KeylessConditional { join_type: KeylessConditionalJoinKind },
    ProjectedCross,
    MixedKeyedResidual,
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

    pub(crate) fn from_join_node_with_strategy_facts(
        join: &JoinNode,
        facts: KeyedHashJoinStrategyInputFacts,
    ) -> std::result::Result<Option<Self>, KeyedHashJoinStrategyDecision> {
        if join_node_execution_capability(join) != Some(JoinExecutionCapability::KeyedHash) {
            return Ok(Self::from_non_keyed_hash_join_node(join));
        }
        Self::from_keyed_hash_strategy_decision(keyed_hash_join_strategy_decision(join, facts)).map(Some)
    }

    fn from_keyed_hash_strategy_decision(
        decision: KeyedHashJoinStrategyDecision,
    ) -> std::result::Result<Self, KeyedHashJoinStrategyDecision> {
        match decision {
            KeyedHashJoinStrategyDecision::Selected { strategy, .. } => match strategy {
                KeyedHashJoinStrategy::ReusableHashEqui => Ok(Self::HashEqui),
                KeyedHashJoinStrategy::OneShotEqui => Ok(Self::OneShotEqui),
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
