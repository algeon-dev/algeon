use crate::{
    capability::{JoinExecutionCapability, KeylessConditionalJoinKind, join_node_execution_capability},
    plan::JoinNode,
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
    FilteredSemiAnti,
    KeylessConditional { join_type: KeylessConditionalJoinKind },
    ProjectedCross,
    MixedKeyedResidual,
}

impl CompiledJoinExecutionStrategy {
    pub(crate) fn from_join_node(join: &JoinNode) -> Option<Self> {
        match join_node_execution_capability(join)? {
            JoinExecutionCapability::KeyedHash => Some(Self::HashEqui),
            JoinExecutionCapability::KeyedFilteredSemiAnti => Some(Self::FilteredSemiAnti),
            JoinExecutionCapability::KeylessConditional(join_type) => Some(Self::KeylessConditional { join_type }),
            JoinExecutionCapability::ProjectedCross => Some(Self::ProjectedCross),
            JoinExecutionCapability::KeyedResidualMixed => Some(Self::MixedKeyedResidual),
        }
    }
}
