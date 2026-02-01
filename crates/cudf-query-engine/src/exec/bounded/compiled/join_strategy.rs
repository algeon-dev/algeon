use crate::{
    bounded::{BoundedJoinNodeCapability, BoundedKeylessConditionalJoinType, bounded_join_node_capability},
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
    KeylessConditional { join_type: BoundedKeylessConditionalJoinType },
    ProjectedCross,
    MixedKeyedResidual,
}

impl CompiledJoinExecutionStrategy {
    pub(crate) fn from_join_node(join: &JoinNode) -> Option<Self> {
        match bounded_join_node_capability(join)? {
            BoundedJoinNodeCapability::KeyedHash => Some(Self::HashEqui),
            BoundedJoinNodeCapability::KeyedFilteredSemiAnti => Some(Self::FilteredSemiAnti),
            BoundedJoinNodeCapability::KeylessConditional(join_type) => Some(Self::KeylessConditional { join_type }),
            BoundedJoinNodeCapability::ProjectedCross => Some(Self::ProjectedCross),
            BoundedJoinNodeCapability::KeyedResidualMixed => Some(Self::MixedKeyedResidual),
        }
    }
}
