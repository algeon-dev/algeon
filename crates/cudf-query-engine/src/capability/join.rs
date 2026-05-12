use crate::plan::{JoinNode, JoinSide, JoinType};

/// Node-local native join execution capability.
///
/// This describes the operator strategy a single join node can use inside a
/// compiled pipeline. It deliberately does not name whole-query bounded
/// families or root/multi-join topology paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum JoinExecutionCapability {
    KeyedHash,
    KeyedFilteredSemiAnti,
    KeylessConditional(KeylessConditionalJoinKind),
    ProjectedCross,
    KeyedResidualMixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KeylessConditionalJoinKind {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
}

impl KeylessConditionalJoinKind {
    pub(crate) fn try_from_join(join: &JoinNode) -> Option<Self> {
        if !join.spec.keys.is_empty()
            || join.spec.residual_predicate.is_none()
            || join.spec.output_projection.is_empty()
        {
            return None;
        }

        match join.spec.join_type {
            JoinType::Inner => Some(Self::Inner),
            JoinType::Left if join.spec.build_side == JoinSide::Right => Some(Self::Left),
            JoinType::Full => Some(Self::Full),
            JoinType::LeftSemi if join.spec.build_side == JoinSide::Left => Some(Self::LeftSemi),
            JoinType::LeftAnti if join.spec.build_side == JoinSide::Left => Some(Self::LeftAnti),
            JoinType::Left | JoinType::Cross => None,
            JoinType::LeftSemi | JoinType::LeftAnti => None,
        }
    }
}

pub(crate) fn join_node_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
    single_join_node_execution_capability(join).or_else(|| multi_join_subtree_join_execution_capability(join))
}

pub(crate) fn single_join_node_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
    if keyed_hash_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedHash);
    }
    if keyed_filtered_semi_anti_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedFilteredSemiAnti);
    }
    if let Some(join_type) = KeylessConditionalJoinKind::try_from_join(join) {
        return Some(JoinExecutionCapability::KeylessConditional(join_type));
    }
    if projected_cross_join_capable(join) {
        return Some(JoinExecutionCapability::ProjectedCross);
    }
    if keyed_residual_mixed_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedResidualMixed);
    }
    None
}

pub(crate) fn multi_join_subtree_join_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
    if keyed_hash_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedHash);
    }
    if keyed_filtered_semi_anti_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedFilteredSemiAnti);
    }
    if multi_join_keyed_residual_mixed_join_capable(join) {
        return Some(JoinExecutionCapability::KeyedResidualMixed);
    }
    None
}

pub(crate) fn keyed_hash_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

pub(crate) fn keyed_filtered_semi_anti_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Right
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(crate) fn projected_cross_join_capable(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Cross
}

pub(crate) fn keyed_residual_mixed_join_capable(join: &JoinNode) -> bool {
    if join.spec.keys.is_empty() || join.spec.residual_predicate.is_none() || join.spec.output_projection.is_empty() {
        return false;
    }

    match join.spec.join_type {
        JoinType::Inner | JoinType::Full => true,
        JoinType::Left => join.spec.build_side == JoinSide::Right,
        JoinType::LeftSemi | JoinType::LeftAnti => {
            join.spec.build_side == JoinSide::Right
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        },
        JoinType::Cross => false,
    }
}

pub(crate) fn multi_join_keyed_residual_mixed_join_capable(join: &JoinNode) -> bool {
    if join.spec.keys.is_empty() || join.spec.residual_predicate.is_none() || join.spec.output_projection.is_empty() {
        return false;
    }

    match join.spec.join_type {
        JoinType::Inner => true,
        JoinType::LeftSemi | JoinType::LeftAnti => {
            join.spec.build_side == JoinSide::Right
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        },
        JoinType::Left | JoinType::Full | JoinType::Cross => false,
    }
}
