use crate::plan::{JoinNode, JoinSide, JoinType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedKeylessConditionalJoinType {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
}

impl BoundedKeylessConditionalJoinType {
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

pub(crate) fn bounded_root_hash_join_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

pub(crate) fn bounded_root_filtered_join_compatible(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Right
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(crate) fn bounded_root_keyless_conditional_join_compatible(join: &JoinNode) -> bool {
    BoundedKeylessConditionalJoinType::try_from_join(join).is_some()
}

pub(crate) fn bounded_root_keyless_conditional_join_input_subplan_pipeline_compatible(join: &JoinNode) -> bool {
    matches!(BoundedKeylessConditionalJoinType::try_from_join(join), Some(BoundedKeylessConditionalJoinType::Inner))
}

pub(crate) fn bounded_root_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_compatible(
    join: &JoinNode,
) -> bool {
    matches!(BoundedKeylessConditionalJoinType::try_from_join(join), Some(BoundedKeylessConditionalJoinType::Inner))
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(crate) fn bounded_root_projected_cross_join_compatible(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Cross
}

pub(crate) fn bounded_root_mixed_join_compatible(join: &JoinNode) -> bool {
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

pub(crate) fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_root_compatible(
    join: &JoinNode,
) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
}

pub(crate) fn bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_root_compatible(
    join: &JoinNode,
) -> bool {
    join.spec.join_type == JoinType::LeftSemi
        && join.spec.build_side == JoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

pub(crate) fn bounded_root_join_compatible(join: &JoinNode) -> bool {
    bounded_root_hash_join_compatible(join)
        || bounded_root_filtered_join_compatible(join)
        || bounded_root_keyless_conditional_join_compatible(join)
        || bounded_root_projected_cross_join_compatible(join)
        || bounded_root_mixed_join_compatible(join)
}

pub(crate) fn bounded_multi_join_compatible(join: &JoinNode) -> bool {
    bounded_root_hash_join_compatible(join)
        || bounded_root_filtered_join_compatible(join)
        || bounded_multi_join_mixed_keyed_residual_compatible(join)
}

pub(crate) fn bounded_multi_join_mixed_keyed_residual_compatible(join: &JoinNode) -> bool {
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
