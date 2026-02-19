use std::collections::BTreeMap;

use crate::{
    expr::Expr,
    plan::{JoinNode, JoinSide, JoinType, NodeId, QueryPlan, estimated_output_bytes, estimated_output_rows},
};

pub(crate) const LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD: u64 = 64_000_000;
pub(crate) const LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;
pub(crate) const LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
pub(crate) const SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD: u64 = 1_000_000;
pub(crate) const SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD: u64 = 64 * 1024 * 1024;

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
pub(crate) enum KeyedHashJoinStrategy {
    ReusableHashEqui,
    OneShotEqui,
    PartitionedHashEqui,
    PartitionedMixedKeyedResidual,
}

impl KeyedHashJoinStrategy {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ReusableHashEqui => "reusable_hash_equi",
            Self::OneShotEqui => "one_shot_equi",
            Self::PartitionedHashEqui => "partitioned_hash_equi",
            Self::PartitionedMixedKeyedResidual => "partitioned_mixed_keyed_residual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeyedHashJoinStrategyInputFacts {
    pub(crate) estimated_build_rows: Option<u64>,
    pub(crate) observed_build_rows: Option<u64>,
    pub(crate) build_side_bytes: Option<u64>,
    pub(crate) probe_side_bytes: Option<u64>,
    pub(crate) probe_reuse_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum KeyedHashJoinStrategyDecision {
    Selected { strategy: KeyedHashJoinStrategy, facts: BTreeMap<&'static str, String> },
    Rejected { reason: &'static str, facts: BTreeMap<&'static str, String> },
}

impl KeyedHashJoinStrategyDecision {
    pub(crate) fn facts(&self) -> &BTreeMap<&'static str, String> {
        match self {
            Self::Selected { facts, .. } | Self::Rejected { facts, .. } => facts,
        }
    }
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

pub(crate) fn keyed_hash_join_strategy_decision_for_plan(
    plan: &QueryPlan,
    node_id: NodeId,
    join: &JoinNode,
) -> Option<KeyedHashJoinStrategyDecision> {
    if join_residual_requires_post_join_filter(join) {
        return Some(keyed_hash_join_strategy_decision(
            join,
            keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join),
        ));
    }
    if keyed_hash_join_capable(join) {
        return Some(keyed_hash_join_strategy_decision(
            join,
            keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join),
        ));
    }
    if one_shot_equi_join_supported(join) {
        let decision =
            keyed_hash_join_strategy_decision(join, keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join));
        if matches!(
            decision,
            KeyedHashJoinStrategyDecision::Selected {
                strategy: KeyedHashJoinStrategy::OneShotEqui | KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
                ..
            }
        ) {
            return Some(decision);
        }
    }
    None
}

pub(crate) fn keyed_hash_join_strategy_input_facts_for_plan(
    plan: &QueryPlan,
    _node_id: NodeId,
    join: &JoinNode,
) -> KeyedHashJoinStrategyInputFacts {
    let build_input = match join.spec.build_side {
        JoinSide::Left => join.inputs[0],
        JoinSide::Right => join.inputs[1],
    };
    KeyedHashJoinStrategyInputFacts {
        estimated_build_rows: estimated_output_rows(plan, build_input),
        observed_build_rows: None,
        build_side_bytes: estimated_output_bytes(plan, build_input),
        probe_side_bytes: estimated_output_bytes(plan, probe_input_for_join(join)),
        probe_reuse_count: 1,
    }
}

pub(crate) fn keyed_hash_join_strategy_decision(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
) -> KeyedHashJoinStrategyDecision {
    if join_residual_requires_post_join_filter(join) {
        if one_shot_equi_join_supported(join) {
            return KeyedHashJoinStrategyDecision::Selected {
                strategy: KeyedHashJoinStrategy::OneShotEqui,
                facts: strategy_facts(
                    join,
                    input,
                    KeyedHashJoinStrategy::OneShotEqui,
                    Some("residual_requires_post_join_filter"),
                    Some("residual_requires_post_join_filter"),
                    None,
                ),
            };
        }

        return KeyedHashJoinStrategyDecision::Rejected {
            reason: "post_join_residual_filter_requires_one_shot_equi_join",
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::OneShotEqui,
                None,
                Some("residual_requires_post_join_filter"),
                Some("one_shot_equi_requires_keyed_inner_left_or_full_join"),
            ),
        };
    }

    let large_build = input.observed_build_rows.is_some_and(|rows| rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD)
        || input.estimated_build_rows.is_some_and(|rows| rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD)
        || input.build_side_bytes.is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD);
    let small_build = input.observed_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
        || input.estimated_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
        || input.build_side_bytes.is_some_and(|bytes| bytes <= SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD);
    let large_probe = input.probe_side_bytes.is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD);
    let reusable_rejected_reason = if large_build {
        Some("build_side_too_large")
    } else if large_probe && !small_build {
        Some("probe_side_too_large")
    } else {
        None
    };

    let prefer_reusable =
        join.spec.build_artifact_id.is_some() || input.probe_reuse_count > 1 || reusable_rejected_reason.is_none();
    if prefer_reusable {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::ReusableHashEqui,
            facts: strategy_facts(join, input, KeyedHashJoinStrategy::ReusableHashEqui, None, None, None),
        };
    }

    if partitioned_equi_join_supported(join) {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::PartitionedHashEqui,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::PartitionedHashEqui,
                reusable_rejected_reason,
                None,
                None,
            ),
        };
    }

    if partitioned_keyed_residual_join_supported(join) {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
                reusable_rejected_reason,
                None,
                None,
            ),
        };
    }

    if one_shot_equi_join_supported(join) {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::OneShotEqui,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::OneShotEqui,
                reusable_rejected_reason,
                Some("partitioned_hash_equi_unsupported"),
                None,
            ),
        };
    }

    KeyedHashJoinStrategyDecision::Rejected {
        reason: "large_keyed_hash_join_strategy_unsupported",
        facts: strategy_facts(
            join,
            input,
            KeyedHashJoinStrategy::OneShotEqui,
            reusable_rejected_reason,
            Some("partitioned_hash_equi_unsupported"),
            Some("one_shot_equi_requires_keyed_inner_left_or_full_join"),
        ),
    }
}

pub(crate) fn partitioned_equi_join_supported(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

pub(crate) fn partitioned_keyed_residual_join_supported(join: &JoinNode) -> bool {
    keyed_residual_mixed_join_capable(join)
}

pub(crate) fn one_shot_equi_join_supported(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.residual_predicate.is_none() || join.spec.join_type == JoinType::Inner)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

pub(crate) fn join_residual_requires_post_join_filter(join: &JoinNode) -> bool {
    join.spec.residual_predicate.as_ref().is_some_and(expr_requires_post_join_filter)
}

fn expr_requires_post_join_filter(expr: &Expr) -> bool {
    match expr {
        Expr::Case { .. } => true,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Extract { expr, .. }
        | Expr::Alias { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Like { expr, .. } => expr_requires_post_join_filter(expr),
        Expr::Binary { op: crate::expr::BinaryOp::Or, .. } => true,
        Expr::Binary { left, right, .. } => {
            expr_requires_post_join_filter(left) || expr_requires_post_join_filter(right)
        },
        Expr::InList { expr, list, .. } => {
            expr_requires_post_join_filter(expr) || list.iter().any(expr_requires_post_join_filter)
        },
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

fn strategy_facts(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    strategy: KeyedHashJoinStrategy,
    reusable_rejected_reason: Option<&'static str>,
    partitioned_rejected_reason: Option<&'static str>,
    one_shot_rejected_reason: Option<&'static str>,
) -> BTreeMap<&'static str, String> {
    let mut facts = BTreeMap::from([
        ("join_strategy", strategy.as_str().to_string()),
        ("join_type", join.spec.join_type.to_string()),
        ("build_side", join.spec.build_side.to_string()),
        ("key_count", join.spec.keys.len().to_string()),
        ("probe_reuse_count", input.probe_reuse_count.to_string()),
        ("build_artifact_id_present", join.spec.build_artifact_id.is_some().to_string()),
        ("large_keyed_hash_build_row_threshold", LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD.to_string()),
        ("large_keyed_hash_build_bytes_threshold", LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD.to_string()),
        ("large_keyed_hash_probe_bytes_threshold", LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD.to_string()),
        ("small_reusable_hash_build_row_threshold", SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD.to_string()),
        ("small_reusable_hash_build_bytes_threshold", SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD.to_string()),
    ]);
    if let Some(rows) = input.estimated_build_rows {
        facts.insert("estimated_build_rows", rows.to_string());
    }
    if let Some(rows) = input.observed_build_rows {
        facts.insert("observed_build_rows", rows.to_string());
    }
    if let Some(bytes) = input.build_side_bytes {
        facts.insert("build_side_bytes", bytes.to_string());
    }
    if let Some(bytes) = input.probe_side_bytes {
        facts.insert("probe_side_bytes", bytes.to_string());
    }
    if let Some(reason) = reusable_rejected_reason {
        facts.insert("reusable_hash_join_rejected_reason", reason.to_string());
    }
    if let Some(reason) = partitioned_rejected_reason {
        facts.insert("partitioned_hash_join_rejected_reason", reason.to_string());
    }
    if let Some(reason) = one_shot_rejected_reason {
        facts.insert("one_shot_equi_rejected_reason", reason.to_string());
    }
    facts
}

fn probe_input_for_join(join: &JoinNode) -> NodeId {
    match join.spec.build_side {
        JoinSide::Left => join.inputs[1],
        JoinSide::Right => join.inputs[0],
    }
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
