//! Join-strategy capability analysis.
//!
//! Classifies each [`JoinNode`] into a [`KeyedHashJoinStrategyDecision`] using
//! the byte/row thresholds defined as crate-level constants. Decisions are
//! consumed by capability admission, the bounded executor's strategy
//! selection, and the join-strategy metric rows.

use std::collections::BTreeMap;

use arrow::datatypes::DataType;

use crate::{
    expr::{BinaryOp, Expr},
    plan::{JoinNode, JoinSide, JoinType, NodeId, QueryPlan, estimated_output_bytes, estimated_output_rows},
};

/// Past this build scale, reusable hash residency is risky enough to prefer partitioning.
pub(crate) const LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD: u64 = 64_000_000;
/// Byte-size large-build gate used alongside row evidence.
pub(crate) const LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;
/// Large probes can turn one reusable build into too much gather/output pressure.
pub(crate) const LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
/// Builds below this row scale are cheap enough to prebuild even for one probe.
pub(crate) const SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD: u64 = 1_000_000;
/// Byte-size small-build gate used alongside row evidence.
pub(crate) const SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD: u64 = 64 * 1024 * 1024;
/// Small probes avoid the scheduling overhead of the partitioned path.
pub(crate) const SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD: u64 = 256 * 1024 * 1024;
/// Large-but-contained right builds can still be cheaper than partitioning when
/// a left join's preserved probe side is already small.
pub(crate) const MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD: u64 = 128_000_000;
/// Byte guard paired with [`MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD`].
pub(crate) const MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;

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
    KeylessOuterCross,
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
    pub(crate) estimated_output_rows: Option<u64>,
    pub(crate) estimated_output_bytes: Option<u64>,
    pub(crate) build_key_unique: Option<bool>,
    pub(crate) probe_reuse_count: u64,
    pub(crate) reusable_prebuild_supported: bool,
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
            JoinType::Left | JoinType::Cross | JoinType::LeftMark => None,
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
    if keyless_outer_cross_join_capable(join) {
        return Some(JoinExecutionCapability::KeylessOuterCross);
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
    let input = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join);
    if let Some(decision) = partitioned_mixed_keyed_residual_strategy_decision(join, input) {
        return Some(decision);
    }
    if one_shot_equi_join_supported(join) {
        let decision = keyed_hash_join_strategy_decision(join, input);
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
    node_id: NodeId,
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
        estimated_output_rows: estimated_output_rows(plan, node_id),
        estimated_output_bytes: estimated_output_bytes(plan, node_id),
        build_key_unique: None,
        probe_reuse_count: 1,
        reusable_prebuild_supported: reusable_hash_prebuild_supported_for_plan(plan, join),
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
    let small_build = !large_build
        && (input.observed_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
            || input.estimated_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
            || input.build_side_bytes.is_some_and(|bytes| bytes <= SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD));
    let build_size_known =
        input.observed_build_rows.is_some() || input.estimated_build_rows.is_some() || input.build_side_bytes.is_some();
    let large_probe = input.probe_side_bytes.is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD);
    let small_probe =
        !large_probe && input.probe_side_bytes.is_some_and(|bytes| bytes <= SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD);
    let single_use_without_artifact = join.spec.build_artifact_id.is_none() && input.probe_reuse_count <= 1;
    let one_shot_contained_left_join_build = input
        .observed_build_rows
        .or(input.estimated_build_rows)
        .is_some_and(|rows| rows <= MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD)
        && input.build_side_bytes.is_some_and(|bytes| bytes <= MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD);
    let one_shot_contained_left_join_output = input.build_key_unique == Some(true);
    let reusable_rejected_reason = if !input.reusable_prebuild_supported {
        Some("build_key_prebuild_reservation_unsupported")
    } else if large_build {
        Some("build_side_too_large")
    } else if large_probe && !small_build {
        Some("probe_side_too_large")
    } else if input.probe_side_bytes.is_none()
        && input.build_side_bytes.is_some()
        && join.spec.build_side == JoinSide::Left
        && !small_build
    {
        Some("probe_side_unestimated")
    } else if single_use_without_artifact && build_size_known && !small_build {
        Some("single_use_build_not_small")
    } else {
        None
    };

    if let Some(decision) = partitioned_mixed_keyed_residual_strategy_decision(join, input) {
        return decision;
    }

    if one_shot_equi_join_supported(join)
        && join.spec.join_type == JoinType::Left
        && join.spec.build_side == JoinSide::Right
        && input.reusable_prebuild_supported
        && single_use_without_artifact
        && !small_build
        && small_probe
        && one_shot_contained_left_join_build
        && one_shot_contained_left_join_output
    {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::OneShotEqui,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::OneShotEqui,
                reusable_rejected_reason,
                Some("small_preserved_probe_prefers_one_shot_left_join"),
                None,
            ),
        };
    }

    let prefer_reusable = join.spec.build_artifact_id.is_some()
        || input.probe_reuse_count > 1
        || (reusable_rejected_reason.is_none() && (!build_size_known || small_build));
    if prefer_reusable {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::ReusableHashEqui,
            facts: strategy_facts(join, input, KeyedHashJoinStrategy::ReusableHashEqui, None, None, None),
        };
    }

    if one_shot_equi_join_supported(join) && small_build && small_probe {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::OneShotEqui,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::OneShotEqui,
                reusable_rejected_reason,
                Some("small_non_reusable_join_prefers_one_shot"),
                None,
            ),
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

fn partitioned_mixed_keyed_residual_strategy_decision(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
) -> Option<KeyedHashJoinStrategyDecision> {
    if !partitioned_keyed_residual_join_supported(join) || join_residual_requires_post_join_filter(join) {
        return None;
    }
    let large_build = input.observed_build_rows.is_some_and(|rows| rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD)
        || input.estimated_build_rows.is_some_and(|rows| rows >= LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD)
        || input.build_side_bytes.is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD);
    let large_probe = input.probe_side_bytes.is_some_and(|bytes| bytes >= LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD);
    let small_build = !large_build
        && (input.observed_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
            || input.estimated_build_rows.is_some_and(|rows| rows <= SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD)
            || input.build_side_bytes.is_some_and(|bytes| bytes <= SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD));
    let build_size_known =
        input.observed_build_rows.is_some() || input.estimated_build_rows.is_some() || input.build_side_bytes.is_some();
    if !large_build && !large_probe && (!build_size_known || small_build) {
        return None;
    }
    let reusable_rejected_reason = if !input.reusable_prebuild_supported {
        Some("build_key_prebuild_reservation_unsupported")
    } else if large_build {
        Some("build_side_too_large")
    } else if large_probe && !small_build {
        Some("probe_side_too_large")
    } else {
        Some("keyed_residual_requires_partitioned_mixed_strategy")
    };
    Some(KeyedHashJoinStrategyDecision::Selected {
        strategy: KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
        facts: strategy_facts(
            join,
            input,
            KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
            reusable_rejected_reason,
            None,
            None,
        ),
    })
}

pub(crate) fn keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
) -> bool {
    if join.spec.join_type != JoinType::Left
        || join.spec.build_side != JoinSide::Right
        || input.build_key_unique.is_some()
    {
        return false;
    }
    if matches!(
        keyed_hash_join_strategy_decision(join, input),
        KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::OneShotEqui, .. }
    ) {
        return false;
    }
    matches!(
        keyed_hash_join_strategy_decision(
            join,
            KeyedHashJoinStrategyInputFacts { build_key_unique: Some(true), ..input },
        ),
        KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::OneShotEqui, .. }
    )
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
        && (join.spec.residual_predicate.is_none()
            || (join.spec.join_type == JoinType::Inner && join_residual_requires_post_join_filter(join)))
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
        | Expr::TryCast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Alias { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Like { expr, .. } => expr_requires_post_join_filter(expr),
        Expr::Concat(exprs) => exprs.iter().any(expr_requires_post_join_filter),
        Expr::Binary { op: BinaryOp::Divide, output_type: Some(DataType::Decimal128(_, _)), .. } => true,
        Expr::Binary { left, right, .. } => {
            expr_requires_post_join_filter(left) || expr_requires_post_join_filter(right)
        },
        Expr::InList { expr, list, .. } => {
            expr_requires_post_join_filter(expr) || list.iter().any(expr_requires_post_join_filter)
        },
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => true,
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
        ("reusable_prebuild_supported", input.reusable_prebuild_supported.to_string()),
        ("large_keyed_hash_build_row_threshold", LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD.to_string()),
        ("large_keyed_hash_build_bytes_threshold", LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD.to_string()),
        ("large_keyed_hash_probe_bytes_threshold", LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD.to_string()),
        ("small_reusable_hash_build_row_threshold", SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD.to_string()),
        ("small_reusable_hash_build_bytes_threshold", SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD.to_string()),
        ("small_one_shot_probe_bytes_threshold", SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD.to_string()),
        (
            "max_small_probe_left_one_shot_build_row_threshold",
            MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD.to_string(),
        ),
        (
            "max_small_probe_left_one_shot_build_bytes_threshold",
            MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD.to_string(),
        ),
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
    if let Some(rows) = input.estimated_output_rows {
        facts.insert("estimated_output_rows", rows.to_string());
    }
    if let Some(bytes) = input.estimated_output_bytes {
        facts.insert("estimated_output_bytes", bytes.to_string());
    }
    if let Some(unique) = input.build_key_unique {
        facts.insert("build_key_unique", unique.to_string());
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

fn reusable_hash_prebuild_supported_for_plan(plan: &QueryPlan, join: &JoinNode) -> bool {
    let build_input = match join.spec.build_side {
        JoinSide::Left => join.inputs[0],
        JoinSide::Right => join.inputs[1],
    };
    let Some(input) = plan.node(build_input) else {
        return true;
    };
    let schema = input.output_schema();
    join.spec.keys.iter().all(|key| {
        let column = match join.spec.build_side {
            JoinSide::Left => &key.left,
            JoinSide::Right => &key.right,
        };
        schema
            .field_with_name(column.name())
            .map(|field| reusable_hash_prebuild_supported_type(field.data_type()))
            .unwrap_or(true)
    })
}

fn reusable_hash_prebuild_supported_type(data_type: &DataType) -> bool {
    !matches!(
        data_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Struct(_)
            | DataType::Map(_, _)
            | DataType::Union(_, _)
    )
}

pub(crate) fn keyed_filtered_semi_anti_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark)
        && join.spec.build_side == JoinSide::Right
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        && (join.spec.join_type != JoinType::LeftMark || join.spec.mark_column_name.is_some())
}

pub(crate) fn projected_cross_join_capable(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Cross
}

pub(crate) fn keyless_outer_cross_join_capable(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
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
        JoinType::Cross | JoinType::LeftMark => false,
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
        JoinType::Left | JoinType::Full | JoinType::Cross | JoinType::LeftMark => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSpec},
    };

    fn single_use_join(join_type: JoinType) -> JoinNode {
        JoinNode {
            inputs: [NodeId::from_index_for_internal_use(0), NodeId::from_index_for_internal_use(1)],
            spec: JoinSpec::new(
                join_type,
                vec![JoinKey::new(
                    ColumnRef::input(0, "join_key", "join_key"),
                    ColumnRef::input(1, "join_key", "join_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_value", "left_value"), "left_value"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
            ]),
        }
    }

    fn single_use_inner_join() -> JoinNode {
        single_use_join(JoinType::Inner)
    }

    fn single_use_inner_join_with_residual(residual: Expr) -> JoinNode {
        let mut join = single_use_inner_join();
        join.spec = join.spec.with_residual_predicate(residual);
        join
    }

    fn single_use_left_join() -> JoinNode {
        single_use_join(JoinType::Left)
    }

    fn input_facts(
        estimated_build_rows: Option<u64>,
        observed_build_rows: Option<u64>,
        probe_side_bytes: Option<u64>,
    ) -> KeyedHashJoinStrategyInputFacts {
        KeyedHashJoinStrategyInputFacts {
            estimated_build_rows,
            observed_build_rows,
            build_side_bytes: Some(SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD / 2),
            probe_side_bytes,
            estimated_output_rows: None,
            estimated_output_bytes: None,
            build_key_unique: None,
            probe_reuse_count: 1,
            reusable_prebuild_supported: true,
        }
    }

    fn selected_strategy(decision: KeyedHashJoinStrategyDecision) -> KeyedHashJoinStrategy {
        match decision {
            KeyedHashJoinStrategyDecision::Selected { strategy, .. } => strategy,
            KeyedHashJoinStrategyDecision::Rejected { reason, facts } => {
                panic!("expected selected strategy, got rejection {reason} with facts {facts:?}");
            },
        }
    }

    #[test]
    fn known_small_probe_single_use_inner_join_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(Some(10), None, Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2)),
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn supported_keyed_residual_inner_join_stays_on_mixed_strategy() {
        let join = single_use_inner_join_with_residual(Expr::binary(
            Expr::column("left_value"),
            BinaryOp::Gt,
            Expr::column("right_value"),
        ));

        assert!(!one_shot_equi_join_supported(&join));
        assert_eq!(join_node_execution_capability(&join), Some(JoinExecutionCapability::KeyedResidualMixed));
    }

    #[test]
    fn post_filter_only_keyed_residual_inner_join_uses_one_shot_filter_path() {
        let join = single_use_inner_join_with_residual(Expr::is_distinct_from(
            Expr::column("left_value"),
            Expr::column("right_value"),
        ));
        let decision = keyed_hash_join_strategy_decision(
            &join,
            input_facts(Some(10), None, Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2)),
        );

        assert!(one_shot_equi_join_supported(&join));
        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::OneShotEqui);
    }

    #[test]
    fn unknown_probe_medium_single_use_inner_join_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(Some(SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None, None),
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_tiny_estimated_build_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(&single_use_inner_join(), input_facts(Some(10), None, None));

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_tiny_observed_without_estimate_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(&single_use_inner_join(), input_facts(None, Some(31), None));

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_medium_observed_without_estimate_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(None, Some(SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None),
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn left_join_with_small_probe_unique_build_and_contained_large_build_prefers_one_shot() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: None,
                estimated_output_bytes: None,
                build_key_unique: Some(true),
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::OneShotEqui);
    }

    #[test]
    fn left_join_with_small_probe_and_small_build_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(10),
                observed_build_rows: None,
                build_side_bytes: Some(1024),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: None,
                estimated_output_bytes: None,
                build_key_unique: Some(true),
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn left_join_with_small_probe_and_uncontained_large_build_stays_partitioned() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD + 1),
                observed_build_rows: None,
                build_side_bytes: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD + 1),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: None,
                estimated_output_bytes: None,
                build_key_unique: Some(true),
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::PartitionedHashEqui);
    }

    #[test]
    fn left_join_with_small_probe_and_no_output_containment_evidence_stays_partitioned() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: None,
                estimated_output_bytes: None,
                build_key_unique: None,
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::PartitionedHashEqui);
    }

    #[test]
    fn left_join_with_small_probe_and_contained_build_requests_uniqueness_probe() {
        assert!(keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: None,
                estimated_output_bytes: None,
                build_key_unique: None,
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        ));
    }

    #[test]
    fn left_join_with_small_probe_and_contained_output_estimate_without_exact_proof_stays_partitioned() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
                estimated_output_rows: Some(1),
                estimated_output_bytes: Some(1024),
                build_key_unique: None,
                probe_reuse_count: 1,
                reusable_prebuild_supported: true,
            },
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::PartitionedHashEqui);
    }
}
