//! Join-strategy capability analysis.
//!
//! Classifies each [`JoinNode`] into a [`KeyedHashJoinStrategyDecision`] using
//! the byte/row thresholds captured in [`KeyedHashJoinStrategyOptions`]. Decisions are
//! consumed by capability admission, the bounded executor's strategy
//! selection, and the join-strategy metric rows.

use std::collections::BTreeMap;

use arrow::datatypes::DataType;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::{BinaryOp, Expr},
    plan::{JoinNode, JoinSide, JoinType, NodeId, QueryPlan, estimated_output_bytes, estimated_output_rows},
};

/// Threshold policy used by capability admission and runtime keyed-hash join
/// strategy selection.
///
/// This is an expert Rust surface only. Keep server/session/user-facing knobs
/// out until the admission and runtime contracts have a reviewed tuning model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct KeyedHashJoinStrategyOptions {
    #[doc(hidden)]
    pub large_keyed_hash_build_row_threshold: u64,
    #[doc(hidden)]
    pub large_keyed_hash_build_bytes_threshold: u64,
    #[doc(hidden)]
    pub large_keyed_hash_probe_bytes_threshold: u64,
    #[doc(hidden)]
    pub small_reusable_hash_build_row_threshold: u64,
    #[doc(hidden)]
    pub small_reusable_hash_build_bytes_threshold: u64,
    #[doc(hidden)]
    pub small_one_shot_probe_bytes_threshold: u64,
    #[doc(hidden)]
    pub max_small_probe_left_one_shot_build_row_threshold: u64,
    #[doc(hidden)]
    pub max_small_probe_left_one_shot_build_bytes_threshold: u64,
}

impl KeyedHashJoinStrategyOptions {
    /// Past this build scale, reusable hash residency is risky enough to prefer partitioning.
    pub const DEFAULT_LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD: u64 = 64_000_000;
    /// Byte-size large-build gate used alongside row evidence.
    pub const DEFAULT_LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;
    /// Large probes can turn one reusable build into too much gather/output pressure.
    pub const DEFAULT_LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
    /// Builds below this row scale are cheap enough to prebuild even for one probe.
    pub const DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD: u64 = 1_000_000;
    /// Byte-size small-build gate used alongside row evidence.
    pub const DEFAULT_SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD: u64 = 64 * 1024 * 1024;
    /// Small probes avoid the scheduling overhead of the partitioned path.
    pub const DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD: u64 = 256 * 1024 * 1024;
    /// Large-but-contained right builds can still be cheaper than partitioning
    /// when a left join's preserved probe side is already small.
    pub const DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD: u64 = 128_000_000;
    /// Byte guard paired with
    /// [`DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD`](Self::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD).
    pub const DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;

    pub fn large_keyed_hash_build_row_threshold(&self) -> u64 {
        self.large_keyed_hash_build_row_threshold
    }

    pub fn large_keyed_hash_build_bytes_threshold(&self) -> u64 {
        self.large_keyed_hash_build_bytes_threshold
    }

    pub fn large_keyed_hash_probe_bytes_threshold(&self) -> u64 {
        self.large_keyed_hash_probe_bytes_threshold
    }

    pub fn small_reusable_hash_build_row_threshold(&self) -> u64 {
        self.small_reusable_hash_build_row_threshold
    }

    pub fn small_reusable_hash_build_bytes_threshold(&self) -> u64 {
        self.small_reusable_hash_build_bytes_threshold
    }

    pub fn small_one_shot_probe_bytes_threshold(&self) -> u64 {
        self.small_one_shot_probe_bytes_threshold
    }

    pub fn max_small_probe_left_one_shot_build_row_threshold(&self) -> u64 {
        self.max_small_probe_left_one_shot_build_row_threshold
    }

    pub fn max_small_probe_left_one_shot_build_bytes_threshold(&self) -> u64 {
        self.max_small_probe_left_one_shot_build_bytes_threshold
    }

    /// Validates internal consistency for keyed hash-join strategy thresholds.
    ///
    /// Zero remains accepted because this policy is still an expert Rust surface
    /// and callers may use zero to disable narrow strategy branches. The checks
    /// only reject threshold orderings that make "small" and "large" evidence
    /// overlap contradictorily.
    pub fn validate(&self) -> Result<()> {
        validate_keyed_hash_join_threshold_order(
            "large_keyed_hash_build_row_threshold",
            self.large_keyed_hash_build_row_threshold,
            "small_reusable_hash_build_row_threshold",
            self.small_reusable_hash_build_row_threshold,
        )?;
        validate_keyed_hash_join_threshold_order(
            "large_keyed_hash_build_bytes_threshold",
            self.large_keyed_hash_build_bytes_threshold,
            "small_reusable_hash_build_bytes_threshold",
            self.small_reusable_hash_build_bytes_threshold,
        )?;
        validate_keyed_hash_join_threshold_order(
            "large_keyed_hash_probe_bytes_threshold",
            self.large_keyed_hash_probe_bytes_threshold,
            "small_one_shot_probe_bytes_threshold",
            self.small_one_shot_probe_bytes_threshold,
        )?;
        Ok(())
    }

    pub fn with_large_keyed_hash_build_row_threshold(mut self, rows: u64) -> Self {
        self.large_keyed_hash_build_row_threshold = rows;
        self
    }

    pub fn with_large_keyed_hash_build_bytes_threshold(mut self, bytes: u64) -> Self {
        self.large_keyed_hash_build_bytes_threshold = bytes;
        self
    }

    pub fn with_large_keyed_hash_probe_bytes_threshold(mut self, bytes: u64) -> Self {
        self.large_keyed_hash_probe_bytes_threshold = bytes;
        self
    }

    pub fn with_small_reusable_hash_build_row_threshold(mut self, rows: u64) -> Self {
        self.small_reusable_hash_build_row_threshold = rows;
        self
    }

    pub fn with_small_reusable_hash_build_bytes_threshold(mut self, bytes: u64) -> Self {
        self.small_reusable_hash_build_bytes_threshold = bytes;
        self
    }

    pub fn with_small_one_shot_probe_bytes_threshold(mut self, bytes: u64) -> Self {
        self.small_one_shot_probe_bytes_threshold = bytes;
        self
    }

    pub fn with_max_small_probe_left_one_shot_build_row_threshold(mut self, rows: u64) -> Self {
        self.max_small_probe_left_one_shot_build_row_threshold = rows;
        self
    }

    pub fn with_max_small_probe_left_one_shot_build_bytes_threshold(mut self, bytes: u64) -> Self {
        self.max_small_probe_left_one_shot_build_bytes_threshold = bytes;
        self
    }
}

fn validate_keyed_hash_join_threshold_order(
    large_field: &'static str,
    large_value: u64,
    small_field: &'static str,
    small_value: u64,
) -> Result<()> {
    if large_value != 0 && small_value != 0 && large_value < small_value {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.keyed_hash_join_strategy_options.validate",
            "keyed hash-join strategy threshold ordering is inconsistent",
        )
        .with_fact("large_field", large_field)
        .with_fact("large_value", large_value)
        .with_fact("small_field", small_field)
        .with_fact("small_value", small_value)
        .with_fact("reason", "large_threshold_below_small_threshold"));
    }
    Ok(())
}

impl Default for KeyedHashJoinStrategyOptions {
    fn default() -> Self {
        Self {
            large_keyed_hash_build_row_threshold: Self::DEFAULT_LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD,
            large_keyed_hash_build_bytes_threshold: Self::DEFAULT_LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD,
            large_keyed_hash_probe_bytes_threshold: Self::DEFAULT_LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD,
            small_reusable_hash_build_row_threshold: Self::DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD,
            small_reusable_hash_build_bytes_threshold: Self::DEFAULT_SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD,
            small_one_shot_probe_bytes_threshold: Self::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD,
            max_small_probe_left_one_shot_build_row_threshold:
                Self::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD,
            max_small_probe_left_one_shot_build_bytes_threshold:
                Self::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD,
        }
    }
}

/// Node-local native join execution capability.
///
/// This describes the operator strategy a single join node can use inside a
/// compiled pipeline. It deliberately does not name whole-query bounded
/// families or root/multi-join topology paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum JoinExecutionCapability {
    KeyedHash,
    KeyedFilteredSemiAnti,
    KeylessConditional(KeylessConditionalJoinKind),
    KeylessOuterCross,
    ProjectedCross,
    KeyedResidualMixed,
}

/// Physical strategy selected for a keyed hash-capable join after size,
/// reuse, and residual-expression facts have been evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum KeyedHashJoinStrategy {
    /// Prebuild and retain a hash-table artifact because the build side is
    /// reused, explicitly materialized, or small enough to keep reusable.
    ReusableHashEqui,
    /// Build and probe for this join instance only, without preserving a
    /// reusable hash-table artifact for later probes.
    OneShotEqui,
    /// Partition a large keyed equi-join so build/probe memory is bounded by
    /// partition-sized work units.
    PartitionedHashEqui,
    /// Partition a keyed join whose residual predicate is evaluated within the
    /// mixed keyed/residual path instead of as a reusable equi hash join.
    PartitionedMixedKeyedResidual,
}

impl KeyedHashJoinStrategy {
    #[doc(hidden)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReusableHashEqui => "reusable_hash_equi",
            Self::OneShotEqui => "one_shot_equi",
            Self::PartitionedHashEqui => "partitioned_hash_equi",
            Self::PartitionedMixedKeyedResidual => "partitioned_mixed_keyed_residual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct KeyedHashJoinStrategyInputFacts {
    #[doc(hidden)]
    pub estimated_build_rows: Option<u64>,
    #[doc(hidden)]
    pub observed_build_rows: Option<u64>,
    #[doc(hidden)]
    pub build_side_bytes: Option<u64>,
    #[doc(hidden)]
    pub probe_side_bytes: Option<u64>,
    #[doc(hidden)]
    pub estimated_output_rows: Option<u64>,
    #[doc(hidden)]
    pub estimated_output_bytes: Option<u64>,
    #[doc(hidden)]
    pub build_key_unique: Option<bool>,
    #[doc(hidden)]
    pub probe_reuse_count: u64,
    #[doc(hidden)]
    pub reusable_prebuild_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub enum KeyedHashJoinStrategyDecision {
    Selected { strategy: KeyedHashJoinStrategy, facts: BTreeMap<&'static str, String> },
    Rejected { reason: &'static str, facts: BTreeMap<&'static str, String> },
}

impl KeyedHashJoinStrategyDecision {
    #[doc(hidden)]
    pub fn facts(&self) -> &BTreeMap<&'static str, String> {
        match self {
            Self::Selected { facts, .. } | Self::Rejected { facts, .. } => facts,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum KeylessConditionalJoinKind {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
}

impl KeylessConditionalJoinKind {
    #[doc(hidden)]
    pub fn try_from_join(join: &JoinNode) -> Option<Self> {
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

#[doc(hidden)]
pub fn join_node_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
    single_join_node_execution_capability(join).or_else(|| multi_join_subtree_join_execution_capability(join))
}

#[doc(hidden)]
pub fn single_join_node_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
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

#[doc(hidden)]
pub fn multi_join_subtree_join_execution_capability(join: &JoinNode) -> Option<JoinExecutionCapability> {
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

#[doc(hidden)]
pub fn keyed_hash_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

#[doc(hidden)]
pub fn keyed_hash_join_strategy_decision_for_plan(
    plan: &QueryPlan,
    node_id: NodeId,
    join: &JoinNode,
) -> Option<KeyedHashJoinStrategyDecision> {
    keyed_hash_join_strategy_decision_for_plan_with_options(
        plan,
        node_id,
        join,
        &plan.keyed_hash_join_strategy_options(),
    )
}

#[doc(hidden)]
pub fn keyed_hash_join_strategy_decision_for_plan_with_options(
    plan: &QueryPlan,
    node_id: NodeId,
    join: &JoinNode,
    options: &KeyedHashJoinStrategyOptions,
) -> Option<KeyedHashJoinStrategyDecision> {
    if join_residual_requires_post_join_filter(join) {
        return Some(keyed_hash_join_strategy_decision_with_options(
            join,
            keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join),
            options,
        ));
    }
    if keyed_hash_join_capable(join) {
        return Some(keyed_hash_join_strategy_decision_with_options(
            join,
            keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join),
            options,
        ));
    }
    let input = keyed_hash_join_strategy_input_facts_for_plan(plan, node_id, join);
    if let Some(decision) = partitioned_mixed_keyed_residual_strategy_decision(join, input, options) {
        return Some(decision);
    }
    if one_shot_equi_join_supported(join) {
        let decision = keyed_hash_join_strategy_decision_with_options(join, input, options);
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

#[doc(hidden)]
pub fn keyed_hash_join_strategy_input_facts_for_plan(
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

#[cfg(test)]
#[doc(hidden)]
pub fn keyed_hash_join_strategy_decision(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
) -> KeyedHashJoinStrategyDecision {
    keyed_hash_join_strategy_decision_with_options(join, input, &KeyedHashJoinStrategyOptions::default())
}

#[doc(hidden)]
pub fn keyed_hash_join_strategy_decision_with_options(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> KeyedHashJoinStrategyDecision {
    if join_residual_requires_post_join_filter(join) {
        if one_shot_equi_join_supported(join) {
            return KeyedHashJoinStrategyDecision::Selected {
                strategy: KeyedHashJoinStrategy::OneShotEqui,
                facts: strategy_facts(
                    join,
                    input,
                    KeyedHashJoinStrategy::OneShotEqui,
                    options,
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
                options,
                None,
                Some("residual_requires_post_join_filter"),
                Some("one_shot_equi_requires_keyed_inner_left_or_full_join"),
            ),
        };
    }

    let large_build = keyed_hash_join_build_is_large(&input, options);
    let small_build = !large_build
        && (input.observed_build_rows.is_some_and(|rows| rows <= options.small_reusable_hash_build_row_threshold)
            || input.estimated_build_rows.is_some_and(|rows| rows <= options.small_reusable_hash_build_row_threshold)
            || input.build_side_bytes.is_some_and(|bytes| bytes <= options.small_reusable_hash_build_bytes_threshold));
    let build_size_known =
        input.observed_build_rows.is_some() || input.estimated_build_rows.is_some() || input.build_side_bytes.is_some();
    let large_probe =
        input.probe_side_bytes.is_some_and(|bytes| bytes >= options.large_keyed_hash_probe_bytes_threshold);
    let small_probe = !large_probe
        && input.probe_side_bytes.is_some_and(|bytes| bytes <= options.small_one_shot_probe_bytes_threshold);
    let single_use_without_artifact = join.spec.build_artifact_id.is_none() && input.probe_reuse_count <= 1;
    let one_shot_contained_left_join_build = input
        .observed_build_rows
        .or(input.estimated_build_rows)
        .is_some_and(|rows| rows <= options.max_small_probe_left_one_shot_build_row_threshold)
        && input
            .build_side_bytes
            .is_some_and(|bytes| bytes <= options.max_small_probe_left_one_shot_build_bytes_threshold);
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

    if let Some(decision) = partitioned_mixed_keyed_residual_strategy_decision(join, input, options) {
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
                options,
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
            facts: strategy_facts(join, input, KeyedHashJoinStrategy::ReusableHashEqui, options, None, None, None),
        };
    }

    if one_shot_equi_join_supported(join) && small_build && small_probe {
        return KeyedHashJoinStrategyDecision::Selected {
            strategy: KeyedHashJoinStrategy::OneShotEqui,
            facts: strategy_facts(
                join,
                input,
                KeyedHashJoinStrategy::OneShotEqui,
                options,
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
                options,
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
                options,
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
                options,
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
            options,
            reusable_rejected_reason,
            Some("partitioned_hash_equi_unsupported"),
            Some("one_shot_equi_requires_keyed_inner_left_or_full_join"),
        ),
    }
}

fn partitioned_mixed_keyed_residual_strategy_decision(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> Option<KeyedHashJoinStrategyDecision> {
    if !partitioned_keyed_residual_join_supported(join) || join_residual_requires_post_join_filter(join) {
        return None;
    }
    let large_build = keyed_hash_join_build_is_large(&input, options);
    let large_probe =
        input.probe_side_bytes.is_some_and(|bytes| bytes >= options.large_keyed_hash_probe_bytes_threshold);
    let small_build = !large_build
        && (input.observed_build_rows.is_some_and(|rows| rows <= options.small_reusable_hash_build_row_threshold)
            || input.estimated_build_rows.is_some_and(|rows| rows <= options.small_reusable_hash_build_row_threshold)
            || input.build_side_bytes.is_some_and(|bytes| bytes <= options.small_reusable_hash_build_bytes_threshold));
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
            options,
            reusable_rejected_reason,
            None,
            None,
        ),
    })
}

/// Whether the build side already exceeds the large-build thresholds.
///
/// Large-build evidence is monotone in observed rows and bytes: once a build
/// is large under partial observation it stays large, so callers may commit to
/// partition-wise consumption before the whole build side is materialized.
#[doc(hidden)]
pub fn keyed_hash_join_build_is_large(
    input: &KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> bool {
    input.observed_build_rows.is_some_and(|rows| rows >= options.large_keyed_hash_build_row_threshold)
        || input.estimated_build_rows.is_some_and(|rows| rows >= options.large_keyed_hash_build_row_threshold)
        || input.build_side_bytes.is_some_and(|bytes| bytes >= options.large_keyed_hash_build_bytes_threshold)
}

#[cfg(test)]
#[doc(hidden)]
pub fn keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
) -> bool {
    keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options(
        join,
        input,
        &KeyedHashJoinStrategyOptions::default(),
    )
}

#[doc(hidden)]
pub fn keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness_with_options(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> bool {
    if join.spec.join_type != JoinType::Left
        || join.spec.build_side != JoinSide::Right
        || input.build_key_unique.is_some()
    {
        return false;
    }
    if matches!(
        keyed_hash_join_strategy_decision_with_options(join, input, options),
        KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::OneShotEqui, .. }
    ) {
        return false;
    }
    matches!(
        keyed_hash_join_strategy_decision_with_options(
            join,
            KeyedHashJoinStrategyInputFacts { build_key_unique: Some(true), ..input },
            options,
        ),
        KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::OneShotEqui, .. }
    )
}

#[doc(hidden)]
pub fn partitioned_equi_join_supported(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

#[doc(hidden)]
pub fn partitioned_keyed_residual_join_supported(join: &JoinNode) -> bool {
    !keyed_residual_left_match_join_capable(join) && keyed_residual_mixed_join_capable(join)
}

#[doc(hidden)]
pub fn one_shot_equi_join_supported(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && (join.spec.residual_predicate.is_none()
            || (join.spec.join_type == JoinType::Inner && join_residual_requires_post_join_filter(join)))
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

#[doc(hidden)]
pub fn join_residual_requires_post_join_filter(join: &JoinNode) -> bool {
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
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
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
    options: &KeyedHashJoinStrategyOptions,
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
        ("large_keyed_hash_build_row_threshold", options.large_keyed_hash_build_row_threshold.to_string()),
        ("large_keyed_hash_build_bytes_threshold", options.large_keyed_hash_build_bytes_threshold.to_string()),
        ("large_keyed_hash_probe_bytes_threshold", options.large_keyed_hash_probe_bytes_threshold.to_string()),
        ("small_reusable_hash_build_row_threshold", options.small_reusable_hash_build_row_threshold.to_string()),
        ("small_reusable_hash_build_bytes_threshold", options.small_reusable_hash_build_bytes_threshold.to_string()),
        ("small_one_shot_probe_bytes_threshold", options.small_one_shot_probe_bytes_threshold.to_string()),
        (
            "max_small_probe_left_one_shot_build_row_threshold",
            options.max_small_probe_left_one_shot_build_row_threshold.to_string(),
        ),
        (
            "max_small_probe_left_one_shot_build_bytes_threshold",
            options.max_small_probe_left_one_shot_build_bytes_threshold.to_string(),
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
            .map(|field| hash_join_prebuild_reservation_supported_type(field.data_type()))
            .unwrap_or(true)
    })
}

fn hash_join_prebuild_reservation_supported_type(data_type: &DataType) -> bool {
    // Flat strings use cuDF's data-dependent exact pre-build contract; only
    // structural nested keys are rejected here.
    !matches!(
        data_type,
        DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Struct(_)
            | DataType::Map(_, _)
            | DataType::Union(_, _)
    )
}

#[doc(hidden)]
pub fn keyed_filtered_semi_anti_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark)
        && join.spec.build_side == JoinSide::Right
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        && (join.spec.join_type != JoinType::LeftMark || join.spec.mark_column_name.is_some())
}

#[doc(hidden)]
pub fn projected_cross_join_capable(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.join_type == JoinType::Cross
}

#[doc(hidden)]
pub fn keyless_outer_cross_join_capable(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::Left | JoinType::Full)
        && (join.spec.join_type != JoinType::Left || join.spec.build_side == JoinSide::Right)
}

#[doc(hidden)]
pub fn keyed_residual_mixed_join_capable(join: &JoinNode) -> bool {
    // Null-aware anti semantics are implemented only by the filtered semi/anti strategy.
    if join.spec.keys.is_empty()
        || join.spec.residual_predicate.is_none()
        || join.spec.output_projection.is_empty()
        || join.spec.null_aware
    {
        return false;
    }

    match join.spec.join_type {
        JoinType::Inner | JoinType::Full => true,
        JoinType::Left => join.spec.build_side == JoinSide::Right,
        JoinType::LeftSemi | JoinType::LeftAnti => {
            keyed_residual_left_match_join_capable(join)
                || (join.spec.build_side == JoinSide::Right
                    && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left))
        },
        JoinType::Cross | JoinType::LeftMark => false,
    }
}

#[doc(hidden)]
pub fn keyed_residual_left_match_join_capable(join: &JoinNode) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Left
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

#[doc(hidden)]
pub fn multi_join_keyed_residual_mixed_join_capable(join: &JoinNode) -> bool {
    matches!(join.spec.join_type, JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti)
        && keyed_residual_mixed_join_capable(join)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSpec},
    };

    type Options = KeyedHashJoinStrategyOptions;

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
            build_side_bytes: Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD / 2),
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
            input_facts(Some(10), None, Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2)),
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
    fn null_aware_anti_join_executes_only_on_the_filtered_strategy() {
        let mut join = single_use_join(JoinType::LeftAnti);
        join.spec = join
            .spec
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "left_value", "left_value"),
                "left_value",
            )])
            .with_null_aware(true);
        assert_eq!(join_node_execution_capability(&join), Some(JoinExecutionCapability::KeyedFilteredSemiAnti));

        join.spec = join.spec.with_residual_predicate(Expr::binary(
            Expr::column("left_value"),
            BinaryOp::Gt,
            Expr::column("right_value"),
        ));
        assert_eq!(join_node_execution_capability(&join), None);
    }

    #[test]
    fn post_filter_only_keyed_residual_inner_join_uses_one_shot_filter_path() {
        let join = single_use_inner_join_with_residual(Expr::is_distinct_from(
            Expr::column("left_value"),
            Expr::column("right_value"),
        ));
        let decision = keyed_hash_join_strategy_decision(
            &join,
            input_facts(Some(10), None, Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2)),
        );

        assert!(one_shot_equi_join_supported(&join));
        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::OneShotEqui);
    }

    #[test]
    fn unknown_probe_medium_single_use_inner_join_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None, None),
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
            input_facts(None, Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None),
        );

        assert_eq!(selected_strategy(decision), KeyedHashJoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn keyed_hash_join_strategy_options_change_selection_and_reported_thresholds() {
        let join = single_use_inner_join();
        let input = KeyedHashJoinStrategyInputFacts {
            estimated_build_rows: Some(10),
            observed_build_rows: None,
            build_side_bytes: Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD * 2),
            probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
            estimated_output_rows: None,
            estimated_output_bytes: None,
            build_key_unique: None,
            probe_reuse_count: 1,
            reusable_prebuild_supported: true,
        };
        let options = KeyedHashJoinStrategyOptions::default()
            .with_small_reusable_hash_build_row_threshold(5)
            .with_small_reusable_hash_build_bytes_threshold(1024);

        assert_eq!(
            selected_strategy(keyed_hash_join_strategy_decision(&join, input)),
            KeyedHashJoinStrategy::ReusableHashEqui
        );
        let decision = keyed_hash_join_strategy_decision_with_options(&join, input, &options);

        assert_eq!(selected_strategy(decision.clone()), KeyedHashJoinStrategy::PartitionedHashEqui);
        assert_eq!(decision.facts().get("small_reusable_hash_build_row_threshold"), Some(&"5".to_string()));
        assert_eq!(decision.facts().get("small_reusable_hash_build_bytes_threshold"), Some(&"1024".to_string()));
    }

    #[test]
    fn keyed_hash_join_strategy_options_accept_zero_threshold_overrides() {
        KeyedHashJoinStrategyOptions::default().with_large_keyed_hash_build_row_threshold(0).validate().unwrap();
        KeyedHashJoinStrategyOptions::default().with_large_keyed_hash_build_bytes_threshold(0).validate().unwrap();
        KeyedHashJoinStrategyOptions::default().with_large_keyed_hash_probe_bytes_threshold(0).validate().unwrap();
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(0).validate().unwrap();
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_bytes_threshold(0).validate().unwrap();
        KeyedHashJoinStrategyOptions::default().with_small_one_shot_probe_bytes_threshold(0).validate().unwrap();
    }

    #[test]
    fn left_join_with_small_probe_unique_build_and_contained_large_build_prefers_one_shot() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_left_join(),
            KeyedHashJoinStrategyInputFacts {
                estimated_build_rows: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
                estimated_build_rows: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD + 1),
                observed_build_rows: None,
                build_side_bytes: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD + 1),
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
                estimated_build_rows: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
                estimated_build_rows: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
                estimated_build_rows: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD),
                observed_build_rows: None,
                build_side_bytes: Some(Options::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD),
                probe_side_bytes: Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2),
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
