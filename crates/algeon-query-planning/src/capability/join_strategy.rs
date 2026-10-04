use std::collections::BTreeMap;

use crate::{
    capability::{KeyedHashJoinStrategyInputFacts, KeyedHashJoinStrategyOptions, KeylessConditionalJoinKind},
    plan::{JoinNode, JoinSide, JoinType},
};

/// The executor a join node runs, named once for planning, reports, and the
/// engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinStrategy {
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
    FilteredSemiAnti,
    KeylessConditional {
        join_type: KeylessConditionalJoinKind,
    },
    KeylessOuterCross,
    ProjectedCross,
    MixedKeyedResidual,
}

impl JoinStrategy {
    /// Stable diagnostic label, the `join_strategy` fact value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReusableHashEqui => "reusable_hash_equi",
            Self::OneShotEqui => "one_shot_equi",
            Self::PartitionedHashEqui => "partitioned_hash_equi",
            Self::PartitionedMixedKeyedResidual => "partitioned_mixed_keyed_residual",
            Self::FilteredSemiAnti => "filtered_semi_anti",
            Self::KeylessConditional { .. } => "keyless_conditional",
            Self::KeylessOuterCross => "keyless_outer_cross",
            Self::ProjectedCross => "projected_cross",
            Self::MixedKeyedResidual => "mixed_keyed_residual",
        }
    }
}

/// A chosen join strategy, with the size evidence behind it when sizes
/// decided the choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinStrategyChoice {
    strategy: JoinStrategy,
    evidence: Option<JoinStrategyEvidence>,
}

impl JoinStrategyChoice {
    pub(crate) fn with_evidence(evidence: JoinStrategyEvidence) -> Self {
        Self { strategy: evidence.strategy, evidence: Some(evidence) }
    }

    pub(crate) const fn without_evidence(strategy: JoinStrategy) -> Self {
        Self { strategy, evidence: None }
    }

    pub const fn strategy(&self) -> JoinStrategy {
        self.strategy
    }

    pub const fn evidence(&self) -> Option<&JoinStrategyEvidence> {
        self.evidence.as_ref()
    }
}

/// The size facts and thresholds a keyed strategy choice read, and why each
/// alternative lost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinStrategyEvidence {
    strategy: JoinStrategy,
    join_type: JoinType,
    build_side: JoinSide,
    key_count: usize,
    build_artifact_id_present: bool,
    input: KeyedHashJoinStrategyInputFacts,
    options: KeyedHashJoinStrategyOptions,
    reusable_rejected_reason: Option<&'static str>,
    partitioned_rejected_reason: Option<&'static str>,
}

impl JoinStrategyEvidence {
    pub(crate) fn new(
        join: &JoinNode,
        input: KeyedHashJoinStrategyInputFacts,
        strategy: JoinStrategy,
        options: &KeyedHashJoinStrategyOptions,
        reusable_rejected_reason: Option<&'static str>,
        partitioned_rejected_reason: Option<&'static str>,
    ) -> Self {
        Self {
            strategy,
            join_type: join.spec.join_type.clone(),
            build_side: join.spec.build_side,
            key_count: join.spec.keys.len(),
            build_artifact_id_present: join.spec.build_artifact_id.is_some(),
            input,
            options: *options,
            reusable_rejected_reason,
            partitioned_rejected_reason,
        }
    }

    pub const fn input(&self) -> &KeyedHashJoinStrategyInputFacts {
        &self.input
    }

    pub const fn reusable_rejected_reason(&self) -> Option<&'static str> {
        self.reusable_rejected_reason
    }

    pub const fn partitioned_rejected_reason(&self) -> Option<&'static str> {
        self.partitioned_rejected_reason
    }

    /// The evidence as the `join_strategy` decision facts the metric rows and
    /// diagnostics record.
    pub fn facts(&self) -> BTreeMap<&'static str, String> {
        let options = &self.options;
        let input = &self.input;
        let mut facts = BTreeMap::from([
            ("join_strategy", self.strategy.as_str().to_string()),
            ("join_type", self.join_type.to_string()),
            ("build_side", self.build_side.to_string()),
            ("key_count", self.key_count.to_string()),
            ("probe_reuse_count", input.probe_reuse_count.to_string()),
            ("build_artifact_id_present", self.build_artifact_id_present.to_string()),
            ("reusable_prebuild_supported", input.reusable_prebuild_supported.to_string()),
            ("large_keyed_hash_build_row_threshold", options.large_keyed_hash_build_row_threshold.to_string()),
            ("large_keyed_hash_build_bytes_threshold", options.large_keyed_hash_build_bytes_threshold.to_string()),
            ("large_keyed_hash_probe_bytes_threshold", options.large_keyed_hash_probe_bytes_threshold.to_string()),
            ("small_reusable_hash_build_row_threshold", options.small_reusable_hash_build_row_threshold.to_string()),
            (
                "small_reusable_hash_build_bytes_threshold",
                options.small_reusable_hash_build_bytes_threshold.to_string(),
            ),
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
        let optional = [
            ("estimated_build_rows", input.estimated_build_rows.map(|rows| rows.to_string())),
            ("observed_build_rows", input.observed_build_rows.map(|rows| rows.to_string())),
            ("build_side_bytes", input.build_side_bytes.map(|bytes| bytes.to_string())),
            ("probe_side_bytes", input.probe_side_bytes.map(|bytes| bytes.to_string())),
            ("estimated_output_rows", input.estimated_output_rows.map(|rows| rows.to_string())),
            ("estimated_output_bytes", input.estimated_output_bytes.map(|bytes| bytes.to_string())),
            ("build_key_unique", input.build_key_unique.map(|unique| unique.to_string())),
            ("reusable_hash_join_rejected_reason", self.reusable_rejected_reason.map(str::to_string)),
            ("partitioned_hash_join_rejected_reason", self.partitioned_rejected_reason.map(str::to_string)),
        ];
        facts.extend(optional.into_iter().filter_map(|(key, value)| value.map(|value| (key, value))));
        facts
    }
}
