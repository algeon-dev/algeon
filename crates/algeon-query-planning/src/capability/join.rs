//! Join operator forms and strategy choice.
//!
//! [`classify_join`] derives one [`JoinForm`] per join node from its spec;
//! the form names the executors that can run the join. [`choose_join_strategy`]
//! picks one of them from size facts and the thresholds in
//! [`KeyedHashJoinStrategyOptions`]. Capability admission, the compiled join
//! plan, the bounded executor's refinement, and the join-strategy metric rows
//! read these two functions.

use arrow::datatypes::DataType;

use super::{
    join_strategy::{JoinStrategy, JoinStrategyChoice, JoinStrategyEvidence},
    rejection::{Reason, Rejection, RejectionOwner},
};
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
    pub(super) large_keyed_hash_build_row_threshold: u64,
    pub(super) large_keyed_hash_build_bytes_threshold: u64,
    pub(super) large_keyed_hash_probe_bytes_threshold: u64,
    pub(super) small_reusable_hash_build_row_threshold: u64,
    pub(super) small_reusable_hash_build_bytes_threshold: u64,
    pub(super) small_one_shot_probe_bytes_threshold: u64,
    pub(super) max_small_probe_left_one_shot_build_row_threshold: u64,
    pub(super) max_small_probe_left_one_shot_build_bytes_threshold: u64,
}

impl KeyedHashJoinStrategyOptions {
    /// Past this build scale, reusable hash residency is risky enough to prefer partitioning.
    pub(crate) const DEFAULT_LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD: u64 = 64_000_000;
    /// Byte-size large-build gate used alongside row evidence.
    pub(crate) const DEFAULT_LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;
    /// Large probes can turn one reusable build into too much gather/output pressure.
    pub(crate) const DEFAULT_LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
    /// Builds below this row scale are cheap enough to prebuild even for one probe.
    pub(crate) const DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD: u64 = 1_000_000;
    /// Byte-size small-build gate used alongside row evidence.
    pub(crate) const DEFAULT_SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD: u64 = 64 * 1024 * 1024;
    /// Small probes avoid the scheduling overhead of the partitioned path.
    pub(crate) const DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD: u64 = 256 * 1024 * 1024;
    /// Large-but-contained right builds can still be cheaper than partitioning
    /// when a left join's preserved probe side is already small.
    pub(crate) const DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD: u64 = 128_000_000;
    /// Byte guard paired with
    /// [`DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD`](Self::DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_ROW_THRESHOLD).
    pub(crate) const DEFAULT_MAX_SMALL_PROBE_LEFT_ONE_SHOT_BUILD_BYTES_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;

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

    #[cfg(test)]
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

    #[cfg(test)]
    pub fn with_small_one_shot_probe_bytes_threshold(mut self, bytes: u64) -> Self {
        self.small_one_shot_probe_bytes_threshold = bytes;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeylessConditionalJoinKind {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
}

/// The operator form of one join node, derived once from its spec.
///
/// Each form names the executors that can run the join
/// ([`JoinForm::strategies`]); strategy choice picks among them from size
/// facts and never widens the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinForm {
    /// Equality keys without a residual: an inner join, a left join that
    /// builds its right side, or a full join.
    KeyedEqui,
    /// An inner join with equality keys whose residual only a filter after
    /// matching can evaluate.
    KeyedEquiPostFilter,
    /// Equality keys without a residual: a semi, anti, or mark join that
    /// builds its right side and outputs only left columns.
    KeyedFilteredSemiAnti,
    /// Equality keys and a residual the mixed join evaluates while matching.
    /// `left_match` marks a semi or anti join that builds its left side.
    KeyedResidual { left_match: bool },
    /// A residual without equality keys.
    KeylessConditional(KeylessConditionalJoinKind),
    /// A left (building its right side) or full join without keys or residual.
    KeylessOuterCross,
    /// A cross join without keys or residual.
    ProjectedCross,
}

impl JoinForm {
    /// Every strategy that can execute this form.
    pub fn strategies(self) -> &'static [JoinStrategy] {
        match self {
            Self::KeyedEqui => {
                &[JoinStrategy::ReusableHashEqui, JoinStrategy::OneShotEqui, JoinStrategy::PartitionedHashEqui]
            },
            Self::KeyedEquiPostFilter => &[JoinStrategy::OneShotEqui],
            Self::KeyedFilteredSemiAnti => &[JoinStrategy::FilteredSemiAnti],
            Self::KeyedResidual { left_match: false } => {
                &[JoinStrategy::MixedKeyedResidual, JoinStrategy::PartitionedMixedKeyedResidual]
            },
            Self::KeyedResidual { left_match: true } => &[JoinStrategy::MixedKeyedResidual],
            Self::KeylessConditional(KeylessConditionalJoinKind::Inner) => {
                &[JoinStrategy::KeylessConditional { join_type: KeylessConditionalJoinKind::Inner }]
            },
            Self::KeylessConditional(KeylessConditionalJoinKind::Left) => {
                &[JoinStrategy::KeylessConditional { join_type: KeylessConditionalJoinKind::Left }]
            },
            Self::KeylessConditional(KeylessConditionalJoinKind::Full) => {
                &[JoinStrategy::KeylessConditional { join_type: KeylessConditionalJoinKind::Full }]
            },
            Self::KeylessConditional(KeylessConditionalJoinKind::LeftSemi) => {
                &[JoinStrategy::KeylessConditional { join_type: KeylessConditionalJoinKind::LeftSemi }]
            },
            Self::KeylessConditional(KeylessConditionalJoinKind::LeftAnti) => {
                &[JoinStrategy::KeylessConditional { join_type: KeylessConditionalJoinKind::LeftAnti }]
            },
            Self::KeylessOuterCross => &[JoinStrategy::KeylessOuterCross],
            Self::ProjectedCross => &[JoinStrategy::ProjectedCross],
        }
    }

    /// Whether the build runs ahead of other ready work. A filtered semi,
    /// anti, or mark build is a key set the probe side waits on.
    pub fn builds_first(self) -> bool {
        self == Self::KeyedFilteredSemiAnti
    }
}

/// Classifies a join node into the operator form that executes it, or
/// rejects a join no native executor runs.
pub(crate) fn classify_join(node_id: NodeId, join: &JoinNode) -> std::result::Result<JoinForm, Rejection> {
    let spec = &join.spec;
    let rejection = |reason| {
        Rejection::new(
            RejectionOwner::Node(node_id),
            reason,
            [
                ("node_id", node_id.to_string()),
                ("join_type", spec.join_type.to_string()),
                ("key_count", spec.keys.len().to_string()),
                ("has_residual", spec.residual_predicate.is_some().to_string()),
                ("build_side", spec.build_side.to_string()),
            ],
        )
    };
    if spec.output_projection.is_empty() {
        return Err(rejection(Reason::UnsupportedJoinNodeCapability));
    }
    // Semi, anti, and mark joins output only left columns; node validation
    // rejects any other projection, so the forms below need not check it.
    let build_right = spec.build_side == JoinSide::Right;
    let form = match (spec.keys.is_empty(), &spec.residual_predicate) {
        (false, None) => match spec.join_type {
            JoinType::Inner | JoinType::Full => Some(JoinForm::KeyedEqui),
            JoinType::Left => build_right.then_some(JoinForm::KeyedEqui),
            JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } => {
                build_right.then_some(JoinForm::KeyedFilteredSemiAnti)
            },
            JoinType::Cross => None,
        },
        (false, Some(residual)) => {
            // Null-aware anti semantics are implemented only by the filtered
            // semi/anti executor, which takes no residual.
            let left_match = matches!(spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti { .. }) && !build_right;
            let mixed = !spec.join_type.is_null_aware()
                && match spec.join_type {
                    JoinType::Inner | JoinType::Full | JoinType::LeftSemi | JoinType::LeftAnti { .. } => true,
                    JoinType::Left => build_right,
                    JoinType::Cross | JoinType::LeftMark { .. } => false,
                };
            match (mixed, expr_requires_post_join_filter(residual)) {
                (false, _) => None,
                (true, false) => Some(JoinForm::KeyedResidual { left_match }),
                (true, true) if spec.join_type == JoinType::Inner => Some(JoinForm::KeyedEquiPostFilter),
                (true, true) => return Err(rejection(Reason::PostJoinResidualFilterRequiresOneShotEquiJoin)),
            }
        },
        (true, Some(_)) => match spec.join_type {
            JoinType::Inner => Some(KeylessConditionalJoinKind::Inner),
            JoinType::Left => build_right.then_some(KeylessConditionalJoinKind::Left),
            JoinType::Full => Some(KeylessConditionalJoinKind::Full),
            JoinType::LeftSemi => (!build_right).then_some(KeylessConditionalJoinKind::LeftSemi),
            JoinType::LeftAnti { .. } => (!build_right).then_some(KeylessConditionalJoinKind::LeftAnti),
            JoinType::Cross | JoinType::LeftMark { .. } => None,
        }
        .map(JoinForm::KeylessConditional),
        (true, None) => match spec.join_type {
            JoinType::Left => build_right.then_some(JoinForm::KeylessOuterCross),
            JoinType::Full => Some(JoinForm::KeylessOuterCross),
            JoinType::Cross => Some(JoinForm::ProjectedCross),
            JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } => None,
        },
    };
    form.ok_or_else(|| rejection(Reason::UnsupportedJoinNodeCapability))
}

/// Chooses the strategy that executes `join` in `form` from size facts.
///
/// The choice is always one of [`JoinForm::strategies`]. Missing size facts
/// still yield an executable strategy: an unknown keyed build keeps the
/// reusable hash table.
pub(crate) fn choose_join_strategy(
    form: JoinForm,
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> JoinStrategyChoice {
    match form {
        JoinForm::KeyedEqui => keyed_equi_strategy_choice(join, input, options),
        JoinForm::KeyedEquiPostFilter => JoinStrategyChoice::with_evidence(JoinStrategyEvidence::new(
            join,
            input,
            JoinStrategy::OneShotEqui,
            options,
            Some("residual_requires_post_join_filter"),
            Some("residual_requires_post_join_filter"),
        )),
        JoinForm::KeyedResidual { left_match: false } => {
            partitioned_mixed_keyed_residual_evidence(join, input, options)
                .map_or(JoinStrategyChoice::without_evidence(JoinStrategy::MixedKeyedResidual), |evidence| {
                    JoinStrategyChoice::with_evidence(evidence)
                })
        },
        JoinForm::KeyedResidual { left_match: true } => {
            JoinStrategyChoice::without_evidence(JoinStrategy::MixedKeyedResidual)
        },
        JoinForm::KeyedFilteredSemiAnti => JoinStrategyChoice::without_evidence(JoinStrategy::FilteredSemiAnti),
        JoinForm::KeylessConditional(join_type) => {
            JoinStrategyChoice::without_evidence(JoinStrategy::KeylessConditional { join_type })
        },
        JoinForm::KeylessOuterCross => JoinStrategyChoice::without_evidence(JoinStrategy::KeylessOuterCross),
        JoinForm::ProjectedCross => JoinStrategyChoice::without_evidence(JoinStrategy::ProjectedCross),
    }
}

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

fn keyed_equi_strategy_choice(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> JoinStrategyChoice {
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
    let evidence = |strategy, reusable_rejected, partitioned_rejected| {
        JoinStrategyChoice::with_evidence(JoinStrategyEvidence::new(
            join,
            input,
            strategy,
            options,
            reusable_rejected,
            partitioned_rejected,
        ))
    };

    if join.spec.join_type == JoinType::Left
        && input.reusable_prebuild_supported
        && single_use_without_artifact
        && !small_build
        && small_probe
        && one_shot_contained_left_join_build
        && one_shot_contained_left_join_output
    {
        return evidence(
            JoinStrategy::OneShotEqui,
            reusable_rejected_reason,
            Some("small_preserved_probe_prefers_one_shot_left_join"),
        );
    }

    let prefer_reusable = join.spec.build_artifact_id.is_some()
        || input.probe_reuse_count > 1
        || (reusable_rejected_reason.is_none() && (!build_size_known || small_build));
    if prefer_reusable {
        return evidence(JoinStrategy::ReusableHashEqui, None, None);
    }
    if small_build && small_probe {
        return evidence(
            JoinStrategy::OneShotEqui,
            reusable_rejected_reason,
            Some("small_non_reusable_join_prefers_one_shot"),
        );
    }
    evidence(JoinStrategy::PartitionedHashEqui, reusable_rejected_reason, None)
}

/// Evidence for partitioning a keyed residual join, when its build or probe
/// side is large enough that the mixed join should run partition by partition.
fn partitioned_mixed_keyed_residual_evidence(
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> Option<JoinStrategyEvidence> {
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
        "build_key_prebuild_reservation_unsupported"
    } else if large_build {
        "build_side_too_large"
    } else if large_probe && !small_build {
        "probe_side_too_large"
    } else {
        "keyed_residual_requires_partitioned_mixed_strategy"
    };
    Some(JoinStrategyEvidence::new(
        join,
        input,
        JoinStrategy::PartitionedMixedKeyedResidual,
        options,
        Some(reusable_rejected_reason),
        None,
    ))
}

/// Whether the build side already exceeds the large-build thresholds.
///
/// Large-build evidence is monotone in observed rows and bytes: once a build
/// is large under partial observation it stays large, so callers may commit to
/// partition-wise consumption before the whole build side is materialized.
pub fn keyed_hash_join_build_is_large(
    input: &KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> bool {
    input.observed_build_rows.is_some_and(|rows| rows >= options.large_keyed_hash_build_row_threshold)
        || input.estimated_build_rows.is_some_and(|rows| rows >= options.large_keyed_hash_build_row_threshold)
        || input.build_side_bytes.is_some_and(|bytes| bytes >= options.large_keyed_hash_build_bytes_threshold)
}

/// Whether measuring build-key uniqueness could move a keyed left join from
/// partitioning to a one-shot join: the choice is not one-shot without the
/// measurement and would be with a unique build key.
pub fn keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
    form: JoinForm,
    join: &JoinNode,
    input: KeyedHashJoinStrategyInputFacts,
    options: &KeyedHashJoinStrategyOptions,
) -> bool {
    if form != JoinForm::KeyedEqui || join.spec.join_type != JoinType::Left || input.build_key_unique.is_some() {
        return false;
    }
    if keyed_equi_strategy_choice(join, input, options).strategy() == JoinStrategy::OneShotEqui {
        return false;
    }
    keyed_equi_strategy_choice(join, KeyedHashJoinStrategyInputFacts { build_key_unique: Some(true), ..input }, options)
        .strategy()
        == JoinStrategy::OneShotEqui
}

/// Whether a probe kernel of `strategy` pulls its probe input chunk by
/// chunk, so a frame-stream probe input stays chunked instead of being
/// concatenated first.
pub(crate) fn join_probe_streams_chunks(join: &JoinNode, form: JoinForm, strategy: JoinStrategy) -> bool {
    match form {
        JoinForm::KeyedEqui | JoinForm::KeyedFilteredSemiAnti | JoinForm::KeylessOuterCross => {
            match join.spec.join_type {
                JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } => true,
                JoinType::Left => join.spec.build_side == JoinSide::Right,
                JoinType::Full | JoinType::Cross => false,
            }
        },
        JoinForm::KeyedEquiPostFilter => strategy == JoinStrategy::OneShotEqui,
        JoinForm::KeyedResidual { left_match } => left_match && strategy == JoinStrategy::MixedKeyedResidual,
        JoinForm::KeylessConditional(_) | JoinForm::ProjectedCross => false,
    }
}

fn expr_requires_post_join_filter(expr: &Expr) -> bool {
    expr.any(|expr| {
        matches!(
            expr,
            Expr::Case { .. }
                | Expr::IsDistinctFrom { .. }
                | Expr::IsNotDistinctFrom { .. }
                | Expr::Binary { op: BinaryOp::Divide, output_type: Some(DataType::Decimal128(_, _)), .. }
        )
    })
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
            inputs: [NodeId::from_index_for_test(0), NodeId::from_index_for_test(1)],
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

    fn form(join: &JoinNode) -> Option<JoinForm> {
        classify_join(NodeId::from_index_for_test(2), join).ok()
    }

    fn choice(
        join: &JoinNode,
        input: KeyedHashJoinStrategyInputFacts,
        options: &KeyedHashJoinStrategyOptions,
    ) -> JoinStrategyChoice {
        choose_join_strategy(form(join).expect("test join has an operator form"), join, input, options)
    }

    fn keyed_hash_join_strategy_decision(
        join: &JoinNode,
        input: KeyedHashJoinStrategyInputFacts,
    ) -> JoinStrategyChoice {
        choice(join, input, &Options::default())
    }

    fn selected_strategy(choice: JoinStrategyChoice) -> JoinStrategy {
        choice.strategy()
    }

    #[test]
    fn known_small_probe_single_use_inner_join_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(Some(10), None, Some(Options::DEFAULT_SMALL_ONE_SHOT_PROBE_BYTES_THRESHOLD / 2)),
        );

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn supported_keyed_residual_inner_join_stays_on_mixed_strategy() {
        let join = single_use_inner_join_with_residual(Expr::binary(
            Expr::column("left_value"),
            BinaryOp::Gt,
            Expr::column("right_value"),
        ));

        assert_eq!(form(&join), Some(JoinForm::KeyedResidual { left_match: false }));
    }

    #[test]
    fn null_aware_anti_join_executes_only_on_the_filtered_strategy() {
        let mut join = single_use_join(JoinType::LeftAnti { null_aware: true });
        join.spec = join.spec.with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "left_value", "left_value"),
            "left_value",
        )]);
        assert_eq!(form(&join), Some(JoinForm::KeyedFilteredSemiAnti));

        join.spec = join.spec.with_residual_predicate(Expr::binary(
            Expr::column("left_value"),
            BinaryOp::Gt,
            Expr::column("right_value"),
        ));
        assert_eq!(form(&join), None);
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

        assert_eq!(form(&join), Some(JoinForm::KeyedEquiPostFilter));
        assert_eq!(selected_strategy(decision), JoinStrategy::OneShotEqui);
    }

    #[test]
    fn unknown_probe_medium_single_use_inner_join_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None, None),
        );

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_tiny_estimated_build_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(&single_use_inner_join(), input_facts(Some(10), None, None));

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_tiny_observed_without_estimate_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(&single_use_inner_join(), input_facts(None, Some(31), None));

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
    }

    #[test]
    fn unknown_probe_medium_observed_without_estimate_stays_reusable() {
        let decision = keyed_hash_join_strategy_decision(
            &single_use_inner_join(),
            input_facts(None, Some(Options::DEFAULT_SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD / 2), None),
        );

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
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

        assert_eq!(selected_strategy(keyed_hash_join_strategy_decision(&join, input)), JoinStrategy::ReusableHashEqui);
        let decision = choice(&join, input, &options);
        let facts = decision.evidence().expect("a keyed choice carries its evidence").facts();

        assert_eq!(selected_strategy(decision), JoinStrategy::PartitionedHashEqui);
        assert_eq!(facts.get("small_reusable_hash_build_row_threshold"), Some(&"5".to_string()));
        assert_eq!(facts.get("small_reusable_hash_build_bytes_threshold"), Some(&"1024".to_string()));
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

        assert_eq!(selected_strategy(decision), JoinStrategy::OneShotEqui);
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

        assert_eq!(selected_strategy(decision), JoinStrategy::ReusableHashEqui);
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

        assert_eq!(selected_strategy(decision), JoinStrategy::PartitionedHashEqui);
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

        assert_eq!(selected_strategy(decision), JoinStrategy::PartitionedHashEqui);
    }

    #[test]
    fn left_join_with_small_probe_and_contained_build_requests_uniqueness_probe() {
        assert!(keyed_hash_join_strategy_should_probe_left_one_shot_build_key_uniqueness(
            JoinForm::KeyedEqui,
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
            &Options::default(),
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

        assert_eq!(selected_strategy(decision), JoinStrategy::PartitionedHashEqui);
    }
}
