use crate::plan::{JoinSide, JoinType, NodeId, PlanNode, QueryPlan, estimated_output_bytes, estimated_output_rows};

/// Stable rule name used in optimizer diagnostics.
pub const JOIN_BUILD_SIDE_SELECTION_RULE_NAME: &str = "join_build_side_selection";

const DEFAULT_SMALL_BUILD_SIDE_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_SMALL_BUILD_SIDE_ROWS: u64 = 1_000_000;

/// Configuration for choosing smaller hash-join build sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinBuildSideSelectionOptions {
    /// Whether the pass is allowed to change join build-side orientation.
    pub enabled: bool,
    /// Byte threshold under which a side is treated as a small build candidate.
    pub small_build_side_bytes: u64,
    /// Row threshold under which a side is treated as a small build candidate.
    pub small_build_side_rows: u64,
}

impl JoinBuildSideSelectionOptions {
    /// Returns options with the pass disabled and default thresholds preserved.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            small_build_side_bytes: DEFAULT_SMALL_BUILD_SIDE_BYTES,
            small_build_side_rows: DEFAULT_SMALL_BUILD_SIDE_ROWS,
        }
    }
}

impl Default for JoinBuildSideSelectionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            small_build_side_bytes: DEFAULT_SMALL_BUILD_SIDE_BYTES,
            small_build_side_rows: DEFAULT_SMALL_BUILD_SIDE_ROWS,
        }
    }
}

/// Diagnostics for build-side selection candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinBuildSideSelectionDiagnostics {
    /// Number of join candidates considered.
    pub considered_count: usize,
    /// Number of joins whose build side changed.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<JoinBuildSideSelectionCandidateDiagnostics>,
}

impl JoinBuildSideSelectionDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate build-side selection evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinBuildSideSelectionCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Join node considered by the pass.
    pub candidate_node_id: NodeId,
    /// Build side before optimization.
    pub before_build_side: JoinSide,
    /// Build side after optimization when changed.
    pub after_build_side: Option<JoinSide>,
    /// Estimated rows on the left input, if known.
    pub left_estimated_rows: Option<u64>,
    /// Estimated rows on the right input, if known.
    pub right_estimated_rows: Option<u64>,
    /// Estimated bytes on the left input, if known.
    pub left_estimated_bytes: Option<u64>,
    /// Estimated bytes on the right input, if known.
    pub right_estimated_bytes: Option<u64>,
    /// True when the returned plan uses the selected build side.
    pub applied: bool,
    /// Rejection reason when no change was applied.
    pub rejected_reason: Option<JoinBuildSideSelectionRejectedReason>,
}

/// Reasons a build-side candidate was not changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinBuildSideSelectionRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// Join shape cannot safely flip build/probe orientation.
    UnsupportedJoinShape,
    /// Existing build side was already best or estimates were insufficient.
    NoBetterBuildSide,
}

pub(crate) struct JoinBuildSideSelectionResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: JoinBuildSideSelectionDiagnostics,
}

pub(crate) fn optimize_join_build_side_selection(
    plan: QueryPlan,
    options: &JoinBuildSideSelectionOptions,
) -> JoinBuildSideSelectionResult {
    if !options.enabled {
        return JoinBuildSideSelectionResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: JoinBuildSideSelectionDiagnostics::default(),
        };
    }

    let mut nodes = plan.nodes().to_vec();
    let mut diagnostics = JoinBuildSideSelectionDiagnostics::default();
    let mut changed = false;

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_build_side_selection_supported(join) {
            continue;
        }

        diagnostics.considered_count += 1;
        let left = SideEstimate::for_node(&plan, join.inputs[0]);
        let right = SideEstimate::for_node(&plan, join.inputs[1]);
        let before = join.spec.build_side;
        let selected = select_build_side(before, left, right, options);
        let mut candidate = JoinBuildSideSelectionCandidateDiagnostics {
            rule_name: JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
            candidate_node_id: node.id,
            before_build_side: before,
            after_build_side: None,
            left_estimated_rows: left.rows,
            right_estimated_rows: right.rows,
            left_estimated_bytes: left.bytes,
            right_estimated_bytes: right.bytes,
            applied: false,
            rejected_reason: Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide),
        };

        let Some(after) = selected else {
            diagnostics.candidates.push(candidate);
            continue;
        };
        if after == before {
            diagnostics.candidates.push(candidate);
            continue;
        }

        if let Some(rewritten) = nodes.get_mut(node.id.index())
            && let PlanNode::Join(rewritten_join) = &mut rewritten.kind
        {
            rewritten_join.spec.build_side = after;
            candidate.after_build_side = Some(after);
            candidate.applied = true;
            candidate.rejected_reason = None;
            diagnostics.applied_count += 1;
            changed = true;
        }
        diagnostics.candidates.push(candidate);
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    let sink = plan.sink().clone();
    let root = plan.root_id();
    JoinBuildSideSelectionResult { plan: QueryPlan::from_rewritten_parts(nodes, root, sink), change, diagnostics }
}

fn join_build_side_selection_supported(join: &crate::plan::JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.build_artifact_id.is_none()
}

fn select_build_side(
    current: JoinSide,
    left: SideEstimate,
    right: SideEstimate,
    options: &JoinBuildSideSelectionOptions,
) -> Option<JoinSide> {
    if let Some(side) = smaller_known_side(left.bytes, right.bytes) {
        return Some(side);
    }
    if let Some(side) = smaller_known_side(left.rows, right.rows) {
        return Some(side);
    }

    let other = opposite_side(current);
    let other_estimate = match other {
        JoinSide::Left => left,
        JoinSide::Right => right,
    };
    if other_estimate.is_small(options) { Some(other) } else { None }
}

fn smaller_known_side(left: Option<u64>, right: Option<u64>) -> Option<JoinSide> {
    match (left, right) {
        (Some(left), Some(right)) if left < right => Some(JoinSide::Left),
        (Some(left), Some(right)) if right < left => Some(JoinSide::Right),
        _ => None,
    }
}

fn opposite_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

#[derive(Debug, Clone, Copy)]
struct SideEstimate {
    rows: Option<u64>,
    bytes: Option<u64>,
}

impl SideEstimate {
    fn for_node(plan: &QueryPlan, node_id: NodeId) -> Self {
        Self { rows: estimated_output_rows(plan, node_id), bytes: estimated_output_bytes(plan, node_id) }
    }

    fn is_small(self, options: &JoinBuildSideSelectionOptions) -> bool {
        self.bytes.is_some_and(|bytes| bytes <= options.small_build_side_bytes)
            || self.rows.is_some_and(|rows| rows <= options.small_build_side_rows)
    }
}
