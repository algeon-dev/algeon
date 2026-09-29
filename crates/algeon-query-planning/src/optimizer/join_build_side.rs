use crate::{
    capability::classify_join,
    error::{QueryError, QueryErrorCode, Result},
    expr::{ColumnRef, Expr},
    plan::{
        JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, RewrittenNodes, advisory_output_bytes,
        advisory_output_rows, estimated_output_bytes, estimated_output_rows,
    },
};

/// Stable rule name used in optimizer diagnostics.
pub const JOIN_BUILD_SIDE_SELECTION_RULE_NAME: &str = "join_build_side_selection";

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
    /// Default byte threshold under which a side is treated as a small build candidate.
    pub const DEFAULT_SMALL_BUILD_SIDE_BYTES: u64 = 64 * 1024 * 1024;
    /// Default row threshold under which a side is treated as a small build candidate.
    pub const DEFAULT_SMALL_BUILD_SIDE_ROWS: u64 = 1_000_000;

    /// Returns options with the pass disabled and default thresholds preserved.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            small_build_side_bytes: Self::DEFAULT_SMALL_BUILD_SIDE_BYTES,
            small_build_side_rows: Self::DEFAULT_SMALL_BUILD_SIDE_ROWS,
        }
    }

    /// Validates internal consistency for build-side selection thresholds.
    pub fn validate(&self) -> Result<()> {
        validate_nonzero_join_build_side_option("small_build_side_bytes", self.small_build_side_bytes)?;
        validate_nonzero_join_build_side_option("small_build_side_rows", self.small_build_side_rows)?;
        Ok(())
    }
}

fn validate_nonzero_join_build_side_option(field: &'static str, value: u64) -> Result<()> {
    if value == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.join_build_side_selection_options.validate",
            "join build-side selection option must be greater than zero",
        )
        .with_fact("field", field)
        .with_fact("value", value)
        .with_fact("reason", "zero"));
    }
    Ok(())
}

impl Default for JoinBuildSideSelectionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            small_build_side_bytes: Self::DEFAULT_SMALL_BUILD_SIDE_BYTES,
            small_build_side_rows: Self::DEFAULT_SMALL_BUILD_SIDE_ROWS,
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
    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate build-side selection evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinBuildSideSelectionCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
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
    /// Frontend advisory rows on the left input, if the frontend estimated them.
    pub left_advisory_rows: Option<u64>,
    /// Frontend advisory rows on the right input, if the frontend estimated them.
    pub right_advisory_rows: Option<u64>,
    /// Frontend advisory bytes on the left input, if the frontend estimated them.
    pub left_advisory_bytes: Option<u64>,
    /// Frontend advisory bytes on the right input, if the frontend estimated them.
    pub(crate) right_advisory_bytes: Option<u64>,
    /// Configured byte threshold under which a side is treated as a small build candidate.
    pub(crate) small_build_side_bytes: u64,
    /// Configured row threshold under which a side is treated as a small build candidate.
    pub(crate) small_build_side_rows: u64,
    /// Evidence that decided the selected side, when a side was selected.
    pub selection_basis: Option<JoinBuildSideSelectionBasis>,
    /// True when the returned plan uses the selected build side.
    pub applied: bool,
    /// Rejection reason when no change was applied.
    pub rejected_reason: Option<JoinBuildSideSelectionRejectedReason>,
}

/// Evidence class that decided a build side, in the order the pass consults it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinBuildSideSelectionBasis {
    /// Both sides had conservative byte estimates and one was smaller.
    ConservativeBytes,
    /// Both sides had conservative row estimates and one was smaller.
    ConservativeRows,
    /// The current build side is conservatively small.
    SmallCurrentSide,
    /// The other side is conservatively small.
    SmallOtherSide,
    /// A frontend advisory byte estimate stood in for an unknown side.
    AdvisoryBytes,
    /// A frontend advisory row estimate stood in for an unknown side.
    AdvisoryRows,
    /// Exactly one side is keyed on a grouped-aggregate key set.
    GroupedAggregateKeyset,
}

impl JoinBuildSideSelectionBasis {
    /// Stable diagnostic string for this basis.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ConservativeBytes => "conservative_bytes",
            Self::ConservativeRows => "conservative_rows",
            Self::SmallCurrentSide => "small_current_side",
            Self::SmallOtherSide => "small_other_side",
            Self::AdvisoryBytes => "advisory_bytes",
            Self::AdvisoryRows => "advisory_rows",
            Self::GroupedAggregateKeyset => "grouped_aggregate_keyset",
        }
    }
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
    pub(crate) plan: QueryPlan,
    pub(crate) change: super::NativeOptimizerChange,
    pub(crate) diagnostics: JoinBuildSideSelectionDiagnostics,
}

pub(crate) fn optimize_join_build_side_selection(
    plan: QueryPlan,
    options: &JoinBuildSideSelectionOptions,
) -> Result<JoinBuildSideSelectionResult> {
    if !options.enabled {
        return Ok(JoinBuildSideSelectionResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: JoinBuildSideSelectionDiagnostics::default(),
        });
    }

    let mut nodes = plan.nodes().to_vec();
    let mut diagnostics = JoinBuildSideSelectionDiagnostics::default();
    let mut changed = false;

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_build_side_selection_supported(node.id, join) {
            continue;
        }

        diagnostics.considered_count += 1;
        let left = SideEstimate::for_join_side(&plan, join, JoinSide::Left);
        let right = SideEstimate::for_join_side(&plan, join, JoinSide::Right);
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
            left_advisory_rows: left.advisory_rows,
            right_advisory_rows: right.advisory_rows,
            left_advisory_bytes: left.advisory_bytes,
            right_advisory_bytes: right.advisory_bytes,
            small_build_side_bytes: options.small_build_side_bytes,
            small_build_side_rows: options.small_build_side_rows,
            selection_basis: selected.map(|(_, basis)| basis),
            applied: false,
            rejected_reason: Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide),
        };

        let Some((after, _)) = selected else {
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
    let root = plan.root_id();
    Ok(JoinBuildSideSelectionResult { plan: plan.rewrite(RewrittenNodes::in_place(nodes, root))?, change, diagnostics })
}

/// Keyed inner, semi, and anti joins whose build side no reusable artifact
/// fixes are candidates when a native executor runs either direction.
fn join_build_side_selection_supported(node_id: NodeId, join: &JoinNode) -> bool {
    if join.spec.keys.is_empty()
        || join.spec.build_artifact_id.is_some()
        || !matches!(join.spec.join_type, JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti { .. })
    {
        return false;
    }
    [JoinSide::Left, JoinSide::Right].into_iter().all(|build_side| {
        let mut oriented = join.clone();
        oriented.spec = oriented.spec.with_build_side(build_side);
        classify_join(node_id, &oriented).is_ok()
    })
}

fn select_build_side(
    current: JoinSide,
    left: SideEstimate,
    right: SideEstimate,
    options: &JoinBuildSideSelectionOptions,
) -> Option<(JoinSide, JoinBuildSideSelectionBasis)> {
    if let Some(side) = smaller_known_side(left.bytes, right.bytes) {
        return Some((side, JoinBuildSideSelectionBasis::ConservativeBytes));
    }
    if let Some(side) = smaller_known_side(left.rows, right.rows) {
        return Some((side, JoinBuildSideSelectionBasis::ConservativeRows));
    }
    let current_estimate = match current {
        JoinSide::Left => left,
        JoinSide::Right => right,
    };
    if current_estimate.is_small(options) {
        return Some((current, JoinBuildSideSelectionBasis::SmallCurrentSide));
    }
    let other = opposite_side(current);
    let other_estimate = match other {
        JoinSide::Left => left,
        JoinSide::Right => right,
    };
    if other_estimate.is_small(options) {
        return Some((other, JoinBuildSideSelectionBasis::SmallOtherSide));
    }
    // Conservative evidence could not order the sides and neither side is
    // certainly small. A frontend advisory estimate now stands in for each
    // side that has no conservative estimate; a conservative estimate is never
    // replaced by an advisory one, so a wrong advisory can only lose to a
    // known bound, never override it.
    if let Some(side) = smaller_known_side(left.bytes.or(left.advisory_bytes), right.bytes.or(right.advisory_bytes)) {
        return Some((side, JoinBuildSideSelectionBasis::AdvisoryBytes));
    }
    if let Some(side) = smaller_known_side(left.rows.or(left.advisory_rows), right.rows.or(right.advisory_rows)) {
        return Some((side, JoinBuildSideSelectionBasis::AdvisoryRows));
    }
    if let Some(side) = grouped_aggregate_keyset_side(left, right) {
        return Some((side, JoinBuildSideSelectionBasis::GroupedAggregateKeyset));
    }

    None
}

fn grouped_aggregate_keyset_side(left: SideEstimate, right: SideEstimate) -> Option<JoinSide> {
    match (left.has_grouped_aggregate_keyset, right.has_grouped_aggregate_keyset) {
        (true, false) => Some(JoinSide::Left),
        (false, true) => Some(JoinSide::Right),
        _ => None,
    }
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
    advisory_rows: Option<u64>,
    advisory_bytes: Option<u64>,
    has_grouped_aggregate_keyset: bool,
}

impl SideEstimate {
    fn for_join_side(plan: &QueryPlan, join: &JoinNode, side: JoinSide) -> Self {
        let node_id = join.inputs[join_side_index(side)];
        Self {
            rows: estimated_output_rows(plan, node_id).or_else(|| bounded_left_semijoin_rows(plan, node_id)),
            bytes: estimated_output_bytes(plan, node_id).or_else(|| bounded_left_semijoin_bytes(plan, node_id)),
            advisory_rows: advisory_output_rows(plan, node_id),
            advisory_bytes: advisory_output_bytes(plan, node_id),
            has_grouped_aggregate_keyset: join_side_keys_are_grouped_aggregate_keyset(plan, join, side),
        }
    }

    fn is_small(self, options: &JoinBuildSideSelectionOptions) -> bool {
        self.bytes.is_some_and(|bytes| bytes <= options.small_build_side_bytes)
            || self.rows.is_some_and(|rows| rows <= options.small_build_side_rows)
    }
}

fn bounded_left_semijoin_rows(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Projection(projection) => bounded_left_semijoin_rows(plan, projection.input),
        PlanNode::Filter(filter) => bounded_left_semijoin_rows(plan, filter.input),
        PlanNode::Window(window) => bounded_left_semijoin_rows(plan, window.input),
        PlanNode::Sort(sort) => bounded_left_semijoin_rows(plan, sort.input),
        PlanNode::Limit(limit) => bounded_left_semijoin_rows(plan, limit.input).map(|rows| {
            limit.spec.fetch.map_or(rows.saturating_sub(limit.spec.skip as u64), |fetch| {
                rows.min(limit.spec.skip.saturating_add(fetch) as u64).saturating_sub(limit.spec.skip as u64)
            })
        }),
        PlanNode::Join(join)
            if matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti { .. })
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left) =>
        {
            estimated_output_rows(plan, join.inputs[0]).or_else(|| bounded_left_semijoin_rows(plan, join.inputs[0]))
        },
        PlanNode::Source(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Union(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

fn bounded_left_semijoin_bytes(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Projection(projection) => bounded_left_semijoin_bytes(plan, projection.input),
        PlanNode::Filter(filter) => bounded_left_semijoin_bytes(plan, filter.input),
        PlanNode::Sort(sort) => bounded_left_semijoin_bytes(plan, sort.input),
        PlanNode::Join(join)
            if matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti { .. })
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left) =>
        {
            estimated_output_bytes(plan, join.inputs[0]).or_else(|| bounded_left_semijoin_bytes(plan, join.inputs[0]))
        },
        PlanNode::Source(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

fn join_side_keys_are_grouped_aggregate_keyset(plan: &QueryPlan, join: &JoinNode, side: JoinSide) -> bool {
    !join.spec.keys.is_empty()
        && join.spec.keys.iter().all(|key| {
            let input_id = join.inputs[join_side_index(side)];
            let column = match side {
                JoinSide::Left => &key.left,
                JoinSide::Right => &key.right,
            };
            output_column_is_grouped_aggregate_key(plan, input_id, column)
        })
}

fn output_column_is_grouped_aggregate_key(plan: &QueryPlan, node_id: NodeId, column: &ColumnRef) -> bool {
    resolve_column_field_index(plan, node_id, column)
        .is_some_and(|field_index| output_field_is_grouped_aggregate_key(plan, node_id, field_index))
}

fn output_field_is_grouped_aggregate_key(plan: &QueryPlan, node_id: NodeId, field_index: usize) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if field_index >= node.output_schema().fields().len() {
        return false;
    }
    match &node.kind {
        PlanNode::Source(_) => false,
        PlanNode::Filter(filter) => output_field_is_grouped_aggregate_key(plan, filter.input, field_index),
        PlanNode::Sort(sort) => output_field_is_grouped_aggregate_key(plan, sort.input, field_index),
        PlanNode::Limit(limit) => output_field_is_grouped_aggregate_key(plan, limit.input, field_index),
        PlanNode::Projection(projection) => projection
            .outputs
            .get(field_index)
            .is_some_and(|output| expr_is_grouped_aggregate_key(plan, projection.input, &output.expr)),
        PlanNode::Aggregate(aggregate) => {
            aggregate.spec.grouping_sets.is_none() && field_index < aggregate.spec.group_exprs.len()
        },
        PlanNode::Join(join) => join.spec.output_projection.get(field_index).is_some_and(|output| {
            let input_id = join.inputs[join_side_index(output.side)];
            resolve_column_field_index(plan, input_id, &output.column).is_some_and(|input_field_index| {
                output_field_is_grouped_aggregate_key(plan, input_id, input_field_index)
            })
        }),
        PlanNode::Window(window) => plan.node(window.input).is_some_and(|input| {
            field_index < input.output_schema().fields().len()
                && output_field_is_grouped_aggregate_key(plan, window.input, field_index)
        }),
        PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

fn expr_is_grouped_aggregate_key(plan: &QueryPlan, input_id: NodeId, expr: &Expr) -> bool {
    match expr {
        Expr::Column(column) => output_column_is_grouped_aggregate_key(plan, input_id, column),
        Expr::Alias { expr, .. } => expr_is_grouped_aggregate_key(plan, input_id, expr),
        _ => false,
    }
}

fn resolve_column_field_index(plan: &QueryPlan, input_id: NodeId, column: &ColumnRef) -> Option<usize> {
    let node = plan.node(input_id)?;
    let mut matches = node
        .output_schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| (field.name() == column.name()).then_some(index));
    let index = matches.next()?;
    matches.next().is_none().then_some(index)
}

const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}
