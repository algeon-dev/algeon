use std::collections::BTreeMap;

use arrow::datatypes::SchemaRef;

use crate::{
    expr::{AggregateFunction, ColumnRef, Expr},
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode,
        QueryPlan, SortNode,
    },
    runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy},
};

const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_FAMILY: &str = "keyed_hash_join_resident_subplan_pipeline";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER: &str =
    "bounded_keyed_hash_join_resident_subplan_lower_level_contracts";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_SHAPE_CONTRACT: &str =
    "sort_root_keyed_inner_hash_join_with_projection_wrapped_keyed_inner_join_and_scalar_revenue_aggregate";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT: &str =
    "source_contract,aggregate_contract,final_sort_contract";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS: &str = "source_read,operator_output,final_sort";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_STATUSES: &str = "source_read:blocked_by_lower_level,operator_output:blocked_by_lower_level,build_side:modeled,hash_table:modeled,probe_side:modeled,filtered_probe_state:modeled,mixed_size_data:modeled,gather_map:modeled,join_output:modeled,final_sort:blocked_by_lower_level,materialization:modeled";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_REQUIREMENTS: &str = "source_read:source_contract:blocked_by_lower_level,operator_output:aggregate_contract:blocked_by_lower_level,build_side:query_engine_build_side_source_reservation:modeled_before_execution,hash_table:hash_join_pre_build_reservation_size_bytes:modeled_by_cudf_rs_pre_build_sizing,probe_side:query_engine_probe_side_source_reservation:modeled_by_ordered_bounded_join_execution_path,filtered_probe_state:filtered_join_retained_probe_state:modeled_by_cudf_rs_filtered_probe_state_observability,mixed_size_data:mixed_keyed_residual_or_conditional_full_retained_size_data:modeled_by_cudf_rs_retained_join_size_data_observability,gather_map:join_gather_map_reservation_size_bytes:modeled_by_cudf_rs_join_gather_map_sizing,join_output:dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes:modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing,final_sort:final_sort_contract:blocked_by_lower_level,materialization:query_engine_explicit_sink_materialization_reservation:modeled_by_explicit_sink_boundary";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_MODELED_RESERVATIONS: &str =
    "build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";

/// Bounded join admission result, ordered from the narrowest root shape to
/// progressively broader bounded pipeline families.
pub(crate) enum BoundedJoinPlanSupport {
    /// The sink input is one supported root join with source/filter/projection
    /// leaf inputs.
    RootJoin,
    /// One supported root join followed by a post-root unary pipeline.
    SingleJoinPostRootPipeline,
    /// A keyless conditional root join whose inputs may contain bounded
    /// projection-wrapped subplans such as left-anti joins or scalar aggregates.
    KeylessConditionalJoinInputSubplanPipeline,
    /// A keyless conditional root join whose left and right inputs are
    /// projection-wrapped aggregates over bounded join subplans.
    KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    /// A supported multi-join subtree followed by a post-root unary pipeline.
    MultiJoinTreePostRootPipeline,
    /// The plan contains joins but does not match any bounded join family.
    Unsupported,
}

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

/// A normalized bounded join pipeline.
///
/// `stages` are stored in execution order: the `join` runs first, followed by
/// each post-root stage from the join output toward the sink.
pub(crate) struct BoundedJoinPipelineShape<'a> {
    pub(crate) join_id: NodeId,
    pub(crate) join: &'a JoinNode,
    pub(crate) stages: Vec<BoundedJoinPipelineStage<'a>>,
}

/// A post-root bounded pipeline stage.
///
/// Stages are executed in vector order after the root join or join subtree has
/// produced a resident GPU frame.
pub(crate) enum BoundedJoinPipelineStage<'a> {
    /// Apply a boolean predicate to the resident join output.
    Filter { node_id: NodeId, node: &'a FilterNode },
    /// Apply a projection and use `output_schema` as the resulting schema.
    Projection { node_id: NodeId, node: &'a ProjectionNode, output_schema: &'a SchemaRef },
    /// Apply a grouped or scalar aggregate to the resident input frame.
    Aggregate { node_id: NodeId, node: &'a AggregateNode, output_schema: &'a SchemaRef },
    /// Apply a global sort to the resident input frame.
    Sort { node_id: NodeId, node: &'a SortNode },
    /// Apply a limit that must be directly above a global sort in the plan.
    Limit { node_id: NodeId, node: &'a LimitNode },
}

/// Why a candidate did not satisfy a bounded join pipeline contract.
pub(crate) enum BoundedJoinPipelineRejection<'a> {
    /// A referenced plan node was absent.
    MissingNode { node_id: NodeId, message: &'static str },
    /// The root or subtree join does not belong to the admitted join family.
    UnsupportedJoin { join: &'a JoinNode },
    /// A reachable plan node is not admitted at its position in the shape.
    UnsupportedShapeNode { node_id: NodeId, node_kind: &'static str },
    /// A `Limit` node was not directly above a `Sort` node.
    LimitNotDirectlyAboveSort,
    /// No join was found along the sink-input spine.
    MissingRootJoin,
    /// A single-join pipeline candidate reached more than one join.
    MultipleReachableJoins,
    /// A multi-join pipeline candidate reached only one join.
    SingleReachableJoin,
}

pub(crate) fn analyze_bounded_join_plan_support(plan: &QueryPlan) -> BoundedJoinPlanSupport {
    if let Some(node) = plan.node(plan.sink().input())
        && let PlanNode::Join(join) = &node.kind
        && bounded_root_join_compatible(join)
        && reachable_join_count(plan) == 1
        && bounded_join_source_leaf_inputs_compatible(plan, join)
    {
        return BoundedJoinPlanSupport::RootJoin;
    }

    match analyze_bounded_join_pipeline(plan) {
        Ok(_) => BoundedJoinPlanSupport::SingleJoinPostRootPipeline,
        Err(_) => match analyze_bounded_keyless_conditional_join_input_subplan_pipeline(plan) {
            Ok(_) => BoundedJoinPlanSupport::KeylessConditionalJoinInputSubplanPipeline,
            Err(_) => {
                match analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(plan) {
                    Ok(_) => BoundedJoinPlanSupport::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
                    Err(_) => match analyze_bounded_multi_join_pipeline(plan) {
                        Ok(_) => BoundedJoinPlanSupport::MultiJoinTreePostRootPipeline,
                        Err(_) => BoundedJoinPlanSupport::Unsupported,
                    },
                }
            },
        },
    }
}

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    if matches_keyed_hash_join_resident_subplan_pipeline(plan) {
        facts.insert("bounded_join_blocked_join_family", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_FAMILY.to_string());
        facts.insert("bounded_join_specific_blocker", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER.to_string());
        facts.insert("bounded_join_shape_blocker", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER.to_string());
        facts
            .insert("blocked_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS.to_string());
        facts
            .insert("missing_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS.to_string());
        facts
            .insert("modeled_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_MODELED_RESERVATIONS.to_string());
        facts.insert(
            "bounded_join_reservation_statuses",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_STATUSES.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_requirements",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_REQUIREMENTS.to_string(),
        );
        facts.insert(
            "bounded_join_required_shape_contract",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_SHAPE_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_required_lower_level_contract",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_lower_level_blockers",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
    } else if let Some(join) = first_unsupported_keyed_residual_semi_anti_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_only_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left_semi_left_anti".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_only_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_only_join_shape_or_orientation".to_string(),
        );
    } else if let Some(join) = first_keyed_residual_full_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_full_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_full".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_full_shape_contract(plan, join));
        facts.insert("bounded_join_shape_blocker", "bounded_mixed_keyed_residual_full_join_shape".to_string());
    } else if let Some(join) = first_keyed_residual_left_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_join_shape_or_orientation".to_string(),
        );
    }
    facts
}

pub(crate) fn analyze_bounded_join_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    analyze_bounded_join_pipeline_shape(plan, true)
}

pub(crate) fn analyze_bounded_multi_join_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let shape = analyze_bounded_join_pipeline_shape(plan, false)?;
    let join_count = bounded_join_subtree_join_count(plan, shape.join_id)?;
    if join_count <= 1 {
        return Err(BoundedJoinPipelineRejection::SingleReachableJoin);
    }
    validate_bounded_multi_join_subtree(plan, shape.join_id)?;
    Ok(shape)
}

pub(crate) fn analyze_bounded_keyless_conditional_join_input_subplan_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let shape = analyze_bounded_join_pipeline_shape(plan, false)?;
    if !bounded_root_keyless_conditional_join_input_subplan_pipeline_compatible(shape.join) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: shape.join });
    }
    if !bounded_join_inputs_include_resident_subplan(plan, shape.join)? {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: shape.join });
    }
    validate_bounded_keyless_conditional_join_input_subplan(plan, shape.join.inputs[0])?;
    validate_bounded_keyless_conditional_join_input_subplan(plan, shape.join.inputs[1])?;
    Ok(shape)
}

pub(crate) fn analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let shape = analyze_bounded_join_pipeline_shape(plan, false)?;
    if !bounded_root_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_compatible(shape.join) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: shape.join });
    }
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_post_root_pipeline(&shape)?;
    validate_projection_wrapped_aggregate_over_join_subplan_input(
        plan,
        shape.join.inputs[0],
        AggregateOverJoinSubplanInputKind::Grouped,
    )?;
    validate_projection_wrapped_aggregate_over_join_subplan_input(
        plan,
        shape.join.inputs[1],
        AggregateOverJoinSubplanInputKind::Scalar,
    )?;
    Ok(shape)
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

fn analyze_bounded_join_pipeline_shape(
    plan: &QueryPlan,
    require_single_reachable_join: bool,
) -> std::result::Result<BoundedJoinPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let mut current = plan.sink().input();
    let mut stages = Vec::new();

    loop {
        let node = plan.node(current).ok_or(BoundedJoinPipelineRejection::MissingNode {
            node_id: current,
            message: "bounded join pipeline references a missing plan node",
        })?;
        match &node.kind {
            PlanNode::Join(join) => {
                if !bounded_root_join_compatible(join) {
                    return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
                }
                if require_single_reachable_join && reachable_join_count(plan) != 1 {
                    return Err(BoundedJoinPipelineRejection::MultipleReachableJoins);
                }
                if require_single_reachable_join {
                    validate_bounded_join_source_leaf_inputs(plan, join)?;
                }
                stages.reverse();
                return Ok(BoundedJoinPipelineShape { join_id: current, join, stages });
            },
            PlanNode::Filter(filter) => {
                stages.push(BoundedJoinPipelineStage::Filter { node_id: current, node: filter });
                current = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(BoundedJoinPipelineStage::Projection {
                    node_id: current,
                    node: projection,
                    output_schema: &node.facts.output_schema,
                });
                current = projection.input;
            },
            PlanNode::Aggregate(aggregate) => {
                stages.push(BoundedJoinPipelineStage::Aggregate {
                    node_id: current,
                    node: aggregate,
                    output_schema: &node.facts.output_schema,
                });
                current = aggregate.input;
            },
            PlanNode::Sort(sort) => {
                stages.push(BoundedJoinPipelineStage::Sort { node_id: current, node: sort });
                current = sort.input;
            },
            PlanNode::Limit(limit) => {
                let child = plan.node(limit.input).ok_or(BoundedJoinPipelineRejection::MissingNode {
                    node_id: limit.input,
                    message: "bounded join pipeline limit references a missing input",
                })?;
                if !matches!(child.kind, PlanNode::Sort(_)) {
                    return Err(BoundedJoinPipelineRejection::LimitNotDirectlyAboveSort);
                }
                stages.push(BoundedJoinPipelineStage::Limit { node_id: current, node: limit });
                current = limit.input;
            },
            PlanNode::Source(_) => return Err(BoundedJoinPipelineRejection::MissingRootJoin),
        }
    }
}

fn bounded_join_source_leaf_inputs_compatible(plan: &QueryPlan, join: &JoinNode) -> bool {
    validate_bounded_join_source_leaf_inputs(plan, join).is_ok()
}

fn validate_bounded_join_source_leaf_inputs<'a>(
    plan: &'a QueryPlan,
    join: &'a JoinNode,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    validate_bounded_join_source_leaf_input(plan, join.inputs[0])?;
    validate_bounded_join_source_leaf_input(plan, join.inputs[1])
}

fn validate_bounded_join_source_leaf_input(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded join input references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_bounded_join_source_leaf_input(plan, filter.input),
        PlanNode::Projection(projection) => validate_bounded_join_source_leaf_input(plan, projection.input),
        PlanNode::Join(_) => Err(BoundedJoinPipelineRejection::MultipleReachableJoins),
        PlanNode::Aggregate(_) => {
            Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "limit" }),
    }
}

fn validate_bounded_keyless_conditional_join_input_subplan(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    validate_bounded_keyless_conditional_join_input_subplan_inner(plan, node_id, false)
}

fn validate_bounded_keyless_conditional_join_input_subplan_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    join_reached_through_projection: bool,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded keyless conditional join input-subplan pipeline references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Projection(projection) => {
            validate_bounded_keyless_conditional_join_input_subplan_inner(plan, projection.input, true)
        },
        PlanNode::Aggregate(aggregate) => {
            if !aggregate.spec.group_exprs.is_empty() {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id,
                    node_kind: "grouped_aggregate",
                });
            }
            validate_bounded_join_source_leaf_input(plan, aggregate.input)
        },
        PlanNode::Join(join) => {
            if !join_reached_through_projection {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "bare_join" });
            }
            if !bounded_keyless_conditional_join_input_subplan_join_compatible(join) {
                return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
            }
            validate_bounded_join_source_leaf_input(plan, join.inputs[0])?;
            validate_bounded_join_source_leaf_input(plan, join.inputs[1])
        },
        PlanNode::Source(_) | PlanNode::Filter(_) => validate_bounded_join_source_leaf_input(plan, node_id),
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "limit" }),
    }
}

fn bounded_keyless_conditional_join_input_subplan_join_compatible(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::LeftAnti && bounded_root_filtered_join_compatible(join)
}

#[derive(Debug, Clone, Copy)]
enum AggregateOverJoinSubplanInputKind {
    /// The input must be a grouped aggregate over a bounded join subplan.
    Grouped,
    /// The input must be a scalar/global aggregate over a bounded join subplan.
    Scalar,
}

fn validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_post_root_pipeline<'a>(
    shape: &BoundedJoinPipelineShape<'a>,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    let mut seen_projection = false;
    let mut seen_sort = false;
    for stage in &shape.stages {
        match stage {
            BoundedJoinPipelineStage::Projection { node_id, .. } => {
                if seen_projection || seen_sort {
                    return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                        node_id: *node_id,
                        node_kind: "projection",
                    });
                }
                seen_projection = true;
            },
            BoundedJoinPipelineStage::Sort { node_id, .. } => {
                if seen_sort {
                    return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                        node_id: *node_id,
                        node_kind: "sort",
                    });
                }
                seen_sort = true;
            },
            BoundedJoinPipelineStage::Filter { node_id, .. } => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id: *node_id,
                    node_kind: "filter",
                });
            },
            BoundedJoinPipelineStage::Aggregate { node_id, .. } => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id: *node_id,
                    node_kind: "aggregate",
                });
            },
            BoundedJoinPipelineStage::Limit { node_id, .. } => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id: *node_id,
                    node_kind: "limit",
                });
            },
        }
    }
    Ok(())
}

fn validate_projection_wrapped_aggregate_over_join_subplan_input(
    plan: &QueryPlan,
    node_id: NodeId,
    kind: AggregateOverJoinSubplanInputKind,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded keyless conditional join aggregate-over-join subplan inputs pipeline references a missing plan node",
    })?;
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "bare_aggregate_input" });
    };

    let aggregate_node = plan.node(projection.input).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id: projection.input,
        message: "bounded keyless conditional join aggregate-over-join subplan inputs pipeline projection references a missing aggregate",
    })?;
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: projection.input,
            node_kind: "non_aggregate_input",
        });
    };

    match kind {
        AggregateOverJoinSubplanInputKind::Grouped if aggregate.spec.group_exprs.is_empty() => {
            return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                node_id: projection.input,
                node_kind: "scalar_aggregate",
            });
        },
        AggregateOverJoinSubplanInputKind::Scalar if !aggregate.spec.group_exprs.is_empty() => {
            return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                node_id: projection.input,
                node_kind: "grouped_aggregate",
            });
        },
        AggregateOverJoinSubplanInputKind::Grouped | AggregateOverJoinSubplanInputKind::Scalar => {},
    }

    validate_bounded_join_subtree_rooted_at_join(plan, aggregate.input)
}

fn validate_bounded_join_subtree_rooted_at_join(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded keyless conditional join aggregate-over-join subplan inputs pipeline references a missing join subtree",
    })?;
    match &node.kind {
        PlanNode::Join(_) => validate_bounded_multi_join_subtree(plan, node_id),
        PlanNode::Projection(projection) => validate_bounded_join_subtree_rooted_at_join(plan, projection.input),
        _ => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "non_join_subtree_root" }),
    }
}

fn bounded_join_inputs_include_resident_subplan<'a>(
    plan: &'a QueryPlan,
    join: &'a JoinNode,
) -> std::result::Result<bool, BoundedJoinPipelineRejection<'a>> {
    Ok(bounded_join_input_contains_resident_subplan(plan, join.inputs[0])?
        || bounded_join_input_contains_resident_subplan(plan, join.inputs[1])?)
}

fn bounded_join_input_contains_resident_subplan(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<bool, BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded keyless conditional join input-subplan pipeline references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Projection(projection) => bounded_join_input_contains_resident_subplan(plan, projection.input),
        PlanNode::Aggregate(_) | PlanNode::Join(_) => Ok(true),
        PlanNode::Source(_) => Ok(false),
        PlanNode::Filter(filter) => bounded_join_input_contains_resident_subplan(plan, filter.input),
        PlanNode::Sort(_) | PlanNode::Limit(_) => Ok(true),
    }
}

fn validate_bounded_multi_join_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded multi-join pipeline references a missing plan node",
    })?;
    match &node.kind {
        PlanNode::Join(join) => {
            if !bounded_multi_join_compatible(join) {
                return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
            }
            validate_bounded_multi_join_subtree(plan, join.inputs[0])?;
            validate_bounded_multi_join_subtree(plan, join.inputs[1])?;
            Ok(())
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_bounded_multi_join_subtree(plan, filter.input),
        PlanNode::Projection(projection) => validate_bounded_multi_join_subtree(plan, projection.input),
        PlanNode::Aggregate(_) => {
            Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "limit" }),
    }
}

fn bounded_join_subtree_join_count(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<usize, BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded multi-join pipeline references a missing plan node",
    })?;
    let mut count = usize::from(matches!(node.kind, PlanNode::Join(_)));
    for input in node.input_ids() {
        count += bounded_join_subtree_join_count(plan, *input)?;
    }
    Ok(count)
}

fn matches_keyed_hash_join_resident_subplan_pipeline(plan: &QueryPlan) -> bool {
    let Some((final_sort, root_join_id)) = final_sort_root_join_input(plan) else {
        return false;
    };
    let counts = reachable_node_kind_counts(plan);
    if counts.joins != 2 || counts.aggregates != 3 || counts.sorts != 1 || counts.limits != 0 {
        return false;
    }

    let Some(root_join) = join_node(plan, root_join_id) else {
        return false;
    };
    keyed_inner_hash_join_without_residual(root_join)
        && final_sort_is_ascending_supplier_key(final_sort)
        && root_join_matches_q15_revenue_max_join(plan, root_join)
        && validate_q15_supplier_revenue_branch(plan, root_join.inputs[0])
        && validate_q15_scalar_max_revenue_branch(plan, root_join.inputs[1])
}

fn final_sort_root_join_input(plan: &QueryPlan) -> Option<(&SortNode, NodeId)> {
    let node = plan.node(plan.sink().input())?;
    let PlanNode::Sort(sort) = &node.kind else {
        return None;
    };
    Some((sort, sort.input))
}

fn final_sort_is_ascending_supplier_key(sort: &SortNode) -> bool {
    let [sort_expr] = sort.spec.expressions.as_slice() else {
        return false;
    };
    !sort_expr.descending && expr_has_role(&sort_expr.expr, Q15Role::SupplierKey)
}

fn root_join_matches_q15_revenue_max_join(plan: &QueryPlan, join: &JoinNode) -> bool {
    if !join_has_one_key_with_roles(join, Q15Role::TotalRevenue, Q15Role::MaxRevenue) {
        return false;
    }
    root_join_outputs_expected_q15_left_roles(plan, join)
}

fn root_join_outputs_expected_q15_left_roles(plan: &QueryPlan, join: &JoinNode) -> bool {
    if join.spec.output_projection.is_empty()
        || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        return false;
    }

    let Some(left_input_schema) = plan.node(join.inputs[0]).map(|node| node.output_schema()) else {
        return false;
    };
    let projected_roles =
        join.spec.output_projection.iter().filter_map(q15_role_for_join_output_column).collect::<Vec<_>>();
    if projected_roles.len() != join.spec.output_projection.len()
        || !projected_roles.contains(&Q15Role::SupplierKey)
        || !projected_roles.contains(&Q15Role::TotalRevenue)
    {
        return false;
    }

    [Q15Role::SupplierName, Q15Role::SupplierAddress, Q15Role::SupplierPhone]
        .iter()
        .all(|role| !schema_has_role(left_input_schema, *role) || projected_roles.contains(role))
}

fn validate_q15_supplier_revenue_branch(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some((join_id, projection_count)) = peel_projection_chain(plan, node_id, 2) else {
        return false;
    };
    if projection_count == 0 {
        return false;
    }
    let Some(join) = join_node(plan, join_id) else {
        return false;
    };

    keyed_inner_hash_join_without_residual(join)
        && ((join_has_one_key_with_roles(join, Q15Role::SupplierKey, Q15Role::RevenueSupplierNo)
            && validate_source_filter_projection_leaf_pipeline(plan, join.inputs[0])
            && validate_q15_projection_wrapped_grouped_revenue_aggregate(plan, join.inputs[1]))
            || (join_has_one_key_with_roles(join, Q15Role::RevenueSupplierNo, Q15Role::SupplierKey)
                && validate_q15_projection_wrapped_grouped_revenue_aggregate(plan, join.inputs[0])
                && validate_source_filter_projection_leaf_pipeline(plan, join.inputs[1])))
}

fn validate_q15_scalar_max_revenue_branch(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some(scalar_aggregate) = aggregate_node(plan, node_id) else {
        return false;
    };
    if !scalar_aggregate.spec.group_exprs.is_empty() {
        return false;
    }
    let [aggregate_expr] = scalar_aggregate.spec.aggr_exprs.as_slice() else {
        return false;
    };
    aggregate_expr.function == AggregateFunction::Max
        && expr_has_any_role(&aggregate_expr.expr, &[Q15Role::TotalRevenue, Q15Role::RevenueMeasure])
        && validate_q15_projected_grouped_revenue_aggregate(plan, scalar_aggregate.input, 1, 2)
}

fn validate_q15_projection_wrapped_grouped_revenue_aggregate(plan: &QueryPlan, node_id: NodeId) -> bool {
    validate_q15_projected_grouped_revenue_aggregate(plan, node_id, 1, 1)
}

fn validate_q15_projected_grouped_revenue_aggregate(
    plan: &QueryPlan,
    node_id: NodeId,
    min_projection_count: usize,
    max_projection_count: usize,
) -> bool {
    let Some((aggregate_id, projection_count)) = peel_projection_chain(plan, node_id, max_projection_count) else {
        return false;
    };
    if projection_count < min_projection_count {
        return false;
    }
    let Some(aggregate) = aggregate_node(plan, aggregate_id) else {
        return false;
    };
    projection_chain_outputs_role(plan, node_id, max_projection_count, Q15Role::TotalRevenue)
        && validate_q15_grouped_revenue_aggregate(plan, aggregate)
        && validate_source_filter_projection_leaf_pipeline(plan, aggregate.input)
}

fn validate_q15_grouped_revenue_aggregate(plan: &QueryPlan, aggregate: &AggregateNode) -> bool {
    let [group_expr] = aggregate.spec.group_exprs.as_slice() else {
        return false;
    };
    let [aggregate_expr] = aggregate.spec.aggr_exprs.as_slice() else {
        return false;
    };
    expr_has_any_role(group_expr, &[Q15Role::RevenueSupplierNo, Q15Role::SupplierKey])
        && aggregate_expr.function == AggregateFunction::Sum
        && aggregate_argument_has_role(plan, aggregate, &aggregate_expr.expr, Q15Role::RevenueMeasure)
}

fn peel_projection_chain(plan: &QueryPlan, node_id: NodeId, max_projection_count: usize) -> Option<(NodeId, usize)> {
    let mut current = node_id;
    let mut projection_count = 0;
    while let Some(projection) = projection_node(plan, current) {
        if projection_count == max_projection_count {
            return None;
        }
        projection_count += 1;
        current = projection.input;
    }
    Some((current, projection_count))
}

fn validate_source_filter_projection_leaf_pipeline(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(_) => true,
        PlanNode::Filter(filter) => validate_source_filter_projection_leaf_pipeline(plan, filter.input),
        PlanNode::Projection(projection) => validate_source_filter_projection_leaf_pipeline(plan, projection.input),
        PlanNode::Join(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => false,
    }
}

fn keyed_inner_hash_join_without_residual(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
}

fn join_has_one_key_with_roles(join: &JoinNode, left_role: Q15Role, right_role: Q15Role) -> bool {
    let [key] = join.spec.keys.as_slice() else {
        return false;
    };
    column_has_role(&key.left, left_role) && column_has_role(&key.right, right_role)
}

fn aggregate_argument_has_role(plan: &QueryPlan, aggregate: &AggregateNode, expr: &Expr, role: Q15Role) -> bool {
    if expr_has_role(expr, role) {
        return true;
    }
    let Expr::Column(column) = expr else {
        return false;
    };
    projection_defines_column_with_role(plan, aggregate.input, column, role)
}

fn projection_defines_column_with_role(plan: &QueryPlan, node_id: NodeId, column: &ColumnRef, role: Q15Role) -> bool {
    let Some(projection) = projection_node(plan, node_id) else {
        return false;
    };
    projection.expressions.iter().any(|expr| expr_output_name_matches_column(expr, column) && expr_has_role(expr, role))
}

fn projection_chain_outputs_role(
    plan: &QueryPlan,
    node_id: NodeId,
    max_projection_count: usize,
    role: Q15Role,
) -> bool {
    let mut current = node_id;
    let mut projection_count = 0;
    while let Some(projection) = projection_node(plan, current) {
        if projection_count == max_projection_count {
            return false;
        }
        if projection.expressions.iter().any(|expr| expr_output_has_role(expr, role)) {
            return true;
        }
        projection_count += 1;
        current = projection.input;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Q15Role {
    SupplierKey,
    SupplierName,
    SupplierAddress,
    SupplierPhone,
    RevenueSupplierNo,
    RevenueMeasure,
    TotalRevenue,
    MaxRevenue,
}

fn q15_role_for_join_output_column(column: &crate::plan::JoinOutputColumn) -> Option<Q15Role> {
    q15_role_for_names([column.output_name.as_str(), column.column.display_name(), column.column.name()])
}

fn schema_has_role(schema: &SchemaRef, role: Q15Role) -> bool {
    schema.fields().iter().any(|field| name_has_role(field.name(), role))
}

fn expr_has_any_role(expr: &Expr, roles: &[Q15Role]) -> bool {
    roles.iter().any(|role| expr_has_role(expr, *role))
}

fn expr_output_name_matches_column(expr: &Expr, column: &ColumnRef) -> bool {
    expr_output_name(expr).is_some_and(|name| name == column.display_name() || name == column.name())
}

fn expr_output_has_role(expr: &Expr, role: Q15Role) -> bool {
    expr_output_name(expr).is_some_and(|name| name_has_role(name, role))
}

fn expr_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias),
        _ => None,
    }
}

fn expr_has_role(expr: &Expr, role: Q15Role) -> bool {
    match expr {
        Expr::Column(column) => column_has_role(column, role),
        Expr::Alias { expr, alias } => name_has_role(alias, role) || expr_has_role(expr, role),
        Expr::Binary { left, right, .. } => expr_has_role(left, role) || expr_has_role(right, role),
        Expr::Cast { expr, .. }
        | Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Extract { expr, .. } => expr_has_role(expr, role),
        Expr::InList { expr, list, .. } => {
            expr_has_role(expr, role) || list.iter().any(|item| expr_has_role(item, role))
        },
        Expr::Case { branches, else_expr } => {
            expr_has_role(else_expr, role)
                || branches.iter().any(|branch| expr_has_role(&branch.when, role) || expr_has_role(&branch.then, role))
        },
        Expr::Literal(_) => false,
    }
}

fn column_has_role(column: &ColumnRef, role: Q15Role) -> bool {
    q15_role_for_names([column.display_name(), column.name()]).is_some_and(|column_role| column_role == role)
}

fn name_has_role(name: &str, role: Q15Role) -> bool {
    q15_role_for_name(name).is_some_and(|name_role| name_role == role)
}

fn q15_role_for_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<Q15Role> {
    names.into_iter().find_map(q15_role_for_name)
}

fn q15_role_for_name(name: &str) -> Option<Q15Role> {
    match normalized_role_name(name).as_str() {
        "ssuppkey" | "suppkey" | "supplierkey" => Some(Q15Role::SupplierKey),
        "sname" | "suppliername" => Some(Q15Role::SupplierName),
        "saddress" | "supplieraddress" => Some(Q15Role::SupplierAddress),
        "sphone" | "supplierphone" => Some(Q15Role::SupplierPhone),
        "supplierno" | "revenuesupplierno" | "lsuppkey" => Some(Q15Role::RevenueSupplierNo),
        "revenue" | "revenueamount" | "extendedprice" | "lextendedprice" => Some(Q15Role::RevenueMeasure),
        "totalrevenue" => Some(Q15Role::TotalRevenue),
        "maxrevenue" | "maximumrevenue" | "maxrevenuetotalrevenue" => Some(Q15Role::MaxRevenue),
        _ => None,
    }
}

fn normalized_role_name(name: &str) -> String {
    name.chars().filter(|character| character.is_ascii_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn projection_node(plan: &QueryPlan, node_id: NodeId) -> Option<&ProjectionNode> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Projection(projection) => Some(projection),
        _ => None,
    }
}

fn aggregate_node(plan: &QueryPlan, node_id: NodeId) -> Option<&AggregateNode> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Aggregate(aggregate) => Some(aggregate),
        _ => None,
    }
}

fn join_node(plan: &QueryPlan, node_id: NodeId) -> Option<&JoinNode> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Join(join) => Some(join),
        _ => None,
    }
}

#[derive(Default)]
struct ReachableNodeKindCounts {
    joins: usize,
    aggregates: usize,
    sorts: usize,
    limits: usize,
}

fn reachable_node_kind_counts(plan: &QueryPlan) -> ReachableNodeKindCounts {
    let mut reachable = vec![false; plan.nodes().len()];
    let mut stack = vec![plan.sink().input()];
    let mut counts = ReachableNodeKindCounts::default();
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            continue;
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        match &node.kind {
            PlanNode::Join(_) => counts.joins += 1,
            PlanNode::Aggregate(_) => counts.aggregates += 1,
            PlanNode::Sort(_) => counts.sorts += 1,
            PlanNode::Limit(_) => counts.limits += 1,
            PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) => {},
        }
        stack.extend(node.input_ids().iter().copied());
    }
    counts
}

fn first_unsupported_keyed_residual_semi_anti_join(plan: &QueryPlan) -> Option<&JoinNode> {
    let multi_join = reachable_join_count(plan) > 1;
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        if join.spec.keys.is_empty()
            || join.spec.residual_predicate.is_none()
            || !matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        {
            return None;
        }

        if multi_join {
            (!bounded_multi_join_mixed_keyed_residual_compatible(join)).then_some(join)
        } else {
            (!bounded_root_mixed_join_compatible(join) || root_join_leaf_blocker(plan, join).is_some()).then_some(join)
        }
    })
}

fn first_keyed_residual_full_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Full)
            .then_some(join)
    })
}

fn first_keyed_residual_left_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Left)
            .then_some(join)
    })
}

fn mixed_keyed_residual_left_only_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        if join.spec.build_side != JoinSide::Right {
            "multi_join_mixed_keyed_residual_semi_anti_requires_right_build_side"
        } else if join.spec.output_projection.is_empty() {
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_projection"
        } else if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_only_projection"
        } else {
            "bounded_multi_join_mixed_keyed_residual_semi_anti_shape_unsupported"
        }
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_semi_anti_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_semi_anti_requires_left_projection"
    } else if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
        "root_mixed_keyed_residual_semi_anti_requires_left_only_projection"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_semi_anti_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_only_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        if join.spec.build_side != JoinSide::Right
            || join.spec.output_projection.is_empty()
            || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
        {
            "multi_join_mixed_keyed_residual_left_only_join_requires_right_build_left_projection".to_string()
        } else {
            "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
        }
    } else if join.spec.build_side != JoinSide::Right
        || join.spec.output_projection.is_empty()
        || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        "root_mixed_keyed_residual_left_only_join_requires_right_build_left_projection".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_full_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_full_topology_unsupported"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_full_shape_unsupported"
    }
}

fn mixed_keyed_residual_full_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_left_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_left_topology_unsupported"
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_left_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_join_requires_right_build".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn root_join_leaf_blocker(plan: &QueryPlan, join: &JoinNode) -> Option<&'static str> {
    join.inputs.iter().find_map(|input| root_join_input_leaf_blocker(plan, *input))
}

fn root_join_input_leaf_blocker(plan: &QueryPlan, node_id: NodeId) -> Option<&'static str> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(_) => None,
        PlanNode::Filter(filter) => root_join_input_leaf_blocker(plan, filter.input),
        PlanNode::Projection(projection) => root_join_input_leaf_blocker(plan, projection.input),
        PlanNode::Join(_) => Some("bounded_root_join_nested_join_input_topology_unsupported"),
        PlanNode::Aggregate(_) => Some("bounded_root_join_aggregate_leaf_unsupported"),
        PlanNode::Sort(_) => Some("bounded_root_join_sort_leaf_unsupported"),
        PlanNode::Limit(_) => Some("bounded_root_join_limit_leaf_unsupported"),
    }
}

fn reachable_join_count(plan: &QueryPlan) -> usize {
    let mut reachable = vec![false; plan.nodes().len()];
    let mut stack = vec![plan.sink().input()];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            continue;
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        if matches!(node.kind, PlanNode::Join(_)) {
            count += 1;
        }
        stack.extend(node.input_ids().iter().copied());
    }
    count
}
