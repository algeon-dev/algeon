use arrow::datatypes::SchemaRef;

use crate::{
    expr::{AggregateFunction, ColumnQualifier, ColumnRef, Expr},
    plan::{
        AggregateNode, FilterNode, JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode,
        QueryPlan, SortNode,
    },
    source::SourceNode,
};

mod compatibility;
mod facts;
mod q15;
mod shape;

pub(crate) use compatibility::{
    BoundedKeylessConditionalJoinType,
    bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_root_compatible, bounded_multi_join_compatible,
    bounded_multi_join_mixed_keyed_residual_compatible, bounded_root_filtered_join_compatible,
    bounded_root_hash_join_compatible, bounded_root_join_compatible,
    bounded_root_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_compatible,
    bounded_root_keyless_conditional_join_compatible,
    bounded_root_keyless_conditional_join_input_subplan_pipeline_compatible, bounded_root_mixed_join_compatible,
    bounded_root_projected_cross_join_compatible,
};
pub(crate) use facts::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
use shape::{
    AggregateOverJoinSubplanInputKind, analyze_bounded_join_pipeline_shape,
    bounded_join_inputs_include_resident_subplan, bounded_join_source_leaf_inputs_compatible,
    bounded_join_subtree_join_count, peel_projection_chain, projection_node, reachable_join_count,
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_post_root_pipeline,
    validate_bounded_keyless_conditional_join_input_subplan, validate_bounded_multi_join_subtree,
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_post_root_pipeline,
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_sides,
    validate_projection_wrapped_aggregate_over_join_subplan_input, validate_source_filter_projection_leaf_pipeline,
};

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
    /// A keyed residual root join whose inputs are a bounded join subtree and a
    /// projection-wrapped grouped local-Parquet aggregate, followed by a
    /// projection/global-aggregate/projection post-root pipeline.
    MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    /// The plan contains joins but does not match any bounded join family.
    Unsupported,
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

/// Normalized shape for the Q17-style family. The name is intentionally
/// topology-based: admission must not depend on query id, table names, column
/// names, or literal constants.
pub(crate) struct BoundedMixedKeyedResidualJoinResidentGroupedAggregateInputPipelineShape<'a> {
    pub(crate) join_id: NodeId,
    pub(crate) join: &'a JoinNode,
    pub(crate) stages: Vec<BoundedJoinPipelineStage<'a>>,
    pub(crate) join_subtree_side: JoinSide,
    pub(crate) grouped_aggregate_side: JoinSide,
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
        Err(_) => {
            match analyze_bounded_keyless_conditional_join_input_subplan_pipeline(plan) {
                Ok(_) => BoundedJoinPlanSupport::KeylessConditionalJoinInputSubplanPipeline,
                Err(_) => {
                    match analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(plan) {
                    Ok(_) => BoundedJoinPlanSupport::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
                    Err(_) => match analyze_bounded_multi_join_pipeline(plan) {
                        Ok(_) => BoundedJoinPlanSupport::MultiJoinTreePostRootPipeline,
                        Err(_) => match analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan) {
                            Ok(_) => BoundedJoinPlanSupport::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
                            Err(_) => BoundedJoinPlanSupport::Unsupported,
                        },
                    },
                }
                },
            }
        },
    }
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

pub(crate) fn analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<
    BoundedMixedKeyedResidualJoinResidentGroupedAggregateInputPipelineShape<'_>,
    BoundedJoinPipelineRejection<'_>,
> {
    let shape = analyze_bounded_join_pipeline_shape(plan, false)?;
    if !bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_root_compatible(shape.join) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: shape.join });
    }
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_post_root_pipeline(&shape)?;
    if validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_sides(
        plan,
        shape.join,
        JoinSide::Left,
        JoinSide::Right,
    )
    .is_ok()
    {
        return Ok(BoundedMixedKeyedResidualJoinResidentGroupedAggregateInputPipelineShape {
            join_id: shape.join_id,
            join: shape.join,
            stages: shape.stages,
            join_subtree_side: JoinSide::Left,
            grouped_aggregate_side: JoinSide::Right,
        });
    }
    validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_sides(
        plan,
        shape.join,
        JoinSide::Right,
        JoinSide::Left,
    )?;
    Ok(BoundedMixedKeyedResidualJoinResidentGroupedAggregateInputPipelineShape {
        join_id: shape.join_id,
        join: shape.join,
        stages: shape.stages,
        join_subtree_side: JoinSide::Right,
        grouped_aggregate_side: JoinSide::Left,
    })
}
