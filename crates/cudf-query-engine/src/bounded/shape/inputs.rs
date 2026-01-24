use super::*;

pub(in crate::bounded) fn bounded_join_source_leaf_inputs_compatible(plan: &QueryPlan, join: &JoinNode) -> bool {
    validate_bounded_join_source_leaf_inputs(plan, join).is_ok()
}

pub(in crate::bounded) fn validate_bounded_join_source_leaf_inputs<'a>(
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

pub(in crate::bounded) fn validate_bounded_keyless_conditional_join_input_subplan(
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
pub(in crate::bounded) enum AggregateOverJoinSubplanInputKind {
    /// The input must be a grouped aggregate over a bounded join subplan.
    Grouped,
    /// The input must be a scalar/global aggregate over a bounded join subplan.
    Scalar,
}

pub(in crate::bounded) fn validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_post_root_pipeline<
    'a,
>(
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

pub(in crate::bounded) fn validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_post_root_pipeline<'a>(
    shape: &BoundedJoinPipelineShape<'a>,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    let [
        BoundedJoinPipelineStage::Projection { .. },
        BoundedJoinPipelineStage::Aggregate { node_id: aggregate_id, node: aggregate, .. },
        BoundedJoinPipelineStage::Projection { .. },
    ] = shape.stages.as_slice()
    else {
        let node_id = shape.stages.first().map(pipeline_stage_node_id).unwrap_or(shape.join_id);
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "post_root_pipeline" });
    };

    if !aggregate.spec.group_exprs.is_empty() {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: *aggregate_id,
            node_kind: "grouped_post_root_aggregate",
        });
    }
    Ok(())
}

pub(in crate::bounded) fn validate_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline<'a>(
    shape: &BoundedJoinPipelineShape<'a>,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    let [
        BoundedJoinPipelineStage::Projection { .. },
        BoundedJoinPipelineStage::Aggregate { node_id: aggregate_id, node: aggregate, .. },
        BoundedJoinPipelineStage::Projection { .. },
        BoundedJoinPipelineStage::Sort { .. },
        BoundedJoinPipelineStage::Limit { .. },
    ] = shape.stages.as_slice()
    else {
        let node_id = shape.stages.first().map(pipeline_stage_node_id).unwrap_or(shape.join_id);
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id,
            node_kind: "post_root_topk_pipeline",
        });
    };

    if aggregate.spec.group_exprs.is_empty() {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: *aggregate_id,
            node_kind: "scalar_post_root_aggregate",
        });
    }
    Ok(())
}

pub(in crate::bounded) fn validate_mixed_keyed_residual_join_resident_grouped_aggregate_input_sides<'a>(
    plan: &'a QueryPlan,
    root_join: &'a JoinNode,
    join_subtree_side: JoinSide,
    grouped_aggregate_side: JoinSide,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    validate_projection_wrapped_join_subtree_input_structural(
        plan,
        root_join.inputs[join_side_index(join_subtree_side)],
    )?;
    validate_projection_wrapped_grouped_local_parquet_aggregate_input_structural(
        plan,
        root_join.inputs[join_side_index(grouped_aggregate_side)],
    )?;
    if root_join.spec.output_projection.iter().any(|column| column.side != join_subtree_side) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    let Some(predicate) = root_join.spec.residual_predicate.as_ref() else {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    };
    if !expr_references_input(predicate, join_side_index(join_subtree_side))
        || !expr_references_input(predicate, join_side_index(grouped_aggregate_side))
    {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    Ok(())
}

pub(in crate::bounded) fn validate_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_sides<'a>(
    plan: &'a QueryPlan,
    root_join: &'a JoinNode,
    join_subtree_side: JoinSide,
    grouped_aggregate_side: JoinSide,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'a>> {
    if join_subtree_side != JoinSide::Left || grouped_aggregate_side != JoinSide::Right {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    validate_bounded_multi_join_subtree(plan, root_join.inputs[join_side_index(join_subtree_side)])?;
    if bounded_join_subtree_join_count(plan, root_join.inputs[join_side_index(join_subtree_side)])? < 2 {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: root_join.inputs[join_side_index(join_subtree_side)],
            node_kind: "single_join_subtree",
        });
    }
    validate_projection_wrapped_filtered_grouped_local_parquet_aggregate_input_structural(
        plan,
        root_join.inputs[join_side_index(grouped_aggregate_side)],
    )?;
    if root_join.spec.output_projection.iter().any(|column| column.side != join_subtree_side) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    Ok(())
}

fn validate_projection_wrapped_join_subtree_input_structural(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let Some((join_id, projection_count)) = peel_projection_chain(plan, node_id, 3) else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "projection_chain" });
    };
    if projection_count == 0 {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "bare_join_subtree" });
    }
    validate_bounded_multi_join_subtree(plan, join_id)?;
    if bounded_join_subtree_join_count(plan, join_id)? == 0 {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: join_id,
            node_kind: "non_join_subtree",
        });
    }
    Ok(())
}

fn validate_projection_wrapped_grouped_local_parquet_aggregate_input_structural(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded mixed keyed residual join grouped aggregate input pipeline references a missing input",
    })?;
    let PlanNode::Projection(projection) = &node.kind else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id,
            node_kind: "bare_grouped_aggregate_input",
        });
    };
    let aggregate_node = plan.node(projection.input).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id: projection.input,
        message: "bounded mixed keyed residual join grouped aggregate input pipeline projection references a missing aggregate",
    })?;
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: projection.input,
            node_kind: "non_aggregate_input",
        });
    };
    if aggregate.spec.group_exprs.is_empty() {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: projection.input,
            node_kind: "scalar_aggregate",
        });
    }
    validate_source_filter_projection_leaf_pipeline_rooted_in_local_parquet(plan, aggregate.input)
}

fn validate_projection_wrapped_filtered_grouped_local_parquet_aggregate_input_structural(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let mut current = node_id;
    let mut projection_count = 0;
    loop {
        let node = plan.node(current).ok_or(BoundedJoinPipelineRejection::MissingNode {
            node_id: current,
            message: "bounded hash join resident grouped aggregate input references a missing input",
        })?;
        match &node.kind {
            PlanNode::Projection(projection) => {
                projection_count += 1;
                current = projection.input;
            },
            PlanNode::Filter(filter) => {
                current = filter.input;
            },
            PlanNode::Aggregate(aggregate) => {
                if projection_count == 0 {
                    return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                        node_id,
                        node_kind: "bare_grouped_aggregate_input",
                    });
                }
                if aggregate.spec.group_exprs.is_empty() {
                    return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                        node_id: current,
                        node_kind: "scalar_aggregate",
                    });
                }
                return validate_source_filter_projection_leaf_pipeline_rooted_in_local_parquet(plan, aggregate.input);
            },
            PlanNode::Source(_) => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id: current,
                    node_kind: "source",
                });
            },
            PlanNode::Join(_) => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id: current, node_kind: "join" });
            },
            PlanNode::Sort(_) => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id: current, node_kind: "sort" });
            },
            PlanNode::Limit(_) => {
                return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id: current,
                    node_kind: "limit",
                });
            },
        }
    }
}

pub(in crate::bounded) fn validate_projection_wrapped_aggregate_over_join_subplan_input(
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

pub(in crate::bounded) fn bounded_join_inputs_include_resident_subplan<'a>(
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
fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn column_qualifier_input_index(column: &ColumnRef) -> Option<usize> {
    match column.qualifier() {
        Some(ColumnQualifier::Input { index }) => Some(*index),
        Some(ColumnQualifier::Node { .. } | ColumnQualifier::Relation { .. }) | None => None,
    }
}

fn expr_references_input(expr: &Expr, input_index: usize) -> bool {
    match expr {
        Expr::Column(column) => column_qualifier_input_index(column) == Some(input_index),
        Expr::Alias { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Extract { expr, .. } => expr_references_input(expr, input_index),
        Expr::Binary { left, right, .. } => {
            expr_references_input(left, input_index) || expr_references_input(right, input_index)
        },
        Expr::InList { expr, list, .. } => {
            expr_references_input(expr, input_index) || list.iter().any(|item| expr_references_input(item, input_index))
        },
        Expr::Case { branches, else_expr } => {
            expr_references_input(else_expr, input_index)
                || branches.iter().any(|branch| {
                    expr_references_input(&branch.when, input_index) || expr_references_input(&branch.then, input_index)
                })
        },
        Expr::Literal(_) => false,
    }
}

fn pipeline_stage_node_id(stage: &BoundedJoinPipelineStage<'_>) -> NodeId {
    match stage {
        BoundedJoinPipelineStage::Filter { node_id, .. }
        | BoundedJoinPipelineStage::Projection { node_id, .. }
        | BoundedJoinPipelineStage::Aggregate { node_id, .. }
        | BoundedJoinPipelineStage::Sort { node_id, .. }
        | BoundedJoinPipelineStage::Limit { node_id, .. } => *node_id,
    }
}
pub(in crate::bounded) fn validate_source_filter_projection_leaf_pipeline(plan: &QueryPlan, node_id: NodeId) -> bool {
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

fn validate_source_filter_projection_leaf_pipeline_rooted_in_local_parquet(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<(), BoundedJoinPipelineRejection<'_>> {
    let node = plan.node(node_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id,
        message: "bounded mixed keyed residual join grouped aggregate input pipeline references a missing aggregate source",
    })?;
    match &node.kind {
        PlanNode::Source(source) => {
            if matches!(source.as_ref(), SourceNode::LocalParquet(_)) {
                Ok(())
            } else {
                Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                    node_id,
                    node_kind: "non_local_parquet_source",
                })
            }
        },
        PlanNode::Filter(filter) => {
            validate_source_filter_projection_leaf_pipeline_rooted_in_local_parquet(plan, filter.input)
        },
        PlanNode::Projection(projection) => {
            validate_source_filter_projection_leaf_pipeline_rooted_in_local_parquet(plan, projection.input)
        },
        PlanNode::Join(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "join" }),
        PlanNode::Aggregate(_) => {
            Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" })
        },
        PlanNode::Sort(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "sort" }),
        PlanNode::Limit(_) => Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "limit" }),
    }
}
