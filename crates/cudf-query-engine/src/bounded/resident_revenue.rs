use super::*;

// Phase A keeps this as a legacy transitional recognizer for the existing
// resident-subplan family. The role-name checks below must not be copied into
// new families; Phase B.5 decomposes this topology into dataflow contracts.
pub(crate) fn analyze_keyed_hash_join_resident_subplan_pipeline(
    plan: &QueryPlan,
) -> std::result::Result<BoundedKeyedHashJoinResidentSubplanPipelineShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let (final_sort_id, final_sort, root_join_id) = final_sort_root_join_input(plan)?;
    let counts = reachable_node_kind_counts(plan);
    if counts.joins != 2
        || (counts.aggregates != 2 && counts.aggregates != 3)
        || counts.sorts != 1
        || counts.limits != 0
    {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: plan.sink().input(),
            node_kind: "keyed_hash_join_resident_subplan_pipeline_reachable_counts",
        });
    }

    let root_join = join_node(plan, root_join_id).ok_or(BoundedJoinPipelineRejection::MissingRootJoin)?;
    if !keyed_inner_hash_join_without_residual(root_join) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    if !final_sort_is_ascending_supplier_key(final_sort) {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id: final_sort_id, node_kind: "sort" });
    }

    analyze_keyed_hash_join_resident_subplan_pipeline_sides(
        plan,
        final_sort_id,
        final_sort,
        root_join_id,
        root_join,
        JoinSide::Left,
        JoinSide::Right,
    )
    .or_else(|_| {
        analyze_keyed_hash_join_resident_subplan_pipeline_sides(
            plan,
            final_sort_id,
            final_sort,
            root_join_id,
            root_join,
            JoinSide::Right,
            JoinSide::Left,
        )
    })
}

fn analyze_keyed_hash_join_resident_subplan_pipeline_sides<'a>(
    plan: &'a QueryPlan,
    final_sort_id: NodeId,
    final_sort: &'a SortNode,
    root_join_id: NodeId,
    root_join: &'a JoinNode,
    supplier_revenue_side: JoinSide,
    scalar_max_side: JoinSide,
) -> std::result::Result<BoundedKeyedHashJoinResidentSubplanPipelineShape<'a>, BoundedJoinPipelineRejection<'a>> {
    if !join_has_one_key_with_roles_by_side(
        root_join,
        supplier_revenue_side,
        ResidentRevenueRole::TotalRevenue,
        ResidentRevenueRole::MaxRevenue,
    ) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }
    if !root_join_outputs_expected_resident_revenue_supplier_roles(plan, root_join, supplier_revenue_side) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join: root_join });
    }

    let supplier_revenue = analyze_resident_revenue_supplier_revenue_branch(
        plan,
        root_join.inputs[join_side_index(supplier_revenue_side)],
    )?;
    let scalar_max =
        analyze_resident_revenue_scalar_max_revenue_branch(plan, root_join.inputs[join_side_index(scalar_max_side)])?;
    Ok(BoundedKeyedHashJoinResidentSubplanPipelineShape {
        final_sort_id,
        final_sort,
        root_join_id,
        root_join,
        supplier_revenue_side,
        supplier_revenue,
        scalar_max,
    })
}

fn final_sort_root_join_input(
    plan: &QueryPlan,
) -> std::result::Result<(NodeId, &SortNode, NodeId), BoundedJoinPipelineRejection<'_>> {
    let final_sort_id = plan.sink().input();
    let node = plan.node(final_sort_id).ok_or(BoundedJoinPipelineRejection::MissingNode {
        node_id: final_sort_id,
        message: "keyed hash join resident subplan pipeline references a missing final sort node",
    })?;
    let PlanNode::Sort(sort) = &node.kind else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: final_sort_id,
            node_kind: bounded_resident_revenue_node_kind(&node.kind),
        });
    };
    Ok((final_sort_id, sort, sort.input))
}

fn final_sort_is_ascending_supplier_key(sort: &SortNode) -> bool {
    let [sort_expr] = sort.spec.expressions.as_slice() else {
        return false;
    };
    !sort_expr.descending && expr_has_role(&sort_expr.expr, ResidentRevenueRole::SupplierKey)
}

fn root_join_outputs_expected_resident_revenue_supplier_roles(
    plan: &QueryPlan,
    join: &JoinNode,
    supplier_revenue_side: JoinSide,
) -> bool {
    if join.spec.output_projection.is_empty()
        || join.spec.output_projection.iter().any(|column| column.side != supplier_revenue_side)
    {
        return false;
    }

    let Some(supplier_input_schema) =
        plan.node(join.inputs[join_side_index(supplier_revenue_side)]).map(|node| node.output_schema())
    else {
        return false;
    };
    let projected_roles =
        join.spec.output_projection.iter().filter_map(resident_revenue_role_for_join_output_column).collect::<Vec<_>>();
    if projected_roles.len() != join.spec.output_projection.len()
        || !projected_roles.contains(&ResidentRevenueRole::SupplierKey)
        || !projected_roles.contains(&ResidentRevenueRole::TotalRevenue)
    {
        return false;
    }

    [ResidentRevenueRole::SupplierName, ResidentRevenueRole::SupplierAddress, ResidentRevenueRole::SupplierPhone]
        .iter()
        .all(|role| !schema_has_role(supplier_input_schema, *role) || projected_roles.contains(role))
}

fn analyze_resident_revenue_supplier_revenue_branch(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<BoundedKeyedHashJoinResidentSupplierRevenueBranchShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let (join_id, projection_stages) = peel_resident_revenue_projection_stages(plan, node_id, 2)?;
    let Some(join) = join_node(plan, join_id) else {
        let node_kind =
            plan.node(join_id).map(|node| bounded_resident_revenue_node_kind(&node.kind)).unwrap_or("missing");
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id: join_id, node_kind });
    };

    if !keyed_inner_hash_join_without_residual(join) {
        return Err(BoundedJoinPipelineRejection::UnsupportedJoin { join });
    }

    if join_has_one_key_with_roles(join, ResidentRevenueRole::SupplierKey, ResidentRevenueRole::RevenueSupplierNo)
        && validate_source_filter_projection_leaf_pipeline(plan, join.inputs[0])
        && validate_resident_revenue_projection_wrapped_grouped_revenue_aggregate(plan, join.inputs[1])
    {
        return Ok(BoundedKeyedHashJoinResidentSupplierRevenueBranchShape {
            join_id,
            join,
            supplier_source_side: JoinSide::Left,
            grouped_revenue_side: JoinSide::Right,
            projection_stages,
        });
    }
    if join_has_one_key_with_roles(join, ResidentRevenueRole::RevenueSupplierNo, ResidentRevenueRole::SupplierKey)
        && validate_resident_revenue_projection_wrapped_grouped_revenue_aggregate(plan, join.inputs[0])
        && validate_source_filter_projection_leaf_pipeline(plan, join.inputs[1])
    {
        return Ok(BoundedKeyedHashJoinResidentSupplierRevenueBranchShape {
            join_id,
            join,
            supplier_source_side: JoinSide::Right,
            grouped_revenue_side: JoinSide::Left,
            projection_stages,
        });
    }

    Err(BoundedJoinPipelineRejection::UnsupportedJoin { join })
}

fn analyze_resident_revenue_scalar_max_revenue_branch(
    plan: &QueryPlan,
    node_id: NodeId,
) -> std::result::Result<BoundedKeyedHashJoinResidentScalarMaxBranchShape<'_>, BoundedJoinPipelineRejection<'_>> {
    let Some(scalar_aggregate) = aggregate_node(plan, node_id) else {
        let node_kind =
            plan.node(node_id).map(|node| bounded_resident_revenue_node_kind(&node.kind)).unwrap_or("missing");
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind });
    };
    if !scalar_aggregate.spec.group_exprs.is_empty() {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "grouped_aggregate" });
    }
    let [aggregate_expr] = scalar_aggregate.spec.aggr_exprs.as_slice() else {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" });
    };
    let (grouped_aggregate_id, projection_stages) =
        peel_resident_revenue_projection_stages(plan, scalar_aggregate.input, 2)?;
    if projection_stages.is_empty() {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
            node_id: scalar_aggregate.input,
            node_kind: "aggregate",
        });
    }
    if aggregate_expr.function != AggregateFunction::Max
        || !expr_has_any_role(
            &aggregate_expr.expr,
            &[ResidentRevenueRole::TotalRevenue, ResidentRevenueRole::RevenueMeasure],
        )
        || aggregate_node(plan, grouped_aggregate_id)
            .is_none_or(|aggregate| !validate_resident_revenue_grouped_revenue_aggregate(plan, aggregate))
        || !aggregate_node(plan, grouped_aggregate_id)
            .is_some_and(|aggregate| validate_source_filter_projection_leaf_pipeline(plan, aggregate.input))
    {
        return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind: "aggregate" });
    }
    let grouped_revenue_root_id = projection_stage_node_id(&projection_stages[0]);
    let projection_stages_after_grouped_revenue = projection_stages.into_iter().skip(1).collect::<Vec<_>>();
    let output_schema = plan
        .node(node_id)
        .ok_or(BoundedJoinPipelineRejection::MissingNode {
            node_id,
            message: "keyed hash join resident subplan pipeline references a missing scalar aggregate node",
        })?
        .output_schema();
    Ok(BoundedKeyedHashJoinResidentScalarMaxBranchShape {
        grouped_revenue_root_id,
        projection_stages_after_grouped_revenue,
        aggregate_id: node_id,
        aggregate: scalar_aggregate,
        output_schema,
    })
}

fn peel_resident_revenue_projection_stages(
    plan: &QueryPlan,
    node_id: NodeId,
    max_projection_count: usize,
) -> std::result::Result<(NodeId, Vec<BoundedJoinPipelineStage<'_>>), BoundedJoinPipelineRejection<'_>> {
    let mut current = node_id;
    let mut stages = Vec::new();
    while let Some(node) = plan.node(current) {
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        if stages.len() == max_projection_count {
            return Err(BoundedJoinPipelineRejection::UnsupportedShapeNode {
                node_id: current,
                node_kind: "projection",
            });
        }
        stages.push(BoundedJoinPipelineStage::Projection {
            node_id: current,
            node: projection,
            output_schema: &node.facts.output_schema,
        });
        current = projection.input;
    }
    if plan.node(current).is_none() {
        return Err(BoundedJoinPipelineRejection::MissingNode {
            node_id: current,
            message: "keyed hash join resident subplan pipeline references a missing projected branch input",
        });
    }
    stages.reverse();
    Ok((current, stages))
}

fn bounded_resident_revenue_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}

fn projection_stage_node_id(stage: &BoundedJoinPipelineStage<'_>) -> NodeId {
    let BoundedJoinPipelineStage::Projection { node_id, .. } = stage else {
        unreachable!("resident revenue projection chain contains only projection stages")
    };
    *node_id
}

fn validate_resident_revenue_projection_wrapped_grouped_revenue_aggregate(plan: &QueryPlan, node_id: NodeId) -> bool {
    validate_resident_revenue_projected_grouped_revenue_aggregate(plan, node_id, 1, 1)
}

fn validate_resident_revenue_projected_grouped_revenue_aggregate(
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
    projection_chain_outputs_role(plan, node_id, max_projection_count, ResidentRevenueRole::TotalRevenue)
        && validate_resident_revenue_grouped_revenue_aggregate(plan, aggregate)
        && validate_source_filter_projection_leaf_pipeline(plan, aggregate.input)
}

fn validate_resident_revenue_grouped_revenue_aggregate(plan: &QueryPlan, aggregate: &AggregateNode) -> bool {
    let [group_expr] = aggregate.spec.group_exprs.as_slice() else {
        return false;
    };
    let [aggregate_expr] = aggregate.spec.aggr_exprs.as_slice() else {
        return false;
    };
    expr_has_any_role(group_expr, &[ResidentRevenueRole::RevenueSupplierNo, ResidentRevenueRole::SupplierKey])
        && aggregate_expr.function == AggregateFunction::Sum
        && aggregate_argument_has_role(plan, aggregate, &aggregate_expr.expr, ResidentRevenueRole::RevenueMeasure)
}

fn keyed_inner_hash_join_without_residual(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
}

fn join_has_one_key_with_roles(
    join: &JoinNode,
    left_role: ResidentRevenueRole,
    right_role: ResidentRevenueRole,
) -> bool {
    let [key] = join.spec.keys.as_slice() else {
        return false;
    };
    column_has_role(&key.left, left_role) && column_has_role(&key.right, right_role)
}

fn join_has_one_key_with_roles_by_side(
    join: &JoinNode,
    supplier_revenue_side: JoinSide,
    supplier_revenue_role: ResidentRevenueRole,
    scalar_max_role: ResidentRevenueRole,
) -> bool {
    match supplier_revenue_side {
        JoinSide::Left => join_has_one_key_with_roles(join, supplier_revenue_role, scalar_max_role),
        JoinSide::Right => join_has_one_key_with_roles(join, scalar_max_role, supplier_revenue_role),
    }
}

fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn aggregate_argument_has_role(
    plan: &QueryPlan,
    aggregate: &AggregateNode,
    expr: &Expr,
    role: ResidentRevenueRole,
) -> bool {
    if expr_has_role(expr, role) {
        return true;
    }
    let Expr::Column(column) = expr else {
        return false;
    };
    projection_defines_column_with_role(plan, aggregate.input, column, role)
}

fn projection_defines_column_with_role(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &ColumnRef,
    role: ResidentRevenueRole,
) -> bool {
    let Some(projection) = projection_node(plan, node_id) else {
        return false;
    };
    projection.expressions().any(|expr| expr_output_name_matches_column(expr, column) && expr_has_role(expr, role))
}

fn projection_chain_outputs_role(
    plan: &QueryPlan,
    node_id: NodeId,
    max_projection_count: usize,
    role: ResidentRevenueRole,
) -> bool {
    let mut current = node_id;
    let mut projection_count = 0;
    while let Some(projection) = projection_node(plan, current) {
        if projection_count == max_projection_count {
            return false;
        }
        if projection.expressions().any(|expr| expr_output_has_role(expr, role)) {
            return true;
        }
        projection_count += 1;
        current = projection.input;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentRevenueRole {
    SupplierKey,
    SupplierName,
    SupplierAddress,
    SupplierPhone,
    RevenueSupplierNo,
    RevenueMeasure,
    TotalRevenue,
    MaxRevenue,
}

fn resident_revenue_role_for_join_output_column(column: &crate::plan::JoinOutputColumn) -> Option<ResidentRevenueRole> {
    resident_revenue_role_for_names([column.output_name.as_str(), column.column.display_name(), column.column.name()])
}

fn schema_has_role(schema: &SchemaRef, role: ResidentRevenueRole) -> bool {
    schema.fields().iter().any(|field| name_has_role(field.name(), role))
}

fn expr_has_any_role(expr: &Expr, roles: &[ResidentRevenueRole]) -> bool {
    roles.iter().any(|role| expr_has_role(expr, *role))
}

fn expr_output_name_matches_column(expr: &Expr, column: &ColumnRef) -> bool {
    expr_output_name(expr).is_some_and(|name| name == column.display_name() || name == column.name())
}

fn expr_output_has_role(expr: &Expr, role: ResidentRevenueRole) -> bool {
    expr_output_name(expr).is_some_and(|name| name_has_role(name, role))
}

fn expr_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias),
        _ => None,
    }
}

fn expr_has_role(expr: &Expr, role: ResidentRevenueRole) -> bool {
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

fn column_has_role(column: &ColumnRef, role: ResidentRevenueRole) -> bool {
    resident_revenue_role_for_names([column.display_name(), column.name()])
        .is_some_and(|column_role| column_role == role)
}

fn name_has_role(name: &str, role: ResidentRevenueRole) -> bool {
    resident_revenue_role_for_name(name).is_some_and(|name_role| name_role == role)
}

fn resident_revenue_role_for_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<ResidentRevenueRole> {
    names.into_iter().find_map(resident_revenue_role_for_name)
}

fn resident_revenue_role_for_name(name: &str) -> Option<ResidentRevenueRole> {
    match normalized_role_name_without_disambiguator(name).as_str() {
        "ssuppkey" | "suppkey" | "supplierkey" => Some(ResidentRevenueRole::SupplierKey),
        "sname" | "suppliername" => Some(ResidentRevenueRole::SupplierName),
        "saddress" | "supplieraddress" => Some(ResidentRevenueRole::SupplierAddress),
        "sphone" | "supplierphone" => Some(ResidentRevenueRole::SupplierPhone),
        "supplierno" | "revenuesupplierno" | "lsuppkey" => Some(ResidentRevenueRole::RevenueSupplierNo),
        "revenue" | "revenueamount" | "extendedprice" | "lextendedprice" => Some(ResidentRevenueRole::RevenueMeasure),
        "totalrevenue" => Some(ResidentRevenueRole::TotalRevenue),
        "maxrevenue" | "maximumrevenue" | "maxrevenuetotalrevenue" => Some(ResidentRevenueRole::MaxRevenue),
        _ => None,
    }
}

fn normalized_role_name_without_disambiguator(name: &str) -> String {
    let normalized = normalized_role_name(name);
    for role_name in [
        "ssuppkey",
        "suppkey",
        "supplierkey",
        "sname",
        "suppliername",
        "saddress",
        "supplieraddress",
        "sphone",
        "supplierphone",
        "supplierno",
        "revenuesupplierno",
        "lsuppkey",
        "revenue",
        "revenueamount",
        "extendedprice",
        "lextendedprice",
        "totalrevenue",
        "maxrevenue",
        "maximumrevenue",
        "maxrevenuetotalrevenue",
    ] {
        if normalized
            .strip_prefix(role_name)
            .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|character| character.is_ascii_digit()))
        {
            return role_name.to_string();
        }
    }
    normalized
}

fn normalized_role_name(name: &str) -> String {
    name.chars().filter(|character| character.is_ascii_alphanumeric()).flat_map(char::to_lowercase).collect()
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
