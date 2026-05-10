use super::*;

pub(super) fn matches_keyed_hash_join_resident_subplan_pipeline(plan: &QueryPlan) -> bool {
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
        && (root_join_matches_q15_revenue_max_join(plan, root_join, JoinSide::Left, JoinSide::Right)
            || root_join_matches_q15_revenue_max_join(plan, root_join, JoinSide::Right, JoinSide::Left))
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

fn root_join_matches_q15_revenue_max_join(
    plan: &QueryPlan,
    join: &JoinNode,
    supplier_revenue_side: JoinSide,
    scalar_max_side: JoinSide,
) -> bool {
    if !join_has_one_key_with_roles_by_side(join, supplier_revenue_side, Q15Role::TotalRevenue, Q15Role::MaxRevenue) {
        return false;
    }
    root_join_outputs_expected_q15_supplier_roles(plan, join, supplier_revenue_side)
        && validate_q15_supplier_revenue_branch(plan, join.inputs[join_side_index(supplier_revenue_side)])
        && validate_q15_scalar_max_revenue_branch(plan, join.inputs[join_side_index(scalar_max_side)])
}

fn root_join_outputs_expected_q15_supplier_roles(
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
        join.spec.output_projection.iter().filter_map(q15_role_for_join_output_column).collect::<Vec<_>>();
    if projected_roles.len() != join.spec.output_projection.len()
        || !projected_roles.contains(&Q15Role::SupplierKey)
        || !projected_roles.contains(&Q15Role::TotalRevenue)
    {
        return false;
    }

    [Q15Role::SupplierName, Q15Role::SupplierAddress, Q15Role::SupplierPhone]
        .iter()
        .all(|role| !schema_has_role(supplier_input_schema, *role) || projected_roles.contains(role))
}

fn validate_q15_supplier_revenue_branch(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some((join_id, _projection_count)) = peel_projection_chain(plan, node_id, 2) else {
        return false;
    };
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

fn join_has_one_key_with_roles_by_side(
    join: &JoinNode,
    supplier_revenue_side: JoinSide,
    supplier_revenue_role: Q15Role,
    scalar_max_role: Q15Role,
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
    projection.expressions().any(|expr| expr_output_name_matches_column(expr, column) && expr_has_role(expr, role))
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
        if projection.expressions().any(|expr| expr_output_has_role(expr, role)) {
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
    match normalized_role_name_without_disambiguator(name).as_str() {
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
