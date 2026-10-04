use super::*;

pub(super) fn compatible_join_indices_domain(
    left_keys: &TableView<'_>,
    right_keys: &TableView<'_>,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    ResolvedExecution::binary(left_keys.execution_domain(), right_keys.execution_domain(), ctx, operation)
}

pub(super) fn compatible_conditional_join_domain(
    left: &TableView<'_>,
    right: &TableView<'_>,
    predicate: &AstExpression,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(right.execution_domain());
    predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(left.execution_domain(), secondaries, ctx, operation)
}

pub(super) fn compatible_mixed_join_domain(
    tables: MixedJoinTables<'_, '_>,
    predicate: &AstExpression,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(tables.right_equality.execution_domain());
    secondaries.push(tables.left_conditional.execution_domain());
    secondaries.push(tables.right_conditional.execution_domain());
    predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(tables.left_equality.execution_domain(), secondaries, ctx, operation)
}

pub(super) fn compatible_filter_join_indices_domain(
    inputs: FilterJoinInputs<'_, '_>,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let mut secondaries = Vec::new();
    secondaries.push(inputs.right_indices.execution_domain());
    secondaries.push(inputs.left.execution_domain());
    secondaries.push(inputs.right.execution_domain());
    inputs.predicate.literal_domains(&mut secondaries);
    ResolvedExecution::many(inputs.left_indices.execution_domain(), secondaries, ctx, operation)
}
