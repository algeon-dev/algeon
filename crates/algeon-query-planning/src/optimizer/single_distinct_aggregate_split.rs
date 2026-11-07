use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::{Field, Schema, SchemaRef};

use self::SingleDistinctAggregateSplitRejectedReason as Reason;
use super::{NativeOptimizerChange, rewrite};
use crate::{
    error::Result,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, alias_wrapped_column},
    naming::unique_alias_with_suffix,
    plan::{
        AggregateNode, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan,
        RewrittenNodes,
    },
    prepared::{
        AVG_COUNT_STATE_TYPE, aggregate_expr_is_exact_nunique, aggregate_spec_has_distinct,
        avg_declared_sum_state_type, avg_final_expr, avg_merge_aggregates, avg_partial_aggregates, merged_count_field,
        zero_filled_count_expr,
    },
};

/// Stable rule name used in optimizer diagnostics.
pub const SINGLE_DISTINCT_AGGREGATE_SPLIT_RULE_NAME: &str = "single_distinct_aggregate_split";

/// Configuration for splitting an aggregate whose exact distinct counts share
/// one argument into two grouped aggregates that consume chunked input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleDistinctAggregateSplitOptions {
    /// Whether the pass may split eligible aggregates.
    pub enabled: bool,
}

impl SingleDistinctAggregateSplitOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for SingleDistinctAggregateSplitOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for single-distinct aggregate splits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SingleDistinctAggregateSplitDiagnostics {
    /// Number of aggregates with an exact distinct count considered.
    pub considered_count: usize,
    /// Number of aggregates split into two grouped aggregates.
    pub applied_count: usize,
    /// Per-candidate evidence in plan node order.
    pub candidates: Vec<SingleDistinctAggregateSplitCandidateDiagnostics>,
}

impl SingleDistinctAggregateSplitDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Evidence for one aggregate with an exact distinct count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleDistinctAggregateSplitCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Aggregate node considered before rewrite.
    pub aggregate_node_id: NodeId,
    /// Input column every exact distinct count reads, when they share one.
    pub distinct_column: Option<String>,
    /// True when the aggregate was split.
    pub applied: bool,
    /// Rejection reason when the aggregate was kept.
    pub rejected_reason: Option<SingleDistinctAggregateSplitRejectedReason>,
}

/// Reasons an aggregate with an exact distinct count was kept whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleDistinctAggregateSplitRejectedReason {
    /// An aggregate argument is not an input column.
    NonColumnAggregateArgument,
    /// An exact distinct count has a `FILTER` clause.
    FilteredDistinctAggregate,
    /// Exact distinct counts read different input columns.
    MultipleDistinctArguments,
    /// The aggregate has grouping-set masks.
    GroupingSets,
    /// The distinct argument is also a group key.
    DistinctArgumentIsGroupKey,
    /// Another aggregate has no partial state with a merge: MEDIAN, a
    /// statistical aggregate, a DISTINCT aggregate other than a count, or an
    /// AVG whose argument and output types have no native state that
    /// reproduces DataFusion.
    NonDecomposableAggregate,
    /// The input or declared output schema does not resolve the aggregate's
    /// columns.
    UnresolvedSchema,
}

pub(crate) struct SingleDistinctAggregateSplitResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: NativeOptimizerChange,
    pub(crate) diagnostics: SingleDistinctAggregateSplitDiagnostics,
}

/// Rewrites `Aggregate(keys; NUnique(x), SUM/MIN/MAX/COUNT/AVG(y)...)` into a
/// partial aggregate grouped by `keys + x` and a merge aggregate grouped by
/// `keys`, where each exact distinct count becomes `COUNT(x)` over the partial
/// groups. AVG carries its SUM and COUNT state through both aggregates, and a
/// projection above the merge computes its value.
///
/// An exact distinct count needs its whole input in one aggregate, while both
/// replacement aggregates merge chunk-local partial state. The partial
/// aggregate has one row per `(keys, x)` group.
pub(crate) fn optimize_single_distinct_aggregate_split(
    plan: QueryPlan,
    options: &SingleDistinctAggregateSplitOptions,
) -> Result<SingleDistinctAggregateSplitResult> {
    let has_candidate = plan.nodes().iter().any(|node| distinct_aggregate(&node.kind).is_some());
    if !options.enabled || !has_candidate {
        return Ok(SingleDistinctAggregateSplitResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: SingleDistinctAggregateSplitDiagnostics::default(),
        });
    }

    let mut diagnostics = SingleDistinctAggregateSplitDiagnostics::default();
    let mut nodes = Vec::<QueryNode>::with_capacity(plan.node_count() + 2);
    let mut images = BTreeMap::new();
    for node in plan.nodes() {
        let kind = rewrite::clone_plan_node_with_inputs(&node.kind, |input| images[&input]);
        let split = distinct_aggregate(&kind).and_then(|aggregate| {
            let input_schema = nodes[aggregate.input.index()].output_schema();
            let (distinct_column, split) = split_aggregate(aggregate, input_schema, node.output_schema());
            diagnostics.considered_count += 1;
            diagnostics.applied_count += usize::from(split.is_ok());
            diagnostics.candidates.push(SingleDistinctAggregateSplitCandidateDiagnostics {
                rule_name: SINGLE_DISTINCT_AGGREGATE_SPLIT_RULE_NAME,
                aggregate_node_id: node.id,
                distinct_column,
                applied: split.is_ok(),
                rejected_reason: split.as_ref().err().copied(),
            });
            split.ok()
        });
        let image = match split {
            Some(split) => split.push_nodes(&mut nodes, node.facts.clone()),
            None => push_node(&mut nodes, kind, node.facts.clone()),
        };
        images.insert(node.id, image);
    }
    if diagnostics.applied_count == 0 {
        return Ok(SingleDistinctAggregateSplitResult { plan, change: NativeOptimizerChange::Unchanged, diagnostics });
    }
    let root = images[&plan.root_id()];
    Ok(SingleDistinctAggregateSplitResult {
        plan: plan.rewrite(RewrittenNodes::rebuilt(nodes, root, images))?,
        change: NativeOptimizerChange::Changed,
        diagnostics,
    })
}

fn distinct_aggregate(kind: &PlanNode) -> Option<&AggregateNode> {
    match kind {
        PlanNode::Aggregate(aggregate) if aggregate_spec_has_distinct(&aggregate.spec) => Some(aggregate),
        _ => None,
    }
}

fn push_node(nodes: &mut Vec<QueryNode>, kind: PlanNode, facts: PlanFacts) -> NodeId {
    let id = NodeId(nodes.len());
    nodes.push(QueryNode { id, kind, facts });
    id
}

/// The two aggregates, and the projection that finalizes AVG state or the
/// counts of a global merge, that replace one aggregate.
struct AggregateSplit {
    partial: AggregateNode,
    partial_schema: SchemaRef,
    merge_spec: AggregateSpec,
    merge_schema: SchemaRef,
    /// Restores the original schema from AVG state, and from counts that a
    /// global merge sums over no partial rows.
    projection: Option<Vec<ProjectionOutput>>,
}

impl AggregateSplit {
    /// Appends the replacement nodes and returns the one that produces the
    /// original output.
    fn push_nodes(self, nodes: &mut Vec<QueryNode>, original_facts: PlanFacts) -> NodeId {
        let partial = push_node(nodes, PlanNode::Aggregate(self.partial), PlanFacts::new(self.partial_schema));
        let merge = AggregateNode { input: partial, spec: self.merge_spec };
        let Some(outputs) = self.projection else {
            return push_node(nodes, PlanNode::Aggregate(merge), original_facts);
        };
        let merge = push_node(nodes, PlanNode::Aggregate(merge), PlanFacts::new(self.merge_schema));
        push_node(nodes, PlanNode::Projection(ProjectionNode::new(merge, outputs)), original_facts)
    }
}

type SplitOutcome = std::result::Result<AggregateSplit, SingleDistinctAggregateSplitRejectedReason>;

/// Returns the shared distinct input column, when there is one, and the split
/// or the reason the aggregate stays whole.
fn split_aggregate(
    aggregate: &AggregateNode,
    input_schema: &Schema,
    output_schema: &Schema,
) -> (Option<String>, SplitOutcome) {
    match shared_distinct_column(&aggregate.spec) {
        Ok(column) => {
            let split = split_on_distinct_column(aggregate, &column, input_schema, output_schema);
            (Some(column), split)
        },
        Err(reason) => (None, Err(reason)),
    }
}

fn shared_distinct_column(
    spec: &AggregateSpec,
) -> std::result::Result<String, SingleDistinctAggregateSplitRejectedReason> {
    let mut shared: Option<&str> = None;
    for expr in spec.aggr_exprs.iter().filter(|expr| aggregate_expr_is_exact_nunique(expr)) {
        let column = alias_wrapped_column(&expr.expr)
            .ok_or(SingleDistinctAggregateSplitRejectedReason::NonColumnAggregateArgument)?;
        if expr.filter.is_some() {
            return Err(SingleDistinctAggregateSplitRejectedReason::FilteredDistinctAggregate);
        }
        match shared {
            Some(name) if name != column.name() => {
                return Err(SingleDistinctAggregateSplitRejectedReason::MultipleDistinctArguments);
            },
            _ => shared = Some(column.name()),
        }
    }
    shared.map(str::to_string).ok_or(SingleDistinctAggregateSplitRejectedReason::UnresolvedSchema)
}

fn split_on_distinct_column(
    aggregate: &AggregateNode,
    distinct_column: &str,
    input_schema: &Schema,
    output_schema: &Schema,
) -> SplitOutcome {
    let spec = &aggregate.spec;
    if spec.grouping_sets.is_some() {
        return Err(Reason::GroupingSets);
    }
    if spec.group_exprs.iter().filter_map(alias_wrapped_column).any(|key| key.name() == distinct_column) {
        return Err(Reason::DistinctArgumentIsGroupKey);
    }
    let key_count = spec.group_exprs.len();
    if output_schema.fields().len() != key_count + spec.aggr_exprs.len() {
        return Err(Reason::UnresolvedSchema);
    }
    let distinct_field = input_schema.field_with_name(distinct_column).map_err(|_| Reason::UnresolvedSchema)?;
    let key_fields = output_schema.fields()[..key_count].iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    // AVG state columns sit beside the original outputs in the merge schema.
    let mut taken = output_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut unique_name = |base: &str| {
        let name = unique_alias_with_suffix(base, |candidate| taken.contains(candidate));
        taken.insert(name.clone());
        name
    };

    let distinct_key = unique_name(distinct_column);
    let mut partial_group_exprs = spec.group_exprs.clone();
    partial_group_exprs.push(Expr::column(distinct_column).alias(distinct_key.as_str()));
    let mut partial_fields = key_fields.clone();
    partial_fields.push(distinct_field.clone().with_name(distinct_key.as_str()));
    let mut partial_exprs = Vec::new();
    let mut merge_exprs = Vec::with_capacity(spec.aggr_exprs.len());
    let mut merge_fields = key_fields;
    // Output index and the projection expression that restores it.
    let mut finalized_outputs = Vec::<(usize, Expr)>::new();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let output_field = output_schema.field(key_count + index);
        if output_field.name() != &expr.alias {
            return Err(Reason::UnresolvedSchema);
        }
        if aggregate_expr_is_exact_nunique(expr) {
            merge_exprs.push(AggregateExpr::new(
                AggregateFunction::Count,
                Expr::column(distinct_key.as_str()),
                expr.alias.as_str(),
            ));
            merge_fields.push(output_field.clone());
            continue;
        }
        if expr.distinct
            || !matches!(
                expr.function,
                AggregateFunction::Sum
                    | AggregateFunction::Count
                    | AggregateFunction::Min
                    | AggregateFunction::Max
                    | AggregateFunction::Avg
            )
        {
            return Err(Reason::NonDecomposableAggregate);
        }
        // The partial aggregate is grouped, and grouped aggregates read
        // columns; a literal count-star argument exists only for global ones.
        let Some(argument) = alias_wrapped_column(&expr.expr) else {
            return Err(Reason::NonColumnAggregateArgument);
        };
        if expr.function == AggregateFunction::Avg {
            let input_field = input_schema.field_with_name(argument.name()).map_err(|_| Reason::UnresolvedSchema)?;
            let sum_type = avg_declared_sum_state_type(input_field.data_type(), output_field.data_type())
                .map_err(|_| Reason::NonDecomposableAggregate)?;
            let sum = unique_name(format!("__algeon_native_distinct_partial_{index}_sum").as_str());
            let count = unique_name(format!("__algeon_native_distinct_partial_{index}_count").as_str());
            partial_exprs.extend(
                avg_partial_aggregates(&expr.expr, sum.as_str(), count.as_str())
                    .map(|partial| partial.with_filter(expr.filter.clone())),
            );
            partial_fields.push(Field::new(sum.as_str(), sum_type.clone(), true));
            partial_fields.push(Field::new(count.as_str(), AVG_COUNT_STATE_TYPE, false));
            merge_exprs.extend(avg_merge_aggregates(sum.as_str(), count.as_str()));
            merge_fields.push(Field::new(sum.as_str(), sum_type, true));
            merge_fields
                .push(merged_count_field(Field::new(count.as_str(), AVG_COUNT_STATE_TYPE, false), key_count == 0));
            let avg =
                avg_final_expr(Expr::column(sum.as_str()), Expr::column(count.as_str()), output_field.data_type())
                    .map_err(|_| Reason::NonDecomposableAggregate)?;
            finalized_outputs.push((key_count + index, avg.alias(expr.alias.as_str())));
            continue;
        }
        let merge_function = match expr.function {
            AggregateFunction::Count => AggregateFunction::Sum,
            function => function,
        };
        let state = unique_name(format!("__algeon_native_distinct_partial_{index}").as_str());
        partial_exprs.push(
            AggregateExpr::new(expr.function, expr.expr.clone(), state.as_str()).with_filter(expr.filter.clone()),
        );
        let is_count = expr.function == AggregateFunction::Count;
        // A partial SUM, MIN or MAX is null when no row of its group passes
        // the filter or every argument is null.
        partial_fields.push(Field::new(state.as_str(), output_field.data_type().clone(), !is_count));
        merge_exprs.push(AggregateExpr::new(merge_function, Expr::column(state.as_str()), expr.alias.as_str()));
        let global_count = is_count && key_count == 0;
        if global_count {
            let zero_filled = zero_filled_count_expr(Expr::column(expr.alias.as_str()), output_field.data_type())
                .map_err(|_| Reason::NonDecomposableAggregate)?;
            finalized_outputs.push((key_count + index, zero_filled.alias(expr.alias.as_str())));
        }
        merge_fields.push(merged_count_field(output_field.clone(), global_count));
    }

    let projection = (!finalized_outputs.is_empty()).then(|| {
        output_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                let expr = match finalized_outputs.iter().find(|(output_index, _)| *output_index == index) {
                    Some((_, expr)) => expr.clone(),
                    None => Expr::column(field.name().as_str()),
                };
                ProjectionOutput::new(expr, field.as_ref().clone())
            })
            .collect()
    });
    let merge_group_exprs =
        output_schema.fields()[..key_count].iter().map(|field| Expr::column(field.name().as_str())).collect::<Vec<_>>();
    Ok(AggregateSplit {
        partial: AggregateNode { input: aggregate.input, spec: AggregateSpec::new(partial_group_exprs, partial_exprs) },
        partial_schema: Schema::new(partial_fields).into(),
        merge_spec: AggregateSpec::new(merge_group_exprs, merge_exprs),
        merge_schema: Schema::new(merge_fields).into(),
        projection,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::DataType;

    use super::*;
    use crate::{
        expr::{CaseWhen, GroupingSetsSpec, LiteralValue},
        plan::QueryPlanBuilder,
        prepared::{aggregate_spec_requires_whole_input, avg_final_expr},
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn grouped_distinct_count_with_mergeable_aggregates_splits_into_two_streaming_aggregates() {
        let plan = aggregate_plan(
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "sy"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cy"),
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("y"), "lo"),
                    AggregateExpr::new(AggregateFunction::Max, Expr::column("y"), "hi"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cf")
                        .with_filter(Some(Expr::column("f"))),
                ],
            ),
            vec![
                Field::new("k", DataType::Int32, true),
                count_field("dx"),
                value_field("sy"),
                count_field("cy"),
                value_field("lo"),
                value_field("hi"),
                count_field("cf"),
            ],
        );

        let result = split(&plan);

        assert_eq!(result.change, NativeOptimizerChange::Changed);
        assert_eq!(result.diagnostics.applied_count, 1);
        assert_eq!(result.diagnostics.candidates[0].distinct_column.as_deref(), Some("x"));
        assert_eq!(result.plan.output_schema(), plan.output_schema());
        let (partial, merge) = partial_and_merge(&result.plan, result.plan.root_id());
        assert!(!aggregate_spec_requires_whole_input(&partial.spec));
        assert!(!aggregate_spec_requires_whole_input(&merge.spec));
        assert_eq!(partial.spec.group_exprs, vec![Expr::column("k"), Expr::column("x").alias("x")]);
        assert_eq!(
            partial.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.filter.clone())).collect::<Vec<_>>(),
            vec![
                (AggregateFunction::Sum, None),
                (AggregateFunction::Count, None),
                (AggregateFunction::Min, None),
                (AggregateFunction::Max, None),
                (AggregateFunction::Count, Some(Expr::column("f"))),
            ]
        );
        assert_eq!(merge.spec.group_exprs, vec![Expr::column("k")]);
        assert_eq!(
            merge.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.alias.as_str())).collect::<Vec<_>>(),
            vec![
                (AggregateFunction::Count, "dx"),
                (AggregateFunction::Sum, "sy"),
                (AggregateFunction::Sum, "cy"),
                (AggregateFunction::Min, "lo"),
                (AggregateFunction::Max, "hi"),
                (AggregateFunction::Sum, "cf"),
            ]
        );
        assert_eq!(merge.spec.aggr_exprs[0].expr, Expr::column("x"));
    }

    #[test]
    fn global_distinct_count_split_fills_counts_of_empty_input_with_zero() {
        let plan = aggregate_plan(
            AggregateSpec::new(
                Vec::new(),
                vec![
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cy"),
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "sy"),
                ],
            ),
            vec![count_field("dx"), count_field("cy"), value_field("sy")],
        );

        let result = split(&plan);

        assert_eq!(result.change, NativeOptimizerChange::Changed);
        assert_eq!(result.plan.output_schema(), plan.output_schema());
        let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
            panic!("a global split restores counts in a projection: {}", result.plan);
        };
        let (partial, merge) = partial_and_merge(&result.plan, projection.input);
        assert_eq!(partial.spec.group_exprs, vec![Expr::column("x").alias("x")]);
        assert!(merge.spec.group_exprs.is_empty());
        let zero_filled = Expr::case(
            vec![CaseWhen::new(Expr::is_null(Expr::column("cy")), Expr::literal(LiteralValue::Int64(0)))],
            Expr::column("cy"),
        )
        .alias("cy");
        assert_eq!(
            projection.expressions().cloned().collect::<Vec<_>>(),
            vec![Expr::column("dx"), zero_filled, Expr::column("sy")]
        );
    }

    #[test]
    fn grouped_distinct_count_with_avg_carries_sum_and_count_state_to_a_finalizing_projection() {
        let plan = aggregate_plan(
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("y"), "ay")
                        .with_filter(Some(Expr::column("f"))),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "ad"),
                ],
            ),
            vec![
                Field::new("k", DataType::Int32, true),
                count_field("dx"),
                Field::new("ay", DataType::Float64, true),
                Field::new("ad", DataType::Decimal128(13, 6), true),
            ],
        );

        let result = split(&plan);

        assert_eq!(result.change, NativeOptimizerChange::Changed);
        assert_eq!(result.plan.output_schema(), plan.output_schema());
        let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
            panic!("a split with AVG finalizes it in a projection: {}", result.plan);
        };
        let (partial, merge) = partial_and_merge(&result.plan, projection.input);
        assert!(!aggregate_spec_requires_whole_input(&partial.spec));
        assert_eq!(
            partial.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.filter.clone())).collect::<Vec<_>>(),
            vec![
                (AggregateFunction::Sum, Some(Expr::column("f"))),
                (AggregateFunction::Count, Some(Expr::column("f"))),
                (AggregateFunction::Sum, None),
                (AggregateFunction::Count, None),
            ]
        );
        let partial_schema = result.plan.node(merge.input).unwrap().output_schema();
        assert_eq!(partial_schema.field(2).data_type(), &DataType::Float64);
        assert_eq!(partial_schema.field(4).data_type(), &DataType::Decimal128(38, 2));
        let states = partial.spec.aggr_exprs.iter().map(|expr| expr.alias.as_str()).collect::<Vec<_>>();
        assert_eq!(
            merge.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.alias.as_str())).collect::<Vec<_>>(),
            vec![
                (AggregateFunction::Count, "dx"),
                (AggregateFunction::Sum, states[0]),
                (AggregateFunction::Sum, states[1]),
                (AggregateFunction::Sum, states[2]),
                (AggregateFunction::Sum, states[3]),
            ]
        );
        let avg = |sum: &str, count: &str, output: DataType, alias: &str| {
            avg_final_expr(Expr::column(sum), Expr::column(count), &output).unwrap().alias(alias)
        };
        assert_eq!(
            projection.expressions().cloned().collect::<Vec<_>>(),
            vec![
                Expr::column("k"),
                Expr::column("dx"),
                avg(states[0], states[1], DataType::Float64, "ay"),
                avg(states[2], states[3], DataType::Decimal128(13, 6), "ad"),
            ]
        );
    }

    #[test]
    fn global_distinct_count_with_avg_merges_state_whose_count_is_null_over_empty_input() {
        let plan = aggregate_plan(
            AggregateSpec::new(
                Vec::new(),
                vec![
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "ad"),
                ],
            ),
            vec![count_field("dx"), Field::new("ad", DataType::Decimal128(13, 6), true)],
        );

        let result = split(&plan);

        assert_eq!(result.change, NativeOptimizerChange::Changed);
        assert_eq!(result.plan.output_schema(), plan.output_schema());
        let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
            panic!("a split with AVG finalizes it in a projection: {}", result.plan);
        };
        let merge_schema = result.plan.node(projection.input).unwrap().output_schema();
        let count_state = merge_schema.field(2);
        assert!(count_state.is_nullable(), "a global merge over no partial rows has a null count: {count_state:?}");
        assert_eq!(
            projection.expressions().nth(1).cloned(),
            Some(
                avg_final_expr(
                    Expr::column(merge_schema.field(1).name().as_str()),
                    Expr::column(count_state.name().as_str()),
                    &DataType::Decimal128(13, 6),
                )
                .unwrap()
                .alias("ad")
            )
        );
    }

    #[test]
    fn aggregates_without_one_shared_unfiltered_distinct_argument_or_mergeable_state_stay_whole() {
        use SingleDistinctAggregateSplitRejectedReason as Reason;

        let distinct_x = || AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx");
        let cases = [
            (
                "two distinct arguments",
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![distinct_x(), AggregateExpr::new(AggregateFunction::NUnique, Expr::column("y"), "dy")],
                ),
                None,
                Reason::MultipleDistinctArguments,
            ),
            (
                "filtered distinct",
                AggregateSpec::new(vec![Expr::column("k")], vec![distinct_x().with_filter(Some(Expr::column("f")))]),
                None,
                Reason::FilteredDistinctAggregate,
            ),
            (
                "median",
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![distinct_x(), AggregateExpr::new(AggregateFunction::Median, Expr::column("y"), "my")],
                ),
                Some("x"),
                Reason::NonDecomposableAggregate,
            ),
            (
                "statistical",
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![distinct_x(), AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("y"), "my")],
                ),
                Some("x"),
                Reason::NonDecomposableAggregate,
            ),
            (
                "avg over a decimal DataFusion sums as Decimal256",
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![distinct_x(), AggregateExpr::new(AggregateFunction::Avg, Expr::column("wide"), "aw")],
                ),
                Some("x"),
                Reason::NonDecomposableAggregate,
            ),
            (
                "grouping sets",
                AggregateSpec::new(vec![Expr::column("k")], vec![distinct_x()])
                    .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
                Some("x"),
                Reason::GroupingSets,
            ),
        ];

        for (label, spec, distinct_column, reason) in cases {
            let mut fields = vec![Field::new("k", DataType::Int32, true)];
            if spec.grouping_sets.is_some() {
                fields.push(Field::new("__grouping_id", DataType::UInt8, false));
            }
            fields.extend(spec.aggr_exprs.iter().map(|expr| match expr.function {
                AggregateFunction::Avg => Field::new(expr.alias.as_str(), DataType::Decimal128(34, 6), true),
                _ => count_field(expr.alias.as_str()),
            }));
            let plan = aggregate_plan(spec, fields);

            let result = split(&plan);

            assert_eq!(result.change, NativeOptimizerChange::Unchanged, "{label}");
            assert_eq!(result.plan, plan, "{label}");
            assert_eq!(result.diagnostics.considered_count, 1, "{label}");
            let candidate = &result.diagnostics.candidates[0];
            assert_eq!(candidate.distinct_column.as_deref(), distinct_column, "{label}");
            assert_eq!(candidate.rejected_reason, Some(reason), "{label}");
        }
    }

    fn split(plan: &QueryPlan) -> SingleDistinctAggregateSplitResult {
        optimize_single_distinct_aggregate_split(plan.clone(), &SingleDistinctAggregateSplitOptions::default()).unwrap()
    }

    fn aggregate_plan(spec: AggregateSpec, output_fields: Vec<Field>) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
            Field::new("f", DataType::Boolean, true),
            Field::new("d", DataType::Decimal128(9, 2), true),
            Field::new("wide", DataType::Decimal128(30, 2), true),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
        let aggregate = builder.add_aggregate(source, spec, Arc::new(Schema::new(output_fields))).unwrap();
        builder.finish_gpu_resident(aggregate).unwrap()
    }

    /// The merge aggregate at `merge_id` and the partial aggregate it reads.
    fn partial_and_merge(plan: &QueryPlan, merge_id: NodeId) -> (&AggregateNode, &AggregateNode) {
        let aggregate = |id: NodeId| match &plan.node(id).unwrap().kind {
            PlanNode::Aggregate(aggregate) => aggregate,
            _ => panic!("node {id} should be an aggregate: {plan}"),
        };
        let merge = aggregate(merge_id);
        (aggregate(merge.input), merge)
    }

    fn count_field(name: &str) -> Field {
        Field::new(name, DataType::Int64, false)
    }

    fn value_field(name: &str) -> Field {
        Field::new(name, DataType::Int64, true)
    }
}
