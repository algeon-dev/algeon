use super::*;
use crate::bounded::BoundedJoinPipelineStage;

pub(in crate::exec::bounded) type ProjectionStage<'a> = (&'a ProjectionNode, &'a arrow::datatypes::SchemaRef);

#[derive(Clone)]
pub(in crate::exec::bounded) struct BoundedPlan<'a> {
    pub(in crate::exec::bounded) source_id: NodeId,
    pub(in crate::exec::bounded) source: &'a LocalParquetSource,
    pub(in crate::exec::bounded) pre_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) filter: Option<&'a FilterNode>,
    pub(in crate::exec::bounded) post_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) aggregate_id: NodeId,
    pub(in crate::exec::bounded) aggregate_input_schema: &'a arrow::datatypes::SchemaRef,
    pub(in crate::exec::bounded) aggregate: (&'a AggregateNode, &'a arrow::datatypes::SchemaRef),
    pub(in crate::exec::bounded) post_aggregate_projection: Option<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) sort: Option<&'a SortNode>,
    pub(in crate::exec::bounded) limit: Option<&'a LimitNode>,
}

fn take_projection_stack<'a>(
    plan: &'a QueryPlan,
    current: &mut NodeId,
    missing_message: &'static str,
) -> Result<Vec<ProjectionStage<'a>>> {
    let mut projections = Vec::new();
    loop {
        let Some(node) = plan.node(*current) else {
            return Err(invalid_node_reference(*current, missing_message));
        };
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        projections.push((projection, &node.facts.output_schema));
        *current = projection.input;
    }
    projections.reverse();
    Ok(projections)
}

impl<'a> BoundedPlan<'a> {
    pub(in crate::exec::bounded) fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let mut current = plan.sink().input();
        let limit = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Limit(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let sort = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Sort(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };
        if limit.is_some() && sort.is_none() {
            return Err(unsupported_bounded_shape(
                "bounded execution does not support limit without a preceding global sort",
                "limit_not_directly_above_sort",
            ));
        }

        let post_aggregate_projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => None,
            },
            None => {
                return Err(invalid_node_reference(current, "bounded query plan references a missing sort input node"));
            },
        };

        let aggregate_id = current;
        let aggregate_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing aggregate node"))?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently requires a grouped sum aggregate after source-local operators",
                "missing_grouped_aggregate",
            ));
        };
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(aggregate.input, "bounded query plan references a missing aggregate input node")
        })?;
        current = aggregate.input;

        let post_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing aggregate input node")?;

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let pre_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing source input node")?;

        let source_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing source node"))?;
        let PlanNode::Source(source_node) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded aggregate execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_projection,
            sort,
            limit,
        })
    }

    pub(in crate::exec::bounded) fn analyze_projection_wrapped_local_aggregate_subplan(
        plan: &'a QueryPlan,
        root_id: NodeId,
    ) -> Result<Self> {
        let mut current = root_id;
        let post_aggregate_projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => {
                    return Err(unsupported_bounded_shape(
                        "bounded mixed keyed residual join grouped aggregate input requires a projection-wrapped aggregate",
                        "bounded_join_plan_shape_unsupported",
                    )
                    .with_fact("node_id", root_id)
                    .with_fact("node_kind", bounded_local_aggregate_subplan_node_kind(&node.kind)));
                },
            },
            None => {
                return Err(invalid_node_reference(
                    current,
                    "bounded mixed keyed residual join grouped aggregate input references a missing projection node",
                ));
            },
        };

        let aggregate_id = current;
        let aggregate_node = plan.node(current).ok_or_else(|| {
            invalid_node_reference(
                current,
                "bounded mixed keyed residual join grouped aggregate input references a missing aggregate node",
            )
        })?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded mixed keyed residual join grouped aggregate input requires a grouped aggregate",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", aggregate_id)
            .with_fact("node_kind", bounded_local_aggregate_subplan_node_kind(&aggregate_node.kind)));
        };
        if aggregate.spec.group_exprs.is_empty() {
            return Err(unsupported_bounded_shape(
                "bounded mixed keyed residual join grouped aggregate input requires grouped aggregate state",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", aggregate_id)
            .with_fact("node_kind", "scalar_aggregate"));
        }
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(
                aggregate.input,
                "bounded mixed keyed residual join grouped aggregate input references a missing aggregate input node",
            )
        })?;
        current = aggregate.input;

        let post_filter_projections = take_projection_stack(
            plan,
            &mut current,
            "bounded mixed keyed residual join grouped aggregate input references a missing aggregate input node",
        )?;

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let pre_filter_projections = take_projection_stack(
            plan,
            &mut current,
            "bounded mixed keyed residual join grouped aggregate input references a missing source input node",
        )?;

        let source_node = plan.node(current).ok_or_else(|| {
            invalid_node_reference(
                current,
                "bounded mixed keyed residual join grouped aggregate input references a missing source node",
            )
        })?;
        let PlanNode::Source(source_node) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded mixed keyed residual join grouped aggregate input supports only local parquet sources",
                "non_local_parquet_source",
            )
            .with_fact("node_id", current)
            .with_fact("node_kind", bounded_local_aggregate_subplan_node_kind(&source_node.kind)));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded mixed keyed residual join grouped aggregate input supports only local parquet sources",
                "non_local_parquet_source",
            )
            .with_fact("node_id", current));
        };

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_projection,
            sort: None,
            limit: None,
        })
    }

    pub(in crate::exec::bounded) fn analyze_projection_wrapped_local_aggregate_subplan_with_post_stages(
        plan: &'a QueryPlan,
        root_id: NodeId,
    ) -> Result<(Self, Vec<BoundedJoinPipelineStage<'a>>)> {
        let mut current = root_id;
        let mut post_aggregate_stages = Vec::new();
        loop {
            let node = plan.node(current).ok_or_else(|| {
                invalid_node_reference(
                    current,
                    "bounded resident grouped aggregate input references a missing post-aggregate node",
                )
            })?;
            match &node.kind {
                PlanNode::Projection(projection) => {
                    post_aggregate_stages.push(BoundedJoinPipelineStage::Projection {
                        node_id: current,
                        node: projection,
                        output_schema: &node.facts.output_schema,
                    });
                    current = projection.input;
                },
                PlanNode::Filter(filter) => {
                    post_aggregate_stages.push(BoundedJoinPipelineStage::Filter { node_id: current, node: filter });
                    current = filter.input;
                },
                PlanNode::Aggregate(_) => break,
                _ => {
                    return Err(unsupported_bounded_shape(
                        "bounded resident grouped aggregate input requires projection/filter stages over a grouped aggregate",
                        "bounded_join_plan_shape_unsupported",
                    )
                    .with_fact("node_id", current)
                    .with_fact("node_kind", bounded_local_aggregate_subplan_node_kind(&node.kind)));
                },
            }
        }
        if !post_aggregate_stages.iter().any(|stage| matches!(stage, BoundedJoinPipelineStage::Projection { .. })) {
            return Err(unsupported_bounded_shape(
                "bounded resident grouped aggregate input requires a projection-wrapped aggregate",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", root_id)
            .with_fact("node_kind", "bare_grouped_aggregate_input"));
        }
        post_aggregate_stages.reverse();

        let aggregate_id = current;
        let aggregate_node = plan.node(current).ok_or_else(|| {
            invalid_node_reference(
                current,
                "bounded resident grouped aggregate input references a missing aggregate node",
            )
        })?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            unreachable!("loop exits only at aggregate nodes");
        };
        if aggregate.spec.group_exprs.is_empty() {
            return Err(unsupported_bounded_shape(
                "bounded resident grouped aggregate input requires grouped aggregate state",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", aggregate_id)
            .with_fact("node_kind", "scalar_aggregate"));
        }
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(
                aggregate.input,
                "bounded resident grouped aggregate input references a missing aggregate input node",
            )
        })?;
        current = aggregate.input;

        let post_filter_projections = take_projection_stack(
            plan,
            &mut current,
            "bounded resident grouped aggregate input references a missing aggregate input node",
        )?;

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let pre_filter_projections = take_projection_stack(
            plan,
            &mut current,
            "bounded resident grouped aggregate input references a missing source input node",
        )?;

        let source_node = plan.node(current).ok_or_else(|| {
            invalid_node_reference(current, "bounded resident grouped aggregate input references a missing source node")
        })?;
        let PlanNode::Source(source_node) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded resident grouped aggregate input supports only local parquet sources",
                "non_local_parquet_source",
            )
            .with_fact("node_id", current)
            .with_fact("node_kind", bounded_local_aggregate_subplan_node_kind(&source_node.kind)));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded resident grouped aggregate input supports only local parquet sources",
                "non_local_parquet_source",
            )
            .with_fact("node_id", current));
        };

        Ok((
            Self {
                source_id: current,
                source,
                pre_filter_projections,
                filter,
                post_filter_projections,
                aggregate_id,
                aggregate_input_schema: &aggregate_input_node.facts.output_schema,
                aggregate: (aggregate, &aggregate_node.facts.output_schema),
                post_aggregate_projection: None,
                sort: None,
                limit: None,
            },
            post_aggregate_stages,
        ))
    }

    pub(in crate::exec::bounded) fn validate_capabilities(&self) -> Result<()> {
        if let Some(filter) = self.filter {
            lower_expr(&filter.predicate)?;
        }

        for (projection, output_schema) in &self.pre_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        for (projection, output_schema) in &self.post_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        validate_aggregate_capability(self.aggregate.0, self.aggregate.1)?;
        validate_final_aggregate_capability(self.aggregate.0, self.aggregate.1)?;

        if let Some((projection, output_schema)) = self.post_aggregate_projection {
            validate_projection_capability(projection, output_schema)?;
        }

        if let Some(sort) = self.sort {
            lower_sort_spec(&sort.spec)?;
        }
        if let Some(limit) = self.limit {
            topk_candidate_rows(limit.spec)?;
        }

        Ok(())
    }
}

fn bounded_local_aggregate_subplan_node_kind(node: &PlanNode) -> &'static str {
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
