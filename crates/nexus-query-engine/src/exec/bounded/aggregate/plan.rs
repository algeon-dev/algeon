use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;

pub(in crate::exec::bounded) type ProjectionStage<'a> = (&'a ProjectionNode, &'a ArrowSchemaRef);

#[derive(Clone, Copy)]
pub(in crate::exec::bounded) enum BoundedUnaryStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode, output_schema: &'a ArrowSchemaRef },
}

#[derive(Clone)]
pub(in crate::exec::bounded) struct BoundedPlan<'a> {
    pub(in crate::exec::bounded) source_id: NodeId,
    pub(in crate::exec::bounded) source: &'a LocalParquetSource,
    pub(in crate::exec::bounded) pre_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) filter: Option<&'a FilterNode>,
    pub(in crate::exec::bounded) post_filter_projections: Vec<ProjectionStage<'a>>,
    pub(in crate::exec::bounded) aggregate_id: NodeId,
    pub(in crate::exec::bounded) aggregate_input_schema: &'a ArrowSchemaRef,
    pub(in crate::exec::bounded) aggregate: (&'a AggregateNode, &'a ArrowSchemaRef),
    pub(in crate::exec::bounded) post_aggregate_stages: Vec<BoundedUnaryStage<'a>>,
    pub(in crate::exec::bounded) sort: Option<&'a SortNode>,
    pub(in crate::exec::bounded) post_sort_projections: Vec<BoundedUnaryStage<'a>>,
    pub(in crate::exec::bounded) limit: Option<(NodeId, &'a LimitNode)>,
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

fn take_post_aggregate_stage_stack<'a>(
    plan: &'a QueryPlan,
    current: &mut NodeId,
    missing_message: &'static str,
) -> Result<Vec<BoundedUnaryStage<'a>>> {
    let mut stages = Vec::new();
    loop {
        let Some(node) = plan.node(*current) else {
            return Err(invalid_node_reference(*current, missing_message));
        };
        match &node.kind {
            PlanNode::Filter(filter) => {
                stages.push(BoundedUnaryStage::Filter { node_id: *current, node: filter });
                *current = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(BoundedUnaryStage::Projection {
                    node_id: *current,
                    node: projection,
                    output_schema: &node.facts.output_schema,
                });
                *current = projection.input;
            },
            _ => break,
        }
    }
    stages.reverse();
    Ok(stages)
}

impl<'a> BoundedPlan<'a> {
    pub(in crate::exec::bounded) fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let mut current = plan.sink().input();
        let limit = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Limit(node)) => {
                let limit_id = current;
                current = node.input;
                Some((limit_id, node))
            },
            _ => None,
        };

        let limit_input_projections = take_post_aggregate_stage_stack(
            plan,
            &mut current,
            "bounded query plan references a missing limit input node",
        )?;

        let sort = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Sort(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };
        let (post_sort_projections, aggregate_tail_projections) =
            if sort.is_some() { (limit_input_projections, Vec::new()) } else { (Vec::new(), limit_input_projections) };

        let mut post_aggregate_stages = take_post_aggregate_stage_stack(
            plan,
            &mut current,
            "bounded query plan references a missing sort input node",
        )?;
        if sort.is_none() {
            post_aggregate_stages.extend(aggregate_tail_projections);
        }

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
        reject_position_delete_visible_scan_limit_for_specialized_bounded_aggregate(source)?;

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_stages,
            sort,
            post_sort_projections,
            limit,
        })
    }

    pub(in crate::exec::bounded) fn analyze_aggregate_subplan(
        plan: &'a QueryPlan,
        aggregate_id: NodeId,
    ) -> Result<Self> {
        let aggregate_node = plan.node(aggregate_id).ok_or_else(|| {
            invalid_node_reference(aggregate_id, "bounded query plan references a missing aggregate node")
        })?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded aggregate subplan requires an aggregate root",
                "missing_grouped_aggregate",
            ));
        };
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(aggregate.input, "bounded query plan references a missing aggregate input node")
        })?;
        let mut current = aggregate.input;

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
                "bounded aggregate subplan currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded aggregate subplan currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };
        reject_position_delete_visible_scan_limit_for_specialized_bounded_aggregate(source)?;

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_stages: Vec::new(),
            sort: None,
            post_sort_projections: Vec::new(),
            limit: None,
        })
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
        if !aggregate_spec_requires_whole_input(&self.aggregate.0.spec) {
            validate_final_aggregate_capability(self.aggregate.0, self.aggregate.1)?;
        }

        for stage in &self.post_aggregate_stages {
            validate_bounded_unary_stage_capability(stage)?;
        }

        if let Some(sort) = self.sort {
            lower_sort_spec(&sort.spec)?;
        }
        if let Some((_, limit)) = self.limit {
            topk_candidate_rows(limit.spec)?;
        }
        for stage in &self.post_sort_projections {
            validate_bounded_unary_stage_capability(stage)?;
        }

        Ok(())
    }

    pub(in crate::exec::bounded) fn has_source_adjacent_stages(&self) -> bool {
        !self.pre_filter_projections.is_empty() || self.filter.is_some() || !self.post_filter_projections.is_empty()
    }
}

fn reject_position_delete_visible_scan_limit_for_specialized_bounded_aggregate(
    source: &LocalParquetSource,
) -> Result<()> {
    if !source.has_position_delete_visible_scan_limit() {
        return Ok(());
    }
    Err(unsupported_bounded_shape(
        "specialized bounded aggregate execution does not own Iceberg position-delete visible-row scan-limit stopping",
        "position_delete_visible_scan_limit_requires_generic_source_execution",
    ))
}

fn validate_bounded_unary_stage_capability(stage: &BoundedUnaryStage<'_>) -> Result<()> {
    match stage {
        BoundedUnaryStage::Filter { node, .. } => {
            lower_expr(&node.predicate)?;
        },
        BoundedUnaryStage::Projection { node, output_schema, .. } => {
            validate_projection_capability(node, output_schema)?;
        },
    }
    Ok(())
}
