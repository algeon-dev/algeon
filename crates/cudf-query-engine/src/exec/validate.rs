use std::collections::BTreeSet;

use arrow::datatypes::{Field, SchemaRef};

use super::*;
use crate::{
    datatypes::is_supported_arrow_type,
    plan::{QueryNode, UnionNode},
};

mod expressions;
mod join;
mod types;

use expressions::{comma_separated, validate_expr, validate_projection_output_expr};
use join::validate_join_capability;
use types::require_boolean;

pub(super) fn validate_query_plan(plan: &QueryPlan) -> Result<()> {
    validate_reachability(plan)?;
    let mut available_columns_by_node = Vec::with_capacity(plan.node_count());

    for (index, node) in plan.nodes().iter().enumerate() {
        if node.id.index() != index {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.plan.validate",
                "query plan node id must match its position",
            )
            .with_fact("node", node.id)
            .with_fact("position", index));
        }

        validate_schema_supported(node.output_schema(), "cudf_query_engine.plan.validate_schema")?;
        validate_output_names_unique(node)?;
        let available_columns = match &node.kind {
            PlanNode::Source(source) => validate_source_node(source)?,
            PlanNode::Filter(filter) => {
                let input = input_node(plan, filter.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, filter.input)?;
                let predicate_type = validate_expr(&filter.predicate, input.output_schema(), input_available)?;
                require_boolean(&predicate_type, "filter predicate")?;
                lower_expr(&filter.predicate)?;
                filter_available_columns(input_available, node.output_schema())
            },
            PlanNode::Projection(projection) => {
                let input = input_node(plan, projection.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, projection.input)?;
                for output in &projection.outputs {
                    validate_projection_output_expr(output, input.output_schema(), input_available)?;
                }
                validate_projection_capability(projection, &node.facts.output_schema)?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Aggregate(aggregate) => {
                let input = input_node(plan, aggregate.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, aggregate.input)?;
                for expr in &aggregate.spec.group_exprs {
                    validate_expr(expr, input.output_schema(), input_available)?;
                }
                for expr in &aggregate.spec.aggr_exprs {
                    validate_expr(&expr.expr, input.output_schema(), input_available)?;
                }
                validate_aggregate_capability(aggregate, node.output_schema())?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Window(window) => {
                let input = input_node(plan, window.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, window.input)?;
                for expr in &window.spec.partition_by {
                    validate_expr(expr, input.output_schema(), input_available)?;
                }
                for sort_expr in &window.spec.order_by {
                    validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
                }
                for function in &window.spec.functions {
                    match function {
                        crate::plan::WindowFunction::Aggregate { expr, .. } => {
                            validate_expr(expr, input.output_schema(), input_available)?;
                        },
                        crate::plan::WindowFunction::Rank { .. } => {},
                    }
                }
                validate_window_capability(window, input.output_schema(), node.output_schema())?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Sort(sort) => {
                let input = input_node(plan, sort.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, sort.input)?;
                for sort_expr in &sort.spec.expressions {
                    validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
                }
                lower_sort_spec(&sort.spec)?;
                filter_available_columns(input_available, node.output_schema())
            },
            PlanNode::Limit(limit) => {
                let input = input_node(plan, limit.input, node.id)?;
                topk_candidate_rows(limit.spec)?;
                let input_available = available_columns(&available_columns_by_node, limit.input)?;
                filter_available_columns(input_available, input.output_schema())
            },
            PlanNode::Join(join) => {
                let left = input_node(plan, join.inputs[0], node.id)?;
                let right = input_node(plan, join.inputs[1], node.id)?;
                let left_available = available_columns(&available_columns_by_node, join.inputs[0])?;
                let right_available = available_columns(&available_columns_by_node, join.inputs[1])?;
                validate_join_capability(
                    join,
                    left.output_schema(),
                    left_available,
                    right.output_schema(),
                    right_available,
                    node.output_schema(),
                )?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Union(union) => {
                validate_union_node(plan, union, node.id, node.output_schema(), &available_columns_by_node)?;
                schema_column_names(node.output_schema())
            },
        };
        available_columns_by_node.push(available_columns);
    }

    let sink_input = input_node(plan, plan.sink().input(), NodeId::from_index_for_internal_use(plan.node_count()))?;
    let sink_available = available_columns(&available_columns_by_node, plan.sink().input())?;
    if let ResultSink::HostArrow { schema, .. } = plan.sink() {
        validate_schema_supported(schema, "cudf_query_engine.plan.validate_sink_schema")?;
        if schema.fields().len() != sink_input.output_schema().fields().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.plan.validate_sink",
                "host sink schema field count must match the root node output",
            )
            .with_fact("root_node", sink_input.id)
            .with_fact("sink_fields", schema.fields().len())
            .with_fact("root_fields", sink_input.output_schema().fields().len()));
        }
        for (index, (sink_field, root_field)) in
            schema.fields().iter().zip(sink_input.output_schema().fields().iter()).enumerate()
        {
            if !sink_available.contains(root_field.name()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.plan.validate_sink",
                    "host sink root schema references a column not produced by the root node",
                )
                .with_fact("column", root_field.name())
                .with_fact("root_node", sink_input.id)
                .with_fact("available_columns", comma_separated(sink_available)));
            }
            if !fields_match_except_name(sink_field.as_ref(), root_field.as_ref()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.plan.validate_sink",
                    "host sink schema field must match the root node output by ordinal except for field name",
                )
                .with_fact("root_node", sink_input.id)
                .with_fact("field_index", index)
                .with_fact("sink_field", sink_field.name())
                .with_fact("root_field", root_field.name())
                .with_fact("sink_type", format!("{:?}", sink_field.data_type()))
                .with_fact("root_type", format!("{:?}", root_field.data_type()))
                .with_fact("sink_nullable", sink_field.is_nullable())
                .with_fact("root_nullable", root_field.is_nullable())
                .with_fact("sink_metadata", format!("{:?}", sink_field.metadata()))
                .with_fact("root_metadata", format!("{:?}", root_field.metadata())));
            }
        }
    }

    Ok(())
}

fn fields_match_except_name(sink_field: &Field, root_field: &Field) -> bool {
    sink_field.clone().with_name(root_field.name().clone()) == *root_field
}

fn validate_reachability(plan: &QueryPlan) -> Result<()> {
    let mut reachable = vec![false; plan.node_count()];
    let mut stack = vec![plan.sink().input()];
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.plan.validate_reachability",
                "query plan sink reaches a missing node",
            )
            .with_fact("node", id));
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        stack.extend(node.input_ids().iter().copied());
    }

    if let Some((index, _)) = reachable.iter().enumerate().find(|(_, reachable)| !**reachable) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate_reachability",
            "query plan contains an unreachable node",
        )
        .with_fact("node", NodeId::from_index_for_internal_use(index))
        .with_fact("sink_input", plan.sink().input()));
    }
    Ok(())
}

fn validate_source_node(source: &crate::source::SourceNode) -> Result<BTreeSet<String>> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            let mut available = BTreeSet::new();
            available.extend(source.read_columns.iter().cloned());
            available.extend(source.partition_columns.iter().map(|column| column.name.clone()));
            available.extend(source.missing_columns.iter().map(|column| column.name.clone()));
            if source.is_zero_column_count_source() {
                available.insert(ZERO_COLUMN_COUNT_COLUMN.to_string());
            }
            Ok(available)
        },
        crate::source::SourceNode::GpuResident(source) => {
            if source.input_name.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.gpu_resident_source.validate",
                    "GPU-resident source input name must not be empty",
                )
                .with_fact("reason", "empty_input_name"));
            }
            Ok(schema_column_names(&source.schema))
        },
    }
}

fn input_node(plan: &QueryPlan, input: NodeId, consumer: NodeId) -> Result<&QueryNode> {
    plan.node(input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate",
            "query plan node references a missing input",
        )
        .with_fact("consumer", consumer)
        .with_fact("input", input)
    })
}

fn available_columns(columns_by_node: &[BTreeSet<String>], input: NodeId) -> Result<&BTreeSet<String>> {
    columns_by_node.get(input.index()).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate",
            "query plan node references an input that has not been validated",
        )
        .with_fact("input", input)
    })
}

fn validate_schema_supported(schema: &SchemaRef, operation: &'static str) -> Result<()> {
    for field in schema.fields() {
        if !is_supported_arrow_type(field.data_type()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                operation,
                "schema field type is not supported",
            )
            .with_fact("field", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    Ok(())
}

fn validate_output_names_unique(node: &QueryNode) -> Result<()> {
    let output_names = node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    for (duplicate_output_index, name) in output_names.iter().enumerate() {
        if let Some(first_output_index) = output_names[..duplicate_output_index].iter().position(|seen| seen == name) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.plan.validate_output_names",
                "query plan node output schema contains duplicate executable column names",
            )
            .with_fact("node_id", node.id)
            .with_fact("node_kind", plan_node_kind(&node.kind))
            .with_fact("schema_behavior", node.facts.schema_behavior)
            .with_fact("duplicate_output_name", *name)
            .with_fact("first_output_index", first_output_index)
            .with_fact("duplicate_output_index", duplicate_output_index)
            .with_fact("output_names", output_names.join(",")));
        }
    }
    Ok(())
}

fn plan_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
    }
}

fn validate_union_node(
    plan: &QueryPlan,
    union: &UnionNode,
    node_id: NodeId,
    output_schema: &SchemaRef,
    available_columns_by_node: &[BTreeSet<String>],
) -> Result<()> {
    if union.inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.union.validate",
            "union node requires at least one input",
        )
        .with_fact("node_id", node_id));
    }
    for (input_index, input) in union.inputs.iter().enumerate() {
        let input_node = input_node(plan, *input, node_id)?;
        let input_available = available_columns(available_columns_by_node, *input)?;
        if input_node.output_schema().fields().len() != output_schema.fields().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.union.validate",
                "union input schema field count must match the union output schema",
            )
            .with_fact("node_id", node_id)
            .with_fact("input_index", input_index)
            .with_fact("input_node", *input)
            .with_fact("input_fields", input_node.output_schema().fields().len())
            .with_fact("output_fields", output_schema.fields().len()));
        }
        for (field_index, (input_field, output_field)) in
            input_node.output_schema().fields().iter().zip(output_schema.fields().iter()).enumerate()
        {
            if !input_available.contains(input_field.name()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.union.validate",
                    "union input schema references a column not produced by the input node",
                )
                .with_fact("node_id", node_id)
                .with_fact("input_node", *input)
                .with_fact("column", input_field.name())
                .with_fact("available_columns", comma_separated(input_available)));
            }
            if input_field.as_ref() != output_field.as_ref() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.union.validate",
                    "union input field must match the union output field by ordinal",
                )
                .with_fact("node_id", node_id)
                .with_fact("input_index", input_index)
                .with_fact("input_node", *input)
                .with_fact("field_index", field_index)
                .with_fact("input_field", input_field.name())
                .with_fact("output_field", output_field.name())
                .with_fact("input_type", format!("{:?}", input_field.data_type()))
                .with_fact("output_type", format!("{:?}", output_field.data_type()))
                .with_fact("input_nullable", input_field.is_nullable())
                .with_fact("output_nullable", output_field.is_nullable()));
            }
        }
    }
    Ok(())
}

fn validate_window_capability(
    window: &crate::plan::WindowNode,
    input_schema: &SchemaRef,
    output_schema: &SchemaRef,
) -> Result<()> {
    if window.spec.functions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.window.validate",
            "window node requires at least one function",
        ));
    }
    if output_schema.fields().len() != input_schema.fields().len() + window.spec.functions.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.window.validate",
            "window output schema must append one field per window function",
        )
        .with_fact("input_fields", input_schema.fields().len())
        .with_fact("function_count", window.spec.functions.len())
        .with_fact("output_fields", output_schema.fields().len()));
    }
    for (index, (input_field, output_field)) in input_schema.fields().iter().zip(output_schema.fields()).enumerate() {
        if input_field.as_ref() != output_field.as_ref() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.window.validate",
                "window output schema must preserve input fields before appended window outputs",
            )
            .with_fact("field_index", index)
            .with_fact("input_field", input_field.name())
            .with_fact("output_field", output_field.name()));
        }
    }
    for (index, function) in window.spec.functions.iter().enumerate() {
        let field = output_schema.fields().get(input_schema.fields().len() + index).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.window.validate",
                "window output field is missing",
            )
            .with_fact("function_index", index)
        })?;
        if field.name() != function.alias() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.window.validate",
                "window output field name must match function alias",
            )
            .with_fact("function_index", index)
            .with_fact("alias", function.alias())
            .with_fact("field", field.name()));
        }
        match function {
            crate::plan::WindowFunction::Aggregate { function, frame, .. } => {
                match frame {
                    crate::plan::WindowFrame::WholePartition => {
                        if !window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "cudf_query_engine.window.validate",
                                "native whole-partition aggregate windows require no ORDER BY",
                            ));
                        }
                    },
                    crate::plan::WindowFrame::RowsUnboundedPrecedingToCurrent => {
                        if window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "cudf_query_engine.window.validate",
                                "native cumulative aggregate windows require ORDER BY",
                            ));
                        }
                        if !matches!(
                            function,
                            crate::expr::AggregateFunction::Sum
                                | crate::expr::AggregateFunction::Min
                                | crate::expr::AggregateFunction::Max
                        ) {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "cudf_query_engine.window.validate",
                                "native cumulative aggregate window function is unsupported",
                            )
                            .with_fact("function", format!("{function:?}")));
                        }
                    },
                }
                for expr in &window.spec.partition_by {
                    lower_column_expr(expr, "cudf_query_engine.window.validate_partition")?;
                }
            },
            crate::plan::WindowFunction::Rank { .. } => {
                if window.spec.order_by.is_empty() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "cudf_query_engine.window.validate",
                        "native rank windows require ORDER BY",
                    ));
                }
                if !matches!(field.data_type(), arrow::datatypes::DataType::UInt64) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "cudf_query_engine.window.validate",
                        "native rank output must be UInt64",
                    )
                    .with_fact("field", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
                }
            },
        }
    }
    Ok(())
}

fn schema_column_names(schema: &SchemaRef) -> BTreeSet<String> {
    schema.fields().iter().map(|field| field.name().clone()).collect()
}

fn filter_available_columns(available: &BTreeSet<String>, output_schema: &SchemaRef) -> BTreeSet<String> {
    output_schema
        .fields()
        .iter()
        .filter(|field| available.contains(field.name()))
        .map(|field| field.name().clone())
        .collect()
}
