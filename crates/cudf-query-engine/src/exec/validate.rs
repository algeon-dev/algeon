use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef, TimeUnit};

use super::*;
use crate::{
    datatypes::{is_supported_arrow_cast, is_supported_arrow_type},
    expr::{CaseWhen, ColumnQualifier, ColumnRef},
    plan::{JoinNode, JoinOutputColumn, JoinSide, JoinType, QueryNode},
};

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
                for expr in &projection.expressions {
                    validate_expr(expr, input.output_schema(), input_available)?;
                }
                validate_projection_capability(projection, node.output_schema())?;
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
        };
        available_columns_by_node.push(available_columns);
    }

    let sink_input = input_node(plan, plan.sink().input(), NodeId::from_index_for_internal_use(plan.node_count()))?;
    let sink_available = available_columns(&available_columns_by_node, plan.sink().input())?;
    if let ResultSink::HostArrow { schema, .. } = plan.sink() {
        validate_schema_supported(schema, "cudf_query_engine.plan.validate_sink_schema")?;
        for field in schema.fields() {
            if !sink_available.contains(field.name()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.plan.validate_sink",
                    "host sink schema references a column not produced by the root node",
                )
                .with_fact("column", field.name())
                .with_fact("root_node", sink_input.id)
                .with_fact("available_columns", comma_separated(sink_available)));
            }
        }
    }

    Ok(())
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
            if source.is_zero_column_count_source() {
                available.insert(ZERO_COLUMN_COUNT_COLUMN.to_string());
            }
            Ok(available)
        },
    }
}

fn validate_join_capability(
    join: &JoinNode,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    if join.spec.join_type != JoinType::Inner {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native join execution currently supports only inner equi joins",
        )
        .with_fact("join_type", join.spec.join_type));
    }
    if join.spec.keys.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "native join requires at least one equality key",
        ));
    }
    if join.spec.residual_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native join execution does not yet support residual predicates",
        )
        .with_fact("join_type", join.spec.join_type));
    }

    for key in &join.spec.keys {
        validate_join_column_ref(&key.left, JoinSide::Left)?;
        validate_join_column_ref(&key.right, JoinSide::Right)?;
        let left_type = validate_column(key.left.name(), left_schema, left_available)?;
        let right_type = validate_column(key.right.name(), right_schema, right_available)?;
        require_comparable_types(&left_type, &right_type)?;
    }

    validate_join_output_projection(
        &join.spec.output_projection,
        left_schema,
        left_available,
        right_schema,
        right_available,
        output_schema,
    )
}

fn validate_join_column_ref(column: &ColumnRef, side: JoinSide) -> Result<()> {
    let expected_input = match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    };
    match column.qualifier() {
        Some(ColumnQualifier::Input { index }) if *index == expected_input => Ok(()),
        Some(ColumnQualifier::Input { index }) => Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "join column qualifier references the wrong input side",
        )
        .with_fact("column", column.name())
        .with_fact("side", side)
        .with_fact("qualifier_input", *index)
        .with_fact("expected_input", expected_input)),
        Some(qualifier) => Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "join execution requires input-qualified column references",
        )
        .with_fact("column", column.name())
        .with_fact("side", side)
        .with_fact("qualifier", qualifier)),
        None => Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "join execution requires input-qualified column references",
        )
        .with_fact("column", column.name())
        .with_fact("side", side)),
    }
}

fn validate_join_output_projection(
    projection: &[JoinOutputColumn],
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    let output_names = projection.iter().map(|column| column.output_name.clone()).collect::<Vec<_>>();
    let schema_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if output_names != schema_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "join output projection names must match output schema fields",
        )
        .with_fact("projection_output_names", output_names.join(","))
        .with_fact("schema_output_names", schema_names.join(",")));
    }

    for (index, output) in projection.iter().enumerate() {
        validate_join_column_ref(&output.column, output.side)?;
        let (schema, available) = match output.side {
            JoinSide::Left => (left_schema, left_available),
            JoinSide::Right => (right_schema, right_available),
        };
        let data_type = validate_column(output.column.name(), schema, available)?;
        let expected = output_schema.field(index).data_type();
        if &data_type != expected {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.join.validate",
                "join output projection type must match output schema field",
            )
            .with_fact("output_name", output.output_name.as_str())
            .with_fact("column", output.column.name())
            .with_fact("side", output.side)
            .with_fact("column_type", format!("{data_type:?}"))
            .with_fact("output_type", format!("{expected:?}")));
        }
    }
    Ok(())
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

fn validate_expr(expr: &Expr, input_schema: &SchemaRef, available_columns: &BTreeSet<String>) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => validate_column(column.name(), input_schema, available_columns),
        Expr::Literal(value) => validate_literal(value),
        Expr::Not(expr) => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_boolean(&data_type, "NOT argument")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::IsNull(expr) | Expr::IsNotNull(expr) => {
            validate_expr(expr, input_schema, available_columns)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Cast { expr, target_type } => {
            let source_type = validate_expr(expr, input_schema, available_columns)?;
            if !is_supported_arrow_cast(&source_type, target_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.plan.validate_expr",
                    "native cast source and target types are not supported",
                )
                .with_fact("source_type", format!("{source_type:?}"))
                .with_fact("target_type", format!("{target_type:?}")));
            }
            Ok(target_type.clone())
        },
        Expr::Binary { left, op, right } => validate_binary_expr(left, *op, right, input_schema, available_columns),
        Expr::Like { expr, escape, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "LIKE input")?;
            if let Some(escape) = escape {
                validate_like_escape_contract(escape)?;
            }
            Ok(ArrowDataType::Boolean)
        },
        Expr::Substring { expr, start, length } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_string(&data_type, "SUBSTRING input")?;
            validate_substring_bounds(*start, *length)?;
            Ok(ArrowDataType::Utf8)
        },
        Expr::InList { expr, list, .. } => validate_in_list_expr(expr, list, input_schema, available_columns),
        Expr::Case { branches, else_expr } => validate_case_expr(branches, else_expr, input_schema, available_columns),
        Expr::Extract { expr, .. } => {
            let data_type = validate_expr(expr, input_schema, available_columns)?;
            require_date_or_timestamp(&data_type, "EXTRACT input")?;
            Ok(ArrowDataType::Int32)
        },
        Expr::Alias { expr, .. } => validate_expr(expr, input_schema, available_columns),
    }
}

fn validate_column(
    name: &str,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if !available_columns.contains(name) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate_expr",
            "native expression references a column that is not produced by its input",
        )
        .with_fact("column", name)
        .with_fact("available_columns", comma_separated(available_columns)));
    }
    let field = input_schema.field_with_name(name).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.plan.validate_expr",
            "native expression references a column missing from its input schema",
        )
        .with_fact("column", name)
        .with_fact("arrow_error", error.to_string())
    })?;
    if !is_supported_arrow_type(field.data_type()) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native expression references an unsupported column type",
        )
        .with_fact("column", name)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(field.data_type().clone())
}

fn validate_literal(value: &LiteralValue) -> Result<ArrowDataType> {
    if matches!(value, LiteralValue::IntervalMonthDayNano { .. }) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "interval literals are supported only inside datetime interval arithmetic",
        )
        .with_fact("literal", value));
    }
    let data_type = value.data_type();
    if !is_supported_arrow_type(&data_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "literal type is not supported",
        )
        .with_fact("literal", value)
        .with_fact("data_type", format!("{data_type:?}")));
    }
    Ok(data_type)
}

fn validate_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    let left_interval = interval_literal(left);
    let right_interval = interval_literal(right);
    if left_interval.is_some() || right_interval.is_some() {
        return validate_interval_binary_expr(
            left,
            op,
            right,
            left_interval,
            right_interval,
            input_schema,
            available_columns,
        );
    }

    let left_type = validate_expr(left, input_schema, available_columns)?;
    let right_type = validate_expr(right, input_schema, available_columns)?;
    if let Some(output_type) = temporal_duration_binary_output_type(op, &left_type, &right_type)? {
        return Ok(output_type);
    }
    match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
            require_arithmetic_type(&left_type, "binary arithmetic left operand")?;
            require_arithmetic_type(&right_type, "binary arithmetic right operand")?;
            require_compatible_types(&left_type, &right_type, "binary arithmetic operands")?;
            Ok(left_type)
        },
        BinaryOp::Modulo => {
            require_integral_type(&left_type, "modulo left operand")?;
            require_integral_type(&right_type, "modulo right operand")?;
            require_compatible_types(&left_type, &right_type, "modulo operands")?;
            Ok(left_type)
        },
        BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
            require_comparable_types(&left_type, &right_type)?;
            Ok(ArrowDataType::Boolean)
        },
        BinaryOp::And | BinaryOp::Or => {
            require_boolean(&left_type, "boolean binary left operand")?;
            require_boolean(&right_type, "boolean binary right operand")?;
            Ok(ArrowDataType::Boolean)
        },
    }
}

fn validate_interval_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    left_interval: Option<(i32, i32, i64)>,
    right_interval: Option<(i32, i32, i64)>,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if left_interval.is_some() == right_interval.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native interval arithmetic requires exactly one interval literal operand",
        ));
    }
    let operand = if left_interval.is_some() {
        if op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        right
    } else {
        if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native interval arithmetic supports only datetime plus/minus interval",
            ));
        }
        right_interval.expect("right interval was checked above");
        left
    };
    let operand_type = validate_expr(operand, input_schema, available_columns)?;
    require_date_or_timestamp(&operand_type, "interval arithmetic datetime operand")?;
    Ok(operand_type)
}

fn validate_in_list_expr(
    expr: &Expr,
    list: &[Expr],
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if list.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native IN list must not be empty",
        ));
    }
    let expr_type = validate_expr(expr, input_schema, available_columns)?;
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native IN list values must be literals",
            ));
        };
        if matches!(value, LiteralValue::Null(_)) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.plan.validate_expr",
                "native IN list does not yet model SQL null-containing list semantics",
            ));
        }
        let item_type = validate_literal(value)?;
        require_compatible_types(&expr_type, &item_type, "IN list value")?;
    }
    Ok(ArrowDataType::Boolean)
}

fn validate_case_expr(
    branches: &[CaseWhen],
    else_expr: &Expr,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    if branches.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native CASE requires at least one WHEN branch",
        ));
    }
    let else_type = validate_expr(else_expr, input_schema, available_columns)?;
    for branch in branches {
        let when_type = validate_expr(&branch.when, input_schema, available_columns)?;
        require_boolean(&when_type, "CASE WHEN predicate")?;
        let then_type = validate_expr(&branch.then, input_schema, available_columns)?;
        require_compatible_types(&else_type, &then_type, "CASE branch result")?;
    }
    Ok(else_type)
}

fn interval_literal(expr: &Expr) -> Option<(i32, i32, i64)> {
    match expr {
        Expr::Literal(LiteralValue::IntervalMonthDayNano { months, days, nanoseconds }) => {
            Some((*months, *days, *nanoseconds))
        },
        _ => None,
    }
}

fn validate_substring_bounds(start: i32, length: Option<i32>) -> Result<()> {
    if start < 1 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native substring requires a one-based positive start position",
        )
        .with_fact("start", start));
    }
    if let Some(length) = length
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native substring requires a non-negative in-range length",
        )
        .with_fact("start", start)
        .with_fact("length", length));
    }
    Ok(())
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn validate_like_escape_contract(escape: &str) -> Result<()> {
    if escape.len() <= 1 && escape.is_ascii() {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.plan.validate_expr",
            "native LIKE escape must be empty or one ASCII byte",
        )
        .with_fact("escape", escape))
    }
}

fn require_boolean(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Boolean) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected boolean")
    }
}

fn require_string(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected string")
    }
}

fn require_date_or_timestamp(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if matches!(data_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None)) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected date or timezone-free timestamp")
    }
}

fn require_integral_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_integral_type(data_type) { Ok(()) } else { unsupported_type(context, data_type, "expected integral") }
}

fn require_arithmetic_type(data_type: &ArrowDataType, context: &'static str) -> Result<()> {
    if is_numeric_type(data_type) || is_decimal_type(data_type) {
        Ok(())
    } else {
        unsupported_type(context, data_type, "expected numeric or decimal")
    }
}

fn require_compatible_types(left: &ArrowDataType, right: &ArrowDataType, context: &'static str) -> Result<()> {
    if left == right || is_supported_arrow_cast(right, left) || is_supported_arrow_cast(left, right) {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native expression operand types are not compatible",
        )
        .with_fact("context", context)
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

fn require_comparable_types(left: &ArrowDataType, right: &ArrowDataType) -> Result<()> {
    if left == right
        || ((is_numeric_type(left) || is_decimal_type(left)) && (is_numeric_type(right) || is_decimal_type(right)))
        || (is_string_type(left) && is_string_type(right))
        || (is_temporal_type(left) && is_temporal_type(right))
    {
        Ok(())
    } else {
        Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native comparison operand types are not compatible",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}")))
    }
}

fn temporal_duration_binary_output_type(
    op: BinaryOp,
    left: &ArrowDataType,
    right: &ArrowDataType,
) -> Result<Option<ArrowDataType>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Ok(None);
    }
    let left_temporal = is_date_or_timestamp_type(left);
    let right_temporal = is_date_or_timestamp_type(right);
    let left_duration = is_duration_type(left);
    let right_duration = is_duration_type(right);

    match (op, left_temporal, right_temporal, left_duration, right_duration) {
        (BinaryOp::Add, true, false, false, true) | (BinaryOp::Subtract, true, false, false, true) => {
            Ok(Some(left.clone()))
        },
        (BinaryOp::Add, false, true, true, false) => Ok(Some(right.clone())),
        (_, true | false, true | false, true, true) | (_, true, true, false, false) => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.plan.validate_expr",
            "native temporal arithmetic supports only datetime plus/minus duration",
        )
        .with_fact("left_type", format!("{left:?}"))
        .with_fact("right_type", format!("{right:?}"))),
        _ => Ok(None),
    }
}

fn unsupported_type(context: &'static str, data_type: &ArrowDataType, expected: &'static str) -> Result<()> {
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "cudf_query_engine.plan.validate_expr",
        "native expression type does not satisfy its operator contract",
    )
    .with_fact("context", context)
    .with_fact("expected", expected)
    .with_fact("data_type", format!("{data_type:?}")))
}

fn is_integral_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
    )
}

fn is_numeric_type(data_type: &ArrowDataType) -> bool {
    is_integral_type(data_type) || matches!(data_type, ArrowDataType::Float32 | ArrowDataType::Float64)
}

fn is_decimal_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Decimal128(_, _))
}

fn is_string_type(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn is_temporal_type(data_type: &ArrowDataType) -> bool {
    is_date_or_timestamp_type(data_type) || is_duration_type(data_type)
}

fn is_date_or_timestamp_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Date32
            | ArrowDataType::Timestamp(
                TimeUnit::Second | TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond,
                None
            )
    )
}

fn is_duration_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Duration(
            TimeUnit::Second | TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond
        )
    )
}

fn comma_separated(columns: &BTreeSet<String>) -> String {
    columns.iter().cloned().collect::<Vec<_>>().join(",")
}
