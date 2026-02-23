use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};

use super::{
    expressions::{validate_column, validate_literal},
    types::{
        is_decimal_type, is_numeric_type, require_arithmetic_type, require_boolean, require_comparable_types,
        require_compatible_types, require_date_or_timestamp, require_integral_type, require_string,
        require_timestamp_second,
    },
    *,
};
use crate::{
    datatypes::is_supported_arrow_cast,
    expr::{BinaryOp, ColumnQualifier, ColumnRef, DatePart, LiteralValue},
    plan::{JoinNode, JoinOutputColumn, JoinSide, JoinType},
};

pub(super) fn validate_join_capability(
    join: &JoinNode,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    if !matches!(
        join.spec.join_type,
        JoinType::Inner
            | JoinType::Left
            | JoinType::Full
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::Cross
            | JoinType::LeftMark
    ) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native join execution currently supports inner, left, full, left semi, left anti, left mark, and cross joins",
        )
        .with_fact("join_type", join.spec.join_type));
    }
    if join.spec.join_type == JoinType::Cross && !join.spec.keys.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "native cross join must not carry equality keys",
        )
        .with_fact("key_count", join.spec.keys.len()));
    }
    if join.spec.join_type != JoinType::Cross
        && !keyless_predicate_free_outer_join(join)
        && join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "native join requires equality keys or a residual predicate",
        ));
    }
    if join.spec.join_type == JoinType::Cross && join.spec.residual_predicate.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native cross join execution does not support residual predicates; use conditional join IR instead",
        )
        .with_fact("join_type", join.spec.join_type));
    }
    if join.spec.join_type == JoinType::Left && join.spec.build_side != JoinSide::Right {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native reusable hash left join execution requires the right side to be the build side",
        )
        .with_fact("join_type", join.spec.join_type)
        .with_fact("build_side", join.spec.build_side));
    }
    if join.spec.join_type == JoinType::LeftMark && join.spec.build_side != JoinSide::Right {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native left mark join execution requires the right side to be the build side",
        )
        .with_fact("join_type", join.spec.join_type)
        .with_fact("build_side", join.spec.build_side));
    }
    if join.spec.join_type == JoinType::LeftMark && join.spec.mark_column_name.is_none() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "native left mark join requires a mark_column_name in the join spec",
        )
        .with_fact("join_type", join.spec.join_type));
    }
    if join.spec.join_type != JoinType::LeftMark && join.spec.mark_column_name.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "mark_column_name is only valid for left mark joins",
        )
        .with_fact("join_type", join.spec.join_type)
        .with_fact("mark_column_name", join.spec.mark_column_name.as_deref().unwrap_or("")));
    }
    if let Some(build_artifact_id) = &join.spec.build_artifact_id {
        if build_artifact_id.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.join.validate",
                "native reusable hash-join build artifact id must not be empty",
            )
            .with_fact("reason", "empty_build_artifact_id"));
        }
        if join.spec.keys.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "cudf_query_engine.join.validate",
                "native reusable hash-join build artifacts require equality keys",
            )
            .with_fact("build_artifact_id", build_artifact_id)
            .with_fact("join_type", join.spec.join_type));
        }
        if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "cudf_query_engine.join.validate",
                "native durable hash-join build artifacts are supported only for reusable hash-join probe types",
            )
            .with_fact("build_artifact_id", build_artifact_id)
            .with_fact("join_type", join.spec.join_type));
        }
    }

    for key in &join.spec.keys {
        validate_join_column_ref(&key.left, JoinSide::Left)?;
        validate_join_column_ref(&key.right, JoinSide::Right)?;
        let left_type = validate_column(key.left.name(), left_schema, left_available)?;
        let right_type = validate_column(key.right.name(), right_schema, right_available)?;
        require_comparable_types(&left_type, &right_type)?;
    }
    if let Some(predicate) = &join.spec.residual_predicate {
        let predicate_type =
            match validate_join_residual_expr(predicate, left_schema, left_available, right_schema, right_available) {
                Ok(data_type) => data_type,
                Err(_error) if join_residual_post_filter_supported(join) => validate_post_join_residual_expr(
                    predicate,
                    left_schema,
                    left_available,
                    right_schema,
                    right_available,
                )?,
                Err(error) => return Err(error),
            };
        require_boolean(&predicate_type, "join residual predicate")?;
        validate_join_residual_column_sides(predicate)?;
    }

    validate_join_type_projection(join)?;
    if join.spec.join_type == JoinType::LeftMark {
        validate_mark_join_output_projection(join, left_schema, left_available, output_schema)?;
    } else {
        validate_join_output_projection(
            &join.spec.output_projection,
            left_schema,
            left_available,
            right_schema,
            right_available,
            output_schema,
        )?;
    }
    Ok(())
}

fn keyless_predicate_free_outer_join(join: &JoinNode) -> bool {
    join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && matches!(join.spec.join_type, JoinType::Left | JoinType::Full)
}

fn validate_join_type_projection(join: &JoinNode) -> Result<()> {
    if !matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark) {
        return Ok(());
    }

    if let Some(output) = join.spec.output_projection.iter().find(|column| column.side == JoinSide::Right) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "left semi, left anti, and left mark joins can only project left-side output columns",
        )
        .with_fact("join_type", join.spec.join_type)
        .with_fact("side", output.side)
        .with_fact("column", output.column.name())
        .with_fact("output_name", output.output_name.as_str()));
    }

    Ok(())
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

fn validate_mark_join_output_projection(
    join: &JoinNode,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    let mark_column_name = join.spec.mark_column_name.as_deref().unwrap_or("mark");
    let expected_output_count = join.spec.output_projection.len() + 1;
    if output_schema.fields().len() != expected_output_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "left mark join output schema must have projection columns plus one mark column",
        )
        .with_fact("output_schema_len", output_schema.fields().len())
        .with_fact("expected_output_count", expected_output_count));
    }
    // Validate left-side projection columns.
    for (index, output) in join.spec.output_projection.iter().enumerate() {
        validate_join_column_ref(&output.column, output.side)?;
        let data_type = validate_column(output.column.name(), left_schema, left_available)?;
        let schema_field = output_schema.field(index);
        if output.output_name != *schema_field.name() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.join.validate",
                "left mark join output projection name must match output schema field",
            )
            .with_fact("output_name", output.output_name.as_str())
            .with_fact("schema_name", schema_field.name()));
        }
        if &data_type != schema_field.data_type() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.join.validate",
                "left mark join output projection type must match output schema field",
            )
            .with_fact("output_name", output.output_name.as_str())
            .with_fact("column_type", format!("{data_type:?}"))
            .with_fact("output_type", format!("{:?}", schema_field.data_type())));
        }
    }
    // Validate the mark column at the last position.
    let mark_field = output_schema.field(join.spec.output_projection.len());
    if mark_field.name() != mark_column_name {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "left mark join mark column name must match output schema",
        )
        .with_fact("mark_column_name", mark_column_name)
        .with_fact("schema_name", mark_field.name()));
    }
    if *mark_field.data_type() != ArrowDataType::Boolean {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "left mark join mark column must be Boolean",
        )
        .with_fact("mark_column_type", format!("{:?}", mark_field.data_type())));
    }
    Ok(())
}

fn validate_join_residual_column_sides(expr: &Expr) -> Result<()> {
    let mut left = false;
    let mut right = false;
    collect_join_residual_sides(expr, &mut left, &mut right)?;
    if !left || !right {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "native join residual predicates must reference at least one column from each input",
        )
        .with_fact("expression", expr)
        .with_fact("references_left", left)
        .with_fact("references_right", right));
    }
    Ok(())
}

fn collect_join_residual_sides(expr: &Expr, left: &mut bool, right: &mut bool) -> Result<()> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(ColumnQualifier::Input { index: 0 }) => {
                *left = true;
                Ok(())
            },
            Some(ColumnQualifier::Input { index: 1 }) => {
                *right = true;
                Ok(())
            },
            qualifier => Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "join residual column must be input-qualified",
            )
            .with_fact("column", column.name())
            .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(_) => Ok(()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Abs(inner)
        | Expr::Sqrt { expr: inner, .. }
        | Expr::Round { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => collect_join_residual_sides(inner, left, right),
        Expr::Binary { left: lhs, right: rhs, .. } => {
            collect_join_residual_sides(lhs, left, right)?;
            collect_join_residual_sides(rhs, left, right)
        },
        Expr::IsDistinctFrom { left: lhs, right: rhs } | Expr::IsNotDistinctFrom { left: lhs, right: rhs } => {
            collect_join_residual_sides(lhs, left, right)?;
            collect_join_residual_sides(rhs, left, right)
        },
        Expr::InList { expr, list, .. } => {
            collect_join_residual_sides(expr, left, right)?;
            for item in list {
                collect_join_residual_sides(item, left, right)?;
            }
            Ok(())
        },
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_join_residual_sides(expr, left, right)?;
            }
            Ok(())
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_join_residual_sides(&branch.when, left, right)?;
                collect_join_residual_sides(&branch.then, left, right)?;
            }
            collect_join_residual_sides(else_expr, left, right)
        },
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn validate_join_residual_expr(
    expr: &Expr,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(ColumnQualifier::Input { index: 0 }) => validate_column(column.name(), left_schema, left_available),
            Some(ColumnQualifier::Input { index: 1 }) => validate_column(column.name(), right_schema, right_available),
            qualifier => Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "join residual column must be input-qualified",
            )
            .with_fact("column", column.name())
            .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(value) => validate_literal(value),
        Expr::Not(expr) => {
            let data_type =
                validate_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_boolean(&data_type, "join residual NOT argument")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::IsNull(expr) | Expr::IsNotNull(expr) => {
            validate_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Cast { expr, target_type } => {
            let source_type =
                validate_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            if !is_supported_join_residual_ast_cast(&source_type, target_type) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "native join residual cast is not supported by the cudf AST contract",
                )
                .with_fact("source_type", format!("{source_type:?}"))
                .with_fact("target_type", format!("{target_type:?}")));
            }
            Ok(target_type.clone())
        },
        Expr::Binary { left, op, right, .. } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_join_interval_binary_expr(
                left,
                *op,
                right,
                left_schema,
                left_available,
                right_schema,
                right_available,
            )
        },
        Expr::Binary { left, op, right, .. } => {
            let left_type =
                validate_join_residual_expr(left, left_schema, left_available, right_schema, right_available)?;
            let right_type =
                validate_join_residual_expr(right, left_schema, left_available, right_schema, right_available)?;
            match op {
                BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
                    require_arithmetic_type(&left_type, "join residual arithmetic left operand")?;
                    require_arithmetic_type(&right_type, "join residual arithmetic right operand")?;
                    require_compatible_types(&left_type, &right_type, "join residual arithmetic operands")?;
                    Ok(left_type)
                },
                BinaryOp::Modulo => {
                    require_integral_type(&left_type, "join residual modulo left operand")?;
                    require_integral_type(&right_type, "join residual modulo right operand")?;
                    require_compatible_types(&left_type, &right_type, "join residual modulo operands")?;
                    Ok(left_type)
                },
                BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                    require_integral_type(&left_type, "join residual bitwise left operand")?;
                    require_integral_type(&right_type, "join residual bitwise right operand")?;
                    require_compatible_types(&left_type, &right_type, "join residual bitwise operands")?;
                    Ok(left_type)
                },
                BinaryOp::ShiftLeft | BinaryOp::ShiftRight => Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "native join residual AST does not support bit-shift operators",
                )
                .with_fact("operator", op.to_string())),
                BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                    require_comparable_types(&left_type, &right_type)?;
                    Ok(ArrowDataType::Boolean)
                },
                BinaryOp::And | BinaryOp::Or => {
                    require_boolean(&left_type, "join residual logical left operand")?;
                    require_boolean(&right_type, "join residual logical right operand")?;
                    Ok(ArrowDataType::Boolean)
                },
            }
        },
        Expr::InList { expr, list, .. } => {
            let expr_type =
                validate_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "native join residual IN list must not be empty",
                ));
            }
            for item in list {
                let Expr::Literal(value) = item else {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.validate_residual",
                        "native join residual IN list values must be literals",
                    ));
                };
                if matches!(value, LiteralValue::Null(_)) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.validate_residual",
                        "native join residual IN list does not yet model SQL null-containing list semantics",
                    ));
                }
                let item_type = validate_literal(value)?;
                require_compatible_types(&expr_type, &item_type, "join residual IN list value")?;
            }
            Ok(ArrowDataType::Boolean)
        },
        Expr::Alias { expr, .. } => {
            validate_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)
        },
        Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn validate_post_join_residual_expr(
    expr: &Expr,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(ColumnQualifier::Input { index: 0 }) => validate_column(column.name(), left_schema, left_available),
            Some(ColumnQualifier::Input { index: 1 }) => validate_column(column.name(), right_schema, right_available),
            qualifier => Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "post-join residual column must be input-qualified",
            )
            .with_fact("column", column.name())
            .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string()))),
        },
        Expr::Literal(value) => validate_literal(value),
        Expr::Not(expr) => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_boolean(&data_type, "post-join residual NOT argument")?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::IsNull(expr) | Expr::IsNotNull(expr) => {
            validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::Cast { expr, target_type } => {
            validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            Ok(target_type.clone())
        },
        Expr::Lower(expr) | Expr::Upper(expr) => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_string(&data_type, "post-join residual string case input")?;
            Ok(data_type)
        },
        Expr::StringLength { expr, output_type } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_string(&data_type, "post-join residual string length input")?;
            require_integral_type(output_type, "post-join residual string length output")?;
            Ok(output_type.clone())
        },
        Expr::RegexpReplace { expr, output_type, .. } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_string(&data_type, "post-join residual REGEXP_REPLACE input")?;
            require_string(output_type, "post-join residual REGEXP_REPLACE output")?;
            Ok(output_type.clone())
        },
        Expr::DateTrunc { field, expr, output_type } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            match field {
                DatePart::Minute => {
                    require_timestamp_second(&data_type, "post-join residual DATE_TRUNC minute input")?;
                    require_timestamp_second(output_type, "post-join residual DATE_TRUNC minute output")?;
                    Ok(output_type.clone())
                },
                DatePart::Year | DatePart::Month | DatePart::Day => Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "post-join residual DATE_TRUNC does not support this field",
                )
                .with_fact("field", field.to_string())),
            }
        },
        Expr::Extract { expr, .. } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_date_or_timestamp(&data_type, "post-join residual EXTRACT input")?;
            Ok(ArrowDataType::Int32)
        },
        Expr::Abs(expr) => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_arithmetic_type(&data_type, "post-join residual ABS input")?;
            Ok(data_type)
        },
        Expr::Sqrt { expr, output_type } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_arithmetic_type(&data_type, "post-join residual SQRT input")?;
            require_arithmetic_type(output_type, "post-join residual SQRT output")?;
            Ok(output_type.clone())
        },
        Expr::Round { expr, output_type, .. } => {
            let data_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            require_arithmetic_type(&data_type, "post-join residual ROUND input")?;
            require_arithmetic_type(output_type, "post-join residual ROUND output")?;
            Ok(output_type.clone())
        },
        Expr::Concat(exprs) => {
            if exprs.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "post-join residual CONCAT requires at least one argument",
                ));
            }
            let mut output_type = ArrowDataType::Utf8;
            for expr in exprs {
                let data_type =
                    validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
                require_string(&data_type, "post-join residual CONCAT input")?;
                if data_type == ArrowDataType::Utf8View {
                    output_type = ArrowDataType::Utf8View;
                } else if data_type == ArrowDataType::LargeUtf8 && output_type != ArrowDataType::Utf8View {
                    output_type = ArrowDataType::LargeUtf8;
                }
            }
            Ok(output_type)
        },
        Expr::Binary { left, op, right, .. } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_post_join_interval_binary_expr(
                left,
                *op,
                right,
                left_schema,
                left_available,
                right_schema,
                right_available,
            )
        },
        Expr::Binary { left, op, right, output_type } => {
            let left_type =
                validate_post_join_residual_expr(left, left_schema, left_available, right_schema, right_available)?;
            let right_type =
                validate_post_join_residual_expr(right, left_schema, left_available, right_schema, right_available)?;
            match op {
                BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide => {
                    require_arithmetic_type(&left_type, "post-join residual arithmetic left operand")?;
                    require_arithmetic_type(&right_type, "post-join residual arithmetic right operand")?;
                    require_compatible_types(&left_type, &right_type, "post-join residual arithmetic operands")?;
                    Ok(output_type.clone().unwrap_or(left_type))
                },
                BinaryOp::Modulo => {
                    require_integral_type(&left_type, "post-join residual modulo left operand")?;
                    require_integral_type(&right_type, "post-join residual modulo right operand")?;
                    require_compatible_types(&left_type, &right_type, "post-join residual modulo operands")?;
                    Ok(left_type)
                },
                BinaryOp::BitwiseAnd
                | BinaryOp::BitwiseOr
                | BinaryOp::BitwiseXor
                | BinaryOp::ShiftLeft
                | BinaryOp::ShiftRight => {
                    require_integral_type(&left_type, "post-join residual bitwise left operand")?;
                    require_integral_type(&right_type, "post-join residual bitwise right operand")?;
                    require_compatible_types(&left_type, &right_type, "post-join residual bitwise operands")?;
                    Ok(left_type)
                },
                BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                    require_comparable_types(&left_type, &right_type)?;
                    Ok(ArrowDataType::Boolean)
                },
                BinaryOp::And | BinaryOp::Or => {
                    require_boolean(&left_type, "post-join residual logical left operand")?;
                    require_boolean(&right_type, "post-join residual logical right operand")?;
                    Ok(ArrowDataType::Boolean)
                },
            }
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            let left_type =
                validate_post_join_residual_expr(left, left_schema, left_available, right_schema, right_available)?;
            let right_type =
                validate_post_join_residual_expr(right, left_schema, left_available, right_schema, right_available)?;
            require_comparable_types(&left_type, &right_type)?;
            Ok(ArrowDataType::Boolean)
        },
        Expr::InList { expr, list, .. } => {
            let expr_type =
                validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)?;
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "post-join residual IN list must not be empty",
                ));
            }
            for item in list {
                let Expr::Literal(value) = item else {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.validate_residual",
                        "post-join residual IN list values must be literals",
                    ));
                };
                if matches!(value, LiteralValue::Null(_)) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.join.validate_residual",
                        "post-join residual IN list does not yet model SQL null-containing list semantics",
                    ));
                }
                let item_type = validate_literal(value)?;
                require_compatible_types(&expr_type, &item_type, "post-join residual IN list value")?;
            }
            Ok(ArrowDataType::Boolean)
        },
        Expr::Case { branches, else_expr } => {
            if branches.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.join.validate_residual",
                    "post-join residual CASE requires at least one WHEN branch",
                ));
            }
            let else_type = validate_post_join_residual_expr(
                else_expr,
                left_schema,
                left_available,
                right_schema,
                right_available,
            )?;
            for branch in branches {
                let when_type = validate_post_join_residual_expr(
                    &branch.when,
                    left_schema,
                    left_available,
                    right_schema,
                    right_available,
                )?;
                require_boolean(&when_type, "post-join residual CASE WHEN predicate")?;
                let then_type = validate_post_join_residual_expr(
                    &branch.then,
                    left_schema,
                    left_available,
                    right_schema,
                    right_available,
                )?;
                require_compatible_types(&else_type, &then_type, "post-join residual CASE branch result")?;
            }
            Ok(else_type)
        },
        Expr::Alias { expr, .. } => {
            validate_post_join_residual_expr(expr, left_schema, left_available, right_schema, right_available)
        },
        Expr::TryCast { .. } | Expr::Like { .. } | Expr::Substring { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "post-join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn join_residual_post_filter_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && !join.spec.output_projection.is_empty()
        && join.spec.residual_predicate.as_ref().is_some_and(residual_expr_requires_post_filter)
}

fn residual_expr_requires_post_filter(expr: &Expr) -> bool {
    match expr {
        Expr::Case { .. } => true,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Alias { expr, .. } => residual_expr_requires_post_filter(expr),
        Expr::Concat(exprs) => exprs.iter().any(residual_expr_requires_post_filter),
        Expr::Like { expr, .. } | Expr::Substring { expr, .. } => residual_expr_requires_post_filter(expr),
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => true,
        Expr::Binary { op: BinaryOp::Or, .. } => true,
        Expr::Binary { left, right, .. } => {
            residual_expr_requires_post_filter(left) || residual_expr_requires_post_filter(right)
        },
        Expr::InList { expr, list, .. } => {
            residual_expr_requires_post_filter(expr) || list.iter().any(residual_expr_requires_post_filter)
        },
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

fn validate_join_interval_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "join residual interval arithmetic requires exactly one interval literal operand",
        ));
    }
    let interval = if left_interval { left } else { right };
    validate_fixed_day_interval_literal(interval)?;
    let operand = if left_interval {
        if op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "join residual interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        right
    } else {
        if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "join residual interval arithmetic supports only datetime plus/minus interval",
            ));
        }
        left
    };
    let operand_type =
        validate_join_residual_expr(operand, left_schema, left_available, right_schema, right_available)?;
    if operand_type != ArrowDataType::Date32 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "join residual AST interval arithmetic currently supports only Date32 fixed-day operands",
        )
        .with_fact("operand_type", format!("{operand_type:?}")));
    }
    Ok(operand_type)
}

fn validate_post_join_interval_binary_expr(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    left_schema: &SchemaRef,
    left_available: &BTreeSet<String>,
    right_schema: &SchemaRef,
    right_available: &BTreeSet<String>,
) -> Result<ArrowDataType> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "post-join residual interval arithmetic requires exactly one interval literal operand",
        ));
    }
    let operand = if left_interval {
        if op != BinaryOp::Add {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "post-join residual interval arithmetic supports interval + datetime but not interval minus datetime",
            ));
        }
        right
    } else {
        if !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.join.validate_residual",
                "post-join residual interval arithmetic supports only datetime plus/minus interval",
            ));
        }
        right_interval.then_some(()).expect("right interval was checked above");
        left
    };
    let operand_type =
        validate_post_join_residual_expr(operand, left_schema, left_available, right_schema, right_available)?;
    require_date_or_timestamp(&operand_type, "post-join residual interval datetime operand")?;
    Ok(operand_type)
}

fn expr_is_interval_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }))
}

fn validate_fixed_day_interval_literal(expr: &Expr) -> Result<()> {
    let Expr::Literal(LiteralValue::IntervalMonthDayNano { months, nanoseconds, .. }) = expr else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "join residual interval arithmetic requires an interval literal",
        ));
    };
    const NANOS_PER_DAY: i64 = 86_400_000_000_000;
    if *months != 0 || *nanoseconds % NANOS_PER_DAY != 0 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "join residual AST interval arithmetic currently supports only fixed day-resolution intervals",
        )
        .with_fact("months", *months)
        .with_fact("nanoseconds", *nanoseconds));
    }
    Ok(())
}

fn is_supported_join_residual_ast_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(target_type, ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64)
        && (is_numeric_type(source_type) || is_decimal_type(source_type))
        && is_supported_arrow_cast(source_type, target_type)
}
