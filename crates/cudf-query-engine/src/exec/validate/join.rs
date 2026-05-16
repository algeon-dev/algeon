use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};

use super::{
    expressions::{validate_column, validate_literal},
    types::{
        is_decimal_type, is_numeric_type, require_arithmetic_type, require_boolean, require_comparable_types,
        require_compatible_types, require_integral_type,
    },
    *,
};
use crate::{
    datatypes::is_supported_arrow_cast,
    expr::{ColumnQualifier, ColumnRef},
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
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross
    ) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.validate",
            "native join execution currently supports inner, left, full, left semi, left anti, and cross joins",
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
    if join.spec.join_type != JoinType::Cross && join.spec.keys.is_empty() && join.spec.residual_predicate.is_none() {
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
            validate_join_residual_expr(predicate, left_schema, left_available, right_schema, right_available)?;
        require_boolean(&predicate_type, "join residual predicate")?;
        validate_join_residual_column_sides(predicate)?;
    }

    validate_join_type_projection(join)?;
    validate_join_output_projection(
        &join.spec.output_projection,
        left_schema,
        left_available,
        right_schema,
        right_available,
        output_schema,
    )
}

fn validate_join_type_projection(join: &JoinNode) -> Result<()> {
    if !matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti) {
        return Ok(());
    }

    if let Some(output) = join.spec.output_projection.iter().find(|column| column.side == JoinSide::Right) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.join.validate",
            "left semi and left anti joins can only project left-side output columns",
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
        | Expr::Alias { expr: inner, .. } => collect_join_residual_sides(inner, left, right),
        Expr::Binary { left: lhs, right: rhs, .. } => {
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
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => Err(QueryError::new(
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
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.join.validate_residual",
            "native join residual predicates do not yet support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn is_supported_join_residual_ast_cast(source_type: &ArrowDataType, target_type: &ArrowDataType) -> bool {
    matches!(target_type, ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64)
        && (is_numeric_type(source_type) || is_decimal_type(source_type))
        && is_supported_arrow_cast(source_type, target_type)
}
