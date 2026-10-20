use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{Field, SchemaRef};
use datafusion::{
    common::JoinSide as DataFusionJoinSide,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
        },
    },
    physical_plan::joins::utils::JoinFilter,
};
use query_planning::expr::{
    BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen, ColumnRef as NativeColumnRef, Expr as NativeExpr,
    LiteralValue as NativeLiteralValue,
};

use super::{
    LoweredNativeInput,
    expr::{
        alias_if_needed, collect_required_field_indices, lower_binary_operator, lower_literal,
        lower_validated_join_residual_expr, physical_expr_field_from_validated,
    },
    join::CanonicalJoin,
    naming::unique_name_with_suffix,
};
use crate::{
    capability::CapabilityReason,
    native::{
        LoweringFailure,
        expression_contract::{ValidatedJoinResidual, case_expr_data_type},
    },
};

/// Residual predicate lowered for a native join, plus the input columns and
/// pre-join synthetic expressions needed to evaluate it.
pub(super) struct LoweredJoinResidual {
    pub(super) predicate: NativeExpr,
    pub(super) left_required: BTreeSet<usize>,
    pub(super) right_required: BTreeSet<usize>,
    pub(super) materializations: JoinResidualMaterializations,
}

/// Side-partitioned materializations. They are keyed by original DataFusion
/// side because projections are inserted before canonical join input ordering.
#[derive(Default)]
pub(super) struct JoinResidualMaterializations {
    left: Vec<JoinResidualMaterialization>,
    right: Vec<JoinResidualMaterialization>,
}

impl JoinResidualMaterializations {
    fn side_mut(&mut self, side: DataFusionJoinSide) -> Result<&mut Vec<JoinResidualMaterialization>, LoweringFailure> {
        match side {
            DataFusionJoinSide::Left => Ok(&mut self.left),
            DataFusionJoinSide::Right => Ok(&mut self.right),
            DataFusionJoinSide::None => Err(CapabilityReason::UnsupportedExpression.into()),
        }
    }

    pub(super) fn side(&self, side: DataFusionJoinSide) -> Result<&[JoinResidualMaterialization], LoweringFailure> {
        match side {
            DataFusionJoinSide::Left => Ok(&self.left),
            DataFusionJoinSide::Right => Ok(&self.right),
            DataFusionJoinSide::None => Err(CapabilityReason::UnsupportedExpression.into()),
        }
    }
}

/// Synthetic column appended to one join input so the residual predicate can
/// reference an expression that cannot be represented directly in the native
/// join predicate AST.
pub(super) struct JoinResidualMaterialization {
    pub(super) alias: String,
    pub(super) expr: NativeExpr,
    pub(super) field: Field,
}

#[derive(Debug, Clone, Copy)]
struct JoinFilterLoweringContext<'a> {
    contract: &'a ValidatedJoinResidual<'a>,
    filter: &'a JoinFilter,
    left_schema: &'a SchemaRef,
    right_schema: &'a SchemaRef,
    canonical: CanonicalJoin,
    bindings: Option<JoinResidualInputBindings<'a>>,
}

/// Executable join inputs available when residual expressions reference columns.
#[derive(Debug, Clone, Copy)]
pub(super) struct JoinResidualInputBindings<'a> {
    pub(super) left: &'a LoweredNativeInput,
    pub(super) right: &'a LoweredNativeInput,
}

pub(super) fn lower_join_filter(
    contract: &ValidatedJoinResidual<'_>,
    canonical: CanonicalJoin,
    bindings: Option<JoinResidualInputBindings<'_>>,
) -> Result<LoweredJoinResidual, LoweringFailure> {
    let filter = contract.filter();
    debug_assert!(Arc::ptr_eq(contract.expression().expr(), filter.expression()));
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let mut materializations = JoinResidualMaterializations::default();
    let context = JoinFilterLoweringContext {
        contract,
        filter,
        left_schema: contract.left_schema(),
        right_schema: contract.right_schema(),
        canonical,
        bindings,
    };
    // Each materialization site becomes a synthetic column of its input, and
    // the residual reads that column in its place.
    let substitutions = contract
        .materializations()
        .iter()
        .map(|site| {
            let column = materialize_join_residual_expr(
                &site.expr,
                site.side,
                context,
                &mut left_required,
                &mut right_required,
                &mut materializations,
            )?;
            Ok((Arc::clone(&site.expr), column))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let predicate = if contract.contains_decimal_division() {
        // Evaluate checked arithmetic after matching keys. Pre-join
        // materialization would also evaluate rows with no matching key.
        let mut required = BTreeSet::new();
        collect_required_field_indices(filter.expression(), filter.schema(), &mut required)?;
        let columns = required
            .into_iter()
            .map(|index| {
                let field = filter.schema().field(index);
                let expr = lower_join_filter_column(
                    &Column::new(field.name(), index),
                    context,
                    &mut left_required,
                    &mut right_required,
                )?;
                let NativeExpr::Column(column) = expr else { unreachable!("join column lowering returns a column") };
                Ok((index, column))
            })
            .collect::<Result<_, LoweringFailure>>()?;
        lower_validated_join_residual_expr(contract.expression(), &columns, &substitutions)?
    } else {
        lower_join_filter_expr(filter.expression(), context, &substitutions, &mut left_required, &mut right_required)?
    };
    debug_assert!(
        !left_required.is_empty() && !right_required.is_empty(),
        "validated join residual must reference both inputs"
    );
    Ok(LoweredJoinResidual { predicate, left_required, right_required, materializations })
}

/// Lowers the residual operators the cuDF join AST evaluates, or that an
/// inner join evaluates as a post-join filter. Every other single-input
/// subexpression is a contract materialization site and arrives here as a
/// substitution.
fn lower_join_filter_expr(
    expr: &Arc<dyn PhysicalExpr>,
    context: JoinFilterLoweringContext<'_>,
    substitutions: &[(Arc<dyn PhysicalExpr>, NativeExpr)],
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringFailure> {
    if let Some((_, column)) = substitutions.iter().find(|(site, _)| Arc::ptr_eq(site, expr)) {
        return Ok(column.clone());
    }
    if let Some(column) = expr.downcast_ref::<Column>() {
        return lower_join_filter_column(column, context, left_required, right_required);
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return Ok(lower_literal(literal.value()).expect("validated join-residual literal must lower"));
    }
    let mut lower = |child: &Arc<dyn PhysicalExpr>| {
        lower_join_filter_expr(child, context, substitutions, left_required, right_required)
    };
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let left = lower(binary.left())?;
        let right = lower(binary.right())?;
        return Ok(match binary.op() {
            Operator::IsDistinctFrom => NativeExpr::is_distinct_from(left, right),
            Operator::IsNotDistinctFrom => NativeExpr::is_not_distinct_from(left, right),
            op => NativeExpr::binary(left, lower_binary_operator(op), right),
        });
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Ok(!lower(not.arg())?);
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower(is_null.arg())?));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower(is_not_null.arg())?));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return Ok(NativeExpr::cast(lower(cast.expr())?, cast.cast_type().clone()));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        debug_assert!(!in_list.is_empty(), "empty IN list cannot carry a join-residual validation proof");
        let value = lower(in_list.expr())?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let literal = item.downcast_ref::<Literal>().expect("validated join-residual IN entries are literals");
                debug_assert!(
                    !literal.value().is_null(),
                    "null IN entry cannot carry a join-residual validation proof"
                );
                lower(item)
            })
            .collect::<Result<Vec<_>, LoweringFailure>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(value, list)
        } else {
            NativeExpr::in_list(value, list)
        });
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return lower_join_filter_case_expr(case, context, &mut lower);
    }
    unreachable!("validated join residual must have a lowering branch")
}

fn lower_join_filter_case_expr(
    case: &CaseExpr,
    context: JoinFilterLoweringContext<'_>,
    lower: &mut impl FnMut(&Arc<dyn PhysicalExpr>) -> Result<NativeExpr, LoweringFailure>,
) -> Result<NativeExpr, LoweringFailure> {
    let base_expr = case.expr().map(&mut *lower).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => NativeExpr::binary(base_expr.clone(), NativeBinaryOp::Eq, lower(when)?),
                None => lower(when)?,
            };
            Ok(NativeCaseWhen::new(when, lower(then)?))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower(else_expr)?,
        None => {
            let data_type = case_expr_data_type(case, context.filter.schema())
                .expect("validated join-residual CASE must have an output type");
            NativeExpr::literal(NativeLiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

/// Appends `expr`, which reads only `side`, to that input as a synthetic
/// column and returns the residual's reference to it. The general expression
/// lowering produces the column, as it does for a join key expression.
fn materialize_join_residual_expr(
    expr: &Arc<dyn PhysicalExpr>,
    side: DataFusionJoinSide,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
    materializations: &mut JoinResidualMaterializations,
) -> Result<NativeExpr, LoweringFailure> {
    let (schema, required) = match side {
        DataFusionJoinSide::Left => (context.left_schema, left_required),
        DataFusionJoinSide::Right => (context.right_schema, right_required),
        DataFusionJoinSide::None => unreachable!("validated materialized residual must belong to left or right"),
    };
    let mut filter_indices = BTreeSet::new();
    collect_required_field_indices(expr, context.filter.schema(), &mut filter_indices)?;
    let mut columns = BTreeMap::new();
    for filter_index in filter_indices {
        let column_index = context
            .filter
            .column_indices()
            .get(filter_index)
            .expect("validated residual column must have a side mapping");
        debug_assert_eq!(column_index.side, side, "a materialization site reads one input");
        let field =
            schema.fields().get(column_index.index).expect("validated residual column must exist in its side schema");
        required.insert(column_index.index);
        let native_name = side_native_name(context, side, column_index.index, field.name())?;
        columns.insert(filter_index, NativeColumnRef::unqualified(native_name));
    }
    let validated = context.contract.expression().trusted_descendant(expr);
    let native_expr = lower_validated_join_residual_expr(&validated, &columns, &[])?;
    let side_materializations =
        materializations.side_mut(side).expect("validated materialization side must be left or right");
    let alias = unique_synthetic_join_residual_alias(side, side_materializations.len(), schema, side_materializations);
    let field = physical_expr_field_from_validated(&validated, alias.as_str())
        .expect("validated materialized residual must have a native field");
    side_materializations.push(JoinResidualMaterialization {
        alias: alias.clone(),
        expr: alias_if_needed(native_expr, alias.as_str()),
        field,
    });
    let native_input_index = context
        .canonical
        .native_input_index_for_datafusion_side(side)
        .expect("validated materialization side must map to a native input");
    Ok(NativeExpr::input_column(native_input_index, alias.as_str(), alias.as_str()))
}

fn unique_synthetic_join_residual_alias(
    side: DataFusionJoinSide,
    index: usize,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> String {
    // The `__algeon_native_join_residual_*` prefix is visible in native plans and tests.
    // Treat it as stable unless a migration updates diagnostics consumers.
    let side_name = match side {
        DataFusionJoinSide::Left => "left",
        DataFusionJoinSide::Right => "right",
        DataFusionJoinSide::None => "none",
    };
    let base = format!("__algeon_native_join_residual_{side_name}_{index}");
    unique_name_with_suffix(base.as_str(), |alias| {
        synthetic_join_residual_alias_is_available(alias, input_schema, materializations)
    })
}

fn synthetic_join_residual_alias_is_available(
    alias: &str,
    input_schema: &SchemaRef,
    materializations: &[JoinResidualMaterialization],
) -> bool {
    input_schema.index_of(alias).is_err()
        && materializations.iter().all(|materialization| materialization.alias != alias)
}

fn lower_join_filter_column(
    column: &Column,
    context: JoinFilterLoweringContext<'_>,
    left_required: &mut BTreeSet<usize>,
    right_required: &mut BTreeSet<usize>,
) -> Result<NativeExpr, LoweringFailure> {
    let column_index = context
        .filter
        .column_indices()
        .get(column.index())
        .expect("validated residual column must have a side mapping");
    let (datafusion_side, schema, required) = match column_index.side {
        DataFusionJoinSide::Left => (DataFusionJoinSide::Left, context.left_schema, left_required),
        DataFusionJoinSide::Right => (DataFusionJoinSide::Right, context.right_schema, right_required),
        DataFusionJoinSide::None => unreachable!("validated residual column must belong to a join input"),
    };
    let field =
        schema.fields().get(column_index.index).expect("validated residual column must exist in its side schema");
    required.insert(column_index.index);
    let native_input_index = context
        .canonical
        .native_input_index_for_datafusion_side(datafusion_side)
        .expect("validated residual side must map to a native input");
    let native_name = side_native_name(context, datafusion_side, column_index.index, field.name())?;
    Ok(NativeExpr::input_column(native_input_index, native_name, field.name()))
}

fn side_native_name<'a>(
    context: JoinFilterLoweringContext<'a>,
    side: DataFusionJoinSide,
    frontend_index: usize,
    fallback_name: &'a str,
) -> Result<&'a str, LoweringFailure> {
    let Some(bindings) = context.bindings else {
        return Ok(fallback_name);
    };
    let input = match side {
        DataFusionJoinSide::Left => bindings.left,
        DataFusionJoinSide::Right => bindings.right,
        DataFusionJoinSide::None => unreachable!("validated residual binding side must be left or right"),
    };
    Ok(input.binding_for_frontend_index(frontend_index)?.native_name.as_str())
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        common::{JoinSide, ScalarValue},
        config::ConfigOptions,
        functions::{string, unicode},
        logical_expr::{Operator, ScalarUDF},
        physical_expr::{
            PhysicalExpr, ScalarFunctionExpr,
            expressions::{
                CaseExpr, CastExpr, IsNotNullExpr, IsNullExpr, LikeExpr, NotExpr, binary, col, in_list, lit,
            },
        },
        physical_plan::joins::utils::{ColumnIndex, JoinFilter},
    };

    use super::*;
    use crate::native::{expression_contract::validate_join_residual, lowering::join::CanonicalJoinKind};

    fn scalar_function(
        name: &str,
        function: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        output_type: DataType,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(ScalarFunctionExpr::new(
            name,
            function,
            args,
            Arc::new(Field::new(name, output_type, true)),
            Arc::new(ConfigOptions::new()),
        ))
    }

    fn residual_schemas(data_type: DataType) -> (SchemaRef, SchemaRef, SchemaRef) {
        (
            Arc::new(Schema::new(vec![Field::new("left", data_type.clone(), true)])),
            Arc::new(Schema::new(vec![Field::new("right", data_type.clone(), true)])),
            Arc::new(Schema::new(vec![
                Field::new("left", data_type.clone(), true),
                Field::new("right", data_type, true),
            ])),
        )
    }

    fn assert_join_residual_parity(
        label: &str,
        expr: Arc<dyn PhysicalExpr>,
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
        filter_schema: &SchemaRef,
    ) -> LoweredJoinResidual {
        let filter = JoinFilter::new(
            expr,
            vec![ColumnIndex { index: 0, side: JoinSide::Left }, ColumnIndex { index: 0, side: JoinSide::Right }],
            Arc::clone(filter_schema),
        );
        let validated = validate_join_residual(&filter, left_schema, right_schema)
            .unwrap_or_else(|error| panic!("{label} must validate: {error:?}"));
        let lowered = lower_join_filter(&validated, CanonicalJoin::new(CanonicalJoinKind::Inner, false), None)
            .unwrap_or_else(|error| panic!("{label} must lower: {error:?}"));
        assert!(!lowered.left_required.is_empty(), "{label} must retain a left input dependency");
        assert!(!lowered.right_required.is_empty(), "{label} must retain a right input dependency");
        lowered
    }

    /// The single left-input materialization of `lowered`, which the residual reads by its alias.
    fn left_materialization<'l>(label: &str, lowered: &'l LoweredJoinResidual) -> &'l NativeExpr {
        let [materialization] = lowered.materializations.side(DataFusionJoinSide::Left).unwrap() else {
            panic!("{label} must materialize one left-input expression");
        };
        let alias = materialization.alias.as_str();
        assert!(
            lowered.predicate.any(|expr| matches!(expr, NativeExpr::Column(column) if column.name() == alias)),
            "{label} residual must read {alias}: {}",
            lowered.predicate
        );
        let NativeExpr::Alias { expr, .. } = &materialization.expr else {
            panic!("{label} materialization must carry its alias: {:?}", materialization.expr);
        };
        expr
    }

    #[test]
    fn every_supported_join_residual_shape_validates_then_lowers() {
        let (left_schema, right_schema, schema) = residual_schemas(DataType::Int32);
        let left = || col("left", &schema).unwrap();
        let right = || col("right", &schema).unwrap();
        let cross_side = || binary(left(), Operator::Eq, right(), &schema).unwrap();

        assert_join_residual_parity("columns and binary", cross_side(), &left_schema, &right_schema, &schema);

        let not_and_null = binary(
            Arc::new(NotExpr::new(cross_side())),
            Operator::And,
            binary(Arc::new(IsNullExpr::new(left())), Operator::Or, Arc::new(IsNotNullExpr::new(right())), &schema)
                .unwrap(),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("not and null predicates", not_and_null, &left_schema, &right_schema, &schema);

        let ast_cast = binary(
            Arc::new(CastExpr::new(left(), DataType::Int64, None)),
            Operator::Eq,
            Arc::new(CastExpr::new(right(), DataType::Int64, None)),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("AST casts", ast_cast, &left_schema, &right_schema, &schema);

        let materialized_cast = binary(
            Arc::new(CastExpr::new(left(), DataType::Decimal128(10, 0), None)),
            Operator::Eq,
            Arc::new(CastExpr::new(right(), DataType::Decimal128(10, 0), None)),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("materialized casts", materialized_cast, &left_schema, &right_schema, &schema);

        let in_list_expr = binary(
            in_list(
                left(),
                vec![lit(ScalarValue::Int32(Some(1))), lit(ScalarValue::Int32(Some(2)))],
                &false,
                schema.as_ref(),
            )
            .unwrap(),
            Operator::And,
            cross_side(),
            &schema,
        )
        .unwrap();
        assert_join_residual_parity("IN list", in_list_expr, &left_schema, &right_schema, &schema);

        let single_side_case: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(binary(left(), Operator::Gt, lit(ScalarValue::Int32(Some(0))), &schema).unwrap(), left())],
                Some(lit(ScalarValue::Int32(Some(0)))),
            )
            .unwrap(),
        );
        let materialized_case = binary(single_side_case, Operator::Eq, right(), &schema).unwrap();
        assert_join_residual_parity(
            "single-side materialized CASE",
            materialized_case,
            &left_schema,
            &right_schema,
            &schema,
        );

        let cross_side_case: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(cross_side(), lit(ScalarValue::Boolean(Some(true))))],
                Some(lit(ScalarValue::Boolean(Some(false)))),
            )
            .unwrap(),
        );
        assert_join_residual_parity("cross-side CASE", cross_side_case, &left_schema, &right_schema, &schema);

        let (left_schema, right_schema, schema) = residual_schemas(DataType::Utf8);
        let left = || col("left", &schema).unwrap();
        let right = || col("right", &schema).unwrap();
        let supported_functions = [
            scalar_function(
                "substring",
                unicode::substr(),
                vec![left(), lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(Some(2)))],
                DataType::Utf8,
            ),
            scalar_function("lower", string::lower(), vec![left()], DataType::Utf8),
            scalar_function("upper", string::upper(), vec![left()], DataType::Utf8),
        ];
        for function in supported_functions {
            let name = function.downcast_ref::<ScalarFunctionExpr>().unwrap().name().to_string();
            let predicate = binary(function, Operator::Eq, right(), &schema).unwrap();
            assert_join_residual_parity(&name, predicate, &left_schema, &right_schema, &schema);
        }

        let text = |value: &str| lit(ScalarValue::Utf8(Some(value.to_string())));
        let or_right_is_x = |expr| {
            binary(expr, Operator::Or, binary(right(), Operator::Eq, text("x"), &schema).unwrap(), &schema).unwrap()
        };
        let like = Arc::new(LikeExpr::new(false, false, left(), text("a%")));
        let lowered = assert_join_residual_parity("LIKE", or_right_is_x(like), &left_schema, &right_schema, &schema);
        assert!(matches!(left_materialization("LIKE", &lowered), NativeExpr::Like { .. }));

        let concatenation = binary(left(), Operator::StringConcat, text("!"), &schema).unwrap();
        let predicate = binary(concatenation, Operator::Eq, right(), &schema).unwrap();
        let lowered = assert_join_residual_parity("||", predicate, &left_schema, &right_schema, &schema);
        assert!(left_materialization("||", &lowered).any(|expr| matches!(expr, NativeExpr::Concat(_))));

        for operator in [Operator::IsDistinctFrom, Operator::IsNotDistinctFrom] {
            let single_input = binary(left(), operator, text("x"), &schema).unwrap();
            let label = format!("single-input {operator}");
            let lowered =
                assert_join_residual_parity(&label, or_right_is_x(single_input), &left_schema, &right_schema, &schema);
            assert!(matches!(
                left_materialization(&label, &lowered),
                NativeExpr::IsDistinctFrom { .. } | NativeExpr::IsNotDistinctFrom { .. }
            ));

            let cross_input = binary(left(), operator, right(), &schema).unwrap();
            let lowered = assert_join_residual_parity(
                &format!("cross-input {operator}"),
                cross_input,
                &left_schema,
                &right_schema,
                &schema,
            );
            assert!(matches!(
                lowered.predicate,
                NativeExpr::IsDistinctFrom { .. } | NativeExpr::IsNotDistinctFrom { .. }
            ));
        }

        let (left_schema, right_schema, schema) = residual_schemas(DataType::Int32);
        let left = || col("left", &schema).unwrap();
        let right = || col("right", &schema).unwrap();
        let halved = binary(left(), Operator::Divide, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
        let predicate = binary(halved, Operator::Eq, right(), &schema).unwrap();
        let lowered = assert_join_residual_parity(
            "integer division by a literal",
            predicate,
            &left_schema,
            &right_schema,
            &schema,
        );
        assert!(matches!(
            left_materialization("integer division by a literal", &lowered),
            NativeExpr::Binary { op: NativeBinaryOp::Divide, .. }
        ));
    }
}
