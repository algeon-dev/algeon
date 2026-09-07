//! RAPIDS implementation of the planning backend capability contract.
//!
//! Every answer here comes from the same component rules execution binds with:
//! the expression checks *are* the lowering, with the lowered result dropped.
//! That keeps a plan from reaching a kernel and only then discovering an
//! unsupported binding that was decidable during planning.

use arrow::datatypes::DataType as ArrowDataType;
use cudf::ops::binary::{BinaryOperator, infer_binary_output_dtype};
use nexus_query_planning::backend::{AggregateBinding, AnalysisContext, BackendCapabilities, ExpressionUse};

use crate::{
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_type},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateSpec, BinaryOp, Expr},
    native_error::cudf_query_error,
    plan::ProjectionOutput,
    prepared::{lower_aggregate_spec, lower_expr, lower_filter_predicate_expr, lower_projection_output_expr},
};

/// The backend capability the native engine executes with.
#[derive(Debug, Clone, Copy, Default)]
pub struct RapidsBackend;

/// The backend capability the native engine executes with.
pub const fn rapids_backend() -> &'static RapidsBackend {
    &RapidsBackend
}

/// An analysis context bound to the native engine's backend.
pub const fn rapids_analysis_context() -> AnalysisContext<'static> {
    AnalysisContext::new(rapids_backend())
}

impl BackendCapabilities for RapidsBackend {
    fn supports_type(&self, data_type: &ArrowDataType) -> bool {
        is_supported_arrow_type(data_type)
    }

    fn binary_output_type(&self, op: BinaryOp, left: &ArrowDataType, right: &ArrowDataType) -> Result<ArrowDataType> {
        let left_dtype = cudf_dtype_from_arrow_type(left).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.plan.validate_expr",
                "native decimal arithmetic left operand type is not supported",
            )
            .with_fact("left_type", format!("{left:?}"))
        })?;
        let right_dtype = cudf_dtype_from_arrow_type(right).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.plan.validate_expr",
                "native decimal arithmetic right operand type is not supported",
            )
            .with_fact("right_type", format!("{right:?}"))
        })?;
        let output_dtype =
            infer_binary_output_dtype(left_dtype, right_dtype, cudf_binary_operator(op)?).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.plan.validate_expr",
                    "failed to infer native decimal arithmetic output type",
                    error,
                )
                .with_fact("left_type", format!("{left:?}"))
                .with_fact("right_type", format!("{right:?}"))
                .with_fact("operator", op.to_string())
            })?;
        cudf::interop::arrow_host::arrow_type_from_cudf_dtype(output_dtype).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.plan.validate_expr",
                "failed to map inferred native decimal arithmetic output type to Arrow",
                error,
            )
            .with_fact("left_type", format!("{left:?}"))
            .with_fact("right_type", format!("{right:?}"))
            .with_fact("operator", op.to_string())
        })
    }

    fn check_expression(&self, expr: &Expr, expression_use: ExpressionUse) -> Result<()> {
        match expression_use {
            ExpressionUse::Value => lower_expr(expr).map(|_| ()),
            ExpressionUse::FilterPredicate => lower_filter_predicate_expr(expr).map(|_| ()),
        }
    }

    fn check_projection_output(&self, output: &ProjectionOutput) -> Result<()> {
        lower_projection_output_expr(output).map(|_| ())
    }

    fn aggregate_output_names(&self, spec: &AggregateSpec, binding: AggregateBinding) -> Result<Vec<String>> {
        let lowered = match binding {
            AggregateBinding::SingleStage => lower_aggregate_spec(spec)?,
            AggregateBinding::FinalMerge => crate::prepared::lower_final_aggregate_spec(spec)?,
        };
        Ok(lowered.output_names)
    }
}

fn cudf_binary_operator(op: BinaryOp) -> Result<BinaryOperator> {
    match op {
        BinaryOp::Add => Ok(BinaryOperator::Add),
        BinaryOp::Subtract => Ok(BinaryOperator::Sub),
        BinaryOp::Multiply => Ok(BinaryOperator::Mul),
        BinaryOp::Divide => Ok(BinaryOperator::Div),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "nexus_query_engine.plan.validate_expr",
            "native decimal arithmetic cannot infer output type for this operator",
        )
        .with_fact("operator", op.to_string())),
    }
}
