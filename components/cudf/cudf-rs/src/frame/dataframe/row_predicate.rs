use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::{
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        scalar::Scalar,
        types::{DataType, TypeId},
    },
    internal::resolve::missing_column_error,
    ops::{
        Expr,
        binary::BinaryOperator,
        expr::{self, LitValue},
        filter,
        unary::UnaryOperator,
    },
    runtime::{ExecutionOverrides, execution_domain::ExecutionDomain},
};

struct LoweredRowPredicate {
    nodes: Vec<cudf_sys::cudf_row_predicate_node_t>,
    scalars: Vec<Scalar>,
    root_index: i32,
}

impl DataFrame {
    /// Filter rows using a validated fused predicate expression.
    ///
    /// This is the preferred high-level API for multi-condition row
    /// filtering. It avoids chained intermediate `DataFrame`
    /// materializations by lowering a supported predicate subset to a
    /// dedicated row-predicate execution path and applying one final table
    /// filter.
    ///
    /// V1 supported predicate syntax is intentionally narrower than the full
    /// [`Expr`] DSL:
    ///
    /// - column references and literals
    /// - comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
    /// - null checks: `is_null`, `is_not_null`
    /// - boolean composition: `!`, `&`, `|`
    /// - alias wrappers around a supported inner predicate
    ///
    /// Unsupported predicate nodes are evaluated through the general
    /// expression engine and applied as a boolean mask until the fused
    /// row-predicate path supports those expressions directly.
    ///
    /// The predicate must evaluate to a row-aligned `Bool8`. Rows where the
    /// predicate is `false` or null are dropped.
    pub fn filter_expr(&self, predicate: Expr) -> Result<Self> {
        self.filter_expr_with_domain(predicate, Arc::clone(&self.domain), "dataframe::filter_expr")
    }

    /// Filter rows where all predicates are true.
    ///
    /// This is the preferred ergonomic wrapper for common multi-condition
    /// conjunctions. It is equivalent to combining the predicates with `&`
    /// and passing the result to [`DataFrame::filter_expr`].
    ///
    /// Empty input is the logical identity and returns `self.clone()`.
    pub fn filter_all<I>(&self, predicates: I) -> Result<Self>
    where I: IntoIterator<Item = Expr> {
        let mut predicates = predicates.into_iter();
        let Some(mut combined) = predicates.next() else {
            return Ok(self.clone());
        };
        for predicate in predicates {
            combined = combined & predicate;
        }
        self.filter_expr_with_domain(combined, Arc::clone(&self.domain), "dataframe::filter_all")
    }

    /// Filter rows where any predicate is true.
    ///
    /// This is the preferred ergonomic wrapper for common multi-condition
    /// disjunctions. It is equivalent to combining the predicates with `|`
    /// and passing the result to [`DataFrame::filter_expr`].
    ///
    /// Empty input returns an empty DataFrame with the same schema.
    pub fn filter_any<I>(&self, predicates: I) -> Result<Self>
    where I: IntoIterator<Item = Expr> {
        let mut predicates = predicates.into_iter();
        let Some(mut combined) = predicates.next() else {
            return self.empty_with_same_schema();
        };
        for predicate in predicates {
            combined = combined | predicate;
        }
        self.filter_expr_with_domain(combined, Arc::clone(&self.domain), "dataframe::filter_any")
    }

    pub(crate) fn filter_expr_in_context(&self, predicate: Expr, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.filter_expr_with_domain(predicate, output_domain, "dataframe::with_execution::filter_expr")
    }

    pub(crate) fn filter_all_in_context<I>(&self, predicates: I, ctx: &ExecutionOverrides<'_>) -> Result<Self>
    where I: IntoIterator<Item = Expr> {
        let mut predicates = predicates.into_iter();
        let Some(mut combined) = predicates.next() else {
            return Ok(self.clone());
        };
        for predicate in predicates {
            combined = combined & predicate;
        }
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.filter_expr_with_domain(combined, output_domain, "dataframe::with_execution::filter_all")
    }

    pub(crate) fn filter_any_in_context<I>(&self, predicates: I, ctx: &ExecutionOverrides<'_>) -> Result<Self>
    where I: IntoIterator<Item = Expr> {
        let mut predicates = predicates.into_iter();
        let Some(mut combined) = predicates.next() else {
            let output_domain = ctx.resolve_output_domain(&self.domain)?;
            return self.empty_with_same_schema_in_domain(output_domain);
        };
        for predicate in predicates {
            combined = combined | predicate;
        }
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.filter_expr_with_domain(combined, output_domain, "dataframe::with_execution::filter_any")
    }

    fn filter_expr_with_domain(
        &self,
        predicate: Expr,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let lowered = match lower_row_predicate(self, &predicate, operation) {
            Ok(lowered) => lowered,
            Err(err) if should_use_mask_fallback(&err) => {
                return self.filter_expr_via_boolean_mask(predicate, output_domain, operation);
            },
            Err(err) => return Err(err),
        };
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            ExecutionDomain::insert_stream_dependency(&self.domain, &output_domain, operation)?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let tv = self.build_table_view()?;
        let result_table = filter::apply_row_predicate_impl(
            &tv,
            &lowered.nodes,
            lowered.root_index,
            &lowered.scalars,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn filter_expr_via_boolean_mask(
        &self,
        predicate: Expr,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let mask = expr::eval_in_output_domain(self, Arc::clone(&output_domain), operation, &predicate)?;
        let mask_view = mask.view()?;
        filter::validate_boolean_mask(&mask_view, self.num_rows(), "DataFrame row count", operation)?;
        let execution = crate::runtime::execution_resolver::ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.domain, mask_view.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());
        let tv = self.build_table_view()?;
        let result_table = filter::apply_boolean_mask_impl(
            &tv,
            &mask_view,
            &output_domain,
            Some(execution.exec_options()),
            operation,
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }
}

fn should_use_mask_fallback(err: &Error) -> bool {
    err.code() == ErrorCode::UnsupportedOperation
        && err.detail_value("resource") == Some("row_predicate")
        && err.detail_value("reason") == Some("unsupported_expr_kind")
}

fn lower_row_predicate(df: &DataFrame, expr: &Expr, operation: &'static str) -> Result<LoweredRowPredicate> {
    let mut lowered = LoweredRowPredicate { nodes: Vec::new(), scalars: Vec::new(), root_index: -1 };
    let (root_index, output_type) = lower_expr(df, expr, &mut lowered, operation)?;
    if output_type.id() != TypeId::Bool8 {
        return Err(Error::new_with(ErrorCode::TypeMismatch, operation, "row predicate must evaluate to Bool8")
            .with_fact(FACT_LEFT_TYPE_ID, output_type.id())
            .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Bool8));
    }
    lowered.root_index = root_index;
    Ok(lowered)
}

fn lower_expr(
    df: &DataFrame,
    expr: &Expr,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    match expr {
        Expr::Column(name) => {
            let column_index = df.column_index(name).ok_or_else(|| missing_column_error(operation, name))?;
            let dtype = df.series(name)?.dtype()?;
            let index = push_node(
                lowered,
                cudf_sys::cudf_row_predicate_node_t {
                    kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_COLUMN_REF,
                    lhs_index: -1,
                    rhs_index: -1,
                    column_index: crate::core::error::usize_to_i32(column_index, operation, "column_index")?,
                    binary_op: cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
                    unary_op: cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
                    literal: std::ptr::null_mut(),
                },
            )?;
            Ok((index, dtype))
        },
        Expr::Literal(literal) => lower_literal(df, literal, lowered, operation),
        Expr::BinaryOp { lhs, op, rhs } => lower_binary_expr(df, lhs, *op, rhs, lowered, operation),
        Expr::BinaryOpWithType { .. } => unsupported_predicate(operation, expr_kind(expr)),
        Expr::UnaryOp { op, input } => lower_unary_expr(df, *op, input, lowered, operation),
        Expr::IsNull(input) => lower_is_null_expr(df, input, lowered, operation),
        Expr::IsNotNull(input) => {
            let (is_null_index, _) = lower_is_null_expr(df, input, lowered, operation)?;
            let not_index = push_node(
                lowered,
                cudf_sys::cudf_row_predicate_node_t {
                    kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_UNARY_OP,
                    lhs_index: is_null_index,
                    rhs_index: -1,
                    column_index: -1,
                    binary_op: cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
                    unary_op: cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
                    literal: std::ptr::null_mut(),
                },
            )?;
            Ok((not_index, DataType::new(TypeId::Bool8)))
        },
        Expr::Alias { inner, .. } => lower_expr(df, inner, lowered, operation),
        Expr::Cast { .. }
        | Expr::FillNaScalar { .. }
        | Expr::FillNaExpr { .. }
        | Expr::FillNullPolicy { .. }
        | Expr::Scan { .. }
        | Expr::Reduce { .. }
        | Expr::StringOp { .. }
        | Expr::DateTimeComponent { .. }
        | Expr::DateTimeOffset { .. }
        | Expr::InList { .. }
        | Expr::IfElse { .. } => unsupported_predicate(operation, expr_kind(expr)),
    }
}

fn lower_literal(
    df: &DataFrame,
    literal: &LitValue,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    lower_literal_with_target(df, literal, None, lowered, operation)
}

fn lower_literal_with_target(
    df: &DataFrame,
    literal: &LitValue,
    target_type: Option<DataType>,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    let mut scalar = literal.to_scalar_in_domain(df.execution_domain())?;
    if let Some(target_type) = target_type {
        scalar = align_decimal_literal_scalar(df, scalar, target_type, operation)?;
    }
    let dtype = scalar.data_type()?;
    if !row_predicate_literal_dtype_is_supported(dtype) {
        return unsupported_predicate(operation, "literal");
    }
    let handle = scalar.raw_handle();
    lowered.scalars.push(scalar);
    let index = push_node(
        lowered,
        cudf_sys::cudf_row_predicate_node_t {
            kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_LITERAL,
            lhs_index: -1,
            rhs_index: -1,
            column_index: -1,
            binary_op: cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
            unary_op: cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
            literal: handle,
        },
    )?;
    Ok((index, dtype))
}

fn align_decimal_literal_scalar(
    df: &DataFrame,
    scalar: Scalar,
    target_type: DataType,
    operation: &'static str,
) -> Result<Scalar> {
    if !target_type.id().is_decimal() {
        return Ok(scalar);
    }
    let source_type = scalar.data_type()?;
    if source_type == target_type {
        return Ok(scalar);
    }
    let Some((raw, source_scale)) = scalar_decimal_or_integral_raw(&scalar, source_type)? else {
        return Ok(scalar);
    };
    let raw = rescale_decimal_raw(raw, source_scale, target_type.scale(), operation)?;
    Scalar::decimal_from_raw_in_domain(raw, target_type, df.execution_domain())
}

fn scalar_decimal_or_integral_raw(scalar: &Scalar, source_type: DataType) -> Result<Option<(i128, i32)>> {
    let raw = match source_type.id() {
        TypeId::Decimal32 => i128::from(scalar.decimal32_raw_value()?),
        TypeId::Decimal64 => i128::from(scalar.decimal64_raw_value()?),
        TypeId::Decimal128 => scalar.decimal128_raw_value()?,
        TypeId::Int8 => i128::from(scalar.value::<i8>()?),
        TypeId::Int16 => i128::from(scalar.value::<i16>()?),
        TypeId::Int32 => i128::from(scalar.value::<i32>()?),
        TypeId::Int64 => i128::from(scalar.value::<i64>()?),
        TypeId::UInt8 => i128::from(scalar.value::<u8>()?),
        TypeId::UInt16 => i128::from(scalar.value::<u16>()?),
        TypeId::UInt32 => i128::from(scalar.value::<u32>()?),
        TypeId::UInt64 => i128::from(scalar.value::<u64>()?),
        _ => return Ok(None),
    };
    let scale = if source_type.id().is_decimal() { source_type.scale() } else { 0 };
    Ok(Some((raw, scale)))
}

fn rescale_decimal_raw(raw: i128, source_scale: i32, target_scale: i32, operation: &'static str) -> Result<i128> {
    let scale_delta = source_scale.checked_sub(target_scale).ok_or_else(|| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal literal scale alignment overflowed")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string())
    })?;
    if scale_delta >= 0 {
        let multiplier = checked_pow10(scale_delta as u32, operation)?;
        raw.checked_mul(multiplier).ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal literal raw value overflows target scale")
                .with_detail("source_scale", source_scale.to_string())
                .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        let divisor = checked_pow10(scale_delta.unsigned_abs(), operation)?;
        if raw % divisor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                operation,
                "decimal literal cannot be represented exactly at target scale",
            )
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(raw / divisor)
    }
}

fn checked_pow10(exponent: u32, operation: &'static str) -> Result<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "decimal scale factor overflows i128")
                .with_detail("exponent", exponent.to_string())
        })?;
    }
    Ok(value)
}

fn row_predicate_literal_dtype_is_supported(dtype: DataType) -> bool {
    dtype.id().is_numeric()
        || dtype.id().is_decimal()
        || dtype.id().is_timestamp()
        || dtype.id().is_duration()
        || dtype.id() == TypeId::Bool8
        || dtype.id() == TypeId::String
}

fn lower_binary_expr(
    df: &DataFrame,
    lhs: &Expr,
    op: BinaryOperator,
    rhs: &Expr,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    let (lhs_index, lhs_type, rhs_index, rhs_type) = match (lhs, rhs) {
        (Expr::Literal(literal), rhs) if op.is_comparison() => {
            let (rhs_index, rhs_type) = lower_expr(df, rhs, lowered, operation)?;
            let (lhs_index, lhs_type) = lower_literal_with_target(df, literal, Some(rhs_type), lowered, operation)?;
            (lhs_index, lhs_type, rhs_index, rhs_type)
        },
        (lhs, Expr::Literal(literal)) if op.is_comparison() => {
            let (lhs_index, lhs_type) = lower_expr(df, lhs, lowered, operation)?;
            let (rhs_index, rhs_type) = lower_literal_with_target(df, literal, Some(lhs_type), lowered, operation)?;
            (lhs_index, lhs_type, rhs_index, rhs_type)
        },
        _ => {
            let (lhs_index, lhs_type) = lower_expr(df, lhs, lowered, operation)?;
            let (rhs_index, rhs_type) = lower_expr(df, rhs, lowered, operation)?;
            (lhs_index, lhs_type, rhs_index, rhs_type)
        },
    };

    let output_type = match op {
        BinaryOperator::Equal
        | BinaryOperator::NotEqual
        | BinaryOperator::Less
        | BinaryOperator::Greater
        | BinaryOperator::LessEqual
        | BinaryOperator::GreaterEqual => {
            validate_row_predicate_comparison_types(lhs_type.id(), rhs_type.id(), op, operation)?;
            crate::ops::binary::infer_binary_output_dtype_in(lhs_type, rhs_type, op, operation)?
        },
        BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
            if lhs_type.id() != TypeId::Bool8 || rhs_type.id() != TypeId::Bool8 {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "logical row-predicate operators require Bool8 operands",
                )
                .with_fact(FACT_LEFT_TYPE_ID, lhs_type.id())
                .with_fact(FACT_RIGHT_TYPE_ID, rhs_type.id())
                .with_detail("operator", op.name()));
            }
            DataType::new(TypeId::Bool8)
        },
        _ => return unsupported_predicate(operation, "binary_operator"),
    };

    let index = push_node(
        lowered,
        cudf_sys::cudf_row_predicate_node_t {
            kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_BINARY_OP,
            lhs_index,
            rhs_index,
            column_index: -1,
            binary_op: op.to_raw(),
            unary_op: cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
            literal: std::ptr::null_mut(),
        },
    )?;
    Ok((index, output_type))
}

fn validate_row_predicate_comparison_types(
    lhs_type: TypeId,
    rhs_type: TypeId,
    op: BinaryOperator,
    operation: &'static str,
) -> Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ComparisonCategory {
        String,
        Bool,
        NumericOrDecimal,
        Temporal,
        Other,
    }

    fn category(type_id: TypeId) -> ComparisonCategory {
        match type_id {
            TypeId::String => ComparisonCategory::String,
            TypeId::Bool8 => ComparisonCategory::Bool,
            _ if type_id.numeric_rank().is_some() || type_id.is_decimal() => ComparisonCategory::NumericOrDecimal,
            _ if type_id.is_timestamp() || type_id.is_duration() => ComparisonCategory::Temporal,
            _ => ComparisonCategory::Other,
        }
    }

    if lhs_type == rhs_type {
        return Ok(());
    }

    let lhs_category = category(lhs_type);
    let rhs_category = category(rhs_type);
    if lhs_category == rhs_category {
        return Ok(());
    }

    if lhs_category == ComparisonCategory::Other || rhs_category == ComparisonCategory::Other {
        return Ok(());
    }

    Err(Error::new_with(
        ErrorCode::TypeMismatch,
        operation,
        "row-predicate comparison operands are not supported for fused filtering",
    )
    .with_fact(FACT_LEFT_TYPE_ID, lhs_type)
    .with_fact(FACT_RIGHT_TYPE_ID, rhs_type)
    .with_detail("operator", op.name()))
}

fn lower_unary_expr(
    df: &DataFrame,
    op: UnaryOperator,
    input: &Expr,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    let (input_index, input_type) = lower_expr(df, input, lowered, operation)?;
    match op {
        UnaryOperator::Not => {
            if input_type.id() != TypeId::Bool8 {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "logical row-predicate NOT requires a Bool8 operand",
                )
                .with_fact(FACT_LEFT_TYPE_ID, input_type.id())
                .with_fact(FACT_RIGHT_TYPE_ID, TypeId::Bool8));
            }
            let index = push_node(
                lowered,
                cudf_sys::cudf_row_predicate_node_t {
                    kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_UNARY_OP,
                    lhs_index: input_index,
                    rhs_index: -1,
                    column_index: -1,
                    binary_op: cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
                    unary_op: op.to_raw(),
                    literal: std::ptr::null_mut(),
                },
            )?;
            Ok((index, DataType::new(TypeId::Bool8)))
        },
        _ => unsupported_predicate(operation, "unary_operator"),
    }
}

fn lower_is_null_expr(
    df: &DataFrame,
    input: &Expr,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    let (input_index, _) = lower_expr(df, input, lowered, operation)?;
    let index = push_node(
        lowered,
        cudf_sys::cudf_row_predicate_node_t {
            kind: cudf_sys::cudf_row_predicate_node_kind_t::CUDF_ROW_PREDICATE_NODE_IS_NULL,
            lhs_index: input_index,
            rhs_index: -1,
            column_index: -1,
            binary_op: cudf_sys::cudf_binary_operator_t::CUDF_BINARY_EQUAL,
            unary_op: cudf_sys::cudf_unary_operator_t::CUDF_UNARY_NOT,
            literal: std::ptr::null_mut(),
        },
    )?;
    Ok((index, DataType::new(TypeId::Bool8)))
}

fn push_node(lowered: &mut LoweredRowPredicate, node: cudf_sys::cudf_row_predicate_node_t) -> Result<i32> {
    lowered.nodes.push(node);
    crate::core::error::usize_to_i32(lowered.nodes.len() - 1, "dataframe::row_predicate::push_node", "node_index")
}

fn unsupported_predicate<T>(operation: &'static str, expr_kind: &str) -> Result<T> {
    Err(Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "predicate expression is not supported by fused row filtering",
    )
    .with_detail("resource", "row_predicate")
    .with_detail("reason", "unsupported_expr_kind")
    .with_detail("expr_kind", expr_kind.to_string())
    .with_detail("remedy", "rewrite the predicate into currently supported fused-filter operators"))
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Column(_) => "column",
        Expr::Literal(_) => "literal",
        Expr::BinaryOp { .. } => "binary_op",
        Expr::BinaryOpWithType { .. } => "binary_op_with_type",
        Expr::UnaryOp { .. } => "unary_op",
        Expr::Cast { .. } => "cast",
        Expr::IsNull(_) => "is_null",
        Expr::IsNotNull(_) => "is_not_null",
        Expr::Alias { .. } => "alias",
        Expr::FillNaScalar { .. } => "fillna_scalar",
        Expr::FillNaExpr { .. } => "fillna_expr",
        Expr::FillNullPolicy { .. } => "fill_null_policy",
        Expr::Scan { .. } => "scan",
        Expr::Reduce { .. } => "reduce",
        Expr::StringOp { .. } => "string_op",
        Expr::DateTimeComponent { .. } => "datetime_component",
        Expr::DateTimeOffset { .. } => "datetime_offset",
        Expr::InList { .. } => "in_list",
        Expr::IfElse { .. } => "if_else",
    }
}
