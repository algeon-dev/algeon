use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::{
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{DataType, TypeId},
    },
    internal::resolve::missing_column_error,
    ops::{Expr, binary::BinaryOperator, expr::LitValue, filter, unary::UnaryOperator},
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
    /// Unsupported predicate nodes fail explicitly before the low-level
    /// filtering path runs.
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
        let lowered = lower_row_predicate(self, &predicate, operation)?;
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
}

fn lower_row_predicate(df: &DataFrame, expr: &Expr, operation: &'static str) -> Result<LoweredRowPredicate> {
    let mut lowered = LoweredRowPredicate { nodes: Vec::new(), scalars: Vec::new(), root_index: -1 };
    let (root_index, output_type) = lower_expr(df, expr, &mut lowered, operation)?;
    if output_type.id() != TypeId::Bool8 {
        return Err(Error::new(ErrorKind::TypeError, "row predicate must evaluate to Bool8")
            .with_operation(operation)
            .with_context("predicate_output_type", format!("{:?}", output_type.id()).to_lowercase()));
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
        Expr::Literal(literal) => lower_literal(df, literal, lowered),
        Expr::BinaryOp { lhs, op, rhs } => lower_binary_expr(df, lhs, *op, rhs, lowered, operation),
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
        | Expr::IfElse { .. } => unsupported_predicate(operation, expr_kind(expr)),
    }
}

fn lower_literal(df: &DataFrame, literal: &LitValue, lowered: &mut LoweredRowPredicate) -> Result<(i32, DataType)> {
    let scalar = literal.to_scalar_in_domain(df.execution_domain())?;
    let dtype = scalar.data_type()?;
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

fn lower_binary_expr(
    df: &DataFrame,
    lhs: &Expr,
    op: BinaryOperator,
    rhs: &Expr,
    lowered: &mut LoweredRowPredicate,
    operation: &'static str,
) -> Result<(i32, DataType)> {
    let (lhs_index, lhs_type) = lower_expr(df, lhs, lowered, operation)?;
    let (rhs_index, rhs_type) = lower_expr(df, rhs, lowered, operation)?;

    let output_type = match op {
        BinaryOperator::Equal
        | BinaryOperator::NotEqual
        | BinaryOperator::Less
        | BinaryOperator::Greater
        | BinaryOperator::LessEqual
        | BinaryOperator::GreaterEqual => {
            validate_row_predicate_comparison_types(lhs_type.id(), rhs_type.id(), op, operation)?;
            crate::ops::binary::infer_binary_output_type(lhs_type.id(), rhs_type.id(), op)?
        },
        BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
            if lhs_type.id() != TypeId::Bool8 || rhs_type.id() != TypeId::Bool8 {
                return Err(Error::new(ErrorKind::TypeError, "logical row-predicate operators require Bool8 operands")
                    .with_operation(operation)
                    .with_context("lhs_type_id", format!("{:?}", lhs_type.id()).to_lowercase())
                    .with_context("rhs_type_id", format!("{:?}", rhs_type.id()).to_lowercase())
                    .with_context("operator", format!("{:?}", op).to_lowercase()));
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

    Err(Error::new(ErrorKind::NotSupported, "row-predicate comparison operands are not supported for fused filtering")
        .with_operation(operation)
        .with_context("lhs_type_id", format!("{:?}", lhs_type).to_lowercase())
        .with_context("rhs_type_id", format!("{:?}", rhs_type).to_lowercase())
        .with_context("operator", format!("{:?}", op).to_lowercase()))
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
                return Err(Error::new(ErrorKind::TypeError, "logical row-predicate NOT requires a Bool8 operand")
                    .with_operation(operation)
                    .with_context("input_type_id", format!("{:?}", input_type.id()).to_lowercase()));
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
    Err(Error::new(ErrorKind::NotSupported, "predicate expression is not supported by fused row filtering")
        .with_operation(operation)
        .with_context("expr_kind", expr_kind.to_string()))
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Column(_) => "column",
        Expr::Literal(_) => "literal",
        Expr::BinaryOp { .. } => "binary_op",
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
        Expr::IfElse { .. } => "if_else",
    }
}
