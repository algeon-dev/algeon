use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::{logical_expr::Operator, physical_expr::PhysicalExpr, physical_expr_common::physical_expr::fmt_sql};

/// Longest SQL rendering a rejection keeps. Reports and client messages carry
/// the text, so a very large expression is cut at a character boundary.
const MAX_RENDERED_SQL_CHARS: usize = 160;

/// The expression capability whose contract rejected a physical expression.
///
/// These variants are adapter-internal machine facts. They deliberately do
/// not contain report labels or formatting so validation and lowering can
/// share the contract without depending on reporting surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpressionCapability {
    PhysicalExpression,
    NegativeExpression,
    DynamicFilter,
    DataFusionTypeResolution,
    Literal,
    CompatibleTypes,
    CastPair,
    TryCastPair,
    BinaryOperator,
    BinaryOperandTypes,
    IntervalArithmetic,
    LikeCaseInsensitive,
    LikePattern,
    InList,
    ScalarFunction,
    ScalarFunctionArguments,
    SubstringArguments,
    ExtractField,
    JoinResidualShape,
    JoinResidualSides,
    JoinResidualSideSchema,
    JoinResidualCastTarget,
    JoinResidualScalarFunction,
    CaseActivationSize,
}

/// Stable coarse classification retained when the expression rejection is
/// projected to the public planning fallback contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpressionRejectionKind {
    UnsupportedExpression,
    UnsupportedType,
}

/// The offending part of a rejected expression, when the rejecting contract
/// can name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpressionRejectionSubject {
    Unnamed,
    /// DataFusion scalar function name.
    ScalarFunction(String),
    /// Source and target types of a rejected `CAST` or `TRY_CAST`.
    Cast {
        source: DataType,
        target: DataType,
    },
    LiteralType(DataType),
    DynamicFilterError(String),
    /// Binary operator and the operand types the contract refused.
    BinaryOperator {
        operator: Operator,
        left: DataType,
        right: DataType,
    },
    /// Unary operator spelling and the operand type the contract refused.
    UnaryOperator {
        operator: &'static str,
        operand: DataType,
    },
    /// The type of an expression that the operator it feeds (an aggregate,
    /// sort, window, or join key) cannot accept.
    ValueType(DataType),
}

/// Structured rejection emitted by the shared expression contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpressionRejection {
    kind: ExpressionRejectionKind,
    capability: ExpressionCapability,
    subject: ExpressionRejectionSubject,
    /// The innermost rejected expression in SQL form, without column indexes.
    sql: Option<String>,
}

impl ExpressionRejection {
    pub(super) const fn unsupported_expression(capability: ExpressionCapability) -> Self {
        Self {
            kind: ExpressionRejectionKind::UnsupportedExpression,
            capability,
            subject: ExpressionRejectionSubject::Unnamed,
            sql: None,
        }
    }

    pub(super) const fn unsupported_type(capability: ExpressionCapability) -> Self {
        Self {
            kind: ExpressionRejectionKind::UnsupportedType,
            capability,
            subject: ExpressionRejectionSubject::Unnamed,
            sql: None,
        }
    }

    /// Rejects `expr`, whose `data_type` the operator it feeds cannot accept.
    pub(in crate::native) fn unsupported_value_type(expr: &Arc<dyn PhysicalExpr>, data_type: DataType) -> Self {
        Self::unsupported_type(ExpressionCapability::PhysicalExpression)
            .with_subject(ExpressionRejectionSubject::ValueType(data_type))
            .at_expression(expr)
    }

    pub(super) fn with_subject(mut self, subject: ExpressionRejectionSubject) -> Self {
        self.subject = subject;
        self
    }

    /// Names `subject` for a rejection raised by an operator's own checks: one
    /// that no nested expression located and that no contract named already.
    pub(super) fn named_by_operator(self, subject: Option<ExpressionRejectionSubject>) -> Self {
        match subject {
            Some(subject) if self.sql.is_none() && self.subject == ExpressionRejectionSubject::Unnamed => {
                self.with_subject(subject)
            },
            _ => self,
        }
    }

    /// Records `expr` as the rejected expression unless a nested expression
    /// was already recorded. Validation recurses into children first, so the
    /// innermost rejected expression is the one kept.
    pub(super) fn at_expression(mut self, expr: &Arc<dyn PhysicalExpr>) -> Self {
        if self.sql.is_none() {
            self.sql = Some(render_sql(expr.as_ref()));
        }
        self
    }

    /// Names the enclosing scalar function on an argument-contract rejection
    /// that does not already carry a subject.
    pub(super) fn for_scalar_function(self, name: &str) -> Self {
        if self.capability == ExpressionCapability::ScalarFunctionArguments
            && self.subject == ExpressionRejectionSubject::Unnamed
        {
            self.with_subject(ExpressionRejectionSubject::ScalarFunction(name.to_string()))
        } else {
            self
        }
    }

    pub(crate) const fn kind(&self) -> ExpressionRejectionKind {
        self.kind
    }

    pub(crate) const fn capability(&self) -> ExpressionCapability {
        self.capability
    }

    pub(crate) const fn subject(&self) -> &ExpressionRejectionSubject {
        &self.subject
    }

    pub(crate) fn sql(&self) -> Option<&str> {
        self.sql.as_deref()
    }
}

fn render_sql(expr: &dyn PhysicalExpr) -> String {
    let sql = fmt_sql(expr).to_string();
    match sql.char_indices().nth(MAX_RENDERED_SQL_CHARS) {
        Some((end, _)) => format!("{}...", &sql[..end]),
        None => sql,
    }
}
