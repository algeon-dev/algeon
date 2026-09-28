use arrow::datatypes::DataType;

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
}

/// Structured rejection emitted by the shared expression contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpressionRejection {
    kind: ExpressionRejectionKind,
    capability: ExpressionCapability,
    subject: ExpressionRejectionSubject,
}

impl ExpressionRejection {
    pub(super) const fn unsupported_expression(capability: ExpressionCapability) -> Self {
        Self {
            kind: ExpressionRejectionKind::UnsupportedExpression,
            capability,
            subject: ExpressionRejectionSubject::Unnamed,
        }
    }

    pub(super) const fn unsupported_type(capability: ExpressionCapability) -> Self {
        Self {
            kind: ExpressionRejectionKind::UnsupportedType,
            capability,
            subject: ExpressionRejectionSubject::Unnamed,
        }
    }

    pub(super) fn with_subject(mut self, subject: ExpressionRejectionSubject) -> Self {
        self.subject = subject;
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
}
