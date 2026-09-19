/// The expression capability whose contract rejected a physical expression.
///
/// These variants are adapter-internal machine facts. They deliberately do
/// not contain report labels or formatting so validation and lowering can
/// share the contract without depending on reporting surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum ExpressionCapability {
    PhysicalExpression,
    DataFusionTypeResolution,
    Literal,
    CompatibleTypes,
    CastPair,
    TryCastPair,
    BinaryOperator,
    BinaryOperandTypes,
    ScalarFunction,
    ScalarFunctionArguments,
    JoinResidualShape,
    JoinResidualSides,
    JoinResidualSideSchema,
    JoinResidualCastTarget,
    JoinResidualScalarFunction,
}

/// Stable coarse classification retained when the expression rejection is
/// projected to the public planning fallback contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum ExpressionRejectionKind {
    UnsupportedExpression,
    UnsupportedType,
}

/// Structured rejection emitted by the shared expression contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) struct ExpressionRejection {
    kind: ExpressionRejectionKind,
    capability: ExpressionCapability,
}

impl ExpressionRejection {
    pub(super) const fn unsupported_expression(capability: ExpressionCapability) -> Self {
        Self { kind: ExpressionRejectionKind::UnsupportedExpression, capability }
    }

    pub(super) const fn unsupported_type(capability: ExpressionCapability) -> Self {
        Self { kind: ExpressionRejectionKind::UnsupportedType, capability }
    }

    pub(in crate::native) const fn kind(self) -> ExpressionRejectionKind {
        self.kind
    }

    pub(in crate::native) const fn capability(self) -> ExpressionCapability {
        self.capability
    }
}
