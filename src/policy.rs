#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GpuFallbackPolicy {
    GpuOnly,
    #[default]
    GpuPreferred,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InsufficientIslandSize,
    PlanPropertyMismatch,
    ExplicitCpuPolicy,
}

impl FallbackReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InsufficientIslandSize => "insufficient_island_size",
            Self::PlanPropertyMismatch => "plan_property_mismatch",
            Self::ExplicitCpuPolicy => "explicit_cpu_policy",
        }
    }
}
