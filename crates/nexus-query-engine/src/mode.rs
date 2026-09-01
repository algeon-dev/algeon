//! Aggregate and report execution-mode labels (wire DTOs).
//!
//! These labels describe aggregate evaluation contracts and preserved report
//! vocabulary, not admission or product modes; the bounded resource
//! controller owns the single bounded execution path. Result placement is
//! represented by `QueryOutputMode` in the execution layer.

use std::{fmt, str::FromStr};

/// Execution contract for floating statistical aggregates such as
/// `STDDEV_SAMP`, `STDDEV_POP`, `VAR_SAMP`, and `VAR_POP`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatisticalAggregateExecutionMode {
    /// Preserve DataFusion's order-sensitive Welford accumulator bits by
    /// materializing to host and evaluating in DataFusion input order.
    #[default]
    DataFusionExactHostWelford,
    /// Use libcudf's native grouped/scalar statistical kernels with each
    /// aggregate's declared degrees of freedom; floating-point bits may differ
    /// from DataFusion Welford order.
    GpuNativeTolerant,
}

impl StatisticalAggregateExecutionMode {
    /// Stable snake-case mode label used in env vars and reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DataFusionExactHostWelford => "datafusion_exact_host_welford",
            Self::GpuNativeTolerant => "gpu_native_tolerant",
        }
    }

    /// Floating parity contract required by this execution mode.
    pub const fn tolerance_contract(self) -> StatisticalAggregateToleranceContract {
        match self {
            Self::DataFusionExactHostWelford => StatisticalAggregateToleranceContract::Exact,
            Self::GpuNativeTolerant => StatisticalAggregateToleranceContract::FloatTolerant,
        }
    }

    /// Returns true when execution must preserve DataFusion's exact Welford bits.
    pub const fn is_datafusion_exact(self) -> bool {
        matches!(self, Self::DataFusionExactHostWelford)
    }
}

impl fmt::Display for StatisticalAggregateExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for StatisticalAggregateExecutionMode {
    type Err = StatisticalAggregateExecutionModeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "datafusion_exact_host_welford" | "datafusion_exact" | "exact" => Ok(Self::DataFusionExactHostWelford),
            "gpu_native_tolerant" | "gpu_native" | "tolerant" => Ok(Self::GpuNativeTolerant),
            other => Err(StatisticalAggregateExecutionModeParseError { value: other.to_string() }),
        }
    }
}

/// Error returned when parsing a statistical aggregate execution-mode label
/// fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatisticalAggregateExecutionModeParseError {
    value: String,
}

impl StatisticalAggregateExecutionModeParseError {
    /// Returns the unrecognized label after ASCII-lowercase normalization.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for StatisticalAggregateExecutionModeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid statistical aggregate execution mode: {}", self.value)
    }
}

impl std::error::Error for StatisticalAggregateExecutionModeParseError {}

/// Tolerance contract declared by a floating statistical aggregate execution
/// mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatisticalAggregateToleranceContract {
    /// Results are expected to match the compatibility contract exactly.
    Exact,
    /// Results are floating-point equivalent under a declared tolerance.
    FloatTolerant,
}

impl StatisticalAggregateToleranceContract {
    /// Stable snake-case tolerance-contract label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::FloatTolerant => "float_tolerant",
        }
    }
}

/// Capability-envelope status used by gate-first contract reports.
///
/// The single bounded controller owns admission. [`Self::as_str`] emits the
/// stable serialized wire labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityEnvelopeStatus {
    /// Every reported gate is modeled for bounded execution.
    Admitted,
    /// Structural support is known, but a prerequisite owner fact is not yet bound.
    Deferred,
    /// At least one gate or capability proof blocks bounded execution.
    Rejected,
}

impl CapabilityEnvelopeStatus {
    /// Stable wire label emitted by report serializers.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "bounded",
            Self::Deferred => "deferred_bounded",
            Self::Rejected => "unsupported_bounded",
        }
    }
}

impl fmt::Display for CapabilityEnvelopeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
