//! Native pipeline execution-mode labels.
//!
//! These labels describe the bounded pipeline execution contract, not the
//! final result placement. Result placement is represented by `QueryOutputMode`
//! in the execution layer.

use std::{fmt, str::FromStr};

/// Coarse native pipeline mode selected before admission or execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePipelineExecutionMode {
    /// Deterministic strict bounded execution with gate-first contract checks.
    #[default]
    StrictBounded,
    /// Adaptive execution with its own admission, actions, and report surface.
    Adaptive,
}

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

/// Backward-compatible alias for the original `STDDEV_SAMP` mode name.
pub type StddevSampExecutionMode = StatisticalAggregateExecutionMode;

/// Backward-compatible alias for the original `STDDEV_SAMP` parse error.
pub type StddevSampExecutionModeParseError = StatisticalAggregateExecutionModeParseError;

impl NativePipelineExecutionMode {
    /// Stable snake-case mode label used in env vars and reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StrictBounded => "strict_bounded",
            Self::Adaptive => "adaptive",
        }
    }
}

impl fmt::Display for NativePipelineExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NativePipelineExecutionMode {
    type Err = NativePipelineExecutionModeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "strict_bounded" | "strict" => Ok(Self::StrictBounded),
            "adaptive" => Ok(Self::Adaptive),
            other => Err(NativePipelineExecutionModeParseError { value: other.to_string() }),
        }
    }
}

/// Error returned when parsing a native pipeline execution-mode label fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePipelineExecutionModeParseError {
    value: String,
}

impl NativePipelineExecutionModeParseError {
    /// Returns the unrecognized label after ASCII-lowercase normalization.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for NativePipelineExecutionModeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid native pipeline execution mode: {}", self.value)
    }
}

impl std::error::Error for NativePipelineExecutionModeParseError {}

/// Strict bounded execution labels used by gate-first contract reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrictBoundedExecutionMode {
    /// Every reported gate is modeled for strict bounded execution.
    StrictBounded,
    /// At least one gate or capability proof blocks strict bounded execution.
    UnsupportedStrictBounded,
}

impl StrictBoundedExecutionMode {
    /// Stable snake-case strict bounded report label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StrictBounded => "strict_bounded",
            Self::UnsupportedStrictBounded => "unsupported_strict_bounded",
        }
    }

    /// Returns the coarse native pipeline mode for this strict report label.
    pub const fn native_mode(self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::StrictBounded
    }
}

impl fmt::Display for StrictBoundedExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Adaptive execution labels reserved for the adaptive report surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveExecutionMode {
    /// Adaptive execution was selected for this candidate.
    Adaptive,
    /// Adaptive mode was selected but the candidate was still unsupported.
    UnsupportedAdaptive,
}

impl AdaptiveExecutionMode {
    /// Stable snake-case adaptive report label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::UnsupportedAdaptive => "unsupported_adaptive",
        }
    }

    /// Returns the coarse native pipeline mode for this adaptive report label.
    pub const fn native_mode(self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::Adaptive
    }
}

impl fmt::Display for AdaptiveExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
