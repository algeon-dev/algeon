//! Native pipeline execution-mode labels.
//!
//! These labels describe the bounded pipeline execution contract, not the
//! final result placement. Result placement is represented by `QueryOutputMode`
//! in the execution layer.

use std::fmt;

/// Coarse native pipeline mode selected before admission or execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePipelineExecutionMode {
    /// Deterministic strict bounded execution with gate-first contract checks.
    StrictBounded,
    /// Adaptive execution with its own admission, actions, and report surface.
    Adaptive,
}

impl NativePipelineExecutionMode {
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

/// Strict bounded execution labels used by gate-first contract reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrictBoundedExecutionMode {
    /// Every reported gate is modeled for strict bounded execution.
    StrictBounded,
    /// At least one gate or capability proof blocks strict bounded execution.
    UnsupportedStrictBounded,
}

impl StrictBoundedExecutionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StrictBounded => "strict_bounded",
            Self::UnsupportedStrictBounded => "unsupported_strict_bounded",
        }
    }

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
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::UnsupportedAdaptive => "unsupported_adaptive",
        }
    }

    pub const fn native_mode(self) -> NativePipelineExecutionMode {
        NativePipelineExecutionMode::Adaptive
    }
}

impl fmt::Display for AdaptiveExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
