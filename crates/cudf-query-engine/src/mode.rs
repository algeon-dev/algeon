//! Native pipeline execution-mode labels.
//!
//! These labels describe the bounded pipeline execution contract, not the
//! final result placement. Result placement is represented by `QueryOutputMode`
//! in the execution layer.
#![deny(missing_docs)]

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
