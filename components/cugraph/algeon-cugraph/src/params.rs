//! Strongly-typed parameter wrappers for cuGraph algorithm configuration.
//!
//! These new types enforce invariants at construction time, making it impossible
//! to pass invalid values (e.g., zero iteration counts, negative scaling
//! ratios) to algorithm functions.
//!
//! # Overview
//!
//! | Type | Constraint | Use Cases |
//! |------|-----------|-----------|
//! | [`IterationCount`] | `> 0` (u32) | `max_iter`, `max_iterations` |
//! | [`BfsDepthLimit`] | `>= 0` (usize) | BFS `depth_limit` |
//! | [`PositiveF64`] | `> 0.0`, finite | `scaling_ratio` |
//! | [`UnitF64`] | `∈ (0.0, 1.0)`, finite | *(currently unused: kept for future algorithms)* |
//! | [`NonNegativeF64`] | `>= 0.0`, finite | `epsilon`, `edge_weight_influence`, `gravity`, `jitter_tolerance` |
//! | [`ClosedUnitF64`] | `∈ [0.0, 1.0]`, finite | `alpha` (damping factor), `barnes_hut_theta` |
//! | [`FractionF64`] | `∈ (0.0, 1.0]`, finite | ECG `threshold` and `resolution` |
//! | [`HalfOpenUnitF64`] | `∈ [0.0, 1.0)`, finite | spectral `evs_tolerance` |
//! | [`ClusterCount`] | `>= 2` (usize) | spectral `n_clusters` |
//!
//! # Examples
//!
//! ```
//! use cugraph::params::{IterationCount, PositiveF64, UnitF64};
//!
//! let iters = IterationCount::new(100).unwrap();
//! assert_eq!(iters.get(), 100);
//!
//! let eps = PositiveF64::new(1e-5).unwrap();
//! assert_eq!(eps.get(), 1e-5);
//!
//! let alpha = UnitF64::new(0.85).unwrap();
//! assert_eq!(alpha.get(), 0.85);
//! ```

use std::{fmt, num::NonZeroU32};

use crate::error::{Error, ErrorKind, Result};

pub(crate) mod defaults {
    use super::{ClosedUnitF64, FractionF64, IterationCount, NonNegativeF64, NonZeroU32, PositiveF64};

    pub(crate) const PAGERANK_ALPHA: ClosedUnitF64 = ClosedUnitF64(0.85);
    pub(crate) const PAGERANK_EPSILON: NonNegativeF64 = NonNegativeF64(1e-5);
    pub(crate) const PAGERANK_MAX_ITERATIONS: IterationCount = IterationCount(NonZeroU32::new(100).unwrap());

    pub(crate) const FORCE_ATLAS2_MAX_ITER: IterationCount = IterationCount(NonZeroU32::new(500).unwrap());
    pub(crate) const FORCE_ATLAS2_OVERLAP_SCALING_RATIO: PositiveF64 = PositiveF64(2.0);
    pub(crate) const FORCE_ATLAS2_EDGE_WEIGHT_INFLUENCE: NonNegativeF64 = NonNegativeF64(1.0);
    pub(crate) const FORCE_ATLAS2_JITTER_TOLERANCE: NonNegativeF64 = NonNegativeF64(1.0);
    pub(crate) const FORCE_ATLAS2_BARNES_HUT_THETA: ClosedUnitF64 = ClosedUnitF64(0.5);
    pub(crate) const FORCE_ATLAS2_SCALING_RATIO: PositiveF64 = PositiveF64(2.0);
    pub(crate) const FORCE_ATLAS2_GRAVITY: NonNegativeF64 = NonNegativeF64(1.0);

    pub(crate) const LOUVAIN_MAX_LEVEL: IterationCount = IterationCount(NonZeroU32::new(100).unwrap());
    pub(crate) const LOUVAIN_THRESHOLD: NonNegativeF64 = NonNegativeF64(1e-7);
    pub(crate) const LOUVAIN_RESOLUTION: PositiveF64 = PositiveF64(1.0);

    pub(crate) const ECG_THRESHOLD: FractionF64 = FractionF64(1e-7);
    pub(crate) const ECG_RESOLUTION: FractionF64 = FractionF64(1.0);
}

/// A strictly positive iteration count.
///
/// Wraps a [`NonZeroU32`] to guarantee that iteration-count parameters (like
/// `max_iter`) are always at least 1. This prevents silent no-ops from passing
/// zero iterations to algorithm functions.
///
/// # Construction
///
/// ```
/// use cugraph::params::IterationCount;
///
/// // From a runtime value (fallible):
/// let count = IterationCount::new(500).unwrap();
/// assert_eq!(count.get(), 500);
///
/// // TryFrom conversions:
/// let count: IterationCount = 100u32.try_into().unwrap();
/// let count: IterationCount = 100i32.try_into().unwrap();
/// let count: IterationCount = 100usize.try_into().unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IterationCount(NonZeroU32);

impl IterationCount {
    /// Create a new `IterationCount` from a `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is zero.
    pub fn new(value: u32) -> Result<Self> {
        NonZeroU32::new(value).map(IterationCount).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "IterationCount::new")
                .with_message("iteration count must be greater than zero")
        })
    }

    /// Create a new `IterationCount` from a [`NonZeroU32`] (infallible).
    pub const fn from_non_zero(value: NonZeroU32) -> Self {
        IterationCount(value)
    }

    /// Get the underlying value as a `u32`.
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Get the underlying [`NonZeroU32`].
    pub const fn into_inner(self) -> NonZeroU32 {
        self.0
    }
}

impl fmt::Display for IterationCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<IterationCount> for u32 {
    fn from(count: IterationCount) -> u32 {
        count.get()
    }
}

impl From<IterationCount> for usize {
    fn from(count: IterationCount) -> usize {
        count.get() as usize
    }
}

impl TryFrom<IterationCount> for i32 {
    type Error = Error;

    fn try_from(count: IterationCount) -> Result<i32> {
        i32::try_from(count.get()).map_err(|_| {
            Error::new(ErrorKind::InvalidInput, "IterationCount::try_into<i32>")
                .with_message("iteration count exceeds i32::MAX")
                .with_context("value", count.get().to_string())
        })
    }
}

impl TryFrom<u32> for IterationCount {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<i32> for IterationCount {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        if value <= 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "IterationCount::try_from<i32>")
                .with_message("iteration count must be greater than zero")
                .with_context("value", value.to_string()));
        }
        Self::new(value as u32)
    }
}

impl TryFrom<usize> for IterationCount {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self> {
        let val_u32 = u32::try_from(value).map_err(|_| {
            Error::new(ErrorKind::InvalidInput, "IterationCount::try_from<usize>")
                .with_message("iteration count exceeds u32::MAX")
                .with_context("value", value.to_string())
        })?;
        Self::new(val_u32)
    }
}

/// A BFS depth limit: the number of hops to expand from the sources.
///
/// `0` expands no hops and reaches only the sources. Use `None` at the API
/// boundary to request unlimited traversal. A limit larger than the graph's
/// vertex type can index behaves as unlimited.
///
/// # Construction
///
/// ```
/// use cugraph::params::BfsDepthLimit;
///
/// let limit = BfsDepthLimit::new(3);
/// assert_eq!(limit.get(), 3);
///
/// assert_eq!(BfsDepthLimit::new(0).get(), 0);
/// assert!(BfsDepthLimit::try_from(-1i32).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BfsDepthLimit(usize);

impl BfsDepthLimit {
    /// Create a new `BfsDepthLimit` from a `usize`.
    pub const fn new(value: usize) -> Self {
        BfsDepthLimit(value)
    }

    /// Get the underlying value as `usize`.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for BfsDepthLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<BfsDepthLimit> for usize {
    fn from(limit: BfsDepthLimit) -> usize {
        limit.get()
    }
}

impl From<usize> for BfsDepthLimit {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl From<u32> for BfsDepthLimit {
    fn from(value: u32) -> Self {
        Self::new(value as usize)
    }
}

impl TryFrom<i32> for BfsDepthLimit {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        usize::try_from(value).map(Self::new).map_err(|_| {
            Error::new(ErrorKind::InvalidInput, "BfsDepthLimit::try_from<i32>")
                .with_message("BFS depth limit must not be negative")
                .with_context("value", value.to_string())
        })
    }
}

/// A strictly positive, finite `f64` value.
///
/// Used for parameters that must be greater than zero, such as scaling ratios.
///
/// # Construction
///
/// ```
/// use cugraph::params::PositiveF64;
///
/// let eps = PositiveF64::new(1e-5).unwrap();
/// assert_eq!(eps.get(), 1e-5);
///
/// // Zero and negative values are rejected:
/// assert!(PositiveF64::new(0.0).is_err());
/// assert!(PositiveF64::new(-1.0).is_err());
///
/// // Infinity and NaN are rejected:
/// assert!(PositiveF64::new(f64::INFINITY).is_err());
/// assert!(PositiveF64::new(f64::NAN).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PositiveF64(f64);

impl PositiveF64 {
    /// Create a new `PositiveF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is not finite and
    /// positive (i.e., `<= 0.0`, `NaN`, or infinity).
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::new(ErrorKind::InvalidInput, "PositiveF64::new")
                .with_message("value must be a finite number greater than zero")
                .with_context("value", format!("{value}")));
        }
        Ok(PositiveF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for PositiveF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<PositiveF64> for f64 {
    fn from(val: PositiveF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for PositiveF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A finite `f64` value in the open interval `(0.0, 1.0)`.
///
/// Used for parameters that represent fractions or probabilities, such as the
/// PageRank damping factor (`alpha`) and Barnes-Hut trade-off (`theta`).
///
/// # Construction
///
/// ```
/// use cugraph::params::UnitF64;
///
/// let alpha = UnitF64::new(0.85).unwrap();
/// assert_eq!(alpha.get(), 0.85);
///
/// // Boundary values are rejected (open interval):
/// assert!(UnitF64::new(0.0).is_err());
/// assert!(UnitF64::new(1.0).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UnitF64(f64);

impl UnitF64 {
    /// Create a new `UnitF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is not in the open
    /// interval `(0.0, 1.0)` or is not finite.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(Error::new(ErrorKind::InvalidInput, "UnitF64::new")
                .with_message("value must be in the open interval (0.0, 1.0)")
                .with_context("value", format!("{value}")));
        }
        Ok(UnitF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for UnitF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<UnitF64> for f64 {
    fn from(val: UnitF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for UnitF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A non-negative, finite `f64` value (i.e., `>= 0.0`).
///
/// Used for parameters that may be zero but not negative, such as
/// `edge_weight_influence` (where 0.0 means "no influence").
///
/// # Construction
///
/// ```
/// use cugraph::params::NonNegativeF64;
///
/// let influence = NonNegativeF64::new(0.0).unwrap();
/// assert_eq!(influence.get(), 0.0);
///
/// let influence = NonNegativeF64::new(1.5).unwrap();
/// assert_eq!(influence.get(), 1.5);
///
/// // Negative values are rejected:
/// assert!(NonNegativeF64::new(-0.1).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegativeF64(f64);

impl NonNegativeF64 {
    /// Create a new `NonNegativeF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is negative, `NaN`, or
    /// infinite.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::new(ErrorKind::InvalidInput, "NonNegativeF64::new")
                .with_message("value must be a finite non-negative number")
                .with_context("value", format!("{value}")));
        }
        Ok(NonNegativeF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for NonNegativeF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NonNegativeF64> for f64 {
    fn from(val: NonNegativeF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for NonNegativeF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A finite `f64` value in the closed interval `[0.0, 1.0]`.
///
/// Used for parameters that represent interpolation weights or trade-offs
/// where the boundary values are valid (e.g., Barnes-Hut theta where 0.0 means
/// "maximum accuracy" and 1.0 means "maximum speed").
///
/// # Construction
///
/// ```
/// use cugraph::params::ClosedUnitF64;
///
/// let theta = ClosedUnitF64::new(0.5).unwrap();
/// assert_eq!(theta.get(), 0.5);
///
/// // Boundary values are accepted (closed interval):
/// assert!(ClosedUnitF64::new(0.0).is_ok());
/// assert!(ClosedUnitF64::new(1.0).is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ClosedUnitF64(f64);

impl ClosedUnitF64 {
    /// Create a new `ClosedUnitF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is not in the closed
    /// interval `[0.0, 1.0]` or is not finite.
    pub fn new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::new(ErrorKind::InvalidInput, "ClosedUnitF64::new")
                .with_message("value must be in the closed interval [0.0, 1.0]")
                .with_context("value", format!("{value}")));
        }
        Ok(ClosedUnitF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for ClosedUnitF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ClosedUnitF64> for f64 {
    fn from(val: ClosedUnitF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for ClosedUnitF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A finite `f64` value in the half-open interval `(0.0, 1.0]`.
///
/// Used for parameters libcugraph requires to be positive and at most one,
/// such as the ECG convergence `threshold` and `resolution`.
///
/// # Construction
///
/// ```
/// use cugraph::params::FractionF64;
///
/// assert_eq!(FractionF64::new(1.0).unwrap().get(), 1.0);
///
/// // Zero and values above one are rejected:
/// assert!(FractionF64::new(0.0).is_err());
/// assert!(FractionF64::new(1.5).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FractionF64(f64);

impl FractionF64 {
    /// Create a new `FractionF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is not in the interval
    /// `(0.0, 1.0]` or is not finite.
    pub fn new(value: f64) -> Result<Self> {
        if !(value > 0.0 && value <= 1.0) {
            return Err(Error::new(ErrorKind::InvalidInput, "FractionF64::new")
                .with_message("value must be in the interval (0.0, 1.0]")
                .with_context("value", format!("{value}")));
        }
        Ok(FractionF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for FractionF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<FractionF64> for f64 {
    fn from(val: FractionF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for FractionF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A finite `f64` value in the half-open interval `[0.0, 1.0)`.
///
/// Used for solver tolerances libcugraph requires to be non-negative and
/// below one, such as the spectral clustering eigensolver and k-means
/// tolerances.
///
/// # Construction
///
/// ```
/// use cugraph::params::HalfOpenUnitF64;
///
/// assert_eq!(HalfOpenUnitF64::new(0.0).unwrap().get(), 0.0);
///
/// // One and values above it are rejected:
/// assert!(HalfOpenUnitF64::new(1.0).is_err());
/// assert!(HalfOpenUnitF64::new(-0.1).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct HalfOpenUnitF64(f64);

impl HalfOpenUnitF64 {
    /// Create a new `HalfOpenUnitF64` from an `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is not in the interval
    /// `[0.0, 1.0)` or is not finite.
    pub fn new(value: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(Error::new(ErrorKind::InvalidInput, "HalfOpenUnitF64::new")
                .with_message("value must be in the interval [0.0, 1.0)")
                .with_context("value", format!("{value}")));
        }
        Ok(HalfOpenUnitF64(value))
    }

    /// Get the underlying `f64` value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for HalfOpenUnitF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<HalfOpenUnitF64> for f64 {
    fn from(val: HalfOpenUnitF64) -> f64 {
        val.get()
    }
}

impl TryFrom<f64> for HalfOpenUnitF64 {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        Self::new(value)
    }
}

/// A cluster count of at least two.
///
/// libcugraph's spectral clustering rejects a single cluster, so the smallest
/// valid partition count is two.
///
/// # Construction
///
/// ```
/// use cugraph::params::ClusterCount;
///
/// assert_eq!(ClusterCount::new(2).unwrap().get(), 2);
/// assert!(ClusterCount::new(1).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterCount(usize);

impl ClusterCount {
    /// Create a new `ClusterCount` from a `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidInput`] if `value` is below two.
    pub fn new(value: usize) -> Result<Self> {
        if value < 2 {
            return Err(Error::new(ErrorKind::InvalidInput, "ClusterCount::new")
                .with_message("cluster count must be at least 2")
                .with_context("value", value.to_string()));
        }
        Ok(ClusterCount(value))
    }

    /// Get the underlying value as `usize`.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for ClusterCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ClusterCount> for usize {
    fn from(count: ClusterCount) -> usize {
        count.get()
    }
}

impl TryFrom<usize> for ClusterCount {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn iteration_count_accepts_positive_values() {
        assert_eq!(IterationCount::new(1).unwrap().get(), 1);
        assert_eq!(IterationCount::new(100).unwrap().get(), 100);
        assert_eq!(IterationCount::new(u32::MAX).unwrap().get(), u32::MAX);
    }

    #[test]
    fn iteration_count_rejects_zero() {
        let err = IterationCount::new(0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("greater than zero"));
    }

    #[test]
    fn iteration_count_try_from_u32() {
        let count: IterationCount = 42u32.try_into().unwrap();
        assert_eq!(count.get(), 42);

        assert!(IterationCount::try_from(0u32).is_err());
    }

    #[test]
    fn iteration_count_try_from_i32() {
        let count: IterationCount = 500i32.try_into().unwrap();
        assert_eq!(count.get(), 500);

        let err = IterationCount::try_from(0i32).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);

        let err = IterationCount::try_from(-1i32).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn iteration_count_try_from_usize() {
        let count: IterationCount = 1000usize.try_into().unwrap();
        assert_eq!(count.get(), 1000);

        assert!(IterationCount::try_from(0usize).is_err());

        // Value exceeding u32::MAX
        let huge = u32::MAX as usize + 1;
        let err = IterationCount::try_from(huge).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("u32::MAX"));
    }

    #[test]
    fn iteration_count_from_non_zero() {
        let nz = NonZeroU32::new(7).unwrap();
        let count = IterationCount::from_non_zero(nz);
        assert_eq!(count.get(), 7);
    }

    #[test]
    fn iteration_count_into_primitives() {
        let count = IterationCount::new(42).unwrap();
        assert_eq!(u32::from(count), 42);
        assert_eq!(usize::from(count), 42);
        assert_eq!(i32::try_from(count).unwrap(), 42);
    }

    #[test]
    fn iteration_count_try_into_i32_rejects_overflow() {
        let count = IterationCount::new(u32::MAX).unwrap();
        let err = i32::try_from(count).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn iteration_count_display() {
        let count = IterationCount::new(123).unwrap();
        assert_eq!(count.to_string(), "123");
    }

    #[test]
    fn iteration_count_ordering() {
        let a = IterationCount::new(10).unwrap();
        let b = IterationCount::new(20).unwrap();
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, IterationCount::new(10).unwrap());
    }

    #[test]
    fn iteration_count_into_inner() {
        let count = IterationCount::new(42).unwrap();
        let nz: NonZeroU32 = count.into_inner();
        assert_eq!(nz.get(), 42);
    }

    #[test]
    fn bfs_depth_limit_accepts_zero_as_the_sources_only_limit() {
        assert_eq!(BfsDepthLimit::new(0).get(), 0);
        assert_eq!(BfsDepthLimit::new(100).get(), 100);
    }

    #[test]
    fn bfs_depth_limit_try_from_i32_rejects_negative_values() {
        let limit: BfsDepthLimit = 7i32.try_into().unwrap();
        assert_eq!(limit.get(), 7);
        assert_eq!(BfsDepthLimit::try_from(0i32).unwrap().get(), 0);

        let err = BfsDepthLimit::try_from(-1i32).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn bfs_depth_limit_into_primitives() {
        let limit = BfsDepthLimit::new(42);
        assert_eq!(usize::from(limit), 42);
        assert_eq!(BfsDepthLimit::from(9usize).get(), 9);
        assert_eq!(BfsDepthLimit::from(5u32).get(), 5);
    }

    #[test]
    fn bfs_depth_limit_display() {
        let limit = BfsDepthLimit::new(12);
        assert_eq!(limit.to_string(), "12");
    }

    #[test]
    fn positive_f64_accepts_positive_finite_values() {
        assert_eq!(PositiveF64::new(0.001).unwrap().get(), 0.001);
        assert_eq!(PositiveF64::new(1.0).unwrap().get(), 1.0);
        assert_eq!(PositiveF64::new(1e10).unwrap().get(), 1e10);
        assert_eq!(PositiveF64::new(f64::MIN_POSITIVE).unwrap().get(), f64::MIN_POSITIVE);
    }

    #[test]
    fn positive_f64_rejects_zero() {
        let err = PositiveF64::new(0.0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("greater than zero"));
    }

    #[test]
    fn positive_f64_rejects_negative() {
        let err = PositiveF64::new(-1.0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn positive_f64_rejects_infinity() {
        assert!(PositiveF64::new(f64::INFINITY).is_err());
        assert!(PositiveF64::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn positive_f64_rejects_nan() {
        assert!(PositiveF64::new(f64::NAN).is_err());
    }

    #[test]
    fn positive_f64_try_from_f64() {
        let val: PositiveF64 = 2.5f64.try_into().unwrap();
        assert_eq!(val.get(), 2.5);

        assert!(PositiveF64::try_from(0.0).is_err());
    }

    #[test]
    fn positive_f64_into_f64() {
        let val = PositiveF64::new(3.15).unwrap();
        assert_eq!(f64::from(val), 3.15);
    }

    #[test]
    fn positive_f64_display() {
        let val = PositiveF64::new(2.5).unwrap();
        assert_eq!(val.to_string(), "2.5");
    }

    #[test]
    fn positive_f64_ordering() {
        let a = PositiveF64::new(1.0).unwrap();
        let b = PositiveF64::new(2.0).unwrap();
        assert!(a < b);
    }

    #[test]
    fn unit_f64_accepts_values_in_open_interval() {
        assert_eq!(UnitF64::new(0.5).unwrap().get(), 0.5);
        assert_eq!(UnitF64::new(0.001).unwrap().get(), 0.001);
        assert_eq!(UnitF64::new(0.999).unwrap().get(), 0.999);
        assert_eq!(UnitF64::new(0.85).unwrap().get(), 0.85);
    }

    #[test]
    fn unit_f64_rejects_zero() {
        let err = UnitF64::new(0.0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("(0.0, 1.0)"));
    }

    #[test]
    fn unit_f64_rejects_one() {
        let err = UnitF64::new(1.0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn unit_f64_rejects_negative() {
        assert!(UnitF64::new(-0.1).is_err());
    }

    #[test]
    fn unit_f64_rejects_greater_than_one() {
        assert!(UnitF64::new(1.01).is_err());
    }

    #[test]
    fn unit_f64_rejects_nan_and_infinity() {
        assert!(UnitF64::new(f64::NAN).is_err());
        assert!(UnitF64::new(f64::INFINITY).is_err());
        assert!(UnitF64::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn unit_f64_try_from_f64() {
        let val: UnitF64 = 0.5f64.try_into().unwrap();
        assert_eq!(val.get(), 0.5);

        assert!(UnitF64::try_from(0.0).is_err());
        assert!(UnitF64::try_from(1.0).is_err());
    }

    #[test]
    fn unit_f64_into_f64() {
        let val = UnitF64::new(0.85).unwrap();
        assert_eq!(f64::from(val), 0.85);
    }

    #[test]
    fn unit_f64_display() {
        let val = UnitF64::new(0.85).unwrap();
        assert_eq!(val.to_string(), "0.85");
    }

    #[test]
    fn non_negative_f64_accepts_zero() {
        assert_eq!(NonNegativeF64::new(0.0).unwrap().get(), 0.0);
    }

    #[test]
    fn non_negative_f64_accepts_positive() {
        assert_eq!(NonNegativeF64::new(1.5).unwrap().get(), 1.5);
        assert_eq!(NonNegativeF64::new(100.0).unwrap().get(), 100.0);
    }

    #[test]
    fn non_negative_f64_rejects_negative() {
        let err = NonNegativeF64::new(-0.001).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("non-negative"));
    }

    #[test]
    fn non_negative_f64_rejects_nan_and_infinity() {
        assert!(NonNegativeF64::new(f64::NAN).is_err());
        assert!(NonNegativeF64::new(f64::INFINITY).is_err());
        assert!(NonNegativeF64::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn non_negative_f64_try_from_f64() {
        let val: NonNegativeF64 = 0.0f64.try_into().unwrap();
        assert_eq!(val.get(), 0.0);

        assert!(NonNegativeF64::try_from(-1.0).is_err());
    }

    #[test]
    fn non_negative_f64_into_f64() {
        let val = NonNegativeF64::new(2.5).unwrap();
        assert_eq!(f64::from(val), 2.5);
    }

    #[test]
    fn non_negative_f64_display() {
        let val = NonNegativeF64::new(0.0).unwrap();
        assert_eq!(val.to_string(), "0");
    }

    #[test]
    fn closed_unit_f64_accepts_boundaries() {
        assert_eq!(ClosedUnitF64::new(0.0).unwrap().get(), 0.0);
        assert_eq!(ClosedUnitF64::new(1.0).unwrap().get(), 1.0);
        assert_eq!(ClosedUnitF64::new(0.5).unwrap().get(), 0.5);
    }

    #[test]
    fn closed_unit_f64_rejects_out_of_range() {
        assert!(ClosedUnitF64::new(-0.001).is_err());
        assert!(ClosedUnitF64::new(1.001).is_err());
    }

    #[test]
    fn closed_unit_f64_rejects_nan_and_infinity() {
        assert!(ClosedUnitF64::new(f64::NAN).is_err());
        assert!(ClosedUnitF64::new(f64::INFINITY).is_err());
    }

    #[test]
    fn closed_unit_f64_try_from_f64() {
        let val: ClosedUnitF64 = 0.5f64.try_into().unwrap();
        assert_eq!(val.get(), 0.5);

        assert!(ClosedUnitF64::try_from(-0.1).is_err());
        assert!(ClosedUnitF64::try_from(1.1).is_err());
    }

    #[test]
    fn closed_unit_f64_into_f64() {
        let val = ClosedUnitF64::new(0.5).unwrap();
        assert_eq!(f64::from(val), 0.5);
    }

    #[test]
    fn closed_unit_f64_display() {
        let val = ClosedUnitF64::new(0.5).unwrap();
        assert_eq!(val.to_string(), "0.5");
    }

    #[test]
    fn fraction_f64_accepts_the_upper_bound_but_not_zero() {
        assert_eq!(FractionF64::new(1.0).unwrap().get(), 1.0);
        assert_eq!(FractionF64::new(f64::MIN_POSITIVE).unwrap().get(), f64::MIN_POSITIVE);
        assert!(FractionF64::new(0.0).is_err());
    }

    #[test]
    fn fraction_f64_rejects_out_of_range_and_non_finite_values() {
        assert!(FractionF64::new(-0.1).is_err());
        assert!(FractionF64::new(1.0 + f64::EPSILON).is_err());
        assert!(FractionF64::new(f64::NAN).is_err());
        assert!(FractionF64::new(f64::INFINITY).is_err());
        assert_eq!(FractionF64::new(2.0).unwrap_err().kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn half_open_unit_f64_accepts_zero_but_not_one() {
        assert_eq!(HalfOpenUnitF64::new(0.0).unwrap().get(), 0.0);
        assert_eq!(HalfOpenUnitF64::new(0.999).unwrap().get(), 0.999);
        assert!(HalfOpenUnitF64::new(1.0).is_err());
    }

    #[test]
    fn half_open_unit_f64_rejects_out_of_range_and_non_finite_values() {
        assert!(HalfOpenUnitF64::new(-f64::MIN_POSITIVE).is_err());
        assert!(HalfOpenUnitF64::new(f64::NAN).is_err());
        assert!(HalfOpenUnitF64::new(f64::INFINITY).is_err());
        assert_eq!(HalfOpenUnitF64::new(2.0).unwrap_err().kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn cluster_count_requires_at_least_two_clusters() {
        assert_eq!(ClusterCount::new(2).unwrap().get(), 2);
        assert!(ClusterCount::new(1).is_err());
        assert_eq!(ClusterCount::new(0).unwrap_err().kind(), ErrorKind::InvalidInput);
    }
}
