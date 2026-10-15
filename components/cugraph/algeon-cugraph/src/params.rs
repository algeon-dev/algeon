//! Strongly-typed parameter wrappers for cuGraph algorithm configuration.
//!
//! Enforces range, non-zero, and finiteness invariants at construction time.
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
/// `max_iter`) are always at least 1. This prevents no-ops that finish without an
/// error from passing zero iterations to algorithm functions.
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
/// Upstream libcugraph spectral clustering requires at least two clusters.
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
