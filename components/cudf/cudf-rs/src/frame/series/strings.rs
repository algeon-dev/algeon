//! `Series` string namespace for GPU string operations.

use super::owned::Series;
use crate::{
    core::{error::Result, scalar::Scalar},
    ops::strings::{self, StripSide},
};

impl Series {
    /// Enter the string namespace for string-specific series methods.
    pub fn str(&self) -> SeriesStringNamespace<'_> {
        SeriesStringNamespace(self)
    }
}

/// String namespace for [`Series`].
#[must_use]
pub struct SeriesStringNamespace<'series>(&'series Series);

impl SeriesStringNamespace<'_> {
    // -- Case conversion ---------------------------------------------------

    /// Convert each string element to lowercase. Inherits the source Series' name.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn lower(&self) -> Result<Series> {
        let cv = self.0.column.view()?;
        let col = strings::to_lower(&cv)?;
        self.0.with_inherited_name(col)
    }

    /// Convert each string element to uppercase. Inherits the source Series' name.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn upper(&self) -> Result<Series> {
        let cv = self.0.column.view()?;
        let col = strings::to_upper(&cv)?;
        self.0.with_inherited_name(col)
    }

    // -- Find / contains / starts_with / ends_with -------------------------

    /// Element-wise literal substring containment test.
    ///
    /// Returns a `BOOL8` Series: `true` where the string contains `pat`.
    /// Null input elements produce null output. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.contains(pat, regex=False)`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn contains(&self, pat: &str) -> Result<Series> {
        let cv = self.0.column.view()?;
        let scalar = Scalar::from_str_in_domain(pat, self.0.column.execution_domain())?;
        let col = strings::str_contains(&cv, &scalar)?;
        self.0.with_inherited_name(col)
    }

    /// Element-wise starts-with test.
    ///
    /// Returns a `BOOL8` Series. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.startswith(pat)`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn starts_with(&self, pat: &str) -> Result<Series> {
        let cv = self.0.column.view()?;
        let scalar = Scalar::from_str_in_domain(pat, self.0.column.execution_domain())?;
        let col = strings::starts_with(&cv, &scalar)?;
        self.0.with_inherited_name(col)
    }

    /// Element-wise ends-with test.
    ///
    /// Returns a `BOOL8` Series. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.endswith(pat)`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn ends_with(&self, pat: &str) -> Result<Series> {
        let cv = self.0.column.view()?;
        let scalar = Scalar::from_str_in_domain(pat, self.0.column.execution_domain())?;
        let col = strings::ends_with(&cv, &scalar)?;
        self.0.with_inherited_name(col)
    }

    // -- Strip / trim -------------------------------------------------------

    /// Strip whitespace from both sides of each string.
    ///
    /// Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.strip()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn strip(&self) -> Result<Series> {
        self.strip_chars(StripSide::Both, "")
    }

    /// Strip whitespace from the left (beginning) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.lstrip()`.
    pub fn lstrip(&self) -> Result<Series> {
        self.strip_chars(StripSide::Left, "")
    }

    /// Strip whitespace from the right (end) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.rstrip()`.
    pub fn rstrip(&self) -> Result<Series> {
        self.strip_chars(StripSide::Right, "")
    }

    /// Strip specific characters from the given side(s) of each string.
    ///
    /// `to_strip` is the set of characters to remove. An empty string
    /// means strip whitespace.
    pub fn strip_chars(&self, side: StripSide, to_strip: &str) -> Result<Series> {
        let cv = self.0.column.view()?;
        let scalar = Scalar::from_str_in_domain(to_strip, self.0.column.execution_domain())?;
        let col = strings::strip(&cv, side, &scalar)?;
        self.0.with_inherited_name(col)
    }

    // -- Attributes ---------------------------------------------------------

    /// Character count (not byte count) per string.
    ///
    /// Returns an `INT32` Series. Null input elements produce null output.
    /// Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.len()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new INT32 column.
    pub fn len(&self) -> Result<Series> {
        let cv = self.0.column.view()?;
        let col = strings::count_characters(&cv)?;
        self.0.with_inherited_name(col)
    }

    /// Byte count per string.
    ///
    /// Returns an `INT32` Series. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new INT32 column.
    pub fn byte_count(&self) -> Result<Series> {
        let cv = self.0.column.view()?;
        let col = strings::count_bytes(&cv)?;
        self.0.with_inherited_name(col)
    }
}
