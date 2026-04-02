//! Series facade methods for GPU string operations.

use super::owned::Series;
use crate::{
    core::{error::Result, scalar::Scalar},
    ops::strings::{self, StripSide},
};

impl Series {
    // -- Case conversion ---------------------------------------------------

    /// Convert each string element to lowercase. Inherits the source Series' name.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn str_lower(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = strings::to_lower(&cv)?;
        self.with_inherited_name(col)
    }

    /// Convert each string element to uppercase. Inherits the source Series' name.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn str_upper(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = strings::to_upper(&cv)?;
        self.with_inherited_name(col)
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
    pub fn str_contains(&self, pat: &str) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = Scalar::from_string(pat)?;
        let col = strings::str_contains(&cv, &scalar)?;
        self.with_inherited_name(col)
    }

    /// Element-wise starts-with test.
    ///
    /// Returns a `BOOL8` Series. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.startswith(pat)`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn str_startswith(&self, pat: &str) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = Scalar::from_string(pat)?;
        let col = strings::starts_with(&cv, &scalar)?;
        self.with_inherited_name(col)
    }

    /// Element-wise ends-with test.
    ///
    /// Returns a `BOOL8` Series. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.str.endswith(pat)`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn str_endswith(&self, pat: &str) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = Scalar::from_string(pat)?;
        let col = strings::ends_with(&cv, &scalar)?;
        self.with_inherited_name(col)
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
    pub fn str_strip(&self) -> Result<Series> {
        self.str_strip_chars(StripSide::Both, "")
    }

    /// Strip whitespace from the left (beginning) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.lstrip()`.
    pub fn str_lstrip(&self) -> Result<Series> {
        self.str_strip_chars(StripSide::Left, "")
    }

    /// Strip whitespace from the right (end) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.rstrip()`.
    pub fn str_rstrip(&self) -> Result<Series> {
        self.str_strip_chars(StripSide::Right, "")
    }

    /// Strip specific characters from the given side(s) of each string.
    ///
    /// `to_strip` is the set of characters to remove. An empty string
    /// means strip whitespace.
    pub fn str_strip_chars(&self, side: StripSide, to_strip: &str) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = Scalar::from_string(to_strip)?;
        let col = strings::strip(&cv, side, &scalar)?;
        self.with_inherited_name(col)
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
    pub fn str_len(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = strings::count_characters(&cv)?;
        self.with_inherited_name(col)
    }

    /// Byte count per string.
    ///
    /// Returns an `INT32` Series. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new INT32 column.
    pub fn str_byte_count(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = strings::count_bytes(&cv)?;
        self.with_inherited_name(col)
    }
}
