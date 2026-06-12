//! `Series` string namespace for GPU string operations.

use super::{context::default_execution_context, owned::Series};
use crate::{
    core::error::Result,
    ops::strings::{self, StripSide},
    runtime::ExecutionOverrides,
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
        self.0.string_unary_with_context(strings::to_lower_impl, &default_execution_context(), "series::str::lower")
    }

    /// Convert each string element to lowercase using the explicit execution-context override.
    pub fn lower_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_unary_with_context(strings::to_lower_impl, ctx, "series::str::lower_in")
    }

    /// Convert each string element to uppercase. Inherits the source Series' name.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn upper(&self) -> Result<Series> {
        self.0.string_unary_with_context(strings::to_upper_impl, &default_execution_context(), "series::str::upper")
    }

    /// Convert each string element to uppercase using the explicit execution-context override.
    pub fn upper_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_unary_with_context(strings::to_upper_impl, ctx, "series::str::upper_in")
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
        self.0.string_scalar_in_context(
            pat,
            &default_execution_context(),
            "series::str::contains",
            strings::str_contains_impl,
        )
    }

    /// Element-wise literal substring containment test using the explicit execution-context override.
    pub fn contains_in(&self, pat: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_scalar_in_context(pat, ctx, "series::str::contains_in", strings::str_contains_impl)
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
        self.0.string_scalar_in_context(
            pat,
            &default_execution_context(),
            "series::str::starts_with",
            strings::starts_with_impl,
        )
    }

    /// Element-wise starts-with test using the explicit execution-context override.
    pub fn starts_with_in(&self, pat: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_scalar_in_context(pat, ctx, "series::str::starts_with_in", strings::starts_with_impl)
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
        self.0.string_scalar_in_context(
            pat,
            &default_execution_context(),
            "series::str::ends_with",
            strings::ends_with_impl,
        )
    }

    /// Element-wise ends-with test using the explicit execution-context override.
    pub fn ends_with_in(&self, pat: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_scalar_in_context(pat, ctx, "series::str::ends_with_in", strings::ends_with_impl)
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

    /// Strip whitespace from both sides of each string using the explicit execution-context override.
    pub fn strip_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.strip_chars_in(StripSide::Both, "", ctx)
    }

    /// Strip whitespace from the left (beginning) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.lstrip()`.
    pub fn lstrip(&self) -> Result<Series> {
        self.strip_chars(StripSide::Left, "")
    }

    /// Strip whitespace from the left of each string using the explicit execution-context override.
    pub fn lstrip_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.strip_chars_in(StripSide::Left, "", ctx)
    }

    /// Strip whitespace from the right (end) of each string.
    ///
    /// Equivalent to cuDF Python's `Series.str.rstrip()`.
    pub fn rstrip(&self) -> Result<Series> {
        self.strip_chars(StripSide::Right, "")
    }

    /// Strip whitespace from the right of each string using the explicit execution-context override.
    pub fn rstrip_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.strip_chars_in(StripSide::Right, "", ctx)
    }

    /// Strip specific characters from the given side(s) of each string.
    ///
    /// `to_strip` is the set of characters to remove. An empty string
    /// means strip whitespace.
    pub fn strip_chars(&self, side: StripSide, to_strip: &str) -> Result<Series> {
        self.0.strip_in_context(side, to_strip, &default_execution_context(), "series::str::strip_chars")
    }

    /// Strip specific characters using the explicit execution-context override.
    pub fn strip_chars_in(&self, side: StripSide, to_strip: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.strip_in_context(side, to_strip, ctx, "series::str::strip_chars_in")
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
        self.0.string_unary_with_context(
            strings::count_characters_impl,
            &default_execution_context(),
            "series::str::len",
        )
    }

    /// Character count per string using the explicit execution-context override.
    pub fn len_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_unary_with_context(strings::count_characters_impl, ctx, "series::str::len_in")
    }

    /// Byte count per string.
    ///
    /// Returns an `INT32` Series. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new INT32 column.
    pub fn byte_count(&self) -> Result<Series> {
        self.0.string_unary_with_context(
            strings::count_bytes_impl,
            &default_execution_context(),
            "series::str::byte_count",
        )
    }

    /// Byte count per string using the explicit execution-context override.
    pub fn byte_count_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_unary_with_context(strings::count_bytes_impl, ctx, "series::str::byte_count_in")
    }
}
