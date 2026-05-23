//! `Series` string namespace for GPU string operations.

use super::{context::default_execution_context, owned::Series};
use crate::{
    core::{error::Result, types::DataType},
    ops::strings::{self, StringSliceOptions, StripSide},
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

    /// Convert each string element to lowercase.
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

    /// Convert each string element to uppercase.
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
    /// Null input elements produce null output.
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

    /// SQL `LIKE` pattern match.
    ///
    /// Returns a `BOOL8` Series. `%` matches zero or more characters, `_`
    /// matches one character, and null input elements produce null output.
    /// This is not a regex API.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column. The selected stream is
    /// synchronized before returning because libcudf temporarily borrows the
    /// host pattern bytes.
    pub fn like(&self, pattern: &str) -> Result<Series> {
        self.0.like_in_context(pattern, None, &default_execution_context(), "series::str::like")
    }

    /// SQL `LIKE` pattern match using an explicit escape character.
    ///
    /// `escape` must be empty or a single ASCII character.
    pub fn like_escape(&self, pattern: &str, escape: &str) -> Result<Series> {
        self.0.like_in_context(
            pattern,
            if escape.is_empty() { None } else { Some(escape) },
            &default_execution_context(),
            "series::str::like_escape",
        )
    }

    /// SQL `LIKE` pattern match using the explicit execution-context override.
    pub fn like_in(&self, pattern: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.like_in_context(pattern, None, ctx, "series::str::like_in")
    }

    /// SQL `LIKE` pattern match with an escape character and explicit execution-context override.
    pub fn like_escape_in(&self, pattern: &str, escape: &str, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.like_in_context(
            pattern,
            if escape.is_empty() { None } else { Some(escape) },
            ctx,
            "series::str::like_escape_in",
        )
    }

    /// Element-wise starts-with test.
    ///
    /// Returns a `BOOL8` Series.
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
    /// Returns a `BOOL8` Series.
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

    // -- Slice ---------------------------------------------------------------

    /// Slice each string using zero-based character positions.
    ///
    /// `StringSliceOptions::start` and `stop` use a half-open `[start, stop)`
    /// range over Unicode character positions, not bytes. `None` leaves a
    /// bound open. Null input elements produce null output.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn slice(&self, options: StringSliceOptions) -> Result<Series> {
        self.0.string_slice_in_context(options, &default_execution_context(), "series::str::slice")
    }

    /// Slice each string using the explicit execution-context override.
    pub fn slice_in(&self, options: StringSliceOptions, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.string_slice_in_context(options, ctx, "series::str::slice_in")
    }

    // -- Strip / trim -------------------------------------------------------

    /// Strip whitespace from both sides of each string.
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
    /// Returns an `INT32` Series.
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

    // -- Numeric conversion (TRY_CAST string -> integer) -------------------

    /// Parse each string element as an integer of `output_type`, returning
    /// null for rows that do not parse cleanly or overflow the target type.
    /// Null input rows stay null. Mirrors SQL `TRY_CAST(s AS <int_type>)`.
    ///
    /// `output_type` must be an integral data type.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column of `output_type`.
    pub fn try_to_integers(&self, output_type: &DataType) -> Result<Series> {
        self.0.try_to_integers_in_context(output_type, &default_execution_context(), "series::str::try_to_integers")
    }

    /// `try_to_integers` using the explicit execution-context override.
    pub fn try_to_integers_in(&self, output_type: &DataType, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.0.try_to_integers_in_context(output_type, ctx, "series::str::try_to_integers_in")
    }
}
