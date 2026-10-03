//! `Series` string namespace for GPU string operations.

use super::{context::SeriesOps, owned::Series};
use crate::{
    core::{error::Result, types::DataType},
    ops::strings::{
        self, ReplaceWithBackrefsOutputSizePreflight, SeriesPreparedReplaceWithBackrefs, StringSliceOptions, StripSide,
    },
    runtime::{BudgetedExecutionOutcome, Exec, ExecutionBudgetToken},
};

impl<'a> SeriesOps<'a> {
    /// Enter the string namespace for string-specific series methods.
    pub fn str(&self) -> SeriesStringOps<'a> {
        SeriesStringOps { series: self.series, ctx: self.ctx }
    }
}

/// String operations of one [`Series`], run where an [`Exec`] selects.
#[must_use]
pub struct SeriesStringOps<'a> {
    series: &'a Series,
    ctx: Exec<'a>,
}

impl SeriesStringOps<'_> {
    // -- Case conversion ---------------------------------------------------

    /// Convert each string element to lowercase.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn lower(&self) -> Result<Series> {
        self.series.string_unary_with_context(strings::to_lower_impl, self.ctx, "series::str::lower")
    }

    /// Convert each string element to uppercase.
    ///
    /// The input must be a `STRING` column.
    ///
    /// # Cost
    /// GPU-allocating: produces a new STRING column.
    pub fn upper(&self) -> Result<Series> {
        self.series.string_unary_with_context(strings::to_upper_impl, self.ctx, "series::str::upper")
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
        self.series.string_scalar_in_context(pat, self.ctx, "series::str::contains", strings::str_contains_impl)
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
        self.series.like_in_context(pattern, None, self.ctx, "series::str::like")
    }

    /// SQL `LIKE` pattern match using an explicit escape character.
    ///
    /// `escape` must be empty or a single ASCII character.
    pub fn like_escape(&self, pattern: &str, escape: &str) -> Result<Series> {
        self.series.like_in_context(
            pattern,
            if escape.is_empty() { None } else { Some(escape) },
            self.ctx,
            "series::str::like_escape",
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
        self.series.string_scalar_in_context(pat, self.ctx, "series::str::starts_with", strings::starts_with_impl)
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
        self.series.string_scalar_in_context(pat, self.ctx, "series::str::ends_with", strings::ends_with_impl)
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
        self.series.string_slice_in_context(options, self.ctx, "series::str::slice")
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
        self.series.strip_in_context(side, to_strip, self.ctx, "series::str::strip_chars")
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
        self.series.string_unary_with_context(strings::count_characters_impl, self.ctx, "series::str::len")
    }

    /// Byte count per string.
    ///
    /// Returns an `INT32` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new INT32 column.
    pub fn byte_count(&self) -> Result<Series> {
        self.series.string_unary_with_context(strings::count_bytes_impl, self.ctx, "series::str::byte_count")
    }

    // -- Regex replace with back-references --------------------------------

    /// Exact retained-output and temporary-workspace preflight for replace-with-backrefs.
    pub fn replace_with_backrefs_output_size(
        &self,
        pattern: &str,
        replacement: &str,
    ) -> Result<ReplaceWithBackrefsOutputSizePreflight> {
        self.series.replace_with_backrefs_output_size_in_context(
            pattern,
            replacement,
            self.ctx,
            "series::str::replace_with_backrefs_output_size",
        )
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
        self.series.try_to_integers_in_context(output_type, self.ctx, "series::str::try_to_integers")
    }
}

impl Series {
    /// Prepare reusable replace-with-backrefs state after one size pass.
    ///
    /// The returned handle retains this series and the completed size-pass state.
    /// Prefer [`prepare_replace_with_backrefs_with_budget`](Self::prepare_replace_with_backrefs_with_budget)
    /// under bounded execution so retained state is charged to the operator token.
    pub fn prepare_replace_with_backrefs(
        &self,
        pattern: &str,
        replacement: &str,
    ) -> Result<SeriesPreparedReplaceWithBackrefs> {
        strings::prepare_replace_with_backrefs(self, pattern, replacement)
    }

    /// Prepare reusable replace-with-backrefs state under a hard budget token.
    pub fn prepare_replace_with_backrefs_with_budget(
        &self,
        pattern: &str,
        replacement: &str,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<SeriesPreparedReplaceWithBackrefs>> {
        strings::prepare_replace_with_backrefs_with_budget(self, pattern, replacement, token)
    }
}
