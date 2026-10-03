pub use crate::core::types::NullEquality;

/// Whether the build table or any later probe table can contain null join keys.
///
/// `MaybeNull` is the conservative default. Use `NoNulls` only when every
/// build/probe key table used with a persistent
/// [`DataFrameHashJoin`](crate::DataFrameHashJoin) is known to be non-nullable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullableJoin {
    /// At least one build/probe key table may contain nulls.
    MaybeNull,
    /// All build/probe key tables are known not to contain nulls.
    NoNulls,
}

impl NullableJoin {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_nullable_join_t {
        match self {
            NullableJoin::MaybeNull => cudf_sys::CUDF_NULLABLE_JOIN_YES,
            NullableJoin::NoNulls => cudf_sys::CUDF_NULLABLE_JOIN_NO,
        }
    }
}

/// Options for reusable duplicate-key [`HashJoin`][crate::ops::join::HashJoin]
/// build artifacts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HashJoinOptions {
    nullable_join: NullableJoin,
    null_eq: NullEquality,
    load_factor: f64,
}

impl Default for HashJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl HashJoinOptions {
    /// Create options using conservative null handling, nulls-equal comparison,
    /// and libcudf's standard 0.5 hash-table load factor.
    pub fn new() -> Self {
        Self { nullable_join: NullableJoin::MaybeNull, null_eq: NullEquality::Equal, load_factor: 0.5 }
    }

    /// Set whether build/probe key tables can contain nulls.
    pub fn nullable_join(mut self, nullable_join: NullableJoin) -> Self {
        self.nullable_join = nullable_join;
        self
    }

    /// Set how null key values are compared.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Set the hash-table load factor.
    ///
    /// libcudf accepts values in `(0, 1]`; invalid values are rejected at the
    /// FFI boundary with an invalid-argument error.
    pub fn load_factor(mut self, load_factor: f64) -> Self {
        self.load_factor = load_factor;
        self
    }

    /// Returns the nullable-join assumption.
    pub fn nullable_join_setting(&self) -> NullableJoin {
        self.nullable_join
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }

    /// Returns the configured hash-table load factor.
    pub fn load_factor_value(&self) -> f64 {
        self.load_factor
    }
}

/// Options for reusable [`FilteredJoin`][crate::ops::join::FilteredJoin]
/// build artifacts used by equality left semi/anti probes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilteredJoinOptions {
    null_eq: NullEquality,
    load_factor: f64,
}

impl Default for FilteredJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl FilteredJoinOptions {
    /// Create options using nulls-equal comparison and libcudf's standard 0.5
    /// hash-table load factor.
    pub fn new() -> Self {
        Self { null_eq: NullEquality::Equal, load_factor: 0.5 }
    }

    /// Set how null key values are compared.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Set the hash-table load factor.
    ///
    /// libcudf accepts values in `(0, 1]`; invalid values are rejected at the
    /// FFI boundary with an invalid-argument error.
    pub fn load_factor(mut self, load_factor: f64) -> Self {
        self.load_factor = load_factor;
        self
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }

    /// Returns the configured hash-table load factor.
    pub fn load_factor_value(&self) -> f64 {
        self.load_factor
    }
}

/// The type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Only rows with matching keys in both sides.
    Inner,
    /// All left rows preserved; right side null-padded for non-matches.
    Left,
    /// All rows from both sides; null-padded where no match.
    Full,
}

/// Options for `DataFrame::merge`, the cuDF-style unified join entry point.
///
/// Use the builder methods to configure the merge. At minimum, either `on()`
/// or both `left_on()` and `right_on()` must be set before calling `merge`.
///
/// # Example
///
/// ```no_run
/// use cudf::{JoinType, MergeOptions};
///
/// let opts = MergeOptions::new()
///     .on(&["id"])
///     .how(JoinType::Left)
///     .suffixes("_x", "_y");
/// ```
#[derive(Debug, Clone)]
pub struct MergeOptions {
    join_type: JoinType,
    on: Option<Vec<String>>,
    left_on: Option<Vec<String>>,
    right_on: Option<Vec<String>>,
    null_eq: NullEquality,
    suffix_left: String,
    suffix_right: String,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeOptions {
    /// Create a new `MergeOptions` with default settings.
    ///
    /// Defaults: inner join, suffixes `("_left", "_right")`, null equality `Equal`.
    pub fn new() -> Self {
        Self {
            join_type: JoinType::Inner,
            on: None,
            left_on: None,
            right_on: None,
            null_eq: NullEquality::Equal,
            suffix_left: "_left".to_string(),
            suffix_right: "_right".to_string(),
        }
    }

    /// Set shared key column names (both sides must have these names).
    pub fn on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set left-side key column names.
    pub fn left_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set right-side key column names.
    pub fn right_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_on = Some(keys.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Set the join type.
    pub fn how(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Shorthand for `how(JoinType::Inner)`.
    pub fn how_inner(self) -> Self {
        self.how(JoinType::Inner)
    }

    /// Shorthand for `how(JoinType::Left)`.
    pub fn how_left(self) -> Self {
        self.how(JoinType::Left)
    }

    /// Shorthand for `how(JoinType::Full)`.
    pub fn how_full(self) -> Self {
        self.how(JoinType::Full)
    }

    /// Set how null keys are compared.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Set suffixes for conflicting payload column names.
    pub fn suffixes(mut self, left: &str, right: &str) -> Self {
        self.suffix_left = left.to_string();
        self.suffix_right = right.to_string();
        self
    }

    /// Returns the configured join type.
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Returns the shared `on` key names, if set.
    pub fn on_keys(&self) -> Option<&[String]> {
        self.on.as_deref()
    }

    /// Returns the left-side key names, if set.
    pub fn left_on_keys(&self) -> Option<&[String]> {
        self.left_on.as_deref()
    }

    /// Returns the right-side key names, if set.
    pub fn right_on_keys(&self) -> Option<&[String]> {
        self.right_on.as_deref()
    }

    /// Returns the configured left suffix.
    pub fn suffix_left(&self) -> &str {
        &self.suffix_left
    }

    /// Returns the configured right suffix.
    pub fn suffix_right(&self) -> &str {
        &self.suffix_right
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }
}
