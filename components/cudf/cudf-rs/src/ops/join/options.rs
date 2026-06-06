pub use crate::core::types::NullEquality;

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
