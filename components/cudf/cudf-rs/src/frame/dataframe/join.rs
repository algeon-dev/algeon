use crate::{core::types::NullEquality, frame::series::Series, runtime::ExecutionPlacement};

/// DataFrame-level equality join key options for query-planning helpers.
///
/// This option shape intentionally excludes join type and suffixes. It is for
/// operations whose result schema is fixed by the helper name, such as
/// [`DataFrame::left_semi_join`](super::DataFrame::left_semi_join) and
/// [`DataFrame::left_anti_join`](super::DataFrame::left_anti_join).
#[derive(Debug, Clone)]
pub struct DataFrameJoinOptions {
    on: Option<Vec<String>>,
    left_on: Option<Vec<String>>,
    right_on: Option<Vec<String>>,
    null_eq: NullEquality,
}

impl Default for DataFrameJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrameJoinOptions {
    /// Create join options with null keys comparing equal by default.
    pub fn new() -> Self {
        Self { on: None, left_on: None, right_on: None, null_eq: NullEquality::Equal }
    }

    /// Set shared key column names present on both sides.
    pub fn on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set left-side key column names.
    pub fn left_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set right-side key column names.
    pub fn right_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set how null join keys compare.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Returns the shared key names, if configured.
    pub fn on_keys(&self) -> Option<&[String]> {
        self.on.as_deref()
    }

    /// Returns the left-side key names, if configured.
    pub fn left_on_keys(&self) -> Option<&[String]> {
        self.left_on.as_deref()
    }

    /// Returns the right-side key names, if configured.
    pub fn right_on_keys(&self) -> Option<&[String]> {
        self.right_on.as_deref()
    }

    /// Returns the null key comparison policy.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }
}

/// Device-side row indices produced by a DataFrame-level join-index helper.
///
/// The wrapped series is an `INT32` gather map over the left DataFrame rows.
/// Producing this value allocates only the index column. Payload data is copied
/// only when the indices are passed to
/// [`DataFrame::materialize_join_indices`](super::DataFrame::materialize_join_indices).
///
/// Cloning this wrapper is cheap: it clones the wrapped [`Series`] handle and
/// shares the same device-side index buffer.
#[derive(Debug, Clone)]
pub struct DataFrameJoinIndices {
    indices: Series,
}

impl DataFrameJoinIndices {
    pub(crate) fn new(indices: Series) -> Self {
        Self { indices }
    }

    /// Borrow the row-index gather map.
    pub fn indices(&self) -> &Series {
        &self.indices
    }

    /// Consume this wrapper and return the underlying row-index series.
    pub fn into_indices(self) -> Series {
        self.indices
    }

    /// Number of output rows represented by this gather map.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` when this gather map contains no rows.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Retained execution placement of the device-side gather map.
    ///
    /// This describes the index buffer itself, not the source left DataFrame
    /// that the indices may later gather from.
    pub fn placement(&self) -> ExecutionPlacement {
        self.indices.placement()
    }
}

/// Device-side row-index pairs produced by a two-sided DataFrame join.
///
/// The left and right gather maps are both `INT32` device columns. They index
/// rows in the left and right input DataFrames respectively. For outer joins,
/// unmatched rows contain libcudf's no-match sentinel; pass each side to
/// [`DataFrame::materialize_join_indices`](super::DataFrame::materialize_join_indices)
/// so the gather path nullifies unmatched payload rows without copying unused
/// columns.
///
/// Cloning this wrapper is cheap: it clones the wrapped [`Series`] handles and
/// shares the same device-side index buffers.
#[derive(Debug, Clone)]
pub struct DataFrameJoinPairIndices {
    left: DataFrameJoinIndices,
    right: DataFrameJoinIndices,
}

impl DataFrameJoinPairIndices {
    pub(crate) fn new(left: Series, right: Series) -> Self {
        Self { left: DataFrameJoinIndices::new(left), right: DataFrameJoinIndices::new(right) }
    }

    /// Borrow the gather map for rows in the left DataFrame.
    pub fn left(&self) -> &DataFrameJoinIndices {
        &self.left
    }

    /// Borrow the gather map for rows in the right DataFrame.
    pub fn right(&self) -> &DataFrameJoinIndices {
        &self.right
    }

    /// Consume this wrapper and return the left and right gather maps.
    pub fn into_sides(self) -> (DataFrameJoinIndices, DataFrameJoinIndices) {
        (self.left, self.right)
    }

    /// Number of output row pairs represented by these gather maps.
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Returns `true` when these gather maps contain no rows.
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// Retained execution placement of the device-side gather maps.
    ///
    /// This describes the index buffers themselves, not either source
    /// DataFrame that the indices may later gather from.
    pub fn placement(&self) -> ExecutionPlacement {
        self.left.placement()
    }
}
