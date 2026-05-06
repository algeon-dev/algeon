use crate::{
    core::{
        error::{Error, ErrorCode, Result},
        types::NullEquality,
    },
    frame::{dataframe::DataFrame, series::Series},
    ops::join::{HashJoin, HashJoinOptions, NullableJoin},
    runtime::ExecutionPlacement,
};

// libcudf join APIs return INT32 row indices (`cudf::size_type`).
const JOIN_GATHER_MAP_INDEX_SIZE_BYTES: usize = std::mem::size_of::<i32>();

fn join_gather_map_reservation_size_bytes(
    output_rows: usize,
    map_count: usize,
    operation: &'static str,
) -> Result<usize> {
    output_rows
        .checked_mul(JOIN_GATHER_MAP_INDEX_SIZE_BYTES)
        .and_then(|single_map_bytes| single_map_bytes.checked_mul(map_count))
        .ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "join gather-map reservation size overflowed usize")
                .with_detail("reason", "size_overflowed_usize")
                .with_detail("output_rows", output_rows.to_string())
                .with_detail("gather_map_count", map_count.to_string())
                .with_detail("index_size_bytes", JOIN_GATHER_MAP_INDEX_SIZE_BYTES.to_string())
        })
}

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

/// Named column projection for [`DataFrame::cross_join_projected`](super::DataFrame::cross_join_projected).
///
/// The result schema is `left_columns` followed by `right_columns`. Output
/// names must be unique across both sides.
#[derive(Debug, Clone)]
pub struct DataFrameCrossJoinOptions {
    left_columns: Vec<String>,
    right_columns: Vec<String>,
    left_output_names: Vec<String>,
    right_output_names: Vec<String>,
}

impl Default for DataFrameCrossJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrameCrossJoinOptions {
    /// Create empty projected cross-join options.
    pub fn new() -> Self {
        Self {
            left_columns: Vec::new(),
            right_columns: Vec::new(),
            left_output_names: Vec::new(),
            right_output_names: Vec::new(),
        }
    }

    /// Set the left-side output columns in result order.
    pub fn left_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self.left_output_names = self.left_columns.clone();
        self
    }

    /// Set the right-side output columns in result order.
    pub fn right_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self.right_output_names = self.right_columns.clone();
        self
    }

    /// Set left-side source columns and output names in result order.
    pub fn left_projection<I, S, O>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = (S, O)>,
        S: AsRef<str>,
        O: AsRef<str>,
    {
        let (source, output): (Vec<_>, Vec<_>) = columns
            .into_iter()
            .map(|(source, output)| (source.as_ref().to_string(), output.as_ref().to_string()))
            .unzip();
        self.left_columns = source;
        self.left_output_names = output;
        self
    }

    /// Set right-side source columns and output names in result order.
    pub fn right_projection<I, S, O>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = (S, O)>,
        S: AsRef<str>,
        O: AsRef<str>,
    {
        let (source, output): (Vec<_>, Vec<_>) = columns
            .into_iter()
            .map(|(source, output)| (source.as_ref().to_string(), output.as_ref().to_string()))
            .unzip();
        self.right_columns = source;
        self.right_output_names = output;
        self
    }

    /// Returns the configured left-side output columns.
    pub fn left_column_names(&self) -> &[String] {
        &self.left_columns
    }

    /// Returns the configured right-side output columns.
    pub fn right_column_names(&self) -> &[String] {
        &self.right_columns
    }

    /// Returns the configured left-side output names.
    pub fn left_output_names(&self) -> &[String] {
        &self.left_output_names
    }

    /// Returns the configured right-side output names.
    pub fn right_output_names(&self) -> &[String] {
        &self.right_output_names
    }
}

/// Named predicate-table columns for DataFrame conditional join helpers.
///
/// The public [`crate::ops::ast::AstExpression`] still uses positional column
/// references. Those positions are resolved against the left and right column
/// lists in this option, so `AstExpression::left_column(0)` refers to the
/// first configured left predicate column.
#[derive(Debug, Clone)]
pub struct DataFramePredicateJoinOptions {
    left_columns: Vec<String>,
    right_columns: Vec<String>,
}

impl Default for DataFramePredicateJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFramePredicateJoinOptions {
    /// Create empty predicate-column options.
    pub fn new() -> Self {
        Self { left_columns: Vec::new(), right_columns: Vec::new() }
    }

    /// Set the left-side columns visible to the AST predicate.
    pub fn left_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self
    }

    /// Set the right-side columns visible to the AST predicate.
    pub fn right_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self
    }

    /// Returns the configured left-side predicate columns.
    pub fn left_column_names(&self) -> &[String] {
        &self.left_columns
    }

    /// Returns the configured right-side predicate columns.
    pub fn right_column_names(&self) -> &[String] {
        &self.right_columns
    }
}

/// Named equality-key and predicate-table columns for DataFrame mixed joins.
///
/// Equality keys use the same `on` vs. `left_on`/`right_on` modes as
/// [`DataFrameJoinOptions`]. Predicate columns define the left/right tables
/// referenced by the supplied [`crate::ops::ast::AstExpression`].
#[derive(Debug, Clone)]
pub struct DataFrameMixedJoinOptions {
    on: Option<Vec<String>>,
    left_on: Option<Vec<String>>,
    right_on: Option<Vec<String>>,
    left_predicate_columns: Vec<String>,
    right_predicate_columns: Vec<String>,
    null_eq: NullEquality,
}

impl Default for DataFrameMixedJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrameMixedJoinOptions {
    /// Create mixed-join options with null keys comparing equal by default.
    pub fn new() -> Self {
        Self {
            on: None,
            left_on: None,
            right_on: None,
            left_predicate_columns: Vec::new(),
            right_predicate_columns: Vec::new(),
            null_eq: NullEquality::Equal,
        }
    }

    /// Set shared equality-key column names present on both sides.
    pub fn on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set left-side equality-key column names.
    pub fn left_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set right-side equality-key column names.
    pub fn right_on<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_on = Some(keys.into_iter().map(|key| key.as_ref().to_string()).collect());
        self
    }

    /// Set the left-side columns visible to the residual AST predicate.
    pub fn left_predicate_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.left_predicate_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self
    }

    /// Set the right-side columns visible to the residual AST predicate.
    pub fn right_predicate_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.right_predicate_columns = columns.into_iter().map(|column| column.as_ref().to_string()).collect();
        self
    }

    /// Set how null equality keys compare.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Returns the shared equality keys, if configured.
    pub fn on_keys(&self) -> Option<&[String]> {
        self.on.as_deref()
    }

    /// Returns the left-side equality keys, if configured.
    pub fn left_on_keys(&self) -> Option<&[String]> {
        self.left_on.as_deref()
    }

    /// Returns the right-side equality keys, if configured.
    pub fn right_on_keys(&self) -> Option<&[String]> {
        self.right_on.as_deref()
    }

    /// Returns the configured left-side predicate columns.
    pub fn left_predicate_column_names(&self) -> &[String] {
        &self.left_predicate_columns
    }

    /// Returns the configured right-side predicate columns.
    pub fn right_predicate_column_names(&self) -> &[String] {
        &self.right_predicate_columns
    }

    /// Returns the null key comparison policy.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }
}

/// DataFrame-level options for building a reusable hash-join artifact.
#[derive(Debug, Clone, Copy)]
pub struct DataFrameHashJoinOptions {
    null_eq: NullEquality,
    nullable_join: NullableJoin,
    load_factor: f64,
}

impl Default for DataFrameHashJoinOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrameHashJoinOptions {
    /// Create conservative hash-join build options.
    pub fn new() -> Self {
        Self { null_eq: NullEquality::Equal, nullable_join: NullableJoin::MaybeNull, load_factor: 0.5 }
    }

    /// Set how null join keys compare.
    pub fn null_equality(mut self, null_eq: NullEquality) -> Self {
        self.null_eq = null_eq;
        self
    }

    /// Set whether build/probe key columns may contain nulls.
    pub fn nullable_join(mut self, nullable_join: NullableJoin) -> Self {
        self.nullable_join = nullable_join;
        self
    }

    /// Declare that all build/probe key columns are non-nullable.
    pub fn no_nulls(self) -> Self {
        self.nullable_join(NullableJoin::NoNulls)
    }

    /// Set the libcudf hash-table load factor.
    pub fn load_factor(mut self, load_factor: f64) -> Self {
        self.load_factor = load_factor;
        self
    }

    /// Returns the configured null equality setting.
    pub fn null_eq(&self) -> NullEquality {
        self.null_eq
    }

    /// Returns the nullable-join assumption.
    pub fn nullable_join_setting(&self) -> NullableJoin {
        self.nullable_join
    }

    /// Returns the configured hash-table load factor.
    pub fn load_factor_value(&self) -> f64 {
        self.load_factor
    }

    pub(crate) fn to_hash_join_options(self) -> HashJoinOptions {
        HashJoinOptions::new()
            .null_equality(self.null_eq)
            .nullable_join(self.nullable_join)
            .load_factor(self.load_factor)
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

    /// Returns bytes retained by this device-side gather map.
    pub fn device_allocated_size_bytes(&self) -> crate::core::error::Result<usize> {
        self.indices
            .device_allocated_size_bytes()
            .map_err(|error| error.reanchor("dataframe::join_indices::device_allocated_size_bytes"))
    }

    /// Return exact bytes that must be reserved for one device-side join gather
    /// map with `output_rows` INT32 row indices.
    ///
    /// This is a pre-allocation planning contract for bounded engines. It does
    /// not allocate the gather map.
    pub fn reservation_size_bytes_for_len(output_rows: usize) -> Result<usize> {
        join_gather_map_reservation_size_bytes(
            output_rows,
            1,
            "dataframe::join_indices::reservation_size_bytes_for_len",
        )
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

    /// Returns bytes retained by both device-side gather maps.
    pub fn device_allocated_size_bytes(&self) -> crate::core::error::Result<usize> {
        Ok(self.left.device_allocated_size_bytes()?.saturating_add(self.right.device_allocated_size_bytes()?))
    }

    /// Return exact bytes that must be reserved for a pair of device-side join
    /// gather maps with `output_rows` INT32 row indices per side.
    ///
    /// This is a pre-allocation planning contract for bounded engines. It does
    /// not allocate either gather map.
    pub fn reservation_size_bytes_for_len(output_rows: usize) -> Result<usize> {
        join_gather_map_reservation_size_bytes(
            output_rows,
            2,
            "dataframe::join_pair_indices::reservation_size_bytes_for_len",
        )
    }
}

/// Device-side row-index pairs produced by probing a reusable DataFrame hash join.
///
/// The `probe` gather map indexes the probe DataFrame passed to the probe call.
/// The `build` gather map indexes the retained build DataFrame inside the
/// [`DataFrameHashJoin`]. For outer probes, unmatched rows contain libcudf's
/// no-match sentinel and should be materialized through
/// [`DataFrameHashJoin::materialize_probe`] or
/// [`DataFrameHashJoin::materialize_build`].
#[derive(Debug, Clone)]
pub struct DataFrameHashJoinPairIndices {
    probe: DataFrameJoinIndices,
    build: DataFrameJoinIndices,
}

impl DataFrameHashJoinPairIndices {
    pub(crate) fn new(probe: Series, build: Series) -> Self {
        Self { probe: DataFrameJoinIndices::new(probe), build: DataFrameJoinIndices::new(build) }
    }

    /// Borrow the gather map for rows in the probe DataFrame.
    pub fn probe(&self) -> &DataFrameJoinIndices {
        &self.probe
    }

    /// Borrow the gather map for rows in the retained build DataFrame.
    pub fn build(&self) -> &DataFrameJoinIndices {
        &self.build
    }

    /// Number of output rows represented by this pair of gather maps.
    pub fn len(&self) -> usize {
        self.probe.len()
    }

    /// Returns `true` when this pair represents no output rows.
    pub fn is_empty(&self) -> bool {
        self.probe.is_empty()
    }

    /// Returns bytes retained by both device-side gather maps.
    pub fn device_allocated_size_bytes(&self) -> crate::core::error::Result<usize> {
        Ok(self.probe.device_allocated_size_bytes()?.saturating_add(self.build.device_allocated_size_bytes()?))
    }

    /// Return exact bytes that must be reserved for a reusable hash-join probe
    /// result with `output_rows` probe/build INT32 gather-map pairs.
    ///
    /// This is a pre-allocation planning contract for bounded engines. It does
    /// not probe the hash table or allocate either gather map.
    pub fn reservation_size_bytes_for_len(output_rows: usize) -> Result<usize> {
        join_gather_map_reservation_size_bytes(
            output_rows,
            2,
            "dataframe::hash_join_pair_indices::reservation_size_bytes_for_len",
        )
    }
}

/// Materialized probe/build sides from a reusable DataFrame hash-join probe.
#[derive(Debug, Clone)]
pub struct DataFrameHashJoinMaterialized {
    /// Projected rows gathered from the probe DataFrame.
    pub probe: DataFrame,
    /// Projected rows gathered from the retained build DataFrame.
    pub build: DataFrame,
}

/// Reusable DataFrame-level hash-join build artifact.
///
/// This type owns the build-side [`DataFrame`] and a low-level reusable
/// [`HashJoin`] built from the named build key columns. Probe methods produce
/// device-side gather maps only; payload columns are copied only when the
/// materialization helpers are called.
#[derive(Debug, Clone)]
pub struct DataFrameHashJoin {
    pub(crate) join: HashJoin,
    pub(crate) build: DataFrame,
    pub(crate) build_keys: Vec<String>,
    pub(crate) options: DataFrameHashJoinOptions,
}

impl DataFrameHashJoin {
    /// Borrow the retained build DataFrame.
    pub fn build(&self) -> &DataFrame {
        &self.build
    }

    /// Returns the resolved build key column names.
    pub fn build_keys(&self) -> &[String] {
        &self.build_keys
    }

    /// Returns the options used to construct this artifact.
    pub fn options(&self) -> DataFrameHashJoinOptions {
        self.options
    }

    /// Returns bytes retained by this reusable hash-join build artifact.
    ///
    /// This excludes the retained build DataFrame and any per-probe
    /// output/index allocations.
    pub fn device_allocated_size_bytes(&self) -> crate::core::error::Result<usize> {
        self.join
            .device_allocated_size_bytes()
            .map_err(|error| error.reanchor("dataframe::hash_join::device_allocated_size_bytes"))
    }

    /// Return exact bytes that must be reserved for a reusable hash-join probe
    /// result with `output_rows` probe/build INT32 gather-map pairs.
    ///
    /// This ergonomic helper delegates to
    /// [`DataFrameHashJoinPairIndices::reservation_size_bytes_for_len`]. The
    /// retained build metadata does not affect per-probe gather-map bytes, and
    /// this method does not probe the hash table or allocate either gather map.
    pub fn probe_gather_map_reservation_size_bytes(&self, output_rows: usize) -> Result<usize> {
        DataFrameHashJoinPairIndices::reservation_size_bytes_for_len(output_rows)
            .map_err(|error| error.reanchor("dataframe::hash_join::probe_gather_map_reservation_size_bytes"))
    }
}
