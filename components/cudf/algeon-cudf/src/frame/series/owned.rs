use std::sync::Arc;

use super::context::default_execution_context;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
        scalar::IntoRuntimeScalar,
        types::{CudfType, DataType, TimestampResolution, TypeId},
    },
    frame::{
        dataframe::DataFrame,
        factorization::{Codebook, DictionaryEncodingOptions, Factorized},
    },
    ops::{binary::BinaryOperator, concatenate, replace::FillPolicy, sorting::SortOptions, unary::UnaryOperator},
    runtime::{
        BoundaryKind, BoundaryObservation, ExecutionPlacement, QueryStageKind, Runtime,
        execution_domain::ExecutionDomain, record_boundary_observation_in_domain,
    },
};

pub(super) const SERIES_VALUE_COUNTS_FALLBACK_NAME: &str = "value";
pub(super) const SERIES_VALUE_COUNTS_COUNT_NAME: &str = "count";

pub(super) fn unsupported_numeric_target_error(operation: &'static str, target: TypeId) -> Error {
    Error::new_with(ErrorCode::UnsupportedDtype, operation, format!("target type {target:?} is not numeric"))
        .with_fact(FACT_TYPE_ID, target)
}

/// A reference-counted GPU column vector.
///
/// `Series` is the cuDF-like user-facing column type. Internally it holds an
/// `Arc<Column>`, so cloning and passing Series between DataFrames is
/// zero-copy on the GPU side.
///
/// Metadata queries (`len()`, `is_empty()`, `null_count()`) are infallible and
/// free; the values are cached at construction time.
///
/// [`std::fmt::Display`] for `Series` is metadata-only. It reports the dtype,
/// length, and nullability summary, but it never materializes values to host
/// memory.
///
/// # Interop
///
/// Use a borrowed view for temporary device access, a prepared export when the
/// consumer needs access or a copy, and an owned transfer when it takes over
/// the device buffers.
#[derive(Clone)]
pub struct Series {
    pub(super) column: Arc<Column>,
    pub(super) cached_len: usize,
    pub(super) cached_null_count: usize,
}

impl TryFrom<Column> for Series {
    type Error = Error;

    fn try_from(col: Column) -> Result<Self> {
        Self::from_column(col)
    }
}

/// Trait for types that can be converted into a `Series`, possibly fallibly.
///
/// This is used by [`DataFrame::with_column`] to accept both `Series` and
/// `Column` transparently.
pub trait IntoSeries {
    /// Convert into a `Series`.
    fn into_series(self) -> Result<Series>;
}

impl IntoSeries for Series {
    fn into_series(self) -> Result<Series> {
        Ok(self)
    }
}

impl IntoSeries for Column {
    fn into_series(self) -> Result<Series> {
        Series::from_column(self)
    }
}

impl Series {
    pub(super) fn require_bool_binary_operands(&self, rhs: &Series, operation: &'static str) -> Result<()> {
        let lhs_type = self.dtype()?.id();
        let rhs_type = rhs.dtype()?.id();

        if lhs_type == TypeId::Bool8 && rhs_type == TypeId::Bool8 {
            return Ok(());
        }

        Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "logical boolean operations require both operands to have dtype Bool8",
        )
        .with_fact(FACT_LEFT_TYPE_ID, lhs_type)
        .with_fact(FACT_RIGHT_TYPE_ID, rhs_type))
    }

    /// Returns the data type of the underlying column.
    pub fn dtype(&self) -> Result<DataType> {
        self.column.view()?.data_type()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// Returns `true` if the series has zero elements.
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> usize {
        self.cached_null_count
    }

    /// Return a diagnostic summary of this series' retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.column.execution_domain().placement()
    }

    /// Returns bytes retained by this series' owned device column.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        self.column.device_allocated_size_bytes().map_err(|error| error.reanchor("series::device_allocated_size_bytes"))
    }

    // -- Access --------------------------------------------------------------

    /// Returns `true` if both series share the same retained provenance
    /// (CUDA stream and memory resource).
    pub fn same_provenance(&self, other: &Series) -> bool {
        ExecutionDomain::same_provenance(self.column.execution_domain(), other.column.execution_domain())
    }

    /// Advanced: drop to the core `Column` type.
    ///
    /// Returns a reference to the underlying `Column`. Use this when you need
    /// to call low-level column operations that are not exposed on `Series`.
    pub fn as_column(&self) -> &Column {
        &self.column
    }

    /// Returns a reference to the shared `Arc<Column>`.
    ///
    /// Crate-internal: used by `DataFrame` to clone the Arc into its column map.
    /// External callers should use [`into_inner`](Self::into_inner) to take
    /// ownership of the `Arc<Column>`.
    pub(in crate::frame) fn column_arc(&self) -> &Arc<Column> {
        &self.column
    }

    /// Clone and return the shared `Arc<Column>` without consuming the `Series`.
    ///
    /// This is the ownership-preserving accessor for callers that need to keep
    /// a retained handle to the underlying column while also keeping the
    /// `Series` wrapper alive.
    pub fn arc(&self) -> Arc<Column> {
        Arc::clone(&self.column)
    }

    /// Consume this Series and return the underlying `Arc<Column>`.
    ///
    /// This is the idiomatic way to unwrap a Series when you no longer need
    /// the Series wrapper.
    pub fn into_inner(self) -> Arc<Column> {
        self.column
    }

    pub(crate) fn view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    /// Return the first `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned.
    pub fn head(&self, n: usize) -> Result<Series> {
        self.head_with_context(n, &default_execution_context(), "series::head")
    }

    /// Return the last `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned.
    pub fn tail(&self, n: usize) -> Result<Series> {
        self.tail_with_context(n, &default_execution_context(), "series::tail")
    }

    /// Return `length` rows starting at `offset`.
    ///
    /// Both `offset` and `length` are clamped to the series length, so
    /// out-of-range values yield fewer rows rather than an error.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Series> {
        self.slice_with_context(offset, length, &default_execution_context(), "series::slice")
    }

    /// Gather rows by an index `Series`.
    ///
    /// The `indices` series must be `Int32`. Out-of-bounds indices produce
    /// null rows in the result.
    pub fn gather(&self, indices: &Series) -> Result<Series> {
        self.gather_with_context(indices, &default_execution_context(), "series::gather")
    }

    /// Filter rows using a BOOL8 mask.
    ///
    /// Rows where `mask` is `true` are kept; `false` and null mask rows are
    /// dropped.
    pub fn filter(&self, mask: &Series) -> Result<Series> {
        self.filter_with_context(mask, &default_execution_context(), "series::filter")
    }

    /// Drop null rows from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.dropna()`.
    pub fn dropna(&self) -> Result<Series> {
        self.dropna_with_context(&default_execution_context(), "series::dropna")
    }

    /// Remove duplicate values from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.drop_duplicates()`.
    pub fn drop_duplicates(&self) -> Result<Series> {
        self.drop_duplicates_with_context(&default_execution_context(), "series::drop_duplicates")
    }

    /// Count the frequency of each distinct non-null value.
    ///
    /// Returns a two-column [`DataFrame`] with one value column and one
    /// `count` column. Null values are excluded from the result.
    pub fn value_counts(&self) -> Result<DataFrame> {
        self.value_counts_with_context(&default_execution_context(), "series::value_counts")
    }

    /// Copy fixed-width column data to a host `Vec<T>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable; use
    /// [`to_optional_vec`](Self::to_optional_vec) instead when nulls are
    /// expected.
    ///
    /// # Cost
    /// Device-to-host copy: one synchronous memcpy of `len * size_of::<T>()` bytes.
    ///
    /// # Errors
    /// - `ErrorKind::InvalidInput` with `ErrorCode::NullUnexpected` if the
    ///   column is nullable.
    /// - `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    ///   column dtype does not match `T`.
    pub fn to_vec_strict<T: CudfType>(&self) -> Result<Vec<T>> {
        let cv = self.column.view()?;
        let null_count = cv.null_count()?;
        if null_count > 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                "series::to_vec_strict",
                "to_vec_strict rejects columns containing nulls",
            )
            .with_detail("null_count", null_count.to_string()));
        }
        cv.to_vec::<T>()
    }

    /// Copy fixed-width column data to host as `Vec<Option<T>>`, encoding
    /// null positions as `None`.
    ///
    /// Non-nullable columns return all `Some`. For nullable columns, this
    /// performs two separate device-to-host transfers (data buffer + null-mask
    /// buffer), each with its own synchronization point, then merges them on
    /// the host.
    ///
    /// # Cost
    /// Two synchronous device-to-host memcpys: `len * size_of::<T>()` bytes
    /// for data, plus `ceil(len / 8)` bytes for the null mask.
    ///
    /// # Errors
    /// Returns `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    /// column dtype does not match `T`.
    pub fn to_optional_vec<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        self.column.view()?.to_optional_vec::<T>()
    }

    /// Copy string column data to a host `Vec<String>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable; use
    /// [`to_optional_strings`](Self::to_optional_strings) or
    /// [`to_strings_lossy`](Self::to_strings_lossy) instead.
    ///
    /// # Errors
    /// - `ErrorKind::InvalidInput` with `ErrorCode::NullUnexpected` if the
    ///   column is nullable.
    /// - `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    ///   underlying column type is not `STRING`.
    pub fn to_strings_strict(&self) -> Result<Vec<String>> {
        let cv = self.column.view()?;
        let null_count = cv.null_count()?;
        if null_count > 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                "series::to_strings_strict",
                "to_strings_strict rejects columns containing nulls",
            )
            .with_detail("null_count", null_count.to_string()));
        }
        cv.to_strings_lossy()
    }

    /// Copy string data to host as `Vec<Option<String>>`, distinguishing null
    /// from empty string.
    ///
    /// Null positions return `None`; valid positions return `Some(s)`.
    /// Non-nullable columns return all `Some`.
    ///
    /// # Errors
    /// Returns `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    /// underlying column type is not `STRING`.
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        self.column.view()?.to_optional_strings()
    }

    /// Copy string column data to a host `Vec<String>`, mapping null elements
    /// to empty strings (`""`).
    ///
    /// This is a lossy conversion: null and genuinely empty strings become
    /// indistinguishable. When null fidelity matters, use
    /// [`to_optional_strings`](Self::to_optional_strings).
    ///
    /// # Errors
    /// Returns `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    /// underlying column type is not `STRING`.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        self.column.view()?.to_strings_lossy()
    }

    // -- Scalar comparison ops -----------------------------------------

    /// Element-wise `self > scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn gt_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Greater, &default_execution_context(), "series::gt_scalar")
    }

    /// Element-wise `self < scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn lt_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Less, &default_execution_context(), "series::lt_scalar")
    }

    /// Element-wise `self == scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn eq_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Equal, &default_execution_context(), "series::eq_scalar")
    }

    // -- Extended scalar comparison ops --------------------------------

    /// Element-wise `self != scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn ne_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::NotEqual, &default_execution_context(), "series::ne_scalar")
    }

    /// Element-wise `self >= scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn ge_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(
            rhs,
            BinaryOperator::GreaterEqual,
            &default_execution_context(),
            "series::ge_scalar",
        )
    }

    /// Element-wise `self <= scalar`.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn le_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::LessEqual, &default_execution_context(), "series::le_scalar")
    }

    // -- Column-vs-column comparison ops -------------------------------

    /// Element-wise `self == rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Equal, &default_execution_context(), "series::eq")
    }

    /// Element-wise `self != rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::NotEqual, &default_execution_context(), "series::ne")
    }

    /// Element-wise `self > rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Greater, &default_execution_context(), "series::gt")
    }

    /// Element-wise `self < rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Less, &default_execution_context(), "series::lt")
    }

    /// Element-wise `self >= rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::GreaterEqual, &default_execution_context(), "series::ge")
    }

    /// Element-wise `self <= rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::LessEqual, &default_execution_context(), "series::le")
    }

    /// Element-wise logical AND. Both operands must be `BOOL8`.
    ///
    /// Returns a `BOOL8` Series suitable for mask composition.
    pub fn logical_and(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_and";
        self.require_bool_binary_operands(rhs, operation)?;
        self.binary_series_in_context(rhs, BinaryOperator::LogicalAnd, &default_execution_context(), operation)
    }

    /// Element-wise logical OR. Both operands must be `BOOL8`.
    ///
    /// Returns a `BOOL8` Series suitable for mask composition.
    pub fn logical_or(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_or";
        self.require_bool_binary_operands(rhs, operation)?;
        self.binary_series_in_context(rhs, BinaryOperator::LogicalOr, &default_execution_context(), operation)
    }

    // -- Arithmetic ops ------------------------------------------------

    /// Element-wise `self + rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Add, &default_execution_context(), "series::add")
    }

    /// Element-wise `self - rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Sub, &default_execution_context(), "series::sub")
    }

    /// Element-wise `self * rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Mul, &default_execution_context(), "series::mul")
    }

    /// Element-wise `self / rhs`. Output type inferred.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Div, &default_execution_context(), "series::div")
    }

    // -- Scalar arithmetic ops -----------------------------------------

    /// Element-wise `self + scalar`.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Add, &default_execution_context(), "series::add_scalar")
    }

    /// Element-wise `self - scalar`.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Sub, &default_execution_context(), "series::sub_scalar")
    }

    /// Element-wise `self * scalar`.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Mul, &default_execution_context(), "series::mul_scalar")
    }

    /// Element-wise `self / scalar`.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Div, &default_execution_context(), "series::div_scalar")
    }

    /// Element-wise `self % rhs` (modulo). Output type inferred. Returns an
    /// `Series`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn rem(&self, rhs: &Series) -> Result<Series> {
        self.binary_series_in_context(rhs, BinaryOperator::Mod, &default_execution_context(), "series::rem")
    }

    /// Element-wise `self % scalar`.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn rem_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.binary_scalar_in_context(rhs, BinaryOperator::Mod, &default_execution_context(), "series::rem_scalar")
    }

    // -- Static constructors for combining Series ----------------------------

    /// Concatenate multiple series into a single `Series`.
    ///
    /// All series must have the same data type.
    ///
    /// # Errors
    /// Returns `ErrorKind::InvalidInput` if the input slice is empty.
    /// Returns an error if the series have mismatched dtypes.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column combining all input data.
    pub fn concat(series: &[&Series]) -> Result<Series> {
        if series.is_empty() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                "series::concat",
                "cannot concatenate empty slice of Series",
            )
            .with_detail("reason", "empty_input")
            .with_detail("remedy", "provide at least one input series"));
        }

        let views: Vec<_> = series
            .iter()
            .map(|s| s.column.view().map_err(|e| e.reanchor("series::concat")))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&ColumnView<'_>> = views.iter().collect();
        let col = concatenate::concatenate_columns(&view_refs).map_err(|e| e.reanchor("series::concat"))?;
        Self::from_column(col)
    }

    /// Ensure this Series is placed in the target runtime.
    ///
    /// If the underlying column already has the target runtime's retained
    /// provenance, this reuses the existing GPU allocation. Otherwise the
    /// column is fenced and deep-copied into the target runtime.
    pub fn copy_into_runtime(&self, runtime: &Runtime) -> Result<Series> {
        let should_copy = !ExecutionDomain::same_provenance(self.column.execution_domain(), runtime.execution_domain());
        let column =
            Self::clone_or_copy_into_domain(&self.column, runtime.execution_domain(), "series::copy_into_runtime")?;
        let series = Self::from_column_arc(column).map_err(|e| e.reanchor("series::copy_into_runtime"))?;
        if should_copy {
            record_boundary_observation_in_domain(
                runtime.execution_domain(),
                BoundaryObservation::new(BoundaryKind::DeviceToDeviceCopy, runtime.placement(), self.len(), 1, false)
                    .with_query_stage(QueryStageKind::OperatorOutput),
            );
        }
        Ok(series)
    }

    /// Clone or deep-copy an `Arc<Column>` into the target domain.
    fn clone_or_copy_into_domain(
        column: &Arc<Column>,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Arc<Column>> {
        if ExecutionDomain::same_provenance(column.execution_domain(), output_domain) {
            return Ok(Arc::clone(column));
        }

        // copy_into_domain installs both fence directions internally.
        column.copy_into_domain(Arc::clone(output_domain), operation).map(Arc::new)
    }

    // -- Convenience methods; unary ops inherit name ---------

    /// Return a BOOL8 mask: `true` where elements are null.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_null(&self) -> Result<Series> {
        self.is_null_with_context(&default_execution_context(), "series::is_null")
    }

    /// Return a BOOL8 mask: `true` where elements are valid (non-null).
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_not_null(&self) -> Result<Series> {
        self.is_not_null_with_context(&default_execution_context(), "series::is_not_null")
    }

    /// Cast this series to the given type.
    ///
    /// Accepts either a [`DataType`] (canonical, supports decimal scale) or a
    /// [`crate::TypeId`] (convenience shorthand, scale defaults to 0).
    ///
    /// Equivalent to cuDF Python's `Series.astype()`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the target dtype.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        self.cast_in_context(target, &default_execution_context(), "series::astype")
    }

    /// Cast this series to a timestamp type.
    ///
    /// The source series should contain integer values representing durations
    /// since the Unix epoch in the specified resolution. If `resolution` is
    /// `None`, defaults to microsecond resolution (matching cuDF Python's
    /// `to_datetime()` default).
    ///
    /// Equivalent to cuDF Python's `cudf.to_datetime(series)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the target timestamp dtype.
    pub fn to_datetime(&self, resolution: Option<TimestampResolution>) -> Result<Series> {
        self.to_datetime_with_context(
            resolution,
            &default_execution_context(),
            "series::to_datetime_duration",
            "series::to_datetime",
        )
    }

    /// Cast this series to a numeric type.
    ///
    /// `target` must be a numeric [`TypeId`] (integer, unsigned integer,
    /// float, or bool). Returns an error if `target` is a non-numeric type
    /// (string, timestamp, duration, etc.).
    ///
    /// Equivalent to cuDF Python's `cudf.to_numeric(series)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the target numeric dtype.
    pub fn to_numeric(&self, target: TypeId) -> Result<Series> {
        if !target.is_numeric() && !target.is_boolean() {
            return Err(unsupported_numeric_target_error("series::to_numeric", target));
        }
        self.cast_in_context(target, &default_execution_context(), "series::to_numeric")
    }

    /// Element-wise natural logarithm.
    ///
    /// Equivalent to cuDF Python's `Series.log()` / `np.log(series)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the transformed values.
    pub fn log(&self) -> Result<Series> {
        self.unary_with_context(UnaryOperator::Log, &default_execution_context(), "series::log")
    }

    /// Replace null values with a scalar.
    ///
    /// Equivalent to cuDF Python's `Series.fillna(scalar)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna<T: CudfType>(&self, value: T) -> Result<Series> {
        self.fillna_with_context(value, &default_execution_context(), "series::fillna")
    }

    /// Replace null values using corresponding elements from another Series.
    ///
    /// Both series must have the same dtype and length.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna_with_series(&self, other: &Series) -> Result<Series> {
        self.fillna_with_series_with_context(other, &default_execution_context(), "series::fillna_with_series")
    }

    /// Forward fill: replace each null with the nearest preceding non-null
    /// value. Leading nulls (with no preceding non-null value) remain null.
    ///
    /// Equivalent to cuDF Python's `Series.ffill()` / `Series.fillna(method='ffill')`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls forward-filled.
    pub fn ffill(&self) -> Result<Series> {
        self.fillna_policy_with_context(FillPolicy::Preceding, &default_execution_context(), "series::ffill")
    }

    /// Backward fill: replace each null with the nearest following non-null
    /// value. Trailing nulls (with no following non-null value) remain null.
    ///
    /// Equivalent to cuDF Python's `Series.bfill()` / `Series.fillna(method='bfill')`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls backward-filled.
    pub fn bfill(&self) -> Result<Series> {
        self.fillna_policy_with_context(FillPolicy::Following, &default_execution_context(), "series::bfill")
    }

    /// Element-wise membership test. Returns a BOOL8 Series where `true`
    /// indicates the element is present in `values`.
    ///
    /// Equivalent to cuDF Python's `Series.isin()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn isin(&self, values: &Series) -> Result<Series> {
        self.isin_with_context(values, &default_execution_context(), "series::isin")
    }

    /// Return a Series with duplicate values removed.
    ///
    /// The order of elements in the result is **not** specified.
    ///
    /// Equivalent to cuDF Python's `Series.unique()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column containing only distinct values.
    pub fn unique(&self) -> Result<Series> {
        self.unique_with_context(&default_execution_context(), "series::unique")
    }

    /// Return the number of distinct non-null values.
    ///
    /// Equivalent to cuDF Python's `Series.nunique()`.
    ///
    /// # Cost
    /// GPU reduction plus synchronization for host extraction. Does not
    /// materialize an intermediate distinct-value column.
    pub fn nunique(&self) -> Result<usize> {
        self.nunique_with_context(&default_execution_context(), "series::nunique")
    }

    /// Build a reusable codebook from the distinct values in this series.
    ///
    /// The returned [`Codebook`] owns both the distinct key series and an
    /// explicit positional code series `0..keys.len()`, making it safe to
    /// reuse for later [`Codebook::encode`] / [`Codebook::decode`] calls.
    ///
    /// # Cost
    /// GPU-allocating: produces the distinct key series plus positional code
    /// storage for the reusable codebook.
    pub fn codebook(&self) -> Result<Codebook> {
        self.codebook_with_context(&default_execution_context(), "series::codebook")
    }

    pub fn codebook_with(&self, options: DictionaryEncodingOptions) -> Result<Codebook> {
        self.codebook_with_options_in_context(&default_execution_context(), options, "series::codebook_with")
    }

    /// Encode this series as row-aligned integer codes plus a reusable codebook.
    ///
    /// Unlike the old tuple API, this returns a typed [`Factorized`] result
    /// that distinguishes row-aligned encoded indices from the reusable
    /// [`Codebook`] used to encode and decode arbitrary series later.
    ///
    /// Equivalent to cuDF Python's `Series.factorize()`, but with an explicit
    /// reusable codebook type.
    ///
    /// # Cost
    /// GPU-allocating: produces row codes plus the codebook keys/codes.
    pub fn factorize(&self) -> Result<Factorized> {
        self.factorize_with_context(&default_execution_context(), "series::factorize")
    }

    pub fn factorize_with(&self, options: DictionaryEncodingOptions) -> Result<Factorized> {
        self.factorize_with_options_in_context(&default_execution_context(), options, "series::factorize_with")
    }

    /// Sort the series according to the given options.
    ///
    /// Equivalent to cuDF Python's `Series.sort_values()`.
    ///
    /// # Cost
    /// GPU-allocating: a new sorted column is produced.
    pub fn sort_values(&self, options: SortOptions) -> Result<Series> {
        self.sort_values_with_context(options, &default_execution_context(), "series::sort_values")
    }
}

// Operator overloads (Add, Sub, Mul, Div) live in the `ops` module.
