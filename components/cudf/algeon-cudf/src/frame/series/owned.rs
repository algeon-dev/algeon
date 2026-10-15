use std::sync::Arc;

use super::context::SeriesOps;
#[cfg(any(test, feature = "test-hooks"))]
use crate::frame::testing_hooks;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_TYPE_ID, Result},
        scalar::ScalarValue,
        types::{CudfType, DataType, TimestampResolution, TypeId},
    },
    frame::{
        dataframe::DataFrame,
        factorization::{Codebook, DictionaryEncodingOptions, Factorized},
    },
    ops::{binary::BinaryOperator, replace::FillPolicy, sorting::SortOptions, unary::UnaryOperator},
    runtime::{
        CompleteReport, DeviceId, ExecutionBudgetReport, ExecutionPlacement, Runtime, execution_domain::ExecutionDomain,
    },
};

pub(super) const SERIES_VALUE_COUNTS_FALLBACK_NAME: &str = "value";
pub(super) const SERIES_VALUE_COUNTS_COUNT_NAME: &str = "count";

pub(super) fn unsupported_numeric_target_error(operation: &'static str, target: TypeId) -> Error {
    Error::new_with(ErrorCode::UnsupportedDtype, operation, format!("target type {target:?} is not numeric"))
        .with_fact(FACT_TYPE_ID, target)
}

/// Reference-counted GPU column vector.
///
/// `Series` wraps an `Arc<Column>`. Cloning and transferring columns across
/// `DataFrame` structures increments reference counts without copying buffer
/// memory.
///
/// Metadata attributes (`len()`, `is_empty()`, `null_count()`) are cached at
/// construction.
///
/// [`std::fmt::Display`] for `Series` is metadata-only. It reports the dtype,
/// length, and nullability summary, but it never materializes values to host
/// memory.
///
/// # Interop
///
/// [`try_prepare_device_column_export`](Self::try_prepare_device_column_export)
/// describes the device buffers to a consumer without copying them;
/// [`try_prepare_host_column_export`](Self::try_prepare_host_column_export)
/// copies the values to host memory.
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

    /// The CUDA device this series' column lives on.
    pub fn device(&self) -> DeviceId {
        self.column.execution_domain().device_id()
    }

    /// Returns bytes retained by this series' owned device column.
    pub fn device_allocated_size_bytes(&self) -> Result<usize> {
        self.column.device_allocated_size_bytes().map_err(|error| error.reanchor("series::device_allocated_size_bytes"))
    }

    /// Returns `true` if both series share the same retained provenance
    /// (CUDA stream and memory resource).
    pub fn same_provenance(&self, other: &Series) -> bool {
        ExecutionDomain::same_provenance(self.column.execution_domain(), other.column.execution_domain())
    }

    /// Borrows the underlying `Column` for operations not exposed on `Series`.
    pub fn as_column(&self) -> &Column {
        &self.column
    }

    /// Borrows the shared `Arc<Column>` so `DataFrame` can clone it into its
    /// column map.
    pub(in crate::frame) fn column_arc(&self) -> &Arc<Column> {
        &self.column
    }

    /// Clones the shared `Arc<Column>` without consuming the `Series`.
    pub fn arc(&self) -> Arc<Column> {
        Arc::clone(&self.column)
    }

    /// Consumes the `Series` and returns the underlying `Arc<Column>`.
    pub fn into_inner(self) -> Arc<Column> {
        self.column
    }

    pub(crate) fn view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    /// Copy fixed-width column data to a host `Vec<T>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable; use
    /// [`to_optional_vec`](Self::to_optional_vec) instead when nulls are
    /// expected.
    ///
    /// # Cost
    /// One synchronous device-to-host memcpy transfers `len * size_of::<T>()` bytes.
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
    /// Two synchronous device-to-host memcpys transfer `len * size_of::<T>()` bytes
    /// for data and `ceil(len / 8)` bytes for the null mask.
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
    /// This conversion is lossy. Null and empty strings become
    /// indistinguishable. When null fidelity matters, use
    /// [`to_optional_strings`](Self::to_optional_strings).
    ///
    /// # Errors
    /// Returns `ErrorKind::InvalidInput` with `ErrorCode::TypeMismatch` if the
    /// underlying column type is not `STRING`.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        self.column.view()?.to_strings_lossy()
    }

    /// Ensure this Series is placed in the target runtime.
    ///
    /// If the underlying column already has the target runtime's retained
    /// provenance, this reuses the existing GPU allocation. Otherwise the
    /// column is fenced and deep-copied into the target runtime.
    pub fn copy_into_runtime(&self, runtime: &Runtime) -> Result<Series> {
        #[cfg(any(test, feature = "test-hooks"))]
        let should_copy = !ExecutionDomain::same_provenance(self.column.execution_domain(), runtime.execution_domain());
        let column =
            Self::clone_or_copy_into_domain(&self.column, runtime.execution_domain(), "series::copy_into_runtime")?;
        let series = Self::from_column_arc(column).map_err(|e| e.reanchor("series::copy_into_runtime"))?;
        #[cfg(any(test, feature = "test-hooks"))]
        if should_copy {
            testing_hooks::record_device_to_device_copy();
        }
        Ok(series)
    }

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
}

// Operator overloads (Add, Sub, Mul, Div) live in the `ops` module.

impl CompleteReport for Series {
    fn complete_report(&self, report: ExecutionBudgetReport) -> Result<ExecutionBudgetReport> {
        let bytes = self.device_allocated_size_bytes()?;
        Ok(report.with_retained_bytes(bytes).with_output_bytes(bytes).with_output_rows(self.len()))
    }
}

impl SeriesOps<'_> {
    /// Return the first `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned.
    pub fn head(&self, n: usize) -> Result<Series> {
        self.series.head_with_context(n, self.ctx, "series::head")
    }

    /// Return the last `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned.
    pub fn tail(&self, n: usize) -> Result<Series> {
        self.series.tail_with_context(n, self.ctx, "series::tail")
    }

    /// Return `length` rows starting at `offset`.
    ///
    /// Both `offset` and `length` are clamped to the series length, so
    /// out-of-range values yield fewer rows and do not return an error.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Series> {
        self.series.slice_with_context(offset, length, self.ctx, "series::slice")
    }

    /// Gather rows by an index `Series`.
    ///
    /// The `indices` series must be `Int32`. Out-of-bounds indices produce
    /// null rows in the result.
    pub fn gather(&self, indices: &Series) -> Result<Series> {
        self.series.gather_with_context(indices, self.ctx, "series::gather")
    }

    /// Filter rows using a BOOL8 mask.
    ///
    /// Rows where `mask` is `true` are kept; `false` and null mask rows are
    /// dropped.
    pub fn filter(&self, mask: &Series) -> Result<Series> {
        self.series.filter_with_context(mask, self.ctx, "series::filter")
    }

    /// Drop null rows from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.dropna()`.
    pub fn dropna(&self) -> Result<Series> {
        self.series.dropna_with_context(self.ctx, "series::dropna")
    }

    /// Remove duplicate values from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.drop_duplicates()`.
    pub fn drop_duplicates(&self) -> Result<Series> {
        self.series.drop_duplicates_with_context(self.ctx, "series::drop_duplicates")
    }

    /// Count the frequency of each distinct non-null value.
    ///
    /// Returns a two-column [`DataFrame`] with one value column and one
    /// `count` column. Null values are excluded from the result.
    pub fn value_counts(&self) -> Result<DataFrame> {
        self.series.value_counts_with_context(self.ctx, "series::value_counts")
    }

    /// Element-wise `self > scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn gt_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Greater, self.ctx, "series::gt_scalar")
    }

    /// Element-wise `self < scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn lt_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Less, self.ctx, "series::lt_scalar")
    }

    /// Element-wise `self == scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn eq_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Equal, self.ctx, "series::eq_scalar")
    }

    /// Element-wise `self != scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn ne_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::NotEqual, self.ctx, "series::ne_scalar")
    }

    /// Element-wise `self >= scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn ge_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::GreaterEqual, self.ctx, "series::ge_scalar")
    }

    /// Element-wise `self <= scalar`.
    ///
    /// Accepts any value that converts into a [`ScalarValue`].
    ///
    /// # Cost
    /// Allocates a new `BOOL8` output column on the device and wraps it in a new `Series`.
    pub fn le_scalar(&self, rhs: impl Into<ScalarValue>) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::LessEqual, self.ctx, "series::le_scalar")
    }

    /// Element-wise `self == rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Equal, self.ctx, "series::eq")
    }

    /// Element-wise `self != rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::NotEqual, self.ctx, "series::ne")
    }

    /// Element-wise `self > rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Greater, self.ctx, "series::gt")
    }

    /// Element-wise `self < rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Less, self.ctx, "series::lt")
    }

    /// Element-wise `self >= rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::GreaterEqual, self.ctx, "series::ge")
    }

    /// Element-wise `self <= rhs`. Returns a `BOOL8` Series.
    ///
    /// # Cost
    /// Allocates a new unnamed `BOOL8` output column on the device and wraps it
    /// in a new `Series`.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::LessEqual, self.ctx, "series::le")
    }

    /// Element-wise logical AND. Both operands must be `BOOL8`.
    ///
    /// Returns a `BOOL8` Series suitable for mask composition.
    pub fn logical_and(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_and";
        self.series.require_bool_binary_operands(rhs, operation)?;
        self.series.binary_series_in_context(rhs, BinaryOperator::LogicalAnd, self.ctx, operation)
    }

    /// Element-wise logical OR. Both operands must be `BOOL8`.
    ///
    /// Returns a `BOOL8` Series suitable for mask composition.
    pub fn logical_or(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_or";
        self.series.require_bool_binary_operands(rhs, operation)?;
        self.series.binary_series_in_context(rhs, BinaryOperator::LogicalOr, self.ctx, operation)
    }

    /// Element-wise `self + rhs`. Output type inferred.
    ///
    /// # Cost
    /// Allocates a new unnamed output column on the device and wraps it in a
    /// new `Series`.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Add, self.ctx, "series::add")
    }

    /// Element-wise `self - rhs`. Output type inferred.
    ///
    /// # Cost
    /// Allocates a new unnamed output column on the device and wraps it in a
    /// new `Series`.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Sub, self.ctx, "series::sub")
    }

    /// Element-wise `self * rhs`. Output type inferred.
    ///
    /// # Cost
    /// Allocates a new unnamed output column on the device and wraps it in a
    /// new `Series`.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Mul, self.ctx, "series::mul")
    }

    /// Element-wise `self / rhs`. Output type inferred.
    ///
    /// # Cost
    /// Allocates a new unnamed output column on the device and wraps it in a
    /// new `Series`.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Div, self.ctx, "series::div")
    }

    /// Element-wise `self + scalar`.
    ///
    /// # Cost
    /// Creates a device scalar from the host value, then produces a new output
    /// column on the device.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Add, self.ctx, "series::add_scalar")
    }

    /// Element-wise `self - scalar`.
    ///
    /// # Cost
    /// Creates a device scalar from the host value, then produces a new output
    /// column on the device.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Sub, self.ctx, "series::sub_scalar")
    }

    /// Element-wise `self * scalar`.
    ///
    /// # Cost
    /// Creates a device scalar from the host value, then produces a new output
    /// column on the device.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Mul, self.ctx, "series::mul_scalar")
    }

    /// Element-wise `self / scalar`.
    ///
    /// # Cost
    /// Creates a device scalar from the host value, then produces a new output
    /// column on the device.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Div, self.ctx, "series::div_scalar")
    }

    /// Element-wise `self % rhs` (modulo). Output type inferred. Returns a
    /// `Series`.
    ///
    /// # Cost
    /// Allocates a new unnamed output column on the device and wraps it in a
    /// new `Series`.
    pub fn rem(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Mod, self.ctx, "series::rem")
    }

    /// Element-wise `self % scalar`.
    ///
    /// # Cost
    /// Creates a device scalar from the host value, then produces a new output
    /// column on the device.
    pub fn rem_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Mod, self.ctx, "series::rem_scalar")
    }

    /// Return a BOOL8 mask: `true` where elements are null.
    ///
    /// # Cost
    /// Allocates a new BOOL8 column on the device.
    pub fn is_null(&self) -> Result<Series> {
        self.series.is_null_with_context(self.ctx, "series::is_null")
    }

    /// Return a BOOL8 mask: `true` where elements are valid (non-null).
    ///
    /// # Cost
    /// Allocates a new BOOL8 column on the device.
    pub fn is_not_null(&self) -> Result<Series> {
        self.series.is_not_null_with_context(self.ctx, "series::is_not_null")
    }

    /// Cast this series to the target type.
    ///
    /// Accepts a [`DataType`] or [`crate::TypeId`]. Raw `TypeId` values default
    /// to scale 0.
    ///
    /// Allocates a new column with the target dtype.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        self.series.cast_in_context(target, self.ctx, "series::astype")
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
    /// Allocates a new column with the target timestamp dtype.
    pub fn to_datetime(&self, resolution: Option<TimestampResolution>) -> Result<Series> {
        self.series.to_datetime_with_context(
            resolution,
            self.ctx,
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
    /// Allocates a new column with the target numeric dtype.
    pub fn to_numeric(&self, target: TypeId) -> Result<Series> {
        if !target.is_numeric() && !target.is_boolean() {
            return Err(unsupported_numeric_target_error("series::to_numeric", target));
        }
        self.series.cast_in_context(target, self.ctx, "series::to_numeric")
    }

    /// Element-wise natural logarithm.
    ///
    /// Equivalent to cuDF Python's `Series.log()` / `np.log(series)`.
    ///
    /// # Cost
    /// Allocates a new column with the transformed values.
    pub fn log(&self) -> Result<Series> {
        self.series.unary_with_context(UnaryOperator::Log, self.ctx, "series::log")
    }

    /// Replace null values with a scalar.
    ///
    /// Equivalent to cuDF Python's `Series.fillna(scalar)`.
    ///
    /// # Cost
    /// Allocates a new column with nulls replaced.
    pub fn fillna<T: CudfType>(&self, value: T) -> Result<Series> {
        self.series.fillna_with_context(value, self.ctx, "series::fillna")
    }

    /// Replace null values using corresponding elements from another Series.
    ///
    /// Both series must have the same dtype and length.
    ///
    /// # Cost
    /// Allocates a new column with nulls replaced.
    pub fn fillna_with_series(&self, other: &Series) -> Result<Series> {
        self.series.fillna_with_series_with_context(other, self.ctx, "series::fillna_with_series")
    }

    /// Forward fill replaces each null with the nearest preceding non-null value.
    /// Leading nulls (with no preceding non-null value) remain null.
    ///
    /// Equivalent to cuDF Python's `Series.ffill()` / `Series.fillna(method='ffill')`.
    ///
    /// # Cost
    /// Allocates a new column with nulls forward-filled.
    pub fn ffill(&self) -> Result<Series> {
        self.series.fillna_policy_with_context(FillPolicy::Preceding, self.ctx, "series::ffill")
    }

    /// Backward fill replaces each null with the nearest following non-null value.
    /// Trailing nulls (with no following non-null value) remain null.
    ///
    /// Equivalent to cuDF Python's `Series.bfill()` / `Series.fillna(method='bfill')`.
    ///
    /// # Cost
    /// Allocates a new column with nulls backward-filled.
    pub fn bfill(&self) -> Result<Series> {
        self.series.fillna_policy_with_context(FillPolicy::Following, self.ctx, "series::bfill")
    }

    /// Element-wise membership test. Returns a BOOL8 Series where `true`
    /// indicates the element is present in `values`.
    ///
    /// Equivalent to cuDF Python's `Series.isin()`.
    ///
    /// # Cost
    /// Allocates a new BOOL8 column.
    pub fn isin(&self, values: &Series) -> Result<Series> {
        self.series.isin_with_context(values, self.ctx, "series::isin")
    }

    /// Return a Series with duplicate values removed.
    ///
    /// The order of elements in the result is not specified.
    ///
    /// Equivalent to cuDF Python's `Series.unique()`.
    ///
    /// # Cost
    /// Allocates a new column containing only distinct values.
    pub fn unique(&self) -> Result<Series> {
        self.series.unique_with_context(self.ctx, "series::unique")
    }

    /// Return the number of distinct non-null values.
    ///
    /// Equivalent to cuDF Python's `Series.nunique()`.
    ///
    /// # Cost
    /// GPU reduction plus synchronization for host extraction. Does not
    /// materialize an intermediate distinct-value column.
    pub fn nunique(&self) -> Result<usize> {
        self.series.nunique_with_context(self.ctx, "series::nunique")
    }

    /// Build a reusable codebook from the distinct values in this series.
    ///
    /// The returned [`Codebook`] owns both the distinct key series and an
    /// explicit positional code series `0..keys.len()`, making it safe to
    /// reuse for later [`Codebook::encode`] / [`Codebook::decode`] calls.
    ///
    /// # Cost
    /// Allocates the distinct key series plus positional code storage for the
    /// reusable codebook.
    pub fn codebook(&self) -> Result<Codebook> {
        self.series.codebook_with_context(self.ctx, "series::codebook")
    }

    pub fn codebook_with(&self, options: DictionaryEncodingOptions) -> Result<Codebook> {
        self.series.codebook_with_options_in_context(self.ctx, options, "series::codebook_with")
    }

    /// Encode this series as row-aligned integer codes plus a reusable codebook.
    ///
    /// Returns a typed [`Factorized`] result that distinguishes row-aligned
    /// encoded indices from the reusable [`Codebook`] used to encode and decode
    /// arbitrary series later.
    ///
    /// Equivalent to cuDF Python's `Series.factorize()`, but with an explicit
    /// reusable codebook type.
    ///
    /// # Cost
    /// Allocates row codes plus the codebook keys/codes.
    pub fn factorize(&self) -> Result<Factorized> {
        self.series.factorize_with_context(self.ctx, "series::factorize")
    }

    pub fn factorize_with(&self, options: DictionaryEncodingOptions) -> Result<Factorized> {
        self.series.factorize_with_options_in_context(self.ctx, options, "series::factorize_with")
    }

    /// Sort the series according to the given options.
    ///
    /// Equivalent to cuDF Python's `Series.sort_values()`.
    ///
    /// # Cost
    /// Allocates a new sorted column.
    pub fn sort_values(&self, options: SortOptions) -> Result<Series> {
        self.series.sort_values_with_context(options, self.ctx, "series::sort_values")
    }
}
