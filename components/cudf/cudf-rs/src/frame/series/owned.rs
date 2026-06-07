use std::sync::Arc;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, TimestampResolution, TypeId},
    },
    internal::temp_table::TempTableView,
    ops::{
        binary, concatenate,
        replace::{self, FillPolicy},
        sorting::SortOptions,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        unary,
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// A named, reference-counted GPU column.
///
/// `Series` is the cuDF-like user-facing column type. Internally it holds an
/// `Arc<Column>`, so cloning, renaming, and passing Series between DataFrames
/// is zero-copy on the GPU side.
///
/// Metadata queries (`len()`, `is_empty()`, `null_count()`) are infallible and
/// free — the values are cached at construction time.
///
/// [`std::fmt::Display`] for `Series` is metadata-only. It reports the name,
/// dtype, length, and nullability summary, but it never materializes values to
/// host memory.
#[derive(Clone)]
pub struct Series {
    pub(super) name: Option<String>,
    pub(super) column: Arc<Column>,
    pub(super) cached_len: usize,
    pub(super) cached_null_count: usize,
}

impl TryFrom<Column> for Series {
    type Error = crate::core::error::Error;

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
    fn scalar_from_value_in_domain<T: CudfType>(&self, value: T) -> Result<Scalar> {
        Scalar::from_value_in_domain(value, self.column.execution_domain())
    }

    fn scalar_from_string_in_domain(&self, value: &str) -> Result<Scalar> {
        Scalar::from_string_in_domain(value, self.column.execution_domain())
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

    /// Consume this Series and return the underlying `Arc<Column>`.
    ///
    /// This is the idiomatic way to unwrap a Series when you no longer need
    /// the name or Series wrapper.
    pub fn into_inner(self) -> Arc<Column> {
        self.column
    }

    /// Returns a borrowed [`ColumnView`] for low-level operations.
    ///
    /// # Advanced
    ///
    /// Use this when you need to call core-layer column operations
    /// (e.g. `device_data_ptr()`, `nullable()`, `validity()`) that are
    /// not exposed on `Series`.
    pub fn as_column_view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    /// Copy fixed-width column data to a host `Vec<T>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable — use
    /// [`to_vec_optional`](Self::to_vec_optional) instead when nulls are
    /// expected.
    ///
    /// # Cost
    /// Device-to-host copy: one synchronous memcpy of `len * size_of::<T>()` bytes.
    ///
    /// # Errors
    /// - `InvalidOperation` if the column is nullable.
    /// - `TypeError` if the column dtype does not match `T`.
    pub fn to_vec_strict<T: CudfType>(&self) -> Result<Vec<T>> {
        let cv = self.column.view()?;
        if cv.null_count()? > 0 {
            return Err(Error::new(ErrorKind::InvalidOperation, "to_vec_strict rejects nullable columns")
                .with_operation("series::to_vec_strict"));
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
    /// Returns `TypeError` if the column dtype does not match `T`.
    pub fn to_vec_optional<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        let cv = self.column.view()?;

        // Bool8 safety: libcudf stores booleans as int8 and makes no guarantee
        // about the byte value of null (masked-off) slots. Rust requires bool
        // to be exactly 0 or 1; any other value is UB. To avoid constructing
        // an invalid bool, we materialize as u8 and only convert valid slots.
        if T::TYPE_ID == TypeId::Bool8 {
            return self.to_vec_optional_bool::<T>(&cv);
        }

        let vals = cv.to_vec::<T>()?;
        match cv.validity()? {
            None => Ok(vals.into_iter().map(Some).collect()),
            Some(mask) => Ok(vals.into_iter().zip(mask).map(|(v, valid)| if valid { Some(v) } else { None }).collect()),
        }
    }

    /// Bool-safe path: materialize raw bytes, convert only valid positions.
    ///
    /// libcudf stores Bool8 as int8 and makes no guarantee about the byte
    /// value of null (masked-off) slots. Rust requires `bool` to be exactly
    /// 0 or 1; any other value is UB. We materialize as `u8` via
    /// `ColumnView::to_raw_bytes` and canonicalize before converting.
    fn to_vec_optional_bool<T: CudfType>(&self, cv: &ColumnView<'_>) -> Result<Vec<Option<T>>> {
        let raw = cv.to_raw_bytes()?;
        let bool_values: Vec<Option<bool>> = match cv.validity()? {
            None => raw.into_iter().map(|b| Some(b != 0)).collect(),
            Some(mask) => raw.into_iter().zip(mask).map(|(b, valid)| if valid { Some(b != 0) } else { None }).collect(),
        };

        // Only `bool` implements `CudfType<TYPE_ID = Bool8>`. Downcasting the
        // completed vector keeps this helper free of representation-level
        // unsafe code while preserving the runtime assertion.
        let boxed: Box<dyn std::any::Any> = Box::new(bool_values);
        boxed.downcast::<Vec<Option<T>>>().map(|typed| *typed).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "internal bool materialization helper requires T = bool")
                .with_operation("series::to_vec_optional_bool")
                .with_context("requested_type_id", format!("{:?}", T::TYPE_ID).to_lowercase())
        })
    }

    /// Copy string column data to a host `Vec<String>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable — use
    /// [`to_strings_optional`](Self::to_strings_optional) or
    /// [`to_strings_lossy`](Self::to_strings_lossy) instead.
    ///
    /// # Errors
    /// - `InvalidOperation` if the column is nullable.
    /// - `TypeError` if the underlying column type is not `STRING`.
    pub fn to_strings_strict(&self) -> Result<Vec<String>> {
        let cv = self.column.view()?;
        if cv.null_count()? > 0 {
            return Err(Error::new(ErrorKind::InvalidOperation, "to_strings_strict rejects nullable columns")
                .with_operation("series::to_strings_strict"));
        }
        cv.to_strings()
    }

    /// Copy string data to host as `Vec<Option<String>>`, distinguishing null
    /// from empty string.
    ///
    /// Null positions return `None`; valid positions return `Some(s)`.
    /// Non-nullable columns return all `Some`.
    ///
    /// # Errors
    /// Returns `TypeError` if the underlying column type is not `STRING`.
    pub fn to_strings_optional(&self) -> Result<Vec<Option<String>>> {
        self.column.view()?.to_optional_strings()
    }

    /// Copy string column data to a host `Vec<String>`, mapping null elements
    /// to empty strings (`""`).
    ///
    /// This is a lossy conversion: null and genuinely empty strings become
    /// indistinguishable. When null fidelity matters, use
    /// [`to_strings_optional`](Self::to_strings_optional).
    ///
    /// # Errors
    /// Returns `TypeError` if the underlying column type is not `STRING`.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        let opt = self.column.view()?.to_optional_strings()?;
        Ok(opt.into_iter().map(|o| o.unwrap_or_default()).collect())
    }

    // -- Scalar comparison ops -----------------------------------------

    /// Element-wise `self > scalar`. Inherits the source Series' name.
    pub fn gt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.gt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self < scalar`. Inherits the source Series' name.
    pub fn lt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.lt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self == scalar`. Inherits the source Series' name.
    pub fn eq_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- String scalar comparison ops ----------------------------------------

    /// Element-wise `self == string_scalar`. Inherits the source Series' name.
    pub fn eq_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self != string_scalar`. Inherits the source Series' name.
    pub fn ne_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.ne_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self > string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn gt_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.gt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self < string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn lt_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.lt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self >= string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn ge_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.ge_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self <= string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn le_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = self.scalar_from_string_in_domain(rhs)?;
        let col = self.column.le_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- Extended scalar comparison ops --------------------------------

    /// Element-wise `self != scalar`. Inherits the source Series' name.
    pub fn ne_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.ne_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self >= scalar`. Inherits the source Series' name.
    pub fn ge_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.ge_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self <= scalar`. Inherits the source Series' name.
    pub fn le_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.le_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- Column-vs-column comparison ops -------------------------------

    /// Element-wise `self == rhs`. Returns an unnamed `BOOL8` Series.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.eq(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self != rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ne(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self > rhs`. Returns an unnamed `BOOL8` Series.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.gt(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self < rhs`. Returns an unnamed `BOOL8` Series.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.lt(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self >= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ge(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self <= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.le(&rhs.column)?;
        Self::from_column(col)
    }

    // -- Arithmetic ops ------------------------------------------------

    /// Element-wise `self + rhs`. Output type inferred. Returns an unnamed Series.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.add(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self - rhs`. Output type inferred. Returns an unnamed Series.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.sub(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self * rhs`. Output type inferred. Returns an unnamed Series.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.mul(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self / rhs`. Output type inferred. Returns an unnamed Series.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.div(&rhs.column)?;
        Self::from_column(col)
    }

    // -- Scalar arithmetic ops -----------------------------------------

    /// Element-wise `self + scalar`. Inherits the source Series' name.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.add_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self - scalar`. Inherits the source Series' name.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.sub_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self * scalar`. Inherits the source Series' name.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.mul_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self / scalar`. Inherits the source Series' name.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.div_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self % rhs` (modulo). Output type inferred. Returns an
    /// unnamed Series.
    pub fn rem(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.modulo(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self % scalar`. Inherits the source Series' name.
    pub fn rem_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.binary_op_scalar(&s, binary::BinaryOperator::Mod)?;
        self.with_inherited_name(col)
    }

    // -- Static constructors for combining Series ----------------------------

    /// Concatenate multiple series into a single unnamed Series.
    ///
    /// All series must have the same data type.
    ///
    /// # Errors
    /// Returns `InvalidOperation` if the input slice is empty.
    /// Returns an error if the series have mismatched dtypes.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column combining all input data.
    pub fn concat(series: &[&Series]) -> Result<Series> {
        if series.is_empty() {
            return Err(Error::new(ErrorKind::InvalidOperation, "cannot concatenate empty slice of Series")
                .with_operation("series::concat"));
        }

        let views: Vec<_> = series.iter().map(|s| s.column.view()).collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&ColumnView<'_>> = views.iter().collect();
        let col = concatenate::concatenate_columns(&view_refs)?;
        Self::from_column(col)
    }

    /// Ensure this Series is placed in the target runtime.
    ///
    /// If the underlying column already has the target runtime's retained
    /// provenance, this reuses the existing GPU allocation. Otherwise the
    /// column is fenced and deep-copied into the target runtime.
    pub fn copy_into_runtime(&self, runtime: &Runtime) -> Result<Series> {
        let column =
            Self::clone_or_copy_into_domain(&self.column, runtime.execution_domain(), "series::copy_into_runtime")?;
        let mut series = Self::from_column_arc(column);
        series.name = self.name.clone();
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

        column.execution_domain().fence_stream(operation)?;

        column.copy_into_domain(Arc::clone(output_domain), operation).map(Arc::new)
    }

    // -- Convenience methods — unary ops inherit name ---------

    /// Return a BOOL8 mask: `true` where elements are null.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_null(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_null(&cv)?;
        self.with_inherited_name(col)
    }

    /// Return a BOOL8 mask: `true` where elements are valid (non-null).
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_not_null(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_valid(&cv)?;
        self.with_inherited_name(col)
    }

    /// Cast this series to the given type. Inherits the source Series' name.
    ///
    /// Accepts either a [`DataType`] (canonical, supports decimal scale) or a
    /// [`crate::TypeId`] (convenience shorthand, scale defaults to 0).
    ///
    /// Equivalent to cuDF Python's `Series.astype()`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the target dtype.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        let dt: DataType = target.into();
        let cv = self.column.view()?;
        let col = unary::cast(&cv, &dt)?;
        self.with_inherited_name(col)
    }

    /// Cast this series to a timestamp type. Inherits the source Series' name.
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
        let res = resolution.unwrap_or(TimestampResolution::Microseconds);
        let source_type = self.dtype()?.id();
        // libcudf does not support direct numeric → timestamp casts; it
        // requires numeric → duration → timestamp.
        if source_type.is_numeric() || source_type.is_boolean() {
            let duration = self.astype(res.duration_type_id())?;
            duration.astype(res.type_id())
        } else {
            self.astype(res.type_id())
        }
    }

    /// Cast this series to a numeric type. Inherits the source Series' name.
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
            return Err(Error::new(ErrorKind::InvalidOperation, format!("target type {target:?} is not numeric"))
                .with_operation("series::to_numeric"));
        }
        self.astype(target)
    }

    /// Replace null values with a scalar. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.fillna(scalar)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna<T: CudfType>(&self, value: T) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = self.scalar_from_value_in_domain(value)?;
        let exec_options = self.column.execution_domain().explicit_exec_options_raw();
        let col = replace::replace_nulls_with_exec_options(&cv, &scalar, exec_options.as_ref(), "series::fillna")?
            .with_execution_domain(Arc::clone(self.column.execution_domain()));
        self.with_inherited_name(col)
    }

    /// Replace null values using corresponding elements from another Series.
    /// Inherits the source Series' name.
    ///
    /// Both series must have the same dtype and length.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna_with_series(&self, other: &Series) -> Result<Series> {
        let operation = "series::fillna_with_series";
        let output_domain = Arc::clone(self.column.execution_domain());
        let cv = self.column.view()?;
        let other_cv = other.column.view()?;
        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            other.column.execution_domain(),
            operation,
        )?;
        if !ExecutionDomain::same_provenance(other.column.execution_domain(), &output_domain) {
            other.column.execution_domain().fence_stream(operation)?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_column_with_exec_options(&cv, &other_cv, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    /// Forward fill: replace each null with the nearest preceding non-null
    /// value. Leading nulls (with no preceding non-null value) remain null.
    /// Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.ffill()` / `Series.fillna(method='ffill')`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls forward-filled.
    pub fn ffill(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let exec_options = self.column.execution_domain().explicit_exec_options_raw();
        let col = replace::replace_nulls_policy_with_exec_options(
            &cv,
            FillPolicy::Preceding,
            exec_options.as_ref(),
            "series::ffill",
        )?
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        self.with_inherited_name(col)
    }

    /// Backward fill: replace each null with the nearest following non-null
    /// value. Trailing nulls (with no following non-null value) remain null.
    /// Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.bfill()` / `Series.fillna(method='bfill')`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls backward-filled.
    pub fn bfill(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let exec_options = self.column.execution_domain().explicit_exec_options_raw();
        let col = replace::replace_nulls_policy_with_exec_options(
            &cv,
            FillPolicy::Following,
            exec_options.as_ref(),
            "series::bfill",
        )?
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        self.with_inherited_name(col)
    }

    /// Element-wise membership test. Returns a BOOL8 Series where `true`
    /// indicates the element is present in `values`. Inherits the source
    /// Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.isin()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn isin(&self, values: &Series) -> Result<Series> {
        let cv = self.column.view()?;
        let values_cv = values.column.view()?;
        let col = cv.isin(&values_cv)?;
        self.with_inherited_name(col)
    }

    /// Return a Series with duplicate values removed. Inherits the source
    /// Series' name.
    ///
    /// The order of elements in the result is **not** specified.
    ///
    /// Equivalent to cuDF Python's `Series.unique()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column containing only distinct values.
    pub fn unique(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let result = stream_compaction::distinct(
            ttv.table(),
            &[0],
            DuplicateKeepOption::KeepAny,
            NullEquality::Equal,
            NanEquality::AllEqual,
        )?
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        let mut cols = result.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }

    /// Return the number of distinct non-null values.
    ///
    /// Equivalent to cuDF Python's `Series.nunique()`.
    ///
    /// # Cost
    /// GPU-allocating: internally computes `unique()` and counts rows.
    pub fn nunique(&self) -> Result<usize> {
        let uniq = self.unique()?;
        // Exclude nulls from the count (matches cuDF Python default).
        Ok(uniq.len() - uniq.null_count())
    }

    /// Encode the series as integer codes and a set of unique values.
    ///
    /// Returns `(codes, uniques)` where `codes` is an `Int32` Series mapping
    /// each element to its index in `uniques`.
    ///
    /// Name behavior:
    /// - named input => `"{name}_codes"` and `"{name}_uniques"`
    /// - unnamed input => both outputs stay unnamed
    ///
    /// Equivalent to cuDF Python's `Series.factorize()`.
    ///
    /// # Cost
    /// GPU-allocating: produces two new columns (codes + uniques).
    pub fn factorize(&self) -> Result<(Series, Series)> {
        let cv = self.column.view()?;
        let (codes, uniques) = cv.factorize()?;
        let codes = Self::from_column(codes)?;
        let uniques = Self::from_column(uniques)?;
        if let Some(name) = self.name() {
            Ok((codes.rename(format!("{name}_codes")), uniques.rename(format!("{name}_uniques"))))
        } else {
            Ok((codes, uniques))
        }
    }

    /// Sort the series according to the given options. Inherits the source
    /// Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.sort_values()`.
    ///
    /// # Cost
    /// GPU-allocating: a new sorted column is produced.
    pub fn sort_values(&self, options: SortOptions) -> Result<Series> {
        let (orders, null_orders) = options.resolve(1)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let sorted = if options.stable {
            crate::ops::sorting::stable_sort(ttv.table(), Some(&orders), null_orders.as_deref())?
        } else {
            crate::ops::sorting::sort(ttv.table(), Some(&orders), null_orders.as_deref())?
        }
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        let mut cols = sorted.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }
}

// Operator overloads (Add, Sub, Mul, Div) live in the `ops` module.
