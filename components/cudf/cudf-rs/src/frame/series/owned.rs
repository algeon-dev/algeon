use std::sync::Arc;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
        scalar::{IntoRuntimeScalar, Scalar},
        types::{CudfType, DataType, TimestampResolution, TypeId},
    },
    frame::{
        dataframe::{DataFrame, DropDuplicatesOptions},
        factorization::{Codebook, CodebookOptions, CodebookOrder, EncodedSeries, Factorized},
    },
    internal::temp_table::TempTableView,
    ops::{
        binary, concatenate, copying, filter as filter_ops,
        groupby::{AggregationRequest, GroupByAgg, NullPolicy, groupby_aggregate_impl},
        replace::{self, FillPolicy},
        sorting::SortOptions,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        unary::{self, UnaryOperator},
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

pub(super) const SERIES_VALUE_COUNTS_FALLBACK_NAME: &str = "value";
pub(super) const SERIES_VALUE_COUNTS_COUNT_NAME: &str = "count";

fn unique_temp_name(reserved: &str, base: &str) -> String {
    if reserved != base {
        return base.to_string();
    }

    let mut suffix = 1usize;
    loop {
        let candidate = format!("{base}{suffix}");
        if candidate != reserved {
            return candidate;
        }
        suffix += 1;
    }
}

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

    pub(super) fn require_bool_binary_operands(&self, rhs: &Series, operation: &'static str) -> Result<()> {
        let lhs_type = self.dtype()?.id();
        let rhs_type = rhs.dtype()?.id();

        if lhs_type == TypeId::Bool8 && rhs_type == TypeId::Bool8 {
            return Ok(());
        }

        Err(Error::new(
            ErrorKind::InvalidArgument,
            "logical boolean operations require both operands to have dtype Bool8",
        )
        .with_operation(operation)
        .with_context("lhs_type_id", format!("{:?}", lhs_type).to_lowercase())
        .with_context("rhs_type_id", format!("{:?}", rhs_type).to_lowercase()))
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
    pub fn view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    /// Build a single-column `TempTableView` over this series' data.
    ///
    /// This is the shared entry point for row-shape ops (`head`, `tail`,
    /// `slice`, `gather`, `filter`, `dropna`, `drop_duplicates`,
    /// `value_counts`). It keeps the `Series` on a direct column-primitive
    /// path instead of detouring through a synthetic single-column
    /// `DataFrame`.
    fn single_column_table_view(&self, operation: &'static str) -> Result<TempTableView<'_>> {
        let cv = self.column.view().map_err(|e| e.with_operation(operation))?;
        TempTableView::from_column_views(vec![cv])
    }

    /// Extract the single result column from a row-shape op, reattaching the
    /// source series name.
    fn take_single_column_result(&self, table: crate::core::table::Table) -> Result<Series> {
        let columns = table.into_columns()?;
        let [col] = <[Column; 1]>::try_from(columns).map_err(|v| {
            Error::new(ErrorKind::InvalidOperation, format!("series row op expected 1 output column, got {}", v.len()))
        })?;
        self.with_inherited_name(col)
    }

    /// Return the first `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned. The output inherits the source
    /// series name, including unnamed inputs.
    pub fn head(&self, n: usize) -> Result<Series> {
        let operation = "series::head";
        let total = self.len();
        let take = n.min(total);
        self.gather_range_inner(0, take, operation)
    }

    /// Return the last `n` rows of the series.
    ///
    /// If `n >= len`, all rows are returned. The output inherits the source
    /// series name, including unnamed inputs.
    pub fn tail(&self, n: usize) -> Result<Series> {
        let operation = "series::tail";
        let total = self.len();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_inner(start, take, operation)
    }

    /// Return `length` rows starting at `offset`.
    ///
    /// Both `offset` and `length` are clamped to the series length, so
    /// out-of-range values yield fewer rows rather than an error.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Series> {
        let operation = "series::slice";
        let total = self.len();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_inner(start, take, operation)
    }

    /// Internal: gather a contiguous range of rows into a new `Series`.
    fn gather_range_inner(&self, start: usize, len: usize, operation: &'static str) -> Result<Series> {
        if len == 0 {
            let domain = Arc::clone(self.column.execution_domain());
            let cv = self.column.view().map_err(|e| e.with_operation(operation))?;
            let empty = Column::empty_with_dtype_in_domain(cv.data_type()?, domain, operation)?;
            return self.with_inherited_name(empty);
        }

        let domain = Arc::clone(self.column.execution_domain());
        let end = start + len;
        let start_i32: i32 = start.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "row range start exceeds i32::MAX").with_operation(operation)
        })?;
        let end_i32: i32 = end.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "row range end exceeds i32::MAX").with_operation(operation)
        })?;
        let indices: Vec<i32> = (start_i32..end_i32).collect();
        let map = Column::from_slice_in_domain(&indices, &domain)?;
        let map_view = map.view()?;

        let ttv = self.single_column_table_view(operation)?;
        let exec_options = domain.explicit_exec_options_raw();
        // Indices are dense `start..end` with bounds already clamped, so
        // DONT_CHECK is safe.
        let result_table = copying::gather_impl(
            ttv.table(),
            &map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            &domain,
            exec_options.as_ref(),
            operation,
        )?;
        self.take_single_column_result(result_table)
    }

    /// Gather rows by an index `Series`.
    ///
    /// The `indices` series must be `Int32`. Out-of-bounds indices produce
    /// null rows in the result.
    pub fn gather(&self, indices: &Series) -> Result<Series> {
        let operation = "series::gather";
        let idx_cv = indices.view().map_err(|e| e.with_operation(operation))?;
        let domain = Arc::clone(self.column.execution_domain());
        ExecutionDomain::require_same_device_for_mixed_inputs(&domain, idx_cv.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(idx_cv.execution_domain(), &domain) {
            idx_cv.execution_domain().fence_stream(operation)?;
        }

        if idx_cv.is_empty()? {
            let cv = self.column.view().map_err(|e| e.with_operation(operation))?;
            let empty = Column::empty_with_dtype_in_domain(cv.data_type()?, domain, operation)?;
            return self.with_inherited_name(empty);
        }

        let ttv = self.single_column_table_view(operation)?;
        let exec_options = domain.explicit_exec_options_raw();
        let result_table = copying::gather_impl(
            ttv.table(),
            &idx_cv,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            &domain,
            exec_options.as_ref(),
            operation,
        )?;
        self.take_single_column_result(result_table)
    }

    /// Filter rows using a BOOL8 mask.
    ///
    /// Rows where `mask` is `true` are kept; `false` and null mask rows are
    /// dropped.
    pub fn filter(&self, mask: &Series) -> Result<Series> {
        let operation = "series::filter";
        let mask_cv = mask.view().map_err(|e| e.with_operation(operation))?;
        let mask_dtype = mask_cv.data_type()?;
        if mask_dtype.id() != TypeId::Bool8 {
            return Err(Error::new(
                ErrorKind::TypeError,
                format!("filter mask must be Bool8, got {:?}", mask_dtype.id()),
            )
            .with_operation(operation)
            .with_context("mask_dtype", format!("{:?}", mask_dtype.id())));
        }
        let mask_len = mask_cv.len()?;
        if mask_len != self.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("filter mask length ({}) does not match series length ({})", mask_len, self.len()),
            )
            .with_operation(operation)
            .with_context("mask_length", mask_len.to_string())
            .with_context("series_length", self.len().to_string()));
        }

        let domain = Arc::clone(self.column.execution_domain());
        ExecutionDomain::require_same_device_for_mixed_inputs(&domain, mask_cv.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(mask_cv.execution_domain(), &domain) {
            mask_cv.execution_domain().fence_stream(operation)?;
        }

        let ttv = self.single_column_table_view(operation)?;
        let exec_options = domain.explicit_exec_options_raw();
        let result_table =
            filter_ops::apply_boolean_mask_impl(ttv.table(), &mask_cv, &domain, exec_options.as_ref(), operation)?;
        self.take_single_column_result(result_table)
    }

    /// Drop null rows from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.dropna()`.
    pub fn dropna(&self) -> Result<Series> {
        let operation = "series::dropna";
        // Mirror DataFrame::dropna with default options: keep rows whose
        // single key column is non-null (threshold = 1 of 1 key columns).
        let domain = Arc::clone(self.column.execution_domain());
        let ttv = self.single_column_table_view(operation)?;
        let exec_options = domain.explicit_exec_options_raw();
        let result_table =
            stream_compaction::drop_nulls_impl(ttv.table(), &[0], 1, &domain, exec_options.as_ref(), operation)?;
        self.take_single_column_result(result_table)
    }

    /// Remove duplicate values from the series.
    ///
    /// Equivalent to the single-column form of cuDF Python's
    /// `Series.drop_duplicates()`.
    pub fn drop_duplicates(&self) -> Result<Series> {
        let operation = "series::drop_duplicates";
        let domain = Arc::clone(self.column.execution_domain());
        let ttv = self.single_column_table_view(operation)?;
        let exec_options = domain.explicit_exec_options_raw();
        let result_table = stream_compaction::distinct_impl(
            ttv.table(),
            &[0],
            DuplicateKeepOption::KeepAny,
            NullEquality::Equal,
            NanEquality::AllEqual,
            &domain,
            exec_options.as_ref(),
            operation,
        )?;
        self.take_single_column_result(result_table)
    }

    /// Count the frequency of each distinct non-null value.
    ///
    /// Returns a two-column [`DataFrame`] with one value column and one
    /// `count` column. Null values are excluded from the result.
    pub fn value_counts(&self) -> Result<DataFrame> {
        let operation = "series::value_counts";
        let value_name =
            self.name().map(str::to_string).unwrap_or_else(|| SERIES_VALUE_COUNTS_FALLBACK_NAME.to_string());
        let domain = Arc::clone(self.column.execution_domain());

        let keys_view = self.column.view().map_err(|e| e.with_operation(operation))?;
        let values_view = self.column.view().map_err(|e| e.with_operation(operation))?;
        let ttv = TempTableView::from_column_views(vec![keys_view])?;
        let aggs = [GroupByAgg::Count { include_nulls: false }];
        let requests = [AggregationRequest { values: &values_view, aggregations: &aggs }];
        let exec_options = domain.explicit_exec_options_raw();
        let gb = groupby_aggregate_impl(
            ttv.table(),
            &requests,
            NullPolicy::Exclude,
            &domain,
            exec_options.as_ref(),
            operation,
        )?;

        let mut key_columns = gb.keys.into_columns()?;
        let key_col = key_columns.pop().ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "groupby aggregate returned no key columns")
                .with_operation(operation)
        })?;
        let mut agg_results = gb.results;
        let mut count_cols = agg_results.pop().ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "groupby aggregate returned no count results")
                .with_operation(operation)
        })?;
        let count_col = count_cols.pop().ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "groupby aggregate returned empty count result")
                .with_operation(operation)
        })?;

        DataFrame::from_named_pairs([
            (value_name, Series::from_column(key_col)?),
            (SERIES_VALUE_COUNTS_COUNT_NAME.to_string(), Series::from_column(count_col)?),
        ])
        .map_err(|e| e.with_operation(operation))
    }

    /// Copy fixed-width column data to a host `Vec<T>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable — use
    /// [`to_optional_vec`](Self::to_optional_vec) instead when nulls are
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
            return Err(Error::new(ErrorKind::InvalidOperation, "to_vec_strict rejects columns containing nulls")
                .with_operation("series::to_vec_strict")
                .with_context("null_count", cv.null_count()?.to_string()));
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
    pub fn to_optional_vec<T: CudfType>(&self) -> Result<Vec<Option<T>>> {
        self.column.view()?.to_optional_vec::<T>()
    }

    /// Copy string column data to a host `Vec<String>`, rejecting nullable
    /// columns.
    ///
    /// Returns an error if the column is nullable — use
    /// [`to_optional_strings`](Self::to_optional_strings) or
    /// [`to_strings_lossy`](Self::to_strings_lossy) instead.
    ///
    /// # Errors
    /// - `InvalidOperation` if the column is nullable.
    /// - `TypeError` if the underlying column type is not `STRING`.
    pub fn to_strings_strict(&self) -> Result<Vec<String>> {
        let cv = self.column.view()?;
        if cv.null_count()? > 0 {
            return Err(Error::new(ErrorKind::InvalidOperation, "to_strings_strict rejects columns containing nulls")
                .with_operation("series::to_strings_strict")
                .with_context("null_count", cv.null_count()?.to_string()));
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
    /// Returns `TypeError` if the underlying column type is not `STRING`.
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
    /// Returns `TypeError` if the underlying column type is not `STRING`.
    pub fn to_strings_lossy(&self) -> Result<Vec<String>> {
        let opt = self.to_optional_strings()?;
        Ok(opt.into_iter().map(|o| o.unwrap_or_default()).collect())
    }

    // -- Scalar comparison ops -----------------------------------------

    /// Element-wise `self > scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn gt_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.gt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self < scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn lt_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.lt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self == scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn eq_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.eq_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- Extended scalar comparison ops --------------------------------

    /// Element-wise `self != scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn ne_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.ne_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self >= scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn ge_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.ge_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self <= scalar`. Inherits the source Series' name.
    ///
    /// Accepts numeric primitives (any [`CudfType`]) and host strings (`&str`,
    /// `String`, `&String`) through the [`IntoRuntimeScalar`] conversion.
    ///
    /// # Cost
    /// GPU-allocating: produces a new `BOOL8` output column on the device and
    /// wraps it in a new `Series`.
    pub fn le_scalar<T: IntoRuntimeScalar>(&self, rhs: T) -> Result<Series> {
        let s = rhs.into_runtime_scalar(self.column.execution_domain())?;
        let col = self.column.le_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- String scalar comparison ops (deprecated aliases) --------------------
    //
    // The generic `*_scalar` methods above now accept `&str`/`String`/`&String`
    // through `IntoRuntimeScalar`. These `*_scalar_str` aliases are kept as a
    // one-release migration courtesy and simply delegate.

    /// Deprecated: use [`Series::eq_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `eq_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn eq_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.eq_scalar(rhs)
    }

    /// Deprecated: use [`Series::ne_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `ne_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn ne_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.ne_scalar(rhs)
    }

    /// Deprecated: use [`Series::gt_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `gt_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn gt_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.gt_scalar(rhs)
    }

    /// Deprecated: use [`Series::lt_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `lt_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn lt_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.lt_scalar(rhs)
    }

    /// Deprecated: use [`Series::ge_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `ge_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn ge_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.ge_scalar(rhs)
    }

    /// Deprecated: use [`Series::le_scalar`] with a `&str` directly.
    #[deprecated(since = "0.2.0", note = "use `le_scalar` directly; `IntoRuntimeScalar` now accepts `&str`")]
    pub fn le_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.le_scalar(rhs)
    }

    // -- Column-vs-column comparison ops -------------------------------

    /// Element-wise `self == rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.eq(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self != rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ne(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self > rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.gt(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self < rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.lt(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self >= rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ge(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self <= rhs`. Returns an unnamed `BOOL8` Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed `BOOL8` output column on the
    /// device and wraps it in a new `Series`.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.le(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise logical AND. Both operands must be `BOOL8`.
    ///
    /// Returns an unnamed `BOOL8` Series suitable for mask composition.
    pub fn logical_and(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_and";
        self.require_bool_binary_operands(rhs, operation)?;
        let col = self
            .column
            .binary_op(&rhs.column, binary::BinaryOperator::LogicalAnd)
            .map_err(|e| e.with_operation(operation))?;
        Self::from_column(col).map_err(|e| e.with_operation(operation))
    }

    /// Element-wise logical OR. Both operands must be `BOOL8`.
    ///
    /// Returns an unnamed `BOOL8` Series suitable for mask composition.
    pub fn logical_or(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::logical_or";
        self.require_bool_binary_operands(rhs, operation)?;
        let col = self
            .column
            .binary_op(&rhs.column, binary::BinaryOperator::LogicalOr)
            .map_err(|e| e.with_operation(operation))?;
        Self::from_column(col).map_err(|e| e.with_operation(operation))
    }

    // -- Arithmetic ops ------------------------------------------------

    /// Element-wise `self + rhs`. Output type inferred. Returns an unnamed Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.add(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self - rhs`. Output type inferred. Returns an unnamed Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.sub(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self * rhs`. Output type inferred. Returns an unnamed Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.mul(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self / rhs`. Output type inferred. Returns an unnamed Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.div(&rhs.column)?;
        Self::from_column(col)
    }

    // -- Scalar arithmetic ops -----------------------------------------

    /// Element-wise `self + scalar`. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.add_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self - scalar`. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.sub_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self * scalar`. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.mul_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self / scalar`. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = self.scalar_from_value_in_domain(rhs)?;
        let col = self.column.div_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self % rhs` (modulo). Output type inferred. Returns an
    /// unnamed Series.
    ///
    /// # Cost
    /// GPU-allocating: produces a new unnamed output column on the device and
    /// wraps it in a new `Series`.
    pub fn rem(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.modulo(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self % scalar`. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: creates a device scalar from the host value, then
    /// produces a new output column on the device.
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
        let mut series = Self::from_column_arc(column).map_err(|e| e.with_operation("series::copy_into_runtime"))?;
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
            return Err(Error::new(ErrorKind::NotSupported, format!("target type {target:?} is not numeric"))
                .with_operation("series::to_numeric"));
        }
        self.astype(target)
    }

    /// Element-wise natural logarithm. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.log()` / `np.log(series)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the transformed values.
    pub fn log(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::unary_operation(&cv, UnaryOperator::Log)?;
        self.with_inherited_name(col)
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
        let col = replace::replace_nulls_with_exec_options(
            &cv,
            &scalar,
            self.column.execution_domain(),
            exec_options.as_ref(),
            "series::fillna",
        )?;
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
        let col = replace::replace_nulls_column_with_exec_options(
            &cv,
            &other_cv,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
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
            self.column.execution_domain(),
            exec_options.as_ref(),
            "series::ffill",
        )?;
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
            self.column.execution_domain(),
            exec_options.as_ref(),
            "series::bfill",
        )?;
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
        Ok(uniq.len().saturating_sub(uniq.null_count()))
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
        self.build_codebook_in_domain(Arc::clone(self.column.execution_domain()), "series::codebook")
    }

    fn first_seen_codebook(&self, operation: &'static str) -> Result<Codebook> {
        const FIRST_SEEN_VALUE_NAME: &str = "__cudf_first_seen_value__";
        const FIRST_SEEN_ROW_IDX_NAME: &str = "__cudf_first_seen_row_idx__";

        let source = self.dropna().map_err(|e| e.with_operation(operation))?;
        let value_name = source.name().unwrap_or(FIRST_SEEN_VALUE_NAME).to_string();
        let row_idx_name = unique_temp_name(&value_name, FIRST_SEEN_ROW_IDX_NAME);
        let source = match source.name() {
            Some(_) => source,
            None => source.rename(FIRST_SEEN_VALUE_NAME),
        };

        let row_idx = Series::from_column(Column::from_sequence_in_domain(
            0i32,
            1i32,
            source.len(),
            Arc::clone(self.column.execution_domain()),
            operation,
        )?)?
        .rename(&row_idx_name);

        let distinct = DataFrame::from_named_pairs([(value_name.clone(), source), (row_idx_name.clone(), row_idx)])
            .map_err(|e| e.with_operation(operation))?
            .drop_duplicates(
                DropDuplicatesOptions::builder()
                    .subset(&[value_name.as_str()])
                    .keep(DuplicateKeepOption::KeepFirst)
                    .build(),
            )
            .map_err(|e| e.with_operation(operation))?
            .sort_values([row_idx_name.as_str()], SortOptions::new().with_stable(true))
            .map_err(|e| e.with_operation(operation))?;

        let keys = distinct.series(&value_name).map_err(|e| e.with_operation(operation))?;
        let keys = match self.name() {
            Some(name) => keys.rename(format!("{name}_keys")),
            None => Series::from_column_arc(keys.into_inner()).map_err(|e| e.with_operation(operation))?,
        };

        Codebook::from_keys_with_fallback(keys, self.name().map(str::to_string))
            .map_err(|e| e.with_operation(operation))
    }

    pub fn codebook_with(&self, options: CodebookOptions) -> Result<Codebook> {
        match options.order_mode() {
            CodebookOrder::Sorted => self.codebook(),
            CodebookOrder::FirstSeen => self.first_seen_codebook("series::codebook_with"),
        }
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
        let cv = self.column.view()?;
        let (row_codes, keys) = cv.factorize_parts()?;
        let row_codes = Self::from_column(row_codes)?;
        let keys = Self::from_column(keys)?;

        let (row_codes, keys) = match self.name() {
            Some(name) => (row_codes.rename(format!("{name}_codes")), keys.rename(format!("{name}_keys"))),
            None => (row_codes, keys),
        };

        let row_codes = EncodedSeries::new(row_codes, self.name().map(str::to_string));
        let codebook = Codebook::from_keys_with_fallback(keys, self.name().map(str::to_string))?;
        Ok(Factorized::new(row_codes, codebook))
    }

    pub fn factorize_with(&self, options: CodebookOptions) -> Result<Factorized> {
        match options.order_mode() {
            CodebookOrder::Sorted => self.factorize(),
            CodebookOrder::FirstSeen => {
                let operation = "series::factorize_with";
                let codebook = self.first_seen_codebook(operation)?;
                let row_codes = codebook.encode(self).map_err(|e| e.with_operation(operation))?;
                Ok(Factorized::new(row_codes, codebook))
            },
        }
    }

    pub fn lookup(&self, keys: &Series, values: &Series) -> Result<Series> {
        let result = crate::ops::lookup::probe(
            &self.view().map_err(|e| e.with_operation("series::lookup"))?,
            &keys.view().map_err(|e| e.with_operation("series::lookup"))?,
            &values.view().map_err(|e| e.with_operation("series::lookup"))?,
        )
        .map_err(|e| e.with_operation("series::lookup"))?;
        let result = Series::from_column(result).map_err(|e| e.with_operation("series::lookup"))?;
        Ok(match values.name() {
            Some(name) => result.rename(name),
            None => result,
        })
    }

    pub fn lookup_as(&self, keys: &Series, values: &Series, output_name: &str) -> Result<Series> {
        self.lookup(keys, values).map(|result| result.rename(output_name))
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
