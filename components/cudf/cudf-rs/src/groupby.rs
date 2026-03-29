//! Low-level GPU-accelerated groupby aggregation.
//!
//! Most users should use [`DataFrame::groupby`](crate::DataFrame::groupby) to
//! create a [`GroupBy`](crate::GroupBy) handle, then call
//! [`agg()`](crate::GroupBy::agg) with an [`AggSpec`](crate::AggSpec).

use std::{os::raw::c_void, ptr, sync::Arc};

use crate::{
    column::{Column, ColumnView},
    error::{self, Error, ErrorKind, Result},
    table::{Table, TableView},
};

/// Aggregation type for groupby operations.
#[derive(Debug, Clone)]
pub enum Aggregation {
    Sum,
    Min,
    Max,
    /// Count non-null values (if `include_nulls` is false) or all values.
    Count {
        include_nulls: bool,
    },
    Mean,
    Median,
    NUnique,
    /// Standard deviation with degrees-of-freedom correction.
    Std {
        ddof: i32,
    },
    /// Variance with degrees-of-freedom correction.
    Variance {
        ddof: i32,
    },
}

/// A single aggregation request: which column to aggregate and how.
pub struct AggregationRequest<'a> {
    /// The values column to aggregate.
    pub values: &'a ColumnView<'a>,
    /// One or more aggregations to apply to this column.
    pub aggregations: &'a [Aggregation],
}

pub use crate::types::NullPolicy;

/// Result of a groupby aggregation.
pub struct GroupByResult {
    /// The unique key groups.
    pub keys: Table,
    /// Aggregation results, one `Vec<Column>` per request. Each inner vec
    /// has one column per aggregation in the corresponding request.
    pub results: Vec<Vec<Column>>,
}

/// Create a C-side aggregation handle from an `Aggregation` enum.
///
/// # Safety
/// Caller must destroy the returned handle via `cudf_aggregation_destroy`.
unsafe fn create_agg_handle(agg: &Aggregation) -> Result<*mut c_void> {
    let mut handle = ptr::null_mut();
    match agg {
        Aggregation::Sum => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(0, &mut handle),
                "groupby::create_agg_handle(sum)",
            )?;
        },
        Aggregation::Min => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(1, &mut handle),
                "groupby::create_agg_handle(min)",
            )?;
        },
        Aggregation::Max => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(2, &mut handle),
                "groupby::create_agg_handle(max)",
            )?;
        },
        Aggregation::Count { include_nulls } => {
            let policy = if *include_nulls { 1 } else { 0 };
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_count(policy, &mut handle),
                "groupby::create_agg_handle(count)",
            )?;
        },
        Aggregation::Mean => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(5, &mut handle),
                "groupby::create_agg_handle(mean)",
            )?;
        },
        Aggregation::Median => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(6, &mut handle),
                "groupby::create_agg_handle(median)",
            )?;
        },
        Aggregation::NUnique => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create(7, &mut handle),
                "groupby::create_agg_handle(nunique)",
            )?;
        },
        Aggregation::Std { ddof } => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_std(*ddof, &mut handle),
                "groupby::create_agg_handle(std)",
            )?;
        },
        Aggregation::Variance { ddof } => {
            // SAFETY: handle is initialized to null and set by the C shim on success.
            error::check_status_in(
                cudf_sys::cudf_aggregation_create_variance(*ddof, &mut handle),
                "groupby::create_agg_handle(variance)",
            )?;
        },
    }
    Ok(handle)
}

/// Perform a groupby aggregation.
///
/// Groups rows by the `keys` table, then applies aggregations from each
/// request to the corresponding values column.
pub fn groupby_aggregate(
    keys: &TableView<'_>,
    requests: &[AggregationRequest<'_>],
    null_policy: NullPolicy,
) -> Result<GroupByResult> {
    let num_requests = i32::try_from(requests.len()).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "number of aggregation requests exceeds i32::MAX")
            .with_operation("groupby::groupby_aggregate")
            .with_context("num_requests", requests.len().to_string())
    })?;

    // Collect values column view handles.
    let value_handles: Vec<*mut c_void> = requests.iter().map(|r| r.values.raw_handle()).collect();

    // Collect agg_counts and create aggregation handles.
    let mut agg_counts: Vec<i32> = Vec::with_capacity(requests.len());
    let mut all_agg_handles: Vec<*mut c_void> = Vec::new();

    for req in requests {
        if req.aggregations.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "each aggregation request must contain at least one aggregation",
            )
            .with_operation("groupby::groupby_aggregate")
            .with_context("request_index", agg_counts.len().to_string())
            .with_context("num_requests", num_requests.to_string()));
        }

        let agg_count = i32::try_from(req.aggregations.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "aggregation count per request exceeds i32::MAX")
                .with_operation("groupby::groupby_aggregate")
                .with_context("request_index", agg_counts.len().to_string())
                .with_context("agg_count", req.aggregations.len().to_string())
        })?;
        agg_counts.push(agg_count);

        for agg in req.aggregations {
            match agg {
                Aggregation::Std { ddof } | Aggregation::Variance { ddof } if *ddof < 0 => {
                    return Err(Error::new(ErrorKind::InvalidOperation, "ddof must be >= 0")
                        .with_operation("groupby::groupby_aggregate")
                        .with_context("request_index", agg_counts.len().to_string())
                        .with_context("ddof", ddof.to_string()));
                },
                _ => {},
            }
            // SAFETY: create_agg_handle calls FFI with valid parameters.
            let handle = unsafe { create_agg_handle(agg)? };
            all_agg_handles.push(handle);
        }
    }

    let mut out_keys = ptr::null_mut();
    // This receives a pointer to a heap-allocated array of column handles
    // (cudf_column_handle* = void**).
    let mut out_values: *mut *mut c_void = ptr::null_mut();
    let mut out_total_columns: i32 = 0;

    // SAFETY: keys handle is valid. value_handles, all_agg_handles, and agg_counts
    // are valid slices with correct lengths.
    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys.raw_handle(),
            value_handles.as_ptr(),
            all_agg_handles.as_ptr(),
            agg_counts.as_ptr(),
            num_requests,
            i32::try_from(all_agg_handles.len()).expect("aggregation count exceeds i32::MAX"),
            null_policy as i32,
            &mut out_keys,
            &mut out_values,
            &mut out_total_columns,
        )
    };

    // Destroy aggregation handles regardless of success/failure.
    for handle in &all_agg_handles {
        // SAFETY: Each handle was returned by a successful create_agg_handle call.
        unsafe { cudf_sys::cudf_aggregation_destroy(*handle) };
    }

    let total_requested_aggs: i32 = agg_counts.iter().sum();
    error::check_status_in_with(
        status,
        "groupby::groupby_aggregate",
        [
            ("num_requests", num_requests.to_string()),
            ("total_requested_aggs", total_requested_aggs.to_string()),
            ("null_policy", (null_policy as i32).to_string()),
        ],
    )?;

    // SAFETY: out_keys was returned by a successful FFI call, transferring ownership.
    let keys_table = unsafe { Table::from_raw(out_keys) };

    // Collect result columns from the flat array.
    let total = out_total_columns as usize;
    // SAFETY: out_values and out_total_columns were set by a successful FFI call.
    // from_raw_parts is valid because the C shim allocated `total` elements.
    let col_array = if total == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(out_values, total) } };

    let mut results: Vec<Vec<Column>> = Vec::with_capacity(requests.len());
    let mut offset = 0;
    for count in &agg_counts {
        let n = *count as usize;
        let mut cols = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: Each column handle in col_array was returned by the FFI call.
            cols.push(unsafe { Column::from_raw(col_array[offset + i]) });
        }
        offset += n;
        results.push(cols);
    }

    // SAFETY: out_values was allocated by the C shim. Column handles were already
    // taken ownership of above, so we only free the array container.
    unsafe { cudf_sys::cudf_free_column_array(out_values) };

    Ok(GroupByResult { keys: keys_table, results })
}

// ===========================================================================
// High-level types: cuDF-Python-like groupby API
// ===========================================================================

/// High-level aggregation type for the user-facing groupby API.
///
/// This mirrors cuDF-Python's aggregation names. Internally each variant maps
/// to the corresponding [`Aggregation`] for the low-level
/// [`groupby_aggregate`] call.
#[derive(Debug, Clone)]
pub enum Agg {
    Sum,
    Min,
    Max,
    Mean,
    Count {
        include_nulls: bool,
    },
    Median,
    NUnique,
    /// Standard deviation with degrees-of-freedom correction.
    Std {
        ddof: u8,
    },
    /// Variance with degrees-of-freedom correction.
    Variance {
        ddof: u8,
    },
}

impl Agg {
    /// Convert to the low-level [`Aggregation`] type.
    fn to_aggregation(&self) -> Aggregation {
        match self {
            Agg::Sum => Aggregation::Sum,
            Agg::Min => Aggregation::Min,
            Agg::Max => Aggregation::Max,
            Agg::Mean => Aggregation::Mean,
            Agg::Count { include_nulls } => Aggregation::Count { include_nulls: *include_nulls },
            Agg::Median => Aggregation::Median,
            Agg::NUnique => Aggregation::NUnique,
            Agg::Std { ddof } => Aggregation::Std { ddof: i32::from(*ddof) },
            Agg::Variance { ddof } => Aggregation::Variance { ddof: i32::from(*ddof) },
        }
    }

    /// Return the default output column suffix for this aggregation.
    fn suffix(&self) -> &'static str {
        match self {
            Agg::Sum => "sum",
            Agg::Min => "min",
            Agg::Max => "max",
            Agg::Mean => "mean",
            Agg::Count { .. } => "count",
            Agg::Median => "median",
            Agg::NUnique => "nunique",
            Agg::Std { .. } => "std",
            Agg::Variance { .. } => "var",
        }
    }
}

/// A single aggregation specification: which column, which aggregation,
/// and an optional output alias.
///
/// If no alias is set the output column is named `"{column}_{agg}"`,
/// e.g. `"price_sum"`.
#[derive(Debug, Clone)]
pub struct AggSpec {
    column: String,
    agg: Agg,
    alias: Option<String>,
}

impl AggSpec {
    /// Create a new aggregation specification for `column` with `agg`.
    pub fn new(column: impl Into<String>, agg: Agg) -> Self {
        Self { column: column.into(), agg, alias: None }
    }

    /// Set an output alias for this aggregation result column.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// The output column name: alias if set, otherwise `"{column}_{suffix}"`.
    fn output_name(&self) -> String {
        match &self.alias {
            Some(alias) => alias.clone(),
            None => format!("{}_{}", self.column, self.agg.suffix()),
        }
    }
}

/// Options for the high-level groupby operation.
#[derive(Debug, Clone)]
pub struct GroupByOptions {
    null_policy: NullPolicy,
}

impl Default for GroupByOptions {
    fn default() -> Self {
        Self { null_policy: NullPolicy::Exclude }
    }
}

impl GroupByOptions {
    /// Set whether null keys are included in the groupby output.
    pub fn include_null_keys(mut self, include: bool) -> Self {
        self.null_policy = if include { NullPolicy::Include } else { NullPolicy::Exclude };
        self
    }
}

/// A pending groupby operation on a [`DataFrame`][crate::DataFrame].
///
/// Created via [`DataFrame::groupby`][crate::DataFrame::groupby].
/// Call [`.agg()`][GroupBy::agg] with [`AggSpec`] items, or use the shortcut
/// methods [`.sum()`][GroupBy::sum], [`.mean()`][GroupBy::mean],
/// [`.count()`][GroupBy::count] to execute.
pub struct GroupBy<'a> {
    df: &'a crate::dataframe::DataFrame,
    keys: Vec<String>,
    options: GroupByOptions,
}

impl<'a> GroupBy<'a> {
    /// Create a new `GroupBy` for the given DataFrame and key column names.
    pub(crate) fn new(df: &'a crate::dataframe::DataFrame, keys: Vec<String>) -> Self {
        Self { df, keys, options: GroupByOptions::default() }
    }

    /// Override the default groupby options.
    pub fn options(mut self, options: GroupByOptions) -> Self {
        self.options = options;
        self
    }

    /// Execute the groupby with the given aggregation specifications.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `specs` is empty
    /// - Any key or value column name is not in the DataFrame's schema
    pub fn agg(self, specs: &[AggSpec]) -> Result<crate::dataframe::DataFrame> {
        use indexmap::IndexMap;

        use crate::{
            dataframe::DataFrame,
            internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names},
        };

        if specs.is_empty() {
            return Err(Error::new(ErrorKind::InvalidOperation, "at least one aggregation spec is required")
                .with_operation("groupby::GroupBy::agg"));
        }

        let columns = self.df.columns_map();

        // Resolve key column names to indices.
        let key_refs: Vec<&str> = self.keys.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, "groupby::GroupBy::agg")?;

        // Validate all value column names exist.
        for spec in specs {
            if !columns.contains_key(spec.column.as_str()) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column '{}' not found in schema", spec.column),
                )
                .with_operation("groupby::GroupBy::agg")
                .with_context("column_name", spec.column.clone()));
            }
        }

        // Build a borrowed key projection over the existing columns.
        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;

        // Group specs by value column (preserving request order).
        let mut value_views: Vec<ColumnView<'_>> = Vec::new();
        let mut request_meta: Vec<(usize, Vec<Aggregation>)> = Vec::new();
        let mut result_names: Vec<String> = Vec::new();

        let mut col_name_to_view_idx: IndexMap<String, usize> = IndexMap::new();
        let mut view_idx_to_group_idx: IndexMap<usize, usize> = IndexMap::new();
        let mut output_order: Vec<(usize, usize)> = Vec::new();

        for spec in specs {
            let view_idx = if let Some(&idx) = col_name_to_view_idx.get(spec.column.as_str()) {
                idx
            } else {
                let arc_col = columns.get(spec.column.as_str()).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in DataFrame", spec.column))
                        .with_operation("dataframe::agg")
                })?;
                let idx = value_views.len();
                value_views.push(arc_col.view()?);
                col_name_to_view_idx.insert(spec.column.clone(), idx);
                idx
            };

            let group_idx = if let Some(&idx) = view_idx_to_group_idx.get(&view_idx) {
                idx
            } else {
                let idx = request_meta.len();
                request_meta.push((view_idx, Vec::new()));
                view_idx_to_group_idx.insert(view_idx, idx);
                idx
            };

            let agg_idx_within_group = request_meta[group_idx].1.len();
            request_meta[group_idx].1.push(spec.agg.to_aggregation());
            output_order.push((group_idx, agg_idx_within_group));

            result_names.push(spec.output_name());
        }

        let agg_slices: Vec<Vec<Aggregation>> = request_meta.iter().map(|(_, aggs)| aggs.clone()).collect();

        let requests: Vec<AggregationRequest<'_>> = request_meta
            .iter()
            .enumerate()
            .map(|(i, (view_idx, _))| AggregationRequest {
                values: &value_views[*view_idx],
                aggregations: &agg_slices[i],
            })
            .collect();

        let gb_result = groupby_aggregate(keys_table.table(), &requests, self.options.null_policy)?;
        let output_domain = Arc::clone(self.df.execution_domain());

        // Reorder result columns to match original spec order.
        let mut grouped_results: Vec<Vec<Option<Column>>> = gb_result
            .results
            .into_iter()
            .map(|cols| {
                cols.into_iter().map(|col| Some(col.with_execution_domain(Arc::clone(&output_domain)))).collect()
            })
            .collect();

        let mut all_result_cols: Vec<Column> = Vec::with_capacity(output_order.len());
        for (request_index, (group_idx, agg_idx)) in output_order.iter().enumerate() {
            let group_count = grouped_results.len();
            let group = grouped_results.get_mut(*group_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result group index out of bounds")
                    .with_operation("groupby::GroupBy::agg")
                    .with_context("request_index", request_index.to_string())
                    .with_context("group_index", group_idx.to_string())
                    .with_context("group_count", group_count.to_string())
            })?;

            let col_slot = group.get_mut(*agg_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result index out of bounds")
                    .with_operation("groupby::GroupBy::agg")
                    .with_context("request_index", request_index.to_string())
                    .with_context("agg_index", agg_idx.to_string())
            })?;

            let col = col_slot.take().ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result column consumed more than once")
                    .with_operation("groupby::GroupBy::agg")
                    .with_context("request_index", request_index.to_string())
                    .with_context("agg_index", agg_idx.to_string())
            })?;

            all_result_cols.push(col);
        }

        // Build the result DataFrame.
        let result_keys_cols = gb_result.keys.with_execution_domain(Arc::clone(&output_domain)).into_columns()?;

        let mut final_cols: Vec<Column> = Vec::with_capacity(result_keys_cols.len() + all_result_cols.len());
        final_cols.extend(result_keys_cols);
        final_cols.extend(all_result_cols);

        let mut schema_names: Vec<String> = Vec::with_capacity(self.keys.len() + result_names.len());
        schema_names.extend(self.keys);
        schema_names.extend(result_names);

        DataFrame::new(schema_names, final_cols)
    }

    /// Shortcut: compute the sum of each named column per group.
    pub fn sum<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Sum)).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the mean of each named column per group.
    pub fn mean<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Mean)).collect();
        self.agg(&specs)
    }

    /// Shortcut: count non-null values of each named column per group.
    pub fn count<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> =
            columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Count { include_nulls: false })).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the median of each named column per group.
    pub fn median<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Median)).collect();
        self.agg(&specs)
    }

    /// Shortcut: count unique non-null values of each named column per group.
    pub fn nunique<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::NUnique)).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the sample standard deviation (ddof=1) of each named
    /// column per group.
    pub fn std<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> =
            columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Std { ddof: 1 })).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the sample variance (ddof=1) of each named column
    /// per group.
    pub fn var<I, S>(self, columns: I) -> Result<crate::dataframe::DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> =
            columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Variance { ddof: 1 })).collect();
        self.agg(&specs)
    }
}
