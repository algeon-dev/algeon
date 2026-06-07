//! High-level DataFrame groupby API.

use std::sync::Arc;

use smallvec::SmallVec;
use typed_builder::TypedBuilder;

use super::agg::{Aggregation, AggregationRequest, NullPolicy, groupby_aggregate, groupby_aggregate_with_exec_options};
use crate::{
    core::{
        column::ColumnView,
        error::{Error, ErrorKind, Result},
    },
    frame::dataframe::DataFrame,
    runtime::{execution_context::ExecutionContext, execution_domain::ExecutionDomain},
};

const GROUPBY_VALUE_VIEW_INLINE_CAPACITY: usize = 8;
const GROUPBY_REQUEST_META_INLINE_CAPACITY: usize = 8;
const GROUPBY_RESULT_NAME_INLINE_CAPACITY: usize = 8;
const GROUPBY_OUTPUT_ORDER_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_SLICE_INLINE_CAPACITY: usize = 8;
const GROUPBY_REQUEST_INLINE_CAPACITY: usize = 8;

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
#[derive(Debug, Clone, TypedBuilder)]
pub struct GroupByOptions {
    #[builder(default)]
    include_null_keys: bool,
}

impl Default for GroupByOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl GroupByOptions {
    pub(crate) fn null_policy(&self) -> NullPolicy {
        if self.include_null_keys { NullPolicy::Include } else { NullPolicy::Exclude }
    }
}

/// A pending groupby operation on a [`DataFrame`][crate::DataFrame].
///
/// Created via [`DataFrame::groupby`][crate::DataFrame::groupby].
/// Call [`.agg()`][GroupBy::agg] with [`AggSpec`] items, or use the shortcut
/// methods [`.sum()`][GroupBy::sum], [`.mean()`][GroupBy::mean],
/// [`.count()`][GroupBy::count] to execute.
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    keys: Vec<String>,
    options: GroupByOptions,
}

impl<'a> GroupBy<'a> {
    /// Create a new `GroupBy` for the given DataFrame and key column names.
    pub(crate) fn new(df: &'a DataFrame, keys: Vec<String>) -> Self {
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
    pub fn agg(self, specs: &[AggSpec]) -> Result<DataFrame> {
        let output_domain = Arc::clone(self.df.execution_domain());
        self.agg_impl(specs, None, output_domain, "groupby::GroupBy::agg")
    }

    /// Execute the groupby with a borrowed per-operation execution override.
    pub fn agg_with_context(self, specs: &[AggSpec], ctx: &ExecutionContext<'_>) -> Result<DataFrame> {
        let output_domain = ctx.resolve_output_domain(self.df.execution_domain())?;
        self.agg_impl(
            specs,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "groupby::GroupBy::agg_with_context",
        )
    }

    #[doc(hidden)]
    pub fn __debug_agg_with_execution_context(
        self,
        specs: &[AggSpec],
        ctx: &ExecutionContext<'_>,
    ) -> Result<DataFrame> {
        self.agg_with_context(specs, ctx)
    }

    fn agg_impl(
        self,
        specs: &[AggSpec],
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<DataFrame> {
        use crate::internal::{
            hash::{FastHashMap, fast_hash_map_with_capacity},
            resolve::{build_borrowed_table_view_from_columns, resolve_column_names},
        };

        if specs.is_empty() {
            return Err(Error::new(ErrorKind::InvalidOperation, "at least one aggregation spec is required")
                .with_operation(operation));
        }

        let columns = self.df.columns_map();

        let key_refs: SmallVec<[&str; GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            self.keys.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, operation)?;
        drop(key_refs);

        for spec in specs {
            if !columns.contains_key(spec.column.as_str()) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column '{}' not found in schema", spec.column),
                )
                .with_operation(operation)
                .with_context("column_name", spec.column.clone()));
            }
        }

        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;

        let mut value_views: SmallVec<[ColumnView<'_>; GROUPBY_VALUE_VIEW_INLINE_CAPACITY]> = SmallVec::new();
        let mut request_meta: SmallVec<[(usize, Vec<Aggregation>); GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            SmallVec::new();
        let mut result_names: SmallVec<[String; GROUPBY_RESULT_NAME_INLINE_CAPACITY]> = SmallVec::new();

        let mut col_name_to_view_idx: FastHashMap<String, usize> = fast_hash_map_with_capacity(specs.len());
        let mut view_idx_to_group_idx: FastHashMap<usize, usize> = fast_hash_map_with_capacity(specs.len());
        let mut output_order: SmallVec<[(usize, usize); GROUPBY_OUTPUT_ORDER_INLINE_CAPACITY]> = SmallVec::new();

        for spec in specs {
            let view_idx = if let Some(&idx) = col_name_to_view_idx.get(spec.column.as_str()) {
                idx
            } else {
                let arc_col = columns.get(spec.column.as_str()).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in DataFrame", spec.column))
                        .with_operation(operation)
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

        let agg_slices: SmallVec<[Vec<Aggregation>; GROUPBY_AGG_SLICE_INLINE_CAPACITY]> =
            request_meta.iter().map(|(_, aggs)| aggs.clone()).collect();

        let requests: SmallVec<[AggregationRequest<'_>; GROUPBY_REQUEST_INLINE_CAPACITY]> = request_meta
            .iter()
            .enumerate()
            .map(|(i, (view_idx, _))| AggregationRequest {
                values: &value_views[*view_idx],
                aggregations: &agg_slices[i],
            })
            .collect();

        let gb_result = if let Some(exec_options) = exec_options.as_ref() {
            groupby_aggregate_with_exec_options(
                keys_table.table(),
                &requests,
                self.options.null_policy(),
                exec_options,
                operation,
            )?
        } else {
            groupby_aggregate(keys_table.table(), &requests, self.options.null_policy())?
        };

        let mut grouped_results: Vec<Vec<Option<crate::Column>>> = gb_result
            .results
            .into_iter()
            .map(|cols| {
                cols.into_iter().map(|col| Some(col.with_execution_domain(Arc::clone(&output_domain)))).collect()
            })
            .collect();

        let mut all_result_cols: Vec<crate::Column> = Vec::with_capacity(output_order.len());
        for (request_index, (group_idx, agg_idx)) in output_order.iter().enumerate() {
            let group_count = grouped_results.len();
            let group = grouped_results.get_mut(*group_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result group index out of bounds")
                    .with_operation(operation)
                    .with_context("request_index", request_index.to_string())
                    .with_context("group_index", group_idx.to_string())
                    .with_context("group_count", group_count.to_string())
            })?;

            let col_slot = group.get_mut(*agg_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result index out of bounds")
                    .with_operation(operation)
                    .with_context("request_index", request_index.to_string())
                    .with_context("agg_index", agg_idx.to_string())
            })?;

            let col = col_slot.take().ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby result column consumed more than once")
                    .with_operation(operation)
                    .with_context("request_index", request_index.to_string())
                    .with_context("agg_index", agg_idx.to_string())
            })?;

            all_result_cols.push(col);
        }

        let result_keys_cols = gb_result.keys.with_execution_domain(Arc::clone(&output_domain)).into_columns()?;

        let mut final_cols: Vec<crate::Column> = Vec::with_capacity(result_keys_cols.len() + all_result_cols.len());
        final_cols.extend(result_keys_cols);
        final_cols.extend(all_result_cols);

        let mut schema_names: Vec<String> = Vec::with_capacity(self.keys.len() + result_names.len());
        schema_names.extend(self.keys);
        schema_names.extend(result_names);

        DataFrame::new(schema_names, final_cols)
    }

    /// Shortcut: compute the sum of each named column per group.
    pub fn sum<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Sum)).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the mean of each named column per group.
    pub fn mean<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Mean)).collect();
        self.agg(&specs)
    }

    /// Shortcut: count non-null values of each named column per group.
    pub fn count<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> =
            columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Count { include_nulls: false })).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the median of each named column per group.
    pub fn median<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Median)).collect();
        self.agg(&specs)
    }

    /// Shortcut: count unique non-null values of each named column per group.
    pub fn nunique<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> = columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::NUnique)).collect();
        self.agg(&specs)
    }

    /// Shortcut: compute the sample standard deviation (ddof=1) of each named
    /// column per group.
    pub fn std<I, S>(self, columns: I) -> Result<DataFrame>
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
    pub fn var<I, S>(self, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let specs: Vec<AggSpec> =
            columns.into_iter().map(|col| AggSpec::new(col.as_ref(), Agg::Variance { ddof: 1 })).collect();
        self.agg(&specs)
    }
}
