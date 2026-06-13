//! A named-column wrapper around GPU columns for ergonomic DataFrame operations.
//!
//! [`DataFrame`] stores columns individually as `Arc<Column>`, enabling
//! schema-only operations like rename, select, without_columns, and clone to
//! share GPU memory via reference counting instead of deep-copying column
//! data.
use std::sync::Arc;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
        table::{Table, TableView},
    },
    frame::series::Series,
    internal::{
        hash::{FastIndexMap, fast_index_map, fast_index_map_with_capacity},
        resolve::validate_no_duplicate_names,
    },
    ops::groupby,
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// A GPU DataFrame: a named, ordered collection of GPU columns.
///
/// `DataFrame` is an immutable value — every mutating operation returns a
/// **new** `DataFrame` rather than modifying in place.
///
/// Columns are stored as `Arc<Column>`, so operations that don't change
/// column data (rename, select, without_columns, clone, with_column for
/// unchanged columns) share GPU memory via reference counting.
///
/// Cloning a `DataFrame` is cheap: it clones the `IndexMap` and bumps
/// `Arc` reference counts — no GPU data is copied.
///
/// # Display
///
/// [`std::fmt::Display`] for `DataFrame` is metadata-only. It reports row and
/// column counts plus per-column schema details, but it does not materialize
/// cell values to host memory.
///
/// This keeps `Display` distinct from row-selection APIs such as
/// [`head`](Self::head), [`tail`](Self::tail), and [`slice`](Self::slice).
/// Use those APIs to select row subsets, then inspect the resulting columns
/// explicitly when you need host-visible values.
#[derive(Clone)]
pub struct DataFrame {
    pub(super) columns: FastIndexMap<String, Arc<Column>>,
    pub(super) row_count: usize,
    pub(super) domain: Arc<ExecutionDomain>,
}

/// Split output of a DataFrame join operation.
///
/// This is the compatibility form containing the raw join sides. Each side
/// has columns `[key_columns..., payload_columns...]`.
#[derive(Debug, Clone)]
pub struct DataFrameJoinSides {
    /// Left-side result: key columns + left payload columns.
    pub left: DataFrame,
    /// Right-side result: key columns + right payload columns.
    pub right: DataFrame,
}

fn empty_dataframe_requires_runtime_error(operation: &'static str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        "creating an empty dataframe requires an explicit runtime because no authoritative input columns exist",
    )
    .with_operation(operation)
    .with_context("remedy", "use Runtime::empty_dataframe()")
}

fn named_column_len(name: &str, column: &Arc<Column>, operation: &'static str) -> Result<usize> {
    column
        .view()
        .map_err(|e| e.with_operation(operation).with_context("column_name", name.to_string()))?
        .len()
        .map_err(|e| e.with_operation(operation).with_context("column_name", name.to_string()))
}

fn dataframe_from_named_column_arcs<I>(named_columns: I, operation: &'static str) -> Result<DataFrame>
where I: IntoIterator<Item = (String, Arc<Column>)> {
    let named_columns = named_columns.into_iter().collect::<Vec<_>>();
    if named_columns.is_empty() {
        return Err(empty_dataframe_requires_runtime_error(operation));
    }

    let names = named_columns.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    validate_no_duplicate_names(&names, operation)?;

    let mut named_columns = named_columns.into_iter();
    let (first_name, first_column) = named_columns.next().ok_or_else(|| {
        crate::core::error::invariant_violation(
            operation,
            "non-empty dataframe construction unexpectedly lost its first column",
        )
    })?;
    let first_rows = named_column_len(&first_name, &first_column, operation)?;

    let mut columns: FastIndexMap<String, Arc<Column>> = fast_index_map_with_capacity(names.len());
    columns.insert(first_name.clone(), first_column);

    for (name, column) in named_columns {
        let rows = named_column_len(&name, &column, operation)?;
        if rows != first_rows {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column '{}' has {} rows but first column has {} rows", name, rows, first_rows),
            )
            .with_operation(operation)
            .with_context("column_name", name)
            .with_context("column_rows", rows.to_string())
            .with_context("expected_rows", first_rows.to_string()));
        }
        columns.insert(name, column);
    }

    let domain = derive_frame_domain_from_columns(&columns, operation)?;
    Ok(DataFrame { columns, row_count: first_rows, domain })
}

pub(crate) fn derive_frame_domain_from_columns(
    columns: &FastIndexMap<String, Arc<Column>>,
    operation: &'static str,
) -> Result<Arc<ExecutionDomain>> {
    let Some((first_name, first_column)) = columns.iter().next() else {
        return Err(empty_dataframe_requires_runtime_error(operation));
    };

    let domain = Arc::clone(first_column.execution_domain());
    for (name, column) in columns.iter().skip(1) {
        ExecutionDomain::require_same_provenance(
            &domain,
            column.execution_domain(),
            operation,
            "dataframe construction requires all columns to already share one retained provenance",
            "use copy_into_runtime(...) to relocate all Series inputs into one Runtime before constructing the DataFrame",
        )
        .map_err(|e| e.with_context("column_name", name.clone()).with_context("frame_column", first_name.clone()))?;
    }

    Ok(domain)
}

pub(crate) fn frame_domain_matches_columns(
    columns: &FastIndexMap<String, Arc<Column>>,
    frame_domain: &Arc<ExecutionDomain>,
) -> bool {
    columns.values().all(|column| ExecutionDomain::same_provenance(frame_domain, column.execution_domain()))
}

pub(crate) fn resolve_frame_domain_from_authoritative_source(
    columns: &FastIndexMap<String, Arc<Column>>,
    authoritative_domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<Arc<ExecutionDomain>> {
    if columns.is_empty() {
        return Ok(Arc::clone(authoritative_domain));
    }

    for (name, column) in columns {
        if !ExecutionDomain::same_provenance(authoritative_domain, column.execution_domain()) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "all columns in a dataframe must share the authoritative retained provenance",
            )
            .with_operation(operation)
            .with_context("left_domain", authoritative_domain.debug_label())
            .with_context("right_domain", column.execution_domain().debug_label())
            .with_context("column_name", name.clone()));
        }
    }

    Ok(Arc::clone(authoritative_domain))
}

impl DataFrame {
    pub(crate) fn empty_in_domain(domain: Arc<ExecutionDomain>) -> Self {
        Self { columns: fast_index_map(), row_count: 0, domain }
    }

    /// Crate-private constructor from column names and owned [`Column`] values.
    ///
    /// Advanced constructor. For most user-facing code, prefer
    /// [`DataFrame::from_columns`] with explicit schema names.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `names.len() != columns.len()`
    /// - Any name is duplicated
    /// - Columns have different row counts
    pub(crate) fn new(names: Vec<String>, columns: Vec<Column>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of columns")
                .with_operation("dataframe::new")
                .with_context("names_count", names.len().to_string())
                .with_context("columns_count", columns.len().to_string()));
        }

        validate_no_duplicate_names(&names, "dataframe::new")?;

        if columns.is_empty() {
            return Err(empty_dataframe_requires_runtime_error("dataframe::new"));
        }

        // Validate all columns have the same row count.
        let first_rows = columns[0].view()?.len()?;
        for (i, col) in columns.iter().enumerate().skip(1) {
            let rows = col.view()?.len()?;
            if rows != first_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column {} has {} rows but column 0 has {} rows", i, rows, first_rows),
                )
                .with_operation("dataframe::new")
                .with_context("column_index", i.to_string())
                .with_context("column_rows", rows.to_string())
                .with_context("expected_rows", first_rows.to_string()));
            }
        }

        let mut map: FastIndexMap<String, Arc<Column>> = fast_index_map_with_capacity(names.len());
        for (name, col) in names.into_iter().zip(columns) {
            map.insert(name, Arc::new(col));
        }

        let domain = derive_frame_domain_from_columns(&map, "dataframe::new")?;
        Ok(Self { columns: map, row_count: first_rows, domain })
    }

    /// Wrap an existing [`Table`] with column names.
    ///
    /// # Errors
    ///
    /// Returns an error if `names.len()` does not match the table's column
    /// count, or if names contain duplicates.
    pub fn from_table(table: Table, names: Vec<String>) -> Result<Self> {
        let view = table.view()?;
        let num_cols = view.num_columns()?;
        let row_count = if num_cols > 0 { view.num_rows()? } else { 0 };
        drop(view);
        let table_domain = Arc::clone(table.execution_domain());

        let expected = names.len();
        if expected != num_cols {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of table columns")
                .with_operation("dataframe::from_table")
                .with_context("names_count", names.len().to_string())
                .with_context("table_columns", num_cols.to_string()));
        }

        validate_no_duplicate_names(&names, "dataframe::from_table")?;

        let cols = table.into_columns()?;
        let mut map: FastIndexMap<String, Arc<Column>> = fast_index_map_with_capacity(names.len());
        for (name, col) in names.into_iter().zip(cols) {
            map.insert(name, Arc::new(col));
        }

        let domain = resolve_frame_domain_from_authoritative_source(&map, &table_domain, "dataframe::from_table")?;
        Ok(Self { columns: map, row_count, domain })
    }

    /// Create a `DataFrame` from an iterator of `(name, Series)` pairs.
    ///
    /// `DataFrame` is the schema authority: the pair key becomes the column
    /// name in the resulting frame, regardless of any name metadata carried by
    /// the `Series` itself. This also means unnamed `Series` are accepted as
    /// long as the caller supplies an explicit output name here.
    ///
    /// GPU data is shared via `Arc` — no device copy.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any name is duplicated
    /// - Columns have different row counts
    pub fn from_columns<I, S>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: Into<String>,
    {
        let iter = pairs.into_iter();
        let (lower, _) = iter.size_hint();
        let mut named_columns = Vec::with_capacity(lower);

        for (name, series) in iter {
            named_columns.push((name.into(), Arc::clone(series.column_arc())));
        }

        dataframe_from_named_column_arcs(named_columns, "dataframe::from_columns")
    }

    /// Construct a `DataFrame` directly from pre-validated parts.
    ///
    /// This skips all validation (duplicate names, row-count consistency)
    /// and is intended for internal reassembly helpers that have already
    /// performed those checks.
    ///
    /// # Safety contract (logic, not memory)
    ///
    /// The caller must ensure:
    /// - No duplicate keys in `columns`
    /// - All columns have `row_count` rows
    pub(crate) fn from_raw_parts(
        columns: FastIndexMap<String, Arc<Column>>,
        row_count: usize,
        domain: Arc<ExecutionDomain>,
    ) -> Self {
        debug_assert!(
            frame_domain_matches_columns(&columns, &domain),
            "dataframe::from_raw_parts requires all column domains to match the frame domain",
        );
        Self { columns, row_count, domain }
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the logical number of rows.
    ///
    /// This is cached DataFrame metadata. Schema-only operations such as
    /// [`select`](Self::select) and [`without_columns`](Self::without_columns) preserve it even if
    /// the resulting DataFrame has zero columns. Bare constructors with no
    /// columns, however, create a `0 x 0` DataFrame because they have no
    /// external row index to preserve.
    pub fn num_rows(&self) -> usize {
        self.row_count
    }

    /// Build an owned [`Table`] by deep-copying each column.
    ///
    /// This is useful when calling APIs that operate on `Table` directly.
    ///
    /// # Cost
    ///
    /// GPU-allocating: deep-copies every column to a new `Table`. Each column
    /// gets its own fresh device allocation.
    pub fn to_table_copy(&self) -> Result<Table> {
        let mut cols: Vec<Column> = Vec::with_capacity(self.columns.len());
        for (name, arc_col) in &self.columns {
            let col = arc_col
                .copy_into_domain(Arc::clone(&self.domain), "dataframe::to_table_copy")
                .map_err(|e| e.with_context("column_name", name.clone()))?;
            cols.push(col);
        }

        let table = if cols.is_empty() {
            Table::empty_in_domain(Arc::clone(&self.domain), "dataframe::to_table_copy")?
        } else {
            Table::from_columns(cols).map_err(|e| e.with_operation("dataframe::to_table_copy"))?
        };
        Ok(table.with_execution_domain(Arc::clone(&self.domain)))
    }

    /// Ensure this DataFrame is placed in the target runtime.
    ///
    /// Columns already in the target runtime's retained provenance are shared
    /// without copying. Columns from a different retained provenance are
    /// fenced and deep-copied into the target runtime.
    pub fn copy_into_runtime(&self, runtime: &Runtime) -> Result<Self> {
        let operation = "dataframe::copy_into_runtime";
        let output_domain = Arc::clone(runtime.execution_domain());
        let columns = self.clone_or_copy_columns_into_domain(&output_domain, operation)?;
        let domain = resolve_frame_domain_from_authoritative_source(&columns, &output_domain, operation)?;
        Ok(Self { columns, row_count: self.row_count, domain })
    }

    /// Consume the `DataFrame`, returning an owned table and column names.
    ///
    /// Columns that are uniquely owned (`Arc` strong count == 1) are moved
    /// without copying. Shared columns (strong count > 1) are deep-copied
    /// so the returned `Table` owns independent device memory.
    ///
    /// # Cost
    ///
    /// Conditional: zero-copy for uniquely-owned columns, GPU-allocating
    /// (deep-copy) for shared columns. If you know all columns are unique
    /// (e.g. the `DataFrame` was just constructed), this is effectively free.
    pub fn try_into_table(self) -> Result<(Table, Vec<String>)> {
        let mut names: Vec<String> = Vec::with_capacity(self.columns.len());
        let mut cols: Vec<Column> = Vec::with_capacity(self.columns.len());

        for (name, arc_col) in self.columns {
            let col = match Arc::try_unwrap(arc_col) {
                Ok(col) => col,
                Err(shared) => shared
                    .copy_into_domain(Arc::clone(&self.domain), "dataframe::try_into_table")
                    .map_err(|e| e.with_context("column_name", name.clone()))?,
            };
            names.push(name);
            cols.push(col);
        }

        let table = if cols.is_empty() {
            Table::empty_in_domain(Arc::clone(&self.domain), "dataframe::try_into_table")?
        } else {
            Table::from_columns(cols).map_err(|e| e.with_operation("dataframe::try_into_table"))?
        };
        let table = table.with_execution_domain(Arc::clone(&self.domain));
        Ok((table, names))
    }

    // -- Internal helpers ----------------------------------------------------

    /// Build a borrowed [`TableView`] from the individual `Arc<Column>`s.
    pub(crate) fn build_table_view(&self) -> Result<TableView<'_>> {
        let views: Vec<ColumnView<'_>> = self.columns.values().map(|col| col.view()).collect::<Result<Vec<_>>>()?;
        TableView::from_borrowed_column_views_in_domain(
            views,
            Some(Arc::clone(&self.domain)),
            "dataframe::build_table_view",
        )
    }

    /// Create a DataFrame from a Table and column names, decomposing the
    /// table into Arc-wrapped columns.
    pub(crate) fn from_table_with_names(table: Table, names: Vec<String>) -> Result<Self> {
        crate::internal::reassemble::dataframe_from_table_and_names(table, names)
    }

    /// Returns owned column names.
    pub(crate) fn column_names_owned(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Returns a reference to the internal column map.
    ///
    /// This is a crate-internal accessor used by high-level types like
    /// [`GroupBy`][crate::ops::groupby::GroupBy].
    pub(crate) fn columns_map(&self) -> &FastIndexMap<String, Arc<Column>> {
        &self.columns
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }
    pub(crate) fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
    pub(crate) fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }
    pub(crate) fn __debug_execution_domain_mr_value(&self) -> usize {
        self.domain.debug_mr_value()
    }

    pub(crate) fn clone_or_copy_columns_into_domain(
        &self,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<FastIndexMap<String, Arc<Column>>> {
        if ExecutionDomain::same_provenance(&self.domain, output_domain) {
            return Ok(self.columns.clone());
        }

        ExecutionDomain::insert_stream_dependency(&self.domain, output_domain, operation)
            .map_err(|e| e.with_context("target_domain", output_domain.debug_label()))?;

        let mut copied: FastIndexMap<String, Arc<Column>> = fast_index_map_with_capacity(self.columns.len());
        for (name, column) in &self.columns {
            let rebound = column.copy_into_domain(Arc::clone(output_domain), operation).map_err(|e| {
                e.with_context("column_name", name.clone()).with_context("target_domain", output_domain.debug_label())
            })?;
            copied.insert(name.clone(), Arc::new(rebound));
        }
        Ok(copied)
    }

    pub(crate) fn validate_series_insertion_domain(
        &self,
        series: &Series,
        operation: &'static str,
        column_name: &str,
    ) -> Result<()> {
        if ExecutionDomain::same_provenance(series.as_column().execution_domain(), &self.domain) {
            return Ok(());
        }

        Err(Error::new(
            ErrorKind::InvalidOperation,
            "mixed-provenance dataframe column insertion requires explicit relocation into one runtime before insertion",
        )
        .with_operation(operation)
        .with_context("column_name", column_name.to_string())
        .with_context("frame_domain", self.domain.debug_label())
        .with_context("series_domain", series.as_column().execution_domain().debug_label())
        .with_context("frame_device", self.domain.device_snapshot().to_string())
        .with_context("series_device", series.as_column().execution_domain().device_snapshot().to_string())
        .with_context(
            "remedy",
            "use copy_into_runtime(...) to relocate the DataFrame or incoming Series into one Runtime before inserting it",
        ))
    }
    /// Start a groupby operation.
    ///
    /// Returns a [`GroupBy`](crate::GroupBy) on which you call [`.agg()`][groupby::GroupBy::agg]
    /// with [`AggSpec`][groupby::AggSpec] items, or use shortcut methods like
    /// `.sum()`, `.mean()`, `.count()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::{GroupByAgg, AggSpec, DataFrame, Runtime};
    /// # let rt = Runtime::new().unwrap();
    /// # let keys = rt.series(&[1i32, 2]).unwrap();
    /// # let vals = rt.series(&[10i64, 20]).unwrap();
    /// # let df = DataFrame::from_columns([("key", keys), ("val", vals)]).unwrap();
    /// let result =
    ///     df.groupby(&["key"]).agg(&[AggSpec::new("val", GroupByAgg::Sum)]).unwrap();
    /// ```
    pub fn groupby<I, S>(&self, keys: I) -> groupby::GroupBy<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        groupby::GroupBy::new(self, keys.into_iter().map(|s| s.as_ref().to_string()).collect())
    }
}

/// Construct a [`DataFrame`](crate::frame::dataframe::DataFrame) from named pairs.
///
/// This macro is a shorthand for [`DataFrame::from_columns`]. Each value
/// must evaluate to a [`Series`](crate::Series). The pair key becomes the
/// DataFrame schema name, overriding any name carried by the Series.
///
/// # Example
///
/// ```no_run
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let x = rt.series(&[1i32, 2, 3]).unwrap();
/// let y = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
/// let df = cudf::dataframe! { "x" => x, "y" => y }.unwrap();
/// ```
///
/// ```compile_fail
/// use cudf::Runtime;
///
/// let rt = Runtime::new().unwrap();
/// let x = rt.column_from_slice(&[1i32, 2, 3]).unwrap();
/// let _ = cudf::dataframe! { "x" => x }.unwrap();
/// ```
#[macro_export]
macro_rules! dataframe {
    ($($name:expr => $series:expr),+ $(,)?) => {{
        $crate::DataFrame::from_columns(vec![
            $(($name.to_string(), $series)),+
        ])
    }};
}
