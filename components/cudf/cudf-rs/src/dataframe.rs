//! A named-column wrapper around GPU columns for ergonomic DataFrame operations.
//!
//! [`DataFrame`] stores columns individually as `Arc<Column>`, enabling
//! zero-copy operations like rename, select, and clone that share GPU memory
//! via reference counting instead of deep-copying all column data.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DataFrame, Series};
//!
//! let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
//! let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();
//!
//! let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
//!
//! assert_eq!(df.num_columns(), 2);
//! let score = df.series("score").unwrap();
//! assert_eq!(score.len(), 3);
//! ```

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use indexmap::IndexMap;

use crate::{
    column::{Column, ColumnView},
    copying,
    dataframe_view::DataFrameView,
    error::{Error, ErrorKind, Result},
    execution_domain::ExecutionDomain,
    filter, groupby,
    io::{
        csv::{self, CsvReaderOptions, CsvWriterOptions},
        parquet::{self, ParquetReaderOptions, ParquetWriterOptions},
    },
    join, replace,
    scalar::Scalar,
    series::Series,
    series_view::SeriesView,
    sorting::{self, SortOptions},
    stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
    table::Table,
    types::{CudfType, DataType},
    unary,
};

/// A GPU DataFrame: a named, ordered collection of GPU columns.
///
/// `DataFrame` is an immutable value — every mutating operation returns a
/// **new** `DataFrame` rather than modifying in place.
///
/// Columns are stored as `Arc<Column>`, so operations that don't change
/// column data (rename, select, clone, with_column for unchanged columns)
/// share GPU memory via reference counting.
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
///
/// # Consuming variants
///
/// Schema-only operations have `_owned` variants that take `self` by value:
/// [`rename_owned`](Self::rename_owned), [`select_owned`](Self::select_owned),
/// [`with_column_owned`](Self::with_column_owned). These avoid the O(N)
/// `Arc::clone` overhead for N columns by moving ownership directly.
/// Prefer them when the source DataFrame is no longer needed.
///
/// GPU-allocating operations (head, tail, filter, sort, etc.) do **not**
/// have `_owned` variants because FFI always allocates new device memory,
/// making the `Arc::clone` cost negligible relative to the GPU allocation.
#[derive(Clone)]
pub struct DataFrame {
    columns: IndexMap<String, Arc<Column>>,
    row_count: usize,
    domain: Arc<ExecutionDomain>,
}

/// Result of a split DataFrame join operation.
///
/// This is the compatibility form containing the raw join sides. Each side
/// has columns `[key_columns..., payload_columns...]`.
#[derive(Debug, Clone)]
pub struct DataFrameJoinResult {
    /// Left-side result: key columns + left payload columns.
    pub left: DataFrame,
    /// Right-side result: key columns + right payload columns.
    pub right: DataFrame,
}

/// Options for [`DataFrame::read_csv_with_options`].
///
/// This is the schema-aware CSV facade for `DataFrame`. Unlike the lower-level
/// [`crate::io::csv::CsvReaderOptions`], it can also resolve DataFrame column
/// names from either an explicit override, the CSV header row, or generated
/// fallback names.
#[derive(Debug, Clone)]
pub struct DataFrameCsvReaderOptions {
    filepath: PathBuf,
    delimiter: u8,
    has_header: bool,
    column_names: Option<Vec<String>>,
}

impl DataFrameCsvReaderOptions {
    /// Create options to read a CSV file into a `DataFrame`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { filepath: path.as_ref().to_path_buf(), delimiter: b',', has_header: true, column_names: None }
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether the file has a header row (default: `true`).
    pub fn has_header(mut self, has: bool) -> Self {
        self.has_header = has;
        self
    }

    /// Override the output DataFrame column names.
    ///
    /// When set, these names take precedence over both header-derived names
    /// and generated fallback names.
    pub fn column_names(mut self, names: &[&str]) -> Self {
        self.column_names = Some(names.iter().map(|s| (*s).to_string()).collect());
        self
    }

    fn to_table_options(&self) -> CsvReaderOptions {
        CsvReaderOptions::new(&self.filepath).delimiter(self.delimiter).has_header(self.has_header)
    }

    fn resolve_column_names(&self, num_columns: usize) -> Result<Vec<String>> {
        let operation = "dataframe::read_csv_with_options";

        if let Some(names) = self.column_names.as_ref() {
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names.clone());
        }

        if self.has_header {
            let names = self.read_header_names()?;
            validate_dataframe_csv_column_name_count(names.len(), num_columns, operation)?;
            return Ok(names);
        }

        Ok((0..num_columns).map(|idx| format!("column_{idx}")).collect())
    }

    fn read_header_names(&self) -> Result<Vec<String>> {
        let operation = "dataframe::read_csv_with_options";

        let mut reader =
            ::csv::ReaderBuilder::new().has_headers(true).delimiter(self.delimiter).from_path(&self.filepath).map_err(
                |source| {
                    Error::new(ErrorKind::IoError, "failed to open CSV file for header parsing")
                        .with_operation(operation)
                        .with_context("path", self.filepath.display().to_string())
                        .set_source(source)
                },
            )?;

        let headers = reader.headers().map_err(|source| {
            Error::new(ErrorKind::IoError, "failed to read CSV header")
                .with_operation(operation)
                .with_context("path", self.filepath.display().to_string())
                .set_source(source)
        })?;

        Ok(headers.iter().map(str::to_owned).collect())
    }
}

/// Options for [`DataFrame::write_csv_with_options`].
///
/// This is the schema-aware CSV facade for `DataFrame`. Column names always
/// come from the DataFrame schema; this type only controls file-format knobs
/// such as delimiter and whether the header row is included.
#[derive(Debug, Clone)]
pub struct DataFrameCsvWriterOptions {
    filepath: PathBuf,
    delimiter: u8,
    include_header: bool,
}

impl DataFrameCsvWriterOptions {
    /// Create options to write a `DataFrame` to a CSV file.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { filepath: path.as_ref().to_path_buf(), delimiter: b',', include_header: true }
    }

    /// Set the field delimiter character (default: `,`).
    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Set whether to include a header row (default: `true`).
    pub fn include_header(mut self, include: bool) -> Self {
        self.include_header = include;
        self
    }

    fn to_table_options(&self, column_names: &[&str]) -> CsvWriterOptions {
        CsvWriterOptions::new(&self.filepath)
            .column_names(column_names)
            .delimiter(self.delimiter)
            .include_header(self.include_header)
    }
}

fn validate_dataframe_csv_column_name_count(
    name_count: usize,
    num_columns: usize,
    operation: &'static str,
) -> Result<()> {
    if name_count != num_columns {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("column name count ({name_count}) does not match table column count ({num_columns})"),
        )
        .with_operation(operation)
        .with_context("column_name_count", name_count.to_string())
        .with_context("table_column_count", num_columns.to_string()));
    }

    Ok(())
}

pub(crate) fn derive_frame_domain_from_columns(
    columns: &IndexMap<String, Arc<Column>>,
    operation: &'static str,
) -> Result<Arc<ExecutionDomain>> {
    let Some((first_name, first_column)) = columns.iter().next() else {
        return Ok(ExecutionDomain::ambient_default());
    };

    let domain = Arc::clone(first_column.execution_domain());
    for (name, column) in columns.iter().skip(1) {
        if !ExecutionDomain::same_identity(&domain, column.execution_domain()) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "all columns in a dataframe must share one execution domain",
            )
            .with_operation(operation)
            .with_context("left_domain", domain.debug_label())
            .with_context("right_domain", column.execution_domain().debug_label())
            .with_context("column_name", name.clone())
            .with_context("frame_column", first_name.clone()));
        }
    }

    Ok(domain)
}

/// Options for [`DataFrame::dropna`].
///
/// Controls which rows are dropped based on null values. Mirrors the
/// `dropna()` API from cuDF Python.
///
/// # Defaults
///
/// - `how` = any (drop if **any** subset column is null)
/// - `subset` = all columns
/// - `thresh` = `None` (when set, overrides `how`)
#[derive(Debug, Clone)]
pub struct DropNaOptions {
    /// Column names to inspect. Empty means all columns.
    subset: Vec<String>,
    /// Minimum number of non-null values in subset required to keep the row.
    /// `None` means derive from `how`.
    thresh: Option<i32>,
    /// If `true`, drop only when **all** subset columns are null.
    /// If `false` (default), drop when **any** subset column is null.
    how_all: bool,
}

impl Default for DropNaOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DropNaOptions {
    /// Create default options: how=any, subset=all columns, no threshold override.
    pub fn new() -> Self {
        Self { subset: Vec::new(), thresh: None, how_all: false }
    }

    /// Specify which columns to check for nulls. If not called (or empty),
    /// all columns are checked.
    pub fn subset(mut self, columns: &[&str]) -> Self {
        self.subset = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Drop rows where **any** of the subset columns is null (default).
    pub fn how_any(mut self) -> Self {
        self.how_all = false;
        self
    }

    /// Drop rows only when **all** subset columns are null.
    pub fn how_all(mut self) -> Self {
        self.how_all = true;
        self
    }

    /// Set an explicit threshold: keep rows with at least `thresh` non-null
    /// values in the subset columns. When set, this overrides `how`.
    pub fn thresh(mut self, thresh: i32) -> Self {
        self.thresh = Some(thresh);
        self
    }
}

/// Options for [`DataFrame::drop_duplicates`].
///
/// Controls deduplication behavior. Mirrors `drop_duplicates()` from cuDF
/// Python.
///
/// # Defaults
///
/// - `subset` = all columns
/// - `keep` = [`DuplicateKeepOption::KeepAny`]
#[derive(Debug, Clone)]
pub struct DistinctOptions {
    /// Column names to use as deduplication keys. Empty means all columns.
    subset: Vec<String>,
    /// Which duplicate occurrence to keep.
    keep: DuplicateKeepOption,
}

impl Default for DistinctOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DistinctOptions {
    /// Create default options: all columns, keep any.
    pub fn new() -> Self {
        Self { subset: Vec::new(), keep: DuplicateKeepOption::KeepAny }
    }

    /// Specify which columns to use as deduplication keys. If not called
    /// (or empty), all columns are used.
    pub fn subset(mut self, columns: &[&str]) -> Self {
        self.subset = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Which duplicate occurrence to keep.
    pub fn keep(mut self, keep: DuplicateKeepOption) -> Self {
        self.keep = keep;
        self
    }
}

impl std::fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Debug shows schema, row count, and column types — NO GPU data copy.
        let num_cols = self.columns.len();
        let num_rows = self.num_rows();

        let mut ds = f.debug_struct("DataFrame");
        ds.field("num_columns", &num_cols);
        ds.field("num_rows", &num_rows);

        // Build a columns summary: [(name, type)] without GPU data copy.
        let columns_info: Vec<(&str, String)> = self
            .columns
            .keys()
            .map(|name| {
                let type_str = self
                    .series_view(name)
                    .and_then(|cv| cv.data_type())
                    .map(|dt| format!("{:?}", dt.id()))
                    .unwrap_or_else(|_| "?".to_string());
                (name.as_str(), type_str)
            })
            .collect();
        ds.field("columns", &columns_info);
        ds.finish()
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display_dataframe(self, f)
    }
}

/// Cached schema metadata for one display column.
struct DisplayColumnMetadata {
    type_str: String,
    nullable: Option<bool>,
    null_count: Option<i32>,
}

impl DisplayColumnMetadata {
    fn unknown() -> Self {
        Self { type_str: "?".to_string(), nullable: None, null_count: None }
    }

    fn details(&self) -> Option<String> {
        let mut details = Vec::new();

        if matches!(self.nullable, Some(true)) {
            details.push("nullable".to_string());
        }

        if let Some(null_count) = self.null_count {
            if null_count > 0 {
                details.push(format!("nulls={null_count}"));
            }
        }

        (!details.is_empty()).then(|| details.join(", "))
    }
}

/// Build one display column schema summary from a named DataFrame column.
fn build_display_column_metadata(df: &DataFrame, name: &str) -> DisplayColumnMetadata {
    let cv = match df.series_view(name) {
        Ok(cv) => cv,
        Err(_) => return DisplayColumnMetadata::unknown(),
    };

    let type_id = match cv.data_type() {
        Ok(dt) => dt.id(),
        Err(_) => return DisplayColumnMetadata::unknown(),
    };

    DisplayColumnMetadata {
        type_str: format!("{:?}", type_id),
        nullable: cv.nullable().ok(),
        null_count: cv.null_count().ok(),
    }
}

/// Format a DataFrame as a human-readable metadata summary.
///
/// Shows row and column counts plus per-column schema metadata. This is
/// intentionally metadata-only and never materializes cell values to host.
fn display_dataframe(df: &DataFrame, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let num_cols = df.num_columns();
    let num_rows = df.num_rows();
    write!(f, "DataFrame [{} columns x {} rows]", num_cols, num_rows)?;

    if num_cols == 0 {
        return Ok(());
    }

    let names: Vec<&str> = df.column_names();
    let display_cols: Vec<DisplayColumnMetadata> =
        names.iter().map(|&name| build_display_column_metadata(df, name)).collect();

    write!(f, "\ncolumns:")?;
    for (name, meta) in names.iter().zip(display_cols.iter()) {
        write!(f, "\n- {name}: {}", meta.type_str)?;
        if let Some(details) = meta.details() {
            write!(f, " ({details})")?;
        }
    }

    Ok(())
}

impl DataFrame {
    /// Create a new `DataFrame` from column names and owned [`Column`] values.
    ///
    /// Advanced constructor. For most user-facing code, prefer
    /// [`DataFrame::from_columns`] with named [`Series`](crate::Series)
    /// values.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `names.len() != columns.len()`
    /// - Any name is duplicated
    /// - Columns have different row counts
    pub fn new(names: Vec<String>, columns: Vec<Column>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of columns")
                .with_operation("dataframe::new")
                .with_context("names_count", names.len().to_string())
                .with_context("columns_count", columns.len().to_string()));
        }

        validate_no_duplicate_names(&names, "dataframe::new")?;

        if columns.is_empty() {
            return Ok(Self { columns: IndexMap::new(), row_count: 0, domain: ExecutionDomain::ambient_default() });
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

        let mut map = IndexMap::with_capacity(names.len());
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
        let row_count = if num_cols > 0 { view.num_rows()? as usize } else { 0 };
        drop(view);

        let expected = names.len() as i32;
        if expected != num_cols {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of table columns")
                .with_operation("dataframe::from_table")
                .with_context("names_count", names.len().to_string())
                .with_context("table_columns", num_cols.to_string()));
        }

        validate_no_duplicate_names(&names, "dataframe::from_table")?;

        let cols = table.into_columns()?;
        let mut map = IndexMap::with_capacity(names.len());
        for (name, col) in names.into_iter().zip(cols) {
            map.insert(name, Arc::new(col));
        }

        let domain = derive_frame_domain_from_columns(&map, "dataframe::from_table")?;
        Ok(Self { columns: map, row_count, domain })
    }

    /// Create a `DataFrame` from a fixed-size array of `(name, Series)` pairs.
    ///
    /// This is the ergonomic constructor for building DataFrames from Series.
    /// Each Series' `Arc<Column>` is shared (zero GPU copy).
    ///
    /// The pair name becomes the DataFrame schema name. The input `Series`
    /// value is treated as the data source; its own `Series::name()` does not
    /// control the output column name.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any name is duplicated
    /// - Columns have different row counts
    pub fn from_columns<const N: usize>(pairs: [(&str, Series); N]) -> Result<Self> {
        Self::from_series(pairs)
    }

    /// Create a `DataFrame` from an iterator of `(name, Series)` pairs.
    ///
    /// Each Series' `Arc<Column>` is shared (zero GPU copy).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any name is duplicated
    /// - Columns have different row counts
    pub fn from_series<I, S>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: Into<String>,
    {
        let iter = pairs.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = IndexMap::with_capacity(lower);

        for (name, series) in iter {
            let name = name.into();
            if map.contains_key(&name) {
                return Err(Error::new(ErrorKind::InvalidOperation, format!("duplicate column name '{}'", name))
                    .with_operation("dataframe::from_series")
                    .with_context("duplicate_name", name));
            }
            map.insert(name, Arc::clone(series.column_arc()));
        }

        if map.is_empty() {
            return Ok(Self { columns: IndexMap::new(), row_count: 0, domain: ExecutionDomain::ambient_default() });
        }

        // Validate all columns have the same row count.
        let mut iter = map.iter();
        let (first_name, first_col) = iter.next().expect("non-empty: checked above");
        let first_rows = first_col
            .view()
            .map_err(|e| e.with_operation("dataframe::from_series").with_context("column_name", first_name.clone()))?
            .len()
            .map_err(|e| e.with_operation("dataframe::from_series").with_context("column_name", first_name.clone()))?;

        for (name, col) in iter {
            let rows = col
                .view()
                .map_err(|e| e.with_operation("dataframe::from_series").with_context("column_name", name.clone()))?
                .len()
                .map_err(|e| e.with_operation("dataframe::from_series").with_context("column_name", name.clone()))?;
            if rows != first_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column '{}' has {} rows but first column has {} rows", name, rows, first_rows),
                )
                .with_operation("dataframe::from_series")
                .with_context("column_name", name.clone())
                .with_context("column_rows", rows.to_string())
                .with_context("expected_rows", first_rows.to_string()));
            }
        }

        let domain = derive_frame_domain_from_columns(&map, "dataframe::from_series")?;
        Ok(Self { columns: map, row_count: first_rows, domain })
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
        columns: IndexMap<String, Arc<Column>>,
        row_count: usize,
        domain: Arc<ExecutionDomain>,
    ) -> Self {
        Self { columns, row_count, domain }
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the logical number of rows.
    ///
    /// This is cached DataFrame metadata. Schema-only operations such as
    /// [`select`](Self::select) and [`drop`](Self::drop) preserve it even if
    /// the resulting DataFrame has zero columns. Bare constructors with no
    /// columns, however, create a `0 x 0` DataFrame because they have no
    /// external row index to preserve.
    pub fn num_rows(&self) -> usize {
        self.row_count
    }

    /// Returns column names in insertion (schema) order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Borrow a named [`SeriesView`] by name — the explicit borrow accessor.
    ///
    /// The returned view carries the column name and implements
    /// `Deref<Target = ColumnView>`, so all low-level column methods are
    /// available directly.
    ///
    /// This is the explicit borrow path. For owned access, use
    /// [`series()`](Self::series) instead.
    ///
    /// # Cost
    ///
    /// Zero-copy (FFI view): creates a borrowed `ColumnView` over existing device memory.
    /// No GPU allocation or copy.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not in the schema.
    pub fn series_view(&self, name: &str) -> Result<SeriesView<'_>> {
        let (key, col) = self.columns.get_key_value(name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                .with_operation("dataframe::series_view")
                .with_context("column_name", name.to_string())
        })?;

        let cv = col.view()?;
        Ok(SeriesView::new(Some(key.as_str()), cv))
    }

    /// Return a named [`Series`] for the given column — the canonical user-facing accessor.
    ///
    /// The returned `Series` shares the underlying `Arc<Column>`, so this is a
    /// zero-copy metadata operation. The series name is set to the DataFrame
    /// column name.
    ///
    /// Use this method for all high-level column access. For a borrowed
    /// [`SeriesView`] without an `Arc` bump, use
    /// [`series_view()`](Self::series_view).
    ///
    /// # Cost
    ///
    /// Zero-copy (Arc-share, metadata-only): returns a new `Series` sharing the underlying
    /// `Column` via `Arc`. No GPU allocation or copy.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not in the schema.
    pub fn series(&self, name: &str) -> Result<Series> {
        let col = self.columns.get(name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                .with_operation("dataframe::series")
                .with_context("column_name", name.to_string())
        })?;

        Ok(Series::from_column_arc(Arc::clone(col))?.rename(name))
    }

    /// Look up the positional index of a column by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.get_index_of(name)
    }

    /// Create a [`DataFrameView`] — a named, borrowed view of all columns.
    ///
    /// The view provides named column access and iteration without copying
    /// any GPU data.
    pub fn view(&self) -> DataFrameView<'_> {
        let columns = self.columns.iter().map(|(name, col)| (name.as_str(), col)).collect();
        DataFrameView::new(columns)
    }

    /// Factorize a named column into integer codes and unique values.
    ///
    /// Returns `(codes, uniques)` where `codes[i]` maps to `uniques[codes[i]]`.
    /// The `codes` Series is named `"{column}_codes"` and `uniques` is named
    /// `"{column}_uniques"`.
    ///
    /// This is equivalent to `pandas.factorize()` but GPU-accelerated.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn factorize(&self, column: &str) -> Result<(Series, Series)> {
        let cv = self.series_view(column)?;
        let (codes_col, uniques_col) = cv.factorize().map_err(|e| {
            let err = if e.operation().is_empty() { e.with_operation("dataframe::factorize") } else { e };
            err.with_context("column_name", column.to_string())
        })?;
        let codes = Series::from_column(codes_col)?.rename(format!("{column}_codes"));
        let uniques = Series::from_column(uniques_col)?.rename(format!("{column}_uniques"));
        Ok((codes, uniques))
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
                .copy()
                .map_err(|e| e.with_operation("dataframe::to_table_copy").with_context("column_name", name.clone()))?;
            cols.push(col);
        }

        Table::from_columns(cols).map_err(|e| e.with_operation("dataframe::to_table_copy"))
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
                Err(shared) => shared.copy().map_err(|e| {
                    e.with_operation("dataframe::try_into_table").with_context("column_name", name.clone())
                })?,
            };
            names.push(name);
            cols.push(col);
        }

        let table = Table::from_columns(cols).map_err(|e| e.with_operation("dataframe::try_into_table"))?;
        Ok((table, names))
    }

    /// Create a borrowed [`TableView`] over this DataFrame's columns.
    ///
    /// The returned [`TempTableView`](crate::TempTableView) bundles the
    /// intermediate column views with the table view so lifetimes are
    /// managed automatically. Call [`.table()`](crate::TempTableView::table)
    /// to get a `&TableView` for passing to libcudf operations.
    ///
    /// # Cost
    ///
    /// Zero-copy: creates a borrowed view over existing column device memory.
    /// Only lightweight FFI view handles are allocated (no GPU data movement).
    pub fn as_table_view(&self) -> Result<crate::internal::temp_table::TempTableView<'_>> {
        self.build_table_view()
    }

    // -- Internal helpers ----------------------------------------------------

    /// Build a [`TempTableView`] from the individual `Arc<Column>`s.
    ///
    /// The returned value owns the column views and the table view,
    /// solving the lifetime management problem where column views must
    /// outlive the table view.
    fn build_table_view(&self) -> Result<crate::internal::temp_table::TempTableView<'_>> {
        let views: Vec<ColumnView<'_>> = self.columns.values().map(|col| col.view()).collect::<Result<Vec<_>>>()?;
        crate::internal::temp_table::TempTableView::from_column_views(views)
    }

    /// Create a DataFrame from a Table and column names, decomposing the
    /// table into Arc-wrapped columns.
    fn from_table_with_names(table: Table, names: Vec<String>) -> Result<Self> {
        crate::internal::reassemble::dataframe_from_table_and_names(table, names)
    }

    /// Returns owned column names.
    fn column_names_owned(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Returns a reference to the internal column map.
    ///
    /// This is a crate-internal accessor used by high-level types like
    /// [`GroupBy`][crate::groupby::GroupBy].
    pub(crate) fn columns_map(&self) -> &IndexMap<String, Arc<Column>> {
        &self.columns
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_kind(&self) -> &'static str {
        self.domain.debug_kind()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_stream_value(&self) -> usize {
        self.domain.debug_stream_value()
    }

    // -- I/O -----------------------------------------------------------------

    /// Read a CSV file into a `DataFrame`.
    ///
    /// If the caller does not provide explicit names through
    /// [`DataFrameCsvReaderOptions::column_names`], names come from the header
    /// row when `has_header == true`, or fall back to generated names such as
    /// `"column_0"` when reading headerless CSV.
    pub fn read_csv(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_csv_with_options(DataFrameCsvReaderOptions::new(path))
    }

    /// Read a CSV file into a `DataFrame` using caller-provided options.
    ///
    /// When `has_header == true`, the high-level facade parses the header row
    /// on the host to recover schema names. This metadata parsing does not
    /// materialize data columns through host memory.
    pub fn read_csv_with_options(options: DataFrameCsvReaderOptions) -> Result<Self> {
        let table = csv::read_csv(&options.to_table_options())?;
        let num_columns = {
            let view = table.view()?;
            view.num_columns()? as usize
        };
        let names = options.resolve_column_names(num_columns)?;
        Self::from_table(table, names)
    }

    /// Write this `DataFrame` to a CSV file, preserving the DataFrame schema
    /// as the output header row.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        self.write_csv_with_options(DataFrameCsvWriterOptions::new(path))
    }

    /// Write this `DataFrame` to a CSV file using caller-provided options.
    ///
    /// The DataFrame's schema names are always used as the output column
    /// names so the high-level round-trip path remains truthful.
    pub fn write_csv_with_options(&self, options: DataFrameCsvWriterOptions) -> Result<()> {
        self.domain.synchronize_for_host_visibility("dataframe::write_csv_with_options")?;
        let names: Vec<&str> = self.column_names();
        let opts = options.to_table_options(&names);
        let tv = self.build_table_view()?;
        csv::write_csv(tv.table(), &opts)
    }

    /// Read a Parquet file into a `DataFrame`, using column names from the
    /// file's schema metadata.
    ///
    /// This is a convenience wrapper around
    /// [`parquet::read_parquet_with_metadata`].
    pub fn read_parquet(path: impl AsRef<Path>) -> Result<Self> {
        Self::read_parquet_with_options(ParquetReaderOptions::new(path))
    }

    /// Read a Parquet file into a `DataFrame` using caller-provided options.
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    pub fn read_parquet_with_options(options: ParquetReaderOptions) -> Result<Self> {
        let result = parquet::read_parquet_with_metadata(options)?;
        Self::from_table(result.table, result.column_names)
    }

    /// Write this `DataFrame` to a Parquet file, embedding column names in
    /// the file's schema metadata.
    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        self.domain.synchronize_for_host_visibility("dataframe::write_parquet")?;
        let names: Vec<&str> = self.column_names();
        let opts = ParquetWriterOptions::new(path).column_names(&names)?;
        let tv = self.build_table_view()?;
        parquet::write_parquet(tv.table(), &opts)
    }

    // -- Row slicing ---------------------------------------------------------

    /// Return the first `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with gathered rows via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn head(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range(0, take)
    }

    /// Return the last `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with gathered rows via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn tail(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range(start, take)
    }

    /// Return `length` rows starting at `offset`.
    ///
    /// Both `offset` and `length` are clamped to the number of rows, so
    /// out-of-range values never produce an error — they simply yield fewer
    /// (or zero) rows. The schema is preserved.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range(start, take)
    }

    /// Gather (select) rows by an index `Series`.
    ///
    /// Produces a new `DataFrame` whose row `i` equals `self[indices[i]]`.
    /// The `indices` series must be `Int32`. Out-of-bounds indices produce
    /// null rows (uses [`copying::gather`] which nullifies out-of-bounds indices).
    /// The schema is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `Int32`.
    pub fn gather(&self, indices: &Series) -> Result<Self> {
        ExecutionDomain::compatible_domain(&self.domain, indices.as_column().execution_domain(), "dataframe::gather")?;
        let cv = indices.column_view()?;
        self.gather_by_column_view(&cv)
    }

    /// Gather (select) rows by a low-level index `ColumnView`.
    ///
    /// This is the advanced entry point. Prefer [`gather`](Self::gather)
    /// with a `Series` for the cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `Int32`.
    pub fn gather_by_column_view(&self, indices: &ColumnView<'_>) -> Result<Self> {
        if indices.is_empty()? {
            return self.empty_with_same_schema();
        }
        let tv = self.build_table_view()?;
        let result_table = copying::gather(tv.table(), indices)?.with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    // -- Filter, sort, deduplicate -------------------------------------------

    /// Filter rows using a `Series` boolean mask (cuDF-style high-level API).
    ///
    /// Rows where `mask` is `true` are kept; `false` or null rows are
    /// dropped. The schema is preserved.
    ///
    /// # Errors
    ///
    /// - `TypeError` if `mask` dtype is not `Bool8`.
    /// - `InvalidOperation` if `mask` length does not match the DataFrame row count.
    pub fn filter(&self, mask: &Series) -> Result<Self> {
        ExecutionDomain::compatible_domain(&self.domain, mask.as_column().execution_domain(), "dataframe::filter")?;
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask(&mask_view)
    }

    /// Filter rows using a low-level `ColumnView` boolean mask.
    ///
    /// This is the low-level entry point. Prefer [`filter`](Self::filter)
    /// with a `Series` mask for the cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` is not `Bool8`, or if its row count
    /// does not match the DataFrame's row count.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new filtered table via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn filter_by_mask(&self, mask: &ColumnView<'_>) -> Result<Self> {
        self.validate_mask_view(mask, "dataframe::filter_by_mask")?;
        let tv = self.build_table_view()?;
        let result_table =
            filter::apply_boolean_mask(tv.table(), mask)?.with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Filter rows to those where `column` values appear in a `Series`.
    ///
    /// Convenience wrapper around [`crate::search::contains`] +
    /// [`DataFrame::filter_by_mask`], equivalent to pandas `df[df[column].isin(values)]`.
    ///
    /// `values` is the haystack (the set to test membership against).
    /// The column and values must have the same data type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin(&self, column: &str, values: &Series) -> Result<Self> {
        let cv = values.column_view()?;
        self.filter_isin_by_column_view(column, &cv)
    }

    /// Filter rows to those where `column` values appear in `values`.
    ///
    /// This is the advanced entry point that accepts a raw `ColumnView`.
    /// Prefer [`filter_isin`](Self::filter_isin) with a `Series` for the
    /// cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin_by_column_view(&self, column: &str, values: &ColumnView<'_>) -> Result<Self> {
        let col_view = self.series_view(column)?;
        let mask = crate::search::contains(values, &col_view)?;
        let mask_view = mask.view()?;
        self.filter_by_mask(&mask_view)
    }

    /// Validate that a `Series` is a valid boolean mask for this DataFrame.
    ///
    /// Returns the `ColumnView` if valid, or an appropriate error.
    fn validate_series_mask<'a>(&self, mask: &'a Series) -> Result<SeriesView<'a>> {
        let mask_view = mask.view()?;
        self.validate_mask_view(&mask_view, "dataframe::filter")?;
        Ok(mask_view)
    }

    /// Validate that a low-level column view is a valid boolean mask.
    fn validate_mask_view(&self, mask: &ColumnView<'_>, operation: &'static str) -> Result<()> {
        let mask_dtype = mask.data_type()?;
        if mask_dtype.id() != crate::types::TypeId::Bool8 {
            return Err(Error::new(
                ErrorKind::TypeError,
                format!("filter mask must be Bool8, got {:?}", mask_dtype.id()),
            )
            .with_operation(operation)
            .with_context("mask_dtype", format!("{:?}", mask_dtype.id())));
        }
        let mask_len = mask.len()?;
        if mask_len != self.row_count {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("filter mask length ({}) does not match DataFrame row count ({})", mask_len, self.row_count),
            )
            .with_operation(operation)
            .with_context("mask_length", mask_len.to_string())
            .with_context("row_count", self.row_count.to_string()));
        }
        Ok(())
    }

    /// Sort rows using cuDF-style [`SortOptions`].
    ///
    /// This is the high-level equivalent of [`sort`](Self::sort), accepting
    /// column names and a `SortOptions` struct instead of raw `&[Order]`
    /// slices.
    ///
    /// # Option vector lengths
    ///
    /// `descending` and `nulls_first` vectors in `SortOptions` must be
    /// either empty (meaning defaults for all keys) or exactly equal in
    /// length to the number of `by` columns. Any other length produces an
    /// `InvalidOperation` error.
    ///
    /// # Cost
    ///
    /// Produces a new GPU-allocated output table.
    ///
    /// If `by` is exactly all columns in schema order, cuDF sorts the table
    /// directly.
    ///
    /// Otherwise, `sort_values()` builds a temporary borrowed keys view and
    /// calls the keyed sort primitive. This does not deep-copy the key
    /// columns, but it still allocates temporary view handles plus the sorted
    /// output.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::{DataFrame, Series, SortOptions};
    ///
    /// let ids = Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    /// let df = DataFrame::from_columns([("id", ids)]).unwrap();
    ///
    /// let sorted = df
    ///     .sort_values(&["id"], SortOptions::new().with_descending(vec![true]))
    ///     .unwrap();
    /// ```
    pub fn sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let by: Vec<String> = by.into_iter().map(|s| s.as_ref().to_string()).collect();
        let by_refs: Vec<&str> = by.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(&self.columns, &by_refs, "dataframe::sort_values")?;
        let (orders, null_orders) = options.resolve(by_refs.len())?;

        let null_order_ref = null_orders.as_deref();

        // Check whether `by` names are exactly all columns in table order.
        let all_in_order =
            key_indices.len() == self.columns.len() && key_indices.iter().enumerate().all(|(i, &idx)| idx == i);

        let tv = self.build_table_view()?;

        let result_table = if all_in_order {
            let sort_fn = if options.stable { sorting::stable_sort } else { sorting::sort };
            sort_fn(tv.table(), Some(&orders), null_order_ref)?
        } else {
            // Project the sort keys directly from the already-built full
            // table view, then reorder the full table according to those keys.
            let keys_table = tv.table().select_columns(&key_indices)?;
            let sort_by_key_fn = if options.stable { sorting::stable_sort_by_key } else { sorting::sort_by_key };
            sort_by_key_fn(tv.table(), &keys_table, Some(&orders), null_order_ref)?
        }
        .with_execution_domain(Arc::clone(&self.domain));

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Return the `n` rows with the largest values in `column`.
    ///
    /// Equivalent to `self.sort_values(&[column], SortOptions::new().with_descending(vec![true]))?.head(n)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn nlargest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values(&[column], SortOptions::new().with_descending(vec![true]))?.head(n)
    }

    /// Return the `n` rows with the smallest values in `column`.
    ///
    /// Equivalent to `self.sort_values(&[column], SortOptions::new())?.head(n)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn nsmallest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values(&[column], SortOptions::new())?.head(n)
    }

    /// Remove duplicate rows based on a subset of columns.
    ///
    /// The order of rows in the output is **not** guaranteed. The full
    /// schema (all columns) is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `subset` is not in the schema.
    pub fn distinct(&self, subset: &[&str], keep: DuplicateKeepOption) -> Result<Self> {
        let key_indices = resolve_column_names(&self.columns, subset, "dataframe::distinct")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let tv = self.build_table_view()?;
        let result_table =
            stream_compaction::distinct(tv.table(), &key_i32, keep, NullEquality::Equal, NanEquality::AllEqual)?
                .with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Drop rows where the specified columns contain nulls.
    ///
    /// A row is kept only if at least `keep_threshold` of the `subset`
    /// columns are non-null. For the common "drop any null" behavior, set
    /// `keep_threshold` equal to `subset.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `subset` is not in the schema.
    pub fn drop_nulls(&self, subset: &[&str], keep_threshold: i32) -> Result<Self> {
        let key_indices = resolve_column_names(&self.columns, subset, "dataframe::drop_nulls")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::drop_nulls(tv.table(), &key_i32, keep_threshold)?
            .with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Drop rows where **any** of the specified `subset` columns contain nulls.
    ///
    /// Equivalent to `self.drop_nulls(subset, subset.len() as i32)` — every
    /// column in `subset` must be non-null for a row to be kept.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with nulls removed via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `subset` is not in the schema.
    pub fn drop_nulls_any(&self, subset: &[&str]) -> Result<Self> {
        let subset_count = subset.len();
        let keep_threshold = i32::try_from(subset_count).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "subset length exceeds i32::MAX")
                .with_operation("dataframe::drop_nulls_any")
                .with_context("subset_count", subset_count.to_string())
        })?;

        self.drop_nulls(subset, keep_threshold).map_err(|e| {
            e.with_operation("dataframe::drop_nulls_any").with_context("subset_count", subset_count.to_string())
        })
    }

    // -- Joins (merge) -------------------------------------------------------

    /// Merge (join) two DataFrames with cuDF-style options.
    ///
    /// This is the unified high-level entry point for joins. Configure the merge
    /// via [`MergeOptions`](crate::MergeOptions).
    ///
    /// # Key modes
    ///
    /// - `on(keys)` — shared key column names present in both sides. Output
    ///   contains one shared key column per key.
    /// - `left_on(keys)` + `right_on(keys)` — asymmetric key names. Both key
    ///   columns are preserved in the output.
    ///
    /// # Suffix handling
    ///
    /// When `on()` is used: if a left payload and right payload column share
    /// the same name, both are suffixed with the configured suffixes (default
    /// `"_left"` / `"_right"`).
    ///
    /// When `left_on()` / `right_on()` is used: both key columns are preserved,
    /// and conflicting payload names get the same suffix treatment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No keys are specified (`on` or `left_on`+`right_on`)
    /// - `left_on` is set without `right_on` or vice versa
    /// - `left_on` and `right_on` have different lengths
    /// - A key column is not found in its respective DataFrame
    pub fn merge(&self, right: &DataFrame, options: join::MergeOptions) -> Result<Self> {
        let operation: &'static str = "dataframe::merge";

        // Resolve key names from options.
        let (left_on_strs, right_on_strs, shared_keys) = resolve_merge_keys(&options, operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        // Validate key counts match.
        if left_on_refs.len() != right_on_refs.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on_refs.len().to_string())
            .with_context("right_on_count", right_on_refs.len().to_string()));
        }

        let join_type = options.join_type();

        // Execute the split join with null_equality from options.
        let split = self.join_split_on_impl_with_null_eq(
            right,
            &left_on_refs,
            &right_on_refs,
            join_type,
            options.null_eq(),
            operation,
        )?;

        // Reassemble with high-level suffix logic.
        merge_join_result_with_suffixes(
            split,
            &left_on_refs,
            shared_keys,
            shared_keys && join_type == join::JoinType::Full,
            options.suffix_left(),
            options.suffix_right(),
            operation,
        )
    }

    /// Merge (join) two DataFrames, returning split left/right results.
    ///
    /// Like [`merge`](Self::merge) but returns a [`DataFrameJoinResult`] with
    /// separate left and right `DataFrame`s instead of merging columns into one.
    /// Each side contains `[key_columns..., payload_columns...]`.
    ///
    /// # Errors
    ///
    /// Same error conditions as [`merge`](Self::merge).
    pub fn merge_split(&self, right: &DataFrame, options: join::MergeOptions) -> Result<DataFrameJoinResult> {
        let operation: &'static str = "dataframe::merge_split";

        let (left_on_strs, right_on_strs, _shared_keys) = resolve_merge_keys(&options, operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        if left_on_refs.len() != right_on_refs.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on_refs.len().to_string())
            .with_context("right_on_count", right_on_refs.len().to_string()));
        }

        self.join_split_on_impl_with_null_eq(
            right,
            &left_on_refs,
            &right_on_refs,
            options.join_type(),
            options.null_eq(),
            operation,
        )
    }
    /// Internal split-join dispatch with configurable null equality.
    fn join_split_on_impl_with_null_eq(
        &self,
        right: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        join_type: join::JoinType,
        null_equality: join::NullEquality,
        operation: &'static str,
    ) -> Result<DataFrameJoinResult> {
        let output_domain = ExecutionDomain::compatible_domain(&self.domain, &right.domain, operation)?;

        if left_on.len() != right_on.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on.len().to_string())
            .with_context("right_on_count", right_on.len().to_string()));
        }

        // Resolve key column names in both sides.
        let left_key_indices = resolve_column_names(&self.columns, left_on, operation)?;
        let right_key_indices = resolve_column_names(&right.columns, right_on, operation)?;

        // Compute payload indices (all columns not in keys, preserving order).
        let left_payload_indices = payload_indices(&self.columns, &left_key_indices);
        let right_payload_indices = payload_indices(&right.columns, &right_key_indices);

        // Build borrowed key and payload table views over the existing columns.
        let lk_table = build_borrowed_table_view_from_columns(&self.columns, &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(&right.columns, &right_key_indices)?;
        let lp_table = build_borrowed_table_view_from_columns(&self.columns, &left_payload_indices)?;
        let rp_table = build_borrowed_table_view_from_columns(&right.columns, &right_payload_indices)?;

        let result = match join_type {
            join::JoinType::Inner => {
                join::inner_join(lk_table.table(), rk_table.table(), lp_table.table(), rp_table.table(), null_equality)?
            },
            join::JoinType::Left => {
                join::left_join(lk_table.table(), rk_table.table(), lp_table.table(), rp_table.table(), null_equality)?
            },
            join::JoinType::Full => {
                join::full_join(lk_table.table(), rk_table.table(), lp_table.table(), rp_table.table(), null_equality)?
            },
        };

        // Build schemas: left side uses left key names, right side uses right key names.
        let left_schema_names = build_join_schema_names(&self.columns, &left_key_indices, &left_payload_indices);
        let right_schema_names = build_join_schema_names(&right.columns, &right_key_indices, &right_payload_indices);

        Ok(DataFrameJoinResult {
            left: DataFrame::from_table(
                result.left.with_execution_domain(Arc::clone(&output_domain)),
                left_schema_names,
            )?,
            right: DataFrame::from_table(result.right.with_execution_domain(output_domain), right_schema_names)?,
        })
    }

    // -- Groupby aggregation -------------------------------------------------

    /// Start a groupby operation.
    ///
    /// Returns a [`GroupBy`] on which you call [`.agg()`][groupby::GroupBy::agg]
    /// with [`AggSpec`][groupby::AggSpec] items, or use shortcut methods like
    /// `.sum()`, `.mean()`, `.count()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::{Agg, AggSpec, DataFrame, Series};
    /// # let keys = Series::new::<i32>("key", &[1, 2]).unwrap();
    /// # let vals = Series::new::<i64>("val", &[10, 20]).unwrap();
    /// # let df = DataFrame::from_columns([("key", keys), ("val", vals)]).unwrap();
    /// let result =
    ///     df.groupby(&["key"]).agg(&[AggSpec::new("val", Agg::Sum)]).unwrap();
    /// ```
    pub fn groupby<I, S>(&self, keys: I) -> groupby::GroupBy<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        groupby::GroupBy::new(self, keys.into_iter().map(|s| s.as_ref().to_string()).collect())
    }

    // -- Concat, rename, select ----------------------------------------------

    /// Concatenate two DataFrames row-wise.
    ///
    /// Both DataFrames must have the same column names in the same order
    /// and matching column types.
    ///
    /// # Errors
    ///
    /// Returns an error if the schemas do not match (different names, count,
    /// order, or types).
    pub fn concat(&self, other: &DataFrame) -> Result<Self> {
        let output_domain = ExecutionDomain::compatible_domain(&self.domain, &other.domain, "dataframe::concat")?;

        // Validate schemas match: same names, same order.
        if self.columns.len() != other.columns.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot concatenate DataFrames with different column counts",
            )
            .with_operation("dataframe::concat")
            .with_context("left_columns", self.columns.len().to_string())
            .with_context("right_columns", other.columns.len().to_string()));
        }

        for (left_name, right_name) in self.columns.keys().zip(other.columns.keys()) {
            if left_name != right_name {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("schema mismatch: column '{}' vs '{}'", left_name, right_name),
                )
                .with_operation("dataframe::concat")
                .with_context("left_column", left_name.clone())
                .with_context("right_column", right_name.clone()));
            }
        }

        // Validate matching types column-by-column up front for clearer errors.
        for (i, ((name, left_col), (_, right_col))) in self.columns.iter().zip(other.columns.iter()).enumerate() {
            let left_type = left_col.view()?.data_type()?.id();
            let right_type = right_col.view()?.data_type()?.id();
            if left_type != right_type {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cannot concatenate DataFrames with mismatched column types",
                )
                .with_operation("dataframe::concat")
                .with_context("column_index", i.to_string())
                .with_context("column_name", name.to_string())
                .with_context("left_type_id", format!("{:?}", left_type).to_lowercase())
                .with_context("right_type_id", format!("{:?}", right_type).to_lowercase()));
            }
        }

        let left_tv = self.build_table_view()?;
        let right_tv = other.build_table_view()?;
        let result_table = crate::concatenate::concatenate_tables(&[left_tv.table(), right_tv.table()])?
            .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Return a new DataFrame with one column renamed.
    ///
    /// This borrowed variant clones the `Arc` references (no GPU copy).
    ///
    /// # Errors
    ///
    /// Returns an error if `old` is not in the schema, or if `new` already
    /// exists in the schema.
    pub fn rename(&self, old: &str, new: &str) -> Result<Self> {
        if !self.columns.contains_key(old) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", old))
                .with_operation("dataframe::rename")
                .with_context("column_name", old.to_string()));
        }

        if old != new && self.columns.contains_key(new) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("cannot rename '{}' to '{}': target name already exists", old, new),
            )
            .with_operation("dataframe::rename")
            .with_context("old_name", old.to_string())
            .with_context("new_name", new.to_string()));
        }

        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            if name == old {
                new_columns.insert(new.to_string(), Arc::clone(col));
            } else {
                new_columns.insert(name.clone(), Arc::clone(col));
            }
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Consuming variant of [`rename`](Self::rename).
    ///
    /// Avoids O(N) `Arc::clone` for N columns; prefer this when the source
    /// DataFrame is no longer needed. This is a pure schema rewrite with
    /// **no GPU data copy**.
    ///
    /// # Cost
    ///
    /// Zero-copy: only the column name in the schema changes. Consuming
    /// `self` avoids the `Arc` refcount bump for unchanged columns.
    ///
    /// # Errors
    ///
    /// Returns an error if `old` is not in the schema, or if `new` already
    /// exists in the schema.
    pub fn rename_owned(self, old: &str, new: &str) -> Result<Self> {
        if !self.columns.contains_key(old) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", old))
                .with_operation("dataframe::rename_owned")
                .with_context("column_name", old.to_string()));
        }

        if old != new && self.columns.contains_key(new) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("cannot rename '{}' to '{}': target name already exists", old, new),
            )
            .with_operation("dataframe::rename_owned")
            .with_context("old_name", old.to_string())
            .with_context("new_name", new.to_string()));
        }

        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in self.columns {
            if name == old {
                new_columns.insert(new.to_string(), col);
            } else {
                new_columns.insert(name, col);
            }
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: self.domain })
    }

    /// Return a new DataFrame with multiple columns renamed at once.
    ///
    /// Each pair in `renames` is `(old_name, new_name)`. All renames are
    /// applied atomically: swapping two column names (e.g. `[("a","b"),("b","a")]`)
    /// is supported.
    ///
    /// This is a zero-copy metadata operation — all columns are shared via
    /// `Arc::clone`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any source name is not in the schema
    /// - Any target name would conflict with an existing column that is not
    ///   itself being renamed
    /// - Multiple renames produce the same target name
    pub fn rename_columns<I, S1, S2>(&self, renames: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        let pairs: Vec<(String, String)> =
            renames.into_iter().map(|(old, new)| (old.as_ref().to_string(), new.as_ref().to_string())).collect();

        // Build a map from old name → new name.
        let mut rename_map = std::collections::HashMap::with_capacity(pairs.len());
        for (old, new) in &pairs {
            if !self.columns.contains_key(old.as_str()) {
                return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", old))
                    .with_operation("dataframe::rename_columns")
                    .with_context("column_name", old.clone()));
            }
            rename_map.insert(old.as_str(), new.as_str());
        }

        // Compute the full target name set and check for conflicts.
        let mut target_names = std::collections::HashSet::with_capacity(self.columns.len());

        for name in self.columns.keys() {
            let target = if let Some(&new_name) = rename_map.get(name.as_str()) { new_name } else { name.as_str() };
            if !target_names.insert(target.to_string()) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("rename would produce duplicate column name '{}'", target),
                )
                .with_operation("dataframe::rename_columns")
                .with_context("target_name", target.to_string()));
            }
        }

        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let target =
                if let Some(&new_name) = rename_map.get(name.as_str()) { new_name.to_string() } else { name.clone() };
            new_columns.insert(target, Arc::clone(col));
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Return a new DataFrame containing only the specified columns.
    ///
    /// The columns appear in the order given in `columns`. Each selected
    /// column is shared via `Arc::clone` (zero GPU copy). The logical row
    /// count is preserved even if `columns` is empty and the result has zero
    /// columns.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `columns` is not in the schema, or if
    /// `columns` contains duplicates.
    pub fn select<I, S>(&self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let columns: Vec<String> = columns.into_iter().map(|name| name.as_ref().to_string()).collect();

        // Reject duplicates to keep schema (name -> index) consistent.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in &columns {
            if !seen.insert(name.as_str()) {
                return Err(Error::new(ErrorKind::InvalidOperation, "selected columns must be unique")
                    .with_operation("dataframe::select")
                    .with_context("duplicate_column", name.clone())
                    .with_context("selected_column_count", columns.len().to_string()));
            }
        }

        let mut new_columns = IndexMap::with_capacity(columns.len());
        for name in &columns {
            let col = self.columns.get(name.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                    .with_operation("dataframe::select")
                    .with_context("column_name", name.clone())
            })?;
            new_columns.insert(name.clone(), Arc::clone(col));
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Consuming variant of [`select`](Self::select).
    ///
    /// Moves columns out of the `IndexMap` instead of cloning the `Arc`,
    /// avoiding the O(N) `Arc::clone` overhead for N selected columns.
    /// Prefer this when the source DataFrame is no longer needed.
    ///
    /// # Cost
    ///
    /// Zero-copy: moves `Arc<Column>` ownership without GPU data copy.
    /// Consuming `self` avoids the refcount bump entirely for selected
    /// columns and drops unselected columns immediately. The logical row
    /// count is preserved even if `columns` is empty and the result has zero
    /// columns.
    pub fn select_owned<I, S>(mut self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let columns: Vec<String> = columns.into_iter().map(|name| name.as_ref().to_string()).collect();

        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in &columns {
            if !seen.insert(name.as_str()) {
                return Err(Error::new(ErrorKind::InvalidOperation, "selected columns must be unique")
                    .with_operation("dataframe::select_owned")
                    .with_context("duplicate_column", name.clone())
                    .with_context("selected_column_count", columns.len().to_string()));
            }
        }

        let mut new_columns = IndexMap::with_capacity(columns.len());
        for name in &columns {
            let col = self.columns.swap_remove(name.as_str()).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                    .with_operation("dataframe::select_owned")
                    .with_context("column_name", name.clone())
            })?;
            new_columns.insert(name.clone(), col);
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: self.domain })
    }

    /// Return a new DataFrame with the specified columns removed.
    ///
    /// Columns not in `columns` are preserved in their original order via
    /// `Arc::clone` (zero GPU copy). Dropping columns is schema-only: the
    /// logical row count is preserved even if all columns are removed.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `columns` is not in the schema.
    pub fn drop<I>(&self, columns: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let to_drop: std::collections::HashSet<String> = columns.into_iter().map(|s| s.as_ref().to_string()).collect();

        for name in &to_drop {
            if !self.columns.contains_key(name.as_str()) {
                return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                    .with_operation("dataframe::drop")
                    .with_context("column_name", name.clone()));
            }
        }

        let mut new_columns = IndexMap::with_capacity(self.columns.len().saturating_sub(to_drop.len()));
        for (name, col) in &self.columns {
            if !to_drop.contains(name.as_str()) {
                new_columns.insert(name.clone(), Arc::clone(col));
            }
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Return a new DataFrame with the specified columns removed.
    ///
    /// Alias for [`drop`](Self::drop) kept for compatibility.
    pub fn drop_columns<I>(&self, columns: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        self.drop(columns)
    }

    // -- Column-level convenience --------------------------------------------

    /// Return a new DataFrame with the given column added or replaced.
    ///
    /// Accepts anything convertible to [`Series`] — both `Column` and
    /// `Series` work transparently. When a `Series` is passed, its
    /// `Arc<Column>` is shared directly (zero GPU copy for the inserted
    /// column as well as all unchanged columns).
    ///
    /// If `name` already exists in the schema, the column at that position
    /// is replaced. If `name` is new, the column is appended.
    ///
    /// The new column must have the same row count as the existing DataFrame
    /// unless the DataFrame has neither rows nor columns. Zero-column
    /// DataFrames that still carry a logical row count must preserve that
    /// row count when the first column is inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if the row counts do not match.
    pub fn with_column(&self, name: &str, col: impl crate::series::IntoSeries) -> Result<Self> {
        let series: Series = col.into_series()?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows {
            if col_rows != self_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
                )
                .with_operation("dataframe::with_column")
                .with_context("column_name", name.to_string())
                .with_context("column_rows", col_rows.to_string())
                .with_context("dataframe_rows", self_rows.to_string()));
            }
        }

        let mut new_columns = self.columns.clone();
        new_columns.insert(name.to_string(), Arc::clone(series.column_arc()));

        let row_count = expected_rows.unwrap_or(col_rows);

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::with_column")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Consuming variant of [`with_column`](Self::with_column).
    ///
    /// Avoids O(N) `Arc::clone` for N columns; prefer this when the source
    /// DataFrame is no longer needed.
    ///
    /// Accepts anything convertible to [`Series`] — both `Column` and
    /// `Series` work transparently.
    ///
    /// # Cost
    ///
    /// Zero-copy for unchanged columns: moves `Arc<Column>` ownership
    /// without GPU data copy. Consuming `self` avoids the refcount bump
    /// for columns that are not being replaced.
    pub fn with_column_owned(mut self, name: &str, col: impl crate::series::IntoSeries) -> Result<Self> {
        let series: Series = col.into_series()?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows {
            if col_rows != self_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
                )
                .with_operation("dataframe::with_column_owned")
                .with_context("column_name", name.to_string())
                .with_context("column_rows", col_rows.to_string())
                .with_context("dataframe_rows", self_rows.to_string()));
            }
        }

        let row_count = expected_rows.unwrap_or(col_rows);
        self.columns.insert(name.to_string(), Arc::clone(series.column_arc()));

        let domain = derive_frame_domain_from_columns(&self.columns, "dataframe::with_column_owned")?;
        Ok(Self { columns: self.columns, row_count, domain })
    }

    /// Return a new DataFrame with multiple columns added or replaced from
    /// `(name, Series)` pairs.
    ///
    /// For each pair, if `name` already exists the column is replaced;
    /// otherwise it is appended. All unchanged columns are shared via
    /// `Arc::clone` (zero GPU copy).
    ///
    /// # Duplicate name handling
    ///
    /// If the same output name appears more than once in `pairs`, the **last
    /// value wins** — earlier entries for that name are silently overwritten.
    /// This matches the semantics of Python's `DataFrame.assign(**kwargs)` and
    /// is intentional: `assign` is a batch "set these columns" operation, not a
    /// schema-definition operation. Methods that define schema shape
    /// ([`select`](Self::select), [`rename_columns`](Self::rename_columns))
    /// reject duplicates with an error.
    ///
    /// If `self` has zero columns but a non-zero logical row count (for
    /// example after [`drop`](Self::drop) or [`select`](Self::select) removes
    /// every column), inserted Series must still match that row count. Only a
    /// truly empty `0 x 0` DataFrame lets the first inserted Series establish
    /// the row count.
    ///
    /// # Errors
    ///
    /// Returns an error if any Series has a different row count than the
    /// DataFrame.
    pub fn assign<I, S>(&self, pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: AsRef<str>,
    {
        let mut new_columns = self.columns.clone();
        let mut expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        for (name, series) in pairs {
            let name = name.as_ref();
            let series_rows = series.len();

            if let Some(self_rows) = expected_rows {
                if series_rows != self_rows {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("series '{}' has {} rows but DataFrame has {} rows", name, series_rows, self_rows),
                    )
                    .with_operation("dataframe::assign")
                    .with_context("column_name", name.to_string())
                    .with_context("series_rows", series_rows.to_string())
                    .with_context("dataframe_rows", self_rows.to_string()));
                }
            } else {
                expected_rows = Some(series_rows);
            }

            new_columns.insert(name.to_string(), Arc::clone(series.column_arc()));
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::assign")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Evaluate expressions against this DataFrame and add/replace the
    /// resulting columns.
    ///
    /// All expressions are evaluated against the **original** state of `self`
    /// — not an incrementally mutated copy. This ensures that expressions can
    /// reference columns that other expressions will overwrite, and the order
    /// of evaluation does not matter.
    ///
    /// Each expression must have a determinable output name: either via
    /// [`.alias()`](crate::expr::Expr::alias) or by being a bare
    /// [`col()`](crate::expr::col) reference. Expressions without a name
    /// produce an error **before** any GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if any expression lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I>(&self, exprs: I) -> Result<Self>
    where I: IntoIterator<Item = crate::expr::Expr> {
        let exprs: Vec<crate::expr::Expr> = exprs.into_iter().collect();

        // Validate all names up-front before doing any GPU work.
        let names: Vec<&str> = exprs
            .iter()
            .map(|e| {
                e.output_name().ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "expression has no output name; use .alias() to name it")
                        .with_operation("dataframe::with_columns")
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Evaluate all expressions against the original DataFrame.
        let results: Vec<Series> = exprs.iter().map(|e| e.eval(self)).collect::<Result<Vec<_>>>()?;

        // Build the new DataFrame.
        let pairs: Vec<(&str, Series)> = names.into_iter().zip(results).collect();
        self.assign(pairs)
    }

    /// Return a new DataFrame with null values in the named column replaced
    /// by a scalar value.
    ///
    /// All other columns are preserved unchanged. The scalar type must be
    /// compatible with the column type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn replace_nulls(&self, column: &str, value: &Scalar) -> Result<Self> {
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::replace_nulls")
                .with_context("column_name", column.to_string())
        })?;

        let cv = col.view()?;
        let exec_options = col.execution_domain().explicit_exec_options_raw();
        let new_col = replace::replace_nulls_with_exec_options(
            &cv,
            value,
            exec_options.as_ref(),
            "dataframe::replace_nulls",
        )?
        .with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::replace_nulls")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    /// Return a new DataFrame with the named column cast to a different type.
    ///
    /// All other columns are preserved unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the cast
    /// is unsupported.
    pub fn cast_column(&self, column: &str, target_type: &DataType) -> Result<Self> {
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::cast_column")
                .with_context("column_name", column.to_string())
        })?;

        let cv = col.view()?;
        let new_col = unary::cast(&cv, target_type)?.with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::cast_column")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    // -- High-level API: cuDF-style naming --------------------------------

    /// Drop rows containing null values.
    ///
    /// Equivalent to cuDF Python's `DataFrame.dropna()`.
    ///
    /// - `how=any` (default): drop rows where **any** subset column is null.
    /// - `how=all`: drop rows only when **all** subset columns are null.
    /// - `thresh`: if set, keep rows with at least `thresh` non-null subset values
    ///   (overrides `how`).
    /// - `subset`: columns to inspect; defaults to all columns.
    ///
    /// # Cost
    /// GPU-allocating: produces a new table with the surviving rows.
    ///
    /// # Errors
    ///
    /// Returns an error if any column in `subset` is not in the schema.
    pub fn dropna(&self, opts: DropNaOptions) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        let subset_len = i32::try_from(subset_names.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "subset length exceeds i32::MAX")
                .with_operation("dataframe::dropna")
        })?;

        let threshold = if let Some(t) = opts.thresh {
            t
        } else if opts.how_all {
            // "all": drop only when every subset column is null → keep if ≥1 non-null.
            1
        } else {
            // "any": drop if any subset column is null → keep if all non-null.
            subset_len
        };

        self.drop_nulls(&subset_names, threshold).map_err(|e| e.with_operation("dataframe::dropna"))
    }

    /// Remove duplicate rows.
    ///
    /// Equivalent to cuDF Python's `DataFrame.drop_duplicates()`.
    ///
    /// - `subset`: columns to use as deduplication keys; defaults to all columns.
    /// - `keep`: which occurrence to keep
    ///   ([`DuplicateKeepOption::KeepAny`] by default).
    ///
    /// # Cost
    /// GPU-allocating: produces a new table with duplicates removed.
    ///
    /// # Errors
    ///
    /// Returns an error if any column in `subset` is not in the schema.
    pub fn drop_duplicates(&self, opts: DistinctOptions) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        self.distinct(&subset_names, opts.keep).map_err(|e| e.with_operation("dataframe::drop_duplicates"))
    }

    /// Replace null values in a single column with a plain Rust value.
    ///
    /// Equivalent to cuDF Python's `DataFrame.fillna({column: value})`.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged (zero-copy via Arc sharing).
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn fillna<T: CudfType>(&self, column: &str, value: T) -> Result<Self> {
        let scalar = Scalar::from_value(value)?;
        self.fillna_with_scalar(column, &scalar)
    }

    /// Replace null values in a single column with a pre-constructed [`Scalar`].
    ///
    /// This is the advanced entry point. Prefer [`fillna`](Self::fillna)
    /// with a plain Rust value for the cuDF-style API.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged (zero-copy via Arc sharing).
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn fillna_with_scalar(&self, column: &str, value: &Scalar) -> Result<Self> {
        self.replace_nulls(column, value).map_err(|e| e.with_operation("dataframe::fillna_with_scalar"))
    }

    /// Replace null values in a single column using corresponding elements
    /// from a [`Series`].
    ///
    /// Both the target column and `fill_series` must have compatible types and
    /// matching lengths.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged (zero-copy via Arc sharing).
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the types
    /// or lengths are incompatible.
    pub fn fillna_with_series(&self, column: &str, fill_series: &Series) -> Result<Self> {
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::fillna_with_series")
                .with_context("column_name", column.to_string())
        })?;

        let output_domain = ExecutionDomain::compatible_domain(
            col.execution_domain(),
            fill_series.as_column().execution_domain(),
            "dataframe::fillna_with_series",
        )?;
        let cv = col.view()?;
        let fill_cv = fill_series.column_view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let new_col = replace::replace_nulls_column_with_exec_options(
            &cv,
            &fill_cv,
            exec_options.as_ref(),
            "dataframe::fillna_with_series",
        )?
        .with_execution_domain(output_domain);

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::fillna_with_series")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }
}

// ---------------------------------------------------------------------------
// Helpers — thin wrappers delegating to `internal` module
// ---------------------------------------------------------------------------

use crate::internal::resolve::{
    build_borrowed_table_view_from_columns, payload_indices, resolve_column_names, validate_no_duplicate_names,
};

impl DataFrame {
    /// Internal: gather a contiguous range of rows into a new `DataFrame`,
    /// preserving the schema.
    fn gather_range(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema();
        }

        let indices: Vec<i32> = (start as i32..(start + len) as i32).collect();
        let map = Column::from_slice(&indices)?;
        let tv = self.build_table_view()?;
        // SAFETY: indices are constructed from `start..start+len` which the
        // callers have already validated against row_count.
        let result_table = unsafe { copying::gather_unchecked(tv.table(), &map.view()?)? }
            .with_execution_domain(Arc::clone(&self.domain));

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Internal: build an empty DataFrame with the same schema (column names
    /// and types, zero rows).
    ///
    /// This helper is for row-space operations that produce no rows at all,
    /// such as empty gathers or zero-length slices. Unlike schema-only
    /// operations such as [`select`](Self::select) and [`drop`](Self::drop),
    /// it intentionally resets the logical row count to zero.
    fn empty_with_same_schema(&self) -> Result<Self> {
        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let cv = col.view()?;
            let type_id = cv.data_type()?.id();
            new_columns.insert(
                name.clone(),
                Arc::new(Column::empty(type_id)?.with_execution_domain(Arc::clone(col.execution_domain()))),
            );
        }
        let domain = if new_columns.is_empty() {
            Arc::clone(&self.domain)
        } else {
            derive_frame_domain_from_columns(&new_columns, "dataframe::empty_with_same_schema")?
        };
        Ok(Self { columns: new_columns, row_count: 0, domain })
    }
}

/// Build the schema name list for a join result side:
/// key column names (in `on` order) followed by payload column names.
fn build_join_schema_names(
    columns: &IndexMap<String, Arc<Column>>,
    key_indices: &[usize],
    payload_indices: &[usize],
) -> Vec<String> {
    let keys_iter = columns.keys().collect::<Vec<_>>();

    let mut names = Vec::with_capacity(key_indices.len() + payload_indices.len());
    for &idx in key_indices {
        names.push(keys_iter[idx].clone());
    }
    for &idx in payload_indices {
        names.push(keys_iter[idx].clone());
    }
    names
}

/// Coalesce two columns by preferring non-null values from `left`; if null,
/// take from `right`. Runs entirely on GPU via `replace_nulls_column`.
fn coalesce_columns(left: &Column, right: &Column, operation: &'static str) -> Result<Column> {
    let output_domain =
        ExecutionDomain::compatible_domain(left.execution_domain(), right.execution_domain(), operation)?;
    let lcv = left.view()?;
    let rcv = right.view()?;

    let ldt = lcv.data_type()?;
    let rdt = rcv.data_type()?;
    if ldt != rdt {
        return Err(Error::new(ErrorKind::InvalidOperation, "cannot coalesce columns with different types")
            .with_operation(operation)
            .with_context("left_type_id", format!("{:?}", ldt.id()).to_lowercase())
            .with_context("right_type_id", format!("{:?}", rdt.id()).to_lowercase()));
    }

    let exec_options = output_domain.explicit_exec_options_raw();
    replace::replace_nulls_column_with_exec_options(&lcv, &rcv, exec_options.as_ref(), operation)
        .map(|col| col.with_execution_domain(output_domain))
}

// ---------------------------------------------------------------------------
// High-level merge helpers
// ---------------------------------------------------------------------------

/// Resolve the key names from `MergeOptions`, returning `(left_on, right_on, shared)`.
///
/// `shared` is `true` when `on()` was used (both sides share the same key names),
/// meaning the output should emit one shared key column per key.
fn resolve_merge_keys(
    options: &join::MergeOptions,
    operation: &'static str,
) -> Result<(Vec<String>, Vec<String>, bool)> {
    match (options.on_keys(), options.left_on_keys(), options.right_on_keys()) {
        // on() mode: shared key names
        (Some(on), None, None) => {
            let keys: Vec<String> = on.to_vec();
            Ok((keys.clone(), keys, true))
        },
        // left_on() + right_on() mode: asymmetric keys
        (None, Some(lo), Some(ro)) => Ok((lo.to_vec(), ro.to_vec(), false)),
        // left_on without right_on
        (None, Some(_), None) => {
            Err(Error::new(ErrorKind::InvalidOperation, "left_on was specified without right_on; set both or use on()")
                .with_operation(operation))
        },
        // right_on without left_on
        (None, None, Some(_)) => {
            Err(Error::new(ErrorKind::InvalidOperation, "right_on was specified without left_on; set both or use on()")
                .with_operation(operation))
        },
        // on() together with left_on/right_on is ambiguous — treat as error
        (Some(_), _, _) => Err(Error::new(
            ErrorKind::InvalidOperation,
            "on() cannot be combined with left_on()/right_on(); use one or the other",
        )
        .with_operation(operation)),
        // Nothing specified
        (None, None, None) => {
            Err(Error::new(ErrorKind::InvalidOperation, "no join keys specified; use on() or left_on() + right_on()")
                .with_operation(operation))
        },
    }
}

/// Reassemble a split join result into a single DataFrame using high-level suffix rules.
///
/// When `shared_keys` is true (from `on()`):
///   - Emit one shared key column per key (coalesced for full join).
///   - Conflicting payload names get `suffix_left` / `suffix_right`.
///
/// When `shared_keys` is false (from `left_on()`/`right_on()`):
///   - Both key columns are preserved (left key name, right key name).
///   - Conflicting payload names get suffixes.
fn merge_join_result_with_suffixes(
    split: DataFrameJoinResult,
    left_on: &[&str],
    shared_keys: bool,
    coalesce_keys: bool,
    suffix_left: &str,
    suffix_right: &str,
    operation: &'static str,
) -> Result<DataFrame> {
    let DataFrameJoinResult { left, right } = split;
    let key_count = left_on.len();
    let row_count = left.row_count.max(right.row_count);

    let left_names: Vec<String> = left.columns.keys().cloned().collect();
    let right_names: Vec<String> = right.columns.keys().cloned().collect();
    let left_cols: Vec<Arc<Column>> = left.columns.into_values().collect();
    let right_cols: Vec<Arc<Column>> = right.columns.into_values().collect();

    if key_count > left_cols.len() || key_count > right_cols.len() {
        return Err(Error::new(ErrorKind::InvalidOperation, "join key count exceeds result column count")
            .with_operation(operation)
            .with_context("num_keys", key_count.to_string())
            .with_context("left_columns", left_cols.len().to_string())
            .with_context("right_columns", right_cols.len().to_string()));
    }

    // Collect left and right payload names (after key columns).
    let left_payload_names: Vec<&str> = left_names[key_count..].iter().map(|s| s.as_str()).collect();
    let right_payload_names: Vec<&str> = right_names[key_count..].iter().map(|s| s.as_str()).collect();

    // Find which payload names collide.
    let left_payload_set: std::collections::HashSet<&str> = left_payload_names.iter().copied().collect();
    let right_payload_set: std::collections::HashSet<&str> = right_payload_names.iter().copied().collect();
    let conflicting: std::collections::HashSet<&str> =
        left_payload_set.intersection(&right_payload_set).copied().collect();

    let mut out_cols: Vec<Arc<Column>> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    // Only shared-key full joins should coalesce key columns. For left_on/right_on
    // joins, the contract is to preserve both key columns separately.
    for i in 0..key_count {
        let key_col = if coalesce_keys {
            Arc::new(coalesce_columns(&left_cols[i], &right_cols[i], operation)?)
        } else {
            Arc::clone(&left_cols[i])
        };
        out_cols.push(key_col);
        out_names.push(left_names[i].clone());
    }

    // Left payload.
    for (idx, name) in left_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&left_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_left));
        } else {
            out_names.push(name.to_string());
        }
    }

    if !shared_keys {
        // Emit right key columns for asymmetric join.
        for i in 0..key_count {
            out_cols.push(Arc::clone(&right_cols[i]));
            out_names.push(right_names[i].clone());
        }
    }

    // Right payload.
    for (idx, name) in right_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&right_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_right));
        } else {
            out_names.push(name.to_string());
        }
    }

    validate_no_duplicate_names(&out_names, operation)?;

    let mut columns = IndexMap::with_capacity(out_names.len());
    for (name, col) in out_names.into_iter().zip(out_cols) {
        columns.insert(name, col);
    }

    let domain = derive_frame_domain_from_columns(&columns, operation)?;
    Ok(DataFrame::from_raw_parts(columns, row_count, domain))
}

/// Construct a [`DataFrame`](crate::dataframe::DataFrame) from named columns.
///
/// This macro is Series-oriented: each value must evaluate to a
/// [`Series`](crate::Series). The pair key becomes the DataFrame schema name.
///
/// # Example
///
/// ```no_run
/// use cudf::Series;
///
/// let x = Series::new::<i32>("source_x", &[1, 2, 3]).unwrap();
/// let y = Series::new::<f64>("source_y", &[10.0, 20.0, 30.0]).unwrap();
/// let df = cudf::dataframe! { "x" => x, "y" => y }.unwrap();
/// ```
///
/// ```compile_fail
/// use cudf::column::Column;
///
/// let x = Column::from_slice(&[1i32, 2, 3]).unwrap();
/// let _ = cudf::dataframe! { "x" => x }.unwrap();
/// ```
#[macro_export]
macro_rules! dataframe {
    ($($name:expr => $series:expr),+ $(,)?) => {{
        $crate::DataFrame::from_series(vec![
            $(($name.to_string(), $series)),+
        ])
    }};
}
