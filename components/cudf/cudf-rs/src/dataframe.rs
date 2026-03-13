//! A named-column wrapper around [`Table`] for ergonomic GPU DataFrame operations.
//!
//! [`DataFrame`] pairs a [`Table`] with an ordered column-name schema backed by
//! [`IndexMap`], giving O(1) name-based column lookup while preserving insertion
//! order.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{column::Column, dataframe::DataFrame};
//!
//! let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let scores = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
//!
//! let df = DataFrame::new(
//!     vec!["id".to_string(), "score".to_string()],
//!     vec![ids, scores],
//! )
//! .unwrap();
//!
//! assert_eq!(df.num_columns(), 2);
//! let col = df.column("score").unwrap();
//! ```

use std::path::Path;

use indexmap::IndexMap;

use crate::{
    column::{Column, ColumnView},
    copying,
    error::{Error, ErrorKind, Result},
    filter,
    groupby::{self, Aggregation, AggregationRequest, NullPolicy},
    io::parquet::{self, ParquetReaderOptions, ParquetWriterOptions},
    join, replace,
    scalar::Scalar,
    sorting::{self, Order},
    stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
    table::Table,
    types::DataType,
    unary,
};

/// A GPU DataFrame: a named, ordered collection of GPU columns.
///
/// `DataFrame` is an immutable value — every mutating operation returns a
/// **new** `DataFrame` rather than modifying in place.
pub struct DataFrame {
    table: Table,
    schema: IndexMap<String, usize>,
}

/// Result of a split DataFrame join operation.
///
/// This is the compatibility form containing the raw join sides. Each side
/// has columns `[key_columns..., payload_columns...]`.
#[derive(Debug)]
pub struct DataFrameJoinResult {
    /// Left-side result: key columns + left payload columns.
    pub left: DataFrame,
    /// Right-side result: key columns + right payload columns.
    pub right: DataFrame,
}

impl std::fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Debug shows schema, row count, and column types — NO GPU data copy.
        let num_cols = self.schema.len();
        let num_rows = self.num_rows().unwrap_or(-1);

        let mut ds = f.debug_struct("DataFrame");
        ds.field("num_columns", &num_cols);
        ds.field("num_rows", &num_rows);

        // Build a columns summary: [(name, type)] without GPU data copy.
        let columns_info: Vec<(&str, String)> = self
            .schema
            .keys()
            .map(|name| {
                let type_str = self
                    .column(name)
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
        display_dataframe(self, f, DISPLAY_MAX_ROWS)
    }
}

/// Maximum number of rows to show in the `Display` output.
const DISPLAY_MAX_ROWS: usize = 10;

/// Host-side materialized values for one display column.
enum DisplayColumnValues {
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Bool8(Vec<bool>),
    Unknown,
}

/// Host-side cached display data for one column.
struct DisplayColumn {
    type_str: String,
    validity: Option<Vec<bool>>,
    values: DisplayColumnValues,
}

impl DisplayColumn {
    fn unknown() -> Self {
        Self { type_str: "?".to_string(), validity: None, values: DisplayColumnValues::Unknown }
    }

    fn cell(&self, row_idx: usize) -> String {
        if let Some(validity) = &self.validity {
            if row_idx < validity.len() && !validity[row_idx] {
                return "null".to_string();
            }
        }

        match &self.values {
            DisplayColumnValues::Int8(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Int16(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Int32(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Int64(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::UInt8(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::UInt16(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::UInt32(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::UInt64(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Float32(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Float64(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Bool8(v) => v.get(row_idx).map(|x| x.to_string()).unwrap_or_else(|| "?".to_string()),
            DisplayColumnValues::Unknown => "?".to_string(),
        }
    }
}

/// Build one display column cache from a named DataFrame column.
fn build_display_column(df: &DataFrame, name: &str) -> DisplayColumn {
    let cv = match df.column(name) {
        Ok(cv) => cv,
        Err(_) => return DisplayColumn::unknown(),
    };

    let type_id = match cv.data_type() {
        Ok(dt) => dt.id(),
        Err(_) => return DisplayColumn::unknown(),
    };

    let validity = cv.validity().ok().flatten();

    use crate::types::TypeId;
    let values = match type_id {
        TypeId::Int8 => cv.to_vec::<i8>().map(DisplayColumnValues::Int8).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Int16 => cv.to_vec::<i16>().map(DisplayColumnValues::Int16).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Int32 => cv.to_vec::<i32>().map(DisplayColumnValues::Int32).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Int64 => cv.to_vec::<i64>().map(DisplayColumnValues::Int64).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::UInt8 => cv.to_vec::<u8>().map(DisplayColumnValues::UInt8).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::UInt16 => cv.to_vec::<u16>().map(DisplayColumnValues::UInt16).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::UInt32 => cv.to_vec::<u32>().map(DisplayColumnValues::UInt32).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::UInt64 => cv.to_vec::<u64>().map(DisplayColumnValues::UInt64).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Float32 => cv.to_vec::<f32>().map(DisplayColumnValues::Float32).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Float64 => cv.to_vec::<f64>().map(DisplayColumnValues::Float64).unwrap_or(DisplayColumnValues::Unknown),
        TypeId::Bool8 => cv.to_vec::<bool>().map(DisplayColumnValues::Bool8).unwrap_or(DisplayColumnValues::Unknown),
        _ => DisplayColumnValues::Unknown,
    };

    DisplayColumn { type_str: format!("{:?}", type_id), validity, values }
}

/// Format a DataFrame as a human-readable table.
///
/// Shows column names, types, up to `max_rows` of data, and a row-count
/// summary line.
fn display_dataframe(df: &DataFrame, f: &mut std::fmt::Formatter<'_>, max_rows: usize) -> std::fmt::Result {
    let num_cols = df.num_columns();
    let num_rows = df.num_rows().unwrap_or(0);

    if num_cols == 0 {
        return write!(f, "DataFrame [0 columns × 0 rows]");
    }

    let names: Vec<&str> = df.column_names();

    // Materialize each display column ONCE (GPU -> host copy once per column).
    let display_cols: Vec<DisplayColumn> = names.iter().map(|&name| build_display_column(df, name)).collect();

    let types: Vec<String> = display_cols.iter().map(|c| c.type_str.clone()).collect();

    // Determine how many rows to display.
    let show_rows = max_rows.min(num_rows as usize);

    // Collect string representations of cell values.
    // rows[row_idx][col_idx] = formatted string.
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(show_rows);
    for row_idx in 0..show_rows {
        let mut row: Vec<String> = Vec::with_capacity(num_cols);
        for col in &display_cols {
            row.push(col.cell(row_idx));
        }
        rows.push(row);
    }

    // Compute column widths: max of (name, type, each cell value).
    let mut widths: Vec<usize> = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let mut w = names[col_idx].len().max(types[col_idx].len());
        for row in &rows {
            w = w.max(row[col_idx].len());
        }
        widths.push(w);
    }

    // Print header: column names.
    write_row(f, &names.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &widths)?;

    // Print type row.
    write_row(f, &types, &widths)?;

    // Print separator.
    write_separator(f, &widths)?;

    // Print data rows.
    for row in &rows {
        write_row(f, row, &widths)?;
    }

    // If there are more rows, indicate truncation.
    if (num_rows as usize) > show_rows {
        writeln!(f, "... {} more rows", num_rows as usize - show_rows)?;
    }

    // Summary line.
    write!(f, "[{} columns × {} rows]", num_cols, num_rows)
}

/// Write one row of values, padded to column widths.
fn write_row(f: &mut std::fmt::Formatter<'_>, values: &[String], widths: &[usize]) -> std::fmt::Result {
    for (i, (val, &w)) in values.iter().zip(widths).enumerate() {
        if i > 0 {
            write!(f, "  ")?;
        }
        write!(f, "{:>width$}", val, width = w)?;
    }
    writeln!(f)
}

/// Write a separator line (dashes under each column).
fn write_separator(f: &mut std::fmt::Formatter<'_>, widths: &[usize]) -> std::fmt::Result {
    for (i, &w) in widths.iter().enumerate() {
        if i > 0 {
            write!(f, "  ")?;
        }
        write!(f, "{}", "-".repeat(w))?;
    }
    writeln!(f)
}

impl DataFrame {
    /// Create a new `DataFrame` from column names and owned columns.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `names.len() != columns.len()`
    /// - Any name is duplicated
    /// - Columns have different row counts (enforced by [`Table::from_columns`])
    pub fn new(names: Vec<String>, columns: Vec<Column>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of columns")
                .with_operation("dataframe::new")
                .with_context("names_count", names.len().to_string())
                .with_context("columns_count", columns.len().to_string()));
        }

        let schema = build_schema(&names, "dataframe::new")?;

        if columns.is_empty() {
            // Table::from_columns with an empty vec is valid and yields a
            // zero-column, zero-row table.
            let table = Table::from_columns(columns)?;
            return Ok(Self { table, schema });
        }

        let table = Table::from_columns(columns)?;
        Ok(Self { table, schema })
    }

    /// Wrap an existing [`Table`] with column names.
    ///
    /// # Errors
    ///
    /// Returns an error if `names.len()` does not match the table's column
    /// count, or if names contain duplicates.
    pub fn from_table(table: Table, names: Vec<String>) -> Result<Self> {
        let num_cols = table.view()?.num_columns()?;
        let expected = names.len() as i32;
        if expected != num_cols {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of names must equal number of table columns")
                .with_operation("dataframe::from_table")
                .with_context("names_count", names.len().to_string())
                .with_context("table_columns", num_cols.to_string()));
        }

        let schema = build_schema(&names, "dataframe::from_table")?;
        Ok(Self { table, schema })
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.schema.len()
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> Result<i32> {
        if self.schema.is_empty() {
            return Ok(0);
        }
        self.table.view()?.num_rows()
    }

    /// Returns column names in insertion (schema) order.
    pub fn column_names(&self) -> Vec<&str> {
        self.schema.keys().map(|s| s.as_str()).collect()
    }

    /// Borrow a column view by name.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not in the schema.
    pub fn column(&self, name: &str) -> Result<ColumnView<'_>> {
        let &idx = self.schema.get(name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                .with_operation("dataframe::column")
                .with_context("column_name", name.to_string())
        })?;

        let view = self.table.view()?;
        // We must ensure the ColumnView outlives the TableView. The
        // TableView's internal handle is freed on drop, but the ColumnView
        // returned by `table_view.column()` only references the parent
        // Table's memory (not the TableView wrapper). The lifetime `'a` on
        // ColumnView is tied to `&self`, which keeps the Table alive.
        view.column(idx as i32)
    }

    /// Look up the positional index of a column by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.schema.get(name).copied()
    }

    /// Factorize a named column into integer codes and unique values.
    ///
    /// Returns `(codes, uniques)` where `codes[i]` maps to `uniques[codes[i]]`.
    /// This is equivalent to `pandas.factorize()` but GPU-accelerated.
    pub fn factorize(&self, column: &str) -> Result<(Column, Column)> {
        let cv = self.column(column)?;
        cv.factorize().map_err(|e| {
            let err = if e.operation().is_empty() { e.with_operation("dataframe::factorize") } else { e };
            err.with_context("column_name", column.to_string())
        })
    }

    /// Returns the schema mapping column names to positional indices.
    pub fn schema(&self) -> &IndexMap<String, usize> {
        &self.schema
    }

    /// Returns a reference to the underlying [`Table`].
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Consume the `DataFrame`, returning the underlying table and names.
    pub fn into_table(self) -> (Table, Vec<String>) {
        let names: Vec<String> = self.schema.into_keys().collect();
        (self.table, names)
    }

    // -- I/O (Parquet) -------------------------------------------------------

    /// Read a Parquet file into a `DataFrame`, using column names from the
    /// file's schema metadata.
    ///
    /// This is a convenience wrapper around
    /// [`parquet::read_parquet_with_metadata`].
    pub fn read_parquet(path: impl AsRef<Path>) -> Result<Self> {
        let opts = ParquetReaderOptions::new(path)?;
        Self::read_parquet_with_options(&opts)
    }

    /// Read a Parquet file into a `DataFrame` using caller-provided options.
    ///
    /// Column names are taken from the Parquet file's schema metadata.
    pub fn read_parquet_with_options(options: &ParquetReaderOptions) -> Result<Self> {
        let result = parquet::read_parquet_with_metadata(options)?;
        Self::from_table(result.table, result.column_names)
    }

    /// Write this `DataFrame` to a Parquet file, embedding column names in
    /// the file's schema metadata.
    pub fn write_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        let names: Vec<&str> = self.column_names();
        let opts = ParquetWriterOptions::new(path)?.column_names(&names);
        let view = self.table.view()?;
        parquet::write_parquet(&view, &opts)
    }

    // -- Row slicing ---------------------------------------------------------

    /// Return the first `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    pub fn head(&self, n: usize) -> Result<Self> {
        let total = self.num_rows()? as usize;
        let take = n.min(total);
        self.gather_range(0, take)
    }

    /// Return the last `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    pub fn tail(&self, n: usize) -> Result<Self> {
        let total = self.num_rows()? as usize;
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
        let total = self.num_rows()? as usize;
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range(start, take)
    }

    // -- Filter, sort, deduplicate -------------------------------------------

    /// Filter rows using a boolean mask column.
    ///
    /// Rows where `mask` is `true` are kept; `false` or null rows are
    /// dropped. The schema is preserved.
    pub fn filter(&self, mask: &ColumnView<'_>) -> Result<Self> {
        let table_view = self.table.view()?;
        let result_table = filter::apply_boolean_mask(&table_view, mask)?;
        Ok(Self { table: result_table, schema: self.schema.clone() })
    }

    /// Filter rows to those where `column` values appear in `values`.
    ///
    /// This is a convenience wrapper around [`crate::search::contains`] +
    /// [`DataFrame::filter`], equivalent to pandas `df[df[column].isin(values)]`.
    ///
    /// `values` is the haystack (the set to test membership against).
    /// The column and values must have the same data type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin(&self, column: &str, values: &ColumnView<'_>) -> Result<Self> {
        let col_view = self.column(column)?;
        let mask = crate::search::contains(values, &col_view)?;
        let mask_view = mask.view()?;
        self.filter(&mask_view)
    }

    /// Sort rows by the given column names and orders.
    ///
    /// `by` and `orders` must have the same length. The schema is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `by` is not in the schema, or if
    /// `by.len() != orders.len()`.
    pub fn sort(&self, by: &[&str], orders: &[Order]) -> Result<Self> {
        if by.len() != orders.len() {
            return Err(Error::new(ErrorKind::InvalidOperation, "number of sort columns must equal number of orders")
                .with_operation("dataframe::sort")
                .with_context("by_count", by.len().to_string())
                .with_context("orders_count", orders.len().to_string()));
        }

        let key_indices = resolve_column_names(&self.schema, by, "dataframe::sort")?;

        // Check whether `by` names are exactly all columns in table order.
        let all_in_order =
            key_indices.len() == self.schema.len() && key_indices.iter().enumerate().all(|(i, &idx)| idx == i);

        let table_view = self.table.view()?;

        let result_table = if all_in_order {
            sorting::sort(&table_view, Some(orders), None)?
        } else {
            // Build a keys-only table, get sorted_order, then gather.
            let keys_table = build_keys_table(&table_view, &key_indices)?;
            let keys_view = keys_table.view()?;
            let order_col = sorting::sorted_order(&keys_view, Some(orders), None)?;
            let order_view = order_col.view()?;
            copying::gather(&table_view, &order_view, copying::OutOfBoundsPolicy::DontCheck)?
        };

        Ok(Self { table: result_table, schema: self.schema.clone() })
    }

    /// Return the `n` rows with the largest values in `column`.
    ///
    /// Equivalent to `self.sort(&[column], &[Descending])?.head(n)`.
    pub fn nlargest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort(&[column], &[Order::Descending])?.head(n)
    }

    /// Return the `n` rows with the smallest values in `column`.
    ///
    /// Equivalent to `self.sort(&[column], &[Ascending])?.head(n)`.
    pub fn nsmallest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort(&[column], &[Order::Ascending])?.head(n)
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
        let key_indices = resolve_column_names(&self.schema, subset, "dataframe::distinct")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let table_view = self.table.view()?;
        let result_table =
            stream_compaction::distinct(&table_view, &key_i32, keep, NullEquality::Equal, NanEquality::AllEqual)?;
        Ok(Self { table: result_table, schema: self.schema.clone() })
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
        let key_indices = resolve_column_names(&self.schema, subset, "dataframe::drop_nulls")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let table_view = self.table.view()?;
        let result_table = stream_compaction::drop_nulls(&table_view, &key_i32, keep_threshold)?;
        Ok(Self { table: result_table, schema: self.schema.clone() })
    }

    /// Drop rows where **any** of the specified `subset` columns contain nulls.
    ///
    /// Equivalent to `self.drop_nulls(subset, subset.len() as i32)` — every
    /// column in `subset` must be non-null for a row to be kept.
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

    // -- Joins ---------------------------------------------------------------

    /// Inner join: returns only rows with matching keys in both DataFrames.
    ///
    /// Output schema is: `left_keys + left_payload + right_payload`.
    /// If a right payload name collides with an existing output column name,
    /// `"_right"` is appended (and numbered if needed).
    pub fn inner_join(&self, right: &DataFrame, on: &[&str]) -> Result<Self> {
        self.join_merged_impl(right, on, JoinKind::Inner, "dataframe::inner_join")
    }

    /// Left join: all left rows are preserved; the right side is null-padded
    /// for non-matching rows.
    ///
    /// Output schema is: `left_keys + left_payload + right_payload`.
    pub fn left_join(&self, right: &DataFrame, on: &[&str]) -> Result<Self> {
        self.join_merged_impl(right, on, JoinKind::Left, "dataframe::left_join")
    }

    /// Full (outer) join: all rows from both sides are preserved, with
    /// null-padding where there is no match.
    ///
    /// Output schema is: `left_keys + left_payload + right_payload`.
    ///
    /// For right-only rows, key columns are coalesced from right keys when
    /// left keys are null.
    pub fn full_join(&self, right: &DataFrame, on: &[&str]) -> Result<Self> {
        self.join_merged_impl(right, on, JoinKind::Full, "dataframe::full_join")
    }

    /// Compatibility API: return split left/right join outputs.
    pub fn inner_join_split(&self, right: &DataFrame, on: &[&str]) -> Result<DataFrameJoinResult> {
        self.join_split_on_impl(right, on, on, JoinKind::Inner, "dataframe::inner_join_split")
    }

    /// Compatibility API: return split left/right join outputs.
    pub fn left_join_split(&self, right: &DataFrame, on: &[&str]) -> Result<DataFrameJoinResult> {
        self.join_split_on_impl(right, on, on, JoinKind::Left, "dataframe::left_join_split")
    }

    /// Compatibility API: return split left/right join outputs.
    pub fn full_join_split(&self, right: &DataFrame, on: &[&str]) -> Result<DataFrameJoinResult> {
        self.join_split_on_impl(right, on, on, JoinKind::Full, "dataframe::full_join_split")
    }

    /// Inner join with asymmetric key names.
    ///
    /// `left_on` names columns in `self`, `right_on` names columns in
    /// `right`. Both slices must have the same length (each pair is matched
    /// positionally). Output key columns use the **left** key names.
    ///
    /// Output schema is: `left_keys + left_payload + right_payload`.
    pub fn inner_join_on(&self, right: &DataFrame, left_on: &[&str], right_on: &[&str]) -> Result<Self> {
        self.join_merged_on_impl(right, left_on, right_on, JoinKind::Inner, "dataframe::inner_join_on")
    }

    /// Left join with asymmetric key names.
    ///
    /// All left rows are preserved; the right side is null-padded for
    /// non-matching rows. Output key columns use the **left** key names.
    pub fn left_join_on(&self, right: &DataFrame, left_on: &[&str], right_on: &[&str]) -> Result<Self> {
        self.join_merged_on_impl(right, left_on, right_on, JoinKind::Left, "dataframe::left_join_on")
    }

    /// Full (outer) join with asymmetric key names.
    ///
    /// All rows from both sides are preserved, with null-padding where
    /// there is no match. Key columns are coalesced and use the **left**
    /// key names.
    pub fn full_join_on(&self, right: &DataFrame, left_on: &[&str], right_on: &[&str]) -> Result<Self> {
        self.join_merged_on_impl(right, left_on, right_on, JoinKind::Full, "dataframe::full_join_on")
    }

    /// Internal merged-join implementation.
    fn join_merged_impl(
        &self,
        right: &DataFrame,
        on: &[&str],
        kind: JoinKind,
        operation: &'static str,
    ) -> Result<Self> {
        self.join_merged_on_impl(right, on, on, kind, operation)
    }

    /// Internal merged-join implementation with asymmetric keys.
    fn join_merged_on_impl(
        &self,
        right: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        kind: JoinKind,
        operation: &'static str,
    ) -> Result<Self> {
        let split = self.join_split_on_impl(right, left_on, right_on, kind, operation)?;
        merge_join_result(split, left_on, kind == JoinKind::Full, operation)
    }

    /// Internal split-join dispatch with asymmetric keys.
    fn join_split_on_impl(
        &self,
        right: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        kind: JoinKind,
        operation: &'static str,
    ) -> Result<DataFrameJoinResult> {
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
        let left_key_indices = resolve_column_names(&self.schema, left_on, operation)?;
        let right_key_indices = resolve_column_names(&right.schema, right_on, operation)?;

        // Compute payload indices (all columns not in keys, preserving order).
        let left_payload_indices = payload_indices(&self.schema, &left_key_indices);
        let right_payload_indices = payload_indices(&right.schema, &right_key_indices);

        // Build keys and payload tables via deep-copy from column views.
        let left_tv = self.table.view()?;
        let right_tv = right.table.view()?;

        let lk_table = build_keys_table(&left_tv, &left_key_indices)?;
        let rk_table = build_keys_table(&right_tv, &right_key_indices)?;
        let lp_table = build_keys_table(&left_tv, &left_payload_indices)?;
        let rp_table = build_keys_table(&right_tv, &right_payload_indices)?;

        let lkv = lk_table.view()?;
        let rkv = rk_table.view()?;
        let lpv = lp_table.view()?;
        let rpv = rp_table.view()?;

        let result = match kind {
            JoinKind::Inner => join::inner_join(&lkv, &rkv, &lpv, &rpv, join::NullEquality::Equal)?,
            JoinKind::Left => join::left_join(&lkv, &rkv, &lpv, &rpv, join::NullEquality::Equal)?,
            JoinKind::Full => join::full_join(&lkv, &rkv, &lpv, &rpv, join::NullEquality::Equal)?,
        };

        // Build schemas: left side uses left key names, right side uses right key names.
        let left_schema_names = build_join_schema_names(&self.schema, &left_key_indices, &left_payload_indices);
        let right_schema_names = build_join_schema_names(&right.schema, &right_key_indices, &right_payload_indices);

        Ok(DataFrameJoinResult {
            left: DataFrame::from_table(result.left, left_schema_names)?,
            right: DataFrame::from_table(result.right, right_schema_names)?,
        })
    }

    // -- Groupby aggregation -------------------------------------------------

    /// Start building a groupby aggregation.
    ///
    /// `keys` specifies the column names to group by. Returns a
    /// [`GroupByBuilder`] on which you chain `.agg()` calls, then
    /// `.execute()` to run the aggregation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// # use cudf::dataframe::DataFrame;
    /// # use cudf::groupby::Aggregation;
    /// # let df = DataFrame::new(
    /// #     vec!["key".into(), "val".into()],
    /// #     vec![Column::from_slice(&[1i32,2]).unwrap(),
    /// #          Column::from_slice(&[10i64,20]).unwrap()],
    /// # ).unwrap();
    /// let result =
    ///     df.groupby(&["key"]).agg("val", Aggregation::Sum).execute().unwrap();
    /// ```
    pub fn groupby<'a>(&'a self, keys: &[&str]) -> GroupByBuilder<'a> {
        GroupByBuilder { df: self, key_names: keys.iter().map(|s| s.to_string()).collect(), agg_requests: Vec::new() }
    }

    // -- Concat, copy, rename, select ----------------------------------------

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
        // Validate schemas match: same names, same order.
        if self.schema.len() != other.schema.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot concatenate DataFrames with different column counts",
            )
            .with_operation("dataframe::concat")
            .with_context("left_columns", self.schema.len().to_string())
            .with_context("right_columns", other.schema.len().to_string()));
        }

        for (left_name, right_name) in self.schema.keys().zip(other.schema.keys()) {
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
        let left_view = self.table.view()?;
        let right_view = other.table.view()?;
        for i in 0..self.schema.len() {
            let left_type = left_view.column(i as i32)?.data_type()?.id();
            let right_type = right_view.column(i as i32)?.data_type()?.id();
            if left_type != right_type {
                let name = self.schema.get_index(i).map(|(n, _)| n.as_str()).unwrap_or("<unknown>");
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

        let result_table = crate::concatenate::concatenate_tables(&[&left_view, &right_view])?;

        Ok(Self { table: result_table, schema: self.schema.clone() })
    }

    /// Deep-copy this DataFrame (both the GPU table data and the schema).
    pub fn copy(&self) -> Result<Self> {
        let table_copy = self.table.copy()?;
        Ok(Self { table: table_copy, schema: self.schema.clone() })
    }

    /// Return a new DataFrame with one column renamed.
    ///
    /// This borrowed variant deep-copies the underlying table to preserve
    /// immutable value semantics from `&self`. For a zero-GPU-work rename,
    /// use [`DataFrame::rename_owned`] which consumes `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if `old` is not in the schema, or if `new` already
    /// exists in the schema.
    pub fn rename(&self, old: &str, new: &str) -> Result<Self> {
        if !self.schema.contains_key(old) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", old))
                .with_operation("dataframe::rename")
                .with_context("column_name", old.to_string()));
        }

        if old != new && self.schema.contains_key(new) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("cannot rename '{}' to '{}': target name already exists", old, new),
            )
            .with_operation("dataframe::rename")
            .with_context("old_name", old.to_string())
            .with_context("new_name", new.to_string()));
        }

        // Rebuild schema with the renamed column, preserving order.
        let mut new_schema = IndexMap::with_capacity(self.schema.len());
        for (name, &idx) in &self.schema {
            if name == old {
                new_schema.insert(new.to_string(), idx);
            } else {
                new_schema.insert(name.clone(), idx);
            }
        }

        let table_copy = self.table.copy()?;
        Ok(Self { table: table_copy, schema: new_schema })
    }

    /// Return a new DataFrame with one column renamed, consuming `self`.
    ///
    /// This is a pure schema rewrite with **no GPU data copy**.
    pub fn rename_owned(self, old: &str, new: &str) -> Result<Self> {
        if !self.schema.contains_key(old) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", old))
                .with_operation("dataframe::rename_owned")
                .with_context("column_name", old.to_string()));
        }

        if old != new && self.schema.contains_key(new) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("cannot rename '{}' to '{}': target name already exists", old, new),
            )
            .with_operation("dataframe::rename_owned")
            .with_context("old_name", old.to_string())
            .with_context("new_name", new.to_string()));
        }

        let mut new_schema = IndexMap::with_capacity(self.schema.len());
        for (name, &idx) in &self.schema {
            if name == old {
                new_schema.insert(new.to_string(), idx);
            } else {
                new_schema.insert(name.clone(), idx);
            }
        }

        Ok(Self { table: self.table, schema: new_schema })
    }

    /// Return a new DataFrame containing only the specified columns.
    ///
    /// The columns appear in the order given in `columns`. Each selected
    /// column is deep-copied from its column view.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `columns` is not in the schema, or if
    /// `columns` contains duplicates.
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        // Reject duplicates to keep schema (name -> index) consistent.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for &name in columns {
            if !seen.insert(name) {
                return Err(Error::new(ErrorKind::InvalidOperation, "selected columns must be unique")
                    .with_operation("dataframe::select")
                    .with_context("duplicate_column", name.to_string())
                    .with_context("selected_column_count", columns.len().to_string()));
            }
        }

        let indices = resolve_column_names(&self.schema, columns, "dataframe::select")?;

        let table_view = self.table.view()?;
        let selected_table = build_keys_table(&table_view, &indices)?;

        let mut new_schema = IndexMap::with_capacity(columns.len());
        for (i, &name) in columns.iter().enumerate() {
            new_schema.insert(name.to_string(), i);
        }

        Ok(Self { table: selected_table, schema: new_schema })
    }

    // -- Column-level convenience --------------------------------------------

    /// Return a new DataFrame with the given column added or replaced.
    ///
    /// If `name` already exists in the schema, the column at that position
    /// is replaced. If `name` is new, the column is appended.
    ///
    /// The new column must have the same row count as the existing DataFrame
    /// (unless the DataFrame is empty with 0 columns).
    ///
    /// # Errors
    ///
    /// Returns an error if the row counts do not match.
    pub fn with_column(&self, name: &str, col: Column) -> Result<Self> {
        let col_view = col.view()?;
        let col_rows = col_view.len()?;
        let self_rows = self.num_rows()?;

        if self.num_columns() > 0 && col_rows != self_rows {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
            )
            .with_operation("dataframe::with_column")
            .with_context("column_name", name.to_string())
            .with_context("column_rows", col_rows.to_string())
            .with_context("dataframe_rows", self_rows.to_string()));
        }

        // Determine whether this is a replacement or an append.
        let replace_idx = self.schema.get(name).copied();

        let table_view = self.table.view()?;
        let num_existing = self.schema.len();
        let new_count = if replace_idx.is_some() { num_existing } else { num_existing + 1 };

        let mut columns: Vec<Column> = Vec::with_capacity(new_count);
        let mut schema = IndexMap::with_capacity(new_count);

        for (col_name, &idx) in &self.schema {
            if Some(idx) == replace_idx {
                // Replace: use the new column (deep-copy so we own it).
                let cv = col.view()?;
                let type_id = cv.data_type()?.id();
                let len = cv.len()?;
                let owned =
                    if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
                schema.insert(name.to_string(), columns.len());
                columns.push(owned);
            } else {
                // Copy existing column.
                let cv = table_view.column(idx as i32)?;
                let type_id = cv.data_type()?.id();
                let len = cv.len()?;
                let owned =
                    if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
                schema.insert(col_name.clone(), columns.len());
                columns.push(owned);
            }
        }

        // Append if this is a new column.
        if replace_idx.is_none() {
            let cv = col.view()?;
            let type_id = cv.data_type()?.id();
            let len = cv.len()?;
            let owned = if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
            schema.insert(name.to_string(), columns.len());
            columns.push(owned);
        }

        let table = Table::from_columns(columns)?;
        Ok(Self { table, schema })
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
        let &target_idx = self.schema.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::replace_nulls")
                .with_context("column_name", column.to_string())
        })?;

        self.rebuild_with_replaced_column(target_idx, "dataframe::replace_nulls", |cv| {
            replace::replace_nulls(cv, value)
        })
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
        let &target_idx = self.schema.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::cast_column")
                .with_context("column_name", column.to_string())
        })?;

        self.rebuild_with_replaced_column(target_idx, "dataframe::cast_column", |cv| unary::cast(cv, target_type))
    }

    /// Internal helper: rebuild the table, replacing one column by applying
    /// a transformation function `f` to its view. All other columns are
    /// deep-copied.
    fn rebuild_with_replaced_column<F>(&self, target_idx: usize, _operation: &'static str, f: F) -> Result<Self>
    where F: FnOnce(&ColumnView<'_>) -> Result<Column> {
        let table_view = self.table.view()?;
        let num_cols = self.schema.len();
        let mut columns: Vec<Column> = Vec::with_capacity(num_cols);
        let mut f_opt = Some(f);

        for (_, &idx) in &self.schema {
            let cv = table_view.column(idx as i32)?;
            if idx == target_idx {
                let f = f_opt.take().expect("rebuild_with_replaced_column: target_idx matched twice");
                columns.push(f(&cv)?);
            } else {
                let type_id = cv.data_type()?.id();
                let len = cv.len()?;
                let owned =
                    if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
                columns.push(owned);
            }
        }

        let table = Table::from_columns(columns)?;
        Ok(Self { table, schema: self.schema.clone() })
    }
}

/// Builder for groupby aggregation on a [`DataFrame`].
///
/// Created via [`DataFrame::groupby`]. Chain `.agg()` calls to add
/// aggregation requests, then call `.execute()` to run.
pub struct GroupByBuilder<'a> {
    df: &'a DataFrame,
    key_names: Vec<String>,
    /// (value_column_name, aggregation)
    agg_requests: Vec<(String, Aggregation)>,
}

impl<'a> GroupByBuilder<'a> {
    /// Add an aggregation on `column` using the given [`Aggregation`].
    ///
    /// The result column will be named `"{column}_{aggregation}"`, e.g.
    /// `"price_sum"`.
    pub fn agg(mut self, column: &str, agg: Aggregation) -> Self {
        self.agg_requests.push((column.to_string(), agg));
        self
    }

    /// Execute the groupby aggregation, returning a new [`DataFrame`].
    ///
    /// The result schema is: `[key_columns..., result_columns...]` where
    /// each result column is named `"{value_column}_{aggregation}"`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No aggregations were added.
    /// - Any key or value column name is not in the DataFrame's schema.
    pub fn execute(&self) -> Result<DataFrame> {
        if self.agg_requests.is_empty() {
            return Err(Error::new(ErrorKind::InvalidOperation, "at least one aggregation is required")
                .with_operation("dataframe::groupby::execute"));
        }

        // Resolve key column names to indices.
        let key_indices = resolve_column_names(
            &self.df.schema,
            &self.key_names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "dataframe::groupby::execute",
        )?;

        // Resolve value column names to indices.
        let value_col_names: Vec<&str> = self.agg_requests.iter().map(|(name, _)| name.as_str()).collect();

        // Validate all value column names exist.
        for &name in &value_col_names {
            if !self.df.schema.contains_key(name) {
                return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                    .with_operation("dataframe::groupby::execute")
                    .with_context("column_name", name.to_string()));
            }
        }

        // Build keys table.
        let table_view = self.df.table.view()?;
        let keys_table = build_keys_table(&table_view, &key_indices)?;
        let keys_view = keys_table.view()?;

        // Group aggregation requests by value column. Each unique value
        // column gets one AggregationRequest with potentially multiple
        // aggregations. We preserve the order of first appearance for
        // output column naming.
        //
        // We need to track: for each request, which aggs, and what
        // names they produce.
        //
        // Strategy: build one AggregationRequest per (column_name, agg)
        // pair — this is the simplest approach and works with the existing
        // groupby API which returns one Column per aggregation per request.

        let mut owned_value_cols: Vec<Column> = Vec::new();
        let mut request_meta: Vec<(usize, Vec<Aggregation>)> = Vec::new(); // (owned_col_index, aggs)
        let mut result_names: Vec<String> = Vec::new();

        // Index dedup: map value column name → index in owned_value_cols
        let mut col_name_to_owned_idx: IndexMap<String, usize> = IndexMap::new();
        // Map owned value-column index → request_meta group index.
        let mut owned_idx_to_group_idx: IndexMap<usize, usize> = IndexMap::new();
        // Output ordering map in original .agg() call order:
        // (group_index_in_request_meta, agg_index_within_group)
        let mut output_order: Vec<(usize, usize)> = Vec::new();

        for (col_name, agg) in &self.agg_requests {
            let schema_idx = *self.df.schema.get(col_name.as_str()).unwrap();
            let owned_idx = if let Some(&idx) = col_name_to_owned_idx.get(col_name.as_str()) {
                idx
            } else {
                let cv = table_view.column(schema_idx as i32)?;
                let type_id = cv.data_type()?.id();
                let len = cv.len()?;
                let owned =
                    if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
                let idx = owned_value_cols.len();
                owned_value_cols.push(owned);
                col_name_to_owned_idx.insert(col_name.clone(), idx);
                idx
            };

            let group_idx = if let Some(&idx) = owned_idx_to_group_idx.get(&owned_idx) {
                idx
            } else {
                let idx = request_meta.len();
                request_meta.push((owned_idx, Vec::new()));
                owned_idx_to_group_idx.insert(owned_idx, idx);
                idx
            };

            let agg_idx_within_group = request_meta[group_idx].1.len();
            request_meta[group_idx].1.push(agg.clone());
            output_order.push((group_idx, agg_idx_within_group));

            result_names.push(format!("{}_{}", col_name, agg_suffix(agg)));
        }

        // Build AggregationRequests.
        let value_views: Vec<_> = owned_value_cols.iter().map(|c| c.view()).collect::<Result<Vec<_>>>()?;

        let agg_slices: Vec<Vec<Aggregation>> = request_meta.iter().map(|(_, aggs)| aggs.clone()).collect();

        let requests: Vec<AggregationRequest<'_>> = request_meta
            .iter()
            .enumerate()
            .map(|(i, (owned_idx, _))| AggregationRequest {
                values: &value_views[*owned_idx],
                aggregations: &agg_slices[i],
            })
            .collect();

        let gb_result = groupby::groupby_aggregate(&keys_view, &requests, NullPolicy::Exclude)?;

        // Reorder aggregate result columns to match original .agg() call order.
        let mut grouped_results: Vec<Vec<Option<Column>>> =
            gb_result.results.into_iter().map(|cols| cols.into_iter().map(Some).collect()).collect();

        let mut all_result_cols: Vec<Column> = Vec::with_capacity(output_order.len());
        for (request_index, (group_idx, agg_idx)) in output_order.iter().enumerate() {
            let group_count = grouped_results.len();
            let group = grouped_results.get_mut(*group_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby aggregation result group index out of bounds")
                    .with_operation("dataframe::groupby::execute")
                    .with_context("request_index", request_index.to_string())
                    .with_context("group_index", group_idx.to_string())
                    .with_context("group_count", group_count.to_string())
            })?;

            let group_agg_count = group.len();
            let col_slot = group.get_mut(*agg_idx).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby aggregation result index out of bounds")
                    .with_operation("dataframe::groupby::execute")
                    .with_context("request_index", request_index.to_string())
                    .with_context("group_index", group_idx.to_string())
                    .with_context("agg_index", agg_idx.to_string())
                    .with_context("group_agg_count", group_agg_count.to_string())
            })?;

            let col = col_slot.take().ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "groupby aggregation result column consumed more than once")
                    .with_operation("dataframe::groupby::execute")
                    .with_context("request_index", request_index.to_string())
                    .with_context("group_index", group_idx.to_string())
                    .with_context("agg_index", agg_idx.to_string())
            })?;

            all_result_cols.push(col);
        }

        // Build the key columns from the keys table via deep-copy.
        let result_keys_view = gb_result.keys.view()?;
        let num_key_cols = result_keys_view.num_columns()?;
        let mut final_cols: Vec<Column> = Vec::with_capacity(num_key_cols as usize + all_result_cols.len());

        for i in 0..num_key_cols {
            let cv = result_keys_view.column(i)?;
            let type_id = cv.data_type()?.id();
            let len = cv.len()?;
            let owned = if len == 0 { Column::empty(type_id)? } else { copy_column_view_to_owned(&cv, type_id, len)? };
            final_cols.push(owned);
        }

        final_cols.extend(all_result_cols);

        // Build schema: key names + result names.
        let mut schema_names: Vec<String> = Vec::with_capacity(self.key_names.len() + result_names.len());
        schema_names.extend(self.key_names.clone());
        schema_names.extend(result_names);

        DataFrame::new(schema_names, final_cols)
    }
}

/// Return the suffix for an aggregation type used in result column naming.
fn agg_suffix(agg: &Aggregation) -> &'static str {
    match agg {
        Aggregation::Sum => "sum",
        Aggregation::Min => "min",
        Aggregation::Max => "max",
        Aggregation::Count { .. } => "count",
        Aggregation::Mean => "mean",
        Aggregation::Median => "median",
        Aggregation::NUnique => "nunique",
        Aggregation::Std { .. } => "std",
        Aggregation::Variance { .. } => "var",
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an `IndexMap` from column names, detecting duplicates.
fn build_schema(names: &[String], operation: &'static str) -> Result<IndexMap<String, usize>> {
    let mut schema = IndexMap::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        if schema.contains_key(name) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("duplicate column name '{}'", name))
                .with_operation(operation)
                .with_context("duplicate_name", name.clone()));
        }
        schema.insert(name.clone(), i);
    }
    Ok(schema)
}

impl DataFrame {
    /// Internal: gather a contiguous range of rows into a new `DataFrame`,
    /// preserving the schema.
    fn gather_range(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema();
        }

        let indices: Vec<i32> = (start as i32..(start + len) as i32).collect();
        let map = Column::from_slice(&indices)?;
        let table_view = self.table.view()?;
        let result_table = copying::gather(&table_view, &map.view()?, copying::OutOfBoundsPolicy::DontCheck)?;

        Ok(Self { table: result_table, schema: self.schema.clone() })
    }

    /// Internal: build an empty DataFrame with the same schema (column names
    /// and types, zero rows).
    fn empty_with_same_schema(&self) -> Result<Self> {
        let empty_cols: Vec<Column> = {
            let view = self.table.view()?;
            let n = view.num_columns()?;
            (0..n)
                .map(|i| {
                    let cv = view.column(i)?;
                    Column::empty(cv.data_type()?.id())
                })
                .collect::<Result<Vec<_>>>()?
        };
        let table = Table::from_columns(empty_cols)?;
        Ok(Self { table, schema: self.schema.clone() })
    }
}

/// Resolve a slice of column name strings to their positional indices,
/// returning an error if any name is missing.
fn resolve_column_names(
    schema: &IndexMap<String, usize>,
    names: &[&str],
    operation: &'static str,
) -> Result<Vec<usize>> {
    names
        .iter()
        .map(|&name| {
            schema.get(name).copied().ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                    .with_operation(operation)
                    .with_context("column_name", name.to_string())
            })
        })
        .collect()
}

/// Build an owned `Table` containing only the columns at the given indices
/// from `source`. This deep-copies each selected column from its
/// `ColumnView`.
fn build_keys_table(source: &crate::table::TableView<'_>, indices: &[usize]) -> Result<Table> {
    let cols: Vec<Column> = indices
        .iter()
        .map(|&idx| {
            let cv = source.column(idx as i32)?;
            let type_id = cv.data_type()?.id();
            let len = cv.len()?;

            if len == 0 {
                return Column::empty(type_id);
            }

            copy_column_view_to_owned(&cv, type_id, len)
        })
        .collect::<Result<Vec<_>>>()?;

    Table::from_columns(cols)
}

/// Deep-copy a `ColumnView` into an owned `Column`.
///
/// Uses `cudf_column_copy_from_view`, which supports fixed-width, string,
/// and nested types.
fn copy_column_view_to_owned(cv: &ColumnView<'_>, _type_id: crate::types::TypeId, _len: i32) -> Result<Column> {
    let mut handle = std::ptr::null_mut();
    crate::error::check_status_in(
        unsafe { cudf_sys::cudf_column_copy_from_view(cv.raw_handle(), &mut handle) },
        "dataframe::copy_column_view",
    )?;
    Ok(unsafe { Column::from_raw(handle) })
}

/// Internal join type dispatch tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinKind {
    Inner,
    Left,
    Full,
}

/// Compute payload column indices: all schema positions NOT in `key_indices`,
/// in schema insertion order.
fn payload_indices(schema: &IndexMap<String, usize>, key_indices: &[usize]) -> Vec<usize> {
    let key_set: std::collections::HashSet<usize> = key_indices.iter().copied().collect();
    schema.values().copied().filter(|idx| !key_set.contains(idx)).collect()
}

/// Build the schema name list for a join result side:
/// key column names (in `on` order) followed by payload column names.
fn build_join_schema_names(
    schema: &IndexMap<String, usize>,
    key_indices: &[usize],
    payload_indices: &[usize],
) -> Vec<String> {
    // Build reverse map: positional index → name.
    let idx_to_name: Vec<&str> = {
        let mut pairs: Vec<(usize, &str)> = schema.iter().map(|(k, &v)| (v, k.as_str())).collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs.into_iter().map(|(_, name)| name).collect()
    };

    let mut names = Vec::with_capacity(key_indices.len() + payload_indices.len());
    for &idx in key_indices {
        names.push(idx_to_name[idx].to_string());
    }
    for &idx in payload_indices {
        names.push(idx_to_name[idx].to_string());
    }
    names
}

/// Merge a split join result into one DataFrame:
/// `left_keys + left_payload + right_payload`.
fn merge_join_result(
    split: DataFrameJoinResult,
    on: &[&str],
    coalesce_keys: bool,
    operation: &'static str,
) -> Result<DataFrame> {
    let DataFrameJoinResult { left, right } = split;

    // Extract join-side columns.
    let left_cols = left.table.into_columns()?;
    let right_cols = right.table.into_columns()?;

    let left_names: Vec<String> = left.schema.into_keys().collect();
    let right_names: Vec<String> = right.schema.into_keys().collect();

    let key_count = on.len();

    if left_cols.len() != left_names.len() || right_cols.len() != right_names.len() {
        return Err(Error::new(ErrorKind::InvalidOperation, "join result schema/column count mismatch")
            .with_operation(operation)
            .with_context("left_columns", left_cols.len().to_string())
            .with_context("left_names_count", left_names.len().to_string())
            .with_context("right_columns", right_cols.len().to_string())
            .with_context("right_names_count", right_names.len().to_string()));
    }

    if key_count > left_cols.len() || key_count > right_cols.len() {
        return Err(Error::new(ErrorKind::InvalidOperation, "join key count exceeds result column count")
            .with_operation(operation)
            .with_context("num_keys", key_count.to_string())
            .with_context("left_columns", left_cols.len().to_string())
            .with_context("right_columns", right_cols.len().to_string()));
    }

    let mut out_cols: Vec<Column> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    // 1) keys from left (coalesce with right for full join if requested).
    for i in 0..key_count {
        let key_col = if coalesce_keys {
            coalesce_columns(&left_cols[i], &right_cols[i], operation)?
        } else {
            left_cols[i].copy()?
        };
        out_cols.push(key_col);
        out_names.push(left_names[i].clone());
    }

    // 2) left payload.
    for i in key_count..left_cols.len() {
        out_cols.push(left_cols[i].copy()?);
        out_names.push(left_names[i].clone());
    }

    // 3) right payload with collision suffixing.
    for i in key_count..right_cols.len() {
        out_cols.push(right_cols[i].copy()?);
        let unique = unique_right_payload_name(&out_names, &right_names[i]);
        out_names.push(unique);
    }

    DataFrame::new(out_names, out_cols)
}

/// Coalesce two columns by preferring non-null values from `left`; if null,
/// take from `right`.
fn coalesce_columns(left: &Column, right: &Column, operation: &'static str) -> Result<Column> {
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

    coalesce_columns_host_fallback(&lcv, &rcv, ldt.id(), operation)
}

fn coalesce_columns_host_fallback(
    left: &ColumnView<'_>,
    right: &ColumnView<'_>,
    type_id: crate::types::TypeId,
    operation: &'static str,
) -> Result<Column> {
    let len = left.len()?;
    if len != right.len()? {
        return Err(Error::new(ErrorKind::InvalidOperation, "cannot coalesce columns with different lengths")
            .with_operation(operation)
            .with_context("left_len", left.len()?.to_string())
            .with_context("right_len", right.len()?.to_string()));
    }

    let left_valid = left.validity()?;
    let right_valid = right.validity()?;

    macro_rules! coalesce_typed {
        ($t:ty) => {{
            let lv = left.to_vec::<$t>()?;
            let rv = right.to_vec::<$t>()?;
            let mut out = Vec::with_capacity(lv.len());
            let mut validity = Vec::with_capacity(lv.len());
            for i in 0..lv.len() {
                let l_ok = left_valid.as_ref().map(|m| m[i]).unwrap_or(true);
                if l_ok {
                    out.push(lv[i]);
                    validity.push(true);
                } else {
                    let r_ok = right_valid.as_ref().map(|m| m[i]).unwrap_or(true);
                    if r_ok {
                        out.push(rv[i]);
                        validity.push(true);
                    } else {
                        out.push(lv[i]);
                        validity.push(false);
                    }
                }
            }
            Column::from_slice_nullable(&out, &validity)
        }};
    }

    use crate::types::TypeId;
    match type_id {
        TypeId::Int8 => coalesce_typed!(i8),
        TypeId::Int16 => coalesce_typed!(i16),
        TypeId::Int32 => coalesce_typed!(i32),
        TypeId::Int64 => coalesce_typed!(i64),
        TypeId::UInt8 => coalesce_typed!(u8),
        TypeId::UInt16 => coalesce_typed!(u16),
        TypeId::UInt32 => coalesce_typed!(u32),
        TypeId::UInt64 => coalesce_typed!(u64),
        TypeId::Float32 => coalesce_typed!(f32),
        TypeId::Float64 => coalesce_typed!(f64),
        TypeId::Bool8 => coalesce_typed!(bool),
        _ => Err(Error::new(ErrorKind::InvalidOperation, "coalesce unsupported for non-fixed-width type")
            .with_operation(operation)
            .with_context("type_id", format!("{:?}", type_id).to_lowercase())),
    }
}

fn unique_right_payload_name(existing: &[String], base: &str) -> String {
    if !existing.iter().any(|n| n == base) {
        return base.to_string();
    }

    let first = format!("{}_right", base);
    if !existing.iter().any(|n| n == &first) {
        return first;
    }

    let mut i = 2usize;
    loop {
        let candidate = format!("{}_right{}", base, i);
        if !existing.iter().any(|n| n == &candidate) {
            return candidate;
        }
        i += 1;
    }
}
