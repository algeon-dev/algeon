use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::error::{Error, ErrorKind, Result},
    frame::series::Series,
    internal::{
        hash::{FastHashSet, fast_hash_set_with_capacity, fast_index_map_with_capacity},
        resolve::missing_column_error,
    },
};

impl DataFrame {
    /// Return a named [`Series`] for the given column — the canonical user-facing accessor.
    ///
    /// The returned `Series` shares the underlying `Arc<Column>`, so this is a
    /// metadata-only operation that does not copy GPU data. The series name is
    /// set to the DataFrame column name.
    ///
    /// # Cost
    ///
    /// Metadata-only, no GPU data copy: returns a new `Series` sharing the
    /// underlying `Column` via `Arc`. This bumps the `Arc` refcount but does
    /// not allocate or copy GPU column data.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not in the schema.
    pub fn series(&self, name: &str) -> Result<Series> {
        let col = self.columns.get(name).ok_or_else(|| missing_column_error("dataframe::series", name))?;

        Ok(Series::from_column_arc(Arc::clone(col)).rename(name))
    }

    /// Look up the positional index of a column by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.get_index_of(name)
    }

    /// Probe for a column by name, returning `None` on miss.
    ///
    /// Use this when a missing column is a normal case, not an error.
    /// For strict access that errors on miss, use [`series()`](Self::series).
    ///
    /// # Cost
    ///
    /// Metadata-only, no GPU data copy: returns a new `Series` sharing the
    /// underlying `Column` via `Arc`.
    pub fn get(&self, name: &str) -> Option<Series> {
        self.columns.get(name).map(|col| Series::from_column_arc(Arc::clone(col)).rename(name))
    }

    /// Returns `true` if a column with the given name exists in the schema.
    pub fn contains(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Iterate over all columns as [`Series`] in schema order.
    ///
    /// Each item is an owned `Series` sharing the underlying `Column` via
    /// `Arc`. No GPU data is copied.
    pub fn iter_series(&self) -> impl Iterator<Item = Series> + '_ {
        self.columns.iter().map(|(name, col)| Series::from_column_arc(Arc::clone(col)).rename(name))
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
        let col = self.columns.get(column).ok_or_else(|| missing_column_error("dataframe::factorize", column))?;
        let cv = col.view()?;
        let (codes_col, uniques_col) = cv.factorize().map_err(|e| {
            let err = if e.operation().is_empty() { e.with_operation("dataframe::factorize") } else { e };
            err.with_context("column_name", column.to_string())
        })?;
        let codes = Series::from_column(codes_col)?.rename(format!("{column}_codes"));
        let uniques = Series::from_column(uniques_col)?.rename(format!("{column}_uniques"));
        Ok((codes, uniques))
    }
    /// Return a new DataFrame containing only the specified columns.
    ///
    /// This is a schema-only operation. The columns appear in the order given
    /// in `columns`. Each selected column is shared via `Arc::clone` without
    /// copying GPU data. The logical row count is preserved even if `columns`
    /// is empty and the result has zero columns.
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
        let mut seen: FastHashSet<&str> = fast_hash_set_with_capacity(columns.len());
        for name in &columns {
            if !seen.insert(name.as_str()) {
                return Err(Error::new(ErrorKind::InvalidOperation, "selected columns must be unique")
                    .with_operation("dataframe::select")
                    .with_context("duplicate_column", name.clone())
                    .with_context("selected_column_count", columns.len().to_string()));
            }
        }

        let mut new_columns = fast_index_map_with_capacity(columns.len());
        for name in &columns {
            let col = self.columns.get(name.as_str()).ok_or_else(|| missing_column_error("dataframe::select", name))?;
            new_columns.insert(name.clone(), Arc::clone(col));
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Returns a new `DataFrame` value that excludes the specified columns.
    ///
    /// This is a schema-only operation. The returned `DataFrame` reuses the
    /// retained columns via `Arc` sharing; it does not deep-copy GPU column
    /// data.
    ///
    /// The original `DataFrame` is unchanged. Columns excluded from the
    /// returned value are not destroyed immediately; their GPU resources are
    /// released only when the last owning reference is dropped. The logical
    /// row count is preserved even if all columns are removed.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `columns` is not in the schema.
    pub fn without_columns<I>(&self, columns: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut to_drop: FastHashSet<String> = FastHashSet::default();
        for name in columns {
            let name = name.as_ref().to_string();
            if !to_drop.insert(name.clone()) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("duplicate column name '{}' in exclusion list", name),
                )
                .with_operation("dataframe::without_columns")
                .with_context("column_name", name));
            }
        }

        for name in &to_drop {
            if !self.columns.contains_key(name.as_str()) {
                return Err(missing_column_error("dataframe::without_columns", name));
            }
        }

        let mut new_columns = fast_index_map_with_capacity(self.columns.len().saturating_sub(to_drop.len()));
        for (name, col) in &self.columns {
            if !to_drop.contains(name.as_str()) {
                new_columns.insert(name.clone(), Arc::clone(col));
            }
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }
}
