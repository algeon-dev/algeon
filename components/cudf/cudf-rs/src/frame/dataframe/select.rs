use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::error::{Error, ErrorKind, Result},
    frame::{
        factorization::{Codebook, Factorized},
        series::Series,
    },
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
        Series::from_column_arc(Arc::clone(col))
            .map(|series| series.rename(name))
            .map_err(|e| e.with_operation("dataframe::series").with_context("column_name", name.to_string()))
    }

    /// Look up the positional index of a column by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.get_index_of(name)
    }

    /// Probe for a column by name, returning `Ok(None)` on miss.
    ///
    /// Use this when a missing column is a normal case, not an error.
    /// For strict access that errors on miss, use [`series()`](Self::series).
    ///
    /// # Cost
    ///
    /// Metadata-only, no GPU data copy: returns a new `Series` sharing the
    /// underlying `Column` via `Arc`.
    pub fn get(&self, name: &str) -> Result<Option<Series>> {
        match self.columns.get(name) {
            Some(col) => Series::from_column_arc(Arc::clone(col))
                .map(|series| Some(series.rename(name)))
                .map_err(|e| e.with_operation("dataframe::get").with_context("column_name", name.to_string())),
            None => Ok(None),
        }
    }

    /// Returns `true` if a column with the given name exists in the schema.
    pub fn contains(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Iterate over all columns as [`Series`] in schema order.
    ///
    /// Each item is an owned `Series` sharing the underlying `Column` via
    /// `Arc`. No GPU data is copied.
    pub fn iter_series(&self) -> impl Iterator<Item = Result<Series>> + '_ {
        self.columns.iter().map(|(name, col)| {
            Series::from_column_arc(Arc::clone(col))
                .map(|series| series.rename(name))
                .map_err(|e| e.with_operation("dataframe::iter_series").with_context("column_name", name.clone()))
        })
    }

    /// Build a reusable codebook from a named column's distinct values.
    ///
    /// The returned [`Codebook`] owns both the distinct key series and an
    /// explicit positional code series, so it can be reused safely across
    /// later `encode` / `decode` calls.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn codebook(&self, column: &str) -> Result<Codebook> {
        let series = self.series(column).map_err(|e| e.with_operation("dataframe::codebook"))?;
        series
            .codebook()
            .map_err(|e| e.with_operation("dataframe::codebook").with_context("column_name", column.to_string()))
    }

    /// Factorize a named column into row-aligned encoded indices plus a reusable codebook.
    ///
    /// This is equivalent to `pandas.factorize()` but returns a typed
    /// [`Factorized`] result instead of a raw `(codes, uniques)` tuple.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn factorize(&self, column: &str) -> Result<Factorized> {
        let series = self.series(column).map_err(|e| e.with_operation("dataframe::factorize"))?;
        series
            .factorize()
            .map_err(|e| e.with_operation("dataframe::factorize").with_context("column_name", column.to_string()))
    }

    pub fn with_lookup(
        &self,
        output_name: &str,
        input_column: &str,
        lookup_keys: &Series,
        lookup_values: &Series,
    ) -> Result<Self> {
        let looked_up = self
            .series(input_column)
            .map_err(|e| e.with_operation("dataframe::with_lookup"))?
            .lookup(lookup_keys, lookup_values)
            .map_err(|e| {
                e.with_operation("dataframe::with_lookup").with_context("input_column", input_column.to_string())
            })?;
        self.with_column(output_name, looked_up).map_err(|e| {
            e.with_operation("dataframe::with_lookup").with_context("output_name", output_name.to_string())
        })
    }

    pub fn with_lookup_from(
        &self,
        output_name: &str,
        input_column: &str,
        lookup_df: &DataFrame,
        key_column: &str,
        value_column: &str,
    ) -> Result<Self> {
        let lookup_keys = lookup_df.series(key_column).map_err(|e| {
            e.with_operation("dataframe::with_lookup_from").with_context("key_column", key_column.to_string())
        })?;
        let lookup_values = lookup_df.series(value_column).map_err(|e| {
            e.with_operation("dataframe::with_lookup_from").with_context("value_column", value_column.to_string())
        })?;
        self.with_lookup(output_name, input_column, &lookup_keys, &lookup_values).map_err(|e| {
            e.with_operation("dataframe::with_lookup_from")
                .with_context("input_column", input_column.to_string())
                .with_context("output_name", output_name.to_string())
        })
    }

    pub fn with_encoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<Self> {
        let encoded = codebook
            .encode(&self.series(input_column).map_err(|e| e.with_operation("dataframe::with_encoded_column"))?)
            .map_err(|e| {
                e.with_operation("dataframe::with_encoded_column")
                    .with_context("input_column", input_column.to_string())
            })?;
        self.with_column(output_name, encoded).map_err(|e| {
            e.with_operation("dataframe::with_encoded_column")
                .with_context("input_column", input_column.to_string())
                .with_context("output_name", output_name.to_string())
        })
    }

    pub fn with_decoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<Self> {
        let decoded = codebook
            .decode_series(&self.series(input_column).map_err(|e| e.with_operation("dataframe::with_decoded_column"))?)
            .map_err(|e| {
                e.with_operation("dataframe::with_decoded_column")
                    .with_context("input_column", input_column.to_string())
            })?;
        self.with_column(output_name, decoded).map_err(|e| {
            e.with_operation("dataframe::with_decoded_column")
                .with_context("input_column", input_column.to_string())
                .with_context("output_name", output_name.to_string())
        })
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
