use std::sync::Arc;

use super::{
    info::{ColumnMemoryIdentity, FrameColumnMemoryAlias, FrameMemoryEffect},
    owned::DataFrame,
};
use crate::{
    core::{column::Column, error::Result},
    frame::{
        factorization::{Codebook, Factorized},
        series::Series,
    },
    internal::{
        hash::{FastHashSet, fast_hash_set_with_capacity, fast_index_map_with_capacity},
        resolve::{duplicate_column_name_error, missing_column_error},
    },
};

struct ResolvedSelectColumn {
    input_name: String,
    output_name: String,
    input_position: usize,
    column: Arc<Column>,
}

impl DataFrame {
    /// Return the [`Series`] values for the given column.
    ///
    /// The returned `Series` shares the underlying `Arc<Column>`, so this is a
    /// metadata-only operation that does not copy GPU data.
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
            .map_err(|e| e.reanchor("dataframe::series").with_column(name.to_string()))
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
                .map(Some)
                .map_err(|e| e.reanchor("dataframe::get").with_column(name.to_string())),
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
                .map_err(|e| e.reanchor("dataframe::iter_series").with_column(name.clone()))
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
        let series = self.series(column).map_err(|e| e.reanchor("dataframe::codebook"))?;
        series.codebook().map_err(|e| e.reanchor("dataframe::codebook").with_column(column.to_string()))
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
        let series = self.series(column).map_err(|e| e.reanchor("dataframe::factorize"))?;
        series.factorize().map_err(|e| e.reanchor("dataframe::factorize").with_column(column.to_string()))
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
            .map_err(|e| e.reanchor("dataframe::with_lookup"))?
            .lookup(lookup_keys, lookup_values)
            .map_err(|e| e.reanchor("dataframe::with_lookup").with_column(input_column.to_string()))?;
        self.with_column(output_name, looked_up).map_err(|e| e.reanchor("dataframe::with_lookup"))
    }

    pub fn with_lookup_from(
        &self,
        output_name: &str,
        input_column: &str,
        lookup_df: &DataFrame,
        key_column: &str,
        value_column: &str,
    ) -> Result<Self> {
        let lookup_keys = lookup_df.series(key_column).map_err(|e| e.reanchor("dataframe::with_lookup_from"))?;
        let lookup_values = lookup_df.series(value_column).map_err(|e| e.reanchor("dataframe::with_lookup_from"))?;
        self.with_lookup(output_name, input_column, &lookup_keys, &lookup_values)
            .map_err(|e| e.reanchor("dataframe::with_lookup_from"))
    }

    pub fn with_encoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<Self> {
        let encoded = codebook
            .encode(&self.series(input_column).map_err(|e| e.reanchor("dataframe::with_encoded_column"))?)
            .map_err(|e| e.reanchor("dataframe::with_encoded_column").with_column(input_column.to_string()))?;
        self.with_column(output_name, encoded).map_err(|e| e.reanchor("dataframe::with_encoded_column"))
    }

    pub fn with_decoded_column(&self, output_name: &str, input_column: &str, codebook: &Codebook) -> Result<Self> {
        let decoded = codebook
            .decode_series(&self.series(input_column).map_err(|e| e.reanchor("dataframe::with_decoded_column"))?)
            .map_err(|e| e.reanchor("dataframe::with_decoded_column").with_column(input_column.to_string()))?;
        self.with_column(output_name, decoded).map_err(|e| e.reanchor("dataframe::with_decoded_column"))
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
        let mappings = columns.into_iter().map(|name| {
            let name = name.as_ref().to_string();
            (name.clone(), name)
        });
        self.select_as(mappings)
    }

    /// Return a schema-only projection with explicit source and output names.
    ///
    /// Multiple outputs may alias one source column. Output names remain
    /// unique, and every output shares the original column owner.
    pub fn select_as<I, S, T>(&self, mappings: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
        T: AsRef<str>,
    {
        let resolved = self.resolve_select_mappings(mappings, "dataframe::select_as")?;
        let mut new_columns = fast_index_map_with_capacity(resolved.len());
        for column in resolved {
            new_columns.insert(column.output_name, column.column);
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }

    /// Return the lower-level memory ownership effect for
    /// [`select`](Self::select) with the same validation contract.
    ///
    /// `select` is schema-only: every output column aliases a retained input
    /// column owner through `Arc::clone`; no GPU data is allocated or copied.
    pub fn select_memory_effect<I, S>(&self, columns: I) -> Result<FrameMemoryEffect>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mappings = columns.into_iter().map(|name| {
            let name = name.as_ref().to_string();
            (name.clone(), name)
        });
        self.select_as_memory_effect(mappings)
    }

    /// Return the ownership proof for [`select_as`](Self::select_as).
    pub fn select_as_memory_effect<I, S, T>(&self, mappings: I) -> Result<FrameMemoryEffect>
    where
        I: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
        T: AsRef<str>,
    {
        let resolved = self.resolve_select_mappings(mappings, "dataframe::select_as_memory_effect")?;
        let output_columns = resolved
            .into_iter()
            .enumerate()
            .map(|(output_position, column)| {
                let identity = ColumnMemoryIdentity::from_column_arc(&column.column);
                FrameColumnMemoryAlias {
                    input_position: column.input_position,
                    input_name: column.input_name,
                    output_position,
                    output_name: column.output_name,
                    identity,
                }
            })
            .collect();
        Ok(FrameMemoryEffect::AliasOfInput { input: self.memory_identity(), output_columns })
    }

    fn resolve_select_mappings<I, S, T>(
        &self,
        mappings: I,
        operation: &'static str,
    ) -> Result<Vec<ResolvedSelectColumn>>
    where
        I: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
        T: AsRef<str>,
    {
        let mappings = mappings
            .into_iter()
            .map(|(input, output)| (input.as_ref().to_string(), output.as_ref().to_string()))
            .collect::<Vec<_>>();

        let mut seen: FastHashSet<&str> = fast_hash_set_with_capacity(mappings.len());
        for (_, output_name) in &mappings {
            if !seen.insert(output_name.as_str()) {
                return Err(duplicate_column_name_error(
                    operation,
                    output_name.clone(),
                    mappings.len(),
                    "selected output columns must be unique",
                ));
            }
        }

        let mut resolved = Vec::with_capacity(mappings.len());
        for (input_name, output_name) in mappings {
            let (input_position, _, column) = self
                .columns
                .get_full(input_name.as_str())
                .ok_or_else(|| missing_column_error(operation, &input_name))?;
            let column = Arc::clone(column);
            resolved.push(ResolvedSelectColumn { input_name, output_name, input_position, column });
        }
        Ok(resolved)
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
        let requested: Vec<String> = columns.into_iter().map(|name| name.as_ref().to_string()).collect();
        let mut to_drop: FastHashSet<String> = FastHashSet::default();
        for name in &requested {
            if !to_drop.insert(name.clone()) {
                return Err(duplicate_column_name_error(
                    "dataframe::without_columns",
                    name.clone(),
                    requested.len(),
                    format!("duplicate column name '{}' in exclusion list", name),
                ));
            }
        }

        for name in &requested {
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
