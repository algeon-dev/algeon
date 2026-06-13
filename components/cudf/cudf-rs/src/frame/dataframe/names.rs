use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::error::Result,
    internal::{
        hash::{
            FastHashMap, FastHashSet, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
            fast_index_map_with_capacity,
        },
        resolve::{duplicate_column_name_error, missing_column_error},
    },
};

impl DataFrame {
    /// Returns column names in insertion (schema) order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    pub fn rename_column(&self, old: &str, new: &str) -> Result<Self> {
        self.rename_columns([(old, new)]).map_err(|e| e.reanchor("dataframe::rename_column"))
    }

    /// Return a new DataFrame with multiple columns renamed at once.
    ///
    /// Each pair in `renames` is `(old_name, new_name)`. All renames are
    /// applied atomically: swapping two column names (e.g. `[("a","b"),("b","a")]`)
    /// is supported.
    ///
    /// This is a schema-only operation. The returned DataFrame rebuilds the
    /// column map and clones the `Arc<Column>` handles without copying GPU
    /// column data.
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
        let mut rename_map: FastHashMap<&str, &str> = fast_hash_map_with_capacity(pairs.len());
        for (old, new) in &pairs {
            if !self.columns.contains_key(old.as_str()) {
                return Err(missing_column_error("dataframe::rename_columns", old));
            }
            if rename_map.contains_key(old.as_str()) {
                return Err(duplicate_column_name_error(
                    "dataframe::rename_columns",
                    old.clone(),
                    pairs.len(),
                    format!("duplicate source column '{}' in rename list", old),
                ));
            }
            rename_map.insert(old.as_str(), new.as_str());
        }

        // Compute the full target name set and check for conflicts.
        let mut target_names: FastHashSet<String> = fast_hash_set_with_capacity(self.columns.len());

        for name in self.columns.keys() {
            let target = if let Some(&new_name) = rename_map.get(name.as_str()) { new_name } else { name.as_str() };
            if !target_names.insert(target.to_string()) {
                return Err(duplicate_column_name_error(
                    "dataframe::rename_columns",
                    target.to_string(),
                    self.columns.len(),
                    format!("rename would produce duplicate column name '{}'", target),
                ));
            }
        }

        let mut new_columns = fast_index_map_with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let target =
                if let Some(&new_name) = rename_map.get(name.as_str()) { new_name.to_string() } else { name.clone() };
            new_columns.insert(target, Arc::clone(col));
        }

        Ok(Self { columns: new_columns, row_count: self.row_count, domain: Arc::clone(&self.domain) })
    }
}
