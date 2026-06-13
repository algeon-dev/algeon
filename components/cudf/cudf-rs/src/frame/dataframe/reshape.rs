use std::sync::Arc;

use typed_builder::TypedBuilder;

use super::owned::DataFrame;
use crate::{
    core::{column::Column, error::Result},
    frame::series::Series,
    internal::{
        hash::FastHashSet,
        resolve::{duplicate_column_name_error_with_contract, missing_column_error},
    },
};

/// Options for [`DataFrame::melt`] (unpivot: wide → long format).
///
/// Transforms a "wide" DataFrame into "long" format by taking one or more
/// columns and turning them into rows. Mirrors the `melt()` API from cuDF
/// Python / pandas.
#[derive(Debug, Clone, TypedBuilder)]
pub struct MeltOptions {
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    pub(super) id_vars: Vec<String>,
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    pub(super) value_vars: Vec<String>,
    #[builder(default = "variable".to_string(), setter(transform = |name: &str| name.to_string()))]
    pub(super) var_name: String,
    #[builder(default = "value".to_string(), setter(transform = |name: &str| name.to_string()))]
    pub(super) value_name: String,
}

impl Default for MeltOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MeltOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DataFrame {
    /// Unpivot a DataFrame from wide format to long format.
    ///
    /// Takes a set of id columns (kept as-is) and a set of value columns
    /// (turned into rows). Produces a DataFrame with columns:
    /// `[...id_vars, var_name, value_name]`.
    ///
    /// Equivalent to cuDF Python's `DataFrame.melt()` / `cudf.melt()`.
    pub fn melt(&self, opts: MeltOptions) -> Result<Self> {
        fn duplicate_name_error(
            name: &str,
            column_name_count: usize,
            message: String,
            reason: &'static str,
            remedy: &'static str,
        ) -> crate::Error {
            duplicate_column_name_error_with_contract(
                "dataframe::melt",
                name.to_string(),
                column_name_count,
                message,
                reason,
                remedy,
            )
        }

        let id_vars = &opts.id_vars;
        let value_vars: Vec<String> = if opts.value_vars.is_empty() {
            let id_set: FastHashSet<&str> = id_vars.iter().map(|s| s.as_str()).collect();
            self.columns.keys().filter(|name| !id_set.contains(name.as_str())).cloned().collect()
        } else {
            opts.value_vars.clone()
        };

        for name in id_vars {
            if !self.columns.contains_key(name) {
                return Err(missing_column_error("dataframe::melt", name));
            }
        }
        for name in &value_vars {
            if !self.columns.contains_key(name) {
                return Err(missing_column_error("dataframe::melt", name));
            }
        }

        let id_set: FastHashSet<&str> = id_vars.iter().map(|s| s.as_str()).collect();
        for name in &value_vars {
            if id_set.contains(name.as_str()) {
                return Err(duplicate_name_error(
                    name,
                    id_vars.len() + value_vars.len(),
                    format!("column '{name}' appears in both id_vars and value_vars"),
                    "melt_name_in_both_id_and_value_vars",
                    "place each melt column in either id_vars or value_vars, not both",
                ));
            }
        }
        if id_set.contains(opts.var_name.as_str()) {
            return Err(duplicate_name_error(
                &opts.var_name,
                id_vars.len() + 2,
                format!("var_name '{}' collides with an id_var column", opts.var_name),
                "melt_var_name_conflicts_with_id_var",
                "choose a var_name that does not collide with id_vars",
            ));
        }
        if id_set.contains(opts.value_name.as_str()) {
            return Err(duplicate_name_error(
                &opts.value_name,
                id_vars.len() + 2,
                format!("value_name '{}' collides with an id_var column", opts.value_name),
                "melt_value_name_conflicts_with_id_var",
                "choose a value_name that does not collide with id_vars",
            ));
        }
        if opts.var_name == opts.value_name {
            return Err(duplicate_name_error(
                &opts.var_name,
                2,
                format!("var_name and value_name must be distinct (got '{}')", opts.var_name),
                "melt_output_names_conflict",
                "choose distinct var_name and value_name",
            ));
        }

        let n = self.row_count;
        let k = value_vars.len();
        let mut result_columns: Vec<(&str, Series)> = Vec::new();

        for id_name in id_vars {
            let id_series = self.series(id_name)?;
            if k == 0 {
                let empty = Series::from_column(Column::empty_in_domain(
                    id_series.dtype()?.id(),
                    Arc::clone(&self.domain),
                    "dataframe::melt",
                )?)?;
                result_columns.push((id_name, empty));
            } else if k == 1 {
                result_columns.push((id_name, id_series));
            } else {
                let copies: Vec<&Series> = (0..k).map(|_| &id_series).collect();
                let tiled = Series::concat(&copies)?;
                result_columns.push((id_name, tiled));
            }
        }

        let var_strings: Vec<&str> = value_vars.iter().flat_map(|name| std::iter::repeat_n(name.as_str(), n)).collect();
        let var_series = Series::from_column(Column::from_strs_in_domain(&var_strings, &self.domain)?)?;
        result_columns.push((&opts.var_name, var_series));

        if value_vars.is_empty() {
            let empty_series = Series::from_column(Column::empty_in_domain(
                crate::TypeId::Int32,
                Arc::clone(&self.domain),
                "dataframe::melt",
            )?)?;
            result_columns.push((&opts.value_name, empty_series));
        } else {
            let value_series_list: Vec<Series> =
                value_vars.iter().map(|name| self.series(name)).collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Series> = value_series_list.iter().collect();
            let concat_values = Series::concat(&refs)?;
            result_columns.push((&opts.value_name, concat_values));
        }

        DataFrame::from_columns(result_columns)
    }
}
