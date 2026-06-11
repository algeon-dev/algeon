use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
        types::DataType,
    },
    frame::series::Series,
    internal::hash::{FastHashMap, FastHashSet, fast_hash_map_with_capacity},
    ops,
    runtime::{execution_context::ExecutionContext, execution_domain::ExecutionDomain},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnOrder {
    Left,
    FirstSeen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaAlignment {
    Exact,
    NullFill,
}

#[derive(Debug, Clone)]
pub struct ConcatByNameOptions {
    pub(super) alignment: SchemaAlignment,
    pub(super) column_order: ColumnOrder,
}

impl Default for ConcatByNameOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcatByNameOptions {
    pub fn new() -> Self {
        Self { alignment: SchemaAlignment::NullFill, column_order: ColumnOrder::Left }
    }

    pub fn alignment(mut self, mode: SchemaAlignment) -> Self {
        self.alignment = mode;
        self
    }

    pub fn column_order(mut self, order: ColumnOrder) -> Self {
        self.column_order = order;
        self
    }
}

fn concat_by_name_output_order(self_df: &DataFrame, others: &[&DataFrame], _order: ColumnOrder) -> Vec<String> {
    let mut seen: FastHashSet<String> = FastHashSet::default();
    let mut ordered = Vec::new();

    for frame in std::iter::once(self_df).chain(others.iter().copied()) {
        for name in frame.column_names() {
            if seen.insert(name.to_string()) {
                ordered.push(name.to_string());
            }
        }
    }

    ordered
}

fn concat_by_name_schema(
    self_df: &DataFrame,
    others: &[&DataFrame],
    output_names: &[String],
    alignment: SchemaAlignment,
    operation: &'static str,
) -> Result<FastHashMap<String, DataType>> {
    let mut dtypes: FastHashMap<String, DataType> = fast_hash_map_with_capacity(output_names.len());

    for frame in std::iter::once(self_df).chain(others.iter().copied()) {
        for name in output_names {
            match frame.get(name).map_err(|e| e.with_operation(operation))? {
                Some(series) => {
                    let dtype = series.dtype().map_err(|e| e.with_operation(operation))?;
                    if let Some(existing) = dtypes.get(name) {
                        if *existing != dtype {
                            return Err(Error::new(
                                ErrorKind::InvalidOperation,
                                format!("column '{name}' has incompatible dtypes for concat_by_name"),
                            )
                            .with_operation(operation)
                            .with_context("column_name", name.clone())
                            .with_context("left_type_id", format!("{:?}", existing.id()).to_lowercase())
                            .with_context("right_type_id", format!("{:?}", dtype.id()).to_lowercase())
                            .with_context("schema_alignment", format!("{alignment:?}").to_lowercase()));
                        }
                    } else {
                        dtypes.insert(name.clone(), dtype);
                    }
                },
                None if alignment == SchemaAlignment::Exact => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("column '{name}' is missing under exact concat_by_name alignment"),
                    )
                    .with_operation(operation)
                    .with_context("column_name", name.clone())
                    .with_context("schema_alignment", format!("{alignment:?}").to_lowercase()));
                },
                None => {},
            }
        }
    }

    Ok(dtypes)
}

impl DataFrame {
    fn align_for_concat_by_name(
        &self,
        output_names: &[String],
        dtypes: &FastHashMap<String, DataType>,
        alignment: SchemaAlignment,
        operation: &'static str,
    ) -> Result<Self> {
        let mut columns = Vec::with_capacity(output_names.len());

        for name in output_names {
            if let Some(series) = self.get(name).map_err(|e| e.with_operation(operation))? {
                columns.push((name.clone(), series));
                continue;
            }

            if alignment == SchemaAlignment::Exact {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column '{name}' is missing under exact concat_by_name alignment"),
                )
                .with_operation(operation)
                .with_context("column_name", name.clone())
                .with_context("schema_alignment", format!("{alignment:?}").to_lowercase()));
            }

            let dtype = dtypes.get(name).ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "aligned concat_by_name is missing a planned dtype")
                    .with_operation(operation)
                    .with_context("column_name", name.clone())
            })?;
            let nulls = Column::nulls_in_domain(dtype.id(), self.row_count, &self.domain, operation)
                .map_err(|e| e.with_operation(operation))?;
            columns.push((
                name.clone(),
                Series::from_column(nulls).map_err(|e| e.with_operation(operation))?.rename(name),
            ));
        }

        DataFrame::from_named_pairs(columns).map_err(|e| e.with_operation(operation))
    }

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
        self.concat_many_impl(&[other], None, None, "dataframe::concat")
    }

    /// Concatenate multiple DataFrames row-wise.
    ///
    /// All DataFrames must have the same column names in the same order
    /// and matching column types. An empty `others` slice returns a clone
    /// of `self`.
    ///
    /// This is more efficient than chaining multiple [`concat`](Self::concat)
    /// calls because it passes all table views to a single FFI call.
    ///
    /// # Errors
    ///
    /// Returns an error if any DataFrame's schema does not match `self`
    /// (different names, count, order, or types).
    pub fn concat_many(&self, others: &[&DataFrame]) -> Result<Self> {
        self.concat_many_impl(others, None, None, "dataframe::concat_many")
    }

    pub fn concat_by_name(&self, other: &DataFrame, options: ConcatByNameOptions) -> Result<Self> {
        self.concat_many_by_name(&[other], options)
    }

    pub fn concat_many_by_name(&self, others: &[&DataFrame], options: ConcatByNameOptions) -> Result<Self> {
        let operation = "dataframe::concat_many_by_name";
        if others.is_empty() {
            return Ok(self.clone());
        }

        let output_names = concat_by_name_output_order(self, others, options.column_order);
        if output_names.is_empty() {
            return self.concat_many_impl(others, None, None, operation);
        }

        let dtypes = concat_by_name_schema(self, others, &output_names, options.alignment, operation)?;
        let aligned_self = self.align_for_concat_by_name(&output_names, &dtypes, options.alignment, operation)?;
        let aligned_others = others
            .iter()
            .map(|df| df.align_for_concat_by_name(&output_names, &dtypes, options.alignment, operation))
            .collect::<Result<Vec<_>>>()?;
        let aligned_other_refs: Vec<&DataFrame> = aligned_others.iter().collect();

        aligned_self.concat_many_impl(&aligned_other_refs, None, None, operation)
    }

    /// Concatenate two DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_in_context(&self, other: &DataFrame, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.concat_many_in_context(&[other], ctx)
    }

    /// Concatenate multiple DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_many_in_context(&self, others: &[&DataFrame], ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let operation = "dataframe::in_context::concat_many";
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        for other in others {
            if !ExecutionDomain::same_provenance(&other.domain, &output_domain) {
                other.domain.fence_stream(operation)?;
            }
        }
        self.concat_many_impl(others, output_domain.explicit_exec_options_raw(), Some(output_domain), operation)
    }

    fn concat_many_impl(
        &self,
        others: &[&DataFrame],
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        if others.is_empty() {
            return Ok(self.clone());
        }

        let output_domain = forced_output_domain.unwrap_or_else(|| Arc::clone(&self.domain));
        let exec_options = exec_options.or_else(|| output_domain.explicit_exec_options_raw());

        for other in others {
            ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, &other.domain, operation)?;
        }
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        for other in others {
            if !ExecutionDomain::same_provenance(&other.domain, &output_domain) {
                other.domain.fence_stream(operation)?;
            }
        }

        for other in others {
            self.validate_concat_schema(other, operation)?;
        }

        let self_tv = self.build_table_view()?;
        let other_tvs: Vec<_> = others.iter().map(|df| df.build_table_view()).collect::<Result<_>>()?;

        let mut table_refs: Vec<&crate::core::table::TableView<'_>> = Vec::with_capacity(1 + others.len());
        table_refs.push(&self_tv);
        for tv in &other_tvs {
            table_refs.push(tv);
        }

        let result_table =
            ops::concatenate::concatenate_tables_with_exec(&table_refs, &output_domain, exec_options.as_ref())?;

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn validate_concat_schema(&self, other: &DataFrame, operation: &'static str) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot concatenate DataFrames with different column counts",
            )
            .with_operation(operation)
            .with_context("left_columns", self.columns.len().to_string())
            .with_context("right_columns", other.columns.len().to_string()));
        }

        for (left_name, right_name) in self.columns.keys().zip(other.columns.keys()) {
            if left_name != right_name {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("schema mismatch: column '{}' vs '{}'", left_name, right_name),
                )
                .with_operation(operation)
                .with_context("left_column", left_name.clone())
                .with_context("right_column", right_name.clone()));
            }
        }

        for (i, ((name, left_col), (_, right_col))) in self.columns.iter().zip(other.columns.iter()).enumerate() {
            let left_type = left_col.view()?.data_type()?.id();
            let right_type = right_col.view()?.data_type()?.id();
            if left_type != right_type {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cannot concatenate DataFrames with mismatched column types",
                )
                .with_operation(operation)
                .with_context("column_index", i.to_string())
                .with_context("column_name", name.to_string())
                .with_context("left_type_id", format!("{:?}", left_type).to_lowercase())
                .with_context("right_type_id", format!("{:?}", right_type).to_lowercase()));
            }
        }

        Ok(())
    }
}
