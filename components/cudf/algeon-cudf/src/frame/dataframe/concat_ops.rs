use std::sync::Arc;

use super::owned::DataFrame;
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::Runtime;
use crate::{
    core::{
        column::Column,
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_COLUMN_INDEX, FACT_EXPECTED_COUNT, FACT_LEFT_TYPE_ID,
            FACT_RIGHT_TYPE_ID, Result,
        },
        table::TableView,
        types::DataType,
    },
    frame::series::Series,
    hash::{FastHashMap, FastHashSet, fast_hash_map_with_capacity},
    internal::resolve::missing_column_error,
    ops,
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaAlignment {
    Exact,
    NullFill,
}

#[derive(Debug, Clone)]
pub struct ConcatByNameOptions {
    pub(super) alignment: SchemaAlignment,
}

impl Default for ConcatByNameOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcatByNameOptions {
    pub fn new() -> Self {
        Self { alignment: SchemaAlignment::NullFill }
    }

    pub fn alignment(mut self, mode: SchemaAlignment) -> Self {
        self.alignment = mode;
        self
    }
}

fn concat_by_name_output_order(self_df: &DataFrame, others: &[&DataFrame]) -> Vec<String> {
    let mut seen: FastHashSet<String> = FastHashSet::default();
    let mut ordered = Vec::new();

    for name in self_df.column_names() {
        if seen.insert(name.to_string()) {
            ordered.push(name.to_string());
        }
    }

    for frame in others {
        for name in frame.column_names() {
            if !seen.contains(name) {
                seen.insert(name.to_string());
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
            match frame.get(name).map_err(|e| e.reanchor(operation))? {
                Some(series) => {
                    let dtype = series.dtype().map_err(|e| e.reanchor(operation))?;
                    if let Some(existing) = dtypes.get(name) {
                        if *existing != dtype {
                            return Err(Error::new_with(
                                ErrorCode::TypeMismatch,
                                operation,
                                format!("column '{name}' has incompatible dtypes for concat_by_name"),
                            )
                            .with_column(name.clone())
                            .with_fact(FACT_LEFT_TYPE_ID, existing.id())
                            .with_fact(FACT_RIGHT_TYPE_ID, dtype.id())
                            .with_detail("schema_alignment", format!("{alignment:?}").to_lowercase()));
                        }
                    } else {
                        dtypes.insert(name.clone(), dtype);
                    }
                },
                None if alignment == SchemaAlignment::Exact => {
                    return Err(missing_column_error(operation, name)
                        .with_detail("schema_alignment", format!("{alignment:?}").to_lowercase()));
                },
                None => {},
            }
        }
    }

    Ok(dtypes)
}

fn concat_by_name_missing_planned_dtype_error(operation: &'static str, column_name: &str) -> Error {
    error::invariant_violation(operation, "aligned concat_by_name is missing a planned dtype")
        .with_column(column_name.to_string())
        .with_detail("resource", "concat_by_name_schema")
        .with_detail("reason", "missing_planned_dtype")
        .with_detail("remedy", "ensure concat_by_name schema planning records a dtype for every output column")
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_concat_by_name_missing_planned_dtype_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let df = DataFrame::from_series([("left", runtime.series(&[1i32])?)])?;
    let output_names = vec!["missing".to_string()];
    let dtypes = FastHashMap::default();
    df.align_for_concat_by_name(&output_names, &dtypes, SchemaAlignment::NullFill, "dataframe::concat_many_by_name")
        .map(|_| ())
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
            if let Some(series) = self.get(name).map_err(|e| e.reanchor(operation))? {
                columns.push((name.clone(), series));
                continue;
            }

            if alignment == SchemaAlignment::Exact {
                return Err(missing_column_error(operation, name)
                    .with_detail("schema_alignment", format!("{alignment:?}").to_lowercase()));
            }

            let dtype = dtypes.get(name).ok_or_else(|| concat_by_name_missing_planned_dtype_error(operation, name))?;
            let nulls = Column::nulls_in_domain(*dtype, self.row_count, &self.domain, operation)
                .map_err(|e| e.reanchor(operation))?;
            columns.push((name.clone(), Series::from_column(nulls).map_err(|e| e.reanchor(operation))?));
        }

        DataFrame::from_series(columns).map_err(|e| e.reanchor(operation))
    }

    pub(crate) fn concat_many_by_name_in_context(
        &self,
        others: &[&DataFrame],
        options: ConcatByNameOptions,
        ctx: Exec<'_>,
    ) -> Result<Self> {
        let operation = "dataframe::concat_many_by_name";
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let output_names = concat_by_name_output_order(self, others);
        if others.is_empty() || output_names.is_empty() {
            return self.concat_many_impl(others, output_domain, operation);
        }

        let dtypes = concat_by_name_schema(self, others, &output_names, options.alignment, operation)?;
        let aligned_self = self.align_for_concat_by_name(&output_names, &dtypes, options.alignment, operation)?;
        let aligned_others = others
            .iter()
            .map(|df| df.align_for_concat_by_name(&output_names, &dtypes, options.alignment, operation))
            .collect::<Result<Vec<_>>>()?;
        let aligned_other_refs: Vec<&DataFrame> = aligned_others.iter().collect();

        aligned_self.concat_many_impl(&aligned_other_refs, output_domain, operation)
    }

    /// Concatenate two DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_in_context(&self, other: &DataFrame, ctx: Exec<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.concat_many_impl(&[other], output_domain, "dataframe::concat")
    }

    /// Concatenate multiple DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_many_in_context(&self, others: &[&DataFrame], ctx: Exec<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let operation = "dataframe::concat_many";
        self.concat_many_impl(others, output_domain, operation)
    }

    fn concat_many_impl(
        &self,
        others: &[&DataFrame],
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if others.is_empty() {
            return self.in_domain(&output_domain, operation);
        }

        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            std::iter::once(&self.domain).chain(others.iter().map(|other| &other.domain)),
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        for other in others {
            self.validate_concat_schema(other, operation)?;
        }

        let self_tv = self.build_table_view()?;
        let other_tvs: Vec<_> = others.iter().map(|df| df.build_table_view()).collect::<Result<_>>()?;

        let mut table_refs: Vec<&TableView<'_>> = Vec::with_capacity(1 + others.len());
        table_refs.push(&self_tv);
        for tv in &other_tvs {
            table_refs.push(tv);
        }

        let result_table =
            ops::concatenate::concatenate_tables_with_exec(&table_refs, &output_domain, execution.exec_options())?;

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    pub(crate) fn validate_concat_schema(&self, other: &DataFrame, operation: &'static str) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                operation,
                "cannot concatenate DataFrames with different column counts",
            )
            .with_fact(FACT_EXPECTED_COUNT, self.columns.len())
            .with_fact(FACT_ACTUAL_COUNT, other.columns.len())
            .with_detail("left_columns", self.columns.len().to_string())
            .with_detail("right_columns", other.columns.len().to_string()));
        }

        for (left_name, right_name) in self.columns.keys().zip(other.columns.keys()) {
            if left_name != right_name {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    operation,
                    format!("schema mismatch: column '{}' vs '{}'", left_name, right_name),
                )
                .with_detail("left_column", left_name.clone())
                .with_detail("right_column", right_name.clone()));
            }
        }

        for (i, ((name, left_col), (_, right_col))) in self.columns.iter().zip(other.columns.iter()).enumerate() {
            let left_type = left_col.view()?.data_type()?.id();
            let right_type = right_col.view()?.data_type()?.id();
            if left_type != right_type {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "cannot concatenate DataFrames with mismatched column types",
                )
                .with_fact(FACT_COLUMN_INDEX, i)
                .with_column(name.to_string())
                .with_fact(FACT_LEFT_TYPE_ID, left_type)
                .with_fact(FACT_RIGHT_TYPE_ID, right_type));
            }
        }

        Ok(())
    }
}
