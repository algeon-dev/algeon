use std::sync::Arc;

use super::owned::DataFrame;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
    },
    frame::series::Series,
    internal::{hash::fast_index_map_with_capacity, resolve::missing_column_error},
    ops::{self, copying, filter},
    runtime::{execution_context::ExecutionOverrides, execution_domain::ExecutionDomain},
};

impl DataFrame {
    /// Gather rows using a borrowed per-operation execution override.
    pub(crate) fn gather_in_context(&self, indices: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let cv = indices.view()?;
        self.gather_by_column_view_impl(
            &cv,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::with_execution::gather",
        )
    }
    pub(crate) fn __debug_gather_in_context(&self, indices: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        self.gather_in_context(indices, ctx)
    }

    /// Filter rows using a borrowed per-operation execution override.
    pub(crate) fn filter_in_context(&self, mask: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::with_execution::filter",
        )
    }
    pub(crate) fn __debug_filter_in_context(&self, mask: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        self.filter_in_context(mask, ctx)
    }

    pub(crate) fn head_in_context(&self, n: usize, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range_in_context(0, take, ctx, "dataframe::with_execution::head")
    }

    pub(crate) fn tail_in_context(&self, n: usize, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_in_context(start, take, ctx, "dataframe::with_execution::tail")
    }

    pub(crate) fn slice_in_context(&self, offset: usize, length: usize, ctx: &ExecutionOverrides<'_>) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_in_context(start, take, ctx, "dataframe::with_execution::slice")
    }

    /// Return the first `n` rows.
    pub fn head(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range(0, take)
    }

    /// Return the last `n` rows.
    pub fn tail(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range(start, take)
    }

    /// Return `length` rows starting at `offset`.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range(start, take)
    }

    /// Gather (select) rows by an index `Series`.
    pub fn gather(&self, indices: &Series) -> Result<Self> {
        let cv = indices.view()?;
        let output_domain = Arc::clone(&self.domain);
        self.gather_by_column_view_impl(
            &cv,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather",
        )
    }

    /// Filter rows using a precomputed `Series` boolean mask.
    pub fn filter(&self, mask: &Series) -> Result<Self> {
        let mask_view = self.validate_series_mask(mask)?;
        let output_domain = Arc::clone(&self.domain);
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::filter",
        )
    }

    /// Filter rows to those where `column` values appear in a `Series`.
    pub fn filter_isin(&self, column: &str, values: &Series) -> Result<Self> {
        let operation = "dataframe::filter_isin";
        let cv = values.view()?;
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;
        let col_view = col.view()?;
        let output_domain = Arc::clone(&self.domain);
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, cv.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(cv.execution_domain(), &output_domain) {
            cv.execution_domain().fence_stream(operation)?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let mask = ops::search::contains_impl(&cv, &col_view, &output_domain, exec_options.as_ref(), operation)?;
        let mask_view = mask.view()?;
        self.filter_by_mask_impl(&mask_view, output_domain.explicit_exec_options_raw(), output_domain, operation)
    }

    fn validate_series_mask<'a>(&self, mask: &'a Series) -> Result<ColumnView<'a>> {
        let mask_view = mask.view()?;
        self.validate_mask_view(&mask_view, "dataframe::filter")?;
        Ok(mask_view)
    }

    fn validate_mask_view(&self, mask: &ColumnView<'_>, operation: &'static str) -> Result<()> {
        let mask_dtype = mask.data_type()?;
        if mask_dtype.id() != crate::core::types::TypeId::Bool8 {
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
}

impl DataFrame {
    fn gather_by_column_view_impl(
        &self,
        indices: &ColumnView<'_>,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, indices.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(indices.execution_domain(), &output_domain) {
            indices.execution_domain().fence_stream(operation)?;
        }

        if indices.is_empty()? {
            return self.empty_with_same_schema_in_domain(output_domain);
        }
        let tv = self.build_table_view()?;
        let result_table = copying::gather_impl(
            &tv,
            indices,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn filter_by_mask_impl(
        &self,
        mask: &ColumnView<'_>,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, mask.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(mask.execution_domain(), &output_domain) {
            mask.execution_domain().fence_stream(operation)?;
        }

        self.validate_mask_view(mask, operation)?;
        let tv = self.build_table_view()?;
        let result_table =
            filter::apply_boolean_mask_impl(&tv, mask, &output_domain, exec_options.as_ref(), operation)?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn gather_range(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema();
        }

        let output_domain = Arc::clone(&self.domain);
        self.gather_range_impl(
            start,
            len,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather_range",
        )
    }

    fn gather_range_in_context(
        &self,
        start: usize,
        len: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.gather_range_impl(start, len, output_domain.explicit_exec_options_raw(), output_domain, operation)
    }

    fn gather_range_impl(
        &self,
        start: usize,
        len: usize,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema_in_domain(output_domain);
        }

        let end = start + len;
        let start_i32: i32 = start.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "gather_range: start exceeds i32::MAX").with_operation(operation)
        })?;
        let end_i32: i32 = end.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "gather_range: start + len exceeds i32::MAX")
                .with_operation(operation)
        })?;
        let indices: Vec<i32> = (start_i32..end_i32).collect();
        let map = Column::from_slice_in_domain(&indices, &output_domain)?;
        let map_view = map.view()?;
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, map_view.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }

        let tv = self.build_table_view()?;
        let result_table = copying::gather_impl(
            &tv,
            &map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    pub(crate) fn empty_with_same_schema(&self) -> Result<Self> {
        self.empty_with_same_schema_in_domain(Arc::clone(&self.domain))
    }

    pub(crate) fn empty_with_same_schema_in_domain(&self, domain: Arc<ExecutionDomain>) -> Result<Self> {
        let mut new_columns = fast_index_map_with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let cv = col.view()?;
            let dtype = cv.data_type()?;
            new_columns.insert(
                name.clone(),
                Arc::new(Column::empty_with_dtype_in_domain(
                    dtype,
                    Arc::clone(&domain),
                    "dataframe::empty_with_same_schema_in_domain",
                )?),
            );
        }
        Ok(Self { columns: new_columns, row_count: 0, domain })
    }
}
