use std::sync::Arc;

use super::owned::DataFrame;
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::Runtime;
use crate::{
    core::{column::ColumnView, error::Result, row_range},
    frame::series::Series,
    internal::resolve::missing_column_error,
    ops::{self, copying, filter},
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

#[cfg(any(test, feature = "test-hooks"))]
fn range_testing_dataframe() -> Result<DataFrame> {
    let runtime = Runtime::new()?;
    let series = runtime.series(&[1i32, 2, 3])?;
    DataFrame::from_series([("x", series)])
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_dataframe_row_range_start_limit_error_for_test() -> Result<DataFrame> {
    range_testing_dataframe()?.__debug_gather_range_for_test(usize::MAX, 1, "dataframe::gather_range")
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_dataframe_row_range_end_limit_error_for_test() -> Result<DataFrame> {
    range_testing_dataframe()?.__debug_gather_range_for_test(i32::MAX as usize, 1, "dataframe::gather_range")
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_dataframe_row_range_overflow_error_for_test() -> Result<DataFrame> {
    range_testing_dataframe()?.__debug_gather_range_for_test(i32::MAX as usize, usize::MAX, "dataframe::gather_range")
}

impl DataFrame {
    /// Gather rows using a borrowed per-operation execution override.
    pub(crate) fn gather_in_context(&self, indices: &Series, ctx: Exec<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let cv = indices.view()?;
        self.gather_by_column_view_impl(&cv, output_domain, "dataframe::gather")
    }

    pub(crate) fn gather_device_allocated_size_in_context(
        &self,
        indices: &Series,
        ctx: Exec<'_>,
        operation: &'static str,
        reservation_contract: &'static str,
    ) -> Result<usize> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let cv = indices.view()?;
        self.gather_device_allocated_size_by_column_view_impl(&cv, output_domain, operation, reservation_contract)
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_gather_in_context(&self, indices: &Series, ctx: Exec<'_>) -> Result<Self> {
        self.gather_in_context(indices, ctx)
    }

    /// Filter rows using a borrowed per-operation execution override.
    pub(crate) fn filter_in_context(&self, mask: &Series, ctx: Exec<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask_impl(&mask_view, output_domain, "dataframe::filter")
    }
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_filter_in_context(&self, mask: &Series, ctx: Exec<'_>) -> Result<Self> {
        self.filter_in_context(mask, ctx)
    }

    pub(crate) fn head_in_context(&self, n: usize, ctx: Exec<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range_in_context(0, take, ctx, "dataframe::head")
    }

    pub(crate) fn tail_in_context(&self, n: usize, ctx: Exec<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_in_context(start, take, ctx, "dataframe::tail")
    }

    pub(crate) fn slice_in_context(&self, offset: usize, length: usize, ctx: Exec<'_>) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_in_context(start, take, ctx, "dataframe::slice")
    }

    pub(crate) fn filter_isin_in_context(&self, column: &str, values: &Series, ctx: Exec<'_>) -> Result<Self> {
        let operation = "dataframe::filter_isin";
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.filter_isin_with_domain(column, values, output_domain, operation)
    }

    fn filter_isin_with_domain(
        &self,
        column: &str,
        values: &Series,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let cv = values.view()?;
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;
        let col_view = col.view()?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [cv.execution_domain(), col_view.execution_domain()],
            operation,
        )?;
        let mask =
            ops::search::contains_impl(&cv, &col_view, execution.output_domain(), execution.exec_options(), operation)?;
        let mask_view = mask.view()?;
        self.filter_by_mask_impl(&mask_view, output_domain, operation)
    }

    fn validate_series_mask<'a>(&self, mask: &'a Series) -> Result<ColumnView<'a>> {
        let mask_view = mask.view()?;
        self.validate_mask_view(&mask_view, "dataframe::filter")?;
        Ok(mask_view)
    }

    fn validate_mask_view(&self, mask: &ColumnView<'_>, operation: &'static str) -> Result<()> {
        filter::validate_boolean_mask(mask, self.row_count, "DataFrame row count", operation)
    }
}

impl DataFrame {
    fn gather_by_column_view_impl(
        &self,
        indices: &ColumnView<'_>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.domain, indices.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        let tv = self.build_table_view()?;
        let result_table = copying::gather_impl(
            &tv,
            indices,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            &output_domain,
            execution.exec_options(),
            operation,
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn gather_device_allocated_size_by_column_view_impl(
        &self,
        indices: &ColumnView<'_>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
        reservation_contract: &'static str,
    ) -> Result<usize> {
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.domain, indices.execution_domain()],
            operation,
        )?;

        if indices.is_empty()? {
            return Ok(0);
        }

        let tv = self.build_table_view()?;
        copying::gather_device_allocated_size_impl(
            &tv,
            indices,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            execution.exec_options(),
            operation,
            reservation_contract,
        )
    }

    fn filter_by_mask_impl(
        &self,
        mask: &ColumnView<'_>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.domain, mask.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        self.validate_mask_view(mask, operation)?;
        let tv = self.build_table_view()?;
        let result_table =
            filter::apply_boolean_mask_impl(&tv, mask, &output_domain, execution.exec_options(), operation)?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_gather_range_for_test(
        &self,
        start: usize,
        len: usize,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = Arc::clone(&self.domain);
        self.gather_range_impl(start, len, output_domain, operation)
    }

    fn gather_range_in_context(
        &self,
        start: usize,
        len: usize,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.gather_range_impl(start, len, output_domain, operation)
    }

    fn gather_range_impl(
        &self,
        start: usize,
        len: usize,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let map = row_range::gather_map_in_domain(start, len, &output_domain, operation)?;
        let map_view = map.view()?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.domain, map_view.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        let tv = self.build_table_view()?;
        let result_table = copying::gather_impl(
            &tv,
            &map_view,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            &output_domain,
            execution.exec_options(),
            operation,
        )?;

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    pub(crate) fn empty_with_same_schema_in_domain(&self, domain: Arc<ExecutionDomain>) -> Result<Self> {
        self.gather_range_impl(0, 0, domain, "dataframe::empty_with_same_schema_in_domain")
    }
}
