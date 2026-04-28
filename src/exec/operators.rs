use std::sync::Arc;

use cudf::{AggSpec, ExecutionOverrides, SortOptions as CudfSortOptions};
use datafusion::{
    common::Result as DataFusionResult,
    physical_expr::{PhysicalExpr, PhysicalSortExpr},
    physical_plan::projection::ProjectionExpr,
};

use super::{
    batch::CudfBatch,
    lowering::{GpuAggregate, lower_expr, lower_sort_exprs},
    metrics::CudfGpuIslandMetrics,
};
use crate::error::Error;

#[derive(Clone)]
pub(super) struct CudfOperatorChain {
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    aggregate: Option<GpuAggregate>,
    sort: Vec<PhysicalSortExpr>,
    pub(super) local_limit: Option<usize>,
}

impl CudfOperatorChain {
    pub(super) fn new(
        filter: Option<Arc<dyn PhysicalExpr>>,
        projection: Vec<ProjectionExpr>,
        aggregate: Option<GpuAggregate>,
        sort: Vec<PhysicalSortExpr>,
        local_limit: Option<usize>,
    ) -> Self {
        Self { filter, projection, aggregate, sort, local_limit }
    }

    pub(super) fn requires_single_output_batch(&self) -> bool {
        self.aggregate.is_some() || !self.sort.is_empty()
    }

    pub(super) fn execute(
        &self,
        ctx: &ExecutionOverrides<'_>,
        remaining_rows: Option<&mut usize>,
        batch: CudfBatch,
        metrics: &CudfGpuIslandMetrics,
    ) -> DataFusionResult<CudfBatch> {
        let mut frame = batch.into_frame();

        if let Some(filter) = &self.filter {
            let predicate = lower_expr(filter)?;
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.with_execution(ctx).filter_expr(predicate).map_err(|error| {
                Error::from_cudf("datafusion_cudf.filter.execute", "failed to execute cudf filter expression", error)
            })?;
            operator_timer.done();
        }

        if !self.projection.is_empty() {
            let mut exprs = Vec::with_capacity(self.projection.len());
            let mut output_names = Vec::with_capacity(self.projection.len());
            for projection_expr in &self.projection {
                let expr = lower_expr(&projection_expr.expr)?;
                exprs.push((projection_expr.alias.as_str(), expr));
                output_names.push(projection_expr.alias.as_str());
            }
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.with_execution(ctx).with_columns(exprs).map_err(|error| {
                Error::from_cudf(
                    "datafusion_cudf.projection.execute",
                    "failed to execute cudf projection expressions",
                    error,
                )
            })?;
            frame = frame.select(output_names).map_err(|error| {
                Error::from_cudf(
                    "datafusion_cudf.projection.select",
                    "failed to select cudf projection output columns",
                    error,
                )
            })?;
            operator_timer.done();
        }

        if let Some(aggregate) = &self.aggregate {
            let keys = aggregate.keys.iter().map(|key| key.input_column.as_str()).collect::<Vec<_>>();
            let specs = aggregate
                .aggs
                .iter()
                .map(|agg| {
                    AggSpec::new(agg.input_column.as_str(), agg.agg.clone()).with_alias(agg.output_name.as_str())
                })
                .collect::<Vec<_>>();
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.groupby(keys).with_execution(ctx).agg(&specs).map_err(|error| {
                Error::from_cudf(
                    "datafusion_cudf.aggregate.execute",
                    "failed to execute cudf groupby aggregate inside GPU island",
                    error,
                )
            })?;
            let key_renames = aggregate
                .keys
                .iter()
                .filter(|key| key.input_column != key.output_name)
                .map(|key| (key.input_column.as_str(), key.output_name.as_str()))
                .collect::<Vec<_>>();
            if !key_renames.is_empty() {
                frame = frame.rename_columns(key_renames).map_err(|error| {
                    Error::from_cudf(
                        "datafusion_cudf.aggregate.rename_keys",
                        "failed to rename cudf groupby key columns to DataFusion aggregate aliases",
                        error,
                    )
                })?;
            }
            operator_timer.done();
        }

        if !self.sort.is_empty() {
            let sort_plan = lower_sort_exprs(&self.sort)?;
            let sort_options =
                CudfSortOptions::new().with_descending(sort_plan.descending).with_nulls_first(sort_plan.nulls_first);
            let operator_timer = metrics.cudf_operator_time.timer();
            frame = frame.with_execution(ctx).sort_values(sort_plan.columns, sort_options).map_err(|error| {
                Error::from_cudf("datafusion_cudf.sort.execute", "failed to execute cudf sort inside GPU island", error)
            })?;
            operator_timer.done();
        }

        if let Some(remaining_rows) = remaining_rows {
            let output_rows = frame.num_rows();
            let rows_to_take = output_rows.min(*remaining_rows);
            if rows_to_take < output_rows {
                let operator_timer = metrics.cudf_operator_time.timer();
                frame = frame.with_execution(ctx).slice(0, rows_to_take).map_err(|error| {
                    Error::from_cudf(
                        "datafusion_cudf.limit.execute",
                        "failed to execute cudf local limit inside GPU island",
                        error,
                    )
                })?;
                operator_timer.done();
            }
            *remaining_rows -= rows_to_take;
        }

        Ok(CudfBatch::new(frame))
    }
}
