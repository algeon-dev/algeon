use std::sync::Arc;

use datafusion::{
    physical_expr::{PhysicalExpr, PhysicalSortExpr},
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use super::lowering::{GpuAggregate, lower_aggregate_exec};
use crate::error::Result as AdapterResult;

#[derive(Debug, Clone)]
pub(super) struct NormalizedGpuIslandPlan {
    pub(super) input: Arc<dyn ExecutionPlan>,
    pub(super) filter: Option<Arc<dyn PhysicalExpr>>,
    pub(super) projection: Vec<ProjectionExpr>,
    pub(super) aggregate: Option<GpuAggregate>,
    pub(super) sort: Vec<PhysicalSortExpr>,
    pub(super) local_limit: Option<usize>,
}

pub(super) fn normalize_gpu_island_plan(
    plan: Arc<dyn ExecutionPlan>,
) -> AdapterResult<Option<NormalizedGpuIslandPlan>> {
    let mut input = Arc::clone(&plan);
    let mut filter = None;
    let mut projection = Vec::new();
    let mut aggregate = None;
    let mut sort = Vec::new();
    let mut local_limit = None;
    let mut operator_count = 0usize;

    if let Some(limit) = input.as_any().downcast_ref::<LocalLimitExec>() {
        local_limit = Some(limit.fetch());
        input = Arc::clone(limit.input());
        operator_count += 1;
    }

    if let Some(sort_exec) = input.as_any().downcast_ref::<SortExec>() {
        sort = sort_exec.expr().to_vec();
        if local_limit.is_none() {
            local_limit = sort_exec.fetch();
        }
        input = Arc::clone(sort_exec.input());
        operator_count += 1;
    }

    if let Some(aggregate_exec) = input.as_any().downcast_ref::<AggregateExec>() {
        aggregate = Some(lower_aggregate_exec(aggregate_exec)?);
        input = Arc::clone(aggregate_exec.input());
        operator_count += 1;
    }

    if let Some(projection_exec) = input.as_any().downcast_ref::<ProjectionExec>() {
        projection = projection_exec.expr().to_vec();
        input = Arc::clone(projection_exec.input());
        operator_count += 1;
    }

    if let Some(filter_exec) = input.as_any().downcast_ref::<FilterExec>() {
        filter = Some(Arc::clone(filter_exec.predicate()));
        input = Arc::clone(filter_exec.input());
        operator_count += 1;
    }

    if operator_count == 0 && input.as_any().downcast_ref::<datafusion::datasource::source::DataSourceExec>().is_none()
    {
        return Ok(None);
    }

    Ok(Some(NormalizedGpuIslandPlan { input, filter, projection, aggregate, sort, local_limit }))
}
