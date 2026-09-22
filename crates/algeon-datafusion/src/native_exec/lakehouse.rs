//! Async bridge between DataFusion task context and native lakehouse pruning.

use std::sync::Arc;

use datafusion::execution::context::TaskContext;
use query_engine::{
    admission::AdmittedPlan, plan::QueryPlan, rapids_backend::rapids_analysis_context, runtime::SourceStorageProfile,
};

use super::config::AlgeonNativeExecutionConfig;
use crate::{
    error::{Error, Result},
    native::lakehouse_pruning::prune_native_query_plan,
};

pub(crate) async fn resolve_query_plan(
    execution_config: &AlgeonNativeExecutionConfig,
    plan: QueryPlan,
    task_context: &TaskContext,
) -> QueryPlan {
    if !execution_config.lakehouse_footer_pruning() {
        return plan;
    }
    let runtime_env = task_context.runtime_env();
    prune_native_query_plan(plan, &runtime_env, execution_config.source_storage_profiles()).await
}

/// The artifact a native node executes: the one decided at planning, or, with
/// footer pruning on, that artifact admitted again for the resolved plan.
pub(crate) async fn resolve_admitted_plan(
    lakehouse_footer_pruning: bool,
    source_storage_profiles: &[SourceStorageProfile],
    admitted: &Arc<AdmittedPlan>,
    task_context: &TaskContext,
) -> Result<Arc<AdmittedPlan>> {
    if !lakehouse_footer_pruning {
        return Ok(Arc::clone(admitted));
    }
    let runtime_env = task_context.runtime_env();
    let resolved = prune_native_query_plan(admitted.plan().clone(), &runtime_env, source_storage_profiles).await;
    admitted.with_pruned_sources(resolved, rapids_analysis_context()).map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native_exec.footer_pruning",
            "the footer-pruned native plan does not refine its admitted plan",
            error,
        )
    })
}
