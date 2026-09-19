//! Async bridge between DataFusion task context and native lakehouse pruning.

use datafusion::execution::context::TaskContext;
use query_engine::plan::QueryPlan;

use super::config::AlgeonNativeExecutionConfig;

pub(crate) async fn resolve_query_plan(
    execution_config: &AlgeonNativeExecutionConfig,
    plan: QueryPlan,
    task_context: &TaskContext,
) -> QueryPlan {
    if !execution_config.lakehouse_footer_pruning() {
        return plan;
    }
    let runtime_env = task_context.runtime_env();
    crate::native::lakehouse_pruning::prune_native_query_plan(
        plan,
        &runtime_env,
        execution_config.source_storage_profiles(),
    )
    .await
}
