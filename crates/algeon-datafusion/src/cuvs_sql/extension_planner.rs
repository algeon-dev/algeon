use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Result,
    logical_expr::{LogicalPlan, UserDefinedLogicalNode, physical_planning_context::PhysicalPlanningContext},
    physical_expr::{LexOrdering, create_physical_sort_exprs},
    physical_plan::ExecutionPlan,
    physical_planner::{ExtensionPlanner, PhysicalPlanner},
};

use super::{CuvsAlgoExec, CuvsLogicalNode};
use crate::{native_exec::AlgeonNativeExecutionConfig, session::installed_algeon_native_optimizer_config};

#[derive(Debug)]
pub struct CuvsExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for CuvsExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<CuvsLogicalNode>() else {
            return Ok(None);
        };
        let native_execution_config = installed_algeon_native_optimizer_config(session)
            .map(|config| config.native_execution_config().clone())
            .unwrap_or_else(AlgeonNativeExecutionConfig::bounded_default);
        let input_orderings = node
            .input_orderings()
            .iter()
            .zip(logical_inputs)
            .map(|(ordering, input)| {
                create_physical_sort_exprs(ordering, input.schema(), session.execution_props(), planning_ctx)
                    .map(LexOrdering::new)
            })
            .collect::<Result<Vec<_>>>()?;
        CuvsAlgoExec::try_new(
            node.operation(),
            node.validated_options().clone(),
            physical_inputs.to_vec(),
            input_orderings,
            node.output_schema(),
            native_execution_config,
        )
        .map(|exec| Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
    }
}
