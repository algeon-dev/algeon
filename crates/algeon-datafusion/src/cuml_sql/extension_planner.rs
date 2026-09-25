use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Result,
    logical_expr::{LogicalPlan, UserDefinedLogicalNode, physical_planning_context::PhysicalPlanningContext},
    physical_plan::ExecutionPlan,
    physical_planner::{ExtensionPlanner, PhysicalPlanner},
};

use super::{CumlAlgoExec, CumlLogicalNode};
use crate::{native_exec::AlgeonNativeExecutionConfig, session::installed_algeon_native_optimizer_config};

#[derive(Debug)]
pub struct CumlExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for CumlExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<CumlLogicalNode>() else {
            return Ok(None);
        };
        let native_execution_config = installed_algeon_native_optimizer_config(session)
            .map(|config| config.native_execution_config().clone())
            .unwrap_or_else(AlgeonNativeExecutionConfig::bounded_default);
        CumlAlgoExec::try_new(
            node.operation(),
            node.validated_options().clone(),
            node.model().cloned(),
            physical_inputs.to_vec(),
            node.output_schema(),
            native_execution_config,
        )
        .map(|exec| Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
    }
}
