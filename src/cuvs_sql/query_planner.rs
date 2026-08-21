use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Result,
    execution::context::QueryPlanner,
    logical_expr::{LogicalPlan, UserDefinedLogicalNode, physical_planning_context::PhysicalPlanningContext},
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

use super::{CuvsAlgoExec, CuvsLogicalNode};
use crate::native_exec::NexusNativeExecutionConfig;

pub struct CuvsQueryPlanner {
    extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
}

impl CuvsQueryPlanner {
    /// Builds the Nexus planner from caller-compatible extension planners plus
    /// the cuVS physical extension. Caller extensions retain their order and
    /// run before cuVS recognition.
    pub fn new(extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>) -> Self {
        let mut extension_planners = extension_planners;
        extension_planners.push(Arc::new(CuvsExtensionPlanner));
        Self { extension_planners }
    }
}

impl Default for CuvsQueryPlanner {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl fmt::Debug for CuvsQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CuvsQueryPlanner").field("extension_planner_count", &self.extension_planners.len()).finish()
    }
}

#[async_trait]
impl QueryPlanner for CuvsQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(self.extension_planners.clone())
            .create_physical_plan(logical_plan, session)
            .await
    }
}

#[derive(Debug)]
pub struct CuvsExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for CuvsExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<CuvsLogicalNode>() else {
            return Ok(None);
        };
        // Retain the installed backend's admission source and policy even
        // though current cuVS operators fail closed until they gain a
        // bounded peak-memory contract. This prevents a future supported
        // cuVS path from silently escaping the selected ledger/domain.
        let native_execution_config = crate::session::installed_nexus_native_optimizer_config(session)
            .map(|config| config.native_execution_config().clone())
            .unwrap_or_else(NexusNativeExecutionConfig::bounded_default);
        CuvsAlgoExec::try_new(
            node.operation(),
            node.validated_options().clone(),
            physical_inputs.to_vec(),
            node.output_schema(),
            native_execution_config,
        )
        .map(|exec| Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
    }
}
