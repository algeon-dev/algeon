use std::{collections::BTreeMap, fmt, sync::Arc};

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
use crate::{native_exec::AlgeonNativeExecutionConfig, session::installed_algeon_native_optimizer_config};

#[derive(Clone)]
pub struct AlgeonQueryPlanner {
    caller_extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
    family_extension_planners: BTreeMap<&'static str, Arc<dyn ExtensionPlanner + Send + Sync>>,
}

impl AlgeonQueryPlanner {
    /// Builds the Algeon planner from caller-owned extension planners.
    pub fn new(caller_extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>) -> Self {
        Self { caller_extension_planners, family_extension_planners: BTreeMap::new() }
    }

    pub(crate) fn with_family(
        mut self,
        family_id: &'static str,
        planner: Arc<dyn ExtensionPlanner + Send + Sync>,
    ) -> Self {
        self.family_extension_planners.entry(family_id).or_insert(planner);
        self
    }

    pub(crate) fn contains_family(&self, family_id: &str) -> bool {
        self.family_extension_planners.contains_key(family_id)
    }

    fn extension_planners(&self) -> Vec<Arc<dyn ExtensionPlanner + Send + Sync>> {
        self.caller_extension_planners.iter().chain(self.family_extension_planners.values()).cloned().collect()
    }
}

impl Default for AlgeonQueryPlanner {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl fmt::Debug for AlgeonQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlgeonQueryPlanner")
            .field("caller_extension_planner_count", &self.caller_extension_planners.len())
            .field("family_ids", &self.family_extension_planners.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[async_trait]
impl QueryPlanner for AlgeonQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(self.extension_planners())
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
        let native_execution_config = installed_algeon_native_optimizer_config(session)
            .map(|config| config.native_execution_config().clone())
            .unwrap_or_else(AlgeonNativeExecutionConfig::bounded_default);
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
