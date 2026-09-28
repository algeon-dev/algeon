use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Result,
    logical_expr::{LogicalPlan, UserDefinedLogicalNode, physical_planning_context::PhysicalPlanningContext},
    physical_plan::ExecutionPlan,
    physical_planner::{ExtensionPlanner, PhysicalPlanner},
};
#[cfg(feature = "cugraph")]
use datafusion::{common::DataFusionError, execution::SessionState};

use super::CugraphLogicalNode;
#[cfg(not(feature = "cugraph"))]
use super::planning::CugraphFeatureDisabledExec;
#[cfg(feature = "cugraph")]
use super::{
    CugraphAlgoExec,
    planning::{literal_bfs_source_plan, validate_edge_source_schema},
};
#[cfg(feature = "cugraph")]
use crate::session::{planning_native_execution_config, session_state_with_native_execution_config};

#[derive(Debug)]
pub struct CugraphExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for CugraphExtensionPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<CugraphLogicalNode>() else {
            return Ok(None);
        };

        #[cfg(not(feature = "cugraph"))]
        {
            let _ = (planner, logical_inputs, session);
            let exec = CugraphFeatureDisabledExec::new(node.spec().algorithm, node.output_schema(), None, None)?;
            return Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>));
        }

        #[cfg(feature = "cugraph")]
        {
            let session_state = session.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
                DataFusionError::Plan("cuGraph extension planning requires a DataFusion SessionState".to_string())
            })?;
            let native_execution_config = planning_native_execution_config(session);
            let planning_state =
                session_state_with_native_execution_config(session_state, native_execution_config.clone())
                    .unwrap_or_else(|| session_state.clone());
            let mut children = Vec::with_capacity(logical_inputs.len());
            for input in logical_inputs {
                children.push(planner.create_physical_plan(input, &planning_state).await?);
            }
            let edge_source = children.remove(0);
            let vertex_domain = validate_edge_source_schema(node.spec(), &edge_source.schema())?;
            let mut predicate_sources = Vec::with_capacity(children.len() + 1);
            if let Some(source) = literal_bfs_source_plan(&planning_state, node.spec(), &vertex_domain).await? {
                predicate_sources.push(source);
            }
            predicate_sources.extend(children);
            let exec = CugraphAlgoExec::try_new_with_reserved_gpu_result_input_name(
                node.spec().clone(),
                edge_source,
                predicate_sources,
                None,
                None,
                native_execution_config,
                node.gpu_result_input_name().to_string(),
            )?;
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}
