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

#[cfg(not(feature = "cugraph"))]
use super::planning::CugraphFeatureDisabledExec;
#[cfg(feature = "cugraph")]
use super::{
    CugraphAlgoExec,
    planning::{literal_bfs_source_plan, validate_edge_source_schema},
};
use super::{CugraphLogicalNode, GpuIslandHostBoundaryConfig};
#[cfg(feature = "cugraph")]
use crate::{
    planner::native::InputPlanningSink,
    report::PlanningReport,
    session::{planning_native_execution_config, session_state_with_native_execution_config},
};

#[derive(Debug)]
pub struct CugraphExtensionPlanner {
    host_boundary_config: GpuIslandHostBoundaryConfig,
}

impl CugraphExtensionPlanner {
    pub(crate) fn new(host_boundary_config: GpuIslandHostBoundaryConfig) -> Self {
        Self { host_boundary_config }
    }
}

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
            let _ = (planner, logical_inputs, session, self.host_boundary_config);
            let exec = CugraphFeatureDisabledExec::new(node.spec().algorithm, node.output_schema(), None, None)?;
            return Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>));
        }

        #[cfg(feature = "cugraph")]
        {
            let session_state = session.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
                DataFusionError::Plan("cuGraph extension planning requires a DataFusion SessionState".to_string())
            })?;
            let native_execution_config = planning_native_execution_config(session);
            // Each input is planned on its own state because a native input
            // root becomes a GPU-resident producer named for its role. The
            // outcomes of that planning are labelled with the function and the
            // relation, such as `cugraph_pagerank.edges`, and travel with the
            // graph to the report of the query root.
            let function = node.spec().algorithm.function_name();
            let planning_state = |role: &str| {
                let sink = InputPlanningSink::default();
                let state = session_state_with_native_execution_config(
                    session_state,
                    native_execution_config.clone(),
                    format!("{}.{role}", node.gpu_result_input_name()),
                    sink.clone(),
                )
                .unwrap_or_else(|| session_state.clone());
                (state, sink)
            };
            let mut children = Vec::with_capacity(logical_inputs.len());
            let mut input_planning = PlanningReport::new();
            for (index, (input, relation)) in logical_inputs.iter().zip(node.roles()).enumerate() {
                let role = if index == 0 { "edge".to_string() } else { format!("predicate_{index}") };
                let (state, sink) = planning_state(&role);
                children.push(planner.create_physical_plan(input, &state).await?);
                let mut report = sink.take();
                report.label_relation_input(&format!("{function}.{}", relation.as_str()));
                input_planning.append(report);
            }
            let edge_source = children.remove(0);
            let vertex_domain = validate_edge_source_schema(node.spec(), &edge_source.schema())?;
            let mut predicate_sources = Vec::with_capacity(children.len() + 1);
            if let Some(source) =
                literal_bfs_source_plan(&planning_state("bfs_source").0, node.spec(), &vertex_domain).await?
            {
                predicate_sources.push(source);
            }
            predicate_sources.extend(children);
            let exec = CugraphAlgoExec::try_new_with_gpu_result_input_name(
                node.spec().clone(),
                edge_source,
                predicate_sources,
                None,
                None,
                native_execution_config,
                self.host_boundary_config,
                node.gpu_result_input_name().to_string(),
            )?
            .with_input_planning_report(input_planning);
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}
