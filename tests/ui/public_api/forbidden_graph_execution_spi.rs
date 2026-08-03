use datafusion_nexus::native_exec::graph::{
    GraphExecutionPlan, GraphExecutionPlanDiscovery, register_graph_execution_plan_discovery,
};

struct ExternalGraphPlan;

impl GraphExecutionPlan for ExternalGraphPlan {
    fn graph_model_ask_for_admission(
        &self,
    ) -> datafusion::common::Result<Option<datafusion_nexus::native_exec::graph::GraphAdmissionModelAsk>> {
        Ok(None)
    }
}

fn main() {
    let _: Option<GraphExecutionPlanDiscovery> = None;
    let _ = register_graph_execution_plan_discovery;
}
