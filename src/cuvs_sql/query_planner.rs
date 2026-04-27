use std::{fmt, sync::Arc};

use arrow::datatypes::DataType;
use async_trait::async_trait;
use datafusion::{
    common::Result,
    execution::{SessionState, context::QueryPlanner},
    logical_expr::{LogicalPlan, UserDefinedLogicalNode},
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

use super::{CuvsAlgoExec, CuvsDenseVectorBinding, CuvsLogicalNode, CuvsVectorRelationBinding};
use crate::{
    native_exec::NexusNativeExecutionConfig, planner::optimize_native_relational_child_to_gpu_producer,
    session::installed_nexus_native_optimizer_config,
};

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
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(self.extension_planners.clone())
            .create_physical_plan(logical_plan, session_state)
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
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<CuvsLogicalNode>() else {
            return Ok(None);
        };
        let optimizer_config = installed_nexus_native_optimizer_config(session_state)
            .map(|config| config.with_datafusion_config_options(session_state.config_options()))
            .transpose()?;
        let native_execution_config = optimizer_config
            .as_ref()
            .map(|config| config.native_execution_config().clone())
            .unwrap_or_else(NexusNativeExecutionConfig::unbounded);
        let physical_inputs = match &optimizer_config {
            Some(config) => physical_inputs
                .iter()
                .zip(node.validated_options().bindings())
                .map(|(input, binding)| {
                    if requires_cuvs_vector_ingress(input.as_ref(), binding) {
                        // Numeric list support is intentionally owned by the
                        // vector executor. A generic relational native source
                        // has no proof-bearing List<Float32> contract, so
                        // lowering this child would fail before the cuVS
                        // device witness can validate uniform dimensions.
                        Ok(Arc::clone(input))
                    } else {
                        optimize_native_relational_child_to_gpu_producer(Arc::clone(input), config)
                    }
                })
                .collect::<Result<Vec<_>>>()?,
            None => physical_inputs.to_vec(),
        };
        CuvsAlgoExec::try_new(
            node.operation(),
            node.validated_options().clone(),
            physical_inputs,
            node.output_schema(),
            native_execution_config,
        )
        .map(|exec| Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
    }
}

fn requires_cuvs_vector_ingress(input: &dyn ExecutionPlan, binding: &CuvsVectorRelationBinding) -> bool {
    let CuvsDenseVectorBinding::VectorColumn(column) = binding.vector() else {
        return false;
    };
    let schema = input.schema();
    let Ok(field) = schema.field_with_name(column) else {
        // Logical validation already rejects a missing vector field. Keep the
        // physical planner fail-closed if an extension planner changes the
        // child schema unexpectedly.
        return false;
    };
    matches!(field.data_type(), DataType::FixedSizeList(_, _) | DataType::List(_) | DataType::LargeList(_))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::{ExecutionPlan, empty::EmptyExec};

    use super::*;

    #[test]
    fn variable_and_fixed_size_list_bindings_stay_with_cuvs_ingress() {
        let options = super::super::options::validate_options(
            super::super::CuvsOperation::KMeans,
            r#"{"input":{"id":"id","vector":{"column":"embedding"}},"n_clusters":1}"#,
        )
        .expect("valid KMeans vector-column options");
        let binding = &options.bindings()[0];

        for data_type in [
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
        ] {
            let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("embedding", data_type, false),
            ]))));
            assert!(requires_cuvs_vector_ingress(input.as_ref(), binding));
        }
    }
}
