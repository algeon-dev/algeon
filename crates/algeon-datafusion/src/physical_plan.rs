//! The CPU boundary marker for subtrees that must stay on DataFusion.

use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream},
};

pub(crate) fn cpu_physical_boundary(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    Arc::new(DataFusionCpuBoundaryExec::new(input))
}

/// Transparent node marking a subtree native planning leaves on DataFusion.
///
/// Metadata table functions place it above their results. Fragment search
/// stops at it, and coverage reports it as a CPU boundary. Execution delegates
/// to the input unchanged.
#[derive(Debug)]
pub(crate) struct DataFusionCpuBoundaryExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl DataFusionCpuBoundaryExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self { properties: Arc::clone(input.properties()), input }
    }

    #[cfg(test)]
    pub(crate) fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl DisplayAs for DataFusionCpuBoundaryExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataFusionCpuBoundaryExec")
    }
}

impl ExecutionPlan for DataFusionCpuBoundaryExec {
    fn name(&self) -> &str {
        "DataFusionCpuBoundaryExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = children.try_into().map_err(|children: Vec<Arc<dyn ExecutionPlan>>| {
            DataFusionError::Internal(format!("coverage CPU boundary expected one child, got {}", children.len()))
        })?;
        Ok(Arc::new(Self::new(input)))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, task_context)
    }
}

/// Stable `Name(Child,Child)` rendering of a plan tree for reports.
pub(crate) fn physical_plan_shape(plan: &dyn ExecutionPlan) -> String {
    let children = plan.children();
    if children.is_empty() {
        return plan.name().to_string();
    }

    let mut shape = String::new();
    shape.push_str(plan.name());
    shape.push('(');
    for (index, child) in children.iter().enumerate() {
        if index > 0 {
            shape.push(',');
        }
        shape.push_str(&physical_plan_shape(child.as_ref()));
    }
    shape.push(')');
    shape
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema},
        datasource::{TableProvider, empty::EmptyTable},
        execution::context::SessionContext,
        physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions},
    };

    use super::*;

    #[tokio::test]
    async fn cpu_boundary_exec_preserves_properties_and_delegates_single_child() {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let table = EmptyTable::new(schema);
        let input = table.scan(&ctx.state(), None, &[], None).await.unwrap();
        let boundary = cpu_physical_boundary(Arc::clone(&input));

        assert_eq!(boundary.name(), "DataFusionCpuBoundaryExec");
        assert_eq!(boundary.children().len(), 1);
        assert!(Arc::ptr_eq(boundary.children()[0], &input));
        assert!(Arc::ptr_eq(boundary.properties(), input.properties()));

        let boundary_exec =
            boundary.downcast_ref::<DataFusionCpuBoundaryExec>().expect("must downcast to DataFusionCpuBoundaryExec");
        assert!(Arc::ptr_eq(boundary_exec.input(), &input));

        let recompute = || ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
        let replaced = Arc::clone(&boundary)
            .replace_children(vec![Arc::clone(&input)], recompute())
            .expect("single child replacement succeeds");
        assert_eq!(replaced.name(), "DataFusionCpuBoundaryExec");

        let err = Arc::clone(&boundary).replace_children(vec![], recompute()).unwrap_err();
        assert!(err.to_string().contains("coverage CPU boundary expected one child, got 0"));

        let err2 = Arc::clone(&boundary)
            .replace_children(vec![Arc::clone(&input), Arc::clone(&input)], recompute())
            .unwrap_err();
        assert!(err2.to_string().contains("coverage CPU boundary expected one child, got 2"));
    }
}
