use std::sync::Arc;

use datafusion::{
    common::Result as DataFusionResult,
    logical_expr::{
        Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::TableFactor,
};

use super::{
    CumlLogicalNode, CumlOperation,
    model_store::{CumlModelStore, bind_operation_model, require_model_store},
    options::{dense_relation_function, validate_options},
};
use crate::gpu_functions::dense_relation::table_function_name;

#[derive(Debug, Default)]
/// DataFusion SQL relation planner that binds named cuML function arguments.
///
/// Unrecognized relations retain their original DataFusion planning path.
pub struct CumlRelationPlanner {
    model_store: Option<Arc<CumlModelStore>>,
}

/// Planner without a models root; `cuml_forest_predict` reports
/// `model_root_unconfigured`.
pub fn cuml_relation_planner() -> Arc<dyn RelationPlanner> {
    cuml_relation_planner_with_store(None)
}

pub(crate) fn cuml_relation_planner_with_store(model_store: Option<Arc<CumlModelStore>>) -> Arc<dyn RelationPlanner> {
    Arc::new(CumlRelationPlanner { model_store })
}

impl RelationPlanner for CumlRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<RelationPlanning> {
        let Some(operation) =
            table_function_name(&relation, context).and_then(|name| CumlOperation::from_sql_function_name(&name))
        else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        require_model_store(operation, self.model_store.as_deref())?;
        let call = dense_relation_function(operation).plan_call(relation, context)?;
        let model = bind_operation_model(
            operation,
            self.model_store.as_deref(),
            &validate_options(operation, &call.options_json)?,
        )?;
        let node = CumlLogicalNode::try_new_with_model(operation, call.inputs, &call.options_json, model)?
            .with_input_orderings(call.input_orderings)?;
        let plan = LogicalPlan::Extension(Extension { node: Arc::new(node) });
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, call.alias))))
    }
}
