use std::sync::Arc;

use datafusion::{
    common::Result as DataFusionResult,
    logical_expr::{
        Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::TableFactor,
};

use super::{CuvsLogicalNode, CuvsOperation, options::dense_relation_function};
use crate::gpu_functions::dense_relation::table_function_name;

#[derive(Debug, Default)]
/// DataFusion SQL relation planner that binds named cuVS function arguments.
///
/// Unrecognized relations retain their original DataFusion planning path.
pub struct CuvsRelationPlanner;

/// Creates a shared cuVS relation planner for installation on a session builder.
pub fn cuvs_relation_planner() -> Arc<dyn RelationPlanner> {
    Arc::new(CuvsRelationPlanner)
}

impl RelationPlanner for CuvsRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<RelationPlanning> {
        let Some(operation) =
            table_function_name(&relation, context).and_then(|name| CuvsOperation::from_sql_function_name(&name))
        else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let call = dense_relation_function(operation).plan_call(relation, context)?;
        let node = CuvsLogicalNode::try_new(operation, call.inputs, &call.options_json)?
            .with_input_orderings(call.input_orderings)?;
        let plan = LogicalPlan::Extension(Extension { node: Arc::new(node) });
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, call.alias))))
    }
}
