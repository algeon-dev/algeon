use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Result,
    execution::context::QueryPlanner,
    logical_expr::{LogicalPlan, planner::RelationPlanner},
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

#[derive(Clone)]
pub struct AlgeonQueryPlanner {
    caller_extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
    family_planners: BTreeMap<&'static str, FamilyPlanners>,
}

/// Planners a GPU SQL family installed on the session. The relation planner is
/// kept so a later install can find it in the builder's list by identity.
#[derive(Clone)]
struct FamilyPlanners {
    extension: Arc<dyn ExtensionPlanner + Send + Sync>,
    relation: Arc<dyn RelationPlanner>,
}

impl AlgeonQueryPlanner {
    /// Builds the Algeon planner from caller-owned extension planners.
    pub fn new(caller_extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>) -> Self {
        Self { caller_extension_planners, family_planners: BTreeMap::new() }
    }

    pub(crate) fn with_family(
        mut self,
        family_id: &'static str,
        extension: Arc<dyn ExtensionPlanner + Send + Sync>,
        relation: Arc<dyn RelationPlanner>,
    ) -> Self {
        self.family_planners.entry(family_id).or_insert(FamilyPlanners { extension, relation });
        self
    }

    pub(crate) fn contains_family(&self, family_id: &str) -> bool {
        self.family_planners.contains_key(family_id)
    }

    pub(crate) fn family_relation_planner(&self, family_id: &str) -> Option<&Arc<dyn RelationPlanner>> {
        self.family_planners.get(family_id).map(|family| &family.relation)
    }

    fn extension_planners(&self) -> Vec<Arc<dyn ExtensionPlanner + Send + Sync>> {
        self.caller_extension_planners
            .iter()
            .cloned()
            .chain(self.family_planners.values().map(|family| Arc::clone(&family.extension)))
            .collect()
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
            .field("family_ids", &self.family_planners.keys().collect::<Vec<_>>())
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
