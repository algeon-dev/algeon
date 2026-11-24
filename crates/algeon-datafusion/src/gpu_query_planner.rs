//! The one `QueryPlanner` a session with GPU SQL families installs.

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, Result},
    execution::{SessionState, SessionStateBuilder, context::QueryPlanner},
    logical_expr::{
        Explain, ExplainFormat, LogicalPlan, PlanType, StringifiedPlan, ToStringifiedPlan, planner::RelationPlanner,
    },
    physical_plan::{ExecutionPlan, displayable, explain::ExplainExec},
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

use crate::native_exec::{project_plan_for_display, project_plan_for_display_ref};

/// Physical planner composing caller extension planners with the extension
/// planner of each installed GPU SQL family.
///
/// DataFusion accepts one `QueryPlanner` per session, so each family install
/// adds its planners here; the session's planner is never replaced. Caller
/// planners run first, families in id order. Installing a family twice keeps
/// the first registration.
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
        let planner = DefaultPhysicalPlanner::with_extension_planners(self.extension_planners());
        if let LogicalPlan::Explain(explain) = logical_plan {
            return plan_explain(&planner, explain, logical_plan, session).await;
        }
        planner.create_physical_plan(logical_plan, session).await
    }
}

/// Plans `EXPLAIN` while projecting native execution wrappers into per-node
/// display trees for indent and tree formats.
async fn plan_explain(
    planner: &DefaultPhysicalPlanner,
    explain: &Explain,
    logical_plan: &LogicalPlan,
    session: &dyn Session,
) -> Result<Arc<dyn ExecutionPlan>> {
    let config = &session.config_options().explain;
    if !explain.logical_optimization_succeeded
        || config.logical_plan_only
        || matches!(explain.explain_format, ExplainFormat::PostgresJSON | ExplainFormat::Graphviz)
    {
        return planner.create_physical_plan(logical_plan, session).await;
    }

    match explain.explain_format {
        ExplainFormat::Tree => {
            let optimized_plan = planner.create_physical_plan(explain.plan.as_ref(), session).await?;
            let projected = project_plan_for_display(&optimized_plan);
            let stringified_plans = vec![StringifiedPlan::new(
                PlanType::FinalPhysicalPlan,
                displayable(projected.as_ref())
                    .set_tree_maximum_render_width(config.tree_maximum_render_width)
                    .tree_render()
                    .to_string(),
            )];
            Ok(Arc::new(ExplainExec::new(Arc::clone(explain.schema.inner()), stringified_plans, explain.verbose)))
        },
        ExplainFormat::Indent => {
            let show_statistics = explain.show_statistics.unwrap_or(config.show_statistics);
            let mut stringified_plans = Vec::new();
            if !config.physical_plan_only {
                stringified_plans.clone_from(&explain.stringified_plans);
                stringified_plans.push(explain.plan.to_stringified(PlanType::FinalLogicalPlan));
            }

            // Build the unoptimized initial physical plan by running the
            // planner on a clone of the session with no physical optimizer
            // rules, then run `optimize_physical_plan` once so every optimizer
            // trace and the final plan share one pass.
            let initial_plan_result = if let Some(session_state) = session.as_any().downcast_ref::<SessionState>() {
                let unoptimized_session = SessionStateBuilder::new_from_existing(session_state.clone())
                    .with_physical_optimizer_rules(Vec::new())
                    .build();
                planner.create_physical_plan(explain.plan.as_ref(), &unoptimized_session).await
            } else if !explain.verbose {
                match planner.create_physical_plan(explain.plan.as_ref(), session).await {
                    Ok(final_plan) => {
                        let final_display = project_plan_for_display(&final_plan);
                        stringified_plans.push(StringifiedPlan::new(
                            PlanType::FinalPhysicalPlan,
                            displayable(final_display.as_ref())
                                .set_show_statistics(show_statistics)
                                .set_show_schema(config.show_schema)
                                .indent(false)
                                .to_string(),
                        ));
                        return Ok(Arc::new(ExplainExec::new(
                            Arc::clone(explain.schema.inner()),
                            stringified_plans,
                            false,
                        )));
                    },
                    Err(DataFusionError::Context(optimizer_name, error)) => {
                        stringified_plans.push(StringifiedPlan::new(
                            PlanType::OptimizedPhysicalPlan { optimizer_name },
                            error.to_string(),
                        ));
                        return Ok(Arc::new(ExplainExec::new(
                            Arc::clone(explain.schema.inner()),
                            stringified_plans,
                            false,
                        )));
                    },
                    Err(error) => {
                        stringified_plans
                            .push(StringifiedPlan::new(PlanType::PhysicalPlanError, error.strip_backtrace()));
                        return Ok(Arc::new(ExplainExec::new(
                            Arc::clone(explain.schema.inner()),
                            stringified_plans,
                            false,
                        )));
                    },
                }
            } else {
                return planner.create_physical_plan(logical_plan, session).await;
            };

            match initial_plan_result {
                Ok(input) => {
                    let initial_display = project_plan_for_display(&input);
                    stringified_plans.push(StringifiedPlan::new(
                        PlanType::InitialPhysicalPlan,
                        displayable(initial_display.as_ref())
                            .set_show_statistics(show_statistics)
                            .set_show_schema(config.show_schema)
                            .indent(explain.verbose)
                            .to_string(),
                    ));

                    if explain.verbose {
                        if !show_statistics {
                            stringified_plans.push(StringifiedPlan::new(
                                PlanType::InitialPhysicalPlanWithStats,
                                displayable(initial_display.as_ref())
                                    .set_show_statistics(true)
                                    .indent(explain.verbose)
                                    .to_string(),
                            ));
                        }
                        if !config.show_schema {
                            stringified_plans.push(StringifiedPlan::new(
                                PlanType::InitialPhysicalPlanWithSchema,
                                displayable(initial_display.as_ref())
                                    .set_show_schema(true)
                                    .indent(explain.verbose)
                                    .to_string(),
                            ));
                        }
                    }

                    let optimized_plan = planner.optimize_physical_plan(input, session, |plan, optimizer| {
                        let optimizer_name = optimizer.name().to_string();
                        let plan_type = PlanType::OptimizedPhysicalPlan { optimizer_name };
                        let projected = project_plan_for_display_ref(plan);
                        let rendered_plan = projected.as_deref().unwrap_or(plan);
                        stringified_plans.push(StringifiedPlan::new(
                            plan_type,
                            displayable(rendered_plan)
                                .set_show_statistics(show_statistics)
                                .set_show_schema(config.show_schema)
                                .indent(explain.verbose)
                                .to_string(),
                        ));
                    });

                    match optimized_plan {
                        Ok(final_plan) => {
                            let final_display = project_plan_for_display(&final_plan);
                            stringified_plans.push(StringifiedPlan::new(
                                PlanType::FinalPhysicalPlan,
                                displayable(final_display.as_ref())
                                    .set_show_statistics(show_statistics)
                                    .set_show_schema(config.show_schema)
                                    .indent(explain.verbose)
                                    .to_string(),
                            ));

                            if explain.verbose {
                                if !show_statistics {
                                    stringified_plans.push(StringifiedPlan::new(
                                        PlanType::FinalPhysicalPlanWithStats,
                                        displayable(final_display.as_ref())
                                            .set_show_statistics(true)
                                            .indent(explain.verbose)
                                            .to_string(),
                                    ));
                                }
                                if !config.show_schema {
                                    stringified_plans.push(StringifiedPlan::new(
                                        PlanType::FinalPhysicalPlanWithSchema,
                                        displayable(final_display.as_ref())
                                            .set_show_schema(true)
                                            .indent(explain.verbose)
                                            .to_string(),
                                    ));
                                }
                            }
                        },
                        Err(DataFusionError::Context(optimizer_name, error)) => {
                            let plan_type = PlanType::OptimizedPhysicalPlan { optimizer_name };
                            stringified_plans.push(StringifiedPlan::new(plan_type, error.to_string()));
                        },
                        Err(error) => return Err(error),
                    }
                },
                Err(error) => {
                    stringified_plans.push(StringifiedPlan::new(PlanType::PhysicalPlanError, error.strip_backtrace()));
                },
            }

            Ok(Arc::new(ExplainExec::new(Arc::clone(explain.schema.inner()), stringified_plans, explain.verbose)))
        },
        ExplainFormat::PostgresJSON | ExplainFormat::Graphviz => unreachable!("delegated above"),
    }
}
