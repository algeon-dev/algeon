use super::*;

#[derive(Debug)]
pub struct CudfPhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    last_report: Mutex<Option<PlanningReport>>,
}

impl CudfPhysicalOptimizerRule {
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self { config, last_report: Mutex::new(None) }
    }

    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub fn last_report(&self) -> Option<PlanningReport> {
        self.last_report.lock().expect("planning report mutex poisoned").clone()
    }
}

impl PhysicalOptimizerRule for CudfPhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let OptimizationResult { plan, report } = optimize_plan(plan, &self.config)?;
        *self.last_report.lock().expect("planning report mutex poisoned") = Some(report);
        Ok(plan)
    }

    fn name(&self) -> &str {
        CUDF_PHYSICAL_OPTIMIZER_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

pub fn optimize_plan(
    plan: Arc<dyn ExecutionPlan>,
    config: &CudfOptimizerConfig,
) -> DataFusionResult<OptimizationResult> {
    let mut report = PlanningReport::new();
    let plan = optimize_node(plan, config, &mut report)?;
    Ok(OptimizationResult { plan, report })
}

fn optimize_node(
    plan: Arc<dyn ExecutionPlan>,
    config: &CudfOptimizerConfig,
    report: &mut PlanningReport,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    if let Some(candidate) = inspect_composite_candidate(plan.as_ref()) {
        return handle_candidate(plan, candidate, config, report);
    }

    let children = plan.children();
    let plan = if children.is_empty() {
        plan
    } else {
        let new_children = children
            .into_iter()
            .map(|child| optimize_node(Arc::clone(child), config, report))
            .collect::<DataFusionResult<Vec<_>>>()?;
        plan.with_new_children(new_children)?
    };

    let Some(candidate) = inspect_candidate(plan.as_ref()) else {
        return Ok(plan);
    };

    handle_candidate(plan, candidate, config, report)
}

fn handle_candidate(
    plan: Arc<dyn ExecutionPlan>,
    candidate: InspectedCandidate,
    config: &CudfOptimizerConfig,
    report: &mut PlanningReport,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let rejection = rejection_reason(&candidate, config);
    if let Some(reason) = rejection {
        report.push_rejected(candidate.public.clone(), reason);
        if config.fallback_policy == GpuFallbackPolicy::GpuOnly {
            return Err(candidate_rejected_error(&candidate.public, reason, config).into());
        }
        return Ok(plan);
    }

    report.push_rewritten(candidate.public.clone());
    crate::exec::CudfGpuIslandExec::try_new_from_plan_with_execution_config(plan, config.execution_config().clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(Into::into)
}

fn rejection_reason(candidate: &InspectedCandidate, config: &CudfOptimizerConfig) -> Option<FallbackReason> {
    if config.fallback_policy == GpuFallbackPolicy::CpuOnly {
        return Some(FallbackReason::ExplicitCpuPolicy);
    }
    if let Some(reason) = candidate.rejection {
        return Some(reason);
    }
    if candidate.public.operator_count < config.min_operators_per_island {
        return Some(FallbackReason::InsufficientIslandSize);
    }
    None
}

#[derive(Debug, Clone)]
struct InspectedCandidate {
    public: GpuCandidate,
    rejection: Option<FallbackReason>,
}

fn inspect_composite_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        if let Some(sort) = limit.input().as_any().downcast_ref::<SortExec>() {
            if let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>() {
                let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
                let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
                let rejection = first_rejection([
                    aggregate_rejection(aggregate),
                    sort_rejection(sort),
                    projection_rejection(projection),
                    expr_rejection(filter.predicate()),
                ]);
                return Some(InspectedCandidate {
                    public: GpuCandidate::with_operator_count(
                        "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(child)))))",
                        sort.schema().fields().len(),
                        5,
                    ),
                    rejection,
                });
            }

            let projection = sort.input().as_any().downcast_ref::<ProjectionExec>()?;
            let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
            let rejection = first_rejection([
                projection_rejection(projection),
                sort_rejection(sort),
                expr_rejection(filter.predicate()),
            ]);
            return Some(InspectedCandidate {
                public: GpuCandidate::with_operator_count(
                    "LocalLimitExec(SortExec(ProjectionExec(FilterExec(child))))",
                    sort.schema().fields().len(),
                    4,
                ),
                rejection,
            });
        }

        let projection = limit.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([projection_rejection(projection), expr_rejection(filter.predicate())]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "LocalLimitExec(ProjectionExec(FilterExec(child)))",
                projection.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        if let Some(aggregate) = sort.input().as_any().downcast_ref::<AggregateExec>() {
            let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
            let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
            let rejection = first_rejection([
                aggregate_rejection(aggregate),
                sort_rejection(sort),
                projection_rejection(projection),
                expr_rejection(filter.predicate()),
            ]);
            return Some(InspectedCandidate {
                public: GpuCandidate::with_operator_count(
                    "SortExec(AggregateExec(ProjectionExec(FilterExec(child))))",
                    sort.schema().fields().len(),
                    4,
                ),
                rejection,
            });
        }

        let projection = sort.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([
            projection_rejection(projection),
            sort_rejection(sort),
            expr_rejection(filter.predicate()),
        ]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "SortExec(ProjectionExec(FilterExec(child)))",
                sort.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        let projection = aggregate.input().as_any().downcast_ref::<ProjectionExec>()?;
        let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
        let rejection = first_rejection([
            aggregate_rejection(aggregate),
            projection_rejection(projection),
            expr_rejection(filter.predicate()),
        ]);
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count(
                "AggregateExec(ProjectionExec(FilterExec(child)))",
                aggregate.schema().fields().len(),
                3,
            ),
            rejection,
        });
    }

    let projection = plan.as_any().downcast_ref::<ProjectionExec>()?;
    let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
    let rejection = first_rejection([projection_rejection(projection), expr_rejection(filter.predicate())]);
    Some(InspectedCandidate {
        public: GpuCandidate::with_operator_count(
            "ProjectionExec(FilterExec(child))",
            projection.schema().fields().len(),
            2,
        ),
        rejection,
    })
}

fn inspect_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    if let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>()
        && data_source.downcast_to_file_source::<ParquetSource>().is_some()
    {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("LocalParquetScan", data_source.schema().fields().len(), 1),
            rejection: None,
        });
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("ProjectionExec(child)", projection.schema().fields().len(), 1),
            rejection: projection_rejection(projection),
        });
    }

    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("FilterExec(child)", filter.schema().fields().len(), 1),
            rejection: expr_rejection(filter.predicate()),
        });
    }

    None
}

fn first_rejection(rejections: impl IntoIterator<Item = Option<FallbackReason>>) -> Option<FallbackReason> {
    rejections.into_iter().flatten().next()
}
