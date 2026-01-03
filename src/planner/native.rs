use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningDiagnostics {
    pub input_plan_root: String,
    pub input_plan_shape: String,
    pub report: PlanningReport,
    pub source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    pub bounded_join_reservation_facts: BTreeMap<String, String>,
}

#[derive(Debug)]
pub struct CudfNativePhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    phase: NativeOptimizerPhase,
    state: Arc<CudfNativePhysicalOptimizerRuleState>,
}

#[derive(Debug)]
struct CudfNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<Option<String>>,
    last_input_plan_shape: Mutex<Option<String>>,
    last_report: Mutex<Option<PlanningReport>>,
    last_source_diagnostics: Mutex<Vec<cudf_query_engine::SourceDiagnostics>>,
    last_bounded_join_reservation_facts: Mutex<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerPhase {
    CountOnlyProbe,
    Final,
}

impl CudfNativePhysicalOptimizerRule {
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self {
            config,
            phase: NativeOptimizerPhase::Final,
            state: Arc::new(CudfNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(None),
                last_input_plan_shape: Mutex::new(None),
                last_report: Mutex::new(None),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_bounded_join_reservation_facts: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            phase: NativeOptimizerPhase::CountOnlyProbe,
            state: Arc::clone(&self.state),
        }
    }

    pub fn last_report(&self) -> Option<PlanningReport> {
        self.state.last_report.lock().expect("native planning report mutex poisoned").clone()
    }

    pub fn last_source_diagnostics(&self) -> Vec<cudf_query_engine::SourceDiagnostics> {
        self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned").clone()
    }

    pub fn last_planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.last_report().map(|report| NativePlanningDiagnostics {
            input_plan_root: self
                .state
                .last_input_plan_root
                .lock()
                .expect("native input plan root mutex poisoned")
                .clone()
                .unwrap_or_default(),
            input_plan_shape: self
                .state
                .last_input_plan_shape
                .lock()
                .expect("native input plan shape mutex poisoned")
                .clone()
                .unwrap_or_default(),
            report,
            source_diagnostics: self.last_source_diagnostics(),
            bounded_join_reservation_facts: self
                .state
                .last_bounded_join_reservation_facts
                .lock()
                .expect("native bounded join reservation facts mutex poisoned")
                .clone(),
        })
    }
}

impl PhysicalOptimizerRule for CudfNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if plan.as_any().downcast_ref::<NativeCudfExec>().is_some() {
            return Ok(plan);
        }
        if let Some(native_plan) = peel_cooperative_native_exec(&plan) {
            return Ok(native_plan);
        }

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            Some(plan.name().to_string());
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            Some(physical_plan_shape(plan.as_ref()));

        let Some(native_execution_config) = self.config.native_execution_config() else {
            *self.state.last_report.lock().expect("native planning report mutex poisoned") =
                Some(PlanningReport::new());
            *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") = Vec::new();
            *self
                .state
                .last_bounded_join_reservation_facts
                .lock()
                .expect("native bounded join reservation facts mutex poisoned") = BTreeMap::new();
            return Ok(plan);
        };

        let optimized = match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => {
                optimize_count_only_native_plan(Arc::clone(&plan), native_execution_config)?
            },
            NativeOptimizerPhase::Final => optimize_final_native_plan(Arc::clone(&plan), native_execution_config)?,
        };
        let optimized = apply_native_fallback_policy(Arc::clone(&plan), optimized, &self.config)?;
        *self.state.last_report.lock().expect("native planning report mutex poisoned") = Some(optimized.report.clone());
        *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") =
            optimized.source_diagnostics.clone();
        *self
            .state
            .last_bounded_join_reservation_facts
            .lock()
            .expect("native bounded join reservation facts mutex poisoned") =
            optimized.bounded_join_reservation_facts.clone();
        Ok(optimized.plan)
    }

    fn name(&self) -> &str {
        match self.phase {
            NativeOptimizerPhase::CountOnlyProbe => CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerPhase::Final => CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn apply_native_fallback_policy(
    original_plan: Arc<dyn ExecutionPlan>,
    attempt: NativeOptimizationAttempt,
    config: &CudfOptimizerConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    match config.fallback_policy {
        GpuFallbackPolicy::GpuPreferred => Ok(attempt),
        GpuFallbackPolicy::CpuOnly => Ok(NativeOptimizationAttempt {
            plan: original_plan,
            report: native_cpu_only_report(&attempt.report),
            source_diagnostics: attempt.source_diagnostics,
            bounded_join_reservation_facts: BTreeMap::new(),
            rewritten: false,
        }),
        GpuFallbackPolicy::GpuOnly => {
            if let Some((candidate, reason)) = first_native_rejection(&attempt.report) {
                return Err(candidate_rejected_error(candidate, reason, config).into());
            }
            Ok(attempt)
        },
    }
}

fn native_cpu_only_report(report: &PlanningReport) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } | CandidateOutcome::Rejected { candidate, .. } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: FallbackReason::ExplicitCpuPolicy }
        },
    }))
}

fn first_native_rejection(report: &PlanningReport) -> Option<(&GpuCandidate, FallbackReason)> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { candidate, reason } => Some((candidate, *reason)),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

#[derive(Debug, Clone)]
struct NativeOptimizationAttempt {
    plan: Arc<dyn ExecutionPlan>,
    report: PlanningReport,
    source_diagnostics: Vec<cudf_query_engine::SourceDiagnostics>,
    bounded_join_reservation_facts: BTreeMap<String, String>,
    rewritten: bool,
}

fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering = crate::native::lower_count_only_physical_plan_to_native(lowering_input);
    replace_lowered_root_or_keep_plan(plan, lowering, native_execution_config)
}

fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    if lowering.plans.len() == 1 && lowering.rewritten_count() == 1 {
        return replace_lowered_root_or_keep_plan(plan, lowering, native_execution_config);
    }

    let mut report = lowering.report.clone();
    let mut source_diagnostics = lowering.source_diagnostics.clone();
    let mut bounded_join_reservation_facts = BTreeMap::new();
    let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
        Arc::clone(&plan),
        native_execution_config,
        &mut report,
        &mut source_diagnostics,
        &mut bounded_join_reservation_facts,
    )?
    else {
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            source_diagnostics,
            bounded_join_reservation_facts,
            rewritten: false,
        });
    };

    Ok(NativeOptimizationAttempt {
        plan: rewritten_plan,
        report,
        source_diagnostics,
        bounded_join_reservation_facts,
        rewritten: true,
    })
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    lowering: crate::native::NativeLoweringResult,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            source_diagnostics: lowering.source_diagnostics,
            bounded_join_reservation_facts: BTreeMap::new(),
            rewritten: false,
        });
    }
    if let Some(bounded_join_reservation_facts) =
        native_execution_boundary_rejection_facts(&lowering.plans[0], native_execution_config)
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_boundary_report(&lowering.report),
            source_diagnostics: lowering.source_diagnostics,
            bounded_join_reservation_facts,
            rewritten: false,
        });
    }

    let plan = NativeCudfExec::try_new(lowering.plans[0].clone(), native_execution_config.clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)?;
    let bounded_join_reservation_facts =
        native_execution_boundary_modeled_facts(&lowering.plans[0], native_execution_config).unwrap_or_default();
    Ok(NativeOptimizationAttempt {
        plan,
        report: lowering.report,
        source_diagnostics: lowering.source_diagnostics,
        bounded_join_reservation_facts,
        rewritten: true,
    })
}

fn optimize_native_children_for_host_arrow_boundary(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    report: &mut PlanningReport,
    source_diagnostics: &mut Vec<cudf_query_engine::SourceDiagnostics>,
    bounded_join_reservation_facts: &mut BTreeMap<String, String>,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    if !host_arrow_boundary_allows_child_replacement(plan.as_ref()) {
        return Ok(None);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(None);
    }

    let mut rewritten = false;
    let mut child_reports = Vec::new();
    let mut child_source_diagnostics = Vec::new();
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        let attempt = optimize_final_native_plan(Arc::clone(child), native_execution_config)?;
        rewritten |= attempt.rewritten;
        child_reports.push(attempt.report);
        child_source_diagnostics.extend(attempt.source_diagnostics);
        bounded_join_reservation_facts.extend(attempt.bounded_join_reservation_facts);
        new_children.push(attempt.plan);
    }

    if !rewritten {
        return Ok(None);
    }

    for child_report in child_reports {
        append_planning_report(report, &child_report);
    }
    source_diagnostics.extend(child_source_diagnostics);
    plan.with_new_children(new_children).map(Some)
}

fn append_planning_report(report: &mut PlanningReport, extra: &PlanningReport) {
    *report = PlanningReport::from_outcomes(report.outcomes().iter().cloned().chain(extra.outcomes().iter().cloned()));
}

fn host_arrow_boundary_allows_child_replacement(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<FilterExec>().is_some()
        || plan.as_any().downcast_ref::<ProjectionExec>().is_some()
        || plan.as_any().downcast_ref::<SortExec>().is_some()
        || plan.as_any().downcast_ref::<LocalLimitExec>().is_some()
        || plan.as_any().downcast_ref::<AggregateExec>().is_some()
}

fn native_execution_boundary_rejection_facts(
    plan: &cudf_query_engine::QueryPlan,
    execution_config: &NativeCudfExecutionConfig,
) -> Option<BTreeMap<String, String>> {
    execution_config
        .memory_control_policy()
        .is_bounded()
        .then(|| plan.bounded_join_unsupported_shape_facts(execution_config.boundary_policy()))
        .flatten()
        .map(|facts| facts.into_iter().map(|(key, value)| (key.to_string(), value)).collect())
}

fn native_execution_boundary_modeled_facts(
    plan: &cudf_query_engine::QueryPlan,
    execution_config: &NativeCudfExecutionConfig,
) -> Option<BTreeMap<String, String>> {
    if execution_config.memory_control_policy().is_bounded() && plan.contains_join() {
        plan.bounded_join_modeled_facts(execution_config.boundary_policy())
            .map(|facts| facts.into_iter().map(|(key, value)| (key.to_string(), value)).collect())
    } else {
        None
    }
}

fn native_execution_boundary_report(report: &PlanningReport) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => CandidateOutcome::Rejected {
            candidate: candidate.clone(),
            reason: FallbackReason::NativeBoundedJoinUnsupported,
        },
        CandidateOutcome::Rejected { candidate, reason } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: *reason }
        },
    }))
}

fn peel_output_requirement_exec(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.name() == "OutputRequirementExec"
        && let [child] = plan.children().as_slice()
    {
        return Arc::clone(child);
    }
    plan
}

fn peel_cooperative_native_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() != "CooperativeExec" {
        return None;
    }
    let children = plan.children();
    let [child] = children.as_slice() else {
        return None;
    };
    child.as_any().downcast_ref::<NativeCudfExec>().is_some().then(|| Arc::clone(child))
}

fn physical_plan_shape(plan: &dyn ExecutionPlan) -> String {
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
