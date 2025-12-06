use std::{
    fmt,
    sync::{Arc, Mutex},
};

use datafusion::{
    common::{Result as DataFusionResult, ScalarValue, config::ConfigOptions},
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, filter::FilterExec, projection::ProjectionExec},
};

use crate::{
    error::{Error, ErrorCode},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{GpuCandidate, PlanningReport},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    min_operators_per_island: usize,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self { fallback_policy: GpuFallbackPolicy::default(), min_operators_per_island: 2 }
    }
}

impl CudfOptimizerConfig {
    pub fn new(fallback_policy: GpuFallbackPolicy) -> Self {
        Self { fallback_policy, ..Self::default() }
    }

    pub fn fallback_policy(&self) -> GpuFallbackPolicy {
        self.fallback_policy
    }

    pub fn min_operators_per_island(&self) -> usize {
        self.min_operators_per_island
    }

    pub fn with_min_operators_per_island(mut self, min_operators_per_island: usize) -> Self {
        self.min_operators_per_island = min_operators_per_island.max(1);
        self
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub plan: Arc<dyn ExecutionPlan>,
    pub report: PlanningReport,
}

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
        "datafusion_cudf_gpu_island"
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

    report.push_rewritten(candidate.public);
    crate::exec::CudfGpuIslandExec::try_new_from_plan(plan)
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(Into::into)
}

fn rejection_reason(candidate: &InspectedCandidate, config: &CudfOptimizerConfig) -> Option<FallbackReason> {
    if config.fallback_policy == GpuFallbackPolicy::CpuOnly {
        return Some(FallbackReason::ExplicitCpuPolicy);
    }
    if !candidate.supported_expressions {
        return Some(FallbackReason::UnsupportedExpression);
    }
    if candidate.public.operator_count < config.min_operators_per_island {
        return Some(FallbackReason::InsufficientIslandSize);
    }
    None
}

fn candidate_rejected_error(candidate: &GpuCandidate, reason: FallbackReason, config: &CudfOptimizerConfig) -> Error {
    Error::new(
        ErrorCode::GpuCandidateRejected,
        "datafusion_cudf.optimize_plan",
        "GPU candidate rejected by adapter policy",
    )
    .with_fact("candidate_shape", candidate.shape)
    .with_fact("fallback_reason", reason.code())
    .with_fact("fallback_policy", config.fallback_policy())
    .with_fact("operator_count", candidate.operator_count)
    .with_fact("min_operators_per_island", config.min_operators_per_island())
}

#[derive(Debug, Clone)]
struct InspectedCandidate {
    public: GpuCandidate,
    supported_expressions: bool,
}

fn inspect_composite_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    let projection = plan.as_any().downcast_ref::<ProjectionExec>()?;
    let filter = projection.input().as_any().downcast_ref::<FilterExec>()?;
    let supported_expressions = projection_exprs_supported(projection) && expr_supported(filter.predicate());
    Some(InspectedCandidate {
        public: GpuCandidate::with_operator_count(
            "ProjectionExec(FilterExec(child))",
            projection.schema().fields().len(),
            2,
        ),
        supported_expressions,
    })
}

fn inspect_candidate(plan: &dyn ExecutionPlan) -> Option<InspectedCandidate> {
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("ProjectionExec(child)", projection.schema().fields().len(), 1),
            supported_expressions: projection_exprs_supported(projection),
        });
    }

    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        return Some(InspectedCandidate {
            public: GpuCandidate::with_operator_count("FilterExec(child)", filter.schema().fields().len(), 1),
            supported_expressions: expr_supported(filter.predicate()),
        });
    }

    None
}

fn projection_exprs_supported(projection: &ProjectionExec) -> bool {
    projection.expr().iter().all(|expr| expr_supported(&expr.expr))
}

fn expr_supported(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if expr.as_any().downcast_ref::<Column>().is_some() {
        return true;
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return literal_supported(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return binary_operator_supported(binary.op())
            && expr_supported(binary.left())
            && expr_supported(binary.right());
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return expr_supported(not.arg());
    }
    false
}

fn literal_supported(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Boolean(Some(_))
            | ScalarValue::Int8(Some(_))
            | ScalarValue::Int16(Some(_))
            | ScalarValue::Int32(Some(_))
            | ScalarValue::Int64(Some(_))
            | ScalarValue::UInt8(Some(_))
            | ScalarValue::UInt16(Some(_))
            | ScalarValue::UInt32(Some(_))
            | ScalarValue::UInt64(Some(_))
            | ScalarValue::Float32(Some(_))
            | ScalarValue::Float64(Some(_))
    )
}

fn binary_operator_supported(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::And
            | Operator::Or
    )
}

impl fmt::Display for GpuFallbackPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuOnly => f.write_str("GpuOnly"),
            Self::GpuPreferred => f.write_str("GpuPreferred"),
            Self::CpuOnly => f.write_str("CpuOnly"),
        }
    }
}
