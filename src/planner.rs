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
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

use crate::{
    error::{Error, ErrorCode},
    exec::CudfExecutionConfig,
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{GpuCandidate, PlanningReport},
    session::CUDF_PHYSICAL_OPTIMIZER_RULE_NAME,
};

#[derive(Debug, Clone)]
pub struct CudfOptimizerConfig {
    fallback_policy: GpuFallbackPolicy,
    min_operators_per_island: usize,
    execution_config: CudfExecutionConfig,
}

impl Default for CudfOptimizerConfig {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            min_operators_per_island: 2,
            execution_config: CudfExecutionConfig::default(),
        }
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

    pub fn execution_config(&self) -> &CudfExecutionConfig {
        &self.execution_config
    }

    pub fn with_min_operators_per_island(mut self, min_operators_per_island: usize) -> Self {
        self.min_operators_per_island = min_operators_per_island.max(1);
        self
    }

    pub fn with_execution_config(mut self, execution_config: CudfExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    /// Use one caller-provided cudf runtime for GPU islands produced by this optimizer.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.execution_config = self.execution_config.clone().with_runtime(runtime);
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

    report.push_rewritten(candidate.public);
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

fn projection_rejection(projection: &ProjectionExec) -> Option<FallbackReason> {
    projection.expr().iter().find_map(|expr| expr_rejection(&expr.expr))
}

fn sort_rejection(sort: &SortExec) -> Option<FallbackReason> {
    sort.expr()
        .iter()
        .any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none())
        .then_some(FallbackReason::UnsupportedExpression)
}

fn aggregate_rejection(aggregate: &AggregateExec) -> Option<FallbackReason> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Some(FallbackReason::UnsupportedOperator);
    }

    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Some(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Some(FallbackReason::UnsupportedOperator);
    }
    if group_by.expr().iter().any(|(expr, _alias)| expr.as_any().downcast_ref::<Column>().is_none()) {
        return Some(FallbackReason::UnsupportedExpression);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Some(FallbackReason::UnsupportedExpression);
    }

    aggregate.aggr_expr().iter().find_map(|expr| aggregate_expr_rejection(expr))
}

fn aggregate_expr_rejection(
    expr: &datafusion::physical_expr::aggregate::AggregateFunctionExpr,
) -> Option<FallbackReason> {
    if expr.is_distinct() || !expr.order_bys().is_empty() {
        return Some(FallbackReason::UnsupportedExpression);
    }
    if !matches!(expr.fun().name().to_ascii_lowercase().as_str(), "sum" | "min" | "max" | "count" | "avg") {
        return Some(FallbackReason::UnsupportedOperator);
    }
    let expressions = expr.expressions();
    if expressions.len() != 1 || expressions[0].as_any().downcast_ref::<Column>().is_none() {
        return Some(FallbackReason::UnsupportedExpression);
    }
    None
}

fn expr_rejection(expr: &Arc<dyn PhysicalExpr>) -> Option<FallbackReason> {
    if expr.as_any().downcast_ref::<Column>().is_some() {
        return None;
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return (!literal_supported(literal.value())).then_some(FallbackReason::UnsupportedExpression);
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if !binary_operator_supported(binary.op()) {
            return Some(FallbackReason::UnsupportedExpression);
        }
        return first_rejection([expr_rejection(binary.left()), expr_rejection(binary.right())]);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return expr_rejection(not.arg());
    }
    Some(FallbackReason::UnsupportedExpression)
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
