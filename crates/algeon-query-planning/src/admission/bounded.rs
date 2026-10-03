//! Admission boundary for compiled bounded pipelines.
//!
//! Capability-first bounded-pipeline admission evidence and rejections. This
//! is the single bounded admission contract; "capability" appears only in
//! preserved wire names. Runtime execution remains in
//! `exec::bounded`.

use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType as ArrowDataType, Schema};

use super::shared::{
    CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission, capability_first_bounded_pipeline_admission,
};
use crate::{
    backend::AnalysisContext,
    capability::{Reason, Rejection, RejectionOwner, UnsupportedPart},
    error::{QueryError, QueryErrorCode, QueryErrorKind, Result},
    expr::{AggregateFunction, AggregateSpec, Expr},
    mode::{CapabilityEnvelopeStatus, StatisticalAggregateExecutionMode},
    pipeline::CompiledPipeline,
    plan::{NodeId, PlanFingerprint, PlanNode, QueryPlan, ResultSink},
    policy::PlanningPolicy,
    prepared::{
        DataFusionExactHostAccumulator, aggregate_spec_has_floating_statistical_aggregate,
        datafusion_exact_host_accumulator, datafusion_exact_host_group_key_supported,
    },
    validation::validate_query_plan,
};

/// A plan that passed validation, capability analysis, compilation and the
/// compiled contracts exactly once.
///
/// Only [`admit`] and [`AdmittedPlan::with_pruned_sources`] construct it.
/// Execution and reporting read this artifact instead of deciding again; it
/// owns the plan it admitted, so no other plan, sink or mode can reach the
/// pipeline it carries.
#[derive(Debug)]
pub struct AdmittedPlan {
    plan: QueryPlan,
    pipeline: CompiledPipeline,
    fingerprint: PlanFingerprint,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
}

impl AdmittedPlan {
    /// The admitted plan, in the sink form execution runs.
    pub fn plan(&self) -> &QueryPlan {
        &self.plan
    }

    /// Compiled pipeline admitted for bounded execution.
    pub fn pipeline(&self) -> &CompiledPipeline {
        &self.pipeline
    }

    /// Execution identity of the admitted plan.
    pub const fn fingerprint(&self) -> PlanFingerprint {
        self.fingerprint
    }

    /// Execution mode the admission decision was made under.
    pub const fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.statistical_aggregate_execution_mode
    }

    /// Execution policy the pipeline was admitted under; execution reads its
    /// join and dynamic-filter thresholds from here.
    pub const fn policy(&self) -> PlanningPolicy {
        self.policy
    }

    /// Admits the plan footer pruning resolved from this one.
    ///
    /// Resolution may narrow Parquet row-group selections and append
    /// reader-predicate conjuncts; any other difference is an internal
    /// invariant violation. The resolved plan is admitted again under this
    /// artifact's mode and policy, and a rejection is the typed execution
    /// error, never a fallback.
    pub fn with_pruned_sources(&self, pruned: QueryPlan, ctx: AnalysisContext<'_>) -> Result<Arc<Self>> {
        if let Some(changed) = self.plan.first_change_beyond_source_refinement(&pruned) {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.admission.pruned_sources",
                "footer pruning changed more than source row-group selections and reader predicates",
            )
            .with_fact("changed", changed)
            .with_fact("admitted_plan_fingerprint", self.fingerprint.to_string()));
        }
        match admit(&pruned, self.statistical_aggregate_execution_mode, self.policy, ctx)? {
            AdmissionDecision::Admitted(admitted) => Ok(admitted),
            AdmissionDecision::Rejected(rejection) => Err(rejection.execution_error(&pruned)),
        }
    }
}

/// The one admission decision for a native plan: it runs as a compiled
/// pipeline, or it is rejected with the first contract it misses.
#[derive(Debug)]
pub enum AdmissionDecision {
    /// The compiled pipeline the plan executes as.
    Admitted(Arc<AdmittedPlan>),
    /// No compiled contract covers the plan.
    Rejected(AdmissionRejection),
}

impl AdmissionDecision {
    /// Admitted plan, when admitted.
    pub fn admission(&self) -> Option<&Arc<AdmittedPlan>> {
        match self {
            Self::Admitted(admission) => Some(admission),
            Self::Rejected(_) => None,
        }
    }

    /// Rejection evidence, when rejected.
    pub fn rejection(&self) -> Option<&Rejection> {
        match self {
            Self::Admitted(_) => None,
            Self::Rejected(rejection) => Some(rejection.rejection()),
        }
    }
}

/// Why a plan has no compiled execution.
#[derive(Debug, Clone)]
pub struct AdmissionRejection {
    rejection: Rejection,
    pipeline: Option<Box<CompiledPipeline>>,
    cause: RejectionCause,
}

#[derive(Debug, Clone)]
enum RejectionCause {
    CompiledContract,
    ExecutionMode,
    /// The validator found the plan well formed but outside what the backend
    /// supports. The error keeps the validator's code, operation, message,
    /// and facts.
    Validation(Arc<QueryError>),
}

impl AdmissionRejection {
    /// The first operator, edge, or pipeline contract the plan misses.
    pub fn rejection(&self) -> &Rejection {
        &self.rejection
    }

    /// Whether the requested statistical aggregate execution mode, rather than
    /// a compiled contract, refuses the plan.
    pub fn is_execution_mode_refusal(&self) -> bool {
        matches!(self.cause, RejectionCause::ExecutionMode)
    }

    /// The unsupported-construct error plan validation reported, when
    /// validation refused the plan.
    pub fn validation_error(&self) -> Option<&QueryError> {
        match &self.cause {
            RejectionCause::Validation(error) => Some(error),
            RejectionCause::CompiledContract | RejectionCause::ExecutionMode => None,
        }
    }

    /// The typed error a caller that must execute `plan` reports. A
    /// validation refusal reports the validator's own error.
    pub fn execution_error(&self, plan: &QueryPlan) -> QueryError {
        match &self.cause {
            RejectionCause::Validation(error) => error
                .facts()
                .iter()
                .fold(QueryError::new(error.code(), error.operation(), error.message()), |rebuilt, (key, value)| {
                    rebuilt.with_fact(key, value)
                }),
            RejectionCause::CompiledContract | RejectionCause::ExecutionMode => {
                compiled_bounded_pipeline_required_error(plan, self.pipeline.as_deref(), &self.rejection)
            },
        }
    }

    fn validation(error: QueryError) -> Self {
        let mut facts = error.facts().iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
        facts.insert("validation_code", error.code().as_str().to_string());
        facts.insert("validation_operation", error.operation().to_string());
        Self {
            rejection: Rejection {
                owner: RejectionOwner::Plan,
                reason: Reason::PlanValidation(UnsupportedPart::from_code(error.code())),
                facts,
            },
            pipeline: None,
            cause: RejectionCause::Validation(Arc::new(error)),
        }
    }
}

/// Decides once whether `plan` runs as a compiled pipeline under the given
/// execution mode and policy.
///
/// The plan is validated first. A malformed plan has no correct fallback and
/// is the validator's error; a well-formed plan that uses a type, operator,
/// or expression the backend does not support is rejected with the
/// validator's code and operation in the rejection facts.
pub fn admit(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
    ctx: AnalysisContext<'_>,
) -> Result<AdmissionDecision> {
    let ctx = ctx.with_policy(policy);
    if let Err(error) = validate_query_plan(plan, ctx) {
        if error.kind() != QueryErrorKind::Unsupported {
            return Err(error);
        }
        return Ok(AdmissionDecision::Rejected(AdmissionRejection::validation(error)));
    }
    let admission = capability_first_bounded_pipeline_admission(plan, ctx)?;
    if let Some(rejection) = statistical_aggregate_execution_mode_rejection(plan, statistical_aggregate_execution_mode)
    {
        return Ok(AdmissionDecision::Rejected(AdmissionRejection {
            rejection,
            pipeline: None,
            cause: RejectionCause::ExecutionMode,
        }));
    }
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => {
            let admitted =
                admitted_plan_from_capability_first(admission, plan, statistical_aggregate_execution_mode, policy)?;
            Ok(AdmissionDecision::Admitted(Arc::new(admitted)))
        },
        CapabilityFirstCandidateStatus::Rejected => {
            let rejection =
                admission.rejection().expect("rejected capability-first admission should carry a rejection").clone();
            Ok(AdmissionDecision::Rejected(AdmissionRejection {
                rejection,
                pipeline: admission.into_pipeline().map(Box::new),
                cause: RejectionCause::CompiledContract,
            }))
        },
    }
}

/// Rejects a plan whose floating statistical aggregates the selected execution
/// mode cannot run.
///
/// Only [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`]
/// narrows support: its host kernel reproduces DataFusion's accumulators for a
/// fixed set of group-key and aggregate types, so a frontend that may fall
/// back keeps the other types on DataFusion instead of failing during
/// execution.
fn statistical_aggregate_execution_mode_rejection(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Option<Rejection> {
    if !statistical_aggregate_execution_mode.is_datafusion_exact() {
        return None;
    }
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return None;
        };
        if !aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec) {
            return None;
        }
        let input = plan.node(aggregate.input)?;
        datafusion_exact_host_aggregate_rejection(node.id, &aggregate.spec, input.output_schema(), node.output_schema())
    })
}

fn datafusion_exact_host_aggregate_rejection(
    node_id: NodeId,
    spec: &AggregateSpec,
    input_schema: &Schema,
    output_schema: &Schema,
) -> Option<Rejection> {
    let mut output_types = output_schema.fields().iter().map(|field| field.data_type());
    for expr in &spec.group_exprs {
        let input_type = column_input_type(expr, input_schema);
        let output_type = output_types.next();
        if let (Some(input), Some(output)) = (input_type, output_type)
            && datafusion_exact_host_group_key_supported(input, output)
        {
            continue;
        }
        return Some(datafusion_exact_host_rejection(
            node_id,
            Reason::DatafusionExactStatisticalGroupKeyUnsupported,
            [
                ("group_key", expr.to_string()),
                ("input_data_type", data_type_label(input_type)),
                ("output_data_type", data_type_label(output_type)),
            ],
        ));
    }
    for expr in &spec.aggr_exprs {
        let input_type = column_input_type(&expr.expr, input_schema);
        let output_type = output_types.next();
        match output_type.and_then(|output| datafusion_exact_host_accumulator(expr, input_type, output)) {
            // DataFusion's ungrouped SUM and AVG add each input batch with
            // Arrow's lane-wise sum kernel, so their floating-point result
            // depends on batch boundaries; its grouped accumulators, like the
            // host kernel, add one row at a time.
            Some(DataFusionExactHostAccumulator::FloatSum | DataFusionExactHostAccumulator::Avg)
                if spec.group_exprs.is_empty() =>
            {
                return Some(datafusion_exact_host_rejection(
                    node_id,
                    Reason::DatafusionExactUngroupedFloatSumUnsupported,
                    [
                        ("aggregate_function", expr.function.as_str().to_string()),
                        ("aggregate_alias", expr.alias.clone()),
                    ],
                ));
            },
            Some(_) => continue,
            None => {},
        }
        let part = if expr.distinct || matches!(expr.function, AggregateFunction::NUnique | AggregateFunction::Median) {
            UnsupportedPart::Operator
        } else {
            UnsupportedPart::DataType
        };
        return Some(datafusion_exact_host_rejection(
            node_id,
            Reason::DatafusionExactStatisticalAggregateUnsupported(part),
            [
                ("aggregate_function", expr.function.as_str().to_string()),
                ("aggregate_alias", expr.alias.clone()),
                ("distinct", expr.distinct.to_string()),
                ("input_data_type", data_type_label(input_type)),
                ("output_data_type", data_type_label(output_type)),
            ],
        ));
    }
    None
}

fn datafusion_exact_host_rejection(
    node_id: NodeId,
    reason: Reason,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> Rejection {
    let mut facts = BTreeMap::from_iter(facts);
    facts.insert(
        "statistical_aggregate_execution_mode",
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford.as_str().to_string(),
    );
    Rejection { owner: RejectionOwner::Node(node_id), reason, facts }
}

fn column_input_type<'a>(expr: &Expr, schema: &'a Schema) -> Option<&'a ArrowDataType> {
    match expr {
        Expr::Column(column) => schema.field_with_name(column.name()).ok().map(|field| field.data_type()),
        Expr::Alias { expr, .. } => column_input_type(expr, schema),
        _ => None,
    }
}

fn data_type_label(data_type: Option<&ArrowDataType>) -> String {
    data_type.map_or_else(|| "none".to_string(), |data_type| format!("{data_type:?}"))
}

fn admitted_plan_from_capability_first(
    admission: CapabilityFirstPipelineAdmission,
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
) -> Result<AdmittedPlan> {
    let pipeline = admission.into_pipeline().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.capability_admission",
            "admitted capability candidate must include a compiled pipeline",
        )
        .with_fact("execution_mode", CapabilityEnvelopeStatus::Admitted.as_str())
        .with_fact("candidate_status", "admitted")
    })?;
    Ok(AdmittedPlan {
        fingerprint: plan.execution_fingerprint(),
        plan: plan.clone(),
        pipeline,
        statistical_aggregate_execution_mode,
        policy,
    })
}

fn compiled_bounded_pipeline_required_error(
    plan: &QueryPlan,
    pipeline: Option<&CompiledPipeline>,
    rejection: &Rejection,
) -> QueryError {
    let sink_kind = match plan.sink() {
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    };
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPipelineContract,
        "query_engine.bounded.compiled_pipeline_required",
        "bounded execution requires an admitted compiled pipeline",
    )
    .with_fact("compiled_pipeline_required", "true")
    .with_fact("compiled_pipeline_rejection_reason", rejection.reason)
    .with_fact("result_sink_kind", sink_kind);

    if let Some(pipeline) = pipeline {
        error = error
            .with_fact("compiled_task_count", pipeline.tasks().len())
            .with_fact("compiled_edge_count", pipeline.edges().len())
            .with_fact("compiled_barrier_count", pipeline.barriers().len())
            .with_fact("compiled_resource_count", pipeline.resources().len())
            .with_fact("compiled_sink_kind", pipeline.sink().kind().as_str());
    }

    // Rejection facts may repeat pipeline-derived keys; the pipeline values
    // above are authoritative, so only absent keys are added here.
    for (key, value) in &rejection.facts {
        error = error.with_fact_if_absent(key, value.as_str());
    }

    error
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::AggregateExpr,
        plan::QueryPlanBuilder,
        source::{LocalParquetSource, SourceNode},
    };

    fn statistical_aggregate_plan(extra: AggregateExpr, extra_output: Field) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("quantity", DataType::Int32, true),
            Field::new("amount", DataType::Int64, true),
            Field::new("sold_date", DataType::Date32, true),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("quantity_count", DataType::Int64, false),
            Field::new("quantity_avg", DataType::Float64, true),
            Field::new("quantity_stddev", DataType::Float64, true),
            extra_output,
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/sales.parquet",
            Arc::clone(&input_schema),
            input_schema.fields().iter().map(|field| field.name().clone()).collect(),
        )));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("key")],
                    vec![
                        AggregateExpr::new(AggregateFunction::Count, Expr::column("quantity"), "quantity_count"),
                        AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "quantity_avg"),
                        AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("quantity"), "quantity_stddev"),
                        extra,
                    ],
                ),
                output_schema,
            )
            .unwrap();
        builder.finish_with_host_sink(aggregate).unwrap()
    }

    #[test]
    fn exact_statistical_mode_rejects_ungrouped_float_sums_that_datafusion_adds_by_batch() {
        let exact = StatisticalAggregateExecutionMode::DataFusionExactHostWelford;
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("quantity", DataType::Int32, true),
            Field::new("amount", DataType::Int64, true),
        ]));
        let ungrouped = |extra: AggregateExpr, extra_output: Field| {
            let mut builder = QueryPlanBuilder::new();
            let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/sales.parquet",
                Arc::clone(&input_schema),
                vec!["quantity".to_string(), "amount".to_string()],
            )));
            let aggregate = builder
                .add_aggregate(
                    source,
                    AggregateSpec::new(
                        Vec::new(),
                        vec![
                            AggregateExpr::new(
                                AggregateFunction::StddevSamp,
                                Expr::column("quantity"),
                                "quantity_stddev",
                            ),
                            extra,
                        ],
                    ),
                    Arc::new(Schema::new(vec![Field::new("quantity_stddev", DataType::Float64, true), extra_output])),
                )
                .unwrap();
            builder.finish_with_host_sink(aggregate).unwrap()
        };

        let avg = ungrouped(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "quantity_avg"),
            Field::new("quantity_avg", DataType::Float64, true),
        );
        let rejection = statistical_aggregate_execution_mode_rejection(&avg, exact)
            .expect("an ungrouped AVG sums by batch in DataFusion");
        assert_eq!(rejection.reason, Reason::DatafusionExactUngroupedFloatSumUnsupported);
        assert_eq!(rejection.facts.get("aggregate_alias").map(String::as_str), Some("quantity_avg"));

        let integer_sum = ungrouped(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum"),
            Field::new("amount_sum", DataType::Int64, true),
        );
        assert!(statistical_aggregate_execution_mode_rejection(&integer_sum, exact).is_none());
    }

    #[test]
    fn exact_statistical_mode_rejects_only_aggregates_outside_the_host_kernel() {
        let exact = StatisticalAggregateExecutionMode::DataFusionExactHostWelford;
        let integer_sum = statistical_aggregate_plan(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum"),
            Field::new("amount_sum", DataType::Int64, true),
        );
        assert!(statistical_aggregate_execution_mode_rejection(&integer_sum, exact).is_none());

        let date_min = statistical_aggregate_plan(
            AggregateExpr::new(AggregateFunction::Min, Expr::column("sold_date"), "first_sold_date"),
            Field::new("first_sold_date", DataType::Date32, true),
        );
        let rejection = statistical_aggregate_execution_mode_rejection(&date_min, exact)
            .expect("the exact host kernel has no Date32 MIN accumulator");
        assert_eq!(rejection.reason, Reason::DatafusionExactStatisticalAggregateUnsupported(UnsupportedPart::DataType));
        assert_eq!(rejection.facts.get("aggregate_alias").map(String::as_str), Some("first_sold_date"));
        assert!(
            statistical_aggregate_execution_mode_rejection(
                &date_min,
                StatisticalAggregateExecutionMode::GpuNativeTolerant
            )
            .is_none()
        );
    }
}
