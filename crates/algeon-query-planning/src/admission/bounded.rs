//! Admission boundary for compiled bounded pipelines.
//!
//! Capability-first bounded-pipeline admission evidence and rejection
//! projection. This is the single bounded admission contract; "capability"
//! appears only in preserved wire names. Runtime execution remains in
//! `exec::bounded`.

use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType as ArrowDataType, Schema};

use super::shared::{
    CapabilityFirstCandidateStatus, CapabilityFirstPipelineAdmission, capability_first_bounded_pipeline_admission,
};
use crate::{
    backend::AnalysisContext,
    capability::{
        CapabilityRejectionCategory, CapabilityRejectionOwner, CapabilityRejectionProjection, CapabilityScope,
    },
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateFunction, AggregateSpec, Expr},
    mode::{CapabilityEnvelopeStatus, StatisticalAggregateExecutionMode},
    pipeline::CompiledPipeline,
    plan::{NodeId, PlanFingerprint, PlanNode, QueryPlan, ResultSink},
    policy::PlanningPolicy,
    prepared::{
        DataFusionExactHostAccumulator, aggregate_spec_has_floating_statistical_aggregate,
        datafusion_exact_host_accumulator, datafusion_exact_host_group_key_supported,
    },
};

/// A plan that passed capability analysis, compilation and the compiled
/// contracts exactly once; the frontends validate a plan before admitting it.
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
        match admit(&pruned, self.statistical_aggregate_execution_mode, ctx.with_policy(self.policy))? {
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
    pub fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        match self {
            Self::Admitted(_) => None,
            Self::Rejected(rejection) => Some(rejection.projection()),
        }
    }
}

/// Why a plan has no compiled execution.
#[derive(Debug, Clone)]
pub struct AdmissionRejection {
    projection: CapabilityRejectionProjection,
    pipeline: Option<CompiledPipeline>,
    execution_mode_refusal: bool,
}

impl AdmissionRejection {
    /// The first operator, edge, or pipeline contract the plan misses.
    pub fn projection(&self) -> &CapabilityRejectionProjection {
        &self.projection
    }

    /// Whether the requested statistical aggregate execution mode, rather than
    /// a compiled contract, refuses the plan.
    pub fn is_execution_mode_refusal(&self) -> bool {
        self.execution_mode_refusal
    }

    /// The typed error a caller that must execute `plan` reports.
    pub fn execution_error(&self, plan: &QueryPlan) -> QueryError {
        compiled_bounded_pipeline_required_error(plan, self.pipeline.as_ref(), &self.projection)
    }
}

/// Decides once whether `plan` runs as a compiled pipeline under the given
/// execution mode and the policy `ctx` carries.
pub fn admit(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    ctx: AnalysisContext<'_>,
) -> Result<AdmissionDecision> {
    let admission = capability_first_bounded_pipeline_admission(plan, ctx)?;
    if let Some(projection) = statistical_aggregate_execution_mode_rejection(plan, statistical_aggregate_execution_mode)
    {
        return Ok(AdmissionDecision::Rejected(AdmissionRejection {
            projection,
            pipeline: None,
            execution_mode_refusal: true,
        }));
    }
    match admission.candidate_status() {
        CapabilityFirstCandidateStatus::Admitted => {
            let admitted = admitted_plan_from_capability_first(
                admission,
                plan,
                statistical_aggregate_execution_mode,
                ctx.policy(),
            )?;
            Ok(AdmissionDecision::Admitted(Arc::new(admitted)))
        },
        CapabilityFirstCandidateStatus::Rejected => {
            let projection = admission
                .rejection_projection()
                .expect("rejected capability-first admission should carry a projection")
                .clone();
            Ok(AdmissionDecision::Rejected(AdmissionRejection {
                projection,
                pipeline: admission.into_pipeline(),
                execution_mode_refusal: false,
            }))
        },
    }
}

/// Rejects a plan whose floating statistical aggregates the selected execution
/// mode cannot run.
///
/// Only [`StatisticalAggregateExecutionMode::DataFusionExactHostWelford`]
/// narrows support: its host kernel reproduces DataFusion's accumulators for a
/// fixed set of group-key and aggregate types. Frontends call this before
/// replacing a DataFusion plan so the other types stay on DataFusion instead
/// of failing during execution.
fn statistical_aggregate_execution_mode_rejection(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Option<CapabilityRejectionProjection> {
    if !statistical_aggregate_execution_mode.is_datafusion_exact() {
        return None;
    }
    let unreachable = plan.unreachable_node_ids();
    plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).find_map(|node| {
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
) -> Option<CapabilityRejectionProjection> {
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
            CapabilityRejectionCategory::UnsupportedArrowDtype,
            "datafusion_exact_statistical_group_key_unsupported",
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
                    CapabilityRejectionCategory::UnsupportedNodeOperator,
                    "datafusion_exact_ungrouped_float_sum_unsupported",
                    [
                        ("aggregate_function", expr.function.as_str().to_string()),
                        ("aggregate_alias", expr.alias.clone()),
                    ],
                ));
            },
            Some(_) => continue,
            None => {},
        }
        let category =
            if expr.distinct || matches!(expr.function, AggregateFunction::NUnique | AggregateFunction::Median) {
                CapabilityRejectionCategory::UnsupportedNodeOperator
            } else {
                CapabilityRejectionCategory::UnsupportedArrowDtype
            };
        return Some(datafusion_exact_host_rejection(
            node_id,
            category,
            "datafusion_exact_statistical_aggregate_unsupported",
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
    category: CapabilityRejectionCategory,
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityRejectionProjection {
    let mut facts = BTreeMap::from_iter(facts);
    facts.insert(
        "statistical_aggregate_execution_mode",
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford.as_str().to_string(),
    );
    CapabilityRejectionProjection {
        owner: CapabilityRejectionOwner::Node(node_id),
        category,
        contract_scope: CapabilityScope::Node,
        boundary: "datafusion_exact_statistical_aggregate",
        reason,
        facts,
    }
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
    projection: &CapabilityRejectionProjection,
) -> QueryError {
    let sink_kind = match plan.sink() {
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    };
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "query_engine.bounded.compiled_pipeline_required",
        "bounded execution requires an admitted compiled pipeline",
    )
    .with_fact("compiled_pipeline_required", "true")
    .with_fact("compiled_pipeline_rejection_reason", projection.reason)
    .with_fact("result_sink_kind", sink_kind);

    if let Some(pipeline) = pipeline {
        error = error
            .with_fact("compiled_task_count", pipeline.tasks().len())
            .with_fact("compiled_edge_count", pipeline.edges().len())
            .with_fact("compiled_barrier_count", pipeline.barriers().len())
            .with_fact("compiled_resource_count", pipeline.resources().len())
            .with_fact("compiled_sink_kind", pipeline.sink().kind().as_str());
    }

    // Projection facts may repeat pipeline-derived keys; the pipeline values
    // above are authoritative, so only absent keys are added here.
    for (key, value) in &projection.facts {
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
        let aggregate = builder.add_aggregate(
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
        );
        builder.finish_with_host_sink(aggregate)
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
            let aggregate = builder.add_aggregate(
                source,
                AggregateSpec::new(
                    Vec::new(),
                    vec![
                        AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("quantity"), "quantity_stddev"),
                        extra,
                    ],
                ),
                Arc::new(Schema::new(vec![Field::new("quantity_stddev", DataType::Float64, true), extra_output])),
            );
            builder.finish_with_host_sink(aggregate)
        };

        let avg = ungrouped(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "quantity_avg"),
            Field::new("quantity_avg", DataType::Float64, true),
        );
        let rejection = statistical_aggregate_execution_mode_rejection(&avg, exact)
            .expect("an ungrouped AVG sums by batch in DataFusion");
        assert_eq!(rejection.reason, "datafusion_exact_ungrouped_float_sum_unsupported");
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
        assert_eq!(rejection.reason, "datafusion_exact_statistical_aggregate_unsupported");
        assert!(matches!(rejection.category, CapabilityRejectionCategory::UnsupportedArrowDtype));
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
