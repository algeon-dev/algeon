//! Capability-first bounded-pipeline admission with one decision path.

use std::collections::BTreeMap;

use super::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    capability::{
        CapabilityAnalysis, CapabilityEnvelope, CapabilityEnvelopeAdmission, CapabilityRejectionCategory,
        CapabilityRejectionOwner, CapabilityRejectionProjection, ContractQuality, ContractStatus,
        ReservationRequirement, ReservationRequirementScope,
    },
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::QueryPlan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CapabilityFirstCandidateStatus {
    Admitted,
    Rejected,
}

#[derive(Debug, Clone)]
pub(super) struct CapabilityFirstPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: Option<CompiledPipeline>,
    envelope: Option<CapabilityEnvelope>,
    rejection_projection: Option<CapabilityRejectionProjection>,
    candidate_status: CapabilityFirstCandidateStatus,
}

impl CapabilityFirstPipelineAdmission {
    pub(super) fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    pub(super) fn envelope(&self) -> Option<CapabilityEnvelope> {
        self.envelope
    }

    pub(super) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(super) fn rejection_reason(&self) -> Option<&'static str> {
        self.rejection_projection.as_ref().map(|projection| projection.reason)
    }

    pub(super) fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        self.rejection_projection.as_ref()
    }

    pub(super) fn into_parts(self) -> (CapabilityAnalysis, Option<CompiledPipeline>) {
        (self.analysis, self.pipeline)
    }

    fn admitted(analysis: CapabilityAnalysis, envelope: CapabilityEnvelope, pipeline: CompiledPipeline) -> Self {
        Self {
            analysis,
            pipeline: Some(pipeline),
            envelope: Some(envelope),
            rejection_projection: None,
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
        }
    }

    fn rejected(
        analysis: CapabilityAnalysis,
        pipeline: Option<CompiledPipeline>,
        projection: CapabilityRejectionProjection,
    ) -> Self {
        Self {
            analysis,
            pipeline,
            envelope: None,
            rejection_projection: Some(projection),
            candidate_status: CapabilityFirstCandidateStatus::Rejected,
        }
    }
}

pub(super) fn capability_first_bounded_pipeline_admission(
    plan: &QueryPlan,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    capability_first_bounded_pipeline_admission_impl(plan, false)
}

pub(super) fn capability_first_bounded_cugraph_pipeline_admission(
    plan: &QueryPlan,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    capability_first_bounded_pipeline_admission_impl(plan, true)
}

fn capability_first_bounded_pipeline_admission_impl(
    plan: &QueryPlan,
    cugraph_execution_attested: bool,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    if !plan.contains_join() && !plan.contains_vector_algorithm() {
        return Ok(None);
    }

    let mut analysis = plan.capability_analysis();
    if cugraph_execution_attested {
        rewrite_graph_requirements_for_cugraph_execution(&mut analysis);
    }
    let envelope = match analysis.capability_envelope_admission() {
        CapabilityEnvelopeAdmission::Admitted { envelope } => envelope,
        CapabilityEnvelopeAdmission::Rejected { projection } => {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
        },
    };
    let pipeline = match compile_pipeline(plan, &analysis) {
        Ok(pipeline) => pipeline,
        Err(error) if compiled_pipeline_graph_is_unsupported_envelope(&error) => {
            let projection = admission_projection(
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
                "compiled_pipeline",
                "unsupported_compiled_pipeline_envelope",
                [("pipeline_compile_error", error.message().to_string())],
            );
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
        },
        Err(error) => return Err(error),
    };
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(analysis, envelope, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported => {
            let projection = admission_projection(
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
                "compiled_pipeline",
                "unsupported_compiled_pipeline_envelope",
                [
                    ("compiled_task_count", pipeline.tasks().len().to_string()),
                    ("compiled_edge_count", pipeline.edges().len().to_string()),
                ],
            );
            Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, Some(pipeline), projection)))
        },
    }
}

fn rewrite_graph_requirements_for_cugraph_execution(analysis: &mut CapabilityAnalysis) {
    let replacement = ReservationRequirement::modeled(
        crate::execution_contracts::ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        ContractQuality::CapabilityProof,
    );
    for node in &mut analysis.nodes {
        for requirement in &mut node.local_requirements {
            if requirement.status == ContractStatus::BlockedByLowerLevel
                && requirement.required_contract == crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT
            {
                *requirement = replacement.clone();
            }
        }
    }
    analysis.requirements = analysis
        .nodes
        .iter()
        .flat_map(|node| node.local_requirements.iter())
        .chain(analysis.edges.iter().flat_map(|edge| edge.reservation_requirements.iter()))
        .chain(analysis.invariants.iter().flat_map(|invariant| invariant.reservation_requirements.iter()))
        .cloned()
        .collect();
}

fn admission_projection(
    category: CapabilityRejectionCategory,
    memory_boundary: &'static str,
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityRejectionProjection {
    CapabilityRejectionProjection {
        owner: CapabilityRejectionOwner::Admission(reason),
        category,
        contract_scope: ReservationRequirementScope::Subgraph,
        memory_boundary,
        reason,
        facts: BTreeMap::from_iter(facts),
    }
}

fn compiled_pipeline_graph_is_unsupported_envelope(error: &QueryError) -> bool {
    error.code() == QueryErrorCode::InvalidPlan
        && matches!(error.operation(), "nexus_query_engine.pipeline.compile" | "nexus_query_engine.pipeline.validate")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        dynamic_filter::{
            DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
            DynamicFilterProducer, DynamicFilterRoute,
        },
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, ColumnRef, Expr, SortExpr, SortSpec},
        pipeline::{PipelineTaskKind, SinkBoundaryKind},
        plan::{
            JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, ProjectionOutput, QueryPlan, QueryPlanBuilder,
        },
        source::{LocalParquetSource, SourceNode},
    };

    #[test]
    fn capability_admission_accepts_root_full_sort_left_semi_grouped_aggregate() {
        let plan = root_full_sort_left_semi_grouped_aggregate_plan();

        let admission = capability_first_bounded_pipeline_admission(&plan)
            .expect("left-semi full-sort aggregate admission should not error")
            .expect("join plan should produce capability admission evidence");

        assert!(
            matches!(admission.candidate_status(), CapabilityFirstCandidateStatus::Admitted),
            "rejection={:?}",
            admission.rejection_projection()
        );
        assert_eq!(admission.envelope(), Some(CapabilityEnvelope::RootFullSortJoinAggregate));
        assert!(admission.pipeline().is_some());
    }

    #[test]
    fn capability_admission_accepts_gpu_resident_sink_for_join_pipeline() {
        let plan = root_full_sort_left_semi_grouped_aggregate_plan().clone_with_gpu_resident_sink();

        let admission = capability_first_bounded_pipeline_admission(&plan)
            .expect("gpu-resident sink admission should not error")
            .expect("join plan should produce capability admission evidence");

        assert!(
            matches!(admission.candidate_status(), CapabilityFirstCandidateStatus::Admitted),
            "rejection={:?}",
            admission.rejection_projection()
        );
        let pipeline = admission.pipeline().expect("admitted plan should have a compiled pipeline");
        assert_eq!(pipeline.sink().kind(), SinkBoundaryKind::GpuResidentOutput);
        assert!(
            pipeline.tasks().iter().all(|task| task.kind() != PipelineTaskKind::SinkMaterialize),
            "gpu-resident sink must not compile a host materialization task"
        );
    }

    #[test]
    fn capability_admission_demotes_dynamic_filter_control_cycle_to_unsupported_envelope() {
        let plan = dynamic_filter_control_cycle_plan();
        let analysis = plan.capability_analysis();
        let compile_error = compile_pipeline(&plan, &analysis).expect_err("raw compiler should reject the cycle");

        assert_eq!(compile_error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(compile_error.operation(), "nexus_query_engine.pipeline.validate");

        let admission = capability_first_bounded_pipeline_admission(&plan)
            .expect("admission should classify the cycle as an unsupported envelope")
            .expect("join plan should be considered by bounded admission");

        assert!(matches!(admission.candidate_status(), CapabilityFirstCandidateStatus::Rejected));
        assert_eq!(admission.rejection_reason(), Some("unsupported_compiled_pipeline_envelope"));
        assert!(admission.pipeline().is_none());

        assert!(
            crate::admission::bounded_join_modeled_facts(
                &plan,
                crate::execution_contracts::ExecutionBoundaryPolicy::default()
            )
            .is_none(),
            "unsupported compiled envelopes must not emit modeled bounded-join facts"
        );
        let unsupported_facts = crate::admission::bounded_join_unsupported_shape_facts(
            &plan,
            crate::execution_contracts::ExecutionBoundaryPolicy::default(),
        )
        .expect("unsupported compiled envelope should emit unsupported bounded-join facts");
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_reason").map(String::as_str),
            Some("unsupported_compiled_pipeline_envelope")
        );
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_boundary").map(String::as_str),
            Some("compiled_pipeline")
        );
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_category").map(String::as_str),
            Some("unsupported_compiled_pipeline_envelope")
        );
        assert!(
            unsupported_facts
                .get("bounded_join_rejection_facts")
                .is_some_and(|facts| facts.contains("pipeline_compile_error="))
        );
    }

    fn root_full_sort_left_semi_grouped_aggregate_plan() -> QueryPlan {
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("order_key", DataType::Int64, false),
            Field::new("priority", DataType::Utf8, false),
        ]));
        let right_schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int64, false)]));
        let aggregate_schema = Arc::new(Schema::new(vec![
            Field::new("priority", DataType::Utf8, false),
            Field::new("order_count", DataType::UInt64, false),
        ]));

        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/orders.parquet",
            Arc::clone(&left_schema),
            vec!["order_key".to_string(), "priority".to_string()],
        )));
        let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/lineitem.parquet",
            Arc::clone(&right_schema),
            vec!["order_key".to_string()],
        )));
        let semi = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::LeftSemi,
                vec![JoinKey::new(
                    ColumnRef::input(0, "order_key", "order_key"),
                    ColumnRef::input(1, "order_key", "order_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_key", "order_key"), "order_key"),
                JoinOutputColumn::left(ColumnRef::input(0, "priority", "priority"), "priority"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&left_schema),
        );
        let aggregate = builder.add_aggregate(
            semi,
            AggregateSpec::new(
                vec![Expr::column("priority")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_key"), "order_count")],
            ),
            Arc::clone(&aggregate_schema),
        );
        let projection = builder.add_projection(
            aggregate,
            vec![
                ProjectionOutput::new(Expr::column("priority"), Field::new("priority", DataType::Utf8, false)),
                ProjectionOutput::new(Expr::column("order_count"), Field::new("order_count", DataType::UInt64, false)),
            ],
        );
        let sort = builder.add_sort(
            projection,
            SortSpec::new(vec![SortExpr::new(Expr::column("priority"), false, false)]),
            aggregate_schema,
        );
        builder.finish_with_host_sink(sort)
    }

    fn dynamic_filter_control_cycle_plan() -> QueryPlan {
        let source_schema = Arc::new(Schema::new(vec![Field::new("shared_key", DataType::Int32, false)]));
        let build_schema = Arc::new(Schema::new(vec![Field::new("build_key", DataType::Int32, false)]));
        let mut builder = QueryPlanBuilder::new();
        let shared_probe = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/shared-probe.parquet",
            Arc::clone(&source_schema),
            vec!["shared_key".to_string()],
        )));
        let build_seed = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/build-seed.parquet",
            Arc::clone(&build_schema),
            vec!["build_key".to_string()],
        )));
        let build_subtree = builder.add_join(
            build_seed,
            shared_probe,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "build_key", "build_key"),
                    ColumnRef::input(1, "shared_key", "shared_key"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "build_key", "build_key"),
                "build_key",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&build_schema),
        );
        let root = builder.add_join(
            shared_probe,
            build_subtree,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "shared_key", "shared_key"),
                    ColumnRef::input(1, "build_key", "build_key"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "shared_key", "shared_key"),
                "shared_key",
            )])
            .with_build_side(JoinSide::Right),
            source_schema,
        );
        let plan = builder.finish_with_host_sink(root);
        plan.with_dynamic_filter_routes(vec![DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::HashJoinBuild {
                join_node_id: root,
                build_side: JoinSide::Right,
                build_input: build_subtree,
            },
            DynamicFilterConsumer::LocalParquetSource { source_node_id: shared_probe },
            vec!["build_key".to_string()],
            vec!["shared_key".to_string()],
            vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains],
            DynamicFilterConsumerLowering::ReaderPredicate,
        )])
    }
}
