//! Capability-first bounded-pipeline admission shared by strict and adaptive facades.

use std::collections::BTreeMap;

use super::compiled::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    capability::{
        CapabilityAnalysis, CapabilityEnvelopeAdmission, CapabilityRejectionCategory, CapabilityRejectionOwner,
        CapabilityRejectionProjection, ReservationRequirementScope, StrictBoundedEnvelope,
    },
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{CompiledPipeline, compile_pipeline},
    plan::{QueryPlan, ResultSink},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded) enum CapabilityFirstCandidateStatus {
    Admitted,
    Rejected,
}

#[derive(Debug, Clone)]
pub(in crate::exec::bounded) struct CapabilityFirstPipelineAdmission {
    analysis: CapabilityAnalysis,
    pipeline: Option<CompiledPipeline>,
    envelope: Option<StrictBoundedEnvelope>,
    rejection_projection: Option<CapabilityRejectionProjection>,
    candidate_status: CapabilityFirstCandidateStatus,
}

impl CapabilityFirstPipelineAdmission {
    pub(in crate::exec::bounded) fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    pub(in crate::exec::bounded) fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    pub(in crate::exec::bounded) fn envelope(&self) -> Option<StrictBoundedEnvelope> {
        self.envelope
    }

    pub(in crate::exec::bounded) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(in crate::exec::bounded) fn rejection_reason(&self) -> Option<&'static str> {
        self.rejection_projection.as_ref().map(|projection| projection.reason)
    }

    pub(in crate::exec::bounded) fn rejection_projection(&self) -> Option<&CapabilityRejectionProjection> {
        self.rejection_projection.as_ref()
    }

    pub(in crate::exec::bounded) fn into_parts(self) -> (CapabilityAnalysis, Option<CompiledPipeline>) {
        (self.analysis, self.pipeline)
    }

    fn admitted(analysis: CapabilityAnalysis, envelope: StrictBoundedEnvelope, pipeline: CompiledPipeline) -> Self {
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

pub(in crate::exec::bounded) fn capability_first_bounded_pipeline_admission(
    plan: &QueryPlan,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    let envelope = match analysis.strict_bounded_envelope_admission() {
        CapabilityEnvelopeAdmission::Admitted { envelope } => envelope,
        CapabilityEnvelopeAdmission::Rejected { projection } => {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
        },
    };
    if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        let projection = admission_projection(
            CapabilityRejectionCategory::UnsupportedSinkBoundary,
            "result_sink",
            "unsupported_sink_boundary",
            [("result_sink_kind", result_sink_kind(plan.sink()).to_string())],
        );
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
    }

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

fn result_sink_kind(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
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
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlan, QueryPlanBuilder},
        source::{LocalParquetSource, SourceNode},
    };

    #[test]
    fn strict_admission_demotes_dynamic_filter_control_cycle_to_unsupported_envelope() {
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
            plan.bounded_join_modeled_facts(crate::runtime::ExecutionBoundaryPolicy::default()).is_none(),
            "unsupported compiled envelopes must not emit modeled bounded-join facts"
        );
        let unsupported_facts = plan
            .bounded_join_unsupported_shape_facts(crate::runtime::ExecutionBoundaryPolicy::default())
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
