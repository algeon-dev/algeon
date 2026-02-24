//! Capability-first bounded-pipeline admission shared by strict and adaptive facades.

use super::compiled::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    capability::CapabilityAnalysis,
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
    candidate_status: CapabilityFirstCandidateStatus,
    rejection_reason: Option<&'static str>,
}

impl CapabilityFirstPipelineAdmission {
    pub(in crate::exec::bounded) fn analysis(&self) -> &CapabilityAnalysis {
        &self.analysis
    }

    pub(in crate::exec::bounded) fn pipeline(&self) -> Option<&CompiledPipeline> {
        self.pipeline.as_ref()
    }

    pub(in crate::exec::bounded) fn candidate_status(&self) -> CapabilityFirstCandidateStatus {
        self.candidate_status
    }

    pub(in crate::exec::bounded) fn rejection_reason(&self) -> Option<&'static str> {
        self.rejection_reason
    }

    pub(in crate::exec::bounded) fn into_parts(self) -> (CapabilityAnalysis, Option<CompiledPipeline>) {
        (self.analysis, self.pipeline)
    }

    fn admitted(analysis: CapabilityAnalysis, pipeline: CompiledPipeline) -> Self {
        Self {
            analysis,
            pipeline: Some(pipeline),
            candidate_status: CapabilityFirstCandidateStatus::Admitted,
            rejection_reason: None,
        }
    }

    fn rejected(
        analysis: CapabilityAnalysis,
        pipeline: Option<CompiledPipeline>,
        rejection_reason: &'static str,
    ) -> Self {
        Self {
            analysis,
            pipeline,
            candidate_status: CapabilityFirstCandidateStatus::Rejected,
            rejection_reason: Some(rejection_reason),
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
    if analysis.has_rejections() {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, "missing_strict_contracts")));
    }
    if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, "unsupported_sink_boundary")));
    }

    let pipeline = match compile_pipeline(plan, &analysis) {
        Ok(pipeline) => pipeline,
        Err(error) if compiled_pipeline_graph_is_unsupported_envelope(&error) => {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(
                analysis,
                None,
                "unsupported_compiled_pipeline_envelope",
            )));
        },
        Err(error) => return Err(error),
    };
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(analysis, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported => Ok(Some(CapabilityFirstPipelineAdmission::rejected(
            analysis,
            Some(pipeline),
            "unsupported_compiled_pipeline_envelope",
        ))),
    }
}

fn compiled_pipeline_graph_is_unsupported_envelope(error: &QueryError) -> bool {
    error.code() == QueryErrorCode::InvalidPlan
        && matches!(error.operation(), "cudf_query_engine.pipeline.compile" | "cudf_query_engine.pipeline.validate")
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
        assert_eq!(compile_error.operation(), "cudf_query_engine.pipeline.validate");

        let admission = capability_first_bounded_pipeline_admission(&plan)
            .expect("admission should classify the cycle as an unsupported envelope")
            .expect("join plan should be considered by bounded admission");

        assert!(matches!(admission.candidate_status(), CapabilityFirstCandidateStatus::Rejected));
        assert_eq!(admission.rejection_reason(), Some("unsupported_compiled_pipeline_envelope"));
        assert!(admission.pipeline().is_none());
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
