//! Capability-first bounded-pipeline admission with one decision path.

use super::{CompiledBoundedPipelineSupport, compiled_bounded_query_plan_support};
use crate::{
    capability::{CapabilityAnalysis, CapabilityEnvelope, CapabilityEnvelopeAdmission, CapabilityRejectionProjection},
    error::Result,
    pipeline::{CompiledPipeline, PipelineCompilation, PipelineTaskKind, ResidentDataForm, compile_pipeline_outcome},
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
    capability_first_bounded_pipeline_admission_impl(plan)
}

fn capability_first_bounded_pipeline_admission_impl(
    plan: &QueryPlan,
) -> Result<Option<CapabilityFirstPipelineAdmission>> {
    if !plan.contains_join() && !plan.contains_vector_algorithm() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    if let Some(projection) = analysis.first_local_rejection_projection() {
        return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
    }
    let envelope_admission = analysis.capability_envelope_admission();
    let pipeline = match compile_pipeline_outcome(plan, &analysis)? {
        PipelineCompilation::Compiled(pipeline) => pipeline,
        // The compiler is the authority on constructability: when it cannot
        // build a pipeline, its structural reason (e.g. a dependency cycle)
        // reports the failure, not a plan-family envelope verdict about a
        // pipeline that does not exist.
        PipelineCompilation::Rejected(projection) => {
            return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, None, projection)));
        },
    };
    let envelope = if pipeline_is_within_migrated_execution_scope(&pipeline) {
        analysis.capability_envelope_label()
    } else {
        match envelope_admission {
            CapabilityEnvelopeAdmission::Admitted { envelope } => envelope,
            CapabilityEnvelopeAdmission::Rejected { projection } => {
                return Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, Some(pipeline), projection)));
            },
        }
    };
    match compiled_bounded_query_plan_support(plan, &analysis, &pipeline)? {
        CompiledBoundedPipelineSupport::Supported => {
            Ok(Some(CapabilityFirstPipelineAdmission::admitted(analysis, envelope, pipeline)))
        },
        CompiledBoundedPipelineSupport::Unsupported(projection) => {
            Ok(Some(CapabilityFirstPipelineAdmission::rejected(analysis, Some(pipeline), projection)))
        },
    }
}

/// Execution capabilities the compiled task and resource contracts already
/// decide on their own.
///
/// Join task groups with unary, limit, final-sort, and aggregate stages are
/// admitted by those contracts alone: the named plan-family recognisers keep
/// no veto over such a pipeline and only supply its compatibility label. A
/// resource fanned out to several consumers is covered when it is produced by
/// a source read or is a complete immutable frame: frames are refcounted
/// snapshots, so every consumer borrows and releases them through the same
/// path as shared source reads, with no cursor to alias (dynamic-filter
/// application is a no-op for non-source frames, and the contract graph still
/// refuses unresolvable sharing at compile time). Everything else still needs
/// a capability no compiled contract carries yet — windowed and union
/// composition, shared frame streams, and which join kernels accept another
/// join's retained output — so it keeps the recogniser gate until those
/// contracts exist. Delete this function, and the branch that calls it, once
/// they do.
fn pipeline_is_within_migrated_execution_scope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinMaterialize).count() >= 1
        && pipeline.tasks().iter().all(|task| {
            matches!(
                task.kind(),
                PipelineTaskKind::SourceRead
                    | PipelineTaskKind::GpuResidentSource
                    | PipelineTaskKind::LiteralBatchSource
                    | PipelineTaskKind::Unary
                    | PipelineTaskKind::Limit
                    | PipelineTaskKind::FinalSort
                    | PipelineTaskKind::Aggregate
                    | PipelineTaskKind::JoinBuild
                    | PipelineTaskKind::JoinProbe
                    | PipelineTaskKind::JoinMaterialize
                    | PipelineTaskKind::SinkMaterialize
            )
        })
        && pipeline.resources().iter().all(|resource| {
            resource.consumers().len() <= 1
                || matches!(resource.data_form(), ResidentDataForm::Frame)
                || pipeline.tasks().iter().any(|task| {
                    task.id() == resource.producer()
                        && matches!(
                            task.kind(),
                            PipelineTaskKind::SourceRead
                                | PipelineTaskKind::GpuResidentSource
                                | PipelineTaskKind::LiteralBatchSource
                        )
                })
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        capability::CapabilityRejectionOwner,
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
    fn capability_admission_reports_dynamic_filter_control_cycle_as_a_dependency_cycle() {
        let plan = dynamic_filter_control_cycle_plan();

        let admission = capability_first_bounded_pipeline_admission(&plan)
            .expect("admission should classify the control cycle")
            .expect("join plan should be considered by bounded admission");

        assert!(matches!(admission.candidate_status(), CapabilityFirstCandidateStatus::Rejected));
        let projection =
            admission.rejection_projection().expect("a rejected candidate carries its rejection projection");
        assert_eq!(projection.reason, "compiled_pipeline_dependency_cycle");
        assert_eq!(projection.owner, CapabilityRejectionOwner::Admission("compiled_pipeline_dependency_cycle"));
        assert_eq!(projection.boundary, "compiled_pipeline");
        assert!(projection.facts.contains_key("task_id"), "the cycle names the task it closed on: {:?}", projection);
        assert!(admission.pipeline().is_none());

        assert!(
            crate::admission::bounded_join_modeled_facts(&plan).is_none(),
            "a rejected compiled pipeline must not emit modeled bounded-join facts"
        );
        let unsupported_facts = crate::admission::bounded_join_unsupported_shape_facts(&plan)
            .expect("a rejected compiled pipeline emits unsupported bounded-join facts");
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_reason").map(String::as_str),
            Some("compiled_pipeline_dependency_cycle")
        );
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_boundary").map(String::as_str),
            Some("compiled_pipeline")
        );
        assert_eq!(
            unsupported_facts.get("bounded_join_rejection_category").map(String::as_str),
            Some("unsupported_compiled_pipeline_envelope")
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
