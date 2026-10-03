//! Shadow memory-estimate contract tests, bound to the engine's
//! [`RapidsBackend`](query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].

use query_engine::{
    pipeline::{ShadowBytes, ShadowMemoryModels, ShadowResourceModel, compile_pipeline},
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn shadow_envelope_releases_last_consumers_and_separates_borrowed_allocations() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let projection = builder
        .add_projection(source, vec![ProjectionOutput::new(Expr::column("key"), schema.field(0).clone())])
        .unwrap();
    let root = builder.add_limit(projection, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_gpu_resident(root).unwrap();
    let pipeline =
        compile_pipeline(&plan, &plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()))).unwrap();
    let mut models = ShadowMemoryModels {
        task_workspace: vec![ShadowBytes::estimated(1, "fixture workspace"); pipeline.tasks().len()],
        resources: pipeline
            .resources()
            .iter()
            .map(|resource| {
                let producer = &pipeline.tasks()[resource.producer().index()];
                ShadowResourceModel {
                    borrowed: producer.plan_node() == Some(source),
                    bytes: ShadowBytes::estimated(
                        if producer.plan_node() == Some(source) { 100 } else { 20 },
                        "fixture allocation",
                    ),
                }
            })
            .collect(),
    };
    let estimate = pipeline.shadow_memory_estimate(&plan, models.clone()).unwrap();
    assert_eq!(estimate.query_owned_peak.value(), Some(41));
    assert_eq!(estimate.borrowed_resident_peak.value(), Some(100));
    let limit = estimate.envelopes.iter().find(|envelope| envelope.plan_node == Some(root.index())).unwrap();
    assert_eq!(limit.borrowed_resident_bytes.value(), Some(0), "input owner ends after projection");
    models.task_workspace.last_mut().unwrap().clone_from(&ShadowBytes::unknown("fixture missing kernel model"));
    let unknown = pipeline.shadow_memory_estimate(&plan, models).unwrap();
    assert_eq!(unknown.query_owned_peak.value(), None);
    assert_eq!(unknown.borrowed_resident_peak.value(), Some(100));
}
