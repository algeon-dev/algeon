//! Opt-in pre-execution diagnostics; never an execution or admission policy.

use std::time::Instant;

use cudf::interop::arrow_host::estimate_record_batches_import_required_budget_bytes;
use query_planning::{
    pipeline::{
        CompiledPipeline, PipelineTaskKind, ShadowBytes, ShadowMemoryModels, ShadowPipelineEstimate,
        ShadowResourceModel,
    },
    plan::{PlanNode, QueryPlan},
    source::SourceNode,
};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One explicitly requested pre-execution shadow capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeShadowEstimate {
    pub pipeline: ShadowPipelineEstimate,
    /// Time spent collecting component costs and composing this estimate.
    pub collection_elapsed_ns: u64,
    /// Frontend output volumes retain their own byte model, separate from allocation models.
    pub frontend_outputs: Vec<serde_json::Value>,
}

/// Collects available component models against the already compiled schedule.
/// Decode, hash, gather, aggregate, and sink-copy costs currently remain unknown.
/// Literal import uses cuDF's existing aligned allocation model; resident input
/// handles are borrowed and their unknown size never increases query-owned bytes.
/// Serializes the plan's frontend advisory volumes for a diagnostic artifact.
///
/// A plan without a compiled schedule still reports this group, so the record
/// shape belongs to the advisory signal rather than to the pipeline estimate.
pub fn frontend_output_records(plan: &QueryPlan) -> Vec<serde_json::Value> {
    plan.advisory_outputs()
        .iter()
        .map(|binding| {
            serde_json::json!({
                "node": binding.node.index(),
                "schema": format!("{:?}", binding.schema),
                "scope": "WholeOutput",
                "frontend": binding.frontend,
                "frontend_version": binding.frontend_version,
                "provenance": "FrontendPlannerEstimate",
                "rows": format!("{:?}", binding.statistics.rows),
                "bytes": format!("{:?}", binding.statistics.bytes),
                "unknown_reason": binding.unknown_reason.map(|reason| format!("{reason:?}")),
            })
        })
        .collect()
}

pub fn estimate_compiled_pipeline(plan: &QueryPlan, pipeline: &CompiledPipeline) -> Result<NativeShadowEstimate> {
    let started = Instant::now();
    let task_workspace = pipeline
        .tasks()
        .iter()
        .map(|task| match task.kind() {
            PipelineTaskKind::GpuResidentSource | PipelineTaskKind::LiteralBatchSource => {
                ShadowBytes::estimated(0, "source_import_retained_allocation_model_v1")
            },
            kind => ShadowBytes::unknown(format!("component_workspace_model_unavailable:{}", kind.as_str())),
        })
        .collect();
    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let task = &pipeline.tasks()[resource.producer().index()];
            match task.plan_node().and_then(|node| plan.node(node)).map(|node| &node.kind) {
                Some(PlanNode::Source(source)) => match source.as_ref() {
                    SourceNode::GpuResident(_) => ShadowResourceModel {
                        borrowed: true,
                        bytes: ShadowBytes::unknown("borrowed_resident_size_unavailable"),
                    },
                    SourceNode::LiteralBatch(source) => {
                        let bytes = match estimate_record_batches_import_required_budget_bytes(&source.batches) {
                            Ok(bytes) => {
                                ShadowBytes::estimated(bytes as u64, "cudf_arrow_import_aligned_allocations_v1")
                            },
                            Err(error) => ShadowBytes::unknown(format!("cudf_arrow_import_model:{}", error.code())),
                        };
                        ShadowResourceModel { borrowed: false, bytes }
                    },
                    SourceNode::LocalParquet(_) => ShadowResourceModel {
                        borrowed: false,
                        bytes: ShadowBytes::unknown("component_decode_allocation_model_unavailable"),
                    },
                },
                _ => ShadowResourceModel {
                    borrowed: false,
                    bytes: ShadowBytes::unknown(format!(
                        "component_retained_allocation_model_unavailable:{}",
                        task.kind().as_str()
                    )),
                },
            }
        })
        .collect();
    let frontend_outputs = frontend_output_records(plan);
    let pipeline = pipeline.shadow_memory_estimate(plan, ShadowMemoryModels { task_workspace, resources })?;
    Ok(NativeShadowEstimate {
        pipeline,
        collection_elapsed_ns: started.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
        frontend_outputs,
    })
}
