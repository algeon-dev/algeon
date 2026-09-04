//! Diagnostic memory composition over the executable dependency graph.

use serde::{Deserialize, Serialize};

use super::{CompiledPipeline, ResidentLifetime};
use crate::plan::QueryPlan;

/// A modelled allocation quantity. Unknown never contributes an invented zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "precision", rename_all = "snake_case")]
pub enum ShadowBytes {
    Estimated { bytes: u64, model: String },
    Unknown { reason: String },
}

impl ShadowBytes {
    pub fn estimated(bytes: u64, model: impl Into<String>) -> Self {
        Self::Estimated { bytes, model: model.into() }
    }

    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown { reason: reason.into() }
    }

    pub const fn value(&self) -> Option<u64> {
        match self {
            Self::Estimated { bytes, .. } => Some(*bytes),
            Self::Unknown { .. } => None,
        }
    }
}

/// Component-owned allocation models in compiled task/resource index order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowMemoryModels {
    /// Additional simultaneous allocation beyond the task's retained outputs.
    pub task_workspace: Vec<ShadowBytes>,
    pub resources: Vec<ShadowResourceModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowResourceModel {
    /// Borrowed resident allocations belong to an external owner and are
    /// excluded from the query's allocation envelope.
    pub borrowed: bool,
    pub bytes: ShadowBytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowResourceLifetime {
    pub resource: usize,
    pub producer: usize,
    pub consumers: Vec<usize>,
    pub sink_output: bool,
    pub model: ShadowResourceModel,
}

/// One possible-overlap envelope, not a simulated execution trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowTaskEnvelope {
    pub task: usize,
    pub plan_node: Option<usize>,
    pub live_resources: Vec<usize>,
    pub overlapping_workspaces: Vec<usize>,
    pub query_owned_bytes: ShadowBytes,
    pub borrowed_resident_bytes: ShadowBytes,
}

/// Independent diagnostic artifact; none of these quantities is an admission input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowPipelineEstimate {
    pub schema: String,
    pub execution_scope: String,
    pub query_owned_peak: ShadowBytes,
    pub borrowed_resident_peak: ShadowBytes,
    pub task_workspace: Vec<ShadowBytes>,
    pub resources: Vec<ShadowResourceLifetime>,
    pub envelopes: Vec<ShadowTaskEnvelope>,
}

impl CompiledPipeline {
    /// Combines component allocation models using dependency precedence and
    /// last-consumer lifetimes. Unordered work may overlap. The resulting
    /// envelope can overestimate mutually incompatible overlaps; it never
    /// assumes that task id order is an execution timeline.
    pub fn shadow_memory_estimate(
        &self,
        plan: &QueryPlan,
        models: ShadowMemoryModels,
    ) -> crate::error::Result<ShadowPipelineEstimate> {
        self.validate()?;
        if models.task_workspace.len() != self.tasks().len() || models.resources.len() != self.resources().len() {
            return Err(crate::error::QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "pipeline.shadow_memory_estimate",
                "shadow models must use the compiled task and resource index spaces",
            ));
        }
        let count = self.tasks().len();
        let mut precedes = vec![vec![false; count]; count];
        for edge in self.edges() {
            precedes[edge.from().index()][edge.to().index()] = true;
        }
        for through in 0..count {
            let through_row = precedes[through].clone();
            for row in &mut precedes {
                if row[through] {
                    for (reaches, through_reaches) in row.iter_mut().zip(&through_row) {
                        *reaches |= *through_reaches;
                    }
                }
            }
        }
        let mut envelopes = Vec::with_capacity(count);
        for task in self.tasks() {
            let at = task.id().index();
            let live_resources = self
                .resources()
                .iter()
                .filter(|resource| {
                    let producer = resource.producer().index();
                    if precedes[at][producer] {
                        return false;
                    }
                    match resource.lifetime() {
                        ResidentLifetime::TaskLocal { task } => {
                            let owner = task.index();
                            owner == at || (!precedes[owner][at] && !precedes[at][owner])
                        },
                        ResidentLifetime::BetweenTasks { .. } => {
                            resource.consumers().iter().any(|consumer| !precedes[consumer.index()][at])
                        },
                        ResidentLifetime::SinkOutput { .. } => true,
                    }
                })
                .map(|resource| resource.id().index())
                .collect::<Vec<_>>();
            let overlapping_workspaces = (0..count)
                .filter(|other| *other == at || (!precedes[*other][at] && !precedes[at][*other]))
                .collect::<Vec<_>>();
            let owned = live_resources
                .iter()
                .filter(|id| !models.resources[**id].borrowed)
                .map(|id| &models.resources[*id].bytes)
                .chain(overlapping_workspaces.iter().map(|id| &models.task_workspace[*id]));
            let borrowed = live_resources
                .iter()
                .filter(|id| models.resources[**id].borrowed)
                .map(|id| &models.resources[*id].bytes);
            envelopes.push(ShadowTaskEnvelope {
                task: at,
                plan_node: task.plan_node().map(|node| node.index()),
                query_owned_bytes: sum(owned),
                borrowed_resident_bytes: sum(borrowed),
                live_resources,
                overlapping_workspaces,
            });
        }
        Ok(ShadowPipelineEstimate {
            schema: "nexus.shadow_pipeline_estimate.v1".to_owned(),
            execution_scope: crate::util::log_safe_fingerprint(&crate::plan::canonical_execution_scope_fingerprint(
                plan,
            )),
            query_owned_peak: maximum(envelopes.iter().map(|envelope| &envelope.query_owned_bytes)),
            borrowed_resident_peak: maximum(envelopes.iter().map(|envelope| &envelope.borrowed_resident_bytes)),
            task_workspace: models.task_workspace,
            resources: self
                .resources()
                .iter()
                .zip(models.resources)
                .map(|(resource, model)| ShadowResourceLifetime {
                    resource: resource.id().index(),
                    producer: resource.producer().index(),
                    consumers: resource.consumers().iter().map(|task| task.index()).collect(),
                    sink_output: matches!(resource.lifetime(), ResidentLifetime::SinkOutput { .. }),
                    model,
                })
                .collect(),
            envelopes,
        })
    }
}

fn sum<'a>(values: impl Iterator<Item = &'a ShadowBytes>) -> ShadowBytes {
    let mut total = 0u64;
    for value in values {
        let Some(bytes) = value.value() else {
            return ShadowBytes::unknown("component_cost_unavailable");
        };
        let Some(sum) = total.checked_add(bytes) else {
            return ShadowBytes::unknown("arithmetic_overflow");
        };
        total = sum;
    }
    ShadowBytes::estimated(total, "compiled_dependency_overlap_envelope_v1")
}

fn maximum<'a>(values: impl Iterator<Item = &'a ShadowBytes>) -> ShadowBytes {
    let mut peak = 0u64;
    for value in values {
        let Some(bytes) = value.value() else {
            return ShadowBytes::unknown("incomplete_overlap_envelope");
        };
        peak = peak.max(bytes);
    }
    ShadowBytes::estimated(peak, "compiled_dependency_overlap_envelope_v1")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{Expr, LimitSpec},
        pipeline::compile_pipeline,
        plan::{ProjectionOutput, QueryPlanBuilder},
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn shadow_envelope_releases_last_consumers_and_separates_borrowed_allocations() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let projection =
            builder.add_projection(source, vec![ProjectionOutput::new(Expr::column("key"), schema.field(0).clone())]);
        let root = builder.add_limit(projection, LimitSpec::fetch(10), schema);
        let plan = builder.finish_gpu_resident(root);
        let pipeline = compile_pipeline(&plan, &plan.capability_analysis()).unwrap();
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
}
