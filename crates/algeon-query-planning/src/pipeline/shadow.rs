//! Diagnostic memory composition over the executable dependency graph.
//!
//! The estimate is a diagnostic artifact and never an admission input: the
//! query domain's allocator enforces allocation limits during execution
//! (`website/docs/design/admission.mdx#allocation-enforcement`). Overlap is
//! derived from the edge relation alone. Two tasks neither of which reaches the
//! other may run at the same time, so their workspaces and live resources are
//! summed; the envelope therefore bounds a schedule the executor may never
//! produce.

use serde::{Deserialize, Serialize};

use super::{CompiledPipeline, ResidentLifetime};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::QueryPlan,
};

/// A modelled allocation quantity. Unknown never contributes an invented zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "precision", rename_all = "snake_case")]
pub enum ShadowBytes {
    /// A size with the name of the component model that produced it.
    Estimated {
        /// Modeled allocation size in bytes.
        bytes: u64,
        /// Identity of the component allocation model.
        model: String,
    },
    /// No model covers the quantity. Any sum or maximum that includes an
    /// unknown is itself unknown.
    Unknown {
        /// Why no model can supply this quantity.
        reason: String,
    },
}

impl ShadowBytes {
    /// Records a byte estimate together with its component-model identity.
    pub fn estimated(bytes: u64, model: impl Into<String>) -> Self {
        Self::Estimated { bytes, model: model.into() }
    }

    /// Records missing size evidence without substituting a numeric estimate.
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown { reason: reason.into() }
    }

    /// Returns estimated bytes, or `None` when the size is unknown.
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
    /// Allocation models in compiled resource index order.
    pub resources: Vec<ShadowResourceModel>,
}

/// Allocation estimate and ownership attribution for one compiled resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowResourceModel {
    /// Borrowed resident allocations belong to an external owner and are
    /// excluded from the query's allocation envelope.
    pub borrowed: bool,
    /// Estimated allocation bytes, preserving unknown quantities.
    pub bytes: ShadowBytes,
}

/// Resource model paired with the producer and consumers that determine lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowResourceLifetime {
    pub(crate) resource: usize,
    pub(crate) producer: usize,
    pub(crate) consumers: Vec<usize>,
    pub(crate) sink_output: bool,
    /// Component allocation model for this resource.
    pub model: ShadowResourceModel,
}

/// One possible-overlap envelope derived from the edge relation. It does not
/// simulate an execution trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowTaskEnvelope {
    /// Zero-based task index within the compiled graph.
    pub task: usize,
    /// Native plan-node index, or `None` for a task without a plan node.
    pub plan_node: Option<usize>,
    /// Resources that may be resident while this task runs: produced by a
    /// task this one does not precede, and not yet released because some
    /// consumer does not precede this task either.
    pub live_resources: Vec<usize>,
    /// This task and every task unordered relative to it, whose workspace may
    /// be allocated at the same time.
    pub overlapping_workspaces: Vec<usize>,
    /// Combined query-owned bytes in this possible-overlap envelope.
    pub query_owned_bytes: ShadowBytes,
    /// Combined externally owned resident bytes in this envelope.
    pub borrowed_resident_bytes: ShadowBytes,
}

/// Independent diagnostic artifact; none of these quantities is an admission input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowPipelineEstimate {
    pub(crate) schema: String,
    pub(crate) execution_scope: String,
    /// Maximum modeled query-owned envelope; unknown if any contributing size is unknown.
    pub query_owned_peak: ShadowBytes,
    /// Maximum modeled externally owned resident envelope.
    pub borrowed_resident_peak: ShadowBytes,
    /// Workspace allocation estimates in compiled task index order.
    pub task_workspace: Vec<ShadowBytes>,
    /// Resource estimates paired with their compiled lifetimes.
    pub resources: Vec<ShadowResourceLifetime>,
    /// Possible-overlap estimates in compiled task order.
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
    ) -> Result<ShadowPipelineEstimate> {
        self.validate()?;
        if models.task_workspace.len() != self.tasks().len() || models.resources.len() != self.resources().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "pipeline.shadow_memory_estimate",
                "shadow models must use the compiled task and resource index spaces",
            ));
        }
        let count = self.tasks().len();
        // Transitive closure of the edge relation (Warshall): afterwards
        // `precedes[a][b]` means task `a` finishes before task `b` starts.
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
            schema: "algeon.shadow_pipeline_estimate".to_owned(),
            execution_scope: format!("blake3:{}", plan.execution_fingerprint()),
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
    ShadowBytes::estimated(total, "compiled_dependency_overlap_envelope")
}

fn maximum<'a>(values: impl Iterator<Item = &'a ShadowBytes>) -> ShadowBytes {
    let mut peak = 0u64;
    for value in values {
        let Some(bytes) = value.value() else {
            return ShadowBytes::unknown("incomplete_overlap_envelope");
        };
        peak = peak.max(bytes);
    }
    ShadowBytes::estimated(peak, "compiled_dependency_overlap_envelope")
}
