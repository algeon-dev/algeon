//! Frontend-neutral graph execution discovery and admission contracts.
//!
//! The adapter core owns the DataFusion plan discovery SPI. Graph frontends
//! implement it without creating a core-to-frontend dependency.

use datafusion::physical_plan::ExecutionPlan;
use query_planning::admission::AttemptCapabilityEvidence;

use super::native_node::native_node;
use crate::{
    error::{Error, Result},
    native_exec::SealedAttemptMetricsTarget,
};

/// Frontend-neutral graph-model diagnostics exposed by a graph execution node.
///
/// Capability evidence issuance remains in the adapter planner after it has
/// recognized this crate-private execution type; graph frontends do not get a
/// caller-controlled evidence SPI.
pub(crate) trait GraphExecutionPlan: Send + Sync {
    /// Issues capability evidence for the exact native graph fragment this
    /// execution node submits after staging its GPU-resident inputs.
    fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence>;

    /// Returns the one fixed-cardinality post-seal metric target for this
    /// graph execution root. The adapter discovers it through this neutral
    /// SPI, so core execution does not depend on a graph frontend.
    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget;

    /// Returns a typed rejection when this graph root intentionally owns its
    /// result at the host boundary and cannot participate as a GPU producer.
    fn gpu_resident_output_rejection(&self) -> Option<Error> {
        None
    }
}

pub(crate) fn graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    native_node(plan).and_then(|node| node.graph_execution_plan())
}
