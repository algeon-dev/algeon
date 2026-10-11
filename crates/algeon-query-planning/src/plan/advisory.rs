//! Advisory data-volume signals attached by a frontend planner.
//!
//! These signals describe a frontend's own estimate of a node's whole output
//! (all partitions). They are optimizer hints only: they never replace the
//! conservative estimates in [`super::estimates`], the proved bounds in
//! [`super::row_bounds`], or [`super::PlanFacts`], and they must not become
//! a capability, admission, or grant input.

/// One advisory quantity with the frontend's own precision claim.
///
/// The embedding frontend's planner combined its source statistics with its
/// own operator estimation rules. The value was never checked against native
/// execution and can be wrong by orders of magnitude under key skew or
/// correlated predicates.
///
/// `Absent` is a distinct state from zero so consumers can tell "no evidence"
/// from "estimated empty".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdvisoryStatistic {
    /// The frontend claims the value is exact for its plan.
    Exact(u64),
    /// The frontend derived the value from estimation rules.
    Inexact(u64),
    /// The frontend had no estimate.
    Absent,
}

impl AdvisoryStatistic {
    /// Returns the estimated value regardless of precision.
    pub const fn value(self) -> Option<u64> {
        match self {
            Self::Exact(value) | Self::Inexact(value) => Some(value),
            Self::Absent => None,
        }
    }
}

/// Advisory row and byte estimates for one node's whole output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdvisoryOutputStatistics {
    /// Estimated output rows.
    pub rows: AdvisoryStatistic,
    /// Estimated output bytes in the frontend's own byte model, which is not
    /// device residency.
    pub bytes: AdvisoryStatistic,
}

impl AdvisoryOutputStatistics {
    /// Returns true when neither rows nor bytes carry a value.
    pub const fn is_absent(self) -> bool {
        matches!((self.rows, self.bytes), (AdvisoryStatistic::Absent, AdvisoryStatistic::Absent))
    }
}

/// Why an advisory volume has no usable estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdvisoryUnknownReason {
    /// The frontend had no statistics for the occurrence.
    MissingStatistics,
    /// The frontend's statistics computation returned an error.
    StatisticsComputationFailed,
    /// The native occurrence's schema no longer matches the statistics.
    OutputSchemaChanged,
    /// A limit clipped the row estimate; the byte model has no per-row figure
    /// to clip bytes by (see `QueryPlanBuilder::add_limit`).
    LimitBytesUnavailable,
    /// The frontend's join estimate fell back to the Cartesian product of its
    /// inputs, which it reports as unknown.
    CartesianFallback,
    /// An optimizer rule changed the node or one of its dependencies, so the
    /// estimate describes a subtree that no longer exists.
    RewrittenSubtree,
    /// Two frontend occurrences with different statistics were interned to one
    /// native node, so neither value can be attributed to it.
    ConflictingOccurrences,
}

/// Frontend statistics bound to one native output occurrence and schema.
///
/// The statistics cover the occurrence's complete output across all
/// partitions; they never describe a device allocation scope.
///
/// These records are diagnostic input only; build-side selection reads a
/// join's own `JoinSpec::advisory_output_statistics`, not these bindings. They
/// are not native bounds and cannot justify capability, admission, or grant
/// size. Rewrites invalidate a record when its occurrence, schema, or
/// dependencies change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisoryOutputBinding {
    /// Native occurrence the statistics describe; remapped by every rewrite.
    pub node: super::NodeId,
    /// Output schema the frontend estimated for; must equal the node's output
    /// schema when bound (`QueryPlanBuilder::set_advisory_output`).
    pub schema: super::SchemaRef,
    /// Frontend that produced the estimate, for diagnostics only.
    pub frontend: String,
    /// Version string of that frontend, for diagnostics only.
    pub frontend_version: String,
    pub statistics: AdvisoryOutputStatistics,
    /// Explains absent quantities; a present quantity retains its precision.
    pub unknown_reason: Option<AdvisoryUnknownReason>,
}

impl AdvisoryOutputBinding {
    pub(super) fn invalidate(&mut self, reason: AdvisoryUnknownReason) {
        self.statistics.rows = AdvisoryStatistic::Absent;
        self.statistics.bytes = AdvisoryStatistic::Absent;
        self.unknown_reason = Some(reason);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{Expr, LimitSpec},
        plan::{PlanNode, ProjectionOutput, QueryPlanBuilder, RewrittenNodes},
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn advisory_rewrite_invalidates_changed_dependency_without_changing_execution_identity() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
        let root = builder
            .add_projection(source, vec![ProjectionOutput::new(Expr::column("key"), schema.field(0).clone())])
            .unwrap();
        let plain = builder.clone().finish_gpu_resident(root).unwrap();
        builder
            .set_advisory_output(AdvisoryOutputBinding {
                node: root,
                schema: Arc::clone(&schema),
                frontend: "test frontend".to_owned(),
                frontend_version: "1".to_owned(),
                statistics: AdvisoryOutputStatistics {
                    rows: AdvisoryStatistic::Inexact(100),
                    bytes: AdvisoryStatistic::Inexact(800),
                },
                unknown_reason: None,
            })
            .unwrap();
        let plan = builder.clone().finish_gpu_resident(root).unwrap();
        assert_eq!(plain.execution_fingerprint(), plan.execution_fingerprint());
        let unchanged = plan.rewrite(RewrittenNodes::in_place(plan.nodes().to_vec(), root)).unwrap();
        assert_eq!(unchanged.advisory_outputs(), plan.advisory_outputs());

        let mut nodes = plan.nodes().to_vec();
        nodes[source.index()].kind = PlanNode::Source(Box::new(SourceNode::GpuResident(GpuResidentSource::new(
            "different input",
            Arc::clone(&schema),
        ))));
        let rewritten = plan.rewrite(RewrittenNodes::in_place(nodes, root)).unwrap();
        assert!(rewritten.advisory_outputs()[0].statistics.is_absent());
        assert_eq!(rewritten.advisory_outputs()[0].unknown_reason, Some(AdvisoryUnknownReason::RewrittenSubtree));

        let limited = builder.add_limit(root, LimitSpec { skip: 90, fetch: Some(20) }).unwrap();
        let limited = builder.finish_gpu_resident(limited).unwrap();
        let limit_statistics =
            limited.advisory_outputs().iter().find(|binding| binding.node == limited.root_id()).unwrap();
        assert_eq!(limit_statistics.statistics.rows, AdvisoryStatistic::Inexact(10));
        assert_eq!(limit_statistics.statistics.bytes, AdvisoryStatistic::Absent);
    }
}
