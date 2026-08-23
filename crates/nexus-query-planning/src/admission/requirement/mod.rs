//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

use arrow::datatypes::{DataType, Schema, SchemaRef, TimeUnit};

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{
            AttemptReservationConsumerState, AttemptReservationProducerOccurrence, AttemptReservationSchedule,
            AttemptReservationShape,
        },
    },
    capability::ReservationRequirement,
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::ReservationKind,
    expr::{AggregateFunction, Expr, LiteralValue, alias_wrapped_column, representation_cast_preserves_identity},
    graph_memory::{
        GraphMemoryAdmissionEvidence, GraphMemoryOwnerTermFact, GraphMemoryOwnerTermRole, GraphSourceStageLifetime,
        graph_output_is_final, query_plan_hard_row_upper_bound, query_plan_node_hard_row_upper_bound,
    },
    plan::{
        AggregateNode, JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, ResultSink,
        canonical_execution_scope_fingerprint,
        column_evidence::{PlanColumnEvidence, PlanColumnEvidenceAnalysis},
        estimated_local_parquet_source_bytes, estimated_reachable_local_parquet_source_bytes,
    },
    source::SourceNode,
};

/// Trustworthiness of the descriptive working-set hint.
///
/// Ordering is `Unknown < Modeled < Measured`; aggregation retains the weakest
/// evidence so a multi-fragment invocation never overstates its estimate.
mod capability;
mod composed;
mod owner_terms;

pub use capability::{
    AttemptCapabilityEvidence, ComposedGraphProducerCandidate, ComposedProducerBindingDiagnostics,
    QueryPlanCapabilityProof,
};
pub(super) use capability::{checked_owner_sum, owner_inventory_covers_scopes};
#[cfg(test)]
use composed::{ComposedConsumerAnalysis, composed_consumer_additional_peak_bytes_for_plan};
pub(super) use composed::{
    ComposedConsumerBranchBound, ComposedGraphProducerOutputEvidence, NormalizedComposedProducer,
    arrow_fixed_width_cudf_type_id, bounded_consumer_frame_bytes, bounded_consumer_frame_bytes_with_evidence,
    bounded_retained_output_frame_bytes_with_evidence, composed_consumer_bound_with_peak_terms,
    composed_schedule_error, consumer_output_column_with_optional_evidence, cugraph_graph_execution_capability,
    fixed_width_consumer_output_column, generic_graph_execution_blocker, has_reachable_sort,
    is_generic_graph_execution_blocker, limit_owner_terms, resident_side_input_frame_upper_bound_bytes,
    sort_metadata_for_composed_consumer, sort_owner_terms, validate_sort_metadata_descriptor,
};
#[doc(hidden)]
pub use owner_terms::AttemptBoundDerivation;
pub use owner_terms::EstimateQuality;
pub(super) use owner_terms::{
    AllocationClass, AllocationOwnerTerm, FiniteBytes, LimitMetadataDescriptor, LimitRangeGatherPreflight,
    OccurrenceId, OwnerId, OwnerTermId, OwnerTermRole, SortKeyMetadataDescriptor, SortMetadataDescriptor,
    SortPreflight, aggregate_owner_state_is_supported, aggregate_owner_terms_for_shapes, generic_aggregate_owner_terms,
    generic_gpu_resident_owner_terms, graph_owner_terms_from_evidence, has_mixed_local_gpu_sources, limit_owner_error,
    literal_batch_source_owner_terms, local_parquet_source_owner_terms, merge_owner_terms, owner_derivation_error,
    owner_term_error, sort_owner_error, validate_owner_term,
};

#[cfg(test)]
mod tests;
