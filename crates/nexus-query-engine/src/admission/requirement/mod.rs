//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

#[cfg(test)]
use std::collections::BTreeSet;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Schema, SchemaRef, TimeUnit};

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{
            AttemptReservationConsumerState, AttemptReservationProducerOccurrence, AttemptReservationSchedule,
            AttemptReservationShape, canonical_execution_scope_fingerprint,
        },
        string_payload::{PlanColumnEvidence, PlanColumnEvidenceAnalysis},
    },
    capability::ReservationRequirement,
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateFunction, Expr, LiteralValue, alias_wrapped_column, representation_cast_preserves_identity},
    graph_memory::{
        GraphMemoryAdmissionEvidence, GraphMemoryOwnerTermFact, GraphMemoryOwnerTermRole, GraphSourceStageLifetime,
        graph_output_is_final, query_plan_hard_row_upper_bound, query_plan_node_hard_row_upper_bound,
    },
    plan::{
        AggregateNode, JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, ResultSink,
        estimated_local_parquet_source_bytes, estimated_reachable_local_parquet_source_bytes,
    },
    runtime::ReservationKind,
    source::SourceNode,
};

/// Trustworthiness of the descriptive working-set hint.
///
/// Ordering is `Unknown < Modeled < Measured`; aggregation retains the weakest
/// evidence so a multi-fragment invocation never overstates its estimate.
mod capability;
mod composed;
mod owner_terms;

pub(crate) use capability::*;
pub use capability::{AttemptCapabilityEvidence, ComposedGraphProducerCandidate, QueryPlanCapabilityProof};
pub(crate) use composed::*;
pub use owner_terms::EstimateQuality;
pub(crate) use owner_terms::*;

#[cfg(test)]
mod tests;
