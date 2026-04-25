use nexus_query_engine::admission::{
    bounded_join_modeled_facts as modeled_facts, bounded_join_unsupported_shape_facts as unsupported_facts,
};

use super::*;

#[path = "bounded_join_admission/aggregate_over_join_subplans.rs"]
mod aggregate_over_join_subplans;
#[path = "bounded_join_admission/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_admission/plans.rs"]
pub(crate) mod plans;
#[path = "bounded_join_admission/reservation_contract.rs"]
mod reservation_contract;
#[path = "bounded_join_admission/residual_resident_grouped_aggregate.rs"]
mod residual_resident_grouped_aggregate;
pub(crate) use aggregate_over_join_subplans::{
    AggregateOverJoinSubplanInputsVariant, keyless_conditional_aggregate_over_join_subplan_inputs_plan,
};
pub(crate) use mixed_residual::admitted_mixed_residual_multi_join_plan;
use plans::*;
pub(crate) use residual_resident_grouped_aggregate::{
    ResidualResidentGroupedAggregateInputVariant, residual_resident_grouped_aggregate_input_plan,
};

include!("bounded_join_admission/support.rs");
include!("bounded_join_admission/strategy.rs");
include!("bounded_join_admission/capability_projection.rs");
include!("bounded_join_admission/resident_subplan.rs");
include!("bounded_join_admission/root_topk.rs");
include!("bounded_join_admission/conditional_subplan.rs");
