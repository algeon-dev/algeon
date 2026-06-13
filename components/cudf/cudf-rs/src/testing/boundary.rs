use crate::runtime::{BoundaryObservation, last_boundary_observation, record_boundary_observation};

/// Reset the last query-facing boundary observation used by integration tests.
pub fn reset_boundary_observation_for_test() {
    record_boundary_observation(None);
}

/// Read the last query-facing boundary observation, if any.
pub fn last_boundary_observation_for_test() -> Option<BoundaryObservation> {
    last_boundary_observation()
}
