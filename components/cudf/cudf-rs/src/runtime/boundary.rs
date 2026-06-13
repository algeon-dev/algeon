use std::sync::{Arc, Mutex};

use super::{ExecutionPlacement, execution_domain::ExecutionDomain};

static LAST_BOUNDARY_OBSERVATION: Mutex<Option<BoundaryObservation>> = Mutex::new(None);

/// Query-facing classification of a host/device or ownership boundary.
///
/// This is a diagnostic shape for adapters and tests. It is not a scheduler,
/// metrics backend, or proof that an operation is zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundaryKind {
    /// Host-resident input was copied into GPU-owned cudf-rs state.
    HostToDevice,
    /// GPU-resident cudf-rs state was materialized into host-visible output.
    DeviceToHost,
    /// GPU-resident state was copied into another device execution placement.
    DeviceToDeviceCopy,
    /// Arrow C Device ABI roots were handed across an interop boundary.
    DeviceAbiHandoff,
    /// A prepared interop export was activated as a read lease.
    InteropLease,
    /// Ownership was transferred through a RAPIDS interop owned-transfer root.
    OwnedTransfer,
}

/// Stable diagnostic observation for one host/device or interop boundary.
///
/// The row and column counts describe the logical table-shaped payload at the
/// boundary. `synchronized_for_host` reports whether the boundary had to make
/// the retained stream host-visible before returning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryObservation {
    /// Boundary category.
    pub kind: BoundaryKind,
    /// Retained cudf-rs execution placement associated with the boundary.
    pub placement: ExecutionPlacement,
    /// Logical row count at the boundary.
    pub rows: usize,
    /// Logical column count at the boundary.
    pub columns: usize,
    /// Whether the path synchronized for host-visible results.
    pub synchronized_for_host: bool,
}

/// Opt-in observer for query-facing host/device and interop boundaries.
///
/// Observers are attached to a [`crate::Runtime`] with
/// [`crate::RuntimeBuilder::with_boundary_observer`]. Callbacks run
/// synchronously after a successful boundary operation records its diagnostic
/// observation.
pub trait BoundaryObserver: Send + Sync + 'static {
    /// Observe one successful boundary event.
    fn observe(&self, observation: BoundaryObservation);
}

impl<F> BoundaryObserver for F
where F: Fn(BoundaryObservation) + Send + Sync + 'static
{
    fn observe(&self, observation: BoundaryObservation) {
        self(observation);
    }
}

pub(crate) type BoundaryObserverRef = Arc<dyn BoundaryObserver>;

pub(crate) fn record_boundary_observation(observation: Option<BoundaryObservation>) {
    if let Ok(mut last) = LAST_BOUNDARY_OBSERVATION.lock() {
        *last = observation;
    }
}

pub(crate) fn record_boundary_observation_in_domain(domain: &Arc<ExecutionDomain>, observation: BoundaryObservation) {
    record_boundary_observation(Some(observation));
    domain.observe_boundary(observation);
}

pub(crate) fn record_boundary_observation_for_observer(
    observer: Option<&BoundaryObserverRef>,
    observation: BoundaryObservation,
) {
    record_boundary_observation(Some(observation));
    if let Some(observer) = observer {
        observer.observe(observation);
    }
}

pub(crate) fn last_boundary_observation() -> Option<BoundaryObservation> {
    LAST_BOUNDARY_OBSERVATION.lock().ok().and_then(|last| *last)
}
