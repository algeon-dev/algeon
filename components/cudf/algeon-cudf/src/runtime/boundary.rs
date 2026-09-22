use std::{cell::Cell, sync::Arc};

use super::{ExecutionPlacement, execution_domain::ExecutionDomain};

thread_local! {
    static LAST_BOUNDARY_OBSERVATION: Cell<Option<BoundaryObservation>> = const { Cell::new(None) };
}

/// Query-facing classification of a host/device or ownership boundary.
///
/// This is a diagnostic shape for adapters and tests. It is not a scheduler,
/// metrics backend, or proof that an operation is zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundaryKind {
    /// Query-stage checkpoint without a host/device transfer by itself.
    QueryStage,
    /// Host-resident input was copied into GPU-owned cudf state.
    HostToDevice,
    /// GPU-resident cudf state was materialized into host-visible output.
    DeviceToHost,
    /// GPU-resident state was copied into another device execution placement.
    DeviceToDeviceCopy,
    /// Arrow C Device ABI roots were handed across an interop boundary.
    DeviceAbiHandoff,
    /// A prepared interop export was activated as a read lease.
    InteropLease,
    /// Ownership was transferred through a RAPIDS interop owned-transfer root.
    OwnedTransfer,
    /// Device-resident evidence was reduced to one bounded host-visible
    /// control scalar without materializing the input payload.
    BoundedControlScalar,
}

/// Query-stage taxonomy for memory and boundary observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryStageKind {
    /// File or external source read produced GPU-resident cudf state.
    SourceRead,
    /// Query operator produced a GPU-resident output.
    OperatorOutput,
    /// GPU-resident data was materialized into a final host-visible result.
    FinalMaterialize,
    /// External data was imported into cudf.
    InteropImport,
    /// cudf data was exported across an interop boundary.
    InteropExport,
}

/// Typed memory and transfer facts attached to a query-stage observation.
///
/// `payload_bytes` and `device_allocated_bytes` describe the GPU-resident
/// output. `uploaded_bytes` and `downloaded_bytes` are populated only for
/// observations that actually cross a host/device transfer boundary. They are
/// query accounting counters derived from the resulting logical payload, not
/// exact PCIe or driver-level transfer byte counts.
/// `reservation_bytes` is intentionally optional so bounded query engines can
/// attach or compare requested reservations without inventing free-form detail
/// keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueryMemoryObservation {
    /// Logical payload bytes represented by the observed output.
    pub payload_bytes: Option<usize>,
    /// Actual device allocation footprint represented by the observed output.
    pub device_allocated_bytes: Option<usize>,
    /// Payload-derived host-to-device bytes attributed to this boundary.
    pub uploaded_bytes: Option<usize>,
    /// Payload-derived device-to-host bytes attributed to this boundary.
    pub downloaded_bytes: Option<usize>,
    /// Optional bounded-execution reservation size associated with this stage.
    pub reservation_bytes: Option<usize>,
}

impl QueryMemoryObservation {
    /// Empty memory facts for observations that do not compute accounting.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create memory facts for a GPU-resident output.
    pub fn device_resident(payload_bytes: Option<usize>, device_allocated_bytes: Option<usize>) -> Self {
        Self {
            payload_bytes,
            device_allocated_bytes,
            uploaded_bytes: None,
            downloaded_bytes: None,
            reservation_bytes: None,
        }
    }

    /// Attach a host-to-device upload counter.
    pub fn with_uploaded_bytes(mut self, uploaded_bytes: usize) -> Self {
        self.uploaded_bytes = Some(uploaded_bytes);
        self
    }

    /// Attach a device-to-host download counter.
    pub fn with_downloaded_bytes(mut self, downloaded_bytes: usize) -> Self {
        self.downloaded_bytes = Some(downloaded_bytes);
        self
    }

    /// Attach a bounded-execution reservation fact.
    pub fn with_reservation_bytes(mut self, reservation_bytes: usize) -> Self {
        self.reservation_bytes = Some(reservation_bytes);
        self
    }
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
    /// Optional query-stage category for query engine accounting.
    pub query_stage: Option<QueryStageKind>,
    /// Retained cudf execution placement associated with the boundary.
    pub placement: ExecutionPlacement,
    /// Logical row count at the boundary.
    pub rows: usize,
    /// Logical column count at the boundary.
    pub columns: usize,
    /// Whether the path synchronized for host-visible results.
    pub synchronized_for_host: bool,
    /// Typed memory, transfer, and reservation facts for this observation.
    pub memory: QueryMemoryObservation,
}

impl BoundaryObservation {
    /// Construct an observation without query-stage memory facts.
    pub fn new(
        kind: BoundaryKind,
        placement: ExecutionPlacement,
        rows: usize,
        columns: usize,
        synchronized_for_host: bool,
    ) -> Self {
        Self {
            kind,
            query_stage: None,
            placement,
            rows,
            columns,
            synchronized_for_host,
            memory: QueryMemoryObservation::empty(),
        }
    }

    /// Attach a query stage classification.
    pub fn with_query_stage(mut self, query_stage: QueryStageKind) -> Self {
        self.query_stage = Some(query_stage);
        self
    }

    /// Attach typed memory and transfer facts.
    pub fn with_memory(mut self, memory: QueryMemoryObservation) -> Self {
        self.memory = memory;
        self
    }
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
    LAST_BOUNDARY_OBSERVATION.with(|last| last.set(observation));
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
    LAST_BOUNDARY_OBSERVATION.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::runtime::DeviceId;

    fn placement(workflow_id: u64) -> ExecutionPlacement {
        ExecutionPlacement {
            device: DeviceId::DEFAULT,
            workflow_id,
            bundle_id: workflow_id + 100,
            stream_debug: workflow_id as usize + 200,
            memory_resource_debug: workflow_id as usize + 300,
        }
    }

    fn observation(workflow_id: u64) -> BoundaryObservation {
        BoundaryObservation::new(BoundaryKind::QueryStage, placement(workflow_id), 1, 1, false)
    }

    #[test]
    fn last_boundary_observation_is_thread_local_for_parallel_tests() {
        let parent_observation = observation(1);
        let child_observation = observation(2);

        record_boundary_observation(Some(parent_observation));
        thread::spawn(move || {
            assert_eq!(last_boundary_observation(), None);
            record_boundary_observation(Some(child_observation));
            assert_eq!(last_boundary_observation(), Some(child_observation));
        })
        .join()
        .unwrap();

        assert_eq!(last_boundary_observation(), Some(parent_observation));
    }
}
