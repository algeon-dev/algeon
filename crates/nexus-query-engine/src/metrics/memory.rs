use super::{
    CompiledTaskId, PlanNodeId, QueryId, QueryMetrics, optional_row_groups, optional_str, optional_u64, optional_usize,
};
use crate::runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind};

/// Outcome of one memory-reservation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReservationEventOutcome {
    /// Requested reservation was granted.
    Granted,
    /// Requested reservation was denied.
    Denied,
    /// Existing reservation state was changed without a new admission decision.
    Applied,
}

impl MemoryReservationEventOutcome {
    /// Stable snake-case label emitted in debug rows.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Granted => "granted",
            Self::Denied => "denied",
            Self::Applied => "applied",
        }
    }
}

impl std::fmt::Display for MemoryReservationEventOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reservation timeline action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReservationEventAction {
    /// Initial reservation request.
    Reserve,
    /// Reservation grew by a positive delta.
    Grow,
    /// Reservation shrank by a negative delta.
    Shrink,
    /// Reservation was fully released.
    Release,
    /// Reservation ownership moved across boundaries.
    Transfer,
    /// Alias acquired a shared reservation.
    AliasAcquire,
    /// Alias changed its report-facing label.
    AliasRelabel,
    /// Alias released its shared reservation.
    AliasRelease,
}

impl MemoryReservationEventAction {
    /// Stable snake-case label emitted in debug rows.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reserve => "reserve",
            Self::Grow => "grow",
            Self::Shrink => "shrink",
            Self::Release => "release",
            Self::Transfer => "transfer",
            Self::AliasAcquire => "alias_acquire",
            Self::AliasRelabel => "alias_relabel",
            Self::AliasRelease => "alias_release",
        }
    }
}

impl std::fmt::Display for MemoryReservationEventAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One memory-reservation timeline row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationEvent {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Reservation class affected by this event.
    pub reservation_kind: ReservationKind,
    /// Memory space whose budget was charged.
    pub memory_space: MemorySpaceId,
    /// Stable boundary label from the reservation request.
    pub memory_boundary: String,
    /// Reservation action applied at this point in the timeline.
    pub action: MemoryReservationEventAction,
    /// Requested bytes for reserve/grow decisions.
    pub requested_bytes: u64,
    /// Bytes granted or otherwise associated with this event.
    pub granted_bytes: u64,
    /// Signed change to live charged bytes.
    pub charged_delta_bytes: i128,
    /// Live reserved bytes after the event.
    pub current_reserved_bytes: u64,
    /// Budget limit in bytes for the memory space.
    pub budget_bytes: u64,
    /// Admission or application outcome.
    pub outcome: MemoryReservationEventOutcome,
    pub node_id: Option<PlanNodeId>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub path: Option<String>,
    /// Present on source-read events; selects row groups within the file.
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
    pub compiled_task_id: Option<CompiledTaskId>,
    pub resident_resource_id: Option<String>,
    pub source_read_contract_quality: Option<String>,
    /// Bytes used for admission sizing; may differ from the requested reservation.
    pub source_read_planning_bytes: Option<u64>,
    /// Byte estimate carried for diagnostics only; not used for admission.
    pub source_read_estimated_bytes: Option<u64>,
}

impl MemoryReservationEvent {
    /// Builds a granted reserve event from a reservation request.
    pub fn granted(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::from_request(
            request,
            MemoryReservationEventOutcome::Granted,
            MemoryReservationEventAction::Reserve,
            granted_bytes,
            i128::from(granted_bytes),
            granted_bytes,
        )
    }

    /// Builds a denied reserve event from a reservation request.
    pub fn denied(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::from_request(
            request,
            MemoryReservationEventOutcome::Denied,
            MemoryReservationEventAction::Reserve,
            granted_bytes,
            0,
            request.budget_bytes.saturating_sub(granted_bytes),
        )
    }

    /// Builds an event row from a request and explicit timeline state.
    pub fn from_request(
        request: &MemoryReservationRequest,
        outcome: MemoryReservationEventOutcome,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) -> Self {
        Self {
            query_id: request.query_id.clone(),
            reservation_kind: request.reservation_kind,
            memory_space: request.memory_space,
            memory_boundary: request.memory_boundary.clone(),
            action,
            requested_bytes: request.requested_bytes,
            granted_bytes,
            charged_delta_bytes,
            current_reserved_bytes,
            budget_bytes: request.budget_bytes,
            outcome,
            node_id: request.node_id.clone(),
            stage: request.stage.clone(),
            chunk_index: request.chunk_index,
            file_index: request.file_index,
            path: request.path.clone(),
            row_groups: request.row_groups.clone(),
            attempt: request.attempt,
            max_attempts: request.max_attempts,
            compiled_task_id: request.compiled_task_id.clone(),
            resident_resource_id: request.resident_resource_id.clone(),
            source_read_contract_quality: request.source_read_contract_quality.clone(),
            source_read_planning_bytes: request.source_read_planning_bytes,
            source_read_estimated_bytes: request.source_read_estimated_bytes,
        }
    }

    /// Serializes the event as a stable comma-separated debug row.
    pub fn to_stable_debug_row(&self) -> String {
        format!(
            "reservation_kind={},memory_space={},memory_boundary={},stage={},action={},outcome={},requested_bytes={},granted_bytes={},remaining_bytes={},current_reserved_bytes={},budget_bytes={},node_id={},compiled_task_id={},resident_resource_id={},chunk_index={},file_index={},path={},row_groups={},attempt={},max_attempts={},source_read_contract_quality={},source_read_planning_bytes={},source_read_estimated_bytes={}",
            self.reservation_kind,
            self.memory_space,
            self.memory_boundary,
            optional_str(self.stage.as_deref()),
            self.action,
            self.outcome,
            self.requested_bytes,
            self.granted_bytes,
            self.budget_bytes.saturating_sub(self.current_reserved_bytes),
            self.current_reserved_bytes,
            self.budget_bytes,
            optional_str(self.node_id.as_ref().map(PlanNodeId::as_str)),
            optional_str(self.compiled_task_id.as_ref().map(CompiledTaskId::as_str)),
            optional_str(self.resident_resource_id.as_deref()),
            optional_usize(self.chunk_index),
            optional_usize(self.file_index),
            optional_str(self.path.as_deref()),
            optional_row_groups(self.row_groups.as_deref()),
            optional_usize(self.attempt),
            optional_usize(self.max_attempts),
            optional_str(self.source_read_contract_quality.as_deref()),
            optional_u64(self.source_read_planning_bytes),
            optional_u64(self.source_read_estimated_bytes),
        )
    }
}

impl QueryMetrics {
    /// Records a resident DataFrame memory observation in bytes.
    pub fn record_device_memory_observation(&mut self, bytes: u64) {
        self.peak_device_memory_bytes = Some(self.peak_device_memory_bytes.map_or(bytes, |peak| peak.max(bytes)));
        self.resident_dataframe_peak_bytes =
            Some(self.resident_dataframe_peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }

    /// Adds a granted reservation to cumulative counters.
    pub fn record_reservation_granted(&mut self, requested_bytes: u64, granted_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted_bytes);
    }

    /// Adds a denied reservation to cumulative counters.
    pub fn record_reservation_denied(&mut self, requested_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(1);
    }

    /// Records a granted reservation event and updates live/task peaks.
    pub fn record_reservation_granted_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_granted(request.requested_bytes, granted_bytes);
        let mut event = MemoryReservationEvent::granted(request, granted_bytes);
        if event.query_id.is_none() {
            event.query_id = self.inherited_query_id();
        }
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a denied reservation event and updates live/task peaks.
    pub fn record_reservation_denied_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_denied(request.requested_bytes);
        let mut event = MemoryReservationEvent::denied(request, granted_bytes);
        if event.query_id.is_none() {
            event.query_id = self.inherited_query_id();
        }
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a granted reservation timeline event with an explicit action.
    pub fn record_reservation_granted_timeline_event(
        &mut self,
        request: &MemoryReservationRequest,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) {
        self.record_reservation_granted(request.requested_bytes, granted_bytes);
        let mut event = MemoryReservationEvent::from_request(
            request,
            MemoryReservationEventOutcome::Granted,
            action,
            granted_bytes,
            charged_delta_bytes,
            current_reserved_bytes,
        );
        if event.query_id.is_none() {
            event.query_id = self.inherited_query_id();
        }
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a denied reservation timeline event with an explicit action.
    pub fn record_reservation_denied_timeline_event(
        &mut self,
        request: &MemoryReservationRequest,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        current_reserved_bytes: u64,
    ) {
        self.record_reservation_denied(request.requested_bytes);
        let mut event = MemoryReservationEvent::from_request(
            request,
            MemoryReservationEventOutcome::Denied,
            action,
            granted_bytes,
            0,
            current_reserved_bytes,
        );
        if event.query_id.is_none() {
            event.query_id = self.inherited_query_id();
        }
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records an already-built reservation timeline event.
    pub fn record_reservation_timeline_event(&mut self, mut event: MemoryReservationEvent) {
        if event.query_id.is_none() {
            event.query_id = self.inherited_query_id();
        }
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    fn record_reservation_live_peak(&mut self, current_reserved_bytes: u64) {
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(current_reserved_bytes);
    }
}
