use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use super::*;
use crate::{
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveTaskMemoryHistoryShape},
    metrics::{
        CompiledTaskId, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
        PlanNodeId, QueryId,
    },
    mode::NativePipelineExecutionMode,
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    runtime::{
        ExecutionContext, MemoryGrant, MemorySpaceId, cache::local_parquet::*,
        gpu::governor::MemoryGrantLiveReservation,
    },
};

#[derive(Debug, Clone)]
struct ActiveReservationEventContext {
    reservation_kind: ReservationKind,
    memory_space: MemorySpaceId,
    memory_boundary: String,
    budget_bytes: u64,
    query_id: Option<QueryId>,
    node_id: Option<PlanNodeId>,
    stage: Option<String>,
    chunk_index: Option<usize>,
    file_index: Option<usize>,
    path: Option<String>,
    row_groups: Option<Vec<usize>>,
    attempt: Option<usize>,
    max_attempts: Option<usize>,
    compiled_task_id: Option<CompiledTaskId>,
    resident_resource_id: Option<String>,
    source_read_contract_quality: Option<String>,
    source_read_planning_bytes: Option<u64>,
    source_read_estimated_bytes: Option<u64>,
}

impl ActiveReservationEventContext {
    fn from_request(request: &MemoryReservationRequest) -> Self {
        Self {
            reservation_kind: request.reservation_kind,
            memory_space: request.memory_space,
            memory_boundary: request.memory_boundary.clone(),
            budget_bytes: request.budget_bytes,
            query_id: request.query_id.clone(),
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

    fn event(
        &self,
        action: MemoryReservationEventAction,
        requested_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) -> MemoryReservationEvent {
        MemoryReservationEvent {
            query_id: self.query_id.clone(),
            reservation_kind: self.reservation_kind,
            memory_space: self.memory_space,
            memory_boundary: self.memory_boundary.clone(),
            action,
            requested_bytes,
            granted_bytes: 0,
            charged_delta_bytes,
            current_reserved_bytes,
            budget_bytes: self.budget_bytes,
            outcome: MemoryReservationEventOutcome::Applied,
            node_id: self.node_id.clone(),
            stage: self.stage.clone(),
            chunk_index: self.chunk_index,
            file_index: self.file_index,
            path: self.path.clone(),
            row_groups: self.row_groups.clone(),
            attempt: self.attempt,
            max_attempts: self.max_attempts,
            compiled_task_id: self.compiled_task_id.clone(),
            resident_resource_id: self.resident_resource_id.clone(),
            source_read_contract_quality: self.source_read_contract_quality.clone(),
            source_read_planning_bytes: self.source_read_planning_bytes,
            source_read_estimated_bytes: self.source_read_estimated_bytes,
        }
    }
}

/// Controls whether an active reservation charges capacity itself or only
/// keeps already-charged bytes alive through another reservation lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum ActiveReservationSemantics {
    /// Charged bytes backing a resident frame or source read. Aliases may keep
    /// those bytes reserved after the owner is released.
    OwnedResident,
    /// Charged scratch or intermediate work space. It can be transferred to a
    /// later reservation without creating a second capacity charge.
    Work,
    /// Non-charging view of existing bytes. Releasing the last alias also
    /// releases any owner charge that was deferred while aliases were alive.
    Alias,
}

#[derive(Debug)]
struct ActiveReservationLineage {
    alias_count: Cell<usize>,
    deferred_charged_release_bytes: Cell<u64>,
}

impl ActiveReservationLineage {
    fn new() -> Self {
        Self { alias_count: Cell::new(0), deferred_charged_release_bytes: Cell::new(0) }
    }
}

pub(in crate::exec) struct ActiveReservation {
    memory_space: MemorySpaceId,
    reserved_bytes: Rc<Cell<u64>>,
    live_accounting: Option<MemoryGrant>,
    protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
    source_cache_key: Option<GpuLocalParquetSourceCacheKey>,
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    event_context: ActiveReservationEventContext,
    semantics: ActiveReservationSemantics,
    lineage: Option<Rc<ActiveReservationLineage>>,
    release_action: MemoryReservationEventAction,
    bytes: u64,
    charged_bytes: u64,
    released: bool,
}

impl ActiveReservation {
    pub(in crate::exec) fn owned_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            bytes,
            request,
            ActiveReservationSemantics::OwnedResident,
        )
    }

    pub(in crate::exec) fn work(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            bytes,
            request,
            ActiveReservationSemantics::Work,
        )
    }

    fn charged(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        semantics: ActiveReservationSemantics,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            source_cache_key: None,
            source_cache_pin: None,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::Release,
            bytes,
            charged_bytes: bytes,
            released: false,
        }
    }

    pub(in crate::exec) fn borrowed_source_cache(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        source_cache_key: GpuLocalParquetSourceCacheKey,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        {
            let mut protected = protected_source_cache_keys.borrow_mut();
            let count = protected.entry(source_cache_key.clone()).or_insert(0);
            *count = count.saturating_add(1);
        }
        Self {
            memory_space,
            reserved_bytes,
            live_accounting: None,
            protected_source_cache_keys,
            source_cache_key: Some(source_cache_key),
            source_cache_pin,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    pub(in crate::exec) fn borrowed_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            live_accounting: None,
            protected_source_cache_keys,
            source_cache_key: None,
            source_cache_pin: None,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    fn alias(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
        lineage: Rc<ActiveReservationLineage>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        lineage.alias_count.set(lineage.alias_count.get().saturating_add(1));
        Self {
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            source_cache_key: None,
            source_cache_pin,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(lineage),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    pub(in crate::exec) fn memory_space(&self) -> MemorySpaceId {
        self.memory_space
    }

    pub(in crate::exec) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(in crate::exec) fn charged_bytes(&self) -> u64 {
        self.charged_bytes
    }

    pub(in crate::exec) fn retain_grant_live_reservation(&self) -> Option<MemoryGrantLiveReservation> {
        self.live_accounting.as_ref().and_then(|grant| grant.retain_live_reservation(self.charged_bytes))
    }

    pub(in crate::exec) fn is_alias(&self) -> bool {
        matches!(self.semantics, ActiveReservationSemantics::Alias)
    }

    fn transferred(mut self, request: &MemoryReservationRequest) -> Self {
        assert!(!self.released, "cannot transfer a released reservation");
        assert!(!self.is_alias(), "cannot transfer an alias reservation");
        let charged_bytes = self.charged_bytes;
        let semantics = self.semantics;
        self.charged_bytes = 0;
        self.released = true;
        Self {
            memory_space: self.memory_space,
            reserved_bytes: Rc::clone(&self.reserved_bytes),
            live_accounting: self.live_accounting.clone(),
            protected_source_cache_keys: Rc::clone(&self.protected_source_cache_keys),
            source_cache_key: None,
            source_cache_pin: None,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics,
            lineage: self.lineage.clone(),
            release_action: MemoryReservationEventAction::Release,
            bytes: request.requested_bytes,
            charged_bytes,
            released: false,
        }
    }

    fn relabeled_alias(mut self, request: &MemoryReservationRequest) -> Self {
        assert!(!self.released, "cannot relabel a released reservation");
        assert!(self.is_alias(), "cannot relabel a non-alias reservation as an alias");
        let source_cache_key = self.source_cache_key.take();
        let source_cache_pin = self.source_cache_pin.take();
        let lineage = self.lineage.clone();
        self.released = true;
        Self {
            memory_space: self.memory_space,
            reserved_bytes: Rc::clone(&self.reserved_bytes),
            live_accounting: self.live_accounting.clone(),
            protected_source_cache_keys: Rc::clone(&self.protected_source_cache_keys),
            source_cache_key,
            source_cache_pin,
            event_context: ActiveReservationEventContext::from_request(request),
            semantics: ActiveReservationSemantics::Alias,
            lineage,
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes: request.requested_bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    fn release(&mut self) -> Option<MemoryReservationEvent> {
        if self.released {
            return None;
        }
        if let Some(source_cache_key) = self.source_cache_key.take() {
            let mut protected = self.protected_source_cache_keys.borrow_mut();
            if let Some(count) = protected.get_mut(&source_cache_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    protected.remove(&source_cache_key);
                }
            }
        }
        self.source_cache_pin.take();
        if self.source_cache_key.is_none() {
            match (self.semantics, self.lineage.as_ref()) {
                (ActiveReservationSemantics::Alias, Some(lineage)) => {
                    lineage.alias_count.set(lineage.alias_count.get().saturating_sub(1));
                    let deferred_release = if lineage.alias_count.get() == 0 {
                        let bytes = lineage.deferred_charged_release_bytes.get();
                        lineage.deferred_charged_release_bytes.set(0);
                        bytes
                    } else {
                        0
                    };
                    if deferred_release > 0 {
                        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(deferred_release));
                        self.record_live_delta(-i128::from(deferred_release));
                        self.released = true;
                        return Some(self.event_context.event(
                            self.release_action,
                            self.bytes,
                            -i128::from(deferred_release),
                            self.reserved_bytes.get(),
                        ));
                    }
                },
                (_, Some(lineage)) => {
                    if lineage.alias_count.get() > 0 {
                        lineage
                            .deferred_charged_release_bytes
                            .set(lineage.deferred_charged_release_bytes.get().saturating_add(self.charged_bytes));
                        self.charged_bytes = 0;
                    }
                },
                (_, None) => {},
            }
        }
        let charged_delta_bytes = -i128::from(self.charged_bytes);
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.charged_bytes));
        self.record_live_delta(charged_delta_bytes);
        self.released = true;
        Some(self.event_context.event(self.release_action, self.bytes, charged_delta_bytes, self.reserved_bytes.get()))
    }

    fn grow(&mut self, bytes: u64) -> u64 {
        if bytes == 0 {
            return 0;
        }
        assert!(!self.released, "cannot grow a released reservation");
        self.bytes = self.bytes.saturating_add(bytes);
        self.charged_bytes = self.charged_bytes.saturating_add(bytes);
        bytes
    }

    fn shrink_to(&mut self, bytes: u64) -> u64 {
        if bytes >= self.bytes {
            return 0;
        }
        assert!(!self.released, "cannot shrink a released reservation");
        let refunded = (self.bytes - bytes).min(self.charged_bytes);
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(refunded));
        self.record_live_delta(-i128::from(refunded));
        self.bytes = bytes;
        self.charged_bytes = self.charged_bytes.saturating_sub(refunded);
        refunded
    }

    fn promote_charged_bytes_to_baseline(&mut self, promoted_bytes: u64) {
        if promoted_bytes == 0 {
            return;
        }
        assert!(!self.released, "cannot promote a released reservation");
        let promoted_charged_bytes = promoted_bytes.min(self.charged_bytes);
        self.charged_bytes = self.charged_bytes.saturating_sub(promoted_charged_bytes);
    }

    fn record_live_delta(&self, delta_bytes: i128) {
        if let Some(grant) = &self.live_accounting {
            grant.record_live_reservation_delta(delta_bytes);
        }
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

pub(in crate::exec) struct ResidentGpuFrame {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
}

pub(in crate::exec) type CommittedPartial = ResidentGpuFrame;

pub(in crate::exec::bounded) fn reconcile_work_reservation_to_output(
    reservations: &mut ReservationTracker,
    work_reservation: &mut ActiveReservation,
    output_bytes: u64,
    output_request: &MemoryReservationRequest,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        reservations.try_grow(work_reservation, output_request.clone().with_requested_bytes(delta), metrics)?;
    } else {
        reservations.shrink_to_observed_with_request(work_reservation, output_bytes, output_request.clone(), metrics);
    }
    Ok(())
}

#[derive(Debug)]
pub(in crate::exec) struct ReservationTracker {
    query_id: Option<QueryId>,
    memory_space: MemorySpaceId,
    budget: ReservationBudget,
    reserved_bytes: Rc<Cell<u64>>,
    local_parquet_source_cache: Option<Arc<GpuLocalParquetSourceCache>>,
    protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
    source_cache_baseline_bytes: u64,
    compiled_task_id: Option<CompiledTaskId>,
    resident_resource_id: Option<String>,
    adaptive_task_refinement: Option<AdaptiveTaskReservationRefinement>,
}

#[derive(Debug, Clone)]
enum ReservationBudget {
    Static(u64),
    GrantBackedStatic { budget_bytes: u64, grant: MemoryGrant },
    AdaptiveGrant(MemoryGrant),
}

impl ReservationBudget {
    fn bytes(&self) -> u64 {
        match self {
            Self::Static(bytes) => *bytes,
            Self::GrantBackedStatic { budget_bytes, .. } => *budget_bytes,
            Self::AdaptiveGrant(grant) => grant.budget_bytes(),
        }
    }

    fn live_accounting(&self) -> Option<MemoryGrant> {
        match self {
            Self::Static(_) => None,
            Self::GrantBackedStatic { grant, .. } => Some(grant.clone()),
            Self::AdaptiveGrant(grant) => Some(grant.clone()),
        }
    }

    fn live_reserved_bytes(&self, tracker_reserved_bytes: u64) -> u64 {
        match self {
            Self::Static(_) => tracker_reserved_bytes,
            Self::GrantBackedStatic { grant, .. } => grant.live_reserved_bytes(),
            Self::AdaptiveGrant(grant) => grant.live_reserved_bytes(),
        }
    }

    fn remaining_live_reservation_bytes(&self, tracker_remaining_bytes: u64) -> Result<u64> {
        match self {
            Self::Static(_) => Ok(tracker_remaining_bytes),
            Self::GrantBackedStatic { grant, .. } => {
                Ok(tracker_remaining_bytes.min(grant.remaining_live_reservation_bytes()?))
            },
            Self::AdaptiveGrant(grant) => Ok(tracker_remaining_bytes.min(grant.remaining_live_reservation_bytes()?)),
        }
    }

    fn try_reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        match self {
            Self::Static(_) => Ok(true),
            Self::GrantBackedStatic { grant, .. } => grant.try_reserve_live_bytes(bytes),
            Self::AdaptiveGrant(grant) => grant.try_reserve_live_bytes(bytes),
        }
    }

    fn release_live_bytes(&self, bytes: u64) {
        match self {
            Self::Static(_) => {},
            Self::GrantBackedStatic { grant, .. } | Self::AdaptiveGrant(grant) => {
                grant.release_live_bytes(bytes);
            },
        }
    }
}

#[derive(Debug, Clone)]
struct AdaptiveTaskReservationRefinement {
    memory_history: Arc<AdaptiveMemoryHistory>,
    task_shapes: BTreeMap<String, AdaptiveTaskMemoryHistoryShape>,
}

impl ReservationTracker {
    pub(in crate::exec) fn new(memory_space: MemorySpaceId, budget_bytes: u64) -> Self {
        Self::new_with_budget(memory_space, ReservationBudget::Static(budget_bytes))
    }

    fn new_with_budget(memory_space: MemorySpaceId, budget: ReservationBudget) -> Self {
        Self {
            query_id: None,
            memory_space,
            budget,
            reserved_bytes: Rc::new(Cell::new(0)),
            local_parquet_source_cache: None,
            protected_source_cache_keys: Rc::new(RefCell::new(HashMap::new())),
            source_cache_baseline_bytes: 0,
            compiled_task_id: None,
            resident_resource_id: None,
            adaptive_task_refinement: None,
        }
    }

    pub(in crate::exec) fn new_for_context(
        context: &ExecutionContext,
        budget_bytes: u64,
        metrics: &mut QueryMetrics,
    ) -> Result<Self> {
        let budget = match (context.native_pipeline_execution_mode(), context.memory_grant().cloned()) {
            (NativePipelineExecutionMode::Adaptive, Some(grant)) => ReservationBudget::AdaptiveGrant(grant),
            (_, Some(grant)) => ReservationBudget::GrantBackedStatic { budget_bytes, grant },
            (_, None) => ReservationBudget::Static(budget_bytes),
        };
        let mut tracker = Self::new_with_budget(context.device_memory_space(), budget);
        tracker.query_id = context.query_id().cloned();
        tracker.local_parquet_source_cache = context.local_parquet_source_cache();
        if let Some(cache) = context.local_parquet_source_cache() {
            let retained_bytes = cache.retained_device_bytes()?;
            if retained_bytes > 0 {
                if !cache.is_governor_backed() {
                    tracker.try_reserve_source_cache_baseline(retained_bytes, metrics)?;
                }
                metrics.record_source_cache_retained_bytes(retained_bytes);
            }
        }
        Ok(tracker)
    }

    pub(in crate::exec) fn new_for_compiled_context(
        context: &ExecutionContext,
        plan: &QueryPlan,
        pipeline: &CompiledPipeline,
        budget_bytes: u64,
        metrics: &mut QueryMetrics,
    ) -> Result<Self> {
        let mut tracker = Self::new_for_context(context, budget_bytes, metrics)?;
        if context.native_pipeline_execution_mode() == NativePipelineExecutionMode::Adaptive
            && let Some(memory_history) = context.adaptive_memory_history()
        {
            tracker.adaptive_task_refinement = Some(AdaptiveTaskReservationRefinement {
                memory_history,
                task_shapes: pipeline
                    .tasks()
                    .iter()
                    .map(|task| (task.id().to_string(), AdaptiveTaskMemoryHistoryShape::for_plan_task(plan, task)))
                    .collect(),
            });
        }
        Ok(tracker)
    }

    pub(in crate::exec) fn budget(&self) -> u64 {
        self.budget.bytes()
    }

    pub(in crate::exec) fn reserved_bytes(&self) -> u64 {
        self.raw_reserved_bytes().saturating_sub(self.source_cache_baseline_bytes)
    }

    pub(in crate::exec) fn remaining_bytes(&self) -> u64 {
        self.budget().saturating_sub(self.raw_reserved_bytes())
    }

    pub(in crate::exec) fn acknowledge_requested_grant_shrink(&self) -> Result<bool> {
        match &self.budget {
            ReservationBudget::Static(_) | ReservationBudget::GrantBackedStatic { .. } => Ok(true),
            ReservationBudget::AdaptiveGrant(grant) => {
                grant.try_apply_requested_shrink(self.budget.live_reserved_bytes(self.raw_reserved_bytes()))
            },
        }
    }

    pub(in crate::exec) fn can_reserve(&self, requested_bytes: u64) -> bool {
        self.reservable_remaining_bytes().is_ok_and(|remaining| requested_bytes <= remaining)
    }

    pub(in crate::exec) fn set_diagnostic_context(
        &mut self,
        compiled_task_id: Option<impl ToString>,
        resident_resource_id: Option<impl ToString>,
    ) {
        self.compiled_task_id = compiled_task_id.map(CompiledTaskId::new);
        self.resident_resource_id = resident_resource_id.map(|value| value.to_string());
    }

    pub(in crate::exec) fn clear_diagnostic_context(&mut self) {
        self.compiled_task_id = None;
        self.resident_resource_id = None;
    }

    pub(in crate::exec) fn available_bytes_including(&self, reservation: &ActiveReservation) -> u64 {
        self.assert_reservation_belongs_to_tracker(reservation, "available_bytes_including");
        reservation.charged_bytes().saturating_add(self.reservable_remaining_bytes().unwrap_or(0))
    }

    pub(in crate::exec) fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        self.try_reserve_owned_resident(request, metrics)
    }

    pub(in crate::exec) fn try_reserve_owned_resident(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        self.try_reserve_charged(request, ActiveReservationSemantics::OwnedResident, metrics)
    }

    pub(in crate::exec) fn try_reserve_work(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        self.try_reserve_charged(request, ActiveReservationSemantics::Work, metrics)
    }

    fn try_reserve_charged(
        &mut self,
        request: MemoryReservationRequest,
        semantics: ActiveReservationSemantics,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let request = self.refine_adaptive_task_reservation_request(self.stamp_request(request, metrics));
        let remaining = self.reservable_remaining_bytes()?;
        if request.requested_bytes > remaining {
            self.reclaim_source_cache_for_request(request.requested_bytes, None, metrics)?;
        }
        let remaining = self.reservable_remaining_bytes()?;
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        if !self.reserve_live_bytes(request.requested_bytes)? {
            let remaining = self.reservable_remaining_bytes()?;
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Reserve,
            request.requested_bytes,
            i128::from(request.requested_bytes),
            self.raw_reserved_bytes(),
        );
        let reservation = match semantics {
            ActiveReservationSemantics::OwnedResident => ActiveReservation::owned_resident(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
                self.budget.live_accounting(),
                Rc::clone(&self.protected_source_cache_keys),
                request.requested_bytes,
                &request,
            ),
            ActiveReservationSemantics::Work => ActiveReservation::work(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
                self.budget.live_accounting(),
                Rc::clone(&self.protected_source_cache_keys),
                request.requested_bytes,
                &request,
            ),
            ActiveReservationSemantics::Alias => unreachable!("alias reservations do not charge capacity"),
        };
        Ok(reservation)
    }

    fn refine_adaptive_task_reservation_request(&self, request: MemoryReservationRequest) -> MemoryReservationRequest {
        let Some(refinement) = &self.adaptive_task_refinement else {
            return request;
        };
        let Some(compiled_task_id) = request.compiled_task_id.as_ref().map(|value| value.as_str()) else {
            return request;
        };
        let Some(shape) = refinement.task_shapes.get(compiled_task_id) else {
            return request;
        };
        let Some(recommended_bytes) = refinement.memory_history.recommended_task_reservation_bytes_for_shape(
            shape,
            request.requested_bytes,
            request.requested_bytes,
        ) else {
            return request;
        };
        if recommended_bytes < request.requested_bytes {
            request.with_requested_bytes(recommended_bytes)
        } else {
            request
        }
    }

    pub(in crate::exec) fn alias_reservation(
        &mut self,
        source: &ActiveReservation,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        self.assert_reservation_belongs_to_tracker(source, "alias_reservation");
        let request = self.stamp_request(request, metrics);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::AliasAcquire,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        ActiveReservation::alias(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            source.live_accounting.clone(),
            Rc::clone(&self.protected_source_cache_keys),
            source.source_cache_pin.clone(),
            Rc::clone(source.lineage.as_ref().expect("generic alias requires a reservation lineage")),
            request.requested_bytes,
            &request,
        )
    }

    pub(in crate::exec) fn transfer(
        &mut self,
        reservation: ActiveReservation,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        self.assert_reservation_belongs_to_tracker(&reservation, "transfer");
        assert!(
            !reservation.is_alias(),
            "reservation tracker invariant failed during transfer: alias reservations cannot transfer ownership"
        );
        let request = self.stamp_request(request, metrics);
        let transferred = reservation.transferred(&request);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::Transfer,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        transferred
    }

    pub(in crate::exec) fn relabel_alias_reservation(
        &mut self,
        reservation: ActiveReservation,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        self.assert_reservation_belongs_to_tracker(&reservation, "relabel_alias_reservation");
        assert!(
            reservation.is_alias(),
            "reservation tracker invariant failed during relabel_alias_reservation: only alias reservations can be relabeled"
        );
        let request = self.stamp_request(request, metrics);
        let relabeled = reservation.relabeled_alias(&request);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::AliasRelabel,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        relabeled
    }

    pub(in crate::exec) fn borrow_accounted_source_cache(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        bytes: u64,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        let request = self.stamp_request(
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, bytes, self.budget())
                .with_memory_boundary("source_cache_borrowed")
                .with_stage("source_cache_hit"),
            metrics,
        );
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::AliasAcquire,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        ActiveReservation::borrowed_source_cache(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            Rc::clone(&self.protected_source_cache_keys),
            key,
            source_cache_pin,
            bytes,
            &request,
        )
    }

    pub(in crate::exec) fn borrow_resident(
        &self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        let request = self.stamp_request(request, metrics);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::AliasAcquire,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        ActiveReservation::borrowed_resident(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            Rc::clone(&self.protected_source_cache_keys),
            request.requested_bytes,
            &request,
        )
    }

    pub(in crate::exec) fn release(&mut self, mut reservation: ActiveReservation, metrics: &mut QueryMetrics) {
        self.assert_reservation_belongs_to_tracker(&reservation, "release");
        if let Some(event) = reservation.release() {
            // Release and alias-release events describe live ledger movement only; aggregate
            // requested/granted counters intentionally remain historical admission counters.
            let mut event = event;
            if event.compiled_task_id.is_none() {
                event.compiled_task_id = self.compiled_task_id.clone();
            }
            if event.query_id.is_none() {
                event.query_id = self.query_id.clone();
            }
            if event.resident_resource_id.is_none() {
                event.resident_resource_id = self.resident_resource_id.clone();
            }
            metrics.record_reservation_timeline_event(event);
        }
    }

    pub(in crate::exec) fn try_grow(
        &mut self,
        reservation: &mut ActiveReservation,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
        self.validate_reservation_belongs_to_tracker(reservation, "try_grow")?;
        let request = self.stamp_request(request, metrics);
        let remaining = self.reservable_remaining_bytes()?;
        if request.requested_bytes > remaining {
            self.reclaim_source_cache_for_request(request.requested_bytes, None, metrics)?;
        }
        let remaining = self.reservable_remaining_bytes()?;
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Grow,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        if !self.reserve_live_bytes(request.requested_bytes)? {
            let remaining = self.reservable_remaining_bytes()?;
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Grow,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        let charged_delta = reservation.grow(request.requested_bytes);
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Grow,
            request.requested_bytes,
            i128::from(charged_delta),
            self.raw_reserved_bytes(),
        );
        Ok(())
    }

    pub(in crate::exec) fn shrink_to_observed(
        &mut self,
        reservation: &mut ActiveReservation,
        observed_bytes: u64,
        metrics: &mut QueryMetrics,
    ) {
        self.assert_reservation_belongs_to_tracker(reservation, "shrink_to_observed");
        let refunded = reservation.shrink_to(observed_bytes);
        if refunded > 0 {
            metrics.record_reservation_timeline_event(reservation.event_context.event(
                MemoryReservationEventAction::Shrink,
                observed_bytes,
                -i128::from(refunded),
                self.raw_reserved_bytes(),
            ));
        }
    }

    pub(in crate::exec) fn shrink_to_observed_with_request(
        &mut self,
        reservation: &mut ActiveReservation,
        observed_bytes: u64,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) {
        self.assert_reservation_belongs_to_tracker(reservation, "shrink_to_observed_with_request");
        let refunded = reservation.shrink_to(observed_bytes);
        if refunded > 0 {
            let request = self.stamp_request(request, metrics);
            metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
                &request,
                MemoryReservationEventOutcome::Applied,
                MemoryReservationEventAction::Shrink,
                0,
                -i128::from(refunded),
                self.raw_reserved_bytes(),
            ));
        }
    }

    pub(in crate::exec) fn promote_source_cache_insert(
        &mut self,
        reservation: &mut ActiveReservation,
        retained_bytes: u64,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
        protected_key: &GpuLocalParquetSourceCacheKey,
    ) -> Result<()> {
        self.validate_reservation_belongs_to_tracker(reservation, "promote_source_cache_insert")?;
        if retained_bytes == 0 {
            return Ok(());
        }

        let request = self.stamp_request(request, metrics);
        if self.local_parquet_source_cache.as_ref().is_some_and(|cache| cache.is_governor_backed()) {
            metrics.record_reservation_granted_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                retained_bytes,
                0,
                self.raw_reserved_bytes(),
            );
            return Ok(());
        }
        let additional_bytes = retained_bytes.saturating_sub(reservation.charged_bytes());
        let remaining = self.reservable_remaining_bytes()?;
        if additional_bytes > remaining {
            self.reclaim_source_cache_for_request(additional_bytes, Some(protected_key), metrics)?;
        }
        let remaining = self.reservable_remaining_bytes()?;
        if additional_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }

        if additional_bytes > 0 && !self.reserve_live_bytes(additional_bytes)? {
            let remaining = self.reservable_remaining_bytes()?;
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        reservation.promote_charged_bytes_to_baseline(retained_bytes);
        self.source_cache_baseline_bytes = self.source_cache_baseline_bytes.saturating_add(retained_bytes);
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Reserve,
            retained_bytes,
            i128::from(additional_bytes),
            self.raw_reserved_bytes(),
        );
        Ok(())
    }

    pub(in crate::exec) fn release_source_cache_baseline_bytes(&mut self, bytes: u64, metrics: &mut QueryMetrics) {
        let released = bytes.min(self.source_cache_baseline_bytes);
        if released == 0 {
            return;
        }
        self.source_cache_baseline_bytes = self.source_cache_baseline_bytes.saturating_sub(released);
        self.release_live_bytes(released);
        let request = self.stamp_request(
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, released, self.budget())
                .with_memory_boundary("source_cache_retained")
                .with_stage("source_cache_eviction"),
            metrics,
        );
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::Release,
            0,
            -i128::from(released),
            self.raw_reserved_bytes(),
        ));
    }

    pub(in crate::exec::bounded) fn reclaim_source_cache_for_external_working_set(
        &mut self,
        working_set_bytes: u64,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
        let Some(cache) = self.local_parquet_source_cache.as_ref().map(Arc::clone) else {
            return Ok(());
        };
        if working_set_bytes == 0 || self.source_cache_baseline_bytes == 0 {
            return Ok(());
        }
        let evicted_bytes = cache.evict_at_least_protecting(working_set_bytes, &self.source_cache_protected_keys())?;
        self.release_source_cache_baseline_bytes(evicted_bytes, metrics);
        metrics.record_source_cache_retained_bytes(cache.retained_device_bytes()?);
        Ok(())
    }

    pub(in crate::exec) fn source_cache_protected_keys_with(
        &self,
        additional_key: &GpuLocalParquetSourceCacheKey,
    ) -> HashSet<GpuLocalParquetSourceCacheKey> {
        let mut protected = self.source_cache_protected_keys();
        protected.insert(additional_key.clone());
        protected
    }

    pub(in crate::exec) fn source_cache_protected_key_count(&self, key: &GpuLocalParquetSourceCacheKey) -> usize {
        self.protected_source_cache_keys.borrow().get(key).copied().unwrap_or(0)
    }

    fn try_reserve_source_cache_baseline(&mut self, retained_bytes: u64, metrics: &mut QueryMetrics) -> Result<()> {
        let request =
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, retained_bytes, self.budget())
                .with_memory_boundary("source_cache_retained")
                .with_stage("source_cache_retained");
        let request = self.stamp_request(request, metrics);
        let remaining = self.reservable_remaining_bytes()?;
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        if !self.reserve_live_bytes(request.requested_bytes)? {
            let remaining = self.reservable_remaining_bytes()?;
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        self.source_cache_baseline_bytes = self.source_cache_baseline_bytes.saturating_add(request.requested_bytes);
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Reserve,
            request.requested_bytes,
            i128::from(request.requested_bytes),
            self.raw_reserved_bytes(),
        );
        Ok(())
    }

    fn reclaim_source_cache_for_request(
        &mut self,
        requested_bytes: u64,
        additional_protected_key: Option<&GpuLocalParquetSourceCacheKey>,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
        let Some(cache) = self.local_parquet_source_cache.as_ref().map(Arc::clone) else {
            return Ok(());
        };
        let remaining = self.reservable_remaining_bytes()?;
        if requested_bytes <= remaining {
            return Ok(());
        }
        let required_bytes = requested_bytes.saturating_sub(remaining);
        if required_bytes == 0 || self.source_cache_baseline_bytes == 0 {
            return Ok(());
        }

        let mut protected = self.source_cache_protected_keys();
        if let Some(additional_protected_key) = additional_protected_key {
            protected.insert(additional_protected_key.clone());
        }
        let evicted_bytes = cache.evict_at_least_protecting(required_bytes, &protected)?;
        self.release_source_cache_baseline_bytes(evicted_bytes, metrics);
        metrics.record_source_cache_retained_bytes(cache.retained_device_bytes()?);
        Ok(())
    }

    fn source_cache_protected_keys(&self) -> HashSet<GpuLocalParquetSourceCacheKey> {
        self.protected_source_cache_keys.borrow().keys().cloned().collect()
    }

    fn stamp_request(&self, mut request: MemoryReservationRequest, metrics: &QueryMetrics) -> MemoryReservationRequest {
        request.memory_space = self.memory_space;
        request.budget_bytes = self.budget();
        if request.query_id.is_none() {
            request.query_id =
                self.query_id.clone().or_else(|| metrics.query_id().is_assigned().then(|| metrics.query_id().clone()));
        }
        if request.compiled_task_id.is_none() {
            request.compiled_task_id = self.compiled_task_id.clone();
        }
        if request.resident_resource_id.is_none() {
            request.resident_resource_id = self.resident_resource_id.clone();
        }
        request
    }

    fn raw_reserved_bytes(&self) -> u64 {
        self.reserved_bytes.get()
    }

    pub(in crate::exec) fn reservable_remaining_bytes(&self) -> Result<u64> {
        self.budget.remaining_live_reservation_bytes(self.remaining_bytes())
    }

    fn reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        if !self.budget.try_reserve_live_bytes(bytes)? {
            return Ok(false);
        }
        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(bytes));
        Ok(true)
    }

    fn release_live_bytes(&self, bytes: u64) {
        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_sub(bytes));
        self.budget.release_live_bytes(bytes);
    }

    fn reservation_belongs_to_tracker(&self, reservation: &ActiveReservation) -> bool {
        reservation.memory_space() == self.memory_space && Rc::ptr_eq(&reservation.reserved_bytes, &self.reserved_bytes)
    }

    fn assert_reservation_belongs_to_tracker(&self, reservation: &ActiveReservation, operation: &'static str) {
        assert!(
            self.reservation_belongs_to_tracker(reservation),
            "reservation tracker invariant failed during {operation}: reservation memory_space={} tracker memory_space={}",
            reservation.memory_space(),
            self.memory_space
        );
    }

    fn validate_reservation_belongs_to_tracker(
        &self,
        reservation: &ActiveReservation,
        operation: &'static str,
    ) -> Result<()> {
        if self.reservation_belongs_to_tracker(reservation) {
            return Ok(());
        }

        Err(QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.bounded.reservation_tracker",
            "active reservation does not belong to this reservation tracker",
        )
        .with_fact("operation", operation)
        .with_fact("reservation_memory_space", reservation.memory_space())
        .with_fact("tracker_memory_space", self.memory_space))
    }
}

impl Drop for ReservationTracker {
    fn drop(&mut self) {
        if self.source_cache_baseline_bytes > 0 {
            self.release_live_bytes(self.source_cache_baseline_bytes);
            self.source_cache_baseline_bytes = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{GrantRequest, MemoryGovernor};

    fn test_request(
        kind: ReservationKind,
        boundary: &'static str,
        bytes: u64,
        budget: u64,
    ) -> MemoryReservationRequest {
        MemoryReservationRequest::new(kind, bytes, budget).with_memory_boundary(boundary).with_stage("test")
    }

    #[test]
    fn adaptive_reservation_tracker_observes_live_grant_growth() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 40, 40))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );

        let first = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "first", 30, reservations.budget()),
            &mut metrics,
        )?;
        let denied = match reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "before_growth", 20, reservations.budget()),
            &mut metrics,
        ) {
            Ok(_) => panic!("only 10 bytes should remain before grant growth"),
            Err(error) => error,
        };
        assert_eq!(denied.fact_value("granted_bytes"), Some("10"));

        assert!(grant.try_grow_to(70)?);
        assert_eq!(reservations.budget(), 70);
        assert_eq!(reservations.remaining_bytes(), 40);

        let second = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "after_growth", 20, reservations.budget()),
            &mut metrics,
        )?;

        let granted = metrics
            .reservation_events
            .iter()
            .rev()
            .find(|event| {
                event.outcome == MemoryReservationEventOutcome::Granted
                    && event.action == MemoryReservationEventAction::Reserve
                    && event.memory_boundary == "after_growth"
            })
            .expect("post-growth reservation should be recorded");
        assert_eq!(granted.budget_bytes, 70);

        reservations.release(second, &mut metrics);
        reservations.release(first, &mut metrics);
        Ok(())
    }

    #[test]
    fn static_reservation_tracker_ignores_later_grant_growth() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(8, 40, 40))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations =
            ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, grant.budget_bytes());

        let first = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "first", 30, reservations.budget()),
            &mut metrics,
        )?;
        assert!(grant.try_grow_to(70)?);
        assert_eq!(grant.budget_bytes(), 70);
        assert_eq!(reservations.budget(), 40);
        assert_eq!(reservations.remaining_bytes(), 10);

        let denied = match reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "after_external_growth", 20, reservations.budget()),
            &mut metrics,
        ) {
            Ok(_) => panic!("static reservation tracker must keep the original budget"),
            Err(error) => error,
        };
        assert_eq!(denied.fact_value("granted_bytes"), Some("10"));

        reservations.release(first, &mut metrics);
        Ok(())
    }

    #[test]
    fn static_reservation_tracker_acknowledges_grant_shrink_as_noop() -> Result<()> {
        let reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);

        assert!(reservations.acknowledge_requested_grant_shrink()?);
        assert_eq!(reservations.budget(), 100);
        Ok(())
    }

    #[test]
    fn grant_backed_static_tracker_retains_live_output_bytes() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 10, 40))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::GrantBackedStatic { budget_bytes: 40, grant: grant.clone() },
        );

        let active = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "strict_output", 30, reservations.budget()),
            &mut metrics,
        )?;
        assert_eq!(grant.live_reserved_bytes(), 30);

        let retained = active.retain_grant_live_reservation().expect("strict grant should retain live output bytes");
        drop(active);

        assert_eq!(grant.live_reserved_bytes(), 30);
        assert!(grant.try_shrink_to(0)?);
        assert_eq!(grant.budget_bytes(), 30);

        drop(retained);
        assert_eq!(grant.live_reserved_bytes(), 0);
        assert!(grant.try_shrink_to(0)?);
        assert_eq!(grant.budget_bytes(), 10);
        Ok(())
    }

    #[test]
    fn adaptive_reservation_tracker_acknowledges_shrink_without_underflowing_live_reservations() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 80))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let live = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "live", 55, reservations.budget()),
            &mut metrics,
        )?;

        assert_eq!(grant.request_shrink_to(40)?, Some(40));
        assert!(!reservations.acknowledge_requested_grant_shrink()?);

        assert_eq!(reservations.reserved_bytes(), 55);
        assert_eq!(grant.budget_bytes(), 55);
        assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
        assert_eq!(governor.available_bytes()?, 45);

        reservations.release(live, &mut metrics);
        Ok(())
    }

    #[test]
    fn adaptive_shrink_ack_uses_grant_scoped_live_bytes_across_trackers() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 90))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut first_tracker = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let mut second_tracker = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let first_live = first_tracker.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "first_live", 60, first_tracker.budget()),
            &mut metrics,
        )?;
        let second_live = second_tracker.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "second_live", 20, second_tracker.budget()),
            &mut metrics,
        )?;

        assert_eq!(grant.live_reserved_bytes(), 80);
        assert_eq!(grant.request_shrink_to(40)?, Some(40));
        assert!(!second_tracker.acknowledge_requested_grant_shrink()?);

        assert_eq!(second_tracker.raw_reserved_bytes(), 20);
        assert_eq!(grant.budget_bytes(), 80);
        assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
        assert_eq!(governor.available_bytes()?, 20);

        first_tracker.release(first_live, &mut metrics);
        assert!(second_tracker.acknowledge_requested_grant_shrink()?);
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes()?, None);
        assert_eq!(governor.available_bytes()?, 60);

        second_tracker.release(second_live, &mut metrics);
        assert_eq!(grant.live_reserved_bytes(), 0);
        Ok(())
    }

    #[test]
    fn adaptive_reservation_tracker_denies_when_shared_grant_live_budget_is_exhausted() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 70))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut first_tracker = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let mut second_tracker = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let first_live = first_tracker.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "first_live", 50, first_tracker.budget()),
            &mut metrics,
        )?;

        let denied = match second_tracker.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "second_live", 30, second_tracker.budget()),
            &mut metrics,
        ) {
            Ok(_) => panic!("second tracker must be capped by the shared grant live footprint"),
            Err(error) => error,
        };

        assert_eq!(denied.fact_value("granted_bytes"), Some("20"));
        assert_eq!(grant.live_reserved_bytes(), 50);

        first_tracker.release(first_live, &mut metrics);
        assert_eq!(grant.live_reserved_bytes(), 0);
        Ok(())
    }

    #[test]
    fn adaptive_reservation_tracker_borrowed_resident_does_not_double_charge_grant_live_bytes() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 70))?.expect("initial grant should fit");
        let retained_input = grant.retain_live_reservation(50).expect("input handle should retain live bytes");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );

        let borrowed = reservations.borrow_resident(
            test_request(ReservationKind::SourceRead, "gpu_resident_source", 50, reservations.budget()),
            &mut metrics,
        );

        assert_eq!(reservations.raw_reserved_bytes(), 0);
        assert_eq!(borrowed.bytes(), 50);
        assert_eq!(borrowed.charged_bytes(), 0);
        assert_eq!(grant.live_reserved_bytes(), 50);

        let work = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "consumer_work", 20, reservations.budget()),
            &mut metrics,
        )?;
        assert_eq!(grant.live_reserved_bytes(), 70);

        reservations.release(borrowed, &mut metrics);
        assert_eq!(grant.live_reserved_bytes(), 70);
        reservations.release(work, &mut metrics);
        assert_eq!(grant.live_reserved_bytes(), 50);
        drop(retained_input);
        assert_eq!(grant.live_reserved_bytes(), 0);
        Ok(())
    }

    #[test]
    fn alias_reservation_over_borrowed_resident_releases_without_charging() {
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);

        let borrowed = reservations.borrow_resident(
            test_request(ReservationKind::SourceRead, "gpu_resident_source", 40, reservations.budget()),
            &mut metrics,
        );
        let lineage = Rc::clone(borrowed.lineage.as_ref().expect("borrowed resident should carry lineage"));
        let alias = reservations.alias_reservation(
            &borrowed,
            test_request(ReservationKind::OperatorOutput, "borrowed_alias", 40, reservations.budget()),
            &mut metrics,
        );

        assert!(alias.is_alias());
        assert_eq!(alias.charged_bytes(), 0);
        assert_eq!(reservations.reserved_bytes(), 0);
        assert_eq!(reservations.raw_reserved_bytes(), 0);
        assert_eq!(lineage.alias_count.get(), 1);

        reservations.release(alias, &mut metrics);
        assert_eq!(lineage.alias_count.get(), 0);
        assert_eq!(lineage.deferred_charged_release_bytes.get(), 0);
        assert_eq!(reservations.reserved_bytes(), 0);

        reservations.release(borrowed, &mut metrics);
        assert_eq!(lineage.deferred_charged_release_bytes.get(), 0);
        assert_eq!(reservations.reserved_bytes(), 0);
        assert_eq!(reservations.raw_reserved_bytes(), 0);
    }

    #[test]
    fn adaptive_reservation_tracker_completes_shrink_after_live_bytes_drop() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 80))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );
        let live = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "live", 55, reservations.budget()),
            &mut metrics,
        )?;

        assert_eq!(grant.request_shrink_to(40)?, Some(40));
        assert!(!reservations.acknowledge_requested_grant_shrink()?);
        reservations.release(live, &mut metrics);
        assert!(reservations.acknowledge_requested_grant_shrink()?);

        assert_eq!(reservations.reserved_bytes(), 0);
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes()?, None);
        assert_eq!(governor.available_bytes()?, 60);
        Ok(())
    }

    #[test]
    fn adaptive_reservation_tracker_shrink_ack_counts_source_cache_baseline() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor.try_request(GrantRequest::new(7, 30, 80))?.expect("initial grant should fit");
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::AdaptiveGrant(grant.clone()),
        );

        reservations.try_reserve_source_cache_baseline(55, &mut metrics)?;
        assert_eq!(reservations.reserved_bytes(), 0);
        assert_eq!(reservations.raw_reserved_bytes(), 55);
        assert_eq!(grant.request_shrink_to(40)?, Some(40));

        assert!(!reservations.acknowledge_requested_grant_shrink()?);

        assert_eq!(grant.budget_bytes(), 55);
        assert_eq!(grant.requested_shrink_target_bytes()?, Some(40));
        assert_eq!(governor.available_bytes()?, 45);
        Ok(())
    }

    #[test]
    fn reconcile_work_reservation_to_output_grows_with_output_request_context() -> Result<()> {
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let mut work = reservations.try_reserve_work(
            test_request(ReservationKind::OperatorOutput, "operator_work", 40, reservations.budget()),
            &mut metrics,
        )?;
        let output_request =
            test_request(ReservationKind::OperatorOutput, "operator_output", 60, reservations.budget());

        reconcile_work_reservation_to_output(&mut reservations, &mut work, 60, &output_request, &mut metrics)?;

        assert_eq!(work.bytes(), 60);
        assert_eq!(reservations.reserved_bytes(), 60);
        let actions = metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
        assert_eq!(actions, [MemoryReservationEventAction::Reserve, MemoryReservationEventAction::Grow]);
        let boundaries =
            metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
        assert_eq!(boundaries, ["operator_work", "operator_output"]);

        let output = reservations.transfer(work, output_request, &mut metrics);
        reservations.release(output, &mut metrics);
        Ok(())
    }

    #[test]
    fn reconcile_work_reservation_to_output_shrinks_with_output_request_context() -> Result<()> {
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let mut work = reservations.try_reserve_work(
            test_request(ReservationKind::OperatorOutput, "operator_work", 60, reservations.budget()),
            &mut metrics,
        )?;
        let output_request =
            test_request(ReservationKind::OperatorOutput, "operator_output", 40, reservations.budget());

        reconcile_work_reservation_to_output(&mut reservations, &mut work, 40, &output_request, &mut metrics)?;

        assert_eq!(work.bytes(), 40);
        assert_eq!(reservations.reserved_bytes(), 40);
        let actions = metrics.reservation_events.iter().map(|event| event.action).collect::<Vec<_>>();
        assert_eq!(actions, [MemoryReservationEventAction::Reserve, MemoryReservationEventAction::Shrink]);
        let deltas = metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).collect::<Vec<_>>();
        assert_eq!(deltas, [60, -20]);
        let boundaries =
            metrics.reservation_events.iter().map(|event| event.memory_boundary.as_str()).collect::<Vec<_>>();
        assert_eq!(boundaries, ["operator_work", "operator_output"]);

        let output = reservations.transfer(work, output_request, &mut metrics);
        reservations.release(output, &mut metrics);
        Ok(())
    }

    #[test]
    fn reconcile_work_reservation_to_output_denial_leaves_release_to_caller() -> Result<()> {
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 50);
        let mut work = reservations.try_reserve_work(
            test_request(ReservationKind::OperatorOutput, "operator_work", 40, reservations.budget()),
            &mut metrics,
        )?;
        let output_request =
            test_request(ReservationKind::OperatorOutput, "operator_output", 60, reservations.budget());

        let error =
            reconcile_work_reservation_to_output(&mut reservations, &mut work, 60, &output_request, &mut metrics)
                .expect_err("grow beyond budget should be denied");

        assert_eq!(error.fact_value("memory_boundary"), Some("operator_output"));
        assert_eq!(error.fact_value("requested_bytes"), Some("20"));
        assert_eq!(error.fact_value("granted_bytes"), Some("10"));
        assert_eq!(work.bytes(), 40);
        assert_eq!(reservations.reserved_bytes(), 40);
        let denied = metrics
            .reservation_events
            .iter()
            .find(|event| event.outcome == MemoryReservationEventOutcome::Denied)
            .expect("grow denial should be recorded");
        assert_eq!(denied.action, MemoryReservationEventAction::Grow);
        assert_eq!(denied.memory_boundary, "operator_output");

        reservations.release(work, &mut metrics);
        assert_eq!(reservations.reserved_bytes(), 0);
        Ok(())
    }
}
