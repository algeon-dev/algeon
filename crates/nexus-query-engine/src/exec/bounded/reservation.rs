use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use super::*;
use crate::{
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveTaskMemoryHistoryShape},
    exec::types::{GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey},
    metrics::{MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome},
    mode::NativePipelineExecutionMode,
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    runtime::{ExecutionContext, MemorySpaceId},
};

#[derive(Debug, Clone)]
struct ActiveReservationEventContext {
    reservation_kind: ReservationKind,
    memory_space: MemorySpaceId,
    memory_boundary: String,
    budget_bytes: u64,
    node_id: Option<String>,
    stage: Option<String>,
    chunk_index: Option<usize>,
    file_index: Option<usize>,
    path: Option<String>,
    row_groups: Option<Vec<usize>>,
    attempt: Option<usize>,
    max_attempts: Option<usize>,
    compiled_task_id: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum ActiveReservationSemantics {
    OwnedResident,
    Work,
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
    protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
    source_cache_key: Option<GpuLocalParquetSourceCacheKey>,
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
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            protected_source_cache_keys,
            bytes,
            request,
            ActiveReservationSemantics::OwnedResident,
        )
    }

    pub(in crate::exec) fn work(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            protected_source_cache_keys,
            bytes,
            request,
            ActiveReservationSemantics::Work,
        )
    }

    fn charged(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        semantics: ActiveReservationSemantics,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            protected_source_cache_keys,
            source_cache_key: None,
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
            protected_source_cache_keys,
            source_cache_key: Some(source_cache_key),
            event_context: ActiveReservationEventContext::from_request(request),
            semantics: ActiveReservationSemantics::Alias,
            lineage: None,
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    fn alias(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        lineage: Rc<ActiveReservationLineage>,
        bytes: u64,
        request: &MemoryReservationRequest,
    ) -> Self {
        lineage.alias_count.set(lineage.alias_count.get().saturating_add(1));
        Self {
            memory_space,
            reserved_bytes,
            protected_source_cache_keys,
            source_cache_key: None,
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
            protected_source_cache_keys: Rc::clone(&self.protected_source_cache_keys),
            source_cache_key: None,
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
        let lineage = self.lineage.clone();
        self.released = true;
        Self {
            memory_space: self.memory_space,
            reserved_bytes: Rc::clone(&self.reserved_bytes),
            protected_source_cache_keys: Rc::clone(&self.protected_source_cache_keys),
            source_cache_key,
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
        self.released = true;
        Some(self.event_context.event(self.release_action, self.bytes, charged_delta_bytes, self.reserved_bytes.get()))
    }

    fn grow(&mut self, bytes: u64) -> u64 {
        if bytes == 0 {
            return 0;
        }
        assert!(!self.released, "cannot grow a released reservation");
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_add(bytes));
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
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[derive(Debug)]
pub(in crate::exec) struct ReservationTracker {
    memory_space: MemorySpaceId,
    budget_bytes: u64,
    reserved_bytes: Rc<Cell<u64>>,
    local_parquet_source_cache: Option<Arc<GpuLocalParquetSourceCache>>,
    protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
    source_cache_baseline_bytes: u64,
    compiled_task_id: Option<String>,
    resident_resource_id: Option<String>,
    adaptive_task_refinement: Option<AdaptiveTaskReservationRefinement>,
}

#[derive(Debug, Clone)]
struct AdaptiveTaskReservationRefinement {
    memory_history: Arc<AdaptiveMemoryHistory>,
    task_shapes: BTreeMap<String, AdaptiveTaskMemoryHistoryShape>,
}

impl ReservationTracker {
    pub(in crate::exec) fn new(memory_space: MemorySpaceId, budget_bytes: u64) -> Self {
        Self {
            memory_space,
            budget_bytes,
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
        let mut tracker = Self::new(context.device_memory_space(), budget_bytes);
        tracker.local_parquet_source_cache = context.local_parquet_source_cache();
        if let Some(cache) = context.local_parquet_source_cache() {
            let retained_bytes = cache.retained_device_bytes()?;
            if retained_bytes > 0 {
                tracker.try_reserve_source_cache_baseline(retained_bytes, metrics)?;
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
        self.budget_bytes
    }

    pub(in crate::exec) fn reserved_bytes(&self) -> u64 {
        self.raw_reserved_bytes().saturating_sub(self.source_cache_baseline_bytes)
    }

    pub(in crate::exec) fn remaining_bytes(&self) -> u64 {
        self.budget_bytes.saturating_sub(self.raw_reserved_bytes())
    }

    pub(in crate::exec) fn can_reserve(&self, requested_bytes: u64) -> bool {
        requested_bytes <= self.remaining_bytes()
    }

    pub(in crate::exec) fn set_diagnostic_context(
        &mut self,
        compiled_task_id: Option<impl ToString>,
        resident_resource_id: Option<impl ToString>,
    ) {
        self.compiled_task_id = compiled_task_id.map(|value| value.to_string());
        self.resident_resource_id = resident_resource_id.map(|value| value.to_string());
    }

    pub(in crate::exec) fn clear_diagnostic_context(&mut self) {
        self.compiled_task_id = None;
        self.resident_resource_id = None;
    }

    pub(in crate::exec) fn available_bytes_including(&self, reservation: &ActiveReservation) -> u64 {
        self.assert_reservation_belongs_to_tracker(reservation, "available_bytes_including");
        reservation.charged_bytes().saturating_add(self.remaining_bytes())
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
        let request = self.refine_adaptive_task_reservation_request(self.stamp_request(request));
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            self.reclaim_source_cache_for_request(request.requested_bytes, None, metrics)?;
        }
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(request.requested_bytes));
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
                Rc::clone(&self.protected_source_cache_keys),
                request.requested_bytes,
                &request,
            ),
            ActiveReservationSemantics::Work => ActiveReservation::work(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
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
        let Some(compiled_task_id) = request.compiled_task_id.as_deref() else {
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
        let request = self.stamp_request(request);
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
            Rc::clone(&self.protected_source_cache_keys),
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
        let request = self.stamp_request(request);
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
        let request = self.stamp_request(request);
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
        metrics: &mut QueryMetrics,
    ) -> ActiveReservation {
        let request = self.stamp_request(
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, bytes, self.budget_bytes)
                .with_memory_boundary("source_cache_borrowed")
                .with_stage("source_cache_hit"),
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
            bytes,
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
        let request = self.stamp_request(request);
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            self.reclaim_source_cache_for_request(request.requested_bytes, None, metrics)?;
        }
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
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
            let request = self.stamp_request(request);
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

        let request = self.stamp_request(request);
        let additional_bytes = retained_bytes.saturating_sub(reservation.charged_bytes());
        let remaining = self.remaining_bytes();
        if additional_bytes > remaining {
            self.reclaim_source_cache_for_request(additional_bytes, Some(protected_key), metrics)?;
        }
        let remaining = self.remaining_bytes();
        if additional_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }

        if additional_bytes > 0 {
            self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(additional_bytes));
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
        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_sub(released));
        let request = self.stamp_request(
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, released, self.budget_bytes)
                .with_memory_boundary("source_cache_retained")
                .with_stage("source_cache_eviction"),
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
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, retained_bytes, self.budget_bytes)
                .with_memory_boundary("source_cache_retained")
                .with_stage("source_cache_retained");
        let request = self.stamp_request(request);
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_timeline_event(
                &request,
                MemoryReservationEventAction::Reserve,
                remaining,
                self.raw_reserved_bytes(),
            );
            return Err(request.denied_error(remaining));
        }
        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(request.requested_bytes));
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
        let remaining = self.remaining_bytes();
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

    fn stamp_request(&self, mut request: MemoryReservationRequest) -> MemoryReservationRequest {
        request.memory_space = self.memory_space;
        request.budget_bytes = self.budget_bytes;
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

pub(in crate::exec) struct CommittedPartial {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
}

pub(in crate::exec) type ResidentGpuFrame = CommittedPartial;
