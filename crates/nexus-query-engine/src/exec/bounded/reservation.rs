use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet, VecDeque},
    sync::Arc,
};

use super::*;
use crate::{
    adaptive_runtime::{AdaptiveMemoryHistory, AdaptiveTaskMemoryHistoryShape},
    metrics::{
        CompiledTaskId, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
        PlanNodeId, QueryId, ReservationProjectionLedgerHandle,
    },
    mode::NativePipelineExecutionMode,
    observability::{
        RuntimeEventContext, RuntimeExecutionObservationScope, RuntimeObservationPayload, RuntimeReservationAction,
        RuntimeReservationDisposition, RuntimeReservationId, RuntimeSpanId, TaskAttemptId,
    },
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    runtime::{
        ExecutionContext, MemoryGrant, MemorySpaceId, cache::local_parquet::*,
        gpu::governor::MemoryGrantLiveReservation,
    },
};

#[derive(Debug, Clone)]
struct ReservationObservation {
    scope: RuntimeExecutionObservationScope,
    reservation_id: RuntimeReservationId,
    lineage_id: RuntimeReservationId,
    task_attempt_id: Option<TaskAttemptId>,
    span_id: Option<RuntimeSpanId>,
}

#[derive(Debug, Clone)]
struct SourceCacheBaselineObservation {
    observation: Option<ReservationObservation>,
    event_context: ActiveReservationEventContext,
    reservation_projection_ledger: ReservationProjectionLedgerHandle,
    remaining_bytes: u64,
}

impl ReservationObservation {
    fn retarget(&mut self, task_attempt_id: Option<TaskAttemptId>, span_id: Option<RuntimeSpanId>) {
        self.task_attempt_id = task_attempt_id;
        self.span_id = span_id;
    }

    fn record(
        &self,
        context: &ActiveReservationEventContext,
        action: RuntimeReservationAction,
        disposition: RuntimeReservationDisposition,
        requested_bytes: u64,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) {
        self.record_with_denied_action(
            context,
            action,
            None,
            disposition,
            requested_bytes,
            granted_bytes,
            charged_delta_bytes,
            current_reserved_bytes,
        );
    }

    fn record_denied(
        &self,
        context: &ActiveReservationEventContext,
        denied_action: RuntimeReservationAction,
        requested_bytes: u64,
        granted_bytes: u64,
        current_reserved_bytes: u64,
    ) {
        self.record_with_denied_action(
            context,
            RuntimeReservationAction::Deny,
            Some(denied_action),
            RuntimeReservationDisposition::Denied,
            requested_bytes,
            granted_bytes,
            0,
            current_reserved_bytes,
        );
    }

    fn record_with_denied_action(
        &self,
        context: &ActiveReservationEventContext,
        action: RuntimeReservationAction,
        denied_action: Option<RuntimeReservationAction>,
        disposition: RuntimeReservationDisposition,
        requested_bytes: u64,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) {
        let event_context = match self.span_id {
            Some(span_id) => RuntimeEventContext::for_plan(self.scope.plan_execution_id().clone())
                .with_span(span_id, Some(self.scope.plan_span_id())),
            None => RuntimeEventContext::for_plan(self.scope.plan_execution_id().clone())
                .with_span(self.scope.plan_span_id(), None),
        };
        let logical_bytes = self.logical_retained_bytes(context);
        self.scope.observation().record_with_context_sized(event_context, logical_bytes, || {
            RuntimeObservationPayload::Reservation(Box::new(crate::observability::RuntimeReservationObservation {
                reservation_id: self.reservation_id.clone(),
                lineage_id: self.lineage_id.clone(),
                task_attempt_id: context.task_attempt_id.clone().or_else(|| self.task_attempt_id.clone()),
                task_kind: context.task_kind.clone(),
                action,
                denied_action,
                disposition,
                alias: context.alias,
                reservation_kind: context.reservation_kind.as_str().to_string(),
                memory_space: context.memory_space.to_string(),
                memory_boundary: context.memory_boundary.clone(),
                requested_bytes,
                granted_bytes,
                charged_delta_bytes,
                current_reserved_bytes,
                budget_bytes: context.budget_bytes,
                node_id: context.node_id.as_ref().map(ToString::to_string),
                stage: context.stage.clone(),
                chunk_index: context.chunk_index,
                file_index: context.file_index,
                source_location_id: context
                    .path
                    .as_deref()
                    .and_then(|path| self.scope.observation().source_location_id(path)),
                row_groups: context.row_groups.clone(),
                attempt: context.attempt,
                max_attempts: context.max_attempts,
                resident_resource_id: context.resident_resource_id.clone(),
                source_read_contract_quality: context.source_read_contract_quality.clone(),
                source_read_planning_bytes: context.source_read_planning_bytes,
                source_read_estimated_bytes: context.source_read_estimated_bytes,
            }))
        });
    }

    fn logical_retained_bytes(&self, context: &ActiveReservationEventContext) -> u64 {
        fn optional_text_bytes(value: Option<&str>) -> u64 {
            value.map_or(0, |value| u64::try_from(value.len()).unwrap_or(u64::MAX))
        }

        fn decimal_digits(mut value: usize) -> u64 {
            let mut digits = 1u64;
            while value >= 10 {
                value /= 10;
                digits += 1;
            }
            digits
        }

        let task_attempt = context.task_attempt_id.as_ref().or(self.task_attempt_id.as_ref());
        let memory_space_bytes = match context.memory_space {
            MemorySpaceId::Device { device_ordinal } => 4u64.saturating_add(decimal_digits(device_ordinal)),
        };
        let source_location_bytes = context
            .path
            .as_ref()
            .map_or(0, |_| self.scope.observation().source_location_id_retained_bytes().unwrap_or(0));
        42u64
            .saturating_add(u64::try_from(self.scope.plan_execution_id().as_str().len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(self.reservation_id.as_str().len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(self.lineage_id.as_str().len()).unwrap_or(u64::MAX))
            .saturating_add(task_attempt.map_or(0, |attempt| {
                u64::try_from(attempt.task_id.as_str().len()).unwrap_or(u64::MAX).saturating_add(4)
            }))
            .saturating_add(optional_text_bytes(context.task_kind.as_deref()))
            .saturating_add(u64::try_from(context.reservation_kind.as_str().len()).unwrap_or(u64::MAX))
            .saturating_add(memory_space_bytes)
            .saturating_add(u64::try_from(context.memory_boundary.len()).unwrap_or(u64::MAX))
            .saturating_add(optional_text_bytes(context.node_id.as_ref().map(PlanNodeId::as_str)))
            .saturating_add(optional_text_bytes(context.stage.as_deref()))
            .saturating_add(source_location_bytes)
            .saturating_add(
                context
                    .row_groups
                    .as_ref()
                    .map_or(0, |values| u64::try_from(values.len()).unwrap_or(u64::MAX).saturating_mul(8)),
            )
            .saturating_add(optional_text_bytes(context.resident_resource_id.as_deref()))
            .saturating_add(optional_text_bytes(context.source_read_contract_quality.as_deref()))
            .saturating_add(66)
    }
}

#[derive(Debug, Clone)]
struct ReservationDiagnosticContext {
    task_attempt_id: Option<TaskAttemptId>,
    task_kind: Option<String>,
    span_id: Option<RuntimeSpanId>,
    resident_resource_id: Option<String>,
    release_disposition: RuntimeReservationDisposition,
}

impl Default for ReservationDiagnosticContext {
    fn default() -> Self {
        Self {
            task_attempt_id: None,
            task_kind: None,
            span_id: None,
            resident_resource_id: None,
            release_disposition: RuntimeReservationDisposition::ExplicitRelease,
        }
    }
}

/// Restores the previous typed reservation context on every return path and unwind.
#[must_use]
pub(in crate::exec) struct ReservationDiagnosticScope {
    context: Rc<RefCell<ReservationDiagnosticContext>>,
    previous: Option<ReservationDiagnosticContext>,
}

impl Drop for ReservationDiagnosticScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            *self.context.borrow_mut() = previous;
        }
    }
}

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
    task_attempt_id: Option<TaskAttemptId>,
    task_kind: Option<String>,
    resident_resource_id: Option<String>,
    source_read_contract_quality: Option<String>,
    source_read_planning_bytes: Option<u64>,
    source_read_estimated_bytes: Option<u64>,
    alias: bool,
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
            task_attempt_id: request.task_attempt_id.clone(),
            task_kind: request.task_kind.clone(),
            resident_resource_id: request.resident_resource_id.clone(),
            source_read_contract_quality: request.source_read_contract_quality.clone(),
            source_read_planning_bytes: request.source_read_planning_bytes,
            source_read_estimated_bytes: request.source_read_estimated_bytes,
            alias: false,
        }
    }

    fn with_alias(mut self) -> Self {
        self.alias = true;
        self
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
    reservation_projection_ledger: ReservationProjectionLedgerHandle,
    observation: Option<ReservationObservation>,
    semantics: ActiveReservationSemantics,
    lineage: Option<Rc<ActiveReservationLineage>>,
    release_action: MemoryReservationEventAction,
    bytes: u64,
    charged_bytes: u64,
    released: bool,
}

impl ActiveReservation {
    fn owned_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            bytes,
            request,
            reservation_projection_ledger,
            observation,
            ActiveReservationSemantics::OwnedResident,
        )
    }

    fn work(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            bytes,
            request,
            reservation_projection_ledger,
            observation,
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
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
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
            reservation_projection_ledger,
            observation,
            semantics,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::Release,
            bytes,
            charged_bytes: bytes,
            released: false,
        }
    }

    fn borrowed_source_cache(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        source_cache_key: GpuLocalParquetSourceCacheKey,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
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
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger,
            observation,
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    fn borrowed_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        protected_source_cache_keys: Rc<RefCell<HashMap<GpuLocalParquetSourceCacheKey, usize>>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            live_accounting: None,
            protected_source_cache_keys,
            source_cache_key: None,
            source_cache_pin: None,
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger,
            observation,
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
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        lineage.alias_count.set(lineage.alias_count.get().saturating_add(1));
        Self {
            memory_space,
            reserved_bytes,
            live_accounting,
            protected_source_cache_keys,
            source_cache_key: None,
            source_cache_pin,
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger,
            observation,
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
            reservation_projection_ledger: self.reservation_projection_ledger.clone(),
            observation: self.observation.clone(),
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
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger: self.reservation_projection_ledger.clone(),
            observation: self.observation.clone(),
            semantics: ActiveReservationSemantics::Alias,
            lineage,
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes: request.requested_bytes,
            charged_bytes: 0,
            released: false,
        }
    }

    fn release(&mut self, disposition: RuntimeReservationDisposition) -> Option<MemoryReservationEvent> {
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
                        if let Some(observation) = &self.observation {
                            observation.record(
                                &self.event_context,
                                runtime_release_action(disposition),
                                disposition,
                                self.bytes,
                                0,
                                -i128::from(deferred_release),
                                self.reserved_bytes.get(),
                            );
                        }
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
        if let Some(observation) = &self.observation {
            let disposition = if charged_delta_bytes == 0
                && !self.is_alias()
                && self.lineage.as_ref().is_some_and(|lineage| lineage.alias_count.get() > 0)
            {
                RuntimeReservationDisposition::AliasDeferred
            } else {
                disposition
            };
            observation.record(
                &self.event_context,
                runtime_release_action(disposition),
                disposition,
                self.bytes,
                0,
                charged_delta_bytes,
                self.reserved_bytes.get(),
            );
        }
        Some(self.event_context.event(self.release_action, self.bytes, charged_delta_bytes, self.reserved_bytes.get()))
    }

    fn retarget_observation(&mut self, task_attempt_id: Option<TaskAttemptId>, span_id: Option<RuntimeSpanId>) {
        if let Some(observation) = &mut self.observation {
            observation.retarget(task_attempt_id, span_id);
        }
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
        let disposition = if std::thread::panicking() {
            RuntimeReservationDisposition::UnwindRelease
        } else {
            RuntimeReservationDisposition::DropRelease
        };
        if let Some(event) = self.release(disposition) {
            self.reservation_projection_ledger.record_implicit_release(event, self.event_context.task_kind.clone());
        }
    }
}

fn runtime_release_action(disposition: RuntimeReservationDisposition) -> RuntimeReservationAction {
    if disposition == RuntimeReservationDisposition::RollbackRelease {
        RuntimeReservationAction::Rollback
    } else {
        RuntimeReservationAction::Release
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
    source_cache_baseline_observations: VecDeque<SourceCacheBaselineObservation>,
    observation_scope: Option<RuntimeExecutionObservationScope>,
    next_reservation_id: Rc<Cell<u64>>,
    diagnostic_context: Rc<RefCell<ReservationDiagnosticContext>>,
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
        Self::new_with_budget_and_observation(memory_space, budget, None)
    }

    fn new_with_budget_and_observation(
        memory_space: MemorySpaceId,
        budget: ReservationBudget,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Self {
        Self {
            query_id: observation_scope
                .and_then(|scope| scope.observation().query_id())
                .map(|query_id| QueryId::new(query_id.as_str())),
            memory_space,
            budget,
            reserved_bytes: Rc::new(Cell::new(0)),
            local_parquet_source_cache: None,
            protected_source_cache_keys: Rc::new(RefCell::new(HashMap::new())),
            source_cache_baseline_bytes: 0,
            source_cache_baseline_observations: VecDeque::new(),
            observation_scope: observation_scope.cloned(),
            next_reservation_id: Rc::new(Cell::new(1)),
            diagnostic_context: Rc::new(RefCell::new(ReservationDiagnosticContext::default())),
            adaptive_task_refinement: None,
        }
    }

    pub(in crate::exec) fn new_for_context(
        context: &ExecutionContext,
        budget_bytes: u64,
        metrics: &mut QueryMetrics,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let memory_grant = context.memory_grant()?;
        let budget = match (context.native_pipeline_execution_mode(), memory_grant) {
            (NativePipelineExecutionMode::Adaptive, Some(grant)) => ReservationBudget::AdaptiveGrant(grant),
            (_, Some(grant)) => ReservationBudget::GrantBackedStatic { budget_bytes, grant },
            (_, None) => ReservationBudget::Static(budget_bytes),
        };
        let mut tracker =
            Self::new_with_budget_and_observation(context.device_memory_space(), budget, observation_scope);
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
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let mut tracker = Self::new_for_context(context, budget_bytes, metrics, observation_scope)?;
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

    pub(in crate::exec) fn enter_task_observation_scope(
        &self,
        task_attempt_id: TaskAttemptId,
        task_kind: impl Into<String>,
        span_id: RuntimeSpanId,
        resident_resource_id: Option<impl ToString>,
    ) -> ReservationDiagnosticScope {
        self.enter_diagnostic_scope(|context| {
            context.task_attempt_id = Some(task_attempt_id);
            context.task_kind = Some(task_kind.into());
            context.span_id = Some(span_id);
            context.resident_resource_id = resident_resource_id.map(|value| value.to_string());
        })
    }

    pub(in crate::exec) fn enter_resource_observation_scope(
        &self,
        resident_resource_id: impl ToString,
    ) -> ReservationDiagnosticScope {
        self.enter_diagnostic_scope(|context| {
            context.resident_resource_id = Some(resident_resource_id.to_string());
        })
    }

    pub(in crate::exec) fn enter_release_observation_scope(
        &self,
        disposition: RuntimeReservationDisposition,
    ) -> ReservationDiagnosticScope {
        self.enter_diagnostic_scope(|context| context.release_disposition = disposition)
    }

    fn enter_diagnostic_scope(
        &self,
        update: impl FnOnce(&mut ReservationDiagnosticContext),
    ) -> ReservationDiagnosticScope {
        let previous = self.diagnostic_context.borrow().clone();
        update(&mut self.diagnostic_context.borrow_mut());
        ReservationDiagnosticScope { context: Rc::clone(&self.diagnostic_context), previous: Some(previous) }
    }

    fn next_observation_id(&self) -> Option<RuntimeReservationId> {
        self.observation_scope.as_ref()?;
        let next = self.next_reservation_id.get();
        self.next_reservation_id.set(next.saturating_add(1));
        Some(RuntimeReservationId::new(format!("reservation-{next}")))
    }

    fn new_reservation_observation(&self) -> Option<ReservationObservation> {
        let scope = self.observation_scope.clone()?;
        let reservation_id = self.next_observation_id()?;
        let context = self.diagnostic_context.borrow();
        Some(ReservationObservation {
            scope,
            lineage_id: reservation_id.clone(),
            reservation_id,
            task_attempt_id: context.task_attempt_id.clone(),
            span_id: context.span_id,
        })
    }

    fn new_alias_observation(&self, source: &ActiveReservation) -> Option<ReservationObservation> {
        let scope = self.observation_scope.clone()?;
        let reservation_id = self.next_observation_id()?;
        let context = self.diagnostic_context.borrow();
        Some(ReservationObservation {
            scope,
            lineage_id: source
                .observation
                .as_ref()
                .map_or_else(|| reservation_id.clone(), |value| value.lineage_id.clone()),
            reservation_id,
            task_attempt_id: context.task_attempt_id.clone(),
            span_id: context.span_id,
        })
    }

    fn retarget_reservation_observation(&self, reservation: &mut ActiveReservation) {
        let context = self.diagnostic_context.borrow();
        reservation.retarget_observation(context.task_attempt_id.clone(), context.span_id);
    }

    fn record_denied_observation(
        &self,
        request: &MemoryReservationRequest,
        denied_action: RuntimeReservationAction,
        granted_bytes: u64,
    ) {
        let Some(observation) = self.new_reservation_observation() else {
            return;
        };
        observation.record_denied(
            &ActiveReservationEventContext::from_request(request),
            denied_action,
            request.requested_bytes,
            granted_bytes,
            self.raw_reserved_bytes(),
        );
    }

    fn push_source_cache_baseline_observation(
        &mut self,
        bytes: u64,
        source: Option<&ActiveReservation>,
        request: &MemoryReservationRequest,
        metrics: &QueryMetrics,
    ) -> Option<ReservationObservation> {
        let observation = match source {
            Some(source) => self.new_alias_observation(source),
            None => self.new_reservation_observation(),
        };
        self.source_cache_baseline_observations.push_back(SourceCacheBaselineObservation {
            observation: observation.clone(),
            event_context: ActiveReservationEventContext::from_request(request),
            reservation_projection_ledger: metrics.reservation_projection_ledger_handle(),
            remaining_bytes: bytes,
        });
        observation
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
            return Err(request.denied_error(remaining));
        }
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Reserve,
            request.requested_bytes,
            i128::from(request.requested_bytes),
            self.raw_reserved_bytes(),
        );
        let observation = self.new_reservation_observation();
        let reservation_projection_ledger = metrics.reservation_projection_ledger_handle();
        let reservation = match semantics {
            ActiveReservationSemantics::OwnedResident => ActiveReservation::owned_resident(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
                self.budget.live_accounting(),
                Rc::clone(&self.protected_source_cache_keys),
                request.requested_bytes,
                &request,
                reservation_projection_ledger,
                observation.clone(),
            ),
            ActiveReservationSemantics::Work => ActiveReservation::work(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
                self.budget.live_accounting(),
                Rc::clone(&self.protected_source_cache_keys),
                request.requested_bytes,
                &request,
                reservation_projection_ledger,
                observation,
            ),
            ActiveReservationSemantics::Alias => unreachable!("alias reservations do not charge capacity"),
        };
        if let Some(observation) = &reservation.observation {
            observation.record(
                &reservation.event_context,
                RuntimeReservationAction::Reserve,
                RuntimeReservationDisposition::Granted,
                request.requested_bytes,
                request.requested_bytes,
                i128::from(request.requested_bytes),
                self.raw_reserved_bytes(),
            );
        }
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
        let observation = self.new_alias_observation(source);
        let alias = ActiveReservation::alias(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            source.live_accounting.clone(),
            Rc::clone(&self.protected_source_cache_keys),
            source.source_cache_pin.clone(),
            Rc::clone(source.lineage.as_ref().expect("generic alias requires a reservation lineage")),
            request.requested_bytes,
            &request,
            metrics.reservation_projection_ledger_handle(),
            observation,
        );
        if let Some(observation) = &alias.observation {
            observation.record(
                &alias.event_context,
                RuntimeReservationAction::Alias,
                RuntimeReservationDisposition::Applied,
                request.requested_bytes,
                0,
                0,
                self.raw_reserved_bytes(),
            );
        }
        alias
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
        let mut transferred = reservation.transferred(&request);
        self.retarget_reservation_observation(&mut transferred);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::Transfer,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        if let Some(observation) = &transferred.observation {
            observation.record(
                &transferred.event_context,
                RuntimeReservationAction::Transfer,
                RuntimeReservationDisposition::Applied,
                request.requested_bytes,
                0,
                0,
                self.raw_reserved_bytes(),
            );
        }
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
        let mut relabeled = reservation.relabeled_alias(&request);
        self.retarget_reservation_observation(&mut relabeled);
        metrics.record_reservation_timeline_event(MemoryReservationEvent::from_request(
            &request,
            MemoryReservationEventOutcome::Applied,
            MemoryReservationEventAction::AliasRelabel,
            0,
            0,
            self.raw_reserved_bytes(),
        ));
        if let Some(observation) = &relabeled.observation {
            observation.record(
                &relabeled.event_context,
                RuntimeReservationAction::Relabel,
                RuntimeReservationDisposition::Applied,
                request.requested_bytes,
                0,
                0,
                self.raw_reserved_bytes(),
            );
        }
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
        let observation = self.new_reservation_observation();
        let borrowed = ActiveReservation::borrowed_source_cache(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            Rc::clone(&self.protected_source_cache_keys),
            key,
            source_cache_pin,
            bytes,
            &request,
            metrics.reservation_projection_ledger_handle(),
            observation,
        );
        if let Some(observation) = &borrowed.observation {
            observation.record(
                &borrowed.event_context,
                RuntimeReservationAction::Alias,
                RuntimeReservationDisposition::Applied,
                bytes,
                0,
                0,
                self.raw_reserved_bytes(),
            );
        }
        borrowed
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
        let observation = self.new_reservation_observation();
        let borrowed = ActiveReservation::borrowed_resident(
            self.memory_space,
            Rc::clone(&self.reserved_bytes),
            Rc::clone(&self.protected_source_cache_keys),
            request.requested_bytes,
            &request,
            metrics.reservation_projection_ledger_handle(),
            observation,
        );
        if let Some(observation) = &borrowed.observation {
            observation.record(
                &borrowed.event_context,
                RuntimeReservationAction::Alias,
                RuntimeReservationDisposition::Applied,
                request.requested_bytes,
                0,
                0,
                self.raw_reserved_bytes(),
            );
        }
        borrowed
    }

    pub(in crate::exec) fn release(&mut self, mut reservation: ActiveReservation, metrics: &mut QueryMetrics) {
        self.assert_reservation_belongs_to_tracker(&reservation, "release");
        self.retarget_reservation_observation(&mut reservation);
        let disposition = self.diagnostic_context.borrow().release_disposition;
        if let Some(event) = reservation.release(disposition) {
            // Release and alias-release events describe live ledger movement only; aggregate
            // requested/granted counters intentionally remain historical admission counters.
            let mut event = event;
            let context = self.diagnostic_context.borrow();
            if event.compiled_task_id.is_none() {
                event.compiled_task_id =
                    context.task_attempt_id.as_ref().map(|attempt| CompiledTaskId::new(attempt.task_id.as_str()));
            }
            if event.query_id.is_none() {
                event.query_id = self.query_id.clone();
            }
            if event.resident_resource_id.is_none() {
                event.resident_resource_id = context.resident_resource_id.clone();
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
            self.record_denied_observation(&request, RuntimeReservationAction::Grow, remaining);
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
            self.record_denied_observation(&request, RuntimeReservationAction::Grow, remaining);
            return Err(request.denied_error(remaining));
        }
        self.retarget_reservation_observation(reservation);
        let charged_delta = reservation.grow(request.requested_bytes);
        metrics.record_reservation_granted_timeline_event(
            &request,
            MemoryReservationEventAction::Grow,
            request.requested_bytes,
            i128::from(charged_delta),
            self.raw_reserved_bytes(),
        );
        if let Some(observation) = &reservation.observation {
            observation.record(
                &ActiveReservationEventContext::from_request(&request),
                RuntimeReservationAction::Grow,
                RuntimeReservationDisposition::Granted,
                request.requested_bytes,
                request.requested_bytes,
                i128::from(charged_delta),
                self.raw_reserved_bytes(),
            );
        }
        Ok(())
    }

    pub(in crate::exec) fn shrink_to_observed(
        &mut self,
        reservation: &mut ActiveReservation,
        observed_bytes: u64,
        metrics: &mut QueryMetrics,
    ) {
        self.assert_reservation_belongs_to_tracker(reservation, "shrink_to_observed");
        self.retarget_reservation_observation(reservation);
        let refunded = reservation.shrink_to(observed_bytes);
        if refunded > 0 {
            metrics.record_reservation_timeline_event(reservation.event_context.event(
                MemoryReservationEventAction::Shrink,
                observed_bytes,
                -i128::from(refunded),
                self.raw_reserved_bytes(),
            ));
            if let Some(observation) = &reservation.observation {
                observation.record(
                    &reservation.event_context,
                    RuntimeReservationAction::Shrink,
                    RuntimeReservationDisposition::Applied,
                    observed_bytes,
                    0,
                    -i128::from(refunded),
                    self.raw_reserved_bytes(),
                );
            }
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
        self.retarget_reservation_observation(reservation);
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
            if let Some(observation) = &reservation.observation {
                observation.record(
                    &ActiveReservationEventContext::from_request(&request),
                    RuntimeReservationAction::Shrink,
                    RuntimeReservationDisposition::Applied,
                    request.requested_bytes,
                    0,
                    -i128::from(refunded),
                    self.raw_reserved_bytes(),
                );
            }
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
            if let Some(observation) = self.new_alias_observation(reservation) {
                observation.record(
                    &ActiveReservationEventContext::from_request(&request),
                    RuntimeReservationAction::SourceCachePromote,
                    RuntimeReservationDisposition::Applied,
                    retained_bytes,
                    retained_bytes,
                    0,
                    self.raw_reserved_bytes(),
                );
            }
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
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
        if let Some(mut observation) =
            self.push_source_cache_baseline_observation(retained_bytes, Some(reservation), &request, metrics)
        {
            let context = self.diagnostic_context.borrow();
            observation.retarget(context.task_attempt_id.clone(), context.span_id);
            observation.record(
                &ActiveReservationEventContext::from_request(&request),
                RuntimeReservationAction::SourceCachePromote,
                RuntimeReservationDisposition::Applied,
                retained_bytes,
                retained_bytes,
                i128::from(additional_bytes),
                self.raw_reserved_bytes(),
            );
        }
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
        let disposition = self.diagnostic_context.borrow().release_disposition;
        self.record_source_cache_baseline_release(
            &request,
            released,
            RuntimeReservationAction::SourceCacheEvict,
            disposition,
            Some(metrics),
        );
    }

    fn record_source_cache_baseline_release(
        &mut self,
        request: &MemoryReservationRequest,
        released: u64,
        action: RuntimeReservationAction,
        terminal_disposition: RuntimeReservationDisposition,
        mut metrics: Option<&mut QueryMetrics>,
    ) {
        let diagnostic = self.diagnostic_context.borrow().clone();
        let mut remaining_release = released;
        let mut current_reserved_bytes = self.raw_reserved_bytes().saturating_add(released);
        while remaining_release > 0 {
            let Some(segment) = self.source_cache_baseline_observations.front_mut() else {
                break;
            };
            let segment_release = remaining_release.min(segment.remaining_bytes);
            segment.remaining_bytes = segment.remaining_bytes.saturating_sub(segment_release);
            remaining_release = remaining_release.saturating_sub(segment_release);
            current_reserved_bytes = current_reserved_bytes.saturating_sub(segment_release);
            let terminal = segment.remaining_bytes == 0;
            let disposition = if terminal { terminal_disposition } else { RuntimeReservationDisposition::Applied };
            if let Some(observation) = &segment.observation {
                let mut observation = observation.clone();
                observation.retarget(diagnostic.task_attempt_id.clone(), diagnostic.span_id);
                observation.record(
                    &ActiveReservationEventContext::from_request(request),
                    action,
                    disposition,
                    segment_release,
                    0,
                    -i128::from(segment_release),
                    current_reserved_bytes,
                );
            }
            let mut public_request = request.clone();
            public_request.task_attempt_id.clone_from(&segment.event_context.task_attempt_id);
            public_request.task_kind.clone_from(&segment.event_context.task_kind);
            public_request.compiled_task_id.clone_from(&segment.event_context.compiled_task_id);
            let event = MemoryReservationEvent::from_request(
                &public_request,
                MemoryReservationEventOutcome::Applied,
                MemoryReservationEventAction::Release,
                0,
                -i128::from(segment_release),
                current_reserved_bytes,
            );
            if let Some(metrics) = metrics.as_deref_mut() {
                if let (Some(task_attempt_id), Some(task_kind)) =
                    (&public_request.task_attempt_id, &public_request.task_kind)
                {
                    metrics.register_task_definition(task_attempt_id.task_id.as_str(), task_kind);
                }
                metrics.record_reservation_timeline_event(event);
            } else {
                segment
                    .reservation_projection_ledger
                    .record_implicit_release(event, segment.event_context.task_kind.clone());
            }
            if terminal {
                self.source_cache_baseline_observations.pop_front();
            }
        }
        debug_assert_eq!(remaining_release, 0);
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
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
            self.record_denied_observation(&request, RuntimeReservationAction::Reserve, remaining);
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
        if let Some(observation) =
            self.push_source_cache_baseline_observation(request.requested_bytes, None, &request, metrics)
        {
            observation.record(
                &ActiveReservationEventContext::from_request(&request),
                RuntimeReservationAction::Reserve,
                RuntimeReservationDisposition::Granted,
                request.requested_bytes,
                request.requested_bytes,
                i128::from(request.requested_bytes),
                self.raw_reserved_bytes(),
            );
        }
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

    fn stamp_request(
        &self,
        mut request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> MemoryReservationRequest {
        request.memory_space = self.memory_space;
        request.budget_bytes = self.budget();
        if request.query_id.is_none() {
            request.query_id =
                self.query_id.clone().or_else(|| metrics.query_id().is_assigned().then(|| metrics.query_id().clone()));
        }
        let context = self.diagnostic_context.borrow();
        if request.task_attempt_id.is_none() {
            request.task_attempt_id.clone_from(&context.task_attempt_id);
        }
        if request.task_kind.is_none() {
            request.task_kind.clone_from(&context.task_kind);
        }
        if request.compiled_task_id.is_none() {
            request.compiled_task_id =
                context.task_attempt_id.as_ref().map(|attempt| CompiledTaskId::new(attempt.task_id.as_str()));
        }
        if request.resident_resource_id.is_none() {
            request.resident_resource_id = context.resident_resource_id.clone();
        }
        drop(context);
        if let (Some(task_attempt_id), Some(task_kind)) = (&request.task_attempt_id, &request.task_kind) {
            metrics.register_task_definition(task_attempt_id.task_id.as_str(), task_kind);
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
            let released = self.source_cache_baseline_bytes;
            self.release_live_bytes(released);
            self.source_cache_baseline_bytes = 0;
            let mut request =
                MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, released, self.budget())
                    .with_memory_boundary("source_cache_retained")
                    .with_stage("tracker_drop");
            request.memory_space = self.memory_space;
            request.query_id = self.query_id.clone();
            self.record_source_cache_baseline_release(
                &request,
                released,
                runtime_release_action(if std::thread::panicking() {
                    RuntimeReservationDisposition::UnwindRelease
                } else {
                    RuntimeReservationDisposition::DropRelease
                }),
                if std::thread::panicking() {
                    RuntimeReservationDisposition::UnwindRelease
                } else {
                    RuntimeReservationDisposition::DropRelease
                },
                None,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        observability::{
            AdmissionRequestId, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationLossReason,
            RuntimeObservationPayload, RuntimeObservationSession, RuntimeQueryLifecycle, RuntimeReservationAction,
            RuntimeReservationDisposition, RuntimeTerminalOutcome, write_runtime_observation_jsonl,
        },
        runtime::{GrantRequest, MemoryGovernor},
        source::LocalParquetSource,
    };

    fn test_request(
        kind: ReservationKind,
        boundary: &'static str,
        bytes: u64,
        budget: u64,
    ) -> MemoryReservationRequest {
        MemoryReservationRequest::new(kind, bytes, budget).with_memory_boundary(boundary).with_stage("test")
    }

    fn observed_projection_with_limits(
        config: RuntimeObservationConfig,
    ) -> Result<(QueryMetrics, crate::observability::SealedRuntimeObservationSnapshot)> {
        let lifecycle = RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new("reservation-cap-parity", None, config),
            PlanExecutionId::new("plan-reservation-cap-parity"),
        );
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(100),
            Some(&scope),
        );
        let request = |kind, boundary, bytes| {
            test_request(kind, boundary, bytes, 100)
                .with_node_id("cap-node")
                .with_task_registration("cap-task", 1, "unary")
        };
        let mut owned =
            reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "reserve", 60), &mut metrics)?;
        reservations.try_grow(&mut owned, request(ReservationKind::OperatorOutput, "grow", 10), &mut metrics)?;
        reservations.shrink_to_observed(&mut owned, 50, &mut metrics);
        let denied =
            reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "denied", 60), &mut metrics);
        assert!(denied.is_err());
        drop(owned);
        drop(reservations);
        metrics.finalize_reservation_projection();
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        Ok((metrics, snapshot))
    }

    fn assert_exact_reservation_projection(actual: &QueryMetrics, expected: &QueryMetrics) {
        assert_eq!(actual.reservation_requested_bytes, expected.reservation_requested_bytes);
        assert_eq!(actual.reservation_granted_bytes, expected.reservation_granted_bytes);
        assert_eq!(actual.reservation_live_peak_bytes, expected.reservation_live_peak_bytes);
        assert_eq!(actual.reservation_denied_count, expected.reservation_denied_count);
        assert_eq!(actual.max_task_reservation_bytes, expected.max_task_reservation_bytes);
        assert_eq!(actual.reservation_events, expected.reservation_events);
        assert_eq!(actual.task_runtime_metrics, expected.task_runtime_metrics);
        assert_eq!(actual.task_operator_metrics, expected.task_operator_metrics);
    }

    #[test]
    fn public_reservation_projection_is_exact_across_trace_record_and_byte_caps() -> Result<()> {
        let (uncapped, complete_trace) =
            observed_projection_with_limits(RuntimeObservationConfig::bounded(64, 64 * 1024))?;
        assert_eq!(uncapped.reservation_requested_bytes, 130);
        assert_eq!(uncapped.reservation_granted_bytes, 70);
        assert_eq!(uncapped.reservation_live_peak_bytes, 70);
        assert_eq!(uncapped.reservation_denied_count, 1);
        assert_eq!(uncapped.reservation_events.len(), 5);
        let task = &uncapped.task_runtime_metrics["cap-task"];
        assert_eq!(task.reservation_requested_bytes, 130);
        assert_eq!(task.reservation_granted_bytes, 70);
        assert_eq!(task.reservation_released_bytes, 70);
        assert_eq!(task.reservation_live_peak_bytes, 70);
        assert!(!complete_trace.footer().truncated);
        assert!(!complete_trace.footer().degraded, "exact reservation preflight must match retained accounting");

        for config in [
            RuntimeObservationConfig::bounded(0, 64 * 1024),
            RuntimeObservationConfig::bounded(1, 64 * 1024),
            RuntimeObservationConfig::bounded(64, 1),
            RuntimeObservationConfig::bounded(64, 512),
        ] {
            let (capped, trace) = observed_projection_with_limits(config)?;
            assert_exact_reservation_projection(&capped, &uncapped);
            assert!(trace.footer().truncated);
            assert!(trace.events().len() < complete_trace.events().len());
            assert!(trace.footer().first_dropped_sequence.is_some());
        }
        let (_, record_limited) = observed_projection_with_limits(RuntimeObservationConfig::bounded(0, 64 * 1024))?;
        assert!(
            record_limited
                .footer()
                .dropped_by_reason
                .iter()
                .any(|loss| loss.reason == RuntimeObservationLossReason::RecordLimit)
        );
        let (_, oversized) = observed_projection_with_limits(RuntimeObservationConfig::bounded(64, 1))?;
        assert!(
            oversized
                .footer()
                .dropped_by_reason
                .iter()
                .any(|loss| loss.reason == RuntimeObservationLossReason::Oversized)
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum TestReleaseMode {
        Explicit,
        Drop,
        Unwind,
    }

    fn projection_for_release_mode(mode: TestReleaseMode) -> Result<QueryMetrics> {
        let mut metrics = QueryMetrics::new("release-projection-parity");
        metrics.record_task_execution("release-task", "unary");
        metrics.record_task_elapsed("release-task", "unary", std::time::Duration::from_millis(7));
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let active = reservations.try_reserve_work(
            test_request(ReservationKind::OperatorOutput, "release-parity", 60, 100).with_task_registration(
                "release-task",
                1,
                "unary",
            ),
            &mut metrics,
        )?;
        match mode {
            TestReleaseMode::Explicit => reservations.release(active, &mut metrics),
            TestReleaseMode::Drop => drop(active),
            TestReleaseMode::Unwind => {
                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _active = active;
                    panic!("exercise reservation unwind cleanup");
                }));
                assert!(unwind.is_err());
            },
        }
        assert_eq!(reservations.reserved_bytes(), 0);
        drop(reservations);
        metrics.finalize_reservation_projection();
        Ok(metrics)
    }

    #[test]
    fn disabled_observation_drop_and_unwind_keep_the_exact_public_projection() -> Result<()> {
        let explicit = projection_for_release_mode(TestReleaseMode::Explicit)?;
        let dropped = projection_for_release_mode(TestReleaseMode::Drop)?;
        let unwound = projection_for_release_mode(TestReleaseMode::Unwind)?;

        assert_exact_reservation_projection(&dropped, &explicit);
        assert_exact_reservation_projection(&unwound, &explicit);
        assert_eq!(explicit.reservation_events.len(), 2);
        assert_eq!(explicit.reservation_events.iter().map(|event| event.charged_delta_bytes).sum::<i128>(), 0);
        let task = &explicit.task_runtime_metrics["release-task"];
        assert_eq!(task.elapsed_time, std::time::Duration::from_millis(7));
        assert_eq!(task.reservation_requested_bytes, 60);
        assert_eq!(task.reservation_granted_bytes, 60);
        assert_eq!(task.reservation_released_bytes, 60);
        assert_eq!(task.live_reserved_bytes, 0);
        assert_eq!(explicit.task_operator_metrics["unary"].task_count, 1);
        Ok(())
    }

    #[test]
    fn cloning_query_metrics_detaches_the_reservation_projection_ledger() -> Result<()> {
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new(MemorySpaceId::Device { device_ordinal: 0 }, 100);
        let active = reservations
            .try_reserve_work(test_request(ReservationKind::OperatorOutput, "clone-ledger", 60, 100), &mut metrics)?;
        let mut cloned = metrics.clone();

        drop(active);
        metrics.finalize_reservation_projection();
        cloned.finalize_reservation_projection();

        assert_eq!(metrics.reservation_events.len(), 2);
        assert_eq!(metrics.reservation_events.iter().map(|event| event.charged_delta_bytes).sum::<i128>(), 0);
        assert_eq!(cloned.reservation_events.len(), 1);
        assert_eq!(cloned.reservation_events[0].charged_delta_bytes, 60);
        Ok(())
    }

    #[test]
    fn adaptive_reservation_tracker_observes_live_grant_growth() -> Result<()> {
        let governor = MemoryGovernor::new(0, 100)?;
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 40))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(8), 40, 40))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 10, 40))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 90))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 70))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 70))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?
            .expect("initial grant should fit");
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
        let grant = governor
            .try_request(GrantRequest::new(AdmissionRequestId::new(7), 30, 80))?
            .expect("initial grant should fit");
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

    #[test]
    fn typed_reservation_lineage_balances_transfer_alias_and_deferred_release() -> Result<()> {
        let lifecycle = RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new(
                "reservation-lineage",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
            ),
            PlanExecutionId::new("plan-reservation-lineage"),
        );
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(100),
            Some(&scope),
        );
        let request = |kind, boundary, bytes| {
            test_request(kind, boundary, bytes, 100).with_node_id("lineage-node").with_task_registration(
                "lineage-task",
                1,
                "unary",
            )
        };
        let mut owned =
            reservations.try_reserve_work(request(ReservationKind::OperatorOutput, "work", 60), &mut metrics)?;
        reservations.try_grow(&mut owned, request(ReservationKind::OperatorOutput, "work_grow", 10), &mut metrics)?;
        reservations.shrink_to_observed(&mut owned, 50, &mut metrics);
        let transferred = reservations.transfer(owned, request(ReservationKind::ProbeSide, "probe", 50), &mut metrics);
        let alias = reservations.alias_reservation(
            &transferred,
            request(ReservationKind::OperatorOutput, "alias", 50),
            &mut metrics,
        );
        let alias = reservations.relabel_alias_reservation(
            alias,
            request(ReservationKind::ProbeSide, "relabeled_alias", 50),
            &mut metrics,
        );
        reservations.release(transferred, &mut metrics);
        assert_eq!(reservations.reserved_bytes(), 50, "owner charge remains live through its alias");
        reservations.release(alias, &mut metrics);
        assert_eq!(reservations.reserved_bytes(), 0);
        drop(reservations);

        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        let events = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Reservation(observation) => Some((
                    &observation.reservation_id,
                    &observation.lineage_id,
                    observation.action,
                    observation.disposition,
                    observation.charged_delta_bytes,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(events.iter().map(|event| event.4).sum::<i128>(), 0);
        assert!(events.iter().any(|event| {
            event.2 == RuntimeReservationAction::Transfer && event.3 == RuntimeReservationDisposition::Applied
        }));
        assert!(events.iter().any(|event| {
            event.2 == RuntimeReservationAction::Relabel && event.3 == RuntimeReservationDisposition::Applied
        }));
        assert!(events.iter().any(|event| {
            event.2 == RuntimeReservationAction::Release
                && event.3 == RuntimeReservationDisposition::AliasDeferred
                && event.4 == 0
        }));
        let alias = events
            .iter()
            .find(|event| event.2 == RuntimeReservationAction::Alias)
            .expect("alias acquisition is recorded");
        let root =
            events.iter().find(|event| event.2 == RuntimeReservationAction::Reserve).expect("root reserve is recorded");
        assert_ne!(alias.0, root.0);
        assert_eq!(alias.1, root.1);

        assert_eq!(metrics.reservation_requested_bytes, 70);
        assert_eq!(metrics.reservation_granted_bytes, 70);
        assert_eq!(metrics.reservation_denied_count, 0);
        assert_eq!(metrics.reservation_events.last().map(|event| event.current_reserved_bytes), Some(0));
        assert!(metrics.reservation_events.iter().all(|event| {
            event.query_id.as_ref().is_some_and(|query_id| query_id.as_str() == "reservation-lineage")
                && event.node_id.as_ref().is_some_and(|node_id| node_id.as_str() == "lineage-node")
                && event.compiled_task_id.as_ref().is_some_and(|task_id| task_id.as_str() == "lineage-task")
        }));
        let task = metrics.task_runtime_metrics.get("lineage-task").expect("typed task projection exists");
        assert_eq!(task.operator_kind, "unary");
        assert_eq!(task.reservation_requested_bytes, 70);
        assert_eq!(task.reservation_granted_bytes, 70);
        assert_eq!(task.reservation_released_bytes, 70);
        assert_eq!(task.reservation_live_peak_bytes, 70);
        Ok(())
    }

    #[test]
    fn source_cache_promotion_and_eviction_keep_typed_meaning_and_legacy_public_totals() -> Result<()> {
        let lifecycle = RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new(
                "source-cache-promotion",
                None,
                RuntimeObservationConfig::bounded(32, 64 * 1024),
            ),
            PlanExecutionId::new("plan-source-cache-promotion"),
        );
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(100),
            Some(&scope),
        );
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        let source = LocalParquetSource::new("/tmp/source-cache-promotion.parquet", schema, vec!["c0".to_string()]);
        let key = GpuLocalParquetSourceCacheKey::from_source(&source);
        let request = |kind, boundary, bytes| {
            test_request(kind, boundary, bytes, 100).with_node_id("source-cache-node").with_task_registration(
                "source-cache-task",
                1,
                "source_read",
            )
        };
        let mut source_reservation = reservations
            .try_reserve_owned_resident(request(ReservationKind::SourceRead, "source_read", 40), &mut metrics)?;
        reservations.promote_source_cache_insert(
            &mut source_reservation,
            60,
            request(ReservationKind::SourceCacheRetained, "source_cache_retained", 60),
            &mut metrics,
            &key,
        )?;
        reservations.release(source_reservation, &mut metrics);
        reservations.release_source_cache_baseline_bytes(60, &mut metrics);
        assert_eq!(reservations.raw_reserved_bytes(), 0);
        drop(reservations);

        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        let typed = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Reservation(observation) => {
                    Some((observation.action, observation.charged_delta_bytes))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(typed.iter().any(|event| event.0 == RuntimeReservationAction::SourceCachePromote && event.1 == 20));
        assert!(typed.iter().any(|event| event.0 == RuntimeReservationAction::SourceCacheEvict && event.1 == -60));
        assert!(!typed.iter().any(|event| event.0 == RuntimeReservationAction::Transfer));
        assert_eq!(typed.iter().map(|event| event.1).sum::<i128>(), 0);

        let promotion = metrics
            .reservation_events
            .iter()
            .find(|event| event.memory_boundary == "source_cache_retained" && event.charged_delta_bytes > 0)
            .expect("legacy promotion row is retained");
        assert_eq!(promotion.action, MemoryReservationEventAction::Reserve);
        assert_eq!(promotion.requested_bytes, 60);
        assert_eq!(promotion.granted_bytes, 60);
        let eviction = metrics
            .reservation_events
            .iter()
            .find(|event| event.stage.as_deref() == Some("source_cache_eviction"))
            .expect("legacy eviction row is retained");
        assert_eq!(eviction.action, MemoryReservationEventAction::Release);
        assert_eq!(eviction.charged_delta_bytes, -60);
        assert_eq!(eviction.current_reserved_bytes, 0);
        assert_eq!(metrics.reservation_requested_bytes, 100);
        assert_eq!(metrics.reservation_granted_bytes, 100);
        let task = &metrics.task_runtime_metrics["source-cache-task"];
        assert_eq!(task.reservation_requested_bytes, 100);
        assert_eq!(task.reservation_granted_bytes, 60);
        assert_eq!(task.reservation_released_bytes, 60);
        assert_eq!(task.live_reserved_bytes, 0);
        Ok(())
    }

    #[test]
    fn typed_reservations_replace_source_paths_with_stable_opaque_ids() -> Result<()> {
        let sensitive_locations = [
            "/srv/private/customer-42/orders.parquet",
            "s3://alice:secret@bucket/private/key.parquet?token=top-secret",
            "catalog/customer/partition=private",
        ];
        let source_location_key = [0x5a; blake3::KEY_LEN];
        let session = RuntimeObservationSession::with_source_location_test_key(
            "source-location-privacy",
            None,
            RuntimeObservationConfig::bounded(32, 64 * 1024),
            source_location_key,
        );
        let expected_ids = sensitive_locations
            .iter()
            .map(|location| session.handle().source_location_id(location).expect("enabled session has a digest key"))
            .collect::<Vec<_>>();
        let lifecycle = RuntimeQueryLifecycle::new(session, PlanExecutionId::new("plan-source-location-privacy"));
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(16),
            Some(&scope),
        );
        for (index, location) in sensitive_locations.iter().enumerate() {
            let reservation = reservations.try_reserve_owned_resident(
                test_request(ReservationKind::SourceRead, "private_source", 1, reservations.budget())
                    .with_path(location)
                    .with_file_index(index)
                    .with_row_groups(vec![index, index + 2]),
                &mut metrics,
            )?;
            reservations.release(reservation, &mut metrics);
        }
        drop(reservations);
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        assert!(!snapshot.footer().degraded, "Parquet row-group preflight must use the exact retained size");
        let ids = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Reservation(observation) => observation.source_location_id.clone(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ids.len(), sensitive_locations.len() * 2, "reserve and release retain opaque correlation");
        for (index, expected) in expected_ids.iter().enumerate() {
            assert_eq!(&ids[index * 2], expected);
            assert_eq!(&ids[index * 2 + 1], expected);
            assert!(expected.as_str().starts_with("source_location:blake3_keyed_256:"));
            assert!(!expected.as_str().contains(":len:"));
        }
        assert_ne!(ids[0], ids[2]);
        assert_ne!(ids[2], ids[4]);

        let mut jsonl = Vec::new();
        write_runtime_observation_jsonl(&snapshot, &mut jsonl).expect("privacy snapshot exports");
        let jsonl = String::from_utf8(jsonl).expect("JSONL is UTF-8");
        for location in sensitive_locations {
            assert!(!jsonl.contains(location), "raw source location leaked: {location}");
        }
        assert!(!jsonl.contains("alice:secret"));
        assert!(!jsonl.contains("top-secret"));
        assert!(!jsonl.contains(&"5a".repeat(blake3::KEY_LEN)));
        assert!(jsonl.contains("source_location_id"));
        assert!(
            metrics
                .reservation_events
                .iter()
                .any(|event| { event.path.as_deref() == Some("/srv/private/customer-42/orders.parquet") }),
            "required public reservation diagnostics remain unchanged"
        );
        Ok(())
    }

    #[test]
    fn typed_reservation_drop_unwind_and_denial_are_distinct_and_balanced() -> Result<()> {
        let lifecycle = RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new(
                "reservation-terminal",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
            ),
            PlanExecutionId::new("plan-reservation-terminal"),
        );
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(100),
            Some(&scope),
        );

        let dropped = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "drop", 20, reservations.budget()),
            &mut metrics,
        )?;
        drop(dropped);
        assert_eq!(reservations.reserved_bytes(), 0);

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _unwinding = reservations
                .try_reserve_owned_resident(
                    test_request(ReservationKind::OperatorOutput, "unwind", 30, reservations.budget()),
                    &mut metrics,
                )
                .expect("unwind reservation fits");
            panic!("reservation unwind probe");
        }));
        assert!(unwind.is_err());
        assert_eq!(reservations.reserved_bytes(), 0);

        let live = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "live", 80, reservations.budget()),
            &mut metrics,
        )?;
        let denial = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::ProbeSide, "denied", 30, reservations.budget()),
            &mut metrics,
        );
        assert!(denial.is_err());
        reservations.release(live, &mut metrics);
        drop(reservations);

        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        let events = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Reservation(observation) => {
                    Some((observation.action, observation.disposition, observation.charged_delta_bytes))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(events.iter().map(|event| event.2).sum::<i128>(), 0);
        assert!(events.iter().any(|event| event.1 == RuntimeReservationDisposition::DropRelease));
        assert!(events.iter().any(|event| event.1 == RuntimeReservationDisposition::UnwindRelease));
        assert!(events.iter().any(|event| {
            event.0 == RuntimeReservationAction::Deny && event.1 == RuntimeReservationDisposition::Denied
        }));
        Ok(())
    }

    #[test]
    fn typed_reservation_rollback_and_cleanup_releases_are_balanced() -> Result<()> {
        let lifecycle = RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new(
                "reservation-cleanup",
                None,
                RuntimeObservationConfig::bounded(32, 64 * 1024),
            ),
            PlanExecutionId::new("plan-reservation-cleanup"),
        );
        let scope = lifecycle.execution_scope();
        let mut metrics = QueryMetrics::default();
        let mut reservations = ReservationTracker::new_with_budget_and_observation(
            MemorySpaceId::Device { device_ordinal: 0 },
            ReservationBudget::Static(100),
            Some(&scope),
        );
        let rollback = reservations.try_reserve_work(
            test_request(ReservationKind::GatherMap, "rollback", 30, reservations.budget()),
            &mut metrics,
        )?;
        {
            let _rollback_scope =
                reservations.enter_release_observation_scope(RuntimeReservationDisposition::RollbackRelease);
            reservations.release(rollback, &mut metrics);
        }
        let cleanup = reservations.try_reserve_owned_resident(
            test_request(ReservationKind::OperatorOutput, "cleanup", 40, reservations.budget()),
            &mut metrics,
        )?;
        {
            let _cleanup_scope =
                reservations.enter_release_observation_scope(RuntimeReservationDisposition::CleanupRelease);
            reservations.release(cleanup, &mut metrics);
        }
        reservations.try_reserve_source_cache_baseline(20, &mut metrics)?;
        reservations.release_source_cache_baseline_bytes(20, &mut metrics);
        drop(reservations);

        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Succeeded).expect("observation seals");
        let events = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Reservation(observation) => {
                    Some((observation.action, observation.disposition, observation.charged_delta_bytes))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(events.iter().map(|event| event.2).sum::<i128>(), 0);
        assert!(events.iter().any(|event| {
            event.0 == RuntimeReservationAction::Rollback && event.1 == RuntimeReservationDisposition::RollbackRelease
        }));
        assert!(events.iter().any(|event| {
            event.0 == RuntimeReservationAction::Release && event.1 == RuntimeReservationDisposition::CleanupRelease
        }));
        Ok(())
    }
}
