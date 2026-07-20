use super::*;

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
pub(super) enum ReservationBudget {
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

    pub(super) fn new_with_budget(memory_space: MemorySpaceId, budget: ReservationBudget) -> Self {
        Self::new_with_budget_and_observation(memory_space, budget, None)
    }

    pub(super) fn new_with_budget_and_observation(
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

    pub(super) fn try_reserve_source_cache_baseline(
        &mut self,
        retained_bytes: u64,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
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

    pub(super) fn raw_reserved_bytes(&self) -> u64 {
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
