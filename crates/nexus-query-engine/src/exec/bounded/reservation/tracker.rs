use super::*;

pub(in crate::exec) struct ReservationTracker {
    query_id: Option<QueryId>,
    memory_space: MemorySpaceId,
    budget: ReservationBudget,
    reserved_bytes: Rc<Cell<u64>>,
    observation_scope: Option<RuntimeExecutionObservationScope>,
    next_reservation_id: Rc<Cell<u64>>,
    diagnostic_context: Rc<RefCell<ReservationDiagnosticContext>>,
}

#[derive(Debug, Clone)]
pub(super) enum ReservationBudget {
    Static(u64),
    GrantBackedStatic { budget_bytes: u64, grant: MemoryGrant },
}

impl ReservationBudget {
    fn bytes(&self) -> u64 {
        match self {
            Self::Static(bytes) => *bytes,
            Self::GrantBackedStatic { budget_bytes, .. } => *budget_bytes,
        }
    }

    fn live_accounting(&self) -> Option<MemoryGrant> {
        match self {
            Self::Static(_) => None,
            Self::GrantBackedStatic { grant, .. } => Some(grant.clone()),
        }
    }

    fn remaining_live_reservation_bytes(&self, tracker_remaining_bytes: u64) -> Result<u64> {
        match self {
            Self::Static(_) => Ok(tracker_remaining_bytes),
            Self::GrantBackedStatic { grant, .. } => {
                Ok(tracker_remaining_bytes.min(grant.remaining_live_reservation_bytes()?))
            },
        }
    }

    fn try_reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        match self {
            Self::Static(_) => Ok(true),
            Self::GrantBackedStatic { grant, .. } => grant.try_reserve_live_bytes(bytes),
        }
    }
}

impl ReservationTracker {
    #[cfg(test)]
    pub(in crate::exec) fn new(memory_space: MemorySpaceId, budget_bytes: u64) -> Self {
        Self::new_with_budget(memory_space, ReservationBudget::Static(budget_bytes))
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate reservation contract tests"))]
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
            observation_scope: observation_scope.cloned(),
            next_reservation_id: Rc::new(Cell::new(1)),
            diagnostic_context: Rc::new(RefCell::new(ReservationDiagnosticContext::default())),
        }
    }

    pub(in crate::exec) fn new_for_context(
        context: &ExecutionContext,
        budget_bytes: u64,
        _metrics: &mut QueryMetrics,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let memory_grant = context.memory_grant()?;
        let budget = match memory_grant {
            Some(grant) => ReservationBudget::GrantBackedStatic { budget_bytes, grant },
            None => ReservationBudget::Static(budget_bytes),
        };
        let mut tracker =
            Self::new_with_budget_and_observation(context.device_memory_space(), budget, observation_scope);
        tracker.query_id = context.query_id().cloned();
        Ok(tracker)
    }

    pub(in crate::exec) fn new_for_compiled_context(
        context: &ExecutionContext,
        _plan: &QueryPlan,
        _pipeline: &CompiledPipeline,
        budget_bytes: u64,
        metrics: &mut QueryMetrics,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        Self::new_for_context(context, budget_bytes, metrics, observation_scope)
    }

    pub(in crate::exec) fn budget(&self) -> u64 {
        self.budget.bytes()
    }

    pub(in crate::exec) fn reserved_bytes(&self) -> u64 {
        self.raw_reserved_bytes()
    }

    pub(in crate::exec) fn remaining_bytes(&self) -> u64 {
        self.budget().saturating_sub(self.raw_reserved_bytes())
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
                request.requested_bytes,
                &request,
                reservation_projection_ledger,
                observation.clone(),
            ),
            ActiveReservationSemantics::Work => ActiveReservation::work(
                self.memory_space,
                Rc::clone(&self.reserved_bytes),
                self.budget.live_accounting(),
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
        // Release the RefCell borrow before metrics registration can re-borrow diagnostic state.
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
