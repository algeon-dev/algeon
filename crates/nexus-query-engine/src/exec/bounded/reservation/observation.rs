use super::*;

#[derive(Debug, Clone)]
pub(super) struct ReservationObservation {
    pub(super) scope: RuntimeExecutionObservationScope,
    pub(super) reservation_id: RuntimeReservationId,
    pub(super) lineage_id: RuntimeReservationId,
    pub(super) task_attempt_id: Option<TaskAttemptId>,
    pub(super) span_id: Option<RuntimeSpanId>,
}

impl ReservationObservation {
    pub(super) fn retarget(&mut self, task_attempt_id: Option<TaskAttemptId>, span_id: Option<RuntimeSpanId>) {
        self.task_attempt_id = task_attempt_id;
        self.span_id = span_id;
    }

    pub(super) fn record(
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

    pub(super) fn record_denied(
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
            value.map_or(0, |value| value.len() as u64)
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
            _ => u64::MAX,
        };
        let source_location_bytes = context
            .path
            .as_ref()
            .map_or(0, |_| self.scope.observation().source_location_id_retained_bytes().unwrap_or(0));
        42u64
            .saturating_add(self.scope.plan_execution_id().as_str().len() as u64)
            .saturating_add(self.reservation_id.as_str().len() as u64)
            .saturating_add(self.lineage_id.as_str().len() as u64)
            .saturating_add(task_attempt.map_or(0, |attempt| (attempt.task_id.as_str().len() as u64).saturating_add(4)))
            .saturating_add(optional_text_bytes(context.task_kind.as_deref()))
            .saturating_add(context.reservation_kind.as_str().len() as u64)
            .saturating_add(memory_space_bytes)
            .saturating_add(context.memory_boundary.len() as u64)
            .saturating_add(optional_text_bytes(context.node_id.as_ref().map(PlanNodeId::as_str)))
            .saturating_add(optional_text_bytes(context.stage.as_deref()))
            .saturating_add(source_location_bytes)
            .saturating_add(context.row_groups.as_ref().map_or(0, |values| (values.len() as u64).saturating_mul(8)))
            .saturating_add(optional_text_bytes(context.resident_resource_id.as_deref()))
            .saturating_add(optional_text_bytes(context.source_read_contract_quality.as_deref()))
            .saturating_add(66)
    }
}

#[derive(Debug, Clone)]
pub(super) struct ActiveReservationEventContext {
    pub(super) reservation_kind: ReservationKind,
    pub(super) memory_space: MemorySpaceId,
    pub(super) memory_boundary: String,
    pub(super) budget_bytes: u64,
    pub(super) query_id: Option<QueryId>,
    pub(super) node_id: Option<PlanNodeId>,
    pub(super) stage: Option<String>,
    pub(super) chunk_index: Option<usize>,
    pub(super) file_index: Option<usize>,
    pub(super) path: Option<String>,
    pub(super) row_groups: Option<Vec<usize>>,
    pub(super) attempt: Option<usize>,
    pub(super) max_attempts: Option<usize>,
    pub(super) compiled_task_id: Option<CompiledTaskId>,
    pub(super) task_attempt_id: Option<TaskAttemptId>,
    pub(super) task_kind: Option<String>,
    pub(super) resident_resource_id: Option<String>,
    pub(super) source_read_contract_quality: Option<String>,
    pub(super) source_read_planning_bytes: Option<u64>,
    pub(super) source_read_estimated_bytes: Option<u64>,
    pub(super) alias: bool,
}

impl ActiveReservationEventContext {
    pub(super) fn from_request(request: &MemoryReservationRequest) -> Self {
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

    pub(super) fn with_alias(mut self) -> Self {
        self.alias = true;
        self
    }

    pub(super) fn event(
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
