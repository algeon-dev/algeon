use super::*;

pub(super) const GRAPH_TICKET_ASK_SAFETY_NUMERATOR: u64 = 6;
pub(super) const GRAPH_TICKET_ASK_SAFETY_DENOMINATOR: u64 = 5;
pub(super) const GRAPH_TICKET_NON_GRAPH_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GraphAdmissionGrantAsk {
    cold_bytes: u64,
    warm_bytes: u64,
}

impl NexusNativeExecutionConfig {
    /// Execute through an admitted multi-tenant query-service handle.
    ///
    /// The handle supplies the service-owned CUDA stream pool, source-fetch
    /// gates, deployment IO capability, and tenant device-budget slice used by
    /// native execution. The handle must remain unfinished until every plan
    /// node built from this config has completed execution; dropping the final
    /// handle releases the service admission slot.
    ///
    /// Source storage profiles are service-owned on this path. Register them
    /// on `QueryServiceConfig::source_storage_profiles`; profiles attached to
    /// this per-plan config are used only for standalone execution contexts.
    pub fn with_query_service_handle(mut self, handle: Arc<QueryHandle>) -> Self {
        self.memory_resource_policy = MemoryResourcePolicy::ExternalRuntime;
        self.memory_control_policy = handle.memory_control_policy();
        self.boundary_policy = handle.boundary_policy();
        self.source_scheduler_policy = handle.source_scheduler_policy();
        self.source_chunk_planning_mode = handle.source_chunk_planning_mode();
        self.lakehouse_cache_fill_config = handle.lakehouse_cache_fill_config();
        self.native_pipeline_execution_mode = handle.native_pipeline_execution_mode();
        self.stddev_samp_execution_mode = handle.stddev_samp_execution_mode();
        self.dynamic_filter_execution_options = handle.dynamic_filter_execution_options();
        self.keyed_hash_join_strategy_options = handle.keyed_hash_join_strategy_options();
        self.aggregate_strategy_execution_options = handle.aggregate_strategy_execution_options();
        self.source_chunk_planning_options = handle.source_chunk_planning_options();
        self.query_service_handle = Some(handle);
        self.query_admission_facts_recorded = Some(Arc::new(AtomicBool::new(false)));
        self.query_admission_source = None;
        self.query_admission_ticket = None;
        self.requested_grant_min_bytes = None;
        self.requested_grant_bytes = None;
        self.unbounded_native_execution_limiter = None;
        self
    }

    /// Set a request-scoped minimum query-service grant floor.
    ///
    /// The value is validated by query-service admission when a plan ticket is
    /// executed. It must be within the backend tenant grant band and device
    /// capacity. `None` uses the backend tenant minimum.
    pub fn with_requested_grant_min_bytes(mut self, requested_grant_min_bytes: Option<u64>) -> Self {
        self.requested_grant_min_bytes = requested_grant_min_bytes;
        self
    }

    /// Request-scoped minimum query-service grant floor in bytes.
    pub fn requested_grant_min_bytes(&self) -> Option<u64> {
        self.requested_grant_min_bytes
    }

    /// Set a request-scoped preferred query-service grant override.
    ///
    /// The value is validated by query-service admission when a plan ticket is
    /// executed. It must be within the backend tenant grant band and device
    /// capacity. `None` uses the backend default want.
    pub fn with_requested_grant_bytes(mut self, requested_grant_bytes: Option<u64>) -> Self {
        self.requested_grant_bytes = requested_grant_bytes;
        self
    }

    /// Request-scoped preferred query-service grant override in bytes.
    pub fn requested_grant_bytes(&self) -> Option<u64> {
        self.requested_grant_bytes
    }

    #[cfg(any(feature = "server", test))]
    pub(crate) fn with_unbounded_native_execution_limiter(mut self, limiter: UnboundedNativeExecutionLimiter) -> Self {
        self.unbounded_native_execution_limiter = Some(limiter);
        self
    }

    pub(crate) fn with_query_admission_source(mut self, source: QueryAdmissionSource) -> Self {
        if self.query_service_handle.is_none() {
            self.query_admission_source = Some(source);
            self.query_admission_facts_recorded = None;
            self.query_admission_ticket = None;
        }
        self
    }

    pub(crate) fn with_plan_query_admission_ticket(mut self) -> Self {
        if self.query_service_handle.is_none()
            && self.query_admission_ticket.is_none()
            && let Some(source) = self.query_admission_source.clone()
        {
            let observation_capture = self.metrics_capture.as_ref().map(NativeMetricsCapture::new_execution);
            let observation = self.new_runtime_observation(observation_capture.as_ref());
            self.query_admission_ticket = Some(QueryAdmissionTicket::new_with_observation_and_capture(
                source,
                self.requested_grant_min_bytes,
                self.requested_grant_bytes,
                observation,
                observation_capture,
            ));
        }
        self
    }

    pub(super) fn new_plan_query_admission_ticket_with_graph_ask(
        &self,
        graph_ask: Option<GraphAdmissionGrantAsk>,
        placement_hint: QueryPlacementHint,
    ) -> Option<Arc<QueryAdmissionTicket>> {
        if self.query_service_handle.is_some() {
            return None;
        }
        let source = self.query_admission_source.clone()?;
        let observation_capture = self.metrics_capture.as_ref().map(NativeMetricsCapture::new_execution);
        let observation = self.new_runtime_observation(observation_capture.as_ref());
        Some(QueryAdmissionTicket::new_with_placement_hint_and_warm_graph_grant_and_observation_and_capture(
            source,
            self.requested_grant_min_bytes,
            self.requested_grant_bytes.or(graph_ask.map(|ask| ask.cold_bytes)),
            placement_hint,
            #[cfg(feature = "cugraph")]
            graph_ask.map(|ask| ask.warm_bytes),
            observation,
            observation_capture,
        ))
    }

    pub(super) fn new_runtime_observation(
        &self,
        capture: Option<&NativeExecutionMetricsCapture>,
    ) -> crate::native_exec::NativeRuntimeObservation {
        let query_id = self.query_id.clone().unwrap_or_else(QueryId::mint);
        let trace_id = self.trace_id.as_ref().map(|trace_id| RuntimeTraceId::new(trace_id.as_str()));
        match capture {
            Some(capture) => capture.new_runtime_observation(query_id, trace_id),
            None => {
                RuntimeObservationOwner::new(query_id, trace_id, RuntimeObservationConfig::native_execution()).into()
            },
        }
    }

    pub(crate) fn new_plan_query_admission_ticket_with_graph_model_ask(
        &self,
        graph_model_ask: Option<GraphAdmissionModelAsk>,
        placement_hint: QueryPlacementHint,
    ) -> nexus_query_engine::error::Result<Option<Arc<QueryAdmissionTicket>>> {
        let graph_ask = self.graph_aware_requested_grants_for_model_ask(graph_model_ask)?;
        Ok(self.new_plan_query_admission_ticket_with_graph_ask(graph_ask, placement_hint))
    }

    #[cfg(test)]
    pub(crate) fn graph_aware_requested_grant_bytes_for_model_ask(
        &self,
        graph_model_ask_bytes: Option<u64>,
    ) -> nexus_query_engine::error::Result<Option<u64>> {
        if self.requested_grant_bytes.is_some() {
            return Ok(None);
        }
        let Some(graph_model_ask_bytes) = graph_model_ask_bytes else {
            return Ok(None);
        };
        let Some(source) = self.query_admission_source.clone() else {
            return Ok(None);
        };
        let priced = graph_ticket_requested_grant_bytes(graph_model_ask_bytes);
        let band = source.requested_grant_band(QueryGrantRequest {
            requested_grant_min_bytes: self.requested_grant_min_bytes,
            requested_grant_bytes: None,
        })?;
        Ok(Some(priced.clamp(band.min_bytes, band.want_bytes)))
    }

    pub(super) fn graph_aware_requested_grants_for_model_ask(
        &self,
        graph_model_ask: Option<GraphAdmissionModelAsk>,
    ) -> nexus_query_engine::error::Result<Option<GraphAdmissionGrantAsk>> {
        if self.requested_grant_bytes.is_some() {
            return Ok(None);
        }
        let Some(graph_model_ask) = graph_model_ask else {
            return Ok(None);
        };
        let Some(source) = self.query_admission_source.clone() else {
            return Ok(None);
        };
        let band = source.requested_grant_band(QueryGrantRequest {
            requested_grant_min_bytes: self.requested_grant_min_bytes,
            requested_grant_bytes: None,
        })?;
        let price = |bytes| graph_ticket_requested_grant_bytes(bytes).clamp(band.min_bytes, band.want_bytes);
        Ok(Some(GraphAdmissionGrantAsk {
            cold_bytes: price(graph_model_ask.cold_bytes()),
            warm_bytes: price(graph_model_ask.warm_bytes()),
        }))
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn with_graph_aware_plan_query_admission_ticket(
        mut self,
        graph_model_ask: Option<GraphAdmissionModelAsk>,
        placement_hint: QueryPlacementHint,
    ) -> nexus_query_engine::error::Result<Self> {
        let graph_ask = self.graph_aware_requested_grants_for_model_ask(graph_model_ask)?;
        if let Some(ticket) = self.new_plan_query_admission_ticket_with_graph_ask(graph_ask, placement_hint) {
            self.query_admission_ticket = Some(ticket);
        }
        Ok(self)
    }

    pub(crate) fn query_admission_source(&self) -> Option<QueryAdmissionSource> {
        self.query_admission_source.clone()
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn with_existing_query_admission_ticket(mut self, ticket: Arc<QueryAdmissionTicket>) -> Self {
        if self.query_service_handle.is_none() {
            self.query_admission_ticket = Some(ticket);
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id(&self) -> Option<u64> {
        self.query_admission_ticket.as_ref().map(|ticket| ticket.id())
    }

    pub(super) fn query_admission_ticket_for_task_context(
        &self,
        task_context: &TaskContext,
    ) -> Option<Arc<QueryAdmissionTicket>> {
        query_admission_ticket_from_task_context(task_context)
            .or_else(|| self.query_admission_ticket.as_ref().map(Arc::clone))
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id_for_task_context(&self, task_context: &TaskContext) -> Option<u64> {
        self.query_admission_ticket_for_task_context(task_context).map(|ticket| ticket.id())
    }

    #[cfg(any(test, feature = "cugraph"))]
    pub(crate) fn query_admission_ticket(&self) -> Option<Arc<QueryAdmissionTicket>> {
        self.query_admission_ticket.as_ref().map(Arc::clone)
    }

    pub(crate) fn unbounded_native_execution_limiter(&self) -> Option<UnboundedNativeExecutionLimiter> {
        self.unbounded_native_execution_limiter.clone()
    }

    #[cfg(test)]
    pub(crate) fn unbounded_native_max_concurrent_queries(&self) -> Option<std::num::NonZeroUsize> {
        self.unbounded_native_execution_limiter.as_ref().map(UnboundedNativeExecutionLimiter::max_concurrent_queries)
    }

    /// Returns whether execution is bound to a QueryService admission handle.
    pub(crate) fn has_query_service_handle(&self) -> bool {
        self.query_service_handle.is_some()
    }

    pub(crate) fn has_query_admission_authority(&self) -> bool {
        self.query_service_handle.is_some()
            || self.query_admission_source.is_some()
            || self.query_admission_ticket.is_some()
    }

    #[cfg(feature = "server")]
    pub(crate) fn query_admission_source_matches(
        &self,
        service: &nexus_query_engine::runtime::QueryService,
        tenant_id: &nexus_query_engine::runtime::TenantId,
    ) -> bool {
        self.query_admission_source
            .as_ref()
            .is_some_and(|source| source.service_matches(service) && source.tenant_id() == tenant_id)
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn is_standalone_bounded(&self) -> bool {
        matches!(self.memory_control_policy, MemoryControlPolicy::Bounded { .. })
            && !self.has_query_admission_authority()
    }
}

pub(super) fn graph_ticket_requested_grant_bytes(graph_model_ask_bytes: u64) -> u64 {
    let padded = ceil_saturating_ratio(
        graph_model_ask_bytes,
        GRAPH_TICKET_ASK_SAFETY_NUMERATOR,
        GRAPH_TICKET_ASK_SAFETY_DENOMINATOR,
    );
    padded.saturating_add(GRAPH_TICKET_NON_GRAPH_FLOOR_BYTES)
}

pub(super) fn ceil_saturating_ratio(value: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return u64::MAX;
    }
    let product = u128::from(value).saturating_mul(u128::from(numerator));
    let rounded = product.saturating_add(u128::from(denominator - 1)) / u128::from(denominator);
    rounded.min(u128::from(u64::MAX)) as u64
}
