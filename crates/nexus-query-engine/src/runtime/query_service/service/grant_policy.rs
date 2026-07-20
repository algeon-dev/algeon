//! Tenant grant bands, validation, and reservation budget helpers.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::runtime::query_service) struct TenantGrantPolicy {
    pub(in crate::runtime::query_service) min_bytes: u64,
    pub(in crate::runtime::query_service) want_bytes: u64,
}

impl TenantGrantPolicy {
    pub(in crate::runtime::query_service::service) fn with_requested_grant_bytes(
        self,
        requested_grant_bytes: u64,
    ) -> Self {
        Self { want_bytes: requested_grant_bytes, ..self }
    }

    pub(in crate::runtime::query_service::service) fn with_requested_grant_min_bytes(
        self,
        requested_grant_min_bytes: u64,
    ) -> Self {
        Self { min_bytes: requested_grant_min_bytes, ..self }
    }
}

impl From<TenantGrantPolicy> for QueryGrantBand {
    fn from(policy: TenantGrantPolicy) -> Self {
        Self { min_bytes: policy.min_bytes, want_bytes: policy.want_bytes }
    }
}

impl QueryService {
    pub fn tenant_budget(&self, tenant_id: &TenantId) -> Result<u64> {
        self.tenant_grant_policy(tenant_id).map(|grant| grant.want_bytes)
    }

    /// Returns the configured grant band for a tenant.
    pub fn tenant_grant_band(&self, tenant_id: &TenantId) -> Result<QueryGrantBand> {
        self.tenant_grant_policy(tenant_id).map(Into::into)
    }

    /// Resolve the effective preferred grant for a tenant request.
    pub fn tenant_requested_budget(&self, tenant_id: &TenantId, requested_grant_bytes: Option<u64>) -> Result<u64> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(|grant| grant.want_bytes)
    }

    /// Resolve the effective preferred grant for a tenant request with an
    /// optional request-specific grant floor.
    pub fn tenant_requested_budget_with_grant_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<u64> {
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(|grant| grant.want_bytes)
    }

    /// Resolve the effective grant band for a tenant request.
    pub fn tenant_requested_grant_band_with_grant_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryGrantBand> {
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(Into::into)
    }

    /// Largest device capacity represented by this query service.
    pub fn max_device_capacity_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.inner.fallback_memory_governor.capacity_bytes(),
            |resources| resources.devices.contexts().map(|context| context.capacity_bytes()).max().unwrap_or(0),
        )
    }

    /// Largest physical device-total byte count represented by this query service.
    ///
    /// Admission-only services have no CUDA inventory, so they fall back to the
    /// configured governor capacity used by deterministic CPU-only tests.
    pub fn max_device_total_memory_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.max_device_capacity_bytes(),
            |resources| resources.devices.contexts().map(|context| context.facts().total_bytes).max().unwrap_or(0),
        )
    }

    pub(in crate::runtime::query_service::service) fn tenant_grant_policy(
        &self,
        tenant_id: &TenantId,
    ) -> Result<TenantGrantPolicy> {
        self.inner.tenant_grants.get(tenant_id).copied().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "unknown tenant id for query service admission",
            )
            .with_fact("tenant_id", tenant_id.as_str())
        })
    }

    pub(in crate::runtime::query_service::service) fn tenant_grant_policy_for_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<TenantGrantPolicy> {
        let grant_policy = self.tenant_grant_policy(tenant_id)?;
        if let Some(requested_grant_bytes) = grant_request.requested_grant_bytes {
            self.validate_requested_grant_bytes(tenant_id, grant_policy, requested_grant_bytes)?;
        }
        if let Some(requested_grant_min_bytes) = grant_request.requested_grant_min_bytes {
            self.validate_requested_grant_min_bytes(tenant_id, grant_policy, requested_grant_min_bytes)?;
        }
        let mut effective_policy = grant_policy;
        if let Some(requested_grant_min_bytes) = grant_request.requested_grant_min_bytes {
            effective_policy = effective_policy.with_requested_grant_min_bytes(requested_grant_min_bytes);
        }
        if let Some(requested_grant_bytes) = grant_request.requested_grant_bytes {
            effective_policy = effective_policy.with_requested_grant_bytes(requested_grant_bytes);
        }
        if effective_policy.want_bytes < effective_policy.min_bytes {
            return Err(invalid_requested_grant_range_error(
                tenant_id,
                grant_policy,
                grant_request,
                "requested_grant_below_requested_minimum",
            ));
        }
        Ok(effective_policy)
    }

    pub(in crate::runtime::query_service::service) fn validate_requested_grant_bytes(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        requested_grant_bytes: u64,
    ) -> Result<()> {
        if requested_grant_bytes < grant_policy.min_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_below_minimum",
            ));
        }
        if requested_grant_bytes > grant_policy.want_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_above_tenant_want",
            ));
        }
        let max_device_capacity_bytes = self.max_device_capacity_bytes();
        if requested_grant_bytes > max_device_capacity_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                max_device_capacity_bytes,
                "requested_grant_above_device_capacity",
            ));
        }
        Ok(())
    }

    pub(in crate::runtime::query_service::service) fn validate_requested_grant_min_bytes(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        requested_grant_min_bytes: u64,
    ) -> Result<()> {
        if requested_grant_min_bytes < grant_policy.min_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_min_below_tenant_minimum",
            ));
        }
        if requested_grant_min_bytes > grant_policy.want_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_min_above_tenant_want",
            ));
        }
        let max_device_capacity_bytes = self.max_device_capacity_bytes();
        if requested_grant_min_bytes > max_device_capacity_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                max_device_capacity_bytes,
                "requested_grant_min_above_device_capacity",
            ));
        }
        if self.inner.config.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
            let idle_grant_ceiling = self.guaranteed_min_idle_grant_ceiling_bytes()?;
            if requested_grant_min_bytes > idle_grant_ceiling {
                return Err(invalid_requested_grant_min_error(
                    tenant_id,
                    grant_policy,
                    requested_grant_min_bytes,
                    max_device_capacity_bytes,
                    "requested_grant_min_above_guaranteed_min_slot_ceiling",
                )
                .with_fact("guaranteed_min_idle_grant_ceiling_bytes", idle_grant_ceiling)
                .with_fact("max_concurrent_queries", self.inner.config.max_concurrent_queries)
                .with_fact("max_tenant_min_device_budget_bytes", self.inner.max_tenant_min_bytes)
                .with_fact("slot_guarantee", self.inner.config.slot_guarantee.as_str()));
            }
        }
        Ok(())
    }

    pub(in crate::runtime::query_service::service) fn guaranteed_min_idle_grant_ceiling_bytes(&self) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(self.max_device_capacity_bytes());
        }
        let Some(resources) = &self.inner.resources else {
            return Ok(self
                .inner
                .fallback_memory_governor
                .capacity_bytes()
                .saturating_sub(self.reserved_min_bytes_for_remaining_slots(0)));
        };
        let mut ceiling = 0u64;
        for context in resources.devices.contexts() {
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device_at_idle(resources, context.device())?;
            ceiling = ceiling.max(context.capacity_bytes().saturating_sub(reserved_for_remaining_slots));
        }
        Ok(ceiling)
    }

    pub(in crate::runtime::query_service::service) fn remaining_slots_after_current(
        &self,
        active_queries: usize,
    ) -> usize {
        self.inner.config.max_concurrent_queries.saturating_sub(active_queries.saturating_add(1))
    }

    pub(in crate::runtime::query_service::service) fn reserved_min_bytes_for_remaining_slots(
        &self,
        active_queries: usize,
    ) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        let remaining_slots = self
            .remaining_slots_after_current(active_queries)
            .min(self.future_slots_after_current_on_fallback_device(active_queries));
        self.inner.max_tenant_min_bytes.saturating_mul(u64::try_from(remaining_slots).unwrap_or(u64::MAX))
    }

    pub(in crate::runtime::query_service::service) fn reserved_min_bytes_for_empty_slots(
        &self,
        active_queries: usize,
    ) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        let empty_slots = self
            .inner
            .config
            .max_concurrent_queries
            .saturating_sub(active_queries)
            .min(self.future_slots_on_fallback_device(active_queries));
        self.inner.max_tenant_min_bytes.saturating_mul(u64::try_from(empty_slots).unwrap_or(u64::MAX))
    }

    pub(in crate::runtime::query_service::service) fn future_slots_after_current_on_fallback_device(
        &self,
        active_queries: usize,
    ) -> usize {
        self.inner.config.max_queries_per_device.map_or(usize::MAX, |max_queries_per_device| {
            max_queries_per_device.saturating_sub(active_queries.saturating_add(1))
        })
    }

    pub(in crate::runtime::query_service::service) fn future_slots_on_fallback_device(
        &self,
        active_queries: usize,
    ) -> usize {
        self.inner
            .config
            .max_queries_per_device
            .map_or(usize::MAX, |max_queries_per_device| max_queries_per_device.saturating_sub(active_queries))
    }

    pub(in crate::runtime::query_service::service) fn future_min_slots_on_device(
        &self,
        context: &DeviceContext,
        min_bytes: u64,
        active_after_selected: usize,
    ) -> Result<usize> {
        if min_bytes == 0 {
            return Ok(usize::MAX);
        }
        let memory_slots = usize::try_from(context.available_bytes()? / min_bytes).unwrap_or(usize::MAX);
        let capped_slots = self.inner.config.max_queries_per_device.map_or(memory_slots, |max_queries_per_device| {
            memory_slots.min(max_queries_per_device.saturating_sub(active_after_selected))
        });
        Ok(capped_slots)
    }

    pub(in crate::runtime::query_service::service) fn idle_future_min_slots_on_device(
        &self,
        capacity_bytes: u64,
        min_bytes: u64,
        active_after_selected: usize,
    ) -> usize {
        if min_bytes == 0 {
            return usize::MAX;
        }
        let memory_slots = usize::try_from(capacity_bytes / min_bytes).unwrap_or(usize::MAX);
        self.inner.config.max_queries_per_device.map_or(memory_slots, |max_queries_per_device| {
            memory_slots.min(max_queries_per_device.saturating_sub(active_after_selected))
        })
    }

    pub(in crate::runtime::query_service::service) fn reserved_min_bytes_for_selected_device_at_idle(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let remaining_slots = self.remaining_slots_after_current(0);
        if remaining_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.idle_future_min_slots_on_device(context.capacity_bytes(), self.inner.max_tenant_min_bytes, 0);
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= remaining_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = remaining_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots =
            self.idle_future_min_slots_on_device(selected_context.capacity_bytes(), self.inner.max_tenant_min_bytes, 1);
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    pub(in crate::runtime::query_service::service) fn reserved_min_bytes_for_selected_device(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let remaining_slots = self.remaining_slots_after_current(active_queries);
        if remaining_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.future_min_slots_on_device(context, self.inner.max_tenant_min_bytes, context.active_queries())?;
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= remaining_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = remaining_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots = self.future_min_slots_on_device(
            selected_context,
            self.inner.max_tenant_min_bytes,
            selected_context.active_queries().saturating_add(1),
        )?;
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    pub(in crate::runtime::query_service::service) fn reserved_min_bytes_for_selected_device_empty_slots(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let empty_slots = self.inner.config.max_concurrent_queries.saturating_sub(active_queries);
        if empty_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.future_min_slots_on_device(context, self.inner.max_tenant_min_bytes, context.active_queries())?;
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= empty_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = empty_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots = self.future_min_slots_on_device(
            selected_context,
            self.inner.max_tenant_min_bytes,
            selected_context.active_queries(),
        )?;
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    pub(in crate::runtime::query_service::service) fn grant_budget_bytes(
        &self,
        available_bytes: u64,
        grant_policy: TenantGrantPolicy,
        reserved_for_remaining_slots: u64,
    ) -> Option<u64> {
        let required_floor = grant_policy.min_bytes.saturating_add(reserved_for_remaining_slots);
        if available_bytes < required_floor {
            return None;
        }
        Some(
            grant_policy
                .want_bytes
                .min(available_bytes.saturating_sub(reserved_for_remaining_slots))
                .max(grant_policy.min_bytes),
        )
    }
}

pub(in crate::runtime::query_service::service) fn invalid_requested_grant_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    requested_grant_bytes: u64,
    max_device_capacity_bytes: u64,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant",
        "requested query grant bytes must be within the tenant grant band and device capacity",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("requested_grant_bytes", requested_grant_bytes)
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("max_device_capacity_bytes", max_device_capacity_bytes)
    .with_fact("reason", reason)
}

pub(in crate::runtime::query_service::service) fn invalid_requested_grant_min_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    requested_grant_min_bytes: u64,
    max_device_capacity_bytes: u64,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant_min",
        "requested query grant minimum bytes must be within the tenant grant band and device capacity",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("requested_grant_min_bytes", requested_grant_min_bytes)
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("max_device_capacity_bytes", max_device_capacity_bytes)
    .with_fact("reason", reason)
}

pub(in crate::runtime::query_service::service) fn invalid_requested_grant_range_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    grant_request: QueryGrantRequest,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant_range",
        "requested query grant preferred bytes must be at least the requested minimum bytes",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact(
        "requested_grant_min_bytes",
        grant_request.requested_grant_min_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
    )
    .with_fact(
        "requested_grant_bytes",
        grant_request.requested_grant_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
    )
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("reason", reason)
}
