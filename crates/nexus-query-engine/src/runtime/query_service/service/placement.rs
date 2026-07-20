//! Device placement, governor grants, and query slot acquisition.

use super::*;

pub(in crate::runtime::query_service::service) fn unhealthy_placement_device_error(
    device: cudf::DeviceId,
    health: cudf::runtime::DeviceHealthSnapshot,
) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::CudfPersistentFailure,
        "nexus_query_engine.query_service.placement",
        "CUDA device is unhealthy and cannot admit new queries",
    )
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("device_health", "unhealthy")
    .with_fact("device_fault_count", health.fault_count())
    .with_fact("reason", "device_health_rejected_admission");
    if let Some(first_fault) = health.first_fault() {
        error = error
            .with_fact("first_device_fault_kind", format!("{:?}", first_fault.kind()))
            .with_fact("first_device_fault_operation", first_fault.operation());
    }
    error
}

pub(in crate::runtime::query_service::service) fn resolve_fallback_placement_device(
    device_ordinal: usize,
) -> Result<cudf::DeviceId> {
    #[cfg(test)]
    if let Ok(raw_ordinal) = i32::try_from(device_ordinal)
        && let Some(device) = cudf::testing::device_safety::injected_device_id(raw_ordinal)
    {
        return Ok(device);
    }

    cudf::DeviceId::new(device_ordinal).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.query_service.placement",
            "failed to resolve fallback CUDA device during query placement",
            error,
        )
    })
}

impl QueryService {
    pub(in crate::runtime::query_service::service) fn try_grant_from_governor(
        &self,
        governor: &MemoryGovernor,
        admission_request_id: AdmissionRequestId,
        grant_policy: TenantGrantPolicy,
        qos: QueryQos,
        deadline: Option<Instant>,
        active_queries: usize,
    ) -> Result<Option<MemoryGrant>> {
        let available_bytes = governor.available_bytes()?;
        if available_bytes < grant_policy.min_bytes {
            return Ok(None);
        }
        let reserved_for_remaining_slots = self.reserved_min_bytes_for_remaining_slots(active_queries);
        let Some(want_bytes) = self.grant_budget_bytes(available_bytes, grant_policy, reserved_for_remaining_slots)
        else {
            return Ok(None);
        };
        let mut request = GrantRequest::new(admission_request_id, grant_policy.min_bytes, want_bytes);
        request = request.with_qos(qos);
        if let Some(deadline) = deadline {
            request = request.with_deadline(deadline);
        }
        governor.try_request(request)
    }

    pub(in crate::runtime::query_service) fn try_place_query(
        &self,
        tenant_id: &TenantId,
        admission_request_id: AdmissionRequestId,
        grant_policy: TenantGrantPolicy,
        #[cfg(feature = "cugraph")] warm_graph_grant_policy: Option<TenantGrantPolicy>,
        qos: QueryQos,
        placement_hint: &QueryPlacementHint,
        deadline: Option<Instant>,
        active_queries: usize,
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let Some(resources) = &self.inner.resources else {
            let fallback_device = resolve_fallback_placement_device(self.inner.config.device_ordinal)?;
            let health = cudf::runtime::device_health_snapshot(fallback_device);
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                return Err(unhealthy_placement_device_error(fallback_device, health));
            }
            if !self.fallback_active_query_cap_allows(active_queries) {
                return Ok(None);
            }
            let Some(memory_grant) = self.try_grant_from_governor(
                &self.inner.fallback_memory_governor,
                admission_request_id,
                grant_policy,
                qos,
                deadline,
                active_queries,
            )?
            else {
                return Ok(None);
            };
            return Ok(Some(PlacedQuery {
                device_ordinal: self.inner.config.device_ordinal,
                memory_grant,
                device_slot: None,
                grant_policy,
                #[cfg(feature = "cugraph")]
                warm_graph_reservation: None,
            }));
        };

        let placement_request = PlacementRequest::with_grant_range(grant_policy.min_bytes, grant_policy.want_bytes)
            .with_tenant_id(tenant_id.as_str())
            .with_placement_hint(placement_hint.clone());
        let mut any_device_can_fit_minimum = false;
        let mut healthy_candidates = 0usize;
        let mut first_unhealthy = None;
        for device in self.placement_candidates(resources, &placement_request)? {
            let health = cudf::runtime::device_health_snapshot(device);
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                first_unhealthy.get_or_insert((device, health));
                continue;
            }
            healthy_candidates = healthy_candidates.saturating_add(1);
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() >= grant_policy.min_bytes {
                any_device_can_fit_minimum = true;
            }
            let available_bytes = context.available_bytes()?;
            if available_bytes < grant_policy.min_bytes {
                continue;
            }
            // This helper accounts for the query being selected; compute it
            // before the device slot increments the active-query counter.
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, active_queries)?;
            let Some(device_slot) = context.try_acquire_query_slot(self.inner.config.max_queries_per_device) else {
                continue;
            };
            #[cfg(feature = "cugraph")]
            let warm_graph_reservation = match (warm_graph_grant_policy, placement_hint.resident_graph_key()) {
                (Some(_warm_policy), Some(resident_graph_key)) => {
                    resources.try_reserve_warm_graph(context, resident_graph_key)?
                },
                _ => None,
            };
            #[cfg(feature = "cugraph")]
            let selected_grant_policy = if warm_graph_reservation.is_some() {
                warm_graph_grant_policy.expect("warm reservation requires warm graph grant policy")
            } else {
                grant_policy
            };
            #[cfg(not(feature = "cugraph"))]
            let selected_grant_policy = grant_policy;
            let Some(want_bytes) =
                self.grant_budget_bytes(available_bytes, selected_grant_policy, reserved_for_remaining_slots)
            else {
                slot_notifications.release_slot(device_slot);
                continue;
            };
            let mut request = GrantRequest::new(admission_request_id, selected_grant_policy.min_bytes, want_bytes);
            request = request.with_qos(qos);
            if let Some(deadline) = deadline {
                request = request.with_deadline(deadline);
            }
            match context.try_grant_query(request) {
                Ok(Some(memory_grant)) => {
                    return Ok(Some(PlacedQuery {
                        device_ordinal: device.ordinal(),
                        memory_grant,
                        device_slot: Some(device_slot),
                        grant_policy: selected_grant_policy,
                        #[cfg(feature = "cugraph")]
                        warm_graph_reservation,
                    }));
                },
                Ok(None) => slot_notifications.release_slot(device_slot),
                Err(error) => {
                    slot_notifications.release_slot(device_slot);
                    return Err(error);
                },
            }
        }
        if any_device_can_fit_minimum {
            return Ok(None);
        }
        if healthy_candidates == 0
            && let Some((device, health)) = first_unhealthy
        {
            return Err(unhealthy_placement_device_error(device, health));
        }
        Err(QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.placement",
            "no configured CUDA device has enough capacity for the query minimum budget",
        )
        .with_fact("tenant_id", tenant_id.as_str())
        .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
        .with_fact("reason", "query_min_budget_exceeds_device_capacity"))
    }

    pub(in crate::runtime::query_service) fn attach_admission_release_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
        let service = self.clone();
        grant.notify_on_release(Arc::new(move || {
            let mut guard = match service.inner.admission.lock() {
                Ok(guard) => guard,
                Err(_error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to lock query service admission for grant release wakeup"
                    );
                    return;
                },
            };
            if guard.cleanup_in_progress != 0 {
                return;
            }
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    pub(in crate::runtime::query_service) fn attach_admission_budget_decrease_wakeup(
        &self,
        grant: &MemoryGrant,
    ) -> Result<()> {
        let service = self.clone();
        grant.notify_on_budget_decrease(Arc::new(move || {
            let mut guard = match service.inner.admission.lock() {
                Ok(guard) => guard,
                Err(_error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to lock query service admission for grant budget-decrease wakeup"
                    );
                    return;
                },
            };
            if guard.cleanup_in_progress != 0 {
                return;
            }
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    pub(in crate::runtime::query_service::service) fn placement_candidates(
        &self,
        resources: &QueryServiceResources,
        request: &PlacementRequest,
    ) -> Result<Vec<cudf::DeviceId>> {
        let candidates = self.inner.config.placement_policy.candidates(&resources.devices, request)?;
        #[cfg(feature = "cugraph")]
        if let Some(resident_graph_key) = request.placement_hint.resident_graph_key() {
            let mut hit_ordinals = BTreeSet::new();
            for device in &candidates {
                if let Some(cache) = resources.resident_graph_cache_for_device(device.ordinal()) {
                    match cache.contains_resident_key(resident_graph_key) {
                        Ok(true) => {
                            hit_ordinals.insert(device.ordinal());
                        },
                        Ok(false) => {},
                        Err(error) => {
                            eprintln!(
                                "[nexus-query-engine] warning: failed to inspect resident graph cache for placement affinity on device {}: {error}",
                                device.ordinal()
                            );
                        },
                    }
                }
            }
            return Ok(rank_cache_affine_candidates(candidates, &hit_ordinals));
        }
        Ok(candidates)
    }

    pub(in crate::runtime::query_service::service) fn device_active_query_cap_allows(
        &self,
        context: &DeviceContext,
    ) -> bool {
        self.inner
            .config
            .max_queries_per_device
            .is_none_or(|max_queries_per_device| context.active_queries() < max_queries_per_device)
    }

    pub(in crate::runtime::query_service::service) fn placement_policy_admits_ordinal(&self, ordinal: usize) -> bool {
        placement_policy_admits_ordinal(self.inner.config.placement_policy, ordinal)
    }

    pub(in crate::runtime::query_service::service) fn fallback_active_query_cap_allows(
        &self,
        active_queries: usize,
    ) -> bool {
        self.inner
            .config
            .max_queries_per_device
            .is_none_or(|max_queries_per_device| active_queries < max_queries_per_device)
    }
}
