//! Public request/snapshot DTOs for query-service admission.

use super::*;

/// Point-in-time service admission state for fairness and overload reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceAdmissionSnapshot {
    /// Number of currently admitted query handles.
    pub active_queries: usize,
    /// Number of admissions currently waiting in the service queue.
    pub queued_queries: usize,
    /// Queue depth by configured tenant, in admission rotation order.
    pub queued_by_tenant: Vec<TenantQueueDepth>,
    /// Tenant that will be considered first on the next admission decision.
    pub next_tenant_id: Option<TenantId>,
}

/// Process-wide active-query slots currently held on one service device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryServiceDeviceAdmissionSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Slots held across every `QueryService` in this process.
    pub process_active_query_slots: usize,
}

/// Per-device resident graph cache state for service diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryServiceGraphCacheSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Whether a positive resident graph cache cap configured this device.
    pub enabled: bool,
    /// Available plus checked-out entries.
    pub entries: usize,
    /// Device bytes retained by all entries.
    pub resident_bytes: u64,
    /// Device bytes held by checked-out entries.
    pub in_use_bytes: u64,
    /// Successful exact-runtime checkouts.
    pub hits: u64,
    /// Cold graph executions elected by the build coordinator.
    pub misses: u64,
    /// Entries evicted under capacity or governor pressure.
    pub evictions: u64,
    /// Graph-build generations currently in flight.
    pub builds_in_flight: usize,
    /// Follower waits joined since service creation.
    pub build_waits: u64,
}

/// Per-tenant queue depth in a [`QueryServiceAdmissionSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQueueDepth {
    /// Tenant identifier.
    pub tenant_id: TenantId,
    /// Number of queued admission requests for this tenant.
    pub queued_queries: usize,
}

/// Request-scoped query-service grant overrides.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryGrantRequest {
    /// Optional per-request minimum grant floor in bytes.
    pub requested_grant_min_bytes: Option<u64>,
    /// Optional per-request preferred grant in bytes.
    pub requested_grant_bytes: Option<u64>,
}

impl QueryGrantRequest {
    /// Creates an empty request that uses the tenant grant band unchanged.
    pub const fn new() -> Self {
        Self { requested_grant_min_bytes: None, requested_grant_bytes: None }
    }

    /// Creates a request with only a preferred grant override.
    pub const fn with_requested_grant_bytes(requested_grant_bytes: u64) -> Self {
        Self { requested_grant_min_bytes: None, requested_grant_bytes: Some(requested_grant_bytes) }
    }

    /// Creates a request with only a minimum grant floor override.
    pub const fn with_requested_grant_min_bytes(requested_grant_min_bytes: u64) -> Self {
        Self { requested_grant_min_bytes: Some(requested_grant_min_bytes), requested_grant_bytes: None }
    }

    /// Sets the per-request preferred grant override.
    #[must_use]
    pub const fn prefer(mut self, requested_grant_bytes: u64) -> Self {
        self.requested_grant_bytes = Some(requested_grant_bytes);
        self
    }

    /// Sets the per-request minimum grant floor override.
    #[must_use]
    pub const fn floor(mut self, requested_grant_min_bytes: u64) -> Self {
        self.requested_grant_min_bytes = Some(requested_grant_min_bytes);
        self
    }
}

/// Complete request-scoped metadata for one query-service admission attempt.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct QueryAdmissionRequest {
    /// Priority and adaptive budget-growth metadata.
    pub qos: QueryQos,
    /// Optional request-specific memory grant overrides.
    pub grant_request: QueryGrantRequest,
    /// Optional placement hints used only to order candidate devices.
    pub placement_hint: QueryPlacementHint,
    /// Optional exact-runtime graph-cache hit price.
    ///
    /// QueryService uses this grant request only after it has reserved both an
    /// idle stream-pool runtime and the matching resident graph cache entry.
    /// Device-level cache presence alone never selects this price.
    #[cfg(feature = "cugraph")]
    pub warm_graph_grant_request: Option<QueryGrantRequest>,
    runtime_observation: RuntimeObservationHandle,
    runtime_observation_owner: Option<RuntimeObservationOwner>,
    runtime_observation_activity: Option<Result<QueryActivityLease>>,
}

pub(in crate::runtime::query_service::service) struct AdmissionObservation {
    pub(in crate::runtime::query_service::service) handle: RuntimeObservationHandle,
    pub(in crate::runtime::query_service::service) owner: Option<RuntimeObservationOwner>,
    pub(in crate::runtime::query_service::service) activity: Option<QueryActivityLease>,
    pub(in crate::runtime::query_service::service) activity_error: Option<QueryError>,
}

impl AdmissionObservation {
    pub(in crate::runtime::query_service::service) fn as_ref(&self) -> &RuntimeObservationHandle {
        &self.handle
    }

    pub(in crate::runtime::query_service::service) fn reject_activity_error(
        &mut self,
        admission_request_id: AdmissionRequestId,
        tenant_id: &TenantId,
    ) -> Result<()> {
        let Some(error) = self.activity_error.take() else {
            return Ok(());
        };
        record_admission_lifecycle(
            &self.handle,
            admission_request_id,
            tenant_id,
            admission_rejection(&error),
            0,
            None,
            None,
        );
        Err(error)
    }
}

/// Opaque RAII token retained until query-service cleanup is complete.
///
/// The native engine never invokes the value. Dropping it returns activity
/// ownership to the adapter coordinator after queued admission, contexts,
/// grants, device slots, and cleanup work have all been released.
pub struct QueryActivityLease {
    _lease: Box<dyn Send + Sync>,
}

impl QueryActivityLease {
    pub fn new(lease: impl Send + Sync + 'static) -> Self {
        Self { _lease: Box::new(lease) }
    }
}

impl std::fmt::Debug for QueryActivityLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("QueryActivityLease").finish_non_exhaustive()
    }
}

impl QueryAdmissionRequest {
    /// Creates a request with default QoS, tenant-default grants, and no placement hint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets caller-supplied QoS metadata.
    pub fn with_qos(mut self, qos: QueryQos) -> Self {
        self.qos = qos;
        self
    }

    /// Sets request-scoped minimum/preferred grant overrides.
    pub fn with_grant_request(mut self, grant_request: QueryGrantRequest) -> Self {
        self.grant_request = grant_request;
        self
    }

    /// Sets optional placement metadata.
    pub fn with_placement_hint(mut self, placement_hint: QueryPlacementHint) -> Self {
        self.placement_hint = placement_hint;
        self
    }

    /// Attaches the query observation owner created by the adapter before enqueue.
    pub fn with_runtime_observation_owner(mut self, owner: RuntimeObservationOwner) -> Self {
        self.runtime_observation = owner.handle();
        self.runtime_observation_owner = Some(owner);
        self
    }

    /// Attaches coordinator-owned recording access without query-seal authority.
    pub fn with_runtime_observation_handle(mut self, observation: RuntimeObservationHandle) -> Self {
        self.runtime_observation = observation;
        self.runtime_observation_owner = None;
        self
    }

    /// Retains an adapter-owned activity token through delayed native cleanup.
    pub fn with_runtime_observation_activity(mut self, activity: QueryActivityLease) -> Self {
        self.runtime_observation_activity = Some(Ok(activity));
        self
    }

    /// Carries fallible adapter activity acquisition into native admission.
    ///
    /// Coordinated callers must pass acquisition failure through this field so
    /// admission rejects structurally instead of silently omitting the lease.
    pub fn with_runtime_observation_activity_result(mut self, activity: Result<QueryActivityLease>) -> Self {
        self.runtime_observation_activity = Some(activity);
        self
    }

    pub(in crate::runtime::query_service::service) fn into_runtime_observation(self) -> AdmissionObservation {
        let (activity, activity_error) = match self.runtime_observation_activity {
            Some(Ok(activity)) => (Some(activity), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        AdmissionObservation {
            handle: self.runtime_observation,
            owner: self.runtime_observation_owner,
            activity,
            activity_error,
        }
    }

    /// Sets the grant request that is valid only for a reserved exact-runtime
    /// resident graph cache hit.
    #[cfg(feature = "cugraph")]
    pub fn with_warm_graph_grant_request(mut self, grant_request: QueryGrantRequest) -> Self {
        self.warm_graph_grant_request = Some(grant_request);
        self
    }
}

/// Effective grant band for one query-service tenant request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryGrantBand {
    /// Minimum grant bytes admission must provide before the query can run.
    pub min_bytes: u64,
    /// Preferred grant bytes admission tries to provide when capacity allows.
    pub want_bytes: u64,
}
