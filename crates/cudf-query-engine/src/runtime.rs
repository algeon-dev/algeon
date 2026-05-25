//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

#![deny(missing_docs)]

use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use object_store::ObjectStoreExt;

use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    error::{QueryError, QueryErrorCode, Result},
    mode::NativePipelineExecutionMode,
};

mod boundary;
mod bounded_join;
mod io_capability;
mod policy;
mod query_service;
mod reservation;

pub use boundary::{
    BuildMaterializationBoundary, ExecutionBoundaryPolicy, ProbeExecutionBoundary, ResultMaterializationBoundary,
    SourceReadBoundary,
};
pub use bounded_join::{
    BoundedJoinReservationContract, BoundedJoinReservationRequirement, BoundedJoinReservationStatus,
};
pub use io_capability::{
    DEFAULT_GDS_LOCAL_MOUNT_ROOT, DeploymentIoCapability, GdsAvailabilityStatus, GdsDriverProbeStatus,
    KVIKIO_COMPAT_MODE_ENV, KvikioCompatModeStatus, NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH, NvidiaFsProbeStatus,
    SourceGdsEligibility, source_gds_eligibility, source_gds_eligibility_with_default_mounts,
};
pub use policy::{DevicePlacement, MemoryControlPolicy, MemoryResourcePolicy, MemorySpaceId, ReservationKind};
pub use query_service::{
    QueryHandle, QueryService, QueryServiceAdmissionSnapshot, QueryServiceConfig, QueryServiceTenantConfig, TenantId,
    TenantQueryContext, TenantQueueDepth,
};
pub use reservation::{MemoryReservationRequest, RetryFacts};

/// Shared cooperative cancellation flag for native query execution.
///
/// Cancellation is observed only at native-engine boundaries. It does not
/// preempt a libcudf/CUDA call already in flight, but it prevents the next
/// source chunk, pipeline task, operator, or materialization step from starting
/// once the flag is set.
#[derive(Debug, Clone)]
pub struct ExecutionCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ExecutionCancellation {
    /// Creates a non-cancelled execution flag.
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Marks the execution as cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns a structured native error if cancellation has been requested.
    pub fn check(&self, operation: &'static str) -> Result<()> {
        if self.is_cancelled() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::Cancelled,
                operation,
                "native query execution was cancelled",
            )
            .with_fact("cancellation_requested", true));
        }
        Ok(())
    }
}

impl Default for ExecutionCancellation {
    fn default() -> Self {
        Self::new()
    }
}

/// Native source scheduling controls.
///
/// The scheduler preserves source order by default and uses `max_source_reads_in_flight`
/// as an upper bound on active source reads. A value of `1` keeps the historical
/// serialized source path. Values greater than `1` allow bounded execution to
/// prefetch the next source chunk while chunk-local operators consume the
/// current chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSchedulerPolicy {
    max_source_reads_in_flight: usize,
}

impl SourceSchedulerPolicy {
    /// Serialized source execution.
    pub const fn serialized() -> Self {
        Self { max_source_reads_in_flight: 1 }
    }

    /// Creates a source scheduler policy.
    pub const fn new(max_source_reads_in_flight: usize) -> Self {
        Self { max_source_reads_in_flight }
    }

    /// Maximum active source reads allowed by the scheduler.
    pub const fn max_source_reads_in_flight(self) -> usize {
        self.max_source_reads_in_flight
    }

    /// Returns whether source prefetch is enabled.
    pub const fn enables_prefetch(self) -> bool {
        self.max_source_reads_in_flight > 1
    }

    fn validate(self) -> Result<()> {
        if self.max_source_reads_in_flight == 0 {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_scheduler_policy.validate",
                "source scheduler max_source_reads_in_flight must be greater than zero",
            )
            .with_fact("field", "max_source_reads_in_flight")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for SourceSchedulerPolicy {
    fn default() -> Self {
        Self { max_source_reads_in_flight: 2 }
    }
}

/// Fallback host-buffer budget used when physical RAM cannot be detected.
pub const DEFAULT_HOST_FETCH_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
/// Maximum default host-buffer budget derived from physical RAM.
pub const DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Default service-wide concurrent object-store SourceFetch operation cap.
pub const DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS: usize = 64;
/// Default service-wide concurrent GDS DMA operation cap.
pub const DEFAULT_GDS_MAX_CONCURRENT_OPS: usize = 32;
/// Default per-chunk size threshold for selecting future GDS direct-device reads.
pub const DEFAULT_GDS_MIN_CHUNK_BYTES: u64 = 4 * 1024 * 1024;

/// Deployment-level source IO substrate preference.
///
/// This is a service policy, not a query-planning decision. `Auto` allows the
/// runtime to choose a future GDS direct-device path only when the deployment
/// and per-source eligibility checks pass. `HostBuffer` forces the
/// `SourceFetch + SourceDecode` host-buffer path. `DirectDevice` requires GDS
/// readiness at service/context construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SourceFetchSubstrate {
    /// Probe deployment capability and choose per-source IO strategy later.
    #[default]
    Auto,
    /// Force host-buffer fetch/decode for every source.
    HostBuffer,
    /// Require GDS direct-device reads for eligible local sources.
    DirectDevice,
}

impl SourceFetchSubstrate {
    /// Stable diagnostic string for this substrate policy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::HostBuffer => "host_buffer",
            Self::DirectDevice => "direct_device",
        }
    }
}

impl fmt::Display for SourceFetchSubstrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Source IO strategy selected for a source chunk.
///
/// This is a runtime/source-compile decision, not a planner admission result.
/// Remote object-store locations must continue to use host-buffer fetch/decode;
/// direct-device is reserved for future GDS-capable local/RDMA sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFetchStrategy {
    /// Fetch compressed bytes into host memory, then decode into a device frame.
    HostBuffer,
    /// Read directly through a future GDS/KvikIO path under the GDS DMA gate.
    DirectDevice,
}

impl SourceFetchStrategy {
    /// Stable diagnostic string for this strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostBuffer => "host_buffer",
            Self::DirectDevice => "direct_device",
        }
    }
}

impl fmt::Display for SourceFetchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable reason for a [`SourceFetchStrategyDecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFetchStrategyReason {
    /// Service config forced host-buffer IO.
    SubstrateHostBuffer,
    /// The source passed deployment, path, mount, and size checks for GDS.
    GdsEligible,
    /// Deployment-level GDS capability is unavailable.
    DeploymentUnavailable,
    /// Remote object-store/network locations cannot use direct-device reads.
    RemoteObjectStore,
    /// The source URI scheme is not a supported local/RDMA filesystem scheme.
    UnsupportedScheme,
    /// The local source path is outside configured GDS-capable roots.
    NotOnGdsCapableMount,
    /// The source chunk is below the GDS minimum chunk-size threshold.
    BelowMinChunkBytes,
}

impl SourceFetchStrategyReason {
    /// Stable diagnostic string for this reason.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SubstrateHostBuffer => "substrate_host_buffer",
            Self::GdsEligible => "gds_eligible",
            Self::DeploymentUnavailable => "deployment_unavailable",
            Self::RemoteObjectStore => "remote_object_store",
            Self::UnsupportedScheme => "unsupported_scheme",
            Self::NotOnGdsCapableMount => "not_on_gds_capable_mount",
            Self::BelowMinChunkBytes => "below_min_chunk_bytes",
        }
    }
}

impl fmt::Display for SourceFetchStrategyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<SourceGdsEligibility> for SourceFetchStrategyReason {
    fn from(eligibility: SourceGdsEligibility) -> Self {
        match eligibility {
            SourceGdsEligibility::Eligible => Self::GdsEligible,
            SourceGdsEligibility::DeploymentUnavailable => Self::DeploymentUnavailable,
            SourceGdsEligibility::RemoteObjectStore => Self::RemoteObjectStore,
            SourceGdsEligibility::UnsupportedScheme => Self::UnsupportedScheme,
            SourceGdsEligibility::NotOnGdsCapableMount => Self::NotOnGdsCapableMount,
            SourceGdsEligibility::BelowMinChunkBytes => Self::BelowMinChunkBytes,
        }
    }
}

/// Runtime source IO decision and report-facing diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceFetchStrategyDecision {
    /// Selected source IO strategy.
    pub strategy: SourceFetchStrategy,
    /// Stable reason for the selected strategy.
    pub reason: SourceFetchStrategyReason,
    /// Raw GDS eligibility result used by `Auto` and `DirectDevice`.
    pub gds_eligibility: SourceGdsEligibility,
    /// Effective service-level substrate policy.
    pub substrate: SourceFetchSubstrate,
}

impl SourceFetchStrategyDecision {
    /// Stable report facts for source IO strategy diagnostics.
    pub fn diagnostic_facts(&self) -> Vec<(&'static str, String)> {
        vec![
            ("source_fetch_strategy", self.strategy.as_str().to_string()),
            ("source_fetch_strategy_reason", self.reason.as_str().to_string()),
            ("source_fetch_substrate", self.substrate.as_str().to_string()),
            ("source_gds_eligibility", self.gds_eligibility.as_str().to_string()),
        ]
    }
}

/// Service-level source IO configuration shared by native executions.
///
/// These knobs protect physical resources shared across queries: host fetch
/// buffers, object-store fetch operations, and future GDS DMA operations. They
/// deliberately live outside per-query memory-control policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceFetchConfig {
    /// Substrate policy used when compiling source IO tasks.
    pub substrate: SourceFetchSubstrate,
    /// Max in-flight host-buffer bytes across queries. `None` auto-sizes from host RAM.
    pub host_fetch_budget_bytes: Option<u64>,
    /// Max concurrent object-store host-fetch operations across queries.
    pub host_fetch_max_concurrent_ops: usize,
    /// Max concurrent future GDS DMA operations across queries.
    pub gds_max_concurrent_ops: usize,
    /// Per-chunk size below which GDS-eligible sources fall back to host-buffer IO.
    pub gds_min_chunk_bytes: u64,
}

impl SourceFetchConfig {
    /// Validates the source IO config against deployment capability diagnostics.
    pub fn validate_for_deployment(&self, deployment: &DeploymentIoCapability) -> Result<()> {
        if let Some(host_fetch_budget_bytes) = self.host_fetch_budget_bytes {
            validate_host_fetch_budget_bytes(host_fetch_budget_bytes)?;
        }
        validate_host_fetch_max_concurrent_ops(self.host_fetch_max_concurrent_ops)?;
        validate_gds_max_concurrent_ops(self.gds_max_concurrent_ops)?;
        validate_gds_min_chunk_bytes(self.gds_min_chunk_bytes)?;
        if matches!(self.substrate, SourceFetchSubstrate::DirectDevice) && !deployment.gds_available {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_fetch_config.validate",
                "direct-device source fetch substrate requires available GDS deployment capability",
            )
            .with_fact("source_fetch_substrate", self.substrate.as_str())
            .with_fact("gds_available", deployment.gds_available)
            .with_fact("gds_status", deployment.gds_status.as_str()));
        }
        Ok(())
    }

    /// Selects the source IO strategy for a location and optional chunk-size estimate.
    ///
    /// This decision is intentionally pure and does not start IO. It centralizes
    /// the Phase 5 contract: remote object-store/network sources remain
    /// host-buffer reads, `HostBuffer` forces that path, and `Auto` /
    /// `DirectDevice` select direct-device only when GDS eligibility succeeds.
    pub fn decide_source_fetch_strategy(
        &self,
        deployment: &DeploymentIoCapability,
        source_location: &str,
        chunk_bytes: Option<u64>,
    ) -> SourceFetchStrategyDecision {
        let gds_eligibility = source_gds_eligibility_with_default_mounts(
            deployment,
            source_location,
            chunk_bytes,
            self.gds_min_chunk_bytes,
        );
        if matches!(self.substrate, SourceFetchSubstrate::HostBuffer) {
            return SourceFetchStrategyDecision {
                strategy: SourceFetchStrategy::HostBuffer,
                reason: SourceFetchStrategyReason::SubstrateHostBuffer,
                gds_eligibility,
                substrate: self.substrate,
            };
        }

        let reason = SourceFetchStrategyReason::from(gds_eligibility);
        let strategy = if matches!(gds_eligibility, SourceGdsEligibility::Eligible) {
            SourceFetchStrategy::DirectDevice
        } else {
            SourceFetchStrategy::HostBuffer
        };
        SourceFetchStrategyDecision { strategy, reason, gds_eligibility, substrate: self.substrate }
    }
}

impl Default for SourceFetchConfig {
    fn default() -> Self {
        Self {
            substrate: SourceFetchSubstrate::Auto,
            host_fetch_budget_bytes: None,
            host_fetch_max_concurrent_ops: DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
            gds_max_concurrent_ops: DEFAULT_GDS_MAX_CONCURRENT_OPS,
            gds_min_chunk_bytes: DEFAULT_GDS_MIN_CHUNK_BYTES,
        }
    }
}

/// Returns the default service-wide host-buffer fetch budget.
///
/// Linux hosts use 25% of `MemTotal` from `/proc/meminfo`. Other hosts, or
/// hosts where memory detection fails, use [`DEFAULT_HOST_FETCH_BUDGET_BYTES`].
pub fn default_host_fetch_budget_bytes() -> u64 {
    default_host_fetch_budget_bytes_from_physical_memory(detect_physical_memory_bytes())
}

fn default_host_fetch_budget_bytes_from_physical_memory(physical_memory_bytes: Option<u64>) -> u64 {
    physical_memory_bytes
        .and_then(|bytes| bytes.checked_div(4))
        .map(|bytes| bytes.min(DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES))
        .filter(|bytes| *bytes > 0)
        .unwrap_or(DEFAULT_HOST_FETCH_BUDGET_BYTES)
}

fn detect_physical_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[derive(Debug)]
struct HostFetchBudgetInner {
    budget_bytes: u64,
    state: Mutex<HostFetchBudgetState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct HostFetchBudgetState {
    in_flight_bytes: u64,
}

/// Service-wide host-memory budget for object-store source fetches.
///
/// The budget accounts compressed Parquet bytes held between `SourceFetch`
/// completion and `SourceDecode` consumption. It is shared by all cloned
/// execution contexts that carry the same budget handle.
#[derive(Debug, Clone)]
pub struct HostFetchBudget {
    inner: Arc<HostFetchBudgetInner>,
}

impl HostFetchBudget {
    /// Creates a host fetch budget with a positive byte capacity.
    pub fn new(budget_bytes: u64) -> Result<Self> {
        validate_host_fetch_budget_bytes(budget_bytes)?;
        Ok(Self {
            inner: Arc::new(HostFetchBudgetInner {
                budget_bytes,
                state: Mutex::new(HostFetchBudgetState::default()),
                available: Condvar::new(),
            }),
        })
    }

    /// Byte capacity shared by all in-flight host-buffer fetches.
    pub fn budget_bytes(&self) -> u64 {
        self.inner.budget_bytes
    }

    /// Acquires a host-buffer permit, blocking while the budget is exhausted.
    pub fn acquire(&self, bytes: u64) -> Result<HostFetchPermit> {
        validate_host_fetch_request_bytes(bytes, self.budget_bytes())?;
        let mut state = self.lock_state("cudf_query_engine.host_fetch_budget.acquire")?;
        while !host_fetch_bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "cudf_query_engine.host_fetch_budget.acquire",
                    "host fetch budget mutex was poisoned while waiting for capacity",
                )
            })?;
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
        Ok(HostFetchPermit { budget: self.clone(), bytes })
    }

    fn grow_permit(&self, permit: &mut HostFetchPermit, new_bytes: u64) -> Result<()> {
        if new_bytes <= permit.bytes {
            return Ok(());
        }
        let delta = new_bytes - permit.bytes;
        validate_host_fetch_request_bytes(new_bytes, self.budget_bytes())?;
        let mut state = self.lock_state("cudf_query_engine.host_fetch_budget.grow")?;
        while !host_fetch_bytes_fit(state.in_flight_bytes, delta, self.budget_bytes()) {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "cudf_query_engine.host_fetch_budget.grow",
                    "host fetch budget mutex was poisoned while waiting for growth capacity",
                )
            })?;
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(delta);
        permit.bytes = new_bytes;
        Ok(())
    }

    fn release(&self, bytes: u64) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.in_flight_bytes = state.in_flight_bytes.saturating_sub(bytes);
                self.inner.available.notify_all();
            },
            Err(_error) => {
                eprintln!("[cudf-query-engine] warning: failed to release poisoned host fetch budget state");
            },
        }
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, HostFetchBudgetState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "host fetch budget mutex was poisoned")
        })
    }
}

impl Default for HostFetchBudget {
    fn default() -> Self {
        Self::new(default_host_fetch_budget_bytes()).expect("default host fetch budget is non-zero")
    }
}

/// RAII permit for bytes charged to a [`HostFetchBudget`].
#[derive(Debug)]
pub struct HostFetchPermit {
    budget: HostFetchBudget,
    bytes: u64,
}

impl HostFetchPermit {
    /// Number of bytes currently charged to the budget.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Grows the permit after a fetch discovers a larger actual payload.
    pub fn grow_to(&mut self, new_bytes: u64) -> Result<()> {
        let budget = self.budget.clone();
        budget.grow_permit(self, new_bytes)
    }
}

impl Drop for HostFetchPermit {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

fn host_fetch_bytes_fit(in_flight_bytes: u64, requested_bytes: u64, budget_bytes: u64) -> bool {
    in_flight_bytes.saturating_add(requested_bytes) <= budget_bytes
}

fn validate_host_fetch_budget_bytes(budget_bytes: u64) -> Result<()> {
    if budget_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.host_fetch_budget.validate",
            "host fetch budget bytes must be greater than zero",
        )
        .with_fact("field", "host_fetch_budget_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn validate_host_fetch_request_bytes(requested_bytes: u64, budget_bytes: u64) -> Result<()> {
    if requested_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.host_fetch_budget.request",
            "host fetch budget request bytes must be greater than zero",
        )
        .with_fact("field", "host_fetch_request_bytes")
        .with_fact("reason", "zero"));
    }
    if requested_bytes > budget_bytes {
        return Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "cudf_query_engine.host_fetch_budget.request",
            "object-store source fetch exceeds the configured host fetch budget",
        )
        .with_fact("requested_bytes", requested_bytes)
        .with_fact("budget_bytes", budget_bytes));
    }
    Ok(())
}

#[derive(Debug)]
struct SourceFetchConcurrencyGateInner {
    max_concurrent_ops: usize,
    state: Mutex<SourceFetchConcurrencyGateState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct SourceFetchConcurrencyGateState {
    in_flight_ops: usize,
}

#[derive(Debug, Clone)]
struct SourceFetchConcurrencyGate {
    inner: Arc<SourceFetchConcurrencyGateInner>,
}

impl SourceFetchConcurrencyGate {
    fn new(max_concurrent_ops: usize) -> Result<Self> {
        validate_host_fetch_max_concurrent_ops(max_concurrent_ops)?;
        Ok(Self {
            inner: Arc::new(SourceFetchConcurrencyGateInner {
                max_concurrent_ops,
                state: Mutex::new(SourceFetchConcurrencyGateState::default()),
                available: Condvar::new(),
            }),
        })
    }

    fn max_concurrent_ops(&self) -> usize {
        self.inner.max_concurrent_ops
    }

    fn acquire(&self) -> Result<SourceFetchOperationPermit> {
        let mut state = self.lock_state("cudf_query_engine.source_fetch_concurrency.acquire")?;
        while state.in_flight_ops >= self.max_concurrent_ops() {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "cudf_query_engine.source_fetch_concurrency.acquire",
                    "source fetch concurrency mutex was poisoned while waiting for a permit",
                )
            })?;
        }
        state.in_flight_ops = state.in_flight_ops.saturating_add(1);
        Ok(SourceFetchOperationPermit { gate: self.clone() })
    }

    fn release(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.in_flight_ops = state.in_flight_ops.saturating_sub(1);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[cudf-query-engine] warning: failed to release poisoned source fetch concurrency state");
            },
        }
    }

    fn lock_state(
        &self,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, SourceFetchConcurrencyGateState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "source fetch concurrency mutex was poisoned")
        })
    }
}

impl Default for SourceFetchConcurrencyGate {
    fn default() -> Self {
        Self::new(DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS).expect("default host fetch concurrency is non-zero")
    }
}

pub(crate) struct SourceFetchOperationPermit {
    gate: SourceFetchConcurrencyGate,
}

impl Drop for SourceFetchOperationPermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

fn validate_host_fetch_max_concurrent_ops(max_concurrent_ops: usize) -> Result<()> {
    if max_concurrent_ops == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch_concurrency.validate",
            "host fetch max concurrent operations must be greater than zero",
        )
        .with_fact("field", "host_fetch_max_concurrent_ops")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

#[derive(Debug)]
#[allow(dead_code)]
struct GdsConcurrencyGateInner {
    max_concurrent_ops: usize,
    state: Mutex<GdsConcurrencyGateState>,
    available: Condvar,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct GdsConcurrencyGateState {
    in_flight_ops: usize,
}

#[derive(Debug, Clone)]
struct GdsConcurrencyGate {
    inner: Arc<GdsConcurrencyGateInner>,
}

#[allow(dead_code)]
impl GdsConcurrencyGate {
    fn new(max_concurrent_ops: usize) -> Result<Self> {
        validate_gds_max_concurrent_ops(max_concurrent_ops)?;
        Ok(Self {
            inner: Arc::new(GdsConcurrencyGateInner {
                max_concurrent_ops,
                state: Mutex::new(GdsConcurrencyGateState::default()),
                available: Condvar::new(),
            }),
        })
    }

    fn max_concurrent_ops(&self) -> usize {
        self.inner.max_concurrent_ops
    }

    fn acquire(&self) -> Result<GdsOperationPermit> {
        let mut state = self.lock_state("cudf_query_engine.gds_concurrency.acquire")?;
        while state.in_flight_ops >= self.max_concurrent_ops() {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "cudf_query_engine.gds_concurrency.acquire",
                    "GDS concurrency mutex was poisoned while waiting for a permit",
                )
            })?;
        }
        state.in_flight_ops = state.in_flight_ops.saturating_add(1);
        Ok(GdsOperationPermit { gate: self.clone() })
    }

    fn release(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.in_flight_ops = state.in_flight_ops.saturating_sub(1);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[cudf-query-engine] warning: failed to release poisoned GDS concurrency state");
            },
        }
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, GdsConcurrencyGateState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "GDS concurrency mutex was poisoned")
        })
    }
}

impl Default for GdsConcurrencyGate {
    fn default() -> Self {
        Self::new(DEFAULT_GDS_MAX_CONCURRENT_OPS).expect("default GDS concurrency is non-zero")
    }
}

#[allow(dead_code)]
pub(crate) struct GdsOperationPermit {
    gate: GdsConcurrencyGate,
}

impl Drop for GdsOperationPermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

fn validate_gds_max_concurrent_ops(max_concurrent_ops: usize) -> Result<()> {
    if max_concurrent_ops == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gds_concurrency.validate",
            "GDS max concurrent operations must be greater than zero",
        )
        .with_fact("field", "gds_max_concurrent_ops")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn validate_gds_min_chunk_bytes(gds_min_chunk_bytes: u64) -> Result<()> {
    if gds_min_chunk_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_fetch_config.validate",
            "GDS minimum chunk bytes must be greater than zero",
        )
        .with_fact("field", "gds_min_chunk_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

/// Fallback and upper bound for the default service stream pool size.
///
/// The pool creates streams lazily, so this is a concurrency cap rather than a
/// startup allocation count. [`default_service_stream_pool_size_for_device`]
/// uses this value as the fallback when device SM count detection is
/// unavailable and as the upper bound when detection succeeds.
pub const DEFAULT_SERVICE_STREAM_POOL_SIZE: usize = 64;

/// Returns the default service stream pool size for a CUDA device.
///
/// The default is `min(device_multiprocessor_count * 2, 64)`. If the
/// multiprocessor count cannot be detected, this returns
/// [`DEFAULT_SERVICE_STREAM_POOL_SIZE`]. The detected value is cached per
/// device ordinal for the process lifetime.
pub fn default_service_stream_pool_size_for_device(device: cudf::DeviceId) -> usize {
    static DEFAULT_SIZES: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let sizes = DEFAULT_SIZES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sizes = match sizes.lock() {
        Ok(sizes) => sizes,
        Err(_error) => {
            return default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device));
        },
    };
    *sizes
        .entry(device.ordinal())
        .or_insert_with(|| default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device)))
}

fn default_service_stream_pool_size_from_sm_count(sm_count: Option<usize>) -> usize {
    sm_count
        .filter(|count| *count > 0)
        .and_then(|count| count.checked_mul(2))
        .map(|size| size.min(DEFAULT_SERVICE_STREAM_POOL_SIZE))
        .unwrap_or(DEFAULT_SERVICE_STREAM_POOL_SIZE)
}

// TODO: make nvml a dependent of cudf-rs?
fn detect_device_multiprocessor_count(device: cudf::DeviceId) -> Option<usize> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device_index = u32::try_from(device.ordinal()).ok()?;
    let attributes = nvml.device_by_index(device_index).ok()?.attributes().ok()?;
    nonzero_sm_count(attributes.multiprocessor_count)
}

fn nonzero_sm_count(count: u32) -> Option<usize> {
    (count > 0).then(|| usize::try_from(count).ok()).flatten()
}

#[derive(Debug)]
struct ServiceStreamPoolInner {
    device: cudf::DeviceId,
    pool_size: usize,
    state: Mutex<ServiceStreamPoolState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct ServiceStreamPoolState {
    idle: Vec<Arc<cudf::Runtime>>,
    created: usize,
}

/// Service-level CUDA stream pool for short-lived native worker tasks.
///
/// The pool owns a bounded set of same-device cuDF runtimes. Each runtime
/// carries one CUDA stream and the process-local device memory manager. Streams
/// are created lazily up to `pool_size`; dropping a [`StreamLease`] returns the
/// runtime to the pool for reuse. This avoids creating a CUDA stream for every
/// bounded source-read worker while preserving the existing reservation-aware
/// allocator contract.
#[derive(Debug, Clone)]
pub struct ServiceStreamPool {
    inner: Arc<ServiceStreamPoolInner>,
}

impl ServiceStreamPool {
    /// Create a stream pool for `device`.
    ///
    /// `pool_size` is the maximum number of simultaneously leased streams.
    /// Runtime/stream creation is lazy: construction records the limit but does
    /// not allocate CUDA streams immediately.
    pub fn new(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        Ok(Self {
            inner: Arc::new(ServiceStreamPoolInner {
                device,
                pool_size,
                state: Mutex::new(ServiceStreamPoolState::default()),
                available: Condvar::new(),
            }),
        })
    }

    /// Return the process-global default pool for `device` and `pool_size`.
    ///
    /// This is the fallback used by native execution when a service has not
    /// injected its own pool yet. Pools are keyed by `(device ordinal,
    /// pool_size)` so differently sized service tests do not silently share a
    /// capacity contract.
    pub fn for_device(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        static POOLS: OnceLock<Mutex<HashMap<(usize, usize), ServiceStreamPool>>> = OnceLock::new();
        let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut pools = pools.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.service_stream_pool.registry",
                "service stream pool registry mutex was poisoned",
            )
        })?;
        let key = (device.ordinal(), pool_size);
        if let Some(pool) = pools.get(&key) {
            return Ok(pool.clone());
        }
        let pool = Self::new(device, pool_size)?;
        pools.insert(key, pool.clone());
        Ok(pool)
    }

    /// Return the process-global default-sized pool for `device`.
    pub fn default_for_device(device: cudf::DeviceId) -> Result<Self> {
        Self::for_device(device, default_service_stream_pool_size_for_device(device))
    }

    /// CUDA device served by this pool.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Maximum number of simultaneously leased streams.
    pub fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    /// Acquire a CUDA stream/runtime lease, blocking when all pool slots are busy.
    pub fn acquire(&self) -> Result<StreamLease> {
        enum AcquireAction {
            Reuse(Arc<cudf::Runtime>),
            Create,
        }

        let action = loop {
            let mut state = self.lock_state("cudf_query_engine.service_stream_pool.acquire")?;
            if let Some(runtime) = state.idle.pop() {
                break AcquireAction::Reuse(runtime);
            }
            if state.created < self.pool_size() {
                state.created += 1;
                break AcquireAction::Create;
            }
            let _guard = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "cudf_query_engine.service_stream_pool.acquire",
                    "service stream pool mutex was poisoned while waiting for a stream",
                )
            })?;
        };

        let runtime = match action {
            AcquireAction::Reuse(runtime) => runtime,
            AcquireAction::Create => match self.build_runtime() {
                Ok(runtime) => Arc::new(runtime),
                Err(error) => {
                    self.discard_created_slot();
                    return Err(error);
                },
            },
        };
        Ok(StreamLease { pool: self.clone(), runtime: Some(runtime) })
    }

    fn build_runtime(&self) -> Result<cudf::Runtime> {
        cudf::Runtime::builder().with_device(self.device()).with_device_memory_manager().build().map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.service_stream_pool.runtime",
                "failed to initialize service stream pool runtime",
                error,
            )
        })
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, ServiceStreamPoolState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "service stream pool mutex was poisoned")
        })
    }

    fn discard_created_slot(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.created = state.created.saturating_sub(1);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[cudf-query-engine] warning: failed to update poisoned service stream pool state");
            },
        }
    }

    fn release(&self, runtime: Arc<cudf::Runtime>) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.idle.push(runtime);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[cudf-query-engine] warning: failed to return runtime to poisoned service stream pool");
            },
        }
    }
}

/// RAII handle for a runtime borrowed from a [`ServiceStreamPool`].
///
/// Dropping the lease synchronizes the runtime stream and returns it to the
/// pool. Synchronization keeps source-read output safe to consume on another
/// runtime before the stream is reused by an unrelated worker.
#[derive(Debug)]
pub struct StreamLease {
    pool: ServiceStreamPool,
    runtime: Option<Arc<cudf::Runtime>>,
}

impl StreamLease {
    /// Borrow the leased cuDF runtime.
    pub fn runtime(&self) -> &cudf::Runtime {
        self.runtime.as_ref().expect("stream lease runtime is present").as_ref()
    }

    /// Clone the shared runtime handle for a short-lived execution context.
    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(self.runtime.as_ref().expect("stream lease runtime is present"))
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        if let Err(error) = runtime.synchronize() {
            eprintln!("[cudf-query-engine] warning: failed to synchronize service stream before return: {error}");
        }
        self.pool.release(runtime);
    }
}

pub(crate) struct SourceReadStreamLease {
    context: ExecutionContext,
    _stream_lease: StreamLease,
}

impl SourceReadStreamLease {
    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

fn validate_service_stream_pool_size(pool_size: usize) -> Result<()> {
    if pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.service_stream_pool.validate",
            "service stream pool size must be greater than zero",
        )
        .with_fact("field", "service_stream_pool_size")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct S3ObjectStoreKey {
    pub(crate) bucket: String,
    pub(crate) region: String,
}

pub(crate) struct ObjectStoreFetchResult {
    pub(crate) bytes: Result<Bytes>,
    pub(crate) elapsed: Duration,
}

struct SourceObjectStoreRegistry {
    stores: Mutex<HashMap<S3ObjectStoreKey, Arc<dyn object_store::ObjectStore>>>,
}

impl SourceObjectStoreRegistry {
    fn new() -> Self {
        Self { stores: Mutex::new(HashMap::new()) }
    }

    fn s3_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn object_store::ObjectStore>> {
        let mut stores = self.stores.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_object_store.registry",
                "source object-store registry mutex was poisoned",
            )
        })?;
        if let Some(store) = stores.get(&key) {
            return Ok(Arc::clone(store));
        }
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(key.bucket.as_str())
            .with_region(key.region.as_str())
            .build()
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "cudf_query_engine.source_object_store.s3",
                    "failed to build S3 object-store client for native source fetch",
                )
                .with_fact("bucket", key.bucket.as_str())
                .with_fact("region", key.region.as_str())
                .with_source(error)
            })?;
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(store);
        stores.insert(key, Arc::clone(&store));
        Ok(store)
    }
}

impl fmt::Debug for SourceObjectStoreRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cached_store_count = self.stores.lock().map_or(0, |stores| stores.len());
        f.debug_struct("SourceObjectStoreRegistry").field("cached_store_count", &cached_store_count).finish()
    }
}

fn default_source_fetch_runtime() -> Arc<tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    Arc::clone(RUNTIME.get_or_init(|| {
        Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("cudf-source-fetch")
                .enable_all()
                .build()
                .expect("source fetch runtime should initialize"),
        )
    }))
}

/// Immutable execution configuration shared by native query execution.
///
/// The context owns or shares the [`cudf::Runtime`] used for all device work in
/// a query. The runtime determines the CUDA device, stream, and memory resource;
/// the query engine layers its own chunk-planning, reservation, retry, and
/// boundary policies on top of that runtime. Cloning the context is cheap and
/// keeps the same runtime/cache handles.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Shared cudf runtime used for GPU work.
    ///
    /// Runtime ownership is explicit: callers may inject an `Arc` to share a
    /// stream, allocator, and device across query plans, or allow the builder to
    /// create a fresh runtime from the configured device/memory-resource policy.
    runtime: Arc<cudf::Runtime>,
    /// Runtime allocator construction policy recorded for diagnostics.
    ///
    /// This does not replace the allocator owned by an injected runtime. When an
    /// external runtime is supplied, the runtime's existing memory resource is
    /// authoritative and this field records [`MemoryResourcePolicy::ExternalRuntime`].
    memory_policy: MemoryResourcePolicy,
    /// Engine-level memory control contract for bounded execution.
    ///
    /// Bounded fields drive chunk planning, reservation admission, and
    /// retry/fan-in decisions. Source chunk byte and row-group settings are
    /// planning thresholds where row-group metadata permits exact splitting.
    memory_control_policy: MemoryControlPolicy,
    /// Execution boundary contract used by source reads, build/probe reuse, and
    /// result materialization.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source scheduling policy for bounded local Parquet reads.
    source_scheduler_policy: SourceSchedulerPolicy,
    /// Service-level source IO substrate and gate configuration.
    source_fetch_config: SourceFetchConfig,
    /// Service-level pool used by bounded source-read workers.
    service_stream_pool: ServiceStreamPool,
    /// Service-wide host-memory budget for object-store fetch buffers.
    host_fetch_budget: HostFetchBudget,
    /// Service-wide operation-count gate for object-store fetches.
    source_fetch_concurrency: SourceFetchConcurrencyGate,
    /// Service-wide operation-count gate for future GDS direct-device reads.
    gds_concurrency: GdsConcurrencyGate,
    /// Service-scoped async runtime used for object-store source fetches.
    source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    /// Cached object-store clients used by source fetch workers.
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    /// Deployment-level IO capability probes used for readiness/reporting.
    deployment_io_capability: DeploymentIoCapability,
    /// Cooperative cancellation flag observed at native execution boundaries.
    cancellation: ExecutionCancellation,
    /// Selected native bounded-pipeline execution mode.
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Optional adaptive-only memory history store.
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    /// Optional durable GPU hash-join build cache.
    ///
    /// Required when the boundary policy asks for durable build
    /// materialization. Query-scoped build materialization does not use this
    /// cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    /// Optional durable GPU local-Parquet source cache.
    ///
    /// Bounded execution charges retained cache bytes to the reservation ledger
    /// before admitting source or operator work.
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
}

impl ExecutionContext {
    /// Starts a builder for an [`ExecutionContext`].
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around an owned cudf runtime.
    ///
    /// The resulting context records [`MemoryResourcePolicy::ExternalRuntime`]
    /// because the runtime, including its allocator and stream, has already been
    /// constructed by the caller.
    pub fn from_runtime(runtime: cudf::Runtime) -> Self {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Self {
        let service_stream_pool =
            ServiceStreamPool::new(runtime.device(), default_service_stream_pool_size_for_device(runtime.device()))
                .expect("default service stream pool size is non-zero");
        let source_fetch_config = SourceFetchConfig::default();
        Self {
            runtime,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_fetch_config,
            service_stream_pool,
            host_fetch_budget: HostFetchBudget::default(),
            source_fetch_concurrency: SourceFetchConcurrencyGate::default(),
            gds_concurrency: GdsConcurrencyGate::default(),
            source_fetch_runtime: default_source_fetch_runtime(),
            source_object_stores: Arc::new(SourceObjectStoreRegistry::new()),
            deployment_io_capability: DeploymentIoCapability::probe_current_host(),
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
        }
    }

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    /// Returns the CUDA stream owned by the runtime.
    pub fn stream(&self) -> &cudf::runtime::CudaStream {
        self.runtime.stream()
    }

    /// Returns the device memory resource owned by the runtime.
    pub fn memory_resource(&self) -> &cudf::core::memory::DeviceMemoryResource {
        self.runtime.memory_resource()
    }

    /// Returns a shared handle to the runtime.
    ///
    /// The caller receives the same runtime object, not a new runtime with
    /// copied configuration.
    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(&self.runtime)
    }

    /// Acquires a same-device leased context for a scheduled source-read worker.
    pub(crate) fn acquire_source_read_lease(&self) -> Result<SourceReadStreamLease> {
        let stream_lease = self.service_stream_pool.acquire()?;
        let context = self.with_stream_lease(&stream_lease)?;
        Ok(SourceReadStreamLease { context, _stream_lease: stream_lease })
    }

    /// Builds a short-lived execution context from a borrowed stream lease.
    ///
    /// The returned context shares this context's query policies, caches, and
    /// cancellation flag, but executes GPU work on the leased runtime. The
    /// caller must keep `stream_lease` alive until all work submitted through
    /// the returned context has completed.
    pub fn with_stream_lease(&self, stream_lease: &StreamLease) -> Result<Self> {
        validate_service_stream_pool_device(&stream_lease.pool, self.runtime.device())?;
        Ok(Self {
            runtime: stream_lease.shared_runtime(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_fetch_config: self.source_fetch_config.clone(),
            service_stream_pool: self.service_stream_pool.clone(),
            host_fetch_budget: self.host_fetch_budget.clone(),
            source_fetch_concurrency: self.source_fetch_concurrency.clone(),
            gds_concurrency: self.gds_concurrency.clone(),
            source_fetch_runtime: Arc::clone(&self.source_fetch_runtime),
            source_object_stores: Arc::clone(&self.source_object_stores),
            deployment_io_capability: self.deployment_io_capability.clone(),
            cancellation: self.cancellation.clone(),
            native_pipeline_execution_mode: self.native_pipeline_execution_mode,
            adaptive_memory_history: self.adaptive_memory_history.as_ref().map(Arc::clone),
            hash_join_build_cache: self.hash_join_build_cache.as_ref().map(Arc::clone),
            local_parquet_source_cache: self.local_parquet_source_cache.as_ref().map(Arc::clone),
        })
    }

    /// Returns the cooperative cancellation flag for this execution.
    pub fn cancellation(&self) -> &ExecutionCancellation {
        &self.cancellation
    }

    /// Fails with a structured cancellation error if the execution was cancelled.
    pub fn check_cancellation(&self, operation: &'static str) -> Result<()> {
        self.cancellation.check(operation)
    }

    /// Reports the effective device placement derived from the runtime.
    pub fn device_placement(&self) -> DevicePlacement {
        DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
    }

    /// Reports the default device memory space used for bounded reservation accounting.
    pub fn device_memory_space(&self) -> MemorySpaceId {
        MemorySpaceId::Device { device_ordinal: self.runtime.device().ordinal() }
    }

    /// Returns the memory-resource construction policy recorded for this context.
    pub fn memory_policy(&self) -> MemoryResourcePolicy {
        self.memory_policy
    }

    /// Returns the engine-level memory control policy.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    /// Returns the execution boundary policy.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Returns source scheduling controls for bounded source execution.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Returns the service-level source IO configuration.
    pub fn source_fetch_config(&self) -> &SourceFetchConfig {
        &self.source_fetch_config
    }

    /// Returns the effective source IO substrate preference.
    pub fn source_fetch_substrate(&self) -> SourceFetchSubstrate {
        self.source_fetch_config.substrate
    }

    /// Returns the service-level CUDA stream pool used by source-read workers.
    pub fn service_stream_pool(&self) -> ServiceStreamPool {
        self.service_stream_pool.clone()
    }

    /// Returns the service-wide host-memory budget for object-store source fetches.
    pub fn host_fetch_budget(&self) -> HostFetchBudget {
        self.host_fetch_budget.clone()
    }

    /// Maximum object-store SourceFetch operations admitted concurrently.
    pub fn host_fetch_max_concurrent_ops(&self) -> usize {
        self.source_fetch_concurrency.max_concurrent_ops()
    }

    /// Maximum future GDS DMA operations admitted concurrently.
    pub fn gds_max_concurrent_ops(&self) -> usize {
        self.gds_concurrency.max_concurrent_ops()
    }

    /// Per-chunk threshold below which future GDS-eligible reads use host-buffer IO.
    pub fn gds_min_chunk_bytes(&self) -> u64 {
        self.source_fetch_config.gds_min_chunk_bytes
    }

    /// Selects the source IO strategy for a location and optional chunk-size estimate.
    pub fn source_fetch_strategy_for_location(
        &self,
        source_location: &str,
        chunk_bytes: Option<u64>,
    ) -> SourceFetchStrategyDecision {
        self.source_fetch_config.decide_source_fetch_strategy(
            &self.deployment_io_capability,
            source_location,
            chunk_bytes,
        )
    }

    pub(crate) fn acquire_source_fetch_operation(&self) -> Result<SourceFetchOperationPermit> {
        self.source_fetch_concurrency.acquire()
    }

    pub(crate) fn s3_object_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn object_store::ObjectStore>> {
        self.source_object_stores.s3_store(key)
    }

    pub(crate) fn spawn_object_store_fetch(
        &self,
        store: Arc<dyn object_store::ObjectStore>,
        location: object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        Self::spawn_object_store_fetch_on_runtime(self.source_fetch_runtime.as_ref(), store, location, on_result)
    }

    /// Returns deployment-level IO capability diagnostics.
    pub fn deployment_io_capability(&self) -> &DeploymentIoCapability {
        &self.deployment_io_capability
    }

    #[cfg(test)]
    fn fetch_object_store_bytes_on_runtime(
        source_fetch_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn object_store::ObjectStore>,
        location: object_store::path::Path,
    ) -> Result<Bytes> {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self::spawn_object_store_fetch_on_runtime(source_fetch_runtime, store, location, move |result| {
            let _ = sender.send(result.bytes);
        });
        receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_fetch.receive",
                "object-store source fetch worker stopped before returning a result",
            )
            .with_source(error)
        })?
    }

    fn spawn_object_store_fetch_on_runtime(
        source_fetch_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn object_store::ObjectStore>,
        location: object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        source_fetch_runtime.spawn(async move {
            let timer = Instant::now();
            let bytes = async move {
                let result = store.get(&location).await?;
                result.bytes().await
            }
            .await
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "cudf_query_engine.source_fetch.object_store",
                    "failed to fetch native Parquet source bytes through object_store",
                )
                .with_source(error)
            });
            on_result(ObjectStoreFetchResult { bytes, elapsed: timer.elapsed() });
        });
    }

    /// Returns the selected bounded-pipeline execution mode.
    pub fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.native_pipeline_execution_mode
    }

    /// Returns the adaptive-only memory history store, when configured.
    pub fn adaptive_memory_history(&self) -> Option<Arc<AdaptiveMemoryHistory>> {
        self.adaptive_memory_history.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU hash-join build cache, when configured.
    ///
    /// A clone of the `Arc` is returned so callers can retain the cache without
    /// borrowing the context.
    pub fn hash_join_build_cache(&self) -> Option<Arc<crate::exec::GpuHashJoinBuildCache>> {
        self.hash_join_build_cache.as_ref().map(Arc::clone)
    }

    /// Returns the durable GPU local-Parquet source cache, when configured.
    pub fn local_parquet_source_cache(&self) -> Option<Arc<crate::exec::GpuLocalParquetSourceCache>> {
        self.local_parquet_source_cache.as_ref().map(Arc::clone)
    }
}

/// Builder for [`ExecutionContext`].
///
/// The builder separates runtime construction policy from execution policy. If
/// a runtime is supplied, it is used as-is and `device_ordinal` /
/// `memory_policy` no longer construct CUDA state. If no runtime is supplied,
/// the builder creates a cudf runtime from the requested device ordinal and
/// memory-resource policy.
#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    /// Prebuilt runtime to use instead of constructing one.
    runtime: Option<Arc<cudf::Runtime>>,
    /// CUDA device ordinal used only when this builder constructs the runtime.
    device_ordinal: Option<usize>,
    /// Memory resource used only when this builder constructs the runtime.
    memory_policy: MemoryResourcePolicy,
    /// Engine-level bounded/unbounded memory contract.
    memory_control_policy: MemoryControlPolicy,
    /// Source/build/probe/result boundary contract.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source read scheduling contract.
    source_scheduler_policy: SourceSchedulerPolicy,
    /// Optional service-level source IO configuration.
    source_fetch_config: Option<SourceFetchConfig>,
    /// Optional caller-owned service stream pool.
    service_stream_pool: Option<ServiceStreamPool>,
    /// Pool size for the default process-global service stream pool.
    service_stream_pool_size: Option<usize>,
    /// Optional caller-owned host fetch budget.
    host_fetch_budget: Option<HostFetchBudget>,
    /// Host fetch budget size used when constructing the default budget.
    host_fetch_budget_bytes: Option<u64>,
    /// Max object-store SourceFetch operations allowed concurrently.
    host_fetch_max_concurrent_ops: Option<usize>,
    /// Optional caller-supplied deployment IO capability diagnostic.
    deployment_io_capability: Option<DeploymentIoCapability>,
    /// Cooperative cancellation flag observed by native execution.
    cancellation: ExecutionCancellation,
    /// Native bounded-pipeline execution mode.
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Optional adaptive-only memory history store.
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    /// Optional durable hash-join build cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
    /// Optional durable local-Parquet source cache.
    local_parquet_source_cache: Option<Arc<crate::exec::GpuLocalParquetSourceCache>>,
}

impl ExecutionContextBuilder {
    /// Creates a builder with default runtime and execution policies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses an already-constructed cudf runtime.
    ///
    /// This marks the memory-resource policy as
    /// [`MemoryResourcePolicy::ExternalRuntime`] because allocator/device setup
    /// has happened outside the query engine.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Uses an already-shared cudf runtime.
    ///
    /// Prefer this over building separate runtimes when an adapter needs one
    /// CUDA stream, allocator, or device policy across multiple executions.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Selects the CUDA device ordinal for runtime construction.
    ///
    /// Ignored when a runtime is supplied with [`Self::with_runtime`] or
    /// [`Self::with_shared_runtime`].
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Selects the memory resource used when constructing a runtime.
    ///
    /// Ignored for an injected runtime except for the recorded
    /// `ExternalRuntime` diagnostic state.
    pub fn with_memory_policy(mut self, memory_policy: MemoryResourcePolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    /// Sets the engine-level memory control contract.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Sets the execution boundary policy.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Sets bounded source scheduling controls.
    pub fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    /// Sets the service-level source IO configuration.
    ///
    /// Existing per-field builder setters for host-buffer budget and
    /// object-store operation count override the corresponding values in this
    /// config so older call sites can keep using their narrower knobs.
    pub fn with_source_fetch_config(mut self, source_fetch_config: SourceFetchConfig) -> Self {
        self.source_fetch_config = Some(source_fetch_config);
        self
    }

    /// Uses a caller-owned service stream pool for bounded source-read workers.
    pub fn with_service_stream_pool(mut self, service_stream_pool: ServiceStreamPool) -> Self {
        self.service_stream_pool = Some(service_stream_pool);
        self
    }

    /// Sets the size of the default process-global service stream pool.
    ///
    /// Ignored when a pool is supplied with [`Self::with_service_stream_pool`].
    pub fn with_service_stream_pool_size(mut self, pool_size: usize) -> Self {
        self.service_stream_pool_size = Some(pool_size);
        self
    }

    /// Uses a caller-owned host fetch budget for object-store source reads.
    pub fn with_host_fetch_budget(mut self, host_fetch_budget: HostFetchBudget) -> Self {
        self.host_fetch_budget = Some(host_fetch_budget);
        self
    }

    /// Sets the size of the default host fetch budget.
    ///
    /// Ignored when a budget is supplied with [`Self::with_host_fetch_budget`].
    pub fn with_host_fetch_budget_bytes(mut self, budget_bytes: u64) -> Self {
        self.host_fetch_budget_bytes = Some(budget_bytes);
        self
    }

    /// Sets the service-wide object-store SourceFetch operation cap.
    pub fn with_host_fetch_max_concurrent_ops(mut self, max_concurrent_ops: usize) -> Self {
        self.host_fetch_max_concurrent_ops = Some(max_concurrent_ops);
        self
    }

    /// Supplies deployment IO capability diagnostics instead of probing the current host.
    pub fn with_deployment_io_capability(mut self, capability: DeploymentIoCapability) -> Self {
        self.deployment_io_capability = Some(capability);
        self
    }

    /// Sets the cooperative cancellation flag for native execution.
    pub fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Selects the native bounded-pipeline execution mode.
    pub fn with_native_pipeline_execution_mode(mut self, mode: NativePipelineExecutionMode) -> Self {
        self.native_pipeline_execution_mode = mode;
        self
    }

    /// Provides the adaptive-only memory history store.
    pub fn with_adaptive_memory_history(mut self, memory_history: Arc<AdaptiveMemoryHistory>) -> Self {
        self.adaptive_memory_history = Some(memory_history);
        self
    }

    /// Provides the cache required by durable GPU hash-join materialization.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<crate::exec::GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Provides the durable GPU local-Parquet source cache.
    ///
    /// Bounded execution charges retained source frames through the reservation
    /// ledger before using cached frames or retaining newly read frames.
    pub fn with_local_parquet_source_cache(mut self, cache: Arc<crate::exec::GpuLocalParquetSourceCache>) -> Self {
        self.local_parquet_source_cache = Some(cache);
        self
    }

    /// Validates policies, constructs or records the runtime, and returns a context.
    ///
    /// Validation failures carry report-facing facts such as
    /// `missing_hash_join_build_cache`; do not churn those fact keys casually.
    pub fn build(self) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;
        self.boundary_policy.validate(self.hash_join_build_cache.is_some())?;
        self.source_scheduler_policy.validate()?;
        if let Some(service_stream_pool_size) = self.service_stream_pool_size {
            validate_service_stream_pool_size(service_stream_pool_size)?;
        }
        let deployment_io_capability =
            self.deployment_io_capability.unwrap_or_else(DeploymentIoCapability::probe_current_host);
        let mut source_fetch_config = self.source_fetch_config.unwrap_or_default();
        if let Some(host_fetch_budget_bytes) = self.host_fetch_budget_bytes {
            source_fetch_config.host_fetch_budget_bytes = Some(host_fetch_budget_bytes);
        }
        if let Some(max_concurrent_ops) = self.host_fetch_max_concurrent_ops {
            source_fetch_config.host_fetch_max_concurrent_ops = max_concurrent_ops;
        }
        source_fetch_config.validate_for_deployment(&deployment_io_capability)?;
        let host_fetch_budget = match self.host_fetch_budget {
            Some(budget) => budget,
            None => HostFetchBudget::new(
                source_fetch_config.host_fetch_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
        };
        let source_fetch_concurrency =
            SourceFetchConcurrencyGate::new(source_fetch_config.host_fetch_max_concurrent_ops)?;
        let gds_concurrency = GdsConcurrencyGate::new(source_fetch_config.gds_max_concurrent_ops)?;
        let source_fetch_runtime = default_source_fetch_runtime();
        let source_object_stores = Arc::new(SourceObjectStoreRegistry::new());
        if let Some(runtime) = self.runtime {
            let service_stream_pool_size = self
                .service_stream_pool_size
                .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
            let service_stream_pool = match self.service_stream_pool {
                Some(pool) => {
                    validate_service_stream_pool_device(&pool, runtime.device())?;
                    pool
                },
                None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
            };
            return Ok(ExecutionContext {
                runtime,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
                boundary_policy: self.boundary_policy,
                source_scheduler_policy: self.source_scheduler_policy,
                source_fetch_config,
                service_stream_pool,
                host_fetch_budget,
                source_fetch_concurrency,
                gds_concurrency,
                source_fetch_runtime,
                source_object_stores,
                deployment_io_capability,
                cancellation: self.cancellation,
                native_pipeline_execution_mode: self.native_pipeline_execution_mode,
                adaptive_memory_history: self.adaptive_memory_history,
                hash_join_build_cache: self.hash_join_build_cache,
                local_parquet_source_cache: self.local_parquet_source_cache,
            });
        }

        let mut builder = cudf::Runtime::builder();
        if let Some(device_ordinal) = self.device_ordinal {
            let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
                QueryError::from_cudf("cudf_query_engine.runtime.device", "failed to resolve CUDA device", error)
            })?;
            builder = builder.with_device(device);
        }

        let memory_policy = match self.memory_policy {
            MemoryResourcePolicy::ExternalRuntime => MemoryResourcePolicy::DefaultCuda,
            other => other,
        };

        match memory_policy {
            MemoryResourcePolicy::DefaultCuda | MemoryResourcePolicy::ExternalRuntime => {},
            MemoryResourcePolicy::Pool { initial_pool_size, maximum_pool_size } => {
                builder = builder.with_pool_memory_resource(initial_pool_size, maximum_pool_size);
            },
        }

        let runtime = builder.build().map_err(|error| {
            QueryError::from_cudf("cudf_query_engine.runtime.build", "failed to initialize cudf runtime", error)
        })?;
        let service_stream_pool_size = self
            .service_stream_pool_size
            .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
        let service_stream_pool = match self.service_stream_pool {
            Some(pool) => {
                validate_service_stream_pool_device(&pool, runtime.device())?;
                pool
            },
            None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
        };
        Ok(ExecutionContext {
            runtime: Arc::new(runtime),
            memory_policy,
            memory_control_policy: self.memory_control_policy,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_fetch_config,
            service_stream_pool,
            host_fetch_budget,
            source_fetch_concurrency,
            gds_concurrency,
            source_fetch_runtime,
            source_object_stores,
            deployment_io_capability,
            cancellation: self.cancellation,
            native_pipeline_execution_mode: self.native_pipeline_execution_mode,
            adaptive_memory_history: self.adaptive_memory_history,
            hash_join_build_cache: self.hash_join_build_cache,
            local_parquet_source_cache: self.local_parquet_source_cache,
        })
    }
}

fn validate_service_stream_pool_device(pool: &ServiceStreamPool, runtime_device: cudf::DeviceId) -> Result<()> {
    if pool.device() != runtime_device {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.service_stream_pool.device",
            "service stream pool device must match the execution runtime device",
        )
        .with_fact("pool_device_ordinal", pool.device().ordinal())
        .with_fact("runtime_device_ordinal", runtime_device.ordinal()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_store_fetch_returns_bytes_from_full_get() {
        let runtime = default_source_fetch_runtime();
        let store = Arc::new(object_store::memory::InMemory::new());
        let location = object_store::path::Path::from("part-0.parquet");
        runtime.block_on(store.put(&location, b"abcdef".as_ref().into())).unwrap();

        let bytes = ExecutionContext::fetch_object_store_bytes_on_runtime(runtime.as_ref(), store, location).unwrap();

        assert_eq!(bytes.as_ref(), b"abcdef");
    }

    #[test]
    fn service_stream_pool_default_size_is_sm_aware_with_cap_and_fallback() {
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(1)), 2);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(16)), 32);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(32)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(128)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(Some(0)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
        assert_eq!(default_service_stream_pool_size_from_sm_count(None), DEFAULT_SERVICE_STREAM_POOL_SIZE);
    }

    #[test]
    fn service_stream_pool_sm_count_filters_zero_counts() {
        assert_eq!(nonzero_sm_count(142), Some(142));
        assert_eq!(nonzero_sm_count(1), Some(1));
        assert_eq!(nonzero_sm_count(0), None);
    }

    #[test]
    fn default_host_fetch_budget_is_capped() {
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(Some(128 * 1024 * 1024 * 1024)),
            DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES
        );
        assert_eq!(
            default_host_fetch_budget_bytes_from_physical_memory(Some(4 * 1024 * 1024 * 1024)),
            1024 * 1024 * 1024
        );
        assert_eq!(default_host_fetch_budget_bytes_from_physical_memory(None), DEFAULT_HOST_FETCH_BUDGET_BYTES);
    }

    #[test]
    fn host_fetch_budget_tracks_acquire_grow_and_release() {
        let budget = HostFetchBudget::new(1024).unwrap();
        let mut first = budget.acquire(256).unwrap();
        assert_eq!(first.bytes(), 256);
        first.grow_to(512).unwrap();
        assert_eq!(first.bytes(), 512);
        let second = budget.acquire(512).unwrap();
        drop(first);
        let third = budget.acquire(512).unwrap();
        drop(second);
        drop(third);
    }

    #[test]
    fn host_fetch_budget_rejects_oversized_fetches() {
        let budget = HostFetchBudget::new(1024).unwrap();
        let error = budget.acquire(2048).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
        assert_eq!(error.fact_value("requested_bytes"), Some("2048"));
        assert_eq!(error.fact_value("budget_bytes"), Some("1024"));
    }

    #[test]
    fn execution_context_records_host_fetch_concurrency_limit() {
        let context = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(7).build().unwrap();
        assert_eq!(context.host_fetch_max_concurrent_ops(), 7);
    }

    #[test]
    fn execution_context_rejects_zero_host_fetch_concurrency_limit() {
        let error = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(0).build().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("host_fetch_max_concurrent_ops"));
    }

    fn available_gds_capability() -> DeploymentIoCapability {
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::Present { path: std::path::PathBuf::from(NVIDIA_FS_PROC_PATH) },
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::Available,
        )
    }

    #[test]
    fn source_fetch_config_defaults_match_phase5_contract() {
        let config = SourceFetchConfig::default();
        assert_eq!(config.substrate, SourceFetchSubstrate::Auto);
        assert_eq!(config.host_fetch_budget_bytes, None);
        assert_eq!(config.host_fetch_max_concurrent_ops, DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
        assert_eq!(config.gds_max_concurrent_ops, DEFAULT_GDS_MAX_CONCURRENT_OPS);
        assert_eq!(config.gds_min_chunk_bytes, DEFAULT_GDS_MIN_CHUNK_BYTES);
    }

    #[test]
    fn source_fetch_strategy_auto_keeps_remote_object_store_on_host_buffer() {
        let config = SourceFetchConfig::default();
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "s3://bucket/path/file.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(decision.reason, SourceFetchStrategyReason::RemoteObjectStore);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::RemoteObjectStore);
        assert_eq!(
            decision.diagnostic_facts(),
            vec![
                ("source_fetch_strategy", "host_buffer".to_string()),
                ("source_fetch_strategy_reason", "remote_object_store".to_string()),
                ("source_fetch_substrate", "auto".to_string()),
                ("source_gds_eligibility", "remote_object_store".to_string()),
            ]
        );
    }

    #[test]
    fn source_fetch_strategy_auto_selects_direct_device_only_for_gds_eligible_local_chunks() {
        let config = SourceFetchConfig::default();
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(decision.reason, SourceFetchStrategyReason::GdsEligible);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
    }

    #[test]
    fn source_fetch_strategy_host_buffer_substrate_overrides_gds_eligible_sources() {
        let config = SourceFetchConfig { substrate: SourceFetchSubstrate::HostBuffer, ..SourceFetchConfig::default() };
        let decision = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );

        assert_eq!(decision.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(decision.reason, SourceFetchStrategyReason::SubstrateHostBuffer);
        assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
    }

    #[test]
    fn source_fetch_strategy_direct_device_still_falls_back_for_ineligible_sources() {
        let config =
            SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..SourceFetchConfig::default() };

        let remote = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "https://example.test/file.parquet",
            Some(64 * 1024 * 1024),
        );
        assert_eq!(remote.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(remote.reason, SourceFetchStrategyReason::RemoteObjectStore);

        let small = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(1024),
        );
        assert_eq!(small.strategy, SourceFetchStrategy::HostBuffer);
        assert_eq!(small.reason, SourceFetchStrategyReason::BelowMinChunkBytes);

        let eligible = config.decide_source_fetch_strategy(
            &available_gds_capability(),
            "/opt/dlami/nvme/tpch/sf10/lineitem.parquet",
            Some(64 * 1024 * 1024),
        );
        assert_eq!(eligible.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(eligible.reason, SourceFetchStrategyReason::GdsEligible);
    }

    #[test]
    fn execution_context_exposes_effective_source_fetch_strategy() {
        let context = ExecutionContext::builder()
            .with_deployment_io_capability(available_gds_capability())
            .with_source_fetch_config(SourceFetchConfig::default())
            .build()
            .unwrap();

        let decision = context
            .source_fetch_strategy_for_location("/opt/dlami/nvme/tpch/sf10/lineitem.parquet", Some(64 * 1024 * 1024));

        assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
        assert_eq!(decision.substrate, SourceFetchSubstrate::Auto);
    }

    #[test]
    fn source_fetch_config_rejects_zero_gds_controls() {
        let deployment = available_gds_capability();
        let zero_ops = SourceFetchConfig { gds_max_concurrent_ops: 0, ..SourceFetchConfig::default() };
        let error = zero_ops.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("gds_max_concurrent_ops"));

        let zero_min_chunk = SourceFetchConfig { gds_min_chunk_bytes: 0, ..SourceFetchConfig::default() };
        let error = zero_min_chunk.validate_for_deployment(&deployment).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("gds_min_chunk_bytes"));
    }

    #[test]
    fn direct_device_source_fetch_config_requires_gds_available() {
        let unavailable = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::Available,
        );
        let config =
            SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..SourceFetchConfig::default() };
        let error = config.validate_for_deployment(&unavailable).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("source_fetch_substrate"), Some("direct_device"));
        assert_eq!(error.fact_value("gds_available"), Some("false"));
        assert_eq!(error.fact_value("gds_status"), Some("driver_missing"));

        config.validate_for_deployment(&available_gds_capability()).unwrap();
    }

    #[test]
    fn gds_concurrency_gate_limits_in_flight_ops() {
        let gate = GdsConcurrencyGate::new(1).unwrap();
        let permit = gate.acquire().unwrap();
        let gate_for_thread = gate.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let (acquired_sender, acquired_receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            let permit = gate_for_thread.acquire().unwrap();
            acquired_sender.send(()).unwrap();
            drop(permit);
        });
        started_receiver.recv().unwrap();
        assert!(acquired_receiver.try_recv().is_err());
        drop(permit);
        acquired_receiver.recv().unwrap();
        handle.join().unwrap();
    }
}
