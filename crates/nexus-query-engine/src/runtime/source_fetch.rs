//! Source fetch substrate policies and service-wide IO gates.

use std::{
    fmt,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use super::{
    DeploymentIoCapability, ExecutionCancellation, SourceGdsEligibility, source_gds_eligibility_with_default_mounts,
};
use crate::error::{QueryError, QueryErrorCode, Result};

/// Fallback host-buffer budget used when physical RAM cannot be detected.
pub const DEFAULT_HOST_FETCH_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
/// Maximum default host-buffer budget derived from physical RAM.
pub const DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Default service-wide concurrent object-store SourceFetch operation cap.
pub const DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS: usize = 64;
/// Default service-wide concurrent background lakehouse cache-fill operation cap.
pub const DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS: usize = 4;
/// Default service-wide concurrent GDS DMA operation cap.
pub const DEFAULT_GDS_MAX_CONCURRENT_OPS: usize = 32;
/// Default per-chunk size threshold for selecting future GDS direct-device reads.
pub const DEFAULT_GDS_MIN_CHUNK_BYTES: u64 = 4 * 1024 * 1024;

/// Service-level source IO substrate preference.
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
    /// Max in-flight background complete-object cache-fill bytes across queries. `None` auto-sizes from host RAM.
    pub lakehouse_cache_fill_budget_bytes: Option<u64>,
    /// Max concurrent background complete-object cache-fill operations across queries.
    pub lakehouse_cache_fill_max_concurrent_ops: usize,
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
        if let Some(lakehouse_cache_fill_budget_bytes) = self.lakehouse_cache_fill_budget_bytes {
            validate_lakehouse_cache_fill_budget_bytes(lakehouse_cache_fill_budget_bytes)?;
        }
        validate_host_fetch_max_concurrent_ops(self.host_fetch_max_concurrent_ops)?;
        validate_lakehouse_cache_fill_max_concurrent_ops(self.lakehouse_cache_fill_max_concurrent_ops)?;
        validate_gds_max_concurrent_ops(self.gds_max_concurrent_ops)?;
        validate_gds_min_chunk_bytes(self.gds_min_chunk_bytes)?;
        if matches!(self.substrate, SourceFetchSubstrate::DirectDevice) && !deployment.gds_available {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_fetch_config.validate",
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
    /// the Milestone 5 contract: remote object-store/network sources remain
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
            lakehouse_cache_fill_budget_bytes: None,
            lakehouse_cache_fill_max_concurrent_ops: DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS,
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

pub(super) fn default_host_fetch_budget_bytes_from_physical_memory(physical_memory_bytes: Option<u64>) -> u64 {
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
        let mut state = self.lock_state("nexus_query_engine.host_fetch_budget.acquire")?;
        while !host_fetch_bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.host_fetch_budget.acquire",
                    "host fetch budget mutex was poisoned while waiting for capacity",
                )
            })?;
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
        Ok(HostFetchPermit { budget: self.clone(), bytes })
    }

    /// Attempts to acquire a host-buffer permit without waiting.
    pub fn try_acquire(&self, bytes: u64) -> Result<Option<HostFetchPermit>> {
        validate_host_fetch_request_bytes(bytes, self.budget_bytes())?;
        let mut state = self.lock_state("nexus_query_engine.host_fetch_budget.try_acquire")?;
        if !host_fetch_bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
            return Ok(None);
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
        Ok(Some(HostFetchPermit { budget: self.clone(), bytes }))
    }

    /// Acquires a host-buffer permit with a bounded wait.
    pub fn acquire_with_timeout(
        &self,
        cancellation: &ExecutionCancellation,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<HostFetchPermit>> {
        validate_host_fetch_request_bytes(bytes, self.budget_bytes())?;
        let start = Instant::now();
        let mut state = self.lock_state(operation)?;
        loop {
            cancellation.check(operation)?;
            if host_fetch_bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
                state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
                return Ok(Some(HostFetchPermit { budget: self.clone(), bytes }));
            }
            let Some(remaining) = timeout.checked_sub(start.elapsed()) else {
                return Ok(None);
            };
            let wait = remaining.min(Duration::from_millis(100));
            let (next_state, _wait_result) = self.inner.available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "host fetch budget mutex was poisoned while waiting for capacity",
                )
            })?;
            state = next_state;
        }
    }

    fn grow_permit(&self, permit: &mut HostFetchPermit, new_bytes: u64) -> Result<()> {
        if new_bytes <= permit.bytes {
            return Ok(());
        }
        let delta = new_bytes - permit.bytes;
        validate_host_fetch_request_bytes(new_bytes, self.budget_bytes())?;
        let mut state = self.lock_state("nexus_query_engine.host_fetch_budget.grow")?;
        while !host_fetch_bytes_fit(state.in_flight_bytes, delta, self.budget_bytes()) {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.host_fetch_budget.grow",
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
                eprintln!("[nexus-query-engine] warning: failed to release poisoned host fetch budget state");
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
            "nexus_query_engine.host_fetch_budget.validate",
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
            "nexus_query_engine.host_fetch_budget.request",
            "host fetch budget request bytes must be greater than zero",
        )
        .with_fact("field", "host_fetch_request_bytes")
        .with_fact("reason", "zero"));
    }
    if requested_bytes > budget_bytes {
        return Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.host_fetch_budget.request",
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

// Cloneable permit gate for bounded object-store fetch concurrency.
#[derive(Debug, Clone)]
pub(super) struct SourceFetchConcurrencyGate {
    inner: Arc<SourceFetchConcurrencyGateInner>,
}

impl SourceFetchConcurrencyGate {
    pub(super) fn new(max_concurrent_ops: usize) -> Result<Self> {
        validate_host_fetch_max_concurrent_ops(max_concurrent_ops)?;
        Ok(Self {
            inner: Arc::new(SourceFetchConcurrencyGateInner {
                max_concurrent_ops,
                state: Mutex::new(SourceFetchConcurrencyGateState::default()),
                available: Condvar::new(),
            }),
        })
    }

    pub(super) fn max_concurrent_ops(&self) -> usize {
        self.inner.max_concurrent_ops
    }

    pub(super) fn acquire(&self) -> Result<SourceFetchOperationPermit> {
        let mut state = self.lock_state("nexus_query_engine.source_fetch_concurrency.acquire")?;
        while state.in_flight_ops >= self.max_concurrent_ops() {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.source_fetch_concurrency.acquire",
                    "source fetch concurrency mutex was poisoned while waiting for a permit",
                )
            })?;
        }
        state.in_flight_ops = state.in_flight_ops.saturating_add(1);
        Ok(SourceFetchOperationPermit { gate: self.clone() })
    }

    pub(super) fn try_acquire(&self) -> Result<Option<SourceFetchOperationPermit>> {
        let mut state = self.lock_state("nexus_query_engine.source_fetch_concurrency.try_acquire")?;
        if state.in_flight_ops >= self.max_concurrent_ops() {
            return Ok(None);
        }
        state.in_flight_ops = state.in_flight_ops.saturating_add(1);
        Ok(Some(SourceFetchOperationPermit { gate: self.clone() }))
    }

    pub(super) fn acquire_with_timeout(
        &self,
        cancellation: &ExecutionCancellation,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<SourceFetchOperationPermit>> {
        let start = Instant::now();
        let mut state = self.lock_state(operation)?;
        loop {
            cancellation.check(operation)?;
            if state.in_flight_ops < self.max_concurrent_ops() {
                state.in_flight_ops = state.in_flight_ops.saturating_add(1);
                return Ok(Some(SourceFetchOperationPermit { gate: self.clone() }));
            }
            let Some(remaining) = timeout.checked_sub(start.elapsed()) else {
                return Ok(None);
            };
            let wait = remaining.min(Duration::from_millis(100));
            let (next_state, _wait_result) = self.inner.available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "source fetch concurrency mutex was poisoned while waiting for a permit",
                )
            })?;
            state = next_state;
        }
    }

    fn release(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.in_flight_ops = state.in_flight_ops.saturating_sub(1);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release poisoned source fetch concurrency state");
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
            "nexus_query_engine.source_fetch_concurrency.validate",
            "host fetch max concurrent operations must be greater than zero",
        )
        .with_fact("field", "host_fetch_max_concurrent_ops")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn validate_lakehouse_cache_fill_budget_bytes(budget_bytes: u64) -> Result<()> {
    if budget_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.lakehouse_cache.background_fill_budget.validate",
            "lakehouse cache fill budget bytes must be greater than zero",
        )
        .with_fact("field", "lakehouse_cache_fill_budget_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn validate_lakehouse_cache_fill_max_concurrent_ops(max_concurrent_ops: usize) -> Result<()> {
    if max_concurrent_ops == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.lakehouse_cache.background_fill_concurrency.validate",
            "lakehouse cache fill max concurrent operations must be greater than zero",
        )
        .with_fact("field", "lakehouse_cache_fill_max_concurrent_ops")
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

// Cloneable permit gate reserved for future GDS operation concurrency.
#[derive(Debug, Clone)]
pub(super) struct GdsConcurrencyGate {
    inner: Arc<GdsConcurrencyGateInner>,
}

#[allow(dead_code)]
impl GdsConcurrencyGate {
    pub(super) fn new(max_concurrent_ops: usize) -> Result<Self> {
        validate_gds_max_concurrent_ops(max_concurrent_ops)?;
        Ok(Self {
            inner: Arc::new(GdsConcurrencyGateInner {
                max_concurrent_ops,
                state: Mutex::new(GdsConcurrencyGateState::default()),
                available: Condvar::new(),
            }),
        })
    }

    pub(super) fn max_concurrent_ops(&self) -> usize {
        self.inner.max_concurrent_ops
    }

    pub(super) fn acquire(&self) -> Result<GdsOperationPermit> {
        let mut state = self.lock_state("nexus_query_engine.gds_concurrency.acquire")?;
        while state.in_flight_ops >= self.max_concurrent_ops() {
            state = self.inner.available.wait(state).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.gds_concurrency.acquire",
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
                eprintln!("[nexus-query-engine] warning: failed to release poisoned GDS concurrency state");
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
            "nexus_query_engine.gds_concurrency.validate",
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
            "nexus_query_engine.source_fetch_config.validate",
            "GDS minimum chunk bytes must be greater than zero",
        )
        .with_fact("field", "gds_min_chunk_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_fetch_budget_try_acquire_does_not_wait_when_capacity_is_held() {
        let budget = HostFetchBudget::new(10).expect("budget should be valid");
        let held = budget.acquire(10).expect("initial permit should fit");

        let next = budget.try_acquire(1).expect("try acquire should not error");

        assert!(next.is_none());
        drop(held);
        assert!(budget.try_acquire(1).expect("released capacity should fit").is_some());
    }

    #[test]
    fn source_fetch_concurrency_try_acquire_does_not_wait_when_capacity_is_held() {
        let gate = SourceFetchConcurrencyGate::new(1).expect("gate should be valid");
        let held = gate.acquire().expect("initial permit should fit");

        let next = gate.try_acquire().expect("try acquire should not error");

        assert!(next.is_none());
        drop(held);
        assert!(gate.try_acquire().expect("released capacity should fit").is_some());
    }
}
