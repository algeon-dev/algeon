//! Service-wide lakehouse cache-fill resource policy and gates.

use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::ExecutionCancellation;

/// Conservative fallback when physical-memory detection is unavailable.
pub const DEFAULT_LAKEHOUSE_CACHE_FILL_BUDGET_BYTES: u64 = LakehouseCacheFillConfig::DEFAULT_BUDGET_BYTES;
/// Auto-sizing cap so cache maintenance does not crowd other host state.
pub const DEFAULT_LAKEHOUSE_CACHE_FILL_BUDGET_CAP_BYTES: u64 = LakehouseCacheFillConfig::DEFAULT_BUDGET_CAP_BYTES;
/// Default bounded wait for blocking cache maintenance resource acquisition.
pub const DEFAULT_LAKEHOUSE_CACHE_FILL_ACQUIRE_TIMEOUT: Duration = LakehouseCacheFillConfig::DEFAULT_ACQUIRE_TIMEOUT;
/// Default complete-object cache maintenance concurrency.
pub const DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS: usize = LakehouseCacheFillConfig::DEFAULT_MAX_CONCURRENT_OPS;

/// Service-wide resources used only by complete-object lakehouse cache fills.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LakehouseCacheFillConfig {
    /// Max in-flight complete-object bytes. `None` auto-sizes from host RAM.
    pub budget_bytes: Option<u64>,
    /// Timeout for blocking byte-budget and operation-gate acquisition.
    pub acquire_timeout: Duration,
    /// Max concurrent complete-object cache maintenance operations.
    pub max_concurrent_ops: usize,
}

impl LakehouseCacheFillConfig {
    pub const DEFAULT_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
    pub const DEFAULT_BUDGET_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;
    pub const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);
    pub const DEFAULT_MAX_CONCURRENT_OPS: usize = 4;

    pub fn validate(&self) -> Result<()> {
        if let Some(budget_bytes) = self.budget_bytes {
            validate_budget_bytes(budget_bytes)?;
        }
        if self.acquire_timeout.is_zero() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.lakehouse_cache.fill_config.validate",
                "lakehouse cache fill acquire timeout must be greater than zero",
            )
            .with_fact("field", "lakehouse_cache_fill_acquire_timeout")
            .with_fact("reason", "zero"));
        }
        if self.max_concurrent_ops == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.lakehouse_cache.fill_config.validate",
                "lakehouse cache fill max concurrent operations must be greater than zero",
            )
            .with_fact("field", "lakehouse_cache_fill_max_concurrent_ops")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for LakehouseCacheFillConfig {
    fn default() -> Self {
        Self {
            budget_bytes: None,
            acquire_timeout: Self::DEFAULT_ACQUIRE_TIMEOUT,
            max_concurrent_ops: Self::DEFAULT_MAX_CONCURRENT_OPS,
        }
    }
}

pub fn default_lakehouse_cache_fill_budget_bytes() -> u64 {
    default_lakehouse_cache_fill_budget_bytes_from_physical_memory(detect_physical_memory_bytes())
}

pub(super) fn default_lakehouse_cache_fill_budget_bytes_from_physical_memory(
    physical_memory_bytes: Option<u64>,
) -> u64 {
    physical_memory_bytes
        .and_then(|bytes| bytes.checked_div(4))
        .map(|bytes| bytes.min(LakehouseCacheFillConfig::DEFAULT_BUDGET_CAP_BYTES))
        .filter(|bytes| *bytes > 0)
        .unwrap_or(LakehouseCacheFillConfig::DEFAULT_BUDGET_BYTES)
}

fn detect_physical_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[derive(Debug)]
struct CacheFillByteBudgetInner {
    budget_bytes: u64,
    state: Mutex<CacheFillByteBudgetState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct CacheFillByteBudgetState {
    in_flight_bytes: u64,
}

/// Service-wide byte budget for complete-object lakehouse cache maintenance.
#[derive(Debug, Clone)]
pub struct CacheFillByteBudget {
    inner: Arc<CacheFillByteBudgetInner>,
}

impl CacheFillByteBudget {
    pub fn new(budget_bytes: u64) -> Result<Self> {
        validate_budget_bytes(budget_bytes)?;
        Ok(Self {
            inner: Arc::new(CacheFillByteBudgetInner {
                budget_bytes,
                state: Mutex::new(CacheFillByteBudgetState::default()),
                available: Condvar::new(),
            }),
        })
    }

    pub fn budget_bytes(&self) -> u64 {
        self.inner.budget_bytes
    }

    pub fn try_acquire(&self, bytes: u64) -> Result<Option<CacheFillBytePermit>> {
        validate_request_bytes(bytes, self.budget_bytes())?;
        let mut state = self.lock_state("query_engine.lakehouse_cache.fill_budget.try_acquire")?;
        if !bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
            return Ok(None);
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
        Ok(Some(CacheFillBytePermit { budget: self.clone(), bytes }))
    }

    pub fn acquire_with_timeout(
        &self,
        cancellation: &ExecutionCancellation,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillBytePermit>> {
        validate_request_bytes(bytes, self.budget_bytes())?;
        let start = Instant::now();
        let mut state = self.lock_state(operation)?;
        loop {
            cancellation.check(operation)?;
            if bytes_fit(state.in_flight_bytes, bytes, self.budget_bytes()) {
                state.in_flight_bytes = state.in_flight_bytes.saturating_add(bytes);
                return Ok(Some(CacheFillBytePermit { budget: self.clone(), bytes }));
            }
            let Some(remaining) = timeout.checked_sub(start.elapsed()) else {
                return Ok(None);
            };
            let wait = remaining.min(Duration::from_millis(100));
            let (next_state, _) = self.inner.available.wait_timeout(state, wait).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "lakehouse cache fill byte-budget mutex was poisoned while waiting",
                )
            })?;
            state = next_state;
        }
    }

    fn try_grow_permit(&self, permit: &mut CacheFillBytePermit, new_bytes: u64) -> Result<bool> {
        if new_bytes <= permit.bytes {
            return Ok(true);
        }
        validate_request_bytes(new_bytes, self.budget_bytes())?;
        let delta = new_bytes - permit.bytes;
        let mut state = self.lock_state("query_engine.lakehouse_cache.fill_budget.grow")?;
        // Growth never waits: two fills that each hold part of the budget and
        // wait for the remainder deadlock while holding their shares and the
        // object-store worker. A fill that cannot grow abandons its cache
        // write instead; the reader still gets its data.
        if !bytes_fit(state.in_flight_bytes, delta, self.budget_bytes()) {
            return Ok(false);
        }
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(delta);
        permit.bytes = new_bytes;
        Ok(true)
    }

    fn release(&self, bytes: u64) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.in_flight_bytes = state.in_flight_bytes.saturating_sub(bytes);
                self.inner.available.notify_all();
            },
            Err(_) => eprintln!("[query-engine] warning: failed to release poisoned cache fill byte budget"),
        }
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, CacheFillByteBudgetState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, operation, "lakehouse cache fill byte-budget mutex was poisoned")
        })
    }
}

impl Default for CacheFillByteBudget {
    fn default() -> Self {
        Self::new(default_lakehouse_cache_fill_budget_bytes()).expect("default cache fill byte budget is non-zero")
    }
}

/// RAII permit for bytes charged to a [`CacheFillByteBudget`].
#[derive(Debug)]
pub struct CacheFillBytePermit {
    budget: CacheFillByteBudget,
    bytes: u64,
}

impl CacheFillBytePermit {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Grows this permit without waiting. Returns `false` when the budget
    /// cannot cover the growth; the caller abandons its cache write and the
    /// reader still gets its data.
    pub fn try_grow(&mut self, new_bytes: u64) -> Result<bool> {
        let budget = self.budget.clone();
        budget.try_grow_permit(self, new_bytes)
    }
}

impl Drop for CacheFillBytePermit {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

#[derive(Debug)]
struct CacheFillConcurrencyGateInner {
    max_concurrent_ops: usize,
    state: Mutex<CacheFillConcurrencyGateState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct CacheFillConcurrencyGateState {
    in_flight_ops: usize,
}

#[derive(Debug, Clone)]
pub(super) struct CacheFillConcurrencyGate {
    inner: Arc<CacheFillConcurrencyGateInner>,
}

impl CacheFillConcurrencyGate {
    pub(super) fn new(max_concurrent_ops: usize) -> Result<Self> {
        let config = LakehouseCacheFillConfig { max_concurrent_ops, ..Default::default() };
        config.validate()?;
        Ok(Self {
            inner: Arc::new(CacheFillConcurrencyGateInner {
                max_concurrent_ops,
                state: Mutex::new(CacheFillConcurrencyGateState::default()),
                available: Condvar::new(),
            }),
        })
    }

    pub(super) fn max_concurrent_ops(&self) -> usize {
        self.inner.max_concurrent_ops
    }

    pub(super) fn acquire_with_timeout(
        &self,
        cancellation: &ExecutionCancellation,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillOperationPermit>> {
        let start = Instant::now();
        let mut state = self.lock_state(operation)?;
        loop {
            cancellation.check(operation)?;
            if state.in_flight_ops < self.max_concurrent_ops() {
                state.in_flight_ops = state.in_flight_ops.saturating_add(1);
                return Ok(Some(CacheFillOperationPermit { gate: self.clone() }));
            }
            let Some(remaining) = timeout.checked_sub(start.elapsed()) else {
                return Ok(None);
            };
            let wait = remaining.min(Duration::from_millis(100));
            let (next_state, _) = self.inner.available.wait_timeout(state, wait).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "lakehouse cache fill concurrency mutex was poisoned while waiting",
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
            Err(_) => eprintln!("[query-engine] warning: failed to release poisoned cache fill concurrency"),
        }
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, CacheFillConcurrencyGateState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, operation, "lakehouse cache fill concurrency mutex was poisoned")
        })
    }
}

#[doc(hidden)]
pub struct CacheFillOperationPermit {
    gate: CacheFillConcurrencyGate,
}

impl Drop for CacheFillOperationPermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

fn bytes_fit(in_flight_bytes: u64, requested_bytes: u64, budget_bytes: u64) -> bool {
    in_flight_bytes.saturating_add(requested_bytes) <= budget_bytes
}

fn validate_budget_bytes(budget_bytes: u64) -> Result<()> {
    if budget_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.fill_budget.validate",
            "lakehouse cache fill budget bytes must be greater than zero",
        )
        .with_fact("field", "lakehouse_cache_fill_budget_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn validate_request_bytes(requested_bytes: u64, budget_bytes: u64) -> Result<()> {
    if requested_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.fill_budget.request",
            "lakehouse cache fill request bytes must be greater than zero",
        )
        .with_fact("field", "lakehouse_cache_fill_request_bytes")
        .with_fact("reason", "zero"));
    }
    if requested_bytes > budget_bytes {
        return Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "query_engine.lakehouse_cache.fill_budget.request",
            "lakehouse cache fill exceeds its configured byte budget",
        )
        .with_fact("requested_bytes", requested_bytes)
        .with_fact("budget_bytes", budget_bytes));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_fill_byte_budget_try_acquire_does_not_wait_when_capacity_is_held() {
        let budget = CacheFillByteBudget::new(10).unwrap();
        let held = budget.try_acquire(10).unwrap().unwrap();
        assert!(budget.try_acquire(1).unwrap().is_none());
        drop(held);
        assert!(budget.try_acquire(1).unwrap().is_some());
    }
}
