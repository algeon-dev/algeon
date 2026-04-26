//! Query-keyed metrics forwarding and explicit local observation ownership.

use std::{
    collections::VecDeque,
    fmt,
    sync::{Arc, Mutex, Weak},
};

use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    metrics::QueryMetrics,
    observability::{
        RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationLimits, RuntimeObservationOwner,
        RuntimeQueryId, RuntimeTraceId, SealedRuntimeObservationSnapshot,
    },
    runtime::{LakehouseCacheFillEventSnapshot, QueryActivityLease},
};

use crate::native_exec::admission::QueryAdmissionFacts;

/// Owned adapter-side evidence transferred across the native execution boundary.
pub(crate) type RuntimeTraceSnapshot = SealedRuntimeObservationSnapshot;
pub(crate) type NativeObservationActivityFactory = Arc<dyn Fn() -> Option<QueryActivityLease> + Send + Sync + 'static>;

/// Observation authority selected for one native query boundary.
///
/// Server-coordinated queries receive only a stable recording handle; local
/// execution keeps the owner needed to seal and retain a standalone snapshot.
#[derive(Clone, Debug)]
pub(crate) enum NativeRuntimeObservation {
    Local(RuntimeObservationOwner),
    Coordinated(RuntimeObservationHandle),
}

impl NativeRuntimeObservation {
    pub(crate) fn handle(&self) -> RuntimeObservationHandle {
        match self {
            Self::Local(owner) => owner.handle(),
            Self::Coordinated(handle) => handle.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn into_local_owner(self) -> RuntimeObservationOwner {
        match self {
            Self::Local(owner) => owner,
            Self::Coordinated(_) => panic!("coordinated observation has no local seal owner"),
        }
    }
}

impl From<RuntimeObservationOwner> for NativeRuntimeObservation {
    fn from(owner: RuntimeObservationOwner) -> Self {
        Self::Local(owner)
    }
}

/// Query-keyed sink for native metrics captured while DataFusion wrapper metrics are populated.
pub(crate) trait NativeMetricsCaptureSink: Send + Sync {
    fn record_admission_facts(&self, query_id: &str, facts: QueryAdmissionFacts);

    fn record_query_metrics(&self, query_id: &str, metrics: &QueryMetrics);

    fn record_lakehouse_cache_fill_events(&self, query_id: &str, events: &[LakehouseCacheFillEventSnapshot]);
}

/// Cloneable config-time wiring used to create one durable owner per execution wrapper.
///
/// This value never owns runtime snapshots. It is therefore safe to retain in a
/// session or application config and clone into independently planned queries.
#[derive(Clone)]
pub(crate) struct NativeMetricsCapture {
    query_id: Arc<str>,
    sink: Option<Weak<dyn NativeMetricsCaptureSink>>,
    coordinated_observation: Option<RuntimeObservationHandle>,
    coordinated_activity: Option<NativeObservationActivityFactory>,
    observation_config: RuntimeObservationConfig,
    metric_capture_limits: RuntimeObservationLimits,
}

/// Deterministic accounting for a bounded standalone observation collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeRuntimeObservationRetentionStats {
    pub(crate) capacity: usize,
    pub(crate) retained: usize,
    pub(crate) recorded_total: u64,
    pub(crate) dropped_total: u64,
    pub(crate) evicted_total: u64,
    pub(crate) drained_total: u64,
}

#[derive(Debug)]
struct LocalObservationCollector {
    capacity: usize,
    observations: VecDeque<Arc<RuntimeTraceSnapshot>>,
    recorded_total: u64,
    dropped_total: u64,
    evicted_total: u64,
    drained_total: u64,
}

impl LocalObservationCollector {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            observations: VecDeque::with_capacity(capacity),
            recorded_total: 0,
            dropped_total: 0,
            evicted_total: 0,
            drained_total: 0,
        }
    }

    fn record(&mut self, observation: Arc<RuntimeTraceSnapshot>) {
        if self.observations.iter().any(|retained| retained.is_same_snapshot(&observation)) {
            return;
        }
        self.recorded_total = self.recorded_total.saturating_add(1);
        if self.capacity == 0 {
            self.dropped_total = self.dropped_total.saturating_add(1);
            return;
        }
        if self.observations.len() == self.capacity {
            let _ = self.observations.pop_front();
            self.evicted_total = self.evicted_total.saturating_add(1);
        }
        self.observations.push_back(observation);
    }

    fn stats(&self) -> NativeRuntimeObservationRetentionStats {
        NativeRuntimeObservationRetentionStats {
            capacity: self.capacity,
            retained: self.observations.len(),
            recorded_total: self.recorded_total,
            dropped_total: self.dropped_total,
            evicted_total: self.evicted_total,
            drained_total: self.drained_total,
        }
    }

    #[cfg(test)]
    fn drain(&mut self) -> Vec<Arc<RuntimeTraceSnapshot>> {
        let drained = self.observations.drain(..).collect::<Vec<_>>();
        self.drained_total = self.drained_total.saturating_add(u64::try_from(drained.len()).unwrap_or(u64::MAX));
        drained
    }
}

/// Metrics bridge plus the optional local observation root for one execution.
///
/// Standalone clones share one bounded collector across fragments and admission
/// epochs. Coordinated server captures have no local collector or seal owner.
#[derive(Clone)]
pub(crate) struct NativeExecutionMetricsCapture {
    wiring: NativeMetricsCapture,
    local_observations: Option<Arc<Mutex<LocalObservationCollector>>>,
}

impl NativeMetricsCapture {
    /// Maximum sealed observations retained by one execution before the
    /// deterministic oldest-first eviction policy applies.
    pub(crate) const DEFAULT_RUNTIME_OBSERVATION_CAPACITY: usize = 256;

    pub(crate) fn without_sink(query_id: impl Into<Arc<str>>) -> Self {
        Self {
            query_id: query_id.into(),
            sink: None,
            coordinated_observation: None,
            coordinated_activity: None,
            observation_config: RuntimeObservationConfig::native_execution(),
            metric_capture_limits: RuntimeObservationConfig::native_execution().limits(),
        }
    }

    #[cfg(all(feature = "server", test))]
    pub(crate) fn new(query_id: impl Into<Arc<str>>, sink: Arc<dyn NativeMetricsCaptureSink>) -> Self {
        Self {
            query_id: query_id.into(),
            sink: Some(Arc::downgrade(&sink)),
            coordinated_observation: None,
            coordinated_activity: None,
            observation_config: RuntimeObservationConfig::native_execution(),
            metric_capture_limits: RuntimeObservationConfig::native_execution().limits(),
        }
    }

    #[cfg(any(feature = "server", test))]
    pub(crate) fn coordinated(query_id: impl Into<Arc<str>>, observation: RuntimeObservationHandle) -> Self {
        let metric_capture_limits = observation.config().limits();
        Self {
            query_id: query_id.into(),
            sink: None,
            coordinated_observation: Some(observation),
            coordinated_activity: None,
            observation_config: RuntimeObservationConfig::disabled(),
            metric_capture_limits,
        }
    }

    #[cfg(any(feature = "server", test))]
    pub(crate) fn with_observation_activity_factory(mut self, factory: NativeObservationActivityFactory) -> Self {
        self.coordinated_activity = Some(factory);
        self
    }

    pub(crate) fn new_observation_activity(&self) -> Option<Result<QueryActivityLease>> {
        self.coordinated_observation.as_ref()?;
        Some(match &self.coordinated_activity {
            Some(factory) => factory().ok_or_else(coordinated_activity_unavailable_error),
            None => Err(QueryError::new(
                QueryErrorCode::Internal,
                "datafusion_nexus.native_exec.admission.observation_activity",
                "coordinated observation activity factory is missing",
            )
            .with_fact("reason", "coordinated_activity_factory_missing")
            .with_fact("admission_stage", "observation_activity")),
        })
    }

    #[cfg(feature = "server")]
    pub(crate) fn query_id(&self) -> &str {
        &self.query_id
    }

    #[cfg(test)]
    pub(crate) fn with_runtime_observation_config(mut self, config: RuntimeObservationConfig) -> Self {
        self.observation_config = config;
        self.metric_capture_limits = config.limits();
        self
    }

    pub(crate) const fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.observation_config
    }

    pub(crate) const fn metric_capture_limits(&self) -> RuntimeObservationLimits {
        self.metric_capture_limits
    }

    pub(crate) fn new_execution(&self) -> NativeExecutionMetricsCapture {
        self.new_execution_with_capacity(Self::DEFAULT_RUNTIME_OBSERVATION_CAPACITY)
    }

    #[cfg(test)]
    pub(crate) fn new_execution_with_test_capacity(&self, capacity: usize) -> NativeExecutionMetricsCapture {
        self.new_execution_with_capacity(capacity)
    }

    fn new_execution_with_capacity(&self, capacity: usize) -> NativeExecutionMetricsCapture {
        NativeExecutionMetricsCapture {
            wiring: self.clone(),
            local_observations: self
                .coordinated_observation
                .is_none()
                .then(|| Arc::new(Mutex::new(LocalObservationCollector::new(capacity)))),
        }
    }
}

impl NativeExecutionMetricsCapture {
    pub(crate) fn new_runtime_observation(
        &self,
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
    ) -> NativeRuntimeObservation {
        match &self.wiring.coordinated_observation {
            Some(observation) => NativeRuntimeObservation::Coordinated(observation.clone()),
            None => {
                let owner = RuntimeObservationOwner::new(query_id, trace_id, self.wiring.observation_config);
                // QueryService owns admitted local roots through asynchronous
                // cleanup, so the terminal snapshot cannot be synchronously
                // returned to the wrapper. Deliver it directly to the one
                // local collector; this is not sink publication, and the
                // collector identity-dedupes synchronous return paths.
                let collector = self
                    .local_observations
                    .as_ref()
                    .expect("standalone execution capture owns a local observation collector")
                    .clone();
                owner.register_seal_observer(move |snapshot| {
                    collector.lock().unwrap_or_else(std::sync::PoisonError::into_inner).record(Arc::new(snapshot));
                });
                NativeRuntimeObservation::Local(owner)
            },
        }
    }

    pub(crate) fn new_observation_activity(&self) -> Option<Result<QueryActivityLease>> {
        self.wiring.new_observation_activity()
    }

    pub(crate) const fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.wiring.runtime_observation_config()
    }

    pub(crate) const fn metric_capture_limits(&self) -> RuntimeObservationLimits {
        self.wiring.metric_capture_limits()
    }

    #[cfg(test)]
    pub(crate) fn runtime_observations(&self) -> Vec<Arc<RuntimeTraceSnapshot>> {
        self.local_observations.as_ref().map_or_else(Vec::new, |collector| {
            collector.lock().unwrap_or_else(std::sync::PoisonError::into_inner).observations.iter().cloned().collect()
        })
    }

    #[cfg(test)]
    pub(crate) fn drain_runtime_observations(&self) -> Vec<Arc<RuntimeTraceSnapshot>> {
        self.local_observations.as_ref().map_or_else(Vec::new, |collector| {
            collector.lock().unwrap_or_else(std::sync::PoisonError::into_inner).drain()
        })
    }

    pub(crate) fn retention_stats(&self) -> NativeRuntimeObservationRetentionStats {
        self.local_observations.as_ref().map_or(
            NativeRuntimeObservationRetentionStats {
                capacity: 0,
                retained: 0,
                recorded_total: 0,
                dropped_total: 0,
                evicted_total: 0,
                drained_total: 0,
            },
            |collector| collector.lock().unwrap_or_else(std::sync::PoisonError::into_inner).stats(),
        )
    }

    #[cfg(test)]
    pub(crate) fn shares_owner_with(&self, other: &Self) -> bool {
        match (&self.local_observations, &other.local_observations) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, _) | (_, None) => false,
        }
    }

    pub(super) fn record_query_metrics(&self, metrics: &QueryMetrics) {
        let wiring = &self.wiring;
        if let Some(sink) = wiring.sink.as_ref().and_then(Weak::upgrade) {
            if metrics.query_id().as_str() == wiring.query_id.as_ref() {
                sink.record_query_metrics(&wiring.query_id, metrics);
            } else {
                sink.record_query_metrics(&wiring.query_id, &metrics.clone().with_query_id(wiring.query_id.as_ref()));
            }
        }
    }

    pub(super) fn record_lakehouse_cache_fill_events(&self, events: &[LakehouseCacheFillEventSnapshot]) {
        let wiring = &self.wiring;
        if let Some(sink) = wiring.sink.as_ref().and_then(Weak::upgrade) {
            sink.record_lakehouse_cache_fill_events(&wiring.query_id, events);
        }
    }

    pub(crate) fn record_runtime_observation(&self, observation: Arc<RuntimeTraceSnapshot>) {
        if let Some(collector) = &self.local_observations {
            collector.lock().unwrap_or_else(std::sync::PoisonError::into_inner).record(observation);
        }
    }

    pub(super) fn record_admission_facts(&self, facts: QueryAdmissionFacts) {
        let wiring = &self.wiring;
        if let Some(sink) = wiring.sink.as_ref().and_then(Weak::upgrade) {
            sink.record_admission_facts(&wiring.query_id, facts);
        }
    }
}

fn coordinated_activity_unavailable_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "datafusion_nexus.native_exec.admission.observation_activity",
        "query coordinator rejected observation activity acquisition",
    )
    .with_fact("reason", "query_coordinator_activity_rejected")
    .with_fact("admission_stage", "observation_activity")
}

impl fmt::Debug for NativeMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeMetricsCapture").field("query_id", &self.query_id).finish_non_exhaustive()
    }
}

impl fmt::Debug for NativeExecutionMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeExecutionMetricsCapture")
            .field("query_id", &self.wiring.query_id)
            .field("retention", &self.retention_stats())
            .finish_non_exhaustive()
    }
}
