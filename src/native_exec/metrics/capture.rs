//! Runtime-observation authority carried across native execution wrappers.

use std::{fmt, sync::Arc};

use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    observability::{
        RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationLimits, RuntimeObservationOwner,
        RuntimeQueryId, RuntimeTraceId,
    },
    runtime::QueryActivityLease,
};

/// Activity authority selected for one coordinated native query boundary.
pub(crate) type NativeObservationActivityFactory = Arc<dyn Fn() -> Option<QueryActivityLease> + Send + Sync + 'static>;

/// Observation authority selected for one native query boundary.
///
/// A local owner seals its snapshot into the owning attempt only; this adapter
/// layer deliberately retains no second snapshot store.
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

/// Cloneable config-time observation wiring.
#[derive(Clone)]
pub(crate) struct NativeMetricsCapture {
    query_id: Option<Arc<str>>,
    coordinated_observation: Option<RuntimeObservationHandle>,
    coordinated_activity: Option<NativeObservationActivityFactory>,
    observation_config: RuntimeObservationConfig,
    metric_capture_limits: RuntimeObservationLimits,
}

impl NativeMetricsCapture {
    pub(crate) fn standalone() -> Self {
        let observation_config = RuntimeObservationConfig::native_execution();
        Self {
            query_id: None,
            coordinated_observation: None,
            coordinated_activity: None,
            observation_config,
            metric_capture_limits: observation_config.limits(),
        }
    }

    #[cfg(any(feature = "server", test))]
    pub(crate) fn coordinated(query_id: impl Into<Arc<str>>, observation: RuntimeObservationHandle) -> Self {
        let metric_capture_limits = observation.config().limits();
        Self {
            query_id: Some(query_id.into()),
            coordinated_observation: Some(observation),
            coordinated_activity: None,
            observation_config: RuntimeObservationConfig::disabled(),
            metric_capture_limits,
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn query_id(&self) -> &str {
        self.query_id.as_deref().expect("coordinated server capture carries a query id")
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
        NativeExecutionMetricsCapture { wiring: self.clone() }
    }
}

/// Per-wrapper projection configuration. It owns neither terminal state nor
/// sealed runtime observations.
#[derive(Clone)]
pub(crate) struct NativeExecutionMetricsCapture {
    wiring: NativeMetricsCapture,
}

impl NativeExecutionMetricsCapture {
    pub(crate) fn new_runtime_observation(
        &self,
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
    ) -> NativeRuntimeObservation {
        match &self.wiring.coordinated_observation {
            Some(observation) => NativeRuntimeObservation::Coordinated(observation.clone()),
            None => NativeRuntimeObservation::Local(RuntimeObservationOwner::new(
                query_id,
                trace_id,
                self.wiring.observation_config,
            )),
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
        f.debug_struct("NativeMetricsCapture")
            .field("coordinated", &self.coordinated_observation.is_some())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for NativeExecutionMetricsCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeExecutionMetricsCapture").finish_non_exhaustive()
    }
}
