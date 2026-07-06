//! Process-scoped GPU backend facade for embedded DataFusion sessions.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use datafusion::execution::SessionStateBuilder;
use nexus_query_engine::{
    DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, MemoryControlPolicy, NvidiaFsProbeStatus,
    QueryService, QueryServiceConfig, QueryServiceTenantConfig, TenantId,
};

use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, Error, ErrorCode, GpuFallbackPolicy,
    NativeCudfExecutionConfig, Result, native_exec::QueryAdmissionSource,
    session::install_cudf_native_rule_with_session_config, session_config::DataFusionNexusSessionConfig,
};

const DEFAULT_BACKEND_TENANT_ID: &str = "default";
const DEFAULT_ADMISSION_QUEUE_MULTIPLIER: usize = 64;

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    query_service: QueryService,
    tenant_id: TenantId,
    optimizer_config: CudfOptimizerConfig,
    device_ordinals: Vec<usize>,
}

/// Backend memory policy.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NexusMemoryPolicy {
    /// Auto-size bounded execution from selected device capacity.
    BoundedAuto { device_ordinals: Vec<usize>, budget_fraction: Option<f64> },
    /// Explicit bounded execution budget and source chunk controls.
    Bounded {
        device_ordinals: Vec<usize>,
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    },
    /// No native bounded-memory controls. Backend admission currently requires bounded memory.
    Unbounded { device_ordinals: Vec<usize> },
}

impl Default for NexusMemoryPolicy {
    fn default() -> Self {
        Self::BoundedAuto { device_ordinals: vec![0], budget_fraction: None }
    }
}

/// Backend admission policy. Grant-band and slot-guarantee controls land in Phase C.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NexusAdmissionPolicy {
    max_concurrent_queries: usize,
    max_queued: Option<usize>,
    wait_timeout: Duration,
}

impl NexusAdmissionPolicy {
    pub fn with_max_concurrent_queries(mut self, max_concurrent_queries: usize) -> Self {
        self.max_concurrent_queries = max_concurrent_queries;
        self
    }

    pub fn with_max_queued(mut self, max_queued: usize) -> Self {
        self.max_queued = Some(max_queued);
        self
    }

    pub fn with_wait_timeout(mut self, wait_timeout: Duration) -> Self {
        self.wait_timeout = wait_timeout;
        self
    }

    pub fn max_concurrent_queries(&self) -> usize {
        self.max_concurrent_queries
    }

    pub fn max_queued(&self) -> usize {
        self.max_queued.unwrap_or_else(|| admission_queue_len(self.max_concurrent_queries))
    }

    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout
    }
}

impl Default for NexusAdmissionPolicy {
    fn default() -> Self {
        Self { max_concurrent_queries: 2, max_queued: None, wait_timeout: Duration::from_secs(60) }
    }
}

/// Builder for [`NexusGpuBackend`].
#[derive(Debug, Clone)]
pub struct NexusGpuBackendBuilder {
    fallback_policy: GpuFallbackPolicy,
    memory_policy: NexusMemoryPolicy,
    admission_policy: NexusAdmissionPolicy,
    tenant_id: TenantId,
}

impl NexusGpuBackend {
    pub fn builder() -> NexusGpuBackendBuilder {
        NexusGpuBackendBuilder::default()
    }

    /// Install this backend's native optimizer into a DataFusion session builder.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(self.optimizer_config.clone()));
        let session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&self.optimizer_config).frozen_for_backend_admission();
        Ok(install_cudf_native_rule_with_session_config(builder, rule, session_config))
    }

    /// Shared query-service admission owner used by sessions installed from this backend.
    pub fn query_service(&self) -> &QueryService {
        &self.query_service
    }

    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    pub fn optimizer_config(&self) -> &CudfOptimizerConfig {
        &self.optimizer_config
    }

    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    #[cfg(feature = "server")]
    pub(crate) fn from_optimizer_config_for_admission(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(optimizer_config, device_ordinals, admission_policy, tenant_id, false)
    }

    #[cfg(all(test, feature = "server"))]
    pub(crate) fn from_optimizer_config_for_admission_only(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(optimizer_config, device_ordinals, admission_policy, tenant_id, true)
    }

    #[cfg(feature = "server")]
    fn from_optimizer_config_inner(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
        admission_only: bool,
    ) -> Result<Option<Self>> {
        let Some(resolved) = ResolvedBackendMemoryPolicy::from_execution_config(
            device_ordinals,
            optimizer_config.native_execution_config().clone(),
        )?
        else {
            return Ok(None);
        };
        Self::build_bounded(optimizer_config, resolved, admission_policy, tenant_id, admission_only).map(Some)
    }

    fn build_bounded(
        optimizer_config: CudfOptimizerConfig,
        resolved: ResolvedBackendMemoryPolicy,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
        admission_only: bool,
    ) -> Result<Self> {
        validate_admission_policy(&admission_policy)?;
        let min_budget = partitioned_query_budget_bytes(
            resolved.planning_device_budget_bytes,
            admission_policy.max_concurrent_queries,
        )?;
        let mut execution_config =
            resolved.execution_config.clone().with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: min_budget,
                max_source_chunk_bytes: resolved.max_source_chunk_bytes,
                max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
                max_retry_attempts: resolved.max_retry_attempts,
                min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
            });

        let service_config = backend_query_service_config(
            &execution_config,
            &admission_policy,
            &resolved,
            admission_policy.max_concurrent_queries,
        );
        let tenant_config = QueryServiceTenantConfig::with_grant_range(
            tenant_id.clone(),
            min_budget,
            resolved.planning_device_budget_bytes,
        );
        let device_ordinals = resolved.device_ordinals.clone();
        let query_service = if admission_only {
            QueryService::new_admission_only(service_config, vec![tenant_config], nongds_deployment())
        } else {
            QueryService::new_with_device_ordinals(
                service_config,
                resolved.device_ordinals.clone(),
                vec![tenant_config],
            )
        }
        .map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.query_service",
                "failed to initialize backend query service admission",
                error,
            )
        })?;
        execution_config = execution_config
            .with_query_admission_source(QueryAdmissionSource::new(query_service.clone(), tenant_id.clone()));
        let optimizer_config = optimizer_config.with_native_execution_config(execution_config);
        Ok(NexusGpuBackend { query_service, tenant_id, optimizer_config, device_ordinals })
    }
}

impl Default for NexusGpuBackendBuilder {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            memory_policy: NexusMemoryPolicy::default(),
            admission_policy: NexusAdmissionPolicy::default(),
            tenant_id: TenantId::new(DEFAULT_BACKEND_TENANT_ID),
        }
    }
}

impl NexusGpuBackendBuilder {
    pub fn fallback_policy(mut self, fallback_policy: GpuFallbackPolicy) -> Self {
        self.fallback_policy = fallback_policy;
        self
    }

    pub fn memory_policy(mut self, memory_policy: NexusMemoryPolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    pub fn admission(mut self, admission_policy: NexusAdmissionPolicy) -> Self {
        self.admission_policy = admission_policy;
        self
    }

    pub fn tenant_id(mut self, tenant_id: impl Into<TenantId>) -> Self {
        self.tenant_id = tenant_id.into();
        self
    }

    pub fn build(self) -> Result<NexusGpuBackend> {
        self.build_inner(false)
    }

    #[cfg(test)]
    pub(crate) fn build_admission_only(self) -> Result<NexusGpuBackend> {
        self.build_inner(true)
    }

    fn build_inner(self, admission_only: bool) -> Result<NexusGpuBackend> {
        let resolved = ResolvedBackendMemoryPolicy::resolve(&self.memory_policy)?;
        let optimizer_config = CudfOptimizerConfig::new(self.fallback_policy)
            .with_native_execution_config(resolved.execution_config.clone());
        NexusGpuBackend::build_bounded(
            optimizer_config,
            resolved,
            self.admission_policy,
            self.tenant_id,
            admission_only,
        )
    }
}

#[derive(Debug)]
struct ResolvedBackendMemoryPolicy {
    device_ordinals: Vec<usize>,
    planning_device_budget_bytes: u64,
    execution_config: NativeCudfExecutionConfig,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
}

impl ResolvedBackendMemoryPolicy {
    #[cfg(feature = "server")]
    fn from_execution_config(
        device_ordinals: Vec<usize>,
        execution_config: NativeCudfExecutionConfig,
    ) -> Result<Option<Self>> {
        validate_device_ordinals(&device_ordinals)?;
        let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = execution_config.memory_control_policy()
        else {
            return Ok(None);
        };
        validate_positive_u64("device_budget_bytes", device_budget_bytes)?;
        validate_positive_u64("max_source_chunk_bytes", max_source_chunk_bytes)?;
        validate_positive_usize("max_row_groups_per_chunk", max_row_groups_per_chunk)?;
        validate_positive_usize("max_retry_attempts", max_retry_attempts)?;
        validate_positive_usize("min_retry_chunk_row_groups", min_retry_chunk_row_groups)?;
        Ok(Some(Self {
            execution_config: execution_config.with_device_ordinal(device_ordinals[0]),
            device_ordinals,
            planning_device_budget_bytes: device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        }))
    }

    fn resolve(policy: &NexusMemoryPolicy) -> Result<Self> {
        match policy {
            NexusMemoryPolicy::BoundedAuto { device_ordinals, budget_fraction } => {
                validate_device_ordinals(device_ordinals)?;
                if let Some(fraction) = budget_fraction {
                    validate_budget_fraction(*fraction)?;
                }
                let first_device = device_ordinals[0];
                let budget = budget_fraction.map_or_else(
                    || NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device(first_device),
                    |fraction| {
                        NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                            first_device,
                            fraction,
                        )
                    },
                );
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    planning_device_budget_bytes: budget,
                    execution_config: NativeCudfExecutionConfig::bounded(
                        budget,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                    )
                    .with_device_ordinal(first_device),
                    max_source_chunk_bytes: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                    max_row_groups_per_chunk: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                    max_retry_attempts: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                    min_retry_chunk_row_groups: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                })
            },
            NexusMemoryPolicy::Bounded {
                device_ordinals,
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            } => {
                validate_device_ordinals(device_ordinals)?;
                validate_positive_u64("device_budget_bytes", *device_budget_bytes)?;
                validate_positive_u64("max_source_chunk_bytes", *max_source_chunk_bytes)?;
                validate_positive_usize("max_row_groups_per_chunk", *max_row_groups_per_chunk)?;
                validate_positive_usize("max_retry_attempts", *max_retry_attempts)?;
                validate_positive_usize("min_retry_chunk_row_groups", *min_retry_chunk_row_groups)?;
                let execution_config = NativeCudfExecutionConfig::bounded(
                    *device_budget_bytes,
                    *max_source_chunk_bytes,
                    *max_row_groups_per_chunk,
                    *max_retry_attempts,
                    *min_retry_chunk_row_groups,
                )
                .with_device_ordinal(device_ordinals[0]);
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    planning_device_budget_bytes: *device_budget_bytes,
                    execution_config,
                    max_source_chunk_bytes: *max_source_chunk_bytes,
                    max_row_groups_per_chunk: *max_row_groups_per_chunk,
                    max_retry_attempts: *max_retry_attempts,
                    min_retry_chunk_row_groups: *min_retry_chunk_row_groups,
                })
            },
            NexusMemoryPolicy::Unbounded { device_ordinals } => {
                validate_device_ordinals(device_ordinals)?;
                Err(invalid_backend_config("memory_policy", "NexusGpuBackend admission requires bounded memory policy"))
            },
        }
    }
}

fn backend_query_service_config(
    execution_config: &NativeCudfExecutionConfig,
    admission_policy: &NexusAdmissionPolicy,
    resolved: &ResolvedBackendMemoryPolicy,
    max_concurrent_queries: usize,
) -> QueryServiceConfig {
    QueryServiceConfig {
        device_ordinal: resolved.device_ordinals[0],
        source_fetch: execution_config.source_fetch_config().clone(),
        source_storage_profiles: execution_config.source_storage_profiles().to_vec(),
        boundary_policy: execution_config.boundary_policy(),
        source_scheduler_policy: execution_config.source_scheduler_policy(),
        source_chunk_planning_mode: execution_config.source_chunk_planning_mode(),
        native_pipeline_execution_mode: execution_config.native_pipeline_execution_mode(),
        stddev_samp_execution_mode: execution_config.stddev_samp_execution_mode(),
        dynamic_filter_execution_options: execution_config.dynamic_filter_execution_options(),
        keyed_hash_join_strategy_options: execution_config.keyed_hash_join_strategy_options(),
        aggregate_strategy_execution_options: execution_config.aggregate_strategy_execution_options(),
        source_chunk_planning_options: execution_config.source_chunk_planning_options(),
        stream_pool_size: Some(backend_stream_pool_size(
            max_concurrent_queries,
            execution_config.source_scheduler_policy().max_source_reads_in_flight(),
        )),
        device_capacity_bytes: Some(resolved.planning_device_budget_bytes),
        max_concurrent_queries,
        max_admission_queue_len: admission_policy.max_queued(),
        admission_wait_timeout: admission_policy.wait_timeout(),
        max_source_chunk_bytes: resolved.max_source_chunk_bytes,
        max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
        max_retry_attempts: resolved.max_retry_attempts,
        min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
        ..QueryServiceConfig::default()
    }
}

fn backend_stream_pool_size(max_concurrent_queries: usize, max_source_reads_in_flight: usize) -> usize {
    let per_query_streams = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_concurrent_queries.saturating_mul(per_query_streams).max(max_concurrent_queries)
}

fn admission_queue_len(max_concurrent_queries: usize) -> usize {
    max_concurrent_queries.saturating_mul(DEFAULT_ADMISSION_QUEUE_MULTIPLIER).max(DEFAULT_ADMISSION_QUEUE_MULTIPLIER)
}

fn partitioned_query_budget_bytes(capacity_bytes: u64, max_concurrent_queries: usize) -> Result<u64> {
    capacity_bytes
        .checked_div(u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX))
        .filter(|budget| *budget > 0)
        .ok_or_else(|| invalid_backend_config("device_budget_bytes", "is too small for max_concurrent_queries"))
}

fn validate_admission_policy(policy: &NexusAdmissionPolicy) -> Result<()> {
    validate_positive_usize("max_concurrent_queries", policy.max_concurrent_queries)?;
    validate_positive_usize("max_queued", policy.max_queued())?;
    if policy.wait_timeout.is_zero() {
        return Err(invalid_backend_config("wait_timeout", "must be greater than zero"));
    }
    Ok(())
}

fn validate_device_ordinals(device_ordinals: &[usize]) -> Result<()> {
    if device_ordinals.is_empty() {
        return Err(invalid_backend_config("device_ordinals", "must not be empty"));
    }
    let mut seen = BTreeSet::new();
    for ordinal in device_ordinals {
        if !seen.insert(*ordinal) {
            return Err(invalid_backend_config("device_ordinals", "must not contain duplicates"));
        }
    }
    Ok(())
}

fn validate_budget_fraction(fraction: f64) -> Result<()> {
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || fraction == 0.0 {
        return Err(invalid_backend_config("budget_fraction", "must be greater than zero and at most one"));
    }
    Ok(())
}

fn validate_positive_u64(field: &'static str, value: u64) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn validate_positive_usize(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn invalid_backend_config(field: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.backend.config", message).with_fact("field", field)
}

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[cfg(test)]
mod tests {
    use datafusion::{
        common::config::{ConfigOptions, ExtensionOptions},
        execution::SessionStateBuilder,
    };

    use super::*;
    use crate::session_config::DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS;

    #[test]
    fn backend_admission_only_lowers_to_query_service_and_planning_floor() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(2))
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().config().max_concurrent_queries, 2);
        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(backend.query_service().config().device_capacity_bytes, Some(4096));
        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 4096);
        assert_eq!(
            backend.optimizer_config().native_execution_config().memory_control_policy(),
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 2048,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            }
        );
    }

    #[test]
    fn backend_rejects_invalid_admission_policy() {
        let error = NexusGpuBackend::builder()
            .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(0))
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("max_concurrent_queries"));
    }

    #[test]
    fn backend_rejects_duplicate_devices() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::BoundedAuto { device_ordinals: vec![0, 0], budget_fraction: None })
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.fact_value("field"), Some("device_ordinals"));
    }

    #[test]
    fn backend_install_freezes_native_session_keys() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
        let extension = state
            .config_options()
            .extensions
            .get::<DataFusionNexusSessionConfig>()
            .expect("backend install must register datafusion-nexus session config");
        assert!(extension.is_backend_admission_frozen());

        for option in DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS {
            let mut extension = extension.clone();
            let error = extension.set(option.key, "1").unwrap_err();
            assert!(error.to_string().contains(&option.full_key()), "{} should reject", option.full_key());
            assert!(error.to_string().contains("NexusGpuBackend"));
        }
    }

    #[test]
    fn backend_freeze_rejects_direct_session_config_mutation_at_planning_time() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
        let mut extension = state
            .config_options()
            .extensions
            .get::<DataFusionNexusSessionConfig>()
            .expect("backend install must register datafusion-nexus session config")
            .clone();
        extension.native_device_budget_bytes += 1;
        let mut options = ConfigOptions::default();
        options.extensions.insert(extension);

        let error = backend.optimizer_config().with_datafusion_config_options(&options).unwrap_err();

        assert!(error.to_string().contains("native.device_budget_bytes"));
        assert!(error.to_string().contains("NexusGpuBackend"));
    }

    #[test]
    fn backend_admission_rejects_non_frozen_session_extension() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let mut options = ConfigOptions::default();
        options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(backend.optimizer_config()));

        let error = backend.optimizer_config().with_datafusion_config_options(&options).unwrap_err();

        assert!(error.to_string().contains("native.memory_policy"));
        assert!(error.to_string().contains("NexusGpuBackend"));
    }
}
