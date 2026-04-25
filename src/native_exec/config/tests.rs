use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        DeploymentIoCapability, DeviceFacts, GdsDriverProbeStatus, IoLifecycle, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryService, QueryServiceConfig, QueryServiceTenantConfig, TenantId,
    },
};

#[cfg(debug_assertions)]
use super::context::StandaloneBoundedContextDebugGuard;
use super::{
    device_budget::{
        auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory,
        auto_sized_device_budget_bytes_for_device_with_inventory, auto_sized_device_budget_bytes_with_inventory,
    },
    *,
};
use crate::native_exec::admission::{QueryAdmissionTicket, task_context_with_query_admission_ticket};

enum FakeInventory {
    Facts(DeviceFacts),
    Failure,
}

impl GpuInventory for FakeInventory {
    fn device_count(&self) -> Result<usize> {
        Ok(1)
    }

    fn facts(&self, _device_ordinal: usize) -> Result<DeviceFacts> {
        match self {
            Self::Facts(facts) => Ok(facts.clone()),
            Self::Failure => Err(QueryError::new(
                QueryErrorCode::CudfTemporaryFailure,
                "datafusion_nexus.test.gpu_inventory",
                "simulated inventory failure",
            )),
        }
    }
}

fn device_facts(total_bytes: u64) -> DeviceFacts {
    DeviceFacts { device: 0, total_bytes, free_bytes: total_bytes / 2, sm_count: Some(16), utilization_pct: None }
}

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[test]
fn auto_sized_budget_uses_total_memory_minus_headroom() {
    let total = 8 * 1024 * 1024 * 1024;
    let inventory = FakeInventory::Facts(device_facts(total));

    assert_eq!(
        auto_sized_device_budget_bytes_with_inventory(&inventory),
        total - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
    );
}

#[test]
fn auto_sized_budget_falls_back_when_inventory_fails() {
    let inventory = FakeInventory::Failure;

    assert_eq!(
        auto_sized_device_budget_bytes_with_inventory(&inventory),
        NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES
    );
}

#[test]
fn auto_sized_budget_for_device_uses_requested_ordinal() {
    struct OrdinalInventory;

    impl GpuInventory for OrdinalInventory {
        fn device_count(&self) -> Result<usize> {
            Ok(2)
        }

        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            Ok(device_facts(match device_ordinal {
                0 => 8 * 1024 * 1024 * 1024,
                1 => 16 * 1024 * 1024 * 1024,
                other => panic!("unexpected device ordinal {other}"),
            }))
        }
    }

    assert_eq!(
        auto_sized_device_budget_bytes_for_device_with_inventory(&OrdinalInventory, 1),
        16 * 1024 * 1024 * 1024 - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
    );
}

#[test]
fn auto_sized_budget_fraction_preserves_headroom_floor() {
    let total = 8 * 1024 * 1024 * 1024;
    let inventory = FakeInventory::Facts(device_facts(total));

    assert_eq!(
        auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(&inventory, 0, 1.0),
        total - NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES
    );
    assert_eq!(auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(&inventory, 0, 0.5), total / 2);
}

#[test]
fn execution_context_carries_configured_correlation_ids() {
    let context = NativeCudfExecutionConfig::unbounded()
        .with_query_id("query-config")
        .with_trace_id("trace-config")
        .execution_context()
        .unwrap();

    assert_eq!(context.context().query_id().map(QueryId::as_str), Some("query-config"));
    assert_eq!(context.context().trace_id().map(TraceId::as_str), Some("trace-config"));
}

#[test]
fn standalone_bounded_and_unbounded_contexts_reject_parquet_io_after_generation_close() {
    let configs =
        [NativeCudfExecutionConfig::bounded(1024 * 1024, 512 * 1024, 1, 1, 1), NativeCudfExecutionConfig::unbounded()];

    for config in configs {
        let lifecycle = IoLifecycle::new();
        let context = config.with_io_lifecycle(lifecycle.clone()).execution_context().unwrap();
        lifecycle.close();

        let error = context.context().acquire_io_lease("datafusion_nexus.test.parquet_io.after_close").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("io_lifecycle_closed"));
    }
}

#[test]
fn acquired_query_service_context_is_stamped_with_server_io_generation() {
    let lifecycle = IoLifecycle::new();
    let config =
        NativeCudfExecutionConfig::bounded(1024 * 1024, 512 * 1024, 1, 1, 1).with_io_lifecycle(lifecycle.clone());
    let context = NativeCudfExecutionContext::standalone(ExecutionContext::builder().build().unwrap(), None);
    let context = config.stamp_acquired_service_context(context).unwrap();
    lifecycle.close();

    let error = context.context().acquire_io_lease("datafusion_nexus.test.service_parquet_io.after_close").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("io_lifecycle_closed"));
}

#[test]
fn service_acquired_context_preserves_service_graph_calibration_without_adapter_override() {
    let service_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
    let config = NativeCudfExecutionConfig::bounded_auto_sized();

    assert_eq!(
        config.graph_memory_calibration_override_for_acquired_service_context(service_override),
        service_override
    );
}

#[test]
fn service_acquired_context_applies_only_configured_adapter_graph_calibration_fields() {
    let service_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);

    let raw_reset = NativeCudfExecutionConfig::bounded_auto_sized().with_graph_raw_input_edges_override(None);
    assert_eq!(
        raw_reset.graph_memory_calibration_override_for_acquired_service_context(service_override),
        GraphMemoryCalibrationOverride::default()
    );

    let full_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(99);
    let config = NativeCudfExecutionConfig::bounded_auto_sized().with_graph_memory_calibration_override(full_override);
    assert_eq!(config.graph_memory_calibration_override_for_acquired_service_context(service_override), full_override);
}

#[cfg(debug_assertions)]
#[test]
fn standalone_bounded_debug_guard_is_per_device_and_releases() {
    {
        let _first = StandaloneBoundedContextDebugGuard::new(10_001);
        let _other_device = StandaloneBoundedContextDebugGuard::new(10_002);
    }

    let _again_after_release = StandaloneBoundedContextDebugGuard::new(10_001);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "concurrent unadmitted bounded native contexts")]
fn standalone_bounded_debug_guard_panics_on_same_device_overlap() {
    let _first = StandaloneBoundedContextDebugGuard::new(10_003);
    let _second = StandaloneBoundedContextDebugGuard::new(10_003);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "concurrent unadmitted bounded native contexts")]
fn standalone_bounded_debug_guard_panics_on_same_task_context_owner_overlap() {
    let _first = StandaloneBoundedContextDebugGuard::new_with_owner(10_004, Some(42));
    let _nested = StandaloneBoundedContextDebugGuard::new_with_owner(10_004, Some(42));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "concurrent unadmitted bounded native contexts")]
fn standalone_bounded_debug_guard_panics_on_different_task_context_owner_overlap() {
    let _first = StandaloneBoundedContextDebugGuard::new_with_owner(10_005, Some(42));
    let _second = StandaloneBoundedContextDebugGuard::new_with_owner(10_005, Some(43));
}

#[test]
fn plan_query_admission_ticket_is_shared_by_config_clones() {
    let tenant_id = TenantId::new("tenant");
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let service = QueryService::new_admission_only(
        QueryServiceConfig { device_capacity_bytes: Some(1024 * 1024), ..QueryServiceConfig::default() },
        vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), 256, 1024 * 1024)],
        deployment,
    )
    .unwrap();
    let source = QueryAdmissionSource::new(service, tenant_id);
    let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1)
        .with_query_id("ticket-query")
        .with_trace_id("ticket-trace")
        .with_query_admission_source(source)
        .with_requested_grant_min_bytes(Some(384))
        .with_requested_grant_bytes(Some(512));
    let config = base.clone().with_plan_query_admission_ticket();

    let cloned = config.clone().with_plan_query_admission_ticket();
    let separate_plan = base.with_plan_query_admission_ticket();

    assert_eq!(config.query_admission_ticket_id(), cloned.query_admission_ticket_id());
    assert_ne!(config.query_admission_ticket_id(), separate_plan.query_admission_ticket_id());
    let ticket = config.query_admission_ticket().expect("plan ticket should be created");
    assert_eq!(ticket.requested_grant_min_bytes(), Some(384));
    assert_eq!(ticket.requested_grant_bytes(), Some(512));
    let owner = ticket.observation_owner();
    assert_eq!(owner.query_id().map(|query_id| query_id.as_str()), Some("ticket-query"));
    assert_eq!(owner.trace_id().map(|trace_id| trace_id.as_str()), Some("ticket-trace"));
}

#[test]
fn query_service_handle_clears_plan_admission_source_to_prevent_double_admission() {
    let tenant_id = TenantId::new("tenant");
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let service = QueryService::new_admission_only(
        QueryServiceConfig {
            device_capacity_bytes: Some(1024 * 1024),
            max_concurrent_queries: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), 256, 1024 * 1024)],
        deployment,
    )
    .unwrap();
    let source = QueryAdmissionSource::new(service.clone(), tenant_id.clone());
    let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1)
        .with_query_admission_source(source)
        .with_requested_grant_min_bytes(Some(384))
        .with_requested_grant_bytes(Some(512))
        .with_plan_query_admission_ticket();
    let handle = Arc::new(service.try_admit_query(tenant_id.clone()).unwrap().expect("query should admit"));

    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 0);
    let config = base.with_query_service_handle(handle);

    assert!(config.has_query_service_handle());
    assert!(config.query_admission_source.is_none());
    assert!(config.query_admission_ticket().is_none());
    assert_eq!(config.requested_grant_min_bytes(), None);
    assert_eq!(config.requested_grant_bytes(), None);
    let snapshot = service.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1);
    assert_eq!(snapshot.queued_queries, 0);
    assert!(service.try_admit_query(tenant_id).unwrap().is_none());
}

#[test]
fn query_admission_ticket_selection_prefers_ambient_query_admission_ticket() {
    let root_tenant = TenantId::new("root");
    let root_service = QueryService::new_admission_only(
        QueryServiceConfig {
            device_capacity_bytes: Some(2048),
            max_concurrent_queries: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(root_tenant.clone(), 256, 2048)],
        nongds_deployment(),
    )
    .unwrap();
    let child_tenant = TenantId::new("child");
    let child_service = QueryService::new_admission_only(
        QueryServiceConfig {
            device_capacity_bytes: Some(2048),
            max_concurrent_queries: 1,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::with_grant_range(child_tenant.clone(), 256, 2048)],
        nongds_deployment(),
    )
    .unwrap();
    let ambient_ticket =
        QueryAdmissionTicket::new(QueryAdmissionSource::new(root_service.clone(), root_tenant), None, Some(1536));
    let child_config = NativeCudfExecutionConfig::bounded(256, 512, 1, 3, 1)
        .with_query_admission_source(QueryAdmissionSource::new(child_service.clone(), child_tenant))
        .with_requested_grant_bytes(Some(512))
        .with_plan_query_admission_ticket();
    let child_ticket_id = child_config.query_admission_ticket_id();
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ambient_ticket));

    assert_ne!(Some(ambient_ticket.id()), child_ticket_id);
    assert_eq!(
        child_config.query_admission_ticket_id_for_task_context(&task_context),
        Some(ambient_ticket.id()),
        "embedded child config ticket must not win when an ambient root ticket exists"
    );
}

#[test]
fn query_service_handle_admission_facts_record_once_across_config_clones() {
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let service = QueryService::new_admission_only(
        QueryServiceConfig { max_concurrent_queries: 1, ..QueryServiceConfig::default() },
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        deployment,
    )
    .unwrap();
    let handle = Arc::new(service.try_admit_query("tenant").unwrap().expect("query should admit"));
    let config = NativeCudfExecutionConfig::bounded_auto_sized().with_query_service_handle(handle);
    let cloned = config.clone();
    let marker = config.query_admission_facts_recorded.as_ref().expect("direct handle configs must have a marker");
    let cloned_marker =
        cloned.query_admission_facts_recorded.as_ref().expect("cloned direct handle configs must have a marker");

    assert!(Arc::ptr_eq(marker, cloned_marker));
    assert!(!marker.swap(true, Ordering::AcqRel));
    assert!(cloned_marker.swap(true, Ordering::AcqRel));
}

#[test]
fn config_validation_errors_carry_configured_correlation_ids() {
    let error = NativeCudfExecutionConfig::unbounded()
        .with_query_id("query-config-error")
        .with_trace_id("trace-config-error")
        .with_source_chunk_planning_options(
            SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(0),
        )
        .validate_query_policies()
        .unwrap_err();

    assert_eq!(error.fact_value("query_id"), Some("query-config-error"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-config-error"));
    let native = error
        .source_error()
        .and_then(|source| source.downcast_ref::<QueryError>())
        .expect("adapter error should retain native source error");
    assert_eq!(native.query_id().map(QueryId::as_str), Some("query-config-error"));
    assert_eq!(native.trace_id().map(TraceId::as_str), Some("trace-config-error"));
}
