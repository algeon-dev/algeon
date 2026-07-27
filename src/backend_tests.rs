use datafusion::{
    common::config::{ConfigOptions, ExtensionOptions},
    execution::SessionStateBuilder,
    prelude::SessionConfig,
};

use super::*;
use crate::{
    planner::FinalPlanRequirement, session::installed_nexus_native_optimizer_config,
    session_config::DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS,
};

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
    assert_eq!(backend.query_service().config().slot_guarantee, SlotGuarantee::GuaranteedMinimum);
    assert_eq!(backend.device_ordinals(), &[0, 1]);
    assert_eq!(backend.query_service().config().device_capacity_bytes, Some(4096));
    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 4096);
    let handle = backend
        .query_service()
        .try_admit_query(backend.tenant_id().clone())
        .unwrap()
        .expect("first query should admit");
    assert_eq!(handle.device_budget_bytes(), 3584);
    assert_eq!(
        backend.optimizer_config().native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 512,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    );
}

#[test]
fn backend_propagates_query_service_local_parquet_source_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(2048));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0, 1],
        planning_device_budget_bytes: 4096,
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let backend = NexusGpuBackend::build_bounded(
        optimizer_config,
        resolved,
        NexusAdmissionPolicy::default().with_max_concurrent_queries(2),
        TenantId::new("tenant"),
        true,
    )
    .unwrap();

    assert_eq!(backend.query_service().config().local_parquet_source_cache_max_retained_bytes, Some(2048));
    assert_eq!(
        backend
            .optimizer_config()
            .native_execution_config()
            .query_service_local_parquet_source_cache_max_retained_bytes(),
        Some(2048)
    );
}

#[test]
fn backend_rejects_zero_query_service_local_parquet_source_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(0));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0],
        planning_device_budget_bytes: 4096,
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = NexusGpuBackend::build_bounded(
        optimizer_config,
        resolved,
        NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
        TenantId::new("tenant"),
        true,
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_source_cache_max_retained_bytes"));
}

#[test]
fn backend_builder_propagates_query_service_local_parquet_source_cache_capacity() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .query_service_local_parquet_source_cache_max_retained_bytes(Some(2048))
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(2))
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().local_parquet_source_cache_max_retained_bytes, Some(2048));
    assert_eq!(
        backend
            .optimizer_config()
            .native_execution_config()
            .query_service_local_parquet_source_cache_max_retained_bytes(),
        Some(2048)
    );
}

#[test]
fn backend_builder_propagates_query_memory_domain_policy() {
    let policy = QueryMemoryDomainPolicy::new().with_ceiling_multiplier(3);
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .query_service_query_memory_domain(Some(policy))
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(2))
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().query_memory_domain, Some(policy));
    assert_eq!(backend.optimizer_config().native_execution_config().query_service_query_memory_domain(), Some(policy));
}

#[test]
fn backend_builder_rejects_zero_query_service_local_parquet_source_cache_capacity() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .query_service_local_parquet_source_cache_max_retained_bytes(Some(0))
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(1))
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_source_cache_max_retained_bytes"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_propagates_query_service_resident_graph_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_resident_graph_cache_max_retained_bytes(Some(2048));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0, 1],
        planning_device_budget_bytes: 4096,
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let backend = NexusGpuBackend::build_bounded(
        optimizer_config,
        resolved,
        NexusAdmissionPolicy::default().with_max_concurrent_queries(2),
        TenantId::new("tenant"),
        true,
    )
    .unwrap();

    assert_eq!(backend.query_service().config().resident_graph_cache_max_retained_bytes, Some(2048));
    assert_eq!(
        backend.optimizer_config().native_execution_config().query_service_resident_graph_cache_max_retained_bytes(),
        Some(2048)
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_rejects_zero_query_service_resident_graph_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_resident_graph_cache_max_retained_bytes(Some(0));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0],
        planning_device_budget_bytes: 4096,
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = NexusGpuBackend::build_bounded(
        optimizer_config,
        resolved,
        NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
        TenantId::new("tenant"),
        true,
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("resident_graph_cache_max_retained_bytes"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_builder_propagates_query_service_resident_graph_cache_capacity() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .query_service_resident_graph_cache_max_retained_bytes(Some(2048))
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(2))
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().resident_graph_cache_max_retained_bytes, Some(2048));
    assert_eq!(
        backend.optimizer_config().native_execution_config().query_service_resident_graph_cache_max_retained_bytes(),
        Some(2048)
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_builder_rejects_zero_query_service_resident_graph_cache_capacity() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .query_service_resident_graph_cache_max_retained_bytes(Some(0))
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(1))
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("resident_graph_cache_max_retained_bytes"));
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
fn backend_admission_policy_propagates_placement_and_per_device_cap() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(2)
                .with_max_queries_per_device(Some(2))
                .with_placement_policy(QueryPlacementPolicy::BestFit),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().max_queries_per_device, Some(2));
    assert_eq!(backend.query_service().config().placement_policy, QueryPlacementPolicy::BestFit);
}

#[test]
fn backend_admission_only_pinned_placement_sets_service_device_ordinal() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(1)
                .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 1 }),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().device_ordinal, 1);
    assert_eq!(backend.query_service().config().placement_policy, QueryPlacementPolicy::Pinned { device_ordinal: 1 });
}

#[test]
fn backend_rejects_pinned_placement_outside_configured_devices() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(1)
                .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 1 }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("placement_policy"));
    assert_eq!(error.fact_value("pinned_device_ordinal"), Some("1"));
}

#[test]
fn backend_rejects_invalid_grant_band() {
    let error = NexusGpuBackend::builder()
        .admission(
            NexusAdmissionPolicy::default()
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(2048), want: MemoryShare::Bytes(1024) }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.want"));
}

#[test]
fn backend_rejects_policy_default_grant_min() {
    let error = NexusGpuBackend::builder()
        .admission(
            NexusAdmissionPolicy::default()
                .with_grant_band(GrantBand { min: MemoryShare::PolicyDefault, want: MemoryShare::PolicyDefault }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_guaranteed_minimum_grant_floor_that_cannot_fill_slots() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(2)
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_optimistic_slot_guarantee_allows_floor_above_empty_slot_reservation() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(2)
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) })
                .with_slot_guarantee(SlotGuarantee::Optimistic),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().config().slot_guarantee, SlotGuarantee::Optimistic);
    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 60);
}

#[test]
fn backend_default_optimistic_strict_want_keeps_per_device_arrival_cushion() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_slot_guarantee(SlotGuarantee::Optimistic),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_uses_effective_slots_per_device() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(5)
                .with_slot_guarantee(SlotGuarantee::Optimistic),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_uses_pinned_device_slots() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_slot_guarantee(SlotGuarantee::Optimistic)
                .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 0 }),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_respects_per_device_cap() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(5)
                .with_max_queries_per_device(Some(2))
                .with_slot_guarantee(SlotGuarantee::Optimistic),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 700);
}

#[test]
fn backend_policy_default_grant_want_round_trips_without_pinning_full_capacity() {
    let policy =
        NexusAdmissionPolicy::default().with_max_concurrent_queries(5).with_slot_guarantee(SlotGuarantee::Optimistic);
    assert_eq!(policy.grant().want, MemoryShare::PolicyDefault);
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(policy.max_concurrent_queries())
                .with_slot_guarantee(policy.slot_guarantee())
                .with_grant_band(policy.grant()),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 400);
}

#[cfg(feature = "server")]
#[test]
fn backend_default_optimistic_adaptive_want_uses_full_capacity() {
    let adaptive = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded(800, 64, 1, 1, 1)
            .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive),
    );
    let adaptive = NexusGpuBackend::from_optimizer_config_for_admission_only(
        adaptive,
        vec![0],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(3).with_slot_guarantee(SlotGuarantee::Optimistic),
        TenantId::new("default"),
    )
    .unwrap()
    .unwrap();

    assert_eq!(adaptive.query_service().tenant_budget(adaptive.tenant_id()).unwrap(), 800);
}

#[test]
fn backend_explicit_grant_want_overrides_optimistic_strict_default() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 800,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_slot_guarantee(SlotGuarantee::Optimistic)
                .with_grant_want(MemoryShare::Bytes(700)),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 700);
}

#[test]
fn backend_guaranteed_minimum_floor_counts_selected_devices() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 200,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(2)
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(100), want: MemoryShare::Bytes(100) }),
        )
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.device_ordinals(), &[0, 1]);
    assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 100);
}

#[test]
fn backend_rejects_guaranteed_minimum_floor_that_only_fits_as_aggregate_bytes() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_pinned_guaranteed_minimum_floor_that_needs_unpinned_device() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(2)
                .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 0 })
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_guaranteed_minimum_floor_that_needs_more_slots_than_per_device_cap() {
    let error = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_max_queries_per_device(Some(1))
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(50), want: MemoryShare::Bytes(50) }),
        )
        .build_admission_only()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("max_queries_per_device"));
    assert_eq!(error.fact_value("bound_field"), Some("max_concurrent_queries"));
    assert_eq!(error.fact_value("max_queries_per_device"), Some("1"));
    assert_eq!(error.fact_value("placement_eligible_device_count"), Some("2"));
    assert_eq!(error.fact_value("max_concurrent_queries"), Some("3"));
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
        if option.key.starts_with("request.") {
            extension.set(option.key, "1").unwrap();
            continue;
        }
        let error = extension.set(option.key, "1").unwrap_err();
        assert!(error.to_string().contains(&option.full_key()), "{} should reject", option.full_key());
        assert!(error.to_string().contains("query-service admission"));
    }
}

#[test]
fn backend_installs_configured_final_plan_requirements() {
    let backend = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .with_no_datafusion_cpu_requirement()
        .build_admission_only()
        .unwrap();

    assert_eq!(backend.optimizer_config().final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);

    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
    let installed = installed_nexus_native_optimizer_config(&state).expect("backend install must retain native config");
    assert_eq!(installed.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
}

#[test]
fn backend_install_rejects_conflicting_preexisting_session_extension() {
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
    let mut config = SessionConfig::new();
    config
        .options_mut()
        .extensions
        .insert(DataFusionNexusSessionConfig::from_optimizer_config(&NexusNativeOptimizerConfig::default()));

    let error = backend.install_on(SessionStateBuilder::new().with_config(config)).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("session_config_conflicts_with_backend_admission"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_install_rejects_preexisting_cugraph_sql_registration() {
    use crate::{cugraph_sql::CugraphSqlConfig, session::NexusSessionStateBuilderExt};

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

    let builder = SessionStateBuilder::new().with_cugraph_sql(CugraphSqlConfig::default());
    let error = backend.install_on(builder).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("preexisting_cugraph_sql_without_backend_admission"));
    assert!(error.fact_value("table_function").is_some_and(|name| name.starts_with("cugraph_")));
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
    assert!(error.to_string().contains("query-service admission"));
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

    assert!(error.to_string().contains("session config must be frozen"));
    assert!(error.to_string().contains("query-service admission"));
}
