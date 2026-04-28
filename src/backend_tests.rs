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

fn backend_builder(device_ordinals: &[usize]) -> NexusGpuBackendBuilder {
    NexusGpuBackend::builder().device_memory_ownership(
        device_ordinals
            .iter()
            .copied()
            .map(|device_ordinal| (device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)),
    )
}

#[test]
fn backend_requires_an_exact_device_ownership_declaration_before_engine_build() {
    let missing = NexusGpuBackend::builder().build().unwrap_err();
    assert_eq!(missing.code(), ErrorCode::InvalidRequest);
    assert_eq!(missing.fact_value("field"), Some("device_memory_ownership"));
    assert_eq!(missing.fact_value("reason"), Some("missing_devices"));
    assert_eq!(missing.fact_value("device_ordinals"), Some("0"));

    let extra = backend_builder(&[0, 1]).build().unwrap_err();
    assert_eq!(extra.fact_value("reason"), Some("extra_devices"));
    assert_eq!(extra.fact_value("device_ordinals"), Some("1"));

    let duplicate = NexusGpuBackend::builder()
        .device_memory_ownership([
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
            (0, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 1 }),
        ])
        .build()
        .unwrap_err();
    assert_eq!(duplicate.fact_value("reason"), Some("duplicate_device"));
    assert_eq!(duplicate.fact_value("device_ordinal"), Some("0"));
}

#[test]
fn backend_exact_ownership_validation_preserves_static_partitions() {
    let declarations = [
        (2, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 }),
        (0, GpuMemoryOwnership::WholeDeviceExclusive),
    ];
    let validated = validate_device_memory_ownership(&[0, 2], &declarations).unwrap();
    assert_eq!(validated[&0], GpuMemoryOwnership::WholeDeviceExclusive);
    assert_eq!(validated[&2], GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 });
    assert_eq!(
        validated.into_iter().collect::<Vec<_>>(),
        vec![
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
            (2, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 }),
        ]
    );
}

#[test]
fn backend_test_controller_projection_lowers_policy_and_planning_floor() {
    let admission_policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(2);
    let backend = backend_builder(&[0, 1])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(admission_policy.clone())
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.controller_config().unwrap().max_concurrent_queries, 2);
    assert_eq!(backend.controller_config().unwrap().slot_guarantee, SlotGuarantee::GuaranteedMinimum);
    assert_eq!(backend.admission_policy(), &admission_policy);
    assert_eq!(backend.admission_policy().max_queries_per_device(), None);
    assert_eq!(backend.device_ordinals(), &[0, 1]);
    assert_eq!(backend.controller_config().unwrap().device_capacity_bytes, Some(4096));
    assert_eq!(backend.tenant_budget().unwrap(), 4096);
    assert_eq!(backend.admitted_grant_bytes().unwrap(), 3584);
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
fn backend_propagates_static_partition_to_pure_engine_validation() {
    let error = NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 0 })])
        .build()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.operation(), "datafusion_nexus.backend.gpu_core");
    assert_eq!(error.fact_value("field"), Some("memory_ownership.allocation_limit_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn backend_rejects_global_concurrency_below_device_count_without_inflating_it() {
    let error = backend_builder(&[0, 1])
        .memory_policy(NexusMemoryPolicy::BoundedAuto { device_ordinals: vec![0, 1], budget_fraction: None })
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(1))
        .build()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("max_concurrent_queries"));
    assert_eq!(error.fact_value("reason"), Some("below_device_count"));
    assert_eq!(error.fact_value("max_concurrent_queries"), Some("1"));
    assert_eq!(error.fact_value("device_count"), Some("2"));

    let policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(1);
    let distributed = distributed_active_query_limits(&policy, 2).unwrap_err();
    assert_eq!(distributed.fact_value("reason"), Some("below_device_count"));
}

#[test]
fn backend_rejects_query_service_local_parquet_source_cache_domain_in_p0() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(2048));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0, 1],
        device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 },
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = build_test_controller_projection(
        optimizer_config,
        resolved,
        vec![(0, GpuMemoryOwnership::WholeDeviceExclusive), (1, GpuMemoryOwnership::WholeDeviceExclusive)],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(2),
        TenantId::new("tenant"),
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[test]
fn backend_rejects_zero_query_service_local_parquet_source_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(0));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0],
        device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 },
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = build_test_controller_projection(
        optimizer_config,
        resolved,
        vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
        TenantId::new("tenant"),
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[test]
fn backend_builder_rejects_query_service_local_parquet_source_cache_domain_in_p0() {
    let error = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[test]
fn backend_builder_propagates_query_memory_domain_policy() {
    let policy = QueryMemoryDomainPolicy::new().with_ceiling_multiplier(3);
    let backend = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.controller_config().unwrap().query_memory_domain, Some(policy));
    assert_eq!(backend.optimizer_config().native_execution_config().query_service_query_memory_domain(), Some(policy));
}

#[test]
fn backend_builder_rejects_zero_query_service_local_parquet_source_cache_capacity() {
    let error = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_rejects_query_service_resident_graph_cache_domain_in_p0() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_resident_graph_cache_max_retained_bytes(Some(2048));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0, 1],
        device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 },
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = build_test_controller_projection(
        optimizer_config,
        resolved,
        vec![(0, GpuMemoryOwnership::WholeDeviceExclusive), (1, GpuMemoryOwnership::WholeDeviceExclusive)],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(2),
        TenantId::new("tenant"),
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_rejects_zero_query_service_resident_graph_cache_capacity() {
    let execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 1, 3, 1)
        .with_query_service_resident_graph_cache_max_retained_bytes(Some(0));
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config.clone());
    let resolved = ResolvedBackendMemoryPolicy {
        device_ordinals: vec![0],
        device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 },
        execution_config,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let error = build_test_controller_projection(
        optimizer_config,
        resolved,
        vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
        TenantId::new("tenant"),
    )
    .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_builder_rejects_query_service_resident_graph_cache_domain_in_p0() {
    let error = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[cfg(feature = "cugraph")]
#[test]
fn backend_builder_rejects_zero_query_service_resident_graph_cache_capacity() {
    let error = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("field"), Some("cache_domains"));
    assert_eq!(error.fact_value("reason"), Some("cache_domains_must_be_disabled"));
}

#[test]
fn backend_rejects_invalid_admission_policy() {
    let error = backend_builder(&[0])
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(0))
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("max_concurrent_queries"));
}

#[test]
fn backend_admission_policy_propagates_placement_and_per_device_cap() {
    let admission_policy = NexusAdmissionPolicy::default()
        .with_max_concurrent_queries(2)
        .with_max_queries_per_device(Some(2))
        .with_placement_policy(QueryPlacementPolicy::BestFit);
    let backend = backend_builder(&[0, 1])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(admission_policy.clone())
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.controller_config().unwrap().max_queries_per_device, Some(2));
    assert_eq!(backend.controller_config().unwrap().placement_policy, QueryPlacementPolicy::BestFit);
    assert_eq!(backend.admission_policy(), &admission_policy);
}

#[test]
fn backend_test_controller_projection_preserves_pinned_device_ordinal() {
    let backend = backend_builder(&[0, 1])
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
                .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 1 }),
        )
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.controller_config().unwrap().device_ordinal, 1);
    assert_eq!(
        backend.controller_config().unwrap().placement_policy,
        QueryPlacementPolicy::Pinned { device_ordinal: 1 }
    );
}

#[test]
fn backend_rejects_pinned_placement_outside_configured_devices() {
    let error = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("placement_policy"));
    assert_eq!(error.fact_value("pinned_device_ordinal"), Some("1"));
}

#[test]
fn backend_rejects_invalid_grant_band() {
    let error = backend_builder(&[0])
        .admission(
            NexusAdmissionPolicy::default()
                .with_grant_band(GrantBand { min: MemoryShare::Bytes(2048), want: MemoryShare::Bytes(1024) }),
        )
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.want"));
}

#[test]
fn backend_rejects_policy_default_grant_min() {
    let error = backend_builder(&[0])
        .admission(
            NexusAdmissionPolicy::default()
                .with_grant_band(GrantBand { min: MemoryShare::PolicyDefault, want: MemoryShare::PolicyDefault }),
        )
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_guaranteed_minimum_grant_floor_that_cannot_fill_slots() {
    let error = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_optimistic_slot_guarantee_allows_floor_above_empty_slot_reservation() {
    let backend = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.controller_config().unwrap().slot_guarantee, SlotGuarantee::Optimistic);
    assert_eq!(backend.tenant_budget().unwrap(), 60);
}

#[test]
fn backend_default_optimistic_strict_want_keeps_per_device_arrival_cushion() {
    let backend = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_uses_effective_slots_per_device() {
    let backend = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_uses_pinned_device_slots() {
    let backend = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 600);
}

#[test]
fn backend_default_optimistic_strict_want_respects_per_device_cap() {
    let backend = backend_builder(&[0, 1])
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
                .with_max_concurrent_queries(4)
                .with_max_queries_per_device(Some(2))
                .with_slot_guarantee(SlotGuarantee::Optimistic),
        )
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 700);
}

#[test]
fn backend_policy_default_grant_want_round_trips_without_pinning_full_capacity() {
    let policy =
        NexusAdmissionPolicy::default().with_max_concurrent_queries(5).with_slot_guarantee(SlotGuarantee::Optimistic);
    assert_eq!(policy.grant().want, MemoryShare::PolicyDefault);
    let backend = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 400);
}

#[cfg(feature = "server")]
#[test]
fn backend_default_optimistic_adaptive_want_uses_full_capacity() {
    let adaptive = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded(800, 64, 1, 1, 1)
            .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive),
    );
    let adaptive = test_controller_projection_from_optimizer_config(
        adaptive,
        vec![0],
        vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        NexusAdmissionPolicy::default().with_max_concurrent_queries(3).with_slot_guarantee(SlotGuarantee::Optimistic),
        TenantId::new("default"),
    )
    .unwrap()
    .unwrap();

    assert_eq!(adaptive.tenant_budget().unwrap(), 800);
}

#[test]
fn backend_explicit_grant_want_overrides_optimistic_strict_default() {
    let backend = backend_builder(&[0])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.tenant_budget().unwrap(), 700);
}

#[test]
fn backend_guaranteed_minimum_floor_counts_selected_devices() {
    let backend = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.device_ordinals(), &[0, 1]);
    assert_eq!(backend.tenant_budget().unwrap(), 100);
}

#[test]
fn backend_rejects_guaranteed_minimum_floor_that_only_fits_as_aggregate_bytes() {
    let error = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_pinned_guaranteed_minimum_floor_that_needs_unpinned_device() {
    let error = backend_builder(&[0, 1])
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
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("grant.min"));
}

#[test]
fn backend_rejects_global_concurrency_above_aggregate_per_device_cap() {
    let error = backend_builder(&[0, 1])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0, 1],
            device_budget_bytes: 100,
            max_source_chunk_bytes: 64,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(3).with_max_queries_per_device(Some(1)))
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("field"), Some("max_queries_per_device"));
    assert_eq!(error.fact_value("device_count"), Some("2"));
    assert_eq!(error.fact_value("max_queries_per_device"), Some("1"));
    assert_eq!(error.fact_value("max_concurrent_queries"), Some("3"));
}

#[test]
fn backend_rejects_duplicate_devices() {
    let error = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::BoundedAuto { device_ordinals: vec![0, 0], budget_fraction: None })
        .build_test_controller_projection()
        .unwrap_err();

    assert_eq!(error.fact_value("field"), Some("device_ordinals"));
}

#[test]
fn backend_install_freezes_native_session_keys() {
    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
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
    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .with_no_datafusion_cpu_requirement()
        .build_test_controller_projection()
        .unwrap();

    assert_eq!(backend.optimizer_config().final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);

    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
    let installed = installed_nexus_native_optimizer_config(&state).expect("backend install must retain native config");
    assert_eq!(installed.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
}

#[test]
fn backend_install_rejects_conflicting_preexisting_session_extension() {
    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
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

#[test]
fn backend_install_rejects_preexisting_cugraph_sql_registration() {
    use crate::{cugraph_sql::CugraphSqlConfig, session::NexusSessionStateBuilderExt};

    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
        .unwrap();

    let builder = SessionStateBuilder::new().with_cugraph_sql(CugraphSqlConfig::default());
    let error = backend.install_on(builder).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("preexisting_cugraph_sql_without_backend_admission"));
    assert!(error.fact_value("table_function").is_some_and(|name| name.starts_with("cugraph_")));
}

#[test]
fn backend_install_with_cugraph_sql_registers_the_public_family() {
    use crate::cugraph_sql::CugraphSqlConfig;

    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
        .unwrap();

    let state = backend
        .install_on_with_cugraph_sql(SessionStateBuilder::new(), CugraphSqlConfig::default())
        .expect("backend installer must own cuGraph SQL registration")
        .build();
    assert!(state.table_functions().contains_key("cugraph_pagerank"));
}

#[test]
fn backend_freeze_rejects_direct_session_config_mutation_at_planning_time() {
    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
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
    let backend = backend_builder(&[0])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 4096,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build_test_controller_projection()
        .unwrap();
    let mut options = ConfigOptions::default();
    options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(backend.optimizer_config()));

    let error = backend.optimizer_config().with_datafusion_config_options(&options).unwrap_err();

    assert!(error.to_string().contains("session config must be frozen"));
    assert!(error.to_string().contains("query-service admission"));
}
