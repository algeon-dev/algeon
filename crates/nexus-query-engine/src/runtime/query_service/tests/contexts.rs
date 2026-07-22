#[test]
fn strict_query_service_context_uses_hard_limited_query_domain_members() {
    let config = QueryServiceConfig {
        native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();
    let grant = handle.lifetime.memory_grant().unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let token = tenant_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("test_strict_policy"))
        .unwrap();

    assert_eq!(token.enforcement(), cudf::ExecutionBudgetEnforcement::QueryDomainMember);
    assert_eq!(grant.hard_cap_bytes(), Some(16 * 1024 * 1024));
}
#[test]
fn concurrent_runtimes_single_flight_query_memory_domain_creation() {
    const CONTEXTS: usize = 8;
    let config = QueryServiceConfig {
        max_concurrent_queries: CONTEXTS,
        native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
        query_memory_domain: Some(QueryMemoryDomainPolicy::default()),
        slot_guarantee: SlotGuarantee::Optimistic,
        stream_pool_size: Some(CONTEXTS * 3),
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();
    let handle = service.admit_query("tenant").unwrap();
    let grant = handle.lifetime.memory_grant().unwrap();
    let grant_key = grant.identity_key();
    let device = service.inner.resources.as_ref().unwrap().devices.context_by_ordinal(handle.device_ordinal).unwrap();
    let capacity_bytes = device.capacity_bytes();
    let runtimes = (0..CONTEXTS)
        .map(|_| device.stream_pool().acquire().unwrap())
        .map(|lease| (lease.shared_runtime().unwrap(), lease))
        .collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(CONTEXTS));
    let threads = runtimes
        .into_iter()
        .map(|(runtime, lease)| {
            let grant = grant.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let domain = super::super::memory_domain::create_grant_memory_domain(
                    runtime.as_ref(),
                    Some(&grant),
                    Some(QueryMemoryDomainPolicy::default()),
                    capacity_bytes,
                )
                .unwrap()
                .unwrap();
                drop(lease);
                domain
            })
        })
        .collect::<Vec<_>>();
    let domains = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();

    assert!(domains.iter().all(|domain| Arc::ptr_eq(&domains[0], domain)));
    assert!(!grant_memory_domain_creation_locks().lock().unwrap().contains_key(&grant_key));
}
#[test]
fn query_service_context_carries_local_parquet_source_cache() {
    let config = QueryServiceConfig {
        local_parquet_source_cache_max_retained_bytes: Some(4096),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let resources = service.inner.resources.as_ref().unwrap();
    let service_cache = resources.local_parquet_source_cache_for_device(0).expect("device cache should be installed");
    let context_cache = tenant_context
        .context()
        .local_parquet_source_cache()
        .expect("service context should carry the per-device source cache");

    assert_eq!(context_cache.max_retained_device_bytes(), 4096);
    assert!(context_cache.is_governor_backed());
    assert!(Arc::ptr_eq(&service_cache, &context_cache));
}
#[test]
fn expired_shutdown_deadline_runs_local_cache_phase_and_returns_component_timeout() {
    let config = QueryServiceConfig {
        local_parquet_source_cache_max_retained_bytes: Some(4096),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();
    let cache = service.inner.resources.as_ref().unwrap().local_parquet_source_cache_for_device(0).unwrap();

    let error = service.close_and_wait_for_quiescence_until(Instant::now()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("local_source_cache_drain_timeout"));
    assert!(cache.is_empty().unwrap());
}
#[test]
fn query_service_source_cache_is_shared_across_stream_runtimes() {
    let config = QueryServiceConfig {
        local_parquet_source_cache_max_retained_bytes: Some(4096),
        max_concurrent_queries: 2,
        stream_pool_size: Some(6),
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let first_handle = service.admit_query("tenant").unwrap();
    let second_handle = service.admit_query("tenant").unwrap();
    let first_context = first_handle.acquire_context().unwrap();
    let second_context = second_handle.acquire_context().unwrap();

    let first_runtime = first_context.context().shared_runtime().unwrap();
    let second_runtime = second_context.context().shared_runtime().unwrap();
    assert!(
        !Arc::ptr_eq(&first_runtime, &second_runtime),
        "holding the first context should force the second stream-pool runtime"
    );

    let first_cache = first_context.context().local_parquet_source_cache().expect("first runtime should have a cache");
    let second_cache =
        second_context.context().local_parquet_source_cache().expect("second runtime should have a cache");

    assert_eq!(first_cache.max_retained_device_bytes(), 4096);
    assert_eq!(second_cache.max_retained_device_bytes(), 4096);
    assert!(
        Arc::ptr_eq(&first_cache, &second_cache),
        "same-device service contexts share one runtime-aware cache with one retained-byte cap"
    );
}
#[test]
fn local_parquet_cache_is_query_neutral_and_checkout_retains_current_query() {
    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::DataFrame;

    use crate::{
        runtime::{
            cache::local_parquet::GpuLocalParquetSourceCacheKey,
            resident::{GpuDataFrame, GpuResultHandle},
        },
        source::LocalParquetSource,
    };

    let service = QueryService::new_with_deployment(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(500),
            local_parquet_source_cache_max_retained_bytes: Some(1024 * 1024),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let source = LocalParquetSource::new(
        "/tmp/datafusion-nexus-query-neutral-cache.parquet",
        Arc::clone(&schema),
        vec!["value".to_string()],
    );
    let key = GpuLocalParquetSourceCacheKey::from_source(&source);

    let q1 = service.admit_query("tenant").unwrap();
    let q1_context = q1.acquire_context().unwrap();
    let runtime = q1_context.context().shared_runtime().unwrap();
    let frame = GpuDataFrame::new(DataFrame::from_record_batch(runtime.as_ref(), &batch).unwrap());
    let retained_device_bytes = frame.device_allocated_size_bytes().unwrap();
    let q1_result = GpuResultHandle::new(q1_context.context(), frame, Arc::clone(&schema)).unwrap();
    let cache = q1_context.context().local_parquet_source_cache().unwrap();
    let inserted =
        cache.insert_if_absent_pinned(q1_context.context(), key.clone(), q1_result, retained_device_bytes).unwrap();
    assert!(inserted.handle.is_some());
    drop(inserted.handle);
    drop(runtime);
    drop(q1_context);
    q1.finish();
    drop(q1);

    let q2 = service.admit_query("tenant").expect("cache entry must not retain q1 admission");
    let q2_context = q2.acquire_context().unwrap();
    let (q2_output, cached_bytes) = cache.get(&key, q2_context.context(), &schema).unwrap().expect("q2 cache hit");
    assert_eq!(cached_bytes, retained_device_bytes);
    assert_eq!(q2_output.to_host_record_batch().unwrap(), batch);

    q2.finish();
    let expired = match cache.get(&key, q2_context.context(), &schema) {
        Ok(_) => panic!("finished query must reject cache lookup"),
        Err(error) => error,
    };
    assert_eq!(expired.fact_value("reason"), Some("query_handle_released"));
    drop(q2_context);
    drop(q2);
    assert!(service.try_admit_query("tenant").unwrap().is_none(), "escaped q2 output retains q2 admission");

    drop(q2_output);
    let replacement = service.admit_query("tenant").expect("final q2 output drop releases admission");
    drop(replacement);
}
#[test]
fn query_service_context_carries_graph_calibration_override() {
    let graph_memory_calibration_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
    let config = QueryServiceConfig {
        graph_memory_calibration_override,
        // Runtime-backed services contend for process-global governor claims in
        // unit tests; optimistic claims can coexist across test cases.
        slot_guarantee: SlotGuarantee::Optimistic,
        ..QueryServiceConfig::default()
    };
    let service = QueryService::new_with_deployment(
        config,
        vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
        non_gds_deployment(),
    )
    .unwrap();

    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();

    assert_eq!(tenant_context.context().graph_memory_calibration_override(), graph_memory_calibration_override);
}
#[test]
fn query_service_released_handle_rejects_context_acquisition() {
    let service = test_service(1, 8);
    let handle = service.admit_query("small").unwrap();
    handle.finish();

    let error = handle.acquire_context().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("tenant_id"), Some("small"));
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
}
#[test]
fn query_service_context_keeps_admission_until_context_drop() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    drop(handle);

    assert!(service.try_admit_query("small").unwrap().is_none());
    drop(tenant_context);
    service.admit_query("small").expect("cleanup should wake the blocked admission");
}
#[test]
fn finished_query_rejects_context_derivations_until_escaped_context_drops() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let escaped_context = tenant_context.context().clone();
    drop(tenant_context);

    handle.finish();

    for error in [
        escaped_context.runtime().unwrap_err(),
        escaped_context.memory_grant().unwrap_err(),
        escaped_context.gpu_execution_pool().unwrap_err(),
        escaped_context.service_stream_pool().acquire().unwrap_err(),
    ] {
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    }
    assert!(service.try_admit_query("small").unwrap().is_none());

    drop(escaped_context);
    let next = service.admit_query("small").expect("the final context anchor releases admission");
    drop(next);
}
#[test]
fn finish_vs_runtime_derivation_is_linearizable_and_releases_once() {
    let service = runtime_test_service(1, 8);
    for _ in 0..16 {
        let handle = Arc::new(service.admit_query("small").unwrap());
        let tenant_context = handle.acquire_context().unwrap();
        let escaped_context = tenant_context.context().clone();
        drop(tenant_context);
        let barrier = Arc::new(Barrier::new(3));
        let finish_handle = Arc::clone(&handle);
        let finish_barrier = Arc::clone(&barrier);
        let finish = std::thread::spawn(move || {
            finish_barrier.wait();
            finish_handle.finish();
        });
        let derive_barrier = Arc::clone(&barrier);
        let derive = std::thread::spawn(move || {
            derive_barrier.wait();
            escaped_context.runtime()
        });

        barrier.wait();
        finish.join().unwrap();
        let runtime = match derive.join().unwrap() {
            Ok(runtime) => Some(runtime),
            Err(error) => {
                assert_eq!(error.code(), QueryErrorCode::Cancelled);
                assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
                None
            },
        };
        drop(handle);
        if runtime.is_some() {
            assert!(
                service.try_admit_query("small").unwrap().is_none(),
                "a runtime linearized before finish must retain admission"
            );
        }
        drop(runtime);
        let next = service.admit_query("small").expect("each raced query releases admission exactly once");
        drop(next);
    }
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
}
#[test]
fn absolute_shutdown_deadline_includes_admission_close_lock_contention() {
    let service = test_service(1, 8);
    let admission = service.lock_admission().unwrap();
    let deadline = Instant::now() + Duration::from_millis(20);
    let waiting_service = service.clone();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let wait = std::thread::spawn(move || {
        result_tx.send(waiting_service.close_and_wait_for_quiescence_until(deadline)).unwrap();
    });

    let result = result_rx.recv_timeout(Duration::from_millis(200));
    drop(admission);
    wait.join().unwrap();
    let error = result
        .expect("shutdown must return its mutex timeout before the contended lock is released")
        .expect_err("admission mutex contention must consume the shutdown deadline");

    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(error.fact_value("deadline_stage"), Some("admission_mutex"));
    assert_eq!(
        service.try_admit_query("small").unwrap_err().fact_value("reason"),
        Some("admission_closed"),
        "admission-lock timeout must leave the service fail-closed"
    );

    service.close_and_wait_for_quiescence(Duration::from_secs(1)).unwrap();
}
#[test]
fn close_signal_is_nonblocking_and_fail_closed_during_admission_lock_contention() {
    let service = test_service(1, 8);
    let admission = service.lock_admission().unwrap();
    let closing_service = service.clone();
    let (closed_tx, closed_rx) = std::sync::mpsc::channel();
    let close = std::thread::spawn(move || closed_tx.send(closing_service.close()).unwrap());

    let close_result = closed_rx.recv_timeout(Duration::from_millis(50));
    drop(admission);
    let close_result = close_result.expect("close signal must not wait for the admission-state mutex");
    close.join().unwrap();
    close_result.unwrap();
    assert_eq!(service.try_admit_query("small").unwrap_err().fact_value("reason"), Some("admission_closed"));
}
#[test]
fn resource_free_expired_shutdown_completes_epoch_cleanup_but_cannot_succeed() {
    let service = test_service(1, 8);
    let error = service
        .close_and_wait_for_quiescence_until(Instant::now())
        .expect_err("the complete shutdown cannot succeed after its absolute deadline");

    assert_eq!(error.fact_value("reason"), Some("query_service_resource_epoch_timeout"));
    assert_eq!(service.try_admit_query("small").unwrap_err().fact_value("reason"), Some("admission_closed"));
}
#[test]
fn quiescence_reached_after_absolute_deadline_is_not_accepted() {
    let service = test_service(1, 8);
    service.lock_admission().unwrap().active_queries = 1;
    let deadline = Instant::now() + Duration::from_millis(100);
    let waiting_service = service.clone();
    let wait = std::thread::spawn(move || waiting_service.close_and_wait_for_quiescence_until(deadline));
    while service.inner.shutdown_waiters.load(Ordering::Acquire) == 0 {
        std::thread::yield_now();
    }

    let mut state = service.lock_admission().unwrap();
    let release_at = deadline + Duration::from_millis(20);
    std::thread::sleep(release_at.saturating_duration_since(Instant::now()));
    state.active_queries = 0;
    service.inner.notify_admission_available();
    drop(state);

    let error = wait.join().unwrap().expect_err("late quiescence must preserve the component timeout");
    assert_eq!(error.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(error.fact_value("deadline_stage"), Some("admission_quiescence"));
}
#[test]
fn query_service_handle_binds_secondary_context_to_dedicated_stream() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let first = handle.acquire_context().unwrap();
    let second = handle.acquire_context().unwrap();
    let first_runtime = first.context().shared_runtime().unwrap();
    let second_runtime = second.context().shared_runtime().unwrap();

    // Secondary acquires keep the same admitted device while binding a dedicated
    // CUDA stream so concurrent native fragments can each create a live
    // query-domain budget token (cuDF rejects two live tokens on one stream).
    assert_eq!(first_runtime.placement().device, second_runtime.placement().device);
    assert!(!Arc::ptr_eq(&first_runtime, &second_runtime));
    let lease = first_runtime.lease_anchor();
    assert!(!second_runtime.matches_lease_anchor(&lease));
    assert_ne!(first_runtime.placement().stream_debug, second_runtime.placement().stream_debug);

    // Concurrent live query-domain budget tokens must succeed on both contexts.
    let left = first
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("secondary_stream_left"))
        .expect("first fragment budget token");
    let right = second
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("secondary_stream_right"))
        .expect("second fragment budget token");
    drop(left);
    drop(right);
}
#[test]
fn escaped_context_causes_bounded_shutdown_timeout_until_final_anchor_drops() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let retained_context = tenant_context.context().clone();
    drop(tenant_context);
    drop(handle);

    let timeout = service.close_and_wait_for_quiescence(Duration::from_millis(10)).unwrap_err();
    assert_eq!(timeout.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(timeout.fact_value("deadline_stage"), Some("admission_quiescence"));

    for error in [
        retained_context.runtime().unwrap_err(),
        retained_context.memory_grant().unwrap_err(),
        retained_context.gpu_execution_pool().unwrap_err(),
    ] {
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("query_service_epoch_closed"));
    }

    drop(retained_context);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    let replacement =
        try_guaranteed_runtime_test_service().expect("the final query anchor releases the guaranteed governor claim");
    drop(replacement);
}
#[test]
fn held_runtime_and_produced_object_delay_claim_release_until_last_provenance_drop() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let retained_context = tenant_context.context().clone();
    let runtime = retained_context.runtime().unwrap();
    let series = runtime.series(&[1_i32, 2, 3]).expect("produce a retained GPU series");
    drop(runtime);
    drop(tenant_context);
    drop(retained_context);
    drop(handle);

    let timeout = service.close_and_wait_for_quiescence(Duration::from_millis(10)).unwrap_err();
    assert_eq!(timeout.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(timeout.fact_value("deadline_stage"), Some("admission_quiescence"));
    let error = try_guaranteed_runtime_test_service()
        .expect_err("retained produced object keeps the prior governor claim live");
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));

    drop(series);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    let replacement =
        try_guaranteed_runtime_test_service().expect("last retained provenance drop releases the prior governor claim");
    drop(replacement);
}
#[test]
fn execution_budget_token_retains_query_domain_until_drop() {
    let service = try_guaranteed_runtime_test_service().expect("guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let token = tenant_context
        .context()
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1).with_boundary("query_epoch_retention"))
        .expect("query-domain execution budget token");
    drop(tenant_context);
    handle.finish();

    assert!(
        service.try_admit_query("tenant").unwrap().is_none(),
        "the budget token's retained runtime must keep query admission"
    );
    drop(token);
    let next = service.admit_query("tenant").expect("dropping the budget token releases query admission");
    drop(next);
}
#[test]
fn retained_service_stream_pool_keeps_query_admitted_and_rejects_post_finish_acquire() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let retained_pool = tenant_context.context().service_stream_pool();
    drop(tenant_context);
    handle.finish();

    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    let error = retained_pool.acquire().expect_err("closed query epoch must reject stream acquisition");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    drop(retained_pool);
    let replacement = service.admit_query("tenant").expect("final stream-pool anchor releases query admission");
    drop(replacement);
}
#[test]
fn acquired_service_stream_and_derived_runtime_hold_query_until_last_counted_drop() {
    use cudf::testing::prelude::RuntimeTestingExt;

    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let retained_pool = tenant_context.context().service_stream_pool();
    let stream_lease = retained_pool.acquire().expect("service stream acquisition");
    let runtime = stream_lease.runtime().expect("counted derived runtime");
    let shared_runtime = stream_lease.shared_runtime().expect("counted shared derived runtime");

    assert_eq!(runtime.testing_retained_lifetime_count(), 1);
    assert_eq!(shared_runtime.testing_retained_lifetime_count(), 1);
    drop(shared_runtime);
    drop(retained_pool);
    drop(tenant_context);
    handle.finish();

    let error = stream_lease.runtime().expect_err("query finish must reject new runtime derivation from the lease");
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    assert!(service.try_admit_query("tenant").unwrap().is_none());

    drop(stream_lease);
    assert!(
        service.try_admit_query("tenant").unwrap().is_none(),
        "the runtime derived before finish retains the query after its stream is released"
    );

    drop(runtime);
    let replacement = service.admit_query("tenant").expect("the final stream runtime releases query admission");
    drop(replacement);
}
#[test]
fn cloned_memory_grant_and_live_reservation_hold_query_and_reject_post_finish_work() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let grant = tenant_context.context().memory_grant().unwrap().expect("service context carries a memory grant");
    let retained_grant = grant.clone();
    let retained_reservation = retained_grant.retain_live_reservation(1).expect("positive reservation is retained");
    drop(grant);
    drop(tenant_context);
    handle.finish();

    let error = retained_grant.try_grow(1).expect_err("closed query epoch must reject grant mutation");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    assert!(service.try_admit_query("tenant").unwrap().is_none());

    drop(retained_grant);
    assert!(
        service.try_admit_query("tenant").unwrap().is_none(),
        "the live reservation's cloned grant retains query admission"
    );

    drop(retained_reservation);
    let replacement = service.admit_query("tenant").expect("the final memory-grant anchor releases admission");
    drop(replacement);
}
#[tokio::test(flavor = "current_thread")]
async fn cloned_gpu_pool_job_holds_query_and_rejects_post_finish_submission() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let pool = tenant_context.context().gpu_execution_pool().expect("counted GPU execution pool");
    let retained_pool = pool.clone();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let job_release = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = started_tx.send(());
        let _permit = job_release.acquire_owned().await.expect("job release semaphore remains open");
        let _ = finished_tx.send(());
    })
    .unwrap();
    started_rx.await.expect("GPU job starts");
    drop(pool);
    drop(tenant_context);
    handle.finish();

    let spawn_error = retained_pool
        .spawn(GpuExecutionJobKind::QueryExecution, async {})
        .expect_err("closed query epoch must reject detached GPU work");
    assert_eq!(spawn_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(spawn_error.fact_value("reason"), Some("query_handle_released"));
    let run_error = retained_pool
        .run(GpuExecutionJobKind::QueryExecution, async {})
        .await
        .expect_err("closed query epoch must reject awaited GPU work");
    assert_eq!(run_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(run_error.fact_value("reason"), Some("query_handle_released"));
    drop(retained_pool);

    assert!(service.try_admit_query("tenant").unwrap().is_none(), "the in-flight GPU job must retain query admission");

    release.add_permits(1);
    finished_rx.await.expect("GPU job finishes after release");
    let replacement = service.admit_query("tenant").expect("the completed GPU job releases query admission");
    drop(replacement);
}
#[tokio::test(flavor = "current_thread")]
async fn queued_gpu_pool_job_retains_query_without_parking_cleanup_on_worker_permit() {
    let service = try_guaranteed_runtime_test_service().expect("single-worker guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let pool = tenant_context.context().gpu_execution_pool().expect("counted GPU execution pool");
    assert_eq!(pool.snapshot().worker_count, 1);
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let blocker_release = Arc::clone(&release);
    let (blocker_started_tx, blocker_started_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = blocker_started_tx.send(());
        let _permit = blocker_release.acquire_owned().await.expect("blocker release remains open");
    })
    .unwrap();
    blocker_started_rx.await.expect("blocking GPU job starts");

    let (queued_finished_tx, queued_finished_rx) = tokio::sync::oneshot::channel();
    pool.spawn(GpuExecutionJobKind::QueryExecution, async move {
        let _ = queued_finished_tx.send(());
    })
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while pool.snapshot().queued_jobs == 0 {
        assert!(Instant::now() < deadline, "second GPU job did not enter the worker queue");
        std::thread::yield_now();
    }

    drop(tenant_context);
    handle.finish();
    drop(pool);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    release.add_permits(1);
    queued_finished_rx.await.expect("queued job runs after the worker permit is released");
    let next = service.admit_query("tenant").expect("queued and running anchors drain before cleanup");
    drop(next);
}
#[tokio::test(flavor = "current_thread")]
async fn query_handle_gpu_pool_is_counted_and_rejects_work_after_finish() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let retained_pool = handle.gpu_execution_pool().expect("counted query-handle GPU execution pool");

    handle.finish();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    let spawn_error = retained_pool
        .spawn(GpuExecutionJobKind::QueryExecution, async {})
        .expect_err("closed query epoch must reject detached query-handle GPU work");
    assert_eq!(spawn_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(spawn_error.fact_value("reason"), Some("query_handle_released"));
    let run_error = retained_pool
        .run(GpuExecutionJobKind::QueryExecution, async {})
        .await
        .expect_err("closed query epoch must reject awaited query-handle GPU work");
    assert_eq!(run_error.code(), QueryErrorCode::Cancelled);
    assert_eq!(run_error.fact_value("reason"), Some("query_handle_released"));

    drop(retained_pool);
    let replacement = service.admit_query("tenant").expect("the final query-handle GPU pool drop releases admission");
    drop(replacement);
}
#[tokio::test(flavor = "current_thread")]
async fn query_handle_gpu_job_retains_query_epoch_until_joined() {
    let service = try_guaranteed_runtime_test_service_with_concurrency(2).expect("two-worker guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let retained_pool = handle.gpu_execution_pool().expect("counted query-handle GPU execution pool");
    assert_eq!(retained_pool.snapshot().worker_count, 2);
    let job_pool = retained_pool.clone();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let job_release = Arc::clone(&release);
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let job = tokio::spawn(async move {
        job_pool
            .run(GpuExecutionJobKind::QueryExecution, async move {
                let _ = started_tx.send(());
                let _permit = job_release.acquire_owned().await.expect("job release semaphore remains open");
            })
            .await
    });
    started_rx.await.expect("GPU job starts");

    handle.finish();
    drop(retained_pool);
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);

    release.add_permits(1);
    job.await.expect("GPU job task joins").expect("GPU job succeeds after release");
    let replacement = service.admit_query("tenant").expect("joining the final GPU job releases query admission");
    drop(replacement);
}
#[test]
fn active_query_owner_prevents_claim_release_until_cleanup_finishes() {
    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();

    service.close().unwrap();
    let error = try_guaranteed_runtime_test_service().expect_err("active grant keeps prior claim live");
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));

    drop(handle);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    let replacement = try_guaranteed_runtime_test_service().expect("cleanup releases the prior claim");
    drop(replacement);
}
#[test]
fn close_and_wait_fences_delayed_gpu_cleanup_before_claim_release() {
    struct CleanupBlockerGate {
        semaphore: Arc<tokio::sync::Semaphore>,
    }

    impl Drop for CleanupBlockerGate {
        fn drop(&mut self) {
            self.semaphore.close();
        }
    }

    let service = try_guaranteed_runtime_test_service().expect("first guaranteed service");
    let handle = service.admit_query("tenant").unwrap();
    let pool = handle.gpu_execution_pool().unwrap();
    let worker_count = pool.snapshot().worker_count;
    let release = CleanupBlockerGate { semaphore: Arc::new(tokio::sync::Semaphore::new(0)) };
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();

    for _ in 0..worker_count {
        let started_tx = started_tx.clone();
        let finished_tx = finished_tx.clone();
        let release = Arc::clone(&release.semaphore);
        pool.spawn(GpuExecutionJobKind::Cleanup, async move {
            started_tx.send(()).expect("cleanup blocker start receiver remains live");
            let _permit = release.acquire_owned().await.expect("cleanup blocker semaphore remains open");
            finished_tx.send(()).expect("cleanup finish receiver remains live");
        })
        .unwrap();
    }
    drop(started_tx);
    drop(finished_tx);
    for _ in 0..worker_count {
        started_rx.recv_timeout(Duration::from_secs(5)).expect("every GPU worker starts a cleanup blocker");
    }

    drop(handle);
    service.close().unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    let error =
        try_guaranteed_runtime_test_service().expect_err("queued GPU cleanup must retain the prior governor claim");
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));

    let wait_entry = Arc::new(Barrier::new(2));
    let waiter_entry = Arc::clone(&wait_entry);
    let waiter_service = service.clone();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        waiter_entry.wait();
        done_tx
            .send(waiter_service.close_and_wait_for_quiescence(Duration::from_secs(5)))
            .expect("close waiter receiver remains live");
    });
    wait_entry.wait();
    assert!(
        matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "close-and-wait cannot finish while GPU cleanup is queued"
    );

    release.semaphore.add_permits(worker_count);
    for _ in 0..worker_count {
        finished_rx.recv().expect("every delayed cleanup job finishes after release");
    }
    assert!(
        matches!(done_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)),
        "close-and-wait remains blocked by the counted pool after cleanup finishes"
    );
    drop(pool);
    done_rx.recv().expect("close waiter completes after the counted pool drops").unwrap();
    waiter.join().expect("close waiter must not panic");

    let replacement =
        try_guaranteed_runtime_test_service().expect("close-and-wait releases the claim before publishing quiescence");
    drop(replacement);
}
#[test]
fn query_service_waits_for_background_stream_cleanup_quiescence() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let context = handle.acquire_context().unwrap();

    let error = service.wait_for_stream_pools_idle(Duration::ZERO).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("stream_pool_idle_timeout"));
    assert_eq!(error.fact_value("created_runtimes"), Some("1"));
    assert_eq!(error.fact_value("idle_runtimes"), Some("0"));

    drop(context);
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
}
#[test]
fn query_service_quiescence_allows_declared_long_lived_root_contexts() {
    let service = runtime_test_service(1, 8);
    let handle = service.admit_query("large").unwrap();
    let context = handle.acquire_context().unwrap();

    service.wait_for_stream_pools_quiescent(&[(0, 1)], Duration::ZERO).unwrap();

    let error = service.wait_for_stream_pools_quiescent(&[(0, 0)], Duration::ZERO).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
    assert_eq!(error.fact_value("reason"), Some("stream_pool_idle_timeout"));
    assert_eq!(error.fact_value("active_leased_runtimes"), Some("1"));
    assert_eq!(error.fact_value("expected_leased_runtimes"), Some("0"));

    drop(context);
    drop(handle);
    service.wait_for_stream_pools_idle(Duration::from_secs(5)).unwrap();
}
