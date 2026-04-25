use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "cugraph")]
use cugraph::cudf_interop::{SeriesGraphOptions, graph_from_series};

use super::*;
use crate::{
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphConstructionPolicy, GraphEdgeColumnBindings, GraphResidencyOrientation, GraphResidencyPlanAnalysis,
        QueryPlanBuilder, graph_residency_descriptor_for_edge_plan,
    },
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryService,
        QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee,
    },
    source::{LocalParquetFile, LocalParquetSource, SourceNode},
};

fn test_resident_graph_entry_key() -> GpuResidentGraphCacheEntryKey {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/preparation-pin-edges.parquet")
                .with_file_size_bytes(1)
                .with_stable_id("local:/tmp/preparation-pin-edges.parquet:1:mtime=test"),
        ],
        schema,
        vec!["src".to_string(), "dst".to_string()],
    );
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(&plan, &spec) else {
        panic!("test edge source should produce a resident graph key")
    };
    GpuResidentGraphCacheEntryKey {
        resident_key: descriptor.resident_key(&spec, GraphResidencyOrientation::Forward),
        runtime_key: 1,
    }
}

#[cfg(feature = "cugraph")]
fn test_graph_context(cache: &Arc<GpuResidentGraphCache>) -> ExecutionContext {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_resident_graph_cache(Arc::clone(cache))
        .build()
        .unwrap()
}

#[cfg(feature = "cugraph")]
fn test_parked_forward_graph(
    cache: &GpuResidentGraphCache,
    context: &ExecutionContext,
) -> (cugraph::ParkedGraph<cugraph::Forward>, Weak<cudf::Runtime>) {
    let runtime =
        cache.companion_runtime_for_context(context, "nexus_query_engine.graph_cache.test_companion").unwrap();
    let runtime_owner = Arc::downgrade(&runtime);
    let handle = cugraph::ResourceHandle::for_runtime(Arc::clone(&runtime)).unwrap();
    let entered = handle.enter("nexus_query_engine.graph_cache.test_graph").unwrap();
    let src = runtime.series(&[0_i32, 1, 2]).unwrap();
    let dst = runtime.series(&[1_i32, 2, 0]).unwrap();
    let graph =
        graph_from_series::<cugraph::Forward>(&entered, &src, &dst, None, &SeriesGraphOptions::default().renumber())
            .unwrap();
    let parked = graph.park(&entered).unwrap();
    drop(entered);
    drop(handle);
    drop(runtime);
    (parked, runtime_owner)
}

#[cfg(feature = "cugraph")]
fn guaranteed_graph_cache_service() -> Result<QueryService> {
    QueryService::new_with_deployment(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            max_admission_queue_len: 8,
            admission_wait_timeout: Duration::from_millis(500),
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            resident_graph_cache_max_retained_bytes: Some(64 * 1024 * 1024),
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 8 * 1024 * 1024)],
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        ),
    )
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_preparation_pin_reserves_checkout_for_its_owner() {
    let entry_key = test_resident_graph_entry_key();
    let mut state = GpuResidentGraphCacheState::default();
    state.preparation_pins.insert(entry_key.clone(), 41);

    assert!(!state.consume_checkout_preparation_pin(&entry_key, None));
    assert_eq!(state.preparation_pins.get(&entry_key), Some(&41));
    assert!(!state.consume_checkout_preparation_pin(&entry_key, Some(42)));
    assert_eq!(state.preparation_pins.get(&entry_key), Some(&41));
    assert!(state.consume_checkout_preparation_pin(&entry_key, Some(41)));
    assert!(!state.preparation_pins.contains_key(&entry_key));
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_build_leader_failure_wakes_followers_and_reconciles_counters() {
    let resident_key = test_resident_graph_entry_key().resident_key;
    let cache = GpuResidentGraphCache::new(1);
    let leader = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Leader(leader) => leader,
        GpuResidentGraphBuildJoin::Follower(_) => panic!("first join must lead"),
    };
    let follower = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Follower(follower) => follower,
        GpuResidentGraphBuildJoin::Leader(_) => panic!("second simultaneous join must follow"),
    };
    let cancellation = crate::runtime::ExecutionCancellation::new();
    let waiter = std::thread::spawn(move || {
        follower.wait(
            &cancellation,
            Instant::now() + Duration::from_secs(1),
            "nexus_query_engine.graph_cache.test_build_wait",
        )
    });

    drop(leader);
    waiter.join().unwrap().unwrap();

    let snapshot = cache.snapshot().unwrap();
    assert_eq!(snapshot.misses, 1);
    assert_eq!(snapshot.build_waits, 1);
    assert_eq!(snapshot.builds_in_flight, 0);
    assert!(matches!(cache.join_build(&resident_key).unwrap(), GpuResidentGraphBuildJoin::Leader(_)));
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_build_follower_observes_cancellation() {
    let resident_key = test_resident_graph_entry_key().resident_key;
    let cache = GpuResidentGraphCache::new(1);
    let _leader = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Leader(leader) => leader,
        GpuResidentGraphBuildJoin::Follower(_) => panic!("first join must lead"),
    };
    let follower = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Follower(follower) => follower,
        GpuResidentGraphBuildJoin::Leader(_) => panic!("second simultaneous join must follow"),
    };
    let cancellation = crate::runtime::ExecutionCancellation::new();
    cancellation.cancel();

    let error = follower
        .wait(
            &cancellation,
            Instant::now() + Duration::from_secs(1),
            "nexus_query_engine.graph_cache.test_build_cancel",
        )
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(cache.snapshot().unwrap().build_waits, 1);
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_build_follower_observes_deadline() {
    let resident_key = test_resident_graph_entry_key().resident_key;
    let cache = GpuResidentGraphCache::new(1);
    let _leader = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Leader(leader) => leader,
        GpuResidentGraphBuildJoin::Follower(_) => panic!("first join must lead"),
    };
    let follower = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Follower(follower) => follower,
        GpuResidentGraphBuildJoin::Leader(_) => panic!("second simultaneous join must follow"),
    };

    let outcome = follower
        .wait(
            &crate::runtime::ExecutionCancellation::new(),
            Instant::now(),
            "nexus_query_engine.graph_cache.test_build_deadline",
        )
        .unwrap();

    assert_eq!(outcome, GpuResidentGraphBuildWait::DeadlineExpired);
    assert_eq!(cache.snapshot().unwrap().build_waits, 1);
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_close_and_drain_is_idempotent_and_rejects_new_builds() {
    let resident_key = test_resident_graph_entry_key().resident_key;
    let cache = GpuResidentGraphCache::new(1);
    let _leader = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Leader(leader) => leader,
        GpuResidentGraphBuildJoin::Follower(_) => panic!("first join must lead"),
    };
    let follower = match cache.join_build(&resident_key).unwrap() {
        GpuResidentGraphBuildJoin::Follower(follower) => follower,
        GpuResidentGraphBuildJoin::Leader(_) => panic!("second simultaneous join must follow"),
    };

    cache.close_and_drain_until(Instant::now() + Duration::from_secs(1)).unwrap();
    let outcome = follower
        .wait(
            &crate::runtime::ExecutionCancellation::new(),
            Instant::now() + Duration::from_secs(1),
            "nexus_query_engine.graph_cache.test_close_and_drain",
        )
        .unwrap();
    assert_eq!(outcome, GpuResidentGraphBuildWait::LeaderFinished);

    cache.close_and_drain().unwrap();
    let snapshot = cache.snapshot().unwrap();
    assert_eq!(snapshot.entries, 0);
    assert_eq!(snapshot.builds_in_flight, 0);
    let error = match cache.join_build(&resident_key) {
        Ok(_) => panic!("closed cache must reject a new build"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("resident_graph_cache_closed"));
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_drain_overrun_is_reported_after_non_preemptible_drop_phase() {
    let cache = GpuResidentGraphCache::new(1);
    cache.set_close_and_drain_delay_for_test(Duration::from_millis(40));

    let error = cache
        .close_and_drain_until(Instant::now() + Duration::from_millis(10))
        .expect_err("resident graph drain that overruns the shared deadline must not report success");

    assert_eq!(error.fact_value("reason"), Some("resident_graph_cache_drain_timeout"));
    assert_eq!(cache.snapshot().unwrap().entries, 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_close_destroys_available_parked_graph_and_companion() {
    let cache = Arc::new(GpuResidentGraphCache::new(64 * 1024 * 1024));
    let context = test_graph_context(&cache);
    let resident_key = test_resident_graph_entry_key().resident_key;
    let (parked, companion) = test_parked_forward_graph(&cache, &context);
    assert_eq!(
        cache.insert_if_absent(resident_key, &context, parked, 1024).unwrap(),
        GpuResidentGraphCacheInsert::Inserted
    );
    assert!(companion.upgrade().is_some());

    cache.close_and_drain().unwrap();

    assert!(companion.upgrade().is_none(), "available graph destruction releases its companion runtime");
    assert_eq!(cache.snapshot().unwrap().entries, 0);
    assert_eq!(cache.snapshot().unwrap().resident_bytes, 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn resident_graph_checked_out_return_after_close_is_not_reinserted() {
    let cache = Arc::new(GpuResidentGraphCache::new(64 * 1024 * 1024));
    let context = test_graph_context(&cache);
    let resident_key = test_resident_graph_entry_key().resident_key;
    let (parked, companion) = test_parked_forward_graph(&cache, &context);
    cache.insert_if_absent(resident_key.clone(), &context, parked, 1024).unwrap();
    let mut checkout = cache.checkout::<cugraph::Forward>(&resident_key, &context, None).unwrap().unwrap();
    let parked = checkout.take_parked();

    cache.close_and_drain().unwrap();
    assert!(companion.upgrade().is_some(), "the checked-out graph still owns its companion runtime");
    checkout.return_parked(parked).unwrap();

    assert!(companion.upgrade().is_none(), "closed-cache return destroys the parked graph");
    assert_eq!(cache.snapshot().unwrap().entries, 0);
    assert_eq!(cache.snapshot().unwrap().resident_bytes, 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn cached_graph_owner_is_query_neutral_while_current_outputs_retain_query() {
    let service = guaranteed_graph_cache_service().expect("graph-cache service");
    let first = service.admit_query("tenant").unwrap();
    let first_tenant_context = first.acquire_context().unwrap();
    let first_context = first_tenant_context.context().clone();
    let cache = first_context.resident_graph_cache().expect("service graph cache");
    let resident_key = test_resident_graph_entry_key().resident_key;
    let (parked, companion) = test_parked_forward_graph(&cache, &first_context);
    assert_eq!(
        cache.insert_if_absent(resident_key.clone(), &first_context, parked, 1024).unwrap(),
        GpuResidentGraphCacheInsert::Inserted
    );
    drop(first_context);
    drop(first_tenant_context);
    first.finish();
    drop(first);

    service.wait_for_stream_pools_idle(Duration::from_secs(5)).expect("first query cleanup returns its pooled runtime");
    let second = service.admit_query("tenant").expect("cache entry must not retain the first query slot");
    let second_tenant_context = second.acquire_context().unwrap();
    let second_context = second_tenant_context.context().clone();
    let mut checkout = cache
        .checkout::<cugraph::Forward>(&resident_key, &second_context, None)
        .unwrap()
        .expect("second query reuses the first query's parked graph");
    let parked = checkout.take_parked();
    let (cache_runtime, query_runtime) =
        cache.runtimes_for_context(&second_context, "nexus_query_engine.graph_cache.test_query_neutral").unwrap();
    assert!(Arc::ptr_eq(&cache_runtime, &companion.upgrade().expect("cached graph retains its neutral companion")));
    let handle = cugraph::ResourceHandle::for_runtime(Arc::clone(&cache_runtime)).unwrap();
    let entered = handle.enter("nexus_query_engine.graph_cache.test_query_neutral").unwrap();
    let graph = parked.resume(&entered).unwrap();
    checkout.return_parked(graph.park(&entered).unwrap()).unwrap();
    let output = query_runtime.series(&[7_i32, 11, 13]).unwrap();
    drop(entered);
    drop(handle);
    drop(query_runtime);
    drop(cache_runtime);
    drop(second_tenant_context);
    second.finish();
    let error = match cache
        .runtimes_for_context(&second_context, "nexus_query_engine.graph_cache.test_post_finish_derivation")
    {
        Ok(_) => panic!("closed query epoch must reject cache runtime derivation"),
        Err(error) => error,
    };
    assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    drop(second_context);
    drop(second);

    assert!(
        service.try_admit_query("tenant").unwrap().is_none(),
        "current-query output provenance must retain the second query"
    );
    drop(output);
    let third = service.admit_query("tenant").expect("dropping the current output releases the second query");
    drop(third);

    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    assert!(companion.upgrade().is_none(), "cache shutdown destroys graph before its service companion");
    assert_eq!(cache.snapshot().unwrap().entries, 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn escaped_checked_out_graph_retains_service_epoch_and_claim_until_drop() {
    let service = guaranteed_graph_cache_service().expect("first guaranteed graph-cache service");
    let handle = service.admit_query("tenant").unwrap();
    let tenant_context = handle.acquire_context().unwrap();
    let context = tenant_context.context().clone();
    let cache = context.resident_graph_cache().expect("service graph cache");
    let resident_key = test_resident_graph_entry_key().resident_key;
    let (parked, companion) = test_parked_forward_graph(&cache, &context);
    cache.insert_if_absent(resident_key.clone(), &context, parked, 1024).unwrap();
    let mut checkout = cache.checkout::<cugraph::Forward>(&resident_key, &context, None).unwrap().unwrap();
    let parked = checkout.take_parked();
    drop(context);
    drop(tenant_context);
    drop(handle);

    let replacement_query =
        service.admit_query("tenant").expect("escaped query-neutral parked graph must not retain the first query slot");
    drop(replacement_query);

    service.close().unwrap();
    cache.close_and_drain().unwrap();
    let error = guaranteed_graph_cache_service()
        .expect_err("escaped parked graph companion must retain the prior guaranteed claim");
    assert_eq!(error.fact_value("reason"), Some("shared_governor_guaranteed_minimum_conflict"));
    assert!(companion.upgrade().is_some());

    drop(parked);
    drop(checkout);
    service.close_and_wait_for_quiescence(Duration::from_secs(5)).unwrap();
    assert!(companion.upgrade().is_none());
    let replacement = guaranteed_graph_cache_service().expect("last escaped graph owner releases the claim");
    drop(replacement);
}
