use super::*;

pub(super) async fn run_profile_stress(
    config: &StressConfig,
    profile: &StressProfileConfig,
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
) -> DataFusionResult<ProfileStressResult> {
    let lanes = admit_profile_lanes(Arc::clone(&service), profile.lanes).await?;
    let expected_leases = profile_expected_root_leases(&lanes);
    let mut evidence = ProfileEvidenceReport::default();

    for iteration in 1..=profile.warmup_iterations {
        let batch = prepare_profile_iteration(config, profile, &lanes, "warmup", iteration).await?;
        let batches = batch.into_iter().map(|request| vec![request]).collect();
        let results = execute_profile_batches(Arc::clone(&service), Arc::clone(&cpu_baselines), batches).await?;
        wait_for_service_stream_pools_quiescent(
            Arc::clone(&service),
            expected_leases.clone(),
            config.stream_idle_timeout,
        )
        .await?;
        log_profile_cache_evidence(&results);
        validate_profile_results(&results, config.source_cache_max_retained_bytes.is_some() && iteration >= 2)?;
        evidence.rows.extend(results.into_iter().map(|result| result.evidence));
        eprintln!(
            "[query_service_tpch_stress] profile warmup iteration {iteration}/{} complete",
            profile.warmup_iterations
        );
    }

    let measured_batches = prepare_profile_measurements(config, profile, &lanes).await?;
    wait_for_service_stream_pools_quiescent(Arc::clone(&service), expected_leases.clone(), config.stream_idle_timeout)
        .await?;

    let profiler_capture = CudaProfilerCapture::start(config.cuda_profiler_capture)?;
    let measured_started = Instant::now();
    let execution_result =
        execute_profile_batches(Arc::clone(&service), Arc::clone(&cpu_baselines), measured_batches).await;
    let measured_execution_ms = duration_ms(measured_started.elapsed());
    let quiescence_result =
        wait_for_service_stream_pools_quiescent(Arc::clone(&service), expected_leases, config.stream_idle_timeout)
            .await;
    let profiler_result = profiler_capture.finish();
    let results = execution_result?;
    quiescence_result?;
    profiler_result?;

    log_profile_cache_evidence(&results);
    validate_profile_results(&results, config.source_cache_max_retained_bytes.is_some())?;
    let rows = results.iter().map(|result| result.row.clone()).collect::<Vec<_>>();
    evidence.rows.extend(results.into_iter().map(|result| result.evidence));

    drop(lanes);
    wait_for_service_stream_pools_idle(service, config.stream_idle_timeout).await?;
    Ok(ProfileStressResult { rows, evidence, measured_execution_ms })
}

async fn admit_profile_lanes(service: Arc<QueryService>, lanes: usize) -> DataFusionResult<Vec<ProfileLane>> {
    let mut admitted = Vec::with_capacity(lanes);
    for lane in 1..=lanes {
        let service_for_admission = Arc::clone(&service);
        let handle = tokio::task::spawn_blocking(move || service_for_admission.admit_query("bulk"))
            .await
            .map_err(|error| plan_error(format!("profile lane admission task join failed: {error}")))?
            .map_err(native_error)?;
        admitted.push(ProfileLane { lane, handle: Arc::new(handle) });
    }
    Ok(admitted)
}

fn profile_expected_root_leases(lanes: &[ProfileLane]) -> Vec<(usize, usize)> {
    let mut by_device = BTreeMap::new();
    for lane in lanes {
        *by_device.entry(lane.handle.device_ordinal()).or_insert(0) += 1;
    }
    by_device.into_iter().collect()
}

async fn wait_for_service_stream_pools_quiescent(
    service: Arc<QueryService>,
    expected_leases: Vec<(usize, usize)>,
    timeout: Duration,
) -> DataFusionResult<()> {
    tokio::task::spawn_blocking(move || service.wait_for_stream_pools_quiescent(&expected_leases, timeout))
        .await
        .map_err(|error| plan_error(format!("query service stream-quiescence wait task join failed: {error}")))?
        .map_err(native_error)
}

async fn prepare_profile_iteration(
    config: &StressConfig,
    profile: &StressProfileConfig,
    lanes: &[ProfileLane],
    phase: &'static str,
    iteration: usize,
) -> DataFusionResult<Vec<PreparedProfileRequest>> {
    let query = tpch_query(&profile.query_id)?;
    let mut prepared = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let request = profile_request(profile, lane.lane, iteration);
        let plan = prepare_native_with_handle(config, query, Arc::clone(&lane.handle)).await?;
        prepared.push(PreparedProfileRequest {
            phase,
            lane: lane.lane,
            request,
            handle: Arc::clone(&lane.handle),
            prepared: plan,
        });
    }
    Ok(prepared)
}

async fn prepare_profile_measurements(
    config: &StressConfig,
    profile: &StressProfileConfig,
    lanes: &[ProfileLane],
) -> DataFusionResult<Vec<Vec<PreparedProfileRequest>>> {
    let query = tpch_query(&profile.query_id)?;
    let mut batches = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let mut requests = Vec::with_capacity(profile.measured_iterations);
        for iteration in 1..=profile.measured_iterations {
            let request = profile_request(profile, lane.lane, iteration);
            let plan = prepare_native_with_handle(config, query, Arc::clone(&lane.handle)).await?;
            requests.push(PreparedProfileRequest {
                phase: "measured",
                lane: lane.lane,
                request,
                handle: Arc::clone(&lane.handle),
                prepared: plan,
            });
        }
        batches.push(requests);
    }
    Ok(batches)
}

fn profile_request(profile: &StressProfileConfig, lane: usize, iteration: usize) -> StressRequest {
    StressRequest {
        request_index: (iteration - 1) * profile.lanes + lane,
        tenant_id: TenantId::new("bulk"),
        workload_class: "profile",
        query_id: profile.query_id.clone(),
        iteration,
    }
}

async fn execute_profile_batches(
    service: Arc<QueryService>,
    cpu_baselines: Arc<HashMap<String, CpuBaseline>>,
    batches: Vec<Vec<PreparedProfileRequest>>,
) -> DataFusionResult<Vec<ProfileRequestResult>> {
    let barrier = Arc::new(Barrier::new(batches.len().max(1)));
    let mut tasks = Vec::with_capacity(batches.len());
    for batch in batches {
        let service = Arc::clone(&service);
        let cpu_baselines = Arc::clone(&cpu_baselines);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let mut results = Vec::with_capacity(batch.len());
            for request in batch {
                results.push(execute_prepared_profile_request(&service, &cpu_baselines, request).await?);
            }
            Ok::<Vec<ProfileRequestResult>, DataFusionError>(results)
        }));
    }

    let mut results = Vec::new();
    for task in tasks {
        results.extend(task.await.map_err(|error| plan_error(format!("profile lane task join failed: {error}")))??);
    }
    results.sort_by_key(|result| (result.row.iteration, result.evidence.lane));
    Ok(results)
}

async fn execute_prepared_profile_request(
    service: &QueryService,
    cpu_baselines: &HashMap<String, CpuBaseline>,
    prepared: PreparedProfileRequest,
) -> DataFusionResult<ProfileRequestResult> {
    let PreparedProfileRequest { phase, lane, request, handle, prepared } = prepared;
    let total_started = Instant::now();
    let native_started = Instant::now();
    let admission_sequence = handle.admission_sequence();
    let device_ordinal = handle.device_ordinal();
    let admission_wait_ms = Some(duration_ms(handle.admission_wait()));
    let admission_facts = admission_facts(service, &handle)?;
    let cpu = cpu_baselines.get(&request.query_id);

    let (row, metrics) = match collect_prepared_native(prepared).await {
        Ok(NativeCollectOutcome::Succeeded { planning_row, measurement: native, metrics }) => {
            let parity = match cpu {
                Some(cpu) if cpu.output_fingerprint == native.output_fingerprint => "matched",
                Some(_) => "mismatched",
                None => "cpu_missing",
            };
            let row = StressRow {
                row_type: "query",
                request_index: request.request_index,
                tenant_id: request.tenant_id.to_string(),
                workload_class: request.workload_class,
                query_id: request.query_id.clone(),
                iteration: request.iteration,
                admission_sequence: Some(admission_sequence),
                admission_wait_ms,
                admission_facts,
                total_wall_ms: duration_ms(total_started.elapsed()),
                native_wall_ms: Some(duration_ms(native_started.elapsed())),
                planning_outcome: planning_row
                    .outcome
                    .map_or("none", NativeDataFusionPlanningOutcome::as_str)
                    .to_string(),
                final_disposition: planning_row.final_disposition.as_str().to_string(),
                native_runtime_outcome: "succeeded",
                parity_status: parity,
                cpu_output_rows: cpu.map(|cpu| cpu.output_rows),
                native_output_rows: Some(native.output_rows),
                cpu_output_fingerprint: cpu.map(|cpu| cpu.output_fingerprint.clone()),
                native_output_fingerprint: Some(native.output_fingerprint),
                error: StructuredFailure::default(),
            };
            (row, metrics)
        },
        Ok(NativeCollectOutcome::RuntimeFailed { planning_row, error, metrics }) => {
            let row = failure_row(
                &request,
                total_started.elapsed(),
                Some(admission_sequence),
                admission_wait_ms,
                admission_facts,
                planning_row.outcome.map_or("none", NativeDataFusionPlanningOutcome::as_str),
                planning_row.final_disposition.as_str(),
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            );
            (row, metrics)
        },
        Err(error) => {
            let row = failure_row(
                &request,
                total_started.elapsed(),
                Some(admission_sequence),
                admission_wait_ms,
                admission_facts,
                "unknown",
                "unknown",
                "failed",
                "native_failed",
                StructuredFailure::from_datafusion_error(&error),
                cpu,
                None,
            );
            (row, NativeDataFusionExecutionMetrics::default())
        },
    };

    let evidence = ProfileEvidenceRow {
        phase,
        lane,
        iteration: row.iteration,
        request_index: row.request_index,
        admission_sequence,
        device_ordinal,
        query_id: row.query_id.clone(),
        native_runtime_outcome: row.native_runtime_outcome,
        parity_status: row.parity_status,
        source_cache_hits: metrics.source_cache_hits,
        source_cache_misses: metrics.source_cache_misses,
        source_cache_inserts: metrics.source_cache_inserts,
        source_cache_insert_skipped: metrics.source_cache_insert_skipped,
        source_cache_retained_bytes: metrics.source_cache_retained_bytes,
        native_wall_ms: row.native_wall_ms,
    };
    Ok(ProfileRequestResult { row, evidence })
}

fn validate_profile_results(results: &[ProfileRequestResult], require_cache_only: bool) -> DataFusionResult<()> {
    for result in results {
        let row = &result.row;
        let evidence = &result.evidence;
        if row.native_runtime_outcome != "succeeded" || row.parity_status != "matched" {
            return Err(plan_error(format!(
                "profile {} lane {} iteration {} failed: runtime_outcome={} parity_status={} error={}",
                evidence.phase,
                evidence.lane,
                evidence.iteration,
                row.native_runtime_outcome,
                row.parity_status,
                row.error.message.as_deref().unwrap_or("none"),
            )));
        }
        if require_cache_only
            && (evidence.source_cache_hits == 0
                || evidence.source_cache_misses != 0
                || evidence.source_cache_inserts != 0)
        {
            return Err(plan_error(format!(
                "profile {} lane {} iteration {} was not cache-only: hits={} misses={} inserts={} skipped={}",
                evidence.phase,
                evidence.lane,
                evidence.iteration,
                evidence.source_cache_hits,
                evidence.source_cache_misses,
                evidence.source_cache_inserts,
                evidence.source_cache_insert_skipped,
            )));
        }
    }
    Ok(())
}

fn log_profile_cache_evidence(results: &[ProfileRequestResult]) {
    for result in results {
        let evidence = &result.evidence;
        eprintln!(
            "[query_service_tpch_stress] profile phase={} lane={} iteration={} device={} hits={} misses={} inserts={} skipped={} retained_bytes={} native_wall_ms={}",
            evidence.phase,
            evidence.lane,
            evidence.iteration,
            evidence.device_ordinal,
            evidence.source_cache_hits,
            evidence.source_cache_misses,
            evidence.source_cache_inserts,
            evidence.source_cache_insert_skipped,
            evidence.source_cache_retained_bytes,
            evidence.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
        );
    }
}
