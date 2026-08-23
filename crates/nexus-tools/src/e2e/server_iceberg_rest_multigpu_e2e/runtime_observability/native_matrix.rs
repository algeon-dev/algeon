use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use super::*;

pub(super) async fn run_matrix(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    if params.scenario_selection.runs_native() {
        run_native_matrix(params, evidence).await?;
    }
    if params.scenario_selection.runs_datafusion_baseline() && !params.native_once {
        run_datafusion_baseline_phase(params, evidence).await?;
    }
    Ok(())
}

pub(super) async fn run_native_matrix(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    let mut server = spawn_server(params, evidence, "native-initial").await?;
    setup_local_table(params, &server.channel, "native-initial").await?;

    let effective_waves = if params.native_once { 1 } else { params.waves };
    for wave in 0..effective_waves {
        let (mut rows, wave_evidence) = run_success_wave(params, &server.channel, wave).await?;
        evidence.scenarios.append(&mut rows);
        evidence.waves.push(wave_evidence);
    }
    if !params.native_once {
        evidence.scenarios.push(run_native_join(params, &server.channel).await?);
        evidence.scenarios.push(run_variable_width_projection(params, &server.channel).await?);
    }
    wait_for_idle_with_timeout(params, &server.channel).await?;

    if params.native_once {
        let process = terminate_server(params, &mut server, "TERM", true).await?;
        evidence.processes.push(process);
        return Ok(());
    }

    evidence.scenarios.append(&mut run_structured_failures(params, &server.channel).await?);
    evidence.scenarios.push(run_cancel_query(params, &server.channel).await?);
    evidence.scenarios.push(run_drop_before_first_poll(params, &server.channel).await?);
    evidence.scenarios.push(run_midstream_drop(params, &server.channel).await?);
    evidence.retention = Some(run_retention_pressure(params, &server.channel, evidence).await?);
    wait_for_idle_with_timeout(params, &server.channel).await?;

    let (graceful_exit, graceful_query_ids, client_terminal_results) =
        terminate_gracefully_with_in_flight(params, &mut server).await?;
    evidence.processes.push(graceful_exit);

    let mut recovery = spawn_server(params, evidence, "native-graceful-restart").await?;
    setup_local_table(params, &recovery.channel, "native-graceful-restart").await?;
    let recovery_row = run_success_query(params, &recovery.channel, "graceful_restart_recovery", 0).await?;
    wait_for_idle_with_timeout(params, &recovery.channel).await?;
    evidence.restarts.push(RestartEvidence {
        iteration: 0,
        termination: "graceful_sigterm".to_string(),
        in_flight_query_id: None,
        in_flight_query_ids: graceful_query_ids,
        client_terminal_results,
        exit: evidence.processes.last().expect("graceful process evidence").clone_for_restart(),
        recovery_query_id: recovery_row.query_id.clone(),
        recovery_artifact_state: recovery_row.artifact.state.clone(),
        recovery_idle: true,
    });
    evidence.scenarios.push(recovery_row);

    for iteration in 1..=params.restart_iterations {
        let (query_id, stream) = start_in_flight_projection(params, &recovery.channel, iteration).await?;
        let active_frames =
            require_action_frames(&recovery.channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let active = parse_artifact(&query_id, None, &active_frames, None, None)?;
        if active.state != "active" {
            return Err(format!(
                "hard restart iteration {iteration} query {query_id} must be active before SIGKILL, got {}",
                active.state
            )
            .into());
        }
        let hard_exit = terminate_server(params, &mut recovery, "KILL", false).await?;
        drop(stream);
        if hard_exit.exit_signal != Some(9) {
            return Err(format!(
                "hard restart iteration {iteration} expected signal 9, got code={:?} signal={:?}",
                hard_exit.exit_code, hard_exit.exit_signal
            )
            .into());
        }
        evidence.processes.push(hard_exit);

        let label = format!("native-hard-restart-{iteration}");
        recovery = spawn_server(params, evidence, &label).await?;
        setup_local_table(params, &recovery.channel, &label).await?;
        let recovery_row = run_success_query(params, &recovery.channel, "hard_restart_recovery", iteration).await?;
        wait_for_idle_with_timeout(params, &recovery.channel).await?;
        evidence.restarts.push(RestartEvidence {
            iteration,
            termination: "hard_sigkill".to_string(),
            in_flight_query_id: Some(query_id.clone()),
            in_flight_query_ids: vec![query_id],
            client_terminal_results: vec!["stream_dropped_after_sigkill".to_string()],
            exit: evidence.processes.last().expect("hard process evidence").clone_for_restart(),
            recovery_query_id: recovery_row.query_id.clone(),
            recovery_artifact_state: recovery_row.artifact.state.clone(),
            recovery_idle: true,
        });
        evidence.scenarios.push(recovery_row);
    }

    let final_exit = terminate_server(params, &mut recovery, "TERM", true).await?;
    evidence.processes.push(final_exit);

    let global_byte_pressure = run_global_retained_byte_pressure_phase(params, evidence).await?;
    evidence
        .retention
        .as_mut()
        .expect("native retention evidence is installed before the dedicated byte-pressure phase")
        .global_byte_pressure = Some(global_byte_pressure);
    Ok(())
}

pub(super) fn native_sql() -> String {
    format!(
        "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM {TABLE_NAME} \
         WHERE l_shipdate >= DATE '1994-01-01' \
         AND l_shipdate < DATE '1995-01-01' \
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
    )
}

pub(super) fn cancellation_projection_sql() -> String {
    format!("SELECT l_orderkey, l_partkey FROM {TABLE_NAME}")
}

pub(super) fn broad_projection_sql() -> String {
    format!(
        "SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity, \
         l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, \
         l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct, l_shipmode, l_comment \
         FROM {TABLE_NAME} LIMIT {BROAD_PROJECTION_ROWS}"
    )
}

pub(super) fn native_join_sql() -> String {
    format!(
        "SELECT COUNT(*) AS joined_rows FROM {TABLE_NAME} l \
         JOIN {ORDERS_TABLE_NAME} o ON l.l_orderkey = o.o_orderkey \
         WHERE l.l_orderkey = 1"
    )
}

pub(super) async fn run_success_wave(
    params: &RuntimeParams,
    channel: &Channel,
    wave: usize,
) -> E2eResult<(Vec<ScenarioEvidence>, WaveEvidence)> {
    let started = Instant::now();
    let barrier = Arc::new(Barrier::new(params.concurrency + 1));
    let mut tasks = JoinSet::new();
    for index in 0..params.concurrency {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let params = params.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            run_success_query(&params, &channel, &format!("native_wave_{wave}"), index).await
        });
    }
    barrier.wait().await;
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut peak_active_queries = 0;
    let mut peak_active_query_jobs = 0;
    while !tasks.is_empty() {
        if let Ok(Ok(payload)) = tokio::time::timeout(Duration::from_millis(250), get_admission_stats(channel)).await {
            let (active_queries, active_query_jobs, reported_peak) = admission_activity(&payload)?;
            peak_active_queries = peak_active_queries.max(active_queries);
            peak_active_query_jobs = peak_active_query_jobs.max(active_query_jobs).max(reported_peak);
        }
        while let Some(result) = tasks.try_join_next() {
            match result {
                Ok(Ok(row)) => rows.push(row),
                Ok(Err(error)) => failures.push(error.to_string()),
                Err(error) => failures.push(format!("task join failure: {error}")),
            }
        }
        if !tasks.is_empty() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
    rows.sort_by(|left, right| left.correlation_id.cmp(&right.correlation_id));
    let elapsed = started.elapsed();
    let wave_evidence = WaveEvidence {
        wave,
        started: params.concurrency,
        completed: rows.len() + failures.len(),
        passed: rows.len(),
        failed: failures.len(),
        elapsed_ms: millis(elapsed),
        throughput_queries_per_second: rows.len() as f64 / elapsed.as_secs_f64().max(0.001),
        peak_active_queries,
        peak_active_query_jobs,
        overlap_proven: peak_active_queries >= 2 || peak_active_query_jobs >= 2,
    };
    if !failures.is_empty() {
        return Err(format!("native wave {wave} failures: {failures:?}").into());
    }
    if !wave_evidence.overlap_proven {
        return Err(format!(
            "native wave {wave} did not prove overlap: active_queries={} active_query_jobs={}",
            wave_evidence.peak_active_queries, wave_evidence.peak_active_query_jobs
        )
        .into());
    }
    Ok((rows, wave_evidence))
}

pub(super) async fn run_success_query(
    params: &RuntimeParams,
    channel: &Channel,
    scenario: &str,
    index: usize,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = format!("runtime-e2e-{scenario}-{index}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(native_sql(), None))
        .await
        .map_err(|_| format!("{scenario} GetFlightInfo timed out"))??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| format!("{scenario} DoGet timed out"))??;
    let batches = &collected.batches;
    if batches.iter().map(RecordBatch::num_rows).sum::<usize>() != 1 {
        return Err(format!("{scenario} native aggregate did not return exactly one row").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if artifact.plan_execution_ids.is_empty() {
        return Err(format!("native query {query_id} artifact has no plan_execution_id").into());
    }
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        execution_path: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: batches.len(),
        output_rows: 1,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "query_id".to_string(),
            "correlation_id".to_string(),
            "native_path".to_string(),
            "runtime_succeeded".to_string(),
            "no_multiple_sessions".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

pub(super) async fn run_native_join(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-native-join".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(native_join_sql(), None))
        .await
        .map_err(|_| "native join GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "native join DoGet timed out")??;
    let batches = &collected.batches;
    if batches.len() != 1 || batches[0].num_rows() != 1 || batches[0].num_columns() != 1 {
        return Err(format!("native join expected one scalar result, got {batches:?}").into());
    }
    let value = arrow::util::display::array_value_to_string(batches[0].column(0), 0)?;
    if value != "6" {
        return Err(format!("native join expected TPCH SF1 order 1 to have 6 lineitems, got {value:?}").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if artifact.plan_execution_ids.is_empty() {
        return Err(format!("native join query {query_id} artifact has no plan_execution_id").into());
    }
    Ok(ScenarioEvidence {
        scenario: "native_join".to_string(),
        execution_path: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: 1,
        output_rows: 1,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "exact_join_count=6".to_string(),
            "native_path".to_string(),
            "runtime_succeeded".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

pub(super) async fn run_variable_width_projection(
    params: &RuntimeParams,
    channel: &Channel,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-variable-width-projection".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "variable-width projection GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "variable-width projection DoGet timed out")??;
    let output_rows = collected.batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if output_rows != BROAD_PROJECTION_ROWS || collected.batches.iter().any(|batch| batch.num_columns() != 16) {
        return Err(format!(
            "variable-width projection returned {output_rows} rows across {} columns, expected {BROAD_PROJECTION_ROWS} rows across 16 columns",
            collected.batches.first().map_or(0, RecordBatch::num_columns)
        )
        .into());
    }
    let frame_evidence = collected.frame_evidence();
    validate_variable_frame_evidence(frame_evidence)?;

    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "native")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "succeeded")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: "variable_width_projection".to_string(),
        execution_path: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: collected.batches.len(),
        output_rows,
        encoded_frame_count: Some(frame_evidence.encoded_frame_count),
        record_batch_chunks: Some(frame_evidence.record_batch_chunks),
        max_encoded_frame_bytes: Some(frame_evidence.max_encoded_frame_bytes),
        stats_assertions: vec![
            "broad_projection_columns=16".to_string(),
            format!("record_batch_chunks={}", frame_evidence.record_batch_chunks),
            format!("max_encoded_frame_bytes={}", frame_evidence.max_encoded_frame_bytes),
            format!("frame_limit_bytes={FLIGHT_MAX_ENCODED_FRAME_BYTES}"),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

pub(super) async fn collect_flight_info(channel: &Channel, flight_info: FlightInfo) -> E2eResult<CollectedFlight> {
    let mut batches = Vec::new();
    let encoded_frame_count = Arc::new(AtomicUsize::new(0));
    let max_encoded_frame_bytes = Arc::new(AtomicUsize::new(0));
    for endpoint in flight_info.endpoint {
        let ticket = endpoint.ticket.ok_or("Flight SQL endpoint missing ticket")?;
        let response = FlightServiceClient::new(channel.clone()).do_get(Request::new(ticket)).await?.into_inner();
        let frame_count = Arc::clone(&encoded_frame_count);
        let max_frame_bytes = Arc::clone(&max_encoded_frame_bytes);
        let observed = response.map_err(FlightError::from).map_ok(move |frame| {
            observe_encoded_frame(&frame, &frame_count, &max_frame_bytes);
            frame
        });
        let mut stream = FlightRecordBatchStream::new_from_flight_data(observed);
        while let Some(batch) = stream.try_next().await? {
            batches.push(batch);
        }
    }
    Ok(CollectedFlight {
        batches,
        encoded_frame_count: encoded_frame_count.load(Ordering::Relaxed),
        max_encoded_frame_bytes: max_encoded_frame_bytes.load(Ordering::Relaxed),
    })
}

pub(super) fn first_ticket(flight_info: &FlightInfo) -> E2eResult<Ticket> {
    flight_info
        .endpoint
        .first()
        .and_then(|endpoint| endpoint.ticket.clone())
        .ok_or_else(|| "FlightInfo missing statement ticket".into())
}
