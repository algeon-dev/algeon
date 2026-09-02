use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use super::*;

pub(super) async fn terminate_gracefully_with_in_flight(
    params: &RuntimeParams,
    server: &mut RunningServer,
) -> E2eResult<(ProcessEvidence, Vec<String>, Vec<String>)> {
    let mut statements = Vec::with_capacity(params.concurrency);
    for index in 0..params.concurrency {
        let correlation_id = format!("runtime-e2e-graceful-in-flight-{index}");
        let mut client = correlated_client(&server.channel, &correlation_id);
        let info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
            .await
            .map_err(|_| format!("graceful in-flight {index} GetFlightInfo timed out"))??;
        let query_id = query_id_from_statement_flight_info(&info)?;
        statements.push((query_id, first_ticket(&info)?));
    }

    let start = Arc::new(Barrier::new(statements.len() + 1));
    let release = Arc::new(Barrier::new(statements.len() + 1));
    let (ready_tx, mut ready_rx) = mpsc::channel(statements.len());
    let mut tasks = JoinSet::new();
    let query_ids = statements.iter().map(|(query_id, _)| query_id.clone()).collect::<Vec<_>>();
    for (query_id, ticket) in statements {
        let channel = server.channel.clone();
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);
        let ready_tx = ready_tx.clone();
        tasks.spawn(async move {
            start.wait().await;
            let mut client = FlightSqlServiceClient::new(channel);
            let mut stream = client.do_get(ticket).await?;
            ready_tx.send(query_id.clone()).await?;
            release.wait().await;
            let mut batches = 0usize;
            let mut rows = 0usize;
            loop {
                match stream.try_next().await {
                    Ok(Some(batch)) => {
                        batches += 1;
                        rows += batch.num_rows();
                    },
                    Ok(None) => {
                        return Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                            query_id,
                            format!("eof:batches={batches},rows={rows}"),
                        ));
                    },
                    Err(error) => {
                        return Ok((query_id, format!("stream_error:{error}")));
                    },
                }
            }
        });
    }
    drop(ready_tx);
    start.wait().await;
    let ready_deadline = Instant::now() + params.poll_timeout;
    let mut ready = BTreeSet::new();
    while ready.len() < query_ids.len() {
        let remaining = ready_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(
                format!("only {}/{} graceful restart streams became ready", ready.len(), query_ids.len()).into()
            );
        }
        let query_id = tokio::time::timeout(remaining, ready_rx.recv())
            .await
            .map_err(|_| "graceful restart stream readiness timed out")?
            .ok_or("graceful restart readiness channel closed early")?;
        ready.insert(query_id);
    }
    let mut active = 0usize;
    for query_id in &query_ids {
        let frames = require_action_frames(&server.channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let artifact = parse_artifact(query_id, None, &frames, None, None)?;
        if artifact.state == "active" {
            active += 1;
        }
    }
    if active < 2 || active != query_ids.len() {
        return Err(format!(
            "graceful restart requires all {} concurrent streams active, observed {active}",
            query_ids.len()
        )
        .into());
    }

    let termination = terminate_server(params, server, "TERM", true);
    tokio::pin!(termination);
    let exit = tokio::select! {
        result = &mut termination => {
            release.wait().await;
            result?
        },
        () = tokio::time::sleep(Duration::from_millis(50)) => {
            release.wait().await;
            termination.await?
        },
    };
    let mut terminal_results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        let (query_id, terminal) = result.map_err(|error| format!("graceful client task failed: {error}"))??;
        terminal_results.push(format!("{query_id}:{terminal}"));
    }
    terminal_results.sort();
    if terminal_results.len() != query_ids.len() {
        return Err(format!(
            "graceful restart captured {} client terminal results for {} streams",
            terminal_results.len(),
            query_ids.len()
        )
        .into());
    }
    Ok((exit, query_ids, terminal_results))
}

pub(super) async fn run_retention_pressure(
    params: &RuntimeParams,
    channel: &Channel,
    evidence: &mut RunEvidence,
) -> E2eResult<RetentionEvidence> {
    let (live_query_id, live_stream) = start_in_flight_projection(params, channel, 10_000).await?;
    require_artifact_state(channel, &live_query_id, "active").await?;

    let generation_count = params.observability.retained_count.saturating_add(2);
    let mut generated = Vec::with_capacity(generation_count);
    let mut generated_wire_bytes = 0usize;
    for index in 0..generation_count {
        let row = run_success_query(params, channel, "retention_count_pressure", index).await?;
        generated_wire_bytes = generated_wire_bytes.saturating_add(row.artifact.wire_bytes);
        generated.push((row.query_id.clone(), Instant::now()));
        evidence.scenarios.push(row);
        require_artifact_state(channel, &live_query_id, "active").await?;
    }

    let mut sealed_count = 0usize;
    let mut unavailable = BTreeMap::new();
    let not_found = Vec::new();
    let no_silent_loss = true;
    for (query_id, sealed_at) in &generated {
        match optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await? {
            Some(frames) => {
                let artifact = parse_artifact(query_id, None, &frames, None, None)?;
                match artifact.state.as_str() {
                    "sealed" => sealed_count += 1,
                    "unavailable" => {
                        let reason = artifact.reason.unwrap_or_else(|| "missing_reason".to_string());
                        if reason != "evicted" {
                            return Err(format!(
                                "count retention query {query_id} became unavailable for {reason}, expected evicted"
                            )
                            .into());
                        }
                        unavailable.insert(query_id.clone(), reason);
                    },
                    other => return Err(format!("retention query {query_id} has unexpected state {other}").into()),
                }
            },
            None => {
                let elapsed = sealed_at.elapsed();
                return Err(format!(
                    "count retention query {query_id} became NotFound after {}ms without an unavailable tombstone",
                    millis(elapsed)
                )
                .into());
            },
        }
    }
    if sealed_count > params.observability.retained_count {
        return Err(format!(
            "retention kept {sealed_count} artifacts above count limit {}",
            params.observability.retained_count
        )
        .into());
    }
    if unavailable.is_empty() && not_found.is_empty() {
        return Err("count retention pressure did not evict any generated artifact".into());
    }

    let (ttl_query_id, ttl_sealed_at) = generated.last().expect("retention generation is nonempty").clone();
    let (ttl_terminal_state, ttl_unavailable_elapsed_ms, ttl_not_found_elapsed_ms) =
        wait_through_artifact_ttl(params, channel, &ttl_query_id, ttl_sealed_at).await?;
    require_artifact_state(channel, &live_query_id, "active").await?;

    let per_query_bytes = params.observability.per_query_bytes;
    if per_query_bytes > 2 * 1024 * 1024 {
        return Err(
            "--per-query-bytes above 2 MiB makes the capture-budget statement unsafe; use a smaller value".into()
        );
    }
    let padding_len = usize::try_from(per_query_bytes).expect("validated 2 MiB bound fits usize") + 4096;
    let padded_sql = format!("/*{}*/ {}", "x".repeat(padding_len), native_sql());
    let capture_budget_input_bytes = padded_sql.len();
    let mut client = correlated_client(channel, "runtime-e2e-capture-budget-pressure");
    let info = tokio::time::timeout(params.query_timeout, client.execute(padded_sql, None))
        .await
        .map_err(|_| "capture-budget GetFlightInfo timed out")??;
    let capture_budget_query_id = query_id_from_statement_flight_info(&info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, info))
        .await
        .map_err(|_| "capture-budget DoGet timed out")??;
    if collected.batches.iter().map(RecordBatch::num_rows).sum::<usize>() != 1 {
        return Err("capture-budget Parquet query did not return one row".into());
    }
    let capture_stats = wait_for_sealed_stats(params, channel, &capture_budget_query_id).await?;
    let stats_incomplete = capture_stats.contains("# incomplete=true\n");
    let retained_byte_loss_recorded = capture_stats.contains("# capture_loss.retained_byte_limit=");
    if !stats_incomplete || !retained_byte_loss_recorded {
        return Err(format!(
            "capture-budget query {capture_budget_query_id} did not report bounded retained-byte loss:\n{capture_stats}"
        )
        .into());
    }
    let capture_frames = wait_for_terminal_artifact(params, channel, &capture_budget_query_id).await?;
    let capture_artifact = parse_artifact(&capture_budget_query_id, None, &capture_frames, None, None)?;
    let capture_budget_reason = capture_artifact.reason.clone().unwrap_or_else(|| "missing_reason".to_string());
    if capture_artifact.state != "unavailable" || capture_budget_reason != "capture_budget" {
        return Err(format!(
            "capture-budget query expected unavailable/capture_budget, got {}/{}",
            capture_artifact.state, capture_budget_reason
        )
        .into());
    }
    require_artifact_state(channel, &live_query_id, "active").await?;
    drop(live_stream);

    Ok(RetentionEvidence {
        generated_query_ids: generated.into_iter().map(|(query_id, _)| query_id).collect(),
        live_query_id,
        live_state_during_pressure: "active".to_string(),
        sealed_count,
        unavailable,
        not_found,
        generated_wire_bytes,
        capture_budget: CaptureBudgetEvidence {
            query_id: capture_budget_query_id,
            input_bytes: capture_budget_input_bytes,
            per_query_capture_limit: params.observability.per_query_bytes,
            state: capture_artifact.state,
            reason: capture_budget_reason,
            frame_count: capture_artifact.frame_count,
            stats_incomplete,
            retained_byte_loss_recorded,
        },
        global_byte_pressure: None,
        ttl_query_id,
        ttl_terminal_state,
        ttl_unavailable_elapsed_ms,
        ttl_not_found_elapsed_ms,
        no_silent_loss,
    })
}

pub(super) async fn run_global_retained_byte_pressure_phase(
    params: &RuntimeParams,
    evidence: &mut RunEvidence,
) -> E2eResult<GlobalBytePressureEvidence> {
    let mut pressure_params = params.clone();
    pressure_params.observability = ObservabilityLimits {
        per_query_records: 256,
        per_query_bytes: GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES,
        retained_count: GLOBAL_BYTE_PRESSURE_RETAINED_COUNT,
        retained_bytes: GLOBAL_BYTE_PRESSURE_RETAINED_BYTES,
        ttl_secs: GLOBAL_BYTE_PRESSURE_TTL_SECS,
        shutdown_grace_secs: params.observability.shutdown_grace_secs,
    };
    let label = "native-global-byte-pressure";
    let mut server = spawn_server(&pressure_params, evidence, label).await?;
    let phase_result = async {
        setup_local_table(&pressure_params, &server.channel, label).await?;
        run_global_retained_byte_pressure(&pressure_params, &server.channel, label, &mut evidence.scenarios).await
    }
    .await;
    let termination = terminate_server(&pressure_params, &mut server, "TERM", true).await;
    match (phase_result, termination) {
        (Ok(phase), Ok(process)) => {
            evidence.processes.push(process);
            Ok(phase)
        },
        (Err(error), Ok(process)) => {
            evidence.processes.push(process);
            Err(error)
        },
        (Ok(_), Err(error)) => Err(error),
        (Err(phase_error), Err(termination_error)) => Err(format!(
            "global byte-pressure phase failed: {phase_error:#}; dedicated server termination also failed: {termination_error:#}"
        )
        .into()),
    }
}

pub(super) async fn run_global_retained_byte_pressure(
    params: &RuntimeParams,
    channel: &Channel,
    server_label: &str,
    scenarios: &mut Vec<ScenarioEvidence>,
) -> E2eResult<GlobalBytePressureEvidence> {
    if GLOBAL_BYTE_PRESSURE_MAX_QUERIES.saturating_add(GLOBAL_BYTE_PRESSURE_SETUP_QUERIES)
        >= params.observability.retained_count
    {
        return Err("global byte-pressure count limit does not leave room for every bounded probe query".into());
    }

    let mut generated_query_ids = Vec::new();
    let mut artifact_charge_bytes = BTreeMap::new();
    let mut generated_charge_bytes = 0u64;
    let mut max_observed_generated_retained_bytes = 0u64;
    let mut previously_evicted = BTreeSet::new();
    let mut terminal_snapshot = None;

    for index in 0..GLOBAL_BYTE_PRESSURE_MAX_QUERIES {
        let row = run_success_query(params, channel, "global_retained_byte_pressure", index).await?;
        let stats = get_query_stats(channel, &row.query_id).await?;
        let charge = artifact_charge(row.artifact.wire_bytes, stats.len())?;
        if charge > params.observability.retained_bytes {
            return Err(format!(
                "global byte-pressure query {} individually charged {charge} bytes above cap {}",
                row.query_id, params.observability.retained_bytes
            )
            .into());
        }
        generated_charge_bytes = generated_charge_bytes
            .checked_add(charge)
            .ok_or_else(|| "global byte-pressure generated charge accounting overflowed".to_string())?;
        artifact_charge_bytes.insert(row.query_id.clone(), charge);
        generated_query_ids.push(row.query_id.clone());
        scenarios.push(row);

        let snapshot = observe_global_byte_pressure_snapshot(params, channel, &generated_query_ids).await?;
        if snapshot.retained_bytes > params.observability.retained_bytes {
            return Err(format!(
                "global byte-pressure retained {} generated bytes above cap {}",
                snapshot.retained_bytes, params.observability.retained_bytes
            )
            .into());
        }
        if previously_evicted.iter().any(|query_id| !snapshot.evicted.contains_key(query_id)) {
            return Err("global byte-pressure eviction state regressed to a retained artifact".into());
        }
        for (query_id, observed_charge) in &snapshot.sealed_charge_bytes {
            let expected_charge = artifact_charge_bytes
                .get(query_id)
                .ok_or_else(|| format!("global byte-pressure snapshot returned unknown query {query_id}"))?;
            if observed_charge != expected_charge {
                return Err(format!(
                    "global byte-pressure query {query_id} charge changed from {expected_charge} to {observed_charge}"
                )
                .into());
            }
        }
        previously_evicted.extend(snapshot.evicted.keys().cloned());
        max_observed_generated_retained_bytes = max_observed_generated_retained_bytes.max(snapshot.retained_bytes);

        if !snapshot.evicted.is_empty() && generated_charge_bytes > params.observability.retained_bytes {
            terminal_snapshot = Some(snapshot);
            break;
        }
    }

    let snapshot = terminal_snapshot.ok_or_else(|| {
        format!(
            "global retained-byte pressure did not evict a generated artifact after {} bounded Parquet queries",
            GLOBAL_BYTE_PRESSURE_MAX_QUERIES
        )
    })?;
    let count_limit_not_reached = generated_query_ids.len().saturating_add(GLOBAL_BYTE_PRESSURE_SETUP_QUERIES)
        < params.observability.retained_count;
    if !count_limit_not_reached {
        return Err(format!(
            "global byte-pressure generated {} queries plus {} setup queries against count limit {}",
            generated_query_ids.len(),
            GLOBAL_BYTE_PRESSURE_SETUP_QUERIES,
            params.observability.retained_count
        )
        .into());
    }
    if snapshot.sealed_query_ids.is_empty() {
        return Err("global byte-pressure evicted every generated artifact instead of retaining a bounded set".into());
    }

    Ok(GlobalBytePressureEvidence {
        server_label: server_label.to_string(),
        per_query_bytes: params.observability.per_query_bytes,
        retained_count_limit: params.observability.retained_count,
        retained_byte_limit: params.observability.retained_bytes,
        ttl_secs: params.observability.ttl_secs,
        generated_query_ids,
        artifact_charge_bytes,
        sealed_query_ids_after_pressure: snapshot.sealed_query_ids,
        evicted: snapshot.evicted,
        generated_charge_bytes,
        max_observed_generated_retained_bytes,
        retained_bytes_after_pressure: snapshot.retained_bytes,
        setup_artifacts_known_evicted: true,
        exact_retained_set_proven: true,
        count_limit_not_reached,
        byte_limit_respected: true,
        no_silent_loss: true,
    })
}

pub(super) async fn observe_global_byte_pressure_snapshot(
    params: &RuntimeParams,
    channel: &Channel,
    query_ids: &[String],
) -> E2eResult<GlobalBytePressureSnapshot> {
    let mut sealed_query_ids = Vec::new();
    let mut sealed_charge_bytes = BTreeMap::new();
    let mut evicted = BTreeMap::new();
    let mut retained_bytes = 0u64;
    for query_id in query_ids {
        let frames = optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        let Some(frames) = frames else {
            return Err(format!(
                "global byte-pressure query {query_id} became NotFound before the configured {}s TTL",
                params.observability.ttl_secs
            )
            .into());
        };
        let artifact = parse_artifact(query_id, None, &frames, None, Some(&params.observability))?;
        match artifact.state.as_str() {
            "sealed" => {
                let stats = get_query_stats(channel, query_id).await?;
                let charge = artifact_charge(artifact.wire_bytes, stats.len())?;
                retained_bytes = retained_bytes
                    .checked_add(charge)
                    .ok_or_else(|| "global byte-pressure retained charge accounting overflowed".to_string())?;
                sealed_query_ids.push(query_id.clone());
                sealed_charge_bytes.insert(query_id.clone(), charge);
            },
            "unavailable" if artifact.reason.as_deref() == Some("evicted") => {
                evicted.insert(query_id.clone(), "evicted".to_string());
            },
            state => {
                return Err(format!(
                    "global byte-pressure query {query_id} entered {state}/{:?}, expected sealed or unavailable/evicted",
                    artifact.reason
                )
                .into());
            },
        }
    }
    if sealed_query_ids.len().saturating_add(evicted.len()) != query_ids.len() {
        return Err("global byte-pressure snapshot silently lost a generated query".into());
    }
    Ok(GlobalBytePressureSnapshot { sealed_query_ids, sealed_charge_bytes, evicted, retained_bytes })
}

pub(super) async fn wait_for_sealed_stats(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
) -> E2eResult<String> {
    let deadline = Instant::now() + params.poll_timeout;
    loop {
        let stats = get_query_stats(channel, query_id).await?;
        if stats.contains("# sealed=true\n") {
            return Ok(stats);
        }
        if Instant::now() >= deadline {
            return Err(
                format!("query {query_id} stats did not seal within {:?}:\n{stats}", params.poll_timeout).into()
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

pub(super) async fn wait_for_sealed_artifact(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
) -> E2eResult<Vec<Vec<u8>>> {
    let frames = wait_for_terminal_artifact(params, channel, query_id).await?;
    let state = control_state(query_id, &frames)?;
    if state != "sealed" {
        return Err(format!("query {query_id} artifact became {state}, expected sealed").into());
    }
    let repeated = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
    if repeated != frames {
        return Err(format!("query {query_id} sealed artifact bytes changed across repeated retrieval").into());
    }
    Ok(frames)
}

pub(super) async fn wait_for_terminal_artifact(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
) -> E2eResult<Vec<Vec<u8>>> {
    let deadline = Instant::now() + params.poll_timeout;
    loop {
        let frames = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
        match control_state(query_id, &frames)?.as_str() {
            "sealed" | "unavailable" => return Ok(frames),
            "pending" | "active" if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            },
            state => {
                return Err(
                    format!("query {query_id} artifact remained {state} beyond {:?}", params.poll_timeout).into()
                );
            },
        }
    }
}

pub(super) async fn require_action_frames(
    channel: &Channel,
    action_type: &str,
    body: &[u8],
) -> E2eResult<Vec<Vec<u8>>> {
    action_frames(channel, action_type, body.to_vec()).await
}

pub(super) async fn optional_action_frames(
    channel: &Channel,
    action_type: &str,
    body: &[u8],
) -> E2eResult<Option<Vec<Vec<u8>>>> {
    match action_frames(channel, action_type, body.to_vec()).await {
        Ok(frames) => Ok(Some(frames)),
        Err(error) if boxed_status_code(error.as_ref()) == Some(Code::NotFound) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) async fn require_artifact_state(channel: &Channel, query_id: &str, expected: &str) -> E2eResult<()> {
    let frames = require_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await?;
    let artifact = parse_artifact(query_id, None, &frames, None, None)?;
    if artifact.state != expected {
        return Err(format!("query {query_id} artifact state {}, expected {expected}", artifact.state).into());
    }
    Ok(())
}

pub(super) async fn wait_through_artifact_ttl(
    params: &RuntimeParams,
    channel: &Channel,
    query_id: &str,
    sealed_at: Instant,
) -> E2eResult<(String, u64, u64)> {
    let retained_ttl = Duration::from_secs(params.observability.ttl_secs);
    let full_lifetime = retained_ttl.saturating_mul(2);
    let tolerance = Duration::from_millis(100).min(retained_ttl);
    let unavailable_deadline = sealed_at + retained_ttl + params.poll_timeout;
    let not_found_deadline = sealed_at + full_lifetime + params.poll_timeout;
    let mut unavailable_elapsed = None;
    loop {
        match optional_action_frames(channel, "GetQueryRuntimeArtifact", query_id.as_bytes()).await? {
            Some(frames) => {
                let artifact = parse_artifact(query_id, None, &frames, None, None)?;
                match artifact.state.as_str() {
                    "sealed" if unavailable_elapsed.is_none() => {},
                    "unavailable" if artifact.reason.as_deref() == Some("evicted") => {
                        let elapsed = sealed_at.elapsed();
                        if elapsed.saturating_add(tolerance) < retained_ttl {
                            return Err(format!(
                                "TTL query {query_id} became unavailable after {}ms, before retained TTL {}ms",
                                millis(elapsed),
                                millis(retained_ttl)
                            )
                            .into());
                        }
                        unavailable_elapsed.get_or_insert(elapsed);
                    },
                    other => {
                        return Err(format!("TTL query {query_id} entered unexpected state {other}").into());
                    },
                }
            },
            None => {
                let elapsed = sealed_at.elapsed();
                let Some(unavailable_elapsed) = unavailable_elapsed else {
                    return Err(format!("TTL query {query_id} became NotFound without an unavailable tombstone").into());
                };
                if elapsed.saturating_add(tolerance) < full_lifetime {
                    return Err(format!(
                        "TTL query {query_id} became NotFound after {}ms, before retained+tombstone TTL {}ms",
                        millis(elapsed),
                        millis(full_lifetime)
                    )
                    .into());
                }
                return Ok((
                    "unavailable:evicted_then_not_found".to_string(),
                    millis(unavailable_elapsed),
                    millis(elapsed),
                ));
            },
        }
        if unavailable_elapsed.is_none() && Instant::now() >= unavailable_deadline {
            return Err(format!("TTL query {query_id} did not become unavailable within the bounded TTL poll").into());
        }
        if Instant::now() >= not_found_deadline {
            return Err(format!("TTL query {query_id} tombstone did not expire within the bounded poll").into());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
