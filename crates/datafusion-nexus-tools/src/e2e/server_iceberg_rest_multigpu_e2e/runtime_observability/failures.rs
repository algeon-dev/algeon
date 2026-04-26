use super::*;

pub(super) async fn run_structured_failures(
    params: &RuntimeParams,
    channel: &Channel,
) -> E2eResult<Vec<ScenarioEvidence>> {
    let select = run_structured_failure_case(
        params,
        channel,
        "invalid_select",
        "SELECT missing_runtime_observability_column FROM runtime_observability_lineitem",
        false,
    )
    .await?;
    let ddl = run_structured_failure_case(
        params,
        channel,
        "invalid_ddl_update",
        "DROP TABLE runtime_observability_table_that_does_not_exist",
        true,
    )
    .await?;
    Ok(vec![select, ddl])
}

pub(super) async fn run_structured_failure_case(
    params: &RuntimeParams,
    channel: &Channel,
    scenario: &str,
    sql: &str,
    update: bool,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let error = if update {
        tokio::time::timeout(params.query_timeout, client.execute_update(sql.to_string(), None))
            .await
            .map_err(|_| format!("{scenario} timed out"))?
            .expect_err("invalid DDL update must fail")
    } else {
        tokio::time::timeout(params.query_timeout, client.execute(sql.to_string(), None))
            .await
            .map_err(|_| format!("{scenario} timed out"))?
            .expect_err("invalid SELECT must fail")
    };
    let status = flight_status(error)?;
    if !matches!(status.code(), Code::InvalidArgument | Code::NotFound | Code::Unimplemented) {
        return Err(format!("{scenario} returned unexpected status {status:?}").into());
    }
    let query_id = status_metadata(&status, QUERY_ID_METADATA, scenario)?;
    let response_correlation = status_metadata(&status, CORRELATION_HEADER, scenario)?;
    if response_correlation != correlation_id {
        return Err(format!(
            "{scenario} response correlation id {response_correlation:?} did not match {correlation_id:?}"
        )
        .into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_contains(&stats, "# outcome=failed\n", "failed outcome")?;
    require_stats_contains(&stats, "[structured_error]\n", "structured error section")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("failed"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        execution_path: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "failed".to_string(),
        output_batches: 0,
        output_rows: 0,
        encoded_frame_count: None,
        record_batch_chunks: None,
        max_encoded_frame_bytes: None,
        stats_assertions: vec![
            "structured_error".to_string(),
            "query_id".to_string(),
            "correlation_id".to_string(),
            "runtime_failed".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

pub(super) async fn run_cancel_query(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "cancel_query";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(cancellation_projection_sql(), None))
        .await
        .map_err(|_| "cancel query GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let result = cancel_query(channel, &flight_info).await?;
    if result.result != CANCEL_RESULT_CANCELLING {
        return Err(format!(
            "CancelQuery for {query_id} returned {}, expected {CANCEL_RESULT_CANCELLING}",
            result.result
        )
        .into());
    }
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 0, 0).await
}

pub(super) async fn run_drop_before_first_poll(
    params: &RuntimeParams,
    channel: &Channel,
) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "drop_before_first_poll";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "drop-before-poll GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| "drop-before-poll DoGet timed out")??;
    drop(stream);
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 0, 0).await
}

pub(super) async fn run_midstream_drop(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let scenario = "midstream_drop";
    let correlation_id = format!("runtime-e2e-{scenario}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| "midstream-drop GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let mut stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| "midstream-drop DoGet timed out")??;
    let first = tokio::time::timeout(params.query_timeout, stream.try_next())
        .await
        .map_err(|_| "midstream-drop first batch timed out")??
        .ok_or("midstream-drop stream ended before its first batch")?;
    let rows = first.num_rows();
    drop(stream);
    terminal_scenario(params, channel, started, scenario, query_id, correlation_id, 1, rows).await
}

pub(super) async fn terminal_scenario(
    params: &RuntimeParams,
    channel: &Channel,
    started: Instant,
    scenario: &str,
    query_id: String,
    correlation_id: String,
    output_batches: usize,
    output_rows: usize,
) -> E2eResult<ScenarioEvidence> {
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_contains(&stats, "# outcome=failed\n", "cancelled query outcome")?;
    require_stats_contains(&stats, "[structured_error]\n", "cancelled structured error")?;
    require_stats_contains(&stats, "code=cancelled\n", "cancelled structured error code")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("cancelled"), Some(&params.observability))?;
    Ok(ScenarioEvidence {
        scenario: scenario.to_string(),
        execution_path: "native".to_string(),
        query_id,
        correlation_id,
        outcome: "cancelled".to_string(),
        output_batches,
        output_rows,
        encoded_frame_count: None,
        record_batch_chunks: None,
        max_encoded_frame_bytes: None,
        stats_assertions: vec![
            "sealed".to_string(),
            "structured_cancelled".to_string(),
            "terminal_cancelled".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}

pub(super) async fn start_in_flight_projection(
    params: &RuntimeParams,
    channel: &Channel,
    index: usize,
) -> E2eResult<(String, FlightRecordBatchStream)> {
    let correlation_id = format!("runtime-e2e-in-flight-{index}");
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(params.query_timeout, client.execute(broad_projection_sql(), None))
        .await
        .map_err(|_| format!("in-flight projection {index} GetFlightInfo timed out"))??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket = first_ticket(&flight_info)?;
    let stream = tokio::time::timeout(params.query_timeout, client.do_get(ticket))
        .await
        .map_err(|_| format!("in-flight projection {index} DoGet timed out"))??;
    Ok((query_id, stream))
}

pub(super) fn boxed_status_code(error: &(dyn std::error::Error + Send + Sync + 'static)) -> Option<Code> {
    let flight = error.downcast_ref::<FlightError>()?;
    match flight {
        FlightError::Tonic(status) => Some(status.code()),
        _ => None,
    }
}

pub(super) fn flight_status(error: FlightError) -> E2eResult<tonic::Status> {
    match error {
        FlightError::Tonic(status) => Ok(*status),
        other => Err(format!("expected Flight tonic status, got {other:?}").into()),
    }
}

pub(super) fn status_metadata(status: &tonic::Status, name: &str, context: &str) -> E2eResult<String> {
    status
        .metadata()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| format!("{context} status lacks {name} metadata: {status:?}").into())
}
