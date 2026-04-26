use super::*;

pub(super) async fn run_cpu_phase(params: &RuntimeParams, evidence: &mut RunEvidence) -> E2eResult<()> {
    let mut server = spawn_server(params, evidence, "cpu-fallback").await?;
    setup_local_table(params, &server.channel, "cpu-fallback").await?;
    evidence.scenarios.push(run_cpu_fallback(params, &server.channel).await?);
    wait_for_idle_with_timeout(params, &server.channel).await?;
    let process = terminate_server(params, &mut server, "TERM", true).await?;
    evidence.processes.push(process);
    Ok(())
}

pub(super) async fn run_cpu_fallback(params: &RuntimeParams, channel: &Channel) -> E2eResult<ScenarioEvidence> {
    let started = Instant::now();
    let correlation_id = "runtime-e2e-cpu-fallback".to_string();
    let mut client = correlated_client(channel, &correlation_id);
    let flight_info = tokio::time::timeout(
        params.query_timeout,
        client.execute(
            format!(
                "SELECT l_orderkey FROM {TABLE_NAME} \
                 WHERE l_orderkey % l_partkey >= 0 ORDER BY l_orderkey LIMIT 2"
            ),
            None,
        ),
    )
    .await
    .map_err(|_| "CPU fallback GetFlightInfo timed out")??;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let collected = tokio::time::timeout(params.query_timeout, collect_flight_info(channel, flight_info))
        .await
        .map_err(|_| "CPU fallback DoGet timed out")??;
    let batches = &collected.batches;
    let output_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if output_rows != 2 || batches.iter().any(|batch| batch.num_columns() != 1) {
        return Err(format!("CPU fallback query returned {output_rows} rows, expected two one-column rows").into());
    }
    let values = batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(|row| arrow::util::display::array_value_to_string(batch.column(0), row))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values != ["1", "1"] {
        return Err(format!("CPU fallback query returned {values:?}, expected [\"1\", \"1\"]").into());
    }
    let stats = wait_for_sealed_stats(params, channel, &query_id).await?;
    require_stats_identity(&stats, &query_id, &correlation_id)?;
    require_stats_field(&stats, "planning", "outcome", "rejected_before_execution")?;
    require_stats_field(&stats, "planning", "reason_code", "unsupported_expression")?;
    require_stats_field(&stats, "planning", "unsupported_category", "expression")?;
    require_stats_field(&stats, "planning", "post_native_execution_path", "cpu")?;
    require_stats_field(&stats, "planning", "native_plan_fragment_count", "0")?;
    require_stats_field(&stats, "planning", "rejected_before_execution", "true")?;
    require_stats_field(&stats, "execution", "planning_outcome", "rejected_before_execution")?;
    require_stats_field(&stats, "execution", "post_native_execution_path", "cpu")?;
    require_stats_field(&stats, "execution", "native_runtime_started", "false")?;
    require_stats_field(&stats, "execution", "runtime_outcome", "not_started")?;
    let frames = wait_for_sealed_artifact(params, channel, &query_id).await?;
    let artifact =
        parse_artifact(&query_id, Some(&correlation_id), &frames, Some("succeeded"), Some(&params.observability))?;
    if !artifact.plan_execution_ids.is_empty() {
        return Err(format!(
            "CPU fallback query {query_id} unexpectedly reported native plan_execution_ids {:?}",
            artifact.plan_execution_ids
        )
        .into());
    }
    Ok(ScenarioEvidence {
        scenario: "cpu_fallback".to_string(),
        execution_path: "cpu".to_string(),
        query_id,
        correlation_id,
        outcome: "succeeded".to_string(),
        output_batches: batches.len(),
        output_rows,
        encoded_frame_count: Some(collected.encoded_frame_count),
        record_batch_chunks: Some(collected.record_batch_chunks()),
        max_encoded_frame_bytes: Some(collected.max_encoded_frame_bytes),
        stats_assertions: vec![
            "exact_result=[1,1]".to_string(),
            "unsupported_expression_rejected_before_execution".to_string(),
            "cpu_path".to_string(),
            "native_not_started".to_string(),
            "no_native_plan_execution_id".to_string(),
        ],
        artifact,
        elapsed_ms: millis(started.elapsed()),
    })
}
