use std::{
    collections::BTreeSet,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use tokio::{
    sync::{Barrier, mpsc},
    task::JoinSet,
};
use tonic::transport::Channel;

use super::{
    E2eResult,
    flight::{
        CANCEL_RESULT_CANCELLING, CpuReference, QueryEvidence, Workload, cancel_query, collect_success_from_info,
        get_query_stats, query_id_from_statement_flight_info,
    },
};

pub(super) const CHURN_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) struct ChurnReport {
    pub(super) completed: Vec<QueryEvidence>,
    pub(super) cancelled: Vec<(String, String)>,
    pub(super) dropped: Vec<(String, String)>,
}

#[derive(Clone, Copy)]
enum ChurnMode {
    Complete,
    Cancel,
    Drop,
}

enum ChurnTaskOutput {
    Complete(QueryEvidence),
    Cancel(String),
    Drop(String),
}

pub(super) async fn run_churn(
    channel: &Channel,
    workload: &Workload,
    cpu_reference: &CpuReference,
) -> E2eResult<ChurnReport> {
    // Twenty-four simultaneous DoGet calls exceed the sixteen-query active
    // ceiling. Eight are cancelled, eight drop their streams, and eight must
    // still complete with exact parity.
    let modes = [vec![ChurnMode::Complete; 8], vec![ChurnMode::Cancel; 8], vec![ChurnMode::Drop; 8]].concat();
    let mut statements = Vec::with_capacity(modes.len());
    for mode in modes {
        let mut client = FlightSqlServiceClient::new(channel.clone());
        let flight_info = tokio::time::timeout(Duration::from_secs(60), client.execute(workload.sql.clone(), None))
            .await
            .map_err(|_| "churn GetFlightInfo exceeded 60 seconds")??;
        let query_id = query_id_from_statement_flight_info(&flight_info)?;
        statements.push((mode, query_id, flight_info));
    }

    let start = Arc::new(Barrier::new(statements.len() + 1));
    let cancel_release = Arc::new(Barrier::new(9));
    let (cancel_ready_tx, mut cancel_ready_rx) = mpsc::channel::<String>(8);
    let mut cancel_infos = Vec::new();
    let mut tasks: JoinSet<E2eResult<ChurnTaskOutput>> = JoinSet::new();

    for (mode, query_id, flight_info) in statements {
        if matches!(mode, ChurnMode::Cancel) {
            cancel_infos.push((query_id.clone(), flight_info.clone()));
        }
        let channel = channel.clone();
        let start = Arc::clone(&start);
        let cancel_release = Arc::clone(&cancel_release);
        let cancel_ready_tx = cancel_ready_tx.clone();
        let workload = workload.clone();
        let cpu_reference = cpu_reference.clone();
        tasks.spawn(async move {
            start.wait().await;
            match mode {
                ChurnMode::Complete => {
                    let evidence =
                        collect_success_from_info(&channel, "churn_complete", workload, cpu_reference, flight_info)
                            .await?;
                    Ok(ChurnTaskOutput::Complete(evidence))
                },
                ChurnMode::Cancel => {
                    let ticket = flight_info
                        .endpoint
                        .first()
                        .and_then(|endpoint| endpoint.ticket.clone())
                        .ok_or("cancel churn FlightInfo missing ticket")?;
                    let mut client = FlightSqlServiceClient::new(channel);
                    let mut stream = client.do_get(ticket).await?;
                    cancel_ready_tx.send(query_id.clone()).await?;
                    cancel_release.wait().await;
                    while stream.try_next().await.is_ok_and(|batch| batch.is_some()) {}
                    Ok(ChurnTaskOutput::Cancel(query_id))
                },
                ChurnMode::Drop => {
                    let ticket = flight_info
                        .endpoint
                        .first()
                        .and_then(|endpoint| endpoint.ticket.clone())
                        .ok_or("drop churn FlightInfo missing ticket")?;
                    let mut client = FlightSqlServiceClient::new(channel);
                    let stream = client.do_get(ticket).await?;
                    drop(stream);
                    Ok(ChurnTaskOutput::Drop(query_id))
                },
            }
        });
    }
    drop(cancel_ready_tx);
    start.wait().await;

    let mut ready = BTreeSet::new();
    while ready.len() < cancel_infos.len() {
        let query_id = tokio::time::timeout(Duration::from_secs(30), cancel_ready_rx.recv())
            .await
            .map_err(|_| "cancel churn streams did not become ready within 30 seconds")?
            .ok_or("cancel churn readiness channel closed early")?;
        ready.insert(query_id);
    }
    for (query_id, flight_info) in &cancel_infos {
        if !ready.contains(query_id) {
            return Err(format!("cancel churn stream {query_id} did not become ready").into());
        }
        let result = cancel_query(channel, flight_info).await?;
        if result.result != CANCEL_RESULT_CANCELLING {
            return Err(format!(
                "CancelQuery for active churn query {query_id} returned {}, expected {CANCEL_RESULT_CANCELLING}",
                result.result
            )
            .into());
        }
    }
    cancel_release.wait().await;

    let mut completed = Vec::new();
    let mut cancelled_ids = Vec::new();
    let mut dropped_ids = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result.map_err(|error| format!("churn task failed: {error}"))?? {
            ChurnTaskOutput::Complete(row) => completed.push(row),
            ChurnTaskOutput::Cancel(query_id) => cancelled_ids.push(query_id),
            ChurnTaskOutput::Drop(query_id) => dropped_ids.push(query_id),
        }
    }
    if completed.len() != 8 || cancelled_ids.len() != 8 || dropped_ids.len() != 8 {
        return Err(format!(
            "churn outcome count mismatch: completed={} cancelled={} dropped={}",
            completed.len(),
            cancelled_ids.len(),
            dropped_ids.len()
        )
        .into());
    }

    let mut cancelled = Vec::new();
    for query_id in cancelled_ids {
        let payload = wait_for_cancelled_query_stats(channel, &query_id).await?;
        cancelled.push((query_id, payload));
    }
    let mut dropped = Vec::new();
    for query_id in dropped_ids {
        let payload = wait_for_cancelled_query_stats(channel, &query_id).await?;
        dropped.push((query_id, payload));
    }
    Ok(ChurnReport { completed, cancelled, dropped })
}

async fn wait_for_cancelled_query_stats(channel: &Channel, query_id: &str) -> E2eResult<String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let payload = get_query_stats(channel, query_id).await?;
        if payload.contains("# sealed=true\n") {
            if !payload.contains("# outcome=failed\n")
                || !payload.contains("[structured_error]\n")
                || !payload.contains("code=cancelled\n")
            {
                return Err(format!(
                    "cancelled/dropped query {query_id} lacks structured cancellation evidence:\n{payload}"
                )
                .into());
            }
            if payload.contains("code=internal") || payload.contains("kind=internal") {
                return Err(format!("cancelled/dropped query {query_id} reported internal failure:\n{payload}").into());
            }
            return Ok(payload);
        }
        if Instant::now() >= deadline {
            return Err(format!("query {query_id} was not sealed within 30 seconds:\n{payload}").into());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
