use super::*;

pub(super) fn tracing_correlation_id(correlation_id: Option<&str>) -> Option<&str> {
    correlation_id.filter(|value| !value.is_empty())
}

impl DatafusionAlgeonFlightSqlService {
    pub(super) fn query_span(query_id: &str, correlation_id: Option<&str>, operation: &'static str) -> tracing::Span {
        tracing::info_span!(
            "algeon_datafusion.flight_sql.query",
            query_id,
            correlation_id = tracing_correlation_id(correlation_id),
            operation
        )
    }

    pub(super) fn query_stage_span(
        parent: Option<&tracing::Span>,
        query_id: &str,
        correlation_id: Option<&str>,
        stage: &'static str,
    ) -> tracing::Span {
        match parent {
            Some(parent) => tracing::info_span!(
                parent: parent,
                "algeon_datafusion.flight_sql.stage",
                query_id,
                correlation_id = tracing_correlation_id(correlation_id),
                stage
            ),
            None => tracing::info_span!(
                "algeon_datafusion.flight_sql.stage",
                query_id,
                correlation_id = tracing_correlation_id(correlation_id),
                stage
            ),
        }
    }
}

pub(crate) fn request_correlation_id<T>(request: &Request<T>) -> Result<Option<String>, Status> {
    for key in CORRELATION_ID_METADATA_ALIASES {
        let Some(value) = request.metadata().get(*key) else {
            continue;
        };
        let value = value
            .to_str()
            .map_err(|error| Status::invalid_argument(format!("{key} metadata is not valid ASCII: {error}")))?
            .trim();
        if value.is_empty() {
            continue;
        }
        if value.len() > MAX_CORRELATION_ID_LEN {
            return Err(Status::invalid_argument(format!("{key} metadata exceeds {MAX_CORRELATION_ID_LEN} bytes")));
        }
        return Ok(Some(value.to_string()));
    }
    Ok(None)
}

pub(crate) fn attach_query_response_metadata<T>(
    response: &mut Response<T>,
    query_id: &str,
    correlation_id: Option<&str>,
) {
    if let Ok(value) = MetadataValue::try_from(query_id) {
        response.metadata_mut().insert(QUERY_ID_METADATA_KEY, value);
    }
    if let Some(correlation_id) = correlation_id
        && let Ok(value) = MetadataValue::try_from(correlation_id)
    {
        response.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, value);
    }
}

#[cfg(test)]
mod correlation_tests {
    use std::{
        sync::{Arc, mpsc},
        time::{Duration, Instant},
    };

    use query_planning::correlation::QueryId;
    use query_runtime::observability::{
        QueryBoundaryKind, RUNTIME_OBSERVATION_JSONL_SCHEMA, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeObservationPayload, RuntimeObservationRecordDisposition, RuntimeReservationAction,
        RuntimeReservationDisposition, RuntimeReservationId, RuntimeReservationObservation,
    };
    use tokio::{spawn, task::yield_now, time::timeout};

    use super::*;
    use crate::server::{
        attempt::AttemptOptions,
        coordinator::{
            QueryTerminalOutcome, install_coordinator_render_probe,
            test_support::{cpu_attempt_report, seal_attempt_observation},
        },
        error::{
            ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
        },
        flight::{
            frame_limiter::MAX_FLIGHT_FRAME_BYTES,
            service::{BoundedTestFlightSqlService, bounded_test_service},
        },
    };

    pub(super) fn test_service() -> BoundedTestFlightSqlService {
        bounded_test_service(Duration::from_secs(30))
    }

    fn query_stats(service: &DatafusionAlgeonFlightSqlService, query_id: &str) -> String {
        String::from_utf8(
            service.coordinator.query_stats(query_id).expect("query stats capture should be retained").to_vec(),
        )
        .expect("query stats are UTF-8")
    }

    async fn sealed_query_stats(service: &DatafusionAlgeonFlightSqlService, query_id: &str) -> String {
        timeout(Duration::from_secs(5), async {
            loop {
                let payload = query_stats(service, query_id);
                if payload.contains("# sealed=true\n") {
                    return payload;
                }
                yield_now().await;
            }
        })
        .await
        .expect("attempt report handoff should seal the terminal query")
    }

    #[test]
    pub(super) fn request_correlation_id_accepts_server_key_and_alias() {
        let mut request = Request::new(());
        request.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, MetadataValue::try_from("corr-123").unwrap());

        assert_eq!(request_correlation_id(&request).unwrap().as_deref(), Some("corr-123"));

        let mut alias_request = Request::new(());
        alias_request.metadata_mut().insert("x-correlation-id", MetadataValue::try_from("corr-alias").unwrap());

        assert_eq!(request_correlation_id(&alias_request).unwrap().as_deref(), Some("corr-alias"));
    }

    #[tokio::test]
    async fn get_flight_info_statement_echoes_correlation_id_and_retains_it_in_stats() {
        use prost::Message as _;

        let service = test_service();
        let mut request = Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new()));
        request.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, MetadataValue::try_from("client-corr-42").unwrap());

        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                request,
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();

        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("client-corr-42")
        );
        let correlation_id =
            service.coordinator.ticket_correlation_id(&query_id).expect("statement ticket should be retained");
        assert_eq!(correlation_id.as_deref(), Some("client-corr-42"));
        let payload = query_stats(&service, &query_id);
        assert!(payload.contains("# correlation_id=client-corr-42\n"), "{payload}");
        assert!(
            service
                .cancel_from_flight_info(&response.into_inner().encode_to_vec(), &Owner::Anonymous)
                .expect("ticket cancellation decodes")
        );
        let payload = sealed_query_stats(&service, &query_id).await;
        assert!(payload.contains("# correlation_id=client-corr-42\n"), "{payload}");
    }

    #[tokio::test]
    async fn do_get_only_correlation_id_is_retained_in_stats() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        assert!(!query_stats(&service, &query_id).contains("# correlation_id="));

        let response = service
            .execute_ticket(query_id.clone(), Some("do-get-corr-7".to_string()))
            .await
            .expect("statement do_get should succeed");

        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("do-get-corr-7")
        );
        let payload = query_stats(&service, &query_id);
        assert!(payload.contains("# correlation_id=do-get-corr-7\n"), "{payload}");
    }

    #[tokio::test]
    async fn oversized_native_single_cell_do_get_preserves_pre_d2h_limit_and_correlation_metadata() {
        let service = test_service();
        let correlation_id = "oversized-cell-correlation";
        let mut request = Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new()));
        request.metadata_mut().insert(CORRELATION_ID_METADATA_KEY, MetadataValue::try_from(correlation_id).unwrap());
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery {
                    query: format!("SELECT repeat('x', {}) AS oversized FROM (VALUES (1)) AS t(x)", 5 * 1024 * 1024),
                    transaction_id: None,
                },
                request,
            )
            .await
            .expect("oversized scalar statement must plan and issue a ticket");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();

        let response = service
            .execute_ticket(query_id.clone(), None)
            .await
            .expect("DoGet constructs a lazy response before encoding the oversized row");
        assert_eq!(
            response.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some(query_id.as_str())
        );
        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some(correlation_id)
        );

        let mut stream = response.into_inner();
        let schema = stream.next().await.expect("schema frame").expect("schema frame is valid");
        assert!(schema.encoded_len() <= MAX_FLIGHT_FRAME_BYTES);
        let status = stream
            .next()
            .await
            .expect("oversized row must terminate the DoGet stream")
            .expect_err("one oversized scalar cannot be emitted as FlightData");

        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_eq!(
            status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some(query_id.as_str())
        );
        assert_eq!(
            status.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some(correlation_id)
        );
        assert_eq!(
            status.metadata().get(ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("host_materialization_limit_exceeded")
        );
        assert_eq!(
            status.metadata().get(ERROR_KIND_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("resource_exhausted")
        );
        assert_eq!(
            status.metadata().get(ERROR_STATUS_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("permanent")
        );
        assert_eq!(
            status.metadata().get(ERROR_OPERATION_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("algeon_datafusion.native_exec.host_batch_cursor")
        );
        assert!(status.message().contains("host materialization limit"), "{status:?}");
        assert!(!status.message().to_ascii_lowercase().contains("retry later"), "{status:?}");
        assert!(!status.message().contains("internal server error"), "{status:?}");

        let payload = sealed_query_stats(&service, &query_id).await;
        assert!(payload.contains("code=host_materialization_limit_exceeded\n"), "{payload}");
        assert!(payload.contains("kind=resource_exhausted\n"), "{payload}");
        assert!(payload.contains("status=permanent\n"), "{payload}");
        assert!(payload.contains("operation=algeon_datafusion.native_exec.host_batch_cursor\n"), "{payload}");
        assert!(payload.contains("# correlation_id=oversized-cell-correlation\n"), "{payload}");
    }

    #[tokio::test]
    async fn dropping_do_get_before_first_poll_cancels_statement() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        let attempt_cancellation =
            service.coordinator.ticket_cancellation(&query_id).expect("statement ticket should be retained");

        let response =
            service.execute_ticket(query_id.clone(), None).await.expect("statement do_get should return a lazy stream");
        assert!(!attempt_cancellation.is_cancelled(), "stream creation must not cancel the query");

        drop(response);

        assert!(attempt_cancellation.is_cancelled(), "dropping the DoGet stream must cancel native execution");
        assert_eq!(service.cancellations.len(), 0, "dropped streams must leave no cancellable registry entry");
        let payload = sealed_query_stats(&service, &query_id).await;
        assert!(payload.contains("# outcome=cancelled"), "{payload}");
        assert!(payload.contains("# delivery_outcome=dropped"), "{payload}");
        assert!(payload.contains("# sealed=true"), "{payload}");
        assert!(payload.contains("# attempt_outcome=cancelled"), "{payload}");
        assert!(!payload.contains("[structured_error]"), "the coordinator must not mint an attempt error: {payload}");
    }

    #[tokio::test]
    async fn panicking_lazy_do_get_drop_records_one_panicked_terminal_before_cancellation() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        let attempt_cancellation =
            service.coordinator.ticket_cancellation(&query_id).expect("statement ticket should be retained");
        let response =
            service.execute_ticket(query_id.clone(), None).await.expect("statement do_get should return a lazy stream");

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _response = response;
            panic!("lazy stream panic probe");
        }));

        assert!(unwind.is_err());
        assert_eq!(service.coordinator.terminal(&query_id), Some(QueryTerminalOutcome::Panicked));
        assert!(attempt_cancellation.is_cancelled(), "panic cleanup still cancels native execution");
        assert_eq!(service.cancellations.len(), 0);
        let payload = query_stats(&service, &query_id);
        assert!(payload.contains("# outcome=panicked\n"), "{payload}");
        assert!(payload.contains("# delivery_outcome=dropped\n"), "{payload}");
        assert_eq!(payload.matches("# attempt_id=attempt-").count(), 1, "{payload}");
        assert_eq!(payload.matches("# attempt_outcome=").count(), 1, "{payload}");
        assert!(payload.contains("# attempt_outcome=panicked"), "{payload}");
        assert!(!payload.contains("code=cancelled"), "panic must not be downgraded to cancellation: {payload}");
    }

    #[tokio::test]
    async fn dropping_do_get_midstream_cancels_statement() {
        let service = test_service();
        let response = service
            .get_flight_info_statement(
                CommandStatementQuery { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(FlightDescriptor::new_cmd(Vec::<u8>::new())),
            )
            .await
            .expect("statement flight info should succeed");
        let query_id = response
            .metadata()
            .get(QUERY_ID_METADATA_KEY)
            .and_then(|value| value.to_str().ok())
            .expect("query id response metadata")
            .to_string();
        let attempt_cancellation =
            service.coordinator.ticket_cancellation(&query_id).expect("statement ticket should be retained");

        let mut stream = service
            .execute_ticket(query_id.clone(), None)
            .await
            .expect("statement do_get should return a lazy stream")
            .into_inner();
        stream.next().await.expect("schema frame").expect("schema frame is valid");
        stream.next().await.expect("record-batch frame").expect("record-batch frame is valid");
        assert!(!attempt_cancellation.is_cancelled(), "partial polling must keep the query live");

        drop(stream);

        assert!(attempt_cancellation.is_cancelled(), "dropping a partially polled stream must cancel native work");
        assert_eq!(service.cancellations.len(), 0, "midstream drops must remove cancellation registration");
        let payload = sealed_query_stats(&service, &query_id).await;
        assert!(payload.contains("# outcome=cancelled"), "{payload}");
        assert!(payload.contains("# delivery_outcome=failed"), "{payload}");
        assert!(payload.contains("code=cancelled"), "{payload}");
        assert!(payload.contains("operation=algeon_datafusion.server.flight.cancel_query"), "{payload}");
    }

    #[test]
    fn stream_error_terminalizes_once_with_structured_adapter_metadata() {
        let service = DatafusionAlgeonFlightSqlService::new(
            Arc::new(SessionContext::new()),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        );
        let query_id = "stream-error-terminal";
        let planning = service
            .coordinator
            .register_with_id(query_id, "SELECT private_sql_literal", None, service.runtime_observation_config)
            .expect("coordinator-owned fixture registers");
        let execution = planning.into_execution().unwrap();
        let terminal = CoordinatorExecutionTerminal::new(
            execution,
            query_id.to_string(),
            Arc::from("SELECT private_sql_literal"),
            Instant::now(),
        );

        let _flight_error = datafusion_error_to_flight_error(
            DataFusionError::Execution("runtime batch failed".to_string()),
            query_id,
            &terminal,
            "algeon_datafusion.server.flight.test_stream",
        );

        assert!(terminal.is_finished());
        let payload = query_stats(&service, query_id);
        assert!(payload.contains("# outcome=failed"), "{payload}");
        assert!(payload.contains("# sealed=true"), "{payload}");
        assert!(payload.contains("code=internal"), "{payload}");
        assert!(payload.contains("kind=internal"), "{payload}");
        assert!(payload.contains("status=permanent"), "{payload}");
        assert!(payload.contains("operation=algeon_datafusion.server.flight.test_stream"), "{payload}");
    }

    #[tokio::test]
    async fn rejected_attempt_report_handoff_keeps_cancelled_attempt_outcome_separate_from_delivery_failure() {
        const QUERY_ID: &str = "attempt-report-handoff-rejection";
        let service = test_service();
        let planning = service
            .register_query(QUERY_ID.to_string(), "SELECT 1 AS one".to_string(), None)
            .expect("query registration should create the coordinator record");
        let execution = planning.into_execution().expect("planning record should promote to execution");
        let terminal = CoordinatorExecutionTerminal::new_with_attempt_report(
            execution,
            QUERY_ID.to_string(),
            Arc::from("SELECT 1 AS one"),
            Instant::now(),
            true,
        );
        let attempt = service
            .admission
            .begin_attempt(AttemptOptions {
                query_id: Some(QUERY_ID.to_string()),
                trace_context: None,
                deadline: None,
                observation: service.runtime_observation_config,
            })
            .expect("test backend should begin the attempt")
            .expect("test backend always owns a native attempt");

        // Route the observer to a missing coordinator record. This simulates
        // a handoff rejection while preserving the real terminal's lease and
        // ensures the failure cannot remain blocked on the report gate.
        let mut missing_key = terminal.key().clone();
        missing_key.generation = u64::MAX;
        service.register_attempt_report_handoff_for_test(
            missing_key,
            &attempt,
            CoordinatorReportHandoffTerminal::execution(Arc::clone(&terminal)),
        );
        terminal.finish_success("stream");
        attempt.cancel_before_stream();

        timeout(Duration::from_secs(1), terminal.wait_finished())
            .await
            .expect("rejected report handoff must release the terminal lease");
        assert!(terminal.is_finished());
        assert!(matches!(
            service.coordinator.artifact(QUERY_ID),
            Some(CoordinatorArtifact::Unavailable(reason))
                if reason.as_str() == "attempt_report_delivery_failed"
        ));
        let payload = query_stats(&service, QUERY_ID);
        assert!(payload.contains("# outcome=cancelled\n"), "{payload}");
        assert!(payload.contains("# delivery_outcome=failed\n"), "{payload}");
        assert!(payload.contains("# incomplete=true\n"), "{payload}");
        assert!(payload.contains("# unavailable_reason=attempt_report_delivery_failed\n"), "{payload}");
    }

    #[tokio::test]
    async fn missing_attempt_report_handoff_releases_the_flight_report_gate() {
        const QUERY_ID: &str = "attempt-report-missing";
        let service = test_service();
        let planning = service
            .register_query(QUERY_ID.to_string(), "SELECT 1 AS one".to_string(), None)
            .expect("query registration should create the coordinator record");
        let execution = planning.into_execution().expect("planning record should promote to execution");
        let terminal = CoordinatorExecutionTerminal::new_with_attempt_report(
            execution,
            QUERY_ID.to_string(),
            Arc::from("SELECT 1 AS one"),
            Instant::now(),
            true,
        );
        let handoff = CoordinatorReportHandoffTerminal::execution(Arc::clone(&terminal));

        // This drives the observer's invariant-failure branch directly. A
        // real attempt can never reach it without violating report-before-
        // snapshot publication, so constructing a malformed attempt would
        // test an impossible public contract instead of the Flight gate.
        terminal.finish_success("stream");
        DatafusionAlgeonFlightSqlService::finish_missing_attempt_report_handoff(&handoff, QUERY_ID);

        timeout(Duration::from_secs(1), terminal.wait_finished())
            .await
            .expect("missing report handoff must release the terminal lease");
        assert!(terminal.is_finished());
        assert!(matches!(
            service.coordinator.artifact(QUERY_ID),
            Some(CoordinatorArtifact::Unavailable(reason))
                if reason.as_str() == "attempt_report_delivery_failed"
        ));
        let payload = query_stats(&service, QUERY_ID);
        assert!(payload.contains("# outcome=failed\n"), "{payload}");
        assert!(payload.contains("# incomplete=true\n"), "{payload}");
        assert!(payload.contains("# unavailable_reason=attempt_report_delivery_failed\n"), "{payload}");
    }

    #[tokio::test]
    async fn rejected_planning_attempt_report_handoff_releases_report_gate() {
        const QUERY_ID: &str = "planning-attempt-report-handoff-rejection";
        let service = test_service();
        let mut planning = service
            .register_query(QUERY_ID.to_string(), "SELECT 1 AS one".to_string(), None)
            .expect("query registration should create the coordinator record");
        let attempt = service
            .admission
            .begin_attempt(AttemptOptions {
                query_id: Some(QUERY_ID.to_string()),
                trace_context: None,
                deadline: None,
                observation: service.runtime_observation_config,
            })
            .expect("test backend should begin the attempt")
            .expect("test backend always owns a native attempt");

        // A planning failure seals before any execution terminal exists. Route
        // that observer to a missing record and verify its generation-bound
        // planning callback clears the real record's report gate.
        let mut missing_key = planning.key().clone();
        missing_key.generation = u64::MAX;
        service.register_attempt_report_handoff_for_test(
            missing_key,
            &attempt,
            CoordinatorReportHandoffTerminal::planning(&planning),
        );
        attempt.complete_without_result_stream();
        let status = Status::internal("planning failed after report handoff rejection");
        planning.finish_captured("planning", &status, CapturedError::from_status(&status));

        timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    service.coordinator.artifact(QUERY_ID),
                    Some(CoordinatorArtifact::Unavailable(reason))
                        if reason.as_str() == "attempt_report_delivery_failed"
                ) {
                    return;
                }
                yield_now().await;
            }
        })
        .await
        .expect("planning report handoff failure must not leave finalization waiting on its report gate");

        let payload = query_stats(&service, QUERY_ID);
        assert!(payload.contains("# outcome=succeeded\n"), "{payload}");
        assert!(payload.contains("# delivery_outcome=failed\n"), "{payload}");
        assert!(payload.contains("# incomplete=true\n"), "{payload}");
        assert!(payload.contains("# unavailable_reason=attempt_report_delivery_failed\n"), "{payload}");
    }

    #[tokio::test]
    async fn get_admission_stats_action_returns_current_admission_payload() {
        let service = test_service();

        let action_types = service
            .list_custom_actions()
            .await
            .expect("custom actions should be listed")
            .into_iter()
            .map(|action| action.unwrap().r#type)
            .collect::<Vec<_>>();
        assert!(action_types.contains(&GET_QUERY_STATS_ACTION.to_string()), "{action_types:?}");
        assert!(action_types.contains(&GET_ADMISSION_STATS_ACTION.to_string()), "{action_types:?}");

        let response = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_ADMISSION_STATS_ACTION.to_string(),
                body: Vec::new().into(),
            }))
            .await
            .expect("GetAdmissionStats action should succeed");
        let results = response.into_inner().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(results.len(), 1);
        let payload = String::from_utf8(results[0].body.to_vec()).unwrap();

        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# backend_metrics_backend_generation=1\n"), "{payload}");
        assert!(
            payload.contains("[backend_metrics]\nbackend_generation\tgeneration\tconfigured_devices\n1\t0\t1\n"),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[backend_device_admission]\ndevice_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tresident_owner_bytes\tmax_resident_owner_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n0\t0\t0\t0\t0\t0\t0\t0\tOpen\n"
            ),
            "{payload}"
        );
    }

    /// The GetQueryStats handler waits for another query's render off the
    /// tokio worker: on a single-threaded runtime a witness task must still
    /// run while the handler's waiter blocks, and only the witness releases
    /// the held render. A handler that waits on the worker starves the
    /// witness, so the render never completes and the action times out.
    #[tokio::test(flavor = "current_thread")]
    async fn get_query_stats_action_waits_for_another_render_off_the_tokio_worker() {
        const QUERY_ID: &str = "worker-stats-render";
        let service = DatafusionAlgeonFlightSqlService::new(
            Arc::new(SessionContext::new()),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        );
        let execution = service
            .coordinator
            .register_with_id(QUERY_ID, "select 1", None, RuntimeObservationConfig::bounded(16, 4096))
            .expect("live query registration")
            .into_execution()
            .expect("execution phase");
        let cache = {
            let state = service.coordinator.lock_state();
            let key = state.current.get(QUERY_ID).expect("registered query key");
            Arc::clone(&state.records.get(key).expect("registered query record").live_stats_cache)
        };
        let (wait_tx, wait_rx) = mpsc::channel();
        *cache.wait_observer.lock().unwrap() = Some(wait_tx);
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();

        let render_coordinator = Arc::clone(&service.coordinator);
        let renderer = std::thread::spawn(move || {
            let _probe = install_coordinator_render_probe(move |operation| {
                if operation == "query-stats projection" {
                    entered_tx.send(()).expect("render entry must be observed");
                    release_rx.recv().expect("render release must arrive");
                }
            });
            render_coordinator.query_stats_result(QUERY_ID).expect("held render completes")
        });

        // The handler must wait rather than render itself, so it only runs
        // once the other render is already in flight.
        loop {
            if entered_rx.try_recv().is_ok() {
                break;
            }
            yield_now().await;
        }
        let witness = spawn(async move {
            loop {
                if wait_rx.try_recv().is_ok() {
                    release_tx.send(()).expect("render release must send");
                    return;
                }
                yield_now().await;
            }
        });

        let response = timeout(
            Duration::from_secs(10),
            service.do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_QUERY_STATS_ACTION.to_string(),
                body: QUERY_ID.as_bytes().to_vec().into(),
            })),
        )
        .await
        .expect("the handler must not park the only worker on another render")
        .expect("GetQueryStats action should succeed");
        let results = response.into_inner().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].body.is_empty());

        witness.await.expect("the witness must run while the handler waits");
        renderer.join().expect("held render must complete");
        execution.succeed();
    }

    #[tokio::test]
    async fn create_prepared_statement_is_terminalized_and_addressable_by_its_handle() {
        let service = test_service();
        let result = service
            .do_action_create_prepared_statement(
                ActionCreatePreparedStatementRequest { query: "SELECT 1 AS one".to_string(), transaction_id: None },
                Request::new(arrow_flight::Action { r#type: String::new(), body: Default::default() }),
            )
            .await
            .expect("prepared statement creation succeeds");
        let query_id = std::str::from_utf8(&result.prepared_statement_handle).unwrap();
        let payload = query_stats(&service, query_id);
        assert!(payload.contains("# outcome=succeeded\n"), "{payload}");
        assert!(payload.contains("# sealed=true\n"), "{payload}");
        assert!(matches!(service.coordinator.artifact(query_id), Some(CoordinatorArtifact::Retained(_))));
    }

    #[tokio::test]
    async fn get_query_runtime_artifact_action_returns_atomic_sanitized_frames() {
        const QUERY_ID: &str = "runtime-artifact-rpc-query";
        const SQL_SECRET: &str = "rpc-sql-literal-secret";
        const RAW_SOURCE: &str = "s3://reader:rpc-path-secret@private-bucket/customer.parquet?token=raw-token";

        let service = DatafusionAlgeonFlightSqlService::new(
            Arc::new(SessionContext::new()),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        );
        let planning = service
            .coordinator
            .register_with_id_with_attempt_observation(
                QUERY_ID,
                format!("SELECT '{SQL_SECRET}' FROM '{RAW_SOURCE}'"),
                None,
                service.runtime_observation_config,
            )
            .expect("attempt-owned runtime artifact fixture registers");
        let key = planning.key().clone();
        // Stands in for the statement's attempt, which records into its own
        // observation and hands the sealed snapshot to the coordinator.
        let attempt_observation =
            RuntimeObservationOwner::new(QueryId::new(QUERY_ID), None, service.runtime_observation_config);
        let observation = attempt_observation.handle();
        let execution = planning.into_execution().unwrap();
        let source_location_id =
            observation.source_location_id(RAW_SOURCE).expect("enabled capture derives an opaque source identity");
        assert!(matches!(
            observation.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started }),
            RuntimeObservationRecordDisposition::Retained { .. }
        ));
        let reservation_id = RuntimeReservationId::new("rpc-reservation-1");
        assert!(matches!(
            observation.record(|| RuntimeObservationPayload::Reservation(Box::new(RuntimeReservationObservation {
                reservation_id: reservation_id.clone(),
                lineage_id: reservation_id,
                task_attempt_id: None,
                task_kind: Some("source_read".to_string()),
                action: RuntimeReservationAction::Reserve,
                denied_action: None,
                disposition: RuntimeReservationDisposition::Applied,
                alias: false,
                reservation_kind: "source_input".to_string(),
                memory_space: "device:0".to_string(),
                memory_boundary: "source_read".to_string(),
                requested_bytes: 128,
                granted_bytes: 128,
                charged_delta_bytes: 128,
                current_reserved_bytes: 128,
                budget_bytes: 4096,
                node_id: Some("source-node-1".to_string()),
                stage: Some("read".to_string()),
                chunk_index: Some(0),
                file_index: Some(0),
                source_location_id: Some(source_location_id.clone()),
                row_groups: Some(vec![0]),
                attempt: Some(1),
                max_attempts: Some(1),
                resident_resource_id: None,
                source_read_contract_quality: Some("exact".to_string()),
                source_read_planning_bytes: Some(128),
                source_read_estimated_bytes: Some(128),
            }))),
            RuntimeObservationRecordDisposition::Retained { .. }
        ));

        let action_types = service
            .list_custom_actions()
            .await
            .expect("custom actions should be listed")
            .into_iter()
            .map(|action| action.expect("custom action should be valid").r#type)
            .collect::<Vec<_>>();
        assert!(action_types.contains(&GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string()), "{action_types:?}");

        let active = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string(),
                body: QUERY_ID.as_bytes().to_vec().into(),
            }))
            .await
            .expect("active runtime artifact action should succeed")
            .into_inner()
            .try_collect::<Vec<_>>()
            .await
            .expect("active action stream should be valid");
        assert_eq!(active.len(), 1, "active artifacts must not emit a partial JSONL frame");
        let active_control: serde_json::Value =
            serde_json::from_slice(&active[0].body).expect("control frame should be JSON");
        assert_eq!(
            active_control,
            serde_json::json!({
                "schema": "algeon_datafusion.query_runtime_artifact.control",
                "query_id": QUERY_ID,
                "state": "active",
                "retryable": true,
                "reason": null,
                "incomplete": false,
                "runtime_jsonl_follows": false,
                "attempt": null,
            })
        );

        execution.succeed();
        let (report, snapshot) = seal_attempt_observation(attempt_observation, cpu_attempt_report(QUERY_ID));
        service
            .coordinator
            .attach_attempt_report(&key, Arc::new(report), Some(snapshot), None)
            .expect("the attempt hands its sealed report to the coordinator");

        let sealed = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string(),
                body: QUERY_ID.as_bytes().to_vec().into(),
            }))
            .await
            .expect("sealed runtime artifact action should succeed")
            .into_inner()
            .try_collect::<Vec<_>>()
            .await
            .expect("sealed action stream should be valid");
        assert_eq!(sealed.len(), 2, "sealed artifacts must emit one control frame and one complete JSONL frame");
        let sealed_control: serde_json::Value =
            serde_json::from_slice(&sealed[0].body).expect("control frame should be JSON");
        assert_eq!(sealed_control["schema"], "algeon_datafusion.query_runtime_artifact.control");
        assert_eq!(sealed_control["runtime_jsonl_follows"], true);
        assert_eq!(sealed_control["query_id"], QUERY_ID);
        assert_eq!(sealed_control["state"], "sealed");
        assert_eq!(sealed_control["retryable"], false);
        assert!(sealed_control["attempt"].is_object(), "an attempt-owned artifact projects its attempt report");
        let records = std::str::from_utf8(&sealed[1].body)
            .expect("runtime artifact JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSONL line should be complete"))
            .collect::<Vec<_>>();
        assert_eq!(records.first().and_then(|record| record["record_type"].as_str()), Some("manifest"));
        assert_eq!(
            records.first().and_then(|record| record["data"]["schema"].as_str()),
            Some(RUNTIME_OBSERVATION_JSONL_SCHEMA)
        );
        let footer = records.last().expect("JSONL footer");
        assert_eq!(footer["record_type"], "footer");
        assert_eq!(footer["data"]["complete"], true);
        assert!(!footer["data"]["terminal"].is_null(), "the attempt seals its observation with its outcome");

        let wire_bytes = sealed.iter().flat_map(|frame| frame.body.iter().copied()).collect::<Vec<_>>();
        let wire_text = String::from_utf8(wire_bytes).expect("artifact frames should be UTF-8");
        assert!(wire_text.contains(source_location_id.as_str()), "opaque source identity should be exported");
        for private_value in [SQL_SECRET, RAW_SOURCE, "rpc-path-secret", "raw-token"] {
            assert!(!wire_text.contains(private_value), "runtime artifact leaked private value: {private_value}");
        }
    }

    #[tokio::test]
    async fn unavailable_runtime_artifact_action_emits_only_its_control_frame() {
        const QUERY_ID: &str = "runtime-artifact-unavailable";

        let service = DatafusionAlgeonFlightSqlService::new(
            Arc::new(SessionContext::new()),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        );
        let planning = service
            .coordinator
            .register_with_id(QUERY_ID, "SELECT 1", None, service.runtime_observation_config)
            .expect("coordinator-owned unavailable-artifact fixture registers");
        let hot = planning.hot_counters();
        hot.record_output(usize::MAX, 0);
        hot.record_output(1, 0);
        planning.into_execution().unwrap().succeed();

        let frames = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string(),
                body: QUERY_ID.as_bytes().to_vec().into(),
            }))
            .await
            .expect("unavailable runtime artifact action should succeed")
            .into_inner()
            .try_collect::<Vec<_>>()
            .await
            .expect("unavailable action stream should be valid");

        assert_eq!(frames.len(), 1, "unavailable artifacts have exactly one control frame");
        let control: serde_json::Value = serde_json::from_slice(&frames[0].body).expect("control frame should be JSON");
        assert_eq!(
            control,
            serde_json::json!({
                "schema": "algeon_datafusion.query_runtime_artifact.control",
                "query_id": QUERY_ID,
                "state": "unavailable",
                "retryable": false,
                "reason": "counter_overflow",
                "incomplete": true,
                "runtime_jsonl_follows": false,
                "attempt": null,
            })
        );
        assert_eq!(service.coordinator.artifact_egress_usage(), (0, 0));
    }

    #[tokio::test]
    async fn dropping_runtime_artifact_action_stream_releases_egress_lease() {
        const QUERY_ID: &str = "runtime-artifact-stream-cancel";

        let service = DatafusionAlgeonFlightSqlService::new(
            Arc::new(SessionContext::new()),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        );
        let planning = service.register_query(QUERY_ID.to_string(), "SELECT 1".to_string(), None).unwrap();
        planning.into_execution().unwrap().succeed();
        assert_eq!(service.coordinator.artifact_egress_usage(), (0, 0));

        let response = service
            .do_action_fallback(Request::new(arrow_flight::Action {
                r#type: GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string(),
                body: QUERY_ID.as_bytes().to_vec().into(),
            }))
            .await
            .expect("sealed runtime artifact action should acquire one egress lease");
        let usage_while_open = service.coordinator.artifact_egress_usage();
        assert_eq!(usage_while_open.0, 1);
        assert!(usage_while_open.1 > 0);

        let mut stream = response.into_inner();
        stream.next().await.expect("control frame").expect("control frame should be valid");
        assert_eq!(service.coordinator.artifact_egress_usage(), usage_while_open);

        drop(stream);
        assert_eq!(service.coordinator.artifact_egress_usage(), (0, 0));
    }
}
