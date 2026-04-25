use super::*;

pub(super) fn tracing_correlation_id(correlation_id: Option<&str>) -> Option<&str> {
    correlation_id.filter(|value| !value.is_empty())
}

impl DatafusionCudfFlightSqlService {
    pub(super) fn query_span(query_id: &str, correlation_id: Option<&str>, operation: &'static str) -> tracing::Span {
        tracing::info_span!(
            "datafusion_nexus.flight_sql.query",
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
                "datafusion_nexus.flight_sql.stage",
                query_id,
                correlation_id = tracing_correlation_id(correlation_id),
                stage
            ),
            None => tracing::info_span!(
                "datafusion_nexus.flight_sql.stage",
                query_id,
                correlation_id = tracing_correlation_id(correlation_id),
                stage
            ),
        }
    }
}

pub(in crate::server) fn request_correlation_id<T>(request: &Request<T>) -> Result<Option<String>, Status> {
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

pub(in crate::server) fn attach_query_response_metadata<T>(
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

pub(super) fn finish_adapter_error_status_with_correlation_id(
    observability: &QueryObservability,
    query_id: &str,
    correlation_id: Option<&str>,
    stage: &'static str,
    error: Error,
) -> Status {
    observability.finish_failed(query_id, stage, CapturedError::from_adapter_error(&error));
    status_with_query_id_and_correlation_id(adapter_error_status(error), query_id, correlation_id)
}

#[cfg(test)]
mod correlation_tests {
    use super::*;
    use crate::{
        policy::GpuFallbackPolicy,
        server::auth::{AuthMode, TokenStore},
    };

    pub(super) fn test_service() -> DatafusionCudfFlightSqlService {
        DatafusionCudfFlightSqlService::new(
            Arc::new(SessionContext::new()),
            CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred),
            ServerSessionProfile::Native,
            Arc::new(QueryAdmission::new(1).expect("test admission limit")),
            Duration::from_secs(30),
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            #[cfg(feature = "cugraph")]
            None,
        )
    }

    #[test]
    fn tracing_correlation_field_omits_absent_and_empty_values() {
        assert_eq!(tracing_correlation_id(None), None);
        assert_eq!(tracing_correlation_id(Some("")), None);
        assert_eq!(tracing_correlation_id(Some("corr-123")), Some("corr-123"));
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

    #[test]
    pub(super) fn query_response_metadata_echoes_query_and_correlation_id() {
        let mut response = Response::new(());

        attach_query_response_metadata(&mut response, "query-1", Some("corr-1"));

        assert_eq!(
            response.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("query-1")
        );
        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("corr-1")
        );
    }

    #[test]
    pub(super) fn status_metadata_echoes_query_and_correlation_id() {
        let status = status_with_query_id_and_correlation_id(Status::unavailable("busy"), "query-1", Some("corr-1"));

        assert_eq!(status.metadata().get(QUERY_ID_METADATA_KEY).and_then(|value| value.to_str().ok()), Some("query-1"));
        assert_eq!(
            status.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("corr-1")
        );
    }

    #[tokio::test]
    async fn get_flight_info_statement_echoes_correlation_id_and_retains_it_in_stats() {
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
        let ticket = service.tickets.get(&query_id).expect("statement ticket should be retained");
        assert_eq!(ticket.correlation_id.as_deref(), Some("client-corr-42"));
        drop(ticket);
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
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
        assert!(
            !service
                .observability
                .get(&query_id)
                .expect("query stats capture should be retained")
                .to_query_stats_payload()
                .contains("# correlation_id=")
        );

        let response = service
            .execute_ticket(query_id.clone(), Some("do-get-corr-7".to_string()))
            .await
            .expect("statement do_get should succeed");

        assert_eq!(
            response.metadata().get(CORRELATION_ID_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("do-get-corr-7")
        );
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
        assert!(payload.contains("# correlation_id=do-get-corr-7\n"), "{payload}");
    }

    #[tokio::test]
    async fn dropping_do_get_stream_cancels_statement() {
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
        let native_cancellation = {
            let ticket = service.tickets.get(&query_id).expect("statement ticket should be retained");
            ticket.native_cancellation.clone()
        };

        let response =
            service.execute_ticket(query_id.clone(), None).await.expect("statement do_get should return a lazy stream");
        assert!(!native_cancellation.is_cancelled(), "stream creation must not cancel the query");

        drop(response);

        assert!(native_cancellation.is_cancelled(), "dropping the DoGet stream must cancel native execution");
        assert_eq!(service.cancellations.len(), 0, "dropped streams must leave no cancellable registry entry");
        let payload = service
            .observability
            .get(&query_id)
            .expect("query stats capture should be retained")
            .to_query_stats_payload();
        assert!(payload.contains("# outcome=failed"), "{payload}");
        assert!(payload.contains("# sealed=true"), "{payload}");
        assert!(payload.contains("[structured_error]"), "{payload}");
        assert!(payload.contains("code=cancelled"), "{payload}");
        assert!(payload.contains("operation=datafusion_nexus.server.flight.cancel_query"), "{payload}");
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

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=semaphore\n"), "{payload}");
        assert!(payload.contains("[admission]\nactive_queries\tqueued_queries\tnext_tenant_id\n"), "{payload}");
        assert!(payload.contains("[semaphore]\nmax_in_flight\tavailable_permits\n1\t1\n"), "{payload}");
    }
}
