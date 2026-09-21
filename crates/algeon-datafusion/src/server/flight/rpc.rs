use std::{sync::Arc, time::Instant};

use arrow_flight::{Result as FlightResult, sql::metadata::GetCatalogsBuilder};
use futures::stream::{iter, once, unfold};
use prost::bytes::Bytes;
use tokio::task::spawn_blocking;

use super::*;
use crate::server::{attempt::AttemptOptions, error::adapter_error_to_status, observability::RedactedSql};

#[tonic::async_trait]
impl FlightSqlService for DatafusionAlgeonFlightSqlService {
    type FlightService = Self;

    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<BoxStream<'static, Result<HandshakeResponse, Status>>>, Status> {
        handle_handshake(&self.auth, request).await
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let caller = self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let sql = query.query;
        let parsed_statement = self.parse_flight_statement(&sql);
        let execution_domain = parsed_statement
            .as_ref()
            .map_or(StatementExecutionDomain::Data, |statement| self.classify_statement_candidate(statement));
        let registration = if query.transaction_id.is_some() || execution_domain == StatementExecutionDomain::Metadata {
            self.register_new_query(sql.clone(), correlation_id.clone())
        } else {
            self.register_new_attempt_query(sql.clone(), correlation_id.clone())
        };
        let (query_id, mut planning_lease) = registration.map_err(|(query_id, status)| {
            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
        })?;
        self.ownership.insert(query_id.clone(), caller);
        let query_span =
            DatafusionAlgeonFlightSqlService::query_span(&query_id, correlation_id.as_deref(), "get_flight_info");
        let _query_enter = query_span.enter();
        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            correlation_id = tracing_correlation_id(correlation_id.as_deref()),
            sql = %RedactedSql(&sql),
            "planning Flight SQL statement"
        );
        if query.transaction_id.is_some() {
            let (status, captured) = adapter_error_to_status_and_capture(flight_sql_transactions_unsupported_error());
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            log_statement_status(&query_id, Some(&sql), "planning", planning_started_at, &status);
            planning_lease.fail("planning", captured);
            return Err(status);
        }
        let explain_gpu = match parse_explain_gpu(&sql) {
            Ok(statement) => statement,
            Err(error) => {
                let attempt = match self.admission.begin_attempt(AttemptOptions {
                    query_id: Some(query_id.clone()),
                    trace_context: correlation_id.clone(),
                    deadline: None,
                    observation: self.runtime_observation_config,
                }) {
                    Ok(attempt) => attempt,
                    Err(begin_error) => {
                        self.coordinator.allow_missing_attempt_report(&query_id);
                        let (status, captured) = adapter_error_to_status_and_capture(begin_error);
                        let status =
                            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                        log_statement_status(&query_id, Some(&sql), "planning", planning_started_at, &status);
                        planning_lease.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                };
                if let Some(attempt) = attempt.as_ref() {
                    self.register_attempt_report_handoff(
                        attempt,
                        CoordinatorReportHandoffTerminal::planning(&planning_lease),
                    );
                    attempt.planning_failed_adapter(&error);
                }
                let (status, captured) = adapter_error_to_status_and_capture(error);
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                log_statement_status(&query_id, Some(&sql), "planning", planning_started_at, &status);
                planning_lease.fail("planning", captured);
                return Err(status);
            },
        };
        drop(_query_enter);
        let descriptor = request.into_inner();
        let info_result = if let Some(explain_gpu) = explain_gpu {
            self.explain_gpu_flight_info(
                planning_lease,
                query_id.clone(),
                explain_gpu.inner_sql,
                explain_gpu.output_mode,
                correlation_id.clone(),
                descriptor,
                planning_started_at,
            )
        } else {
            self.statement_flight_info(
                planning_lease,
                query_id.clone(),
                sql.clone(),
                correlation_id.clone(),
                descriptor,
                planning_started_at,
                StatementTicketKind::Statement,
                None,
                parsed_statement,
                execution_domain,
                None,
            )
            .instrument(query_span.clone())
            .await
        };
        let info = match info_result {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                log_statement_status(&query_id, Some(&sql), "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        let mut response = Response::new(info);
        attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
        Ok(response)
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let caller = self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let handle = match std::str::from_utf8(&query.prepared_statement_handle) {
            Ok(handle) => handle.to_string(),
            Err(error) => {
                let error = Error::new(
                    ErrorCode::InvalidRequest,
                    "algeon_datafusion.server.flight.get_flight_info_prepared_statement.handle",
                    format!("prepared statement handle is not valid UTF-8: {error}"),
                );
                return Err(record_non_attempt_failure(self, &correlation_id, error)?);
            },
        };
        self.ownership.authorize(&handle, &caller)?;
        let (sql, execution_domain, planned_forest_model_digests) = match self
            .prepared_statements
            .get(&handle)
            .map(|entry| (entry.sql.clone(), entry.execution_domain, entry.forest_model_digests.clone()))
        {
            Some(entry) => entry,
            None => {
                let error = Error::new(
                    ErrorCode::ObjectNotFound,
                    "algeon_datafusion.server.flight.get_flight_info_prepared_statement.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                );
                return Err(record_non_attempt_failure(self, &correlation_id, error)?);
            },
        };
        if let Some(error) = explain_gpu_endpoint_error(
            &sql,
            "algeon_datafusion.server.flight.get_flight_info_prepared_statement.explain_gpu",
            "EXPLAIN GPU cannot be executed through a prepared statement; use the statement query endpoint",
        ) {
            return Err(record_non_attempt_failure(self, &correlation_id, error)?);
        }
        let registration = match execution_domain {
            StatementExecutionDomain::Metadata => self.register_new_query(sql.clone(), correlation_id.clone()),
            StatementExecutionDomain::Data => self.register_new_attempt_query(sql.clone(), correlation_id.clone()),
        };
        let (query_id, planning_lease) = registration.map_err(|(query_id, status)| {
            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
        })?;
        self.ownership.insert(query_id.clone(), caller);
        let parsed_statement = self.parse_flight_statement(&sql);
        let query_span = DatafusionAlgeonFlightSqlService::query_span(
            &query_id,
            correlation_id.as_deref(),
            "get_flight_info_prepared_statement",
        );
        let _query_enter = query_span.enter();
        self.coordinator.set_sql(&query_id, sql.clone());

        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            correlation_id = tracing_correlation_id(correlation_id.as_deref()),
            sql,
            prepared_statement_handle = handle.as_str(),
            "planning Flight SQL prepared statement"
        );
        drop(_query_enter);
        let info = match self
            .statement_flight_info(
                planning_lease,
                query_id.clone(),
                sql.clone(),
                correlation_id.clone(),
                request.into_inner(),
                planning_started_at,
                StatementTicketKind::PreparedStatement,
                Some(handle.clone()),
                parsed_statement,
                execution_domain,
                Some(planned_forest_model_digests),
            )
            .instrument(query_span.clone())
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                log_statement_status(&query_id, Some(&sql), "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        let mut response = Response::new(info);
        attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
        Ok(response)
    }

    async fn get_flight_info_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        // `CommandGetCatalogs` is a unit message; the builder produces a
        // static schema, and we echo the (Copy) command back as the ticket.
        let schema = GetCatalogsBuilder::new().schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = GetDbSchemasBuilder::from(query.clone()).schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_tables(
        &self,
        query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = GetTablesBuilder::from(query.clone()).schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_table_types(
        &self,
        query: CommandGetTableTypes,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_table_types_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let ticket = Ticket { ticket: query.as_any().encode_to_vec().into() };
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let sql_info = sql_info_data()?;
        let info = FlightInfo::new()
            .try_with_schema(query.into_builder(&sql_info).schema().as_ref())
            .map_err(|error| {
                arrow_error_to_status_for_operation(
                    error,
                    "algeon_datafusion.server.flight.get_flight_info_sql_info.schema",
                    "failed to build Flight SQL SqlInfo FlightInfo schema",
                )
            })?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint);
        Ok(Response::new(info))
    }

    async fn get_flight_info_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_primary_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_exported_keys(
        &self,
        query: CommandGetExportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_imported_keys(
        &self,
        query: CommandGetImportedKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn get_flight_info_cross_reference(
        &self,
        query: CommandGetCrossReference,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let descriptor = request.into_inner();
        let schema = flight_sql_foreign_keys_schema();
        Ok(Response::new(metadata_flight_info(query, descriptor, schema.as_ref())?))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let query_id = std::str::from_utf8(&ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.authorize_id(&request, &query_id)?;
        let request_correlation_id = request_correlation_id(&request)?;
        self.execute_ticket(query_id, request_correlation_id).await
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let query_id = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.authorize_id(&request, &query_id)?;
        let request_correlation_id = request_correlation_id(&request)?;
        self.execute_ticket(query_id, request_correlation_id).await
    }

    async fn do_get_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let mut builder = query.into_builder();
        for name in self.ctx.catalog_names() {
            builder.append(name);
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(|error| {
            arrow_flight_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_catalogs.build",
                "failed to build Flight SQL catalogs metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            schema,
            batch,
            "algeon_datafusion.server.flight.do_get_catalogs.encode",
            "failed to encode Flight SQL catalogs metadata stream",
        )))
    }

    async fn do_get_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let mut builder = GetDbSchemasBuilder::from(query);
        for catalog_name in self.ctx.catalog_names() {
            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                builder.append(&catalog_name, &schema_name);
            }
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(|error| {
            arrow_flight_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_schemas.build",
                "failed to build Flight SQL schemas metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            schema,
            batch,
            "algeon_datafusion.server.flight.do_get_schemas.encode",
            "failed to encode Flight SQL schemas metadata stream",
        )))
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let (schema, batch) = self.build_get_tables_response(query).await?;
        Ok(Response::new(metadata_record_batch_stream(
            schema,
            batch,
            "algeon_datafusion.server.flight.do_get_tables.encode",
            "failed to encode Flight SQL GetTables metadata stream",
        )))
    }

    async fn do_get_table_types(
        &self,
        _query: CommandGetTableTypes,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = build_table_types_batch().map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_table_types.build",
                "failed to build Flight SQL table-types metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "algeon_datafusion.server.flight.do_get_table_types.encode",
            "failed to encode Flight SQL table-types metadata stream",
        )))
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let sql_info = sql_info_data()?;
        let builder = query.into_builder(&sql_info);
        let schema = builder.schema();
        let batch = builder.build();
        let stream = encode_flight_data_stream(schema, once(async { batch })).map_err(|error| {
            arrow_flight_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_sql_info.encode",
                "failed to encode Flight SQL SqlInfo response",
            )
        });
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_primary_keys(
        &self,
        _query: CommandGetPrimaryKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        // DataFusion's catalog does not carry primary-key constraints, so the
        // response is always an empty batch with the Flight SQL standard
        // schema. This lets clients (DataGrip, the Arrow Flight SQL JDBC
        // driver, ADBC) complete introspection without a UNIMPLEMENTED RPC
        // failure tearing down the session.
        let batch = empty_primary_keys_batch().map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_primary_keys.build",
                "failed to build Flight SQL primary-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "algeon_datafusion.server.flight.do_get_primary_keys.encode",
            "failed to encode Flight SQL primary-keys metadata stream",
        )))
    }

    async fn do_get_exported_keys(
        &self,
        _query: CommandGetExportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_exported_keys.build",
                "failed to build Flight SQL exported-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "algeon_datafusion.server.flight.do_get_exported_keys.encode",
            "failed to encode Flight SQL exported-keys metadata stream",
        )))
    }

    async fn do_get_imported_keys(
        &self,
        _query: CommandGetImportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_imported_keys.build",
                "failed to build Flight SQL imported-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "algeon_datafusion.server.flight.do_get_imported_keys.encode",
            "failed to encode Flight SQL imported-keys metadata stream",
        )))
    }

    async fn do_get_cross_reference(
        &self,
        _query: CommandGetCrossReference,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_cross_reference.build",
                "failed to build Flight SQL cross-reference metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "algeon_datafusion.server.flight.do_get_cross_reference.encode",
            "failed to encode Flight SQL cross-reference metadata stream",
        )))
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        let caller = self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let sql = query.query;
        if query.transaction_id.is_some() {
            return Err(record_non_attempt_failure(
                self,
                &correlation_id,
                flight_sql_transactions_unsupported_error(),
            )?);
        }
        if let Some(error) = explain_gpu_endpoint_error(
            &sql,
            "algeon_datafusion.server.flight.do_put_statement_update.explain_gpu",
            "EXPLAIN GPU is a statement-query operation; use the Flight SQL statement query endpoint",
        ) {
            return Err(record_non_attempt_failure(self, &correlation_id, error)?);
        }
        let (query_id, planning_lease) =
            self.register_new_attempt_query(sql.clone(), correlation_id.clone()).map_err(|(query_id, status)| {
                status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
            })?;
        self.ownership.insert(query_id, caller);
        self.execute_update_sql(planning_lease, sql, correlation_id, UpdateStatementKind::Statement, None).await
    }

    async fn do_put_prepared_statement_update(
        &self,
        query: CommandPreparedStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        let caller = self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let handle = match std::str::from_utf8(&query.prepared_statement_handle) {
            Ok(handle) => handle.to_string(),
            Err(error) => {
                let error = Error::new(
                    ErrorCode::InvalidRequest,
                    "algeon_datafusion.server.flight.do_put_prepared_statement_update.handle",
                    format!("prepared statement handle is not valid UTF-8: {error}"),
                );
                return Err(record_non_attempt_failure(self, &correlation_id, error)?);
            },
        };
        self.ownership.authorize(&handle, &caller)?;
        let (sql, already_executed, planned_forest_model_digests) = match self
            .prepared_statements
            .get(&handle)
            .map(|entry| (entry.sql.clone(), entry.ddl_already_executed(), entry.forest_model_digests.clone()))
        {
            Some(entry) => entry,
            None => {
                let error = Error::new(
                    ErrorCode::ObjectNotFound,
                    "algeon_datafusion.server.flight.do_put_prepared_statement_update.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                );
                return Err(record_non_attempt_failure(self, &correlation_id, error)?);
            },
        };
        if let Some(error) = explain_gpu_endpoint_error(
            &sql,
            "algeon_datafusion.server.flight.do_put_prepared_statement_update.explain_gpu",
            "EXPLAIN GPU is a statement-query operation and cannot be executed through a prepared update",
        ) {
            return Err(record_non_attempt_failure(self, &correlation_id, error)?);
        }
        let (query_id, planning_lease) =
            self.register_new_attempt_query(sql.clone(), correlation_id.clone()).map_err(|(query_id, status)| {
                status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
            })?;
        self.ownership.insert(query_id.clone(), caller);
        self.coordinator.set_sql(&query_id, sql.clone());

        // DataGrip-class clients may re-run the held handle after executing a
        // DDL through executeUpdate. A no-result-set DDL handle that already
        // executed once is a no-op on re-execution rather than an "already
        // exists" rejection.
        if already_executed {
            tracing::info!(
                query_id,
                prepared_statement_handle = handle.as_str(),
                "deduplicated repeated prepared DDL execution on the update channel; returning a no-op"
            );
            return self.complete_deduplicated_update(planning_lease, sql, correlation_id).await;
        }

        let record_count = self
            .execute_update_sql(
                planning_lease,
                sql,
                correlation_id,
                UpdateStatementKind::PreparedStatement,
                Some(planned_forest_model_digests),
            )
            .await?;
        if let Some(entry) = self.prepared_statements.get(&handle) {
            entry.mark_executed();
        }
        Ok(record_count)
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        let caller = self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let sql = query.query;
        let (query_id, mut planning_lease) =
            self.register_new_query(sql.clone(), correlation_id.clone()).map_err(|(query_id, status)| {
                status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
            })?;
        let query_span = DatafusionAlgeonFlightSqlService::query_span(
            &query_id,
            correlation_id.as_deref(),
            "create_prepared_statement",
        );
        let _query_enter = query_span.enter();
        if query.transaction_id.is_some() {
            let (status, captured) = adapter_error_to_status_and_capture(flight_sql_transactions_unsupported_error());
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        if let Some(error) = explain_gpu_endpoint_error(
            &sql,
            "algeon_datafusion.server.flight.do_action_create_prepared_statement.explain_gpu",
            "EXPLAIN GPU cannot be prepared; use the Flight SQL statement query endpoint",
        ) {
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        // Plan logically to derive the result-set schema the client needs
        // up-front. This does *not* run the physical optimizer rule and does
        // not consume GPU admission; the physical plan and plan-scoped native
        // ticket happen later when the client executes this prepared statement.
        let state = self.ctx.state();
        let parsed_statement = self.parse_flight_statement(&sql);
        let execution_domain = parsed_statement
            .as_ref()
            .map_or(StatementExecutionDomain::Data, |statement| self.classify_statement_candidate(statement));
        drop(_query_enter);
        let logical_plan = tokio::select! {
            result = async {
                let statement = parsed_statement?;
                state.statement_to_plan(statement).await
            }
                .instrument(DatafusionAlgeonFlightSqlService::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => {
                match result {
                    Ok(plan) => plan,
                    Err(error) => {
                        let (status, captured) = datafusion_error_to_status_and_capture(
                            error,
                            "algeon_datafusion.server.flight.do_action_create_prepared_statement.create_logical_plan",
                            FailedStatement::Other,
                        );
                        let status = status_with_query_id_and_correlation_id(
                            status,
                            &query_id,
                            correlation_id.as_deref(),
                        );
                        planning_lease.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                }
            },
            _ = self.shutdown_token.cancelled() => {
                let (status, captured) = adapter_error_to_status_and_capture(server_shutdown_error());
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id,
                    correlation_id.as_deref(),
                );
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            }
        };
        if execution_domain == StatementExecutionDomain::Metadata
            && let Err(error) = self.verify_metadata_logical_plan(
                &logical_plan,
                "algeon_datafusion.server.flight.do_action_create_prepared_statement.metadata_logical_proof",
            )
        {
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        if let Err(error) = reject_parameterized_plan(
            &logical_plan,
            "algeon_datafusion.server.flight.do_action_create_prepared_statement.parameter_binding",
        ) {
            let (status, captured) = datafusion_error_to_status_and_capture(
                error,
                "algeon_datafusion.server.flight.do_action_create_prepared_statement.parameter_binding",
                FailedStatement::Other,
            );
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        if let Err(error) = self.verify_prepared_statement_plan(&logical_plan) {
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        let dataset_schema = match flight_result_schema(&state, &logical_plan) {
            Ok(schema) => schema,
            Err(error) => {
                let (status, captured) = datafusion_error_to_status_and_capture(
                    error,
                    "algeon_datafusion.server.flight.do_action_create_prepared_statement.dataset_schema",
                    FailedStatement::Other,
                );
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        let dataset_schema_ipc = if self.prepared_statement_has_result_set(&logical_plan) {
            match schema_to_ipc_bytes(&dataset_schema) {
                Ok(bytes) => bytes,
                Err(error) => {
                    let (status, captured) = arrow_error_to_status_and_capture(
                        error,
                        "algeon_datafusion.server.flight.do_action_create_prepared_statement.dataset_schema_ipc",
                        "failed to encode Flight SQL prepared-statement dataset schema",
                    );
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    planning_lease.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            }
        } else {
            Bytes::new()
        };

        // DDL prepares without a result-set schema; classify it once here so
        // neither execution channel re-plans to decide whether a re-run of the
        // held handle should be deduplicated into a no-op.
        let no_result_ddl = matches!(logical_plan, LogicalPlan::Ddl(_));
        // Flight SQL's CreatePreparedStatement response has no query_id field.
        // Reuse the returned opaque handle as this action's query_id so the
        // successful create action remains addressable through GetQueryStats.
        let handle = query_id.clone();
        // The cap check and the insert that consumes it are one operation in
        // the store, so concurrent creates cannot both pass the check.
        if let Err(limit) = self.prepared_statements.insert(
            handle.clone(),
            caller.clone(),
            PreparedStatementEntry::new(sql, no_result_ddl, execution_domain, forest_model_digests(&logical_plan)),
        ) {
            let error = Error::new(
                ErrorCode::ServiceOverloaded,
                "algeon_datafusion.server.flight.do_action_create_prepared_statement.admission",
                format!("{} (limit {})", limit.message(), limit.limit()),
            )
            .with_log_safe_fact("limit", limit.limit());
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("admission", &status, captured);
            return Err(status);
        }
        self.ownership.insert(handle.clone(), caller);
        tracing::info!(prepared_statement_handle = handle.as_str(), "created Flight SQL prepared statement");
        planning_lease.succeed();

        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into_bytes().into(),
            dataset_schema: dataset_schema_ipc,
            parameter_schema: Bytes::new(),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<(), Status> {
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let caller = self.authorize_id(&request, handle)?;
        // Closing an unknown handle is not an error: arrow-flight clients
        // may retry close, and the handle may have been swept by a prior
        // close on the same connection. Trace-log the no-op so client
        // lifecycle bugs (e.g. double-close, close-after-eviction) are
        // observable without breaking the idempotent contract.
        if self.prepared_statements.remove(handle, &caller) {
            self.ownership.remove(handle);
        } else {
            tracing::trace!(
                prepared_statement_handle = handle,
                "close of unknown or already-closed prepared statement (no-op)"
            );
        }
        Ok(())
    }

    async fn do_action_cancel_query(
        &self,
        query: ActionCancelQueryRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<ActionCancelQueryResult, Status> {
        let caller = self.check_auth(&request)?;
        let result = if self.cancel_from_flight_info(&query.info, &caller)? {
            FLIGHT_SQL_CANCEL_RESULT_CANCELLING
        } else {
            FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE
        };
        Ok(ActionCancelQueryResult { result })
    }

    async fn list_custom_actions(&self) -> Option<Vec<Result<arrow_flight::ActionType, Status>>> {
        Some(vec![
            Ok(arrow_flight::ActionType {
                r#type: GET_QUERY_STATS_ACTION.to_string(),
                description: "Returns retained algeon-datafusion query diagnostics for a UTF-8 query_id body"
                    .to_string(),
            }),
            Ok(arrow_flight::ActionType {
                r#type: GET_QUERY_RUNTIME_ARTIFACT_ACTION.to_string(),
                description: "Returns a versioned control frame and sealed native runtime JSONL artifact".to_string(),
            }),
            Ok(arrow_flight::ActionType {
                r#type: GET_ADMISSION_STATS_ACTION.to_string(),
                description: "Returns current algeon-datafusion admission and memory-governor diagnostics".to_string(),
            }),
        ])
    }

    async fn do_action_fallback(
        &self,
        request: Request<arrow_flight::Action>,
    ) -> Result<Response<<Self as FlightService>::DoActionStream>, Status> {
        let caller = self.check_auth(&request)?;
        let action_type = request.get_ref().r#type.as_str();
        if action_type == GET_ADMISSION_STATS_ACTION {
            let payload = self
                .admission
                .diagnostics_payload()
                .map_err(|error| Status::internal(format!("failed to read admission diagnostics: {error}")))?;
            let output = iter(vec![Ok(FlightResult { body: payload.into_bytes().into() })]);
            return Ok(Response::new(Box::pin(output) as <Self as FlightService>::DoActionStream));
        }
        if action_type == GET_QUERY_RUNTIME_ARTIFACT_ACTION {
            self.expire_stale_tickets(Instant::now());
            let query_id = std::str::from_utf8(&request.get_ref().body)
                .map_err(|error| {
                    Status::invalid_argument(format!("GetQueryRuntimeArtifact query_id body is not UTF-8: {error}"))
                })?
                .trim();
            if query_id.is_empty() {
                return Err(Status::invalid_argument("GetQueryRuntimeArtifact query_id body is empty"));
            }
            self.ownership.authorize(query_id, &caller)?;
            let response = self
                .coordinator
                .lease_runtime_artifact(query_id)
                .map_err(|error| {
                    adapter_error_to_status(
                        &error.adapter_error("algeon_datafusion.server.flight.get_query_runtime_artifact.egress"),
                    )
                })?
                .ok_or_else(|| Status::not_found(format!("unknown or expired query_id: {query_id}")))?;
            let output = unfold(response, |mut response| async move {
                response.next_frame().map(|body| (Ok(FlightResult { body }), response))
            });
            return Ok(Response::new(Box::pin(output) as <Self as FlightService>::DoActionStream));
        }
        if action_type != GET_QUERY_STATS_ACTION {
            return Err(Status::invalid_argument(format!(
                "do_action: The defined request is invalid: {action_type:?}"
            )));
        }
        self.expire_stale_tickets(Instant::now());
        let query_id = std::str::from_utf8(&request.get_ref().body)
            .map_err(|error| Status::invalid_argument(format!("GetQueryStats query_id body is not UTF-8: {error}")))?
            .trim();
        if query_id.is_empty() {
            return Err(Status::invalid_argument("GetQueryStats query_id body is empty"));
        }
        self.ownership.authorize(query_id, &caller)?;
        // `lease_query_stats` waits on a `std` Condvar while another query
        // renders its stats. Run it on the blocking pool so this handler
        // never parks a tokio worker on another query's render.
        let coordinator = Arc::clone(&self.coordinator);
        let owned_query_id = query_id.to_string();
        let leased = spawn_blocking(move || coordinator.lease_query_stats(&owned_query_id))
            .await
            .map_err(|error| Status::internal(format!("GetQueryStats blocking task failed: {error}")))?;
        let response = leased
            .map_err(|error| {
                adapter_error_to_status(&error.adapter_error("algeon_datafusion.server.flight.get_query_stats.egress"))
            })?
            .ok_or_else(|| Status::not_found(format!("unknown or expired query_id: {query_id}")))?;
        let output = unfold(response, |mut response| async move {
            response.next_frame().map(|body| (Ok(FlightResult { body }), response))
        });
        Ok(Response::new(Box::pin(output) as <Self as FlightService>::DoActionStream))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

/// Record a Flight request failure that occurs before an attempt begins. The
/// coordinator report remains available to clients, but no attempt exists to
/// provide a terminal report.
fn record_non_attempt_failure(
    service: &DatafusionAlgeonFlightSqlService,
    correlation_id: &Option<String>,
    error: Error,
) -> std::result::Result<Status, Status> {
    let (query_id, planning_lease) =
        service.register_new_query(String::new(), correlation_id.clone()).map_err(|(query_id, status)| {
            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
        })?;
    let (status, captured) = adapter_error_to_status_and_capture(error);
    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
    planning_lease.fail("planning", captured);
    Ok(status)
}

fn explain_gpu_endpoint_error(sql: &str, operation: &'static str, message: &'static str) -> Option<Error> {
    match parse_explain_gpu(sql) {
        Ok(Some(_)) | Err(_) => Some(Error::new(ErrorCode::InvalidRequest, operation, message)),
        Ok(None) => None,
    }
}
