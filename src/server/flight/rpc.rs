use super::*;

#[tonic::async_trait]
impl FlightSqlService for DatafusionCudfFlightSqlService {
    type FlightService = Self;

    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>, Status> {
        handle_handshake(self.auth.mode(), self.auth.tokens(), request).await
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        self.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), correlation_id.clone());
        let query_span =
            DatafusionCudfFlightSqlService::query_span(&query_id, correlation_id.as_deref(), "get_flight_info");
        let _query_enter = query_span.enter();
        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            correlation_id = correlation_id.as_deref().unwrap_or(""),
            sql,
            "planning Flight SQL statement"
        );
        if query.transaction_id.is_some() {
            let status = finish_adapter_error_status_with_correlation_id(
                self.observability.as_ref(),
                &query_id,
                correlation_id.as_deref(),
                "planning",
                flight_sql_transactions_unsupported_error(),
            );
            log_statement_status(&query_id, "planning", planning_started_at, &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        let queued_at = Instant::now();
        drop(_query_enter);
        let permit = match self
            .acquire_statement_permit(&query_id, correlation_id.as_deref())
            .instrument(DatafusionCudfFlightSqlService::query_stage_span(
                Some(&query_span),
                &query_id,
                correlation_id.as_deref(),
                "admission",
            ))
            .await
        {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                self.observability.finish_status(&query_id, "admission", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };
        let admission_wait_ms = queued_at.elapsed().as_millis();
        if admission_wait_ms > 0 || self.admission.max_in_flight() > 1 {
            tracing::info!(
                query_id,
                admission_wait_ms,
                max_in_flight = self.admission.max_in_flight(),
                "admitted Flight SQL statement for planning"
            );
        }
        let info = match self
            .statement_flight_info(
                query_id.clone(),
                sql,
                correlation_id.clone(),
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::Statement,
                None,
            )
            .instrument(query_span.clone())
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                self.observability.finish_status(&query_id, "planning", &status);
                self.observability.seal(&query_id);
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
        self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        self.observability.start_query_with_correlation_id(query_id.clone(), "", correlation_id.clone());
        let query_span = DatafusionCudfFlightSqlService::query_span(
            &query_id,
            correlation_id.as_deref(),
            "get_flight_info_prepared_statement",
        );
        let _query_enter = query_span.enter();
        let handle = match std::str::from_utf8(&query.prepared_statement_handle) {
            Ok(handle) => handle.to_string(),
            Err(error) => {
                let error = Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.get_flight_info_prepared_statement.handle",
                    format!("prepared statement handle is not valid UTF-8: {error}"),
                );
                self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
                self.observability.seal(&query_id);
                return Err(status_with_query_id_and_correlation_id(
                    adapter_error_status(error),
                    &query_id,
                    correlation_id.as_deref(),
                ));
            },
        };
        let sql = match self.prepared_statements.get(&handle).map(|entry| entry.sql.clone()) {
            Some(sql) => sql,
            None => {
                let error = Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.get_flight_info_prepared_statement.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                );
                self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
                self.observability.seal(&query_id);
                return Err(status_with_query_id_and_correlation_id(
                    adapter_error_status(error),
                    &query_id,
                    correlation_id.as_deref(),
                ));
            },
        };
        self.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), correlation_id.clone());

        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            correlation_id = correlation_id.as_deref().unwrap_or(""),
            sql,
            prepared_statement_handle = handle.as_str(),
            "planning Flight SQL prepared statement"
        );
        let queued_at = Instant::now();
        drop(_query_enter);
        let permit = match self
            .acquire_statement_permit(&query_id, correlation_id.as_deref())
            .instrument(DatafusionCudfFlightSqlService::query_stage_span(
                Some(&query_span),
                &query_id,
                correlation_id.as_deref(),
                "admission",
            ))
            .await
        {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                self.observability.finish_status(&query_id, "admission", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };
        let info = match self
            .statement_flight_info(
                query_id.clone(),
                sql,
                correlation_id.clone(),
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::PreparedStatement,
                Some(handle.clone()),
            )
            .instrument(query_span.clone())
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                self.observability.finish_status(&query_id, "planning", &status);
                self.observability.seal(&query_id);
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
        let schema = arrow_flight::sql::metadata::GetCatalogsBuilder::new().schema();
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
                    "datafusion_nexus.server.flight.get_flight_info_sql_info.schema",
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
        self.check_auth(&request)?;
        let request_correlation_id = request_correlation_id(&request)?;
        let query_id = std::str::from_utf8(&ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id, request_correlation_id).await
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let request_correlation_id = request_correlation_id(&request)?;
        let query_id = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
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
                "datafusion_nexus.server.flight.do_get_catalogs.build",
                "failed to build Flight SQL catalogs metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            schema,
            batch,
            "datafusion_nexus.server.flight.do_get_catalogs.encode",
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
                "datafusion_nexus.server.flight.do_get_schemas.build",
                "failed to build Flight SQL schemas metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            schema,
            batch,
            "datafusion_nexus.server.flight.do_get_schemas.encode",
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
            "datafusion_nexus.server.flight.do_get_tables.encode",
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
                "datafusion_nexus.server.flight.do_get_table_types.build",
                "failed to build Flight SQL table-types metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "datafusion_nexus.server.flight.do_get_table_types.encode",
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
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|error| {
                arrow_flight_error_to_status_for_operation(
                    error,
                    "datafusion_nexus.server.flight.do_get_sql_info.encode",
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
                "datafusion_nexus.server.flight.do_get_primary_keys.build",
                "failed to build Flight SQL primary-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "datafusion_nexus.server.flight.do_get_primary_keys.encode",
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
                "datafusion_nexus.server.flight.do_get_exported_keys.build",
                "failed to build Flight SQL exported-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "datafusion_nexus.server.flight.do_get_exported_keys.encode",
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
                "datafusion_nexus.server.flight.do_get_imported_keys.build",
                "failed to build Flight SQL imported-keys metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "datafusion_nexus.server.flight.do_get_imported_keys.encode",
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
                "datafusion_nexus.server.flight.do_get_cross_reference.build",
                "failed to build Flight SQL cross-reference metadata batch",
            )
        })?;
        Ok(Response::new(metadata_record_batch_stream(
            batch.schema(),
            batch,
            "datafusion_nexus.server.flight.do_get_cross_reference.encode",
            "failed to encode Flight SQL cross-reference metadata stream",
        )))
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        self.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), correlation_id.clone());
        if query.transaction_id.is_some() {
            let status = finish_adapter_error_status_with_correlation_id(
                self.observability.as_ref(),
                &query_id,
                correlation_id.as_deref(),
                "planning",
                flight_sql_transactions_unsupported_error(),
            );
            self.observability.seal(&query_id);
            return Err(status);
        }
        self.execute_update_sql(query_id, sql, correlation_id, UpdateStatementKind::Statement).await
    }

    async fn do_put_prepared_statement_update(
        &self,
        query: CommandPreparedStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        self.observability.start_query_with_correlation_id(query_id.clone(), "", correlation_id.clone());
        let handle = match std::str::from_utf8(&query.prepared_statement_handle) {
            Ok(handle) => handle.to_string(),
            Err(error) => {
                let error = Error::new(
                    ErrorCode::InvalidRequest,
                    "datafusion_nexus.server.flight.do_put_prepared_statement_update.handle",
                    format!("prepared statement handle is not valid UTF-8: {error}"),
                );
                self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
                self.observability.seal(&query_id);
                return Err(status_with_query_id_and_correlation_id(
                    adapter_error_status(error),
                    &query_id,
                    correlation_id.as_deref(),
                ));
            },
        };
        let (sql, already_executed) = match self
            .prepared_statements
            .get(&handle)
            .map(|entry| (entry.sql.clone(), entry.ddl_already_executed()))
        {
            Some(entry) => entry,
            None => {
                let error = Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.do_put_prepared_statement_update.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                );
                self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
                self.observability.seal(&query_id);
                return Err(status_with_query_id_and_correlation_id(
                    adapter_error_status(error),
                    &query_id,
                    correlation_id.as_deref(),
                ));
            },
        };
        self.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), correlation_id.clone());

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
            self.observability.finish_success(&query_id);
            self.observability.seal(&query_id);
            self.observability.delete(&query_id);
            return Ok(NO_UPDATED_RECORDS_COUNT);
        }

        let record_count =
            self.execute_update_sql(query_id, sql, correlation_id, UpdateStatementKind::PreparedStatement).await?;
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
        self.check_auth(&request)?;
        let correlation_id = request_correlation_id(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        self.observability.start_query_with_correlation_id(query_id.clone(), sql.clone(), correlation_id.clone());
        let query_span = DatafusionCudfFlightSqlService::query_span(
            &query_id,
            correlation_id.as_deref(),
            "create_prepared_statement",
        );
        let _query_enter = query_span.enter();
        if query.transaction_id.is_some() {
            let status = finish_adapter_error_status_with_correlation_id(
                self.observability.as_ref(),
                &query_id,
                correlation_id.as_deref(),
                "planning",
                flight_sql_transactions_unsupported_error(),
            );
            self.observability.seal(&query_id);
            return Err(status);
        }
        if self.prepared_statements.len() >= MAX_PREPARED_STATEMENTS {
            let error = Error::new(
                ErrorCode::ServiceOverloaded,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.admission",
                format!("too many live prepared statements (limit {MAX_PREPARED_STATEMENTS})"),
            )
            .with_fact("limit", MAX_PREPARED_STATEMENTS);
            self.observability.finish_failed(&query_id, "admission", CapturedError::from_adapter_error(&error));
            self.observability.seal(&query_id);
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        // Plan logically to derive the result-set schema the client needs
        // up-front. This does *not* run the physical optimizer rule and does
        // not consume an admission permit — the real plan / admission /
        // ticket happen later in `get_flight_info_prepared_statement` when
        // the client actually executes this prepared statement.
        let state = self.ctx.state();
        drop(_query_enter);
        let logical_plan = tokio::select! {
            result = state.create_logical_plan(&sql)
                .instrument(DatafusionCudfFlightSqlService::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => {
                result.map_err(|error| {
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(
                            &error,
                            "datafusion_nexus.server.flight.do_action_create_prepared_statement.create_logical_plan",
                        ),
                    );
                    self.observability.seal(&query_id);
                    status_with_query_id_and_correlation_id(
                        datafusion_error_to_status_for_operation(
                            error,
                            "datafusion_nexus.server.flight.do_action_create_prepared_statement.create_logical_plan",
                        ),
                        &query_id,
                        correlation_id.as_deref(),
                    )
                })?
            },
            _ = self.shutdown_token.cancelled() => {
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    server_shutdown_error(),
                );
                self.observability.seal(&query_id);
                return Err(status);
            }
        };
        if let Err(error) = reject_parameterized_plan(
            &logical_plan,
            "datafusion_nexus.server.flight.do_action_create_prepared_statement.parameter_binding",
        ) {
            self.observability.finish_failed(
                &query_id,
                "planning",
                datafusion_error_to_captured_error(
                    &error,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.parameter_binding",
                ),
            );
            self.observability.seal(&query_id);
            return Err(status_with_query_id_and_correlation_id(
                datafusion_error_to_status_for_operation(
                    error,
                    "datafusion_nexus.server.flight.do_action_create_prepared_statement.parameter_binding",
                ),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        let dataset_schema = match self.dataset_schema_for_prepared_statement(&logical_plan) {
            Ok(schema) => schema,
            Err(error) => {
                self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
                self.observability.seal(&query_id);
                return Err(status_with_query_id_and_correlation_id(
                    adapter_error_status(error),
                    &query_id,
                    correlation_id.as_deref(),
                ));
            },
        };
        let dataset_schema_ipc = if self.prepared_statement_has_result_set(&logical_plan) {
            match schema_to_ipc_bytes(&dataset_schema) {
                Ok(bytes) => bytes,
                Err(error) => {
                    let status = status_with_query_id_and_correlation_id(
                        arrow_error_to_status_for_operation(
                            error,
                            "datafusion_nexus.server.flight.do_action_create_prepared_statement.dataset_schema_ipc",
                            "failed to encode Flight SQL prepared-statement dataset schema",
                        ),
                        &query_id,
                        correlation_id.as_deref(),
                    );
                    self.observability.finish_status(&query_id, "planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            }
        } else {
            prost::bytes::Bytes::new()
        };

        // DDL prepares without a result-set schema; classify it once here so
        // neither execution channel re-plans to decide whether a re-run of the
        // held handle should be deduplicated into a no-op.
        let no_result_ddl = matches!(logical_plan, LogicalPlan::Ddl(_));
        // Flight SQL's CreatePreparedStatement response has no query_id field.
        // Reuse the returned opaque handle as this action's query_id so the
        // successful create action remains addressable through GetQueryStats.
        let handle = query_id.clone();
        self.prepared_statements.insert(handle.clone(), PreparedStatementEntry::new(sql, no_result_ddl));
        tracing::info!(prepared_statement_handle = handle.as_str(), "created Flight SQL prepared statement");
        self.observability.finish_success(&query_id);
        self.observability.seal(&query_id);

        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into_bytes().into(),
            dataset_schema: dataset_schema_ipc,
            parameter_schema: prost::bytes::Bytes::new(),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<(), Status> {
        self.check_auth(&request)?;
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        // Closing an unknown handle is not an error: arrow-flight clients
        // may retry close, and the handle may have been swept by a prior
        // close on the same connection. Trace-log the no-op so client
        // lifecycle bugs (e.g. double-close, close-after-eviction) are
        // observable without breaking the idempotent contract.
        if self.prepared_statements.remove(handle).is_none() {
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
        self.check_auth(&request)?;
        let result = if self.cancel_from_flight_info(&query.info)? {
            FLIGHT_SQL_CANCEL_RESULT_CANCELLING
        } else {
            FLIGHT_SQL_CANCEL_RESULT_NOT_CANCELLABLE
        };
        Ok(ActionCancelQueryResult { result })
    }

    async fn list_custom_actions(&self) -> Option<Vec<Result<arrow_flight::ActionType, Status>>> {
        Some(vec![Ok(arrow_flight::ActionType {
            r#type: GET_QUERY_STATS_ACTION.to_string(),
            description: "Returns retained datafusion-nexus query diagnostics for a UTF-8 query_id body".to_string(),
        })])
    }

    async fn do_action_fallback(
        &self,
        request: Request<arrow_flight::Action>,
    ) -> Result<Response<<Self as FlightService>::DoActionStream>, Status> {
        self.check_auth(&request)?;
        if request.get_ref().r#type != GET_QUERY_STATS_ACTION {
            return Err(Status::invalid_argument(format!(
                "do_action: The defined request is invalid: {:?}",
                request.get_ref().r#type
            )));
        }
        let query_id = std::str::from_utf8(&request.get_ref().body)
            .map_err(|error| Status::invalid_argument(format!("GetQueryStats query_id body is not UTF-8: {error}")))?
            .trim();
        if query_id.is_empty() {
            return Err(Status::invalid_argument("GetQueryStats query_id body is empty"));
        }
        let captured = self
            .observability
            .get(query_id)
            .ok_or_else(|| Status::not_found(format!("unknown or expired query_id: {query_id}")))?;
        let output = futures::stream::iter(vec![Ok(arrow_flight::Result {
            body: captured.to_query_stats_payload().into_bytes().into(),
        })]);
        Ok(Response::new(Box::pin(output) as <Self as FlightService>::DoActionStream))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}
