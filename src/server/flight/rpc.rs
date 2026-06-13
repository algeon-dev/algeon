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
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = query.query;
        let planning_started_at = Instant::now();
        tracing::info!(query_id, sql, "planning Flight SQL statement");
        if query.transaction_id.is_some() {
            let status = flight_sql_transactions_unsupported_status(&query_id);
            log_statement_status(&query_id, "planning", planning_started_at, &status);
            return Err(status);
        }
        let queued_at = Instant::now();
        let permit = match self.acquire_statement_permit(&query_id).await {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
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
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::Statement,
            )
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id(status, &query_id);
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        Ok(Response::new(info))
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.check_auth(&request)?;
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let sql = self.prepared_statements.get(&handle).map(|entry| entry.value().clone()).ok_or_else(|| {
            adapter_error_status_with_query_id(
                Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.get_flight_info_prepared_statement.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                ),
                &query_id,
            )
        })?;

        let planning_started_at = Instant::now();
        tracing::info!(
            query_id,
            sql,
            prepared_statement_handle = handle.as_str(),
            "planning Flight SQL prepared statement"
        );
        let queued_at = Instant::now();
        let permit = match self.acquire_statement_permit(&query_id).await {
            Ok(permit) => permit,
            Err(status) => {
                log_statement_status(&query_id, "admission", queued_at, &status);
                return Err(status);
            },
        };
        let info = match self
            .statement_flight_info(
                query_id.clone(),
                sql,
                request.into_inner(),
                planning_started_at,
                permit,
                StatementTicketKind::PreparedStatement,
            )
            .await
        {
            Ok(info) => info,
            Err(status) => {
                let status = status_with_query_id(status, &query_id);
                log_statement_status(&query_id, "planning", planning_started_at, &status);
                return Err(status);
            },
        };
        Ok(Response::new(info))
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
            .map_err(arrow_error_to_status)?
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
        let query_id = std::str::from_utf8(&ticket.statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id).await
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let query_id = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .to_string();
        self.execute_ticket(query_id).await
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
        let batch = builder.build().map_err(arrow_flight_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
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
        let batch = builder.build().map_err(arrow_flight_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let (schema, batch) = self.build_get_tables_response(query).await?;
        Ok(Response::new(metadata_record_batch_stream(schema, batch)))
    }

    async fn do_get_table_types(
        &self,
        _query: CommandGetTableTypes,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = build_table_types_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
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
            .map_err(Status::from);
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
        let batch = empty_primary_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_exported_keys(
        &self,
        _query: CommandGetExportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_imported_keys(
        &self,
        _query: CommandGetImportedKeys,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_get_cross_reference(
        &self,
        _query: CommandGetCrossReference,
        request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.check_auth(&request)?;
        let batch = empty_foreign_keys_batch().map_err(arrow_error_to_status)?;
        Ok(Response::new(metadata_record_batch_stream(batch.schema(), batch)))
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        if query.transaction_id.is_some() {
            return Err(flight_sql_transactions_unsupported_status(&query_id));
        }
        self.execute_update_sql(query_id, query.query, UpdateStatementKind::Statement).await
    }

    async fn do_put_prepared_statement_update(
        &self,
        query: CommandPreparedStatementUpdate,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let handle = std::str::from_utf8(&query.prepared_statement_handle)
            .map_err(|error| {
                adapter_error_status_with_query_id(
                    Error::new(
                        ErrorCode::InvalidRequest,
                        "datafusion_nexus.server.flight.do_put_prepared_statement_update.handle",
                        format!("prepared statement handle is not valid UTF-8: {error}"),
                    ),
                    &query_id,
                )
            })?
            .to_string();
        let sql = self.prepared_statements.get(&handle).map(|entry| entry.value().clone()).ok_or_else(|| {
            adapter_error_status_with_query_id(
                Error::new(
                    ErrorCode::ObjectNotFound,
                    "datafusion_nexus.server.flight.do_put_prepared_statement_update.resolve",
                    format!("unknown prepared statement handle: {handle}"),
                ),
                &query_id,
            )
        })?;

        self.execute_update_sql(query_id, sql, UpdateStatementKind::PreparedStatement).await
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        request: Request<arrow_flight::Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        self.check_auth(&request)?;
        let query_id = Uuid::new_v4().hyphenated().to_string();
        if query.transaction_id.is_some() {
            return Err(flight_sql_transactions_unsupported_status(&query_id));
        }
        if self.prepared_statements.len() >= MAX_PREPARED_STATEMENTS {
            let error = Error::new(
                ErrorCode::ServiceOverloaded,
                "datafusion_nexus.server.flight.do_action_create_prepared_statement.admission",
                format!("too many live prepared statements (limit {MAX_PREPARED_STATEMENTS})"),
            )
            .with_fact("limit", MAX_PREPARED_STATEMENTS);
            let operation = error.operation();
            return Err(status_with_query_id(
                datafusion_error_to_status_for_operation(error.into(), operation),
                &query_id,
            ));
        }
        let sql = query.query;
        // Plan logically to derive the result-set schema the client needs
        // up-front. This does *not* run the physical optimizer rule and does
        // not consume an admission permit — the real plan / admission /
        // ticket happen later in `get_flight_info_prepared_statement` when
        // the client actually executes this prepared statement.
        let state = self.ctx.state();
        let logical_plan = tokio::select! {
            result = state.create_logical_plan(&sql) => {
                result.map_err(|error| {
                    status_with_query_id(
                        datafusion_error_to_status_for_operation(
                            error,
                            "datafusion_nexus.server.flight.do_action_create_prepared_statement.create_logical_plan",
                        ),
                        &query_id,
                    )
                })?
            },
            _ = self.shutdown_token.cancelled() => {
                return Err(status_with_query_id(Status::cancelled("server shutting down"), &query_id));
            }
        };
        if let Err(status) = reject_parameterized_plan(
            &logical_plan,
            "datafusion_nexus.server.flight.do_action_create_prepared_statement.parameter_binding",
        ) {
            return Err(status_with_query_id(status, &query_id));
        }
        let dataset_schema = self
            .dataset_schema_for_prepared_statement(&logical_plan)
            .map_err(|status| status_with_query_id(status, &query_id))?;
        let dataset_schema_ipc =
            schema_to_ipc_bytes(&dataset_schema).map_err(|error| Status::internal(error.to_string()))?;

        let handle = Uuid::new_v4().hyphenated().to_string();
        self.prepared_statements.insert(handle.clone(), sql);
        tracing::info!(prepared_statement_handle = handle.as_str(), "created Flight SQL prepared statement");

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

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}
