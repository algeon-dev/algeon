use super::*;

/// Keeps a Flight-owned cancellation path ahead of the attempt stream's drop
/// implementation. `take_until` and client disconnects may discard this
/// wrapper at any poll boundary; its `Drop` cancels the attempt before Rust
/// drops the inner [`AttemptStream`], so terminalization observes
/// `Cancelled` rather than an unrelated client-drop outcome.
struct FlightExecutionBody {
    inner: Option<SendableRecordBatchStream>,
    cancellation: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
    native_cancellation: ExecutionCancellation,
    attempt_cancellation: Option<nexus_query_engine::attempt::AttemptCancellation>,
    delivery_started: Arc<std::sync::atomic::AtomicBool>,
    terminal: Arc<CoordinatorExecutionTerminal>,
    native_execution_span: tracing::Span,
}

impl FlightExecutionBody {
    fn new(
        inner: SendableRecordBatchStream,
        flight_token: CancellationToken,
        native_cancellation: ExecutionCancellation,
        attempt_cancellation: Option<nexus_query_engine::attempt::AttemptCancellation>,
        delivery_started: Arc<std::sync::atomic::AtomicBool>,
        terminal: Arc<CoordinatorExecutionTerminal>,
        native_execution_span: tracing::Span,
    ) -> Self {
        Self {
            inner: Some(inner),
            cancellation: Box::pin(flight_token.cancelled_owned()),
            native_cancellation,
            attempt_cancellation,
            delivery_started,
            terminal,
            native_execution_span,
        }
    }
}

impl Stream for FlightExecutionBody {
    type Item = datafusion::common::Result<arrow::record_batch::RecordBatch>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let _native_enter = this.native_execution_span.enter();
        if this.cancellation.as_mut().poll(context).is_ready() {
            this.native_cancellation.cancel();
            if let Some(attempt_cancellation) = &this.attempt_cancellation {
                attempt_cancellation.cancel();
            }
            // `take_until` retains its inner stream after its stop future
            // resolves. Release ours now so AttemptStream terminalizes and
            // the report handoff completes before the cancellation tail can
            // release the coordinator execution lease.
            this.inner = None;
            return std::task::Poll::Ready(None);
        }
        let poll = this.inner.as_mut().map_or(std::task::Poll::Ready(None), |stream| {
            this.delivery_started.store(true, std::sync::atomic::Ordering::Release);
            stream.as_mut().poll_next(context)
        });
        if matches!(&poll, std::task::Poll::Ready(None)) {
            // Reaching EOF is the execution-resource boundary. Do not retain
            // the exhausted DataFusion stream (and its admission ticket)
            // until tonic later drops the response wrapper. This is also the
            // only reliable Flight success boundary: an empty synthetic tail
            // may not be polled after stream encoding finishes, which would
            // otherwise drop the coordinator lease as `Dropped`.
            this.inner = None;
            this.terminal.finish_success("stream");
        }
        poll
    }
}

impl Drop for FlightExecutionBody {
    fn drop(&mut self) {
        if self.inner.is_some() {
            self.native_cancellation.cancel();
            if let Some(attempt_cancellation) = &self.attempt_cancellation {
                attempt_cancellation.cancel();
            }
        }
    }
}

/// Owns the diagnostic-only half of an `EXPLAIN GPU` Flight response.
///
/// Unlike an ordinary statement, coverage validation produces a synthetic
/// batch and never transfers the attempt owner into a physical result stream.
/// Keeping that owner beside the Flight body preserves the same terminal
/// contract: EOF succeeds, cancellation cancels, and a client drop commits
/// before its coordinator lease can be released.
struct ExplainGpuExecutionBody {
    batch: Option<arrow::record_batch::RecordBatch>,
    attempt: Option<crate::attempt::NexusAttempt>,
    guard: QueryExecutionGuard,
    finished: bool,
}

impl ExplainGpuExecutionBody {
    fn new(
        batch: arrow::record_batch::RecordBatch,
        attempt: Option<crate::attempt::NexusAttempt>,
        guard: QueryExecutionGuard,
    ) -> Self {
        Self { batch: Some(batch), attempt, guard, finished: false }
    }

    fn cancel_attempt(&mut self) {
        self.guard.native_cancellation.cancel();
        if let Some(attempt) = self.attempt.take() {
            attempt.cancel_before_stream();
        }
    }

    fn cancelled_status(&self) -> (Status, CapturedError) {
        let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.guard.shutdown_token));
        (
            status_with_query_id_and_correlation_id(status, &self.guard.query_id, self.guard.correlation_id.as_deref()),
            captured,
        )
    }
}

impl Stream for ExplainGpuExecutionBody {
    type Item = Result<arrow::record_batch::RecordBatch, FlightError>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return std::task::Poll::Ready(None);
        }
        if this.guard.flight_token.is_cancelled() {
            this.cancel_attempt();
            this.batch = None;
            this.finished = true;
            let (status, captured) = this.cancelled_status();
            this.guard.terminal.finish_captured("stream", &status, captured);
            return std::task::Poll::Ready(Some(Err(FlightError::from(status))));
        }
        if let Some(batch) = this.batch.take() {
            this.guard.delivery_started.store(true, std::sync::atomic::Ordering::Release);
            return std::task::Poll::Ready(Some(Ok(batch)));
        }
        if let Some(attempt) = this.attempt.take() {
            attempt.complete_without_result_stream();
        }
        this.finished = true;
        this.guard.terminal.finish_success("stream");
        std::task::Poll::Ready(None)
    }
}

impl Drop for ExplainGpuExecutionBody {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if std::thread::panicking() {
            // The attempt owner observes the active panic while it is dropped
            // and records the matching panicked terminal report.
            drop(self.attempt.take());
            self.guard.terminal.finish_panicked("stream");
            return;
        }
        // Match the ordinary Flight stream: a client drop is cooperative
        // cancellation. Dropping the attempt before the guard preserves the
        // sealed-attempt handoff, while the guard owns the one delivery
        // classification policy: before the first batch it is dropped; after
        // delivery starts it is a captured cancellation failure.
        self.cancel_attempt();
    }
}

impl DatafusionNexusFlightSqlService {
    fn empty_ddl_response(
        &self,
        query_id: &str,
        correlation_id: Option<&str>,
    ) -> Response<<Self as FlightService>::DoGetStream> {
        let query_id_for_encode = query_id.to_string();
        let correlation_id_for_encode = correlation_id.map(str::to_string);
        let stream = encode_flight_data_stream(
            Arc::new(Schema::empty()),
            futures::stream::empty::<Result<arrow::record_batch::RecordBatch, FlightError>>(),
        )
        .map_err(move |error| {
            status_with_query_id_and_correlation_id(
                arrow_flight_error_to_status_for_operation(
                    error,
                    "datafusion_nexus.server.flight.execute_ticket.empty_ddl_encode",
                    "failed to encode empty Flight SQL DDL result stream",
                ),
                &query_id_for_encode,
                correlation_id_for_encode.as_deref(),
            )
        });
        let mut response = Response::new(Box::pin(stream) as <Self as FlightService>::DoGetStream);
        attach_query_response_metadata(&mut response, query_id, correlation_id);
        response
    }

    pub(super) fn expire_stale_tickets(&self, now: Instant) {
        self.coordinator.maintenance_at(now);
    }

    pub(super) async fn execute_ticket(
        &self,
        query_id: String,
        request_correlation_id: Option<String>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        self.expire_stale_tickets(Instant::now());
        let (mut execution_lease, ticket) =
            self.coordinator.consume_ticket(&query_id, Instant::now()).map_err(|error| {
                let status = status_with_query_id_and_correlation_id(
                    adapter_error_status(
                        Error::new(
                            ErrorCode::InvalidRequest,
                            "datafusion_nexus.server.flight.execute_ticket.resolve",
                            "unknown, expired, or already consumed statement ticket",
                        )
                        .with_log_safe_fact("query_coordinator_error", error.kind().as_str()),
                    ),
                    &query_id,
                    request_correlation_id.as_deref(),
                );
                log_statement_status(&query_id, "ticket", Instant::now(), &status);
                status
            })?;

        // Derive from `shutdown_token` so cancellation cascades to statements
        // that race in between ticket consumption and cancellation registry insert.
        let token = self.shutdown_token.child_token();
        let CoordinatorTicketPayload {
            sql,
            correlation_id,
            prepared_handle,
            kind,
            native_cancellation,
            mut attempt,
            started_at,
            ..
        } = ticket;
        let ticket_kind = match StatementTicketKind::from_code(kind) {
            Ok(ticket_kind) => ticket_kind,
            Err(status) => {
                let captured = CapturedError::from_status(&status);
                execution_lease.finish_captured("ticket", &status, captured);
                return Err(status);
            },
        };
        let correlation_id = correlation_id.or(request_correlation_id);
        self.coordinator.record_correlation_id(&query_id, correlation_id.as_deref());
        let query_span = Self::query_span(&query_id, correlation_id.as_deref(), "do_get_statement");
        {
            let _query_enter = query_span.enter();
            tracing::info!(
                query_id,
                correlation_id = tracing_correlation_id(correlation_id.as_deref()),
                "executing Flight SQL statement ticket"
            );
        }
        let coordinator_capture = if attempt.is_none() {
            let observation = match execution_lease.observation_handle() {
                Ok(observation) => observation,
                Err(error) => {
                    let (status, captured) = adapter_error_to_status_and_capture(
                        error.adapter_error("datafusion_nexus.server.flight.execute_ticket.observation_handle"),
                    );
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    execution_lease.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            };
            Some((observation, execution_lease.activity_factory()))
        } else {
            None
        };
        let output_counters = execution_lease.hot_counters();
        let attempt_cancellation = attempt.as_ref().map(crate::attempt::NexusAttempt::cancellation);
        self.cancellations.insert(
            query_id.clone(),
            token.clone(),
            native_cancellation.clone(),
            attempt_cancellation.clone(),
        );
        let attempt_cancellation_for_stream = attempt_cancellation.clone();
        let coordinator_terminal = CoordinatorExecutionTerminal::new_with_attempt_report(
            execution_lease,
            query_id.clone(),
            started_at,
            attempt.is_some(),
        );
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(
                attempt,
                CoordinatorReportHandoffTerminal::execution(Arc::clone(&coordinator_terminal)),
            );
        }
        // Construct the guard before any post-ticket branch so cancellation
        // registration and observability cleanup stay tied together after the
        // ticket leaves the coordinator's ticketed state.
        let guard = QueryExecutionGuard {
            query_id: query_id.clone(),
            correlation_id: correlation_id.clone(),
            cancellations: Arc::clone(&self.cancellations),
            flight_token: token.clone(),
            native_cancellation: native_cancellation.clone(),
            attempt_cancellation,
            shutdown_token: self.shutdown_token.clone(),
            delivery_started: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            terminal: Arc::clone(&coordinator_terminal),
        };
        if token.is_cancelled() {
            native_cancellation.cancel();
            if let Some(attempt) = attempt.as_ref() {
                attempt.cancel_before_stream();
            }
            let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            coordinator_terminal.finish_captured("planning", &status, captured);
            return Err(status);
        }

        if let Some(output_mode) = ticket_kind.explain_gpu_output_mode() {
            // `EXPLAIN GPU` is a plan-time diagnostic. It uses the same
            // per-query context, coordinator lease, and cancellation wiring as
            // ordinary statements, but deliberately never executes a physical
            // plan or acquires GPU admission.
            let planning = match coordinator_capture.as_ref() {
                Some((observation, activity_factory)) => self.coordinated_query_planning(
                    &query_id,
                    correlation_id.as_deref(),
                    native_cancellation.clone(),
                    observation.clone(),
                    Arc::clone(activity_factory),
                ),
                None => self.attempt_query_planning(correlation_id.as_deref(), native_cancellation.clone()),
            };
            let query_ctx = match attempt.as_ref() {
                Some(attempt) => match attempt.planning_context(&planning.ctx) {
                    Ok(context) => context,
                    Err(error) => {
                        let (status, captured) =
                            datafusion_error_to_status_and_capture(error, ticket_kind.sql_operation());
                        let status =
                            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                        coordinator_terminal.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                },
                None => planning.ctx.clone(),
            };
            let coverage = tokio::select! {
                result = validate_query(&query_ctx, &sql)
                    .instrument(Self::query_stage_span(
                        Some(&query_span),
                        &query_id,
                        correlation_id.as_deref(),
                        "planning",
                    )) => match result {
                    Ok(coverage) => coverage,
                    Err(error) => {
                        if let Some(attempt) = attempt.as_ref() {
                            attempt.planning_failed_adapter(&error);
                        }
                        let (status, captured) = adapter_error_to_status_and_capture(error);
                        let status = status_with_query_id_and_correlation_id(
                            status,
                            &query_id,
                            correlation_id.as_deref(),
                        );
                        coordinator_terminal.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                },
                _ = token.cancelled() => {
                    native_cancellation.cancel();
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.cancel_before_stream();
                    }
                    let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                    let status = status_with_query_id_and_correlation_id(
                        status,
                        &query_id,
                        correlation_id.as_deref(),
                    );
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            };
            let batch = match coverage_batch(&coverage, output_mode, Some(&query_id)) {
                Ok(batch) => batch,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    let (status, captured) = datafusion_error_to_status_and_capture(
                        error,
                        "datafusion_nexus.server.flight.do_get_statement.explain_gpu.project",
                    );
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            };
            let output_counters_for_body = Arc::clone(&output_counters);
            let stream = ExplainGpuExecutionBody::new(batch, attempt.take(), guard).map(move |result| {
                result.inspect(|batch| {
                    output_counters_for_body.record_output(batch.num_rows(), batch.get_array_memory_size());
                })
            });
            let terminal_for_encode = Arc::clone(&coordinator_terminal);
            let query_id_for_encode = query_id.clone();
            let correlation_id_for_encode = correlation_id.clone();
            let stream = encode_flight_data_stream(coverage_schema(), stream).map_err(move |error| {
                let (status, captured) = arrow_flight_error_to_status_and_capture(
                    error,
                    "datafusion_nexus.server.flight.do_get_statement.explain_gpu.encode",
                    "failed to encode EXPLAIN GPU result stream",
                );
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id_for_encode,
                    correlation_id_for_encode.as_deref(),
                );
                terminal_for_encode.finish_captured("flight_encode", &status, captured);
                status
            });
            let mut response = Response::new(Box::pin(stream) as <Self as FlightService>::DoGetStream);
            attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
            return Ok(response);
        }

        // DataGrip-class clients re-run held prepared-statement handles. A
        // no-result-set DDL handle that already executed once is a no-op on
        // re-execution: return an empty result so the client's `execute()`
        // succeeds instead of surfacing DataFusion's "already exists" /
        // "not found" catalog error from the duplicate run.
        if let Some(handle) = &prepared_handle
            && self.prepared_statements.get(handle).is_some_and(|entry| entry.ddl_already_executed())
        {
            tracing::info!(
                query_id,
                prepared_statement_handle = handle.as_str(),
                "deduplicated repeated prepared DDL execution on the query channel; returning an empty result"
            );
            if let Some(attempt) = attempt.as_ref() {
                attempt.complete_without_result_stream();
            }
            coordinator_terminal.finish_success("stream");
            drop(guard);
            return Ok(self.empty_ddl_response(&query_id, correlation_id.as_deref()));
        }

        let planning = match coordinator_capture {
            Some((observation, activity_factory)) => self.coordinated_query_planning(
                &query_id,
                correlation_id.as_deref(),
                native_cancellation.clone(),
                observation,
                activity_factory,
            ),
            None => self.attempt_query_planning(correlation_id.as_deref(), native_cancellation.clone()),
        };
        let query_ctx = match attempt.as_ref() {
            Some(attempt) => match attempt.planning_context(&planning.ctx) {
                Ok(context) => context,
                Err(error) => {
                    let (status, captured) = datafusion_error_to_status_and_capture(error, ticket_kind.sql_operation());
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            },
            None => planning.ctx.clone(),
        };
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    let (status, captured) =
                        datafusion_error_to_status_and_capture(error, ticket_kind.sql_operation());
                    let status = status_with_query_id(status, &query_id);
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                if let Some(attempt) = attempt.as_ref() {
                    attempt.cancel_before_stream();
                }
                let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id,
                    correlation_id.as_deref(),
                );
                coordinator_terminal.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        if let Err(error) =
            reject_drop_object_type_mismatch(&query_ctx, &logical_plan, ticket_kind.sql_operation()).await
        {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed_adapter(&error);
            }
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            coordinator_terminal.finish_captured("planning", &status, captured);
            return Err(status);
        }
        let completes_eagerly_without_result_stream = matches!(&logical_plan, LogicalPlan::Ddl(_));
        let df = tokio::select! {
            result = query_ctx.execute_logical_plan(logical_plan)
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(df) => df,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    // DDL execution and logical-plan wrapping happen before
                    // the physical optimizer runs, so `last_report` is empty
                    // here. The leaf-guard helper no-ops on non-leaf-guard
                    let (status, captured) =
                        datafusion_error_to_status_and_capture(error, ticket_kind.sql_operation());
                    let status = status_with_query_id(status, &query_id);
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                if let Some(attempt) = attempt.as_ref() {
                    attempt.cancel_before_stream();
                }
                let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id,
                    correlation_id.as_deref(),
                );
                coordinator_terminal.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        // `execute_logical_plan` has run any DDL side effect by now, so record
        // the success: a re-run of this prepared handle short-circuits to an
        // empty result (`mark_executed` self-guards on `no_result_ddl`, so
        // query/DML handles are unaffected).
        if let Some(handle) = &prepared_handle
            && let Some(entry) = self.prepared_statements.get(handle)
        {
            entry.mark_executed();
        }
        if completes_eagerly_without_result_stream {
            // DataFusion executes DDL from `execute_logical_plan`; CTAS in
            // particular has already consumed its native result stream and
            // its attempt's sole finalizer. Planning the returned empty
            // DataFrame would manufacture a second native admission.
            if let Some(attempt) = attempt.as_ref() {
                attempt.complete_without_result_stream();
            }
            coordinator_terminal.finish_success("stream");
            coordinator_terminal.wait_finished().await;
            drop(guard);
            return Ok(self.empty_ddl_response(&query_id, correlation_id.as_deref()));
        }
        let schema = df.schema().as_arrow().clone();
        let logical_plan_tables = referenced_table_names(df.logical_plan());
        // Physical planning runs the retained native optimizer rule. Capture
        // the post-optimizer plan and retained diagnostics before execution so
        // query_stats can report both native rewrites and retained DataFusion baselines.
        let plan = tokio::select! {
            result = df.create_physical_plan()
                .instrument(Self::query_stage_span(
                    Some(&query_span),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                )) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    let diagnostics = planning.native_rule.last_planning_diagnostics();
                    log_native_planning_report_for_tables(
                        &query_id,
                        diagnostics.as_ref().map(|diagnostics| &diagnostics.report),
                        &logical_plan_tables,
                    );
                    self.coordinator.capture_planning_failure(
                        &query_id,
                        "physical_planning",
                        "physical_planning_failed",
                        error.to_string(),
                    );
                    let (status, captured) =
                        datafusion_error_to_status_and_capture(error, ticket_kind.execute_stream_operation());
                    let status = status_with_query_id(status, &query_id);
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("execute_stream", &status, captured);
                    return Err(status);
                },
            },
            _ = token.cancelled() => {
                native_cancellation.cancel();
                if let Some(attempt) = attempt.as_ref() {
                    attempt.cancel_before_stream();
                }
                let (status, captured) = adapter_error_to_status_and_capture(cancellation_error(&self.shutdown_token));
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id,
                    correlation_id.as_deref(),
                );
                coordinator_terminal.finish_captured("execute_stream", &status, captured);
                return Err(status);
            },
        };
        let diagnostics = planning.native_rule.last_planning_diagnostics();
        if let Some(attempt) = attempt.as_ref() {
            attempt.physical_planning_finished(plan.as_ref());
        }
        log_native_planning_report_for_tables(
            &query_id,
            diagnostics.as_ref().map(|diagnostics| &diagnostics.report),
            &logical_plan_tables,
        );
        self.coordinator.capture_planning(&query_id, diagnostics, Arc::clone(&plan));
        let native_execution_span =
            Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "native_execution");
        let stream_result: datafusion::common::Result<SendableRecordBatchStream> = match attempt.as_ref() {
            Some(attempt) => attempt
                .execute_physical_plan(Arc::clone(&plan), &query_ctx)
                .map(|stream| Box::pin(stream) as SendableRecordBatchStream),
            None => execute_physical_stream(Arc::clone(&plan), Arc::new(df.task_ctx())),
        };
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                let (status, captured) =
                    datafusion_error_to_status_and_capture(error, ticket_kind.execute_stream_operation());
                let status = status_with_query_id(status, &query_id);
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                coordinator_terminal.finish_captured("execute_stream", &status, captured);
                return Err(status);
            },
        };
        let stream = FlightExecutionBody::new(
            stream,
            token.clone(),
            native_cancellation.clone(),
            attempt_cancellation_for_stream.clone(),
            Arc::clone(&guard.delivery_started),
            Arc::clone(&coordinator_terminal),
            native_execution_span,
        );
        let cancel_token_for_tail = token.clone();
        let shutdown_token_for_tail = self.shutdown_token.clone();
        let query_id_for_tail = query_id.clone();
        let correlation_id_for_tail = correlation_id.clone();
        let terminal_for_body = Arc::clone(&coordinator_terminal);
        let query_id_for_body = query_id.clone();
        let correlation_id_for_body = correlation_id.clone();
        let output_counters_for_output = Arc::clone(&output_counters);
        let streaming_span =
            Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "result_streaming");
        let streaming_span_for_output = streaming_span.clone();
        let stream_operation = ticket_kind.stream_operation();
        // `take_until` ends cleanly on cancellation, so append a one-shot tail
        // that emits CANCELLED as the gRPC trailer when the token fired.
        let body = stream
            .map_ok(move |batch| {
                let _stream_enter = streaming_span_for_output.enter();
                output_counters_for_output.record_output(batch.num_rows(), batch.get_array_memory_size());
                batch
            })
            .map_err(move |error| {
                let _stream_enter = DatafusionNexusFlightSqlService::query_stage_span(
                    None,
                    &query_id_for_body,
                    correlation_id_for_body.as_deref(),
                    "result_streaming",
                )
                .entered();
                datafusion_error_to_flight_error(error, &query_id_for_body, &terminal_for_body, stream_operation)
            });
        let terminal_for_tail = Arc::clone(&coordinator_terminal);
        let tail = futures::stream::once(async move {
            cancel_token_for_tail.is_cancelled().then(|| {
                let (status, captured) =
                    adapter_error_to_status_and_capture(cancellation_error(&shutdown_token_for_tail));
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id_for_tail,
                    correlation_id_for_tail.as_deref(),
                );
                terminal_for_tail.finish_captured("stream", &status, captured);
                Err(FlightError::from(status))
            })
        })
        .filter_map(|opt| async move { opt });
        let terminal_for_end = Arc::clone(&coordinator_terminal);
        let end = futures::stream::once(async move {
            terminal_for_end.wait_finished().await;
            None::<Result<arrow::record_batch::RecordBatch, FlightError>>
        })
        .filter_map(|opt| async move { opt });
        let stream = body.chain(tail).chain(end).map(move |batch| {
            let _guard = &guard;
            batch
        });
        let mut stream = Box::pin(stream);
        let result_streaming_span_for_poll = streaming_span.clone();
        let stream = futures::stream::poll_fn(move |context| {
            let _stream_enter = result_streaming_span_for_poll.enter();
            stream.as_mut().poll_next(context)
        });
        let terminal_for_encode = Arc::clone(&coordinator_terminal);
        let query_id_for_encode = query_id.clone();
        let correlation_id_for_encode = correlation_id.clone();
        let stream = encode_flight_data_stream(Arc::new(schema), stream).map_err(move |error| {
            let (status, captured) = arrow_flight_error_to_status_and_capture(
                error,
                "datafusion_nexus.server.flight.execute_ticket.result_encode",
                "failed to encode Flight SQL result stream",
            );
            let status = status_with_query_id_and_correlation_id(
                status,
                &query_id_for_encode,
                correlation_id_for_encode.as_deref(),
            );
            terminal_for_encode.finish_captured("flight_encode", &status, captured);
            status
        });

        let mut response = Response::new(Box::pin(stream) as <Self as FlightService>::DoGetStream);
        attach_query_response_metadata(&mut response, &query_id, correlation_id.as_deref());
        Ok(response)
    }

    pub(super) fn cancel_from_flight_info(&self, raw_info: &[u8]) -> Result<bool, Status> {
        let info = FlightInfo::decode(raw_info).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained malformed FlightInfo bytes",
                "flight_info_decode",
                error,
            )
        })?;
        let Some(endpoint) = info.endpoint.first() else {
            return Ok(false);
        };
        let Some(ticket) = endpoint.ticket.as_ref() else {
            return Ok(false);
        };
        let any = arrow_flight::sql::Any::decode(&*ticket.ticket).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained a malformed ticket envelope",
                "ticket_any_decode",
                error,
            )
        })?;
        // The ticket may be either a `TicketStatementQuery` (issued by
        // `get_flight_info_statement`) or a `CommandPreparedStatementQuery`
        // (issued by `get_flight_info_prepared_statement`). Both encode the
        // per-execution `query_id` UTF-8 bytes in their handle field; once
        // extracted, the cancellation path is identical.
        let query_id_bytes = if let Some(statement_ticket) = any.unpack::<TicketStatementQuery>().map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action could not unpack statement ticket",
                "statement_ticket_unpack",
                error,
            )
        })? {
            statement_ticket.statement_handle
        } else if let Some(prepared_ticket) = any.unpack::<CommandPreparedStatementQuery>().map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action could not unpack prepared-statement ticket",
                "prepared_ticket_unpack",
                error,
            )
        })? {
            prepared_ticket.prepared_statement_handle
        } else {
            return Ok(false);
        };
        let query_id = std::str::from_utf8(&query_id_bytes).map_err(|error| {
            cancel_invalid_argument_status(
                "Flight SQL cancel action contained a non-UTF-8 query id",
                "query_id_utf8",
                error,
            )
        })?;
        let error = cancellation_error(&self.shutdown_token);
        if self.coordinator.cancel_ticket(query_id, CapturedError::from_adapter_error(&error)) {
            tracing::info!(query_id, outcome = "cancelled", stage = "ticket", "cancelled Flight SQL statement ticket");
            return Ok(true);
        }
        let cancelled = self.cancellations.cancel(query_id);
        Ok(cancelled)
    }
}

pub(super) fn cancel_invalid_argument_status(
    message: &'static str,
    stage: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> Status {
    adapter_error_status(
        Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.server.flight.cancel_query.decode", message)
            .with_fact("cancel_decode_stage", stage)
            .with_source(source),
    )
}
