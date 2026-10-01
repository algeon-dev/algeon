//! Composed native execution node for GPU-resident native fragments.
//!
//! Producer children emit device-resident handles into a shared native context.
//! The consumer plan then uses those handles as GPU inputs and materializes the
//! final result back to host Arrow for DataFusion.

use std::sync::Arc;

use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, SendableRecordBatchStream},
};
use query_engine::{
    admission::{AttemptCapabilityEvidence, AttemptExecutionScope, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput},
    plan::{DeferredGraphExecutionScopeIdentity, QueryPlan},
    runtime::{
        ExecutionContext,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
};

use super::{
    AlgeonNativeExec, GpuResidentProducer,
    admission::attempt_execution_required_error,
    common::{
        NATIVE_ALGEON_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, mark_native_runtime_error,
        native_host_output_stream, native_query_runtime_error, run_gpu_execution_task,
    },
    config::AlgeonNativeExecutionConfig,
    execute_gpu_resident_producer_with_context, gpu_resident_producer, gpu_resident_producer_input_name,
    metrics::AlgeonNativeMetrics,
    prepare_gpu_resident_producer_for_execution,
    producer::prepared_gpu_resident_producer_admission_tickets,
};
use crate::{
    error::{Error, ErrorCode, Result},
    planner::native::Invocation,
};

struct CompletedComposedAdmissionEvidence {
    execution_config: AlgeonNativeExecutionConfig,
}

struct PreparedProducerCapabilityEvidence {
    aggregate: AttemptCapabilityEvidence,
    output_scope: AttemptExecutionScope,
    deferred_output_identity: Option<DeferredGraphExecutionScopeIdentity>,
}

fn prepared_producer_capability_evidence(
    input: &dyn ExecutionPlan,
    tickets: &[Arc<super::admission::QueryAdmissionTicket>],
) -> QueryEngineResult<PreparedProducerCapabilityEvidence> {
    let producer = gpu_resident_producer(input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InternalInvariantViolation,
            "algeon_datafusion.native_composed.admission_evidence",
            "a prepared composed input is not a GPU-resident producer",
        )
        .with_fact("reason", "composed_admission_producer_missing")
    })?;
    let output_evidence = producer.gpu_result_capability_evidence().map_err(|error| {
        QueryError::new(
            QueryErrorCode::InternalInvariantViolation,
            "algeon_datafusion.native_composed.admission_evidence",
            "a prepared composed producer did not expose its output capability evidence",
        )
        .with_fact("reason", "composed_admission_producer_evidence_missing")
        .with_source(error)
    })?;
    let output_scope = output_evidence.execution_scopes().last().cloned().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InternalInvariantViolation,
            "algeon_datafusion.native_composed.admission_evidence",
            "a prepared composed producer's evidence has no execution scope to bind an occurrence to",
        )
        .with_fact("reason", "composed_admission_producer_scope_missing")
    })?;
    let deferred_output_identity =
        output_evidence.deferred_graph_identity_for_scope(output_scope.execution_scope_fingerprint());
    let mut evidence = tickets
        .iter()
        .filter_map(|ticket| ticket.attempt_capability_evidence_for_rebind())
        .map(AttemptCapabilityEvidence::without_execution_manifest)
        .collect::<Vec<_>>();
    evidence.push(output_evidence);
    Ok(PreparedProducerCapabilityEvidence {
        aggregate: AttemptCapabilityEvidence::aggregate(evidence)?,
        output_scope,
        deferred_output_identity,
    })
}

/// Issues the one engine-owned evidence for a prepared composed GPU island.
///
/// The caller supplies the exact consumer plan that will run and the
/// top-level producer occurrences in execution order. Nested producer
/// fragments stay behind each top-level evidence; they are not flattened
/// into a second occurrence manifest.
#[cfg(feature = "cugraph")]
pub(crate) fn composed_admission_capability_evidence(
    consumer_plan: &QueryPlan,
    prepared_gpu_inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<AttemptCapabilityEvidence> {
    resolve_composed_admission_capability_evidence(consumer_plan, prepared_gpu_inputs)
        .map_err(|error| prepared_composed_admission_error(error, consumer_plan, None))
}

pub(super) fn resolve_composed_admission_capability_evidence(
    consumer_plan: &QueryPlan,
    prepared_gpu_inputs: &[Arc<dyn ExecutionPlan>],
) -> QueryEngineResult<AttemptCapabilityEvidence> {
    let mut producer_inputs = Vec::with_capacity(prepared_gpu_inputs.len());
    for input in prepared_gpu_inputs {
        let input_name = gpu_resident_producer_input_name(input.as_ref()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "algeon_datafusion.native_composed.admission_evidence",
                "a prepared composed producer is missing its stable runtime occurrence identity",
            )
            .with_fact("reason", "composed_admission_occurrence_missing")
        })?;
        producer_inputs.push((input_name, input.as_ref()));
    }
    producer_bound_capability_evidence(consumer_plan, producer_inputs)
}

/// Aggregates a consumer plan's evidence with the evidence of every
/// GPU-resident producer it binds, and records each producer output as an
/// occurrence on the consumer scope.
///
/// Each occurrence id is the key under which the consumer's
/// `GpuResidentInputMap` holds that producer's output.
pub(crate) fn producer_bound_capability_evidence<'a>(
    consumer_plan: &QueryPlan,
    producer_inputs: impl IntoIterator<Item = (String, &'a dyn ExecutionPlan)>,
) -> QueryEngineResult<AttemptCapabilityEvidence> {
    let mut consumer_evidence = AttemptCapabilityEvidence::for_query_plan(consumer_plan);
    let mut evidence = Vec::new();
    for (input_name, input) in producer_inputs {
        let tickets = prepared_gpu_resident_producer_admission_tickets(input);
        let producer_evidence = prepared_producer_capability_evidence(input, &tickets)?;
        consumer_evidence.push_producer_occurrence(
            AttemptProducerOccurrence::new(
                input_name,
                producer_evidence.output_scope.execution_scope_fingerprint(),
                producer_evidence.output_scope.output_schema_fingerprint(),
            )
            .with_deferred_graph_identity(producer_evidence.deferred_output_identity),
        )?;
        evidence.push(producer_evidence.aggregate);
    }
    evidence.push(consumer_evidence);
    AttemptCapabilityEvidence::aggregate(evidence)
}

fn prepared_composed_admission_error(
    mut error: QueryError,
    consumer_plan: &QueryPlan,
    execution_config: Option<&AlgeonNativeExecutionConfig>,
) -> Error {
    error = error
        .with_fact("stage", "prepared_composed_admission")
        .with_fact("candidate_plan_node_id", consumer_plan.root_id());
    if let Some(execution_config) = execution_config {
        if let Some(shape) = execution_config.plan_task_mapping_datafusion_shape() {
            error = error.with_fact("candidate_shape", shape);
        }
        if let Some(query_id) = execution_config.query_id() {
            error = error.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = execution_config.trace_id() {
            error = error.with_trace_id(trace_id.clone());
        }
    }
    Error::from_native_with_blocking_diagnostics(
        "algeon_datafusion.native_composed.admission_evidence",
        "failed to derive the complete engine-owned composed admission bound",
        error,
    )
}

/// Replaces provisional producer tickets with the one evidence for the complete
/// composed GPU island. Child preparation is the only phase that can resolve
/// footer-bound graph evidence, while the outer consumer is the only owner of
/// its aggregate/filter capability requirements.
fn complete_composed_admission_evidence(
    execution_config: AlgeonNativeExecutionConfig,
    consumer_plan: &QueryPlan,
    prepared_gpu_inputs: &[Arc<dyn ExecutionPlan>],
    task_context: &TaskContext,
) -> Result<CompletedComposedAdmissionEvidence> {
    let mut producer_tickets = Vec::new();
    for input in prepared_gpu_inputs {
        for ticket in prepared_gpu_resident_producer_admission_tickets(input.as_ref()) {
            if !producer_tickets.iter().any(|existing| Arc::ptr_eq(existing, &ticket)) {
                producer_tickets.push(ticket);
            }
        }
    }

    let canonical_ticket = execution_config.query_admission_ticket_for_task_context(task_context);
    let source = execution_config
        .query_admission_source()
        .or_else(|| producer_tickets.first().map(|ticket| ticket.source()))
        .or_else(|| canonical_ticket.as_ref().map(|ticket| ticket.source()));
    let Some(source) = source else {
        // A direct composed caller without an admission authority remains
        // unchanged; the runtime context will reject it before producer work.
        resolve_composed_admission_capability_evidence(consumer_plan, prepared_gpu_inputs)
            .map_err(|error| prepared_composed_admission_error(error, consumer_plan, Some(&execution_config)))?;
        return Ok(CompletedComposedAdmissionEvidence { execution_config });
    };
    if producer_tickets.iter().any(|ticket| !source.same_backend(&ticket.source())) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_composed.admission_evidence",
            "prepared GPU producers belong to different admission backends",
        )
        .with_fact("reason", "composed_producer_admission_backend_mismatch"));
    }

    if let Some(active_ticket) = canonical_ticket.as_ref()
        && !source.same_backend(&active_ticket.source())
    {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_composed.admission_evidence",
            "the composed root and its active admission ticket belong to different backends",
        )
        .with_fact("reason", "composed_active_admission_backend_mismatch"));
    }

    if let Some((active_ticket, active_evidence)) = canonical_ticket
        .as_ref()
        .and_then(|ticket| ticket.attempt_capability_evidence_for_rebind().map(|evidence| (ticket, evidence)))
    {
        let occurrence_ids = prepared_gpu_inputs
            .iter()
            .filter_map(|input| gpu_resident_producer_input_name(input.as_ref()))
            .collect::<Vec<_>>();
        if occurrence_ids.len() == prepared_gpu_inputs.len()
            && active_evidence.validate_covers_plan(consumer_plan).is_ok()
            && active_evidence.validate_composed_occurrences(consumer_plan, &occurrence_ids).is_ok()
        {
            return Ok(CompletedComposedAdmissionEvidence {
                execution_config: execution_config.with_existing_query_admission_ticket(Arc::clone(active_ticket)),
            });
        }
    }

    let complete_evidence = resolve_composed_admission_capability_evidence(consumer_plan, prepared_gpu_inputs)
        .map_err(|error| prepared_composed_admission_error(error, consumer_plan, Some(&execution_config)))?;

    // The admission stream has already selected its ambient ticket before a
    // footer-bound producer can finish preparation. Rebinding a new ticket
    // here would create a second query-service lifecycle whose release can
    // revoke the deferred host cursor before it reaches EOF. Update that
    // still-unadmitted ambient ticket in place.
    if let Some(active_ticket) = canonical_ticket {
        active_ticket.bind_fragment_evidence([complete_evidence]).map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_composed.admission_evidence",
                "failed to bind the composed fragment to the active admission ticket",
                error,
            )
        })?;
        return Ok(CompletedComposedAdmissionEvidence {
            execution_config: execution_config.with_existing_query_admission_ticket(active_ticket),
        });
    }

    if producer_tickets.is_empty() {
        return Ok(CompletedComposedAdmissionEvidence { execution_config });
    }
    let ticket = match Invocation::of_task(task_context)
        .and_then(|invocation| invocation.attempt_admission_ticket(source, Some(complete_evidence.clone())))
    {
        Some((ticket, true)) => ticket,
        // Another fragment of this invocation minted the ticket first.
        Some((ticket, false)) => {
            ticket.bind_fragment_evidence([complete_evidence]).map_err(|error| {
                Error::from_native(
                    "algeon_datafusion.native_composed.admission_evidence",
                    "failed to bind the composed fragment to the invocation admission ticket",
                    error,
                )
            })?;
            ticket
        },
        None => {
            return Err(attempt_execution_required_error("algeon_datafusion.native_composed.admission_evidence"));
        },
    };
    Ok(CompletedComposedAdmissionEvidence {
        execution_config: execution_config.with_existing_query_admission_ticket(ticket),
    })
}

impl AlgeonNativeExec {
    /// Streams the host result of a unit that reads GPU-resident producers.
    pub(super) fn execute_host_composed(
        &self,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed.execute",
                "native composed cudf execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let admitted = Arc::clone(&self.admitted);
        let execution_config = self.execution_config.clone();
        let account_memory_pool = execution_config.host_output_memory_pool_accounting();
        let gpu_inputs = self.gpu_inputs.clone();
        let metrics = AlgeonNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        )
        .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
        let output_schema = Arc::clone(&schema);
        let output = native_host_output_stream(
            NATIVE_ALGEON_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                let admitted = super::lakehouse::resolve_admitted_plan(
                    execution_config.lakehouse_footer_pruning(),
                    &admitted,
                    task_context.as_ref(),
                )
                .await?;
                let mut prepared_gpu_inputs = Vec::with_capacity(gpu_inputs.len());
                for input in gpu_inputs {
                    prepared_gpu_inputs
                        .push(prepare_gpu_resident_producer_for_execution(input, Arc::clone(&task_context)).await?);
                }
                let completed_admission = complete_composed_admission_evidence(
                    execution_config,
                    admitted.plan(),
                    &prepared_gpu_inputs,
                    task_context.as_ref(),
                )?;
                let execution_config = completed_admission.execution_config;
                let runtime_occurrence_ids = prepared_gpu_inputs
                    .iter()
                    .map(|input| {
                        gpu_resident_producer_input_name(input.as_ref()).ok_or_else(|| {
                            DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "algeon_datafusion.native_composed.execute",
                                    "the prepared composed producer is missing its stable runtime occurrence identity",
                                )
                                .with_fact("reason", "composed_admission_occurrence_missing"),
                            )
                        })
                    })
                    .collect::<DataFusionResult<Vec<_>>>()?;
                // Build one native context for all producer fragments and the consumer so device-resident
                // handles are compatible across the composed execution.
                let context_lease = execution_config
                    .execution_context_async(Arc::clone(&task_context))
                    .await
                    .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
                if let Some(admission_facts) = context_lease.admission_facts() {
                    metrics.record_admission_facts(admission_facts);
                }
                let admitted_composed_projection = context_lease
                    .context()
                    .validate_admitted_composed_attempt(admitted.plan(), &runtime_occurrence_ids)
                    .map_err(|error| {
                        DataFusionError::from(mark_native_runtime_error(
                            Error::from_native_with_blocking_diagnostics(
                                "algeon_datafusion.native_composed.admission",
                                "the admitted composed bound does not match runtime execution",
                                error,
                            ),
                            "composed_admission",
                        ))
                    })?;
                let ordered_runtime_occurrence_ids =
                    admitted_composed_projection.as_ref().map(|_| runtime_occurrence_ids.clone());
                let attempt_ticket = execution_config.query_admission_ticket_for_task_context(task_context.as_ref());
                if let Some(projection) = admitted_composed_projection
                    && let Some(ticket) = &attempt_ticket
                {
                    ticket.record_attempt_composed_schedule_projection(projection).map_err(|error| {
                        DataFusionError::from(mark_native_runtime_error(
                            Error::from_native(
                                "algeon_datafusion.native_composed.admission",
                                "failed to attach the admitted ordered bound to the attempt lifecycle",
                                error,
                            ),
                            "composed_admission",
                        ))
                    })?;
                }
                let gpu_executor = context_lease.context().gpu_executor().map_err(|error| {
                    DataFusionError::from(mark_native_runtime_error(
                        Error::from_native(
                            "algeon_datafusion.native_composed.gpu_execution_pool",
                            "query service closed before composed native execution acquired its GPU pool",
                            error,
                        ),
                        "gpu_execution_pool",
                    ))
                })?;
                let observation = context_lease.context().runtime_observation().clone();
                if let Some(ticket) = &attempt_ticket {
                    ticket.record_attempt_gpu_execution_started();
                }
                let runtime_attempt_ticket = attempt_ticket.clone();
                let cursor_executor = gpu_executor.clone();
                let cursor_observation = observation.clone();
                run_gpu_execution_task(
                    gpu_executor,
                    observation,
                    "algeon_datafusion.native_composed.gpu_execution_pool",
                    move |gpu_stream_scope| async move {
                        let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                        let execution_task_context =
                            context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                        let mut input_map = GpuResidentInputMap::new();
                        for (input_index, child) in prepared_gpu_inputs.into_iter().enumerate() {
                            let (input_name, handle) = execute_gpu_resident_producer_with_context(
                                child.as_ref(),
                                &context,
                                partition,
                                Arc::clone(&execution_task_context),
                            )
                            .await?;
                            if let Some(expected_occurrence_ids) = &ordered_runtime_occurrence_ids {
                                let expected_occurrence_id = &expected_occurrence_ids[input_index];
                                if input_name != *expected_occurrence_id {
                                    return Err(DataFusionError::from(
                                        Error::new(
                                            ErrorCode::InvalidExecutionPlan,
                                            "algeon_datafusion.native_composed.execute",
                                            "a composed producer returned a runtime occurrence different from the admitted schedule",
                                        )
                                        .with_fact("reason", "composed_admission_occurrence_order_mismatch")
                                        .with_fact("scheduled_occurrence_id", expected_occurrence_id)
                                        .with_fact("runtime_occurrence_id", input_name),
                                    ));
                                }
                            }
                            input_map.insert(input_name, handle);
                        }

                        let context = context.with_admitted_gpu_inputs(admitted.plan(), &input_map).map_err(|error| {
                            DataFusionError::from(mark_native_runtime_error(
                                Error::from_native_with_blocking_diagnostics(
                                    "algeon_datafusion.native_composed.admission",
                                    "failed to refine the admitted composed bound with resident evidence",
                                    error,
                                ),
                                "composed_admission",
                            ))
                        })?;
                        if let Some(projection) = context
                            .validate_admitted_composed_attempt(admitted.plan(), &runtime_occurrence_ids)
                            .map_err(|error| {
                                DataFusionError::from(mark_native_runtime_error(
                                    Error::from_native(
                                        "algeon_datafusion.native_composed.admission",
                                        "the refined composed bound does not match runtime execution",
                                        error,
                                    ),
                                    "composed_admission",
                                ))
                            })?
                            && let Some(ticket) = &runtime_attempt_ticket
                        {
                            ticket.record_attempt_composed_schedule_projection(projection).map_err(|error| {
                                DataFusionError::from(mark_native_runtime_error(
                                    Error::from_native(
                                        "algeon_datafusion.native_composed.admission",
                                        "failed to attach the refined ordered bound to the attempt lifecycle",
                                        error,
                                    ),
                                    "composed_admission",
                                ))
                            })?;
                        }

                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        // The consumer query starts after all GPU-resident producer handles are registered. The
                        // native attempt is terminal after this point and is not replayed through DataFusion.
                        let outcome = execute_query_plan(
                            &context,
                            QueryExecutionRequest::admitted(&admitted).with_gpu_inputs(&input_map),
                        );
                        let (output, deferred_metrics) = metrics.finalize_query_outcome_for_deferred_host(outcome, context_lease).map_err(|error| {
                            DataFusionError::from(native_query_runtime_error(
                                "algeon_datafusion.native_composed.execute",
                                "algeon native composed query execution failed",
                                error,
                            ))
                        })?;
                        let QueryOutput::GpuResident(handle) = output else {
                            return Err(DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "algeon_datafusion.native_composed.execute",
                                    "deferred composed host execution did not produce a GPU-resident cursor input",
                                )
                                .with_fact("result_sink", "host_arrow"),
                            ));
                        };
                        let cursor = handle
                            .into_host_batch_cursor_with_host_schema(
                                output_schema,
                                context.cancellation().clone(),
                                HostBatchCursorConfig::default(),
                            )
                            .map_err(|error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "algeon_datafusion.native_composed.host_batch_cursor",
                                    "failed to bind and create deferred composed host-output cursor",
                                    error,
                                ))
                            })?;
                        Ok::<NativeHostOutputCursor, DataFusionError>(NativeHostOutputCursor::new(
                            cursor,
                            cursor_executor,
                            cursor_observation,
                            deferred_metrics,
                        ))
                    },
                )
                .await
            },
        );

        Ok(output)
    }

    pub(super) fn composed_prepared_admission_tickets(&self) -> Vec<Arc<super::admission::QueryAdmissionTicket>> {
        let mut tickets = self.execution_config.query_admission_ticket().into_iter().collect::<Vec<_>>();
        for child in &self.gpu_inputs {
            for ticket in prepared_gpu_resident_producer_admission_tickets(child.as_ref()) {
                if !tickets.iter().any(|existing| Arc::ptr_eq(existing, &ticket)) {
                    tickets.push(ticket);
                }
            }
        }
        tickets
    }

    /// Executes a GPU-resident unit's producers and plan inside the
    /// composed root's shared native context, after
    /// [`GpuResidentProducer::prepare_execution_plan`] resolved them.
    pub(super) async fn execute_composed_gpu_result(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                "composed GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        if !self.prepared {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                "composed GPU-resident producer was not prepared before native context acquisition",
            )
            .with_log_safe_fact("producer_state", "not_prepared")
            .into());
        }

        // Producers run only after the admitted composed bound matches
        // this consumer and the order of the occurrences it binds.
        let runtime_occurrence_ids = self
            .gpu_inputs
            .iter()
            .map(|input| {
                gpu_resident_producer_input_name(input.as_ref()).ok_or_else(|| {
                    DataFusionError::from(
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                            "a composed GPU-resident producer input is missing its occurrence identity",
                        )
                        .with_fact("reason", "composed_admission_occurrence_missing"),
                    )
                })
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        native_context.validate_admitted_composed_attempt(self.admitted.plan(), &runtime_occurrence_ids).map_err(
            |error| {
                DataFusionError::from(Error::from_native(
                    "algeon_datafusion.native_composed_gpu_producer.admission",
                    "the admitted composed bound does not match runtime execution",
                    error,
                ))
            },
        )?;

        let mut input_map = GpuResidentInputMap::new();
        for child in &self.gpu_inputs {
            let (input_name, handle) = execute_gpu_resident_producer_with_context(
                child.as_ref(),
                native_context,
                partition,
                Arc::clone(&task_context),
            )
            .await?;
            input_map.insert(input_name, handle);
        }

        let metrics = AlgeonNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        );
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let outcome = execute_query_plan(
            native_context,
            QueryExecutionRequest::admitted(&self.admitted).with_gpu_inputs(&input_map),
        );
        let output = metrics.finalize_query_outcome(outcome).map_err(|error| {
            DataFusionError::from(native_query_runtime_error(
                "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                "algeon composed GPU-resident relational execution failed",
                error,
            ))
        })?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                "composed GPU-resident producer materialized a host result",
            )
            .with_fact("result_sink", "host_arrow")
            .into());
        };
        Ok((self.gpu_result_input_name().to_string(), handle))
    }
}

#[cfg(test)]
pub(crate) mod tests;
