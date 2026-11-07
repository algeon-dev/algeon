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
use query_engine::exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput};
use query_planning::{
    admission::{AttemptCapabilityEvidence, AttemptExecutionScope, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    plan::{DeferredGraphExecutionScopeIdentity, QueryPlan},
};
use query_runtime::runtime::{
    ExecutionContext,
    resident::{GpuResidentInputMap, GpuResultHandle},
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
    shared_resident::task_context_with_shared_resident_results,
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
/// The caller supplies the consumer plan that will run and the
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
                        // One result scope per island execution; producers and
                        // nested units read shared results through it.
                        let execution_task_context = task_context_with_shared_resident_results(
                            context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context)),
                            &prepared_gpu_inputs,
                        )?;
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

    /// Tickets of this unit and its producer subtree, deduplicated by pointer:
    /// one ticket is reachable through several producers of one invocation.
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
pub(crate) mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use arrow::datatypes::{DataType, Field, Schema};
    use cudf::testing;
    use datafusion::{
        common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
        execution::context::TaskContext,
        physical_expr::PhysicalExpr,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, empty::EmptyExec,
            metrics::ExecutionPlanMetricsSet,
        },
    };
    use futures::future::BoxFuture;
    use query_planning::{
        admission::AttemptCapabilityEvidence,
        expr::{Expr, LimitSpec, SortExpr, SortSpec},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, ProjectionOutput, QueryPlan, QueryPlanBuilder,
        },
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };
    use query_runtime::runtime::{ExecutionContext, QueryAdmissionRequest, resident::GpuResultHandle};
    use tokio::runtime::Builder as TestRuntimeBuilder;

    use super::*;
    use crate::{
        attempt::attempt_ticket_for_test,
        backend::InstalledTestGpuBackend,
        error::{Error, ErrorCode},
        native_exec::{
            AlgeonNativeExec, GraphClaimCalibrationCapture, QueryAdmissionSource, SealedAttemptMetricsTarget,
            SourceChunking,
            admission::{QueryAdmissionTicket, task_context_with_query_admission_ticket},
            graph::GraphExecutionPlan,
        },
    };

    #[derive(Debug)]
    pub(crate) struct FakeGpuResidentProducer {
        input_name: String,
        ticket: Arc<QueryAdmissionTicket>,
        capability_evidence: AttemptCapabilityEvidence,
        graph_execution: bool,
        execute_count: Arc<AtomicUsize>,
        inner: EmptyExec,
    }

    impl FakeGpuResidentProducer {
        /// The fake runs a graph algorithm only when built with `graph_execution`.
        pub(crate) fn graph_execution_plan(&self) -> Option<&dyn GraphExecutionPlan> {
            self.graph_execution.then_some(self as &dyn GraphExecutionPlan)
        }
    }

    #[derive(Debug)]
    pub(crate) struct FakeHostOwnedGraphExec {
        inner: EmptyExec,
    }

    impl DisplayAs for FakeHostOwnedGraphExec {
        fn fmt_as(&self, format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(format, formatter)
        }
    }

    impl ExecutionPlan for FakeHostOwnedGraphExec {
        fn name(&self) -> &str {
            "FakeHostOwnedGraphExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if children.is_empty() {
                Ok(self)
            } else {
                Err(DataFusionError::Execution("fake host-owned graph has no children".to_string()))
            }
        }

        fn execute(
            &self,
            partition: usize,
            task_context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            self.inner.execute(partition, task_context)
        }
    }

    impl GraphExecutionPlan for FakeHostOwnedGraphExec {
        fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
            Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.test.host_owned_graph",
                "fake host-owned graph does not issue a evidence",
            ))
        }

        fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
            SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
        }

        fn gpu_resident_output_rejection(&self) -> Option<Error> {
            Some(
                Error::new(
                    ErrorCode::RequiredGpuOperationUnsupported,
                    "algeon_datafusion.test.host_owned_graph",
                    "fake graph output is host-owned",
                )
                .with_fact("reason", "logical_bfs_gpu_result_unsupported")
                .with_fact("output_boundary", "gpu_resident"),
            )
        }
    }

    impl DisplayAs for FakeGpuResidentProducer {
        fn fmt_as(&self, format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(format, formatter)
        }
    }

    impl ExecutionPlan for FakeGpuResidentProducer {
        fn name(&self) -> &str {
            "FakeGpuResidentProducer"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if children.is_empty() {
                Ok(self)
            } else {
                Err(DataFusionError::Execution("fake GPU-resident producer has no children".to_string()))
            }
        }

        fn execute(
            &self,
            partition: usize,
            task_context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            self.inner.execute(partition, task_context)
        }
    }

    impl GpuResidentProducer for FakeGpuResidentProducer {
        fn gpu_result_input_name(&self) -> &str {
            &self.input_name
        }

        fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
            Ok(self.capability_evidence.clone())
        }

        fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
            vec![Arc::clone(&self.ticket)]
        }

        fn execute_gpu_result_with_context<'a>(
            &'a self,
            _native_context: &'a ExecutionContext,
            _partition: usize,
            _task_context: Arc<TaskContext>,
        ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(DataFusionError::Execution(
                    "fake GPU-resident producer is only used to test composed admission evidence completion"
                        .to_string(),
                ))
            })
        }
    }

    impl GraphExecutionPlan for FakeGpuResidentProducer {
        fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
            Ok(self.capability_evidence.clone())
        }

        fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
            SealedAttemptMetricsTarget::new(&ExecutionPlanMetricsSet::new(), GraphClaimCalibrationCapture::default())
        }
    }

    /// A task context of the invocation whose one admission ticket is `ticket`.
    fn invocation_task_context(ticket: &Arc<QueryAdmissionTicket>) -> Arc<TaskContext> {
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(ticket))
    }

    fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
        backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority")
    }

    fn typed_graph_producer_evidence() -> (Arc<Schema>, AttemptCapabilityEvidence) {
        let edge_schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/composed-admission-evidence.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![10])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&edge_schema),
            vec!["src".to_string(), "dst".to_string()],
        )));
        let graph = builder.add_graph_algorithm(source, Vec::new(), spec.clone()).expect("BFS graph plan must lower");
        let graph_plan = builder.finish_gpu_resident(graph).unwrap();
        let mut evidence = AttemptCapabilityEvidence::for_query_plan(&graph_plan);
        evidence.attest_cugraph_graph_execution_scope(&graph_plan);
        (spec.output_schema, evidence)
    }

    fn typed_relational_producer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/composed-relational-producer.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![10])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            schema.fields().iter().map(|field| field.name().to_owned()).collect(),
        )));
        builder.finish_gpu_resident(source).unwrap()
    }

    fn typed_relational_producer_evidence(schema: Arc<Schema>) -> AttemptCapabilityEvidence {
        let producer_plan = typed_relational_producer_plan(schema);
        AttemptCapabilityEvidence::for_query_plan(&producer_plan)
    }

    fn consumer_plan_for_inputs(input_names: &[&str], schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let inputs = input_names
            .iter()
            .map(|input_name| {
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(*input_name, Arc::clone(&schema))))
            })
            .collect::<Vec<_>>();
        let root = match inputs.as_slice() {
            [input] => *input,
            _ => builder.add_union(inputs, Arc::clone(&schema)).unwrap(),
        };
        builder.finish_with_host_sink(root).unwrap()
    }

    fn consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        consumer_plan_for_inputs(&["fake_input"], schema)
    }

    #[test]
    fn composed_constructor_preserves_host_owned_graph_gpu_output_rejection() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let child = Arc::new(FakeHostOwnedGraphExec { inner: EmptyExec::new(schema) }) as Arc<dyn ExecutionPlan>;

        let error = AlgeonNativeExec::try_new_composed_for_test(
            consumer_plan,
            vec![child],
            AlgeonNativeExecutionConfig::default(),
        )
        .expect_err("host-owned graph results must not be accepted as GPU-resident producers");

        assert_eq!(error.code(), ErrorCode::RequiredGpuOperationUnsupported);
        assert_eq!(error.fact_value("reason"), Some("logical_bfs_gpu_result_unsupported"));
        assert_eq!(error.fact_value("output_boundary"), Some("gpu_resident"));
    }

    fn non_attested_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
        let projected = builder
            .add_projection(
                source,
                schema
                    .fields()
                    .iter()
                    .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
                    .collect(),
            )
            .unwrap();
        builder.finish_with_host_sink(projected).unwrap()
    }

    fn filter_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
        let filtered = builder.add_filter(source, Expr::is_not_null(Expr::column(schema.field(0).name()))).unwrap();
        builder.finish_with_host_sink(filtered).unwrap()
    }

    fn zero_copy_projection_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
        let field = schema.field(0).clone();
        let projected =
            builder.add_projection(source, vec![ProjectionOutput::new(Expr::column(field.name()), field)]).unwrap();
        builder.finish_with_host_sink(projected).unwrap()
    }

    fn limit_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", Arc::clone(&schema))));
        let limited = builder.add_limit(source, LimitSpec::fetch(1)).unwrap();
        builder.finish_with_host_sink(limited).unwrap()
    }

    fn sorted_consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        consumer_plan(schema).with_appended_sort(SortSpec::new(vec![SortExpr::new(
            Expr::column("vertex"),
            false,
            false,
        )]))
    }

    fn complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);
        let task_context = invocation_task_context(&producer_ticket);

        let completed =
            complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
                .expect("evidence completion is CPU-only and does not acquire CUDA");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("complete evidence installs one ticket");

        assert!(Arc::ptr_eq(&completed_ticket, &producer_ticket), "completion keeps the invocation's one ticket");
        let completed_evidence =
            completed_ticket.attempt_capability_evidence().expect("completed ticket retains the composed evidence");
        completed_evidence
            .validate_covers_plan(&consumer_plan)
            .expect("the completed evidence must cover the consumer plan it was derived from");
    }

    fn complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = non_attested_projection_consumer_plan(Arc::clone(&schema));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);
        let completed = complete_composed_admission_evidence(
            config,
            &consumer_plan,
            &prepared_inputs,
            invocation_task_context(&producer_ticket).as_ref(),
        )
        .expect("non-attested one-producer consumer uses ordinary bounded admission");
        let completed_ticket = completed
            .execution_config
            .query_admission_ticket()
            .expect("fallback completion installs one admission ticket");
        let completed_evidence =
            completed_ticket.attempt_capability_evidence().expect("fallback completion retains the selected evidence");
        completed_evidence
            .validate_covers_plan(&consumer_plan)
            .expect("fallback composed evidence must cover the non-attested consumer plan");
    }

    fn complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant() {
        for (label, build) in [
            ("filter", filter_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
            ("projection", zero_copy_projection_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
            ("limit", limit_consumer_plan as fn(Arc<Schema>) -> QueryPlan),
        ] {
            let backend = InstalledTestGpuBackend::new();
            let source = admission_source(&backend);
            let (schema, producer_evidence) = typed_graph_producer_evidence();
            let consumer_plan = build(Arc::clone(&schema));
            let execute_count = Arc::new(AtomicUsize::new(0));
            let (_producer_ticket_attempt, producer_ticket) =
                attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
            let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
                input_name: "fake_input".to_string(),
                ticket: Arc::clone(&producer_ticket),
                capability_evidence: producer_evidence,
                graph_execution: true,
                execute_count: Arc::clone(&execute_count),
                inner: EmptyExec::new(schema),
            })];
            let config =
                AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                    .with_query_admission_source(source);
            let completed = complete_composed_admission_evidence(
                config,
                &consumer_plan,
                &prepared_inputs,
                invocation_task_context(&producer_ticket).as_ref(),
            )
            .unwrap_or_else(|error| panic!("{label} consumer must complete a bounded evidence: {error}"));
            assert_eq!(
                execute_count.load(Ordering::SeqCst),
                0,
                "{label} admission evidence does not execute producers"
            );
            let completed_ticket = completed
                .execution_config
                .query_admission_ticket()
                .expect("bounded consumer completion installs an admission ticket");
            let evidence = completed_ticket
                .attempt_capability_evidence()
                .expect("bounded consumer completion retains the composed evidence")
                .clone();
            let handle = backend
                .backend()
                .session_capability()
                .admit_query_with_request(
                    QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence),
                )
                .unwrap_or_else(|error| panic!("{label} consumer must admit on the CPU-only test backend: {error}"));
            assert_eq!(execute_count.load(Ordering::SeqCst), 0, "{label} admission does not execute producers");
            handle.end_test_attempt();
            drop(handle);
            drop(completed_ticket);
            drop(prepared_inputs);
        }
    }

    #[test]
    fn tests_tg_composed_cases_1() {
        testing::run_named_cases(&[
            (
                "complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path",
                complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path,
            ),
            (
                "complete_composed_admission_evidence_mixed_producers_preserve_engine_selection",
                complete_composed_admission_evidence_mixed_producers_preserve_engine_selection,
            ),
            (
                "complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence",
                complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence,
            ),
            (
                "complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission",
                complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission,
            ),
            (
                "complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence",
                complete_composed_admission_evidence_extends_the_invocation_ticket_with_full_cpu_only_evidence,
            ),
        ]);
    }

    #[test]
    fn tests_tg_composed_cases_2() {
        testing::run_named_cases(&[
            (
                "complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant",
                complete_composed_admission_evidence_bounds_filter_projection_and_limit_before_grant,
            ),
            (
                "complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket",
                complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket,
            ),
            (
                "complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope",
                complete_composed_admission_evidence_aggregates_non_attested_consumer_fallback_scope,
            ),
            ("prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant", || {
                TestRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("case runtime")
                    .block_on(prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant())
            }),
            (
                "complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind",
                complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind,
            ),
        ]);
    }

    fn complete_composed_admission_evidence_preserves_bounded_single_graph_consumer_path() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = sorted_consumer_plan(Arc::clone(&schema));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);

        let completed = complete_composed_admission_evidence(
            config,
            &consumer_plan,
            &prepared_inputs,
            invocation_task_context(&producer_ticket).as_ref(),
        )
        .expect("the existing single graph sort must use the engine composed resolver");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("standalone path retains one admission ticket");
        completed_ticket
            .attempt_capability_evidence()
            .expect("standalone path retains a composed evidence")
            .validate_covers_plan(&consumer_plan)
            .expect("the sorted graph consumer's evidence must cover its own plan");
    }

    fn complete_composed_admission_evidence_composes_relational_producers_with_typed_evidence() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let producer_evidence = typed_relational_producer_evidence(Arc::clone(&schema));
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            capability_evidence: producer_evidence,
            graph_execution: false,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);

        let completed = complete_composed_admission_evidence(
            config,
            &consumer_plan,
            &prepared_inputs,
            invocation_task_context(&producer_ticket).as_ref(),
        )
        .expect("pure relational composition must use the engine composed resolver");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("relational composition retains one ticket");

        completed_ticket
            .attempt_capability_evidence()
            .expect("relational composition retains a composed evidence")
            .validate_covers_plan(&consumer_plan)
            .expect("relational composed evidence must cover the consumer plan");
    }

    fn complete_composed_admission_evidence_mixed_producers_preserve_engine_selection() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, graph_evidence) = typed_graph_producer_evidence();
        let relational_evidence = typed_relational_producer_evidence(Arc::clone(&schema));
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);

        let mixed_plan = consumer_plan_for_inputs(&["graph", "relational"], Arc::clone(&schema));
        let (_mixed_attempt, mixed_ticket) = attempt_ticket_for_test(
            config.query_admission_source().expect("test config has admission authority"),
            None,
        );
        let mixed_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
            Arc::new(FakeGpuResidentProducer {
                input_name: "graph".to_string(),
                ticket: Arc::clone(&mixed_ticket),
                capability_evidence: graph_evidence.clone(),
                graph_execution: true,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }),
            Arc::new(FakeGpuResidentProducer {
                input_name: "relational".to_string(),
                ticket: Arc::clone(&mixed_ticket),
                capability_evidence: relational_evidence.clone(),
                graph_execution: false,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }),
        ];
        let mixed = complete_composed_admission_evidence(
            config.clone(),
            &mixed_plan,
            &mixed_inputs,
            invocation_task_context(&mixed_ticket).as_ref(),
        )
        .expect("adding a complete relational producer must remain supported");
        let mixed_completed_ticket =
            mixed.execution_config.query_admission_ticket().expect("mixed completion installs a ticket");
        let mixed_evidence =
            mixed_completed_ticket.attempt_capability_evidence().expect("mixed completion retains a evidence");
        mixed_evidence
            .validate_covers_plan(&mixed_plan)
            .expect("mixed evidence must cover the mixed-producer consumer plan");

        let three_input_plan = consumer_plan_for_inputs(&["first", "second", "relational"], Arc::clone(&schema));
        let (_three_input_attempt, three_input_ticket) = attempt_ticket_for_test(
            config.query_admission_source().expect("test config has admission authority"),
            None,
        );
        let three_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![
            Arc::new(FakeGpuResidentProducer {
                input_name: "first".to_string(),
                ticket: Arc::clone(&three_input_ticket),
                capability_evidence: graph_evidence.clone(),
                graph_execution: true,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }),
            Arc::new(FakeGpuResidentProducer {
                input_name: "second".to_string(),
                ticket: Arc::clone(&three_input_ticket),
                capability_evidence: graph_evidence,
                graph_execution: true,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }),
            Arc::new(FakeGpuResidentProducer {
                input_name: "relational".to_string(),
                ticket: Arc::clone(&three_input_ticket),
                capability_evidence: relational_evidence,
                graph_execution: false,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }),
        ];
        let three_input = complete_composed_admission_evidence(
            config,
            &three_input_plan,
            &three_inputs,
            invocation_task_context(&three_input_ticket).as_ref(),
        )
        .expect("adding a complete relational producer must not reject two graph producers");
        three_input
            .execution_config
            .query_admission_ticket()
            .expect("three-input completion installs a ticket")
            .attempt_capability_evidence()
            .expect("three-input completion retains a evidence")
            .validate_covers_plan(&three_input_plan)
            .expect("two graph plus one relational producer remains covered");
    }

    fn complete_composed_admission_evidence_preserves_the_active_stream_ticket_before_admission() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let (_producer_ticket_attempt, producer_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let (_active_ticket_attempt, active_ticket) =
            attempt_ticket_for_test(source.clone(), Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: producer_ticket,
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

        let completed =
            complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
                .expect("evidence completion must preserve the already-active stream ticket");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("completion retains the active ticket");

        assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
        assert!(active_ticket.attempt_capability_evidence().is_some());
    }

    fn complete_composed_admission_evidence_rebinds_outer_ticket_without_producer_ticket() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let producer_plan = typed_relational_producer_plan(Arc::clone(&schema));
        let producer = AlgeonNativeExec::try_new_producer_for_test("fake_input", producer_plan)
            .expect("ticketless relational producer should build");
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let (_active_ticket_attempt, active_ticket) = attempt_ticket_for_test(source.clone(), None);
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);

        let completed =
            complete_composed_admission_evidence(config, &consumer_plan, &[Arc::new(producer)], task_context.as_ref())
                .expect("outer ticket must absorb the complete producer and consumer evidence");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("completion retains the active outer ticket");

        assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
        assert!(active_ticket.attempt_capability_evidence().is_some());
    }

    #[test]
    fn complete_composed_admission_evidence_keeps_no_authority_unchanged() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let producer = AlgeonNativeExec::try_new_producer_for_test(
            "fake_input",
            typed_relational_producer_plan(Arc::clone(&schema)),
        )
        .expect("ticketless relational producer should build");
        let consumer_plan = consumer_plan(schema);
        let config = AlgeonNativeExecutionConfig::default();

        let completed = complete_composed_admission_evidence(
            config,
            &consumer_plan,
            &[Arc::new(producer)],
            &TaskContext::default(),
        )
        .expect("a direct caller without admission authority remains unchanged");

        assert!(completed.execution_config.query_admission_ticket().is_none());
        assert!(!completed.execution_config.has_query_admission_authority());
    }

    fn complete_composed_admission_evidence_rejects_backend_mismatch_before_rebind() {
        let backend = InstalledTestGpuBackend::new();
        let producer_source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let (producer_attempt, producer_ticket) =
            attempt_ticket_for_test(producer_source, Some(producer_evidence.clone()));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: producer_ticket,
            capability_evidence: producer_evidence,
            graph_execution: true,
            execute_count: Arc::new(AtomicUsize::new(0)),
            inner: EmptyExec::new(schema),
        })];
        drop(producer_attempt);
        drop(backend);

        let other_backend = InstalledTestGpuBackend::new();
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(admission_source(&other_backend));
        let result =
            complete_composed_admission_evidence(config, &consumer_plan, &prepared_inputs, &TaskContext::default());
        let error = match result {
            Ok(_) => panic!("producer and root admission authorities must match"),
            Err(error) => error,
        };
        assert_eq!(error.fact_value("reason"), Some("composed_producer_admission_backend_mismatch"));
    }

    async fn prepared_composed_producers_rebind_one_ordered_scope_evidence_and_grant() {
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let (schema, producer_evidence) = typed_graph_producer_evidence();
        let consumer_plan = consumer_plan_for_inputs(&["first", "second"], Arc::clone(&schema));
        let (_attempt, invocation_ticket) = attempt_ticket_for_test(source.clone(), None);
        let producers = vec![
            Arc::new(FakeGpuResidentProducer {
                input_name: "first".to_string(),
                ticket: Arc::clone(&invocation_ticket),
                capability_evidence: producer_evidence.clone(),
                graph_execution: true,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }) as Arc<dyn ExecutionPlan>,
            Arc::new(FakeGpuResidentProducer {
                input_name: "second".to_string(),
                ticket: Arc::clone(&invocation_ticket),
                capability_evidence: producer_evidence,
                graph_execution: true,
                execute_count: Arc::new(AtomicUsize::new(0)),
                inner: EmptyExec::new(Arc::clone(&schema)),
            }) as Arc<dyn ExecutionPlan>,
        ];
        let config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_query_admission_source(source);
        let composed = AlgeonNativeExec::try_new_composed_for_test(consumer_plan.clone(), producers, config)
            .expect("the real composed constructor must retain both prepared producer occurrences");

        let task_context = invocation_task_context(&invocation_ticket);
        let prepared = composed
            .prepare_lakehouse_metadata(Arc::clone(&task_context))
            .await
            .expect("CPU-only composed preparation must preserve both producers");
        let prepared_occurrences = prepared
            .gpu_inputs
            .iter()
            .map(|input| gpu_resident_producer_input_name(input.as_ref()).expect("prepared producer identity"))
            .collect::<Vec<_>>();
        assert_eq!(prepared_occurrences, ["first", "second"]);

        let completed = complete_composed_admission_evidence(
            prepared.execution_config.clone(),
            prepared.native_plan(),
            &prepared.gpu_inputs,
            task_context.as_ref(),
        )
        .expect("evidence completion must remain CPU-only before context acquisition");
        let completed_ticket =
            completed.execution_config.query_admission_ticket().expect("composed evidence owns one ticket");
        let completed_evidence =
            completed_ticket.attempt_capability_evidence().expect("composed evidence completion retains one");
        completed_evidence
            .validate_covers_plan(&consumer_plan)
            .expect("prepared producer occurrences must derive evidence covering the consumer plan");

        // The scheduler receives only this opaque evidence. Its selected handle
        // is the immutable grant authority; no GPU work or result handle is
        // created by this assertion.
        let grant = backend
            .backend()
            .session_capability()
            .admit_query_with_request(
                QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(completed_evidence),
            )
            .expect("the installed test profile must admit the composed evidence");
        assert_eq!(grant.device_ordinal(), 0);
        drop(grant);
    }
}
