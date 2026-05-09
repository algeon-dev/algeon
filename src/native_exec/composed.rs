//! Composed native execution node for GPU-resident native fragments.
//!
//! Producer children emit device-resident handles into a shared native context.
//! The consumer plan then uses those handles as GPU inputs and materializes the
//! final result back to host Arrow for DataFusion.

use std::{collections::BTreeSet, sync::Arc};

use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    exec::{execute_query_plan_with_gpu_inputs_observed, output::QueryOutput},
    plan::QueryPlan,
    runtime::{
        ExecutionContext,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
};

use super::{
    admission::query_admission_ticket_from_task_context,
    common::{
        NATIVE_NEXUS_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, conservative_properties,
        gpu_resident_source_names, mark_native_runtime_error, native_host_output_stream, native_plan_one_line,
        native_query_runtime_error, properties_with_equivalence, run_gpu_execution_task, write_native_plan_tree,
    },
    config::NexusNativeExecutionConfig,
    metrics::{
        GraphClaimCalibrationCapture, NativeExecutionMetricsCapture, NexusNativeMetrics, SealedAttemptMetricsTarget,
    },
    producer::{
        GpuResidentProducer, execute_gpu_resident_producer_with_context, gpu_resident_producer_input_name,
        prepare_gpu_resident_producer_for_execution, prepared_gpu_resident_producer_admission_tickets,
        register_gpu_resident_producer_discovery,
    },
};
use crate::{
    attempt::NexusAttemptExtension,
    error::{Error, ErrorCode},
    native_exec::graph::graph_execution_plan,
};

/// DataFusion wrapper for composed native execution with GPU-resident producer fragments.
///
/// The wrapper executes producer children into `GpuResultHandle`s, maps those handles into a
/// `GpuResidentInputMap`, then runs a host-sink native consumer plan that materializes the final
/// result as Arrow for DataFusion.
#[derive(Debug, Clone)]
pub struct NexusNativeComposedExec {
    plan: QueryPlan,
    gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    execution_config: NexusNativeExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NexusNativeComposedExec {
    /// Create composed native execution from GPU-resident producer children.
    ///
    /// The final plan must use a host Arrow sink, every child must advertise a
    /// unique GPU-resident input name, and the child names must match the
    /// consumer plan's declared GPU sources exactly.
    pub fn try_new(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NexusNativeExecutionConfig,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, gpu_inputs, execution_config, None)
    }

    pub(crate) fn try_new_with_equivalence_properties(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NexusNativeExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, gpu_inputs, execution_config, Some(eq_properties))
    }

    fn try_new_with_cache(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NexusNativeExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
    ) -> crate::error::Result<Self> {
        // Constructor validation preserves the composed execution contract: the final consumer must
        // host-materialize, every child must be a GPU-resident producer, and required/provided GPU
        // input names must match exactly.
        if !plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.try_new",
                "native composed DataFusion execution requires a host-arrow result sink",
            )
            .with_fact("result_sink", "gpu_resident"));
        }
        if gpu_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.try_new",
                "native composed execution requires at least one GPU-resident producer",
            ));
        }

        let required_inputs = gpu_resident_source_names(&plan);
        let mut provided_inputs = BTreeSet::new();
        for child in &gpu_inputs {
            let Some(input_name) = gpu_resident_producer_input_name(child.as_ref()) else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_composed.try_new",
                    "native composed execution children must be GPU-resident native producers",
                )
                .with_fact("child_name", child.name()));
            };
            if !provided_inputs.insert(input_name.clone()) {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_composed.try_new",
                    "native composed execution received duplicate GPU-resident producer input names",
                )
                .with_fact("input_name", input_name));
            }
        }

        let missing_inputs = required_inputs.difference(&provided_inputs).map(String::as_str).collect::<Vec<_>>();
        if !missing_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.try_new",
                "native composed execution is missing GPU-resident producer inputs",
            )
            .with_fact("missing_inputs", missing_inputs.join(",")));
        }
        let unused_inputs = provided_inputs.difference(&required_inputs).map(String::as_str).collect::<Vec<_>>();
        if !unused_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.try_new",
                "native composed execution received GPU-resident producer inputs the consumer plan does not declare",
            )
            .with_fact("unused_inputs", unused_inputs.join(",")));
        }
        let cache = match eq_properties {
            Some(eq_properties) => properties_with_equivalence(Arc::clone(plan.output_schema()), eq_properties)?,
            None => conservative_properties(Arc::clone(plan.output_schema())),
        };

        let metrics_capture = execution_config.new_metrics_capture();
        let gpu_inputs = gpu_inputs
            .into_iter()
            .map(|input| {
                let Some(producer) = input.as_ref().downcast_ref::<super::producer::NexusNativeGpuProducerExec>()
                else {
                    return input;
                };
                Arc::new(producer.clone().with_metrics_capture(metrics_capture.clone())) as Arc<dyn ExecutionPlan>
            })
            .collect();
        Ok(Self {
            cache,
            plan,
            gpu_inputs,
            execution_config,
            metrics_capture,
            graph_claim_calibration_capture: GraphClaimCalibrationCapture::default(),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn execution_config(&self) -> &NexusNativeExecutionConfig {
        &self.execution_config
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn gpu_inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.gpu_inputs
    }

    /// Returns the one post-seal attempt projection target for this composed
    /// root. Producer fragments are intentionally not separate owners.
    pub(crate) fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        let producer_captures = self.gpu_inputs.iter().filter_map(|input| {
            super::sealed_attempt_metrics_target(input.as_ref())
                .map(|target| target.graph_claim_calibration_captures().collect::<Vec<_>>())
        });
        SealedAttemptMetricsTarget::new(&self.metrics, self.graph_claim_calibration_capture.clone())
            .with_graph_claim_calibration_captures(producer_captures.flatten())
    }

    pub(crate) async fn prepare_lakehouse_metadata(&self, task_context: Arc<TaskContext>) -> DataFusionResult<Self> {
        let mut prepared = self.clone();
        if self.execution_config.lakehouse_footer_pruning() {
            prepared.plan =
                super::lakehouse::resolve_query_plan(&self.execution_config, self.plan.clone(), task_context.as_ref())
                    .await;
        }
        let mut prepared_inputs = Vec::with_capacity(self.gpu_inputs.len());
        for input in &self.gpu_inputs {
            prepared_inputs
                .push(prepare_gpu_resident_producer_for_execution(Arc::clone(input), Arc::clone(&task_context)).await?);
        }
        prepared.gpu_inputs = prepared_inputs;
        prepared.execution_config = prepared.execution_config.with_lakehouse_footer_pruning(false);
        Ok(prepared)
    }
}

/// Replaces provisional producer tickets with the one proof for the complete
/// composed GPU island.  Child preparation is the only phase that can resolve
/// footer-bound graph evidence, while the outer consumer is the only owner of
/// its aggregate/filter capability requirements.
fn complete_composed_admission_proof(
    execution_config: NexusNativeExecutionConfig,
    consumer_plan: &QueryPlan,
    prepared_gpu_inputs: &[Arc<dyn ExecutionPlan>],
    task_context: &TaskContext,
) -> DataFusionResult<NexusNativeExecutionConfig> {
    let mut producer_tickets = Vec::new();
    for input in prepared_gpu_inputs {
        for ticket in prepared_gpu_resident_producer_admission_tickets(input.as_ref()) {
            if !producer_tickets.iter().any(|existing| Arc::ptr_eq(existing, &ticket)) {
                producer_tickets.push(ticket);
            }
        }
    }

    let mut proofs =
        producer_tickets.iter().filter_map(|ticket| ticket.attempt_capability_proof_for_rebind()).collect::<Vec<_>>();
    let graph_result_input_names = prepared_gpu_inputs
        .iter()
        .filter_map(|input| graph_execution_plan(input.as_ref()).map(|graph| graph.graph_result_input_name()))
        .collect::<Vec<_>>();
    let consumer_proof = match graph_result_input_names.as_slice() {
        [graph_result_input_name] => {
            let passthrough = QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(
                consumer_plan,
                graph_result_input_name,
            );
            let bounded_consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan(
                consumer_plan,
                graph_result_input_name,
            );
            passthrough.or(bounded_consumer).or_else(|| QueryPlanCapabilityProof::for_query_plan(consumer_plan))
        },
        _ => QueryPlanCapabilityProof::for_query_plan(consumer_plan),
    };
    if let Some(consumer_proof) = consumer_proof {
        proofs.push(consumer_proof);
    }
    let Some(first_proof) = proofs.first().cloned() else {
        return Ok(execution_config);
    };
    let Some(first_ticket) = producer_tickets.first() else {
        // The outer consumer alone needs a capability proof, so retain the
        // existing ordinary admission path rather than manufacturing a second
        // ticket without a prepared producer to own its lifecycle.
        return Ok(execution_config);
    };

    let source = execution_config.query_admission_source().unwrap_or_else(|| first_ticket.source());
    if producer_tickets.iter().any(|ticket| !source.same_backend(&ticket.source())) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_composed.admission_proof",
            "prepared GPU producers belong to different admission backends",
        )
        .with_fact("reason", "composed_producer_admission_backend_mismatch")
        .into());
    }

    let complete_proof =
        QueryPlanCapabilityProof::combine(first_proof, proofs.into_iter().skip(1)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_composed.admission_proof",
                "failed to aggregate prepared producer and consumer capability proofs",
                error,
            ))
        })?;
    // The admission stream has already selected its ambient ticket before a
    // footer-bound producer can finish preparation. Rebinding a new ticket
    // here would create a second query-service lifecycle whose release can
    // revoke the deferred host cursor before it reaches EOF. Update that
    // still-unadmitted ambient ticket in place; standalone callers retain the
    // replacement-ticket path because they have no active stream to preserve.
    if let Some(active_ticket) = query_admission_ticket_from_task_context(task_context) {
        if !source.same_backend(&active_ticket.source()) {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.admission_proof",
                "the composed root and its active admission ticket belong to different backends",
            )
            .with_fact("reason", "composed_active_admission_backend_mismatch")
            .into());
        }
        active_ticket.rebind_capability_proof_before_admission(complete_proof).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.native_composed.admission_proof",
                "failed to rebind the active composed admission ticket before query-service admission",
                error,
            ))
        })?;
        return Ok(execution_config.with_existing_query_admission_ticket(active_ticket));
    }

    let ticket = first_ticket.rebind_with_source_and_capability_proof(source, complete_proof);
    if let Some(attempt) = task_context.session_config().options().extensions.get::<NexusAttemptExtension>() {
        attempt.bind_ticket(Arc::clone(&ticket));
    }
    Ok(execution_config.with_existing_query_admission_ticket(ticket))
}

impl DisplayAs for NexusNativeComposedExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "NexusNativeComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode=bounded, memory_control_policy={}, boundary_policy={}, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}",
                self.plan.node_count(),
                self.gpu_inputs.len(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "NexusNativeComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode=bounded, memory_control_policy={}, boundary_policy={}, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}, native_plan={}",
                self.plan.node_count(),
                self.gpu_inputs.len(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
                native_plan_one_line(&self.plan),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "gpu_input_count={}", self.gpu_inputs.len())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(f, "native_pipeline_execution_mode=bounded")?;
                writeln!(f, "memory_control_policy={}", self.execution_config.memory_control_policy())?;
                writeln!(f, "boundary_policy={}", self.execution_config.boundary_policy())?;
                writeln!(f, "input_boundary=gpu_resident")?;
                writeln!(f, "output_boundary=device_to_host")?;
                writeln!(f, "edge_normalize_inserted={}", self.plan.contains_edge_normalize())?;
                write_native_plan_tree(f, &self.plan)
            },
        }
    }
}

impl ExecutionPlan for NexusNativeComposedExec {
    fn name(&self) -> &str {
        "NexusNativeComposedExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.gpu_inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // These children are device-handle producers, not DataFusion row
        // streams. Repartitioning or coalescing one would wrap the producer
        // in a host execution node and break the composed GPU contract.
        vec![false; self.gpu_inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        NexusNativeComposedExec::try_new_with_equivalence_properties(
            self.plan.clone(),
            children,
            self.execution_config.clone(),
            self.cache.equivalence_properties().clone(),
        )
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed.execute",
                "native composed cudf execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let plan = self.plan.clone();
        let execution_config = self.execution_config.clone();
        let account_memory_pool = execution_config.host_output_memory_pool_accounting();
        let gpu_inputs = self.gpu_inputs.clone();
        let metrics = NexusNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        )
        .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
        let output_schema = Arc::clone(&schema);
        let output = native_host_output_stream(
            NATIVE_NEXUS_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                let plan = super::lakehouse::resolve_query_plan(&execution_config, plan, task_context.as_ref()).await;
                let mut prepared_gpu_inputs = Vec::with_capacity(gpu_inputs.len());
                for input in gpu_inputs {
                    prepared_gpu_inputs
                        .push(prepare_gpu_resident_producer_for_execution(input, Arc::clone(&task_context)).await?);
                }
                let execution_config = complete_composed_admission_proof(
                    execution_config,
                    &plan,
                    &prepared_gpu_inputs,
                    task_context.as_ref(),
                )?;
                let runtime_observation_config = execution_config.runtime_observation_config();
                // Build one native context for all producer fragments and the consumer so device-resident
                // handles are compatible across the composed execution.
                let context_lease = execution_config
                    .execution_context_async(Arc::clone(&task_context))
                    .await
                    .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
                if let Some(admission_facts) = context_lease.admission_facts() {
                    metrics.record_admission_facts(admission_facts);
                }
                let gpu_executor = context_lease.context().gpu_executor().map_err(|error| {
                    DataFusionError::from(mark_native_runtime_error(
                        Error::from_native(
                            "datafusion_nexus.native_composed.gpu_execution_pool",
                            "query service closed before composed native execution acquired its GPU pool",
                            error,
                        ),
                        "gpu_execution_pool",
                    ))
                })?;
                let observation = context_lease.context().runtime_observation().clone();
                if let Some(ticket) = super::admission::query_admission_ticket_from_task_context(task_context.as_ref())
                {
                    ticket.record_attempt_gpu_execution_started();
                }
                let cursor_executor = gpu_executor.clone();
                let cursor_observation = observation.clone();
                let cursor_metrics = metrics.clone();
                run_gpu_execution_task(
                    gpu_executor,
                    observation,
                    "datafusion_nexus.native_composed.gpu_execution_pool",
                    move |gpu_stream_scope| async move {
                        let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                        let execution_task_context =
                            context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                        let mut input_map = GpuResidentInputMap::new();
                        for child in prepared_gpu_inputs {
                            let (input_name, handle) = execute_gpu_resident_producer_with_context(
                                child.as_ref(),
                                &context,
                                partition,
                                Arc::clone(&execution_task_context),
                            )
                            .await?;
                            input_map.insert(input_name, handle);
                        }

                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        let gpu_plan = plan.clone_with_deferred_host_materialization_sink();
                        // The consumer query starts after all GPU-resident producer handles are registered. The
                        // native attempt is terminal after this point and is not replayed through DataFusion.
                        let outcome = execute_query_plan_with_gpu_inputs_observed(
                            &context,
                            &gpu_plan,
                            &input_map,
                            runtime_observation_config,
                        );
                        let output = metrics.finalize_query_outcome_for_deferred_host(outcome).map_err(|error| {
                            DataFusionError::from(native_query_runtime_error(
                                "datafusion_nexus.native_composed.execute",
                                "nexus native composed query execution failed",
                                error,
                            ))
                        })?;
                        metrics
                            .record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                        let QueryOutput::GpuResident(handle) = output else {
                            return Err(DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "datafusion_nexus.native_composed.execute",
                                    "deferred composed host execution did not produce a GPU-resident cursor input",
                                )
                                .with_fact("result_sink", "host_arrow"),
                            ));
                        };
                        let handle = GpuResultHandle::new(&context, handle.into_gpu_frame(), output_schema).map_err(
                            |error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "datafusion_nexus.native_composed.host_batch_cursor",
                                    "failed to bind deferred composed output to the DataFusion host schema",
                                    error,
                                ))
                            },
                        )?;
                        let cursor = handle
                            .into_host_batch_cursor(context.cancellation().clone(), HostBatchCursorConfig::default())
                            .map_err(|error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "datafusion_nexus.native_composed.host_batch_cursor",
                                    "failed to create deferred composed host-output cursor",
                                    error,
                                ))
                            })?;
                        Ok::<NativeHostOutputCursor, DataFusionError>(NativeHostOutputCursor::new(
                            cursor,
                            cursor_executor,
                            cursor_observation,
                            cursor_metrics,
                            context_lease,
                        ))
                    },
                )
                .await
            },
        );

        Ok(output)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// GPU-resident form of a relational native consumer with accelerator
/// producer children.
///
/// This is an internal composition boundary. It cannot stream Arrow through
/// [`ExecutionPlan::execute`]; an outer GPU-resident consumer must
/// prepare it before native admission and execute it with the same native
/// context used by the final accelerator operation.
#[derive(Debug, Clone)]
pub struct NexusNativeComposedGpuProducerExec {
    input_name: String,
    plan: QueryPlan,
    gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    execution_config: NexusNativeExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    prepared: bool,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NexusNativeComposedGpuProducerExec {
    fn try_new(
        input_name: impl Into<String>,
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NexusNativeExecutionConfig,
        metrics_capture: Option<NativeExecutionMetricsCapture>,
        eq_properties: EquivalenceProperties,
    ) -> crate::error::Result<Self> {
        let input_name = input_name.into();
        if input_name.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed_gpu_producer.try_new",
                "composed GPU-resident producer input name must not be empty",
            ));
        }
        if plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_composed_gpu_producer.try_new",
                "composed GPU-resident producer requires a GPU-resident sink",
            )
            .with_fact("result_sink", "host_arrow"));
        }
        validate_composed_gpu_inputs(&plan, &gpu_inputs, "datafusion_nexus.native_composed_gpu_producer.try_new")?;
        let cache = properties_with_equivalence(Arc::clone(plan.output_schema()), eq_properties)?;
        register_composed_gpu_producer();
        Ok(Self {
            input_name,
            plan,
            gpu_inputs,
            execution_config,
            metrics_capture,
            prepared: false,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    pub(crate) fn execution_config(&self) -> &NexusNativeExecutionConfig {
        &self.execution_config
    }
}

impl GpuResidentProducer for NexusNativeComposedGpuProducerExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.input_name
    }

    fn prepared_admission_tickets(&self) -> Vec<Arc<super::admission::QueryAdmissionTicket>> {
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

    fn prepare_execution_plan<'a>(
        &'a self,
        _plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            let mut prepared = self.clone();
            prepared.plan =
                super::lakehouse::resolve_query_plan(&self.execution_config, self.plan.clone(), task_context.as_ref())
                    .await
                    .clone_with_gpu_resident_sink();
            let mut prepared_inputs = Vec::with_capacity(self.gpu_inputs.len());
            for input in &self.gpu_inputs {
                prepared_inputs.push(
                    prepare_gpu_resident_producer_for_execution(Arc::clone(input), Arc::clone(&task_context)).await?,
                );
            }
            prepared.gpu_inputs = prepared_inputs;
            prepared.execution_config = prepared.execution_config.with_lakehouse_footer_pruning(false);
            prepared.prepared = true;
            Ok(Arc::new(prepared) as Arc<dyn ExecutionPlan>)
        })
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
        Box::pin(async move {
            if partition != 0 {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_composed_gpu_producer.execute_gpu",
                    "composed GPU-resident producer exposes exactly one output partition",
                )
                .with_fact("partition", partition)
                .into());
            }
            if !self.prepared {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_composed_gpu_producer.execute_gpu",
                    "composed GPU-resident producer was not prepared before native context acquisition",
                )
                .with_log_safe_fact("producer_state", "not_prepared")
                .into());
            }

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

            let metrics = NexusNativeMetrics::new_with_capture_and_shape(
                &self.metrics,
                partition,
                self.metrics_capture.clone(),
                self.execution_config.plan_task_mapping_datafusion_shape(),
            );
            let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
            let outcome = execute_query_plan_with_gpu_inputs_observed(
                native_context,
                &self.plan,
                &input_map,
                self.execution_config.runtime_observation_config(),
            );
            let output = metrics.finalize_query_outcome(outcome).map_err(|error| {
                DataFusionError::from(native_query_runtime_error(
                    "datafusion_nexus.native_composed_gpu_producer.execute_gpu",
                    "nexus composed GPU-resident relational execution failed",
                    error,
                ))
            })?;
            metrics.record_lakehouse_cache_fill_events(&native_context.drain_lakehouse_cache_fill_event_snapshots());
            let QueryOutput::GpuResident(handle) = output else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_composed_gpu_producer.execute_gpu",
                    "composed GPU-resident producer materialized a host result",
                )
                .with_fact("result_sink", "host_arrow")
                .into());
            };
            Ok((self.input_name.clone(), handle))
        })
    }
}

impl DisplayAs for NexusNativeComposedGpuProducerExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "NexusNativeComposedGpuProducerExec: input_name={}, native_nodes={}, gpu_input_count={}, sink=gpu_resident, input_boundary=gpu_resident, output_boundary=device_resident",
                self.input_name,
                self.plan.node_count(),
                self.gpu_inputs.len(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "NexusNativeComposedGpuProducerExec: input_name={}, native_nodes={}, gpu_input_count={}, sink=gpu_resident, input_boundary=gpu_resident, output_boundary=device_resident, native_plan={}",
                self.input_name,
                self.plan.node_count(),
                self.gpu_inputs.len(),
                native_plan_one_line(&self.plan),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "input_name={}", self.input_name)?;
                writeln!(f, "gpu_input_count={}", self.gpu_inputs.len())?;
                writeln!(f, "sink=gpu_resident")?;
                writeln!(f, "input_boundary=gpu_resident")?;
                writeln!(f, "output_boundary=device_resident")?;
                write_native_plan_tree(f, &self.plan)
            },
        }
    }
}

impl ExecutionPlan for NexusNativeComposedGpuProducerExec {
    fn name(&self) -> &str {
        "NexusNativeComposedGpuProducerExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.gpu_inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // Keep DataFusion distribution rules from inserting a row-stream
        // wrapper between this internal producer and its device inputs.
        vec![false; self.gpu_inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        NexusNativeComposedGpuProducerExec::try_new(
            self.input_name.clone(),
            self.plan.clone(),
            children,
            self.execution_config.clone(),
            self.metrics_capture.clone(),
            self.cache.equivalence_properties().clone(),
        )
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)
    }

    fn execute(
        &self,
        _partition: usize,
        _task_context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_composed_gpu_producer.execute",
            "composed GPU-resident producer cannot stream host Arrow directly",
        )
        .with_fact("result_sink", "gpu_resident")
        .into())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn validate_composed_gpu_inputs(
    plan: &QueryPlan,
    gpu_inputs: &[Arc<dyn ExecutionPlan>],
    operation: &'static str,
) -> crate::error::Result<()> {
    if gpu_inputs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            operation,
            "composed native execution requires at least one GPU-resident producer",
        ));
    }
    let required_inputs = gpu_resident_source_names(plan);
    let mut provided_inputs = BTreeSet::new();
    for child in gpu_inputs {
        let Some(input_name) = gpu_resident_producer_input_name(child.as_ref()) else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                operation,
                "composed native execution children must be GPU-resident producers",
            )
            .with_fact("child_name", child.name()));
        };
        if !provided_inputs.insert(input_name.clone()) {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                operation,
                "composed native execution received duplicate GPU-resident input names",
            )
            .with_fact("input_name", input_name));
        }
    }
    let missing_inputs = required_inputs.difference(&provided_inputs).map(String::as_str).collect::<Vec<_>>();
    if !missing_inputs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            operation,
            "composed native execution is missing GPU-resident producer inputs",
        )
        .with_fact("missing_inputs", missing_inputs.join(",")));
    }
    let unused_inputs = provided_inputs.difference(&required_inputs).map(String::as_str).collect::<Vec<_>>();
    if !unused_inputs.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            operation,
            "composed native execution received unused GPU-resident producer inputs",
        )
        .with_fact("unused_inputs", unused_inputs.join(",")));
    }
    Ok(())
}

fn composed_gpu_producer_discovery(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().map(|producer| producer as &dyn GpuResidentProducer)
}

fn register_composed_gpu_producer() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| register_gpu_resident_producer_discovery(composed_gpu_producer_discovery));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        common::{DataFusionError, Result as DataFusionResult},
        execution::context::TaskContext,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, empty::EmptyExec,
        },
    };
    use futures::future::BoxFuture;
    use nexus_query_engine::{
        admission::QueryPlanCapabilityProof,
        plan::{QueryPlan, QueryPlanBuilder},
        runtime::{ExecutionContext, resident::GpuResultHandle},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    use super::*;
    use crate::{
        backend::InstalledTestGpuBackend,
        native_exec::admission::{QueryAdmissionTicket, task_context_with_query_admission_ticket},
    };

    #[derive(Debug)]
    struct FakeGpuResidentProducer {
        input_name: String,
        ticket: Arc<QueryAdmissionTicket>,
        inner: EmptyExec,
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

        fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
            vec![Arc::clone(&self.ticket)]
        }

        fn execute_gpu_result_with_context<'a>(
            &'a self,
            _native_context: &'a ExecutionContext,
            _partition: usize,
            _task_context: Arc<TaskContext>,
        ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
            Box::pin(async {
                Err(DataFusionError::Execution(
                    "fake GPU-resident producer is only used to test composed admission proof completion".to_string(),
                ))
            })
        }
    }

    fn fake_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
        plan.downcast_ref::<FakeGpuResidentProducer>().map(|producer| producer as &dyn GpuResidentProducer)
    }

    fn admission_source(backend: &InstalledTestGpuBackend) -> crate::native_exec::QueryAdmissionSource {
        backend
            .optimizer_config()
            .native_execution_config()
            .query_admission_source()
            .expect("installed backend must provide admission authority")
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]))
    }

    fn producer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/composed-admission-proof.parquet")],
            schema,
            vec!["value".to_string()],
        )));
        builder.finish_gpu_resident(source)
    }

    fn consumer_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fake_input", schema)));
        builder.finish_with_host_sink(source)
    }

    #[test]
    fn complete_composed_admission_proof_rebinds_fake_producer_with_full_cpu_only_proof() {
        register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let schema = schema();
        let producer_proof = QueryPlanCapabilityProof::for_query_plan(&producer_plan(Arc::clone(&schema)))
            .expect("local parquet producer has a capability proof");
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let consumer_proof = QueryPlanCapabilityProof::for_query_plan(&consumer_plan)
            .expect("host-materializing consumer has a capability proof");
        let expected_proof = QueryPlanCapabilityProof::combine(producer_proof.clone(), [consumer_proof])
            .expect("producer and consumer proofs combine");
        let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: Arc::clone(&producer_ticket),
            inner: EmptyExec::new(schema),
        })];
        let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let task_context = TaskContext::default();

        let completed = complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, &task_context)
            .expect("proof completion is CPU-only and does not acquire CUDA");
        let completed_ticket = completed.query_admission_ticket().expect("complete proof installs one ticket");

        assert!(!Arc::ptr_eq(&completed_ticket, &producer_ticket), "completion reissues the prepared ticket");
        assert_eq!(completed_ticket.attempt_capability_proof(), Some(&expected_proof));
    }

    #[test]
    fn complete_composed_admission_proof_preserves_the_active_stream_ticket_before_admission() {
        register_gpu_resident_producer_discovery(fake_gpu_resident_producer);
        let backend = InstalledTestGpuBackend::new();
        let source = admission_source(&backend);
        let schema = schema();
        let producer_proof = QueryPlanCapabilityProof::for_query_plan(&producer_plan(Arc::clone(&schema)))
            .expect("local parquet producer has a capability proof");
        let consumer_plan = consumer_plan(Arc::clone(&schema));
        let consumer_proof = QueryPlanCapabilityProof::for_query_plan(&consumer_plan)
            .expect("host-materializing consumer has a capability proof");
        let expected_proof = QueryPlanCapabilityProof::combine(producer_proof.clone(), [consumer_proof])
            .expect("producer and consumer proofs combine");
        let producer_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof.clone()));
        let active_ticket = QueryAdmissionTicket::new(source.clone(), Some(producer_proof));
        let prepared_inputs: Vec<Arc<dyn ExecutionPlan>> = vec![Arc::new(FakeGpuResidentProducer {
            input_name: "fake_input".to_string(),
            ticket: producer_ticket,
            inner: EmptyExec::new(schema),
        })];
        let config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let task_context =
            task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

        let completed =
            complete_composed_admission_proof(config, &consumer_plan, &prepared_inputs, task_context.as_ref())
                .expect("proof completion must preserve the already-active stream ticket");
        let completed_ticket = completed.query_admission_ticket().expect("completion retains the active ticket");

        assert!(Arc::ptr_eq(&completed_ticket, &active_ticket));
        assert_eq!(active_ticket.attempt_capability_proof(), Some(&expected_proof));
    }
}
