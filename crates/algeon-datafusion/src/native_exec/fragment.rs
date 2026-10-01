//! The one DataFusion node for a native execution unit.
//!
//! A unit runs one admitted native plan. Its children are the GPU-resident
//! producers that plan reads as inputs. Its output either streams host Arrow
//! batches to DataFusion or stays on the GPU for a consumer that binds it by
//! input name; such a producer cannot stream through
//! [`ExecutionPlan::execute`].

use std::{collections::BTreeSet, sync::Arc};

use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use futures::future::BoxFuture;
use query_engine::{
    admission::{AdmittedPlan, AttemptCapabilityEvidence},
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput},
    plan::{QueryPlan, ResultSink, row_bounds::query_plan_hard_row_upper_bound},
    runtime::{ExecutionContext, resident::GpuResultHandle},
};
#[cfg(test)]
use query_engine::{exec::admit_query_plan, mode::StatisticalAggregateExecutionMode, policy::PlanningPolicy};

#[cfg(test)]
use super::metrics::NativeMetricsCapture;
use super::{
    admission::QueryAdmissionTicket,
    common::{
        NATIVE_ALGEON_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, comma_separated,
        conservative_properties, gpu_resident_source_names, mark_native_runtime_error, native_host_output_stream,
        native_plan_one_line, native_query_runtime_error, properties_with_equivalence, run_gpu_execution_task,
        write_native_plan_tree,
    },
    config::AlgeonNativeExecutionConfig,
    graph::graph_execution_plan,
    metrics::{
        AlgeonNativeMetrics, GraphClaimCalibrationCapture, NativeExecutionMetricsCapture, SealedAttemptMetricsTarget,
    },
    producer::{GpuResidentProducer, gpu_resident_producer_input_name},
};
use crate::error::{Error, ErrorCode, Result};

/// Where a native unit's result goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NativeOutput {
    /// Host Arrow batches streamed to DataFusion.
    Host,
    /// A GPU-resident result a consumer binds under `input_name`.
    GpuResident { input_name: String },
}

/// The four unit shapes, {leaf, composed} by {host, GPU-resident} output.
/// The shape names the node and the operation of its errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Host,
    Composed,
    Producer,
    ComposedProducer,
}

impl Shape {
    fn of(composed: bool, output: &NativeOutput) -> Self {
        match (composed, output) {
            (false, NativeOutput::Host) => Self::Host,
            (true, NativeOutput::Host) => Self::Composed,
            (false, NativeOutput::GpuResident { .. }) => Self::Producer,
            (true, NativeOutput::GpuResident { .. }) => Self::ComposedProducer,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Host => "AlgeonNativeExec",
            Self::Composed => "AlgeonNativeComposedExec",
            Self::Producer => "AlgeonNativeGpuProducerExec",
            Self::ComposedProducer => "AlgeonNativeComposedGpuProducerExec",
        }
    }

    const fn try_new_operation(self) -> &'static str {
        match self {
            Self::Host => "algeon_datafusion.native_exec.try_new",
            Self::Composed => "algeon_datafusion.native_composed.try_new",
            Self::Producer => "algeon_datafusion.native_gpu_producer.try_new",
            Self::ComposedProducer => "algeon_datafusion.native_composed_gpu_producer.try_new",
        }
    }

    const fn with_new_children_operation(self) -> &'static str {
        match self {
            Self::Host => "algeon_datafusion.native_exec.with_new_children",
            Self::Composed => "algeon_datafusion.native_composed.with_new_children",
            Self::Producer => "algeon_datafusion.native_gpu_producer.with_new_children",
            Self::ComposedProducer => "algeon_datafusion.native_composed_gpu_producer.with_new_children",
        }
    }
}

/// One native execution unit: an admitted native plan, the GPU-resident
/// producers it reads, and its host or GPU-resident output.
///
/// Created by the Algeon physical optimizer and adapter composition; the
/// public type identity serves [`ExecutionPlan`] inspection, downcasting,
/// metrics collection, and plan diagnostics. Construction is crate-private
/// and validates the sink against the output, the producer children against
/// the plan's GPU-resident sources, and the equivalence properties. Each node
/// exposes one output partition. A host unit executes its producers and its
/// plan in one admitted native context and streams the result as Arrow.
#[derive(Debug, Clone)]
pub struct AlgeonNativeExec {
    pub(super) admitted: Arc<AdmittedPlan>,
    pub(super) gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    output: NativeOutput,
    pub(super) execution_config: AlgeonNativeExecutionConfig,
    pub(super) metrics_capture: Option<NativeExecutionMetricsCapture>,
    pub(super) graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    /// Whether a composed GPU-resident unit resolved its plan and producers
    /// before the composed root acquired the native context.
    pub(super) prepared: bool,
    pub(super) metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl AlgeonNativeExec {
    /// A native unit for `admitted`, reading `gpu_inputs` and writing
    /// `output`. `eq_properties` are the replaced DataFusion subtree's
    /// properties, which the native plan must already reproduce; `None`
    /// declares no ordering.
    pub(crate) fn try_new(
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        output: NativeOutput,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
    ) -> Result<Self> {
        let metrics_capture = execution_config.new_metrics_capture();
        Self::try_new_with_metrics_capture(
            admitted,
            gpu_inputs,
            output,
            execution_config,
            eq_properties,
            metrics_capture,
        )
    }

    fn try_new_with_metrics_capture(
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        output: NativeOutput,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
        metrics_capture: Option<NativeExecutionMetricsCapture>,
    ) -> Result<Self> {
        let shape = Shape::of(!gpu_inputs.is_empty(), &output);
        let operation = shape.try_new_operation();
        let plan = admitted.plan();
        let schema = match &output {
            // The host unit executes the plan in the deferred form it was
            // admitted in and materializes the host batches itself.
            NativeOutput::Host => {
                if !matches!(plan.sink(), ResultSink::DeferredHostArrow { .. }) {
                    return Err(Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        operation,
                        "host native execution requires a deferred host-arrow result sink",
                    )
                    .with_fact("result_sink", result_sink_label(plan.sink())));
                }
                plan.caller_visible_output_schema()
            },
            NativeOutput::GpuResident { input_name } => {
                if input_name.is_empty() {
                    return Err(Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        operation,
                        "GPU-resident producer input name must not be empty",
                    ));
                }
                if plan.sink().materializes_host_boundary() {
                    return Err(Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        operation,
                        "GPU-resident producer requires a GPU-resident result sink",
                    )
                    .with_fact("result_sink", "host_arrow"));
                }
                plan.output_schema()
            },
        };
        validate_gpu_inputs(plan, &gpu_inputs, operation)?;
        let cache = match eq_properties {
            Some(eq_properties) => properties_with_equivalence(Arc::clone(schema), eq_properties)?,
            None => conservative_properties(Arc::clone(schema)),
        };
        // A host root records its producers' native execution under its own
        // capture, so one attempt projection reads the whole unit.
        let gpu_inputs = if output == NativeOutput::Host {
            gpu_inputs
                .into_iter()
                .map(|input| match input.downcast_ref::<Self>() {
                    Some(producer) if producer.gpu_inputs.is_empty() => {
                        let mut producer = producer.clone();
                        producer.metrics_capture.clone_from(&metrics_capture);
                        Arc::new(producer) as Arc<dyn ExecutionPlan>
                    },
                    _ => input,
                })
                .collect()
        } else {
            gpu_inputs
        };
        Ok(Self {
            admitted,
            gpu_inputs,
            output,
            execution_config,
            metrics_capture,
            graph_claim_calibration_capture: GraphClaimCalibrationCapture::default(),
            prepared: false,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// A host unit for `plan` admitted as planning would.
    #[cfg(test)]
    pub(crate) fn try_new_host_for_test(
        plan: QueryPlan,
        execution_config: AlgeonNativeExecutionConfig,
    ) -> Result<Self> {
        Self::try_new_composed_for_test(plan, Vec::new(), execution_config)
    }

    /// A host unit reading `gpu_inputs`, for `plan` admitted as planning would.
    #[cfg(test)]
    pub(crate) fn try_new_composed_for_test(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
    ) -> Result<Self> {
        let admitted = super::admit_for_test(&plan.clone_with_deferred_host_materialization_sink(), &execution_config)?;
        Self::try_new(admitted, gpu_inputs, NativeOutput::Host, execution_config, None)
    }

    /// A GPU-resident leaf producer for `plan`.
    #[cfg(test)]
    pub(crate) fn try_new_producer_for_test(input_name: impl Into<String>, plan: QueryPlan) -> Result<Self> {
        Self::try_new_producer_with_capture_for_test(input_name, plan, None, None, false)
    }

    /// A GPU-resident leaf producer recording into `metrics_capture`.
    #[cfg(test)]
    pub(crate) fn try_new_producer_with_capture_for_test(
        input_name: impl Into<String>,
        plan: QueryPlan,
        metrics_capture: Option<NativeMetricsCapture>,
        plan_task_mapping_datafusion_shape: Option<&str>,
        lakehouse_footer_pruning: bool,
    ) -> Result<Self> {
        let metrics_capture = Some(metrics_capture.unwrap_or_else(NativeMetricsCapture::standalone).new_execution());
        let admitted = admit_query_plan(&plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())
            .map_err(|error| {
                Error::from_native("algeon_datafusion.native_exec.test_admission", "test plan is not admissible", error)
            })?;
        let mut execution_config =
            AlgeonNativeExecutionConfig::default().with_lakehouse_footer_pruning(lakehouse_footer_pruning);
        if let Some(shape) = plan_task_mapping_datafusion_shape {
            execution_config = execution_config.with_plan_task_mapping_datafusion_shape(shape);
        }
        Self::try_new_with_metrics_capture(
            admitted,
            Vec::new(),
            NativeOutput::GpuResident { input_name: input_name.into() },
            execution_config,
            None,
            metrics_capture,
        )
    }

    fn shape(&self) -> Shape {
        Shape::of(!self.gpu_inputs.is_empty(), &self.output)
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        self.admitted.plan()
    }

    #[cfg(feature = "bench-support")]
    #[doc(hidden)]
    pub fn admitted_plan_for_internal_consumer(&self) -> &Arc<AdmittedPlan> {
        &self.admitted
    }

    /// Whether the unit streams host Arrow batches to DataFusion; otherwise
    /// its output stays on the GPU for a consumer unit.
    pub fn materializes_host(&self) -> bool {
        self.output == NativeOutput::Host
    }

    /// The input name a consumer binds this unit's GPU-resident output under.
    pub fn gpu_resident_input_name(&self) -> Option<&str> {
        match &self.output {
            NativeOutput::Host => None,
            NativeOutput::GpuResident { input_name } => Some(input_name),
        }
    }

    pub(crate) fn gpu_inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.gpu_inputs
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn execution_config(&self) -> &AlgeonNativeExecutionConfig {
        &self.execution_config
    }

    pub(crate) fn execution_metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    /// Returns the one post-seal attempt projection target for this host
    /// unit. Its producers are not separate owners; their graph calibration
    /// captures join this target. The attempt owner elects the target before
    /// execution; the node never reads or retains terminal attempt state.
    pub(crate) fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        let producer_captures = self.gpu_inputs.iter().filter_map(|input| {
            super::sealed_attempt_metrics_target(input.as_ref())
                .map(|target| target.graph_claim_calibration_captures().collect::<Vec<_>>())
        });
        SealedAttemptMetricsTarget::new(&self.metrics, self.graph_claim_calibration_capture.clone())
            .with_graph_claim_calibration_captures(producer_captures.flatten())
    }

    /// Resolves footer-pruned metadata for the plan and its producers before
    /// execution acquires the native context.
    pub(crate) async fn prepare_lakehouse_metadata(&self, task_context: Arc<TaskContext>) -> DataFusionResult<Self> {
        let mut prepared = self.clone();
        prepared.admitted = super::lakehouse::resolve_admitted_plan(
            self.execution_config.lakehouse_footer_pruning(),
            &self.admitted,
            task_context.as_ref(),
        )
        .await?;
        let mut prepared_inputs = Vec::with_capacity(self.gpu_inputs.len());
        for input in &self.gpu_inputs {
            prepared_inputs.push(
                super::prepare_gpu_resident_producer_for_execution(Arc::clone(input), Arc::clone(&task_context))
                    .await?,
            );
        }
        prepared.gpu_inputs = prepared_inputs;
        prepared.execution_config = prepared.execution_config.with_lakehouse_footer_pruning(false);
        Ok(prepared)
    }

    /// Streams the host result of a unit without producers.
    fn execute_host_leaf(
        &self,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        // Runtime, allocator, boundary, and memory-control state come from
        // `AlgeonNativeExecutionConfig`; the DataFusion task context scopes
        // backend admission across native fragments in one execution.
        let schema = self.schema();
        let admitted = Arc::clone(&self.admitted);
        let execution_config = self.execution_config.clone();
        let account_memory_pool = execution_config.host_output_memory_pool_accounting();
        let metrics = AlgeonNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        )
        .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
        let output_schema = Arc::clone(&schema);
        let output = native_host_output_stream(
            NATIVE_ALGEON_EXEC_HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                let unresolved_capability_evidence = AttemptCapabilityEvidence::for_query_plan(admitted.plan());
                let admitted = super::lakehouse::resolve_admitted_plan(
                    execution_config.lakehouse_footer_pruning(),
                    &admitted,
                    task_context.as_ref(),
                )
                .await?;
                let plan = admitted.plan();
                if execution_config.lakehouse_footer_pruning()
                    && let Some(ticket) =
                        execution_config.query_admission_ticket_for_task_context(task_context.as_ref())
                {
                    let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
                    if let Some(existing) = ticket.attempt_capability_evidence_for_rebind()
                        && unresolved_capability_evidence
                            .execution_scopes()
                            .iter()
                            .any(|scope| !existing.execution_scopes().iter().any(|bound| bound == scope))
                    {
                        return Err(DataFusionError::from(
                            Error::new(
                                ErrorCode::InvalidExecutionPlan,
                                "algeon_datafusion.native_exec.admission_evidence",
                                "the active native host admission ticket does not attest to this unresolved fragment",
                            )
                            .with_fact("reason", "native_host_admission_evidence_scope_mismatch")
                            .with_fact("ticket_evidence_scope", "different_native_plan"),
                        ));
                    }
                    ticket.bind_fragment_evidence([evidence]).map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "algeon_datafusion.native_exec.admission_evidence",
                            "failed to bind the native host fragment to its attempt admission",
                            error,
                        ))
                    })?;
                }
                // Context construction is the configuration/validation stage. Failures here happen
                // before the native query starts.
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
                            "algeon_datafusion.native_exec.gpu_execution_pool",
                            "query service closed before native execution acquired its GPU pool",
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
                run_gpu_execution_task(
                    gpu_executor,
                    observation,
                    "algeon_datafusion.native_exec.gpu_execution_pool",
                    move |gpu_stream_scope| async move {
                        let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        // Once native execution starts, the attempt is terminal: errors are surfaced with stable
                        // facts by `native_query_runtime_error` and are not replayed through DataFusion.
                        let outcome = execute_query_plan(&context, QueryExecutionRequest::admitted(&admitted));
                        let (output, deferred_metrics) = metrics
                            .finalize_query_outcome_for_deferred_host(outcome, context_lease)
                            .map_err(|error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "algeon_datafusion.native_exec.execute",
                                    "Algeon native query execution failed",
                                    error,
                                ))
                            })?;
                        let QueryOutput::GpuResident(handle) = output else {
                            return Err(DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "algeon_datafusion.native_exec.execute",
                                    "deferred native host execution did not produce a GPU-resident cursor input",
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
                                    "algeon_datafusion.native_exec.host_batch_cursor",
                                    "failed to bind and create deferred native host-output cursor",
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

    /// Executes a producer without inputs inside a composed root's shared
    /// native context. The result stays on the GPU and is returned under its
    /// input name, so the consumer binds it without host materialization.
    fn execute_leaf_gpu_result(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        admitted: &AdmittedPlan,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_gpu_producer.execute_gpu",
                "native GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let metrics = AlgeonNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        );
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let outcome = execute_query_plan(native_context, QueryExecutionRequest::admitted(admitted));
        let output = metrics.finalize_query_outcome(outcome).map_err(|error| {
            DataFusionError::from(native_query_runtime_error(
                "algeon_datafusion.native_gpu_producer.execute_gpu",
                "algeon native GPU-resident fragment execution failed",
                error,
            ))
        })?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_gpu_producer.execute_gpu",
                    "native GPU-resident producer materialized a host result",
                )
                .with_fact("result_sink", "host_arrow"),
            ));
        };
        Ok((self.gpu_result_input_name().to_string(), handle))
    }
}

fn result_sink_label(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
        ResultSink::DeferredHostArrow { .. } => "deferred_host_arrow",
    }
}

/// Checks that the producer children supply exactly the GPU-resident sources
/// the plan declares, each under a distinct input name.
fn validate_gpu_inputs(plan: &QueryPlan, gpu_inputs: &[Arc<dyn ExecutionPlan>], operation: &'static str) -> Result<()> {
    let required_inputs = gpu_resident_source_names(plan);
    if gpu_inputs.is_empty() {
        if required_inputs.is_empty() {
            return Ok(());
        }
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            operation,
            "native execution is missing the GPU-resident producers its plan reads",
        )
        .with_fact("gpu_resident_source_count", required_inputs.len())
        .with_fact("gpu_resident_sources", comma_separated(required_inputs.iter().map(String::as_str))));
    }
    let mut provided_inputs = BTreeSet::new();
    for child in gpu_inputs {
        let Some(input_name) = gpu_resident_producer_input_name(child.as_ref()) else {
            if let Some(rejection) =
                graph_execution_plan(child.as_ref()).and_then(|graph| graph.gpu_resident_output_rejection())
            {
                return Err(rejection);
            }
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
            "composed native execution received GPU-resident producer inputs the plan does not declare",
        )
        .with_fact("unused_inputs", unused_inputs.join(",")));
    }
    Ok(())
}

impl GpuResidentProducer for AlgeonNativeExec {
    /// The name of a GPU-resident unit's output. Only GPU-resident units are
    /// classified as producers, so a host unit never answers.
    fn gpu_result_input_name(&self) -> &str {
        self.gpu_resident_input_name().unwrap_or_default()
    }

    fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        if self.gpu_inputs.is_empty() {
            return Ok(AttemptCapabilityEvidence::for_query_plan(self.native_plan()));
        }
        super::composed::resolve_composed_admission_capability_evidence(self.native_plan(), &self.gpu_inputs).map_err(
            |error| {
                Error::from_native(
                    "algeon_datafusion.native_composed_gpu_producer.admission_evidence",
                    "failed to derive capability evidence for a composed GPU-resident output",
                    error,
                )
            },
        )
    }

    fn hard_output_row_upper_bound(&self) -> Option<u64> {
        self.gpu_inputs.is_empty().then(|| query_plan_hard_row_upper_bound(self.native_plan())).flatten()
    }

    fn execution_metrics(&self) -> Option<&ExecutionPlanMetricsSet> {
        Some(&self.metrics)
    }

    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        if self.gpu_inputs.is_empty() {
            return Vec::new();
        }
        self.composed_prepared_admission_tickets()
    }

    fn prepare_execution_plan<'a>(
        &'a self,
        plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            if self.gpu_inputs.is_empty() && !self.execution_config.lakehouse_footer_pruning() {
                return Ok(plan);
            }
            let mut prepared = self.prepare_lakehouse_metadata(task_context).await?;
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
            if !self.gpu_inputs.is_empty() {
                return self.execute_composed_gpu_result(native_context, partition, task_context).await;
            }
            let admitted = super::lakehouse::resolve_admitted_plan(
                self.execution_config.lakehouse_footer_pruning(),
                &self.admitted,
                task_context.as_ref(),
            )
            .await?;
            self.execute_leaf_gpu_result(native_context, partition, &admitted)
        })
    }
}

impl DisplayAs for AlgeonNativeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plan = self.native_plan();
        let shape = self.shape();
        let input_name = self.gpu_resident_input_name().unwrap_or_default();
        let gpu_input_count = self.gpu_inputs.len();
        let summary = match shape {
            Shape::Host => format!(
                "native_nodes={}, sink=host_arrow, native_pipeline_execution_mode=bounded, output_boundary=device_to_host, edge_normalize_inserted={}",
                plan.node_count(),
                plan.contains_edge_normalize(),
            ),
            Shape::Composed => format!(
                "native_nodes={}, gpu_input_count={gpu_input_count}, sink=host_arrow, native_pipeline_execution_mode=bounded, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}",
                plan.node_count(),
                plan.contains_edge_normalize(),
            ),
            Shape::Producer => format!(
                "input_name={input_name}, native_nodes={}, sink=gpu_resident, output_boundary=device_resident, edge_normalize_inserted={}",
                plan.node_count(),
                plan.contains_edge_normalize(),
            ),
            Shape::ComposedProducer => format!(
                "input_name={input_name}, native_nodes={}, gpu_input_count={gpu_input_count}, sink=gpu_resident, input_boundary=gpu_resident, output_boundary=device_resident",
                plan.node_count(),
            ),
        };
        match t {
            DisplayFormatType::Default => write!(f, "{}: {summary}", shape.name()),
            DisplayFormatType::Verbose => {
                write!(f, "{}: {summary}, native_plan={}", shape.name(), native_plan_one_line(plan))
            },
            DisplayFormatType::TreeRender => {
                for field in summary.split(", ") {
                    writeln!(f, "{field}")?;
                }
                write_native_plan_tree(f, plan)
            },
        }
    }
}

impl ExecutionPlan for AlgeonNativeExec {
    fn name(&self) -> &str {
        self.shape().name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.gpu_inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // Keep DataFusion distribution rules from inserting a row-stream
        // wrapper between this unit and its device inputs.
        vec![false; self.gpu_inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if self.gpu_inputs.is_empty() {
            if !children.is_empty() {
                return Err(Error::new(
                    ErrorCode::InvalidChildCount,
                    self.shape().with_new_children_operation(),
                    "native execution without GPU-resident inputs expected no children",
                )
                .with_fact("actual_child_count", children.len())
                .into());
            }
            return Ok(self);
        }
        Self::try_new_with_metrics_capture(
            Arc::clone(&self.admitted),
            children,
            self.output.clone(),
            self.execution_config.clone(),
            Some(self.cache.equivalence_properties().clone()),
            self.metrics_capture.clone(),
        )
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let shape = self.shape();
        let operation = match shape {
            Shape::Host => "algeon_datafusion.native_exec.execute",
            Shape::Composed => "algeon_datafusion.native_composed.execute",
            Shape::Producer => "algeon_datafusion.native_gpu_producer.execute",
            Shape::ComposedProducer => "algeon_datafusion.native_composed_gpu_producer.execute",
        };
        if !self.materializes_host() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                operation,
                "native GPU-resident producer is an internal device boundary and cannot stream host Arrow directly",
            )
            .with_fact("result_sink", "gpu_resident")
            .into());
        }
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                operation,
                "native execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        if shape == Shape::Composed {
            return self.execute_host_composed(partition, task_context);
        }
        self.execute_host_leaf(partition, task_context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
