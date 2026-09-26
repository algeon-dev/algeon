//! Composed native execution node for GPU-resident native fragments.
//!
//! Producer children emit device-resident handles into a shared native context.
//! The consumer plan then uses those handles as GPU inputs and materializes the
//! final result back to host Arrow for DataFusion.

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
#[cfg(feature = "cugraph")]
use query_engine::plan::row_bounds::query_plan_hard_row_upper_bound;
use query_engine::{
    admission::{AdmittedPlan, AttemptCapabilityEvidence, AttemptExecutionScope, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput},
    plan::{DeferredGraphExecutionScopeIdentity, QueryPlan, ResultSink},
    runtime::{
        ExecutionContext,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
};

use super::{
    GpuResidentProducer,
    admission::attempt_execution_required_error,
    common::{
        NATIVE_ALGEON_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, conservative_properties,
        gpu_resident_source_names, mark_native_runtime_error, native_host_output_stream, native_plan_one_line,
        native_query_runtime_error, properties_with_equivalence, run_gpu_execution_task, write_native_plan_tree,
    },
    config::AlgeonNativeExecutionConfig,
    execute_gpu_resident_producer_with_context, gpu_resident_producer, gpu_resident_producer_input_name,
    metrics::{
        AlgeonNativeMetrics, GraphClaimCalibrationCapture, NativeExecutionMetricsCapture, SealedAttemptMetricsTarget,
    },
    prepare_gpu_resident_producer_for_execution,
    producer::prepared_gpu_resident_producer_admission_tickets,
};
use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::graph::graph_execution_plan,
    planner::native::Invocation,
};

/// DataFusion wrapper for composed native execution with GPU-resident producer fragments.
///
/// Created by Algeon physical optimizer and adapter composition. The public type
/// identity is exposed for [`ExecutionPlan`] inspection, downcasting, metrics
/// collection, and plan diagnostics.
///
/// Construction remains crate-private: constructors validate host sinks, child
/// producer relationships, equivalence properties, and admission-bound execution
/// configuration. The wrapper executes producer children into GPU-resident handles,
/// maps those handles into an input map, and executes a host-sink native consumer
/// plan that materializes the final result as Arrow for DataFusion streaming.
#[derive(Debug, Clone)]
pub struct AlgeonNativeComposedExec {
    admitted: Arc<AdmittedPlan>,
    gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    execution_config: AlgeonNativeExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl AlgeonNativeComposedExec {
    #[cfg(feature = "cugraph")]
    pub(crate) fn hard_output_row_upper_bound(&self) -> Option<u64> {
        query_plan_hard_row_upper_bound(self.native_plan())
    }

    /// Create composed native execution from GPU-resident producer children.
    ///
    /// The final plan must use a host Arrow sink, every child must advertise a
    /// unique GPU-resident input name, and the child names must match the
    /// consumer plan's declared GPU sources exactly.
    #[cfg(test)]
    pub(crate) fn try_new(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
    ) -> Result<Self> {
        let admitted = super::admit_for_test(&plan.clone_with_deferred_host_materialization_sink(), &execution_config)?;
        Self::try_new_with_cache(admitted, gpu_inputs, execution_config, None)
    }

    pub(crate) fn try_new_with_equivalence_properties(
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> Result<Self> {
        Self::try_new_with_cache(admitted, gpu_inputs, execution_config, Some(eq_properties))
    }

    fn try_new_with_cache(
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
    ) -> Result<Self> {
        // Constructor validation preserves the composed execution contract: the final consumer must
        // host-materialize, every child must be a GPU-resident producer, and required/provided GPU
        // input names must match exactly. The consumer executes in the deferred form it was
        // admitted in.
        let plan = admitted.plan();
        if !matches!(plan.sink(), ResultSink::DeferredHostArrow { .. }) {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed.try_new",
                "native composed DataFusion execution requires a host-arrow result sink",
            )
            .with_fact("result_sink", "gpu_resident"));
        }
        if gpu_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed.try_new",
                "native composed execution requires at least one GPU-resident producer",
            ));
        }

        let required_inputs = gpu_resident_source_names(plan);
        let mut provided_inputs = BTreeSet::new();
        for child in &gpu_inputs {
            let Some(input_name) = gpu_resident_producer_input_name(child.as_ref()) else {
                if let Some(rejection) =
                    graph_execution_plan(child.as_ref()).and_then(|graph| graph.gpu_resident_output_rejection())
                {
                    return Err(adapter_error_from_graph_rejection(
                        rejection,
                        "algeon_datafusion.native_composed.try_new",
                    ));
                }
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_composed.try_new",
                    "native composed execution children must be GPU-resident native producers",
                )
                .with_fact("child_name", child.name()));
            };
            if !provided_inputs.insert(input_name.clone()) {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_composed.try_new",
                    "native composed execution received duplicate GPU-resident producer input names",
                )
                .with_fact("input_name", input_name));
            }
        }

        let missing_inputs = required_inputs.difference(&provided_inputs).map(String::as_str).collect::<Vec<_>>();
        if !missing_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed.try_new",
                "native composed execution is missing GPU-resident producer inputs",
            )
            .with_fact("missing_inputs", missing_inputs.join(",")));
        }
        let unused_inputs = provided_inputs.difference(&required_inputs).map(String::as_str).collect::<Vec<_>>();
        if !unused_inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed.try_new",
                "native composed execution received GPU-resident producer inputs the consumer plan does not declare",
            )
            .with_fact("unused_inputs", unused_inputs.join(",")));
        }
        let cache = match eq_properties {
            Some(eq_properties) => {
                properties_with_equivalence(Arc::clone(plan.caller_visible_output_schema()), eq_properties)?
            },
            None => conservative_properties(Arc::clone(plan.caller_visible_output_schema())),
        };

        let metrics_capture = execution_config.new_metrics_capture();
        let gpu_inputs = gpu_inputs
            .into_iter()
            .map(|input| {
                let Some(producer) = input.as_ref().downcast_ref::<super::producer::AlgeonNativeGpuProducerExec>()
                else {
                    return input;
                };
                Arc::new(producer.clone().with_metrics_capture(metrics_capture.clone())) as Arc<dyn ExecutionPlan>
            })
            .collect();
        Ok(Self {
            cache,
            admitted,
            gpu_inputs,
            execution_config,
            metrics_capture,
            graph_claim_calibration_capture: GraphClaimCalibrationCapture::default(),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        self.admitted.plan()
    }

    /// The artifact planning admitted for this node.
    #[cfg(feature = "cugraph")]
    pub(crate) fn admitted(&self) -> &Arc<AdmittedPlan> {
        &self.admitted
    }

    #[cfg(feature = "bench-support")]
    #[doc(hidden)]
    pub fn admitted_plan_for_internal_consumer(&self) -> &Arc<AdmittedPlan> {
        &self.admitted
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn execution_config(&self) -> &AlgeonNativeExecutionConfig {
        &self.execution_config
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn gpu_inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.gpu_inputs
    }

    pub(crate) fn execution_metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
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

    #[cfg(any(feature = "cugraph", test))]
    pub(crate) async fn prepare_lakehouse_metadata(&self, task_context: Arc<TaskContext>) -> DataFusionResult<Self> {
        let mut prepared = self.clone();
        prepared.admitted = super::lakehouse::resolve_admitted_plan(
            self.execution_config.lakehouse_footer_pruning(),
            self.execution_config.source_storage_profiles(),
            &self.admitted,
            task_context.as_ref(),
        )
        .await?;
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

fn adapter_error_from_graph_rejection(rejection: DataFusionError, operation: &'static str) -> Error {
    match rejection {
        DataFusionError::External(source) => match source.downcast::<Error>() {
            Ok(error) => *error,
            Err(source) => Error::new(
                ErrorCode::InvalidExecutionPlan,
                operation,
                "graph execution rejected the GPU-resident output boundary",
            )
            .with_boxed_source(source),
        },
        rejection => Error::new(
            ErrorCode::InvalidExecutionPlan,
            operation,
            "graph execution rejected the GPU-resident output boundary",
        )
        .with_source(rejection),
    }
}

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
            QueryErrorCode::UnsupportedPlanShape,
            "algeon_datafusion.native_composed.admission_evidence",
            "a prepared composed input is not a GPU-resident producer",
        )
        .with_fact("reason", "composed_admission_producer_missing")
    })?;
    let output_evidence = producer.gpu_result_capability_evidence().map_err(|error| {
        QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "algeon_datafusion.native_composed.admission_evidence",
            "a prepared composed producer did not expose its output capability evidence",
        )
        .with_fact("reason", "composed_admission_producer_evidence_missing")
        .with_source(error)
    })?;
    let output_scope = output_evidence.execution_scopes().last().cloned().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
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
) -> DataFusionResult<AttemptCapabilityEvidence> {
    resolve_composed_admission_capability_evidence(consumer_plan, prepared_gpu_inputs)
        .map_err(|error| prepared_composed_admission_error(error, consumer_plan, None))
}

fn resolve_composed_admission_capability_evidence(
    consumer_plan: &QueryPlan,
    prepared_gpu_inputs: &[Arc<dyn ExecutionPlan>],
) -> QueryEngineResult<AttemptCapabilityEvidence> {
    let mut producer_inputs = Vec::with_capacity(prepared_gpu_inputs.len());
    for input in prepared_gpu_inputs {
        let input_name = gpu_resident_producer_input_name(input.as_ref()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
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
) -> DataFusionError {
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
    DataFusionError::from(Error::from_native_with_blocking_diagnostics(
        "algeon_datafusion.native_composed.admission_evidence",
        "failed to derive the complete engine-owned composed admission bound",
        error,
    ))
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
) -> DataFusionResult<CompletedComposedAdmissionEvidence> {
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
        .with_fact("reason", "composed_producer_admission_backend_mismatch")
        .into());
    }

    if let Some(active_ticket) = canonical_ticket.as_ref()
        && !source.same_backend(&active_ticket.source())
    {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_composed.admission_evidence",
            "the composed root and its active admission ticket belong to different backends",
        )
        .with_fact("reason", "composed_active_admission_backend_mismatch")
        .into());
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
        active_ticket.extend_capability_evidence_before_admission([complete_evidence]).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.native_composed.admission_evidence",
                "failed to rebind the active composed admission ticket before query-service admission",
                error,
            ))
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
            ticket.extend_capability_evidence_before_admission([complete_evidence]).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "algeon_datafusion.native_composed.admission_evidence",
                    "failed to extend the invocation admission ticket before query-service admission",
                    error,
                ))
            })?;
            ticket
        },
        None => {
            return Err(attempt_execution_required_error("algeon_datafusion.native_composed.admission_evidence").into());
        },
    };
    Ok(CompletedComposedAdmissionEvidence {
        execution_config: execution_config.with_existing_query_admission_ticket(ticket),
    })
}

impl DisplayAs for AlgeonNativeComposedExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "AlgeonNativeComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode=bounded, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}",
                self.native_plan().node_count(),
                self.gpu_inputs.len(),
                self.native_plan().contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "AlgeonNativeComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode=bounded, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}, native_plan={}",
                self.native_plan().node_count(),
                self.gpu_inputs.len(),
                self.native_plan().contains_edge_normalize(),
                native_plan_one_line(self.native_plan()),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.native_plan().node_count())?;
                writeln!(f, "gpu_input_count={}", self.gpu_inputs.len())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(f, "native_pipeline_execution_mode=bounded")?;
                writeln!(f, "input_boundary=gpu_resident")?;
                writeln!(f, "output_boundary=device_to_host")?;
                writeln!(f, "edge_normalize_inserted={}", self.native_plan().contains_edge_normalize())?;
                write_native_plan_tree(f, self.native_plan())
            },
        }
    }
}

impl ExecutionPlan for AlgeonNativeComposedExec {
    fn name(&self) -> &str {
        "AlgeonNativeComposedExec"
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
        // These children are device-handle producers, not DataFusion row
        // streams. Repartitioning or coalescing one would wrap the producer
        // in a host execution node and break the composed GPU contract.
        vec![false; self.gpu_inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        AlgeonNativeComposedExec::try_new_with_equivalence_properties(
            Arc::clone(&self.admitted),
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
                    execution_config.source_storage_profiles(),
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
                let cursor_metrics = metrics.clone();
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
                        let output = metrics.finalize_query_outcome_for_deferred_host(outcome).map_err(|error| {
                            DataFusionError::from(native_query_runtime_error(
                                "algeon_datafusion.native_composed.execute",
                                "algeon native composed query execution failed",
                                error,
                            ))
                        })?;
                        metrics
                            .record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
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

/// Composed GPU-resident native fragment producer with nested GPU-resident producer inputs.
///
/// Created by Algeon physical optimizer and adapter composition. The public type
/// identity is exposed for [`ExecutionPlan`] inspection, downcasting, metrics
/// collection, and plan diagnostics.
///
/// Construction remains crate-private: constructors validate GPU-resident sinks,
/// nested producer children, equivalence properties, GPU-resident input names,
/// and admission-bound execution configuration. This node produces device-resident
/// intermediate data and cannot directly stream Arrow through
/// [`ExecutionPlan::execute`].
#[derive(Debug, Clone)]
pub struct AlgeonNativeComposedGpuProducerExec {
    input_name: String,
    admitted: Arc<AdmittedPlan>,
    gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    execution_config: AlgeonNativeExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    prepared: bool,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl AlgeonNativeComposedGpuProducerExec {
    pub(crate) fn try_new_with_equivalence_properties(
        input_name: impl Into<String>,
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> Result<Self> {
        let metrics_capture = execution_config.new_metrics_capture();
        Self::try_new(input_name, admitted, gpu_inputs, execution_config, metrics_capture, eq_properties)
    }

    fn try_new(
        input_name: impl Into<String>,
        admitted: Arc<AdmittedPlan>,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: AlgeonNativeExecutionConfig,
        metrics_capture: Option<NativeExecutionMetricsCapture>,
        eq_properties: EquivalenceProperties,
    ) -> Result<Self> {
        let input_name = input_name.into();
        if input_name.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed_gpu_producer.try_new",
                "composed GPU-resident producer input name must not be empty",
            ));
        }
        let plan = admitted.plan();
        if plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_composed_gpu_producer.try_new",
                "composed GPU-resident producer requires a GPU-resident sink",
            )
            .with_fact("result_sink", "host_arrow"));
        }
        validate_composed_gpu_inputs(plan, &gpu_inputs, "algeon_datafusion.native_composed_gpu_producer.try_new")?;
        let cache = properties_with_equivalence(Arc::clone(plan.output_schema()), eq_properties)?;
        Ok(Self {
            input_name,
            admitted,
            gpu_inputs,
            execution_config,
            metrics_capture,
            prepared: false,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        self.admitted.plan()
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn execution_config(&self) -> &AlgeonNativeExecutionConfig {
        &self.execution_config
    }
}

impl GpuResidentProducer for AlgeonNativeComposedGpuProducerExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.input_name
    }

    fn gpu_result_capability_evidence(&self) -> DataFusionResult<AttemptCapabilityEvidence> {
        resolve_composed_admission_capability_evidence(self.native_plan(), &self.gpu_inputs).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.native_composed_gpu_producer.admission_evidence",
                "failed to derive capability evidence for a composed GPU-resident output",
                error,
            ))
        })
    }

    fn execution_metrics(&self) -> Option<&ExecutionPlanMetricsSet> {
        Some(&self.metrics)
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
            prepared.admitted = super::lakehouse::resolve_admitted_plan(
                self.execution_config.lakehouse_footer_pruning(),
                self.execution_config.source_storage_profiles(),
                &self.admitted,
                task_context.as_ref(),
            )
            .await?;
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
            metrics.record_lakehouse_cache_fill_events(&native_context.drain_lakehouse_cache_fill_event_snapshots());
            let QueryOutput::GpuResident(handle) = output else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_composed_gpu_producer.execute_gpu",
                    "composed GPU-resident producer materialized a host result",
                )
                .with_fact("result_sink", "host_arrow")
                .into());
            };
            Ok((self.input_name.clone(), handle))
        })
    }
}

impl DisplayAs for AlgeonNativeComposedGpuProducerExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "AlgeonNativeComposedGpuProducerExec: input_name={}, native_nodes={}, gpu_input_count={}, sink=gpu_resident, input_boundary=gpu_resident, output_boundary=device_resident",
                self.input_name,
                self.native_plan().node_count(),
                self.gpu_inputs.len(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "AlgeonNativeComposedGpuProducerExec: input_name={}, native_nodes={}, gpu_input_count={}, sink=gpu_resident, input_boundary=gpu_resident, output_boundary=device_resident, native_plan={}",
                self.input_name,
                self.native_plan().node_count(),
                self.gpu_inputs.len(),
                native_plan_one_line(self.native_plan()),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "input_name={}", self.input_name)?;
                writeln!(f, "gpu_input_count={}", self.gpu_inputs.len())?;
                writeln!(f, "sink=gpu_resident")?;
                writeln!(f, "input_boundary=gpu_resident")?;
                writeln!(f, "output_boundary=device_resident")?;
                write_native_plan_tree(f, self.native_plan())
            },
        }
    }
}

impl ExecutionPlan for AlgeonNativeComposedGpuProducerExec {
    fn name(&self) -> &str {
        "AlgeonNativeComposedGpuProducerExec"
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
        // wrapper between this internal producer and its device inputs.
        vec![false; self.gpu_inputs.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        AlgeonNativeComposedGpuProducerExec::try_new(
            self.input_name.clone(),
            Arc::clone(&self.admitted),
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
            "algeon_datafusion.native_composed_gpu_producer.execute",
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
) -> Result<()> {
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
            if let Some(rejection) =
                graph_execution_plan(child.as_ref()).and_then(|graph| graph.gpu_resident_output_rejection())
            {
                return Err(adapter_error_from_graph_rejection(rejection, operation));
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
            "composed native execution received unused GPU-resident producer inputs",
        )
        .with_fact("unused_inputs", unused_inputs.join(",")));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
