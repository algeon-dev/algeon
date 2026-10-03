//! Host-materializing native DataFusion execution node.
//!
//! This wrapper is the normal native execution boundary for DataFusion: it
//! executes a DataFusion-free native plan and streams the final result as host
//! Arrow batches.

use std::sync::Arc;

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
use query_engine::{
    admission::{AdmittedPlan, AttemptCapabilityEvidence},
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput},
    plan::{QueryPlan, ResultSink},
};

use super::{
    common::{
        NATIVE_ALGEON_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, conservative_properties,
        mark_native_runtime_error, native_host_output_stream, native_plan_one_line, native_query_runtime_error,
        properties_with_equivalence, run_gpu_execution_task, write_native_plan_tree,
    },
    config::AlgeonNativeExecutionConfig,
    metrics::{
        AlgeonNativeMetrics, GraphClaimCalibrationCapture, NativeExecutionMetricsCapture, SealedAttemptMetricsTarget,
    },
};
use crate::error::{Error, ErrorCode, Result};

/// Leaf DataFusion execution plan for a host-materializing native cuDF query.
///
/// Created by Algeon physical optimizer and adapter composition. The public type
/// identity is exposed for [`ExecutionPlan`] inspection, downcasting, metrics
/// collection, and plan diagnostics.
///
/// Construction remains crate-private: the constructor validates host-materializing
/// sinks, child topologies, equivalence properties, and admission-bound execution
/// configuration. `AlgeonNativeExec` has no children and exposes exactly one output
/// partition. It executes a DataFusion-free native `QueryPlan`, then materializes
/// the final GPU result back to host Arrow for DataFusion streaming.
#[derive(Debug, Clone)]
pub struct AlgeonNativeExec {
    admitted: Arc<AdmittedPlan>,
    execution_config: AlgeonNativeExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl AlgeonNativeExec {
    /// Create a native DataFusion execution plan.
    ///
    /// GPU-resident sinks are rejected here because regular `ExecutionPlan::execute` must stream
    /// host Arrow `RecordBatch` output to DataFusion.
    #[cfg(test)]
    pub(crate) fn try_new(plan: QueryPlan, execution_config: AlgeonNativeExecutionConfig) -> Result<Self> {
        let admitted = super::admit_for_test(&plan.clone_with_deferred_host_materialization_sink(), &execution_config)?;
        Self::try_new_with_cache(admitted, execution_config, None)
    }

    pub(crate) fn try_new_with_equivalence_properties(
        admitted: Arc<AdmittedPlan>,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> Result<Self> {
        Self::try_new_with_cache(admitted, execution_config, Some(eq_properties))
    }

    fn try_new_with_cache(
        admitted: Arc<AdmittedPlan>,
        execution_config: AlgeonNativeExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
    ) -> Result<Self> {
        // The node executes the plan in the deferred form it was admitted in
        // and materializes the host batches itself.
        let plan = admitted.plan();
        if !matches!(plan.sink(), ResultSink::DeferredHostArrow { .. }) {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_exec.try_new",
                "native DataFusion execution requires a deferred host-arrow result sink",
            )
            .with_fact(
                "result_sink",
                match plan.sink() {
                    ResultSink::HostArrow { .. } => "host_arrow",
                    ResultSink::GpuResident { .. } => "gpu_resident",
                    ResultSink::DeferredHostArrow { .. } => "deferred_host_arrow",
                },
            ));
        }
        let cache = match eq_properties {
            Some(eq_properties) => {
                properties_with_equivalence(Arc::clone(plan.caller_visible_output_schema()), eq_properties)?
            },
            None => conservative_properties(Arc::clone(plan.caller_visible_output_schema())),
        };

        let metrics_capture = execution_config.new_metrics_capture();
        Ok(Self {
            cache,
            admitted,
            execution_config,
            metrics_capture,
            graph_claim_calibration_capture: GraphClaimCalibrationCapture::default(),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        self.admitted.plan()
    }

    #[cfg(feature = "bench-support")]
    #[doc(hidden)]
    pub fn admitted_plan_for_internal_consumer(&self) -> &Arc<AdmittedPlan> {
        &self.admitted
    }

    #[cfg(all(test, feature = "cugraph"))]
    pub(crate) fn execution_config(&self) -> &AlgeonNativeExecutionConfig {
        &self.execution_config
    }

    /// Returns the one post-seal attempt projection target for this wrapper.
    /// The attempt owner elects it before execution; the wrapper itself never
    /// reads or retains terminal attempt state.
    pub(crate) fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&self.metrics, self.graph_claim_calibration_capture.clone())
    }

    pub(crate) fn execution_metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    #[cfg(feature = "cugraph")]
    pub(crate) async fn prepare_lakehouse_metadata(&self, task_context: &TaskContext) -> Result<Self> {
        let mut prepared = self.clone();
        prepared.admitted = super::lakehouse::resolve_admitted_plan(
            self.execution_config.lakehouse_footer_pruning(),
            &self.admitted,
            task_context,
        )
        .await?;
        prepared.execution_config = prepared.execution_config.with_lakehouse_footer_pruning(false);
        Ok(prepared)
    }
}

impl DisplayAs for AlgeonNativeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "AlgeonNativeExec: native_nodes={}, sink=host_arrow, native_pipeline_execution_mode=bounded, output_boundary=device_to_host, edge_normalize_inserted={}",
                self.native_plan().node_count(),
                self.native_plan().contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "AlgeonNativeExec: native_nodes={}, sink=host_arrow, native_pipeline_execution_mode=bounded, output_boundary=device_to_host, edge_normalize_inserted={}, native_plan={}",
                self.native_plan().node_count(),
                self.native_plan().contains_edge_normalize(),
                native_plan_one_line(self.native_plan()),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.native_plan().node_count())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(f, "native_pipeline_execution_mode=bounded")?;
                writeln!(f, "output_boundary=device_to_host")?;
                writeln!(f, "edge_normalize_inserted={}", self.native_plan().contains_edge_normalize())?;
                write_native_plan_tree(f, self.native_plan())
            },
        }
    }
}

impl ExecutionPlan for AlgeonNativeExec {
    fn name(&self) -> &str {
        "AlgeonNativeExec"
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
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "algeon_datafusion.native_exec.with_new_children",
                "algeon native execution expected no children",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_exec.execute",
                "algeon native execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
