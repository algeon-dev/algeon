//! Host-materializing native DataFusion execution node.
//!
//! This wrapper is the normal native execution boundary for DataFusion: it
//! executes a DataFusion-free native plan and streams the final result as host
//! Arrow batches.

use std::sync::Arc;

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
use nexus_query_engine::{
    exec::{execute_query_plan_observed, output::QueryOutput},
    plan::QueryPlan,
    runtime::resident::GpuResultHandle,
};

use super::{
    common::{
        NATIVE_CUDF_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, NativeHostOutputCursor, conservative_properties,
        mark_native_runtime_error, native_host_output_stream, native_plan_one_line, native_query_runtime_error,
        properties_with_equivalence, run_gpu_execution_task, write_native_plan_tree,
    },
    config::NativeCudfExecutionConfig,
    metrics::{NativeCudfMetrics, NativeExecutionMetricsCapture},
};
use crate::error::{Error, ErrorCode};

/// Leaf DataFusion execution plan for a host-materializing native cuDF query.
///
/// `NativeCudfExec` has no children and exposes exactly one output partition. It executes a
/// DataFusion-free native `QueryPlan`, then materializes the final GPU result back to host Arrow
/// because normal DataFusion execution consumes a `RecordBatch` stream.
#[derive(Debug, Clone)]
pub struct NativeCudfExec {
    plan: QueryPlan,
    execution_config: NativeCudfExecutionConfig,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfExec {
    /// Create a native DataFusion execution plan.
    ///
    /// GPU-resident sinks are rejected here because regular `ExecutionPlan::execute` must stream
    /// host Arrow `RecordBatch` output to DataFusion.
    pub fn try_new(plan: QueryPlan, execution_config: NativeCudfExecutionConfig) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, execution_config, None)
    }

    pub(crate) fn try_new_with_equivalence_properties(
        plan: QueryPlan,
        execution_config: NativeCudfExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, execution_config, Some(eq_properties))
    }

    fn try_new_with_cache(
        plan: QueryPlan,
        execution_config: NativeCudfExecutionConfig,
        eq_properties: Option<EquivalenceProperties>,
    ) -> crate::error::Result<Self> {
        if !plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_exec.try_new",
                "native DataFusion execution requires a host-arrow result sink",
            )
            .with_fact("result_sink", "gpu_resident"));
        }
        let cache = match eq_properties {
            Some(eq_properties) => properties_with_equivalence(Arc::clone(plan.output_schema()), eq_properties)?,
            None => conservative_properties(Arc::clone(plan.output_schema())),
        };

        let metrics_capture = execution_config.new_metrics_capture();
        Ok(Self { cache, plan, execution_config, metrics_capture, metrics: ExecutionPlanMetricsSet::new() })
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn execution_config(&self) -> &NativeCudfExecutionConfig {
        &self.execution_config
    }

    #[allow(dead_code)] // Standalone callers can drain the wrapper-owned local collector here.
    pub(crate) fn metrics_capture(&self) -> Option<NativeExecutionMetricsCapture> {
        self.metrics_capture.clone()
    }

    pub(crate) async fn prepare_lakehouse_metadata(&self, task_context: &TaskContext) -> Self {
        if !self.execution_config.lakehouse_footer_pruning() {
            return self.clone();
        }
        let mut prepared = self.clone();
        prepared.plan =
            super::lakehouse::resolve_query_plan(&self.execution_config, self.plan.clone(), task_context).await;
        prepared.execution_config = prepared.execution_config.with_lakehouse_footer_pruning(false);
        prepared
    }
}

impl DisplayAs for NativeCudfExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "NativeCudfExec: native_nodes={}, sink=host_arrow, native_pipeline_execution_mode={}, memory_control_policy={}, boundary_policy={}, output_boundary=device_to_host, edge_normalize_inserted={}",
                self.plan.node_count(),
                self.execution_config.native_pipeline_execution_mode(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfExec: native_nodes={}, sink=host_arrow, native_pipeline_execution_mode={}, memory_control_policy={}, boundary_policy={}, output_boundary=device_to_host, edge_normalize_inserted={}, native_plan={}",
                self.plan.node_count(),
                self.execution_config.native_pipeline_execution_mode(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
                native_plan_one_line(&self.plan),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(
                    f,
                    "native_pipeline_execution_mode={}",
                    self.execution_config.native_pipeline_execution_mode()
                )?;
                writeln!(f, "memory_control_policy={}", self.execution_config.memory_control_policy())?;
                writeln!(f, "boundary_policy={}", self.execution_config.boundary_policy())?;
                writeln!(f, "output_boundary=device_to_host")?;
                writeln!(f, "edge_normalize_inserted={}", self.plan.contains_edge_normalize())?;
                write_native_plan_tree(f, &self.plan)
            },
        }
    }
}

impl ExecutionPlan for NativeCudfExec {
    fn name(&self) -> &str {
        "NativeCudfExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
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
                "datafusion_nexus.native_exec.with_new_children",
                "native cudf execution expected no children",
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
                "datafusion_nexus.native_exec.execute",
                "native cudf execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        // Runtime, allocator, boundary, and memory-control state come from
        // `NativeCudfExecutionConfig`; the DataFusion task context scopes
        // backend admission across native fragments in one execution.
        let schema = self.schema();
        let plan = self.plan.clone();
        let execution_config = self.execution_config.clone();
        let runtime_observation_config = execution_config.runtime_observation_config();
        let account_memory_pool = execution_config.host_output_memory_pool_accounting();
        let metrics = NativeCudfMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        );
        let output_schema = Arc::clone(&schema);
        let output = native_host_output_stream(
            NATIVE_CUDF_EXEC_HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                let plan = super::lakehouse::resolve_query_plan(&execution_config, plan, task_context.as_ref()).await;
                // Context construction is the configuration/validation stage. Failures here happen
                // before the native query starts.
                let context_lease = execution_config
                    .execution_context_async(task_context)
                    .await
                    .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
                if let Some(admission_facts) = context_lease.admission_facts() {
                    metrics.record_admission_facts(admission_facts);
                }
                let gpu_execution_pool = context_lease.context().gpu_execution_pool().map_err(|error| {
                    DataFusionError::from(mark_native_runtime_error(
                        Error::from_native(
                            "datafusion_nexus.native_exec.gpu_execution_pool",
                            "query service closed before native execution acquired its GPU pool",
                            error,
                        ),
                        "gpu_execution_pool",
                    ))
                })?;
                let observation = context_lease.context().runtime_observation().clone();
                let cursor_pool = gpu_execution_pool.clone();
                let cursor_observation = observation.clone();
                let cursor_metrics = metrics.clone();
                run_gpu_execution_task(
                    gpu_execution_pool,
                    observation,
                    "datafusion_nexus.native_exec.gpu_execution_pool",
                    move |gpu_stream_scope| async move {
                        let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                        let gpu_plan = plan.clone_with_deferred_host_materialization_sink();
                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        // Once native execution starts there is no CPU fallback; errors are surfaced with stable
                        // facts by `native_query_runtime_error`.
                        let outcome = execute_query_plan_observed(&context, &gpu_plan, runtime_observation_config);
                        let output = metrics.finalize_query_outcome_for_deferred_host(outcome).map_err(|error| {
                            DataFusionError::from(native_query_runtime_error(
                                "datafusion_nexus.native_exec.execute",
                                "native cudf query execution failed",
                                error,
                            ))
                        })?;
                        metrics
                            .record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                        let QueryOutput::GpuResident(handle) = output else {
                            return Err(DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "datafusion_nexus.native_exec.execute",
                                    "deferred native host execution did not produce a GPU-resident cursor input",
                                )
                                .with_fact("result_sink", "host_arrow"),
                            ));
                        };
                        let handle = GpuResultHandle::new(&context, handle.into_gpu_frame(), output_schema).map_err(
                            |error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "datafusion_nexus.native_exec.host_batch_cursor",
                                    "failed to bind deferred native output to the DataFusion host schema",
                                    error,
                                ))
                            },
                        )?;
                        let cursor = handle
                            .into_host_batch_cursor(context.cancellation().clone(), HostBatchCursorConfig::default())
                            .map_err(|error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "datafusion_nexus.native_exec.host_batch_cursor",
                                    "failed to create deferred native host-output cursor",
                                    error,
                                ))
                            })?;
                        Ok::<NativeHostOutputCursor, DataFusionError>(NativeHostOutputCursor::new(
                            cursor,
                            cursor_pool,
                            cursor_observation,
                            cursor_metrics,
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
