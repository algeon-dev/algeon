//! Composed native execution node for GPU-resident native fragments.
//!
//! Producer children emit device-resident handles into a shared native context.
//! The consumer plan then uses those handles as GPU inputs and materializes the
//! final result back to host Arrow for DataFusion.

use std::{collections::BTreeSet, sync::Arc};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
    },
};
use nexus_query_engine::{GpuResidentInputMap, QueryOutput, QueryPlan, execute_query_plan_with_gpu_inputs};

use super::{
    common::{
        NATIVE_CUDF_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER, conservative_properties, gpu_resident_source_names,
        mark_native_runtime_error, native_host_output_stream, native_plan_one_line, native_query_runtime_error,
        properties_with_equivalence, run_gpu_execution_task, write_native_plan_tree,
    },
    config::NativeCudfExecutionConfig,
    metrics::NativeCudfMetrics,
    producer::{execute_gpu_resident_producer_with_context, gpu_resident_producer_input_name},
};
use crate::error::{Error, ErrorCode};

/// DataFusion wrapper for composed native execution with GPU-resident producer fragments.
///
/// The wrapper executes producer children into `GpuResultHandle`s, maps those handles into a
/// `GpuResidentInputMap`, then runs a host-sink native consumer plan that materializes the final
/// result as Arrow for DataFusion.
#[derive(Debug, Clone)]
pub struct NativeCudfComposedExec {
    plan: QueryPlan,
    gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    execution_config: NativeCudfExecutionConfig,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfComposedExec {
    /// Create composed native execution from GPU-resident producer children.
    ///
    /// The final plan must use a host Arrow sink, every child must advertise a
    /// unique GPU-resident input name, and the child names must match the
    /// consumer plan's declared GPU sources exactly.
    pub fn try_new(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NativeCudfExecutionConfig,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, gpu_inputs, execution_config, None)
    }

    pub(crate) fn try_new_with_equivalence_properties(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NativeCudfExecutionConfig,
        eq_properties: EquivalenceProperties,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_cache(plan, gpu_inputs, execution_config, Some(eq_properties))
    }

    fn try_new_with_cache(
        plan: QueryPlan,
        gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
        execution_config: NativeCudfExecutionConfig,
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

        Ok(Self { cache, plan, gpu_inputs, execution_config, metrics: ExecutionPlanMetricsSet::new() })
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn execution_config(&self) -> &NativeCudfExecutionConfig {
        &self.execution_config
    }

    #[cfg_attr(not(feature = "cugraph"), allow(dead_code))]
    pub(crate) fn gpu_inputs(&self) -> &[Arc<dyn ExecutionPlan>] {
        &self.gpu_inputs
    }
}

impl DisplayAs for NativeCudfComposedExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "NativeCudfComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode={}, memory_control_policy={}, boundary_policy={}, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}",
                self.plan.node_count(),
                self.gpu_inputs.len(),
                self.execution_config.native_pipeline_execution_mode(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfComposedExec: native_nodes={}, gpu_input_count={}, sink=host_arrow, native_pipeline_execution_mode={}, memory_control_policy={}, boundary_policy={}, input_boundary=gpu_resident, output_boundary=device_to_host, edge_normalize_inserted={}, native_plan={}",
                self.plan.node_count(),
                self.gpu_inputs.len(),
                self.execution_config.native_pipeline_execution_mode(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
                self.plan.contains_edge_normalize(),
                native_plan_one_line(&self.plan),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "gpu_input_count={}", self.gpu_inputs.len())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(
                    f,
                    "native_pipeline_execution_mode={}",
                    self.execution_config.native_pipeline_execution_mode()
                )?;
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

impl ExecutionPlan for NativeCudfComposedExec {
    fn name(&self) -> &str {
        "NativeCudfComposedExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.gpu_inputs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        NativeCudfComposedExec::try_new_with_equivalence_properties(
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
        let metrics = NativeCudfMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.execution_config.metrics_capture(),
            self.execution_config.plan_task_mapping_datafusion_shape(),
        );
        let output = native_host_output_stream(
            NATIVE_CUDF_COMPOSED_EXEC_HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                // Build one native context for all producer fragments and the consumer so device-resident
                // handles are compatible across the composed execution.
                let context_lease = execution_config
                    .execution_context_async(Arc::clone(&task_context))
                    .await
                    .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
                if let Some(admission_facts) = context_lease.admission_facts() {
                    metrics.record_admission_facts(admission_facts);
                }
                let gpu_execution_pool = context_lease.context().gpu_execution_pool();
                run_gpu_execution_task(
                    gpu_execution_pool,
                    "datafusion_nexus.native_composed.gpu_execution_pool",
                    async move {
                        let context = context_lease.context();
                        let execution_task_context =
                            context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                        let mut input_map = GpuResidentInputMap::new();
                        for child in gpu_inputs {
                            let (input_name, handle) = execute_gpu_resident_producer_with_context(
                                child.as_ref(),
                                context,
                                partition,
                                Arc::clone(&execution_task_context),
                            )
                            .await?;
                            input_map.insert(input_name, handle);
                        }

                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        // The consumer query starts after all GPU-resident producer handles are registered. No
                        // CPU fallback is attempted after this point.
                        let result =
                            execute_query_plan_with_gpu_inputs(context, &plan, &input_map).map_err(|error| {
                                DataFusionError::from(native_query_runtime_error(
                                    "datafusion_nexus.native_composed.execute",
                                    "native cudf composed query execution failed",
                                    error,
                                ))
                            })?;
                        let (output, query_metrics) = result.into_parts();
                        metrics.record_query_metrics(&query_metrics);
                        metrics
                            .record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                        let QueryOutput::HostArrow(batch) = output else {
                            return Err(DataFusionError::from(
                                Error::new(
                                    ErrorCode::InvalidExecutionPlan,
                                    "datafusion_nexus.native_composed.execute",
                                    "native composed cudf execution produced a non-host result for DataFusion",
                                )
                                .with_fact("result_sink", "gpu_resident"),
                            ));
                        };
                        Ok::<RecordBatch, DataFusionError>(batch.record_output(&metrics.baseline))
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
