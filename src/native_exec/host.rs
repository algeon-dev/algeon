use std::{any::Any, sync::Arc};

use arrow::record_batch::RecordBatch;
use cudf_query_engine::{QueryOutput, QueryPlan, execute_query_plan};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::TryStreamExt;

use super::{
    common::{conservative_properties, mark_native_runtime_error, native_query_runtime_error},
    config::NativeCudfExecutionConfig,
    metrics::NativeCudfMetrics,
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
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfExec {
    /// Create a native DataFusion execution plan.
    ///
    /// GPU-resident sinks are rejected here because regular `ExecutionPlan::execute` must stream
    /// host Arrow `RecordBatch` output to DataFusion.
    pub fn try_new(plan: QueryPlan, execution_config: NativeCudfExecutionConfig) -> crate::error::Result<Self> {
        if !plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_exec.try_new",
                "native DataFusion execution requires a host-arrow result sink",
            )
            .with_fact("result_sink", "gpu_resident"));
        }

        Ok(Self {
            cache: conservative_properties(Arc::clone(plan.output_schema())),
            plan,
            execution_config,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Return the lowered native query plan wrapped by this DataFusion execution node.
    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }
}

impl DisplayAs for NativeCudfExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfExec: native_nodes={}, sink=host_arrow, memory_control_policy={}, boundary_policy={}, output_boundary=device_to_host",
                self.plan.node_count(),
                self.execution_config.memory_control_policy(),
                self.execution_config.boundary_policy(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=host_arrow")?;
                writeln!(f, "memory_control_policy={}", self.execution_config.memory_control_policy())?;
                writeln!(f, "boundary_policy={}", self.execution_config.boundary_policy())?;
                writeln!(f, "output_boundary=device_to_host")
            },
        }
    }
}

impl ExecutionPlan for NativeCudfExec {
    fn name(&self) -> &str {
        "NativeCudfExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
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
                "datafusion_cudf.native_exec.with_new_children",
                "native cudf execution expected no children",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        // Native DataFusion execution currently exposes a single final output partition.
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_exec.execute",
                "native cudf execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        // The DataFusion `TaskContext` is intentionally unused: runtime, allocator, boundary, and
        // memory-control state come from `NativeCudfExecutionConfig`.
        let schema = self.schema();
        let plan = self.plan.clone();
        let execution_config = self.execution_config.clone();
        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let output = futures::stream::once(async move {
            // Context construction is the configuration/validation phase. Failures here happen
            // before the native query starts.
            let context = execution_config
                .execution_context()
                .map_err(|error| DataFusionError::from(mark_native_runtime_error(error, "runtime_context")))?;
            // Once native execution starts there is no CPU fallback; errors are surfaced with stable
            // facts by `native_query_runtime_error`.
            let result = execute_query_plan(&context, &plan).map_err(|error| {
                DataFusionError::from(native_query_runtime_error(
                    "datafusion_cudf.native_exec.execute",
                    "native cudf query execution failed",
                    error,
                ))
            })?;
            let (output, query_metrics) = result.into_parts();
            metrics.record_query_metrics(&query_metrics);
            let QueryOutput::HostArrow(batch) = output else {
                return Err(DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_cudf.native_exec.execute",
                        "native cudf execution produced a non-host result for DataFusion",
                    )
                    .with_fact("result_sink", "gpu_resident"),
                ));
            };
            Ok::<Option<RecordBatch>, DataFusionError>(Some(batch.record_output(&metrics.baseline)))
        })
        .try_filter_map(|batch| async move { Ok(batch) });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
