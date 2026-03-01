//! GPU-resident native fragment producer for composed execution.
//!
//! Producers are internal children of composed native execution. Their public
//! wrapper type is exposed for diagnostics and plan inspection, but executing
//! them through the ordinary DataFusion streaming path is invalid.

use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{ExecutionContext, GpuResultHandle, QueryOutput, QueryPlan, execute_query_plan};

use super::{
    common::{comma_separated, conservative_properties, gpu_resident_source_names, native_query_runtime_error},
    metrics::NativeCudfMetrics,
};
use crate::error::{Error, ErrorCode};

pub(crate) trait GpuResidentProducerExec {
    fn gpu_result_input_name(&self) -> &str;

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>>;
}

/// Internal GPU-resident native fragment producer.
///
/// This plan is valid only as a child of composed native execution. It produces
/// a device-resident `GpuResultHandle` for another native plan and is not
/// directly streamable through [`ExecutionPlan::execute`]. The ordinary
/// DataFusion execution path returns `InvalidExecutionPlan`; composed native
/// execution consumes this wrapper through the internal GPU-resident producer
/// path so the result can stay on device.
#[derive(Debug, Clone)]
pub struct NativeCudfGpuProducerExec {
    input_name: String,
    plan: QueryPlan,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfGpuProducerExec {
    /// Create a native producer fragment for composed execution.
    ///
    /// The plan must use a GPU-resident result sink and must not itself depend
    /// on external GPU-resident inputs. `input_name` is the local name the
    /// composed consumer uses to look up the produced handle. The returned
    /// execution plan is intended for composed native consumers, not for direct
    /// `ExecutionPlan::execute` streaming.
    pub fn try_new(input_name: impl Into<String>, plan: QueryPlan) -> crate::error::Result<Self> {
        let input_name = input_name.into();
        if input_name.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_gpu_producer.try_new",
                "native GPU-resident producer input name must not be empty",
            ));
        }
        if plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_gpu_producer.try_new",
                "native GPU-resident producer requires a GPU-resident result sink",
            )
            .with_fact("result_sink", "host_arrow"));
        }
        let source_names = gpu_resident_source_names(&plan);
        if !source_names.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_gpu_producer.try_new",
                "native GPU-resident producer cannot consume external GPU inputs",
            )
            .with_fact("gpu_resident_source_count", source_names.len())
            .with_fact("gpu_resident_sources", comma_separated(source_names.iter().map(String::as_str))));
        }

        Ok(Self {
            input_name,
            cache: conservative_properties(Arc::clone(plan.output_schema())),
            plan,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    /// Execute this producer inside a composed exec's shared native context.
    ///
    /// The result stays on the GPU and is returned as `(input_name, GpuResultHandle)` so the composed
    /// consumer can populate its `GpuResidentInputMap` without host materialization.
    pub(crate) fn execute_gpu_result_with_context(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_gpu_producer.execute_gpu",
                "native GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let result = execute_query_plan(native_context, &self.plan).map_err(|error| {
            DataFusionError::from(native_query_runtime_error(
                "datafusion_nexus.native_gpu_producer.execute_gpu",
                "native cudf GPU-resident fragment execution failed",
                error,
            ))
        })?;
        let (output, query_metrics) = result.into_parts();
        metrics.record_query_metrics(&query_metrics);
        let QueryOutput::GpuResident(handle) = output else {
            return Err(DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.native_gpu_producer.execute_gpu",
                    "native GPU-resident producer materialized a host result",
                )
                .with_fact("result_sink", "host_arrow"),
            ));
        };
        Ok((self.input_name.clone(), handle))
    }
}

impl GpuResidentProducerExec for NativeCudfGpuProducerExec {
    fn gpu_result_input_name(&self) -> &str {
        self.input_name()
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
        Box::pin(async move {
            let _ = task_context;
            NativeCudfGpuProducerExec::execute_gpu_result_with_context(self, native_context, partition)
        })
    }
}

pub(crate) fn gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducerExec> {
    if let Some(producer) = plan.downcast_ref::<NativeCudfGpuProducerExec>() {
        return Some(producer);
    }
    #[cfg(feature = "cugraph")]
    if let Some(producer) = crate::graph_sql::gpu_resident_producer(plan) {
        return Some(producer);
    }
    None
}

pub(crate) fn gpu_resident_producer_input_name(plan: &dyn ExecutionPlan) -> Option<String> {
    gpu_resident_producer(plan).map(|producer| producer.gpu_result_input_name().to_string())
}

pub(crate) async fn execute_gpu_resident_producer_with_context(
    plan: &dyn ExecutionPlan,
    native_context: &ExecutionContext,
    partition: usize,
    task_context: Arc<TaskContext>,
) -> DataFusionResult<(String, GpuResultHandle)> {
    if let Some(producer) = gpu_resident_producer(plan) {
        return producer.execute_gpu_result_with_context(native_context, partition, task_context).await;
    }
    Err(Error::new(
        ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.native_gpu_producer.execute_gpu",
        "execution plan is not a GPU-resident producer",
    )
    .with_fact("plan_name", plan.name())
    .into())
}

impl DisplayAs for NativeCudfGpuProducerExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfGpuProducerExec: input_name={}, native_nodes={}, sink=gpu_resident, output_boundary=device_resident, edge_normalize_inserted={}",
                self.input_name,
                self.plan.node_count(),
                self.plan.contains_edge_normalize(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "input_name={}", self.input_name)?;
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=gpu_resident")?;
                writeln!(f, "output_boundary=device_resident")?;
                writeln!(f, "edge_normalize_inserted={}", self.plan.contains_edge_normalize())
            },
        }
    }
}

impl ExecutionPlan for NativeCudfGpuProducerExec {
    fn name(&self) -> &str {
        "NativeCudfGpuProducerExec"
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
                "datafusion_nexus.native_gpu_producer.with_new_children",
                "native GPU-resident producer expected no children",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(self)
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.native_gpu_producer.execute",
            "native GPU-resident producer is an internal device boundary and cannot stream host Arrow directly",
        )
        .with_fact("result_sink", "gpu_resident")
        .into())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
