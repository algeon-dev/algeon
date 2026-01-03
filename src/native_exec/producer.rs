use std::{any::Any, sync::Arc};

use cudf_query_engine::{ExecutionContext, GpuResultHandle, QueryOutput, QueryPlan, execute_query_plan};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};

use super::{
    common::{comma_separated, conservative_properties, gpu_resident_source_names, native_query_runtime_error},
    metrics::NativeCudfMetrics,
};
use crate::error::{Error, ErrorCode};

/// Internal GPU-resident native fragment producer.
///
/// This plan is valid only as a child of composed native execution. It produces a device-resident
/// `GpuResultHandle` for another native plan and is not directly streamable through
/// `ExecutionPlan::execute`.
#[derive(Debug, Clone)]
pub struct NativeCudfGpuProducerExec {
    input_name: String,
    plan: QueryPlan,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NativeCudfGpuProducerExec {
    pub fn try_new(input_name: impl Into<String>, plan: QueryPlan) -> crate::error::Result<Self> {
        let input_name = input_name.into();
        if input_name.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_gpu_producer.try_new",
                "native GPU-resident producer input name must not be empty",
            ));
        }
        if plan.sink().materializes_host_boundary() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_gpu_producer.try_new",
                "native GPU-resident producer requires a GPU-resident result sink",
            )
            .with_fact("result_sink", "host_arrow"));
        }
        let source_names = gpu_resident_source_names(&plan);
        if !source_names.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_gpu_producer.try_new",
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
    pub(super) fn execute_gpu_result_with_context(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_cudf.native_gpu_producer.execute_gpu",
                "native GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let metrics = NativeCudfMetrics::new(&self.metrics, partition);
        let result = execute_query_plan(native_context, &self.plan).map_err(|error| {
            DataFusionError::from(native_query_runtime_error(
                "datafusion_cudf.native_gpu_producer.execute_gpu",
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
                    "datafusion_cudf.native_gpu_producer.execute_gpu",
                    "native GPU-resident producer materialized a host result",
                )
                .with_fact("result_sink", "host_arrow"),
            ));
        };
        Ok((self.input_name.clone(), handle))
    }
}

impl DisplayAs for NativeCudfGpuProducerExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "NativeCudfGpuProducerExec: input_name={}, native_nodes={}, sink=gpu_resident, output_boundary=device_resident",
                self.input_name,
                self.plan.node_count(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "input_name={}", self.input_name)?;
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=gpu_resident")?;
                writeln!(f, "output_boundary=device_resident")
            },
        }
    }
}

impl ExecutionPlan for NativeCudfGpuProducerExec {
    fn name(&self) -> &str {
        "NativeCudfGpuProducerExec"
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
                "datafusion_cudf.native_gpu_producer.with_new_children",
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
            "datafusion_cudf.native_gpu_producer.execute",
            "native GPU-resident producer is an internal device boundary and cannot stream host Arrow directly",
        )
        .with_fact("result_sink", "gpu_resident")
        .into())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
