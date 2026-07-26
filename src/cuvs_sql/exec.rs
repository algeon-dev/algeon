use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    exec::{execute_query_plan_with_gpu_inputs_observed, output::QueryOutput},
    metrics::QueryMetrics,
    plan::QueryPlan,
    runtime::{ExecutionContext, MemoryControlPolicy, resident::GpuResultHandle},
};

use super::{
    CuvsDenseVectorBinding, CuvsOperation, CuvsValidatedOptions,
    error::{bounded_execution_unsupported, feature_disabled, invalid_call},
    lowering::{CuvsPlanSink, lower_cuvs_query_plan},
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        NativeExecutionMetricsCapture, NexusNativeExecutionConfig, NexusNativeMetrics,
        child_inputs::{
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES, NativeChildBinding, NativeChildInputSpec, StagedNativeChildren,
            stage_native_children,
        },
        producer::{GpuResidentProducer, register_gpu_resident_producer_discovery},
        run_gpu_execution_task,
    },
};

static NEXT_CUVS_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

/// Physical cuVS operator over real DataFusion children.
///
/// Children remain in semantic operation order. Ordinary Arrow inputs are
/// staged once; direct accelerator producers are consumed as GPU handles.
#[derive(Clone, Debug)]
pub struct CuvsAlgoExec {
    operation: CuvsOperation,
    options: CuvsValidatedOptions,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    input_names: Vec<String>,
    input_projections: Vec<Vec<usize>>,
    output_schema: SchemaRef,
    native_execution_config: NexusNativeExecutionConfig,
    gpu_result_input_name: String,
    prepared_staged_inputs: Option<Arc<std::sync::Mutex<Option<StagedNativeChildren>>>>,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl CuvsAlgoExec {
    pub(crate) fn try_new(
        operation: CuvsOperation,
        options: CuvsValidatedOptions,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        output_schema: SchemaRef,
        native_execution_config: NexusNativeExecutionConfig,
    ) -> Result<Self> {
        let id = NEXT_CUVS_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        let input_names =
            (0..inputs.len()).map(|index| format!("__datafusion_nexus_cuvs_input_{id}_{index}")).collect();
        Self::try_new_with_names(
            operation,
            options,
            inputs,
            output_schema,
            native_execution_config,
            input_names,
            format!("__datafusion_nexus_cuvs_result_{id}"),
        )
    }

    fn try_new_with_names(
        operation: CuvsOperation,
        options: CuvsValidatedOptions,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        output_schema: SchemaRef,
        native_execution_config: NexusNativeExecutionConfig,
        input_names: Vec<String>,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        register_cuvs_execution_spi();
        if inputs.len() != options.bindings().len() || input_names.len() != inputs.len() {
            return Err(invalid_call(
                operation,
                "physical_child_count_mismatch",
                "cuVS physical inputs do not match validated relation bindings",
            ));
        }
        let input_projections = inputs
            .iter()
            .zip(options.bindings())
            .map(|(input, binding)| required_projection(operation, input.schema(), binding))
            .collect::<Result<Vec<_>>>()?;
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let metrics_capture = native_execution_config.new_metrics_capture();
        let exec = Self {
            operation,
            options,
            inputs,
            input_names,
            input_projections,
            output_schema,
            native_execution_config,
            gpu_result_input_name,
            prepared_staged_inputs: None,
            metrics_capture,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        };
        let plan = exec.native_plan()?;
        nexus_query_engine::exec::preflight_query_plan(&plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cuvs_sql.preflight",
                "native cuVS query-plan preflight failed",
                error,
            ))
        })?;
        Ok(exec)
    }

    pub fn operation(&self) -> CuvsOperation {
        self.operation
    }

    pub fn native_plan(&self) -> Result<QueryPlan> {
        lower_cuvs_query_plan(
            self.operation,
            &self.options,
            &self
                .inputs
                .iter()
                .zip(&self.input_names)
                .zip(&self.input_projections)
                .zip(self.operation.descriptor().relation_roles.iter())
                .map(|(((input, input_name), projection), role)| NativeChildBinding {
                    input_name: input_name.clone(),
                    role: role.as_str(),
                    schema: input.schema(),
                    projection: projection.clone(),
                })
                .collect::<Vec<_>>(),
            Arc::clone(&self.output_schema),
            CuvsPlanSink::HostArrow,
        )
    }

    fn child_specs(&self) -> Vec<NativeChildInputSpec> {
        self.inputs
            .iter()
            .zip(&self.input_names)
            .zip(&self.input_projections)
            .zip(self.operation.descriptor().relation_roles.iter())
            .map(|(((plan, input_name), projection), role)| NativeChildInputSpec {
                input_name: input_name.clone(),
                role: role.as_str(),
                plan: Arc::clone(plan),
                projection: projection.clone(),
            })
            .collect()
    }

    fn reject_unavailable_execution(&self, memory_control_policy: MemoryControlPolicy) -> Result<()> {
        if matches!(memory_control_policy, MemoryControlPolicy::Bounded { .. }) {
            return Err(bounded_execution_unsupported(
                self.operation,
                self.native_execution_config.native_pipeline_execution_mode(),
            ));
        }
        if !cfg!(feature = "cuvs") {
            return Err(feature_disabled(self.operation));
        }
        Ok(())
    }

    fn take_prepared_staged_inputs(&self) -> Result<StagedNativeChildren> {
        let slot = self.prepared_staged_inputs.as_ref().ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cuvs_sql.execute_gpu",
                    "cuVS GPU-resident producer inputs were not prepared before native context acquisition",
                )
                .with_log_safe_fact("staged_input_state", "not_prepared"),
            )
        })?;
        slot.lock()
            .map_err(|_| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.cuvs_sql.execute_gpu",
                        "cuVS staged-input ownership was poisoned",
                    )
                    .with_log_safe_fact("staged_input_state", "poisoned"),
                )
            })?
            .take()
            .ok_or_else(|| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.cuvs_sql.execute_gpu",
                        "cuVS staged inputs were already consumed",
                    )
                    .with_log_safe_fact("staged_input_state", "already_consumed"),
                )
            })
    }

    async fn execute_staged(
        &self,
        native_context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        staged: StagedNativeChildren,
        sink: CuvsPlanSink,
        metrics: &NexusNativeMetrics,
    ) -> Result<QueryOutput> {
        self.reject_unavailable_execution(native_context.memory_control_policy())?;
        let uploaded = staged.upload(native_context, task_context).await?;
        let host_to_device_uploads = uploaded.host_to_device_uploads.clone();
        let plan = lower_cuvs_query_plan(
            self.operation,
            &self.options,
            &uploaded.bindings,
            Arc::clone(&self.output_schema),
            sink,
        )?;
        let outcome = execute_query_plan_with_gpu_inputs_observed(
            native_context,
            &plan,
            &uploaded.inputs,
            self.native_execution_config.runtime_observation_config(),
        );
        let (result, mut query_metrics, observation) = outcome.into_parts();
        record_cuvs_staging_boundaries(&mut query_metrics, &host_to_device_uploads);
        metrics.record_query_metrics(&query_metrics);
        if let Some(observation) = observation {
            metrics.record_runtime_observation(Arc::new(observation));
        }
        result.map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cuvs_sql.execute",
                "native cuVS vector query execution failed",
                error,
            ))
        })
    }

    async fn execute_gpu_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> Result<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cuvs_sql.execute_gpu",
                "cuVS GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        let staged = self.take_prepared_staged_inputs()?;
        let metrics = NexusNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let output =
            self.execute_staged(native_context, task_context, staged, CuvsPlanSink::GpuResident, &metrics).await?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cuvs_sql.execute_gpu",
                "native cuVS producer materialized an unexpected host result",
            )
            .with_log_safe_fact("result_sink", "host_arrow")
            .into());
        };
        Ok((self.gpu_result_input_name.clone(), handle))
    }
}

impl GpuResidentProducer for CuvsAlgoExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    fn prepare_execution_plan<'a>(
        &'a self,
        _plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            self.reject_unavailable_execution(self.native_execution_config.memory_control_policy())?;
            let staged = stage_native_children(
                self.child_specs(),
                task_context,
                self.native_execution_config.cancellation(),
                DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
            )
            .await?;
            let mut prepared = self.clone();
            prepared.prepared_staged_inputs = Some(Arc::new(std::sync::Mutex::new(Some(staged))));
            Ok(Arc::new(prepared) as Arc<dyn ExecutionPlan>)
        })
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, Result<(String, GpuResultHandle)>> {
        Box::pin(
            async move { self.execute_gpu_result_with_context_impl(native_context, partition, task_context).await },
        )
    }
}

fn cuvs_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<CuvsAlgoExec>().map(|producer| producer as &dyn GpuResidentProducer)
}

fn register_cuvs_execution_spi() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| register_gpu_resident_producer_discovery(cuvs_gpu_resident_producer));
}

impl DisplayAs for CuvsAlgoExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CuvsAlgoExec: operation={}, input_count={}, memory_control_policy={}, input_boundary=gpu_resident_or_host_staged, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.operation.descriptor().sql_function_name,
                self.inputs.len(),
                self.native_execution_config.memory_control_policy(),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "operation={}", self.operation.descriptor().sql_function_name)?;
                writeln!(f, "input_count={}", self.inputs.len())?;
                writeln!(f, "memory_control_policy={}", self.native_execution_config.memory_control_policy())?;
                writeln!(f, "input_boundary=gpu_resident_or_host_staged")?;
                writeln!(f, "output_boundary=device_resident_or_device_to_host")?;
                writeln!(f, "gpu_result_input={}", self.gpu_result_input_name)
            },
        }
    }
}

impl ExecutionPlan for CuvsAlgoExec {
    fn name(&self) -> &str {
        "CuvsAlgoExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.inputs.len()]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != self.inputs.len() {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_nexus.cuvs_sql.with_new_children",
                "cuVS execution received an unexpected child count",
            )
            .with_fact("expected_child_count", self.inputs.len())
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(Arc::new(Self::try_new_with_names(
            self.operation,
            self.options.clone(),
            children,
            Arc::clone(&self.output_schema),
            self.native_execution_config.clone(),
            self.input_names.clone(),
            self.gpu_result_input_name.clone(),
        )?))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cuvs_sql.execute",
                "cuVS execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        self.reject_unavailable_execution(self.native_execution_config.memory_control_policy())?;

        let schema = Arc::clone(&self.output_schema);
        let exec = self.clone();
        let execution_config = self.native_execution_config.clone();
        let metrics = NexusNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let stream = futures::stream::once(async move {
            let staged = stage_native_children(
                exec.child_specs(),
                Arc::clone(&task_context),
                execution_config.cancellation(),
                DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
            )
            .await?;
            let _unbounded_permit = match execution_config.memory_control_policy() {
                MemoryControlPolicy::Unbounded => match execution_config.unbounded_native_execution_limiter() {
                    Some(limiter) => Some(limiter.acquire().await?),
                    None => None,
                },
                MemoryControlPolicy::Bounded { .. } => None,
            };
            let context_lease = execution_config
                .execution_context_async(Arc::clone(&task_context))
                .await
                .map_err(DataFusionError::from)?;
            let gpu_execution_pool = context_lease.context().gpu_execution_pool().map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cuvs_sql.gpu_execution_pool",
                    "query service closed before cuVS acquired its GPU execution pool",
                    error,
                ))
            })?;
            let observation = context_lease.context().runtime_observation().clone();
            run_gpu_execution_task(
                gpu_execution_pool,
                observation,
                "datafusion_nexus.cuvs_sql.gpu_execution_pool",
                move |gpu_stream_scope| async move {
                    let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                    let execution_task_context =
                        context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                    let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                    let output = exec
                        .execute_staged(&context, execution_task_context, staged, CuvsPlanSink::HostArrow, &metrics)
                        .await?;
                    match output {
                        QueryOutput::HostArrow(batch) => Ok(batch),
                        QueryOutput::GpuResident(_) => Err(Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_nexus.cuvs_sql.execute",
                            "native cuVS host execution returned a GPU-resident result",
                        )
                        .with_log_safe_fact("result_sink", "gpu_resident")
                        .into()),
                    }
                },
            )
            .await
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn record_cuvs_staging_boundaries(
    metrics: &mut QueryMetrics,
    uploads: &[crate::native_exec::child_inputs::HostToDeviceUpload],
) {
    for upload in uploads {
        metrics.boundary.record_host_to_device_upload(upload.rows, upload.bytes);
    }
}

fn required_projection(
    operation: CuvsOperation,
    schema: SchemaRef,
    binding: &super::CuvsVectorRelationBinding,
) -> Result<Vec<usize>> {
    let mut names = Vec::new();
    names.push(binding.id());
    match binding.vector() {
        CuvsDenseVectorBinding::Columns(columns) => names.extend(columns.iter().map(String::as_str)),
        CuvsDenseVectorBinding::VectorColumn(column) => names.push(column),
    }
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert((*name).to_string()))
        .map(|name| {
            schema.index_of(name).map_err(|_| {
                invalid_call(
                    operation,
                    "physical_projection_field_missing",
                    format!("cuVS physical input has no field named '{name}'"),
                )
            })
        })
        .collect()
}
