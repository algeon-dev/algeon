//! Shared relation-function execution over DataFusion physical children.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{DataFusionError, Result, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, LexOrdering, OrderingRequirements, Partitioning, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, InputDistributionRequirements, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use futures::future::BoxFuture;
use query_engine::{
    admission::AttemptCapabilityEvidence,
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryOutput, preflight_query_plan},
    expr::Expr as NativeExpr,
    metrics::QueryMetrics,
    plan::{NodeId, ProjectionOutput, QueryPlan, QueryPlanBuilder},
    runtime::{ExecutionContext, resident::GpuResultHandle},
    source::{GpuResidentSource, SourceNode},
};

use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        AlgeonNativeExecutionConfig, AlgeonNativeMetrics, GpuResidentProducer, NativeExecutionMetricsCapture,
        NativeHostOutputCursor,
        admission::QueryAdmissionTicket,
        child_inputs::{
            DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES, HostToDeviceUpload, NativeChildBinding, NativeChildInputSpec,
            StagedNativeChildren, stage_native_children,
        },
        gpu_function_occurrence_from_task_context, gpu_resident_producer, native_host_output_stream,
        prepared_gpu_resident_producer_admission_tickets, producer_bound_capability_evidence, run_gpu_execution_task,
    },
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum RelationFunctionSink {
    HostArrow,
    DeferredHostArrow,
    GpuResident,
}

pub struct RelationFunctionOperations {
    pub execute: &'static str,
    pub execute_gpu: &'static str,
    pub gpu_execution_pool: &'static str,
    pub host_batch_cursor: &'static str,
    pub preflight: &'static str,
    pub with_new_children: &'static str,
}

pub struct RelationFunctionMessages {
    pub preflight_failed: &'static str,
    pub staged_inputs_missing: &'static str,
    pub staged_inputs_poisoned: &'static str,
    pub staged_inputs_consumed: &'static str,
    pub execution_failed: &'static str,
    pub gpu_partition: &'static str,
    pub unexpected_host_result: &'static str,
    pub child_count: &'static str,
    pub host_partition: &'static str,
    pub gpu_pool_closed: &'static str,
    pub deferred_result: &'static str,
    pub cursor_failed: &'static str,
}

pub struct RelationFunctionText {
    pub operations: RelationFunctionOperations,
    pub messages: RelationFunctionMessages,
}

pub(crate) trait RelationFunctionSpec: Clone + std::fmt::Debug + Send + Sync + 'static {
    const EXEC_NAME: &'static str;
    const HOST_OUTPUT_MEMORY_CONSUMER: &'static str;
    fn operation_name(&self) -> &'static str;
    fn binding_count(&self) -> usize;
    fn roles(&self) -> Vec<&'static str>;
    fn required_projection(&self, index: usize, schema: SchemaRef) -> Result<Vec<usize>>;
    /// Lowers the function into a native plan with a host Arrow sink; every
    /// other sink is derived from that plan.
    fn lower(&self, bindings: &[NativeChildBinding], schema: SchemaRef) -> Result<QueryPlan>;
    fn feature_available(&self) -> bool;
    fn feature_disabled(&self) -> DataFusionError;
    fn invalid_child_count(&self) -> DataFusionError;
    const TEXT: RelationFunctionText;
}

#[derive(Clone, Debug)]
pub struct RelationFunctionExec<S> {
    spec: S,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    input_names: Vec<String>,
    input_projections: Vec<Vec<usize>>,
    /// Declared order of each input relation. `row_ordinal` numbers input
    /// rows in arrival order, so DataFusion must deliver this order.
    input_orderings: Vec<Option<LexOrdering>>,
    output_schema: SchemaRef,
    native_host_plan: QueryPlan,
    native_execution_config: AlgeonNativeExecutionConfig,
    gpu_result_input_name: String,
    prepared_staged_inputs: Option<Arc<std::sync::Mutex<Option<StagedNativeChildren>>>>,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

#[expect(private_bounds, reason = "the family spec uses crate-local lowering bindings")]
impl<S: RelationFunctionSpec> RelationFunctionExec<S> {
    pub(crate) fn try_new_with_names(
        spec: S,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        output_schema: SchemaRef,
        native_execution_config: AlgeonNativeExecutionConfig,
        input_names: Vec<String>,
        input_orderings: Vec<Option<LexOrdering>>,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        if inputs.len() != spec.binding_count()
            || input_names.len() != inputs.len()
            || input_orderings.len() != inputs.len()
        {
            return Err(spec.invalid_child_count());
        }
        let input_projections = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| spec.required_projection(index, input.schema()))
            .collect::<Result<Vec<_>>>()?;
        let bindings = planned_bindings(&spec, &inputs, &input_names, &input_projections)?;
        let native_host_plan = spec.lower(&bindings, Arc::clone(&output_schema))?;
        preflight_query_plan(&native_host_plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                S::TEXT.operations.preflight,
                S::TEXT.messages.preflight_failed,
                error,
            ))
        })?;
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let metrics_capture = native_execution_config.new_metrics_capture();
        Ok(Self {
            spec,
            inputs,
            input_names,
            input_projections,
            input_orderings,
            output_schema,
            native_host_plan,
            native_execution_config,
            gpu_result_input_name,
            prepared_staged_inputs: None,
            metrics_capture,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    pub(crate) fn spec(&self) -> &S {
        &self.spec
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        &self.native_host_plan
    }

    /// Host-staged children are uploaded inside this function's own scope, so
    /// only GPU-resident producer children contribute evidence and occurrences.
    fn capability_evidence_for_sink(&self, sink: RelationFunctionSink) -> Result<AttemptCapabilityEvidence> {
        let plan = plan_for_sink(&self.native_host_plan, sink);
        let producer_inputs = self
            .input_names
            .iter()
            .zip(&self.inputs)
            .filter(|(_, input)| gpu_resident_producer(input.as_ref()).is_some())
            .map(|(input_name, input)| (input_name.clone(), input.as_ref()));
        producer_bound_capability_evidence(&plan, producer_inputs).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.relation_function.admission_evidence",
                "failed to derive capability evidence for a relation function and its GPU-resident inputs",
                error,
            ))
        })
    }

    fn child_specs(&self) -> Vec<NativeChildInputSpec> {
        self.inputs
            .iter()
            .zip(&self.input_names)
            .zip(&self.input_projections)
            .zip(self.spec.roles().iter())
            .map(|(((plan, input_name), projection), role)| NativeChildInputSpec {
                input_name: input_name.clone(),
                role,
                plan: Arc::clone(plan),
                projection: projection.clone(),
            })
            .collect()
    }

    fn reject_unavailable_execution(&self) -> Result<()> {
        if !self.spec.feature_available() {
            return Err(self.spec.feature_disabled());
        }
        Ok(())
    }

    fn take_prepared_staged_inputs(&self) -> Result<StagedNativeChildren> {
        let slot = self.prepared_staged_inputs.as_ref().ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    S::TEXT.operations.execute_gpu,
                    S::TEXT.messages.staged_inputs_missing,
                )
                .with_log_safe_fact("staged_input_state", "not_prepared"),
            )
        })?;
        slot.lock()
            .map_err(|_| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        S::TEXT.operations.execute_gpu,
                        S::TEXT.messages.staged_inputs_poisoned,
                    )
                    .with_log_safe_fact("staged_input_state", "poisoned"),
                )
            })?
            .take()
            .ok_or_else(|| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        S::TEXT.operations.execute_gpu,
                        S::TEXT.messages.staged_inputs_consumed,
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
        sink: RelationFunctionSink,
        metrics: &AlgeonNativeMetrics,
    ) -> Result<QueryOutput> {
        self.reject_unavailable_execution()?;
        let uploaded = staged.upload(native_context, task_context).await?;
        let host_to_device_uploads = uploaded.host_to_device_uploads.clone();
        let plan = plan_for_sink(&self.spec.lower(&uploaded.bindings, Arc::clone(&self.output_schema))?, sink);
        let outcome =
            execute_query_plan(native_context, QueryExecutionRequest::new(&plan).with_gpu_inputs(&uploaded.inputs));
        let (result, mut query_metrics, _observation) = outcome.into_parts();
        record_relation_staging_boundaries(&mut query_metrics, &host_to_device_uploads);
        metrics.record_query_metrics(&query_metrics);
        result.map_err(|error| {
            DataFusionError::from(Error::from_native_with_blocking_diagnostics(
                S::TEXT.operations.execute,
                S::TEXT.messages.execution_failed,
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
                S::TEXT.operations.execute_gpu,
                S::TEXT.messages.gpu_partition,
            )
            .with_fact("partition", partition)
            .into());
        }
        let staged = self.take_prepared_staged_inputs()?;
        let metrics = AlgeonNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let output = self
            .execute_staged(native_context, task_context, staged, RelationFunctionSink::GpuResident, &metrics)
            .await?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                S::TEXT.operations.execute_gpu,
                S::TEXT.messages.unexpected_host_result,
            )
            .with_log_safe_fact("result_sink", "host_arrow")
            .into());
        };
        Ok((self.gpu_result_input_name.clone(), handle))
    }
}

impl<S: RelationFunctionSpec> GpuResidentProducer for RelationFunctionExec<S> {
    fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        self.capability_evidence_for_sink(RelationFunctionSink::GpuResident)
    }

    fn host_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        self.capability_evidence_for_sink(RelationFunctionSink::HostArrow)
    }

    fn native_relation_input_names(&self) -> Option<&[String]> {
        Some(&self.input_names)
    }

    fn host_result_query_plan(&self) -> Result<Option<QueryPlan>> {
        Ok(Some(self.native_plan().clone()))
    }

    fn execution_metrics(&self) -> Option<&ExecutionPlanMetricsSet> {
        Some(&self.metrics)
    }

    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        let mut tickets = Vec::new();
        for input in &self.inputs {
            for ticket in prepared_gpu_resident_producer_admission_tickets(input.as_ref()) {
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
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            self.reject_unavailable_execution()?;
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

impl<S: RelationFunctionSpec> DisplayAs for RelationFunctionExec<S> {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "{}: operation={}, input_count={}, memory_control_policy={}, input_boundary=gpu_resident_or_host_staged, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                S::EXEC_NAME,
                self.spec.operation_name(),
                self.inputs.len(),
                self.native_execution_config.memory_control_policy(),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "operation={}", self.spec.operation_name())?;
                writeln!(f, "input_count={}", self.inputs.len())?;
                writeln!(f, "memory_control_policy={}", self.native_execution_config.memory_control_policy())?;
                writeln!(f, "input_boundary=gpu_resident_or_host_staged")?;
                writeln!(f, "output_boundary=device_resident_or_device_to_host")?;
                writeln!(f, "gpu_result_input={}", self.gpu_result_input_name)
            },
        }
    }
}

impl<S: RelationFunctionSpec> ExecutionPlan for RelationFunctionExec<S> {
    fn name(&self) -> &str {
        S::EXEC_NAME
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.inputs.len()]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        self.input_orderings.iter().map(|ordering| ordering.clone().map(OrderingRequirements::from)).collect()
    }

    /// An ordered input must arrive as one partition to keep its order across
    /// partitions. Unordered inputs keep the partition-order concatenation of
    /// host staging, which is more stable than the completion-order interleave
    /// of a coalesce.
    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        InputDistributionRequirements::new(
            self.input_orderings
                .iter()
                .map(|ordering| match ordering {
                    Some(_) => Distribution::SinglePartition,
                    None => Distribution::UnspecifiedDistribution,
                })
                .collect(),
        )
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != self.inputs.len() {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                S::TEXT.operations.with_new_children,
                S::TEXT.messages.child_count,
            )
            .with_fact("expected_child_count", self.inputs.len())
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(Arc::new(Self::try_new_with_names(
            self.spec.clone(),
            children,
            Arc::clone(&self.output_schema),
            self.native_execution_config.clone(),
            self.input_names.clone(),
            self.input_orderings.clone(),
            self.gpu_result_input_name.clone(),
        )?))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                S::TEXT.operations.execute,
                S::TEXT.messages.host_partition,
            )
            .with_fact("partition", partition)
            .into());
        }
        self.reject_unavailable_execution()?;

        let schema = Arc::clone(&self.output_schema);
        let output_schema = Arc::clone(&schema);
        let exec = self.clone();
        let execution_config = self.native_execution_config.clone();
        let account_memory_pool = execution_config.host_output_memory_pool_accounting();
        let metrics = AlgeonNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let stream = native_host_output_stream(
            S::HOST_OUTPUT_MEMORY_CONSUMER,
            schema,
            Arc::clone(&task_context),
            account_memory_pool,
            async move {
                let staged = stage_native_children(
                    exec.child_specs(),
                    Arc::clone(&task_context),
                    execution_config.cancellation(),
                    DEFAULT_MULTI_CHILD_HOST_STAGING_BYTES,
                )
                .await?;
                let host_occurrence = match execution_config.query_admission_ticket_for_task_context(&task_context) {
                    Some(ticket) => {
                        let occurrence_id =
                            gpu_function_occurrence_from_task_context(&task_context, &exec.gpu_result_input_name)?
                                .unwrap_or_else(|| exec.gpu_result_input_name.clone());
                        ticket.begin_host_occurrence(
                            &task_context,
                            &occurrence_id,
                            &exec.gpu_result_input_name,
                            exec.native_plan(),
                        )?
                    },
                    None => None,
                };
                let context_lease = execution_config
                    .execution_context_async(Arc::clone(&task_context))
                    .await
                    .map_err(DataFusionError::from)?;
                let gpu_executor = context_lease.context().gpu_executor().map_err(|error| {
                    DataFusionError::from(Error::from_native(
                        S::TEXT.operations.gpu_execution_pool,
                        S::TEXT.messages.gpu_pool_closed,
                        error,
                    ))
                })?;
                let observation = context_lease.context().runtime_observation().clone();
                let cursor_executor = gpu_executor.clone();
                let cursor_observation = observation.clone();
                let cursor_metrics = metrics.clone();
                run_gpu_execution_task(
                    gpu_executor,
                    observation,
                    S::TEXT.operations.gpu_execution_pool,
                    move |gpu_stream_scope| async move {
                        let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                        let execution_task_context =
                            context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
                        let output = exec
                            .execute_staged(
                                &context,
                                execution_task_context,
                                staged,
                                RelationFunctionSink::DeferredHostArrow,
                                &metrics,
                            )
                            .await?;
                        let QueryOutput::GpuResident(handle) = output else {
                            return Err(Error::new(
                                ErrorCode::InvalidExecutionPlan,
                                S::TEXT.operations.execute,
                                S::TEXT.messages.deferred_result,
                            )
                            .with_log_safe_fact("result_sink", "host_arrow")
                            .into());
                        };
                        let cursor = handle
                            .into_host_batch_cursor_with_host_schema(
                                output_schema,
                                context.cancellation().clone(),
                                HostBatchCursorConfig::default(),
                            )
                            .map_err(|error| {
                                DataFusionError::from(Error::from_native(
                                    S::TEXT.operations.host_batch_cursor,
                                    S::TEXT.messages.cursor_failed,
                                    error,
                                ))
                            })?;
                        let cursor = NativeHostOutputCursor::new(
                            cursor,
                            cursor_executor,
                            cursor_observation,
                            cursor_metrics,
                            context_lease,
                        );
                        if let Some(host_occurrence) = host_occurrence {
                            // The GPU result is now complete and owned by the
                            // host cursor. DataFusion may validly stop polling
                            // that cursor after LIMIT without invalidating the
                            // producer occurrence.
                            host_occurrence.complete();
                        }
                        Ok::<NativeHostOutputCursor, DataFusionError>(cursor)
                    },
                )
                .await
            },
        );
        Ok(stream)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn planned_bindings<S: RelationFunctionSpec>(
    spec: &S,
    inputs: &[Arc<dyn ExecutionPlan>],
    input_names: &[String],
    input_projections: &[Vec<usize>],
) -> Result<Vec<NativeChildBinding>> {
    inputs
        .iter()
        .zip(input_names)
        .zip(input_projections)
        .zip(spec.roles().iter())
        .map(|(((input, input_name), projection), role)| {
            if gpu_resident_producer(input.as_ref()).is_some() {
                return Ok(NativeChildBinding {
                    input_name: input_name.clone(),
                    role,
                    schema: input.schema(),
                    projection: projection.clone(),
                });
            }
            let schema = Arc::new(input.schema().project(projection).map_err(DataFusionError::from)?);
            Ok(NativeChildBinding {
                input_name: input_name.clone(),
                role,
                projection: (0..schema.fields().len()).collect(),
                schema,
            })
        })
        .collect()
}

fn plan_for_sink(host_plan: &QueryPlan, sink: RelationFunctionSink) -> QueryPlan {
    match sink {
        RelationFunctionSink::HostArrow => host_plan.clone(),
        RelationFunctionSink::DeferredHostArrow => host_plan.clone_with_deferred_host_materialization_sink(),
        RelationFunctionSink::GpuResident => host_plan.clone_with_gpu_resident_sink(),
    }
}

pub(crate) fn add_projected_relation_source(builder: &mut QueryPlanBuilder, child: &NativeChildBinding) -> NodeId {
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        child.input_name.clone(),
        Arc::clone(&child.schema),
    )));
    if child.projection.len() == child.schema.fields().len()
        && child.projection.iter().copied().eq(0..child.schema.fields().len())
    {
        source
    } else {
        let outputs = child
            .projection
            .iter()
            .map(|index| {
                let field = child.schema.field(*index);
                ProjectionOutput::new(NativeExpr::column(field.name()), field.as_ref().clone())
            })
            .collect();
        builder.add_projection(source, outputs)
    }
}

fn record_relation_staging_boundaries(metrics: &mut QueryMetrics, uploads: &[HostToDeviceUpload]) {
    for upload in uploads {
        metrics.boundary.record_host_to_device_upload(upload.rows, upload.bytes);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;

    use super::*;

    #[derive(Clone, Debug)]
    struct CountingSpec {
        lower_calls: Arc<AtomicUsize>,
    }

    impl RelationFunctionSpec for CountingSpec {
        const EXEC_NAME: &'static str = "CountingRelationFunctionExec";
        const HOST_OUTPUT_MEMORY_CONSUMER: &'static str = "CountingRelationFunctionExec[host-output-staging]";
        const TEXT: RelationFunctionText = RelationFunctionText {
            operations: RelationFunctionOperations {
                execute: "test.execute",
                execute_gpu: "test.execute_gpu",
                gpu_execution_pool: "test.gpu_execution_pool",
                host_batch_cursor: "test.host_batch_cursor",
                preflight: "test.preflight",
                with_new_children: "test.with_new_children",
            },
            messages: RelationFunctionMessages {
                preflight_failed: "preflight failed",
                staged_inputs_missing: "staged inputs missing",
                staged_inputs_poisoned: "staged inputs poisoned",
                staged_inputs_consumed: "staged inputs consumed",
                execution_failed: "execution failed",
                gpu_partition: "gpu partition",
                unexpected_host_result: "unexpected host result",
                child_count: "child count",
                host_partition: "host partition",
                gpu_pool_closed: "gpu pool closed",
                deferred_result: "deferred result",
                cursor_failed: "cursor failed",
            },
        };

        fn operation_name(&self) -> &'static str {
            "counting"
        }

        fn binding_count(&self) -> usize {
            1
        }

        fn roles(&self) -> Vec<&'static str> {
            vec!["input"]
        }

        fn required_projection(&self, _index: usize, schema: SchemaRef) -> Result<Vec<usize>> {
            Ok((0..schema.fields().len()).collect())
        }

        fn lower(&self, bindings: &[NativeChildBinding], schema: SchemaRef) -> Result<QueryPlan> {
            self.lower_calls.fetch_add(1, Ordering::SeqCst);
            let mut builder = QueryPlanBuilder::new();
            let root = add_projected_relation_source(&mut builder, &bindings[0]);
            Ok(builder.finish_with_host_sink_schema(root, schema))
        }

        fn feature_available(&self) -> bool {
            true
        }

        fn feature_disabled(&self) -> DataFusionError {
            DataFusionError::Internal("feature disabled".to_string())
        }

        fn invalid_child_count(&self) -> DataFusionError {
            DataFusionError::Internal("invalid child count".to_string())
        }
    }

    #[test]
    fn planning_queries_reuse_the_lowered_host_plan() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let input = MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None).expect("memory source");
        let lower_calls = Arc::new(AtomicUsize::new(0));
        let exec = RelationFunctionExec::try_new_with_names(
            CountingSpec { lower_calls: Arc::clone(&lower_calls) },
            vec![input as Arc<dyn ExecutionPlan>],
            Arc::clone(&schema),
            AlgeonNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
            vec!["input".to_string()],
            vec![None],
            "counting_result".to_string(),
        )
        .expect("relation function exec");

        exec.gpu_result_capability_evidence().expect("GPU-result evidence");
        exec.host_result_capability_evidence().expect("host-result evidence");
        let host_plan = exec.host_result_query_plan().expect("host plan").expect("relation host plan");

        assert_eq!(&host_plan, exec.native_plan());
        assert_eq!(lower_calls.load(Ordering::SeqCst), 1);
    }
}
