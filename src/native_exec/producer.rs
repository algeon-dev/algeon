//! GPU-resident native fragment producer for composed execution.
//!
//! Producers are internal children of composed native execution. Their public
//! wrapper type is exposed for diagnostics and plan inspection, but executing
//! them through the ordinary DataFusion streaming path is invalid.

use std::sync::{Arc, Mutex, OnceLock};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        coop::CooperativeExec,
        metrics::{ExecutionPlanMetricsSet, MetricsSet},
    },
};
use futures::future::BoxFuture;
use nexus_query_engine::{
    exec::{execute_query_plan_observed, output::QueryOutput},
    observability::RuntimeObservationConfig,
    plan::QueryPlan,
    runtime::{ExecutionContext, SourceStorageProfile, resident::GpuResultHandle},
};

use super::{
    admission::QueryAdmissionTicket,
    common::{native_plan_one_line, native_query_runtime_error, write_native_plan_tree},
    metrics::{NativeExecutionMetricsCapture, NexusNativeMetrics},
};
#[cfg(any(test, feature = "test-support"))]
use super::{
    common::{comma_separated, conservative_properties, gpu_resident_source_names},
    metrics::NativeMetricsCapture,
};
use crate::error::{Error, ErrorCode};

/// Crate-internal device-resident producer contract used by composed native execution.
pub(crate) trait GpuResidentProducer: Send + Sync {
    fn gpu_result_input_name(&self) -> &str;

    fn rewrite_subsumed_composed_consumer(
        &self,
        _consumer_plan: &QueryPlan,
    ) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
    }

    /// Returns every post-preparation admission ticket owned by this producer
    /// subtree.  A composed root aggregates these immutable proofs with its
    /// own consumer proof before it acquires one context for the whole GPU
    /// island.  The default keeps ordinary relational producers ticket-free.
    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        Vec::new()
    }

    /// Performs fallible async preparation before callers acquire native GPU
    /// admission. Frontends return an equivalent producer plan carrying
    /// resolved metadata and any bounded host inputs required by the GPU island.
    fn prepare_execution_plan<'a>(
        &'a self,
        plan: Arc<dyn ExecutionPlan>,
        _task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move { Ok(plan) })
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>>;
}

/// A frontend hook that recognizes one of its GPU-resident producer nodes.
pub(crate) type GpuResidentProducerDiscovery = for<'a> fn(&'a dyn ExecutionPlan) -> Option<&'a dyn GpuResidentProducer>;

static GPU_RESIDENT_PRODUCER_DISCOVERIES: OnceLock<Mutex<Vec<GpuResidentProducerDiscovery>>> = OnceLock::new();

/// Register a frontend's GPU-resident producer recognizer.
///
/// Registration is idempotent for the same function. Core execution never
/// imports a concrete frontend to discover producer nodes.
pub(crate) fn register_gpu_resident_producer_discovery(discovery: GpuResidentProducerDiscovery) {
    let mut discoveries = GPU_RESIDENT_PRODUCER_DISCOVERIES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("GPU-resident producer discovery registry mutex cannot be poisoned");
    if !discoveries.iter().any(|candidate| std::ptr::fn_addr_eq(*candidate, discovery)) {
        discoveries.push(discovery);
    }
}

/// Internal GPU-resident native fragment producer.
///
/// Valid only under composed native execution; returns a device-resident handle,
/// not a streamable DataFusion plan.
#[derive(Debug, Clone)]
pub struct NexusNativeGpuProducerExec {
    input_name: String,
    plan: QueryPlan,
    metrics_capture: Option<NativeExecutionMetricsCapture>,
    plan_task_mapping_datafusion_shape: Option<String>,
    lakehouse_footer_pruning: bool,
    source_storage_profiles: Vec<SourceStorageProfile>,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl NexusNativeGpuProducerExec {
    /// Create a native producer fragment for composed execution.
    ///
    /// Requires a GPU-resident sink and no external GPU-resident inputs.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn try_new(input_name: impl Into<String>, plan: QueryPlan) -> crate::error::Result<Self> {
        Self::try_new_with_metrics_capture(input_name, plan, None)
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn try_new_for_test(input_name: impl Into<String>, plan: QueryPlan) -> crate::error::Result<Self> {
        Self::try_new(input_name, plan)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn try_new_with_metrics_capture(
        input_name: impl Into<String>,
        plan: QueryPlan,
        metrics_capture: Option<NativeMetricsCapture>,
    ) -> crate::error::Result<Self> {
        Self::try_new_with_metrics_capture_and_shape(input_name, plan, metrics_capture, None, false, Vec::new())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn try_new_with_metrics_capture_and_shape(
        input_name: impl Into<String>,
        plan: QueryPlan,
        metrics_capture: Option<NativeMetricsCapture>,
        plan_task_mapping_datafusion_shape: Option<&str>,
        lakehouse_footer_pruning: bool,
        source_storage_profiles: Vec<SourceStorageProfile>,
    ) -> crate::error::Result<Self> {
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

        let metrics_capture = Some(metrics_capture.unwrap_or_else(NativeMetricsCapture::standalone).new_execution());
        Ok(Self {
            input_name,
            cache: conservative_properties(Arc::clone(plan.output_schema())),
            plan,
            metrics_capture,
            plan_task_mapping_datafusion_shape: plan_task_mapping_datafusion_shape.map(ToString::to_string),
            lakehouse_footer_pruning,
            source_storage_profiles,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub(crate) fn native_plan(&self) -> &QueryPlan {
        &self.plan
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn native_plan_for_test(&self) -> &QueryPlan {
        self.native_plan()
    }

    pub(crate) fn with_metrics_capture(mut self, metrics_capture: Option<NativeExecutionMetricsCapture>) -> Self {
        self.metrics_capture = metrics_capture;
        self
    }

    /// Execute this producer inside a composed exec's shared native context.
    ///
    /// The result stays on the GPU and is returned as `(input_name, GpuResultHandle)` so the composed
    /// consumer can populate its `GpuResidentInputMap` without host materialization.
    fn execute_gpu_result_with_context(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        plan: &QueryPlan,
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

        let metrics = NexusNativeMetrics::new_with_capture_and_shape(
            &self.metrics,
            partition,
            self.metrics_capture.clone(),
            self.plan_task_mapping_datafusion_shape.as_deref(),
        );
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let runtime_observation_config = self.metrics_capture.as_ref().map_or_else(
            RuntimeObservationConfig::native_execution,
            NativeExecutionMetricsCapture::runtime_observation_config,
        );
        let outcome = execute_query_plan_observed(native_context, plan, runtime_observation_config);
        let output = metrics.finalize_query_outcome(outcome).map_err(|error| {
            DataFusionError::from(native_query_runtime_error(
                "datafusion_nexus.native_gpu_producer.execute_gpu",
                "nexus native GPU-resident fragment execution failed",
                error,
            ))
        })?;
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

impl GpuResidentProducer for NexusNativeGpuProducerExec {
    fn gpu_result_input_name(&self) -> &str {
        self.input_name()
    }

    fn prepare_execution_plan<'a>(
        &'a self,
        plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            if !self.lakehouse_footer_pruning {
                return Ok(plan);
            }
            let runtime_env = task_context.runtime_env();
            let resolved_plan = crate::native::lakehouse_pruning::prune_native_query_plan(
                self.plan.clone(),
                &runtime_env,
                &self.source_storage_profiles,
            )
            .await;
            let mut prepared = self.clone();
            prepared.plan = resolved_plan;
            prepared.lakehouse_footer_pruning = false;
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
            let plan = if self.lakehouse_footer_pruning {
                let runtime_env = task_context.runtime_env();
                crate::native::lakehouse_pruning::prune_native_query_plan(
                    self.plan.clone(),
                    &runtime_env,
                    &self.source_storage_profiles,
                )
                .await
            } else {
                self.plan.clone()
            };
            NexusNativeGpuProducerExec::execute_gpu_result_with_context(self, native_context, partition, &plan)
        })
    }
}

fn native_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    if let Some(producer) = plan.downcast_ref::<NexusNativeGpuProducerExec>() {
        return Some(producer);
    }
    None
}

fn ensure_native_gpu_resident_producer_registered() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| register_gpu_resident_producer_discovery(native_gpu_resident_producer));
}

pub(crate) fn gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    ensure_native_gpu_resident_producer_registered();
    let discoveries = {
        GPU_RESIDENT_PRODUCER_DISCOVERIES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .expect("GPU-resident producer discovery registry mutex cannot be poisoned")
            .clone()
    };
    discoveries.iter().find_map(|discovery| discovery(plan)).or_else(|| {
        // DataFusion adds this scheduling-only wrapper around leaf plans after
        // physical planning. It does not alter rows, partitions, or device
        // ownership, so composed execution may safely reach the inner producer
        // directly. Do not generalize this to repartition/coalesce wrappers:
        // those are row-stream operators and would be host boundaries.
        plan.downcast_ref::<CooperativeExec>()
            .and_then(|cooperative| discoveries.iter().find_map(|discovery| discovery(cooperative.input().as_ref())))
    })
}

pub(crate) fn gpu_resident_producer_input_name(plan: &dyn ExecutionPlan) -> Option<String> {
    gpu_resident_producer(plan).map(|producer| producer.gpu_result_input_name().to_string())
}

pub(crate) fn rewrite_gpu_resident_producer_for_subsumed_composed_consumer(
    plan: &dyn ExecutionPlan,
    consumer_plan: &QueryPlan,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    gpu_resident_producer(plan)
        .map(|producer| producer.rewrite_subsumed_composed_consumer(consumer_plan))
        .unwrap_or(Ok(None))
}

/// Returns the admission tickets emitted by a fully prepared producer
/// subtree.  This is deliberately discovered through the producer SPI rather
/// than the attempt extension: a composed query may have several producers,
/// and the extension only retains the most recently bound ticket.
pub(crate) fn prepared_gpu_resident_producer_admission_tickets(
    plan: &dyn ExecutionPlan,
) -> Vec<Arc<QueryAdmissionTicket>> {
    gpu_resident_producer(plan).map_or_else(Vec::new, GpuResidentProducer::prepared_admission_tickets)
}

/// Resolves producer metadata and bounded host inputs before a composed query
/// acquires GPU admission or enters the dedicated GPU executor.
pub(crate) async fn prepare_gpu_resident_producer_for_execution(
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let Some(producer) = gpu_resident_producer(plan.as_ref()) else {
        return Ok(plan);
    };
    producer.prepare_execution_plan(Arc::clone(&plan), task_context).await
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

impl DisplayAs for NexusNativeGpuProducerExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => write!(
                f,
                "NexusNativeGpuProducerExec: input_name={}, native_nodes={}, sink=gpu_resident, output_boundary=device_resident, edge_normalize_inserted={}",
                self.input_name,
                self.plan.node_count(),
                self.plan.contains_edge_normalize(),
            ),
            DisplayFormatType::Verbose => write!(
                f,
                "NexusNativeGpuProducerExec: input_name={}, native_nodes={}, sink=gpu_resident, output_boundary=device_resident, edge_normalize_inserted={}, native_plan={}",
                self.input_name,
                self.plan.node_count(),
                self.plan.contains_edge_normalize(),
                native_plan_one_line(&self.plan),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "input_name={}", self.input_name)?;
                writeln!(f, "native_nodes={}", self.plan.node_count())?;
                writeln!(f, "sink=gpu_resident")?;
                writeln!(f, "output_boundary=device_resident")?;
                writeln!(f, "edge_normalize_inserted={}", self.plan.contains_edge_normalize())?;
                write_native_plan_tree(f, &self.plan)
            },
        }
    }
}

impl ExecutionPlan for NexusNativeGpuProducerExec {
    fn name(&self) -> &str {
        "NexusNativeGpuProducerExec"
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

#[cfg(test)]
mod registry_tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use arrow::datatypes::Schema;
    use datafusion::physical_plan::{ExecutionPlan, empty::EmptyExec};

    use super::*;

    static REENTRANT_REGISTRATION_RAN: AtomicBool = AtomicBool::new(false);

    fn no_match(_: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
        None
    }

    fn register_while_discovering(_: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
        register_gpu_resident_producer_discovery(no_match);
        REENTRANT_REGISTRATION_RAN.store(true, Ordering::Release);
        None
    }

    #[test]
    fn discovery_callbacks_run_without_holding_the_registry_lock() {
        REENTRANT_REGISTRATION_RAN.store(false, Ordering::Release);
        register_gpu_resident_producer_discovery(register_while_discovering);
        let plan = EmptyExec::new(Arc::new(Schema::empty()));

        assert!(gpu_resident_producer(&plan).is_none());
        assert!(REENTRANT_REGISTRATION_RAN.load(Ordering::Acquire));
    }
}
