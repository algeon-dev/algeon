//! The GPU-resident producer contract composed native execution reads.
//!
//! A producer's output stays on the GPU and binds into a consumer's input map
//! by name; executing it through the ordinary DataFusion streaming path is
//! invalid.

use std::{collections::BTreeSet, sync::Arc};

use datafusion::{
    common::Result as DataFusionResult,
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, coop::CooperativeExec, metrics::ExecutionPlanMetricsSet},
};
use futures::future::BoxFuture;
use query_planning::{admission::AttemptCapabilityEvidence, plan::QueryPlan};
use query_runtime::runtime::{ExecutionContext, resident::GpuResultHandle};

use super::{admission::QueryAdmissionTicket, native_node::native_node};
use crate::error::{Error, ErrorCode, Result};

/// Crate-internal device-resident producer contract used by composed native execution.
///
/// A producer's output stays on the GPU and is bound into a composed native
/// consumer's input map instead of streaming Arrow batches. Its capability
/// evidence and row bound describe that output; they are not memory estimates
/// and must not become a predicted-usage admission veto, a second reservation,
/// or a per-operator grant.
pub(crate) trait GpuResidentProducer: Send + Sync {
    /// Name that binds this output in the consumer's GPU-resident input map. It
    /// matches the consumer plan's GPU-resident source name.
    fn gpu_result_input_name(&self) -> &str;

    /// Capability evidence for this producer's output.
    ///
    /// Composed admission aggregates it with the consumer's evidence and binds
    /// the output occurrence to the evidence's last execution scope. The
    /// default error makes composed admission reject a producer that cannot
    /// describe its output.
    fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_exec.gpu_result_capability_evidence",
            format!(
                "GPU-resident producer {} does not expose capability evidence for its output",
                self.gpu_result_input_name()
            ),
        ))
    }

    /// Capability evidence for this function when its result crosses the
    /// ordinary Arrow boundary back into DataFusion. Most producers execute
    /// the same native fragment for resident and host results; frontends whose
    /// sink changes the executable plan override this method.
    fn host_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        self.gpu_result_capability_evidence()
    }

    /// Exact native plan submitted when this frontend node materializes a host
    /// Arrow result. Resident-only internal producers return `None`.
    fn host_result_query_plan(&self) -> Result<Option<QueryPlan>> {
        Ok(None)
    }

    /// Whether this host plan completes explicit graph cardinality fact slots
    /// after Arrow staging under the same admitted occurrence.
    fn host_result_has_deferred_graph_facts(&self) -> bool {
        false
    }

    /// Guaranteed upper bound on output rows, or `None` when no bound is known.
    ///
    /// `Some(n)` must hold for every execution because consumers treat it as
    /// hard cardinality evidence; `None` must not be replaced by an estimate.
    fn hard_output_row_upper_bound(&self) -> Option<u64> {
        None
    }

    /// Returns one output key whose values are unique, when the producer's
    /// algorithm contract guarantees at most one row per key.
    fn unique_output_key_columns(&self) -> Option<&'static [&'static str]> {
        None
    }

    /// Names, in child order, under which this producer binds each child as a
    /// GPU-resident relation input when that child is itself a producer.
    ///
    /// `Some` lets the final native optimizer replace a DataFusion child with a
    /// lowered GPU-resident producer. `None` means the producer owns its
    /// children's placement (cuGraph plans them in a nested session, native
    /// fragments have no DataFusion children), so lowering leaves those
    /// children unchanged.
    fn native_relation_input_names(&self) -> Option<&[String]> {
        None
    }

    /// Rewrites this producer so it also performs a consumer plan it subsumes.
    ///
    /// Called when a composed plan has this producer as its only GPU-resident
    /// input. `Ok(Some(producer))` replaces this producer and reduces the
    /// consumer to passing the replacement's output to the host sink;
    /// `Ok(None)` leaves both unchanged.
    fn rewrite_subsumed_composed_consumer(&self, _consumer_plan: &QueryPlan) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
    }

    /// Returns every post-preparation admission ticket owned by this producer
    /// subtree. A composed root aggregates their capability evidence with its
    /// consumer evidence before it acquires one context for the whole GPU
    /// island. The default keeps ordinary relational producers ticket-free.
    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        Vec::new()
    }

    /// Live metric set this producer records its native execution into.
    ///
    /// Execution clones share the set, and metrics register only while the
    /// producer executes, so the sealed attempt projection reads it after
    /// execution. Producers that record no native metrics return `None`.
    fn execution_metrics(&self) -> Option<&ExecutionPlanMetricsSet> {
        None
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

    /// Executes the prepared producer within `native_context` and returns
    /// [`gpu_result_input_name`](Self::gpu_result_input_name) with the
    /// GPU-resident output handle.
    ///
    /// Callers invoke this on the plan returned by
    /// [`prepare_execution_plan`](Self::prepare_execution_plan), after
    /// acquiring the admitted native context for the whole composed island;
    /// the output belongs to that context. Current producers expose one output
    /// partition and reject any `partition` other than 0.
    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>>;
}

pub(crate) fn gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    native_node(plan).and_then(|node| node.gpu_resident_producer()).or_else(|| {
        // DataFusion adds this scheduling-only wrapper around leaf plans after
        // physical planning. It does not alter rows, partitions, or device
        // ownership, so composed execution may safely reach the inner producer
        // directly. Do not generalize this to repartition/coalesce wrappers:
        // those are row-stream operators and would be host boundaries.
        plan.downcast_ref::<CooperativeExec>()
            .and_then(|cooperative| native_node(cooperative.input().as_ref()))
            .and_then(|node| node.gpu_resident_producer())
    })
}

pub(crate) fn gpu_resident_producer_input_name(plan: &dyn ExecutionPlan) -> Option<String> {
    gpu_resident_producer(plan).map(|producer| producer.gpu_result_input_name().to_string())
}

pub(crate) fn gpu_resident_producer_hard_output_row_upper_bound(plan: &dyn ExecutionPlan) -> Option<u64> {
    gpu_resident_producer(plan).and_then(GpuResidentProducer::hard_output_row_upper_bound)
}

pub(crate) fn gpu_resident_producer_unique_output_key(plan: &dyn ExecutionPlan) -> Option<BTreeSet<String>> {
    let key: BTreeSet<String> = gpu_resident_producer(plan).and_then(|producer| {
        producer.unique_output_key_columns().map(|columns| columns.iter().map(|column| (*column).to_owned()).collect())
    })?;
    key.iter().all(|column| plan.schema().fields().iter().any(|field| field.name() == column)).then_some(key)
}

pub(crate) fn rewrite_gpu_resident_producer_for_subsumed_composed_consumer(
    plan: &dyn ExecutionPlan,
    consumer_plan: &QueryPlan,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
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
        "algeon_datafusion.native_gpu_producer.execute_gpu",
        "execution plan is not a GPU-resident producer",
    )
    .with_fact("plan_name", plan.name())
    .into())
}
