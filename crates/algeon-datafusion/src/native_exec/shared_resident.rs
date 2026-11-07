//! GPU-resident results that several units of one composed island read.
//!
//! The planner moves a subplan that several native units repeat into one
//! producer, which every reading unit lists among its GPU-resident inputs.
//! The units run nested inside the island root's execution, and the root
//! installs one result scope in the task context its producers receive. The
//! first unit to run the producer computes the result and retains it here,
//! and each later unit takes a clone. The scope drops its copy when the last
//! unit takes the result; the device buffers then live as long as a reading
//! unit holds its input map.
//!
//! GPU table function occurrences register here as well. DataFusion copies a
//! CTE or view at each reference, so one SQL invocation can appear several
//! times in an island. Occurrences with one invocation name and one output
//! scope evaluate once, and every later occurrence reads that result, which
//! keeps the references of a function with nondeterministic results
//! consistent. Separate call sites have separate names and pushed result
//! options have separate scopes. Counts exclude the children of reused
//! occurrences because those children execute only for the first reader.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, PoisonError},
};

use datafusion::{
    common::Result as DataFusionResult,
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, coop::CooperativeExec},
};
use query_planning::plan::PlanFingerprint;
use query_runtime::runtime::resident::GpuResultHandle;

use super::{AlgeonNativeExec, producer::gpu_resident_producer};
use crate::error::Result;

/// Shared resident results of one composed island execution, by producer
/// input name. The planner names each shared producer uniquely across the
/// island's units.
#[derive(Default)]
pub(crate) struct SharedResidentResults {
    results: Mutex<BTreeMap<ResidentProducerIdentity, RetainedResult>>,
    reusable_consumers: BTreeMap<ReusableProducerIdentity, usize>,
    shared_consumers: BTreeMap<String, usize>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReusableProducerIdentity {
    invocation: String,
    scope: PlanFingerprint,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ResidentProducerIdentity {
    Shared(String),
    Reusable(ReusableProducerIdentity),
}

struct RetainedResult {
    handle: GpuResultHandle,
    remaining_consumers: usize,
}

impl SharedResidentResults {
    fn for_gpu_inputs(inputs: &[Arc<dyn ExecutionPlan>]) -> Result<Self> {
        let mut reusable_consumers = BTreeMap::new();
        let mut shared_consumers = BTreeMap::new();
        for input in inputs {
            count_consumers(input.as_ref(), &mut reusable_consumers, &mut shared_consumers)?;
        }
        reusable_consumers.retain(|_, consumers| *consumers > 1);
        Ok(Self { reusable_consumers, shared_consumers, ..Self::default() })
    }

    /// Takes the result a previous unit computed for `input_name`, releasing
    /// the retained copy when this is its last consumer.
    pub(crate) fn take(&self, input_name: &str) -> Option<GpuResultHandle> {
        self.take_identity(&ResidentProducerIdentity::Shared(input_name.to_string()))
    }

    fn take_identity(&self, identity: &ResidentProducerIdentity) -> Option<GpuResultHandle> {
        let mut results = self.results.lock().unwrap_or_else(PoisonError::into_inner);
        let retained = results.get_mut(identity)?;
        retained.remaining_consumers -= 1;
        let handle = retained.handle.clone();
        if retained.remaining_consumers == 0 {
            results.remove(identity);
        }
        Some(handle)
    }

    /// Retains `handle` for the remaining units that execute a read. A reused
    /// function skips its inputs, reducing their planned consumer counts.
    pub(crate) fn retain(&self, input_name: &str, handle: &GpuResultHandle) {
        let remaining_consumers = self.shared_consumers.get(input_name).copied().unwrap_or(1) - 1;
        self.retain_identity(ResidentProducerIdentity::Shared(input_name.to_string()), handle, remaining_consumers);
    }

    fn retain_identity(
        &self,
        identity: ResidentProducerIdentity,
        handle: &GpuResultHandle,
        remaining_consumers: usize,
    ) {
        if remaining_consumers == 0 {
            return;
        }
        self.results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(identity, RetainedResult { handle: handle.clone(), remaining_consumers });
    }

    pub(crate) fn reusable_identity(&self, plan: &dyn ExecutionPlan) -> Result<Option<ReusableProducerIdentity>> {
        let Some(identity) = reusable_producer_identity(plan)? else {
            return Ok(None);
        };
        Ok(self.reusable_consumers.contains_key(&identity).then_some(identity))
    }

    pub(crate) fn take_reusable(&self, identity: &ReusableProducerIdentity) -> Option<GpuResultHandle> {
        self.take_identity(&ResidentProducerIdentity::Reusable(identity.clone()))
    }

    pub(crate) fn retain_reusable(&self, identity: ReusableProducerIdentity, handle: &GpuResultHandle) {
        let consumers = self.reusable_consumers[&identity];
        self.retain_identity(ResidentProducerIdentity::Reusable(identity), handle, consumers - 1);
    }
}

fn count_consumers(
    plan: &dyn ExecutionPlan,
    consumers: &mut BTreeMap<ReusableProducerIdentity, usize>,
    shared_consumers: &mut BTreeMap<String, usize>,
) -> Result<()> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return count_consumers(cooperative.input().as_ref(), consumers, shared_consumers);
    }
    let Some(producer) = gpu_resident_producer(plan) else {
        return Ok(());
    };
    if let Some(identity) = reusable_producer_identity(plan)? {
        let count = consumers.entry(identity).or_default();
        *count += 1;
        // A reused parent does not execute its input producers again.
        if *count > 1 {
            return Ok(());
        }
    }
    if plan.downcast_ref::<AlgeonNativeExec>().is_some_and(|unit| unit.shared_consumer_count > 1) {
        let count = shared_consumers.entry(producer.gpu_result_input_name().to_string()).or_default();
        *count += 1;
        if *count > 1 {
            return Ok(());
        }
    }
    for child in plan.children() {
        count_consumers(child.as_ref(), consumers, shared_consumers)?;
    }
    Ok(())
}

fn reusable_producer_identity(plan: &dyn ExecutionPlan) -> Result<Option<ReusableProducerIdentity>> {
    let Some(producer) = gpu_resident_producer(plan) else {
        return Ok(None);
    };
    Ok(producer
        .reusable_gpu_result_scope()?
        .map(|scope| ReusableProducerIdentity { invocation: producer.gpu_result_input_name().to_string(), scope }))
}

/// `task_context` carrying a fresh result scope for one composed island.
///
/// A nested island replaces the scope of the island that stages its host
/// input, and its units read only results of their own island.
pub(crate) fn task_context_with_shared_resident_results(
    task_context: Arc<TaskContext>,
    gpu_inputs: &[Arc<dyn ExecutionPlan>],
) -> DataFusionResult<Arc<TaskContext>> {
    let mut session_config = task_context.session_config().clone();
    session_config.set_extension(Arc::new(SharedResidentResults::for_gpu_inputs(gpu_inputs)?));
    Ok(Arc::new(TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        session_config,
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    )))
}

/// The result scope the enclosing composed island installed, if any.
pub(crate) fn shared_resident_results(task_context: &TaskContext) -> Option<Arc<SharedResidentResults>> {
    task_context.session_config().get_extension::<SharedResidentResults>()
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, coop::CooperativeExec, empty::EmptyExec,
    };
    use query_planning::{
        plan::QueryPlanBuilder,
        source::{LocalParquetFile, LocalParquetSource, SourceNode},
    };

    use super::SharedResidentResults;
    use crate::{
        cugraph_sql::{
            CugraphAlgoExec, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphGraphOptions,
            CugraphLeidenParams, CugraphSimilarityParams, GpuIslandHostBoundaryConfig, result_schema_for_params,
        },
        native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, NativeOutput, admit_for_test},
    };

    fn native_input(columns: &[&str], name: &str) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(
            columns.iter().map(|name| Field::new(*name, DataType::Int64, false)).collect::<Vec<_>>(),
        ));
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(format!("{name}.parquet")).with_row_group_rows(vec![3])],
            schema,
            columns.iter().map(|name| (*name).to_string()).collect(),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source).unwrap().clone_with_gpu_resident_sink();
        let config = AlgeonNativeExecutionConfig::default();
        Arc::new(
            AlgeonNativeExec::try_new(
                admit_for_test(&plan, &config).unwrap(),
                Vec::new(),
                NativeOutput::GpuResident { input_name: name.to_string() },
                config,
                None,
            )
            .unwrap(),
        )
    }

    fn algorithm(
        name: &str,
        params: CugraphAlgorithmParams,
        limit: Option<usize>,
        host_input: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let kind = match &params {
            CugraphAlgorithmParams::Similarity(_) => CugraphAlgorithm::Jaccard,
            CugraphAlgorithmParams::Leiden(_) => CugraphAlgorithm::Leiden,
            _ => unreachable!(),
        };
        let edges = native_input(&["src", "dst"], "edges");
        let edges = if host_input { Arc::new(EmptyExec::new(edges.schema())) as Arc<dyn ExecutionPlan> } else { edges };
        let sides = if kind.is_similarity() { vec![native_input(&["first", "second"], "pairs")] } else { Vec::new() };
        let spec = CugraphAlgorithmSpec {
            algorithm: kind,
            has_weight: false,
            has_edge_id: false,
            graph_options: CugraphGraphOptions { directed: false, ..CugraphGraphOptions::default() },
            output_schema: result_schema_for_params(kind, &params),
            params,
        };
        Arc::new(
            CugraphAlgoExec::try_new_with_gpu_result_input_name(
                spec,
                edges,
                sides,
                None,
                limit,
                AlgeonNativeExecutionConfig::default(),
                GpuIslandHostBoundaryConfig::default(),
                name.to_string(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn repeated_invocations_preserve_output_pushdown_and_independent_calls() {
        let full =
            algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
        let limited = algorithm(
            "similar",
            CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()),
            Some(1),
            false,
        );
        let independent =
            algorithm("other", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
        let wrapped: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(Arc::clone(&full)));
        let results = SharedResidentResults::for_gpu_inputs(&[
            wrapped,
            Arc::clone(&full),
            Arc::clone(&limited),
            Arc::clone(&limited),
            Arc::clone(&independent),
        ])
        .unwrap();
        let full_identity = results.reusable_identity(full.as_ref()).unwrap().unwrap();
        let limited_identity = results.reusable_identity(limited.as_ref()).unwrap().unwrap();
        assert!(full_identity != limited_identity, "LIMIT variants must retain separate outputs");
        assert_eq!(
            results.reusable_consumers[&full_identity], 2,
            "cooperative wrappers do not add a reading occurrence"
        );
        assert_eq!(results.reusable_consumers[&limited_identity], 2);
        assert!(results.reusable_identity(independent.as_ref()).unwrap().is_none());
    }

    #[test]
    fn rewritten_relation_inputs_read_one_result_of_their_invocation() {
        let original =
            algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
        let rewritten = Arc::clone(&original)
            .replace_children(
                vec![native_input(&["src", "dst"], "rewritten_edges"), native_input(&["first", "second"], "pairs")],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        let results = SharedResidentResults::for_gpu_inputs(&[Arc::clone(&original), Arc::clone(&rewritten)]).unwrap();
        let original = results.reusable_identity(original.as_ref()).unwrap().unwrap();
        let rewritten = results.reusable_identity(rewritten.as_ref()).unwrap().unwrap();
        assert!(original == rewritten, "a physical rewrite of one copy's inputs keeps the invocation's one result");
        assert_eq!(results.reusable_consumers[&original], 2);
    }

    #[test]
    fn reused_functions_register_only_executed_reads_of_shared_inputs() {
        let original =
            algorithm("similar", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, false);
        let edge = native_input(&["src", "dst"], "edges");
        let mut shared = edge.downcast_ref::<AlgeonNativeExec>().unwrap().clone();
        shared.shared_consumer_count = 3;
        let graph = Arc::clone(&original)
            .replace_children(
                vec![Arc::new(shared), native_input(&["first", "second"], "pairs")],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        let results =
            SharedResidentResults::for_gpu_inputs(&[Arc::clone(&graph), Arc::clone(&graph), Arc::clone(&graph)])
                .unwrap();
        assert_eq!(results.shared_consumers["edges"], 1, "reused Jaccard occurrences skip their shared edge input");
    }

    #[test]
    fn nondeterministic_functions_and_host_staged_inputs_reuse_one_result() {
        let leiden = algorithm("worlds", CugraphAlgorithmParams::Leiden(CugraphLeidenParams::default()), None, false);
        let host =
            algorithm("host", CugraphAlgorithmParams::Similarity(CugraphSimilarityParams::default()), None, true);
        let results = SharedResidentResults::for_gpu_inputs(&[
            Arc::clone(&leiden),
            Arc::clone(&leiden),
            Arc::clone(&host),
            Arc::clone(&host),
        ])
        .unwrap();
        for plan in [leiden, host] {
            let identity =
                results.reusable_identity(plan.as_ref()).unwrap().expect("every GPU function call is reusable");
            assert_eq!(results.reusable_consumers[&identity], 2);
        }
    }
}
