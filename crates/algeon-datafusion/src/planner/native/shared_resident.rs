//! Subplans shared by the native units of one composed attempt.
//!
//! A query root that lowers over GPU functions becomes one composed island
//! made of the root unit, the functions it reads, and the GPU-resident units
//! lowered from their inputs: cuVS and cuML relation inputs, and cuGraph edge
//! and side inputs, which graph planning lowers in a nested session. All of
//! them run in the context the root acquires. DataFusion inlines each CTE
//! reference, so a CTE that feeds two function inputs, or one input and a
//! join in the root, is planned in two units.
//!
//! This pass hands the island's unit plans to the shared-resident extraction
//! stage, which owns the matching and the retention rule. It admits the
//! producers that stage creates and the units rewritten to read them, then
//! rebinds the island so each reading unit lists the producer among its
//! GPU-resident inputs. The island root's execution computes each shared
//! producer once and releases it after its last reading unit. A graph reads
//! a rewritten edge unit like any composed edge producer, and its admission
//! evidence includes the shared producer's.

use std::sync::Arc;

use datafusion::{
    common::Result as DataFusionResult,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
};
use query_planning::{
    admission::{AdmissionDecision, AdmittedPlan},
    optimizer::{
        NativeOptimizerDiagnostics, SharedResidentExtractionDecision, SharedResidentExtractionDiagnostics,
        SharedResidentExtractionReason, SharedResidentFragments, SharedResidentProducerPlan,
        extract_shared_resident_fragments,
    },
    plan::QueryPlan,
};

use super::capability_envelope::native_admission_decision;
use crate::{
    error::Error,
    native_exec::{
        AlgeonNativeExec, AlgeonNativeExecutionConfig, gpu_resident_producer, gpu_resident_producer_input_name,
    },
};

/// A composed island after shared subplans moved into resident producers.
pub(super) struct SharedIsland {
    pub(super) root: Arc<dyn ExecutionPlan>,
    /// The root unit's admitted plan when it now reads a shared producer.
    pub(super) root_admission: Option<Arc<AdmittedPlan>>,
    pub(super) native_optimizer: NativeOptimizerDiagnostics,
}

/// Shares the subplans that the units of the island rooted at `root` repeat.
///
/// Returns `None` when `root` is not a host unit over GPU-resident inputs.
/// When admission refuses a producer or a rewritten unit, the island keeps
/// its units and the diagnostics record the refusal.
pub(super) fn share_island_subplans(
    root: &Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> DataFusionResult<Option<SharedIsland>> {
    let Some(root_unit) = root.downcast_ref::<AlgeonNativeExec>() else {
        return Ok(None);
    };
    if !root_unit.materializes_host() || root_unit.gpu_inputs().is_empty() {
        return Ok(None);
    }
    let mut units = Vec::new();
    collect_units(root, true, &mut units);
    let SharedResidentFragments { fragments, producers, mut diagnostics } = extract_shared_resident_fragments(
        units.iter().map(|unit| unit.native_plan().clone()).collect(),
    )
    .map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native_optimizer.shared_resident",
            "shared-resident extraction across native units produced a malformed plan",
            error,
        )
    })?;
    let admitted = if producers.is_empty() {
        None
    } else {
        admit_shared_plans(&fragments, &producers, units.len(), native_execution_config)?
    };
    let Some((admitted_units, shared_inputs)) = admitted else {
        if !producers.is_empty() {
            record_not_admitted(&mut diagnostics);
        }
        return Ok(Some(SharedIsland {
            root: Arc::clone(root),
            root_admission: None,
            native_optimizer: shared_resident_diagnostics(diagnostics),
        }));
    };
    let root_admission = admitted_units[0].clone();
    let mut rebind = IslandRebind { admitted_units, shared_inputs, next_unit: 0 };
    Ok(Some(SharedIsland {
        root: rebind.rebind(root, true)?,
        root_admission,
        native_optimizer: shared_resident_diagnostics(diagnostics),
    }))
}

/// Admitted plans of the units that now read shared producers, and the
/// producer nodes each unit reads, by unit index.
type AdmittedIsland = (Vec<Option<Arc<AdmittedPlan>>>, Vec<Vec<Arc<dyn ExecutionPlan>>>);

/// Admits the shared producers and the units rewritten to read them, or
/// returns `None` when admission refuses any of them.
fn admit_shared_plans(
    fragments: &[QueryPlan],
    producers: &[SharedResidentProducerPlan],
    unit_count: usize,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> DataFusionResult<Option<AdmittedIsland>> {
    let mut shared_inputs = vec![Vec::<Arc<dyn ExecutionPlan>>::new(); unit_count];
    for producer in producers {
        let AdmissionDecision::Admitted(admitted) = native_admission_decision(&producer.plan, native_execution_config)?
        else {
            return Ok(None);
        };
        let producer_exec: Arc<dyn ExecutionPlan> = Arc::new(AlgeonNativeExec::try_new_shared_producer(
            admitted,
            producer.input_name.clone(),
            producer.consumer_fragments.len(),
            native_execution_config.clone(),
        )?);
        for fragment in &producer.consumer_fragments {
            shared_inputs[*fragment].push(Arc::clone(&producer_exec));
        }
    }
    let mut admitted_units = vec![None; unit_count];
    for (index, plan) in fragments.iter().enumerate() {
        if shared_inputs[index].is_empty() {
            continue;
        }
        let AdmissionDecision::Admitted(admitted) = native_admission_decision(plan, native_execution_config)? else {
            return Ok(None);
        };
        admitted_units[index] = Some(admitted);
    }
    Ok(Some((admitted_units, shared_inputs)))
}

/// Records that every extracted subplan stays in its consumers because
/// admission refused a resident plan of the island.
fn record_not_admitted(diagnostics: &mut SharedResidentExtractionDiagnostics) {
    for candidate in &mut diagnostics.candidates {
        if candidate.decision == SharedResidentExtractionDecision::Extracted {
            candidate.decision = SharedResidentExtractionDecision::Rejected;
            candidate.reason = SharedResidentExtractionReason::ResidentPlanNotAdmitted;
            candidate.resident_input_name = None;
            diagnostics.rejected_count += 1;
        }
    }
    diagnostics.extracted_count = 0;
}

fn shared_resident_diagnostics(diagnostics: SharedResidentExtractionDiagnostics) -> NativeOptimizerDiagnostics {
    NativeOptimizerDiagnostics { shared_resident_extraction: diagnostics, ..NativeOptimizerDiagnostics::default() }
}

/// Whether `plan` is a native unit of the island: its root, or a unit whose
/// output stays on the device for a consumer in the island. A host unit
/// below a function input is staged through DataFusion in its own context.
fn island_unit(plan: &Arc<dyn ExecutionPlan>, is_root: bool) -> Option<&AlgeonNativeExec> {
    plan.downcast_ref::<AlgeonNativeExec>().filter(|unit| is_root || !unit.materializes_host())
}

/// Whether `plan` is a GPU function the island reads, or the scheduling
/// wrapper DataFusion places above one or above a unit. A function runs its
/// GPU-resident inputs in the island's context, so they belong to the island.
fn reads_island_inputs(plan: &dyn ExecutionPlan) -> bool {
    gpu_resident_producer(plan).is_some()
}

/// Collects the island's units in pre-order, the order
/// [`IslandRebind::rebind`] visits them.
fn collect_units<'a>(plan: &'a Arc<dyn ExecutionPlan>, is_root: bool, units: &mut Vec<&'a AlgeonNativeExec>) {
    if let Some(unit) = island_unit(plan, is_root) {
        units.push(unit);
        for input in unit.gpu_inputs() {
            collect_units(input, false, units);
        }
    } else if reads_island_inputs(plan.as_ref()) {
        for child in plan.children() {
            collect_units(child, false, units);
        }
    }
}

struct IslandRebind {
    admitted_units: Vec<Option<Arc<AdmittedPlan>>>,
    shared_inputs: Vec<Vec<Arc<dyn ExecutionPlan>>>,
    next_unit: usize,
}

impl IslandRebind {
    fn rebind(&mut self, plan: &Arc<dyn ExecutionPlan>, is_root: bool) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if let Some(unit) = island_unit(plan, is_root) {
            let index = self.next_unit;
            self.next_unit += 1;
            let mut inputs = unit
                .gpu_inputs()
                .iter()
                .map(|input| self.rebind(input, false))
                .collect::<DataFusionResult<Vec<_>>>()?;
            let inputs_changed = inputs.iter().zip(unit.gpu_inputs()).any(|(new, old)| !Arc::ptr_eq(new, old));
            let shared = std::mem::take(&mut self.shared_inputs[index]);
            let Some(admitted) = self.admitted_units[index].take() else {
                return if inputs_changed {
                    Arc::clone(plan)
                        .replace_children(inputs, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
                } else {
                    Ok(Arc::clone(plan))
                };
            };
            inputs.extend(shared);
            inputs.sort_by(|left, right| {
                gpu_resident_producer_input_name(left.as_ref()).cmp(&gpu_resident_producer_input_name(right.as_ref()))
            });
            return Ok(Arc::new(unit.with_rebound_plan(admitted, inputs)?));
        }
        if !reads_island_inputs(plan.as_ref()) {
            return Ok(Arc::clone(plan));
        }
        let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
        let rebound = children.iter().map(|child| self.rebind(child, false)).collect::<DataFusionResult<Vec<_>>>()?;
        if rebound.iter().zip(&children).all(|(new, old)| Arc::ptr_eq(new, old)) {
            return Ok(Arc::clone(plan));
        }
        Arc::clone(plan).replace_children(rebound, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use arrow::{
        array::{ArrayRef, Float32Array, Int64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{ParquetReadOptions, SessionContext},
    };
    use parquet::arrow::ArrowWriter;
    use query_planning::{plan::PlanNode, source::SourceNode};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        cuvs_sql::CuvsAlgoExec,
        native_exec::below_native_admission,
        planner::{AlgeonNativeOptimizerConfig, NativePlanningDiagnostics, plan_with_native_diagnostics},
        session::AlgeonSessionStateBuilderExt,
    };

    /// Plans `sql` over a `points(id, d0, d1)` Parquet table with two rows
    /// per id. The directory must outlive the plan's footer reads.
    async fn plan_over_points(sql: &str) -> (TempDir, Arc<dyn ExecutionPlan>, NativePlanningDiagnostics) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("points.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("d0", DataType::Float32, false),
            Field::new("d1", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values((0..16).map(|row| row % 8))) as ArrayRef,
                Arc::new(Float32Array::from_iter_values((0..16).map(|row| if row % 8 < 4 { 0.0 } else { 10.0 }))),
                Arc::new(Float32Array::from_iter_values((0..16).map(|row| if row % 8 < 4 { 1.0 } else { 11.0 }))),
            ],
        )
        .expect("points batch");
        let mut writer = ArrowWriter::try_new(File::create(&path).expect("points file"), schema, None).expect("writer");
        writer.write(&batch).expect("write points");
        writer.close().expect("close points");
        let state = SessionStateBuilder::new_with_default_features()
            .with_algeon_native(AlgeonNativeOptimizerConfig::default())
            .try_with_cuvs_sql()
            .expect("install cuVS SQL")
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet("points", path.to_str().expect("UTF-8 path"), ParquetReadOptions::default())
            .await
            .expect("register points");
        let logical = ctx.state().create_logical_plan(sql).await.expect("logical plan");
        let (plan, diagnostics) = plan_with_native_diagnostics(&ctx.state(), &logical).await.expect("physical plan");
        (dir, plan, diagnostics)
    }

    /// The root unit and the unit lowered from the K-means input relation.
    fn root_and_kmeans_input(plan: &Arc<dyn ExecutionPlan>) -> (&AlgeonNativeExec, &AlgeonNativeExec) {
        let root = below_native_admission(plan).downcast_ref::<AlgeonNativeExec>().expect("the query root lowers");
        let kmeans = root
            .gpu_inputs()
            .iter()
            .find_map(|input| input.downcast_ref::<CuvsAlgoExec>())
            .expect("the root reads the K-means result on the device");
        let input = kmeans.children()[0].downcast_ref::<AlgeonNativeExec>().expect("the K-means input lowers");
        (root, input)
    }

    fn shared_producers(unit: &AlgeonNativeExec) -> Vec<&AlgeonNativeExec> {
        unit.gpu_inputs()
            .iter()
            .filter_map(|input| input.downcast_ref::<AlgeonNativeExec>())
            .filter(|input| input.gpu_resident_input_name().is_some_and(|name| name.starts_with(SHARED_PREFIX)))
            .collect()
    }

    fn resident_source_names(plan: &QueryPlan) -> Vec<&str> {
        plan.nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) => Some(source.input_name.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    const SHARED_PREFIX: &str = "__algeon_shared_resident_";

    #[tokio::test]
    async fn an_aggregate_cte_read_by_kmeans_and_a_join_is_computed_once() {
        let (_dir, plan, diagnostics) = plan_over_points(
            "WITH centers AS (\
               SELECT id, CAST(SUM(d0) AS REAL) AS d0, CAST(SUM(d1) AS REAL) AS d1 FROM points GROUP BY id), \
             assignments AS (\
               SELECT id, cluster_id FROM cuvs_kmeans(\
                 input => (SELECT id, d0, d1 FROM centers ORDER BY id), n_clusters => 2)) \
             SELECT a.cluster_id, COUNT(*) AS members, SUM(c.d0) AS d0_total, SUM(c.d1) AS d1_total \
             FROM assignments a JOIN centers c ON c.id = a.id GROUP BY a.cluster_id",
        )
        .await;
        let (root, kmeans_input) = root_and_kmeans_input(&plan);

        let [shared] = shared_producers(root)[..] else { panic!("the root reads one shared producer") };
        let shared_name = shared.gpu_resident_input_name().expect("a producer output name");
        assert!(matches!(shared.native_plan().root_node().kind, PlanNode::Aggregate(_)));
        let [kmeans_shared] = shared_producers(kmeans_input)[..] else {
            panic!("the K-means input reads one shared producer")
        };
        assert_eq!(kmeans_shared.gpu_resident_input_name(), Some(shared_name));
        assert!(resident_source_names(root.native_plan()).contains(&shared_name));
        assert_eq!(resident_source_names(kmeans_input.native_plan()), [shared_name]);
        // The K-means input keeps its own ORDER BY above the shared source.
        assert!(matches!(kmeans_input.native_plan().root_node().kind, PlanNode::Sort(_)));
        assert!(
            !kmeans_input.native_plan().nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))),
            "the K-means input must not aggregate the points again"
        );

        let extracted = diagnostics
            .native_optimizer
            .shared_resident_extraction
            .candidates
            .iter()
            .filter(|candidate| candidate.decision == SharedResidentExtractionDecision::Extracted)
            .map(|candidate| (candidate.attempt_fragment, candidate.resident_input_name.as_deref()))
            .collect::<Vec<_>>();
        assert_eq!(extracted, [(Some(0), Some(shared_name)), (Some(1), Some(shared_name))]);
    }

    #[tokio::test]
    async fn a_filtered_scan_cte_read_by_kmeans_and_a_join_is_recomputed() {
        let (_dir, plan, diagnostics) = plan_over_points(
            "WITH kept AS (SELECT id, d0, d1 FROM points WHERE d0 >= 0), \
             assignments AS (\
               SELECT id, cluster_id FROM cuvs_kmeans(\
                 input => (SELECT id, d0, d1 FROM kept ORDER BY id), n_clusters => 2)) \
             SELECT a.cluster_id, COUNT(*) AS members, SUM(k.d0) AS d0_total, SUM(k.d1) AS d1_total \
             FROM assignments a JOIN kept k ON k.id = a.id GROUP BY a.cluster_id",
        )
        .await;
        let (root, kmeans_input) = root_and_kmeans_input(&plan);

        assert!(shared_producers(root).is_empty());
        assert!(kmeans_input.gpu_inputs().is_empty());
        let rejected_fragments = diagnostics
            .native_optimizer
            .shared_resident_extraction
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.decision == SharedResidentExtractionDecision::Rejected
                    && candidate.reason == SharedResidentExtractionReason::RootDoesNotReduceRows
            })
            .filter_map(|candidate| candidate.attempt_fragment)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(rejected_fragments, [0, 1].into());
    }
}
