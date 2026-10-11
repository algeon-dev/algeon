//! Native fragments under DataFusion parents.
//!
//! A query root that does not lower as a whole stays on DataFusion, and each
//! child is tried as a host-output native fragment. A fragment replaces its
//! DataFusion child only when the rebuilt parent keeps every contract it held
//! over that child:
//!
//! - the input schema;
//! - boundedness: a fragment is bounded and emits its rows once its input is
//!   exhausted, so it can stand in only for a bounded child, whose parent
//!   still completes and receives the same rows;
//! - DataFusion's sanity check of the parent's input ordering, distribution
//!   and pipeline requirements and of its executable invariants. One
//!   partition satisfies any key distribution, so a co-partitioned join
//!   keeps a single-partition fragment only when its other inputs have one
//!   partition too.
//!
//! A kept fragment adds its report rows and diagnostics to the attempt. A
//! fragment an operator above it cannot keep is reported with that reason,
//! and the search continues below it. A child the cost model leaves on
//! DataFusion keeps its subtree there and reports the cost decision. A child
//! that does not lower gets no report row of its own; the root's row records
//! the first contract the plan misses. Admission evidence is collected later
//! from the final plan, so only kept fragments reach it.

use std::sync::Arc;

use datafusion::{
    common::{Result as DataFusionResult, config::OptimizerOptions},
    datasource::{
        physical_plan::{FileScanConfigBuilder, FileSource, ParquetSource},
        source::DataSourceExec,
    },
    physical_optimizer::sanity_checker::check_plan_sanity,
    physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, ReplaceChildrenOptions,
        coalesce_partitions::CoalescePartitionsExec, coop::CooperativeExec, filter::FilterExec,
        repartition::RepartitionExec,
    },
};
use query_planning::{
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions},
    source::SourceDiagnostics,
};

use super::{
    NativeCapabilityContractDiagnostics,
    attempt::{
        NativeCostSelection, NativeOptimizationAttempt, NativeReplacementSink,
        replace_lowered_root_or_keep_plan_for_sink,
    },
    plan_contains_gpu_execution,
};
use crate::{
    capability::CapabilityReason,
    native::{lower_physical_plan_to_native, scan_predicate_exactly_matches_filter},
    native_exec::AlgeonNativeExecutionConfig,
    physical_plan::DataFusionCpuBoundaryExec,
    report::{CandidateOutcome, NotSupportedDetail, PlanBlocker, PlanningReport},
};

/// Places native fragments below a query root that did not lower as a whole.
pub(super) struct FragmentSearch<'a> {
    pub(super) native_execution_config: &'a AlgeonNativeExecutionConfig,
    pub(super) native_optimizer_options: &'a NativeOptimizerOptions,
    pub(super) cost_selection: NativeCostSelection,
    pub(super) optimizer_options: &'a OptimizerOptions,
}

/// Report rows and diagnostics of the fragments one search kept, abandoned or
/// left on DataFusion by cost.
#[derive(Default)]
struct FragmentEvidence {
    report: PlanningReport,
    native_optimizer: NativeOptimizerDiagnostics,
    source_diagnostics: Vec<SourceDiagnostics>,
    capability_contract: NativeCapabilityContractDiagnostics,
}

impl FragmentEvidence {
    fn of_attempt(attempt: NativeOptimizationAttempt) -> Self {
        Self {
            report: attempt.report,
            native_optimizer: attempt.native_optimizer,
            source_diagnostics: attempt.source_diagnostics,
            capability_contract: attempt.capability_contract,
        }
    }

    fn append(&mut self, other: Self) {
        self.report.append(other.report);
        self.native_optimizer.merge(other.native_optimizer);
        self.source_diagnostics.extend(other.source_diagnostics);
        self.capability_contract.merge(other.capability_contract);
    }

    /// Records the selected candidates of a replacement `parent` could not
    /// keep as not supported for that reason. Their diagnostics describe
    /// native plans that do not run, so they are dropped.
    fn append_abandoned(&mut self, abandoned: Self, parent: &str) {
        self.report.append(PlanningReport::from_outcomes(abandoned.report.outcomes().iter().map(
            |outcome| match outcome {
                CandidateOutcome::Selected { candidate } => CandidateOutcome::NotSupported {
                    candidate: candidate.clone(),
                    capability_reason: CapabilityReason::NativeFragmentParentContractUnsupported,
                    detail: NotSupportedDetail {
                        blocker: Some(Box::new(PlanBlocker::ParentOperator(parent.to_string()))),
                        ..NotSupportedDetail::default()
                    },
                },
                CandidateOutcome::NotSupported { .. } | CandidateOutcome::NotSelectedByCost { .. } => outcome.clone(),
            },
        )));
    }
}

/// A child's replacement and the evidence it carries.
struct Replacement {
    plan: Arc<dyn ExecutionPlan>,
    evidence: FragmentEvidence,
    /// Whether the child lowered as one fragment. `false` means the child was
    /// rebuilt over fragments below it.
    whole: bool,
}

impl FragmentSearch<'_> {
    /// Adds native fragments below the root of `root_attempt` when the root
    /// did not lower and the cost model did not keep it on DataFusion.
    pub(super) fn below_root(
        &self,
        root_attempt: NativeOptimizationAttempt,
    ) -> DataFusionResult<NativeOptimizationAttempt> {
        if root_attempt.rewritten || root_attempt.report.not_selected_by_cost_count() > 0 {
            return Ok(root_attempt);
        }
        let mut evidence = FragmentEvidence::default();
        let plan = self.below(&root_attempt.plan, None, &mut evidence)?;
        let mut attempt = root_attempt;
        attempt.rewritten = !Arc::ptr_eq(&plan, &attempt.plan);
        attempt.plan = plan;
        attempt.report.append(evidence.report);
        attempt.native_optimizer.merge(evidence.native_optimizer);
        attempt.source_diagnostics.extend(evidence.source_diagnostics);
        attempt.capability_contract.merge(evidence.capability_contract);
        Ok(attempt)
    }

    /// `plan`, which stays on DataFusion, with native fragments in place of
    /// the children it can keep.
    ///
    /// `filter_above` is the filter that reads `plan`'s rows through
    /// row-preserving exchanges only.
    fn below(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        filter_above: Option<&FilterExec>,
        evidence: &mut FragmentEvidence,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Metadata table functions run their results on DataFusion.
        if plan.downcast_ref::<DataFusionCpuBoundaryExec>().is_some() {
            return Ok(Arc::clone(plan));
        }
        let filter_above_children =
            plan.downcast_ref::<FilterExec>().or_else(|| filter_above.filter(|_| passes_rows_through(plan.as_ref())));
        let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
        let mut current = Arc::clone(plan);
        for (index, child) in children.iter().enumerate() {
            let mut next = self.replacement(child, filter_above_children, evidence)?;
            while let Some(replacement) = next.take() {
                if let Some(rebuilt) = self.keep(&current, index, child, &replacement.plan)? {
                    current = rebuilt;
                    evidence.append(replacement.evidence);
                    break;
                }
                evidence.append_abandoned(replacement.evidence, current.name());
                if replacement.whole {
                    next = self.rebuilt_over_fragments(child, filter_above_children, evidence)?;
                }
            }
        }
        Ok(current)
    }

    /// The child as one native fragment when it lowers as a whole, else the
    /// child rebuilt over fragments below it; `None` keeps the child.
    fn replacement(
        &self,
        child: &Arc<dyn ExecutionPlan>,
        filter_above: Option<&FilterExec>,
        evidence: &mut FragmentEvidence,
    ) -> DataFusionResult<Option<Replacement>> {
        // GPU functions, graphs and their inputs keep the placement the
        // relation input pass gave them.
        if plan_contains_gpu_execution(child.as_ref()) {
            return Ok(None);
        }
        let input =
            filter_above.and_then(|filter| without_pruning_copy(filter, child)).unwrap_or_else(|| Arc::clone(child));
        let lowering =
            lower_physical_plan_to_native(Arc::clone(&input), &self.native_execution_config.source_read_strategy())?;
        let attempt = replace_lowered_root_or_keep_plan_for_sink(
            input,
            lowering,
            self.native_execution_config,
            self.native_optimizer_options,
            self.cost_selection,
            NativeReplacementSink::HostArrow,
        )?;
        if attempt.rewritten {
            return Ok(Some(Replacement {
                plan: Arc::clone(&attempt.plan),
                evidence: FragmentEvidence::of_attempt(attempt),
                whole: true,
            }));
        }
        if attempt.report.not_selected_by_cost_count() > 0 {
            evidence.append(FragmentEvidence::of_attempt(attempt));
            return Ok(None);
        }
        self.rebuilt_over_fragments(child, filter_above, evidence)
    }

    /// `child` rebuilt over the fragments placed below it, or `None` when none
    /// is kept. Cost-declined candidates below it go to `evidence`.
    fn rebuilt_over_fragments(
        &self,
        child: &Arc<dyn ExecutionPlan>,
        filter_above: Option<&FilterExec>,
        evidence: &mut FragmentEvidence,
    ) -> DataFusionResult<Option<Replacement>> {
        let mut below = FragmentEvidence::default();
        let plan = self.below(child, filter_above, &mut below)?;
        if Arc::ptr_eq(&plan, child) {
            evidence.append(below);
            return Ok(None);
        }
        Ok(Some(Replacement { plan, evidence: below, whole: false }))
    }

    /// `parent` rebuilt with `replacement` as its child `index`, when it keeps
    /// the contracts it held over `child`.
    fn keep(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        index: usize,
        child: &Arc<dyn ExecutionPlan>,
        replacement: &Arc<dyn ExecutionPlan>,
    ) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
        if replacement.schema() != child.schema() || replacement.boundedness() != child.boundedness() {
            return Ok(None);
        }
        let mut children = parent.children().into_iter().cloned().collect::<Vec<_>>();
        children[index] = Arc::clone(replacement);
        let rebuilt = Arc::clone(parent)
            .replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))?;
        Ok(check_plan_sanity(&rebuilt, self.optimizer_options).is_ok().then_some(rebuilt))
    }
}

/// Whether `plan` passes each input row to its output unchanged.
fn passes_rows_through(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some_and(|coalesce| coalesce.fetch().is_none())
        || plan.downcast_ref::<CooperativeExec>().is_some()
}

/// `plan` with the Parquet scan it reads through row-preserving exchanges
/// rebuilt without its copy of `filter`'s predicate. DataFusion copies a
/// filter's predicate into the scan below it for pruning; while the filter
/// stays on DataFusion it evaluates the predicate on every scanned row, so a
/// fragment reads the scan without the copy, keeping the scan's projection,
/// table options and reader factory.
fn without_pruning_copy(filter: &FilterExec, plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if !passes_rows_through(plan.as_ref()) {
        return scan_without_filter_predicate(filter, plan);
    }
    let input = without_pruning_copy(filter, plan.children().first()?)?;
    Arc::clone(plan).replace_children(vec![input], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute)).ok()
}

fn scan_without_filter_predicate(
    filter: &FilterExec,
    child: &Arc<dyn ExecutionPlan>,
) -> Option<Arc<dyn ExecutionPlan>> {
    let (file_scan, parquet_source) =
        child.downcast_ref::<DataSourceExec>()?.downcast_to_file_source::<ParquetSource>()?;
    if !scan_predicate_exactly_matches_filter(parquet_source, Some(filter)) {
        return None;
    }
    let mut source = ParquetSource::new(parquet_source.table_schema().clone())
        .with_table_parquet_options(parquet_source.table_parquet_options().clone());
    if let Some(factory) = parquet_source.parquet_file_reader_factory() {
        source = source.with_parquet_file_reader_factory(Arc::clone(factory));
    }
    let source = match parquet_source.projection() {
        Some(projection) => source.try_pushdown_projection(projection).ok().flatten()?,
        None => Arc::new(source) as Arc<dyn FileSource>,
    };
    Some(DataSourceExec::from_data_source(FileScanConfigBuilder::from(file_scan.clone()).with_source(source).build()))
}
