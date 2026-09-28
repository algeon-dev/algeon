use query_engine::{
    admission::{AdmissionDecision, AdmissionRejection, AdmittedPlan, admit},
    contract_report::{CapabilityContractReport, CapabilityDecisionRow},
    error::QueryError,
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

use crate::{
    error::{Error, Result},
    native_exec::AlgeonNativeExecutionConfig,
    report::GpuCandidate,
};

/// Adapter-side capability-contract diagnostics collected during planning.
///
/// This is a projection of native-engine contract reports for DataFusion report
/// rows. It does not perform admission or pipeline scheduling itself.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeCapabilityContractDiagnostics {
    /// Candidates accepted by the optimizer with capability reports attached.
    pub accepted_candidates: Vec<NativeCapabilityAcceptedCandidateDiagnostics>,
    /// Candidates rejected before execution with their first blocking decision.
    pub rejected_candidates: Vec<NativeCapabilityRejectedCandidateDiagnostics>,
}

impl NativeCapabilityContractDiagnostics {
    /// Appends accepted/rejected candidate evidence from another optimizer stage.
    pub fn merge(&mut self, other: Self) {
        self.accepted_candidates.extend(other.accepted_candidates);
        self.rejected_candidates.extend(other.rejected_candidates);
    }

    /// Returns true when no capability-contract evidence was captured.
    pub fn is_empty(&self) -> bool {
        self.accepted_candidates.is_empty() && self.rejected_candidates.is_empty()
    }
}

/// Capability report and compiled graph facts for an accepted candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityAcceptedCandidateDiagnostics {
    /// Optimizer candidate that produced these diagnostics.
    pub candidate: GpuCandidate,
    /// Capability report.
    pub report: CapabilityContractReport,
    /// Graph-size diagnostics for the compiled pipeline.
    pub compiled_graph: NativeCapabilityCompiledGraphDiagnostics,
}

/// First capability decision that blocked a rejected candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityRejectedCandidateDiagnostics {
    /// Optimizer candidate that was rejected.
    pub candidate: GpuCandidate,
    /// First rejected row from capability analysis.
    pub blocking_contract: CapabilityDecisionRow,
}

/// Report-facing graph summary for a compiled capability envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCapabilityCompiledGraphDiagnostics {
    /// Number of compiled tasks.
    pub task_count: usize,
    /// Number of compiled dependency edges.
    pub edge_count: usize,
    /// Number of compiled barriers.
    pub barrier_count: usize,
    /// Number of resident resources tracked by the graph.
    pub resource_count: usize,
    /// Final sink boundary kind.
    pub sink_kind: String,
}

impl NativeCapabilityCompiledGraphDiagnostics {
    fn from_pipeline(pipeline: &CompiledPipeline) -> Self {
        Self {
            task_count: pipeline.tasks().len(),
            edge_count: pipeline.edges().len(),
            barrier_count: pipeline.barriers().len(),
            resource_count: pipeline.resources().len(),
            sink_kind: pipeline.sink().kind().as_str().to_string(),
        }
    }
}

/// Decides once whether a native candidate runs as a compiled pipeline.
///
/// `plan` is the candidate in the sink form execution runs, so the decision
/// and the executed plan see the same sink. The boundary check and the
/// accepted/rejected diagnostics read this decision instead of re-deriving it.
/// A candidate the validator finds unsupported is rejected like any other
/// missed contract; only a malformed candidate is an error.
pub(super) fn native_admission_decision(
    plan: &QueryPlan,
    execution_config: &AlgeonNativeExecutionConfig,
) -> Result<AdmissionDecision> {
    admit(
        plan,
        execution_config.statistical_aggregate_execution_mode(),
        execution_config.planning_policy(),
        rapids_analysis_context(execution_config.planning_policy()),
    )
    .map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native.admission",
            "native admission could not decide the candidate plan",
            stamp_native_error_correlation(error, execution_config),
        )
    })
}

pub(super) fn accepted_plan_contract_diagnostics(
    candidate: GpuCandidate,
    execution_config: &AlgeonNativeExecutionConfig,
    admission: &AdmittedPlan,
) -> Result<NativeCapabilityContractDiagnostics> {
    let report = stamp_capability_report_correlation(
        CapabilityContractReport::from_admission(admission).map_err(|error| {
            capability_contract_report_error(stamp_native_error_correlation(error, execution_config))
        })?,
        execution_config,
    );
    Ok(NativeCapabilityContractDiagnostics {
        accepted_candidates: vec![NativeCapabilityAcceptedCandidateDiagnostics {
            candidate,
            report,
            compiled_graph: NativeCapabilityCompiledGraphDiagnostics::from_pipeline(admission.pipeline()),
        }],
        rejected_candidates: Vec::new(),
    })
}

pub(super) fn rejected_plan_contract_diagnostics(
    candidate: GpuCandidate,
    execution_config: &AlgeonNativeExecutionConfig,
    rejection: &AdmissionRejection,
) -> Result<NativeCapabilityContractDiagnostics> {
    let blocking_contract = CapabilityDecisionRow::from_rejection(0, rejection.rejection())
        .map_err(|error| capability_contract_report_error(stamp_native_error_correlation(error, execution_config)))?;
    Ok(NativeCapabilityContractDiagnostics {
        accepted_candidates: Vec::new(),
        rejected_candidates: vec![NativeCapabilityRejectedCandidateDiagnostics { candidate, blocking_contract }],
    })
}

fn capability_contract_report_error(error: QueryError) -> Error {
    Error::from_native(
        "algeon_datafusion.native.capability_contract_diagnostics",
        "capability-contract diagnostics could not be built",
        error,
    )
}

fn stamp_capability_report_correlation(
    mut report: CapabilityContractReport,
    execution_config: &AlgeonNativeExecutionConfig,
) -> CapabilityContractReport {
    if let Some(query_id) = execution_config.query_id() {
        report = report.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        report = report.with_trace_id(trace_id.clone());
    }
    report
}

fn stamp_native_error_correlation(mut error: QueryError, execution_config: &AlgeonNativeExecutionConfig) -> QueryError {
    if let Some(query_id) = execution_config.query_id() {
        error = error.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        error = error.with_trace_id(trace_id.clone());
    }
    error
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Int32Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use query_engine::{
        capability::KeyedHashJoinStrategyOptions,
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSpec, JoinType, QueryPlanBuilder},
        source::{LiteralBatchSource, SourceNode},
    };

    use super::*;
    use crate::{error::ErrorCode, native_exec::SourceChunking};

    #[test]
    fn bounded_decision_admits_under_the_configured_planning_policy() {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("literal batch");
        let mut builder = QueryPlanBuilder::new();
        let left = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
        let right =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        let key = |input| ColumnRef::input(input, "k", "k");
        let join = builder
            .add_join(
                left,
                right,
                JoinSpec::new(JoinType::Inner, vec![JoinKey::new(key(0), key(1))])
                    .with_output_projection(vec![JoinOutputColumn::left(key(0), "k")]),
                Arc::clone(&schema),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_keyed_hash_join_strategy_options(
                    KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11),
                );

        let decision = native_admission_decision(&plan, &execution_config).expect("the join plan is decidable");
        let admission = decision.admission().expect("a literal inner join is admitted");

        assert_eq!(admission.policy(), execution_config.planning_policy());
    }

    #[test]
    fn malformed_candidate_fails_instead_of_falling_back() {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("literal batch");
        let mut builder = QueryPlanBuilder::new();
        let left = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
        let right =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        let key = |input| ColumnRef::input(input, "k", "k");
        // The left key names the right input: the IR is malformed, so no
        // DataFusion fallback can be correct for it.
        let join = builder
            .add_join(
                left,
                right,
                JoinSpec::new(JoinType::Inner, vec![JoinKey::new(key(1), key(1))])
                    .with_output_projection(vec![JoinOutputColumn::left(key(0), "k")]),
                Arc::clone(&schema),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"));

        let error = native_admission_decision(&plan, &execution_config).expect_err("a malformed plan is not decidable");

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
    }
}
