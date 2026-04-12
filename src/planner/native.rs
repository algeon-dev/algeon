use std::sync::{Arc, Mutex};

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
use nexus_query_engine::{PlanNode, QueryId};

use super::CudfOptimizerConfig;
#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphAlgoExec;
use crate::{
    Error, ErrorCode,
    native_exec::{
        NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
        admission::QueryAdmissionTicket, plan_is_cpu_fallback_limited, plan_is_query_admission_wrapped,
        plan_is_unbounded_native_execution_limited, wrap_plan_with_cpu_fallback_limit,
        wrap_plan_with_query_admission_ticket, wrap_plan_with_unbounded_native_execution_limit,
    },
    native_report::{NativeJoinDiagnosticSummary, physical_join_diagnostics},
    physical_plan::physical_plan_shape,
    report::PlanningReport,
    session::{CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME, CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME},
};

mod adaptive;
mod attempt;
mod strict_bounded;

pub use adaptive::{NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics};
use attempt::{
    NativeOptimizationAttempt, apply_native_fallback_policy, optimize_count_only_native_plan,
    optimize_final_native_plan, peel_cooperative_native_exec,
};
pub use strict_bounded::{
    NativeStrictBoundedAcceptedCandidateDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, NativeStrictBoundedRejectedCandidateDiagnostics,
    STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};

/// Last-observed native planning diagnostics from a retained optimizer rule.
///
/// The strict bounded and adaptive fields are intentionally separate. Report
/// code should not project adaptive evidence into strict contract rows, or
/// strict preflight failures into adaptive runtime history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningDiagnostics {
    /// Root DataFusion execution node name seen before native optimization.
    pub input_plan_root: String,
    /// Stable textual shape of the input physical plan used for report context.
    pub input_plan_shape: String,
    /// Join diagnostics captured before native replacement can remove
    /// DataFusion join nodes from the post-native plan.
    pub input_join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    /// Candidate rewrite outcomes from the native physical optimizer rule.
    pub report: PlanningReport,
    /// DataFusion-free native optimizer diagnostics from the lowered plan.
    pub native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics,
    /// Source-read diagnostics captured while lowering/planning native sources.
    pub source_diagnostics: Vec<nexus_query_engine::SourceDiagnostics>,
    /// Strict bounded contract evidence; never populated from adaptive rows.
    pub strict_bounded_contract: NativeStrictBoundedContractDiagnostics,
    /// Adaptive admission evidence; never projected into strict contract rows.
    pub adaptive_pipeline: NativeAdaptivePipelineDiagnostics,
}

/// DataFusion physical optimizer rule that rewrites supported plans to native cuDF execution.
///
/// The rule is reusable across sessions when callers want to inspect
/// [`last_planning_diagnostics`](Self::last_planning_diagnostics). Session
/// insertion deduplicates by the public rule names in [`crate::session`].
/// `GpuPreferred` leaves rejected candidates on the original DataFusion plan;
/// `GpuOnly` reports the first native rejection as an optimizer error.
#[derive(Debug)]
pub struct CudfNativePhysicalOptimizerRule {
    config: CudfOptimizerConfig,
    stage: NativeOptimizerStage,
    state: Arc<CudfNativePhysicalOptimizerRuleState>,
}

#[derive(Debug)]
struct CudfNativePhysicalOptimizerRuleState {
    last_input_plan_root: Mutex<Option<String>>,
    last_input_plan_shape: Mutex<Option<String>>,
    last_input_join_diagnostics: Mutex<Vec<NativeJoinDiagnosticSummary>>,
    last_report: Mutex<Option<PlanningReport>>,
    last_native_optimizer: Mutex<nexus_query_engine::NativeOptimizerDiagnostics>,
    last_source_diagnostics: Mutex<Vec<nexus_query_engine::SourceDiagnostics>>,
    last_strict_bounded_contract: Mutex<NativeStrictBoundedContractDiagnostics>,
    last_adaptive_pipeline: Mutex<NativeAdaptivePipelineDiagnostics>,
}

// Distinct rule identities let the count-only probe run before the final rewrite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeOptimizerStage {
    // Dedicated pass for plans where output-requirement peeling exposes a native count.
    CountOnlyProbe,
    // Full native lowering, fallback handling, and fragment composition.
    Final,
}

impl CudfNativePhysicalOptimizerRule {
    /// Create the final native rewrite rule.
    pub fn new(config: CudfOptimizerConfig) -> Self {
        Self {
            config,
            stage: NativeOptimizerStage::Final,
            state: Arc::new(CudfNativePhysicalOptimizerRuleState {
                last_input_plan_root: Mutex::new(None),
                last_input_plan_shape: Mutex::new(None),
                last_input_join_diagnostics: Mutex::new(Vec::new()),
                last_report: Mutex::new(None),
                last_native_optimizer: Mutex::new(nexus_query_engine::NativeOptimizerDiagnostics::default()),
                last_source_diagnostics: Mutex::new(Vec::new()),
                last_strict_bounded_contract: Mutex::new(NativeStrictBoundedContractDiagnostics::default()),
                last_adaptive_pipeline: Mutex::new(NativeAdaptivePipelineDiagnostics::default()),
            }),
        }
    }

    /// Return the immutable config copied into rewritten native exec nodes.
    pub fn config(&self) -> &CudfOptimizerConfig {
        &self.config
    }

    pub(crate) fn count_only_probe_rule(&self) -> Self {
        Self {
            config: self.config.clone(),
            stage: NativeOptimizerStage::CountOnlyProbe,
            state: Arc::clone(&self.state),
        }
    }

    /// Most recent planning report produced by this retained rule, if it has run.
    pub fn last_report(&self) -> Option<PlanningReport> {
        match self.state.last_report.lock() {
            Ok(report) => report.clone(),
            Err(poisoned) => {
                #[cfg(feature = "server")]
                tracing::warn!("native planning report mutex was poisoned; returning the last captured report");
                poisoned.into_inner().clone()
            },
        }
    }

    /// Source diagnostics captured during the most recent native planning attempt.
    pub fn last_source_diagnostics(&self) -> Vec<nexus_query_engine::SourceDiagnostics> {
        self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned").clone()
    }

    /// Structured diagnostics from the most recent run of this retained rule.
    ///
    /// The result is overwritten on each optimizer invocation and is intended
    /// for report harnesses and tests, not as a historical log.
    pub fn last_planning_diagnostics(&self) -> Option<NativePlanningDiagnostics> {
        self.last_report().map(|report| NativePlanningDiagnostics {
            input_plan_root: self
                .state
                .last_input_plan_root
                .lock()
                .expect("native input plan root mutex poisoned")
                .clone()
                .unwrap_or_default(),
            input_plan_shape: self
                .state
                .last_input_plan_shape
                .lock()
                .expect("native input plan shape mutex poisoned")
                .clone()
                .unwrap_or_default(),
            input_join_diagnostics: self
                .state
                .last_input_join_diagnostics
                .lock()
                .expect("native input join diagnostics mutex poisoned")
                .clone(),
            report,
            native_optimizer: self
                .state
                .last_native_optimizer
                .lock()
                .expect("native optimizer diagnostics mutex poisoned")
                .clone(),
            source_diagnostics: self.last_source_diagnostics(),
            strict_bounded_contract: self
                .state
                .last_strict_bounded_contract
                .lock()
                .expect("native strict bounded contract diagnostics mutex poisoned")
                .clone(),
            adaptive_pipeline: self
                .state
                .last_adaptive_pipeline
                .lock()
                .expect("native adaptive pipeline diagnostics mutex poisoned")
                .clone(),
        })
    }
}

impl PhysicalOptimizerRule for CudfNativePhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if plan_is_query_admission_wrapped(plan.as_ref()) {
            return Ok(plan);
        }
        if plan_is_unbounded_native_execution_limited(plan.as_ref()) {
            return Ok(plan);
        }
        if plan_is_cpu_fallback_limited(plan.as_ref()) {
            return Ok(plan);
        }
        if plan.downcast_ref::<NativeCudfExec>().is_some() || plan.downcast_ref::<NativeCudfComposedExec>().is_some() {
            return Ok(plan);
        }
        if let Some(native_plan) = peel_cooperative_native_exec(&plan) {
            return Ok(native_plan);
        }

        *self.state.last_input_plan_root.lock().expect("native input plan root mutex poisoned") =
            Some(plan.name().to_string());
        *self.state.last_input_plan_shape.lock().expect("native input plan shape mutex poisoned") =
            Some(physical_plan_shape(plan.as_ref()));
        *self.state.last_input_join_diagnostics.lock().expect("native input join diagnostics mutex poisoned") =
            physical_join_diagnostics(plan.as_ref());

        let effective_config =
            self.config.with_datafusion_config_options(config)?.with_query_id_if_absent(QueryId::mint());
        let native_execution_config =
            effective_config.native_execution_config().clone().with_plan_query_admission_ticket();
        let effective_config = effective_config.with_native_execution_config(native_execution_config);
        let native_execution_config = effective_config.native_execution_config();
        let native_optimizer_options = effective_config.native_optimizer_options();
        let native_plan_preference_options = effective_config.native_plan_preference_options();

        let optimized = match self.stage {
            NativeOptimizerStage::CountOnlyProbe => optimize_count_only_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
            )?,
            NativeOptimizerStage::Final => optimize_final_native_plan(
                Arc::clone(&plan),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
            )?,
        };
        let mut optimized = apply_native_fallback_policy(Arc::clone(&plan), optimized, &effective_config)?;
        let query_admission_ticket =
            plan_query_admission_ticket_for_optimized_plan(optimized.plan.as_ref(), native_execution_config)?;
        apply_execution_admission_wrappers(
            self.stage,
            &mut optimized,
            &effective_config,
            native_execution_config,
            query_admission_ticket,
        );
        *self.state.last_report.lock().map_err(|_| {
            datafusion::common::DataFusionError::from(Error::new(
                ErrorCode::Internal,
                "datafusion_nexus.native_optimizer.report",
                "native planning report mutex was poisoned",
            ))
        })? = Some(optimized.report.clone());
        *self.state.last_native_optimizer.lock().expect("native optimizer diagnostics mutex poisoned") =
            optimized.native_optimizer.clone();
        *self.state.last_source_diagnostics.lock().expect("native source diagnostics mutex poisoned") =
            optimized.source_diagnostics.clone();
        *self
            .state
            .last_strict_bounded_contract
            .lock()
            .expect("native strict bounded contract diagnostics mutex poisoned") =
            optimized.strict_bounded_contract.clone();
        *self.state.last_adaptive_pipeline.lock().expect("native adaptive pipeline diagnostics mutex poisoned") =
            optimized.adaptive_pipeline.clone();
        Ok(optimized.plan)
    }

    fn name(&self) -> &str {
        match self.stage {
            NativeOptimizerStage::CountOnlyProbe => CUDF_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME,
            NativeOptimizerStage::Final => CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn apply_execution_admission_wrappers(
    stage: NativeOptimizerStage,
    optimized: &mut NativeOptimizationAttempt,
    effective_config: &CudfOptimizerConfig,
    native_execution_config: &NativeCudfExecutionConfig,
    query_admission_ticket: Option<Arc<QueryAdmissionTicket>>,
) {
    if optimized.rewritten
        && let Some(ticket) = query_admission_ticket
    {
        optimized.plan = wrap_plan_with_query_admission_ticket(Arc::clone(&optimized.plan), ticket);
    } else if optimized.rewritten
        && matches!(native_execution_config.memory_control_policy(), nexus_query_engine::MemoryControlPolicy::Unbounded)
        && let Some(limiter) = native_execution_config.unbounded_native_execution_limiter()
    {
        optimized.plan = wrap_plan_with_unbounded_native_execution_limit(Arc::clone(&optimized.plan), limiter);
    } else if !optimized.rewritten
        && stage == NativeOptimizerStage::Final
        && let Some(limiter) = effective_config.cpu_fallback_statement_limiter()
    {
        optimized.plan = wrap_plan_with_cpu_fallback_limit(Arc::clone(&optimized.plan), limiter);
    }
}

fn plan_query_admission_ticket_for_optimized_plan(
    plan: &dyn ExecutionPlan,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<Option<Arc<QueryAdmissionTicket>>> {
    let requested_grant_bytes = graph_aware_requested_grant_bytes_for_plan(plan, native_execution_config)?;
    Ok(native_execution_config.new_plan_query_admission_ticket(requested_grant_bytes))
}

fn graph_aware_requested_grant_bytes_for_plan(
    plan: &dyn ExecutionPlan,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<Option<u64>> {
    if native_execution_config.requested_grant_bytes().is_some() {
        return Ok(None);
    }
    let Some(source) = native_execution_config.query_admission_source() else {
        return Ok(None);
    };
    let graph_model_ask = complete_graph_dominated_model_ask_bytes(
        plan,
        native_execution_config.graph_memory_calibration_override(),
        Some(source.max_device_total_memory_bytes()),
    )?;
    native_execution_config.graph_aware_requested_grant_bytes_for_model_ask(graph_model_ask).map_err(|error| {
        datafusion::common::DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_optimizer.graph_ticket_ask",
            "failed to compute graph-aware query admission ticket ask",
            error,
        ))
    })
}

#[derive(Debug, Default)]
struct GraphModelAskScan {
    max_graph_model_ask_bytes: Option<u64>,
    graph_node_count: usize,
    estimated_graph_node_count: usize,
    disqualified: bool,
    non_graph_native_fragment_count: usize,
}

fn complete_graph_dominated_model_ask_bytes(
    plan: &dyn ExecutionPlan,
    calibration_override: nexus_query_engine::GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> DataFusionResult<Option<u64>> {
    let mut scan = GraphModelAskScan::default();
    collect_graph_model_ask_scan(plan, calibration_override, device_total_memory_bytes, &mut scan)?;
    if scan.disqualified
        || scan.non_graph_native_fragment_count > 0
        || scan.graph_node_count != 1
        || scan.estimated_graph_node_count != scan.graph_node_count
    {
        return Ok(None);
    }
    Ok(scan.max_graph_model_ask_bytes)
}

fn collect_graph_model_ask_scan(
    plan: &dyn ExecutionPlan,
    calibration_override: nexus_query_engine::GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
    scan: &mut GraphModelAskScan,
) -> DataFusionResult<bool> {
    #[cfg(feature = "cugraph")]
    if let Some(cugraph) = plan.downcast_ref::<CugraphAlgoExec>() {
        update_cugraph_model_ask_scan(cugraph, scan)?;
        return Ok(true);
    }

    let mut child_has_graph = false;
    for child in plan.children() {
        child_has_graph |=
            collect_graph_model_ask_scan(child.as_ref(), calibration_override, device_total_memory_bytes, scan)?;
    }

    let mut local_has_graph = false;
    if let Some(native) = plan.downcast_ref::<NativeCudfExec>() {
        local_has_graph =
            update_graph_model_ask_scan(native.native_plan(), calibration_override, device_total_memory_bytes, scan)?;
    } else if let Some(native) = plan.downcast_ref::<NativeCudfComposedExec>() {
        if child_has_graph && native_plan_is_graph_ticket_light_fragment(native.native_plan()) {
            return Ok(true);
        }
        local_has_graph =
            update_graph_model_ask_scan(native.native_plan(), calibration_override, device_total_memory_bytes, scan)?;
    } else if let Some(native) = plan.downcast_ref::<NativeCudfGpuProducerExec>() {
        local_has_graph =
            update_graph_model_ask_scan(native.native_plan(), calibration_override, device_total_memory_bytes, scan)?;
    }
    Ok(local_has_graph || child_has_graph)
}

fn update_graph_model_ask_scan(
    native_plan: &nexus_query_engine::QueryPlan,
    calibration_override: nexus_query_engine::GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
    scan: &mut GraphModelAskScan,
) -> DataFusionResult<bool> {
    let graph_node_count = native_plan_graph_node_count(native_plan);
    if graph_node_count == 0 {
        scan.non_graph_native_fragment_count += 1;
        return Ok(false);
    }
    scan.graph_node_count += graph_node_count;
    if !native_plan_is_graph_dominated(native_plan) {
        scan.disqualified = true;
        return Ok(true);
    }
    for estimate in nexus_query_engine::graph_memory_estimates_for_plan(
        native_plan,
        calibration_override,
        device_total_memory_bytes,
    )
    .map_err(|error| {
        datafusion::common::DataFusionError::from(Error::from_native(
            "datafusion_nexus.native_optimizer.graph_ticket_ask",
            "failed to estimate graph memory from native plan metadata",
            error,
        ))
    })? {
        scan.estimated_graph_node_count += 1;
        scan.max_graph_model_ask_bytes =
            Some(scan.max_graph_model_ask_bytes.map_or(estimate.estimate.requested_bytes, |current: u64| {
                current.max(estimate.estimate.requested_bytes)
            }));
    }
    Ok(true)
}

#[cfg(feature = "cugraph")]
fn update_cugraph_model_ask_scan(cugraph: &CugraphAlgoExec, scan: &mut GraphModelAskScan) -> DataFusionResult<()> {
    scan.graph_node_count += 1;
    if let Some(model_ask) = cugraph.graph_model_ask_bytes_for_admission_ticket()? {
        scan.estimated_graph_node_count += 1;
        scan.max_graph_model_ask_bytes =
            Some(scan.max_graph_model_ask_bytes.map_or(model_ask, |current: u64| current.max(model_ask)));
    }
    Ok(())
}

fn native_plan_graph_node_count(native_plan: &nexus_query_engine::QueryPlan) -> usize {
    native_plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_))).count()
}

fn native_plan_is_graph_dominated(native_plan: &nexus_query_engine::QueryPlan) -> bool {
    let mut has_graph = false;
    for node in native_plan.nodes() {
        match &node.kind {
            PlanNode::GraphAlgorithm(_) => has_graph = true,
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Limit(_)
            | PlanNode::EdgeNormalize(_) => {},
            PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Join(_)
            | PlanNode::Union(_) => {
                return false;
            },
        }
    }
    has_graph
}

fn native_plan_is_graph_ticket_light_fragment(native_plan: &nexus_query_engine::QueryPlan) -> bool {
    native_plan.nodes().iter().all(|node| {
        matches!(node.kind, PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Limit(_))
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use arrow::datatypes::{DataType, Field, Schema};
    #[cfg(feature = "cugraph")]
    use arrow::{
        array::{ArrayRef, Float64Array, Int64Array},
        record_batch::RecordBatch,
    };
    #[cfg(feature = "cugraph")]
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::{empty::EmptyExec, union::UnionExec};
    #[cfg(feature = "cugraph")]
    use nexus_query_engine::GpuResidentSource;
    use nexus_query_engine::{
        DeploymentIoCapability, GdsDriverProbeStatus, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
        GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy, GraphEdgeColumnBindings,
        GraphTriangleCountAllParams, KvikioCompatModeStatus, LocalParquetFile, LocalParquetSource, NvidiaFsProbeStatus,
        QueryPlan, QueryPlanBuilder, QueryService, QueryServiceConfig, QueryServiceTenantConfig, SourceNode, TenantId,
    };

    use super::*;
    #[cfg(feature = "cugraph")]
    use crate::graph_sql::{
        CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphGraphConstructionPolicy,
        CugraphGraphOptions, CugraphTableReference, CugraphTriangleCountAllParams, result_schema_for_params,
    };
    use crate::native_exec::QueryAdmissionSource;
    #[cfg(feature = "cugraph")]
    use crate::native_exec::gpu_resident_producer_input_name;

    fn attempt(rewritten: bool) -> NativeOptimizationAttempt {
        NativeOptimizationAttempt {
            plan: Arc::new(EmptyExec::new(Arc::new(Schema::empty()))),
            report: PlanningReport::new(),
            native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics::default(),
            source_diagnostics: Vec::new(),
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten,
        }
    }

    fn nongds_deployment() -> DeploymentIoCapability {
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        )
    }

    fn backend_admission_config(min_bytes: u64, want_bytes: u64, capacity_bytes: u64) -> NativeCudfExecutionConfig {
        let tenant_id = TenantId::new("tenant");
        let service = QueryService::new_admission_only(
            QueryServiceConfig { device_capacity_bytes: Some(capacity_bytes), ..QueryServiceConfig::default() },
            vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), min_bytes, want_bytes)],
            nongds_deployment(),
        )
        .unwrap();
        NativeCudfExecutionConfig::bounded(min_bytes, 512, 1, 3, 1)
            .with_query_admission_source(QueryAdmissionSource::new(service, tenant_id))
    }

    fn edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    fn triangle_spec() -> GraphAlgorithmSpec {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::TriangleCountAll,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::PythonCugraph,
            },
            GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
        )
    }

    fn edge_source(row_group_rows: Option<Vec<u64>>) -> LocalParquetSource {
        let mut file = LocalParquetFile::new("edges.parquet");
        if let Some(row_group_rows) = row_group_rows {
            file = file.with_row_group_rows(row_group_rows);
        }
        LocalParquetSource::from_files(
            vec![file],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        )
    }

    fn graph_native_plan(row_group_rows: Option<Vec<u64>>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_source(row_group_rows)));
        let spec = triangle_spec();
        let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
        builder.finish_with_host_sink(graph)
    }

    fn bfs_native_plan(row_group_rows: Option<Vec<u64>>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_source(row_group_rows)));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
        builder.finish_with_host_sink(graph)
    }

    fn graph_exec(row_group_rows: Option<Vec<u64>>, config: &NativeCudfExecutionConfig) -> NativeCudfExec {
        NativeCudfExec::try_new(graph_native_plan(row_group_rows), config.clone()).unwrap()
    }

    fn graph_union_native_plan(row_group_rows: Option<Vec<u64>>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_source(row_group_rows)));
        let spec = triangle_spec();
        let output_schema = Arc::clone(&spec.output_schema);
        let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
        let union = builder.add_union(vec![graph, graph], Arc::clone(&output_schema));
        builder.finish_with_host_sink_schema(union, output_schema)
    }

    #[cfg(feature = "cugraph")]
    fn memory_edge_source(rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = edge_schema();
        let src = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64))) as ArrayRef;
        let dst = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64 + 1))) as ArrayRef;
        let weight = Arc::new(Float64Array::from_iter_values((0..rows).map(|_| 1.0))) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![src, dst, weight]).unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap() as Arc<dyn ExecutionPlan>
    }

    #[cfg(feature = "cugraph")]
    fn cugraph_triangle_spec() -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::TriangleCountAll(CugraphTriangleCountAllParams);
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::TriangleCountAll,
            table_name: CugraphTableReference::parse("edges").unwrap(),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: Some("weight".to_string()),
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: false,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::PythonCugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::TriangleCountAll, &params),
            params,
        }
    }

    #[cfg(feature = "cugraph")]
    fn composed_cugraph_consumer_plan(input_name: String, schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
        builder.finish_with_host_sink(source)
    }

    #[test]
    fn final_rewritten_plan_does_not_consume_cpu_fallback_limiter_in_library_mode() {
        let config =
            CudfOptimizerConfig::default().with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
        let native_execution_config = config.native_execution_config().clone();
        let mut optimized = attempt(true);

        apply_execution_admission_wrappers(
            NativeOptimizerStage::Final,
            &mut optimized,
            &config,
            &native_execution_config,
            native_execution_config.query_admission_ticket(),
        );

        assert!(
            !plan_is_cpu_fallback_limited(optimized.plan.as_ref()),
            "rewritten GPU plans must not be wrapped by the CPU fallback limiter"
        );
    }

    #[test]
    fn final_non_rewritten_plan_is_cpu_fallback_limited_when_configured() {
        let config =
            CudfOptimizerConfig::default().with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
        let native_execution_config = config.native_execution_config().clone();
        let mut optimized = attempt(false);

        apply_execution_admission_wrappers(
            NativeOptimizerStage::Final,
            &mut optimized,
            &config,
            &native_execution_config,
            native_execution_config.query_admission_ticket(),
        );

        assert!(
            plan_is_cpu_fallback_limited(optimized.plan.as_ref()),
            "non-rewritten final plans should be wrapped by the configured CPU fallback limiter"
        );
    }

    #[test]
    fn graph_aware_ticket_ask_uses_lowered_graph_model_and_clamps_to_tenant_band() {
        let min = 64 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        let capacity = 4 * 1024 * 1024 * 1024;
        let config = backend_admission_config(min, want, capacity);
        let exec = graph_exec(Some(vec![100, 200]), &config);
        let model_ask =
            complete_graph_dominated_model_ask_bytes(&exec, config.graph_memory_calibration_override(), Some(capacity))
                .unwrap();
        let expected = config.graph_aware_requested_grant_bytes_for_model_ask(model_ask).unwrap();

        let requested = graph_aware_requested_grant_bytes_for_plan(&exec, &config).unwrap();

        assert_eq!(requested, expected);
        assert!(requested.unwrap() < want, "small graph ask should not default to tenant want");
    }

    #[test]
    fn graph_aware_ticket_ask_does_not_override_explicit_requested_grant_bytes() {
        let config = backend_admission_config(64 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .with_requested_grant_bytes(Some(768 * 1024 * 1024));
        let exec = graph_exec(Some(vec![100, 200]), &config);

        assert_eq!(graph_aware_requested_grant_bytes_for_plan(&exec, &config).unwrap(), None);
        let ticket = plan_query_admission_ticket_for_optimized_plan(&exec, &config).unwrap().unwrap();
        assert_eq!(ticket.requested_grant_bytes(), Some(768 * 1024 * 1024));
    }

    #[test]
    fn graph_aware_ticket_ask_keeps_default_ticket_when_graph_rows_are_unknown() {
        let config = backend_admission_config(64 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let exec = graph_exec(None, &config);

        assert_eq!(graph_aware_requested_grant_bytes_for_plan(&exec, &config).unwrap(), None);
        let ticket = plan_query_admission_ticket_for_optimized_plan(&exec, &config).unwrap().unwrap();
        assert_eq!(ticket.requested_grant_bytes(), None);
    }

    #[test]
    fn graph_aware_ticket_ask_keeps_default_ticket_for_non_reserving_bfs() {
        let config = backend_admission_config(64 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let exec = NativeCudfExec::try_new(bfs_native_plan(Some(vec![100, 200])), config.clone()).unwrap();

        assert_eq!(graph_aware_requested_grant_bytes_for_plan(&exec, &config).unwrap(), None);
        let ticket = plan_query_admission_ticket_for_optimized_plan(&exec, &config).unwrap().unwrap();
        assert_eq!(ticket.requested_grant_bytes(), None);
    }

    #[test]
    fn graph_aware_ticket_ask_keeps_default_ticket_when_any_graph_rows_are_unknown() {
        let config = backend_admission_config(64 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let known = Arc::new(graph_exec(Some(vec![100, 200]), &config)) as Arc<dyn ExecutionPlan>;
        let unknown = Arc::new(graph_exec(None, &config)) as Arc<dyn ExecutionPlan>;
        let plan = UnionExec::try_new(vec![known, unknown]).unwrap();

        assert_eq!(graph_aware_requested_grant_bytes_for_plan(plan.as_ref(), &config).unwrap(), None);
        let ticket = plan_query_admission_ticket_for_optimized_plan(plan.as_ref(), &config).unwrap().unwrap();
        assert_eq!(ticket.requested_grant_bytes(), None);
    }

    #[test]
    fn graph_aware_ticket_ask_keeps_default_ticket_for_non_graph_dominated_native_plan() {
        let config = backend_admission_config(64 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let exec = NativeCudfExec::try_new(graph_union_native_plan(Some(vec![100, 200])), config.clone()).unwrap();

        assert_eq!(graph_aware_requested_grant_bytes_for_plan(&exec, &config).unwrap(), None);
        let ticket = plan_query_admission_ticket_for_optimized_plan(&exec, &config).unwrap().unwrap();
        assert_eq!(ticket.requested_grant_bytes(), None);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn graph_aware_ticket_ask_includes_composed_cugraph_producer() {
        let min = 64 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        let capacity = 4 * 1024 * 1024 * 1024;
        let config = backend_admission_config(min, want, capacity);
        let cugraph = Arc::new(
            CugraphAlgoExec::try_new(
                cugraph_triangle_spec(),
                memory_edge_source(300),
                vec![],
                None,
                None,
                config.clone(),
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let input_name = gpu_resident_producer_input_name(cugraph.as_ref()).unwrap();
        let consumer = composed_cugraph_consumer_plan(input_name, cugraph.schema());
        let composed = NativeCudfComposedExec::try_new(consumer, vec![cugraph], config.clone()).unwrap();

        let requested = graph_aware_requested_grant_bytes_for_plan(&composed, &config).unwrap();

        assert!(requested.is_some(), "composed cuGraph producer should contribute a graph-aware ask");
        assert!(requested.unwrap() < want, "composed cuGraph ask should not default to tenant want");
    }
}
