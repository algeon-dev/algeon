use std::{
    mem,
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, Instant},
};

use futures::stream::once;
use query_engine::{
    admission::AttemptCapabilityEvidence,
    plan::{QueryPlan, row_bounds::query_plan_hard_row_upper_bound},
    runtime::cache_observation_fingerprint,
};

use super::{
    AlgeonNativeExecutionConfig, AlgeonNativeMetrics, Boundedness, BoxFuture, CUGRAPH_EDGE_DESTINATION_COLUMN,
    CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN, CUGRAPH_EDGE_WEIGHT_COLUMN, Column, CugraphAlgorithmParams,
    CugraphAlgorithmSpec, CugraphBfsOutputMode, CugraphEdgeCanonicalFacts, CugraphInputRowBounds,
    CugraphPlanDefinition, CugraphResultPlanOptions, DataFusionError, DataFusionResult, DisplayAs, DisplayFormatType,
    EagerCugraphHostOutput, EmissionType, EquivalenceProperties, Error, ErrorCode, ExecutionContext, ExecutionPlan,
    ExecutionPlanMetricsSet, GpuIslandHostBoundaryConfig, GpuResidentProducer, GpuResultHandle,
    GraphAlgorithmRuntimeMetrics, GraphBfsFollowerOutcome, GraphBfsFollowerRoute, GraphBfsLeader,
    GraphBfsPublishedResult, GraphBfsSingleflightJoin, GraphBfsSingleflightKey, GraphExecutionMetrics,
    GraphExecutionPlan, GraphResidencyDescriptor, MetricsSet, NEXT_CUGRAPH_GPU_RESULT_INPUT_ID,
    NativeExecutionMetricsCapture, NativeExpr, Partitioning, PhysicalExpr, PhysicalSortExpr, PlanNode, PlanProperties,
    RecordBatch, RecordBatchStreamAdapter, Result, Schema, SchemaRef, SendableRecordBatchStream, SortOptions,
    SourceNode, StagedCugraphInputs, TaskContext, TreeNodeRecursion, TryStreamExt, UploadedCugraphInputs,
    attach_graph_residency, check_host_staging_cancellation, collect_cugraph_inputs, counted_cudf_runtime,
    cugraph_edge_canonical_facts, cugraph_edge_normalize_inserted, cugraph_execution_capability_evidence,
    cugraph_execution_config_after_staging, cugraph_graph_residency_descriptor, cugraph_schema_query_plan,
    edge_source_graph_vertex_estimate, execute_cugraph_query_to_gpu_with_options, graph_uses_logical_mapping,
    materialize_gpu_output_eagerly, native_graph_spec_for_placement_hint, plan, planned_resident_graph_key,
    preflight_cugraph_query_plan, prepare_cugraph_source_metadata, record_batches_to_gpu, run_gpu_execution_task,
    task_context_with_cugraph_execution_ticket, validate_cugraph_graph_owner_preflight,
    validate_standalone_bounded_cugraph_sources,
};
use crate::{
    attempt::InvocationAttempt,
    native_exec::{
        GraphClaimCalibrationCapture, SealedAttemptMetricsTarget, admission::QueryAdmissionTicket,
        gpu_function_occurrence_from_task_context,
    },
    planner::native::Invocation,
};

#[derive(Clone, Debug)]
pub struct CugraphAlgoExec {
    pub(super) spec: CugraphAlgorithmSpec,
    pub(super) edge_source: Arc<dyn ExecutionPlan>,
    pub(super) predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    pub(super) input_row_bounds: CugraphInputRowBounds,
    pub(super) result_options: CugraphResultPlanOptions,
    pub(super) native_execution_config: AlgeonNativeExecutionConfig,
    pub(super) host_boundary_config: GpuIslandHostBoundaryConfig,
    pub(super) prepared_staged_inputs: Option<Arc<Mutex<Option<StagedCugraphInputs>>>>,
    pub(super) metrics_capture: Option<NativeExecutionMetricsCapture>,
    pub(super) graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    pub(super) edge_source_canonical_facts: CugraphEdgeCanonicalFacts,
    pub(super) graph_residency: Option<GraphResidencyDescriptor>,
    pub(super) gpu_result_input_name: String,
    pub(super) metrics: ExecutionPlanMetricsSet,
    pub(super) cache: Arc<PlanProperties>,
}

impl CugraphAlgoExec {
    #[cfg(test)]
    pub(crate) fn try_new(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: AlgeonNativeExecutionConfig,
    ) -> DataFusionResult<Self> {
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            GpuIslandHostBoundaryConfig::default(),
            Self::next_gpu_result_input_name(),
        )
    }

    pub(crate) fn try_new_with_reserved_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: AlgeonNativeExecutionConfig,
        gpu_result_input_name: String,
    ) -> DataFusionResult<Self> {
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            GpuIslandHostBoundaryConfig::default(),
            gpu_result_input_name,
        )
    }

    pub(crate) fn next_gpu_result_input_name() -> String {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        format!("__algeon_datafusion_cugraph_result_{id}")
    }

    #[cfg(test)]
    pub(crate) fn try_new_with_host_boundary_config(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: AlgeonNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
    ) -> DataFusionResult<Self> {
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            host_boundary_config,
            Self::next_gpu_result_input_name(),
        )
    }

    #[expect(clippy::too_many_arguments, reason = "execution inputs are distinct domain values that share no owner")]
    pub(super) fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: AlgeonNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
        gpu_result_input_name: String,
    ) -> DataFusionResult<Self> {
        let result_options = CugraphResultPlanOptions::new(projection, limit);
        Self::try_new_with_result_options(
            spec,
            edge_source,
            predicate_sources,
            result_options,
            native_execution_config,
            host_boundary_config,
            gpu_result_input_name,
        )
    }

    fn try_new_with_result_options(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        result_options: CugraphResultPlanOptions,
        native_execution_config: AlgeonNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
        gpu_result_input_name: String,
    ) -> DataFusionResult<Self> {
        let input_row_bounds = CugraphInputRowBounds::from_execution_plans(
            edge_source.as_ref(),
            &predicate_sources,
            &native_execution_config.source_read_strategy(),
        )?;
        Self::with_input_row_bounds(
            spec,
            edge_source,
            predicate_sources,
            input_row_bounds,
            result_options,
            native_execution_config,
            host_boundary_config,
            gpu_result_input_name,
        )
        .map_err(DataFusionError::from)
    }

    /// Builds the node from row bounds already computed for these sources, so
    /// only adapter errors remain. Recomputing them would add DataFusion
    /// statistics errors.
    #[expect(clippy::too_many_arguments, reason = "execution inputs are distinct domain values that share no owner")]
    fn with_input_row_bounds(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        input_row_bounds: CugraphInputRowBounds,
        result_options: CugraphResultPlanOptions,
        native_execution_config: AlgeonNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        let schema = projected_schema(&spec.output_schema, result_options.projection.as_ref())?;
        let source_reads = native_execution_config.source_read_strategy();
        let edge_source_canonical_facts = cugraph_edge_canonical_facts(edge_source.as_ref(), &spec, &source_reads);
        let edge_normalize_inserted = cugraph_edge_normalize_inserted(&spec, &edge_source_canonical_facts);
        let edge_vertex_upper_bound = edge_source_graph_vertex_estimate(edge_source.as_ref(), &source_reads);
        preflight_cugraph_query_plan(&CugraphPlanDefinition {
            spec: spec.clone(),
            edge_schema: edge_source.schema(),
            predicate_schemas: predicate_sources.iter().map(|source| source.schema()).collect(),
            predicate_row_upper_bounds: vec![None; predicate_sources.len()],
            result_options: result_options.clone(),
            edge_normalize_inserted,
            edge_row_upper_bound: input_row_bounds.edge_row_upper_bound(),
            edge_vertex_upper_bound,
        })?;
        let graph_residency = cugraph_graph_residency_descriptor(
            &spec,
            Arc::clone(&edge_source),
            edge_normalize_inserted,
            &source_reads,
        )?;
        // Footer pruning is async and can change the exact graph cardinality.
        // Do not let this pre-hydration plan carry an admission ticket; the
        // one immutable ticket is minted after metadata preparation.
        let native_execution_config = native_execution_config.without_query_admission_ticket();
        let metrics_capture = native_execution_config.new_metrics_capture();
        let equivalence_properties = if spec.is_bfs_path_output() {
            match schema.index_of("path_index") {
                Ok(index) => EquivalenceProperties::new_with_orderings(
                    Arc::clone(&schema),
                    [[PhysicalSortExpr::new(Arc::new(Column::new("path_index", index)), SortOptions::default())]],
                ),
                Err(_) => EquivalenceProperties::new(Arc::clone(&schema)),
            }
        } else {
            EquivalenceProperties::new(Arc::clone(&schema))
        };
        Ok(Self {
            spec,
            edge_source,
            predicate_sources,
            input_row_bounds,
            result_options,
            native_execution_config,
            host_boundary_config,
            prepared_staged_inputs: None,
            metrics_capture,
            graph_claim_calibration_capture: GraphClaimCalibrationCapture::default(),
            edge_source_canonical_facts,
            graph_residency,
            gpu_result_input_name,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(PlanProperties::new(
                equivalence_properties,
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }

    pub(crate) fn edge_normalize_inserted(&self) -> bool {
        cugraph_edge_normalize_inserted(&self.spec, &self.edge_source_canonical_facts)
    }

    pub(crate) fn coverage_algorithm_name(&self) -> &'static str {
        self.spec.algorithm.function_name()
    }

    /// Returns the one post-seal attempt projection target for this cuGraph
    /// execution root. It is read only and has no graph/admission authority.
    pub(crate) fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        SealedAttemptMetricsTarget::new(&self.metrics, self.graph_claim_calibration_capture.clone())
    }

    #[cfg(test)]
    pub(crate) fn coverage_gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    pub(super) fn validate_standalone_bounded_sources(&self) -> Result<()> {
        validate_standalone_bounded_cugraph_sources(
            &self.native_execution_config,
            self.edge_source.as_ref(),
            &self.predicate_sources,
        )
    }

    pub(crate) async fn prepare_lakehouse_source_metadata(
        mut self,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<Self> {
        self.edge_source =
            prepare_cugraph_source_metadata(Arc::clone(&self.edge_source), Arc::clone(&task_context)).await?;
        let mut predicate_sources = Vec::with_capacity(self.predicate_sources.len());
        for source in mem::take(&mut self.predicate_sources) {
            predicate_sources.push(prepare_cugraph_source_metadata(source, Arc::clone(&task_context)).await?);
        }
        self.predicate_sources = predicate_sources;
        // Hydration may replace the edge source with its footer-resolved
        // physical shape, so rowset-bound graph provenance must follow it.
        self.graph_residency = cugraph_graph_residency_descriptor(
            &self.spec,
            Arc::clone(&self.edge_source),
            self.edge_normalize_inserted(),
            &self.native_execution_config.source_read_strategy(),
        )?;
        Ok(self)
    }

    pub(super) fn issue_graph_aware_ticket_after_staging(
        mut self,
        input_row_bounds: CugraphInputRowBounds,
        task_context: &TaskContext,
    ) -> Result<(Self, QueryPlan)> {
        let (execution_config, graph_execution_plan) = cugraph_execution_config_after_staging(
            &self.spec,
            self.edge_source.as_ref(),
            &self.predicate_sources,
            &input_row_bounds,
            &self.result_options,
            self.edge_normalize_inserted(),
            &self.native_execution_config,
            task_context,
        )?;
        self.native_execution_config = execution_config;
        self.input_row_bounds = input_row_bounds;
        self.metrics_capture = self.native_execution_config.new_metrics_capture();
        Ok((self, graph_execution_plan))
    }

    pub(super) async fn prepare_staged_attempt(
        self,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<(Self, StagedCugraphInputs)> {
        let prepared = self.prepare_lakehouse_source_metadata(Arc::clone(&task_context)).await?;
        let cancellation = prepared
            .native_execution_config
            .attempt_cancellation(&task_context, "algeon_datafusion.cugraph_sql.host_staging")?;
        let max_staging_bytes = prepared.host_boundary_config.max_input_staging_bytes();
        let max_output_staging_bytes = prepared.host_boundary_config.max_output_staging_bytes();
        let collected_inputs = collect_cugraph_inputs(
            &prepared.spec,
            Arc::clone(&prepared.edge_source),
            prepared.predicate_sources.clone(),
            Arc::clone(&task_context),
            &cancellation,
            max_staging_bytes,
            max_output_staging_bytes,
        )
        .await?;
        let source_reads = prepared.native_execution_config.source_read_strategy();
        let prelogical_row_bounds = collected_inputs.row_bounds(&source_reads)?;
        check_host_staging_cancellation(&cancellation)?;
        validate_cugraph_graph_owner_preflight(
            &prepared.spec,
            prepared.edge_source.as_ref(),
            &prepared.predicate_sources,
            &prelogical_row_bounds,
            &prepared.result_options,
            prepared.edge_normalize_inserted(),
            &prepared.native_execution_config,
        )?;
        check_host_staging_cancellation(&cancellation)?;
        let mut staged_inputs = collected_inputs.prepare_logical_inputs(&prepared.spec)?;
        check_host_staging_cancellation(&cancellation)?;
        let input_row_bounds = staged_inputs.row_bounds(&source_reads)?;
        let (prepared, admission_graph_plan) =
            prepared.issue_graph_aware_ticket_after_staging(input_row_bounds, task_context.as_ref())?;
        staged_inputs.admitted_graph_plan = Some(admission_graph_plan);
        check_host_staging_cancellation(&cancellation)?;
        Ok((prepared, staged_inputs))
    }

    pub(super) fn take_prepared_staged_inputs(&self) -> Result<StagedCugraphInputs> {
        let slot = self.prepared_staged_inputs.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "cuGraph GPU-resident producer inputs were not staged before native context acquisition",
            )
            .with_fact("staged_input_state", "not_prepared")
        })?;
        let mut staged_inputs = slot.lock().map_err(|_| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "cuGraph GPU-resident producer staged-input ownership was poisoned",
            )
            .with_fact("staged_input_state", "poisoned")
        })?;
        staged_inputs.take().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "cuGraph GPU-resident producer staged inputs were already consumed",
            )
            .with_fact("staged_input_state", "already_consumed")
        })
    }

    pub(super) fn bfs_singleflight_key(&self, device_ordinal: usize) -> Option<GraphBfsSingleflightKey> {
        if !self.native_execution_config.graph_execution_services().bfs_singleflight_enabled()
            || !self.native_execution_config.has_query_admission_authority()
            || !self.predicate_sources.is_empty()
        {
            return None;
        }
        let CugraphAlgorithmParams::Bfs(params) = &self.spec.params else {
            return None;
        };
        if params.depth_limit.is_some()
            || params.predicates.has_inputs()
            || params.output.return_target_info
            || params.output.mode == CugraphBfsOutputMode::Path
        {
            return None;
        }
        let source_vertex = params.source.as_i64()?;
        let descriptor = self.graph_residency.as_ref()?;
        let edge_normalize_inserted = self.edge_normalize_inserted();
        let native_spec = native_graph_spec_for_placement_hint(
            &self.spec,
            edge_normalize_inserted,
            &self.predicate_sources,
            &self.edge_source.schema(),
        )?
        .with_edge_normalize_inserted(edge_normalize_inserted);
        let resident_graph_key = planned_resident_graph_key(descriptor, &native_spec)?;
        Some(GraphBfsSingleflightKey {
            device_ordinal,
            resident_graph_key: Some(resident_graph_key),
            source_vertex,
            normalized_output: params.output.mode == CugraphBfsOutputMode::Normalized,
            projection: self.result_options.projection.clone(),
            limit: self.result_options.limit,
            max_output_staging_bytes: self.host_boundary_config.max_output_staging_bytes(),
        })
    }

    async fn execute_gpu_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        let mut staged_inputs = Some(self.take_prepared_staged_inputs()?);
        let runtime = counted_cudf_runtime(
            native_context,
            "algeon_datafusion.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
        if let Some(staged) = staged_inputs.as_ref() {
            staged.validate_admission(native_context)?;
        }
        let Some(key) = self.bfs_singleflight_key(runtime.device().ordinal()) else {
            return self
                .execute_gpu_result_with_context_physical(
                    native_context,
                    partition,
                    task_context,
                    staged_inputs.take().expect("staged inputs are present before the first physical attempt"),
                    None,
                )
                .await;
        };
        let coordinator = self.native_execution_config.graph_execution_services().bfs_singleflight();
        let correlation_fingerprint =
            native_context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str()));
        let attempt_id =
            Invocation::of_task(&task_context).and_then(Invocation::attempt).map(InvocationAttempt::attempt_id);
        loop {
            match coordinator.join_with_attempt(key.clone(), correlation_fingerprint.clone(), attempt_id)? {
                GraphBfsSingleflightJoin::Leader(mut leader) => {
                    match self
                        .execute_gpu_result_with_context_physical(
                            native_context,
                            partition,
                            Arc::clone(&task_context),
                            staged_inputs.take().expect("singleflight leader owns the staged inputs"),
                            Some(&mut leader),
                        )
                        .await
                    {
                        Ok((input_name, handle)) => {
                            let publication: Result<()> = async {
                                if leader.has_followers()? {
                                    leader.publish(GraphBfsPublishedResult::gpu(handle.clone()))?;
                                    // The published frame borrows this attempt's attached domain
                                    // members until every joined follower has copied or dropped
                                    // the handoff. Waiting here keeps leader teardown ordered
                                    // after the last follower copy.
                                    if let Some(drain) = leader.handoff_drain()? {
                                        leader.wait_for_handoff_drain(&drain, native_context.cancellation()).await?;
                                    }
                                } else {
                                    leader.complete_without_result()?;
                                }
                                Ok(())
                            }
                            .await;
                            if let Err(error) = publication {
                                leader.fail();
                                return Err(error.into());
                            }
                            return Ok((input_name, handle));
                        },
                        Err(error) => {
                            leader.fail();
                            return Err(error);
                        },
                    }
                },
                GraphBfsSingleflightJoin::Follower(mut follower) => {
                    match follower.wait(native_context.cancellation()).await? {
                        GraphBfsFollowerOutcome::Ready(result) => {
                            // Release unused host staging before importing or handing off the published result.
                            drop(staged_inputs.take());
                            let started = Instant::now();
                            self.graph_claim_calibration_capture
                                .record_bfs_singleflight_follower(follower.leader_attempt_id());
                            let metrics = AlgeonNativeMetrics::new_with_capture(
                                &self.metrics,
                                partition,
                                self.metrics_capture.clone(),
                            )
                            .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
                            let mut graph_metrics = GraphExecutionMetrics::new(&metrics, native_context);
                            let routed: DataFusionResult<GpuResultHandle> = async {
                                match GraphBfsFollowerRoute::for_gpu_follower(result.kind()) {
                                    GraphBfsFollowerRoute::HandoffGpu => {
                                        let published =
                                            result.gpu_handle().expect("GPU route requires a GPU publication");
                                        // The published handle is retained by the leader's grant: returning it
                                        // directly would fail bound refinement with another attempt's lease.
                                        // Materialize through host staging and re-upload so the follower retains
                                        // its own grant owner, mirroring the host-follower materialization route.
                                        // The download completes the leader's device work before the follower
                                        // re-uploads. The scope claim travels with the copy; the follower's
                                        // own admission evidence re-verifies it before use.
                                        let producer_execution_identity = published.producer_execution_identity();
                                        let output = materialize_gpu_output_eagerly(
                                            published,
                                            task_context.as_ref(),
                                            native_context.cancellation().clone(),
                                            self.host_boundary_config.max_output_staging_bytes(),
                                            None,
                                            &graph_metrics,
                                        )?;
                                        let (batches, _) = output.shared_parts();
                                        let (handle, _) =
                                            self.upload_follower_output(native_context, batches.as_slice())?;
                                        let handle =
                                            handle.with_producer_execution_identity(producer_execution_identity);
                                        follower.record_copy();
                                        self.record_bfs_singleflight_gpu_result_follower_metrics(
                                            &graph_metrics,
                                            &key,
                                            u64::try_from(handle.num_rows()).unwrap_or(u64::MAX),
                                            u64::try_from(handle.num_columns()).unwrap_or(u64::MAX),
                                            started.elapsed(),
                                        );
                                        Ok(handle)
                                    },
                                    GraphBfsFollowerRoute::UploadHost => {
                                        let batches =
                                            result.host_batches().expect("host upload route requires host batches");
                                        let (handle, output_device_bytes) =
                                            self.upload_follower_output(native_context, batches)?;
                                        follower.record_copy();
                                        self.record_bfs_singleflight_gpu_follower_metrics(
                                            &graph_metrics,
                                            &key,
                                            batches,
                                            output_device_bytes,
                                            started.elapsed(),
                                        );
                                        Ok(handle)
                                    },
                                    GraphBfsFollowerRoute::ReuseHost | GraphBfsFollowerRoute::MaterializeGpu => {
                                        unreachable!("GPU follower route must accept a GPU-resident output")
                                    },
                                }
                            }
                            .await;
                            let handle = graph_metrics.finish(routed)?;
                            return Ok((self.gpu_result_input_name.clone(), handle));
                        },
                        GraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    /// Uploads one joined follower's copy of the published BFS result into
    /// its own admitted grant.
    ///
    /// Every follower route pays for its own device memory: the publication
    /// belongs to the leader's attempt, so returning it directly would fail
    /// bound refinement with another attempt's lease. Retaining the upload
    /// under this attempt's resident owner is what makes it an ordinary
    /// executor-published output; without that owner bound refinement sees an
    /// ownerless handle and drops the attempt.
    fn upload_follower_output(
        &self,
        native_context: &ExecutionContext,
        batches: &[RecordBatch],
    ) -> DataFusionResult<(GpuResultHandle, Option<u64>)> {
        let schema = self.schema();
        let frame = record_batches_to_gpu(native_context, Arc::clone(&schema), batches)?;
        let output_device_bytes = frame.device_allocated_size_bytes().ok();
        let resident_owner =
            native_context.retain_resident_output_owner(output_device_bytes.unwrap_or(0)).map_err(|error| {
                Error::from_native(
                    "algeon_datafusion.cugraph_sql.bfs_singleflight_follower",
                    "failed to retain the BFS follower output copy",
                    error,
                )
            })?;
        let handle = GpuResultHandle::new_with_resident_owner(native_context, frame, schema, resident_owner).map_err(
            |error| {
                Error::from_native(
                    "algeon_datafusion.cugraph_sql.bfs_singleflight_follower",
                    "failed to bind the BFS follower output copy",
                    error,
                )
            },
        )?;
        Ok((handle, output_device_bytes))
    }

    pub(super) fn record_bfs_singleflight_gpu_follower_metrics(
        &self,
        graph_metrics: &GraphExecutionMetrics<'_>,
        key: &GraphBfsSingleflightKey,
        batches: &[RecordBatch],
        output_device_bytes: Option<u64>,
        elapsed: Duration,
    ) {
        let output_rows = batches
            .iter()
            .fold(0u64, |rows, batch| rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX)));
        let output_columns = batches.first().map_or(0, |batch| u64::try_from(batch.num_columns()).unwrap_or(u64::MAX));
        graph_metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
            node_id: "bfs_singleflight_follower".to_string(),
            algorithm: "bfs".to_string(),
            runtime_outcome: "succeeded".to_string(),
            input_boundary: "singleflight_host_result".to_string(),
            output_boundary: "gpu_resident".to_string(),
            construction_policy: self.spec.graph_options.construction_policy.as_str().to_string(),
            source_column: CUGRAPH_EDGE_SOURCE_COLUMN.to_string(),
            destination_column: CUGRAPH_EDGE_DESTINATION_COLUMN.to_string(),
            weight_column: None,
            directed: self.spec.graph_options.directed,
            renumber: self.spec.graph_options.renumber,
            algorithm_parameters: format!("source_vertex={}", key.source_vertex),
            algorithm_metadata: "bfs_singleflight=follower".to_string(),
            output_rows,
            output_columns,
            output_device_bytes,
            elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            host_to_device_uploads: 1,
            ..GraphAlgorithmRuntimeMetrics::default()
        });
    }

    pub(super) fn record_bfs_singleflight_gpu_result_follower_metrics(
        &self,
        graph_metrics: &GraphExecutionMetrics<'_>,
        key: &GraphBfsSingleflightKey,
        output_rows: u64,
        output_columns: u64,
        elapsed: Duration,
    ) {
        graph_metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
            node_id: "bfs_singleflight_follower".to_string(),
            algorithm: "bfs".to_string(),
            runtime_outcome: "succeeded".to_string(),
            input_boundary: "singleflight_gpu_result".to_string(),
            output_boundary: "gpu_resident".to_string(),
            construction_policy: self.spec.graph_options.construction_policy.as_str().to_string(),
            source_column: CUGRAPH_EDGE_SOURCE_COLUMN.to_string(),
            destination_column: CUGRAPH_EDGE_DESTINATION_COLUMN.to_string(),
            weight_column: None,
            directed: self.spec.graph_options.directed,
            renumber: self.spec.graph_options.renumber,
            algorithm_parameters: format!("source_vertex={}", key.source_vertex),
            algorithm_metadata: "bfs_singleflight=follower".to_string(),
            output_rows,
            output_columns,
            elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            gpu_resident_input_handoffs: 1,
            ..GraphAlgorithmRuntimeMetrics::default()
        });
    }

    async fn execute_host_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        metrics: &AlgeonNativeMetrics,
        staged_inputs: StagedCugraphInputs,
    ) -> DataFusionResult<EagerCugraphHostOutput> {
        let runtime = counted_cudf_runtime(
            native_context,
            "algeon_datafusion.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
        staged_inputs.validate_admission(native_context)?;
        let Some(key) = self.bfs_singleflight_key(runtime.device().ordinal()) else {
            return Ok(self
                .execute_host_result_with_context_physical(native_context, task_context, metrics, staged_inputs, None)
                .await?
                .record_datafusion_output(metrics));
        };
        let coordinator = self.native_execution_config.graph_execution_services().bfs_singleflight();
        let correlation_fingerprint =
            native_context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str()));
        let attempt_id =
            Invocation::of_task(&task_context).and_then(Invocation::attempt).map(InvocationAttempt::attempt_id);
        let mut staged_inputs = Some(staged_inputs);
        loop {
            match coordinator.join_with_attempt(key.clone(), correlation_fingerprint.clone(), attempt_id)? {
                GraphBfsSingleflightJoin::Leader(mut leader) => {
                    match self
                        .execute_host_result_with_context_physical(
                            native_context,
                            Arc::clone(&task_context),
                            metrics,
                            staged_inputs.take().expect("singleflight leader owns the staged inputs"),
                            Some(&mut leader),
                        )
                        .await
                    {
                        Ok(output) => {
                            let publication = if leader.has_followers()? {
                                leader.publish(output.published_result())
                            } else {
                                leader.complete_without_result()
                            };
                            if let Err(error) = publication {
                                leader.fail();
                                return Err(error.into());
                            }
                            return Ok(output.record_datafusion_output(metrics));
                        },
                        Err(error) => {
                            leader.fail();
                            return Err(error);
                        },
                    }
                },
                GraphBfsSingleflightJoin::Follower(mut follower) => {
                    let started = Instant::now();
                    match follower.wait(native_context.cancellation()).await? {
                        GraphBfsFollowerOutcome::Ready(result) => {
                            // Release unused host staging before reusing or materializing the published result.
                            drop(staged_inputs.take());
                            self.graph_claim_calibration_capture
                                .record_bfs_singleflight_follower(follower.leader_attempt_id());
                            let mut graph_metrics = GraphExecutionMetrics::new(metrics, native_context);
                            let routed: DataFusionResult<EagerCugraphHostOutput> = async {
                                match GraphBfsFollowerRoute::for_host_follower(result.kind()) {
                                    GraphBfsFollowerRoute::ReuseHost => {
                                        let batches =
                                            result.host_batches().expect("host reuse route requires host batches");
                                        follower.record_copy();
                                        self.record_bfs_singleflight_host_follower_metrics(
                                            &graph_metrics,
                                            &key,
                                            batches,
                                            started.elapsed(),
                                        );
                                        Ok(EagerCugraphHostOutput::from_published_result(self.schema(), &result)
                                            .expect("host reuse route requires a host publication"))
                                    },
                                    GraphBfsFollowerRoute::MaterializeGpu => {
                                        let handle = result
                                            .gpu_handle()
                                            .expect("host materialization route requires a GPU publication");
                                        let output = materialize_gpu_output_eagerly(
                                            handle,
                                            task_context.as_ref(),
                                            native_context.cancellation().clone(),
                                            self.host_boundary_config.max_output_staging_bytes(),
                                            None,
                                            &graph_metrics,
                                        )?;
                                        let (batches, _) = output.shared_parts();
                                        follower.record_copy();
                                        self.record_bfs_singleflight_host_follower_metrics_for_input(
                                            &graph_metrics,
                                            &key,
                                            batches.as_slice(),
                                            "singleflight_gpu_result",
                                            started.elapsed(),
                                        );
                                        Ok(output)
                                    },
                                    GraphBfsFollowerRoute::UploadHost | GraphBfsFollowerRoute::HandoffGpu => {
                                        unreachable!("host follower route must produce host Arrow")
                                    },
                                }
                            }
                            .await;
                            return Ok(graph_metrics.finish(routed)?.record_datafusion_output(metrics));
                        },
                        GraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    async fn execute_host_result_with_context_physical(
        &self,
        native_context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        metrics: &AlgeonNativeMetrics,
        staged_inputs: StagedCugraphInputs,
        singleflight_leader: Option<&mut GraphBfsLeader>,
    ) -> DataFusionResult<EagerCugraphHostOutput> {
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let mut graph_metrics = GraphExecutionMetrics::new(metrics, native_context);
        let result = async {
            let UploadedCugraphInputs {
                edge_schema,
                edge_frame,
                predicate_frames,
                host_evidence,
                reservation: _input_reservation,
            } = staged_inputs.upload(native_context, Arc::clone(&task_context)).await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let handle = execute_cugraph_query_to_gpu_with_options(
                native_context,
                &mut graph_metrics,
                &self.plan_definition(),
                edge_frame,
                predicate_frames,
                singleflight_leader,
            )?;
            let output = materialize_gpu_output_eagerly(
                handle,
                task_context.as_ref(),
                native_context.cancellation().clone(),
                self.host_boundary_config.max_output_staging_bytes(),
                Some(&host_evidence),
                &graph_metrics,
            )?;
            Ok(output)
        };
        let result = result.await;
        graph_metrics.finish(result)
    }

    pub(super) fn record_bfs_singleflight_host_follower_metrics(
        &self,
        graph_metrics: &GraphExecutionMetrics<'_>,
        key: &GraphBfsSingleflightKey,
        batches: &[RecordBatch],
        elapsed: Duration,
    ) {
        self.record_bfs_singleflight_host_follower_metrics_for_input(
            graph_metrics,
            key,
            batches,
            "singleflight_host_result",
            elapsed,
        );
    }

    fn record_bfs_singleflight_host_follower_metrics_for_input(
        &self,
        graph_metrics: &GraphExecutionMetrics<'_>,
        key: &GraphBfsSingleflightKey,
        batches: &[RecordBatch],
        input_boundary: &'static str,
        elapsed: Duration,
    ) {
        let output_rows = batches
            .iter()
            .fold(0u64, |rows, batch| rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX)));
        let output_columns = batches.first().map_or(0, |batch| u64::try_from(batch.num_columns()).unwrap_or(u64::MAX));
        graph_metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
            node_id: "bfs_singleflight_follower".to_string(),
            algorithm: "bfs".to_string(),
            runtime_outcome: "succeeded".to_string(),
            input_boundary: input_boundary.to_string(),
            output_boundary: "host_arrow".to_string(),
            construction_policy: self.spec.graph_options.construction_policy.as_str().to_string(),
            source_column: CUGRAPH_EDGE_SOURCE_COLUMN.to_string(),
            destination_column: CUGRAPH_EDGE_DESTINATION_COLUMN.to_string(),
            weight_column: None,
            directed: self.spec.graph_options.directed,
            renumber: self.spec.graph_options.renumber,
            algorithm_parameters: format!("source_vertex={}", key.source_vertex),
            algorithm_metadata: "bfs_singleflight=follower".to_string(),
            output_rows,
            output_columns,
            elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
            ..GraphAlgorithmRuntimeMetrics::default()
        });
    }

    async fn execute_gpu_result_with_context_physical(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
        staged_inputs: StagedCugraphInputs,
        singleflight_leader: Option<&mut GraphBfsLeader>,
    ) -> DataFusionResult<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "cuGraph SQL GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        self.validate_standalone_bounded_sources()?;
        let metrics = AlgeonNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone())
            .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let mut graph_metrics = GraphExecutionMetrics::new(&metrics, native_context);
        let result = async {
            let UploadedCugraphInputs {
                edge_schema,
                edge_frame,
                predicate_frames,
                host_evidence: _,
                reservation: _input_reservation,
            } = staged_inputs.upload(native_context, Arc::clone(&task_context)).await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let handle = execute_cugraph_query_to_gpu_with_options(
                native_context,
                &mut graph_metrics,
                &self.plan_definition(),
                edge_frame,
                predicate_frames,
                singleflight_leader,
            )?;
            Ok((self.gpu_result_input_name.clone(), handle))
        };
        let result = result.await;
        graph_metrics.finish(result)
    }
}

impl GpuResidentProducer for CugraphAlgoExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    fn gpu_result_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        cugraph_deferred_execution_capability_evidence(&self.plan_definition())
    }

    fn host_result_query_plan(&self) -> Result<Option<QueryPlan>> {
        cugraph_schema_query_plan(&self.plan_definition()).map(Some)
    }

    fn host_result_has_deferred_graph_facts(&self) -> bool {
        true
    }

    fn execution_metrics(&self) -> Option<&ExecutionPlanMetricsSet> {
        Some(&self.metrics)
    }

    fn hard_output_row_upper_bound(&self) -> Option<u64> {
        cugraph_schema_query_plan(&self.plan_definition()).ok().as_ref().and_then(query_plan_hard_row_upper_bound)
    }

    fn unique_output_key_columns(&self) -> Option<&'static [&'static str]> {
        cugraph_schema_query_plan(&self.plan_definition()).ok().and_then(|plan| {
            let key = plan.nodes().iter().find_map(|node| match &node.kind {
                PlanNode::GraphAlgorithm(graph) => Some(graph.spec.unique_output_key_columns()),
                _ => None,
            })?;
            (!key.is_empty() && key.iter().all(|column| plan.output_schema().field_with_name(column).is_ok()))
                .then_some(key)
        })
    }

    fn rewrite_subsumed_composed_consumer(&self, consumer_plan: &QueryPlan) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(result_options) = pushed_logical_consumer_options(self, consumer_plan)? else {
            return Ok(None);
        };
        Self::with_input_row_bounds(
            self.spec.clone(),
            Arc::clone(&self.edge_source),
            self.predicate_sources.clone(),
            self.input_row_bounds.clone(),
            result_options,
            self.native_execution_config.clone(),
            self.host_boundary_config,
            self.gpu_result_input_name.clone(),
        )
        .map(|producer| Some(Arc::new(producer) as Arc<dyn ExecutionPlan>))
    }

    fn prepared_admission_tickets(&self) -> Vec<Arc<QueryAdmissionTicket>> {
        self.native_execution_config.query_admission_ticket().into_iter().collect()
    }

    fn prepare_execution_plan<'a>(
        &'a self,
        _plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            let (mut prepared, staged_inputs) = self.clone().prepare_staged_attempt(Arc::clone(&task_context)).await?;
            prepared.prepared_staged_inputs = Some(Arc::new(Mutex::new(Some(staged_inputs))));
            Ok(Arc::new(prepared) as Arc<dyn ExecutionPlan>)
        })
    }

    fn execute_gpu_result_with_context<'a>(
        &'a self,
        native_context: &'a ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, DataFusionResult<(String, GpuResultHandle)>> {
        Box::pin(
            async move { self.execute_gpu_result_with_context_impl(native_context, partition, task_context).await },
        )
    }
}

pub(super) fn pushed_logical_consumer_options(
    producer: &CugraphAlgoExec,
    consumer_plan: &QueryPlan,
) -> Result<Option<CugraphResultPlanOptions>> {
    if producer.spec.is_bfs_path_output()
        || !graph_uses_logical_mapping(&producer.spec, &producer.edge_source.schema())?
        || consumer_plan.output_schema() != &producer.schema()
    {
        return Ok(None);
    }
    let [source_node, remaining @ ..] = consumer_plan.nodes() else {
        return Ok(None);
    };
    let PlanNode::Source(source) = &source_node.kind else {
        return Ok(None);
    };
    let SourceNode::GpuResident(source) = source.as_ref() else {
        return Ok(None);
    };
    if source.input_name != producer.gpu_result_input_name {
        return Ok(None);
    }

    let mut result_options = producer.result_options.clone();
    let mut input = source_node.id;
    for node in remaining {
        match &node.kind {
            PlanNode::Sort(sort)
                if sort.input == input
                    && result_options.subsumed_top_k_sort.is_none()
                    && sort.spec.expressions.iter().all(|expression| {
                        let NativeExpr::Column(column) = &expression.expr else {
                            return false;
                        };
                        producer.schema().field_with_name(column.name()).is_ok()
                    }) =>
            {
                result_options.subsumed_top_k_sort = Some(sort.spec.clone());
            },
            PlanNode::Limit(limit)
                if limit.input == input
                    && limit.spec.skip == 0
                    && limit.spec.fetch.is_some()
                    && result_options.limit.is_none_or(|fetch| Some(fetch) == limit.spec.fetch) =>
            {
                result_options.limit = limit.spec.fetch;
            },
            _ => return Ok(None),
        }
        input = node.id;
    }
    if remaining.is_empty() || consumer_plan.root_id() != input {
        return Ok(None);
    }
    if result_options.subsumed_top_k_sort.is_some() && result_options.limit.is_none() {
        return Ok(None);
    }
    Ok(Some(result_options))
}

impl CugraphAlgoExec {
    /// This node as a GPU-resident producer, when its output can stay on the
    /// device for a native consumer: logical edge-id mapping must be
    /// available for its algorithm.
    pub(crate) fn gpu_resident_producer(&self) -> Option<&dyn GpuResidentProducer> {
        let logical_mapping = graph_uses_logical_mapping(&self.spec, &self.edge_source.schema()).ok()?;
        if self.spec.validate_logical_edge_id_capability(logical_mapping).is_err() {
            return None;
        }
        Some(self)
    }
}

impl CugraphAlgoExec {
    /// The graph-fragment definition this occurrence lowers to.
    pub(super) fn plan_definition(&self) -> CugraphPlanDefinition {
        CugraphPlanDefinition {
            spec: self.spec.clone(),
            edge_schema: self.edge_source.schema(),
            predicate_schemas: self.predicate_sources.iter().map(|source| source.schema()).collect(),
            predicate_row_upper_bounds: self.input_row_bounds.predicate_row_upper_bounds().to_vec(),
            result_options: self.result_options.clone(),
            edge_normalize_inserted: self.edge_normalize_inserted(),
            edge_row_upper_bound: self.input_row_bounds.edge_row_upper_bound(),
            edge_vertex_upper_bound: edge_source_graph_vertex_estimate(
                self.edge_source.as_ref(),
                &self.native_execution_config.source_read_strategy(),
            ),
        }
    }
}

impl GraphExecutionPlan for CugraphAlgoExec {
    fn graph_execution_capability_evidence(&self) -> Result<AttemptCapabilityEvidence> {
        cugraph_deferred_execution_capability_evidence(&self.plan_definition())
    }

    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget {
        self.sealed_attempt_metrics_target()
    }

    fn gpu_resident_output_rejection(&self) -> Option<Error> {
        let logical_mapping = match graph_uses_logical_mapping(&self.spec, &self.edge_source.schema()) {
            Ok(logical_mapping) => logical_mapping,
            Err(error) => return Some(error),
        };
        if let Err(error) = self.spec.validate_logical_edge_id_capability(logical_mapping) {
            return Some(error);
        }
        None
    }
}

fn cugraph_deferred_execution_capability_evidence(
    definition: &CugraphPlanDefinition,
) -> Result<AttemptCapabilityEvidence> {
    let (plan, mut evidence) = cugraph_execution_capability_evidence(definition)?;
    evidence.attest_deferred_cugraph_graph_execution_scope(&plan);
    Ok(evidence)
}

impl DisplayAs for CugraphAlgoExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let weight_column = if self.spec.has_weight { CUGRAPH_EDGE_WEIGHT_COLUMN } else { "<none>" };
        let edge_id_column = if self.spec.has_edge_id { CUGRAPH_EDGE_ID_COLUMN } else { "<none>" };
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CugraphAlgoExec: algorithm={:?}, src_col={}, dst_col={}, weight_col={}, edge_id_col={}, predicate_source_count={}, construction_policy={}, edge_normalize_inserted={}, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.spec.algorithm,
                CUGRAPH_EDGE_SOURCE_COLUMN,
                CUGRAPH_EDGE_DESTINATION_COLUMN,
                weight_column,
                edge_id_column,
                self.predicate_sources.len(),
                self.spec.graph_options.construction_policy.as_str(),
                self.edge_normalize_inserted(),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "algorithm={:?}", self.spec.algorithm)?;
                writeln!(f, "src_col={CUGRAPH_EDGE_SOURCE_COLUMN}")?;
                writeln!(f, "dst_col={CUGRAPH_EDGE_DESTINATION_COLUMN}")?;
                writeln!(f, "weight_col={}", weight_column)?;
                writeln!(f, "edge_id_col={}", edge_id_column)?;
                writeln!(f, "predicate_source_count={}", self.predicate_sources.len())?;
                writeln!(f, "construction_policy={}", self.spec.graph_options.construction_policy.as_str())?;
                writeln!(f, "edge_normalize_inserted={}", self.edge_normalize_inserted())?;
                writeln!(f, "output_boundary=device_resident_or_device_to_host")?;
                writeln!(f, "gpu_result_input={}", self.gpu_result_input_name)
            },
        }
    }
}

impl ExecutionPlan for CugraphAlgoExec {
    fn name(&self) -> &str {
        "CugraphAlgoExec"
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
        let mut children = Vec::with_capacity(1 + self.predicate_sources.len());
        children.push(&self.edge_source);
        children.extend(self.predicate_sources.iter());
        children
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; 1 + self.predicate_sources.len()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let expected = 1 + self.predicate_sources.len();
        if children.len() != expected {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "algeon_datafusion.cugraph_sql.with_new_children",
                "cuGraph SQL execution received an unexpected child count",
            )
            .with_fact("expected_child_count", expected)
            .with_fact("actual_child_count", children.len())
            .into());
        }
        let mut children = children.into_iter();
        let edge_source = children.next().expect("child count checked above");
        let predicate_sources = children.collect();
        Ok(Arc::new(Self::try_new_with_result_options(
            self.spec.clone(),
            edge_source,
            predicate_sources,
            self.result_options.clone(),
            self.native_execution_config.clone(),
            self.host_boundary_config,
            self.gpu_result_input_name.clone(),
        )?))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute",
                "cuGraph SQL execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let exec = self.clone();
        let metrics = AlgeonNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone())
            .with_graph_claim_calibration_capture(self.graph_claim_calibration_capture.clone());
        self.validate_standalone_bounded_sources()?;
        let output = once(async move {
            let (exec, staged_inputs) = exec.prepare_staged_attempt(Arc::clone(&task_context)).await?;
            let execution_config = exec.native_execution_config.clone();
            let task_context = task_context_with_cugraph_execution_ticket(&execution_config, task_context);
            let host_occurrence = match execution_config.query_admission_ticket_for_task_context(&task_context) {
                Some(ticket) => {
                    let occurrence_id =
                        gpu_function_occurrence_from_task_context(&task_context, &exec.gpu_result_input_name)?
                            .unwrap_or_else(|| exec.gpu_result_input_name.clone());
                    ticket.begin_host_occurrence(
                        &task_context,
                        &occurrence_id,
                        &exec.gpu_result_input_name,
                        &cugraph_schema_query_plan(&exec.plan_definition())?,
                    )?
                },
                None => None,
            };
            let context_lease =
                execution_config.execution_context_async(Arc::clone(&task_context)).await.map_err(|error| {
                    DataFusionError::from(
                        error
                            .with_fact("cugraph_sql_stage", "runtime_context")
                            // This graph attempt has entered native runtime context. A
                            // failure is terminal and must not replay through DataFusion.
                            .with_fact("cpu_fallback_after_native_start", false),
                    )
                })?;
            if let Some(admission_facts) = context_lease.admission_facts() {
                metrics.record_admission_facts(admission_facts);
            }
            let gpu_executor = context_lease.context().gpu_executor().map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "algeon_datafusion.cugraph_sql.gpu_execution_pool",
                    "query service closed before cuGraph execution acquired its GPU pool",
                    error,
                ))
            })?;
            let observation = context_lease.context().runtime_observation().clone();
            let output = run_gpu_execution_task(
                gpu_executor,
                observation,
                "algeon_datafusion.cugraph_sql.gpu_execution_pool",
                move |gpu_stream_scope| async move {
                    let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                    let execution_task_context =
                        context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                    let output = exec
                        .execute_host_result_with_context_impl(
                            &context,
                            execution_task_context,
                            &metrics,
                            staged_inputs,
                        )
                        .await?;
                    Ok::<EagerCugraphHostOutput, DataFusionError>(output)
                },
            )
            .await?;
            if let Some(host_occurrence) = host_occurrence {
                host_occurrence.complete();
            }
            Ok::<EagerCugraphHostOutput, DataFusionError>(output)
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

pub(super) fn projected_schema(schema: &SchemaRef, projection: Option<&Vec<usize>>) -> Result<SchemaRef> {
    let Some(projection) = projection else {
        return Ok(Arc::clone(schema));
    };
    let fields = projection
        .iter()
        .map(|index| {
            schema
                .fields()
                .get(*index)
                .map(|field| field.as_ref().clone())
                .ok_or_else(|| plan(format!("cuGraph output projection index {index} is out of range")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}
