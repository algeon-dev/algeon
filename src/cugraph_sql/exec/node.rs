use nexus_query_engine::admission::QueryPlanCapabilityProof;

use super::*;
#[derive(Clone, Debug)]
pub struct CugraphAlgoExec {
    pub(super) spec: CugraphAlgorithmSpec,
    pub(super) edge_source: Arc<dyn ExecutionPlan>,
    pub(super) predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    pub(super) predicate_row_upper_bounds: Vec<Option<u64>>,
    pub(super) projection: Option<Vec<usize>>,
    pub(super) limit: Option<usize>,
    pub(super) native_execution_config: NexusNativeExecutionConfig,
    pub(super) host_boundary_config: GpuIslandHostBoundaryConfig,
    pub(super) prepared_staged_inputs: Option<Arc<std::sync::Mutex<Option<StagedCugraphInputs>>>>,
    pub(super) metrics_capture: Option<NativeExecutionMetricsCapture>,
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
        native_execution_config: NexusNativeExecutionConfig,
    ) -> Result<Self> {
        Self::try_new_with_host_boundary_config(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            GpuIslandHostBoundaryConfig::default(),
        )
    }

    pub(crate) fn try_new_with_host_boundary_config(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NexusNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
    ) -> Result<Self> {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
            host_boundary_config,
            format!("__datafusion_nexus_cugraph_result_{id}"),
        )
    }

    pub(super) fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NexusNativeExecutionConfig,
        host_boundary_config: GpuIslandHostBoundaryConfig,
        gpu_result_input_name: String,
    ) -> Result<Self> {
        register_cugraph_execution_spis();
        let schema = projected_schema(&spec.output_schema, projection.as_ref())?;
        let edge_source_canonical_facts = cugraph_edge_canonical_facts(edge_source.as_ref(), &spec);
        let edge_normalize_inserted = cugraph_edge_normalize_inserted(&spec, &edge_source_canonical_facts);
        let predicate_row_upper_bounds = predicate_sources
            .iter()
            .map(|source| exact_execution_plan_output_rows(source.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        preflight_cugraph_query_plan(
            &spec,
            edge_source.schema(),
            predicate_sources.iter().map(|source| source.schema()).collect(),
            projection.as_ref(),
            limit,
            edge_normalize_inserted,
        )?;
        let graph_residency =
            cugraph_graph_residency_descriptor(&spec, Arc::clone(&edge_source), edge_normalize_inserted);
        let native_execution_config = cugraph_execution_config_with_graph_aware_ticket(
            &spec,
            edge_source.as_ref(),
            &predicate_sources,
            &predicate_row_upper_bounds,
            projection.as_ref(),
            limit,
            edge_normalize_inserted,
            native_execution_config,
        )?;
        let metrics_capture = native_execution_config.new_metrics_capture();
        Ok(Self {
            spec,
            edge_source,
            predicate_sources,
            predicate_row_upper_bounds,
            projection,
            limit,
            native_execution_config,
            host_boundary_config,
            prepared_staged_inputs: None,
            metrics_capture,
            edge_source_canonical_facts,
            graph_residency,
            gpu_result_input_name,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        })
    }

    pub(crate) fn edge_normalize_inserted(&self) -> bool {
        cugraph_edge_normalize_inserted(&self.spec, &self.edge_source_canonical_facts)
    }

    pub(crate) fn graph_model_ask_bytes_for_admission_ticket(&self) -> Result<Option<GraphAdmissionModelAsk>> {
        cugraph_graph_model_ask_bytes(
            &self.spec,
            self.edge_source.as_ref(),
            self.edge_normalize_inserted(),
            &self.native_execution_config,
        )
    }

    pub(crate) fn coverage_algorithm_name(&self) -> &'static str {
        self.spec.algorithm.function_name()
    }

    /// Returns the one post-seal attempt projection target for this cuGraph
    /// execution root. It is read only and has no graph/admission authority.
    pub(crate) fn sealed_attempt_metrics_target(&self) -> crate::native_exec::SealedAttemptMetricsTarget {
        crate::native_exec::SealedAttemptMetricsTarget::new(&self.metrics)
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

    async fn prepare_lakehouse_source_metadata(mut self, task_context: Arc<TaskContext>) -> Result<Self> {
        self.edge_source =
            prepare_cugraph_source_metadata(Arc::clone(&self.edge_source), Arc::clone(&task_context)).await?;
        let mut predicate_sources = Vec::with_capacity(self.predicate_sources.len());
        for source in std::mem::take(&mut self.predicate_sources) {
            predicate_sources.push(prepare_cugraph_source_metadata(source, Arc::clone(&task_context)).await?);
        }
        self.predicate_sources = predicate_sources;
        Ok(self)
    }

    pub(super) fn take_prepared_staged_inputs(&self) -> Result<StagedCugraphInputs> {
        let slot = self.prepared_staged_inputs.as_ref().ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.execute_gpu",
                    "cuGraph GPU-resident producer inputs were not staged before native context acquisition",
                )
                .with_fact("staged_input_state", "not_prepared"),
            )
        })?;
        let mut staged_inputs = slot.lock().map_err(|_| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.execute_gpu",
                    "cuGraph GPU-resident producer staged-input ownership was poisoned",
                )
                .with_fact("staged_input_state", "poisoned"),
            )
        })?;
        staged_inputs.take().ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.execute_gpu",
                    "cuGraph GPU-resident producer staged inputs were already consumed",
                )
                .with_fact("staged_input_state", "already_consumed"),
            )
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
        let native_spec =
            native_graph_spec_for_placement_hint(&self.spec, edge_normalize_inserted, &self.predicate_sources)
                .with_edge_normalize_inserted(edge_normalize_inserted);
        let resident_graph_key = planned_resident_graph_key(descriptor, &native_spec)?;
        Some(GraphBfsSingleflightKey {
            device_ordinal,
            resident_graph_key: Some(resident_graph_key),
            source_vertex,
            normalized_output: params.output.mode == CugraphBfsOutputMode::Normalized,
            projection: self.projection.clone(),
            limit: self.limit,
            max_output_staging_bytes: self.host_boundary_config.max_output_staging_bytes(),
        })
    }

    async fn execute_gpu_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> Result<(String, GpuResultHandle)> {
        let mut staged_inputs = Some(self.take_prepared_staged_inputs()?);
        let runtime = counted_cudf_runtime(
            native_context,
            "datafusion_nexus.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
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
        loop {
            match coordinator.join(key.clone())? {
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
                            let publication: Result<()> = (|| {
                                if leader.has_followers()? {
                                    let publication_metrics = NexusNativeMetrics::new_with_capture(
                                        &self.metrics,
                                        partition,
                                        self.metrics_capture.clone(),
                                    );
                                    let mut graph_metrics = GraphExecutionMetrics::new(
                                        &publication_metrics,
                                        self.native_execution_config.runtime_observation_config(),
                                    );
                                    let output = materialize_gpu_output_eagerly(
                                        handle.clone(),
                                        task_context.as_ref(),
                                        native_context.cancellation().clone(),
                                        self.host_boundary_config.max_output_staging_bytes(),
                                        &mut graph_metrics,
                                    )?;
                                    graph_metrics.finish(Ok(()))?;
                                    leader.publish(output.published_result())?;
                                } else {
                                    leader.complete_without_result()?;
                                }
                                Ok(())
                            })();
                            if let Err(error) = publication {
                                leader.fail();
                                return Err(error);
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
                            drop(staged_inputs.take());
                            let started = Instant::now();
                            let schema = self.schema();
                            let frame = record_batches_to_gpu(native_context, Arc::clone(&schema), result.batches())?;
                            let output_device_bytes = frame.device_allocated_size_bytes().ok();
                            let handle = GpuResultHandle::new(native_context, frame, schema).map_err(|error| {
                                DataFusionError::from(Error::from_native(
                                    "datafusion_nexus.cugraph_sql.bfs_singleflight_follower",
                                    "failed to bind the coalesced BFS follower output",
                                    error,
                                ))
                            })?;
                            coordinator.record_follower_copy();
                            self.record_bfs_singleflight_gpu_follower_metrics(
                                partition,
                                &key,
                                result.batches(),
                                output_device_bytes,
                                started.elapsed(),
                            );
                            return Ok((self.gpu_result_input_name.clone(), handle));
                        },
                        GraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    pub(super) fn record_bfs_singleflight_gpu_follower_metrics(
        &self,
        partition: usize,
        key: &GraphBfsSingleflightKey,
        batches: &[RecordBatch],
        output_device_bytes: Option<u64>,
        elapsed: std::time::Duration,
    ) {
        let output_rows = batches
            .iter()
            .fold(0u64, |rows, batch| rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX)));
        let output_columns = batches.first().map_or(0, |batch| u64::try_from(batch.num_columns()).unwrap_or(u64::MAX));
        let metrics = NexusNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let mut query_metrics = QueryMetrics::default();
        query_metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
            node_id: "bfs_singleflight_follower".to_string(),
            algorithm: "bfs".to_string(),
            runtime_outcome: "succeeded".to_string(),
            input_boundary: "singleflight_host_result".to_string(),
            output_boundary: "gpu_resident".to_string(),
            construction_policy: self.spec.graph_options.construction_policy.as_str().to_string(),
            source_column: self.spec.src_col.clone(),
            destination_column: self.spec.dst_col.clone(),
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
        metrics.record_query_metrics(&query_metrics);
    }

    async fn execute_host_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        metrics: &NexusNativeMetrics,
        staged_inputs: StagedCugraphInputs,
    ) -> Result<EagerCugraphHostOutput> {
        let runtime = counted_cudf_runtime(
            native_context,
            "datafusion_nexus.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
        let Some(key) = self.bfs_singleflight_key(runtime.device().ordinal()) else {
            return Ok(self
                .execute_host_result_with_context_physical(native_context, task_context, metrics, staged_inputs, None)
                .await?
                .record_datafusion_output(metrics));
        };
        let coordinator = self.native_execution_config.graph_execution_services().bfs_singleflight();
        let mut staged_inputs = Some(staged_inputs);
        loop {
            match coordinator.join(key.clone())? {
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
                            drop(staged_inputs.take());
                            coordinator.record_follower_copy();
                            self.record_bfs_singleflight_host_follower_metrics(
                                metrics,
                                &key,
                                result.batches(),
                                started.elapsed(),
                            );
                            return Ok(EagerCugraphHostOutput::from_published_result(self.schema(), &result)
                                .record_datafusion_output(metrics));
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
        metrics: &NexusNativeMetrics,
        staged_inputs: StagedCugraphInputs,
        singleflight_leader: Option<&mut GraphBfsLeader>,
    ) -> Result<EagerCugraphHostOutput> {
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let mut graph_metrics =
            GraphExecutionMetrics::new(metrics, self.native_execution_config.runtime_observation_config());
        let result = async {
            let UploadedCugraphInputs { edge_schema, edge_frame, logical_literal_source, predicate_frames } =
                staged_inputs.upload(native_context, &mut graph_metrics, Arc::clone(&task_context)).await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let path_target = bfs_path_target_vertex(
                native_context,
                &self.spec,
                &predicate_frames,
                task_context.as_ref(),
                self.host_boundary_config.max_output_staging_bytes(),
                &mut graph_metrics,
            )?;
            let prepared = prepare_bfs_logical_gpu_inputs(
                native_context,
                &self.spec,
                edge_schema,
                edge_frame,
                logical_literal_source,
                predicate_frames,
            )?;
            let logical_mapping = prepared.logical_codebook.is_some();
            let handle = execute_cugraph_query_to_gpu_with_options(
                native_context,
                &mut graph_metrics,
                &prepared.execution_spec,
                prepared.edge_schema,
                prepared.edge_frame,
                prepared.predicate_frames,
                &self.predicate_row_upper_bounds,
                if logical_mapping || path_target.is_some() { None } else { self.projection.as_ref() },
                if logical_mapping || path_target.is_some() { None } else { self.limit },
                self.edge_normalize_inserted(),
                singleflight_leader,
            )?;
            let handle = if let Some(codebook) = prepared.logical_codebook {
                let decoded = decode_bfs_logical_gpu_output(
                    native_context,
                    handle,
                    &codebook,
                    bfs_decoded_output_schema_for_host(&self.spec),
                )?;
                if path_target.is_some() {
                    decoded
                } else {
                    apply_gpu_projection_limit(
                        native_context,
                        &mut graph_metrics,
                        decoded,
                        Arc::clone(&self.spec.output_schema),
                        self.projection.as_ref(),
                        self.limit,
                    )?
                }
            } else {
                handle
            };
            if let Some(path_target) = path_target {
                let (path, reservation) = materialize_bfs_path_output(
                    native_context,
                    task_context.as_ref(),
                    handle,
                    &self.spec,
                    path_target,
                    self.projection.as_ref(),
                    self.limit,
                    self.host_boundary_config.max_output_staging_bytes(),
                    &mut graph_metrics,
                )?;
                return Ok(EagerCugraphHostOutput::from_shared_parts(self.schema(), Arc::new(vec![path]), reservation));
            }
            materialize_gpu_output_eagerly(
                handle,
                task_context.as_ref(),
                native_context.cancellation().clone(),
                self.host_boundary_config.max_output_staging_bytes(),
                &mut graph_metrics,
            )
        };
        let result = result.await;
        graph_metrics.finish(result)
    }

    pub(super) fn record_bfs_singleflight_host_follower_metrics(
        &self,
        metrics: &NexusNativeMetrics,
        key: &GraphBfsSingleflightKey,
        batches: &[RecordBatch],
        elapsed: std::time::Duration,
    ) {
        let output_rows = batches
            .iter()
            .fold(0u64, |rows, batch| rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX)));
        let output_columns = batches.first().map_or(0, |batch| u64::try_from(batch.num_columns()).unwrap_or(u64::MAX));
        let mut query_metrics = QueryMetrics::default();
        query_metrics.record_graph_algorithm_runtime(GraphAlgorithmRuntimeMetrics {
            node_id: "bfs_singleflight_follower".to_string(),
            algorithm: "bfs".to_string(),
            runtime_outcome: "succeeded".to_string(),
            input_boundary: "singleflight_host_result".to_string(),
            output_boundary: "host_arrow".to_string(),
            construction_policy: self.spec.graph_options.construction_policy.as_str().to_string(),
            source_column: self.spec.src_col.clone(),
            destination_column: self.spec.dst_col.clone(),
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
        metrics.record_query_metrics(&query_metrics);
    }

    async fn execute_gpu_result_with_context_physical(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
        staged_inputs: StagedCugraphInputs,
        mut singleflight_leader: Option<&mut GraphBfsLeader>,
    ) -> Result<(String, GpuResultHandle)> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.execute_gpu",
                "cuGraph SQL GPU-resident producer exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }
        self.validate_standalone_bounded_sources()?;
        let metrics = NexusNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let mut graph_metrics =
            GraphExecutionMetrics::new(&metrics, self.native_execution_config.runtime_observation_config());
        let result = async {
            let UploadedCugraphInputs { edge_schema, edge_frame, logical_literal_source, predicate_frames } =
                staged_inputs.upload(native_context, &mut graph_metrics, Arc::clone(&task_context)).await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            if is_bfs_path_output(&self.spec) {
                let path_target = bfs_path_target_vertex(
                    native_context,
                    &self.spec,
                    &predicate_frames,
                    task_context.as_ref(),
                    self.host_boundary_config.max_output_staging_bytes(),
                    &mut graph_metrics,
                )?
                .expect("path mode checked");
                let prepared = prepare_bfs_logical_gpu_inputs(
                    native_context,
                    &self.spec,
                    edge_schema,
                    edge_frame,
                    logical_literal_source,
                    predicate_frames,
                )?;
                let handle = execute_cugraph_query_to_gpu_with_options(
                    native_context,
                    &mut graph_metrics,
                    &prepared.execution_spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    &self.predicate_row_upper_bounds,
                    None,
                    None,
                    self.edge_normalize_inserted(),
                    singleflight_leader.as_deref_mut(),
                )?;
                let handle = if let Some(codebook) = prepared.logical_codebook {
                    decode_bfs_logical_gpu_output(
                        native_context,
                        handle,
                        &codebook,
                        bfs_decoded_output_schema_for_host(&self.spec),
                    )?
                } else {
                    handle
                };
                let path_residency_source = handle.clone().into_gpu_frame();
                let (path, _reservation) = materialize_bfs_path_output(
                    native_context,
                    task_context.as_ref(),
                    handle,
                    &self.spec,
                    path_target,
                    self.projection.as_ref(),
                    self.limit,
                    self.host_boundary_config.max_output_staging_bytes(),
                    &mut graph_metrics,
                )?;
                let schema = projected_schema(&self.spec.output_schema, self.projection.as_ref())?;
                let frame = record_batches_to_gpu(native_context, Arc::clone(&schema), &[path])?;
                let frame = retain_mapped_cugraph_frame(
                    frame.into_frame(),
                    &path_residency_source,
                    "datafusion_nexus.cugraph_sql.path_output",
                    "failed to retain BFS path GPU output residency",
                )?;
                let handle = GpuResultHandle::new(native_context, frame, schema).map_err(|error| {
                    DataFusionError::from(Error::from_native(
                        "datafusion_nexus.cugraph_sql.path_output",
                        "failed to bind BFS path output as a GPU-resident result",
                        error,
                    ))
                })?;
                return Ok((self.gpu_result_input_name.clone(), handle));
            }
            let prepared = prepare_bfs_logical_gpu_inputs(
                native_context,
                &self.spec,
                edge_schema,
                edge_frame,
                logical_literal_source,
                predicate_frames,
            )?;
            let handle = execute_cugraph_query_to_gpu_with_options(
                native_context,
                &mut graph_metrics,
                &prepared.execution_spec,
                prepared.edge_schema,
                prepared.edge_frame,
                prepared.predicate_frames,
                &self.predicate_row_upper_bounds,
                if prepared.logical_codebook.is_some() { None } else { self.projection.as_ref() },
                if prepared.logical_codebook.is_some() { None } else { self.limit },
                self.edge_normalize_inserted(),
                singleflight_leader,
            )?;
            let handle = if let Some(codebook) = prepared.logical_codebook {
                let decoded = decode_bfs_logical_gpu_output(
                    native_context,
                    handle,
                    &codebook,
                    Arc::clone(&self.spec.output_schema),
                )?;
                apply_gpu_projection_limit(
                    native_context,
                    &mut graph_metrics,
                    decoded,
                    Arc::clone(&self.spec.output_schema),
                    self.projection.as_ref(),
                    self.limit,
                )?
            } else {
                handle
            };
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

    fn prepare_execution_plan<'a>(
        &'a self,
        _plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            let mut prepared = self.clone().prepare_lakehouse_source_metadata(Arc::clone(&task_context)).await?;
            prepared.prepared_staged_inputs = Some(Arc::new(std::sync::Mutex::new(Some(
                stage_cugraph_inputs(
                    &prepared.spec,
                    Arc::clone(&prepared.edge_source),
                    prepared.predicate_sources.clone(),
                    task_context,
                    prepared.native_execution_config.cancellation(),
                    prepared.host_boundary_config.max_input_staging_bytes(),
                )
                .await?,
            ))));
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

pub(super) fn cugraph_gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducer> {
    plan.downcast_ref::<CugraphAlgoExec>().map(|producer| producer as &dyn GpuResidentProducer)
}

pub(super) fn cugraph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    plan.downcast_ref::<CugraphAlgoExec>().map(|exec| exec as &dyn GraphExecutionPlan)
}

pub(super) fn register_cugraph_execution_spis() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        register_gpu_resident_producer_discovery(cugraph_gpu_resident_producer);
        register_graph_execution_plan_discovery(cugraph_execution_plan);
    });
}

impl GraphExecutionPlan for CugraphAlgoExec {
    fn graph_model_ask_for_admission(&self) -> Result<Option<GraphAdmissionModelAsk>> {
        self.graph_model_ask_bytes_for_admission_ticket()
    }

    fn graph_execution_capability_proof(&self) -> Result<QueryPlanCapabilityProof> {
        cugraph_execution_capability_proof(
            &self.spec,
            self.edge_source.schema(),
            self.predicate_sources.iter().map(|source| source.schema()).collect(),
            &self.predicate_row_upper_bounds,
            self.projection.as_ref(),
            self.limit,
            cugraph_edge_normalize_inserted(&self.spec, &self.edge_source_canonical_facts),
        )
    }

    fn sealed_attempt_metrics_target(&self) -> crate::native_exec::SealedAttemptMetricsTarget {
        self.sealed_attempt_metrics_target()
    }
}

impl DisplayAs for CugraphAlgoExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CugraphAlgoExec: algorithm={:?}, table={}, src_col={}, dst_col={}, weight_col={}, edge_id_col={}, predicate_source_count={}, construction_policy={}, memory_control_policy={}, edge_normalize_inserted={}, output_boundary=device_resident_or_device_to_host, gpu_result_input={}",
                self.spec.algorithm,
                self.spec.table_name,
                self.spec.src_col,
                self.spec.dst_col,
                self.spec.weight_col.as_deref().unwrap_or("<none>"),
                self.spec.edge_id_col.as_deref().unwrap_or("<none>"),
                self.predicate_sources.len(),
                self.spec.graph_options.construction_policy.as_str(),
                self.native_execution_config.memory_control_policy(),
                self.edge_normalize_inserted(),
                self.gpu_result_input_name,
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "algorithm={:?}", self.spec.algorithm)?;
                writeln!(f, "table={}", self.spec.table_name)?;
                writeln!(f, "src_col={}", self.spec.src_col)?;
                writeln!(f, "dst_col={}", self.spec.dst_col)?;
                writeln!(f, "weight_col={}", self.spec.weight_col.as_deref().unwrap_or("<none>"))?;
                writeln!(f, "edge_id_col={}", self.spec.edge_id_col.as_deref().unwrap_or("<none>"))?;
                writeln!(f, "predicate_source_count={}", self.predicate_sources.len())?;
                writeln!(f, "construction_policy={}", self.spec.graph_options.construction_policy.as_str())?;
                writeln!(f, "memory_control_policy={}", self.native_execution_config.memory_control_policy())?;
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = Vec::with_capacity(1 + self.predicate_sources.len());
        children.push(&self.edge_source);
        children.extend(self.predicate_sources.iter());
        children
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; 1 + self.predicate_sources.len()]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        let expected = 1 + self.predicate_sources.len();
        if children.len() != expected {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_nexus.cugraph_sql.with_new_children",
                "cuGraph SQL execution received an unexpected child count",
            )
            .with_fact("expected_child_count", expected)
            .with_fact("actual_child_count", children.len())
            .into());
        }
        let mut children = children.into_iter();
        let edge_source = children.next().expect("child count checked above");
        let predicate_sources = children.collect();
        Ok(Arc::new(Self::try_new_with_gpu_result_input_name(
            self.spec.clone(),
            edge_source,
            predicate_sources,
            self.projection.clone(),
            self.limit,
            self.native_execution_config.clone(),
            self.host_boundary_config,
            self.gpu_result_input_name.clone(),
        )?))
    }

    fn execute(&self, partition: usize, task_context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.execute",
                "cuGraph SQL execution exposes exactly one output partition",
            )
            .with_fact("partition", partition)
            .into());
        }

        let schema = self.schema();
        let exec = self.clone();
        let default_execution_config = self.native_execution_config.clone();
        let metrics = NexusNativeMetrics::new_with_capture(&self.metrics, partition, self.metrics_capture.clone());
        self.validate_standalone_bounded_sources()?;
        let output = futures::stream::once(async move {
            let exec = exec.prepare_lakehouse_source_metadata(Arc::clone(&task_context)).await?;
            let execution_config = cugraph_execution_config_for_task_context(
                exec.edge_source.as_ref(),
                &default_execution_config,
                &task_context,
            );
            let task_context = task_context_with_cugraph_execution_ticket(&execution_config, task_context);
            let staged_inputs = stage_cugraph_inputs(
                &exec.spec,
                Arc::clone(&exec.edge_source),
                exec.predicate_sources.clone(),
                Arc::clone(&task_context),
                execution_config.cancellation(),
                exec.host_boundary_config.max_input_staging_bytes(),
            )
            .await?;
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
                    "datafusion_nexus.cugraph_sql.gpu_execution_pool",
                    "query service closed before cuGraph execution acquired its GPU pool",
                    error,
                ))
            })?;
            let observation = context_lease.context().runtime_observation().clone();
            run_gpu_execution_task(
                gpu_executor,
                observation,
                "datafusion_nexus.cugraph_sql.gpu_execution_pool",
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
                    metrics.record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                    Ok::<EagerCugraphHostOutput, DataFusionError>(output)
                },
            )
            .await
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
                .ok_or_else(|| plan_error(format!("cuGraph output projection index {index} is out of range")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}
