use super::*;
#[derive(Clone, Debug)]
pub struct CugraphAlgoExec {
    pub(super) spec: CugraphAlgorithmSpec,
    pub(super) edge_source: Arc<dyn ExecutionPlan>,
    pub(super) predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    pub(super) projection: Option<Vec<usize>>,
    pub(super) limit: Option<usize>,
    pub(super) native_execution_config: NexusNativeExecutionConfig,
    pub(super) metrics_capture: Option<NativeExecutionMetricsCapture>,
    pub(super) edge_source_canonical_facts: CugraphEdgeCanonicalFacts,
    pub(super) graph_residency: Option<GraphResidencyDescriptor>,
    pub(super) gpu_result_input_name: String,
    pub(super) metrics: ExecutionPlanMetricsSet,
    pub(super) cache: Arc<PlanProperties>,
}

impl CugraphAlgoExec {
    pub(crate) fn try_new(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NexusNativeExecutionConfig,
    ) -> Result<Self> {
        let id = NEXT_CUGRAPH_GPU_RESULT_INPUT_ID.fetch_add(1, Ordering::Relaxed);
        Self::try_new_with_gpu_result_input_name(
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
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
        gpu_result_input_name: String,
    ) -> Result<Self> {
        register_cugraph_execution_spis();
        let schema = projected_schema(&spec.output_schema, projection.as_ref())?;
        let edge_source_canonical_facts = cugraph_edge_canonical_facts(edge_source.as_ref(), &spec);
        let edge_normalize_inserted = cugraph_edge_normalize_inserted(&spec, &edge_source_canonical_facts);
        let graph_residency =
            cugraph_graph_residency_descriptor(&spec, Arc::clone(&edge_source), edge_normalize_inserted);
        let placement_hint =
            cugraph_graph_placement_hint(&spec, graph_residency.as_ref(), edge_normalize_inserted, &predicate_sources);
        let native_execution_config = cugraph_execution_config_with_graph_aware_ticket(
            &spec,
            edge_source.as_ref(),
            edge_normalize_inserted,
            placement_hint,
            native_execution_config,
        )?;
        let metrics_capture = native_execution_config.new_metrics_capture();
        Ok(Self {
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
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

    pub(crate) fn graph_placement_hint_for_admission_ticket(&self) -> QueryPlacementHint {
        cugraph_graph_placement_hint(
            &self.spec,
            self.graph_residency.as_ref(),
            self.edge_normalize_inserted(),
            &self.predicate_sources,
        )
    }

    pub(crate) fn coverage_algorithm_name(&self) -> &'static str {
        self.spec.algorithm.function_name()
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

    async fn prepare_lakehouse_source_metadata(mut self, task_context: &TaskContext) -> Self {
        self.edge_source = prepare_cugraph_source_metadata(Arc::clone(&self.edge_source), task_context).await;
        let mut predicate_sources = Vec::with_capacity(self.predicate_sources.len());
        for source in std::mem::take(&mut self.predicate_sources) {
            predicate_sources.push(prepare_cugraph_source_metadata(source, task_context).await);
        }
        self.predicate_sources = predicate_sources;
        self
    }

    pub(super) fn bfs_singleflight_key(&self, device_ordinal: usize) -> Option<GraphBfsSingleflightKey> {
        if !self.native_execution_config.graph_execution_services().bfs_singleflight_enabled()
            || !self.native_execution_config.has_query_admission_authority()
            || self.native_execution_config.query_service_resident_graph_cache_max_retained_bytes().is_none()
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
        })
    }

    async fn execute_gpu_result_with_context_impl(
        &self,
        native_context: &ExecutionContext,
        partition: usize,
        task_context: Arc<TaskContext>,
    ) -> Result<(String, GpuResultHandle)> {
        let runtime = counted_cudf_runtime(
            native_context,
            "datafusion_nexus.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
        let Some(key) = self.bfs_singleflight_key(runtime.device().ordinal()) else {
            return self.execute_gpu_result_with_context_physical(native_context, partition, task_context, None).await;
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
                            Some(&mut leader),
                        )
                        .await
                    {
                        Ok((input_name, handle)) => {
                            let publication: Result<()> = (|| {
                                if leader.has_followers()? {
                                    let batch = handle.to_host_record_batch().map_err(|error| {
                                        DataFusionError::from(Error::from_native(
                                            "datafusion_nexus.cugraph_sql.bfs_singleflight_publish",
                                            "failed to materialize the leader BFS result for coalesced followers",
                                            error,
                                        ))
                                    })?;
                                    leader.publish(Arc::new(batch))?;
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
                        GraphBfsFollowerOutcome::Ready(batch) => {
                            let started = Instant::now();
                            let schema = self.schema();
                            let frame =
                                record_batches_to_gpu(native_context, Arc::clone(&schema), &[batch.as_ref().clone()])?;
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
                                batch.as_ref(),
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
        batch: &RecordBatch,
        output_device_bytes: Option<u64>,
        elapsed: std::time::Duration,
    ) {
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
            algorithm_metadata: "graph_cache=singleflight_follower,bfs_singleflight=follower".to_string(),
            output_rows: u64::try_from(batch.num_rows()).unwrap_or(u64::MAX),
            output_columns: u64::try_from(batch.num_columns()).unwrap_or(u64::MAX),
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
    ) -> Result<RecordBatch> {
        let runtime = counted_cudf_runtime(
            native_context,
            "datafusion_nexus.cugraph_sql.bfs_singleflight.runtime",
            "query service closed before cuGraph BFS execution acquired its runtime",
        )?;
        let Some(key) = self.bfs_singleflight_key(runtime.device().ordinal()) else {
            return self.execute_host_result_with_context_physical(native_context, task_context, metrics, None).await;
        };
        let coordinator = self.native_execution_config.graph_execution_services().bfs_singleflight();
        loop {
            match coordinator.join(key.clone())? {
                GraphBfsSingleflightJoin::Leader(mut leader) => {
                    match self
                        .execute_host_result_with_context_physical(
                            native_context,
                            Arc::clone(&task_context),
                            metrics,
                            Some(&mut leader),
                        )
                        .await
                    {
                        Ok(batch) => {
                            let publication = (|| {
                                if leader.has_followers()? {
                                    leader.publish(Arc::new(batch.clone()))
                                } else {
                                    leader.complete_without_result()
                                }
                            })();
                            if let Err(error) = publication {
                                leader.fail();
                                return Err(error.into());
                            }
                            return Ok(batch);
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
                        GraphBfsFollowerOutcome::Ready(batch) => {
                            coordinator.record_follower_copy();
                            self.record_bfs_singleflight_host_follower_metrics(
                                metrics,
                                &key,
                                batch.as_ref(),
                                started.elapsed(),
                            );
                            return Ok(batch.as_ref().clone());
                        },
                        GraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    async fn execute_host_result_with_context_physical(
        &self,
        context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        metrics: &NexusNativeMetrics,
        mut singleflight_leader: Option<&mut GraphBfsLeader>,
    ) -> Result<RecordBatch> {
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let mut graph_metrics =
            GraphExecutionMetrics::new(metrics, self.native_execution_config.runtime_observation_config());
        let result = async {
            let edge_schema = self.edge_source.schema();
            let edge_frame = execute_edge_source_to_gpu(
                context,
                &mut graph_metrics,
                Arc::clone(&self.edge_source),
                Arc::clone(&task_context),
            )
            .await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let predicate_frames =
                execute_predicate_sources_to_gpu(context, &mut graph_metrics, &self.predicate_sources, task_context)
                    .await?;
            let path_target = bfs_path_target_vertex(context, &self.spec, &predicate_frames)?;
            let prepared =
                prepare_bfs_logical_gpu_inputs(context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
            let batch = if let Some(codebook) = prepared.logical_codebook {
                let handle = execute_cugraph_query_to_gpu_with_options(
                    context,
                    &mut graph_metrics,
                    &prepared.execution_spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    None,
                    None,
                    self.edge_normalize_inserted(),
                    singleflight_leader.as_deref_mut(),
                )?;
                let decoded_schema = bfs_decoded_output_schema_for_host(&self.spec);
                let decoded = decode_bfs_logical_gpu_output(context, handle, &codebook, decoded_schema)?;
                let batch = decoded.to_host_record_batch().map_err(|error| {
                    DataFusionError::from(Error::from_native(
                        "datafusion_nexus.cugraph_sql.logical_output",
                        "failed to materialize decoded cuGraph logical BFS output",
                        error,
                    ))
                })?;
                let batch = if let Some(target) = path_target {
                    reconstruct_bfs_path_batch(&self.spec, &batch, target)?
                } else {
                    batch
                };
                project_and_limit_host_batch(batch, self.projection.as_ref(), self.limit)?
            } else {
                let batch = execute_cugraph_query_to_host(
                    context,
                    &mut graph_metrics,
                    &self.spec,
                    prepared.edge_schema,
                    prepared.edge_frame,
                    prepared.predicate_frames,
                    if path_target.is_some() { None } else { self.projection.as_ref() },
                    if path_target.is_some() { None } else { self.limit },
                    self.edge_normalize_inserted(),
                    singleflight_leader,
                )?;
                if let Some(target) = path_target {
                    let path = reconstruct_bfs_path_batch(&self.spec, &batch, target)?;
                    project_and_limit_host_batch(path, self.projection.as_ref(), self.limit)?
                } else {
                    batch
                }
            };
            Ok(batch)
        };
        let result = result.await;
        graph_metrics.finish(result)
    }

    pub(super) fn record_bfs_singleflight_host_follower_metrics(
        &self,
        metrics: &NexusNativeMetrics,
        key: &GraphBfsSingleflightKey,
        batch: &RecordBatch,
        elapsed: std::time::Duration,
    ) {
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
            algorithm_metadata: "graph_cache=singleflight_follower,bfs_singleflight=follower".to_string(),
            output_rows: u64::try_from(batch.num_rows()).unwrap_or(u64::MAX),
            output_columns: u64::try_from(batch.num_columns()).unwrap_or(u64::MAX),
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
            if is_bfs_path_output(&self.spec) {
                let edge_schema = self.edge_source.schema();
                let edge_frame = execute_edge_source_to_gpu(
                    native_context,
                    &mut graph_metrics,
                    Arc::clone(&self.edge_source),
                    Arc::clone(&task_context),
                )
                .await?;
                let edge_frame =
                    attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
                let predicate_frames = execute_predicate_sources_to_gpu(
                    native_context,
                    &mut graph_metrics,
                    &self.predicate_sources,
                    Arc::clone(&task_context),
                )
                .await?;
                let path_target =
                    bfs_path_target_vertex(native_context, &self.spec, &predicate_frames)?.expect("path mode checked");
                let prepared = prepare_bfs_logical_gpu_inputs(
                    native_context,
                    &self.spec,
                    edge_schema,
                    edge_frame,
                    predicate_frames,
                )?;
                let (batch, path_residency_source) = if let Some(codebook) = prepared.logical_codebook {
                    let handle = execute_cugraph_query_to_gpu_with_options(
                        native_context,
                        &mut graph_metrics,
                        &prepared.execution_spec,
                        prepared.edge_schema,
                        prepared.edge_frame,
                        prepared.predicate_frames,
                        None,
                        None,
                        self.edge_normalize_inserted(),
                        singleflight_leader.as_deref_mut(),
                    )?;
                    let decoded = decode_bfs_logical_gpu_output(
                        native_context,
                        handle,
                        &codebook,
                        bfs_decoded_output_schema_for_host(&self.spec),
                    )?;
                    let batch = decoded.to_host_record_batch().map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "datafusion_nexus.cugraph_sql.logical_output",
                            "failed to materialize decoded cuGraph logical BFS output",
                            error,
                        ))
                    })?;
                    let path_residency_source = decoded.clone().into_gpu_frame();
                    (batch, path_residency_source)
                } else {
                    let handle = execute_cugraph_query_to_gpu_with_options(
                        native_context,
                        &mut graph_metrics,
                        &self.spec,
                        prepared.edge_schema,
                        prepared.edge_frame,
                        prepared.predicate_frames,
                        None,
                        None,
                        self.edge_normalize_inserted(),
                        singleflight_leader.as_deref_mut(),
                    )?;
                    let batch = handle.to_host_record_batch().map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "datafusion_nexus.cugraph_sql.path_output",
                            "failed to materialize cuGraph BFS output for path reconstruction",
                            error,
                        ))
                    })?;
                    let path_residency_source = handle.clone().into_gpu_frame();
                    (batch, path_residency_source)
                };
                let path = reconstruct_bfs_path_batch(&self.spec, &batch, path_target)?;
                let path = project_and_limit_host_batch(path, self.projection.as_ref(), self.limit)?;
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
            let edge_schema = self.edge_source.schema();
            let edge_frame = execute_edge_source_to_gpu(
                native_context,
                &mut graph_metrics,
                Arc::clone(&self.edge_source),
                Arc::clone(&task_context),
            )
            .await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let predicate_frames = execute_predicate_sources_to_gpu(
                native_context,
                &mut graph_metrics,
                &self.predicate_sources,
                Arc::clone(&task_context),
            )
            .await?;
            let prepared =
                prepare_bfs_logical_gpu_inputs(native_context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
            let handle = execute_cugraph_query_to_gpu_with_options(
                native_context,
                &mut graph_metrics,
                &prepared.execution_spec,
                prepared.edge_schema,
                prepared.edge_frame,
                prepared.predicate_frames,
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
        task_context: &'a TaskContext,
    ) -> BoxFuture<'a, Arc<dyn ExecutionPlan>> {
        Box::pin(async move {
            Arc::new(self.clone().prepare_lakehouse_source_metadata(task_context).await) as Arc<dyn ExecutionPlan>
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

    fn graph_placement_hint_for_admission(&self) -> QueryPlacementHint {
        self.graph_placement_hint_for_admission_ticket()
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
            let exec = exec.prepare_lakehouse_source_metadata(task_context.as_ref()).await;
            let execution_config = cugraph_execution_config_for_task_context(
                exec.edge_source.as_ref(),
                &default_execution_config,
                &task_context,
            );
            let _unbounded_native_execution_permit = match execution_config.memory_control_policy() {
                MemoryControlPolicy::Unbounded => match execution_config.unbounded_native_execution_limiter() {
                    Some(limiter) => Some(limiter.acquire().await?),
                    None => None,
                },
                MemoryControlPolicy::Bounded { .. } => None,
            };
            let task_context = task_context_with_cugraph_execution_ticket(&execution_config, task_context);
            let context_lease =
                execution_config.execution_context_async(Arc::clone(&task_context)).await.map_err(|error| {
                    DataFusionError::from(
                        error
                            .with_fact("cugraph_sql_stage", "runtime_context")
                            .with_fact("cpu_fallback_after_native_start", false),
                    )
                })?;
            if let Some(admission_facts) = context_lease.admission_facts() {
                metrics.record_admission_facts(admission_facts);
            }
            let gpu_execution_pool = context_lease.context().gpu_execution_pool().map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cugraph_sql.gpu_execution_pool",
                    "query service closed before cuGraph execution acquired its GPU pool",
                    error,
                ))
            })?;
            let observation = context_lease.context().runtime_observation().clone();
            run_gpu_execution_task(
                gpu_execution_pool,
                observation,
                "datafusion_nexus.cugraph_sql.gpu_execution_pool",
                move |gpu_stream_scope| async move {
                    let context = context_lease.context().with_runtime_gpu_stream_scope(gpu_stream_scope);
                    let execution_task_context =
                        context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                    let batch =
                        exec.execute_host_result_with_context_impl(&context, execution_task_context, &metrics).await?;
                    metrics.record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                    Ok::<Option<RecordBatch>, DataFusionError>(Some(batch.record_output(&metrics.baseline)))
                },
            )
            .await
        })
        .try_filter_map(|batch| async move { Ok(batch) });

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
