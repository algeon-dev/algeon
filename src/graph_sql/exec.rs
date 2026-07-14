use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, Series, frame::factorization::Codebook};
use datafusion::{
    common::{DataFusionError, Result, stats::Precision},
    datasource::{memory::MemorySourceConfig, physical_plan::FileScanConfig, source::DataSourceExec},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, expressions::Column},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        aggregates::{AggregateExec, AggregateMode},
        collect,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::{TryStreamExt, future::BoxFuture};
use nexus_query_engine::{
    BinaryOp, CaseWhen, ExecutionContext, Expr as NativeExpr, GpuDataFrame, GpuResidentInputMap, GpuResidentSource,
    GpuResultHandle, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmRuntimeMetrics, GraphAlgorithmSpec,
    GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
    GraphBfsPredicateInputs as NativeGraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
    GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams,
    GraphEdgeColumnBindings, GraphEdgePredicateInput as NativeGraphEdgePredicateInput,
    GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams,
    GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams,
    GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams,
    GraphPageRankPersonalizationInput as NativeGraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
    GraphPredicateInput as NativeGraphPredicateInput, GraphResidencyDescriptor, GraphResidencyPlanAnalysis,
    GraphSccParams, GraphSourceFingerprintBypassReason, GraphSpectralClusteringParams, GraphSsspParams,
    GraphTriangleCountAllParams, GraphWccParams, LimitSpec, LiteralValue, MemoryControlPolicy, PlanNode,
    ProjectionOutput, QueryMetrics, QueryOutput, QueryPlacementHint, QueryPlanBuilder, SourceNode, execute_query_plan,
    execute_query_plan_with_gpu_inputs, graph_edge_list_output_rows_for_plan, graph_memory_estimate_for_edge_list,
    graph_residency_descriptor_for_edge_plan, planned_resident_graph_key,
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsFollowerOutcome, CugraphBfsLeader,
    CugraphBfsOutputMode, CugraphBfsSingleflightJoin, CugraphBfsSingleflightKey, CugraphBfsSource,
    CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphKCoreDegreeType, plan_error,
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        GpuResidentProducerExec, GraphAdmissionModelAsk, NativeCudfComposedExec, NativeCudfExec,
        NativeCudfExecutionConfig, NativeCudfGpuProducerExec, NativeCudfMetrics,
        admission::task_context_with_query_admission_ticket, gpu_resident_producer as native_gpu_resident_producer,
        query_admission_ticket_from_task_context, run_gpu_execution_task,
    },
};

static NEXT_CUGRAPH_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct CugraphAlgoExec {
    spec: CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    native_execution_config: NativeCudfExecutionConfig,
    edge_source_canonical_facts: CugraphEdgeCanonicalFacts,
    graph_residency: Option<GraphResidencyDescriptor>,
    gpu_result_input_name: String,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl CugraphAlgoExec {
    pub(crate) fn try_new(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NativeCudfExecutionConfig,
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

    fn try_new_with_gpu_result_input_name(
        spec: CugraphAlgorithmSpec,
        edge_source: Arc<dyn ExecutionPlan>,
        predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        native_execution_config: NativeCudfExecutionConfig,
        gpu_result_input_name: String,
    ) -> Result<Self> {
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
        Ok(Self {
            spec,
            edge_source,
            predicate_sources,
            projection,
            limit,
            native_execution_config,
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

    fn validate_standalone_bounded_sources(&self) -> Result<()> {
        validate_standalone_bounded_cugraph_sources(
            &self.native_execution_config,
            self.edge_source.as_ref(),
            &self.predicate_sources,
        )
    }

    fn bfs_singleflight_key(&self, device_ordinal: usize) -> Option<CugraphBfsSingleflightKey> {
        if !self.native_execution_config.cugraph_bfs_singleflight_enabled()
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
        Some(CugraphBfsSingleflightKey {
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
        let Some(key) = self.bfs_singleflight_key(native_context.runtime().device().ordinal()) else {
            return self.execute_gpu_result_with_context_physical(native_context, partition, task_context, None).await;
        };
        let coordinator = self.native_execution_config.cugraph_bfs_singleflight();
        loop {
            match coordinator.join(key.clone())? {
                CugraphBfsSingleflightJoin::Leader(mut leader) => {
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
                CugraphBfsSingleflightJoin::Follower(mut follower) => {
                    match follower.wait(native_context.cancellation()).await? {
                        CugraphBfsFollowerOutcome::Ready(batch) => {
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
                        CugraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    fn record_bfs_singleflight_gpu_follower_metrics(
        &self,
        partition: usize,
        key: &CugraphBfsSingleflightKey,
        batch: &RecordBatch,
        output_device_bytes: Option<u64>,
        elapsed: std::time::Duration,
    ) {
        let metrics = NativeCudfMetrics::new_with_capture(
            &self.metrics,
            partition,
            self.native_execution_config.metrics_capture(),
        );
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
        metrics: &NativeCudfMetrics,
    ) -> Result<RecordBatch> {
        let Some(key) = self.bfs_singleflight_key(native_context.runtime().device().ordinal()) else {
            return self.execute_host_result_with_context_physical(native_context, task_context, metrics, None).await;
        };
        let coordinator = self.native_execution_config.cugraph_bfs_singleflight();
        loop {
            match coordinator.join(key.clone())? {
                CugraphBfsSingleflightJoin::Leader(mut leader) => {
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
                CugraphBfsSingleflightJoin::Follower(mut follower) => {
                    let started = Instant::now();
                    match follower.wait(native_context.cancellation()).await? {
                        CugraphBfsFollowerOutcome::Ready(batch) => {
                            coordinator.record_follower_copy();
                            self.record_bfs_singleflight_host_follower_metrics(
                                metrics,
                                &key,
                                batch.as_ref(),
                                started.elapsed(),
                            );
                            return Ok(batch.as_ref().clone());
                        },
                        CugraphBfsFollowerOutcome::Retry => continue,
                    }
                },
            }
        }
    }

    async fn execute_host_result_with_context_physical(
        &self,
        context: &ExecutionContext,
        task_context: Arc<TaskContext>,
        metrics: &NativeCudfMetrics,
        mut singleflight_leader: Option<&mut CugraphBfsLeader>,
    ) -> Result<RecordBatch> {
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let edge_schema = self.edge_source.schema();
        let (edge_frame, mut source_metrics) =
            execute_edge_source_to_gpu(context, Arc::clone(&self.edge_source), Arc::clone(&task_context)).await?;
        let edge_frame =
            attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
        let (predicate_frames, predicate_source_metrics) =
            execute_predicate_sources_to_gpu(context, &self.predicate_sources, task_context).await?;
        source_metrics.merge_from(predicate_source_metrics);
        let path_target = bfs_path_target_vertex(context, &self.spec, &predicate_frames)?;
        let prepared = prepare_bfs_logical_gpu_inputs(context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
        let (batch, mut query_metrics) = if let Some(codebook) = prepared.logical_codebook {
            let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
                context,
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
            (project_and_limit_host_batch(batch, self.projection.as_ref(), self.limit)?, query_metrics)
        } else {
            let (batch, query_metrics) = execute_cugraph_query_to_host(
                context,
                &self.spec,
                prepared.edge_schema,
                prepared.edge_frame,
                prepared.predicate_frames,
                if path_target.is_some() { None } else { self.projection.as_ref() },
                if path_target.is_some() { None } else { self.limit },
                self.edge_normalize_inserted(),
                singleflight_leader,
            )?;
            let batch = if let Some(target) = path_target {
                let path = reconstruct_bfs_path_batch(&self.spec, &batch, target)?;
                project_and_limit_host_batch(path, self.projection.as_ref(), self.limit)?
            } else {
                batch
            };
            (batch, query_metrics)
        };
        query_metrics.merge_from(source_metrics);
        metrics.record_query_metrics(&query_metrics);
        Ok(batch)
    }

    fn record_bfs_singleflight_host_follower_metrics(
        &self,
        metrics: &NativeCudfMetrics,
        key: &CugraphBfsSingleflightKey,
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
        mut singleflight_leader: Option<&mut CugraphBfsLeader>,
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
        if is_bfs_path_output(&self.spec) {
            let metrics = NativeCudfMetrics::new_with_capture(
                &self.metrics,
                partition,
                self.native_execution_config.metrics_capture(),
            );
            let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
            let edge_schema = self.edge_source.schema();
            let (edge_frame, mut source_metrics) =
                execute_edge_source_to_gpu(native_context, Arc::clone(&self.edge_source), Arc::clone(&task_context))
                    .await?;
            let edge_frame =
                attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
            let (predicate_frames, predicate_source_metrics) =
                execute_predicate_sources_to_gpu(native_context, &self.predicate_sources, Arc::clone(&task_context))
                    .await?;
            source_metrics.merge_from(predicate_source_metrics);
            let path_target =
                bfs_path_target_vertex(native_context, &self.spec, &predicate_frames)?.expect("path mode checked");
            let prepared =
                prepare_bfs_logical_gpu_inputs(native_context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
            let (batch, mut query_metrics, path_residency_source) = if let Some(codebook) = prepared.logical_codebook {
                let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
                    native_context,
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
                (batch, query_metrics, path_residency_source)
            } else {
                let (handle, query_metrics) = execute_cugraph_query_to_gpu_with_options(
                    native_context,
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
                (batch, query_metrics, path_residency_source)
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
            query_metrics.merge_from(source_metrics);
            metrics.record_query_metrics(&query_metrics);
            return Ok((self.gpu_result_input_name.clone(), handle));
        }
        let metrics = NativeCudfMetrics::new_with_capture(
            &self.metrics,
            partition,
            self.native_execution_config.metrics_capture(),
        );
        let _elapsed_compute = metrics.baseline.elapsed_compute().timer();
        let edge_schema = self.edge_source.schema();
        let (edge_frame, mut source_metrics) =
            execute_edge_source_to_gpu(native_context, Arc::clone(&self.edge_source), Arc::clone(&task_context))
                .await?;
        let edge_frame =
            attach_graph_residency(edge_frame, self.graph_residency.as_ref(), edge_schema.as_ref(), &self.spec);
        let (predicate_frames, predicate_source_metrics) =
            execute_predicate_sources_to_gpu(native_context, &self.predicate_sources, Arc::clone(&task_context))
                .await?;
        source_metrics.merge_from(predicate_source_metrics);
        let prepared =
            prepare_bfs_logical_gpu_inputs(native_context, &self.spec, edge_schema, edge_frame, predicate_frames)?;
        let (handle, mut query_metrics) = execute_cugraph_query_to_gpu_with_options(
            native_context,
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
            let decoded =
                decode_bfs_logical_gpu_output(native_context, handle, &codebook, Arc::clone(&self.spec.output_schema))?;
            apply_gpu_projection_limit(
                native_context,
                decoded,
                Arc::clone(&self.spec.output_schema),
                self.projection.as_ref(),
                self.limit,
            )?
        } else {
            handle
        };
        query_metrics.merge_from(source_metrics);
        metrics.record_query_metrics(&query_metrics);
        Ok((self.gpu_result_input_name.clone(), handle))
    }
}

impl GpuResidentProducerExec for CugraphAlgoExec {
    fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
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

pub(crate) fn gpu_resident_producer(plan: &dyn ExecutionPlan) -> Option<&dyn GpuResidentProducerExec> {
    plan.downcast_ref::<CugraphAlgoExec>().map(|producer| producer as &dyn GpuResidentProducerExec)
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
        let metrics =
            NativeCudfMetrics::new_with_capture(&self.metrics, partition, default_execution_config.metrics_capture());
        self.validate_standalone_bounded_sources()?;
        let output = futures::stream::once(async move {
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
            let gpu_execution_pool = context_lease.context().gpu_execution_pool();
            run_gpu_execution_task(gpu_execution_pool, "datafusion_nexus.cugraph_sql.gpu_execution_pool", async move {
                let context = context_lease.context();
                let execution_task_context =
                    context_lease.task_context().map(Arc::clone).unwrap_or_else(|| Arc::clone(&task_context));
                let batch =
                    exec.execute_host_result_with_context_impl(context, execution_task_context, &metrics).await?;
                metrics.record_lakehouse_cache_fill_events(&context.drain_lakehouse_cache_fill_event_snapshots());
                Ok::<Option<RecordBatch>, DataFusionError>(Some(batch.record_output(&metrics.baseline)))
            })
            .await
        })
        .try_filter_map(|batch| async move { Ok(batch) });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

fn projected_schema(schema: &SchemaRef, projection: Option<&Vec<usize>>) -> Result<SchemaRef> {
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

fn edge_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NativeCudfExecutionConfig,
) -> NativeCudfExecutionConfig {
    if let Some(native) = as_native_cudf_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = as_native_cudf_composed_exec(plan) {
        return native.execution_config().clone();
    }
    default_execution_config.clone()
}

#[cfg(test)]
fn cugraph_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NativeCudfExecutionConfig,
) -> NativeCudfExecutionConfig {
    let execution_config = edge_execution_config(plan, default_execution_config);
    if let Some(ticket) = default_execution_config.query_admission_ticket() {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else {
        execution_config.with_plan_query_admission_ticket()
    }
}

fn cugraph_execution_config_for_task_context(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NativeCudfExecutionConfig,
    task_context: &TaskContext,
) -> NativeCudfExecutionConfig {
    let execution_config = edge_execution_config(plan, default_execution_config);
    if let Some(ticket) = query_admission_ticket_from_task_context(task_context) {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else if let Some(ticket) = default_execution_config.query_admission_ticket() {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else {
        execution_config.with_plan_query_admission_ticket()
    }
}

fn cugraph_execution_config_with_graph_aware_ticket(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    edge_normalize_inserted: bool,
    placement_hint: QueryPlacementHint,
    execution_config: NativeCudfExecutionConfig,
) -> Result<NativeCudfExecutionConfig> {
    let graph_model_ask =
        if execution_config.requested_grant_bytes().is_some() || execution_config.query_admission_source().is_none() {
            None
        } else {
            cugraph_graph_model_ask_bytes(spec, edge_source, edge_normalize_inserted, &execution_config)?
        };
    execution_config.with_graph_aware_plan_query_admission_ticket(graph_model_ask, placement_hint).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to compute cuGraph SQL graph-aware query admission ticket ask",
            error,
        ))
    })
}

fn cugraph_graph_placement_hint(
    spec: &CugraphAlgorithmSpec,
    graph_residency: Option<&GraphResidencyDescriptor>,
    edge_normalize_inserted: bool,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> QueryPlacementHint {
    let Some(graph_residency) = graph_residency else {
        return QueryPlacementHint::default();
    };
    let native_spec = native_graph_spec_for_placement_hint(spec, edge_normalize_inserted, predicate_sources);
    planned_resident_graph_key(graph_residency, &native_spec)
        .map(QueryPlacementHint::for_resident_graph_key)
        .unwrap_or_default()
}

fn cugraph_graph_model_ask_bytes(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    edge_normalize_inserted: bool,
    execution_config: &NativeCudfExecutionConfig,
) -> Result<Option<GraphAdmissionModelAsk>> {
    let Some(native_spec) = native_graph_spec_for_memory_estimate(spec, edge_normalize_inserted) else {
        return Ok(None);
    };
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return Ok(None);
    }
    let calibration_override = execution_config.graph_memory_calibration_override();
    let raw_input_edges = edge_source_output_rows(edge_source)?.or(calibration_override.raw_input_edges);
    let Some(raw_input_edges) = raw_input_edges else {
        return Ok(None);
    };
    let input_edges = cugraph_effective_input_edges(raw_input_edges, spec, edge_normalize_inserted);
    let device_total_memory_bytes =
        execution_config.query_admission_source().map(|source| source.max_device_total_memory_bytes());
    let estimate = graph_memory_estimate_for_edge_list(
        &native_spec,
        &edge_source.schema(),
        input_edges,
        Some(raw_input_edges),
        calibration_override,
        device_total_memory_bytes,
    )
    .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to estimate cuGraph SQL graph memory from planning metadata",
            error,
        ))
    })?;
    Ok(Some(GraphAdmissionModelAsk { cold_bytes: estimate.requested_bytes, warm_bytes: estimate.execution_bytes }))
}

fn edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Result<Option<u64>> {
    let statistics = edge_source.partition_statistics(None)?;
    if let Precision::Exact(rows) = statistics.num_rows {
        return Ok(u64::try_from(rows).ok());
    }
    Ok(native_edge_source_output_rows(edge_source))
}

fn native_edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return None;
    }
    graph_edge_source_output_rows(edge_source)
}

fn native_edge_source_allows_graph_aware_ticket(edge_source: &dyn ExecutionPlan) -> bool {
    if let Some(native) = as_native_cudf_exec(edge_source) {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    if let Some(native) = as_native_cudf_composed_exec(edge_source) {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    if let Some(native) = edge_source.downcast_ref::<NativeCudfGpuProducerExec>() {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    true
}

fn graph_edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    if let Some(native) = as_native_cudf_exec(edge_source) {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    if let Some(native) = as_native_cudf_composed_exec(edge_source) {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    if let Some(native) = edge_source.downcast_ref::<NativeCudfGpuProducerExec>() {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    None
}

fn cugraph_graph_residency_descriptor(
    spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    edge_normalize_inserted: bool,
) -> Option<GraphResidencyDescriptor> {
    if cugraph_spec_uses_logical_vertex_mapping(spec, edge_source.schema().as_ref()) {
        return None;
    }
    let native_spec = native_graph_spec_for_residency(spec, edge_normalize_inserted);
    cugraph_graph_residency_analysis(Arc::clone(&edge_source), &native_spec).descriptor().cloned()
}

fn cugraph_spec_uses_logical_vertex_mapping(spec: &CugraphAlgorithmSpec, edge_schema: &Schema) -> bool {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return false;
    };
    if params.source.is_utf8() {
        return true;
    }
    if !matches!(params.source, CugraphBfsSource::Relation(_)) {
        return false;
    }
    let Ok(source_field) = edge_schema.field_with_name(&spec.src_col) else {
        return true;
    };
    let Ok(destination_field) = edge_schema.field_with_name(&spec.dst_col) else {
        return true;
    };
    is_bfs_logical_vertex_type(source_field.data_type()) && is_bfs_logical_vertex_type(destination_field.data_type())
}

fn cugraph_graph_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> GraphResidencyPlanAnalysis {
    if let Some(native) = as_native_cudf_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = as_native_cudf_composed_exec(edge_source.as_ref()) {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else if let Some(native) = edge_source.as_ref().downcast_ref::<NativeCudfGpuProducerExec>() {
        graph_residency_descriptor_for_edge_plan(native.native_plan(), native_spec)
    } else {
        cugraph_datafusion_edge_plan_residency_analysis(edge_source, native_spec)
    }
}

fn cugraph_datafusion_edge_plan_residency_analysis(
    edge_source: Arc<dyn ExecutionPlan>,
    native_spec: &GraphAlgorithmSpec,
) -> GraphResidencyPlanAnalysis {
    let lowering = crate::native::lower_physical_plan_to_native(edge_source);
    if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
        return GraphResidencyPlanAnalysis::Bypass(GraphSourceFingerprintBypassReason::NonLocalParquetSource);
    }
    let native_plan = &lowering.plans[0];
    graph_residency_descriptor_for_edge_plan(native_plan, native_spec)
}

fn native_graph_spec_for_residency(spec: &CugraphAlgorithmSpec, edge_normalize_inserted: bool) -> GraphAlgorithmSpec {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    let algorithm = native_algorithm(spec.algorithm);
    GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        edge_columns,
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: if edge_normalize_inserted {
                GraphConstructionPolicy::PythonCugraph
            } else {
                GraphConstructionPolicy::RawLibcugraph
            },
        },
        algorithm.definition().default_params(),
    )
    .with_edge_normalize_inserted(edge_normalize_inserted)
}

fn attach_graph_residency(
    frame: GpuDataFrame,
    graph_residency: Option<&GraphResidencyDescriptor>,
    edge_schema: &Schema,
    spec: &CugraphAlgorithmSpec,
) -> GpuDataFrame {
    match graph_residency {
        Some(graph_residency) if graph_residency_matches_edge_schema(graph_residency, edge_schema, spec) => {
            frame.with_graph_residency(graph_residency.clone())
        },
        None => frame,
        Some(_) => frame,
    }
}

fn graph_residency_matches_edge_schema(
    graph_residency: &GraphResidencyDescriptor,
    edge_schema: &Schema,
    spec: &CugraphAlgorithmSpec,
) -> bool {
    let Ok(source_field) = edge_schema.field_with_name(&spec.src_col) else {
        return false;
    };
    let Ok(destination_field) = edge_schema.field_with_name(&spec.dst_col) else {
        return false;
    };
    if source_field.data_type() != &graph_residency.vertex_data_type
        || destination_field.data_type() != &graph_residency.vertex_data_type
    {
        return false;
    }
    graph_residency_optional_dtype_matches(
        graph_residency.weight_data_type.as_ref(),
        spec.weight_col.as_deref(),
        edge_schema,
    ) && graph_residency_optional_dtype_matches(
        graph_residency.edge_id_data_type.as_ref(),
        spec.edge_id_col.as_deref(),
        edge_schema,
    )
}

fn graph_residency_optional_dtype_matches(
    expected: Option<&DataType>,
    column: Option<&str>,
    edge_schema: &Schema,
) -> bool {
    match expected {
        Some(expected) => column
            .and_then(|column| edge_schema.field_with_name(column).ok())
            .is_some_and(|field| field.data_type() == expected),
        None => true,
    }
}

fn native_plan_is_graph_ticket_light_edge_source(native_plan: &nexus_query_engine::QueryPlan) -> bool {
    native_plan.nodes().iter().all(|node| {
        matches!(node.kind, PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Limit(_))
    })
}

fn cugraph_effective_input_edges(
    raw_input_edges: u64,
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> u64 {
    raw_input_edges.saturating_mul(if edge_normalize_inserted && !spec.graph_options.directed { 2 } else { 1 })
}

fn task_context_with_cugraph_execution_ticket(
    execution_config: &NativeCudfExecutionConfig,
    task_context: Arc<TaskContext>,
) -> Arc<TaskContext> {
    if let Some(ticket) = execution_config.query_admission_ticket() {
        task_context_with_query_admission_ticket(task_context, ticket)
    } else {
        task_context
    }
}

fn validate_standalone_bounded_cugraph_sources(
    config: &NativeCudfExecutionConfig,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> Result<()> {
    if !config.is_standalone_bounded() {
        return Ok(());
    }
    validate_standalone_bounded_source_contains_no_native_child("edge", edge_source)?;
    for predicate in predicate_sources {
        validate_standalone_bounded_source_contains_no_native_child("predicate", predicate.as_ref())?;
    }
    Ok(())
}

fn validate_standalone_bounded_source_contains_no_native_child(
    source_kind: &'static str,
    plan: &dyn ExecutionPlan,
) -> Result<()> {
    if plan_contains_native_cudf(plan) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.admission",
            "standalone bounded cuGraph execution cannot execute native cudf child plans without query-service admission",
        )
        .with_fact("source_kind", source_kind)
        .with_fact("remediation", "use NexusGpuBackend::install_on_with_cugraph_sql or unbounded cuGraph execution")
        .into());
    }
    Ok(())
}

fn plan_contains_native_cudf(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NativeCudfExec>().is_some()
        || plan.downcast_ref::<NativeCudfComposedExec>().is_some()
        || plan.downcast_ref::<NativeCudfGpuProducerExec>().is_some()
        || plan.downcast_ref::<CugraphAlgoExec>().is_some()
        || plan.children().iter().any(|child| plan_contains_native_cudf(child.as_ref()))
}

fn as_native_cudf_exec(plan: &dyn ExecutionPlan) -> Option<&NativeCudfExec> {
    if let Some(native) = plan.downcast_ref::<NativeCudfExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.downcast_ref::<NativeCudfExec>()
}

fn as_native_cudf_composed_exec(plan: &dyn ExecutionPlan) -> Option<&NativeCudfComposedExec> {
    if let Some(native) = plan.downcast_ref::<NativeCudfComposedExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.downcast_ref::<NativeCudfComposedExec>()
}

fn cooperative_exec_children(plan: &dyn ExecutionPlan) -> Option<Vec<&Arc<dyn ExecutionPlan>>> {
    (plan.name() == "CooperativeExec").then(|| plan.children())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CugraphEdgeCanonicalFacts {
    add_reverse_present: bool,
    dedup_present: bool,
}

impl CugraphEdgeCanonicalFacts {
    fn satisfies_python_options(self, spec: &CugraphAlgorithmSpec) -> bool {
        self.dedup_present && spec.weight_col.is_none() && (spec.graph_options.directed || self.add_reverse_present)
    }
}

fn cugraph_edge_normalize_inserted(spec: &CugraphAlgorithmSpec, facts: &CugraphEdgeCanonicalFacts) -> bool {
    spec.graph_options.construction_policy == CugraphGraphConstructionPolicy::PythonCugraph
        && !facts.satisfies_python_options(spec)
}

fn cugraph_edge_canonical_facts(plan: &dyn ExecutionPlan, spec: &CugraphAlgorithmSpec) -> CugraphEdgeCanonicalFacts {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        let input_facts = cugraph_edge_canonical_facts(aggregate.input().as_ref(), spec);
        return aggregate_edge_dedup_facts(aggregate, spec, input_facts);
    }

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        if projection_edge_orientation(projection, spec).is_some() {
            return cugraph_edge_canonical_facts(projection.input().as_ref(), spec);
        }
        return CugraphEdgeCanonicalFacts::default();
    }

    if let Some(union) = plan.downcast_ref::<UnionExec>()
        && union_has_reverse_edge_expansion(union, spec)
    {
        return CugraphEdgeCanonicalFacts { add_reverse_present: true, ..Default::default() };
    }

    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return cugraph_edge_canonical_facts(child.as_ref(), spec);
        }
    }

    CugraphEdgeCanonicalFacts::default()
}

#[cfg(test)]
mod tests {
    use std::{fs::File, sync::Arc};

    use arrow::{
        array::{ArrayRef, Float64Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        common::ScalarValue,
        execution::context::TaskContext,
        logical_expr::Operator,
        physical_expr::expressions::{binary, col, lit},
        physical_plan::{empty::EmptyExec, filter::FilterExec},
        prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    };
    use nexus_query_engine::{
        DeploymentIoCapability, Expr, GdsDriverProbeStatus, KvikioCompatModeStatus, LiteralBatchSource,
        LocalParquetFile, LocalParquetSource, NvidiaFsProbeStatus, ProjectionOutput, QueryService, QueryServiceConfig,
        QueryServiceTenantConfig, SortExpr, SortSpec, TenantId,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;
    use crate::{
        graph_sql::{
            CugraphBfsOutputOptions, CugraphBfsParams, CugraphGraphOptions, CugraphPageRankParams, CugraphSqlConfig,
            CugraphTableReference, CugraphTriangleCountAllParams, result_schema_for_params,
        },
        native_exec::{
            QueryAdmissionSource, task_context_with_query_admission_ticket, wrap_plan_with_query_admission_ticket,
        },
        session::CudfSessionStateBuilderExt,
    };

    fn admission_only_source() -> QueryAdmissionSource {
        admission_source_with_band(1024 * 1024, 1024 * 1024, 1024 * 1024)
    }

    fn admission_source_with_band(min_bytes: u64, want_bytes: u64, capacity_bytes: u64) -> QueryAdmissionSource {
        let tenant_id = TenantId::new("cugraph-test");
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let service = QueryService::new_admission_only(
            QueryServiceConfig {
                device_capacity_bytes: Some(capacity_bytes),
                max_concurrent_queries: 1,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), min_bytes, want_bytes)],
            deployment,
        )
        .expect("admission-only query service should build");
        QueryAdmissionSource::new(service, tenant_id)
    }

    fn edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]))
    }

    fn string_edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]))
    }

    fn memory_edge_source(rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = edge_schema();
        let src = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64))) as ArrayRef;
        let dst = Arc::new(Int64Array::from_iter_values((0..rows).map(|row| row as i64 + 1))) as ArrayRef;
        let weight = Arc::new(Float64Array::from_iter_values((0..rows).map(|_| 1.0))) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![src, dst, weight]).unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap() as Arc<dyn ExecutionPlan>
    }

    fn write_edge_parquet(path: &std::path::Path) {
        let schema = edge_schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])) as ArrayRef,
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            schema,
            Some(WriterProperties::builder().set_max_row_group_row_count(Some(2)).build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_string_edge_parquet(path: &std::path::Path) {
        let schema = string_edge_schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["bob", "carol", "alice"])) as ArrayRef,
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            schema,
            Some(WriterProperties::builder().set_max_row_group_row_count(Some(2)).build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    async fn datafusion_parquet_edge_source(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
        datafusion_parquet_edge_source_sql(path, "SELECT src, dst, weight FROM edges").await
    }

    async fn datafusion_parquet_edge_source_sql(path: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
        let config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet("edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
        ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
    }

    #[test]
    fn edge_source_output_rows_rejects_inexact_datafusion_statistics() {
        let edge_source = memory_edge_source(1_000);
        let schema = edge_source.schema();
        let predicate =
            binary(col("src", &schema).unwrap(), Operator::Gt, lit(ScalarValue::Int64(Some(500))), &schema).unwrap();
        let filter = FilterExec::try_new(predicate, edge_source).unwrap();

        assert_eq!(edge_source_output_rows(&filter).unwrap(), None);
    }

    fn native_edge_source(row_group_rows: Vec<u64>, config: &NativeCudfExecutionConfig) -> Arc<dyn ExecutionPlan> {
        native_edge_source_with_root(row_group_rows, config, |_, source| source)
    }

    fn sorted_native_edge_source(
        row_group_rows: Vec<u64>,
        config: &NativeCudfExecutionConfig,
    ) -> Arc<dyn ExecutionPlan> {
        native_edge_source_with_root(row_group_rows, config, |builder, source| {
            builder.add_sort(
                source,
                SortSpec::new(vec![SortExpr::new(Expr::column("src"), false, false)]),
                edge_schema(),
            )
        })
    }

    fn native_edge_source_with_root(
        row_group_rows: Vec<u64>,
        config: &NativeCudfExecutionConfig,
        root: impl FnOnce(&mut QueryPlanBuilder, nexus_query_engine::NodeId) -> nexus_query_engine::NodeId,
    ) -> Arc<dyn ExecutionPlan> {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(row_group_rows)],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let root = root(&mut builder, source);
        let plan = builder.finish_with_host_sink(root);
        Arc::new(NativeCudfExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
    }

    fn native_graph_residency_edge_source_with_stable_id(
        config: &NativeCudfExecutionConfig,
        stable_id: &str,
    ) -> Arc<dyn ExecutionPlan> {
        let file = LocalParquetFile::new("edges.parquet")
            .with_stable_id(stable_id)
            .with_file_size_bytes(1024)
            .with_row_group_rows(vec![100, 200]);
        let source = LocalParquetSource::from_files(
            vec![file],
            edge_schema(),
            vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projected = builder.add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("src"), Field::new("src", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("dst"), Field::new("dst", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("weight"), Field::new("weight", DataType::Float64, false)),
            ],
        );
        let plan = builder.finish_with_host_sink(projected);
        Arc::new(NativeCudfExec::try_new(plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>
    }

    fn native_graph_residency_edge_source(config: &NativeCudfExecutionConfig) -> Arc<dyn ExecutionPlan> {
        native_graph_residency_edge_source_with_stable_id(config, "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z")
    }

    fn triangle_spec() -> CugraphAlgorithmSpec {
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

    fn pagerank_spec() -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::PageRank(CugraphPageRankParams::default());
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::PageRank,
            table_name: CugraphTableReference::parse("edges").unwrap(),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: Some("weight".to_string()),
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: true,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::PageRank, &params),
            params,
        }
    }

    fn bfs_spec() -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::Bfs(Box::new(CugraphBfsParams {
            source: CugraphBfsSource::scalar_i64(0),
            depth_limit: Some(3),
            predicates: Default::default(),
            output: CugraphBfsOutputOptions { mode: CugraphBfsOutputMode::Normalized, return_target_info: false },
        }));
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::Bfs,
            table_name: CugraphTableReference::parse("edges").unwrap(),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: None,
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: true,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::Bfs, &params),
            params,
        }
    }

    fn bfs_utf8_spec() -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::Bfs(Box::new(CugraphBfsParams {
            source: CugraphBfsSource::scalar_utf8("alice"),
            depth_limit: Some(3),
            predicates: Default::default(),
            output: CugraphBfsOutputOptions { mode: CugraphBfsOutputMode::Normalized, return_target_info: false },
        }));
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::Bfs,
            table_name: CugraphTableReference::parse("edges").unwrap(),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: None,
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: true,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::Bfs, &params),
            params,
        }
    }

    fn singleflight_test_exec_with_options(
        source_vertex: i64,
        depth_limit: Option<i64>,
        construction_policy: CugraphGraphConstructionPolicy,
        graph_stable_id: &str,
        output_mode: CugraphBfsOutputMode,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> CugraphAlgoExec {
        let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1)
            .with_query_admission_source(admission_only_source())
            .with_query_service_resident_graph_cache_max_retained_bytes(Some(4096));
        let edge_source = native_graph_residency_edge_source_with_stable_id(&config, graph_stable_id);
        let mut spec = bfs_spec();
        spec.graph_options.construction_policy = construction_policy;
        {
            let CugraphAlgorithmParams::Bfs(params) = &mut spec.params else { unreachable!("BFS test spec") };
            params.source = CugraphBfsSource::scalar_i64(source_vertex);
            params.depth_limit = depth_limit;
            params.output.mode = output_mode;
        }
        spec.output_schema = result_schema_for_params(spec.algorithm, &spec.params);
        CugraphAlgoExec::try_new(spec, edge_source, vec![], projection, limit, config).unwrap()
    }

    fn singleflight_test_exec(source_vertex: i64, depth_limit: Option<i64>) -> CugraphAlgoExec {
        singleflight_test_exec_with_options(
            source_vertex,
            depth_limit,
            CugraphGraphConstructionPolicy::RawLibcugraph,
            "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
            CugraphBfsOutputMode::Normalized,
            None,
            None,
        )
    }

    #[test]
    fn bfs_singleflight_eligibility_is_exact_and_rejects_bounded_depth() {
        let first = singleflight_test_exec(1, None).bfs_singleflight_key(0).expect("deep scalar BFS should qualify");
        let same = singleflight_test_exec(1, None).bfs_singleflight_key(0).expect("same deep BFS should qualify");
        let other_source =
            singleflight_test_exec(2, None).bfs_singleflight_key(0).expect("other scalar BFS should qualify");
        let normalized_graph = singleflight_test_exec_with_options(
            1,
            None,
            CugraphGraphConstructionPolicy::PythonCugraph,
            "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
            CugraphBfsOutputMode::Normalized,
            None,
            None,
        )
        .bfs_singleflight_key(0)
        .expect("normalized scalar BFS should qualify");
        let other_graph = singleflight_test_exec_with_options(
            1,
            None,
            CugraphGraphConstructionPolicy::RawLibcugraph,
            "local:other-edges.parquet:1024:mtime=2026-07-13T00:00:00Z",
            CugraphBfsOutputMode::Normalized,
            None,
            None,
        )
        .bfs_singleflight_key(0)
        .expect("other resident graph should qualify");
        let raw_output = singleflight_test_exec_with_options(
            1,
            None,
            CugraphGraphConstructionPolicy::RawLibcugraph,
            "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
            CugraphBfsOutputMode::Raw,
            None,
            None,
        )
        .bfs_singleflight_key(0)
        .expect("raw-output scalar BFS should qualify");
        let projected = singleflight_test_exec_with_options(
            1,
            None,
            CugraphGraphConstructionPolicy::RawLibcugraph,
            "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
            CugraphBfsOutputMode::Normalized,
            Some(vec![0, 3]),
            None,
        )
        .bfs_singleflight_key(0)
        .expect("projected scalar BFS should qualify");
        let limited = singleflight_test_exec_with_options(
            1,
            None,
            CugraphGraphConstructionPolicy::RawLibcugraph,
            "local:edges.parquet:1024:mtime=2026-07-12T00:00:00Z",
            CugraphBfsOutputMode::Normalized,
            None,
            Some(17),
        )
        .bfs_singleflight_key(0)
        .expect("limited scalar BFS should qualify");

        assert_eq!(first, same);
        assert_ne!(first, other_source, "independent source semantics must never coalesce");
        assert_ne!(first, singleflight_test_exec(1, None).bfs_singleflight_key(1).unwrap());
        assert_ne!(first, other_graph, "different resident graphs must never coalesce");
        assert_ne!(first, normalized_graph, "raw and Python-normalized graph preparations must never coalesce");
        assert_ne!(first, raw_output, "raw and normalized BFS outputs must never coalesce");
        assert_ne!(first, projected, "different projections must never coalesce");
        assert_ne!(first, limited, "different limits must never coalesce");
        assert!(singleflight_test_exec(1, Some(3)).bfs_singleflight_key(0).is_none());
    }

    #[test]
    fn cugraph_execution_config_creates_plan_ticket_for_backend_admission_source() {
        let source = admission_only_source();
        let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let plan = EmptyExec::new(Arc::new(Schema::empty()));

        assert!(base.has_query_admission_authority());
        assert_eq!(base.query_admission_ticket_id(), None);

        let graph_config = cugraph_execution_config(&plan, &base);
        let ticket_id = graph_config
            .query_admission_ticket_id()
            .expect("cuGraph execution should use a plan-scoped backend admission ticket");
        let cloned = cugraph_execution_config(&plan, &graph_config);

        assert_eq!(cloned.query_admission_ticket_id(), Some(ticket_id));
    }

    #[test]
    fn cugraph_algo_exec_mints_graph_aware_ticket_below_tenant_want() {
        let min = 64 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        let capacity = 4 * 1024 * 1024 * 1024;
        let source = admission_source_with_band(min, want, capacity);
        let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);

        let exec = CugraphAlgoExec::try_new(triangle_spec(), memory_edge_source(300), vec![], None, None, config)
            .expect("cuGraph exec should build with graph-aware admission ticket");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
        let requested = ticket.requested_grant_bytes().expect("known edge rows should produce graph-aware ask");
        let warm_requested = ticket
            .warm_graph_requested_grant_bytes()
            .expect("known cacheable graph should carry an exact-hit warm ask");

        assert!(requested >= min);
        assert!(requested < want, "small graph ask should not default to tenant want");
        assert!(warm_requested < requested, "warm ask must omit graph construction bytes");
    }

    #[test]
    fn cugraph_graph_model_ask_prefers_exact_edge_rows_over_configured_raw_edges() {
        let min = 64 * 1024 * 1024;
        let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1);
        let stale_config = config.clone().with_graph_memory_calibration_override(
            nexus_query_engine::GraphMemoryCalibrationOverride::none().with_raw_input_edges(30_000),
        );
        let edge_source = memory_edge_source(300);

        let exact_ask = cugraph_graph_model_ask_bytes(&triangle_spec(), edge_source.as_ref(), true, &config)
            .unwrap()
            .expect("exact rows should produce graph ask");
        let stale_override_ask =
            cugraph_graph_model_ask_bytes(&triangle_spec(), edge_source.as_ref(), true, &stale_config)
                .unwrap()
                .expect("exact rows should win over stale override");

        assert_eq!(stale_override_ask, exact_ask);
        assert!(exact_ask.warm_bytes < exact_ask.cold_bytes);
    }

    #[test]
    fn cugraph_algo_exec_uses_native_edge_source_rows_for_graph_aware_ticket() {
        let min = 64 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        let capacity = 4 * 1024 * 1024 * 1024;
        let source = admission_source_with_band(min, want, capacity);
        let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1).with_query_admission_source(source);
        let edge_source = native_edge_source(vec![100, 200], &config);

        let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should price native edge-source row evidence");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
        let requested =
            ticket.requested_grant_bytes().expect("native edge-source row evidence should produce graph-aware ask");

        assert!(requested >= min);
        assert!(requested < want, "native edge-source graph ask should not default to tenant want");
    }

    #[test]
    fn cugraph_algo_exec_records_graph_residency_for_native_edge_source() {
        let config = NativeCudfExecutionConfig::unbounded();
        let edge_source = native_graph_residency_edge_source(&config);

        let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for native edge source");
        let descriptor = exec.graph_residency.as_ref().expect("native parquet edge source should be keyable");

        assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
        assert_eq!(descriptor.vertex_data_type, DataType::Int64);
        assert_eq!(
            descriptor.weight_data_type,
            Some(DataType::Float64),
            "Python-compatible edge normalization must keep weight in the source fingerprint even for algorithms \
             that do not consume weights directly",
        );
    }

    #[tokio::test]
    async fn cugraph_algo_exec_records_graph_residency_for_datafusion_parquet_edge_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_edge_parquet(&path);
        let config = NativeCudfExecutionConfig::unbounded();
        let edge_source = datafusion_parquet_edge_source(&path).await;

        let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for DataFusion local-Parquet edge source");
        let descriptor = exec.graph_residency.as_ref().expect("DataFusion local-Parquet edge source should be keyable");

        assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
        assert_eq!(descriptor.vertex_data_type, DataType::Int64);
        assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
    }

    #[tokio::test]
    async fn cugraph_algo_exec_bypasses_graph_residency_for_datafusion_parquet_scan_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_edge_parquet(&path);
        let config = NativeCudfExecutionConfig::unbounded();
        let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst, weight FROM edges LIMIT 2").await;

        let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for scan-limited DataFusion local-Parquet edge source");

        assert!(exec.graph_residency.is_none());
    }

    #[tokio::test]
    async fn cugraph_algo_exec_bypasses_graph_residency_for_logical_bfs_datafusion_parquet_edge_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_string_edge_parquet(&path);
        let config = NativeCudfExecutionConfig::unbounded();
        let edge_source = datafusion_parquet_edge_source_sql(&path, "SELECT src, dst FROM edges").await;

        let exec = CugraphAlgoExec::try_new(bfs_utf8_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph logical BFS exec should build for DataFusion local-Parquet edge source");
        let placement_hint = cugraph_graph_placement_hint(
            &exec.spec,
            exec.graph_residency.as_ref(),
            exec.edge_normalize_inserted(),
            &exec.predicate_sources,
        );

        assert!(exec.graph_residency.is_none());
        assert!(placement_hint.is_empty());
    }

    #[tokio::test]
    async fn cugraph_algo_exec_mints_graph_cache_placement_hint_for_datafusion_parquet_edge_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_edge_parquet(&path);
        let source = admission_only_source();
        let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
        let edge_source = datafusion_parquet_edge_source(&path).await;

        let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for keyable DataFusion edge source");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
        let resident_key = ticket
            .placement_hint()
            .resident_graph_key()
            .expect("keyable DataFusion edge source should carry a graph placement hint");

        assert_eq!(resident_key.orientation, nexus_query_engine::GraphResidencyOrientation::Transposed);
        assert_eq!(resident_key.source_column, "src");
        assert_eq!(resident_key.destination_column, "dst");
        assert_eq!(resident_key.weight_column.as_deref(), Some("weight"));
    }

    #[tokio::test]
    async fn cugraph_algo_exec_mints_bfs_graph_cache_placement_hint_for_datafusion_parquet_edge_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_edge_parquet(&path);
        let source = admission_only_source();
        let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
        let edge_source = datafusion_parquet_edge_source(&path).await;

        let exec = CugraphAlgoExec::try_new(bfs_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph BFS exec should build for keyable DataFusion edge source");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
        let resident_key = ticket
            .placement_hint()
            .resident_graph_key()
            .expect("keyable DataFusion BFS edge source should carry a graph placement hint");

        assert_eq!(resident_key.orientation, nexus_query_engine::GraphResidencyOrientation::Forward);
        assert_eq!(resident_key.source_column, "src");
        assert_eq!(resident_key.destination_column, "dst");
        assert_eq!(resident_key.weight_column, None);
    }

    #[tokio::test]
    async fn cugraph_sql_bfs_table_function_records_graph_residency_for_datafusion_parquet_edge_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.parquet");
        write_edge_parquet(&path);
        let config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
        let state = datafusion::execution::SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_cugraph_sql(CugraphSqlConfig::default())
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet("citation_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

        let plan = ctx
            .sql(
                "SELECT COUNT(*) AS reached \
                 FROM cugraph_bfs('citation_edges', 0, 'src', 'dst', NULL, \
                 '{\"depth_limit\": 3, \"output_mode\": \"normalized\"}')",
            )
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let exec = cugraph_exec_in_plan(Arc::clone(&plan)).expect("planned SQL should contain CugraphAlgoExec");
        let exec = exec.downcast_ref::<CugraphAlgoExec>().unwrap();
        let descriptor =
            exec.graph_residency.as_ref().expect("stress-shaped SQL local-Parquet edge source should be keyable");
        let edge_schema = exec.edge_source.schema();
        let placement_hint = cugraph_graph_placement_hint(
            &exec.spec,
            Some(descriptor),
            exec.edge_normalize_inserted(),
            &exec.predicate_sources,
        );

        assert!(graph_residency_matches_edge_schema(descriptor, edge_schema.as_ref(), &exec.spec));
        assert_eq!(
            placement_hint.resident_graph_key().map(|key| key.orientation),
            Some(nexus_query_engine::GraphResidencyOrientation::Forward)
        );
    }

    fn cugraph_exec_in_plan(plan: Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan.downcast_ref::<CugraphAlgoExec>().is_some() {
            return Some(plan);
        }
        plan.children().into_iter().find_map(|child| cugraph_exec_in_plan(Arc::clone(child)))
    }

    #[test]
    fn graph_residency_schema_guard_rejects_descriptor_dtype_mismatch() {
        let config = NativeCudfExecutionConfig::unbounded();
        let edge_source = native_graph_residency_edge_source(&config);
        let spec = triangle_spec();
        let exec = CugraphAlgoExec::try_new(spec.clone(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for native edge source");
        let descriptor = exec.graph_residency.as_ref().expect("native edge source should be keyable");
        let mismatched_schema = Schema::new(vec![
            Field::new("src", DataType::Int32, false),
            Field::new("dst", DataType::Int32, false),
            Field::new("weight", DataType::Float64, false),
        ]);

        assert!(graph_residency_matches_edge_schema(descriptor, edge_schema().as_ref(), &spec));
        assert!(!graph_residency_matches_edge_schema(descriptor, &mismatched_schema, &spec));
    }

    #[test]
    fn cugraph_algo_exec_mints_graph_cache_placement_hint_for_keyable_native_edge_source() {
        let source = admission_only_source();
        let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);
        let edge_source = native_graph_residency_edge_source(&config);

        let exec = CugraphAlgoExec::try_new(pagerank_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build for keyable native edge source");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");
        let resident_key = ticket
            .placement_hint()
            .resident_graph_key()
            .expect("keyable cached graph query should carry a placement hint");

        assert_eq!(resident_key.orientation, nexus_query_engine::GraphResidencyOrientation::Transposed);
        assert_eq!(resident_key.source_column, "src");
        assert_eq!(resident_key.destination_column, "dst");
        assert_eq!(resident_key.weight_column.as_deref(), Some("weight"));
    }

    #[test]
    fn cugraph_algo_exec_keeps_empty_graph_cache_placement_hint_for_unkeyable_edge_source() {
        let source = admission_only_source();
        let config = NativeCudfExecutionConfig::bounded(1024 * 1024, 512, 1, 3, 1).with_query_admission_source(source);

        let exec = CugraphAlgoExec::try_new(pagerank_spec(), memory_edge_source(100), vec![], None, None, config)
            .expect("cuGraph exec should build for non-native edge source");
        let ticket =
            exec.native_execution_config.query_admission_ticket().expect("cuGraph exec should mint a plan ticket");

        assert!(ticket.placement_hint().is_empty());
    }

    #[test]
    fn cugraph_algo_exec_keeps_default_ticket_for_non_light_native_edge_source() {
        let min = 64 * 1024 * 1024;
        let want = 2 * 1024 * 1024 * 1024;
        let capacity = 4 * 1024 * 1024 * 1024;
        let source = admission_source_with_band(min, want, capacity);
        let config = NativeCudfExecutionConfig::bounded(min, 512, 1, 3, 1)
            .with_query_admission_source(source)
            .with_graph_memory_calibration_override(
                nexus_query_engine::GraphMemoryCalibrationOverride::none().with_raw_input_edges(300),
            );
        let edge_source = sorted_native_edge_source(vec![100, 200], &config);

        let exec = CugraphAlgoExec::try_new(triangle_spec(), edge_source, vec![], None, None, config)
            .expect("cuGraph exec should build with non-light native edge source");
        let ticket = exec
            .native_execution_config
            .query_admission_ticket()
            .expect("cuGraph exec should still mint a plan ticket");

        assert_eq!(
            ticket.requested_grant_bytes(),
            None,
            "non-light native preprocessing should keep tenant-default ask"
        );
    }

    #[test]
    fn cugraph_execution_config_keeps_standalone_configs_ticketless() {
        let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1);
        let plan = EmptyExec::new(Arc::new(Schema::empty()));

        let graph_config = cugraph_execution_config(&plan, &base);

        assert_eq!(graph_config.query_admission_ticket_id(), None);
    }

    #[test]
    fn cugraph_execution_config_recovers_wrapped_native_edge_source_ticket() {
        let source = admission_only_source();
        let default_config =
            NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source.clone());
        let native_config = NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(source);
        let native_config = native_config.with_plan_query_admission_ticket();
        let ticket = native_config.query_admission_ticket().expect("native edge source should have a plan ticket");
        let expected_ticket_id = native_config.query_admission_ticket_id();

        let schema = Arc::new(Schema::empty());
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
        let native_plan = builder.finish_with_host_sink(source);
        let native_exec =
            Arc::new(NativeCudfExec::try_new(native_plan, native_config).unwrap()) as Arc<dyn ExecutionPlan>;
        let wrapped = wrap_plan_with_query_admission_ticket(native_exec, ticket);

        assert!(wrapped.downcast_ref::<NativeCudfExec>().is_some());
        assert_eq!(
            cugraph_execution_config(wrapped.as_ref(), &default_config).query_admission_ticket_id(),
            expected_ticket_id
        );
    }

    #[test]
    fn cugraph_execution_config_prefers_ambient_root_ticket() {
        let source = admission_only_source();
        let base = NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let ambient_config = base.clone().with_plan_query_admission_ticket();
        let local_config = base.with_plan_query_admission_ticket();
        let ambient_ticket =
            ambient_config.query_admission_ticket().expect("ambient config should have a root admission ticket");
        let task_context = task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), ambient_ticket);
        let plan = EmptyExec::new(Arc::new(Schema::empty()));

        assert_ne!(ambient_config.query_admission_ticket_id(), local_config.query_admission_ticket_id());
        assert_eq!(
            cugraph_execution_config_for_task_context(&plan, &local_config, &task_context).query_admission_ticket_id(),
            ambient_config.query_admission_ticket_id()
        );
    }

    #[test]
    fn cugraph_execution_context_stamps_resolved_ticket_before_acquire() {
        let source = admission_only_source();
        let execution_config =
            NativeCudfExecutionConfig::bounded(1024, 512, 1, 3, 1).with_query_admission_source(source);
        let execution_config = execution_config.with_plan_query_admission_ticket();
        let raw_context = Arc::new(TaskContext::default());

        assert!(query_admission_ticket_from_task_context(&raw_context).is_none());

        let stamped_context = task_context_with_cugraph_execution_ticket(&execution_config, Arc::clone(&raw_context));
        let stamped_ticket = query_admission_ticket_from_task_context(&stamped_context)
            .expect("cuGraph execution should stamp its resolved ticket onto the TaskContext");

        assert_eq!(Some(stamped_ticket.id()), execution_config.query_admission_ticket_id());
    }

    #[test]
    fn standalone_bounded_cugraph_rejects_native_child_execution_without_admission() {
        let config = NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1);
        let schema = Arc::new(Schema::empty());
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
        let native_plan = builder.finish_with_host_sink(source);
        let native_exec =
            Arc::new(NativeCudfExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;

        let error = validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
            .expect_err("standalone bounded cuGraph must not execute a native edge child without admission");
        assert!(error.to_string().contains("cannot execute native cudf child plans"), "{error}");

        let repartitioned =
            Arc::new(RepartitionExec::try_new(native_exec, Partitioning::UnknownPartitioning(1)).unwrap())
                as Arc<dyn ExecutionPlan>;
        let error = validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
            .expect_err("fallback collect over native children should be rejected");

        assert!(error.to_string().contains("cannot execute native cudf child plans"), "{error}");
    }

    #[test]
    fn admission_backed_cugraph_allows_native_child_execution() {
        let config =
            NativeCudfExecutionConfig::bounded(2048, 512, 1, 3, 1).with_query_admission_source(admission_only_source());
        let schema = Arc::new(Schema::empty());
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
        let native_plan = builder.finish_with_host_sink(source);
        let native_exec =
            Arc::new(NativeCudfExec::try_new(native_plan, config.clone()).unwrap()) as Arc<dyn ExecutionPlan>;
        let repartitioned =
            Arc::new(RepartitionExec::try_new(Arc::clone(&native_exec), Partitioning::UnknownPartitioning(1)).unwrap())
                as Arc<dyn ExecutionPlan>;

        validate_standalone_bounded_cugraph_sources(&config, native_exec.as_ref(), &[])
            .expect("query-service admission owns the native child ledger");
        validate_standalone_bounded_cugraph_sources(&config, repartitioned.as_ref(), &[])
            .expect("query-service admission owns fallback-collected native children");
    }
}

fn preserves_edge_list_canonical_facts(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<SortExec>().is_some_and(|sort| sort.fetch().is_none())
        || matches!(plan.name(), "CoalesceBatchesExec" | "CoalescePartitionsExec") && plan.fetch().is_none()
}

fn aggregate_edge_dedup_facts(
    aggregate: &AggregateExec,
    spec: &CugraphAlgorithmSpec,
    input_facts: CugraphEdgeCanonicalFacts,
) -> CugraphEdgeCanonicalFacts {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return CugraphEdgeCanonicalFacts::default();
    }
    let group_by = aggregate.group_expr();
    if group_by.has_grouping_set()
        || !group_by.null_expr().is_empty()
        || group_by.groups().len() != 1
        || group_by.groups().first().is_some_and(|group| group.iter().any(|is_null| *is_null))
    {
        return CugraphEdgeCanonicalFacts::default();
    }
    let group_exprs = aggregate.group_expr().expr();
    if group_exprs.len() != 2 {
        return CugraphEdgeCanonicalFacts::default();
    }
    let Some(output_indices) = unique_edge_schema_column_indices(aggregate.schema().as_ref(), spec) else {
        return CugraphEdgeCanonicalFacts::default();
    };
    if output_indices.source >= group_exprs.len() || output_indices.destination >= group_exprs.len() {
        return CugraphEdgeCanonicalFacts::default();
    }
    let Some(input_indices) = unique_edge_schema_column_indices(aggregate.input().schema().as_ref(), spec) else {
        return CugraphEdgeCanonicalFacts::default();
    };
    let Some(source_role) = group_exprs
        .get(output_indices.source)
        .and_then(|(expr, _alias)| edge_column_role(expr.as_ref(), input_indices, spec))
    else {
        return CugraphEdgeCanonicalFacts::default();
    };
    let Some(destination_role) = group_exprs
        .get(output_indices.destination)
        .and_then(|(expr, _alias)| edge_column_role(expr.as_ref(), input_indices, spec))
    else {
        return CugraphEdgeCanonicalFacts::default();
    };
    if source_role == destination_role {
        return CugraphEdgeCanonicalFacts::default();
    }

    CugraphEdgeCanonicalFacts { dedup_present: true, add_reverse_present: input_facts.add_reverse_present }
}

fn union_has_reverse_edge_expansion(union: &UnionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let inputs = union.inputs();
    let [left, right] = inputs.as_slice() else {
        return false;
    };
    let Some(left) = edge_projection_provenance(left.as_ref(), spec) else {
        return false;
    };
    let Some(right) = edge_projection_provenance(right.as_ref(), spec) else {
        return false;
    };
    left.input_identity == right.input_identity
        && matches!(
            (left.orientation, right.orientation),
            (EdgeProjectionOrientation::Forward, EdgeProjectionOrientation::Reverse)
                | (EdgeProjectionOrientation::Reverse, EdgeProjectionOrientation::Forward)
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeProjectionOrientation {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeProjectionProvenance {
    orientation: EdgeProjectionOrientation,
    input_identity: EdgeProjectionInputIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EdgeProjectionInputIdentity {
    FileScan {
        object_store_url: String,
        file_groups: Vec<Vec<String>>,
        projection: String,
        filter: String,
        fetch: Option<usize>,
    },
    MemorySourcePointer(String),
}

fn edge_projection_provenance(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionProvenance> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let orientation = projection_edge_orientation(projection, spec)?;
        let input_identity = edge_projection_input_identity(projection.input().as_ref(), spec)?;
        return Some(EdgeProjectionProvenance { orientation, input_identity });
    }

    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return edge_projection_provenance(child.as_ref(), spec);
        }
    }

    let schema = plan.schema();
    if unique_edge_schema_column_indices(schema.as_ref(), spec).is_some() {
        let input_identity = edge_projection_input_identity(plan, spec)?;
        Some(EdgeProjectionProvenance { orientation: EdgeProjectionOrientation::Forward, input_identity })
    } else {
        None
    }
}

fn edge_projection_input_identity(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionInputIdentity> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        if projection_edge_orientation(projection, spec).is_some() {
            return edge_projection_input_identity(projection.input().as_ref(), spec);
        }
        return None;
    }
    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return edge_projection_input_identity(child.as_ref(), spec);
        }
        return None;
    }
    let source = plan.downcast_ref::<DataSourceExec>()?;
    let data_source = source.data_source();
    let data_source_any = data_source.as_ref() as &dyn std::any::Any;
    if let Some(file_scan) = data_source_any.downcast_ref::<FileScanConfig>() {
        let file_groups = file_scan
            .file_groups
            .iter()
            .map(|group| {
                group
                    .files()
                    .iter()
                    .map(|file| {
                        format!(
                            "{}:{}:{:?}:{:?}",
                            file.object_meta.location, file.object_meta.size, file.range, file.partition_values
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        return Some(EdgeProjectionInputIdentity::FileScan {
            object_store_url: format!("{:?}", file_scan.object_store_url),
            file_groups,
            projection: format!("{:?}", file_scan.file_source.projection()),
            filter: format!("{:?}", file_scan.file_source.filter()),
            fetch: file_scan.limit,
        });
    }
    if data_source_any.downcast_ref::<MemorySourceConfig>().is_some() {
        return Some(EdgeProjectionInputIdentity::MemorySourcePointer(format!("{:p}", Arc::as_ptr(data_source))));
    }
    None
}

fn projection_edge_orientation(
    projection: &ProjectionExec,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionOrientation> {
    let output_indices = unique_edge_schema_column_indices(projection.schema().as_ref(), spec)?;
    let input_indices = unique_edge_schema_column_indices(projection.input().schema().as_ref(), spec)?;
    let source_input =
        projection.expr().get(output_indices.source).and_then(|expr| physical_column(expr.expr.as_ref()))?;
    let destination_input =
        projection.expr().get(output_indices.destination).and_then(|expr| physical_column(expr.expr.as_ref()))?;
    if !projection_preserves_weight(projection, spec) {
        return None;
    }

    if column_matches_edge_role(source_input, input_indices.source, &spec.src_col)
        && column_matches_edge_role(destination_input, input_indices.destination, &spec.dst_col)
    {
        Some(EdgeProjectionOrientation::Forward)
    } else if column_matches_edge_role(source_input, input_indices.destination, &spec.dst_col)
        && column_matches_edge_role(destination_input, input_indices.source, &spec.src_col)
    {
        Some(EdgeProjectionOrientation::Reverse)
    } else {
        None
    }
}

fn projection_preserves_weight(projection: &ProjectionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let Some(weight_col) = spec.weight_col.as_deref() else {
        return true;
    };
    let Some(weight_index) = unique_schema_column_index(projection.schema().as_ref(), weight_col) else {
        return false;
    };
    let Some(input_weight_index) = unique_schema_column_index(projection.input().schema().as_ref(), weight_col) else {
        return false;
    };
    projection
        .expr()
        .get(weight_index)
        .and_then(|expr| physical_column(expr.expr.as_ref()))
        .is_some_and(|column| column_matches_edge_role(column, input_weight_index, weight_col))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EdgeSchemaColumnIndices {
    source: usize,
    destination: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeColumnRole {
    Source,
    Destination,
}

fn unique_edge_schema_column_indices(schema: &Schema, spec: &CugraphAlgorithmSpec) -> Option<EdgeSchemaColumnIndices> {
    let source = unique_schema_column_index(schema, &spec.src_col)?;
    let destination = unique_schema_column_index(schema, &spec.dst_col)?;
    (source != destination).then_some(EdgeSchemaColumnIndices { source, destination })
}

fn unique_schema_column_index(schema: &Schema, column_name: &str) -> Option<usize> {
    let mut matching_indices = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| same_edge_column(field.name(), column_name).then_some(index));
    let index = matching_indices.next()?;
    if matching_indices.next().is_some() {
        return None;
    }
    Some(index)
}

fn edge_column_role(
    expr: &dyn datafusion::physical_expr::PhysicalExpr,
    input_indices: EdgeSchemaColumnIndices,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeColumnRole> {
    let column = physical_column(expr)?;
    if column_matches_edge_role(column, input_indices.source, &spec.src_col) {
        Some(EdgeColumnRole::Source)
    } else if column_matches_edge_role(column, input_indices.destination, &spec.dst_col) {
        Some(EdgeColumnRole::Destination)
    } else {
        None
    }
}

fn column_matches_edge_role(column: &Column, index: usize, name: &str) -> bool {
    column.index() == index && same_edge_column(column.name(), name)
}

fn physical_column(expr: &dyn datafusion::physical_expr::PhysicalExpr) -> Option<&Column> {
    expr.downcast_ref::<Column>()
}

fn same_edge_column(actual: &str, expected: &str) -> bool {
    actual == expected
}

async fn execute_edge_source_to_gpu(
    context: &ExecutionContext,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> Result<(GpuDataFrame, QueryMetrics)> {
    if let Some(native) = as_native_cudf_exec(plan.as_ref()) {
        let native_plan = native.native_plan().clone_with_gpu_resident_sink();
        let result = execute_query_plan(context, &native_plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.edge_source_native",
                "native cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, metrics) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_native",
                "native cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok((handle.into_gpu_frame(), metrics));
    }
    if let Some(native) = as_native_cudf_composed_exec(plan.as_ref()) {
        let native_plan = native.native_plan().clone_with_gpu_resident_sink();
        let mut gpu_inputs = GpuResidentInputMap::new();
        for child in native.gpu_inputs() {
            let Some(producer) = native_gpu_resident_producer(child.as_ref()) else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.edge_source_composed",
                    "native composed cuGraph edge-source child was not a GPU-resident producer",
                )
                .with_fact("child_name", child.name())
                .into());
            };
            let (input_name, handle) =
                producer.execute_gpu_result_with_context(context, 0, Arc::clone(&task_context)).await?;
            gpu_inputs.insert(input_name, handle);
        }
        let result = execute_query_plan_with_gpu_inputs(context, &native_plan, &gpu_inputs).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge-source execution failed",
                error,
            ))
        })?;
        let (output, metrics) = result.into_parts();
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok((handle.into_gpu_frame(), metrics));
    }

    let schema = plan.schema();
    let batches = collect(plan, task_context).await?;
    Ok((record_batches_to_gpu(context, schema, &batches)?, QueryMetrics::default()))
}

struct CugraphGpuFrame {
    schema: SchemaRef,
    frame: GpuDataFrame,
}

async fn execute_predicate_sources_to_gpu(
    context: &ExecutionContext,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    task_context: Arc<TaskContext>,
) -> Result<(Vec<CugraphGpuFrame>, QueryMetrics)> {
    let mut frames = Vec::with_capacity(predicate_sources.len());
    let mut metrics = QueryMetrics::default();
    for source in predicate_sources {
        let schema = source.schema();
        let (frame, source_metrics) =
            execute_edge_source_to_gpu(context, Arc::clone(source), Arc::clone(&task_context)).await?;
        metrics.merge_from(source_metrics);
        frames.push(CugraphGpuFrame { schema, frame });
    }
    Ok((frames, metrics))
}

struct PreparedCugraphInputs {
    execution_spec: CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    logical_codebook: Option<Codebook>,
}

fn prepare_bfs_logical_gpu_inputs(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<PreparedCugraphInputs> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    };
    if !bfs_uses_logical_mapping(spec, params, &edge_schema)? {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    }
    if spec.edge_id_col.is_some() || params.predicates.include_edge_ids.is_some() {
        return Err(plan_error(
            "cuGraph BFS logical vertex-id mapping does not yet support edge_id_col or include_edge_ids predicates",
        ));
    }

    let source_edge_frame = edge_frame.clone();
    let edge_frame = edge_frame.into_frame();
    validate_bfs_logical_edge_schema(spec, &edge_schema)?;
    let mut domain_parts = vec![cudf_series(&edge_frame, &spec.src_col)?, cudf_series(&edge_frame, &spec.dst_col)?];
    let literal_source_series = bfs_literal_string_source_series(context, params)?;
    if let Some(source_series) = &literal_source_series {
        domain_parts.push(source_series.clone());
    }
    if let CugraphBfsSource::Relation(source) = &params.source {
        let source_frame = predicate_frames
            .first()
            .ok_or_else(|| plan_error("BFS source_vertices relation source was not planned"))?;
        domain_parts.push(cudf_series(source_frame.frame.frame(), &source.column)?);
    }
    for (index, frame) in predicate_frames.iter().enumerate() {
        collect_bfs_logical_predicate_domain_parts(params, index, frame, &mut domain_parts)?;
    }
    let domain_refs = domain_parts.iter().collect::<Vec<_>>();
    let domain = Series::concat(&domain_refs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to concatenate cuGraph logical BFS mapping domain on GPU",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to concatenate cuGraph logical BFS mapping domain on GPU",
                error,
            ),
        ))
    })?;
    let codebook = domain.codebook().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to build cuGraph logical BFS codebook on GPU",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to build cuGraph logical BFS codebook on GPU",
                error,
            ),
        ))
    })?;

    let mapped_edge = encode_bfs_logical_edge_frame(spec, edge_frame, &codebook)?;
    let mapped_edge_schema = remap_bfs_logical_schema(&edge_schema, &[&spec.src_col, &spec.dst_col])?;
    let mut mapped_predicates = Vec::with_capacity(predicate_frames.len() + 1);
    if let Some(source_series) = literal_source_series {
        let source_code = codebook
            .encode(&source_series)
            .map_err(cudf_logical_mapping_error)?
            .into_series()
            .copy_into_runtime(context.runtime())
            .map_err(cudf_logical_mapping_error)?;
        let source_frame =
            DataFrame::from_series([("__source_vertex", source_code)]).map_err(cudf_logical_mapping_error)?;
        let source_schema =
            Arc::new(Schema::new(vec![arrow::datatypes::Field::new("__source_vertex", DataType::Int32, false)]));
        mapped_predicates.push(CugraphGpuFrame {
            schema: source_schema,
            frame: retain_mapped_cugraph_frame(
                source_frame,
                &source_edge_frame,
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to retain mapped cuGraph logical source frame residency",
            )?,
        });
    }
    for (index, frame) in predicate_frames.into_iter().enumerate() {
        mapped_predicates.push(encode_bfs_logical_predicate_frame(params, index, frame, &codebook)?);
    }

    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok(PreparedCugraphInputs {
        execution_spec,
        edge_schema: mapped_edge_schema,
        edge_frame: retain_mapped_cugraph_frame(
            mapped_edge,
            &source_edge_frame,
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to retain mapped cuGraph logical edge frame residency",
        )?,
        predicate_frames: mapped_predicates,
        logical_codebook: Some(codebook),
    })
}

fn retain_mapped_cugraph_frame(
    frame: DataFrame,
    source: &GpuDataFrame,
    operation: &'static str,
    message: &'static str,
) -> Result<GpuDataFrame> {
    GpuDataFrame::try_new_with_source_residency(frame, source)
        .map_err(|error| DataFusionError::from(Error::from_native(operation, message, error)))
}

fn validate_bfs_logical_edge_schema(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<()> {
    for column in [&spec.src_col, &spec.dst_col] {
        let field = edge_schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{column}'"))
        })?;
        if !is_bfs_logical_vertex_type(field.data_type()) {
            return Err(plan_error(format!(
                "cuGraph BFS logical vertex mapping requires a string edge column '{column}', found {:?}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

fn bfs_uses_logical_mapping(
    spec: &CugraphAlgorithmSpec,
    params: &super::CugraphBfsParams,
    edge_schema: &SchemaRef,
) -> Result<bool> {
    if params.source.is_utf8() {
        return Ok(true);
    }
    if !matches!(params.source, CugraphBfsSource::Relation(_)) {
        return Ok(false);
    }
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.src_col))
    })?;
    let destination_field = edge_schema.field_with_name(&spec.dst_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.dst_col))
    })?;
    Ok(is_bfs_logical_vertex_type(source_field.data_type())
        && is_bfs_logical_vertex_type(destination_field.data_type()))
}

fn bfs_literal_string_source_series(
    context: &ExecutionContext,
    params: &super::CugraphBfsParams,
) -> Result<Option<Series>> {
    let Some(values) = params.source.literal_values() else {
        return Ok(None);
    };
    if !values.first().is_some_and(CugraphBfsSourceVertex::is_utf8) {
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| match value {
            CugraphBfsSourceVertex::Utf8(value) => Ok(value.as_str()),
            CugraphBfsSourceVertex::Int64(_) => Err(plan_error("string BFS source list contained a numeric value")),
        })
        .collect::<Result<Vec<_>>>()?;
    context.runtime().series(values.as_slice()).map(Some).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to create GPU source-vertices series for cuGraph logical BFS mapping",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to create GPU source-vertices series for cuGraph logical BFS mapping",
                error,
            ),
        ))
    })
}

fn is_bfs_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn collect_bfs_logical_predicate_domain_parts(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: &CugraphGpuFrame,
    domain_parts: &mut Vec<Series>,
) -> Result<()> {
    let source_offset = bfs_planned_source_input_offset(params);
    if index < source_offset {
        return Ok(());
    }
    let index = index - source_offset;
    let mut expected_index = 0;
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        domain_parts.push(cudf_series(frame.frame.frame(), &input.source_col)?);
        domain_parts.push(cudf_series(frame.frame.frame(), &input.destination_col)?);
    }
    Ok(())
}

fn encode_bfs_logical_edge_frame(
    spec: &CugraphAlgorithmSpec,
    frame: DataFrame,
    codebook: &Codebook,
) -> Result<DataFrame> {
    let frame = encode_column(frame, &spec.src_col, codebook)?;
    encode_column(frame, &spec.dst_col, codebook)
}

fn encode_bfs_logical_predicate_frame(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: CugraphGpuFrame,
    codebook: &Codebook,
) -> Result<CugraphGpuFrame> {
    let mut expected_index = 0;
    let source_frame = frame.frame.clone();
    let mut raw = frame.frame.into_frame();
    let mut encoded_columns = Vec::new();
    let source_offset = bfs_planned_source_input_offset(params);
    let is_source_frame = index < source_offset;
    if let CugraphBfsSource::Relation(input) = &params.source
        && is_source_frame
    {
        encoded_columns.push(input.column.as_str());
    }
    if is_source_frame {
        for column in &encoded_columns {
            raw = encode_column(raw, column, codebook)?;
        }
        return Ok(CugraphGpuFrame {
            schema: remap_bfs_logical_schema(&frame.schema, &encoded_columns)?,
            frame: retain_mapped_cugraph_frame(
                raw,
                &source_frame,
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to retain mapped cuGraph logical predicate frame residency",
            )?,
        });
    }
    let index = index.saturating_sub(source_offset);
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        encoded_columns.push(input.source_col.as_str());
        encoded_columns.push(input.destination_col.as_str());
    }
    for column in &encoded_columns {
        raw = encode_column(raw, column, codebook)?;
    }
    Ok(CugraphGpuFrame {
        schema: remap_bfs_logical_schema(&frame.schema, &encoded_columns)?,
        frame: retain_mapped_cugraph_frame(
            raw,
            &source_frame,
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to retain mapped cuGraph logical predicate frame residency",
        )?,
    })
}

fn bfs_planned_source_input_offset(params: &super::CugraphBfsParams) -> usize {
    usize::from(params.source.planned_source_vertices_input())
}

fn encode_column(frame: DataFrame, column: &str, codebook: &Codebook) -> Result<DataFrame> {
    let encoded = codebook.encode(&cudf_series(&frame, column)?).map_err(cudf_logical_mapping_error)?.into_series();
    frame.with_column(column, encoded).map_err(cudf_logical_mapping_error)
}

fn remap_bfs_logical_schema(schema: &SchemaRef, columns: &[&str]) -> Result<SchemaRef> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if columns.iter().any(|column| *column == field.name()) {
                arrow::datatypes::Field::new(field.name(), DataType::Int32, false)
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

fn bfs_numeric_execution_schema(params: &super::CugraphBfsParams) -> Result<SchemaRef> {
    let native_schema = bfs_native_result_schema(params.output.return_target_info);
    let fields: Vec<_> = match params.output.mode {
        CugraphBfsOutputMode::Raw => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
        CugraphBfsOutputMode::Normalized => Arc::new(Schema::new(vec![
            normalized_bfs_base_field(&native_schema, "vertex", false)?,
            normalized_bfs_base_field(&native_schema, "distance", true)?,
            normalized_bfs_base_field(&native_schema, "predecessor", true)?,
            arrow::datatypes::Field::new("reachable", DataType::Boolean, false),
        ]))
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .chain(native_schema.fields().iter().skip(3).map(|field| field.as_ref().clone()))
        .collect(),
        CugraphBfsOutputMode::Path => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

fn bfs_native_result_schema(return_target_info: bool) -> SchemaRef {
    let mut native_params = GraphBfsParams::new(0, None);
    native_params.return_target_info = return_target_info;
    GraphAlgorithm::Bfs.definition().result_schema_for_params(&GraphAlgorithmParams::Bfs(Box::new(native_params)))
}

fn normalized_bfs_base_field(
    schema: &SchemaRef,
    name: &'static str,
    nullable: bool,
) -> Result<arrow::datatypes::Field> {
    let field = schema.field_with_name(name).map_err(|error| {
        DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.bfs_result_schema",
                "native BFS descriptor is missing a raw BFS base field",
            )
            .with_fact("column", name)
            .with_source(error),
        )
    })?;
    Ok(arrow::datatypes::Field::new(field.name(), field.data_type().clone(), nullable))
}

fn decode_bfs_logical_gpu_output(
    context: &ExecutionContext,
    handle: GpuResultHandle,
    codebook: &Codebook,
    output_schema: SchemaRef,
) -> Result<GpuResultHandle> {
    let raw_frame = handle.into_gpu_frame();
    let raw = raw_frame.frame();
    let vertex = decode_code_column(raw, "vertex", codebook)?
        .copy_into_runtime(context.runtime())
        .map_err(cudf_logical_mapping_error)?;
    let distance =
        cudf_series(raw, "distance")?.copy_into_runtime(context.runtime()).map_err(cudf_logical_mapping_error)?;
    let predecessor = decode_bfs_predecessor_code_column(raw, codebook)?
        .copy_into_runtime(context.runtime())
        .map_err(cudf_logical_mapping_error)?;
    let mut columns = vec![("vertex", vertex), ("distance", distance), ("predecessor", predecessor)];
    let raw_column_names = raw.column_names();
    if raw_column_names.contains(&"reachable") {
        let reachable =
            cudf_series(raw, "reachable")?.copy_into_runtime(context.runtime()).map_err(cudf_logical_mapping_error)?;
        columns.push(("reachable", reachable));
    }
    if raw_column_names.contains(&"target_found") {
        let target_found = cudf_series(raw, "target_found")?
            .copy_into_runtime(context.runtime())
            .map_err(cudf_logical_mapping_error)?;
        let target_distance = cudf_series(raw, "target_distance")?
            .copy_into_runtime(context.runtime())
            .map_err(cudf_logical_mapping_error)?;
        columns.push(("target_found", target_found));
        columns.push(("target_distance", target_distance));
    }
    let frame = DataFrame::from_series(columns).map_err(cudf_logical_mapping_error)?;
    let frame = GpuDataFrame::try_new_with_source_residency(frame, &raw_frame).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to retain decoded cuGraph logical BFS output residency",
            error,
        ))
    })?;
    GpuResultHandle::new(context, frame, output_schema).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to bind decoded cuGraph logical BFS output",
            error,
        ))
    })
}

fn decode_code_column(frame: &DataFrame, column: &str, codebook: &Codebook) -> Result<Series> {
    let codes = cudf_series(frame, column)?.astype(cudf::TypeId::Int32).map_err(cudf_logical_mapping_error)?;
    codebook.decode_series(&codes).map_err(cudf_logical_mapping_error)
}

fn decode_bfs_predecessor_code_column(frame: &DataFrame, codebook: &Codebook) -> Result<Series> {
    let codes = cudf_series(frame, "predecessor")?.astype(cudf::TypeId::Int32).map_err(cudf_logical_mapping_error)?;
    let code_frame = DataFrame::from_series([("__predecessor_code", codes)]).map_err(cudf_logical_mapping_error)?;
    let code_frame = code_frame
        .with_columns([(
            "__predecessor_code",
            cudf::ops::when(cudf::ops::col("__predecessor_code").ge(0i32))
                .then(cudf::ops::col("__predecessor_code"))
                .otherwise(cudf::ops::lit_null(cudf::DataType::new(cudf::TypeId::Int32))),
        )])
        .map_err(cudf_logical_mapping_error)?;
    codebook.decode_series(&cudf_series(&code_frame, "__predecessor_code")?).map_err(cudf_logical_mapping_error)
}

fn apply_gpu_projection_limit(
    context: &ExecutionContext,
    handle: GpuResultHandle,
    schema: SchemaRef,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<GpuResultHandle> {
    if projection.is_none() && limit.is_none() {
        return Ok(handle);
    }
    let input_name = "__datafusion_nexus_cugraph_logical_output";
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);
    let mut builder = QueryPlanBuilder::new();
    let mut root = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
    let input_schema = Arc::clone(&schema);
    let mut root_schema = schema;
    if let Some(projection) = projection {
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    input_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root_schema = projected_schema(&input_schema, Some(projection))?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    let plan = builder.finish_gpu_resident(root);
    let result = execute_query_plan_with_gpu_inputs(context, &plan, &inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to apply GPU projection/limit to decoded cuGraph logical BFS output",
            error,
        ))
    })?;
    let (output, _) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        return Err(plan_error("decoded cuGraph logical BFS projection did not produce a GPU output"));
    };
    Ok(handle)
}

fn project_and_limit_host_batch(
    batch: RecordBatch,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let mut batch = if let Some(limit) = limit { batch.slice(0, limit.min(batch.num_rows())) } else { batch };
    if let Some(projection) = projection {
        batch = batch.project(projection).map_err(DataFusionError::from)?;
    }
    Ok(batch)
}

fn is_bfs_path_output(spec: &CugraphAlgorithmSpec) -> bool {
    matches!(
        &spec.params,
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
    )
}

fn bfs_decoded_output_schema_for_host(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Arc::clone(&spec.output_schema);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Arc::clone(&spec.output_schema);
    }
    Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("vertex", DataType::Utf8, false),
        arrow::datatypes::Field::new("distance", DataType::Int64, false),
        arrow::datatypes::Field::new("predecessor", DataType::Utf8, true),
    ]))
}

#[derive(Debug, Clone, Eq)]
enum BfsPathVertex {
    Int64(i64),
    Utf8(String),
}

impl PartialEq for BfsPathVertex {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int64(left), Self::Int64(right)) => left == right,
            (Self::Utf8(left), Self::Utf8(right)) => left == right,
            _ => false,
        }
    }
}

impl Hash for BfsPathVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Int64(value) => {
                0_u8.hash(state);
                value.hash(state);
            },
            Self::Utf8(value) => {
                1_u8.hash(state);
                value.hash(state);
            },
        }
    }
}

#[derive(Debug, Clone)]
struct BfsPathNode {
    distance: Option<i64>,
    predecessor: Option<BfsPathVertex>,
}

fn bfs_path_target_vertex(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    predicate_frames: &[CugraphGpuFrame],
) -> Result<Option<BfsPathVertex>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Ok(None);
    }
    let target =
        params.predicates.target_vertices.as_ref().ok_or_else(|| {
            plan_error("BFS path output requires target_vertices_table and target_vertex_col options")
        })?;
    let mut target_index = usize::from(params.source.planned_source_vertices_input());
    if params.predicates.include_vertices.is_some() {
        target_index += 1;
    }
    if params.predicates.exclude_vertices.is_some() {
        target_index += 1;
    }
    let frame = predicate_frames
        .get(target_index)
        .ok_or_else(|| plan_error("BFS path output target predicate source was not planned"))?;
    let handle = GpuResultHandle::new(context, frame.frame.clone(), Arc::clone(&frame.schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.path_output",
            "failed to bind BFS target predicate source for path reconstruction",
            error,
        ))
    })?;
    let batch = handle.to_host_record_batch().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.path_output",
            "failed to materialize BFS target predicate source for path reconstruction",
            error,
        ))
    })?;
    if batch.num_rows() != 1 {
        return Err(plan_error(format!(
            "BFS path output requires exactly one target vertex row, found {}",
            batch.num_rows()
        )));
    }
    let column_index = batch.schema().index_of(&target.column).map_err(DataFusionError::from)?;
    vertex_value(batch.column(column_index).as_ref(), 0).map(Some)
}

fn reconstruct_bfs_path_batch(
    spec: &CugraphAlgorithmSpec,
    batch: &RecordBatch,
    target: BfsPathVertex,
) -> Result<RecordBatch> {
    let CugraphAlgorithmParams::Bfs(_params) = &spec.params else {
        return Err(plan_error("BFS path reconstruction received non-BFS parameters"));
    };
    let vertex_col = batch.schema().index_of("vertex").map_err(DataFusionError::from)?;
    let distance_col = batch.schema().index_of("distance").map_err(DataFusionError::from)?;
    let predecessor_col = batch.schema().index_of("predecessor").map_err(DataFusionError::from)?;
    let mut nodes = HashMap::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let vertex = vertex_value(batch.column(vertex_col).as_ref(), row)?;
        let distance = optional_i64_value(batch.column(distance_col).as_ref(), row)?;
        let predecessor = optional_vertex_value(batch.column(predecessor_col).as_ref(), row)?;
        nodes.insert(vertex, BfsPathNode { distance, predecessor });
    }
    let sources = bfs_path_source_vertices(&nodes);
    let (source, path) = bfs_path_vertices(&nodes, &sources, target.clone())?;
    let source = source.unwrap_or_else(|| target.clone());
    let distances = path
        .iter()
        .enumerate()
        .map(|(index, vertex)| nodes.get(vertex).and_then(|node| node.distance).unwrap_or(index as i64))
        .collect::<Vec<_>>();
    build_bfs_path_batch(Arc::clone(&spec.output_schema), source, target, path, distances)
}

fn bfs_path_source_vertices(nodes: &HashMap<BfsPathVertex, BfsPathNode>) -> HashSet<BfsPathVertex> {
    nodes
        .iter()
        .filter_map(|(vertex, node)| (node.distance == Some(0) && node.predecessor.is_none()).then_some(vertex.clone()))
        .collect()
}

fn bfs_path_vertices(
    nodes: &HashMap<BfsPathVertex, BfsPathNode>,
    sources: &HashSet<BfsPathVertex>,
    target: BfsPathVertex,
) -> Result<(Option<BfsPathVertex>, Vec<BfsPathVertex>)> {
    let Some(target_node) = nodes.get(&target) else {
        return Ok((None, Vec::new()));
    };
    if target_node.distance.is_none() {
        return Ok((None, Vec::new()));
    }
    let mut reversed = Vec::new();
    let mut current = target;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return Err(plan_error("BFS path reconstruction encountered a predecessor cycle"));
        }
        reversed.push(current.clone());
        if sources.contains(&current) {
            let source = current.clone();
            reversed.reverse();
            return Ok((Some(source), reversed));
        }
        let node = nodes.get(&current).ok_or_else(|| {
            plan_error("BFS path reconstruction found a predecessor vertex missing from the BFS output")
        })?;
        let predecessor = node.predecessor.clone().ok_or_else(|| {
            plan_error("BFS path reconstruction reached a vertex without a predecessor before any source")
        })?;
        current = predecessor;
    }
}

fn build_bfs_path_batch(
    schema: SchemaRef,
    source: BfsPathVertex,
    target: BfsPathVertex,
    path: Vec<BfsPathVertex>,
    distances: Vec<i64>,
) -> Result<RecordBatch> {
    let path_index =
        Arc::new(Int64Array::from((0..path.len()).map(|value| value as i64).collect::<Vec<_>>())) as ArrayRef;
    let distance = Arc::new(Int64Array::from(distances)) as ArrayRef;
    let (source_column, target_column, vertex_column): (ArrayRef, ArrayRef, ArrayRef) = match (&source, &target) {
        (BfsPathVertex::Int64(source), BfsPathVertex::Int64(target)) => (
            Arc::new(Int64Array::from(vec![*source; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(vec![*target; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Int64(value) => *value,
                        BfsPathVertex::Utf8(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (BfsPathVertex::Utf8(source), BfsPathVertex::Utf8(target)) => (
            Arc::new(StringArray::from(vec![source.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(vec![target.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Utf8(value) => value.clone(),
                        BfsPathVertex::Int64(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        _ => {
            return Err(plan_error("BFS path output source and target vertex dtypes must match"));
        },
    };
    RecordBatch::try_new(schema, vec![path_index, source_column, target_column, vertex_column, distance])
        .map_err(DataFusionError::from)
}

fn vertex_value(array: &dyn Array, row: usize) -> Result<BfsPathVertex> {
    if array.is_null(row) {
        return Err(plan_error("BFS path vertex values must be non-null"));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(BfsPathVertex::Int64(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(BfsPathVertex::Int64(i64::from(array.value(row))));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    Err(plan_error(format!("BFS path reconstruction does not support vertex array type {:?}", array.data_type())))
}

fn optional_vertex_value(array: &dyn Array, row: usize) -> Result<Option<BfsPathVertex>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let value = vertex_value(array, row)?;
    if value == BfsPathVertex::Int64(-1) {
        return Ok(None);
    }
    Ok(Some(value))
}

fn optional_i64_value(array: &dyn Array, row: usize) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        let value = array.value(row);
        if value == i64::MAX {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        let value = array.value(row);
        if value == i32::MAX {
            return Ok(None);
        }
        return Ok(Some(i64::from(value)));
    }
    Err(plan_error(format!("BFS path reconstruction does not support distance array type {:?}", array.data_type())))
}

fn cudf_logical_mapping_error(error: cudf::Error) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.cugraph_sql.logical_mapping",
        "cuGraph logical BFS GPU mapping failed",
        nexus_query_engine::QueryError::from_cudf(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "cuGraph logical BFS GPU mapping failed",
            error,
        ),
    )
    .into()
}

fn cudf_series(frame: &DataFrame, column: &str) -> Result<Series> {
    frame.series(column).map_err(cudf_logical_mapping_error)
}

fn record_batches_to_gpu(
    context: &ExecutionContext,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<GpuDataFrame> {
    let Some(first_batch) = batches.first() else {
        let arrays =
            schema.fields().iter().map(|field| arrow::array::new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema, arrays).map_err(DataFusionError::from)?;
        return cudf::DataFrame::from_record_batch(context.runtime(), &batch).map(GpuDataFrame::new).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.edge_source_host_import",
                "failed to import empty host Arrow edge-source batch into cudf",
                nexus_query_engine::QueryError::from_cudf(
                    "datafusion_nexus.cugraph_sql.edge_source_host_import",
                    "failed to import empty host Arrow edge-source batch into cudf",
                    error,
                ),
            ))
        });
    };
    let imported = cudf::DataFrame::from_record_batches(context.runtime(), batches).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "failed to import host Arrow edge-source batches into cudf",
            nexus_query_engine::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.edge_source_host_import",
                "failed to import host Arrow edge-source batches into cudf",
                error,
            ),
        ))
    })?;
    if imported.num_columns() != first_batch.num_columns() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "cuGraph edge-source import changed column count",
        )
        .with_fact("expected_column_count", first_batch.num_columns())
        .with_fact("actual_column_count", imported.num_columns())
        .into());
    }
    Ok(GpuDataFrame::new(imported))
}

fn execute_cugraph_query_to_host(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    singleflight_leader: Option<&mut CugraphBfsLeader>,
) -> Result<(RecordBatch, QueryMetrics)> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::HostArrow,
        edge_normalize_inserted,
    )?;
    if let Some(leader) = singleflight_leader {
        leader.record_traversal_submission();
    }
    let result = execute_query_plan_with_gpu_inputs(&graph_context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.execute",
            "native cuGraph SQL query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::HostArrow(batch) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute",
            "native cuGraph SQL query did not produce host Arrow output",
        )
        .into());
    };
    Ok((batch, metrics))
}

fn execute_cugraph_query_to_gpu_with_options(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    singleflight_leader: Option<&mut CugraphBfsLeader>,
) -> Result<(GpuResultHandle, QueryMetrics)> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::GpuResident,
        edge_normalize_inserted,
    )?;
    if let Some(leader) = singleflight_leader {
        leader.record_traversal_submission();
    }
    let result = execute_query_plan_with_gpu_inputs(&graph_context, &plan.plan, &plan.inputs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.execute_gpu",
            "native cuGraph SQL GPU-resident query execution failed",
            error,
        ))
    })?;
    let (output, metrics) = result.into_parts();
    let QueryOutput::GpuResident(handle) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute_gpu",
            "native cuGraph SQL query did not produce GPU-resident output",
        )
        .with_fact("result_sink", "host_arrow")
        .into());
    };
    Ok((handle, metrics))
}

fn graph_algorithm_execution_context(context: &ExecutionContext) -> Result<ExecutionContext> {
    // Graph algorithms remain outside strict bounded admission, but bounded
    // service contexts keep their memory policy so graph workspace-heavy
    // algorithms can price/admit their extra workspace before entering cuGraph.
    Ok(context.clone())
}

#[derive(Debug, Clone, Copy)]
enum CugraphQuerySink {
    HostArrow,
    GpuResident,
}

struct CugraphQueryPlan {
    plan: nexus_query_engine::QueryPlan,
    inputs: GpuResidentInputMap,
}

fn cugraph_query_plan(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    sink: CugraphQuerySink,
    edge_normalize_inserted: bool,
) -> Result<CugraphQueryPlan> {
    let input_name = "__datafusion_nexus_cugraph_edges";
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);

    let mut builder = QueryPlanBuilder::new();
    let input =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&edge_schema))));
    let predicate_nodes = bind_predicate_gpu_inputs(context, &mut builder, &mut inputs, predicate_frames)?;
    let graph_spec = native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes);
    let mut root_schema = Arc::clone(&graph_spec.output_schema);
    let mut root = builder.try_add_graph_algorithm(input, graph_spec).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.plan",
            "failed to build native cuGraph query plan",
            error,
        ))
    })?;
    if let Some(outputs) = bfs_normalized_output_projection(spec, &edge_schema)? {
        root = builder.add_projection(root, outputs);
        root_schema = Arc::clone(&spec.output_schema);
    }
    if let Some(projection) = projection {
        root_schema = projected_schema(&root_schema, Some(projection))?;
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    spec.output_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    let plan = match sink {
        CugraphQuerySink::HostArrow => builder.finish_with_host_sink_schema(root, root_schema),
        CugraphQuerySink::GpuResident => builder.finish_gpu_resident(root),
    };
    Ok(CugraphQueryPlan { plan, inputs })
}

fn bind_predicate_gpu_inputs(
    context: &ExecutionContext,
    builder: &mut QueryPlanBuilder,
    inputs: &mut GpuResidentInputMap,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<Vec<nexus_query_engine::NodeId>> {
    let mut nodes = Vec::with_capacity(predicate_frames.len());
    for (index, CugraphGpuFrame { schema, frame }) in predicate_frames.into_iter().enumerate() {
        let input_name = format!("__datafusion_nexus_cugraph_side_input_{index}");
        let handle = GpuResultHandle::new(context, frame, Arc::clone(&schema)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.gpu_input",
                "failed to bind cuGraph side-input source as a GPU-resident native input",
                error,
            ))
        })?;
        inputs.insert(input_name.clone(), handle);
        nodes.push(builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema))));
    }
    Ok(nodes)
}

fn native_graph_spec(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphAlgorithmSpec {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    let algorithm = native_algorithm(spec.algorithm);
    let params = native_algorithm_params(&spec.params, predicate_nodes);
    let output_schema = algorithm.definition().result_schema_for_params(&params);
    GraphAlgorithmSpec::new(
        algorithm,
        edge_columns,
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: if edge_normalize_inserted {
                GraphConstructionPolicy::PythonCugraph
            } else {
                match spec.graph_options.construction_policy {
                    CugraphGraphConstructionPolicy::PythonCugraph | CugraphGraphConstructionPolicy::RawLibcugraph => {
                        GraphConstructionPolicy::RawLibcugraph
                    },
                }
            },
        },
        params,
        output_schema,
    )
}

fn native_graph_spec_for_placement_hint(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> GraphAlgorithmSpec {
    let mut builder = QueryPlanBuilder::new();
    let predicate_nodes = predicate_sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
                format!("__datafusion_nexus_cugraph_placement_hint_{index}"),
                source.schema(),
            )))
        })
        .collect::<Vec<_>>();
    native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes)
}

fn native_graph_spec_for_memory_estimate(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> Option<GraphAlgorithmSpec> {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    let algorithm = native_algorithm(spec.algorithm);
    let params = native_algorithm_params_for_memory_estimate(&spec.params)?;
    let output_schema = algorithm.definition().result_schema_for_params(&params);
    Some(
        GraphAlgorithmSpec::new(
            algorithm,
            edge_columns,
            GraphConstructionOptions {
                directed: spec.graph_options.directed,
                renumber: spec.graph_options.renumber,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            params,
            output_schema,
        )
        .with_edge_normalize_inserted(edge_normalize_inserted),
    )
}

fn native_algorithm_params_for_memory_estimate(params: &CugraphAlgorithmParams) -> Option<GraphAlgorithmParams> {
    match params {
        // These variants require side-input node ids in the native IR. Fall back
        // to tenant-default admission instead of minting placeholder ids.
        CugraphAlgorithmParams::PersonalizedPageRank(_) => None,
        CugraphAlgorithmParams::Bfs(_) => None,
        _ => Some(native_algorithm_params(params, &[])),
    }
}

fn bfs_normalized_output_projection(
    spec: &CugraphAlgorithmSpec,
    edge_schema: &SchemaRef,
) -> Result<Option<Vec<ProjectionOutput>>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Normalized {
        return Ok(None);
    }

    let unreachable_sentinel = bfs_unreachable_distance_sentinel(spec, edge_schema)?;
    let distance_is_unreachable = NativeExpr::binary(
        NativeExpr::column("distance"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
    );
    let predecessor_is_absent = NativeExpr::binary(
        NativeExpr::column("predecessor"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(-1)),
    );

    let fields = spec.output_schema.fields();
    let mut outputs = vec![
        ProjectionOutput::new(NativeExpr::column("vertex"), fields[0].as_ref().clone()),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(
                    distance_is_unreachable.clone(),
                    NativeExpr::literal(LiteralValue::Null(DataType::Int64)),
                )],
                NativeExpr::column("distance"),
            ),
            fields[1].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(predecessor_is_absent, NativeExpr::literal(LiteralValue::Null(DataType::Int64)))],
                NativeExpr::column("predecessor"),
            ),
            fields[2].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::binary(
                NativeExpr::column("distance"),
                BinaryOp::NotEq,
                NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
            ),
            fields[3].as_ref().clone(),
        ),
    ];
    if params.output.return_target_info {
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_found"), fields[4].as_ref().clone()));
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_distance"), fields[5].as_ref().clone()));
    }
    Ok(Some(outputs))
}

fn bfs_unreachable_distance_sentinel(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<i64> {
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!(
            "cuGraph BFS output normalization could not find source column '{}' in edge source schema",
            spec.src_col
        ))
    })?;
    match source_field.data_type() {
        DataType::Int32 => Ok(i64::from(i32::MAX)),
        DataType::Int64 => Ok(i64::MAX),
        other => Err(plan_error(format!(
            "cuGraph BFS output normalization requires Int32 or Int64 source vertices, found {other:?}"
        ))),
    }
}

fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    algorithm.native_algorithm()
}

fn native_algorithm_params(
    params: &CugraphAlgorithmParams,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphAlgorithmParams {
    match params {
        CugraphAlgorithmParams::PageRank(params) => GraphAlgorithmParams::PageRank(GraphPageRankParams {
            alpha: params.alpha,
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
        }),
        CugraphAlgorithmParams::PersonalizedPageRank(params) => {
            let personalization_node = predicate_nodes.first().copied().expect("personalization node missing");
            GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
                pagerank: GraphPageRankParams {
                    alpha: params.pagerank.alpha,
                    epsilon: params.pagerank.epsilon,
                    max_iterations: params.pagerank.max_iterations,
                },
                personalization: NativeGraphPageRankPersonalizationInput::new(
                    personalization_node,
                    params.personalization.vertex_col.clone(),
                    params.personalization.value_col.clone(),
                ),
            })
        },
        CugraphAlgorithmParams::Bfs(params) => {
            GraphAlgorithmParams::Bfs(Box::new(native_bfs_params(params, predicate_nodes)))
        },
        CugraphAlgorithmParams::Louvain(params) => GraphAlgorithmParams::Louvain(GraphLouvainParams {
            max_level: params.max_level,
            threshold: params.threshold,
            resolution: params.resolution,
        }),
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams {
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams {
                alpha: params.alpha,
                beta: params.beta,
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::Hits(params) => GraphAlgorithmParams::Hits(GraphHitsParams {
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
            normalize: params.normalize,
        }),
        CugraphAlgorithmParams::BetweennessCentrality(params) => {
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                seeds: params.seeds.as_ref().map(|seeds| GraphBetweennessSeeds::I64(seeds.vertices.clone())),
                k: params.k,
                normalized: params.normalized,
                include_endpoints: params.include_endpoints,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                seeds: params.seeds.as_ref().map(|seeds| GraphBetweennessSeeds::I64(seeds.vertices.clone())),
                k: params.k,
                normalized: params.normalized,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::Sssp(params) => {
            GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: params.source_vertex, cutoff: params.cutoff })
        },
        CugraphAlgorithmParams::DegreesAll(_) => GraphAlgorithmParams::DegreesAll(GraphDegreesAllParams),
        CugraphAlgorithmParams::InDegreesAll(_) => GraphAlgorithmParams::InDegreesAll(GraphInDegreesAllParams),
        CugraphAlgorithmParams::OutDegreesAll(_) => GraphAlgorithmParams::OutDegreesAll(GraphOutDegreesAllParams),
        CugraphAlgorithmParams::WeaklyConnectedComponents(_) => {
            GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams)
        },
        CugraphAlgorithmParams::StronglyConnectedComponents(_) => {
            GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams)
        },
        CugraphAlgorithmParams::TriangleCountAll(_) => {
            GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams)
        },
        CugraphAlgorithmParams::CoreNumber(params) => GraphAlgorithmParams::CoreNumber(GraphCoreNumberParams {
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::KCore(params) => GraphAlgorithmParams::KCore(GraphKCoreParams {
            k: params.k,
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::MinimumSpanningTree(_) => GraphAlgorithmParams::MinimumSpanningTree(GraphMstParams),
        CugraphAlgorithmParams::Leiden(params) => GraphAlgorithmParams::Leiden(GraphLeidenParams {
            max_level: params.max_level,
            resolution: params.resolution,
            theta: params.theta,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::Ecg(params) => GraphAlgorithmParams::Ecg(GraphEcgParams {
            min_weight: params.min_weight,
            ensemble_size: params.ensemble_size,
            max_level: params.max_level,
            threshold: params.threshold,
            resolution: params.resolution,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams {
                n_clusters: params.n_clusters,
                n_eigenvectors: params.n_eigenvectors,
                evs_tolerance: params.evs_tolerance,
                evs_max_iterations: params.evs_max_iterations,
                k_means_tolerance: params.k_means_tolerance,
                k_means_max_iterations: params.k_means_max_iterations,
                seed: params.seed,
            })
        },
        CugraphAlgorithmParams::BalancedCutClustering(params) => {
            GraphAlgorithmParams::BalancedCutClustering(GraphSpectralClusteringParams {
                n_clusters: params.n_clusters,
                n_eigenvectors: params.n_eigenvectors,
                evs_tolerance: params.evs_tolerance,
                evs_max_iterations: params.evs_max_iterations,
                k_means_tolerance: params.k_means_tolerance,
                k_means_max_iterations: params.k_means_max_iterations,
                seed: params.seed,
            })
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params {
            max_iter: params.max_iter,
            outbound_attraction_distribution: params.outbound_attraction_distribution,
            lin_log_mode: params.lin_log_mode,
            prevent_overlapping: params.prevent_overlapping,
            overlap_scaling_ratio: params.overlap_scaling_ratio,
            edge_weight_influence: params.edge_weight_influence,
            jitter_tolerance: params.jitter_tolerance,
            barnes_hut_optimize: params.barnes_hut_optimize,
            barnes_hut_theta: params.barnes_hut_theta,
            scaling_ratio: params.scaling_ratio,
            strong_gravity_mode: params.strong_gravity_mode,
            gravity: params.gravity,
            verbose: params.verbose,
            seed: params.seed,
        }),
    }
}

fn native_bfs_params(
    params: &super::CugraphBfsParams,
    predicate_nodes: &[nexus_query_engine::NodeId],
) -> GraphBfsParams {
    let mut native = GraphBfsParams::new(params.source.as_i64().unwrap_or(0), params.depth_limit);
    let mut next = predicate_nodes.iter().copied();
    if params.source.requires_source_vertices_input() {
        native.source_vertices = Some(NativeGraphPredicateInput::new(
            next.next().expect("BFS source-vertices node missing"),
            params.source.source_vertices_column(),
        ));
    }
    native.predicates = NativeGraphBfsPredicateInputs {
        include_vertices: params.predicates.include_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        exclude_vertices: params.predicates.exclude_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        target_vertices: params.predicates.target_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edge_ids: params.predicates.include_edge_ids.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edges: params.predicates.include_edges.as_ref().map(|input| {
            NativeGraphEdgePredicateInput::new(
                next.next().expect("predicate node missing"),
                input.source_col.clone(),
                input.destination_col.clone(),
                input.edge_id_col.clone(),
            )
        }),
    };
    native.return_target_info = params.output.return_target_info;
    if params.output.mode == CugraphBfsOutputMode::Path {
        native.endpoint_policy = GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
    }
    native
}

fn sql_degree_type_to_native(degree_type: CugraphKCoreDegreeType) -> GraphKCoreDegreeType {
    match degree_type {
        CugraphKCoreDegreeType::In => GraphKCoreDegreeType::In,
        CugraphKCoreDegreeType::Out => GraphKCoreDegreeType::Out,
        CugraphKCoreDegreeType::InOut => GraphKCoreDegreeType::InOut,
    }
}
