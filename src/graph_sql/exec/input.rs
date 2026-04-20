use super::*;

pub(super) fn prepare_cugraph_source_metadata<'a>(
    plan: Arc<dyn ExecutionPlan>,
    task_context: &'a TaskContext,
) -> BoxFuture<'a, Arc<dyn ExecutionPlan>> {
    Box::pin(async move {
        if let Some(native) = plan.downcast_ref::<NativeCudfExec>() {
            return Arc::new(native.prepare_lakehouse_metadata(task_context).await) as Arc<dyn ExecutionPlan>;
        }
        if let Some(native) = plan.downcast_ref::<NativeCudfComposedExec>() {
            return Arc::new(native.prepare_lakehouse_metadata(task_context).await) as Arc<dyn ExecutionPlan>;
        }

        let children = plan.children();
        if children.is_empty() {
            return plan;
        }
        let mut prepared_children = Vec::with_capacity(children.len());
        let mut changed = false;
        for child in children {
            let prepared = prepare_cugraph_source_metadata(Arc::clone(child), task_context).await;
            changed |= !Arc::ptr_eq(child, &prepared);
            prepared_children.push(prepared);
        }
        if !changed {
            return plan;
        }
        let original = Arc::clone(&plan);
        plan.with_new_children(prepared_children).unwrap_or(original)
    })
}

pub(super) async fn execute_edge_source_to_gpu(
    context: &ExecutionContext,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> Result<(GpuDataFrame, QueryMetrics)> {
    if let Some(native) = as_native_cudf_exec(plan.as_ref()) {
        let native_plan = crate::native_exec::lakehouse::resolve_query_plan(
            native.execution_config(),
            native.native_plan().clone(),
            task_context.as_ref(),
        )
        .await
        .clone_with_gpu_resident_sink();
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
        let native_plan = crate::native_exec::lakehouse::resolve_query_plan(
            native.execution_config(),
            native.native_plan().clone(),
            task_context.as_ref(),
        )
        .await
        .clone_with_gpu_resident_sink();
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

pub(super) struct CugraphGpuFrame {
    pub(super) schema: SchemaRef,
    pub(super) frame: GpuDataFrame,
}

pub(super) async fn execute_predicate_sources_to_gpu(
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

pub(super) fn cudf_logical_mapping_error(error: cudf::Error) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.cugraph_sql.logical_mapping",
        "cuGraph logical BFS GPU mapping failed",
        nexus_query_engine::error::QueryError::from_cudf(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "cuGraph logical BFS GPU mapping failed",
            error,
        ),
    )
    .into()
}

pub(super) fn cudf_series(frame: &DataFrame, column: &str) -> Result<Series> {
    frame.series(column).map_err(cudf_logical_mapping_error)
}

pub(super) fn record_batches_to_gpu(
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
                nexus_query_engine::error::QueryError::from_cudf(
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
            nexus_query_engine::error::QueryError::from_cudf(
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
