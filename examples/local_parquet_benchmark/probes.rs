use super::*;

pub(super) async fn run_expected_planning_rejection_probe(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ExpectedPlanningRejectedReport> {
    let planned = plan_datafusion_native_sql(path, rejected_benchmark_sql(), config).await?;
    if planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_some() || planned.planning_report.accepted_count() != 0
    {
        return Err(plan_error(format!(
            "expected planning-rejected benchmark mode produced NativeCudfExec: root={}, accepted_count={}, rejected_count={}",
            planned.plan.name(),
            planned.planning_report.accepted_count(),
            planned.planning_report.rejected_count()
        )));
    }
    if planned.planning_report.rejected_count() == 0 {
        return Err(plan_error(format!(
            "expected planning-rejected benchmark mode did not retain a rejection row: root={}, candidates={}",
            planned.plan.name(),
            planned.native_candidate_shapes.join(";")
        )));
    }

    Ok(ExpectedPlanningRejectedReport {
        parquet_file_bytes,
        parquet_row_groups: dataset_layout.row_group_count,
        input_rows: config.input_rows,
        file_count: config.file_count,
        compression: config.compression,
        native_plan_shape: config.native_plan_shape,
        expected_outcome: config.expected_outcome,
        final_plan_root: planned.plan.name().to_string(),
        source_diagnostic,
        planning_report: planned.planning_report,
    })
}

pub(super) fn rejected_benchmark_sql() -> &'static str {
    "SELECT AVG(m00) AS avg_m00 FROM bench"
}

pub(super) async fn run_expected_resource_failure_probe(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ExpectedResourceFailureReport> {
    if !config.native_memory_control_policy.is_bounded() {
        return Err(plan_error(
            "expected resource-failure benchmark mode requires CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY=bounded",
        ));
    }

    let planned = plan_datafusion_native_sql(path, &benchmark_sql(config), config).await?;
    let DataFusionNativePlannedExec { plan, planning: _, planning_report, native_candidate_shapes } = planned;
    if plan.as_any().downcast_ref::<NativeCudfExec>().is_none() || planning_report.accepted_count() != 1 {
        return Err(plan_error(format!(
            "expected resource-failure benchmark mode requires one accepted NativeCudfExec, got root={}, accepted_count={}, rejected_count={}, candidates={}",
            plan.name(),
            planning_report.accepted_count(),
            planning_report.rejected_count(),
            native_candidate_shapes.join(";")
        )));
    }

    let final_plan_root = plan.name().to_string();
    let source_physical_plan_one_line = planning_report
        .rows
        .first()
        .map(|row| row.physical_shape.clone())
        .unwrap_or_else(|| displayable(plan.as_ref()).one_line().to_string());

    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(_) => Err(plan_error(
            "expected resource-failure benchmark mode completed successfully; tighten the bounded memory policy or run as expected_outcome=success",
        )),
        Err(error) => {
            let adapter_error = adapter_error_from_datafusion(&error)?;
            if !is_planned_adapter_resource_failure(adapter_error) {
                return Err(error);
            }
            if adapter_error.fact_value("native_datafusion_phase") != Some("runtime_execution") {
                return Err(plan_error(format!(
                    "expected resource-failure benchmark mode to fail after native execution started, got native_datafusion_phase={:?}",
                    adapter_error.fact_value("native_datafusion_phase")
                )));
            }
            if adapter_error.fact_value("cpu_fallback_after_native_start") != Some("false") {
                return Err(plan_error(format!(
                    "expected resource-failure benchmark mode to preserve no-silent-fallback fact, got cpu_fallback_after_native_start={:?}",
                    adapter_error.fact_value("cpu_fallback_after_native_start")
                )));
            }
            Ok(ExpectedResourceFailureReport {
                parquet_file_bytes,
                parquet_row_groups: dataset_layout.row_group_count,
                input_rows: config.input_rows,
                file_count: config.file_count,
                compression: config.compression,
                native_plan_shape: config.native_plan_shape,
                expected_outcome: config.expected_outcome,
                final_plan_root,
                planning_report,
                native_candidate_shapes,
                source_physical_plan_one_line,
                source_diagnostic,
                error: AdapterRuntimeErrorReport::from_adapter_error(adapter_error),
            })
        },
    }
}

pub(super) fn is_planned_adapter_resource_failure(error: &AdapterError) -> bool {
    matches!(
        error.code(),
        datafusion_cudf::ErrorCode::MemoryReservationFailed | datafusion_cudf::ErrorCode::CudfResourceExhausted
    )
}

pub(super) fn adapter_error_from_datafusion(error: &DataFusionError) -> DataFusionResult<&AdapterError> {
    let DataFusionError::External(source) = error else {
        return Err(plan_error(format!(
            "native execution errors should cross DataFusion as external adapter errors: {error}"
        )));
    };
    source.downcast_ref::<AdapterError>().ok_or_else(|| {
        plan_error(format!("DataFusion external error did not preserve datafusion_cudf::Error: {error}"))
    })
}

pub(super) fn adapter_error_facts_tsv(error: &AdapterError) -> String {
    let facts = error
        .facts()
        .iter()
        .map(|(key, value)| format!("{key}:{}", value.replace(['\t', '\n'], " ")))
        .collect::<Vec<_>>();
    if facts.is_empty() { "none".to_string() } else { facts.join(",") }
}
