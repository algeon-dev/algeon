use super::*;

#[test]
fn accepted_row_preserves_native_not_started_failure_status() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Err(execution::NativeRunFailure {
            runtime_started: false,
            message: "native planning failed\nwith details".to_string(),
            details: Box::default(),
        }),
    );

    assert!(row.cpu_runtime_started);
    assert!(!row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Failed);
    assert_eq!(row.runtime_failure.as_deref(), Some("native planning failed with details"));
    assert_eq!(row.parity_status, validation::ParityStatus::NativeFailed);
    assert_eq!(row.cpu_output_rows, Some(4));
    assert_eq!(row.native_output_rows, None);
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(&row), "bounded_accepted_native_failed");
}

#[test]
fn accepted_row_distinguishes_cpu_planning_failure_from_started_runtime() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Err(execution::CpuRunFailure {
            runtime_started: false,
            message: "cpu physical planning failed\tbefore execution".to_string(),
            details: Box::default(),
        }),
        Ok(timed_native_run("output-fnv1a64:0000000000000001")),
    );

    assert!(!row.cpu_runtime_started);
    assert!(row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.cpu_failure.as_deref(), Some("cpu physical planning failed before execution"));
    assert_eq!(row.runtime_failure, None);
    assert_eq!(row.parity_status, validation::ParityStatus::CpuFailed);
    assert_eq!(row.cpu_output_rows, None);
    assert_eq!(row.native_output_rows, Some(4));
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(&row), "bounded_accepted_cpu_failed");
}

#[test]
fn accepted_row_reports_fingerprint_mismatch_without_canonical_output_fingerprint() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Ok(timed_native_run("output-fnv1a64:0000000000000002")),
    );

    assert!(row.cpu_runtime_started);
    assert!(row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, validation::ParityStatus::Mismatched);
    assert_eq!(row.cpu_output_fingerprint.as_deref(), Some("output-fnv1a64:0000000000000001"));
    assert_eq!(row.native_output_fingerprint.as_deref(), Some("output-fnv1a64:0000000000000002"));
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(&row), "bounded_accepted_parity_mismatched");
}

#[test]
fn native_runtime_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_nexus::error::ErrorCode::MemoryReservationFailed,
        "datafusion_nexus.native_exec.execute",
        "nexus native query execution failed",
    )
    .with_fact("native_code", "memory_reservation_failed")
    .with_fact("native_kind", "resource_exhausted")
    .with_fact("native_status", "temporary")
    .with_fact("native_operation", "nexus_query_engine.memory.reserve")
    .with_fact("memory_boundary", "final_sort")
    .with_fact("requested_bytes", 2048)
    .with_fact("granted_bytes", 1024);
    let datafusion_error = DataFusionError::from(adapter_error);
    let details = execution::FailureDetails::from_datafusion_error(&datafusion_error);

    assert_eq!(details.code.as_deref(), Some("memory_reservation_failed"));
    assert_eq!(details.kind.as_deref(), Some("resource_exhausted"));
    assert_eq!(details.status.as_deref(), Some("temporary"));
    assert_eq!(details.operation.as_deref(), Some("datafusion_nexus.native_exec.execute"));
    assert_eq!(details.facts.get("memory_boundary").map(String::as_str), Some("final_sort"));
    assert_eq!(details.facts.get("requested_bytes").map(String::as_str), Some("2048"));

    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Err(execution::NativeRunFailure {
            runtime_started: true,
            message: datafusion_error.to_string(),
            details: Box::new(details),
        }),
    );
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config: NativeTpchBenchmarkConfig {
            query_ids: vec!["q01".to_string()],
            ..NativeTpchBenchmarkConfig::default()
        },
        devices: Vec::new(),
        rows: vec![row],
    };
    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let row = lines.next().expect("benchmark report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row_fields = row.split('\t').collect::<Vec<_>>();
    let facts = row_fields[header_index(&header_fields, "runtime_failure_facts")];

    assert_eq!(row_fields[header_index(&header_fields, "runtime_outcome")], "failed");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_code")], "memory_reservation_failed");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_kind")], "resource_exhausted");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_status")], "temporary");
    assert_eq!(
        row_fields[header_index(&header_fields, "runtime_failure_operation")],
        "datafusion_nexus.native_exec.execute"
    );
    assert!(facts.contains("memory_boundary=final_sort"));
    assert!(facts.contains("native_operation=nexus_query_engine.memory.reserve"));
    assert!(facts.contains("requested_bytes=2048"));
}

#[test]
fn planning_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_nexus::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.native_planner",
        "native physical planning failed",
    )
    .with_fact("query_id", "q01")
    .with_fact("planner_stage", "native_lowering");
    let row = NativeTpchBenchmarkRow::planning_failure("q01", DataFusionError::from(adapter_error));

    assert_eq!(row.reason_code.as_deref(), Some("physical_planning_failed"));
    assert_eq!(row.unsupported_category.as_deref(), Some("planning"));
    assert!(!row.cpu_runtime_started);
    assert!(!row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::NotStarted);
    assert_eq!(row.runtime_failure_details.code.as_deref(), Some("invalid_execution_plan"));
    assert_eq!(row.runtime_failure_details.kind.as_deref(), Some("planning"));
    assert_eq!(row.runtime_failure_details.status.as_deref(), Some("permanent"));
    assert_eq!(row.runtime_failure_details.operation.as_deref(), Some("datafusion_nexus.native_planner"));
    assert_eq!(row.runtime_failure_details.facts.get("planner_stage").map(String::as_str), Some("native_lowering"));
    assert_eq!(bounded_status(&row), "bounded_planning_failed");
}

#[test]
fn cpu_runtime_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_nexus::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.cpu_collect",
        "cpu collection failed",
    )
    .with_fact("query_id", "q01")
    .with_fact("stage", "collect");
    let datafusion_error = DataFusionError::from(adapter_error);
    let details = execution::FailureDetails::from_datafusion_error(&datafusion_error);
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Err(execution::CpuRunFailure {
            runtime_started: true,
            message: datafusion_error.to_string(),
            details: Box::new(details),
        }),
        Ok(timed_native_run("output-fnv1a64:0000000000000001")),
    );
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config: NativeTpchBenchmarkConfig {
            query_ids: vec!["q01".to_string()],
            ..NativeTpchBenchmarkConfig::default()
        },
        devices: Vec::new(),
        rows: vec![row],
    };
    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let row = lines.next().expect("benchmark report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row_fields = row.split('\t').collect::<Vec<_>>();
    let facts = row_fields[header_index(&header_fields, "cpu_failure_facts")];

    assert_eq!(row_fields[header_index(&header_fields, "parity_status")], "cpu_failed");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_code")], "invalid_execution_plan");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_kind")], "planning");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_status")], "permanent");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_operation")], "datafusion_nexus.cpu_collect");
    assert!(facts.contains("stage=collect"));
    assert!(facts.contains("query_id=q01"));
}
