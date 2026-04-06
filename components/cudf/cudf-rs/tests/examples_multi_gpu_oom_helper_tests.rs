#[path = "common/mod.rs"]
mod common;
#[path = "../examples/multi_gpu/mod.rs"]
mod multi_gpu;

use common::error_context_value;
use cudf::ErrorKind;
use multi_gpu::{
    REQUIRED_VISIBLE_DEVICES, WorkflowReport, WorkflowStatus, validate_exact_visible_devices, validate_final_reports,
};

fn report(label: &'static str, device_ordinal: usize, status: WorkflowStatus, summary: &str) -> WorkflowReport {
    WorkflowReport { label, device_ordinal, status, elapsed_ms: 123, summary: summary.to_string() }
}

#[test]
fn exact_visible_device_validation_accepts_four_devices() {
    validate_exact_visible_devices(REQUIRED_VISIBLE_DEVICES).expect("exactly four visible devices should pass");
}

#[test]
fn exact_visible_device_validation_rejects_non_four_counts() {
    let err = validate_exact_visible_devices(3).expect_err("three visible devices must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "required_visible_devices"), Some("4"));
    assert_eq!(error_context_value(&err, "visible_devices"), Some("3"));

    let err = validate_exact_visible_devices(5).expect_err("five visible devices must fail");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "required_visible_devices"), Some("4"));
    assert_eq!(error_context_value(&err, "visible_devices"), Some("5"));
}

#[test]
fn final_report_validation_accepts_three_completions_and_one_expected_oom() {
    let reports = vec![
        report("gpu0-read-filter-sort", 0, WorkflowStatus::Completed, "rows=128"),
        report("gpu1-groupby-agg", 1, WorkflowStatus::Completed, "groups=7"),
        report("gpu2-merge-concat", 2, WorkflowStatus::Completed, "rows=16"),
        report("gpu3-oom", 3, WorkflowStatus::ExpectedOom, "OutOfMemory at CONCAT"),
    ];

    validate_final_reports(&reports).expect("three completed workflows plus one expected OOM should pass");
}

#[test]
fn final_report_validation_rejects_missing_expected_oom() {
    let reports = vec![
        report("gpu0", 0, WorkflowStatus::Completed, "ok"),
        report("gpu1", 1, WorkflowStatus::Completed, "ok"),
        report("gpu2", 2, WorkflowStatus::Completed, "ok"),
        report("gpu3", 3, WorkflowStatus::Completed, "ok"),
    ];

    let err = validate_final_reports(&reports).expect_err("summary must fail without an expected OOM");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "completed_workflows"), Some("4"));
    assert_eq!(error_context_value(&err, "expected_oom_workflows"), Some("0"));
}

#[test]
fn final_report_validation_rejects_unexpected_failures() {
    let reports = vec![
        report("gpu0", 0, WorkflowStatus::Completed, "ok"),
        report("gpu1", 1, WorkflowStatus::Completed, "ok"),
        report("gpu2", 2, WorkflowStatus::UnexpectedError, "CudaError"),
        report("gpu3", 3, WorkflowStatus::ExpectedOom, "OutOfMemory"),
    ];

    let err = validate_final_reports(&reports).expect_err("unexpected errors must fail the summary");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "unexpected_failures"), Some("1"));
}

#[test]
fn final_report_validation_rejects_wrong_report_count() {
    let reports = vec![
        report("gpu0", 0, WorkflowStatus::Completed, "ok"),
        report("gpu1", 1, WorkflowStatus::Completed, "ok"),
        report("gpu2", 2, WorkflowStatus::ExpectedOom, "OutOfMemory"),
    ];

    let err = validate_final_reports(&reports).expect_err("summary must require four workflow reports");
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(error_context_value(&err, "expected_workflow_reports"), Some("4"));
    assert_eq!(error_context_value(&err, "actual_workflow_reports"), Some("3"));
}
