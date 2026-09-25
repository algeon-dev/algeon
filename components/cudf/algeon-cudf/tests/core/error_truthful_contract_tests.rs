use std::panic::{AssertUnwindSafe, catch_unwind};

use cudf::{
    ErrorCode, ErrorKind, ErrorStatus, TypeId,
    core::error::{FactKeyName, FactValue, kind_for_code, status_for_code},
    testing::{
        core::{
            ffi_cuda_status_error_for_test, ffi_io_status_error_for_test,
            trigger_canonical_key_in_detail_path_for_test, trigger_if_absent_error_annotation_contract_for_test,
            trigger_reanchor_if_needed_same_operation_contract_for_test,
            trigger_same_operation_reanchor_contract_for_test, trigger_truthful_error_contract_for_test,
            trigger_with_column_error_annotation_contract_for_test,
        },
        error::{annotate_foreign_invariant_contract_if_absent_for_test, foreign_invariant_violation_for_test},
    },
};
use crate::common::run_named_cases;

fn kind_projection_is_defined_for_every_public_error_code() {
    for code in ErrorCode::ALL {
        assert!(
            matches!(
                kind_for_code(code),
                ErrorKind::InvalidInput
                    | ErrorKind::NotFound
                    | ErrorKind::Unsupported
                    | ErrorKind::ExecutionContext
                    | ErrorKind::ResourceExhausted
                    | ErrorKind::DependencyFailure
                    | ErrorKind::InvariantViolation
            ),
            "{code} must map to exactly one ErrorKind",
        );
    }
}

fn status_projection_is_defined_for_every_public_error_code() {
    for code in ErrorCode::ALL {
        assert!(
            matches!(status_for_code(code), ErrorStatus::Permanent | ErrorStatus::Temporary | ErrorStatus::Persistent),
            "{code} must map to exactly one ErrorStatus",
        );
    }
}

fn transient_io_failure_is_a_temporary_dependency_leaf() {
    assert_eq!(kind_for_code(ErrorCode::TransientIoFailure), ErrorKind::DependencyFailure);
    assert_eq!(status_for_code(ErrorCode::TransientIoFailure), ErrorStatus::Temporary);
}

fn deterministic_host_materialization_size_failures_are_permanent_resource_leaves() {
    for code in [ErrorCode::SizeOverflow, ErrorCode::HostMaterializationLimitExceeded] {
        assert_eq!(kind_for_code(code), ErrorKind::ResourceExhausted);
        assert_eq!(status_for_code(code), ErrorStatus::Permanent);
    }
}

fn ffi_io_status_promotes_only_known_cuda_copy_transients() {
    let transient = ffi_io_status_error_for_test(
        "CUDA error encountered at: cudf/cpp/src/utilities/cuda_memcpy.cu:43: 1 cudaErrorInvalidValue invalid argument",
        "testing::ffi_status_error_for_test",
    );
    assert_eq!(transient.code(), ErrorCode::TransientIoFailure);
    assert_eq!(transient.kind(), ErrorKind::DependencyFailure);
    assert_eq!(transient.status(), ErrorStatus::Temporary);
    assert!(transient.is_retryable());
    assert_eq!(transient.detail_values("transient_promotion").collect::<Vec<_>>(), vec!["cuda_invalid_value_copy"]);

    let ordinary = ffi_io_status_error_for_test(
        "Failed to open local parquet file: No such file or directory",
        "testing::ffi_status_error_for_test",
    );
    assert_eq!(ordinary.code(), ErrorCode::IoFailure);
    assert_eq!(ordinary.kind(), ErrorKind::DependencyFailure);
    assert_eq!(ordinary.status(), ErrorStatus::Permanent);
    assert!(!ordinary.is_retryable());

    let os_invalid_argument = ffi_io_status_error_for_test(
        "Failed to open local parquet file: invalid argument in thrust path",
        "testing::ffi_status_error_for_test",
    );
    assert_eq!(os_invalid_argument.code(), ErrorCode::IoFailure);
    assert_eq!(os_invalid_argument.status(), ErrorStatus::Permanent);
    assert!(!os_invalid_argument.is_retryable());

    let cuda_invalid_value_without_copy_context = ffi_io_status_error_for_test(
        "cudaErrorInvalidValue invalid argument while opening input path",
        "testing::ffi_status_error_for_test",
    );
    assert_eq!(cuda_invalid_value_without_copy_context.code(), ErrorCode::IoFailure);
    assert_eq!(cuda_invalid_value_without_copy_context.status(), ErrorStatus::Permanent);

    let cuda =
        ffi_cuda_status_error_for_test("cudaErrorInvalidValue invalid argument", "testing::ffi_status_error_for_test");
    assert_eq!(cuda.code(), ErrorCode::CudaRuntime);
    assert_eq!(cuda.status(), ErrorStatus::Temporary);
}

// `Facts` duplicate-insert and wrong-variant tests live as crate-internal
// unit tests in `cudf/src/core/error/facts.rs` because the writer surface
// is not downstream-visible, so the tests cannot live outside the crate.

fn truthful_error_hook_exposes_typed_facts_without_string_parsing() {
    let err = trigger_truthful_error_contract_for_test().unwrap_err();

    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
    assert_eq!(err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(err.facts().len(), 2);
    assert_eq!(err.detail_values("detail").collect::<Vec<_>>(), vec!["alpha", "beta"]);
}

fn diagnostic_string_preserves_machine_fields_and_human_context() {
    let err = trigger_truthful_error_contract_for_test().unwrap_err();
    let diagnostic = err.diagnostic_string();

    assert!(diagnostic.contains("code=ColumnLengthMismatch"), "{diagnostic}");
    assert!(diagnostic.contains("kind=InvalidInput"), "{diagnostic}");
    assert!(diagnostic.contains("status=permanent"), "{diagnostic}");
    assert!(diagnostic.contains("operation=testing::truthful_error_contract::outer"), "{diagnostic}");
    assert!(diagnostic.contains("message=\"column length mismatch\""), "{diagnostic}");
    assert!(diagnostic.contains("facts={expected_rows=3, actual_rows=2}"), "{diagnostic}");
    assert!(diagnostic.contains("details=[detail=\"alpha\", detail=\"beta\"]"), "{diagnostic}");
    assert!(
        diagnostic.contains(
            "call_chain=[testing::truthful_error_contract::inner -> testing::truthful_error_contract::middle]"
        ),
        "{diagnostic}",
    );
    assert!(diagnostic.contains("source=\"root cause\""), "{diagnostic}");
}

fn reanchor_preserves_payload_and_records_call_chain_in_order() {
    let err = trigger_truthful_error_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::outer");
    assert_eq!(
        err.call_chain(),
        ["testing::truthful_error_contract::inner", "testing::truthful_error_contract::middle",],
    );
    assert_eq!(err.message(), "column length mismatch");
    assert_eq!(err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
    assert_eq!(err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(err.detail_values("detail").collect::<Vec<_>>(), vec!["alpha", "beta"]);
    assert_eq!(err.detail_value("called"), None);
    assert_eq!(std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(), Some("root cause"),);
}

fn detail_writer_panics_when_given_a_canonical_fact_snake_case_key() {
    let panic = catch_unwind(AssertUnwindSafe(trigger_canonical_key_in_detail_path_for_test));

    let payload = panic.expect_err("canonical fact key on the detail path must panic the writer");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or_default();
    assert!(
        message.contains("canonical fact key"),
        "expected panic message to mention the canonical-key overlap rule, got: {message}",
    );
}

fn reanchor_does_not_deduplicate_same_operation_labels() {
    let err = trigger_same_operation_reanchor_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::same");
    assert_eq!(err.call_chain(), ["testing::truthful_error_contract::same", "testing::truthful_error_contract::same",],);
}

fn reanchor_if_needed_is_noop_when_operation_is_already_anchored() {
    let err = trigger_reanchor_if_needed_same_operation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::same_if_needed");
    assert!(err.call_chain().is_empty());
    assert_eq!(err.message(), "reanchor_if_needed same operation");
}

fn if_absent_helpers_preserve_existing_payload_and_only_fill_missing_fields() {
    let err = trigger_if_absent_error_annotation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::if_absent");
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(7)));
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.detail_value("resource"), Some("existing_resource"));
    assert_eq!(err.detail_value("reason"), Some("missing_value"));
    assert_eq!(err.detail_values("resource").collect::<Vec<_>>(), vec!["existing_resource"]);
    assert_eq!(err.detail_values("reason").collect::<Vec<_>>(), vec!["missing_value"]);
}

fn with_column_helper_writes_the_canonical_column_name_fact() {
    let err = trigger_with_column_error_annotation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::with_column::outer");
    assert_eq!(err.call_chain(), ["testing::truthful_error_contract::with_column::inner"]);
    assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str("alpha".into())));
    assert_eq!(err.detail_value("column_name"), None);
}

fn foreign_invariant_contract_helper_builds_the_canonical_error_shape() {
    let err = foreign_invariant_violation_for_test();

    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::foreign_invariant_contract::build");
    assert_eq!(err.message(), "foreign metadata contract violated");
    assert_eq!(err.detail_value("resource"), Some("foreign_buffer"));
    assert_eq!(err.detail_value("reason"), Some("invalid_foreign_metadata"));
    assert_eq!(err.detail_value("remedy"), Some("ensure the foreign buffer reports valid metadata"));
    assert_eq!(err.detail_values("resource").collect::<Vec<_>>(), vec!["foreign_buffer"]);
    assert_eq!(err.detail_values("reason").collect::<Vec<_>>(), vec!["invalid_foreign_metadata"]);
    assert_eq!(
        err.detail_values("remedy").collect::<Vec<_>>(),
        vec!["ensure the foreign buffer reports valid metadata"],
    );
}

fn foreign_invariant_contract_annotation_only_fills_missing_details() {
    let err = annotate_foreign_invariant_contract_if_absent_for_test();

    assert_eq!(err.code(), ErrorCode::InvariantViolation);
    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.operation(), "testing::foreign_invariant_contract::annotate");
    assert_eq!(err.message(), "annotate foreign invariant contract");
    assert_eq!(err.detail_value("resource"), Some("existing_resource"));
    assert_eq!(err.detail_value("reason"), Some("missing_reason"));
    assert_eq!(err.detail_value("remedy"), Some("existing_remedy"));
    assert_eq!(err.detail_values("resource").collect::<Vec<_>>(), vec!["existing_resource"]);
    assert_eq!(err.detail_values("reason").collect::<Vec<_>>(), vec!["missing_reason"]);
    assert_eq!(err.detail_values("remedy").collect::<Vec<_>>(), vec!["existing_remedy"]);
}

#[test]
fn error_truthful_contract_tests_cases() {
    run_named_cases(&[
        (
            "kind_projection_is_defined_for_every_public_error_code",
            kind_projection_is_defined_for_every_public_error_code,
        ),
        (
            "status_projection_is_defined_for_every_public_error_code",
            status_projection_is_defined_for_every_public_error_code,
        ),
        ("transient_io_failure_is_a_temporary_dependency_leaf", transient_io_failure_is_a_temporary_dependency_leaf),
        (
            "deterministic_host_materialization_size_failures_are_permanent_resource_leaves",
            deterministic_host_materialization_size_failures_are_permanent_resource_leaves,
        ),
        (
            "ffi_io_status_promotes_only_known_cuda_copy_transients",
            ffi_io_status_promotes_only_known_cuda_copy_transients,
        ),
        (
            "truthful_error_hook_exposes_typed_facts_without_string_parsing",
            truthful_error_hook_exposes_typed_facts_without_string_parsing,
        ),
        (
            "diagnostic_string_preserves_machine_fields_and_human_context",
            diagnostic_string_preserves_machine_fields_and_human_context,
        ),
        (
            "reanchor_preserves_payload_and_records_call_chain_in_order",
            reanchor_preserves_payload_and_records_call_chain_in_order,
        ),
        (
            "detail_writer_panics_when_given_a_canonical_fact_snake_case_key",
            detail_writer_panics_when_given_a_canonical_fact_snake_case_key,
        ),
        ("reanchor_does_not_deduplicate_same_operation_labels", reanchor_does_not_deduplicate_same_operation_labels),
        (
            "reanchor_if_needed_is_noop_when_operation_is_already_anchored",
            reanchor_if_needed_is_noop_when_operation_is_already_anchored,
        ),
        (
            "if_absent_helpers_preserve_existing_payload_and_only_fill_missing_fields",
            if_absent_helpers_preserve_existing_payload_and_only_fill_missing_fields,
        ),
        (
            "with_column_helper_writes_the_canonical_column_name_fact",
            with_column_helper_writes_the_canonical_column_name_fact,
        ),
        (
            "foreign_invariant_contract_helper_builds_the_canonical_error_shape",
            foreign_invariant_contract_helper_builds_the_canonical_error_shape,
        ),
        (
            "foreign_invariant_contract_annotation_only_fills_missing_details",
            foreign_invariant_contract_annotation_only_fills_missing_details,
        ),
    ]);
}
