use std::panic::{AssertUnwindSafe, catch_unwind};

use cudf::{
    ErrorCode, ErrorKind, ErrorStatus, TypeId,
    core::error::{FactKeyName, FactValue, kind_for_code, status_for_code},
};
use strum::IntoEnumIterator;

#[test]
fn kind_projection_is_defined_for_every_public_error_code() {
    let codes: Vec<_> = ErrorCode::iter().collect();
    assert!(!codes.is_empty(), "ErrorCode::iter() must enumerate the public taxonomy");

    for code in codes {
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

#[test]
fn status_projection_is_defined_for_every_public_error_code() {
    let codes: Vec<_> = ErrorCode::iter().collect();
    assert!(!codes.is_empty(), "ErrorCode::iter() must enumerate the public taxonomy");

    for code in codes {
        assert!(
            matches!(status_for_code(code), ErrorStatus::Permanent | ErrorStatus::Temporary | ErrorStatus::Persistent),
            "{code} must map to exactly one ErrorStatus",
        );
    }
}

// `Facts` duplicate-insert and wrong-variant tests live as crate-internal
// unit tests in `cudf-rs/src/core/error/facts.rs` because the writer surface
// is not downstream-visible, so the tests cannot live outside the crate.

#[test]
fn truthful_error_hook_exposes_typed_facts_without_string_parsing() {
    let err = cudf::testing::core::trigger_truthful_error_contract_for_test().unwrap_err();

    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
    assert_eq!(err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(err.facts().len(), 2);
    assert_eq!(err.detail_values("detail").collect::<Vec<_>>(), vec!["alpha", "beta"]);
}

#[test]
fn reanchor_preserves_payload_and_records_call_chain_in_order() {
    let err = cudf::testing::core::trigger_truthful_error_contract_for_test().unwrap_err();

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

#[test]
fn detail_writer_panics_when_given_a_canonical_fact_snake_case_key() {
    let panic = catch_unwind(AssertUnwindSafe(cudf::testing::core::trigger_canonical_key_in_detail_path_for_test));

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

#[test]
fn reanchor_does_not_deduplicate_same_operation_labels() {
    let err = cudf::testing::core::trigger_same_operation_reanchor_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::same");
    assert_eq!(err.call_chain(), ["testing::truthful_error_contract::same", "testing::truthful_error_contract::same",],);
}

#[test]
fn reanchor_if_needed_is_noop_when_operation_is_already_anchored() {
    let err = cudf::testing::core::trigger_reanchor_if_needed_same_operation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::same_if_needed");
    assert!(err.call_chain().is_empty());
    assert_eq!(err.message(), "reanchor_if_needed same operation");
}

#[test]
fn if_absent_helpers_preserve_existing_payload_and_only_fill_missing_fields() {
    let err = cudf::testing::core::trigger_if_absent_error_annotation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::if_absent");
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(7)));
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Int32)));
    assert_eq!(err.detail_value("resource"), Some("existing_resource"));
    assert_eq!(err.detail_value("reason"), Some("missing_value"));
    assert_eq!(err.detail_values("resource").collect::<Vec<_>>(), vec!["existing_resource"]);
    assert_eq!(err.detail_values("reason").collect::<Vec<_>>(), vec!["missing_value"]);
}

#[test]
fn with_column_helper_writes_the_canonical_column_name_fact() {
    let err = cudf::testing::core::trigger_with_column_error_annotation_contract_for_test().unwrap_err();

    assert_eq!(err.operation(), "testing::truthful_error_contract::with_column::outer");
    assert_eq!(err.call_chain(), ["testing::truthful_error_contract::with_column::inner"]);
    assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str("alpha".into())));
    assert_eq!(err.detail_value("column_name"), None);
}

#[test]
fn foreign_invariant_contract_helper_builds_the_canonical_error_shape() {
    let err = cudf::testing::error::foreign_invariant_violation_for_test();

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

#[test]
fn foreign_invariant_contract_annotation_only_fills_missing_details() {
    let err = cudf::testing::error::annotate_foreign_invariant_contract_if_absent_for_test();

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
