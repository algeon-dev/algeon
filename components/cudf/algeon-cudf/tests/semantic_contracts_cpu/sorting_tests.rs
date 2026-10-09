use cudf::{
    ErrorCode, ErrorKind,
    core::error::{FactKeyName, FactValue},
    testing::{run_named_cases, sorting::trigger_sort_key_count_limit_error_for_test},
};

use crate::common::error_context_value;

fn test_testing_hook_locks_sort_key_count_limit_contract() {
    let err = trigger_sort_key_count_limit_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "sorting::sort");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("sort_keys"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(err.fact_value(FactKeyName::KeysCount), Some(&FactValue::Usize(usize::MAX)));
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("reduce the number of sort keys to fit within i32::MAX")
    );
    assert_eq!(error_context_value(&err, "num_keys").as_deref(), None);
}

#[test]
fn sorting_tests_cases() {
    run_named_cases(&[(
        "test_testing_hook_locks_sort_key_count_limit_contract",
        test_testing_hook_locks_sort_key_count_limit_contract,
    )]);
}
