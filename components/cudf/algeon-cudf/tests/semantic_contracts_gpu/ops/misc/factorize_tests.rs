use cudf::{
    ErrorCode, ErrorKind, Exec, Series, TypeId,
    core::{
        Column,
        error::{FactKeyName, FactValue},
    },
    testing::{
        factorization::{
            trigger_codebook_key_count_limit_error_for_test, trigger_factorize_empty_row_code_dtype_error_for_test,
        },
        prelude::*,
    },
};

use crate::common::{
    column_from_slice, column_from_slice_nullable, column_from_strs, dataframe_from_columns, error_context_value,
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, reset_exec_observation, reset_sync_observation,
    series,
};

fn factorize_column(column: Column) -> (Series, Series) {
    let series = Series::from_column(column).unwrap();
    let factorized = series.on(Exec::RETAINED).factorize().unwrap();
    (factorized.row_codes().as_series().clone(), factorized.codebook().keys().clone())
}

fn test_factorize_i32_with_duplicates() {
    let data = vec![429i32, 111, 213, 111, 213, 429, 213];
    let col = column_from_slice(&data).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::Int32);

    assert_eq!(uniques_view.len().unwrap(), 3);
    assert_eq!(codes_view.len().unwrap(), 7);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    for (i, &code) in codes_host.iter().enumerate() {
        assert_eq!(
            data[i], uniques_host[code as usize],
            "data[{}]={} should equal uniques[{}]={}",
            i, data[i], code, uniques_host[code as usize]
        );
    }

    // Verify uniques contain the input distinct values (order is not guaranteed).
    let mut expected_uniques = data;
    expected_uniques.sort();
    expected_uniques.dedup();

    let mut actual_uniques = uniques_host;
    actual_uniques.sort();
    actual_uniques.dedup();

    assert_eq!(actual_uniques, expected_uniques);
}

fn test_factorize_string_column() {
    let strings: &[&str] = &["banana", "apple", "cherry", "apple", "banana"];
    let col = column_from_strs(strings).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::String);

    assert_eq!(uniques_view.len().unwrap(), 3);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_strings_lossy().unwrap();

    for (i, &code) in codes_host.iter().enumerate() {
        assert_eq!(
            strings[i], uniques_host[code as usize],
            "strings[{}]={} should equal uniques[{}]={}",
            i, strings[i], code, uniques_host[code as usize]
        );
    }
}

fn test_factorize_with_nulls() {
    let data = vec![10i32, 20, 30, 20, 10];
    let validity = vec![true, false, true, true, false];
    let col = column_from_slice_nullable(&data, &validity).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 5);

    // Null positions in input should produce null codes.
    let codes_validity = codes_view.to_host_validity().unwrap();
    assert!(codes_validity.is_some(), "codes should have a null mask");
    let codes_mask = codes_validity.unwrap();
    // Indices 1 and 4 were null in the input.
    assert!(!codes_mask[1], "code at null position 1 should be null");
    assert!(!codes_mask[4], "code at null position 4 should be null");

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();
    for (i, (&code, &valid)) in codes_host.iter().zip(codes_mask.iter()).enumerate() {
        if valid {
            assert_eq!(data[i], uniques_host[code as usize], "data[{}]={} should equal uniques[{}]", i, data[i], code,);
        }
    }
}

fn test_factorize_empty_column() {
    let col = column_from_slice::<i32>(&[]).unwrap();
    reset_exec_observation();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 0);
    assert_eq!(uniques_view.len().unwrap(), 0);
    assert_eq!(last_exec_stream_value(), codes.as_column().testing_execution_domain_stream_value());
    assert_eq!(last_exec_mr_value(), codes.as_column().testing_execution_domain_mr_value());
    assert_eq!(
        uniques.as_column().testing_execution_domain_stream_value(),
        codes.as_column().testing_execution_domain_stream_value()
    );
    assert_eq!(
        uniques.as_column().testing_execution_domain_mr_value(),
        codes.as_column().testing_execution_domain_mr_value()
    );
}

fn test_factorize_single_element() {
    let col = column_from_slice(&[42i32]).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 1);
    assert_eq!(uniques_view.len().unwrap(), 1);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    assert_eq!(codes_host, vec![0]);
    assert_eq!(uniques_host, vec![42]);
}

fn test_factorize_already_unique() {
    let data = vec![10i32, 20, 30, 40, 50];
    let col = column_from_slice(&data).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    // All values are unique so uniques == sorted input.
    assert_eq!(uniques_view.len().unwrap(), 5);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<i32>().unwrap();

    // Since input is already sorted and unique, codes should be identity [0,1,2,3,4].
    assert_eq!(codes_host, vec![0, 1, 2, 3, 4]);
    assert_eq!(uniques_host, data);
}

fn test_factorize_f64_column() {
    let data = vec![3.25f64, 2.75, 3.25, 1.5, 2.75];
    let col = column_from_slice(&data).unwrap();
    let (codes, uniques) = factorize_column(col);

    let codes_view = codes.testing_view().unwrap();
    let uniques_view = uniques.testing_view().unwrap();

    assert_eq!(codes_view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(uniques_view.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(uniques_view.len().unwrap(), 3);
    assert_eq!(codes_view.len().unwrap(), 5);

    let codes_host = codes_view.to_vec::<i32>().unwrap();
    let uniques_host = uniques_view.to_vec::<f64>().unwrap();

    for (i, &code) in codes_host.iter().enumerate() {
        assert!(
            (data[i] - uniques_host[code as usize]).abs() < 1e-9,
            "data[{}]={} should equal uniques[{}]={}",
            i,
            data[i],
            code,
            uniques_host[code as usize]
        );
    }
}

fn test_factorize_codes_are_owned_independent_of_input() {
    // Verify codes and uniques are owned (deep-copied) and do not borrow from input.
    let (codes, uniques) = {
        let data = vec![1i32, 2, 1, 3, 2];
        let col = column_from_slice(&data).unwrap();
        factorize_column(col)
        // col is dropped here
    };

    // Should still be readable after input is dropped.
    let codes_host = codes.testing_view().unwrap().to_vec::<i32>().unwrap();
    let uniques_host = uniques.testing_view().unwrap().to_vec::<i32>().unwrap();

    assert_eq!(codes_host.len(), 5);
    assert_eq!(uniques_host.len(), 3);
}

fn test_dataframe_factorize_by_name() {
    let col = column_from_slice(&[10i32, 20, 10, 30, 20]).unwrap();
    let df = dataframe_from_columns(vec!["values".to_string()], vec![col]).unwrap();

    let factorized = df.on(Exec::RETAINED).factorize("values").unwrap();
    let codes_view = factorized.row_codes().testing_view().unwrap();
    let keys_view = factorized.codebook().keys().testing_view().unwrap();

    assert_eq!(codes_view.len().unwrap(), 5);
    assert_eq!(keys_view.len().unwrap(), 3);
}

fn test_dataframe_factorize_missing_column() {
    let col = column_from_slice(&[1i32, 2, 3]).unwrap();
    let df = dataframe_from_columns(vec!["x".to_string()], vec![col]).unwrap();

    let result = df.on(Exec::RETAINED).factorize("nonexistent");
    assert!(result.is_err());
}

fn test_factorize_row_codes_are_not_a_reusable_codebook_for_uniques() {
    let lookup_source = series("lookup_source", &["b", "b", "a", "c"]).unwrap();
    let factorized = lookup_source.on(Exec::RETAINED).factorize().unwrap();
    let row_codes = factorized.row_codes();
    let uniques = factorized.codebook().keys();

    // Row-aligned factorize codes are not a key-to-code mapping. Slicing them to
    // uniques.len() and reusing them as a stable lookup does not yield [0, 1, 2].
    let first_positions = series("first_positions", &[0i32, 1, 2]).unwrap();
    let wrong_lookup_values = row_codes.on(Exec::RETAINED).gather(&first_positions).unwrap();
    assert_eq!(wrong_lookup_values.len(), uniques.len());
    assert_ne!(wrong_lookup_values.to_vec_strict::<i32>().unwrap(), vec![0, 1, 2]);
}

fn test_factorize_uniques_support_safe_positional_codebook_round_trip() {
    let lookup_source = Series::from_column(column_from_strs(&["b", "a", "c", "a"]).unwrap()).unwrap();
    let factorized = lookup_source.on(Exec::RETAINED).factorize().unwrap();
    let codebook = factorized.codebook();
    let query = series("query", &["a", "b", "c"]).unwrap();
    let encoded = codebook.encode(&query, Exec::RETAINED).unwrap();
    let decoded = codebook.decode(&encoded, Exec::RETAINED).unwrap();

    assert_eq!(
        decoded.testing_view().unwrap().to_strings_lossy().unwrap(),
        query.testing_view().unwrap().to_strings_lossy().unwrap()
    );
}

fn test_factorize_retains_input_provenance_and_syncs_on_input_stream_end_to_end() {
    let input = column_from_slice(&[9i32, 7, 9, 8]).unwrap().testing_rebind_execution_domain_explicit(605).unwrap();
    let expected_stream = input.testing_execution_domain_stream_value();
    let expected_mr = input.testing_execution_domain_mr_value();
    let expected_domain = input.testing_execution_domain_id();
    let input = Series::from_column(input).unwrap();

    reset_exec_observation();
    let factorized = input.on(Exec::RETAINED).factorize().unwrap();
    let codes = factorized.row_codes().as_series().clone();
    let uniques = factorized.codebook().keys().clone();

    assert_eq!(codes.as_column().testing_execution_domain_id(), expected_domain);
    assert_eq!(uniques.as_column().testing_execution_domain_id(), expected_domain);
    assert_eq!(codes.as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(uniques.as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(codes.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(uniques.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    let codes_host = codes.testing_view().unwrap().to_vec::<i32>().unwrap();
    let mut uniques_host = uniques.testing_view().unwrap().to_vec::<i32>().unwrap();
    uniques_host.sort_unstable();
    assert_eq!(codes_host.len(), 4);
    assert_eq!(uniques_host, vec![7, 8, 9]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

fn test_testing_hook_locks_factorize_empty_row_code_dtype_contract() {
    let err = trigger_factorize_empty_row_code_dtype_error_for_test().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvariantViolation);
    assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
    assert_eq!(err.operation(), "series::factorize");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::Float64)));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("row_codes"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("unexpected_dtype"));
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("float64"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure the factorization backend emits Int32 row codes")
    );
    assert_eq!(error_context_value(&err, "row_code_type_id").as_deref(), None);
}

fn test_testing_hook_locks_codebook_key_count_limit_contract() {
    let err = trigger_codebook_key_count_limit_error_for_test().unwrap_err();
    let expected = (i32::MAX as usize) + 1;
    let expected_text = expected.to_string();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "codebook::from_keys");
    assert_eq!(err.fact_value(FactKeyName::KeysCount), None);
    assert_eq!(err.detail_value("keys_count"), None);
    assert_eq!(error_context_value(&err, "size").as_deref(), Some(expected_text.as_str()));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("size"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("exceeds_i32_max"));
    assert_eq!(error_context_value(&err, "remedy").as_deref(), Some("reduce size to fit within i32::MAX"));
    assert_eq!(error_context_value(&err, "len").as_deref(), None);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("factorize_tests::test_factorize_i32_with_duplicates", test_factorize_i32_with_duplicates),
    ("factorize_tests::test_factorize_string_column", test_factorize_string_column),
    ("factorize_tests::test_factorize_with_nulls", test_factorize_with_nulls),
    ("factorize_tests::test_factorize_empty_column", test_factorize_empty_column),
    ("factorize_tests::test_factorize_single_element", test_factorize_single_element),
    ("factorize_tests::test_factorize_already_unique", test_factorize_already_unique),
    ("factorize_tests::test_factorize_f64_column", test_factorize_f64_column),
    (
        "factorize_tests::test_factorize_codes_are_owned_independent_of_input",
        test_factorize_codes_are_owned_independent_of_input,
    ),
    ("factorize_tests::test_dataframe_factorize_by_name", test_dataframe_factorize_by_name),
    ("factorize_tests::test_dataframe_factorize_missing_column", test_dataframe_factorize_missing_column),
    (
        "factorize_tests::test_factorize_row_codes_are_not_a_reusable_codebook_for_uniques",
        test_factorize_row_codes_are_not_a_reusable_codebook_for_uniques,
    ),
    (
        "factorize_tests::test_factorize_uniques_support_safe_positional_codebook_round_trip",
        test_factorize_uniques_support_safe_positional_codebook_round_trip,
    ),
    (
        "factorize_tests::test_factorize_retains_input_provenance_and_syncs_on_input_stream_end_to_end",
        test_factorize_retains_input_provenance_and_syncs_on_input_stream_end_to_end,
    ),
    (
        "factorize_tests::test_testing_hook_locks_factorize_empty_row_code_dtype_contract",
        test_testing_hook_locks_factorize_empty_row_code_dtype_contract,
    ),
    (
        "factorize_tests::test_testing_hook_locks_codebook_key_count_limit_contract",
        test_testing_hook_locks_codebook_key_count_limit_contract,
    ),
];
