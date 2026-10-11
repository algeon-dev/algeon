//! Integration tests for GPU string operations.
//!
//! Methods under test: `str().contains`, `str().starts_with`,
//! `str().ends_with`, `str().upper`, `str().lower`, `str().strip`,
//! `str().lstrip`, `str().rstrip`, `str().len`, `str().byte_count`.
//!
//! All tests require a GPU.

use cudf::{DataType, Exec, StringSliceOptions, StripSide, TypeId};

use crate::common::{execution_budget_token, nullable_series, series};

// ===========================================================================
// Case conversion
// ===========================================================================

fn test_str_lower() {
    let s = series("s", &["Hello", "WORLD", "Foo123"]).unwrap();
    let result = s.on(Exec::RETAINED).str().lower().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo123"]);
}

fn test_str_upper() {
    let s = series("s", &["Hello", "world", "Foo123"]).unwrap();
    let result = s.on(Exec::RETAINED).str().upper().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["HELLO", "WORLD", "FOO123"]);
}

fn test_str_upper_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().upper().unwrap();
    let vals = result.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("HELLO".to_string()), None, Some("WORLD".to_string())]);
}

fn test_str_lower_empty() {
    let s = series::<&str>("s", &[]).unwrap();
    let result = s.on(Exec::RETAINED).str().lower().unwrap();
    assert!(result.is_empty());
}

// ===========================================================================
// Contains
// ===========================================================================

fn test_str_contains_basic() {
    let s = series("s", &["hello world", "goodbye", "hello there"]).unwrap();
    let result = s.on(Exec::RETAINED).str().contains("hello").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

fn test_str_contains_empty_pattern_matches_all() {
    let s = series("s", &["abc", "def"]).unwrap();
    let result = s.on(Exec::RETAINED).str().contains("").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true]);
}

fn test_str_contains_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().contains("hello").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

// ===========================================================================
// SQL LIKE
// ===========================================================================

fn test_str_like_sql_wildcards_empty_pattern_escape_and_nulls() {
    let s = nullable_series(
        "s",
        &["abc", "axc", "abbc", "a_c", "", "literal%", "ignored"],
        &[true, true, true, true, true, true, false],
    )
    .unwrap();

    let one_char = s.on(Exec::RETAINED).str().like("a_c").unwrap();
    assert_eq!(one_char.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(
        one_char.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(true), Some(false), Some(true), Some(false), Some(false), None]
    );

    let prefix = s.on(Exec::RETAINED).str().like("a%").unwrap();
    assert_eq!(
        prefix.to_optional_vec::<bool>().unwrap(),
        vec![Some(true), Some(true), Some(true), Some(true), Some(false), Some(false), None]
    );

    let escaped = s.on(Exec::RETAINED).str().like_escape("literal/%", "/").unwrap();
    assert_eq!(
        escaped.to_optional_vec::<bool>().unwrap(),
        vec![Some(false), Some(false), Some(false), Some(false), Some(false), Some(true), None]
    );

    let empty = s.on(Exec::RETAINED).str().like("").unwrap();
    assert_eq!(
        empty.to_optional_vec::<bool>().unwrap(),
        vec![Some(false), Some(false), Some(false), Some(false), Some(true), Some(false), None]
    );
}

fn test_str_like_escape_rejects_multi_character_escape() {
    let s = series("s", &["a_c"]).unwrap();
    let err = s.on(Exec::RETAINED).str().like_escape("a//_c", "//").unwrap_err();
    assert_eq!(err.code(), cudf::ErrorCode::ForeignInvalidArgument);
}

// ===========================================================================
// Starts with / ends with
// ===========================================================================

fn test_str_startswith_basic() {
    let s = series("s", &["hello", "help", "world"]).unwrap();
    let result = s.on(Exec::RETAINED).str().starts_with("hel").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true, false]);
}

fn test_str_endswith_basic() {
    let s = series("s", &["hello", "fellow", "world"]).unwrap();
    let result = s.on(Exec::RETAINED).str().ends_with("llo").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, false]);
}

fn test_str_startswith_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().starts_with("hel").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

fn test_str_endswith_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().ends_with("rld").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

// ===========================================================================
// Slice strings
// ===========================================================================

fn test_str_slice_zero_based_unicode_open_ended_step_and_nulls() {
    let s =
        nullable_series("s", &["abcdef", "\u{4F60}\u{597D}\u{4E16}\u{754C}", "ignored"], &[true, true, false]).unwrap();

    let middle = s.on(Exec::RETAINED).str().slice(StringSliceOptions::new(Some(1), Some(3))).unwrap();
    assert_eq!(middle.dtype().unwrap().id(), TypeId::String);
    assert_eq!(
        middle.to_optional_strings().unwrap(),
        vec![Some("bc".to_string()), Some("\u{597D}\u{4E16}".to_string()), None]
    );

    let open_ended = s.on(Exec::RETAINED).str().slice(StringSliceOptions::new(Some(2), None)).unwrap();
    assert_eq!(
        open_ended.to_optional_strings().unwrap(),
        vec![Some("cdef".to_string()), Some("\u{4E16}\u{754C}".to_string()), None]
    );

    let stepped = s.on(Exec::RETAINED).str().slice(StringSliceOptions::new(None, None).with_step(2)).unwrap();
    assert_eq!(
        stepped.to_optional_strings().unwrap(),
        vec![Some("ace".to_string()), Some("\u{4F60}\u{4E16}".to_string()), None]
    );
}

fn test_str_slice_rejects_zero_step() {
    let s = series("s", &["abc"]).unwrap();
    let err = s.on(Exec::RETAINED).str().slice(StringSliceOptions::new(None, None).with_step(0)).unwrap_err();
    assert_eq!(err.code(), cudf::ErrorCode::ForeignInvalidArgument);
}

// ===========================================================================
// Strip
// ===========================================================================

fn test_str_strip_whitespace() {
    let s = series(
        "s",
        &[
            "  hello  ",
            "\tworld
",
            "  foo",
        ],
    )
    .unwrap();
    let result = s.on(Exec::RETAINED).str().strip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo"]);
}

fn test_str_lstrip_whitespace() {
    let s = series("s", &["  hello  ", "\tworld"]).unwrap();
    let result = s.on(Exec::RETAINED).str().lstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello  ", "world"]);
}

fn test_str_rstrip_whitespace() {
    let s = series("s", &["  hello  ", "world\t"]).unwrap();
    let result = s.on(Exec::RETAINED).str().rstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["  hello", "world"]);
}

fn test_str_strip_specific_chars() {
    let s = series("s", &["xxhelloxx", "xworld"]).unwrap();
    let result = s.on(Exec::RETAINED).str().strip_chars(StripSide::Both, "x").unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world"]);
}

fn test_str_strip_preserves_nulls() {
    let s = nullable_series("s", &["  hi  ", "x", "  there  "], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().strip().unwrap();
    let vals = result.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("hi".to_string()), None, Some("there".to_string())]);
}

// ===========================================================================
// String length
// ===========================================================================

fn test_str_len_basic() {
    let s = series("s", &["hello", "hi", ""]).unwrap();
    let result = s.on(Exec::RETAINED).str().len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![5, 2, 0]);
}

fn test_str_len_utf8() {
    // Multi-byte characters: each CJK character is 1 character but 3 bytes
    let s = series("s", &["abc", "\u{4F60}\u{597D}"]).unwrap(); // "你好"
    let result = s.on(Exec::RETAINED).str().len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 2]); // character count, not byte count
}

fn test_str_byte_count_utf8() {
    let s = series("s", &["abc", "\u{4F60}\u{597D}"]).unwrap();
    let result = s.on(Exec::RETAINED).str().byte_count().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 6]); // byte count: 3 bytes per CJK char
}

fn test_str_len_preserves_nulls() {
    let s = nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().len().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(5), None, Some(5)]);
}

fn test_str_len_empty() {
    let s = series::<&str>("s", &[]).unwrap();
    let result = s.on(Exec::RETAINED).str().len().unwrap();
    assert!(result.is_empty());
}

// ===========================================================================
// replace_with_backrefs size preflight
// ===========================================================================

fn test_replace_with_backrefs_output_size_matches_executed_column() {
    use cudf::{DataFrame, REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT, ops::expr::col};

    assert_eq!(REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT, "regexp_replace_output_size_preflight");

    let s = series("s", &["https://www.example.com/path", "http://other.test/x", "no-match"]).unwrap();
    let pattern = r"^https?://(?:www\.)?([^/]+)/.*$";
    let replacement = r"\1";

    let preflight = s.on(Exec::RETAINED).str().replace_with_backrefs_output_size(pattern, replacement).unwrap();
    let df = DataFrame::from_series([("s", s)]).unwrap();
    let executed = df
        .on(Exec::RETAINED)
        .with_columns(vec![col("s").str().replace_with_backrefs(pattern, replacement).alias("out")])
        .unwrap()
        .series("out")
        .unwrap();
    assert_eq!(preflight.retained_output_bytes(), executed.device_allocated_size_bytes().unwrap());
    assert!(preflight.requested_bytes_upper_bound() >= preflight.retained_output_bytes());
    assert!(preflight.required_budget_bytes() > 0);
}

// ===========================================================================
// prepared replace_with_backrefs prepare / execute
// ===========================================================================

fn test_prepared_replace_with_backrefs_semantics_and_facts() {
    let s = nullable_series(
        "s",
        &["https://www.example.com/path", "http://other.test/x", "no-match", "https://expand.example/long", ""],
        &[true, true, true, true, false],
    )
    .unwrap();
    let pattern = r"^https?://(?:www\.)?([^/]+)/.*$";
    let replacement = r"\1";

    let prepared = s.prepare_replace_with_backrefs(pattern, replacement).unwrap();
    let facts = prepared.output_size_facts();
    assert!(facts.requested_execute_bytes_upper_bound() >= facts.retained_output_bytes());
    assert!(facts.peak_execute_workspace_bytes() > 0);
    assert!(facts.required_budget_bytes() > 0);
    assert!(facts.retained_state_bytes() > 0 || s.is_empty());
    let out = prepared.execute().unwrap();
    assert_eq!(facts.retained_output_bytes(), out.device_allocated_size_bytes().unwrap());

    let vals = out.to_optional_strings().unwrap();
    assert_eq!(
        vals,
        vec![
            Some("example.com".to_string()),
            Some("other.test".to_string()),
            Some("no-match".to_string()),
            Some("expand.example".to_string()),
            None,
        ]
    );
}

fn test_prepared_replace_with_backrefs_empty_and_expanding_capture() {
    let empty = series::<&str>("s", &[]).unwrap();
    let prepared = empty.prepare_replace_with_backrefs(r"(a)", r"x\1y").unwrap();
    let facts = prepared.output_size_facts();
    assert_eq!(facts.retained_output_bytes(), 0);
    let out = prepared.execute().unwrap();
    assert!(out.is_empty());

    let s = series("s", &["a1", "zz", "a99"]).unwrap();
    let prepared = s.prepare_replace_with_backrefs(r"a(\d+)", r"X${1}Y").unwrap();
    let out = prepared.execute().unwrap();
    assert_eq!(out.to_strings_strict().unwrap(), vec!["X1Y", "zz", "X99Y"]);
}

fn test_replace_with_backrefs_preflight_exact_budget_covers_prepare_and_execute() {
    use cudf::{ExecutionBudgetRequest, Runtime};

    let runtime = Runtime::new().unwrap();
    let s = runtime.series(&["abc-123", "def-456"]).unwrap();
    let pattern = r"([a-z]+)-(\d+)";
    let replacement = r"\2/\1";
    let preflight = s.on(Exec::RETAINED).str().replace_with_backrefs_output_size(pattern, replacement).unwrap();
    assert_eq!(preflight.retained_output_bytes(), 26);
    assert!(preflight.requested_bytes_upper_bound() >= preflight.retained_output_bytes());
    assert!(preflight.required_budget_bytes() > 0);

    {
        let token = execution_budget_token(
            &runtime,
            ExecutionBudgetRequest::new(preflight.required_budget_bytes())
                .with_boundary("replace_backrefs_preflight_exact"),
        )
        .unwrap();
        let (prepared, _) =
            s.prepare_replace_with_backrefs_with_budget(pattern, replacement, &token).unwrap().into_parts();
        let (out, _) = prepared.execute_with_budget(&token).unwrap().into_parts();
        assert_eq!(out.to_strings_strict().unwrap(), vec!["123/abc", "456/def"]);
    }

    // The aligned bound is exact for this focused case: one byte less must
    // fail closed in prepare or execute.
    let token = execution_budget_token(
        &runtime,
        ExecutionBudgetRequest::new(preflight.required_budget_bytes() - 1)
            .with_boundary("replace_backrefs_preflight_minus_one"),
    )
    .unwrap();
    match s.prepare_replace_with_backrefs_with_budget(pattern, replacement, &token) {
        Err(_) => {},
        Ok(prepared) => {
            let (prepared, _) = prepared.into_parts();
            assert!(prepared.execute_with_budget(&token).is_err());
        },
    }
}

fn test_prepared_replace_with_backrefs_exact_execute_budget_and_drop_balance() {
    use cudf::{ExecutionBudgetRequest, Runtime};

    let runtime = Runtime::new().unwrap();
    let s = runtime.series(&["abc-123", "def-456"]).unwrap();
    let pattern = r"([a-z]+)-(\d+)";
    let replacement = r"\2/\1";

    // Unused prepared handle must drop cleanly (no double-free / leak panic path).
    {
        let prepared = s.prepare_replace_with_backrefs(pattern, replacement).unwrap();
        assert!(prepared.output_size_facts().required_budget_bytes() > 0);
        drop(prepared);
    }

    let prepared = s.prepare_replace_with_backrefs(pattern, replacement).unwrap();
    let facts = prepared.output_size_facts();
    {
        let token = execution_budget_token(
            &runtime,
            ExecutionBudgetRequest::new(facts.required_budget_bytes()).with_boundary("prepared_replace_execute"),
        )
        .unwrap();
        let (out, report) = prepared.execute_with_budget(&token).unwrap().into_parts();
        assert_eq!(out.to_strings_strict().unwrap(), vec!["123/abc", "456/def"]);
        assert_eq!(report.output_bytes(), Some(out.device_allocated_size_bytes().unwrap()));
        assert_eq!(report.retained_bytes(), Some(facts.retained_state_bytes()));
    }

    // One byte below the prepared execute bound must fail closed without
    // leaving a leaked handle.
    // The previous token must be dropped first because stream reservations are exclusive.
    let prepared = s.prepare_replace_with_backrefs(pattern, replacement).unwrap();
    let required = prepared.output_size_facts().required_budget_bytes();
    let tiny = execution_budget_token(
        &runtime,
        ExecutionBudgetRequest::new(required - 1).with_boundary("prepared_replace_execute_minus_one"),
    )
    .unwrap();
    let err = prepared.execute_with_budget(&tiny).unwrap_err();
    assert_eq!(err.operation(), "series::prepared_replace_with_backrefs::execute_with_budget");
}

// ===========================================================================
// try_to_integers (SQL TRY_CAST string -> integer)
// ===========================================================================

fn test_str_try_to_integers_int32_parses_valid_rejects_invalid() {
    // Valid, valid, parse-failure (alpha), overflow (> i32::MAX), negative.
    let s = series("s", &["42", "0", "abc", "2147483648", "-17"]).unwrap();
    let result = s.on(Exec::RETAINED).str().try_to_integers(&DataType::new(TypeId::Int32)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(42), Some(0), None, None, Some(-17)]);
}

fn test_str_try_to_integers_propagates_input_nulls() {
    let s = nullable_series("s", &["123", "x", "45"], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).str().try_to_integers(&DataType::new(TypeId::Int32)).unwrap();
    assert_eq!(result.to_optional_vec::<i32>().unwrap(), vec![Some(123), None, Some(45)]);
}

fn test_str_try_to_integers_target_type_overflow_yields_null() {
    // 200 fits in UInt8; 256 overflows; -1 is rejected for an unsigned target.
    let s = series("s", &["200", "256", "-1"]).unwrap();
    let result = s.on(Exec::RETAINED).str().try_to_integers(&DataType::new(TypeId::UInt8)).unwrap();
    assert_eq!(result.to_optional_vec::<u8>().unwrap(), vec![Some(200), None, None]);
}

fn test_str_try_to_integers_empty_input() {
    let s = series::<&str>("s", &[]).unwrap();
    let result = s.on(Exec::RETAINED).str().try_to_integers(&DataType::new(TypeId::Int64)).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int64);
}

fn test_str_try_to_integers_rejects_non_integer_target_type() {
    let s = series("s", &["1.5"]).unwrap();
    let err = s
        .on(Exec::RETAINED)
        .str()
        .try_to_integers(&DataType::new(TypeId::Float64))
        .expect_err("Float64 target must be rejected");
    assert!(format!("{err}").contains("integral"), "error should mention integral target: {err}");
}

fn test_expr_try_cast_dispatches_to_strings_try_to_integers() {
    use cudf::{DataFrame, ops::expr::col};

    let s = series("s", &["7", "bad", "11"]).unwrap();
    let df = DataFrame::from_series([("s", s)]).unwrap();
    let result = df.on(Exec::RETAINED).with_columns(vec![col("s").try_cast(TypeId::Int32).alias("v")]).unwrap();
    let series = result.series("v").unwrap();
    assert_eq!(series.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(series.to_optional_vec::<i32>().unwrap(), vec![Some(7), None, Some(11)]);
}

fn test_expr_try_cast_unsupported_pair_returns_error() {
    use cudf::{DataFrame, ops::expr::col};

    let s = series("s", &["7"]).unwrap();
    let df = DataFrame::from_series([("s", s)]).unwrap();
    let err = df
        .on(Exec::RETAINED)
        .with_columns(vec![col("s").try_cast(TypeId::Float64).alias("v")])
        .expect_err("Float64 target must be rejected by TRY_CAST expr");
    assert!(format!("{err}").contains("STRING"), "error should mention STRING source: {err}");
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("strings_tests::test_str_lower", test_str_lower),
    ("strings_tests::test_str_upper", test_str_upper),
    ("strings_tests::test_str_upper_preserves_nulls", test_str_upper_preserves_nulls),
    ("strings_tests::test_str_lower_empty", test_str_lower_empty),
    ("strings_tests::test_str_contains_basic", test_str_contains_basic),
    ("strings_tests::test_str_contains_empty_pattern_matches_all", test_str_contains_empty_pattern_matches_all),
    ("strings_tests::test_str_contains_preserves_nulls", test_str_contains_preserves_nulls),
    (
        "strings_tests::test_str_like_sql_wildcards_empty_pattern_escape_and_nulls",
        test_str_like_sql_wildcards_empty_pattern_escape_and_nulls,
    ),
    (
        "strings_tests::test_str_like_escape_rejects_multi_character_escape",
        test_str_like_escape_rejects_multi_character_escape,
    ),
    ("strings_tests::test_str_startswith_basic", test_str_startswith_basic),
    ("strings_tests::test_str_endswith_basic", test_str_endswith_basic),
    ("strings_tests::test_str_startswith_preserves_nulls", test_str_startswith_preserves_nulls),
    ("strings_tests::test_str_endswith_preserves_nulls", test_str_endswith_preserves_nulls),
    (
        "strings_tests::test_str_slice_zero_based_unicode_open_ended_step_and_nulls",
        test_str_slice_zero_based_unicode_open_ended_step_and_nulls,
    ),
    ("strings_tests::test_str_slice_rejects_zero_step", test_str_slice_rejects_zero_step),
    ("strings_tests::test_str_strip_whitespace", test_str_strip_whitespace),
    ("strings_tests::test_str_lstrip_whitespace", test_str_lstrip_whitespace),
    ("strings_tests::test_str_rstrip_whitespace", test_str_rstrip_whitespace),
    ("strings_tests::test_str_strip_specific_chars", test_str_strip_specific_chars),
    ("strings_tests::test_str_strip_preserves_nulls", test_str_strip_preserves_nulls),
    ("strings_tests::test_str_len_basic", test_str_len_basic),
    ("strings_tests::test_str_len_utf8", test_str_len_utf8),
    ("strings_tests::test_str_byte_count_utf8", test_str_byte_count_utf8),
    ("strings_tests::test_str_len_preserves_nulls", test_str_len_preserves_nulls),
    ("strings_tests::test_str_len_empty", test_str_len_empty),
    (
        "strings_tests::test_replace_with_backrefs_output_size_matches_executed_column",
        test_replace_with_backrefs_output_size_matches_executed_column,
    ),
    (
        "strings_tests::test_prepared_replace_with_backrefs_semantics_and_facts",
        test_prepared_replace_with_backrefs_semantics_and_facts,
    ),
    (
        "strings_tests::test_prepared_replace_with_backrefs_empty_and_expanding_capture",
        test_prepared_replace_with_backrefs_empty_and_expanding_capture,
    ),
    (
        "strings_tests::test_replace_with_backrefs_preflight_exact_budget_covers_prepare_and_execute",
        test_replace_with_backrefs_preflight_exact_budget_covers_prepare_and_execute,
    ),
    (
        "strings_tests::test_prepared_replace_with_backrefs_exact_execute_budget_and_drop_balance",
        test_prepared_replace_with_backrefs_exact_execute_budget_and_drop_balance,
    ),
    (
        "strings_tests::test_str_try_to_integers_int32_parses_valid_rejects_invalid",
        test_str_try_to_integers_int32_parses_valid_rejects_invalid,
    ),
    ("strings_tests::test_str_try_to_integers_propagates_input_nulls", test_str_try_to_integers_propagates_input_nulls),
    (
        "strings_tests::test_str_try_to_integers_target_type_overflow_yields_null",
        test_str_try_to_integers_target_type_overflow_yields_null,
    ),
    ("strings_tests::test_str_try_to_integers_empty_input", test_str_try_to_integers_empty_input),
    (
        "strings_tests::test_str_try_to_integers_rejects_non_integer_target_type",
        test_str_try_to_integers_rejects_non_integer_target_type,
    ),
    (
        "strings_tests::test_expr_try_cast_dispatches_to_strings_try_to_integers",
        test_expr_try_cast_dispatches_to_strings_try_to_integers,
    ),
    (
        "strings_tests::test_expr_try_cast_unsupported_pair_returns_error",
        test_expr_try_cast_unsupported_pair_returns_error,
    ),
];
