//! Integration tests for GPU string operations.
//!
//! Methods under test: `str().contains`, `str().starts_with`,
//! `str().ends_with`, `str().upper`, `str().lower`, `str().strip`,
//! `str().lstrip`, `str().rstrip`, `str().len`, `str().byte_count`.
//!
//! All tests require a GPU.

use cudf::StripSide;

// ===========================================================================
// Case conversion
// ===========================================================================

#[test]
fn test_str_lower() {
    let s = crate::common::series("s", &["Hello", "WORLD", "Foo123"]).unwrap();
    let result = s.str().lower().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo123"]);
}

#[test]
fn test_str_upper() {
    let s = crate::common::series("s", &["Hello", "world", "Foo123"]).unwrap();
    let result = s.str().upper().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["HELLO", "WORLD", "FOO123"]);
}

#[test]
fn test_str_lower_display_omits_input_name() {
    let s = crate::common::series("col", &["ABC"]).unwrap();
    let result = s.str().lower().unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("col"), "lower output should not expose schema names: {}", output);
}

#[test]
fn test_str_upper_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str().upper().unwrap();
    let vals = result.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("HELLO".to_string()), None, Some("WORLD".to_string())]);
}

#[test]
fn test_str_lower_empty() {
    let s = crate::common::series::<&str>("s", &[]).unwrap();
    let result = s.str().lower().unwrap();
    assert!(result.is_empty());
}

// ===========================================================================
// Contains
// ===========================================================================

#[test]
fn test_str_contains_basic() {
    let s = crate::common::series("s", &["hello world", "goodbye", "hello there"]).unwrap();
    let result = s.str().contains("hello").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

#[test]
fn test_str_contains_empty_pattern_matches_all() {
    let s = crate::common::series("s", &["abc", "def"]).unwrap();
    let result = s.str().contains("").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true]);
}

#[test]
fn test_str_contains_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str().contains("hello").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_str_contains_display_omits_input_name() {
    let s = crate::common::series("col", &["abc"]).unwrap();
    let result = s.str().contains("a").unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("col"), "contains output should not expose schema names: {}", output);
}

// ===========================================================================
// Starts with / ends with
// ===========================================================================

#[test]
fn test_str_startswith_basic() {
    let s = crate::common::series("s", &["hello", "help", "world"]).unwrap();
    let result = s.str().starts_with("hel").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true, false]);
}

#[test]
fn test_str_endswith_basic() {
    let s = crate::common::series("s", &["hello", "fellow", "world"]).unwrap();
    let result = s.str().ends_with("llo").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, false]);
}

#[test]
fn test_str_startswith_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str().starts_with("hel").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_str_endswith_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str().ends_with("rld").unwrap();
    let vals = result.to_optional_vec::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

// ===========================================================================
// Strip
// ===========================================================================

#[test]
fn test_str_strip_whitespace() {
    let s = crate::common::series(
        "s",
        &[
            "  hello  ",
            "\tworld
",
            "  foo",
        ],
    )
    .unwrap();
    let result = s.str().strip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo"]);
}

#[test]
fn test_str_lstrip_whitespace() {
    let s = crate::common::series("s", &["  hello  ", "\tworld"]).unwrap();
    let result = s.str().lstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello  ", "world"]);
}

#[test]
fn test_str_rstrip_whitespace() {
    let s = crate::common::series("s", &["  hello  ", "world\t"]).unwrap();
    let result = s.str().rstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["  hello", "world"]);
}

#[test]
fn test_str_strip_specific_chars() {
    let s = crate::common::series("s", &["xxhelloxx", "xworld"]).unwrap();
    let result = s.str().strip_chars(StripSide::Both, "x").unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world"]);
}

#[test]
fn test_str_strip_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["  hi  ", "x", "  there  "], &[true, false, true]).unwrap();
    let result = s.str().strip().unwrap();
    let vals = result.to_optional_strings().unwrap();
    assert_eq!(vals, vec![Some("hi".to_string()), None, Some("there".to_string())]);
}

#[test]
fn test_str_strip_display_omits_input_name() {
    let s = crate::common::series("col", &[" abc "]).unwrap();
    let result = s.str().strip().unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("col"), "strip output should not expose schema names: {}", output);
}

// ===========================================================================
// String length
// ===========================================================================

#[test]
fn test_str_len_basic() {
    let s = crate::common::series("s", &["hello", "hi", ""]).unwrap();
    let result = s.str().len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![5, 2, 0]);
}

#[test]
fn test_str_len_utf8() {
    // Multi-byte characters: each CJK character is 1 character but 3 bytes
    let s = crate::common::series("s", &["abc", "\u{4F60}\u{597D}"]).unwrap(); // "你好"
    let result = s.str().len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 2]); // character count, not byte count
}

#[test]
fn test_str_byte_count_utf8() {
    let s = crate::common::series("s", &["abc", "\u{4F60}\u{597D}"]).unwrap();
    let result = s.str().byte_count().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 6]); // byte count: 3 bytes per CJK char
}

#[test]
fn test_str_len_preserves_nulls() {
    let s = crate::common::nullable_series("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str().len().unwrap();
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(5), None, Some(5)]);
}

#[test]
fn test_str_len_empty() {
    let s = crate::common::series::<&str>("s", &[]).unwrap();
    let result = s.str().len().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_str_len_display_omits_input_name() {
    let s = crate::common::series("col", &["abc"]).unwrap();
    let result = s.str().len().unwrap();
    let output = format!("{}", result);
    assert!(!output.contains("col"), "len output should not expose schema names: {}", output);
}
