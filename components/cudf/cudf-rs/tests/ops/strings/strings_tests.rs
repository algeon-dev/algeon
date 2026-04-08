//! Integration tests for GPU string operations.
//!
//! Methods under test: `str_contains`, `str_startswith`, `str_endswith`,
//! `str_upper`, `str_lower`, `str_strip`, `str_lstrip`, `str_rstrip`,
//! `str_len`, `str_byte_count`.
//!
//! All tests require a GPU.

use cudf::{Series, StripSide};

// ===========================================================================
// Case conversion
// ===========================================================================

#[test]
fn test_str_lower() {
    let s = Series::from_strings("s", &["Hello", "WORLD", "Foo123"]).unwrap();
    let result = s.str_lower().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo123"]);
}

#[test]
fn test_str_upper() {
    let s = Series::from_strings("s", &["Hello", "world", "Foo123"]).unwrap();
    let result = s.str_upper().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["HELLO", "WORLD", "FOO123"]);
}

#[test]
fn test_str_lower_preserves_name() {
    let s = Series::from_strings("col", &["ABC"]).unwrap();
    let result = s.str_lower().unwrap();
    assert_eq!(result.name(), Some("col"));
}

#[test]
fn test_str_upper_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str_upper().unwrap();
    let vals = result.to_strings_optional().unwrap();
    assert_eq!(vals, vec![Some("HELLO".to_string()), None, Some("WORLD".to_string())]);
}

#[test]
fn test_str_lower_empty() {
    let s = Series::from_strings("s", &[]).unwrap();
    let result = s.str_lower().unwrap();
    assert!(result.is_empty());
}

// ===========================================================================
// Contains
// ===========================================================================

#[test]
fn test_str_contains_basic() {
    let s = Series::from_strings("s", &["hello world", "goodbye", "hello there"]).unwrap();
    let result = s.str_contains("hello").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, true]);
}

#[test]
fn test_str_contains_empty_pattern_matches_all() {
    let s = Series::from_strings("s", &["abc", "def"]).unwrap();
    let result = s.str_contains("").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true]);
}

#[test]
fn test_str_contains_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str_contains("hello").unwrap();
    let vals = result.to_vec_optional::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_str_contains_preserves_name() {
    let s = Series::from_strings("col", &["abc"]).unwrap();
    let result = s.str_contains("a").unwrap();
    assert_eq!(result.name(), Some("col"));
}

// ===========================================================================
// Starts with / ends with
// ===========================================================================

#[test]
fn test_str_startswith_basic() {
    let s = Series::from_strings("s", &["hello", "help", "world"]).unwrap();
    let result = s.str_startswith("hel").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, true, false]);
}

#[test]
fn test_str_endswith_basic() {
    let s = Series::from_strings("s", &["hello", "fellow", "world"]).unwrap();
    let result = s.str_endswith("llo").unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![true, false, false]);
}

#[test]
fn test_str_startswith_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str_startswith("hel").unwrap();
    let vals = result.to_vec_optional::<bool>().unwrap();
    assert_eq!(vals, vec![Some(true), None, Some(false)]);
}

#[test]
fn test_str_endswith_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str_endswith("rld").unwrap();
    let vals = result.to_vec_optional::<bool>().unwrap();
    assert_eq!(vals, vec![Some(false), None, Some(true)]);
}

// ===========================================================================
// Strip
// ===========================================================================

#[test]
fn test_str_strip_whitespace() {
    let s = Series::from_strings("s", &["  hello  ", "\tworld\n", "  foo"]).unwrap();
    let result = s.str_strip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world", "foo"]);
}

#[test]
fn test_str_lstrip_whitespace() {
    let s = Series::from_strings("s", &["  hello  ", "\tworld"]).unwrap();
    let result = s.str_lstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello  ", "world"]);
}

#[test]
fn test_str_rstrip_whitespace() {
    let s = Series::from_strings("s", &["  hello  ", "world\t"]).unwrap();
    let result = s.str_rstrip().unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["  hello", "world"]);
}

#[test]
fn test_str_strip_specific_chars() {
    let s = Series::from_strings("s", &["xxhelloxx", "xworld"]).unwrap();
    let result = s.str_strip_chars(StripSide::Both, "x").unwrap();
    let vals = result.to_strings_strict().unwrap();
    assert_eq!(vals, vec!["hello", "world"]);
}

#[test]
fn test_str_strip_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["  hi  ", "x", "  there  "], &[true, false, true]).unwrap();
    let result = s.str_strip().unwrap();
    let vals = result.to_strings_optional().unwrap();
    assert_eq!(vals, vec![Some("hi".to_string()), None, Some("there".to_string())]);
}

#[test]
fn test_str_strip_preserves_name() {
    let s = Series::from_strings("col", &[" abc "]).unwrap();
    let result = s.str_strip().unwrap();
    assert_eq!(result.name(), Some("col"));
}

// ===========================================================================
// String length
// ===========================================================================

#[test]
fn test_str_len_basic() {
    let s = Series::from_strings("s", &["hello", "hi", ""]).unwrap();
    let result = s.str_len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![5, 2, 0]);
}

#[test]
fn test_str_len_utf8() {
    // Multi-byte characters: each CJK character is 1 character but 3 bytes
    let s = Series::from_strings("s", &["abc", "\u{4F60}\u{597D}"]).unwrap(); // "你好"
    let result = s.str_len().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 2]); // character count, not byte count
}

#[test]
fn test_str_byte_count_utf8() {
    let s = Series::from_strings("s", &["abc", "\u{4F60}\u{597D}"]).unwrap();
    let result = s.str_byte_count().unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![3, 6]); // byte count: 3 bytes per CJK char
}

#[test]
fn test_str_len_preserves_nulls() {
    let s = Series::from_strings_nullable("s", &["hello", "x", "world"], &[true, false, true]).unwrap();
    let result = s.str_len().unwrap();
    let vals = result.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(5), None, Some(5)]);
}

#[test]
fn test_str_len_empty() {
    let s = Series::from_strings("s", &[]).unwrap();
    let result = s.str_len().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_str_len_preserves_name() {
    let s = Series::from_strings("col", &["abc"]).unwrap();
    let result = s.str_len().unwrap();
    assert_eq!(result.name(), Some("col"));
}
