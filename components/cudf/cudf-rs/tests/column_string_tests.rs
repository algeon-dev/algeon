//! Tests for string column creation and reading (Task 11).

use cudf::column::Column;
use cudf::types::TypeId;

#[test]
fn test_from_strings_basic() {
    let strings = &["hello", "world", "foo"];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert_eq!(view.null_count().unwrap(), 0);
}

#[test]
fn test_from_strings_roundtrip() {
    let strings = &["hello", "world", "foo"];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["hello", "world", "foo"]);
}

#[test]
fn test_from_strings_empty_strings() {
    let strings = &["", "hello", "", "world", ""];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 5);
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["", "hello", "", "world", ""]);
}

#[test]
fn test_from_strings_single() {
    let strings = &["only"];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 1);
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["only"]);
}

#[test]
fn test_from_strings_empty_column() {
    let strings: &[&str] = &[];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);

    let result = view.to_strings().unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_from_strings_nullable_basic() {
    let strings = &["hello", "world", "foo"];
    let validity = &[true, false, true];
    let col = Column::from_strings_nullable(strings, validity).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 1);

    let mask = view.validity().unwrap().unwrap();
    assert_eq!(mask, vec![true, false, true]);
}

#[test]
fn test_from_strings_nullable_roundtrip() {
    let strings = &["alpha", "beta", "gamma", "delta"];
    let validity = &[true, false, true, false];
    let col = Column::from_strings_nullable(strings, validity).unwrap();
    let view = col.view().unwrap();

    // Non-null strings should round-trip; null positions get empty strings
    let result = view.to_strings().unwrap();
    assert_eq!(result[0], "alpha");
    assert_eq!(result[2], "gamma");

    let mask = view.validity().unwrap().unwrap();
    assert_eq!(mask, vec![true, false, true, false]);
}

#[test]
fn test_from_strings_nullable_mismatched_lengths() {
    let strings = &["a", "b", "c"];
    let validity = &[true, false]; // wrong length
    let result = Column::from_strings_nullable(strings, validity);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
}

#[test]
fn test_from_strings_utf8() {
    let strings = &["héllo", "wörld", "日本語", "🦀"];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["héllo", "wörld", "日本語", "🦀"]);
}

#[test]
fn test_from_strings_column_has_children() {
    // String columns have at least 1 child (offsets)
    let col = Column::from_strings(&["a", "b"]).unwrap();
    let view = col.view().unwrap();
    assert!(view.num_children().unwrap() >= 1);
}

#[test]
fn test_to_strings_rejects_non_string_column() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();

    let result = view.to_strings();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_from_strings_copy_roundtrip() {
    let strings = &["deep", "copy", "test"];
    let col = Column::from_strings(strings).unwrap();
    let copy = col.copy().unwrap();

    // Original can be dropped; copy should still be valid
    drop(col);
    let view = copy.view().unwrap();
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["deep", "copy", "test"]);
}

#[test]
fn test_from_strings_all_empty() {
    let strings = &["", "", ""];
    let col = Column::from_strings(strings).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["", "", ""]);
}

#[test]
fn test_from_strings_nullable_all_null() {
    let strings = &["a", "b", "c"];
    let validity = &[false, false, false];
    let col = Column::from_strings_nullable(strings, validity).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.null_count().unwrap(), 3);
    let mask = view.validity().unwrap().unwrap();
    assert_eq!(mask, vec![false, false, false]);
}

#[test]
fn test_from_strings_nullable_no_nulls() {
    let strings = &["x", "y", "z"];
    let validity = &[true, true, true];
    let col = Column::from_strings_nullable(strings, validity).unwrap();
    let view = col.view().unwrap();

    // null_count should be 0, but column is still nullable (has mask)
    assert_eq!(view.null_count().unwrap(), 0);
    let result = view.to_strings().unwrap();
    assert_eq!(result, vec!["x", "y", "z"]);
}
