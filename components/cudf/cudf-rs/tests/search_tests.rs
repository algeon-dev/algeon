//! Tests for CU-2: contains (isin) and ColumnView::isin.

use cudf::{ErrorKind, column::Column, search};

#[test]
fn test_contains_basic_i32() {
    let haystack = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let needles = Column::from_slice(&[20i32, 40, 60, 80]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false, false]);
}

#[test]
fn test_contains_all_match() {
    let haystack = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let needles = Column::from_slice(&[3i32, 1, 2]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn test_contains_no_match() {
    let haystack = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let needles = Column::from_slice(&[4i32, 5, 6]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_contains_empty_needles() {
    let haystack = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let needles = Column::from_slice(&[] as &[i32]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.len().unwrap(), 0);
}

#[test]
fn test_contains_empty_haystack() {
    let haystack = Column::from_slice(&[] as &[i32]).unwrap();
    let needles = Column::from_slice(&[1i32, 2]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![false, false]);
}

#[test]
fn test_contains_with_duplicates_in_haystack() {
    let haystack = Column::from_slice(&[1i32, 1, 2, 2, 3]).unwrap();
    let needles = Column::from_slice(&[1i32, 3, 4]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn test_contains_float64() {
    let haystack = Column::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
    let needles = Column::from_slice(&[2.0f64, 4.0]).unwrap();

    let hv = haystack.view().unwrap();
    let nv = needles.view().unwrap();

    let result = search::contains(&hv, &nv).unwrap();
    let rv = result.view().unwrap();
    assert_eq!(rv.to_vec::<bool>().unwrap(), vec![true, false]);
}

// ---------------------------------------------------------------------------
// ColumnView::isin
// ---------------------------------------------------------------------------

#[test]
fn test_isin_i32_some_matches() {
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let values = Column::from_slice(&[2i32, 4]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false, true, false]);
}

#[test]
fn test_isin_string_column() {
    let col = Column::from_strings(&["apple", "banana", "cherry"]).unwrap();
    let values = Column::from_strings(&["banana", "date"]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn test_isin_no_matches() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let values = Column::from_slice(&[10i32, 20]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_isin_all_matches() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let values = Column::from_slice(&[3i32, 1, 2]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![true, true, true]);
}

#[test]
fn test_isin_empty_values() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let values = Column::from_slice(&[] as &[i32]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    // Empty values set → nothing can match → all false.
    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn test_isin_column_with_nulls() {
    let col = Column::from_slice_nullable(&[1i32, 0, 3, 0], &[true, false, true, false]).unwrap();
    let values = Column::from_slice(&[1i32, 3]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    // Non-null rows: 1→true, 3→true. Null rows produce null in output.
    assert_eq!(mv.len().unwrap(), 4);
    assert_eq!(mv.null_count().unwrap(), 2);

    let validity = mv.validity().unwrap().expect("mask should be nullable");
    assert_eq!(validity, vec![true, false, true, false]);

    // Valid positions should be true for 1 and 3.
    assert_eq!(mv.to_vec::<bool>().unwrap(), vec![true, false, true, false]);
}

#[test]
fn test_isin_empty_column() {
    let col = Column::from_slice(&[] as &[i32]).unwrap();
    let values = Column::from_slice(&[1i32, 2]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let mask = cv.isin(&vv).unwrap();
    let mv = mask.view().unwrap();
    assert_eq!(mv.len().unwrap(), 0);
}

#[test]
fn test_isin_type_mismatch_error_has_operation_context() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let values = Column::from_slice(&[1.0f64, 2.0]).unwrap();

    let cv = col.view().unwrap();
    let vv = values.view().unwrap();

    let err = match cv.isin(&vv) {
        Ok(_) => panic!("expected type mismatch error"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "column_view::isin");

    let ctx = err.context();
    assert!(ctx.iter().any(|(k, _)| *k == "input_type_id"));
    assert!(ctx.iter().any(|(k, _)| *k == "values_type_id"));
}
