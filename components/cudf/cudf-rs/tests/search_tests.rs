//! Tests for CU-2: contains (isin).

use cudf::column::Column;
use cudf::search;

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
