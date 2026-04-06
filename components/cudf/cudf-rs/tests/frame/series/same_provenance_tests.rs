use cudf::Series;

#[test]
fn same_provenance_ambient_default_returns_true() {
    let a = Series::new("a", &[1i32, 2, 3]).expect("series a");
    let b = Series::new("b", &[4i32, 5, 6]).expect("series b");
    assert!(a.same_provenance(&b), "ambient-default series should share provenance");
}

#[test]
fn same_provenance_self_returns_true() {
    let a = Series::new("x", &[10i64, 20]).expect("series");
    assert!(a.same_provenance(&a), "a series should share provenance with itself");
}
