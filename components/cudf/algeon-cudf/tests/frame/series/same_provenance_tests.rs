fn same_provenance_runtime_seeded_series_returns_true() {
    let a = crate::common::series("a", &[1i32, 2, 3]).expect("series a");
    let b = crate::common::series("b", &[4i32, 5, 6]).expect("series b");
    assert!(a.same_provenance(&b), "runtime-seeded series built through the same helper should share provenance");
}

fn same_provenance_self_returns_true() {
    let a = crate::common::series("x", &[10i64, 20]).expect("series");
    assert!(a.same_provenance(&a), "a series should share provenance with itself");
}

#[test]
fn same_provenance_tests_cases() {
    crate::common::run_named_cases(&[
        ("same_provenance_runtime_seeded_series_returns_true", same_provenance_runtime_seeded_series_returns_true),
        ("same_provenance_self_returns_true", same_provenance_self_returns_true),
    ]);
}
