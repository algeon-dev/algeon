use crate::common::series;

fn same_provenance_runtime_seeded_series_returns_true() {
    let a = series("a", &[1i32, 2, 3]).expect("series a");
    let b = series("b", &[4i32, 5, 6]).expect("series b");
    assert!(a.same_provenance(&b), "runtime-seeded series built through the same helper should share provenance");
}

fn same_provenance_self_returns_true() {
    let a = series("x", &[10i64, 20]).expect("series");
    assert!(a.same_provenance(&a), "a series should share provenance with itself");
}

pub(crate) const CASES: &[(&str, fn())] = &[
    (
        "frame_contracts::same_provenance_tests::same_provenance_runtime_seeded_series_returns_true",
        same_provenance_runtime_seeded_series_returns_true,
    ),
    ("frame_contracts::same_provenance_tests::same_provenance_self_returns_true", same_provenance_self_returns_true),
];
