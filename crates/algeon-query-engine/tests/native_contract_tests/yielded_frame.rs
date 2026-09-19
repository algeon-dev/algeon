use std::panic::{self, AssertUnwindSafe};

use query_engine::exec::YieldedFrameCounterTestHarness;

#[test]
fn yielded_frame_explicit_release_decrements_live_counter_once() {
    let harness = YieldedFrameCounterTestHarness::new();
    assert_eq!(harness.live_bytes(), 0);

    let mut yielded = harness.yield_bytes(64);
    assert_eq!(harness.live_bytes(), 64);

    yielded.take_for_release();
    assert_eq!(harness.live_bytes(), 0);

    drop(yielded);
    assert_eq!(harness.live_bytes(), 0);
}

#[test]
fn yielded_frame_drop_without_release_decrements_live_counter_and_panics_in_debug() {
    let harness = YieldedFrameCounterTestHarness::new();
    let yielded = harness.yield_bytes(128);
    assert_eq!(harness.live_bytes(), 128);

    let result = panic::catch_unwind(AssertUnwindSafe(|| drop(yielded)));
    assert_eq!(harness.live_bytes(), 0);

    if cfg!(debug_assertions) {
        assert!(result.is_err(), "debug builds should flag dropped yielded frames");
    } else {
        assert!(result.is_ok(), "release builds should only decrement the counter");
    }
}

#[test]
fn yielded_frame_double_release_panics_without_double_decrement() {
    let harness = YieldedFrameCounterTestHarness::new();
    let mut yielded = harness.yield_bytes(256);
    assert_eq!(harness.live_bytes(), 256);

    yielded.take_for_release();
    assert_eq!(harness.live_bytes(), 0);

    let result = panic::catch_unwind(AssertUnwindSafe(|| yielded.take_for_release()));
    assert!(result.is_err(), "second release should panic");
    assert_eq!(harness.live_bytes(), 0);
}

#[test]
fn yielded_frame_multiple_yields_release_independently() {
    let harness = YieldedFrameCounterTestHarness::new();
    let mut first = harness.yield_bytes(11);
    let mut second = harness.yield_bytes(17);
    let mut third = harness.yield_bytes(23);
    assert_eq!(harness.live_bytes(), 51);

    second.take_for_release();
    assert_eq!(harness.live_bytes(), 34);
    first.take_for_release();
    assert_eq!(harness.live_bytes(), 23);
    third.take_for_release();
    assert_eq!(harness.live_bytes(), 0);
}
