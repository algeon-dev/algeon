use std::panic::{self, AssertUnwindSafe};

use nexus_query_engine::exec::{ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness};

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

#[test]
fn resident_frame_stream_producer_release_is_idempotent() {
    let mut harness = ResidentFrameStreamProducerTestHarness::new(&[31, 37, 41]).unwrap();
    assert_eq!(harness.reserved_bytes(), 109);

    harness.release_producer_reservations();
    assert_eq!(harness.reserved_bytes(), 0);

    harness.release_producer_reservations();
    assert_eq!(harness.reserved_bytes(), 0);
}

#[test]
fn resident_frame_stream_producer_release_after_partial_iteration_keeps_yielded_counter_owned_by_yields() {
    let counter = YieldedFrameCounterTestHarness::new();
    let mut harness = ResidentFrameStreamProducerTestHarness::new(&[64]).unwrap();
    let mut first = counter.yield_bytes(11);
    let mut second = counter.yield_bytes(13);
    assert_eq!(counter.live_bytes(), 24);
    assert_eq!(harness.reserved_bytes(), 64);

    harness.release_producer_reservations();
    assert_eq!(harness.reserved_bytes(), 0);
    assert_eq!(counter.live_bytes(), 24);

    first.take_for_release();
    assert_eq!(counter.live_bytes(), 13);
    second.take_for_release();
    assert_eq!(counter.live_bytes(), 0);
}

#[test]
fn resident_frame_stream_collect_error_releases_producer_reservations() {
    let mut harness = ResidentFrameStreamProducerTestHarness::new_erroring(&[29, 31]).unwrap();
    assert_eq!(harness.reserved_bytes(), 60);

    let error = harness.collect_remaining_frames_owned_then_release().unwrap_err();
    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Internal);
    assert_eq!(harness.reserved_bytes(), 0);
}

#[test]
fn resident_frame_stream_producer_drop_without_release_decrements_tracker_and_panics_in_debug() {
    let mut harness = ResidentFrameStreamProducerTestHarness::new(&[19, 23]).unwrap();
    assert_eq!(harness.reserved_bytes(), 42);

    let result = panic::catch_unwind(AssertUnwindSafe(|| harness.drop_payload_without_release()));
    assert_eq!(harness.reserved_bytes(), 0);

    if cfg!(debug_assertions) {
        assert!(result.is_err(), "debug builds should flag unreleased stream producers");
    } else {
        assert!(result.is_ok(), "release builds should rely on ActiveReservation drop cleanup");
    }
}
