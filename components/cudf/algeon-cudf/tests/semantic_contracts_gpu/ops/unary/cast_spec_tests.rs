//! Cast behavior across type pairs, including edge cases that the basic unary tests do not cover.

use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, AsArray, Decimal128Array, Float32Array, Float64Array, Int64Array},
    compute::{CastOptions, cast, cast_with_options},
    datatypes::{DataType as ArrowDataType, Field, Int64Type, Schema, TimeUnit},
    error::ArrowError,
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, Exec, Runtime, TimestampResolution, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::{
        expr::col,
        unary::{decimal_rescale_temporary_workspace_reservation_bytes, is_supported_cast},
    },
};

use crate::common::{nullable_series, series, series_opt};

fn decimal128_series_from_arrow(values: Vec<Option<i128>>, precision: u8, scale: i8) -> cudf::Series {
    let amount = Decimal128Array::from(values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(precision, scale), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let df = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    df.series("amount").unwrap()
}

// Integer width widening is lossless.
fn test_cast_i8_to_i16() {
    let s = series::<i8>("x", &[i8::MIN, 0, i8::MAX]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![i8::MIN as i16, 0, i8::MAX as i16]);
}

fn test_cast_i16_to_i32() {
    let s = series::<i16>("x", &[i16::MIN, 0, i16::MAX]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![i16::MIN as i32, 0, i16::MAX as i32]);
}

fn test_cast_i32_to_i64() {
    let s = series::<i32>("x", &[i32::MIN, 0, i32::MAX]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![i32::MIN as i64, 0, i32::MAX as i64]);
}

fn test_cast_i32_to_f32() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

fn test_cast_f32_to_i32_truncates() {
    let s = series::<f32>("x", &[1.9, -2.7, 0.1]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, -2, 0]);
}

fn test_cast_f64_to_f32() {
    let s = series::<f64>("x", &[1.5, 2.5, 3.5]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float32).unwrap();
    let vals = result.to_vec_strict::<f32>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

fn test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls() {
    let amount = decimal128_series_from_arrow(vec![Some(12345), None, Some(-678)], 20, 2);

    let result = amount.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.to_optional_vec::<f64>().unwrap(), vec![Some(123.45), None, Some(-6.78)]);
}

fn test_cast_decimal64_to_f64_uses_decimal_scale() {
    let amount = decimal128_series_from_arrow(vec![Some(12345), Some(-678)], 20, 2)
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();

    let result = amount.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![123.45, -6.78]);
}

fn test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls() {
    let s = nullable_series::<f64>("x", &[123.45, 0.0, -6.78], &[true, false, true]).unwrap();

    let result = s.on(Exec::RETAINED).astype(DataType::fixed_point(TypeId::Decimal128, -2).unwrap()).unwrap();

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2).unwrap());
    assert_eq!(result.null_count(), 1);
    let batch = DataFrame::from_series([("x", result)]).unwrap().to_record_batch().unwrap();
    let values = batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(values.values().as_ref(), &[12345, 0, -678]);
    assert!(values.is_valid(0));
    assert!(values.is_null(1));
    assert!(values.is_valid(2));
}

fn test_cast_bool_to_i32() {
    let s = series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    let vals = result.to_vec_strict::<i32>().unwrap();
    assert_eq!(vals, vec![1, 0, 1]);
}

fn test_cast_i32_to_bool() {
    let s = series::<i32>("x", &[0, 1, 42, -1]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Bool8).unwrap();
    let vals = result.to_vec_strict::<bool>().unwrap();
    assert_eq!(vals, vec![false, true, true, true]);
}

fn test_cast_bool_to_i64() {
    let s = series::<bool>("x", &[false, true]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, 1]);
}

fn test_cast_u8_to_i16() {
    let s = series::<u8>("x", &[0, 127, 255]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int16).unwrap();
    let vals = result.to_vec_strict::<i16>().unwrap();
    assert_eq!(vals, vec![0, 127, 255]);
}

fn test_cast_u32_to_i64() {
    let s = series::<u32>("x", &[0, u32::MAX]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    let vals = result.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![0, u32::MAX as i64]);
}

fn test_cast_preserves_null_mask() {
    let s = nullable_series::<i32>("x", &[10, 0, 30], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 1);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![Some(10.0), None, Some(30.0)]);
}

fn test_cast_all_null_column() {
    let s = nullable_series::<i32>("x", &[0, 0], &[false, false]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();
    assert_eq!(result.null_count(), 2);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    let vals = result.to_optional_vec::<f64>().unwrap();
    assert_eq!(vals, vec![None, None]);
}

fn test_cast_empty_column() {
    let s = series::<i32>("x", &[]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
}

// Casting to the same type leaves the values unchanged.
fn test_cast_same_type() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
}

fn decimal_rescale_workspace_contract_is_scale_sensitive() {
    let scale_zero = DataType::fixed_point(TypeId::Decimal128, 0).unwrap();
    let scale_two = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();

    assert_eq!(decimal_rescale_temporary_workspace_reservation_bytes(&scale_zero, &scale_zero), Some(0));
    assert_eq!(decimal_rescale_temporary_workspace_reservation_bytes(&scale_zero, &scale_two), Some(256));
    assert_eq!(
        decimal_rescale_temporary_workspace_reservation_bytes(
            &DataType::fixed_point(TypeId::Decimal64, 0).unwrap(),
            &scale_two,
        ),
        None
    );
}

fn test_is_supported_cast_numeric_to_float() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    assert!(is_supported_cast(&from, &to).unwrap());
}

fn test_is_supported_cast_decimal_to_float() {
    let to = DataType::new(TypeId::Float64);
    assert!(is_supported_cast(&DataType::fixed_point(TypeId::Decimal64, -2).unwrap(), &to).unwrap());
    assert!(is_supported_cast(&DataType::fixed_point(TypeId::Decimal128, -2).unwrap(), &to).unwrap());
}

fn test_is_supported_cast_float_to_decimal() {
    let from = DataType::new(TypeId::Float64);
    assert!(is_supported_cast(&from, &DataType::fixed_point(TypeId::Decimal128, -2).unwrap()).unwrap());
}

fn test_is_supported_cast_float_to_int() {
    let from = DataType::new(TypeId::Float64);
    let to = DataType::new(TypeId::Int32);
    assert!(is_supported_cast(&from, &to).unwrap());
}

fn test_is_supported_cast_bool_to_int() {
    let from = DataType::new(TypeId::Bool8);
    let to = DataType::new(TypeId::Int32);
    assert!(is_supported_cast(&from, &to).unwrap());
}

// Integer-to-timestamp casts are supported by libcudf.
fn test_cast_i64_to_timestamp_microseconds() {
    let s = series::<i64>("x", &[0, 1_000_000, 1_618_000_000_000_000]).unwrap();
    let dur = s.on(Exec::RETAINED).astype(TypeId::DurationMicroseconds).unwrap();
    let result = dur.on(Exec::RETAINED).astype(TypeId::TimestampMicroseconds).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
    assert_eq!(result.len(), 3);
}

fn test_cast_timestamp_to_i64() {
    // Round-trip: i64 -> duration_us -> timestamp_us -> duration_us -> i64
    let s = series::<i64>("x", &[0, 1000, 2000]).unwrap();
    let dur = s.on(Exec::RETAINED).astype(TypeId::DurationMicroseconds).unwrap();
    let ts = dur.on(Exec::RETAINED).astype(TypeId::TimestampMicroseconds).unwrap();
    let back_dur = ts.on(Exec::RETAINED).astype(TypeId::DurationMicroseconds).unwrap();
    let back = back_dur.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

const EVERY_RESOLUTION: [TimestampResolution; 4] = [
    TimestampResolution::Seconds,
    TimestampResolution::Milliseconds,
    TimestampResolution::Microseconds,
    TimestampResolution::Nanoseconds,
];

/// The same counts as a timestamp series and as a duration series of one resolution.
fn temporal_series(counts: &cudf::Series, resolution: TimestampResolution) -> (cudf::Series, cudf::Series) {
    let durations = counts.on(Exec::RETAINED).astype(resolution.duration_type_id()).unwrap();
    let timestamps = durations.on(Exec::RETAINED).astype(resolution.type_id()).unwrap();
    (timestamps, durations)
}

fn test_cast_timestamp_and_duration_to_i64_returns_the_stored_count_in_the_types_own_unit() {
    let values = [0, 1, -1, 1_700_000_123_456, i64::MIN, i64::MAX, 7];
    let validity = [true, true, true, true, true, true, false];
    let expected = values.iter().zip(validity).map(|(value, valid)| valid.then_some(*value)).collect::<Vec<_>>();
    let counts = nullable_series::<i64>("x", &values, &validity).unwrap();

    for resolution in EVERY_RESOLUTION {
        let (timestamps, durations) = temporal_series(&counts, resolution);
        for (kind, temporal) in [("timestamp", timestamps), ("duration", durations)] {
            assert!(is_supported_cast(&temporal.dtype().unwrap(), &DataType::new(TypeId::Int64)).unwrap());
            let result = temporal.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
            assert_eq!(result.dtype().unwrap().id(), TypeId::Int64, "{kind} {resolution:?}");
            assert_eq!(result.to_optional_vec::<i64>().unwrap(), expected, "{kind} {resolution:?}");
        }
    }
}

fn test_cast_timestamp_and_duration_to_narrower_integers_check_the_stored_count() {
    for resolution in [TimestampResolution::Milliseconds, TimestampResolution::Nanoseconds] {
        let fits = series::<i64>("x", &[i64::from(i32::MIN), 0, i64::from(i32::MAX)]).unwrap();
        let (timestamps, durations) = temporal_series(&fits, resolution);
        for temporal in [timestamps, durations] {
            let kept = temporal.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
            assert_eq!(kept.to_vec_strict::<i32>().unwrap(), vec![i32::MIN, 0, i32::MAX], "{resolution:?}");
        }

        for (count, target, reason) in
            [(i64::from(i32::MAX) + 1, TypeId::Int32, "above_target_range"), (-1, TypeId::UInt64, "below_target_range")]
        {
            let (timestamps, durations) = temporal_series(&series::<i64>("x", &[0, count]).unwrap(), resolution);
            for temporal in [timestamps, durations] {
                let error = temporal.on(Exec::RETAINED).astype(target).expect_err("the count must not fit");
                assert_eq!(error.code(), ErrorCode::ValueOutOfRange, "{resolution:?} {target:?}");
                assert_eq!(error.detail_value("reason"), Some(reason), "{resolution:?} {target:?}");
            }
        }
    }
}

// Timestamp and duration casts between resolutions follow Arrow's multiply and truncate rules.
fn test_cast_integer_to_timestamp_reads_the_integer_as_the_tick_count() {
    let counts = [Some(0), Some(-1), Some(1_700_000_000_123), Some(i64::MAX), Some(i64::MIN), None];
    for resolution in EVERY_RESOLUTION {
        let source = series_opt::<i64>("x", &counts).unwrap();
        assert!(is_supported_cast(&source.dtype().unwrap(), &DataType::new(resolution.type_id())).unwrap());
        let timestamps = source.on(Exec::RETAINED).astype(resolution.type_id()).unwrap();
        assert_eq!(timestamps.dtype().unwrap().id(), resolution.type_id());
        let back = timestamps.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
        assert_eq!(back.to_optional_vec::<i64>().unwrap(), counts, "{resolution:?}");
    }

    let days = series::<i32>("x", &[19_000, -1]).unwrap().on(Exec::RETAINED).astype(TypeId::TimestampDays).unwrap();
    assert_eq!(days.dtype().unwrap().id(), TypeId::TimestampDays);
    let back = days.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![19_000, -1]);
}

fn arrow_time_unit(resolution: TimestampResolution) -> TimeUnit {
    match resolution {
        TimestampResolution::Seconds => TimeUnit::Second,
        TimestampResolution::Milliseconds => TimeUnit::Millisecond,
        TimestampResolution::Microseconds => TimeUnit::Microsecond,
        TimestampResolution::Nanoseconds => TimeUnit::Nanosecond,
        TimestampResolution::Days => unreachable!("Arrow durations and these timestamps have no day unit"),
    }
}

fn nanoseconds_per_tick(resolution: TimestampResolution) -> i64 {
    match resolution {
        TimestampResolution::Days => 86_400_000_000_000,
        TimestampResolution::Seconds => 1_000_000_000,
        TimestampResolution::Milliseconds => 1_000_000,
        TimestampResolution::Microseconds => 1_000,
        TimestampResolution::Nanoseconds => 1,
    }
}

fn arrow_temporal_type(resolution: TimestampResolution, timestamp: bool) -> ArrowDataType {
    let unit = arrow_time_unit(resolution);
    if timestamp { ArrowDataType::Timestamp(unit, None) } else { ArrowDataType::Duration(unit) }
}

/// The stored counts after Arrow's non-safe cast, which DataFusion uses for `CAST` and for coerced casts.
fn arrow_rescaled_counts(
    counts: &[Option<i64>],
    source: ArrowDataType,
    target: ArrowDataType,
) -> Result<Vec<Option<i64>>, ArrowError> {
    let source = cast(&Int64Array::from(counts.to_vec()), &source).unwrap();
    let rescaled = cast_with_options(&source, &target, &CastOptions { safe: false, ..Default::default() })?;
    Ok(cast(&rescaled, &ArrowDataType::Int64).unwrap().as_primitive::<Int64Type>().iter().collect())
}

fn rescaled_counts(
    counts: &[Option<i64>],
    from: TimestampResolution,
    to: TimestampResolution,
    timestamp: bool,
) -> cudf::Result<Vec<Option<i64>>> {
    let (timestamps, durations) = temporal_series(&series_opt::<i64>("x", counts)?, from);
    let source = if timestamp { timestamps } else { durations };
    let target = if timestamp { to.type_id() } else { to.duration_type_id() };
    let rescaled = source.on(Exec::RETAINED).astype(target)?;
    rescaled.on(Exec::RETAINED).astype(TypeId::Int64)?.to_optional_vec::<i64>()
}

fn finer_to_coarser_pairs() -> Vec<(TimestampResolution, TimestampResolution)> {
    let mut pairs = Vec::new();
    for (index, coarser) in EVERY_RESOLUTION.iter().enumerate() {
        pairs.extend(EVERY_RESOLUTION[index + 1..].iter().map(|finer| (*finer, *coarser)));
    }
    pairs
}

fn test_cast_temporal_to_a_coarser_resolution_truncates_toward_zero_like_arrow() {
    // Negative counts that are not a multiple of the divisor separate truncation from libcudf's floor.
    let counts = [
        Some(0),
        Some(1),
        Some(-1),
        Some(999),
        Some(-999),
        Some(1_000_000_001),
        Some(-1_000_000_001),
        Some(-1_500_000_000),
        Some(i64::MAX),
        Some(i64::MIN),
        None,
    ];
    for (from, to) in finer_to_coarser_pairs() {
        for timestamp in [true, false] {
            let expected = arrow_rescaled_counts(
                &counts,
                arrow_temporal_type(from, timestamp),
                arrow_temporal_type(to, timestamp),
            )
            .unwrap();
            let actual = rescaled_counts(&counts, from, to, timestamp).unwrap();
            assert_eq!(actual, expected, "{from:?} to {to:?}, timestamp={timestamp}");
        }
    }
}

fn test_cast_temporal_to_a_finer_resolution_matches_arrow_up_to_the_int64_limits() {
    for (to, from) in finer_to_coarser_pairs() {
        let factor = nanoseconds_per_tick(from) / nanoseconds_per_tick(to);
        let fits = [Some(i64::MAX / factor), Some(i64::MIN / factor), Some(0), Some(1), Some(-1), None];
        for timestamp in [true, false] {
            let (source, target) = (arrow_temporal_type(from, timestamp), arrow_temporal_type(to, timestamp));
            assert_eq!(
                rescaled_counts(&fits, from, to, timestamp).unwrap(),
                arrow_rescaled_counts(&fits, source.clone(), target.clone()).unwrap(),
                "{from:?} to {to:?}, timestamp={timestamp}"
            );

            for (count, reason) in
                [(i64::MAX / factor + 1, "above_target_range"), (i64::MIN / factor - 1, "below_target_range")]
            {
                arrow_rescaled_counts(&[Some(count)], source.clone(), target.clone())
                    .expect_err("Arrow fails where the product leaves Int64");
                let error = rescaled_counts(&[Some(0), Some(count)], from, to, timestamp)
                    .expect_err("the product must not fit Int64");
                assert_eq!(error.code(), ErrorCode::ValueOutOfRange, "{from:?} to {to:?}, timestamp={timestamp}");
                assert_eq!(error.detail_value("reason"), Some(reason), "{from:?} to {to:?}, timestamp={timestamp}");
            }

            // A null row never fails, whatever count the null slot holds.
            let (timestamps, durations) = temporal_series(
                &nullable_series::<i64>("x", &[i64::MAX / factor + 1, 1], &[false, true]).unwrap(),
                from,
            );
            let source = if timestamp { timestamps } else { durations };
            let target = if timestamp { to.type_id() } else { to.duration_type_id() };
            let rescaled = source.on(Exec::RETAINED).astype(target).unwrap();
            let counts = rescaled.on(Exec::RETAINED).astype(TypeId::Int64).unwrap().to_optional_vec::<i64>().unwrap();
            assert_eq!(counts, vec![None, Some(factor)], "{from:?} to {to:?}, timestamp={timestamp}");
        }
    }
}

fn test_cast_dates_to_timestamps_fail_where_the_product_leaves_int64_and_timestamps_to_dates_floor() {
    let days = TimestampResolution::Days;
    let nanos_per_day = nanoseconds_per_tick(days);
    let fits = series_opt::<i64>("x", &[Some(106_751), Some(-106_751), Some(0), None]).unwrap();
    let (dates, _) = temporal_series(&fits, days);
    let nanos = dates.on(Exec::RETAINED).astype(TypeId::TimestampNanoseconds).unwrap();
    let counts = nanos.on(Exec::RETAINED).astype(TypeId::Int64).unwrap().to_optional_vec::<i64>().unwrap();
    assert_eq!(counts, vec![Some(106_751 * nanos_per_day), Some(-106_751 * nanos_per_day), Some(0), None]);

    for (count, reason) in [(106_752, "above_target_range"), (-106_752, "below_target_range")] {
        let (dates, _) = temporal_series(&series::<i64>("x", &[0, count]).unwrap(), days);
        let error = dates.on(Exec::RETAINED).astype(TypeId::TimestampNanoseconds).expect_err("overflows Int64");
        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(error.detail_value("reason"), Some(reason));
    }

    // Arrow takes the calendar date of a timestamp, which floors.
    let counts = [Some(0), Some(-1), Some(86_399_999_999_999), Some(-86_400_000_000_000), Some(i64::MIN), None];
    let expected =
        arrow_rescaled_counts(&counts, ArrowDataType::Timestamp(TimeUnit::Nanosecond, None), ArrowDataType::Date32)
            .unwrap();
    let (timestamps, _) = temporal_series(&series_opt::<i64>("x", &counts).unwrap(), TimestampResolution::Nanoseconds);
    let dates = timestamps.on(Exec::RETAINED).astype(days.type_id()).unwrap();
    let actual = dates.on(Exec::RETAINED).astype(TypeId::Int64).unwrap().to_optional_vec::<i64>().unwrap();
    assert_eq!(actual, expected);
}

fn test_to_datetime_default_microseconds() {
    let s = series::<i64>("ts", &[0, 1_000_000]).unwrap();
    let result = s.on(Exec::RETAINED).to_datetime(None).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

fn test_to_datetime_explicit_milliseconds() {
    let s = series::<i64>("ts", &[0, 1000]).unwrap();
    let result = s.on(Exec::RETAINED).to_datetime(Some(TimestampResolution::Milliseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMilliseconds);
}

fn test_to_datetime_nanoseconds() {
    let s = series::<i64>("ts", &[0, 1_000_000_000]).unwrap();
    let result = s.on(Exec::RETAINED).to_datetime(Some(TimestampResolution::Nanoseconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampNanoseconds);
}

fn test_to_datetime_seconds() {
    let s = series::<i64>("ts", &[0, 86400]).unwrap();
    let result = s.on(Exec::RETAINED).to_datetime(Some(TimestampResolution::Seconds)).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampSeconds);
}

fn test_to_datetime_round_trip() {
    let s = series::<i64>("ts", &[0, 1000, 2000]).unwrap();
    let dt = s.on(Exec::RETAINED).to_datetime(Some(TimestampResolution::Microseconds)).unwrap();
    let dur = dt.on(Exec::RETAINED).astype(TypeId::DurationMicroseconds).unwrap();
    let back = dur.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(back.to_vec_strict::<i64>().unwrap(), vec![0, 1000, 2000]);
}

fn test_to_datetime_preserves_nulls() {
    let s = nullable_series::<i64>("ts", &[100, 0, 300], &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).to_datetime(None).unwrap();
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampMicroseconds);
}

fn test_to_numeric_f64_to_i64() {
    let s = series::<f64>("x", &[1.9, 2.0, 3.1]).unwrap();
    let result = s.on(Exec::RETAINED).to_numeric(TypeId::Int64).unwrap();
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![1, 2, 3]);
}

fn test_to_numeric_i32_to_f64() {
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).to_numeric(TypeId::Float64).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

fn test_to_numeric_bool_to_i32() {
    let s = series::<bool>("x", &[true, false, true]).unwrap();
    let result = s.on(Exec::RETAINED).to_numeric(TypeId::Int32).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 1]);
}

fn test_to_numeric_rejects_non_numeric_target() {
    let s = series::<i32>("x", &[1, 2]).unwrap();
    let err = s.on(Exec::RETAINED).to_numeric(TypeId::String).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::to_numeric");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
}

fn test_to_numeric_rejects_timestamp_target() {
    let s = series::<i32>("x", &[1, 2]).unwrap();
    let err = s.on(Exec::RETAINED).to_numeric(TypeId::TimestampMicroseconds).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "series::to_numeric");
}

fn assert_cast_out_of_range(input: cudf::Series, target: TypeId, reason: &str) {
    let source = format!("{:?}", input.dtype().unwrap().id()).to_lowercase();
    let error = input.on(Exec::RETAINED).astype(target).expect_err("the cast must reject the input");
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange, "{source} -> {target:?}");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.detail_value("reason"), Some(reason), "{source} -> {target:?}");
    assert_eq!(error.detail_value("source_type_id"), Some(source.as_str()));
    assert_eq!(error.detail_value("target_type_id"), Some(format!("{target:?}").to_lowercase().as_str()));
}

fn test_cast_integer_to_integer_accepts_target_extremes_and_rejects_one_past() {
    let kept = series::<i64>("x", &[i64::from(i32::MIN), 0, i64::from(i32::MAX)]).unwrap();
    let kept = kept.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    assert_eq!(kept.to_vec_strict::<i32>().unwrap(), vec![i32::MIN, 0, i32::MAX]);
    assert_cast_out_of_range(
        series::<i64>("x", &[0, i64::from(i32::MAX) + 1]).unwrap(),
        TypeId::Int32,
        "above_target_range",
    );
    assert_cast_out_of_range(
        series::<i64>("x", &[0, i64::from(i32::MIN) - 1]).unwrap(),
        TypeId::Int32,
        "below_target_range",
    );

    let kept = series::<u64>("x", &[0, i64::MAX as u64]).unwrap().on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(kept.to_vec_strict::<i64>().unwrap(), vec![0, i64::MAX]);
    assert_cast_out_of_range(series::<u64>("x", &[i64::MAX as u64 + 1]).unwrap(), TypeId::Int64, "above_target_range");

    let kept = series::<i64>("x", &[0, i64::MAX]).unwrap().on(Exec::RETAINED).astype(TypeId::UInt64).unwrap();
    assert_eq!(kept.to_vec_strict::<u64>().unwrap(), vec![0, i64::MAX as u64]);
    assert_cast_out_of_range(series::<i64>("x", &[-1]).unwrap(), TypeId::UInt64, "below_target_range");

    let kept = series::<i32>("x", &[0, 255]).unwrap().on(Exec::RETAINED).astype(TypeId::UInt8).unwrap();
    assert_eq!(kept.to_vec_strict::<u8>().unwrap(), vec![0, 255]);
    assert_cast_out_of_range(series::<i32>("x", &[256]).unwrap(), TypeId::UInt8, "above_target_range");
    assert_cast_out_of_range(series::<i32>("x", &[-1]).unwrap(), TypeId::UInt8, "below_target_range");

    let kept = series::<u64>("x", &[u64::from(u32::MAX)]).unwrap().on(Exec::RETAINED).astype(TypeId::UInt32).unwrap();
    assert_eq!(kept.to_vec_strict::<u32>().unwrap(), vec![u32::MAX]);
    assert_cast_out_of_range(
        series::<u64>("x", &[u64::from(u32::MAX) + 1]).unwrap(),
        TypeId::UInt32,
        "above_target_range",
    );
    assert_cast_out_of_range(series::<i16>("x", &[128]).unwrap(), TypeId::Int8, "above_target_range");

    let nullable = nullable_series::<i64>("x", &[1, i64::MAX, 3], &[true, false, true]).unwrap();
    let kept = nullable.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    assert_eq!(kept.to_optional_vec::<i32>().unwrap(), vec![Some(1), None, Some(3)]);

    let nan_under_null = nullable_series::<f64>("x", &[1.5, f64::NAN, -2.5], &[true, false, true]).unwrap();
    let kept = nan_under_null.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    assert_eq!(kept.to_optional_vec::<i32>().unwrap(), vec![Some(1), None, Some(-2)]);

    let all_null = nullable_series::<i64>("x", &[i64::MAX, i64::MIN], &[false, false]).unwrap();
    assert_eq!(all_null.on(Exec::RETAINED).astype(TypeId::Int8).unwrap().null_count(), 2);
    let empty = series::<i64>("x", &[]).unwrap();
    assert_eq!(empty.on(Exec::RETAINED).astype(TypeId::Int8).unwrap().len(), 0);
}

fn test_cast_float_to_integer_truncates_in_range_and_rejects_nan_infinity_and_out_of_range() {
    let truncated = series::<f64>("x", &[127.9, -128.9, 0.5, -0.5]).unwrap();
    let truncated = truncated.on(Exec::RETAINED).astype(TypeId::Int8).unwrap();
    assert_eq!(truncated.to_vec_strict::<i8>().unwrap(), vec![127, -128, 0, 0]);
    assert_cast_out_of_range(series::<f64>("x", &[128.0]).unwrap(), TypeId::Int8, "above_target_range");
    assert_cast_out_of_range(series::<f64>("x", &[-129.0]).unwrap(), TypeId::Int8, "below_target_range");
    assert_cast_out_of_range(series::<f64>("x", &[1.0, f64::NAN]).unwrap(), TypeId::Int8, "nan");
    assert_cast_out_of_range(series::<f64>("x", &[f64::INFINITY]).unwrap(), TypeId::Int64, "above_target_range");
    assert_cast_out_of_range(series::<f64>("x", &[f64::NEG_INFINITY]).unwrap(), TypeId::Int64, "below_target_range");

    let unsigned = series::<f64>("x", &[-0.9, 255.9]).unwrap().on(Exec::RETAINED).astype(TypeId::UInt8).unwrap();
    assert_eq!(unsigned.to_vec_strict::<u8>().unwrap(), vec![0, 255]);
    assert_cast_out_of_range(series::<f64>("x", &[-1.0]).unwrap(), TypeId::UInt8, "below_target_range");
    assert_cast_out_of_range(series::<f64>("x", &[256.0]).unwrap(), TypeId::UInt8, "above_target_range");

    let single = series::<f32>("x", &[2_147_483_520.0, -2_147_483_648.0]).unwrap();
    let single = single.on(Exec::RETAINED).astype(TypeId::Int32).unwrap();
    assert_eq!(single.to_vec_strict::<i32>().unwrap(), vec![2_147_483_520, i32::MIN]);
    assert_cast_out_of_range(series::<f32>("x", &[2_147_483_648.0]).unwrap(), TypeId::Int32, "above_target_range");
    assert_cast_out_of_range(series::<f32>("x", &[f32::NAN]).unwrap(), TypeId::Int32, "nan");

    let wide = series::<f64>("x", &[-9_223_372_036_854_775_808.0, 9_223_372_036_854_774_784.0]).unwrap();
    let wide = wide.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(wide.to_vec_strict::<i64>().unwrap(), vec![i64::MIN, 9_223_372_036_854_774_784]);
    assert_cast_out_of_range(
        series::<f64>("x", &[9_223_372_036_854_775_808.0]).unwrap(),
        TypeId::Int64,
        "above_target_range",
    );
    let wide_unsigned = series::<f64>("x", &[18_446_744_073_709_549_568.0]).unwrap();
    let wide_unsigned = wide_unsigned.on(Exec::RETAINED).astype(TypeId::UInt64).unwrap();
    assert_eq!(wide_unsigned.to_vec_strict::<u64>().unwrap(), vec![18_446_744_073_709_549_568]);
    assert_cast_out_of_range(
        series::<f64>("x", &[18_446_744_073_709_551_616.0]).unwrap(),
        TypeId::UInt64,
        "above_target_range",
    );
}

fn test_cast_decimal_to_integer_checks_the_scaled_integer_part() {
    let whole = decimal128_series_from_arrow(vec![Some(i128::from(i64::MAX)), Some(i128::from(i64::MIN)), None], 38, 0);
    let kept = whole.on(Exec::RETAINED).astype(TypeId::Int64).unwrap();
    assert_eq!(kept.to_optional_vec::<i64>().unwrap(), vec![Some(i64::MAX), Some(i64::MIN), None]);
    assert_cast_out_of_range(
        decimal128_series_from_arrow(vec![Some(i128::from(i64::MAX) + 1)], 38, 0),
        TypeId::Int64,
        "above_target_range",
    );
    assert_cast_out_of_range(
        decimal128_series_from_arrow(vec![Some(i128::from(i64::MIN) - 1)], 38, 0),
        TypeId::Int64,
        "below_target_range",
    );
    assert_cast_out_of_range(decimal128_series_from_arrow(vec![Some(-1)], 38, 0), TypeId::UInt32, "below_target_range");

    let fractional = decimal128_series_from_arrow(vec![Some(12_799), Some(-12_899)], 38, 2);
    let kept = fractional.on(Exec::RETAINED).astype(TypeId::Int8).unwrap();
    assert_eq!(kept.to_vec_strict::<i8>().unwrap(), vec![127, -128]);
    assert_cast_out_of_range(
        decimal128_series_from_arrow(vec![Some(12_800)], 38, 2),
        TypeId::Int8,
        "above_target_range",
    );
    assert_cast_out_of_range(
        decimal128_series_from_arrow(vec![Some(-12_900)], 38, 2),
        TypeId::Int8,
        "below_target_range",
    );
}

fn test_cast_integer_to_string_formats_plain_decimal_text() {
    let signed = series::<i64>("x", &[i64::MIN, -1, 0, 7, i64::MAX]).unwrap();
    let text = signed.on(Exec::RETAINED).astype(TypeId::String).unwrap();
    assert_eq!(text.to_strings_strict().unwrap(), vec!["-9223372036854775808", "-1", "0", "7", "9223372036854775807"]);

    let unsigned = series::<u64>("x", &[0, u64::MAX]).unwrap().on(Exec::RETAINED).astype(TypeId::String).unwrap();
    assert_eq!(unsigned.to_strings_strict().unwrap(), vec!["0", "18446744073709551615"]);

    let narrow = series::<i8>("x", &[i8::MIN, i8::MAX]).unwrap().on(Exec::RETAINED).astype(TypeId::String).unwrap();
    assert_eq!(narrow.to_strings_strict().unwrap(), vec!["-128", "127"]);

    let nullable = nullable_series::<i32>("x", &[5, 6, -7], &[true, false, true]).unwrap();
    let text = nullable.on(Exec::RETAINED).astype(TypeId::String).unwrap();
    assert_eq!(text.to_optional_strings().unwrap(), vec![Some("5".to_string()), None, Some("-7".to_string())]);

    let empty = series::<i32>("x", &[]).unwrap().on(Exec::RETAINED).astype(TypeId::String).unwrap();
    assert_eq!(empty.len(), 0);

    assert!(is_supported_cast(&DataType::new(TypeId::Int64), &DataType::new(TypeId::String)).unwrap());
    assert!(!is_supported_cast(&DataType::new(TypeId::Float64), &DataType::new(TypeId::String)).unwrap());
}

fn decimal_target(id: TypeId, scale: i8) -> DataType {
    DataType::fixed_point(id, -i32::from(scale)).unwrap()
}

/// Casts `input` into `id` with `Expr::cast_decimal` and returns the raw values.
fn cast_into_decimal(input: &cudf::Series, id: TypeId, precision: u8, scale: i8) -> cudf::Result<Vec<Option<i128>>> {
    let target = decimal_target(id, scale);
    let frame = DataFrame::from_series([("x", input.clone())])?;
    let result = col("x").cast_decimal(target, precision).eval(&frame)?;
    assert_eq!(result.dtype()?, target);
    // Decimal32 and Decimal64 export as Decimal128 as well, with the same raw values.
    let batch = DataFrame::from_series([("x", result)])?.to_record_batch()?;
    let values = batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap();
    Ok(values.iter().collect())
}

fn arrow_cast_into_decimal(array: ArrayRef, precision: u8, scale: i8) -> Result<Vec<Option<i128>>, ArrowError> {
    let options = CastOptions { safe: false, ..CastOptions::default() };
    let result = cast_with_options(&array, &ArrowDataType::Decimal128(precision, scale), &options)?;
    Ok(result.as_any().downcast_ref::<Decimal128Array>().unwrap().iter().collect())
}

fn assert_decimal_cast_error(error: &cudf::Error, reason: &str) {
    assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.detail_value("reason"), Some(reason));
}

/// The cast must give the values of an Arrow `CAST` in non-safe mode, and fail
/// where Arrow fails.
fn assert_decimal_cast_matches_arrow(name: &str, input: &cudf::Series, array: ArrayRef, precision: u8, scale: i8) {
    let expected = arrow_cast_into_decimal(array, precision, scale);
    let actual = cast_into_decimal(input, TypeId::Decimal128, precision, scale);
    match (expected, actual) {
        (Ok(expected), Ok(actual)) => assert_eq!(actual, expected, "{name} into ({precision}, {scale})"),
        (Err(_), Err(error)) => {
            assert_eq!(error.code(), ErrorCode::ValueOutOfRange, "{name} into ({precision}, {scale}): {error}")
        },
        (expected, actual) => panic!("{name} into ({precision}, {scale}): arrow {expected:?}, cudf {actual:?}"),
    }
}

fn test_cast_integer_to_decimal_checks_the_declared_precision() {
    // Precision 5 at scale 2 holds up to 999.99, so integers up to 999 fit.
    let fits = series_opt::<i64>("x", &[Some(999), Some(-999), Some(0), None]).unwrap();
    assert_eq!(
        cast_into_decimal(&fits, TypeId::Decimal128, 5, 2).unwrap(),
        vec![Some(99_900), Some(-99_900), Some(0), None]
    );
    for overflow in [1000, -1000] {
        let input = series::<i64>("x", &[1, overflow]).unwrap();
        let error = cast_into_decimal(&input, TypeId::Decimal128, 5, 2).unwrap_err();
        assert_decimal_cast_error(&error, "decimal_precision_overflow");
        assert_eq!(error.detail_value("target_precision"), Some("5"));
    }
    // A value under a null mask is not read.
    let masked = nullable_series::<i64>("x", &[7, 1_000_000], &[true, false]).unwrap();
    assert_eq!(cast_into_decimal(&masked, TypeId::Decimal128, 5, 2).unwrap(), vec![Some(700), None]);
    let all_null = nullable_series::<i64>("x", &[1_000_000], &[false]).unwrap();
    assert_eq!(cast_into_decimal(&all_null, TypeId::Decimal128, 5, 2).unwrap(), vec![None]);

    // Narrow storage types keep their own limits: 9 digits for Decimal32, 18 for Decimal64.
    for (id, digits) in [(TypeId::Decimal32, 9u32), (TypeId::Decimal64, 18)] {
        let limit = 10i64.pow(digits) - 1;
        let at_limit = series::<i64>("x", &[limit, -limit]).unwrap();
        assert_eq!(
            cast_into_decimal(&at_limit, id, digits as u8, 0).unwrap(),
            vec![Some(limit.into()), Some((-limit).into())]
        );
        let past_limit = series::<i64>("x", &[limit + 1]).unwrap();
        let error = cast_into_decimal(&past_limit, id, digits as u8, 0).unwrap_err();
        assert_decimal_cast_error(&error, "decimal_precision_overflow");
    }

    let values = [0, 1, -1, 99, 100, 999, 1000, -999, -1000, 12_345, 99_999, 100_000, i64::MAX, i64::MIN];
    for (precision, scale) in [(5, 2), (5, 0), (10, 2), (19, 0), (19, 4), (38, 20), (38, 38), (3, 3)] {
        for value in values {
            let input = series::<i64>("x", &[value]).unwrap();
            let array: ArrayRef = Arc::new(Int64Array::from(vec![value]));
            assert_decimal_cast_matches_arrow(&format!("int64 {value}"), &input, array, precision, scale);
        }
    }
}

fn test_cast_float_to_decimal_rounds_like_arrow_and_rejects_non_finite_values() {
    // Ties round away from zero in both directions; 0.125 and 2.5 are exact in binary.
    let ties = series::<f64>("x", &[0.125, -0.125, 2.5, -2.5, 0.5, -0.5, 1.999]).unwrap();
    assert_eq!(
        cast_into_decimal(&ties, TypeId::Decimal128, 10, 2).unwrap(),
        vec![Some(13), Some(-13), Some(250), Some(-250), Some(50), Some(-50), Some(200)]
    );
    assert_eq!(
        cast_into_decimal(&ties, TypeId::Decimal128, 10, 0).unwrap(),
        vec![Some(0), Some(0), Some(3), Some(-3), Some(1), Some(-1), Some(2)]
    );

    let infinite = series::<f64>("x", &[1.0, f64::INFINITY]).unwrap();
    assert_decimal_cast_error(
        &cast_into_decimal(&infinite, TypeId::Decimal128, 10, 2).unwrap_err(),
        "decimal_precision_overflow",
    );
    let not_a_number = series::<f64>("x", &[1.0, f64::NAN]).unwrap();
    assert_decimal_cast_error(&cast_into_decimal(&not_a_number, TypeId::Decimal128, 10, 2).unwrap_err(), "nan");
    let hidden = nullable_series::<f64>("x", &[1.5, f64::NAN], &[true, false]).unwrap();
    assert_eq!(cast_into_decimal(&hidden, TypeId::Decimal128, 10, 2).unwrap(), vec![Some(150), None]);

    // Precision 5 at scale 2 holds up to 999.99; 999.995 rounds to 1000.00.
    let boundary = series::<f64>("x", &[999.99, -999.99, 999.994]).unwrap();
    assert_eq!(
        cast_into_decimal(&boundary, TypeId::Decimal128, 5, 2).unwrap(),
        vec![Some(99_999), Some(-99_999), Some(99_999)]
    );
    for overflow in [1000.0, -1000.0, 999.996] {
        let input = series::<f64>("x", &[overflow]).unwrap();
        assert_decimal_cast_error(
            &cast_into_decimal(&input, TypeId::Decimal128, 5, 2).unwrap_err(),
            "decimal_precision_overflow",
        );
    }

    let wide = [
        0.0,
        -0.0,
        0.285,
        1.005,
        -1.005,
        123.456,
        -123.456,
        0.1,
        0.7,
        1e-10,
        5e-3,
        99_999.995,
        1e10,
        1e15 + 0.5,
        1e22,
        1e30,
        -1e30,
        9.999e37,
        1e38,
        1e39,
        1.7e300,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    for (precision, scale) in [(10, 2), (10, 0), (15, 6), (20, 10), (38, 0), (38, 4), (38, 38), (6, 3)] {
        for value in wide {
            let input = series::<f64>("x", &[value]).unwrap();
            let array: ArrayRef = Arc::new(Float64Array::from(vec![value]));
            assert_decimal_cast_matches_arrow(&format!("float64 {value:e}"), &input, array, precision, scale);
            let single = value as f32;
            let input = series::<f32>("x", &[single]).unwrap();
            let array: ArrayRef = Arc::new(Float32Array::from(vec![single]));
            assert_decimal_cast_matches_arrow(&format!("float32 {single:e}"), &input, array, precision, scale);
        }
    }
}

fn test_cast_decimal_to_decimal_rounds_half_away_from_zero_and_checks_precision() {
    // Scale 4 into scale 2 drops two digits: the tie at .50 rounds away from zero.
    let source = decimal128_series_from_arrow(
        vec![Some(1_249), Some(1_250), Some(-1_249), Some(-1_250), Some(-1_251), Some(49), Some(50), None],
        38,
        4,
    );
    assert_eq!(
        cast_into_decimal(&source, TypeId::Decimal128, 10, 2).unwrap(),
        vec![Some(12), Some(13), Some(-12), Some(-13), Some(-13), Some(0), Some(1), None]
    );

    // 999.9949 rounds to 999.99 and fits precision 5 at scale 2; 999.9950 rounds to 1000.00.
    let fits = decimal128_series_from_arrow(vec![Some(9_999_949), Some(-9_999_949)], 38, 4);
    assert_eq!(cast_into_decimal(&fits, TypeId::Decimal128, 5, 2).unwrap(), vec![Some(99_999), Some(-99_999)]);
    for overflow in [9_999_950, -9_999_950] {
        let input = decimal128_series_from_arrow(vec![Some(overflow)], 38, 4);
        assert_decimal_cast_error(
            &cast_into_decimal(&input, TypeId::Decimal128, 5, 2).unwrap_err(),
            "decimal_precision_overflow",
        );
    }
    // The same scale into a narrower precision checks the values only.
    let same_scale = decimal128_series_from_arrow(vec![Some(99_999), Some(-99_999)], 38, 2);
    assert_eq!(cast_into_decimal(&same_scale, TypeId::Decimal128, 5, 2).unwrap(), vec![Some(99_999), Some(-99_999)]);
    let too_wide = decimal128_series_from_arrow(vec![Some(100_000)], 38, 2);
    assert_decimal_cast_error(
        &cast_into_decimal(&too_wide, TypeId::Decimal128, 5, 2).unwrap_err(),
        "decimal_precision_overflow",
    );
    let masked = decimal128_series_from_arrow(vec![Some(7), None], 38, 2);
    assert_eq!(cast_into_decimal(&masked, TypeId::Decimal128, 5, 2).unwrap(), vec![Some(7), None]);

    let raw_values: Vec<i128> = vec![
        0,
        5,
        49,
        50,
        51,
        95,
        99,
        100,
        149,
        150,
        151,
        999,
        1_000,
        9_949,
        9_950,
        99_999,
        100_000,
        999_999_999,
        i128::from(i64::MAX),
        10i128.pow(37),
        10i128.pow(38) - 1,
    ];
    let signed: Vec<i128> = raw_values.iter().flat_map(|&raw| [raw, -raw]).collect();
    for (source_precision, source_scale) in [(38u8, 4i8), (38, 0), (30, 10), (20, -3)] {
        // Arrow skips its checks where the source precision proves a cast safe, so the
        // values must stay within the precision they declare.
        for &raw in signed.iter().filter(|raw| raw.unsigned_abs() < 10u128.pow(u32::from(source_precision))) {
            let input = decimal128_series_from_arrow(vec![Some(raw)], source_precision, source_scale);
            let array: ArrayRef = Arc::new(
                Decimal128Array::from(vec![raw]).with_precision_and_scale(source_precision, source_scale).unwrap(),
            );
            for (precision, scale) in [(38, 2), (38, 0), (38, 10), (10, 2), (6, 3), (5, 0), (38, 38), (12, 6)] {
                assert_decimal_cast_matches_arrow(
                    &format!("decimal128({source_precision}, {source_scale}) raw {raw}"),
                    &input,
                    Arc::clone(&array),
                    precision,
                    scale,
                );
            }
        }
    }
}

fn test_cast_decimal_to_narrow_decimal_storage_checks_the_declared_precision() {
    let source = decimal128_series_from_arrow(vec![Some(123_456), Some(-123_450), None], 38, 4);
    assert_eq!(cast_into_decimal(&source, TypeId::Decimal64, 15, 2).unwrap(), vec![Some(1_235), Some(-1_235), None]);
    assert_eq!(cast_into_decimal(&source, TypeId::Decimal32, 9, 1).unwrap(), vec![Some(123), Some(-123), None]);
    let past_decimal32 = decimal128_series_from_arrow(vec![Some(1_000_000_000)], 38, 0);
    assert_decimal_cast_error(
        &cast_into_decimal(&past_decimal32, TypeId::Decimal32, 9, 0).unwrap_err(),
        "decimal_precision_overflow",
    );
    let decimal64 = decimal128_series_from_arrow(vec![Some(150)], 38, 2)
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    assert_eq!(cast_into_decimal(&decimal64, TypeId::Decimal128, 20, 0).unwrap(), vec![Some(2)]);
}

fn test_cast_decimal_scale_beyond_the_target_storage_fails_for_any_values() {
    // Arrow multiplies by 10^(38 + 5) in an i128, which overflows even for zero rows.
    let source = decimal128_series_from_arrow(vec![Some(0)], 20, -5);
    let expected = Decimal128Array::from(vec![0i128]).with_precision_and_scale(20, -5).unwrap();
    assert!(arrow_cast_into_decimal(Arc::new(expected), 38, 38).is_err());
    let error = cast_into_decimal(&source, TypeId::Decimal128, 38, 38).unwrap_err();
    assert_decimal_cast_error(&error, "decimal_scale_overflow");
}

fn test_cast_decimal_rejects_invalid_declarations() {
    let input = series::<i64>("x", &[1]).unwrap();
    let frame = DataFrame::from_series([("x", input)]).unwrap();
    for (id, precision) in
        [(TypeId::Decimal128, 0), (TypeId::Decimal128, 39), (TypeId::Decimal64, 19), (TypeId::Decimal32, 10)]
    {
        let error = col("x").cast_decimal(decimal_target(id, 0), precision).eval(&frame).unwrap_err();
        assert_eq!(error.code(), ErrorCode::UnsupportedDtype, "{id:?} {precision}");
        assert_eq!(error.detail_value("reason"), Some("invalid_decimal_precision"));
    }
    let error = col("x").cast_decimal(DataType::new(TypeId::Int32), 5).eval(&frame).unwrap_err();
    assert_eq!(error.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(error.detail_value("reason"), Some("decimal_precision_requires_decimal_target"));
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("cast_spec_tests::test_cast_i8_to_i16", test_cast_i8_to_i16),
    ("cast_spec_tests::test_cast_i16_to_i32", test_cast_i16_to_i32),
    ("cast_spec_tests::test_cast_i32_to_i64", test_cast_i32_to_i64),
    ("cast_spec_tests::test_cast_i32_to_f32", test_cast_i32_to_f32),
    ("cast_spec_tests::test_cast_f32_to_i32_truncates", test_cast_f32_to_i32_truncates),
    ("cast_spec_tests::test_cast_f64_to_f32", test_cast_f64_to_f32),
    (
        "cast_spec_tests::test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls",
        test_cast_decimal128_to_f64_uses_decimal_scale_and_preserves_nulls,
    ),
    ("cast_spec_tests::test_cast_decimal64_to_f64_uses_decimal_scale", test_cast_decimal64_to_f64_uses_decimal_scale),
    (
        "cast_spec_tests::test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls",
        test_cast_f64_to_decimal128_uses_target_scale_and_preserves_nulls,
    ),
    ("cast_spec_tests::test_cast_bool_to_i32", test_cast_bool_to_i32),
    ("cast_spec_tests::test_cast_i32_to_bool", test_cast_i32_to_bool),
    ("cast_spec_tests::test_cast_bool_to_i64", test_cast_bool_to_i64),
    ("cast_spec_tests::test_cast_u8_to_i16", test_cast_u8_to_i16),
    ("cast_spec_tests::test_cast_u32_to_i64", test_cast_u32_to_i64),
    ("cast_spec_tests::test_cast_preserves_null_mask", test_cast_preserves_null_mask),
    ("cast_spec_tests::test_cast_all_null_column", test_cast_all_null_column),
    ("cast_spec_tests::test_cast_empty_column", test_cast_empty_column),
    ("cast_spec_tests::test_cast_same_type", test_cast_same_type),
    (
        "cast_spec_tests::decimal_rescale_workspace_contract_is_scale_sensitive",
        decimal_rescale_workspace_contract_is_scale_sensitive,
    ),
    ("cast_spec_tests::test_is_supported_cast_numeric_to_float", test_is_supported_cast_numeric_to_float),
    ("cast_spec_tests::test_is_supported_cast_decimal_to_float", test_is_supported_cast_decimal_to_float),
    ("cast_spec_tests::test_is_supported_cast_float_to_decimal", test_is_supported_cast_float_to_decimal),
    ("cast_spec_tests::test_is_supported_cast_float_to_int", test_is_supported_cast_float_to_int),
    ("cast_spec_tests::test_is_supported_cast_bool_to_int", test_is_supported_cast_bool_to_int),
    ("cast_spec_tests::test_cast_i64_to_timestamp_microseconds", test_cast_i64_to_timestamp_microseconds),
    ("cast_spec_tests::test_cast_timestamp_to_i64", test_cast_timestamp_to_i64),
    (
        "cast_spec_tests::test_cast_timestamp_and_duration_to_i64_returns_the_stored_count_in_the_types_own_unit",
        test_cast_timestamp_and_duration_to_i64_returns_the_stored_count_in_the_types_own_unit,
    ),
    (
        "cast_spec_tests::test_cast_timestamp_and_duration_to_narrower_integers_check_the_stored_count",
        test_cast_timestamp_and_duration_to_narrower_integers_check_the_stored_count,
    ),
    (
        "cast_spec_tests::test_cast_integer_to_timestamp_reads_the_integer_as_the_tick_count",
        test_cast_integer_to_timestamp_reads_the_integer_as_the_tick_count,
    ),
    (
        "cast_spec_tests::test_cast_temporal_to_a_coarser_resolution_truncates_toward_zero_like_arrow",
        test_cast_temporal_to_a_coarser_resolution_truncates_toward_zero_like_arrow,
    ),
    (
        "cast_spec_tests::test_cast_temporal_to_a_finer_resolution_matches_arrow_up_to_the_int64_limits",
        test_cast_temporal_to_a_finer_resolution_matches_arrow_up_to_the_int64_limits,
    ),
    (
        "cast_spec_tests::test_cast_dates_to_timestamps_fail_where_the_product_leaves_int64_and_timestamps_to_dates_floor",
        test_cast_dates_to_timestamps_fail_where_the_product_leaves_int64_and_timestamps_to_dates_floor,
    ),
    ("cast_spec_tests::test_to_datetime_default_microseconds", test_to_datetime_default_microseconds),
    ("cast_spec_tests::test_to_datetime_explicit_milliseconds", test_to_datetime_explicit_milliseconds),
    ("cast_spec_tests::test_to_datetime_nanoseconds", test_to_datetime_nanoseconds),
    ("cast_spec_tests::test_to_datetime_seconds", test_to_datetime_seconds),
    ("cast_spec_tests::test_to_datetime_round_trip", test_to_datetime_round_trip),
    ("cast_spec_tests::test_to_datetime_preserves_nulls", test_to_datetime_preserves_nulls),
    ("cast_spec_tests::test_to_numeric_f64_to_i64", test_to_numeric_f64_to_i64),
    ("cast_spec_tests::test_to_numeric_i32_to_f64", test_to_numeric_i32_to_f64),
    ("cast_spec_tests::test_to_numeric_bool_to_i32", test_to_numeric_bool_to_i32),
    ("cast_spec_tests::test_to_numeric_rejects_non_numeric_target", test_to_numeric_rejects_non_numeric_target),
    ("cast_spec_tests::test_to_numeric_rejects_timestamp_target", test_to_numeric_rejects_timestamp_target),
    (
        "cast_spec_tests::test_cast_integer_to_integer_accepts_target_extremes_and_rejects_one_past",
        test_cast_integer_to_integer_accepts_target_extremes_and_rejects_one_past,
    ),
    (
        "cast_spec_tests::test_cast_float_to_integer_truncates_in_range_and_rejects_nan_infinity_and_out_of_range",
        test_cast_float_to_integer_truncates_in_range_and_rejects_nan_infinity_and_out_of_range,
    ),
    (
        "cast_spec_tests::test_cast_decimal_to_integer_checks_the_scaled_integer_part",
        test_cast_decimal_to_integer_checks_the_scaled_integer_part,
    ),
    (
        "cast_spec_tests::test_cast_integer_to_string_formats_plain_decimal_text",
        test_cast_integer_to_string_formats_plain_decimal_text,
    ),
    (
        "cast_spec_tests::test_cast_integer_to_decimal_checks_the_declared_precision",
        test_cast_integer_to_decimal_checks_the_declared_precision,
    ),
    (
        "cast_spec_tests::test_cast_float_to_decimal_rounds_like_arrow_and_rejects_non_finite_values",
        test_cast_float_to_decimal_rounds_like_arrow_and_rejects_non_finite_values,
    ),
    (
        "cast_spec_tests::test_cast_decimal_to_decimal_rounds_half_away_from_zero_and_checks_precision",
        test_cast_decimal_to_decimal_rounds_half_away_from_zero_and_checks_precision,
    ),
    (
        "cast_spec_tests::test_cast_decimal_to_narrow_decimal_storage_checks_the_declared_precision",
        test_cast_decimal_to_narrow_decimal_storage_checks_the_declared_precision,
    ),
    (
        "cast_spec_tests::test_cast_decimal_scale_beyond_the_target_storage_fails_for_any_values",
        test_cast_decimal_scale_beyond_the_target_storage_fails_for_any_values,
    ),
    ("cast_spec_tests::test_cast_decimal_rejects_invalid_declarations", test_cast_decimal_rejects_invalid_declarations),
];
