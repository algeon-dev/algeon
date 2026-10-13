use cudf::{ErrorCode, ErrorKind, Exec, TypeId};

use crate::common::{error_context_value, nullable_series, series};

fn timestamp_days_series(values: &[i32]) -> cudf::Result<cudf::Series> {
    series("ts", values)?
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)?
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
}

fn nullable_timestamp_days_series(values: &[i32], validity: &[bool]) -> cudf::Result<cudf::Series> {
    nullable_series("ts", values, validity)?
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)?
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
}

fn timestamp_series(values: &[i64], duration: TypeId, timestamp: TypeId) -> cudf::Result<cudf::Series> {
    series("ts", values)?.on(Exec::RETAINED).astype(duration)?.on(Exec::RETAINED).astype(timestamp)
}

fn timestamp_us_series(values: &[i64]) -> cudf::Result<cudf::Series> {
    timestamp_series(values, TypeId::DurationMicroseconds, TypeId::TimestampMicroseconds)
}

/// Hour, minute, second, millisecond and microsecond fields of `ts`, in that order.
fn time_fields(ts: &cudf::Series) -> [Vec<i16>; 5] {
    [
        ts.on(Exec::RETAINED).dt().hour().unwrap(),
        ts.on(Exec::RETAINED).dt().minute().unwrap(),
        ts.on(Exec::RETAINED).dt().second().unwrap(),
        ts.on(Exec::RETAINED).dt().millisecond().unwrap(),
        ts.on(Exec::RETAINED).dt().microsecond().unwrap(),
    ]
    .map(|field| {
        assert_eq!(field.dtype().unwrap().id(), TypeId::Int16);
        field.to_vec_strict::<i16>().unwrap()
    })
}

fn test_datetime_extract_year_month_day_from_timestamp_days() {
    let ts = timestamp_days_series(&[0, 10_561, 18_321]).unwrap();

    let year = ts.on(Exec::RETAINED).dt().year().unwrap();
    let month = ts.on(Exec::RETAINED).dt().month().unwrap();
    let day = ts.on(Exec::RETAINED).dt().day().unwrap();

    assert_eq!(year.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(month.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(day.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(year.to_vec_strict::<i16>().unwrap(), vec![1970, 1998, 2020]);
    assert_eq!(month.to_vec_strict::<i16>().unwrap(), vec![1, 12, 2]);
    assert_eq!(day.to_vec_strict::<i16>().unwrap(), vec![1, 1, 29]);
}

fn test_datetime_extract_year_month_day_from_timestamp_microseconds() {
    let ts = timestamp_us_series(&[0, 912_518_055_123_456, 1_583_020_799_999_999]).unwrap();

    assert_eq!(ts.on(Exec::RETAINED).dt().year().unwrap().to_vec_strict::<i16>().unwrap(), vec![1970, 1998, 2020]);
    assert_eq!(ts.on(Exec::RETAINED).dt().month().unwrap().to_vec_strict::<i16>().unwrap(), vec![1, 12, 2]);
    assert_eq!(ts.on(Exec::RETAINED).dt().day().unwrap().to_vec_strict::<i16>().unwrap(), vec![1, 1, 29]);
}

fn test_datetime_extract_time_fields_from_timestamp_microseconds() {
    // 2020-03-04 12:34:56.789123 and the epoch.
    let ts = timestamp_us_series(&[1_583_325_296_789_123, 0]).unwrap();

    assert_eq!(time_fields(&ts), [vec![12, 0], vec![34, 0], vec![56, 0], vec![789, 0], vec![123, 0]]);
}

fn test_datetime_extract_time_fields_are_non_negative_before_epoch() {
    // 1969-12-31 23:59:59.999999 and exactly one day before the epoch.
    let ts = timestamp_us_series(&[-1, -86_400_000_000]).unwrap();

    assert_eq!(time_fields(&ts), [vec![23, 0], vec![59, 0], vec![59, 0], vec![999, 0], vec![999, 0]]);
}

fn test_datetime_extract_time_fields_drop_nanosecond_remainder() {
    // 2020-03-04 12:34:56.789123456
    let ts = timestamp_series(&[1_583_325_296_789_123_456], TypeId::DurationNanoseconds, TypeId::TimestampNanoseconds)
        .unwrap();

    assert_eq!(time_fields(&ts), [vec![12], vec![34], vec![56], vec![789], vec![123]]);
}

fn test_datetime_extract_sub_second_fields_are_zero_for_timestamp_seconds() {
    // 2020-03-04 12:34:56
    let ts = timestamp_series(&[1_583_325_296], TypeId::DurationSeconds, TypeId::TimestampSeconds).unwrap();

    assert_eq!(time_fields(&ts), [vec![12], vec![34], vec![56], vec![0], vec![0]]);
}

fn test_datetime_extract_time_field_preserves_nulls() {
    let ts = nullable_series("ts", &[1_583_325_296_789_123i64, 0, -1], &[true, false, true])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationMicroseconds)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampMicroseconds)
        .unwrap();

    let result = ts.on(Exec::RETAINED).dt().millisecond().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.to_optional_vec::<i16>().unwrap(), vec![Some(789), None, Some(999)]);
}

fn test_datetime_extract_preserves_nulls() {
    let ts = nullable_timestamp_days_series(&[0, 10_561, 18_321], &[true, false, true]).unwrap();

    let result = ts.on(Exec::RETAINED).dt().year().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.to_optional_vec::<i16>().unwrap(), vec![Some(1970), None, Some(2020)]);
}

fn test_datetime_extract_rejects_non_timestamp_with_stable_error() {
    let ints = series("x", &[1i32, 2, 3]).unwrap();

    let err = ints.on(Exec::RETAINED).dt().year().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::dt::year");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), Some("timestamp"));
    assert_eq!(error_context_value(&err, "component").as_deref(), Some("year"));
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "datetime_tests::test_datetime_extract_year_month_day_from_timestamp_days",
        test_datetime_extract_year_month_day_from_timestamp_days,
    ),
    (
        "datetime_tests::test_datetime_extract_year_month_day_from_timestamp_microseconds",
        test_datetime_extract_year_month_day_from_timestamp_microseconds,
    ),
    (
        "datetime_tests::test_datetime_extract_time_fields_from_timestamp_microseconds",
        test_datetime_extract_time_fields_from_timestamp_microseconds,
    ),
    (
        "datetime_tests::test_datetime_extract_time_fields_are_non_negative_before_epoch",
        test_datetime_extract_time_fields_are_non_negative_before_epoch,
    ),
    (
        "datetime_tests::test_datetime_extract_time_fields_drop_nanosecond_remainder",
        test_datetime_extract_time_fields_drop_nanosecond_remainder,
    ),
    (
        "datetime_tests::test_datetime_extract_sub_second_fields_are_zero_for_timestamp_seconds",
        test_datetime_extract_sub_second_fields_are_zero_for_timestamp_seconds,
    ),
    ("datetime_tests::test_datetime_extract_preserves_nulls", test_datetime_extract_preserves_nulls),
    (
        "datetime_tests::test_datetime_extract_time_field_preserves_nulls",
        test_datetime_extract_time_field_preserves_nulls,
    ),
    (
        "datetime_tests::test_datetime_extract_rejects_non_timestamp_with_stable_error",
        test_datetime_extract_rejects_non_timestamp_with_stable_error,
    ),
];
