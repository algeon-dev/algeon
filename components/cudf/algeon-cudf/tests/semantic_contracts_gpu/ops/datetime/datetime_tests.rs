//! Integration tests for datetime extract operations.

use cudf::{ErrorCode, ErrorKind, Exec, TypeId};

use crate::common::{error_context_value, nullable_series, run_named_cases, series};

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

fn timestamp_us_series(values: &[i64]) -> cudf::Result<cudf::Series> {
    series("ts", values)?
        .on(Exec::RETAINED)
        .astype(TypeId::DurationMicroseconds)?
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampMicroseconds)
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

#[test]
fn datetime_tests_cases() {
    run_named_cases(&[
        (
            "test_datetime_extract_year_month_day_from_timestamp_days",
            test_datetime_extract_year_month_day_from_timestamp_days,
        ),
        (
            "test_datetime_extract_year_month_day_from_timestamp_microseconds",
            test_datetime_extract_year_month_day_from_timestamp_microseconds,
        ),
        ("test_datetime_extract_preserves_nulls", test_datetime_extract_preserves_nulls),
        (
            "test_datetime_extract_rejects_non_timestamp_with_stable_error",
            test_datetime_extract_rejects_non_timestamp_with_stable_error,
        ),
    ]);
}
