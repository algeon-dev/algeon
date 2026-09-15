//! Integration tests for datetime extract operations.

use cudf::{
    ErrorCode, ErrorKind, ExecutionOverrides, Runtime, TypeId,
    ops::datetime::{self, DatetimeComponent},
    testing::prelude::*,
};

use crate::common::{
    error_context_value, last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, nullable_series,
    reset_exec_observation, reset_sync_observation, series,
};

fn timestamp_days_series(values: &[i32]) -> cudf::Result<cudf::Series> {
    series("ts", values)?.astype(TypeId::DurationDays)?.astype(TypeId::TimestampDays)
}

fn nullable_timestamp_days_series(values: &[i32], validity: &[bool]) -> cudf::Result<cudf::Series> {
    nullable_series("ts", values, validity)?.astype(TypeId::DurationDays)?.astype(TypeId::TimestampDays)
}

fn timestamp_us_series(values: &[i64]) -> cudf::Result<cudf::Series> {
    series("ts", values)?.astype(TypeId::DurationMicroseconds)?.astype(TypeId::TimestampMicroseconds)
}

fn timestamp_us_series_in_runtime(runtime: &Runtime, values: &[i64]) -> cudf::Result<cudf::Series> {
    runtime.series(values)?.astype(TypeId::DurationMicroseconds)?.astype(TypeId::TimestampMicroseconds)
}

fn test_datetime_extract_year_month_day_from_timestamp_days() {
    let ts = timestamp_days_series(&[0, 10_561, 18_321]).unwrap();

    let year = ts.dt().year().unwrap();
    let month = ts.dt().month().unwrap();
    let day = ts.dt().day().unwrap();

    assert_eq!(year.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(month.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(day.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(year.to_vec_strict::<i16>().unwrap(), vec![1970, 1998, 2020]);
    assert_eq!(month.to_vec_strict::<i16>().unwrap(), vec![1, 12, 2]);
    assert_eq!(day.to_vec_strict::<i16>().unwrap(), vec![1, 1, 29]);
}

fn test_datetime_extract_year_month_day_from_timestamp_microseconds() {
    let ts = timestamp_us_series(&[0, 912_518_055_123_456, 1_583_020_799_999_999]).unwrap();

    assert_eq!(ts.dt().year().unwrap().to_vec_strict::<i16>().unwrap(), vec![1970, 1998, 2020]);
    assert_eq!(ts.dt().month().unwrap().to_vec_strict::<i16>().unwrap(), vec![1, 12, 2]);
    assert_eq!(ts.dt().day().unwrap().to_vec_strict::<i16>().unwrap(), vec![1, 1, 29]);
}

fn test_datetime_extract_preserves_nulls() {
    let ts = nullable_timestamp_days_series(&[0, 10_561, 18_321], &[true, false, true]).unwrap();

    let result = ts.dt().year().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int16);
    assert_eq!(result.null_count(), 1);
    assert_eq!(result.to_optional_vec::<i16>().unwrap(), vec![Some(1970), None, Some(2020)]);
}

fn test_datetime_extract_rejects_non_timestamp_with_stable_error() {
    let ints = series("x", &[1i32, 2, 3]).unwrap();

    let err = ints.dt().year().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "series::dt::year");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), Some("timestamp"));
    assert_eq!(error_context_value(&err, "component").as_deref(), Some("year"));
}

fn test_datetime_add_calendrical_months_rejects_non_timestamp_without_component_context() {
    let ints = series("x", &[1i32, 2, 3]).unwrap();
    let view = ints.as_column().testing_view().unwrap();

    let err = datetime::add_calendrical_months(&view, 1).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "datetime::add_calendrical_months");
    assert_eq!(err.message(), "datetime offset requires timestamp input");
    assert_eq!(error_context_value(&err, "type_id").as_deref(), Some("int32"));
    assert_eq!(error_context_value(&err, "expected_family").as_deref(), Some("timestamp"));
    assert_eq!(error_context_value(&err, "component").as_deref(), None);
}

fn test_datetime_low_level_extract_component_matches_series_namespace() {
    let ts = timestamp_days_series(&[9_204, 10_561]).unwrap();
    let view = ts.as_column().testing_view().unwrap();

    let month = datetime::extract_component(&view, DatetimeComponent::Month).unwrap();
    let day = datetime::extract_component(&view, DatetimeComponent::Day).unwrap();

    assert_eq!(month.testing_view().unwrap().data_type().unwrap().id(), TypeId::Int16);
    assert_eq!(month.testing_view().unwrap().to_vec::<i16>().unwrap(), vec![3, 12]);
    assert_eq!(day.testing_view().unwrap().to_vec::<i16>().unwrap(), vec![15, 1]);
}

fn test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance() {
    let input_runtime = Runtime::builder().build().unwrap();
    let override_runtime = Runtime::builder().build().unwrap();
    let ts = timestamp_us_series_in_runtime(&input_runtime, &[0, 912_518_055_123_456]).unwrap();
    let ctx = ExecutionOverrides::new()
        .with_stream(override_runtime.stream())
        .with_memory_resource(override_runtime.memory_resource());
    let expected_placement = override_runtime.placement();
    let expected_stream = expected_placement.stream_debug;
    let expected_mr = expected_placement.memory_resource_debug;

    reset_exec_observation();
    let result = ts.dt().year_in(&ctx).unwrap();

    let placement = result.placement();
    assert_eq!(placement.stream_debug, expected_stream);
    assert_eq!(placement.memory_resource_debug, expected_mr);
    assert_eq!(last_exec_stream_value(), expected_stream);
    assert_eq!(last_exec_mr_value(), expected_mr);

    reset_sync_observation();
    assert_eq!(result.to_vec_strict::<i16>().unwrap(), vec![1970, 1998]);
    assert_eq!(last_sync_stream_value(), expected_stream);
}

#[test]
fn datetime_tests_cases() {
    crate::common::run_named_cases(&[
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
        (
            "test_datetime_add_calendrical_months_rejects_non_timestamp_without_component_context",
            test_datetime_add_calendrical_months_rejects_non_timestamp_without_component_context,
        ),
        (
            "test_datetime_low_level_extract_component_matches_series_namespace",
            test_datetime_low_level_extract_component_matches_series_namespace,
        ),
        (
            "test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance",
            test_datetime_extract_in_uses_explicit_execution_context_and_output_provenance,
        ),
    ]);
}
