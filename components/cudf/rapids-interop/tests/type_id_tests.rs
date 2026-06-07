use rapids_interop::TypeId;

#[test]
fn second_and_millisecond_time_types_report_eight_byte_width() {
    assert_eq!(TypeId::TimestampDays.fixed_width_size_bytes(), Some(4));
    assert_eq!(TypeId::DurationDays.fixed_width_size_bytes(), Some(4));

    assert_eq!(TypeId::TimestampSeconds.fixed_width_size_bytes(), Some(8));
    assert_eq!(TypeId::TimestampMilliseconds.fixed_width_size_bytes(), Some(8));
    assert_eq!(TypeId::DurationSeconds.fixed_width_size_bytes(), Some(8));
    assert_eq!(TypeId::DurationMilliseconds.fixed_width_size_bytes(), Some(8));
}
