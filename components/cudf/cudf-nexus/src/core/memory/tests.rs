use super::*;

#[test]
fn failure_info_preserves_unknown_allocation_domain_identity() {
    let raw = cudf_sys::cudf_memory_failure_info_t {
        flags: cudf_sys::CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN,
        reserved: [99, 1234, 5678],
        ..Default::default()
    };

    let info = DeviceMemoryFailureInfo::from_sys(raw);

    assert_eq!(
        info.allocation_domain,
        Some(DeviceMemoryFailureAllocationDomain::Unknown { kind: 99, primary_id: 1234, secondary_id: 5678 })
    );
}

#[test]
fn allocator_event_preserves_optional_samples_without_zero_fabrication() {
    let raw = cudf_sys::cudf_allocator_event_t {
        sequence: 17,
        device_ordinal: -1,
        completeness: cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_DOMAIN_IDENTITY
            | cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_ACTION,
        domain_kind: cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN,
        domain_id: 41,
        domain_generation: 9,
        domain_member_id: 3,
        requested_bytes: 4096,
        failure_source: cudf_sys::CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
        action: cudf_sys::CUDF_MEMORY_FAILURE_ACTION_DECLINE as i32,
        // This C ABI struct consists entirely of scalar fields and fixed
        // scalar arrays, so zero is a valid baseline for the omitted
        // optional fields.
        ..unsafe { std::mem::zeroed() }
    };

    let event = AllocatorEvent::from_sys(raw);

    assert_eq!(event.sequence, 17);
    assert_eq!(event.device, None);
    assert_eq!(event.monotonic_time_ns, None);
    assert_eq!(event.pool_usage, None);
    assert_eq!(event.action, Some(AllocatorEventAction::Decline));
    assert_eq!(
        event.allocation_domain,
        Some(AllocatorEventDomainIdentity {
            domain: DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id: 41, member_id: 3 },
            generation: 9,
        })
    );
}
