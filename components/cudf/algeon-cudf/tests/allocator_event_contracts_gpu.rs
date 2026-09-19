use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{
    AllocatorEventAction, AllocatorEventRead, DeviceId, DeviceMemoryFailureAllocationDomain, DeviceMemoryFailureSource,
    DeviceMemoryManager, ErrorKind, QueryMemoryDomainOptions, Runtime,
};

#[test]
fn allocator_event_ring_retains_callback_domain_identity_and_reports_overwrite() {
    let device = DeviceId::current().unwrap();
    let callback_count = Arc::new(AtomicUsize::new(0));
    let observed_callbacks = Arc::clone(&callback_count);
    let manager = DeviceMemoryManager::install_for_device_with_failure_callback_details(
        device,
        Arc::new(move |_| {
            observed_callbacks.fetch_add(1, Ordering::SeqCst);
        }),
    )
    .unwrap();
    let runtime =
        Runtime::builder().with_device(device).with_memory_resource(manager.resource().clone()).build().unwrap();
    let domain_id = 71_001;
    let member_id = 71_101;
    let domain = manager
        .query_memory_domain(domain_id, QueryMemoryDomainOptions::new(1).with_ceiling_bytes(8 * 1024 * 1024))
        .unwrap();
    let domain_identity = domain.identity().unwrap();
    assert_eq!(domain_identity.device, device);
    assert_eq!(domain_identity.domain_id, domain_id);
    assert_ne!(domain_identity.generation, 0);
    let _lease = domain.reserve_stream(runtime.stream(), member_id, 8 * 1024 * 1024).unwrap();

    let before = manager.allocator_event_ring_bounds().unwrap();
    let worker_count = 4;
    let start = Arc::new(Barrier::new(worker_count));
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let runtime = runtime.clone();
            let start = Arc::clone(&start);
            workers.push(scope.spawn(move || {
                start.wait();
                let error = runtime.series(&[1_i32, 2, 3]).unwrap_err();
                assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
            }));
        }
        for worker in workers {
            worker.join().expect("allocator-event worker must not panic");
        }
    });
    let concurrent_after = manager.allocator_event_ring_bounds().unwrap();
    assert_eq!(
        concurrent_after.next_sequence,
        before.next_sequence + worker_count as u64,
        "each concurrent declined allocation must reserve one ring sequence"
    );
    for sequence in before.next_sequence..concurrent_after.next_sequence {
        let event = match manager.read_allocator_event(sequence).unwrap() {
            AllocatorEventRead::Available(event) => event,
            other => panic!("concurrent allocator event {sequence} must be stable, got {other:?}"),
        };
        assert_eq!(event.sequence, sequence);
        assert_eq!(event.device, Some(device));
        assert_eq!(event.source, DeviceMemoryFailureSource::ReservationLimit);
        assert_eq!(
            event.allocation_domain,
            Some(cudf::AllocatorEventDomainIdentity {
                domain: DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id },
                generation: domain_identity.generation,
            })
        );
        assert_eq!(event.action, Some(AllocatorEventAction::Decline));
    }

    let before_overflow = concurrent_after;
    let callback_events = usize::try_from(before_overflow.capacity).unwrap() + 1;
    for _ in 0..callback_events {
        let error = runtime.series(&[1_i32, 2, 3]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
    }

    let after = manager.allocator_event_ring_bounds().unwrap();
    assert_eq!(after.capacity, before_overflow.capacity);
    assert!(
        after.next_sequence >= before_overflow.next_sequence + callback_events as u64,
        "each declined query-domain allocation failure must write an event"
    );
    assert!(
        after.first_available_sequence.is_some_and(|first| first > before_overflow.next_sequence),
        "more events than the fixed capacity must overwrite the oldest sequence"
    );
    assert_eq!(manager.read_allocator_event(before_overflow.next_sequence).unwrap(), AllocatorEventRead::Overwritten);
    assert_eq!(manager.read_allocator_event(after.next_sequence).unwrap(), AllocatorEventRead::NotYetRecorded);

    let last = match manager.read_allocator_event(after.next_sequence - 1).unwrap() {
        AllocatorEventRead::Available(event) => event,
        other => panic!("latest allocator event must be available, got {other:?}"),
    };
    assert_eq!(last.sequence, after.next_sequence - 1);
    assert_eq!(last.device, Some(device));
    assert_eq!(last.source, DeviceMemoryFailureSource::ReservationLimit);
    assert_eq!(last.action, Some(AllocatorEventAction::Decline));
    assert!(last.completeness.action);
    assert_eq!(
        last.allocation_domain,
        Some(cudf::AllocatorEventDomainIdentity {
            domain: DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, member_id },
            generation: domain_identity.generation,
        })
    );
    assert!(callback_count.load(Ordering::SeqCst) >= callback_events);
}
