use std::path::Path;

const STREAM_POOL_HEADER: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/include/cudf/detail/utilities/stream_pool.hpp"));
const STREAM_POOL_SOURCE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/src/utilities/stream_pool.cpp"));
const STREAM_SHIM_HEADER: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/stream.h"));
const STREAM_SHIM_SOURCE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/stream.cpp"));
const DATASOURCE_SOURCE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/src/io/utilities/datasource.cpp"));
const HOST_MEMORY_SOURCE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/src/utilities/host_memory.cpp"));

#[test]
fn fork_stream_provider_contract_has_upstream_fallback_and_ordering_boundary() {
    assert!(STREAM_POOL_HEADER.contains("using stream_provider_fn"));
    assert!(STREAM_POOL_HEADER.contains("void install_stream_provider"));
    assert!(STREAM_POOL_SOURCE.contains("provider != nullptr ? provider(stream, count)"));
    assert!(STREAM_POOL_SOURCE.contains("global_cuda_stream_pool().get_streams(count)"));
    assert!(STREAM_POOL_SOURCE.contains("cudaEventRecord(event, stream)"));
    assert!(STREAM_POOL_SOURCE.contains("cudaStreamWaitEvent(strm, event, 0)"));
    assert!(STREAM_POOL_SOURCE.contains("void join_streams"));
}

#[test]
fn shim_route_contract_rejects_unowned_and_empty_attempt_stream_sets() {
    assert!(STREAM_SHIM_HEADER.contains("cudf_stream_provider_register_attempt"));
    assert!(STREAM_SHIM_HEADER.contains("cudf_stream_provider_unregister_attempt"));
    assert!(STREAM_SHIM_HEADER.contains("cudf_stream_provider_attach_attempt_stream"));
    assert!(STREAM_SHIM_SOURCE.contains("fork parent is not attached to an admitted attempt"));
    assert!(STREAM_SHIM_SOURCE.contains("auxiliary stream set is empty"));
    assert!(STREAM_SHIM_SOURCE.contains("served[index % served.size()]"));
    assert!(STREAM_SHIM_SOURCE.contains("alias fork subset is empty or outside the attempt fork pool"));
    assert!(STREAM_SHIM_SOURCE.contains("attempt_stream_routes.erase(it)"));
}

#[test]
fn audited_backend_helpers_do_not_use_bare_global_stream_pool() {
    assert!(!DATASOURCE_SOURCE.contains("global_cuda_stream_pool().get_stream"));
    assert!(!HOST_MEMORY_SOURCE.contains("global_cuda_stream_pool().get_stream"));
    assert!(DATASOURCE_SOURCE.contains("cudf::get_default_stream()"));
    assert!(HOST_MEMORY_SOURCE.contains("cudf::get_default_stream()"));
    assert!(Path::new(env!("CARGO_MANIFEST_DIR")).join("native/include/cudf_rust/stream.h").is_file());
}

#[test]
fn native_route_registry_accepts_aliases_and_drains_by_parent() {
    let parent = 0x1000usize as *mut std::ffi::c_void;
    let fork_streams = [0x2000usize as *mut std::ffi::c_void, 0x3000usize as *mut std::ffi::c_void];
    let alias = 0x4000usize as *mut std::ffi::c_void;

    // These registry operations only touch the native route map; no CUDA
    // stream is dereferenced until libcudf invokes the installed provider.
    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_install(), cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias, 0, 0), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    }
}

fn fork(stream: *mut std::ffi::c_void, count: usize) -> Vec<usize> {
    let mut forks = vec![std::ptr::null_mut::<std::ffi::c_void>(); count];
    // The probe only reads the route registry; the raw values it returns are
    // compared as identities and never dereferenced.
    let status = unsafe { cudf_sys::test_support::cudf_test_stream_provider_fork(stream, count, forks.as_mut_ptr()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    forks.into_iter().map(|fork| fork as usize).collect()
}

/// An alias carries its own fork subset: forks issued from it stay inside
/// that range and round-robin within it. A whole-pool caller -- the parent or
/// a subset-less alias -- is served only the forks no live subset alias
/// claims, so it can never hand another lane's permit holder one of that
/// lane's streams. Detaching releases exactly that alias's claim.
#[test]
fn whole_pool_callers_avoid_claimed_forks_and_lane_forks_stay_inside_their_lane() {
    let parent = 0x5000usize as *mut std::ffi::c_void;
    let fork_streams = [
        0x6000usize as *mut std::ffi::c_void,
        0x7000usize as *mut std::ffi::c_void,
        0x8000usize as *mut std::ffi::c_void,
        0x9000usize as *mut std::ffi::c_void,
    ];
    let lane_alias = 0xa000usize as *mut std::ffi::c_void;
    let whole_pool_alias = 0xb000usize as *mut std::ffi::c_void;
    let pool: Vec<usize> = fork_streams.iter().map(|fork| *fork as usize).collect();

    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_install(), cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, lane_alias, 2, 4),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, whole_pool_alias, 0, 0),
            cudf_sys::CUDF_SUCCESS
        );
    }

    assert_eq!(fork(lane_alias, 5), vec![pool[2], pool[3], pool[2], pool[3], pool[2]]);
    assert_eq!(fork(parent, 5), vec![pool[0], pool[1], pool[0], pool[1], pool[0]], "the parent avoids claimed forks");
    assert_eq!(
        fork(whole_pool_alias, 5),
        vec![pool[0], pool[1], pool[0], pool[1], pool[0]],
        "a subset-less alias avoids claimed forks"
    );

    // Nested forks stay inside the lane: work libcudf placed on one of the
    // lane's forks forks again from that stream, which must not reach another
    // lane's streams. An unclaimed fork is a whole-pool caller like any other.
    assert_eq!(fork(fork_streams[2], 3), vec![pool[2], pool[3], pool[2]]);
    assert_eq!(fork(fork_streams[0], 3), vec![pool[0], pool[1], pool[0]], "unclaimed forks avoid claimed forks");

    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(lane_alias), cudf_sys::CUDF_SUCCESS);
    }
    assert_eq!(fork(parent, 4), pool, "detach releases the claim with the alias");
    assert_eq!(fork(fork_streams[2], 4), pool, "a released fork is served the whole pool again");

    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(whole_pool_alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}

/// Every fork claimed by lanes leaves a whole-pool caller nothing disjoint to
/// serve. That state means the attempt has no auxiliary permit left for
/// another holder, so the whole pool is the only answer that can be correct.
#[test]
fn a_fully_claimed_pool_falls_back_to_serving_every_fork() {
    let parent = 0x11000usize as *mut std::ffi::c_void;
    let fork_streams = [0x12000usize as *mut std::ffi::c_void, 0x13000usize as *mut std::ffi::c_void];
    let first_lane = 0x14000usize as *mut std::ffi::c_void;
    let second_lane = 0x15000usize as *mut std::ffi::c_void;
    let pool: Vec<usize> = fork_streams.iter().map(|fork| *fork as usize).collect();

    unsafe {
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, first_lane, 0, 1),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, second_lane, 1, 2),
            cudf_sys::CUDF_SUCCESS
        );
    }

    assert_eq!(fork(first_lane, 2), vec![pool[0], pool[0]]);
    assert_eq!(fork(second_lane, 2), vec![pool[1], pool[1]]);
    assert_eq!(fork(parent, 2), pool, "no fork is unclaimed, so the parent is served the whole pool");

    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(first_lane), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(second_lane), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}

/// Two live aliases may not claim the same fork, equal ranges included: each
/// detach releases exactly its own claim, so a shared range would leave one
/// alias confined to streams the other already released.
#[test]
fn alias_fork_subset_rejects_a_range_overlapping_a_live_claim() {
    let parent = 0x16000usize as *mut std::ffi::c_void;
    let fork_streams = [
        0x17000usize as *mut std::ffi::c_void,
        0x18000usize as *mut std::ffi::c_void,
        0x19000usize as *mut std::ffi::c_void,
    ];
    let held = 0x1a000usize as *mut std::ffi::c_void;
    let contender = 0x1b000usize as *mut std::ffi::c_void;

    unsafe {
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, held, 0, 2), cudf_sys::CUDF_SUCCESS);
        for (fork_begin, fork_end) in [(0usize, 2usize), (0, 1), (1, 3), (1, 2)] {
            assert_eq!(
                cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, contender, fork_begin, fork_end),
                cudf_sys::CUDF_ERROR_INVALID_OPERATION,
                "subset [{fork_begin}, {fork_end}) overlaps the live claim [0, 2)"
            );
        }
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, contender, 2, 3),
            cudf_sys::CUDF_SUCCESS,
            "a disjoint claim on the same route is accepted"
        );
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(held), cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, held, 0, 2),
            cudf_sys::CUDF_SUCCESS,
            "the released range is claimable again"
        );
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(held), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(contender), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}

/// An alias subset must name a non-empty range inside the attempt fork pool.
#[test]
fn alias_fork_subset_rejects_empty_inverted_and_out_of_pool_ranges() {
    let parent = 0xc000usize as *mut std::ffi::c_void;
    let fork_streams = [0xd000usize as *mut std::ffi::c_void, 0xe000usize as *mut std::ffi::c_void];
    let alias = 0xf000usize as *mut std::ffi::c_void;

    unsafe {
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
        for (fork_begin, fork_end) in [(1usize, 1usize), (2, 1), (0, 3), (2, 3)] {
            assert_eq!(
                cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias, fork_begin, fork_end),
                cudf_sys::CUDF_ERROR_INVALID_ARGUMENT,
                "subset [{fork_begin}, {fork_end}) is not inside the fork pool"
            );
        }
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias, 0, 2),
            cudf_sys::CUDF_SUCCESS,
            "a rejected subset leaves the alias attachable"
        );
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}
