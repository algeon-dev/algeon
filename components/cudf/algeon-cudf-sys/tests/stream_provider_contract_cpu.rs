use std::path::Path;

use cudf_sys::test_support;

const STREAM_POOL_HEADER: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/include/cudf/detail/utilities/stream_pool.hpp"));
const STREAM_POOL_SOURCE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/src/utilities/stream_pool.cpp"));
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
        assert_eq!(
            cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias, std::ptr::null_mut()),
            cudf_sys::CUDF_SUCCESS,
            "an attach accepts a null out pointer"
        );
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    }
}

/// A route registered with fewer than two forks has no auxiliary fork to
/// serve any alias, only a reserved parent fork, so it is rejected at
/// registration and admits no caller that it cannot isolate.
#[test]
fn a_route_without_an_auxiliary_fork_is_refused() {
    let parent = 0x1a000usize as *mut std::ffi::c_void;
    let reserved_only = [0x1b000usize as *mut std::ffi::c_void];

    let status =
        unsafe { cudf_sys::cudf_stream_provider_register_attempt(parent, reserved_only.as_ptr(), reserved_only.len()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT, "a reserved fork alone backs no alias");
}

/// Attaches `alias` to `parent`, returning the auxiliary fork index the route
/// chose for it.
fn attach(parent: *mut std::ffi::c_void, alias: *mut std::ffi::c_void) -> usize {
    let mut claimed = usize::MAX;
    // Only the native route map is touched; the raw stream values stand in for
    // identities and are never dereferenced.
    let status = unsafe { cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias, &raw mut claimed) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "an attach succeeds while an auxiliary fork is free");
    claimed
}

fn detach(alias: *mut std::ffi::c_void) {
    let status = unsafe { cudf_sys::cudf_stream_provider_detach_attempt_stream(alias) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

fn fork(stream: *mut std::ffi::c_void, count: usize) -> Vec<usize> {
    let mut forks = vec![std::ptr::null_mut::<std::ffi::c_void>(); count];
    // The probe only reads the route registry; the raw values it returns are
    // compared as identities and never dereferenced.
    let status = unsafe { test_support::cudf_test_stream_provider_fork(stream, count, forks.as_mut_ptr()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    forks.into_iter().map(|fork| fork as usize).collect()
}

/// Every active caller of a route owns a fixed, disjoint fork for its
/// lifetime: the parent is served only its reserved fork (registered last),
/// each alias only the auxiliary fork the route claimed for it, and a fork
/// stream only itself, so a nested libcudf fork stays inside the caller that
/// owns it. Nothing resolves to a fork another live caller may be launching
/// kernels on. Detaching releases that alias's index for the next
/// claim.
#[test]
fn every_caller_owns_one_fixed_disjoint_fork() {
    let parent = 0x5000usize as *mut std::ffi::c_void;
    let fork_streams = [
        0x6000usize as *mut std::ffi::c_void,
        0x7000usize as *mut std::ffi::c_void,
        0x8000usize as *mut std::ffi::c_void,
        // The reserved fork of the parent is the last registered entry.
        0x9000usize as *mut std::ffi::c_void,
    ];
    let first_alias = 0xa000usize as *mut std::ffi::c_void;
    let second_alias = 0xa100usize as *mut std::ffi::c_void;
    let third_alias = 0xb000usize as *mut std::ffi::c_void;
    let pool: Vec<usize> = fork_streams.iter().map(|fork| *fork as usize).collect();
    let reserved = pool[pool.len() - 1];

    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_install(), cudf_sys::CUDF_SUCCESS);
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
    }
    assert_eq!(fork(parent, 3), vec![reserved; 3], "the parent is served only its reserved fork");

    assert_eq!(attach(parent, first_alias), 0, "the first claim takes the lowest auxiliary index");
    assert_eq!(attach(parent, second_alias), 1, "the next claim takes the next free auxiliary index");
    assert_eq!(attach(parent, third_alias), 2);

    assert_eq!(fork(first_alias, 3), vec![pool[0]; 3], "an alias forks only its own claimed stream");
    assert_eq!(fork(second_alias, 3), vec![pool[1]; 3]);
    assert_eq!(fork(third_alias, 3), vec![pool[2]; 3]);
    assert_eq!(
        fork(parent, 3),
        vec![reserved; 3],
        "three live aliases leave the parent on its reserved fork, never on theirs"
    );

    // Nested forks stay inside the owner: work libcudf placed on a caller's
    // fork forks again from that stream, which must not reach another
    // caller's stream.
    assert_eq!(fork(fork_streams[0], 2), vec![pool[0]; 2], "a claimed fork nests onto its owner's fork");
    assert_eq!(fork(fork_streams[3], 2), vec![reserved; 2], "the reserved fork nests onto itself");

    detach(second_alias);
    assert_eq!(fork(parent, 2), vec![reserved; 2], "a detach does not move the parent off its reserved fork");
    assert_eq!(attach(parent, second_alias), 1, "the freed index is the lowest free one for a later claim");

    detach(first_alias);
    detach(second_alias);
    detach(third_alias);
    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}

/// An attach onto a route whose auxiliary forks are all claimed fails typed
/// and does not give a second caller a fork a live one already owns. Only a
/// caller without a fresh attempt auxiliary permit can reach that state, since
/// live claims never outnumber permit holders.
#[test]
fn a_full_auxiliary_pool_refuses_a_further_attach() {
    let parent = 0x11000usize as *mut std::ffi::c_void;
    let fork_streams = [
        0x12000usize as *mut std::ffi::c_void,
        0x13000usize as *mut std::ffi::c_void,
        0x13100usize as *mut std::ffi::c_void,
    ];
    let first_alias = 0x14000usize as *mut std::ffi::c_void;
    let second_alias = 0x15000usize as *mut std::ffi::c_void;
    let contender = 0x15100usize as *mut std::ffi::c_void;
    let pool: Vec<usize> = fork_streams.iter().map(|fork| *fork as usize).collect();

    unsafe {
        assert_eq!(
            cudf_sys::cudf_stream_provider_register_attempt(parent, fork_streams.as_ptr(), fork_streams.len()),
            cudf_sys::CUDF_SUCCESS
        );
    }
    assert_eq!(attach(parent, first_alias), 0);
    assert_eq!(attach(parent, second_alias), 1);

    assert_eq!(fork(first_alias, 2), vec![pool[0]; 2]);
    assert_eq!(fork(second_alias, 2), vec![pool[1]; 2]);
    assert_eq!(fork(parent, 2), vec![pool[2]; 2], "a full auxiliary pool leaves the parent on its reserved fork");

    let mut claimed = usize::MAX;
    assert_eq!(
        unsafe { cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, contender, &raw mut claimed) },
        cudf_sys::CUDF_ERROR_INVALID_OPERATION,
        "an attach onto a fully claimed auxiliary pool fails typed rather than sharing a live caller's fork"
    );

    detach(first_alias);
    assert_eq!(attach(parent, contender), 0, "a detach frees exactly that index for the refused caller");

    detach(contender);
    detach(second_alias);
    unsafe {
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
    }
}
