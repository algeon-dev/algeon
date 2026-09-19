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
    assert!(STREAM_SHIM_SOURCE.contains("fork parent is not attached to an admitted attempt"));
    assert!(STREAM_SHIM_SOURCE.contains("auxiliary stream set is empty"));
    assert!(STREAM_SHIM_SOURCE.contains("index % route->fork_streams.size()"));
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
        assert_eq!(cudf_sys::cudf_stream_provider_attach_attempt_stream(parent, alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_detach_attempt_stream(alias), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_SUCCESS);
        assert_eq!(cudf_sys::cudf_stream_provider_unregister_attempt(parent), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    }
}
