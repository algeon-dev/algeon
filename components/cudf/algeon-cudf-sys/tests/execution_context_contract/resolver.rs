use super::*;

#[test]
fn required_exec_options_resolver_rejects_ambient_execution_and_does_not_install_current_mr() {
    let body = function_region(
        EXECUTION_CONTEXT_CPP,
        "cudf_resolve_required_exec_options",
        "void cudf_record_synchronized_stream",
    );

    for required_marker in [
        "options == nullptr",
        "CUDF_EXEC_FLAG_DEVICE_EXPLICIT) == 0",
        "options->stream == nullptr",
        "options->mr == nullptr",
        "options->reserved",
        "guard.activate(options->device)",
    ] {
        assert!(
            body.contains(required_marker),
            "strict resolver must validate `{required_marker}` before production execution"
        );
    }

    assert!(
        source_contract_support::contains_normalized(
            body,
            "cudf::rust::test_support::record_exec_context(reinterpret_cast<std::uintptr_t>((*out)->stream.value()), (*out)->mr_raw);",
        ),
        "strict resolver must observe the resolved stream and memory resource before production execution"
    );

    for forbidden_marker in [
        "cudf_make_exec_context",
        "get_current_device_resource",
        "get_current_device_resource_ref",
        "install_current_device_resource",
        "cudf::get_default_stream",
    ] {
        assert!(
            !body.contains(forbidden_marker),
            "strict resolver must not use ambient/default execution state via `{forbidden_marker}`"
        );
    }
}

#[test]
fn execution_context_memory_resource_ref_retains_type_erased_resource_lifetime() {
    assert!(
        EXECUTION_CONTEXT_INTERNAL_HPP
            .contains("std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr"),
        "26.06 execution contexts must retain the CCCL type-erased resource object that backs \
         copied resource refs"
    );
    assert!(
        source_contract_support::contains_normalized(
            EXECUTION_CONTEXT_CPP,
            "return std::make_shared<cuda::mr::any_resource<cuda::mr::device_accessible>>( rmm::mr::any_device_resource_from_raw(raw));"
        ),
        "execution contexts must build the retained any_resource from the shim-owned raw resource \
         identity, not from ambient current-resource state"
    );
    assert!(
        EXECUTION_CONTEXT_CPP.contains("resource_ref_lifetime"),
        "execution contexts must retain the any_resource lifetime alongside the owner lifetime"
    );
    assert!(
        !EXECUTION_CONTEXT_CPP.contains("device_memory_resource_view"),
        "execution-context code should not depend on the removed pre-26.06 resource-view internals"
    );
}

#[test]
fn permissive_exec_options_resolver_never_swaps_current_resource() {
    let body = function_region(
        EXECUTION_CONTEXT_CPP,
        "cudf_status_t cudf_resolve_exec_options",
        "cudf_status_t\ncudf_resolve_required_exec_options",
    );

    // Device-domain invariant: the current resource is installed once when a
    // device's memory domain is created and never changes afterwards, so
    // resolution may read the tracked identity (the helper carries its own
    // map lock) but must never install, swap, or lock around it.
    for forbidden_marker in [
        "install_current_device_resource",
        "cudf_set_current_device_resource_raw",
        "cudf_current_device_resource_mutex",
        "cudf_current_device_resource_guard",
    ] {
        assert!(
            !body.contains(forbidden_marker),
            "permissive resolver must not touch process/device current-resource state via `{forbidden_marker}`"
        );
    }

    let default_mr_lookups = source_contract_support::normalize_whitespace(body)
        .match_indices("out->mr_raw=cudf_tracked_current_device_resource_raw();")
        .count();
    assert_eq!(
        default_mr_lookups, 2,
        "permissive resolver should only read ambient current-resource identity for the null-options and default-options paths"
    );
}
