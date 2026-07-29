#pragma once

#include <cstdint>

// This internal C ABI keeps RMM's header-defined per-device singleton outside
// the version-scripted shim. It is never installed or exposed to Rust callers.
extern "C" void cudf_rust_rmm_set_current_device_resource(void* resource);
extern "C" int32_t cudf_rust_rmm_current_device_resource_is(void* resource);
