#pragma once

// This internal C ABI keeps RMM's header-defined per-device singleton outside
// the version-scripted shim. It is never installed or exposed to Rust callers.
extern "C" void cudf_rust_rmm_set_current_device_resource(void* resource);
