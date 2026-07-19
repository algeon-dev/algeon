#include "rmm_interop_bridge.hpp"

#include "memory_resource_compat.hpp"

#include <rmm/mr/per_device_resource.hpp>

extern "C" void cudf_rust_rmm_set_current_device_resource(void* resource)
{
  auto* const raw = static_cast<rmm::mr::device_memory_resource*>(resource);
  rmm::mr::set_current_device_resource(rmm::mr::any_device_resource_from_raw(raw));
}
