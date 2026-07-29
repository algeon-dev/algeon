#include "rmm_interop_bridge.hpp"

#include "memory_resource_compat.hpp"

#include <rmm/mr/per_device_resource.hpp>

extern "C" void cudf_rust_rmm_set_current_device_resource(void* resource)
{
  auto* const raw = static_cast<rmm::mr::device_memory_resource*>(resource);
  rmm::mr::set_current_device_resource(rmm::mr::any_device_resource_from_raw(raw));
}

extern "C" int32_t cudf_rust_rmm_current_device_resource_is(void* resource)
{
  auto* const raw = static_cast<rmm::mr::device_memory_resource*>(resource);
  auto expected_resource = rmm::mr::device_memory_resource_pointer{raw};
  auto const expected    = rmm::device_async_resource_ref{expected_resource};
  return rmm::mr::get_current_device_resource_ref() == expected ? 1 : 0;
}
