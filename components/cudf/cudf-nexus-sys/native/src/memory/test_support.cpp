#include "cudf_rust/test_support.h"

#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "memory/internal.hpp"
#include "memory_resource_compat.hpp"
#include "shim_common.h"

#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <atomic>
#include <memory>
#include <thread>

using namespace cudf::rust::memory;

extern "C" cudf_status_t cudf_rust_test_support_memory_manager_query_domain_pool_count_for_id(
  cudf_memory_manager_handle manager, uint64_t domain_id, size_t* out);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_create_stats(
  cudf_memory_resource_handle* out);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream, cudf_memory_resource_handle* out);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_allocation_count(
  cudf_memory_resource_handle handle, int64_t* out);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_active_allocation_count(
  cudf_memory_resource_handle handle, int64_t* out);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_set_current(
  cudf_memory_resource_handle handle);
extern "C" cudf_status_t cudf_rust_test_support_memory_resource_set_current_raw(void* raw_mr);

#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
namespace {

class fail_first_allocation_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  explicit fail_first_allocation_resource_adaptor(rmm::mr::device_memory_resource* upstream)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    if (!failed_once_.exchange(true, std::memory_order_acq_rel)) {
      throw rmm::out_of_memory{"test memory resource forced first allocation failure"};
    }
    return upstream_.allocate(stream, bytes, alignment);
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::atomic<bool> failed_once_{false};
};

}  // namespace

namespace {

struct query_domain_route_pause_state {
  std::atomic<bool> armed{false};
  std::atomic<bool> reached{false};
  std::atomic<bool> released{false};
  std::atomic<std::uint64_t> target_domain_id{0};
};

query_domain_route_pause_state& query_domain_route_pause()
{
  static query_domain_route_pause_state state;
  return state;
}

}  // namespace

namespace cudf::rust::test_support {

void pause_query_domain_route_if_armed(std::uint64_t domain_id)
{
  auto& state = query_domain_route_pause();
  if (!state.armed.load(std::memory_order_acquire) ||
      state.target_domain_id.load(std::memory_order_relaxed) != domain_id) {
    return;
  }
  state.reached.store(true, std::memory_order_release);
  while (!state.released.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  state.armed.store(false, std::memory_order_release);
}

}  // namespace cudf::rust::test_support

std::unique_ptr<rmm::mr::device_memory_resource> cudf_rust_make_fail_first_allocation_resource(
  rmm::mr::device_memory_resource* upstream)
{
  return std::make_unique<fail_first_allocation_resource_adaptor>(upstream);
}
#endif

#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
extern "C" {
int32_t cudf_rust_test_support_query_domain_route_pause_arm(uint64_t domain_id)
{
  auto& state = query_domain_route_pause();
  state.reached.store(false, std::memory_order_relaxed);
  state.released.store(false, std::memory_order_relaxed);
  state.target_domain_id.store(domain_id, std::memory_order_relaxed);
  state.armed.store(true, std::memory_order_release);
  return 1;
}

int32_t cudf_rust_test_support_query_domain_route_pause_reached()
{
  return query_domain_route_pause().reached.load(std::memory_order_acquire) ? 1 : 0;
}

void cudf_rust_test_support_query_domain_route_pause_release()
{
  auto& state = query_domain_route_pause();
  state.released.store(true, std::memory_order_release);
  state.armed.store(false, std::memory_order_release);
}

cudf_status_t cudf_rust_test_support_memory_manager_query_domain_pool_count_for_id(
  cudf_memory_manager_handle manager, uint64_t domain_id, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(manager, "manager is null");
  CUDF_RUST_TRY
  {
    auto manager_state = memory_manager_from_handle(manager);
    if (manager_state == nullptr) {
      set_error("memory manager handle is not registered");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto* adaptor = manager_state->owner->reservation_adaptor;
    if (adaptor == nullptr) {
      set_error("memory manager resource is not reservation-aware");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = adaptor->query_domain_pool_count_for_id(domain_id);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_create_stats(cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state   = std::make_unique<memory_resource_handle_state>();
    state->owner = std::make_shared<memory_resource_owner>();
    auto& owner  = *state->owner;
    owner.device = current_cuda_device();
    owner.upstream =
      std::make_shared<rmm::mr::device_memory_resource_box<rmm::mr::cuda_memory_resource>>();
    auto stats =
      std::make_unique<rmm::mr::device_memory_resource_box<rmm::mr::statistics_resource_adaptor>>(
        rmm::mr::any_device_resource_from_raw(owner.upstream.get()));
    owner.stats = &stats->resource();
    owner.mr    = std::move(stats);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream, cudf_memory_resource_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(upstream, "upstream is null");
  CUDF_RUST_TRY
  {
    auto state              = std::make_unique<memory_resource_handle_state>();
    state->owner            = std::make_shared<memory_resource_owner>();
    auto& owner             = *state->owner;
    auto& upstream_owner    = memory_resource_owner_from_handle(upstream);
    owner.upstream_lifetime = cudf_memory_resource_lifetime_from_handle(upstream);
    owner.async_pool        = upstream_owner.async_pool;
    owner.device            = upstream_owner.device;
    auto* upstream_mr       = cudf_memory_resource_raw_from_handle(upstream);
    owner.mr                = cudf_rust_make_fail_first_allocation_resource(upstream_mr);
    register_memory_resource_lifetime(state->owner);
    *out = reinterpret_cast<cudf_memory_resource_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_allocation_count(
  cudf_memory_resource_handle handle, int64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.stats == nullptr) {
      set_error("memory resource does not expose allocation statistics");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner.stats->get_allocations_counter().total;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_active_allocation_count(
  cudf_memory_resource_handle handle, int64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    if (owner.stats == nullptr) {
      set_error("memory resource does not expose allocation statistics");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *out = owner.stats->get_allocations_counter().value;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_set_current(cudf_memory_resource_handle handle)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto& owner = memory_resource_owner_from_handle(handle);
    cudf_set_current_device_resource_raw(owner.mr.get());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_rust_test_support_memory_resource_set_current_raw(void* raw_mr)
{
  CUDF_RUST_NULL_CHECK(raw_mr, "raw_mr is null");
  CUDF_RUST_TRY
  {
    cudf_set_current_device_resource_raw(static_cast<rmm::mr::device_memory_resource*>(raw_mr));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
#endif

extern "C" {
cudf_status_t cudf_test_memory_manager_query_domain_pool_count_for_id(
  cudf_memory_manager_handle manager, uint64_t domain_id, size_t* out)
{
  return cudf_rust_test_support_memory_manager_query_domain_pool_count_for_id(
    manager, domain_id, out);
}

cudf_status_t cudf_test_memory_resource_create_stats(cudf_memory_resource_handle* out)
{
  return cudf_rust_test_support_memory_resource_create_stats(out);
}

cudf_status_t cudf_test_memory_resource_create_fail_first_allocation(
  cudf_memory_resource_handle upstream, cudf_memory_resource_handle* out)
{
  return cudf_rust_test_support_memory_resource_create_fail_first_allocation(upstream, out);
}

cudf_status_t cudf_test_memory_resource_allocation_count(cudf_memory_resource_handle handle,
                                                         int64_t* out)
{
  return cudf_rust_test_support_memory_resource_allocation_count(handle, out);
}

cudf_status_t cudf_test_memory_resource_active_allocation_count(cudf_memory_resource_handle handle,
                                                                int64_t* out)
{
  return cudf_rust_test_support_memory_resource_active_allocation_count(handle, out);
}

cudf_status_t cudf_test_memory_resource_raw(cudf_memory_resource_handle handle, void** out)
{
  return cudf_memory_resource_raw(handle, out);
}

cudf_status_t cudf_test_memory_resource_set_current(cudf_memory_resource_handle handle)
{
  return cudf_rust_test_support_memory_resource_set_current(handle);
}

cudf_status_t cudf_test_memory_resource_set_current_raw(void* raw_mr)
{
  return cudf_rust_test_support_memory_resource_set_current_raw(raw_mr);
}
}  // extern "C"
