#include "cudf_rust/test_support.h"

#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>

extern "C" cudf_status_t cudf_rust_test_support_io_kvikio_active_io_begin();
extern "C" cudf_status_t cudf_rust_test_support_io_kvikio_active_io_end();

namespace {
constexpr std::size_t k_exec_mr_observation_capacity = 64;

thread_local std::uintptr_t g_last_exec_stream_value = std::numeric_limits<std::uintptr_t>::max();
thread_local std::uintptr_t g_last_exec_sync_stream_value =
  std::numeric_limits<std::uintptr_t>::max();
thread_local std::uint64_t g_exec_observation_sequence             = 0;
thread_local std::uint64_t g_last_exec_sync_sequence               = 0;
thread_local void* g_last_memory_resource_destroy_raw              = nullptr;
thread_local std::uint64_t g_last_memory_resource_destroy_sequence = 0;
thread_local void* g_last_exec_mr_raw                              = nullptr;
thread_local int32_t g_last_exec_device                            = -1;
thread_local std::array<void*, k_exec_mr_observation_capacity> g_exec_mr_observations{};
thread_local std::size_t g_exec_mr_observation_start = 0;
thread_local std::size_t g_exec_mr_observation_count = 0;
thread_local bool g_exec_mr_observation_overflowed   = false;

thread_local bool g_fail_next_column_view_metadata_read = false;
enum class fixed_width_host_import_fault {
  none,
  after_data_copy,
  after_null_mask_copy,
};
thread_local fixed_width_host_import_fault g_next_fixed_width_host_import_fault =
  fixed_width_host_import_fault::none;
enum class strings_host_import_fault {
  none,
  after_offsets_copy,
  after_chars_copy,
  after_null_mask_copy,
  before_make_strings_column,
  after_make_strings_column,
};
thread_local strings_host_import_fault g_next_strings_host_import_fault =
  strings_host_import_fault::none;

thread_local int32_t g_fail_column_handle_release_countdown            = -1;
thread_local bool g_fail_next_table_release_columns_handle_array_alloc = false;
thread_local bool g_fail_next_table_view_negative_num_columns          = false;
thread_local bool g_fail_next_table_view_negative_num_rows             = false;
thread_local bool g_fail_next_groupby_negative_total_columns           = false;
thread_local cudf_status_t g_next_provenance_destroy_failure           = CUDF_SUCCESS;

thread_local int32_t g_apply_boolean_mask_call_count         = 0;
thread_local int32_t g_row_predicate_filter_call_count       = 0;
thread_local int32_t g_row_predicate_compute_mask_call_count = 0;
thread_local int32_t g_row_predicate_final_filter_call_count = 0;

std::atomic<int32_t> g_stream_wait_call_count{0};
std::atomic<std::uintptr_t> g_last_wait_stream_raw{0};
std::atomic<std::uintptr_t> g_last_wait_event_raw{0};
std::atomic<int32_t> g_raw_record_call_count{0};
std::atomic<std::uintptr_t> g_last_raw_record_stream_raw{0};
std::atomic<std::uintptr_t> g_last_raw_record_event_raw{0};
std::atomic<int32_t> g_event_synchronize_call_count{0};
std::atomic<int32_t> g_stream_synchronize_call_count{0};
std::atomic<std::uint64_t> g_allocation_attempt_count{0};
std::mutex g_query_domain_allocation_failure_mutex;
bool g_query_domain_allocation_failure_armed = false;
std::uint64_t g_query_domain_allocation_failure_domain_id = 0;
std::uint64_t g_query_domain_allocation_failure_remaining = 0;
std::mutex g_synchronize_block_mutex;
std::condition_variable g_synchronize_block_cv;
bool g_block_event_synchronize = false;
bool g_fail_event_synchronize  = false;
std::mutex g_event_observation_mutex;
std::unordered_map<std::uintptr_t, int32_t> g_stream_wait_counts_by_stream_raw;
std::unordered_map<std::uintptr_t, int32_t> g_stream_wait_counts_by_event_raw;
std::unordered_map<std::uintptr_t, int32_t> g_raw_record_counts_by_stream_raw;
std::unordered_map<std::uintptr_t, int32_t> g_event_synchronize_counts_by_event_handle_raw;
std::mutex g_stream_synchronize_block_mutex;
std::condition_variable g_stream_synchronize_block_cv;
bool g_block_stream_synchronize = false;
bool g_fail_stream_synchronize  = false;

void reset_exec_observation()
{
  g_last_exec_stream_value                = std::numeric_limits<std::uintptr_t>::max();
  g_last_exec_sync_stream_value           = std::numeric_limits<std::uintptr_t>::max();
  g_exec_observation_sequence             = 0;
  g_last_exec_sync_sequence               = 0;
  g_last_memory_resource_destroy_raw      = nullptr;
  g_last_memory_resource_destroy_sequence = 0;
  g_last_exec_mr_raw                      = nullptr;
  g_last_exec_device                      = -1;
  g_exec_mr_observations.fill(nullptr);
  g_exec_mr_observation_start      = 0;
  g_exec_mr_observation_count      = 0;
  g_exec_mr_observation_overflowed = false;
}

void* exec_mr_observation_at(std::size_t index)
{
  auto const physical = (g_exec_mr_observation_start + index) % g_exec_mr_observations.size();
  return g_exec_mr_observations[physical];
}

bool take_flag(bool& flag)
{
  if (!flag) { return false; }
  flag = false;
  return true;
}

template <typename Enum>
bool take_fault(Enum& current, Enum expected)
{
  if (current != expected) { return false; }
  current = Enum::none;
  return true;
}

std::array<int64_t, 8> g_synthetic_column_view_dummy_storage{};
}  // namespace

namespace cudf::rust::test_support {
void record_exec_context(std::uintptr_t stream_raw, void* mr_raw)
{
  g_last_exec_stream_value = stream_raw;
  g_last_exec_mr_raw       = mr_raw;
  if (g_exec_mr_observation_count < g_exec_mr_observations.size()) {
    auto const physical =
      (g_exec_mr_observation_start + g_exec_mr_observation_count) % g_exec_mr_observations.size();
    g_exec_mr_observations[physical] = mr_raw;
    ++g_exec_mr_observation_count;
  } else {
    g_exec_mr_observations[g_exec_mr_observation_start] = mr_raw;
    g_exec_mr_observation_start = (g_exec_mr_observation_start + 1) % g_exec_mr_observations.size();
    g_exec_mr_observation_overflowed = true;
  }

  int32_t device = -1;
  if (cudaGetDevice(&device) == cudaSuccess) { g_last_exec_device = device; }
}

void record_synchronized_stream(std::uintptr_t stream_raw)
{
  g_last_exec_sync_stream_value = stream_raw;
  g_last_exec_sync_sequence     = ++g_exec_observation_sequence;
}

void record_stream_synchronize()
{
  g_stream_synchronize_call_count.fetch_add(1, std::memory_order_relaxed);
}

bool stream_synchronize_should_fail()
{
  std::scoped_lock lock(g_stream_synchronize_block_mutex);
  return g_fail_stream_synchronize;
}

void wait_for_stream_synchronize_release()
{
  std::unique_lock lock(g_stream_synchronize_block_mutex);
  g_stream_synchronize_block_cv.wait(lock, [] { return !g_block_stream_synchronize; });
}

void record_memory_resource_destroy(void* mr_raw)
{
  g_last_memory_resource_destroy_raw      = mr_raw;
  g_last_memory_resource_destroy_sequence = ++g_exec_observation_sequence;
}

bool take_column_view_metadata_read_failure()
{
  return take_flag(g_fail_next_column_view_metadata_read);
}

bool take_fixed_width_host_import_after_data_copy()
{
  return take_fault(g_next_fixed_width_host_import_fault,
                    fixed_width_host_import_fault::after_data_copy);
}

bool take_fixed_width_host_import_after_null_mask_copy()
{
  return take_fault(g_next_fixed_width_host_import_fault,
                    fixed_width_host_import_fault::after_null_mask_copy);
}

bool take_strings_host_import_after_offsets_copy()
{
  return take_fault(g_next_strings_host_import_fault,
                    strings_host_import_fault::after_offsets_copy);
}

bool take_strings_host_import_after_chars_copy()
{
  return take_fault(g_next_strings_host_import_fault, strings_host_import_fault::after_chars_copy);
}

bool take_strings_host_import_after_null_mask_copy()
{
  return take_fault(g_next_strings_host_import_fault,
                    strings_host_import_fault::after_null_mask_copy);
}

bool take_strings_host_import_before_make_strings_column()
{
  return take_fault(g_next_strings_host_import_fault,
                    strings_host_import_fault::before_make_strings_column);
}

bool take_strings_host_import_after_make_strings_column()
{
  return take_fault(g_next_strings_host_import_fault,
                    strings_host_import_fault::after_make_strings_column);
}

bool take_column_handle_release_failure()
{
  if (g_fail_column_handle_release_countdown < 0) { return false; }
  if (g_fail_column_handle_release_countdown == 0) {
    g_fail_column_handle_release_countdown = -1;
    return true;
  }
  --g_fail_column_handle_release_countdown;
  return false;
}

bool take_table_release_columns_handle_array_alloc_failure()
{
  return take_flag(g_fail_next_table_release_columns_handle_array_alloc);
}

bool take_table_view_negative_num_columns_failure()
{
  return take_flag(g_fail_next_table_view_negative_num_columns);
}

bool take_table_view_negative_num_rows_failure()
{
  return take_flag(g_fail_next_table_view_negative_num_rows);
}

bool take_groupby_negative_total_columns_failure()
{
  return take_flag(g_fail_next_groupby_negative_total_columns);
}

cudf_status_t take_provenance_destroy_failure()
{
  auto const status                 = g_next_provenance_destroy_failure;
  g_next_provenance_destroy_failure = CUDF_SUCCESS;
  return status;
}

void record_apply_boolean_mask() { ++g_apply_boolean_mask_call_count; }
void record_row_predicate_filter() { ++g_row_predicate_filter_call_count; }
void record_row_predicate_compute_mask() { ++g_row_predicate_compute_mask_call_count; }
void record_row_predicate_final_filter() { ++g_row_predicate_final_filter_call_count; }

void record_event_synchronize(std::uintptr_t event_handle_raw)
{
  g_event_synchronize_call_count.fetch_add(1, std::memory_order_relaxed);
  std::scoped_lock lock(g_event_observation_mutex);
  g_event_synchronize_counts_by_event_handle_raw[event_handle_raw] += 1;
}

bool event_synchronize_should_fail()
{
  std::scoped_lock lock(g_synchronize_block_mutex);
  return g_fail_event_synchronize;
}

void wait_for_event_synchronize_release()
{
  std::unique_lock lock(g_synchronize_block_mutex);
  g_synchronize_block_cv.wait(lock, [] { return !g_block_event_synchronize; });
}

void record_stream_wait(std::uintptr_t stream_raw, std::uintptr_t event_raw)
{
  g_stream_wait_call_count.fetch_add(1, std::memory_order_relaxed);
  g_last_wait_stream_raw.store(stream_raw, std::memory_order_relaxed);
  g_last_wait_event_raw.store(event_raw, std::memory_order_relaxed);
  std::scoped_lock lock(g_event_observation_mutex);
  g_stream_wait_counts_by_stream_raw[stream_raw] += 1;
  g_stream_wait_counts_by_event_raw[event_raw] += 1;
}

void record_raw_event_record(std::uintptr_t stream_raw, std::uintptr_t event_raw)
{
  g_raw_record_call_count.fetch_add(1, std::memory_order_relaxed);
  g_last_raw_record_stream_raw.store(stream_raw, std::memory_order_relaxed);
  g_last_raw_record_event_raw.store(event_raw, std::memory_order_relaxed);
  std::scoped_lock lock(g_event_observation_mutex);
  g_raw_record_counts_by_stream_raw[stream_raw] += 1;
}

void record_allocation_attempt(std::size_t bytes) noexcept
{
  if (bytes != 0) { g_allocation_attempt_count.fetch_add(1, std::memory_order_relaxed); }
}

bool take_query_domain_allocation_failure(std::uint64_t domain_id)
{
  std::scoped_lock lock{g_query_domain_allocation_failure_mutex};
  if (!g_query_domain_allocation_failure_armed ||
      g_query_domain_allocation_failure_domain_id != domain_id) {
    return false;
  }
  if (g_query_domain_allocation_failure_remaining != 0) {
    --g_query_domain_allocation_failure_remaining;
    return false;
  }
  g_query_domain_allocation_failure_armed = false;
  return true;
}

}  // namespace cudf::rust::test_support

extern "C" {
void cudf_test_fail_next_table_release_columns_handle_array_alloc()
{
  g_fail_next_table_release_columns_handle_array_alloc = true;
}

void cudf_test_fail_column_handle_release_after(int32_t successful_releases)
{
  g_fail_column_handle_release_countdown = successful_releases;
}

cudf_status_t cudf_test_checked_owning_handle_size_add(size_t left, size_t right, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY
  {
    if (right > std::numeric_limits<size_t>::max() - left) {
      throw std::overflow_error(
        "size accounting overflow while testing "
        "owning-handle checked addition");
    }
    *out = left + right;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_checked_owning_handle_size_mul(size_t left, size_t right, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY
  {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
      throw std::overflow_error(
        "size accounting overflow while testing "
        "owning-handle checked multiplication");
    }
    *out = left * right;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_fail_next_provenance_destroy(cudf_status_t status)
{
  g_next_provenance_destroy_failure = status;
}

void cudf_test_memory_resource_reset_allocation_attempts()
{
  g_allocation_attempt_count.store(0, std::memory_order_relaxed);
}

void cudf_test_query_domain_fail_allocation_after(uint64_t domain_id,
                                                  uint64_t successful_allocations)
{
  std::scoped_lock lock{g_query_domain_allocation_failure_mutex};
  g_query_domain_allocation_failure_domain_id = domain_id;
  g_query_domain_allocation_failure_remaining = successful_allocations;
  g_query_domain_allocation_failure_armed     = true;
}

void cudf_test_query_domain_reset_allocation_failure()
{
  std::scoped_lock lock{g_query_domain_allocation_failure_mutex};
  g_query_domain_allocation_failure_armed = false;
}

cudf_status_t cudf_test_memory_resource_allocation_attempt_count(uint64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_allocation_attempt_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

void cudf_test_exec_reset_observation() { reset_exec_observation(); }

cudf_status_t cudf_test_exec_last_stream_value(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_sync_stream_value(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_sync_stream_value;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_sync_sequence(uint64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_sync_sequence;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_memory_resource_destroy_raw(void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_memory_resource_destroy_raw;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_memory_resource_destroy_sequence(uint64_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_memory_resource_destroy_sequence;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_mr_raw(void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_mr_raw;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_count(size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_exec_mr_observation_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_overflowed(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_exec_mr_observation_overflowed ? 1 : 0;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_observed_mr_raw_at(size_t index, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (index >= g_exec_mr_observation_count) {
    set_error("execution memory-resource observation index is out of range");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  *out = exec_mr_observation_at(index);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_exec_last_device(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_exec_device;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_default_stream_value(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = reinterpret_cast<uintptr_t>(cudf::get_default_stream().value());
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_stream_handle_value(cudf_stream_handle handle, uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out              = 0;
  void* raw         = nullptr;
  auto const status = cudf_stream_raw(handle, &raw);
  if (status != CUDF_SUCCESS) { return status; }
  *out = reinterpret_cast<uintptr_t>(raw);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_current_device_resource_raw(void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    *out = cudf_tracked_current_device_resource_raw();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_row_predicate_reset_observation()
{
  g_apply_boolean_mask_call_count         = 0;
  g_row_predicate_filter_call_count       = 0;
  g_row_predicate_compute_mask_call_count = 0;
  g_row_predicate_final_filter_call_count = 0;
}

cudf_status_t cudf_test_row_predicate_filter_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_filter_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_row_predicate_compute_mask_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_compute_mask_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_row_predicate_final_filter_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_row_predicate_final_filter_call_count;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_apply_boolean_mask_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_apply_boolean_mask_call_count;
  return CUDF_SUCCESS;
}

void cudf_test_fail_next_column_view_metadata_read()
{
  g_fail_next_column_view_metadata_read = true;
}

void cudf_test_fail_next_fixed_width_host_import_after_data_copy()
{
  g_next_fixed_width_host_import_fault = fixed_width_host_import_fault::after_data_copy;
}

void cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy()
{
  g_next_fixed_width_host_import_fault = fixed_width_host_import_fault::after_null_mask_copy;
}

void cudf_test_fail_next_strings_host_import_after_offsets_copy()
{
  g_next_strings_host_import_fault = strings_host_import_fault::after_offsets_copy;
}

void cudf_test_fail_next_strings_host_import_after_chars_copy()
{
  g_next_strings_host_import_fault = strings_host_import_fault::after_chars_copy;
}

void cudf_test_fail_next_strings_host_import_after_null_mask_copy()
{
  g_next_strings_host_import_fault = strings_host_import_fault::after_null_mask_copy;
}

void cudf_test_fail_next_strings_host_import_before_make_strings_column()
{
  g_next_strings_host_import_fault = strings_host_import_fault::before_make_strings_column;
}

void cudf_test_fail_next_strings_host_import_after_make_strings_column()
{
  g_next_strings_host_import_fault = strings_host_import_fault::after_make_strings_column;
}

cudf_status_t cudf_test_column_view_create_synthetic_fixed_width(cudf_type_id_t type_id,
                                                                 int32_t size,
                                                                 int32_t offset,
                                                                 cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (offset < 0) {
    set_error("offset must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("synthetic test view only supports fixed-width types");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto* view =
      new cudf::column_view(dtype,
                            static_cast<cudf::size_type>(size),
                            static_cast<void const*>(g_synthetic_column_view_dummy_storage.data()),
                            nullptr,
                            0,
                            static_cast<cudf::size_type>(offset));
    *out = reinterpret_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_column_view_create_synthetic_string_without_children(
  int32_t size, cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size <= 0) {
    set_error("size must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto* view = new cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                       static_cast<cudf::size_type>(size),
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       {});
    *out       = reinterpret_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_groupby_fail_next_negative_total_columns()
{
  g_fail_next_groupby_negative_total_columns = true;
}

void cudf_test_table_view_fail_next_negative_num_columns()
{
  g_fail_next_table_view_negative_num_columns = true;
}

void cudf_test_table_view_fail_next_negative_num_rows()
{
  g_fail_next_table_view_negative_num_rows = true;
}

void cudf_test_event_reset_observation()
{
  g_stream_wait_call_count.store(0, std::memory_order_relaxed);
  g_last_wait_stream_raw.store(0, std::memory_order_relaxed);
  g_last_wait_event_raw.store(0, std::memory_order_relaxed);
  g_raw_record_call_count.store(0, std::memory_order_relaxed);
  g_last_raw_record_stream_raw.store(0, std::memory_order_relaxed);
  g_last_raw_record_event_raw.store(0, std::memory_order_relaxed);
  g_event_synchronize_call_count.store(0, std::memory_order_relaxed);
  {
    std::scoped_lock lock(g_synchronize_block_mutex);
    g_block_event_synchronize = false;
    g_fail_event_synchronize  = false;
  }
  g_synchronize_block_cv.notify_all();
  std::scoped_lock lock(g_event_observation_mutex);
  g_stream_wait_counts_by_stream_raw.clear();
  g_stream_wait_counts_by_event_raw.clear();
  g_raw_record_counts_by_stream_raw.clear();
  g_event_synchronize_counts_by_event_handle_raw.clear();
}

void cudf_test_event_block_synchronize()
{
  std::scoped_lock lock(g_synchronize_block_mutex);
  g_block_event_synchronize = true;
}

void cudf_test_event_unblock_synchronize()
{
  {
    std::scoped_lock lock(g_synchronize_block_mutex);
    g_block_event_synchronize = false;
  }
  g_synchronize_block_cv.notify_all();
}

void cudf_test_event_fail_synchronize()
{
  std::scoped_lock lock(g_synchronize_block_mutex);
  g_fail_event_synchronize = true;
}

void cudf_test_event_allow_synchronize()
{
  std::scoped_lock lock(g_synchronize_block_mutex);
  g_fail_event_synchronize = false;
}

cudf_status_t cudf_test_event_synchronize_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_event_synchronize_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_synchronize_call_count_for_event_handle_raw(
  uintptr_t event_handle_raw, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::scoped_lock lock(g_event_observation_mutex);
  auto const it = g_event_synchronize_counts_by_event_handle_raw.find(event_handle_raw);
  *out          = it == g_event_synchronize_counts_by_event_handle_raw.end() ? 0 : it->second;
  return CUDF_SUCCESS;
}

void cudf_test_stream_synchronize_reset()
{
  g_stream_synchronize_call_count.store(0, std::memory_order_relaxed);
  {
    std::scoped_lock lock(g_stream_synchronize_block_mutex);
    g_block_stream_synchronize = false;
    g_fail_stream_synchronize  = false;
  }
  g_stream_synchronize_block_cv.notify_all();
}

void cudf_test_stream_block_synchronize()
{
  std::scoped_lock lock(g_stream_synchronize_block_mutex);
  g_block_stream_synchronize = true;
}

void cudf_test_stream_unblock_synchronize()
{
  {
    std::scoped_lock lock(g_stream_synchronize_block_mutex);
    g_block_stream_synchronize = false;
  }
  g_stream_synchronize_block_cv.notify_all();
}

void cudf_test_stream_fail_synchronize()
{
  std::scoped_lock lock(g_stream_synchronize_block_mutex);
  g_fail_stream_synchronize = true;
}

void cudf_test_stream_allow_synchronize()
{
  std::scoped_lock lock(g_stream_synchronize_block_mutex);
  g_fail_stream_synchronize = false;
}

cudf_status_t cudf_test_stream_synchronize_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_stream_synchronize_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_rust_test_support_stream_provider_fork(void* stream, size_t count, void** out_forks);

cudf_status_t cudf_test_stream_provider_fork(void* stream, size_t count, void** out_forks)
{
  return cudf_rust_test_support_stream_provider_fork(stream, count, out_forks);
}

cudf_status_t cudf_test_event_stream_wait_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_stream_wait_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_stream_wait_call_count_for_stream_raw(uintptr_t stream_raw,
                                                                    int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::scoped_lock lock(g_event_observation_mutex);
  auto const it = g_stream_wait_counts_by_stream_raw.find(stream_raw);
  *out          = it == g_stream_wait_counts_by_stream_raw.end() ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_stream_wait_call_count_for_event_raw(uintptr_t event_raw,
                                                                   int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::scoped_lock lock(g_event_observation_mutex);
  auto const it = g_stream_wait_counts_by_event_raw.find(event_raw);
  *out          = it == g_stream_wait_counts_by_event_raw.end() ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_wait_stream_raw(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_wait_stream_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_wait_event_raw(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_wait_event_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_raw_record_call_count(int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_raw_record_call_count.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_raw_record_call_count_for_stream_raw(uintptr_t stream_raw,
                                                                   int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  std::scoped_lock lock(g_event_observation_mutex);
  auto const it = g_raw_record_counts_by_stream_raw.find(stream_raw);
  *out          = it == g_raw_record_counts_by_stream_raw.end() ? 0 : it->second;
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_raw_record_stream_raw(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_raw_record_stream_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

cudf_status_t cudf_test_event_last_raw_record_event_raw(uintptr_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = g_last_raw_record_event_raw.load(std::memory_order_relaxed);
  return CUDF_SUCCESS;
}

int32_t cudf_rust_test_support_query_domain_route_pause_arm(uint64_t domain_id);
int32_t cudf_rust_test_support_query_domain_route_pause_reached();
void cudf_rust_test_support_query_domain_route_pause_release();

int32_t cudf_test_query_domain_route_pause_arm(uint64_t domain_id)
{
  return cudf_rust_test_support_query_domain_route_pause_arm(domain_id);
}

int32_t cudf_test_query_domain_route_pause_reached()
{
  return cudf_rust_test_support_query_domain_route_pause_reached();
}

void cudf_test_query_domain_route_pause_release()
{
  cudf_rust_test_support_query_domain_route_pause_release();
}

cudf_status_t cudf_test_io_kvikio_active_io_begin()
{
  return cudf_rust_test_support_io_kvikio_active_io_begin();
}

cudf_status_t cudf_test_io_kvikio_active_io_end()
{
  return cudf_rust_test_support_io_kvikio_active_io_end();
}

cudf_status_t cudf_test_throw_runtime_error()
{
  cudf_clear_error();
  try {
    throw std::runtime_error("non-io runtime failure");
  } catch (...) {
    return translate_exception();
  }
}

cudf_status_t cudf_test_throw_bad_alloc()
{
  cudf_clear_error();
  try {
    throw std::bad_alloc();
  } catch (...) {
    return translate_exception();
  }
}

cudf_status_t cudf_test_throw_invalid_argument()
{
  cudf_clear_error();
  try {
    throw std::invalid_argument("invalid argument from test shim");
  } catch (...) {
    return translate_exception();
  }
}
}  // extern "C"
