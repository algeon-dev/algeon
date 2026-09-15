#include "cuml_rust/error.h"
#include "shim_common.hpp"

#include <cudf_rust/error_boundary.hpp>
#include <cuml/version_config.hpp>
#include <raft/core/error.hpp>
#include <rmm/error.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace {
constexpr std::int32_t k_no_cuda_error = -1;
struct error_state {
  cuml_error_detail_t detail{};
  bool pending{false};
};
thread_local error_state g_error;

void reset() noexcept
{
  g_error.detail = {};
  g_error.detail.status = CUML_ERROR_UNKNOWN;
  g_error.detail.failure_source = CUML_ERROR_FAILURE_SOURCE_UNKNOWN;
  g_error.detail.cuda_error = k_no_cuda_error;
  g_error.pending = false;
}

void copy_message(char const* message) noexcept
{
  auto const* value = message == nullptr ? "Unknown error" : message;
  auto const size = std::min(std::strlen(value), sizeof(g_error.detail.message) - 1);
  std::memcpy(g_error.detail.message, value, size);
  g_error.detail.message[size] = '\0';
  g_error.pending = true;
}

void set(cuml_status_t status, char const* message, std::int32_t cuda_error = k_no_cuda_error) noexcept
{
  g_error.detail.status = status;
  g_error.detail.failure_source = CUML_ERROR_FAILURE_SOURCE_UNKNOWN;
  g_error.detail.cuda_error = cuda_error;
  copy_message(message);
}

cuml_error_failure_source_t source(cudf_memory_failure_source_t value) noexcept
{
  switch (value) {
    case CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL: return CUML_ERROR_FAILURE_SOURCE_PHYSICAL;
    case CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT: return CUML_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT;
    default: return CUML_ERROR_FAILURE_SOURCE_UNKNOWN;
  }
}

cuml_status_t classify_cuda(std::int32_t* error) noexcept
{
  auto const reported = cudaGetLastError();
  auto const probe = cudaFree(nullptr);
  if (probe != cudaSuccess) {
    auto const synchronized = cudaDeviceSynchronize();
    *error = static_cast<std::int32_t>(probe);
    return probe == synchronized ? CUML_ERROR_FATAL_CUDA : CUML_ERROR_CUDA;
  }
  *error = static_cast<std::int32_t>(reported);
  return CUML_ERROR_CUDA;
}
}

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;

extern "C" cuml_status_t cuml_version_get(int32_t* major, int32_t* minor, int32_t* patch)
{
  cuml_clear_error();
  if (major == nullptr || minor == nullptr || patch == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "version output is null");
  }
  *major = CUML_VERSION_MAJOR;
  *minor = CUML_VERSION_MINOR;
  *patch = CUML_VERSION_PATCH;
  return CUML_SUCCESS;
}

extern "C" cuml_error_detail_t const* cuml_last_error_detail(void)
{
  return g_error.pending ? &g_error.detail : nullptr;
}
extern "C" char const* cuml_last_error_message(void)
{
  return g_error.pending ? g_error.detail.message : nullptr;
}
extern "C" void cuml_clear_error(void) { reset(); }

__attribute__((visibility("hidden"))) cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept
{
  set(status, message);
  return status;
}

cuml_status_t cuml_translate_exception() noexcept
{
  try {
    throw;
  } catch (cudf::rust::memory::attributed_out_of_memory const& error) {
    auto const identity = error.identity();
    set(CUML_ERROR_OUT_OF_MEMORY, error.what(), error.cuda_error());
    g_error.detail.failure_source = source(error.source());
    g_error.detail.domain_id = identity.primary_id;
    g_error.detail.member_id = identity.secondary_id;
    g_error.detail.generation = identity.generation;
    g_error.detail.requested_bytes = error.requested_bytes();
    g_error.detail.alignment = error.alignment();
    return CUML_ERROR_OUT_OF_MEMORY;
  } catch (rmm::out_of_memory const& error) {
    set(CUML_ERROR_OUT_OF_MEMORY, error.what()); return CUML_ERROR_OUT_OF_MEMORY;
  } catch (rmm::bad_alloc const& error) {
    set(CUML_ERROR_OUT_OF_MEMORY, error.what()); return CUML_ERROR_OUT_OF_MEMORY;
  } catch (rmm::cuda_error const& error) {
    std::int32_t cuda_error = k_no_cuda_error;
    auto const status = classify_cuda(&cuda_error);
    set(status, error.what(), cuda_error); return status;
  } catch (raft::logic_error const& error) {
    set(CUML_ERROR_INVALID_ARGUMENT, error.what()); return CUML_ERROR_INVALID_ARGUMENT;
  } catch (raft::exception const& error) {
    set(CUML_ERROR_INVALID_ARGUMENT, error.what()); return CUML_ERROR_INVALID_ARGUMENT;
  } catch (std::bad_alloc const& error) {
    set(CUML_ERROR_HOST_OUT_OF_MEMORY, error.what()); return CUML_ERROR_HOST_OUT_OF_MEMORY;
  } catch (std::overflow_error const& error) {
    set(CUML_ERROR_SIZE_OVERFLOW, error.what()); return CUML_ERROR_SIZE_OVERFLOW;
  } catch (std::invalid_argument const& error) {
    set(CUML_ERROR_INVALID_ARGUMENT, error.what()); return CUML_ERROR_INVALID_ARGUMENT;
  } catch (std::exception const& error) {
    set(CUML_ERROR_UNKNOWN, error.what()); return CUML_ERROR_UNKNOWN;
  } catch (...) {
    set(CUML_ERROR_UNKNOWN, "Unknown error"); return CUML_ERROR_UNKNOWN;
  }
}

cuml_status_t cuml_translate_cleanup_exception_noexcept() noexcept
{
  return cuml_translate_exception();
}
