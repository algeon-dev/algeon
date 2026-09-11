#include "cudf_rust/error.h"

#include "cudf_rust/error_boundary.hpp"

#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <new>
#include <stdexcept>
#include <string>

namespace {

constexpr std::int32_t k_unknown_cuda_error = -1;

struct thread_error_state {
  cudf_error_detail_t detail{};
  bool has_error = false;
};

thread_local thread_error_state g_error;

void reset_error_detail() noexcept
{
  g_error.detail                 = {};
  g_error.detail.status          = CUDF_ERROR_UNKNOWN;
  g_error.detail.failure_source = CUDF_ERROR_FAILURE_SOURCE_UNKNOWN;
  g_error.detail.cuda_error     = k_unknown_cuda_error;
  g_error.has_error              = false;
}

void copy_error_message(char const* message) noexcept
{
  auto const* source = message == nullptr ? "Unknown error" : message;
  auto const length  = std::min(std::strlen(source), sizeof(g_error.detail.message) - 1);
  std::memcpy(g_error.detail.message, source, length);
  g_error.detail.message[length] = '\0';
}

void copy_error_message(std::string const& message) noexcept
{
  copy_error_message(message.c_str());
}

void set_error_detail(cudf_status_t status,
                      cudf_error_failure_source_t source,
                      std::uint64_t domain_id,
                      std::uint64_t member_id,
                      std::uint64_t generation,
                      std::size_t requested_bytes,
                      std::size_t alignment,
                      std::int32_t cuda_error,
                      char const* message) noexcept
{
  g_error.detail.status          = status;
  g_error.detail.failure_source = source;
  g_error.detail.domain_id      = domain_id;
  g_error.detail.member_id      = member_id;
  g_error.detail.generation     = generation;
  g_error.detail.requested_bytes = static_cast<std::uint64_t>(requested_bytes);
  g_error.detail.alignment      = static_cast<std::uint64_t>(alignment);
  g_error.detail.cuda_error     = cuda_error;
  copy_error_message(message);
  g_error.has_error = true;
}

cudf_error_failure_source_t translate_failure_source(cudf_memory_failure_source_t source) noexcept
{
  switch (source) {
    case CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL:
      return CUDF_ERROR_FAILURE_SOURCE_PHYSICAL;
    case CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT:
      return CUDF_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT;
    default:
      return CUDF_ERROR_FAILURE_SOURCE_UNKNOWN;
  }
}

void set_exception_detail(cudf_status_t status,
                          char const* message,
                          cudf_error_failure_source_t source = CUDF_ERROR_FAILURE_SOURCE_UNKNOWN,
                          std::uint64_t domain_id = 0,
                          std::uint64_t member_id = 0,
                          std::uint64_t generation = 0,
                          std::size_t requested_bytes = 0,
                          std::size_t alignment = 0,
                          std::int32_t cuda_error = k_unknown_cuda_error) noexcept
{
  set_error_detail(status,
                   source,
                   domain_id,
                   member_id,
                   generation,
                   requested_bytes,
                   alignment,
                   cuda_error,
                   message);
}

void set_attributed_exception_detail(cudf::rust::memory::attributed_out_of_memory const& error) noexcept
{
  auto const identity = error.identity();
  set_exception_detail(CUDF_ERROR_OUT_OF_MEMORY,
                       error.what(),
                       translate_failure_source(error.source()),
                       identity.primary_id,
                       identity.secondary_id,
                       identity.generation,
                       error.requested_bytes(),
                       error.alignment(),
                       error.cuda_error());
}

}  // namespace

void set_error(std::string const& msg)
{
  // Explicit validation paths do not pass their return status through this
  // helper; the caller's returned status is authoritative in Rust. Exception
  // paths use set_error_detail instead so their allocation facts are kept.
  g_error.detail                 = {};
  g_error.detail.status          = CUDF_ERROR_UNKNOWN;
  g_error.detail.failure_source = CUDF_ERROR_FAILURE_SOURCE_UNKNOWN;
  g_error.detail.cuda_error     = k_unknown_cuda_error;
  copy_error_message(msg);
  g_error.has_error = true;
}

void set_error(char const* msg)
{
  g_error.detail                 = {};
  g_error.detail.status          = CUDF_ERROR_UNKNOWN;
  g_error.detail.failure_source = CUDF_ERROR_FAILURE_SOURCE_UNKNOWN;
  g_error.detail.cuda_error     = k_unknown_cuda_error;
  copy_error_message(msg);
  g_error.has_error = true;
}

extern "C" {
cudf_error_detail_t const* cudf_last_error_detail(void)
{
  return g_error.has_error ? &g_error.detail : nullptr;
}

char const* cudf_last_error_message(void)
{
  return g_error.has_error ? g_error.detail.message : nullptr;
}

void cudf_clear_error(void)
{
  reset_error_detail();
}
}  // extern "C"

namespace {
cudf_status_t translate_exception_impl(cudf_status_t runtime_error_status)
{
  try {
    throw;  // re-throw current exception
  } catch (cudf::rust::memory::attributed_out_of_memory const& e) {
    set_attributed_exception_detail(e);
    return CUDF_ERROR_OUT_OF_MEMORY;
  } catch (rmm::out_of_memory const& e) {
    // A generic RMM exception does not identify which limiting resource
    // failed. Only an owner-attributed exception may claim physical or
    // reservation-limit provenance at this boundary.
    set_exception_detail(CUDF_ERROR_OUT_OF_MEMORY, e.what());
    return CUDF_ERROR_OUT_OF_MEMORY;
  } catch (rmm::bad_alloc const& e) {
    set_exception_detail(CUDF_ERROR_OUT_OF_MEMORY, e.what());
    return CUDF_ERROR_OUT_OF_MEMORY;
  } catch (cudf::fatal_cuda_error const& e) {
    set_exception_detail(CUDF_ERROR_FATAL_CUDA,
                         e.what(),
                         CUDF_ERROR_FAILURE_SOURCE_UNKNOWN,
                         0,
                         0,
                         0,
                         0,
                         0,
                         static_cast<std::int32_t>(e.error_code()));
    return CUDF_ERROR_FATAL_CUDA;
  } catch (cudf::cuda_error const& e) {
    set_exception_detail(CUDF_ERROR_CUDA,
                         e.what(),
                         CUDF_ERROR_FAILURE_SOURCE_UNKNOWN,
                         0,
                         0,
                         0,
                         0,
                         0,
                         static_cast<std::int32_t>(e.error_code()));
    return CUDF_ERROR_CUDA;
  } catch (cudf::data_type_error const& e) {
    set_exception_detail(CUDF_ERROR_TYPE, e.what());
    return CUDF_ERROR_TYPE;
  } catch (cudf::logic_error const& e) {
    set_exception_detail(CUDF_ERROR_INVALID_OPERATION, e.what());
    return CUDF_ERROR_INVALID_OPERATION;
  } catch (std::bad_alloc const& e) {
    set_exception_detail(CUDF_ERROR_OUT_OF_MEMORY, e.what());
    return CUDF_ERROR_OUT_OF_MEMORY;
  } catch (std::out_of_range const& e) {
    set_exception_detail(CUDF_ERROR_INDEX, e.what());
    return CUDF_ERROR_INDEX;
  } catch (std::invalid_argument const& e) {
    set_exception_detail(CUDF_ERROR_INVALID_ARGUMENT, e.what());
    return CUDF_ERROR_INVALID_ARGUMENT;
  } catch (std::overflow_error const& e) {
    set_exception_detail(CUDF_ERROR_SIZE_OVERFLOW, e.what());
    return CUDF_ERROR_SIZE_OVERFLOW;
  } catch (std::ios_base::failure const& e) {
    set_exception_detail(CUDF_ERROR_IO, e.what());
    return CUDF_ERROR_IO;
  } catch (std::runtime_error const& e) {
    set_exception_detail(runtime_error_status, e.what());
    return runtime_error_status;
  } catch (std::exception const& e) {
    set_exception_detail(CUDF_ERROR_UNKNOWN, e.what());
    return CUDF_ERROR_UNKNOWN;
  } catch (...) {
    set_exception_detail(CUDF_ERROR_UNKNOWN, "Unknown error");
    return CUDF_ERROR_UNKNOWN;
  }
}
}  // namespace

// Exception-catching helper used by all C shim functions.
// Returns the appropriate cudf_status_t for the caught exception.
cudf_status_t translate_exception() { return translate_exception_impl(CUDF_ERROR_UNKNOWN); }

cudf_status_t translate_io_exception() { return translate_exception_impl(CUDF_ERROR_IO); }

// Macro used throughout the C shim to wrap function bodies.
// Usage:
//   CUDF_RUST_TRY {
//     ... C++ code ...
//     return CUDF_SUCCESS;
//   } CUDF_RUST_CATCH
//
// Defined in the internal header below.
