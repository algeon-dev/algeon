#include "cuml_rust/test_support.h"
#include "shim_common.hpp"

#include <cudf_rust/error_boundary.hpp>
#include <raft/core/error.hpp>
#include <rmm/error.hpp>

#include <new>
#include <stdexcept>

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;

extern "C" cuml_status_t cuml_test_raise_error(cuml_status_t status, char const* message)
{
  cuml_clear_error();
  if (message == nullptr) return cuml_record_error(CUML_ERROR_NULL, "test error message is null");
  CUML_RUST_TRY {
    if (status == CUML_ERROR_SIZE_OVERFLOW) throw std::overflow_error(message);
    if (status == CUML_ERROR_INVALID_ARGUMENT) throw std::invalid_argument(message);
    throw std::runtime_error(message);
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_test_raise_exception(cuml_test_exception_kind_t kind)
{
  CUML_RUST_TRY {
    switch (kind) {
      case CUML_TEST_ATTRIBUTED_OOM: {
        cudf::rust::memory::memory_failure_domain_identity identity{};
        identity.kind = CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN;
        identity.primary_id = 11;
        identity.secondary_id = 17;
        identity.generation = 23;
        throw cudf::rust::memory::attributed_out_of_memory{
          "injected attributed OOM", CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
          identity, nullptr, 4096, 256, 2};
      }
      case CUML_TEST_RMM_OOM: throw rmm::out_of_memory{"injected RMM OOM"};
      case CUML_TEST_RMM_BAD_ALLOC: throw rmm::bad_alloc{"injected RMM bad_alloc"};
      case CUML_TEST_CUDA_ERROR: throw rmm::cuda_error{"injected CUDA error"};
      case CUML_TEST_RAFT_EXCEPTION: RAFT_FAIL("injected raft exception");
      case CUML_TEST_STD_BAD_ALLOC: throw std::bad_alloc{};
      case CUML_TEST_OVERFLOW: throw std::overflow_error{"injected overflow"};
      case CUML_TEST_INVALID_ARGUMENT: throw std::invalid_argument{"injected invalid argument"};
      case CUML_TEST_UNKNOWN: throw 37;
    }
    throw std::invalid_argument{"unknown test exception kind"};
  } CUML_RUST_CATCH
}
