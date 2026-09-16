#include "cuml_rust/test_support.h"
#include "shim_common.hpp"

#include <cudf_rust/error_boundary.hpp>
#include <raft/core/error.hpp>
#include <rmm/error.hpp>
#include <treelite/c_api.h>

#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

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

namespace {
void check_treelite(int status, char const* operation)
{
  if (status != 0) {
    auto const* detail = TreeliteGetLastError();
    throw std::invalid_argument(std::string{operation} + ": " + (detail == nullptr ? "unknown error" : detail));
  }
}
}  // namespace

extern "C" cuml_status_t cuml_test_forest_gtil_predict(const uint8_t* checkpoint, size_t len, const double* input,
                                                       int32_t rows, int32_t columns, double* output,
                                                       size_t output_capacity, size_t* out_written)
{
  cuml_clear_error();
  if (checkpoint == nullptr || input == nullptr || output == nullptr || out_written == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "GTIL prediction received a null argument");
  }
  *out_written = 0;
  CUML_RUST_TRY {
    TreeliteModelHandle raw = nullptr;
    check_treelite(TreeliteDeserializeModelFromBytes(reinterpret_cast<char const*>(checkpoint), len, &raw),
                   "Treelite checkpoint parse");
    auto model = std::unique_ptr<void, int (*)(TreeliteModelHandle)>{raw, TreeliteFreeModel};
    int features = 0;
    check_treelite(TreeliteQueryNumFeature(model.get(), &features), "Treelite feature query");
    if (rows < 1 || columns != features) throw std::invalid_argument("GTIL rows or feature count are invalid");
    TreeliteGTILConfigHandle raw_config = nullptr;
    check_treelite(TreeliteGTILParseConfig(R"({"predict_type":"default","nthread":1})", &raw_config),
                   "GTIL config");
    auto config = std::unique_ptr<void, int (*)(TreeliteGTILConfigHandle)>{raw_config, TreeliteGTILDeleteConfig};
    uint64_t const* shape = nullptr;
    uint64_t ndim = 0;
    check_treelite(TreeliteGTILGetOutputShape(model.get(), static_cast<uint64_t>(rows), config.get(), &shape, &ndim),
                   "GTIL output shape");
    auto elements = std::size_t{1};
    for (uint64_t i = 0; i < ndim; ++i) elements *= static_cast<std::size_t>(shape[i]);
    if (elements > output_capacity) throw std::invalid_argument("GTIL output capacity is too small");
    check_treelite(TreeliteGTILPredict(model.get(), input, "float64", static_cast<uint64_t>(rows), output,
                                       config.get()),
                   "GTIL predict");
    *out_written = elements;
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
