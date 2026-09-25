#include "cuml_rust/forest_inference.h"
#include "forest_inference_internal.hpp"
#include "shim_common.hpp"

#include <nvforest/detail/exceptions.hpp>
#include <nvforest/exceptions.hpp>
#include <nvforest/forest_model.hpp>
#include <nvforest/handle.hpp>
#include <nvforest/treelite_importer.hpp>
#include <nvforest/version_config.hpp>
#include <raft/core/handle.hpp>
#include <rmm/error.hpp>
#include <treelite/c_api.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;
raft::handle_t& cuml_handle_raft(cuml_handle_t handle);

struct cuml_forest_model {
  nvforest::forest_model forest;
  cuml_forest_metadata_t metadata;
};

namespace {

// nvForest kernels index rows, features, and outputs with 32-bit unsigned
// offsets (input + row * col_count), so every product must fit index_type.
constexpr auto k_max_elements = std::uint64_t{std::numeric_limits<nvforest::index_type>::max()};
// GPU tree alignment recommended by nvForest's importer for 128-byte cache lines.
constexpr auto k_gpu_align_bytes = nvforest::index_type{128};

void check_treelite(int status, char const* operation)
{
  if (status != 0) {
    auto const* detail = TreeliteGetLastError();
    throw std::invalid_argument(std::string{operation} + ": " + (detail == nullptr ? "unknown error" : detail));
  }
}

cuml_treelite_model load_source(std::uint8_t const* bytes, std::size_t len, cuml_forest_format_t format)
{
  auto const* chars = reinterpret_cast<char const*>(bytes);
  TreeliteModelHandle raw = nullptr;
  switch (format) {
    case CUML_FOREST_FORMAT_XGBOOST_JSON:
      check_treelite(TreeliteLoadXGBoostModelFromJSONString(chars, len, "{}", &raw), "XGBoost JSON parse");
      break;
    case CUML_FOREST_FORMAT_XGBOOST_UBJSON:
      check_treelite(TreeliteLoadXGBoostModelFromUBJSONString(chars, len, "{}", &raw), "XGBoost UBJSON parse");
      break;
    case CUML_FOREST_FORMAT_XGBOOST_LEGACY:
      check_treelite(TreeliteLoadXGBoostModelLegacyBinaryFromMemoryBuffer(bytes, len, "{}", &raw),
                     "XGBoost legacy binary parse");
      break;
    case CUML_FOREST_FORMAT_LIGHTGBM: {
      // Treelite reads the LightGBM text model as a NUL-terminated string.
      if (std::memchr(bytes, '\0', len) != nullptr) {
        throw std::invalid_argument("LightGBM text model contains a NUL byte");
      }
      auto const text = std::string{chars, len};
      check_treelite(TreeliteLoadLightGBMModelFromString(text.c_str(), "{}", &raw), "LightGBM text parse");
      break;
    }
    case CUML_FOREST_FORMAT_TREELITE_CHECKPOINT:
      check_treelite(TreeliteDeserializeModelFromBytes(chars, len, &raw), "Treelite checkpoint parse");
      break;
    default: throw std::invalid_argument("forest model format is unsupported");
  }
  return cuml_treelite_model{raw};
}

// nvForest exceptions derive from std::exception only; translate them into
// the shim's taxonomy before the shared boundary classifies them.
template <typename F>
auto nvforest_call(F&& call)
{
  try {
    return call();
  } catch (nvforest::model_import_error const& error) {
    throw std::invalid_argument(error.what());
  } catch (nvforest::type_error const& error) {
    throw std::invalid_argument(error.what());
  } catch (nvforest::detail::bad_cuda_call const& error) {
    throw rmm::cuda_error(error.what());
  }
}

std::int32_t checked_i32(std::uint64_t value, char const* what)
{
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument(std::string{"forest "} + what + " exceeds int32");
  }
  return static_cast<std::int32_t>(value);
}

cuml_forest_metadata_t metadata_of(nvforest::forest_model& forest)
{
  return cuml_forest_metadata_t{
    checked_i32(forest.num_features(), "feature count"),
    checked_i32(forest.num_trees(), "tree count"),
    checked_i32(forest.num_outputs(), "output count"),
    forest.is_double_precision() ? CUML_FOREST_PRECISION_FLOAT64 : CUML_FOREST_PRECISION_FLOAT32,
  };
}

}  // namespace

cuml_forest_model_t cuml_forest_import_treelite(cuml_handle_t handle, TreeliteModelHandle model,
                                                cuml_forest_output_t output, cuml_forest_metadata_t* out_metadata)
{
  if (output != CUML_FOREST_OUTPUT_SCORES && output != CUML_FOREST_OUTPUT_CLASS_INDEX) {
    throw std::invalid_argument("forest output request is unsupported");
  }
  int device = 0;
  auto const device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) throw rmm::cuda_error(cudaGetErrorString(device_status));
  auto stream = cuml_handle_raft(handle).get_stream().value();
  auto result = std::make_unique<cuml_forest_model>();
  // Native precision keeps double thresholds exact; nvForest's device owning
  // buffers allocate through the current RMM resource on this stream.
  result->forest = nvforest_call([&] {
    return nvforest::import_from_treelite_handle(model, nvforest::tree_layout::breadth_first, k_gpu_align_bytes,
                                                 std::nullopt, nvforest::device_type::gpu, device, stream);
  });
  result->metadata = metadata_of(result->forest);
  if (output == CUML_FOREST_OUTPUT_CLASS_INDEX) {
    if (result->metadata.num_outputs < 2) {
      throw std::invalid_argument("class index output requires a model with at least two outputs");
    }
    result->forest.set_row_postprocessing(nvforest::row_op::max_index);
  }
  *out_metadata = result->metadata;
  return result.release();
}

extern "C" cuml_status_t cuml_forest_inspect(const uint8_t* bytes, size_t len, cuml_forest_format_t format,
                                             cuml_forest_metadata_t* out_metadata,
                                             uint8_t** out_checkpoint, size_t* out_checkpoint_len)
{
  cuml_clear_error();
  if (out_metadata == nullptr || out_checkpoint == nullptr || out_checkpoint_len == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "forest inspect output is null");
  }
  *out_checkpoint = nullptr;
  *out_checkpoint_len = 0;
  if (bytes == nullptr) return cuml_record_error(CUML_ERROR_NULL, "forest model bytes are null");
  CUML_RUST_TRY {
    if (len == 0) throw std::invalid_argument("forest model bytes are empty");
    auto model = load_source(bytes, len, format);
    // A host import is the capability proof: it rejects multi-target models,
    // invalid class assignments, and unknown postprocessors before any device work.
    auto forest = nvforest_call([&] {
      return nvforest::import_from_treelite_handle(model.get(), nvforest::tree_layout::breadth_first);
    });
    auto const metadata = metadata_of(forest);
    // Treelite returns a buffer it owns until the next call; copy it out.
    char const* serialized = nullptr;
    std::size_t serialized_len = 0;
    check_treelite(TreeliteSerializeModelToBytes(model.get(), &serialized, &serialized_len),
                   "Treelite checkpoint serialize");
    auto* copy = static_cast<std::uint8_t*>(std::malloc(serialized_len));
    if (copy == nullptr) throw std::bad_alloc{};
    std::memcpy(copy, serialized, serialized_len);
    *out_metadata = metadata;
    *out_checkpoint = copy;
    *out_checkpoint_len = serialized_len;
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_forest_bytes_free(uint8_t* bytes)
{
  cuml_clear_error();
  std::free(bytes);
  return CUML_SUCCESS;
}

extern "C" cuml_status_t cuml_forest_import(cuml_handle_t handle, const uint8_t* checkpoint, size_t len,
                                            cuml_forest_output_t output, cuml_forest_model_t* out_model,
                                            cuml_forest_metadata_t* out_metadata)
{
  cuml_clear_error();
  if (out_model == nullptr || out_metadata == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "forest import output is null");
  }
  *out_model = nullptr;
  if (handle == nullptr || checkpoint == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "forest import received a null argument");
  }
  CUML_RUST_TRY {
    auto model = load_source(checkpoint, len, CUML_FOREST_FORMAT_TREELITE_CHECKPOINT);
    *out_model = cuml_forest_import_treelite(handle, model.get(), output, out_metadata);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_forest_predict(cuml_handle_t handle, cuml_forest_model_t model,
                                             const void* input, int32_t rows, int32_t columns, void* output)
{
  cuml_clear_error();
  if (handle == nullptr || model == nullptr || input == nullptr || output == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "forest prediction received a null argument");
  }
  CUML_RUST_TRY {
    auto const& metadata = model->metadata;
    if (rows < 1 || columns != metadata.num_features) {
      throw std::invalid_argument("forest prediction rows or feature count are invalid");
    }
    auto const row_count = static_cast<std::uint64_t>(rows);
    if (row_count * static_cast<std::uint64_t>(columns) > k_max_elements ||
        row_count * static_cast<std::uint64_t>(metadata.num_outputs) > k_max_elements) {
      throw std::invalid_argument("forest prediction elements exceed nvForest 32-bit indexing");
    }
    auto const forest_handle = nvforest::handle_t{cuml_handle_raft(handle)};
    auto const gpu = nvforest::device_type::gpu;
    nvforest_call([&] {
      if (metadata.precision == CUML_FOREST_PRECISION_FLOAT64) {
        model->forest.predict(forest_handle, static_cast<double*>(output),
                              const_cast<double*>(static_cast<double const*>(input)), row_count, gpu, gpu);
      } else {
        model->forest.predict(forest_handle, static_cast<float*>(output),
                              const_cast<float*>(static_cast<float const*>(input)), row_count, gpu, gpu);
      }
    });
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_forest_close(cuml_forest_model_t model)
{
  cuml_clear_error();
  if (model == nullptr) return CUML_SUCCESS;
  try {
    delete model;
    return CUML_SUCCESS;
  } catch (...) {
    return cuml_translate_cleanup_exception_noexcept();
  }
}

extern "C" cuml_status_t cuml_forest_inference_version_get(int32_t* major, int32_t* minor, int32_t* patch)
{
  cuml_clear_error();
  if (major == nullptr || minor == nullptr || patch == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "forest inference version output is null");
  }
  *major = NVForest_VERSION_MAJOR;
  *minor = NVForest_VERSION_MINOR;
  *patch = NVForest_VERSION_PATCH;
  return CUML_SUCCESS;
}
