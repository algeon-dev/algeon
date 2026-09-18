#include "cuml_rust/random_forest.h"
#include "random_forest_internal.hpp"
#include "shim_common.hpp"

#include <cuml/ensemble/randomforest.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;
raft::handle_t& cuml_handle_raft(cuml_handle_t handle);

struct cuml_rf_training_streams {
  std::shared_ptr<rmm::cuda_stream_pool> pool;
};

namespace {

ML::RF_params make_params(int32_t n_trees,
                          int32_t bootstrap,
                          float max_samples,
                          uint64_t seed,
                          int32_t max_depth,
                          int32_t max_leaves,
                          float max_features,
                          int32_t max_n_bins,
                          int32_t min_samples_leaf,
                          int32_t min_samples_split,
                          float min_impurity_decrease,
                          int32_t max_batch_size,
                          int32_t criterion)
{
  if ((bootstrap != 0 && bootstrap != 1) || n_trees <= 0 || !std::isfinite(max_samples) ||
      max_samples <= 0.0F || max_samples > 1.0F || !std::isfinite(max_features) ||
      max_features <= 0.0F || max_features > 1.0F || !std::isfinite(min_impurity_decrease) ||
      min_impurity_decrease < 0.0F || max_depth < 0 || max_leaves < -1 || max_n_bins < 2 ||
      min_samples_leaf <= 0 || min_samples_split < 2 || max_batch_size <= 0) {
    throw std::invalid_argument("random forest parameters are out of range");
  }
  if (criterion != ML::GINI && criterion != ML::ENTROPY && criterion != ML::MSE) {
    throw std::invalid_argument("random forest split criterion is unsupported");
  }
  return ML::set_rf_params(max_depth,
                           max_leaves,
                           max_features,
                           max_n_bins,
                           min_samples_leaf,
                           min_samples_split,
                           min_impurity_decrease,
                           bootstrap != 0,
                           n_trees,
                           max_samples,
                           seed,
                           static_cast<ML::CRITERION>(criterion),
                           1,
                           max_batch_size);
}

void validate_fit(cuml_handle_t handle,
                  cuml_rf_training_streams_t streams,
                  const void* input,
                  const void* labels,
                  int32_t rows,
                  int32_t columns,
                  int32_t row_major,
                  cuml_rf_forest_t* output)
{
  if (handle == nullptr || streams == nullptr || input == nullptr || labels == nullptr || output == nullptr) {
    throw std::invalid_argument("random forest fit received a null argument");
  }
  *output = nullptr;
  if (rows < 2 || columns < 1 || (row_major != 0 && row_major != 1)) {
    throw std::invalid_argument("random forest dimensions or matrix layout are invalid");
  }
  if (!streams->pool || streams->pool->get_pool_size() < 1) {
    throw std::invalid_argument("random forest training stream pool is empty");
  }
}

void fit_with_sync(raft::handle_t& fit_handle, cuml_rf_training_streams_t streams, auto&& fit)
{
  raft::resource::set_cuda_stream_pool(fit_handle, streams->pool);
  raft::resource::wait_stream_pool_on_stream(fit_handle);
  std::exception_ptr fit_error;
  try {
    fit();
  } catch (...) {
    fit_error = std::current_exception();
  }
  std::exception_ptr sync_error;
  try {
    raft::resource::sync_stream_pool(fit_handle);
  } catch (...) {
    sync_error = std::current_exception();
  }
  if (fit_error) std::rethrow_exception(fit_error);
  if (sync_error) std::rethrow_exception(sync_error);
}

}  // namespace

cuml_treelite_model cuml_rf_build_treelite(cuml_rf_forest const& forest)
{
  TreeliteModelHandle raw = nullptr;
  // Ownership of `raw` starts when cuML writes it, so a throw after that
  // write must free it here.
  try {
    if (forest.type == cuml_rf_forest::kind::classifier) {
      ML::build_treelite_forest(&raw, forest.model.classifier, forest.model.classifier->n_features);
    } else {
      ML::build_treelite_forest(&raw, forest.model.regressor, forest.model.regressor->n_features);
    }
  } catch (...) {
    if (raw != nullptr) TreeliteFreeModel(raw);
    throw;
  }
  return cuml_treelite_model{raw};
}

extern "C" cuml_status_t cuml_rf_training_streams_create(
  cuml_handle_t handle, int32_t count, cuml_rf_training_streams_t* out_streams)
{
  cuml_clear_error();
  if (out_streams == nullptr) return cuml_record_error(CUML_ERROR_NULL, "training stream output is null");
  *out_streams = nullptr;
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "training stream handle is null");
  CUML_RUST_TRY {
    if (count <= 0) throw std::invalid_argument("stream count must be positive");
    auto result = std::make_unique<cuml_rf_training_streams>();
    result->pool = std::make_shared<rmm::cuda_stream_pool>(static_cast<std::size_t>(count));
    *out_streams = result.release();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_rf_training_streams_raw(
  cuml_rf_training_streams_t streams, int32_t index, void** out_stream)
{
  cuml_clear_error();
  if (out_stream == nullptr) return cuml_record_error(CUML_ERROR_NULL, "training stream output is null");
  *out_stream = nullptr;
  if (streams == nullptr) return cuml_record_error(CUML_ERROR_NULL, "training streams are null");
  CUML_RUST_TRY {
    if (!streams->pool || index < 0 || static_cast<std::size_t>(index) >= streams->pool->get_pool_size()) {
      throw std::invalid_argument("training stream index is out of range");
    }
    *out_stream = streams->pool->get_stream(static_cast<std::size_t>(index)).value();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_rf_training_streams_destroy(cuml_rf_training_streams_t streams)
{
  cuml_clear_error();
  if (streams == nullptr) return CUML_SUCCESS;
  try {
    delete streams;
    return CUML_SUCCESS;
  } catch (...) {
    return cuml_translate_cleanup_exception_noexcept();
  }
}

extern "C" cuml_status_t cuml_rf_classifier_fit(
  cuml_handle_t handle, cuml_rf_training_streams_t streams, const float* input,
  int32_t rows, int32_t columns, const int32_t* labels, int32_t n_classes,
  int32_t n_trees, int32_t bootstrap, float max_samples, uint64_t seed,
  int32_t max_depth, int32_t max_leaves, float max_features, int32_t max_n_bins,
  int32_t min_samples_leaf, int32_t min_samples_split, float min_impurity_decrease,
  int32_t max_batch_size, int32_t split_criterion, int32_t input_row_major,
  cuml_rf_forest_t* out_forest)
{
  cuml_clear_error();
  if (out_forest == nullptr) return cuml_record_error(CUML_ERROR_NULL, "forest output is null");
  *out_forest = nullptr;
  CUML_RUST_TRY {
    validate_fit(handle, streams, input, labels, rows, columns, input_row_major, out_forest);
    if (n_classes < 2) throw std::invalid_argument("classifier requires at least two classes");
    auto params = make_params(n_trees, bootstrap, max_samples, seed, max_depth, max_leaves,
                              max_features, max_n_bins, min_samples_leaf, min_samples_split,
                              min_impurity_decrease, max_batch_size, split_criterion);
    auto forest = std::make_unique<ML::RandomForestClassifierF>();
    auto& base = cuml_handle_raft(handle);
    raft::handle_t fit_handle{base};
    fit_with_sync(fit_handle, streams, [&] {
      ML::fit(fit_handle, forest.get(), const_cast<float*>(input), rows, columns,
              const_cast<int32_t*>(labels), n_classes, params,
              rapids_logger::level_enum::warn, nullptr, nullptr, input_row_major != 0);
    });
    auto result = std::make_unique<cuml_rf_forest>();
    result->type = cuml_rf_forest::kind::classifier;
    result->model.classifier = forest.release();
    *out_forest = result.release();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_rf_regressor_fit(
  cuml_handle_t handle, cuml_rf_training_streams_t streams, const float* input,
  int32_t rows, int32_t columns, const float* labels, int32_t n_trees,
  int32_t bootstrap, float max_samples, uint64_t seed, int32_t max_depth,
  int32_t max_leaves, float max_features, int32_t max_n_bins,
  int32_t min_samples_leaf, int32_t min_samples_split, float min_impurity_decrease,
  int32_t max_batch_size, int32_t split_criterion, int32_t input_row_major,
  cuml_rf_forest_t* out_forest)
{
  cuml_clear_error();
  if (out_forest == nullptr) return cuml_record_error(CUML_ERROR_NULL, "forest output is null");
  *out_forest = nullptr;
  CUML_RUST_TRY {
    validate_fit(handle, streams, input, labels, rows, columns, input_row_major, out_forest);
    if (split_criterion != ML::MSE) throw std::invalid_argument("regression requires MSE criterion");
    auto params = make_params(n_trees, bootstrap, max_samples, seed, max_depth, max_leaves,
                              max_features, max_n_bins, min_samples_leaf, min_samples_split,
                              min_impurity_decrease, max_batch_size, split_criterion);
    auto forest = std::make_unique<ML::RandomForestRegressorF>();
    auto& base = cuml_handle_raft(handle);
    raft::handle_t fit_handle{base};
    fit_with_sync(fit_handle, streams, [&] {
      ML::fit(fit_handle, forest.get(), const_cast<float*>(input), rows, columns,
              const_cast<float*>(labels), params, rapids_logger::level_enum::warn,
              nullptr, nullptr, input_row_major != 0);
    });
    auto result = std::make_unique<cuml_rf_forest>();
    result->type = cuml_rf_forest::kind::regressor;
    result->model.regressor = forest.release();
    *out_forest = result.release();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_rf_forest_import(cuml_handle_t handle, cuml_rf_forest_t forest,
                                               cuml_forest_model_t* out_model, cuml_forest_metadata_t* out_metadata)
{
  cuml_clear_error();
  if (out_model == nullptr || out_metadata == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "random forest import output is null");
  }
  *out_model = nullptr;
  if (handle == nullptr || forest == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "random forest import received a null argument");
  }
  CUML_RUST_TRY {
    auto const model = cuml_rf_build_treelite(*forest);
    auto const output = forest->type == cuml_rf_forest::kind::classifier ? CUML_FOREST_OUTPUT_CLASS_INDEX
                                                                          : CUML_FOREST_OUTPUT_SCORES;
    *out_model = cuml_forest_import_treelite(handle, model.get(), output, out_metadata);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_rf_forest_destroy(cuml_rf_forest_t forest)
{
  cuml_clear_error();
  if (forest == nullptr) return CUML_SUCCESS;
  try {
    if (forest->type == cuml_rf_forest::kind::classifier) {
      ML::delete_rf_metadata(forest->model.classifier);
    } else {
      ML::delete_rf_metadata(forest->model.regressor);
    }
    delete forest;
    return CUML_SUCCESS;
  } catch (...) {
    return cuml_translate_cleanup_exception_noexcept();
  }
}
