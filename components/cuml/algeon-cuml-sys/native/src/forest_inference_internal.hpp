#ifndef CUML_RUST_FOREST_INFERENCE_INTERNAL_HPP
#define CUML_RUST_FOREST_INFERENCE_INTERNAL_HPP

#include "cuml_rust/forest_inference.h"

#include <treelite/c_api.h>

#include <memory>

struct cuml_treelite_model_deleter {
  void operator()(void* model) const noexcept { TreeliteFreeModel(model); }
};
using cuml_treelite_model = std::unique_ptr<void, cuml_treelite_model_deleter>;

// Imports a parsed Treelite model at its native precision onto the handle's
// device and stream. nvForest copies the node data, so the caller keeps
// ownership of `model` and may free it after the call. Throws on failure.
cuml_forest_model_t cuml_forest_import_treelite(cuml_handle_t handle, TreeliteModelHandle model,
                                                cuml_forest_output_t output, cuml_forest_metadata_t* out_metadata);

#endif
