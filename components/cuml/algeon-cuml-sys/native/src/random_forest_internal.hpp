#ifndef CUML_RUST_RANDOM_FOREST_INTERNAL_HPP
#define CUML_RUST_RANDOM_FOREST_INTERNAL_HPP

#include "cuml_rust/random_forest.h"
#include "forest_inference_internal.hpp"

#include <cuml/ensemble/randomforest.hpp>

struct cuml_rf_forest {
  enum class kind { classifier, regressor } type;
  union {
    ML::RandomForestClassifierF* classifier;
    ML::RandomForestRegressorF* regressor;
  } model;
};

// Builds a host Treelite model with single-precision thresholds and leaves.
// Classifier leaves hold per-class probabilities averaged across trees.
cuml_treelite_model cuml_rf_build_treelite(cuml_rf_forest const& forest);

#endif
