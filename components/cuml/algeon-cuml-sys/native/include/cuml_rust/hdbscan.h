#ifndef CUML_RUST_HDBSCAN_H
#define CUML_RUST_HDBSCAN_H

#include "cuml_rust/handle.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

cuml_status_t cuml_hdbscan_fit(cuml_handle_t handle,
                               const float* input,
                               int32_t rows,
                               int32_t columns,
                               int32_t min_samples,
                               int32_t min_cluster_size,
                               int32_t max_cluster_size,
                               float cluster_selection_epsilon,
                               int32_t allow_single_cluster,
                               int32_t cluster_selection_method,
                               float alpha,
                               int32_t* n_clusters,
                               int64_t* labels,
                               float* probabilities,
                               float* core_distances,
                               int64_t* single_linkage_children,
                               int64_t* single_linkage_sizes,
                               float* single_linkage_distances,
                               int64_t* mst_sources,
                               int64_t* mst_destinations,
                               float* mst_weights);

#ifdef __cplusplus
}
#endif
#endif
