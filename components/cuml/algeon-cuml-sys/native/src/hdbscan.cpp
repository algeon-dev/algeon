#include "cuml_rust/hdbscan.h"
#include "cuml_rust/error.h"
#include "shim_common.hpp"

#include <cuml/cluster/hdbscan.hpp>
#include <raft/core/handle.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace raft {
class handle_t;
}

raft::handle_t& cuml_handle_raft(cuml_handle_t handle);
cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;

namespace {

std::size_t checked_elements(std::size_t rows, std::size_t columns)
{
  if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::overflow_error("HDBSCAN matrix element count overflows size_t");
  }
  return rows * columns;
}

}  // namespace

extern "C" cuml_status_t cuml_hdbscan_fit(cuml_handle_t handle,
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
                                          float* mst_weights)
{
  cuml_clear_error();
  if (handle == nullptr || input == nullptr || n_clusters == nullptr || labels == nullptr || probabilities == nullptr ||
      core_distances == nullptr || single_linkage_children == nullptr || single_linkage_sizes == nullptr ||
      single_linkage_distances == nullptr || mst_sources == nullptr || mst_destinations == nullptr ||
      mst_weights == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "HDBSCAN received a null argument");
  }
  CUML_RUST_TRY {
    if (rows < 2 || columns < 1 || min_samples < 1 || min_samples >= rows || min_cluster_size < 2 ||
        min_cluster_size > rows || max_cluster_size < 0 || max_cluster_size > rows ||
        !std::isfinite(cluster_selection_epsilon) || cluster_selection_epsilon < 0.0F ||
        (allow_single_cluster != 0 && allow_single_cluster != 1) ||
        (cluster_selection_method != 0 && cluster_selection_method != 1) || !std::isfinite(alpha) || alpha <= 0.0F) {
      throw std::invalid_argument("invalid HDBSCAN dimensions or parameters");
    }
    (void)checked_elements(static_cast<std::size_t>(rows), static_cast<std::size_t>(columns));
    auto& raft_handle = cuml_handle_raft(handle);
    ML::HDBSCAN::Common::HDBSCANParams params;
    params.min_samples = min_samples;
    params.min_cluster_size = min_cluster_size;
    params.max_cluster_size = max_cluster_size;
    params.cluster_selection_epsilon = cluster_selection_epsilon;
    params.allow_single_cluster = allow_single_cluster != 0;
    params.cluster_selection_method = static_cast<ML::HDBSCAN::Common::CLUSTER_SELECTION_METHOD>(cluster_selection_method);
    params.alpha = alpha;
    params.build_algo = ML::HDBSCAN::Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN;

    using Output = ML::HDBSCAN::Common::hdbscan_output<int64_t, float>;
    Output output(raft_handle,
                  rows,
                  labels,
                  probabilities,
                  single_linkage_children,
                  single_linkage_sizes,
                  single_linkage_distances,
                  mst_sources,
                  mst_destinations,
                  mst_weights);
    ML::hdbscan(raft_handle,
                input,
                static_cast<std::size_t>(rows),
                static_cast<std::size_t>(columns),
                ML::distance::DistanceType::L2SqrtExpanded,
                params,
                output,
                core_distances);
    raft_handle.sync_stream();
    *n_clusters = output.get_n_clusters();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
