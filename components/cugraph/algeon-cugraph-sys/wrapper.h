#include <cugraph_c/export.h>
#include <cugraph_c/error.h>
#include <cugraph_c/types.h>
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/random.h>

// Aggregated algorithm APIs.
#include <cugraph_c/algorithms.h>

// Additional APIs not included by algorithms.h.
#include <cugraph_c/graph_functions.h>
#include <cugraph_c/graph_generators.h>

// layout_algorithms.h cannot be included alongside algorithms.h because both
// tree_algorithms.h and layout_algorithms.h define cugraph_layout_result_t.
// Instead, this header forward-declares the functions it needs from layout_algorithms.h.
// The struct cugraph_layout_result_t is already defined via tree_algorithms.h
// (pulled in by algorithms.h).

#ifdef __cplusplus
extern "C" {
#endif

CUGRAPH_EXPORT cugraph_error_code_t cugraph_force_atlas2(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  int max_iter,
  cugraph_type_erased_device_array_view_t* start_vertices,
  cugraph_type_erased_device_array_view_t* x_start,
  cugraph_type_erased_device_array_view_t* y_start,
  bool_t outbound_attraction_distribution,
  bool_t lin_log_mode,
  bool_t prevent_overlapping,
  cugraph_type_erased_device_array_view_t* vertex_radius_vertices,
  cugraph_type_erased_device_array_view_t* vertex_radius_values,
  double overlap_scaling_ratio,
  double edge_weight_influence,
  double jitter_tolerance,
  bool_t barnes_hut_optimize,
  double barnes_hut_theta,
  double scaling_ratio,
  bool_t strong_gravity_mode,
  double gravity,
  cugraph_type_erased_device_array_view_t* vertex_mobility_vertices,
  cugraph_type_erased_device_array_view_t* vertex_mobility_values,
  cugraph_type_erased_device_array_view_t* vertex_mass_vertices,
  cugraph_type_erased_device_array_view_t* vertex_mass_values,
  bool_t verbose,
  bool_t do_expensive_check,
  cugraph_layout_result_t** result,
  cugraph_error_t** error);

CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_vertices(
  cugraph_layout_result_t* result);

CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_x_axis(
  cugraph_layout_result_t* result);

CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_y_axis(
  cugraph_layout_result_t* result);

CUGRAPH_EXPORT void cugraph_layout_result_free(cugraph_layout_result_t* result);

#ifdef __cplusplus
}
#endif
