#include "join_common.hpp"

#include <rmm/mr/cuda_memory_resource.hpp>

using namespace cudf_rust::join;

extern "C" {
cudf_status_t cudf_filter_join_indices_ex(cudf_table_view_handle left,
                                          cudf_table_view_handle right,
                                          cudf_column_view_handle left_indices,
                                          cudf_column_view_handle right_indices,
                                          cudf_ast_expression_handle predicate,
                                          cudf_join_kind_t join_kind,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  return filter_join_indices_impl(left,
                                  right,
                                  left_indices,
                                  right_indices,
                                  predicate,
                                  join_kind,
                                  out_left_indices,
                                  out_right_indices,
                                  options);
}

cudf_status_t cudf_filtered_join_create_ex(cudf_table_view_handle build,
                                           cudf_null_equality_t null_equality,
                                           double load_factor,
                                           cudf_filtered_join_handle* out,
                                           cudf_exec_options_t const* options)
{
  return filtered_join_create_impl(build, null_equality, load_factor, out, options);
}

cudf_status_t cudf_filtered_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
                                                               cudf_null_equality_t null_equality,
                                                               double load_factor,
                                                               size_t* out,
                                                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    (void)null_equality;

    auto* build_view = reinterpret_cast<cudf::table_view*>(build);
    *out             = cudf::filtered_join::pre_build_reservation_size(
      *build_view, load_factor, exec->stream, *exec->mr);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filtered_join_fixed_width_metadata_pre_build_reservation_size(
  cudf_type_id_t const* build_key_types,
  size_t build_key_count,
  size_t build_rows,
  double load_factor,
  size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(build_key_types, "build_key_types is null");
  if (build_key_count == 0) {
    set_error("build_key_count must be greater than zero");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (build_rows > static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
    set_error("build_rows exceeds cudf::size_type maximum");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    alignas(std::max_align_t) static std::byte dummy_storage{};
    std::vector<cudf::column_view> columns;
    columns.reserve(build_key_count);
    for (size_t index = 0; index < build_key_count; ++index) {
      auto const dtype = cudf::data_type{to_cudf_type_id(build_key_types[index])};
      if (!cudf::is_fixed_width(dtype)) {
        set_error("filtered join metadata preflight only supports fixed-width key types");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      columns.emplace_back(dtype,
                           static_cast<cudf::size_type>(build_rows),
                           static_cast<void const*>(&dummy_storage),
                           nullptr,
                           0,
                           0,
                           std::vector<cudf::column_view>{});
    }
    rmm::mr::cuda_memory_resource memory_resource;
    *out = cudf::filtered_join::pre_build_reservation_size(
      cudf::table_view{columns},
      load_factor,
      cudf::get_default_stream(),
      memory_resource);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filtered_join_destroy(cudf_filtered_join_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = reinterpret_cast<filtered_join_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_filtered_join_device_allocated_size(cudf_filtered_join_handle handle,
                                                       size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "filtered join is null");
  CUDF_RUST_TRY
  {
    auto* owner = reinterpret_cast<filtered_join_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->device_allocated_size_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filtered_join_begin_left_semi_probe_ex(cudf_filtered_join_handle join,
                                                          cudf_table_view_handle probe,
                                                          cudf_filtered_join_probe_handle* out,
                                                          cudf_exec_options_t const* options)
{
  return filtered_join_begin_probe_impl(join, probe, out, options, filtered_join_probe_kind::semi);
}

cudf_status_t cudf_filtered_join_begin_left_anti_probe_ex(cudf_filtered_join_handle join,
                                                          cudf_table_view_handle probe,
                                                          cudf_filtered_join_probe_handle* out,
                                                          cudf_exec_options_t const* options)
{
  return filtered_join_begin_probe_impl(join, probe, out, options, filtered_join_probe_kind::anti);
}

cudf_status_t cudf_filtered_join_probe_destroy(cudf_filtered_join_probe_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = reinterpret_cast<filtered_join_probe_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_filtered_join_probe_output_size(cudf_filtered_join_probe_handle handle,
                                                   size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "filtered join probe is null");
  CUDF_RUST_TRY
  {
    auto* owner = reinterpret_cast<filtered_join_probe_owner*>(handle);
    *out        = static_cast<size_t>(owner->probe->output_size());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filtered_join_probe_device_allocated_size(cudf_filtered_join_probe_handle handle,
                                                             size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "filtered join probe is null");
  CUDF_RUST_TRY
  {
    auto* owner = reinterpret_cast<filtered_join_probe_owner*>(handle);
    *out        = owner->device_allocated_size_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_filtered_join_probe_materialize_indices_ex(
  cudf_filtered_join_probe_handle handle,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  return filtered_join_probe_materialize_indices_impl(handle, out_indices, options);
}

cudf_status_t cudf_filtered_join_left_semi_join_size_ex(cudf_filtered_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t* out,
                                                        cudf_exec_options_t const* options)
{
  return filtered_join_size_impl(join, probe, out, options, filtered_join_probe_kind::semi);
}

cudf_status_t cudf_filtered_join_left_anti_join_size_ex(cudf_filtered_join_handle join,
                                                        cudf_table_view_handle probe,
                                                        size_t* out,
                                                        cudf_exec_options_t const* options)
{
  return filtered_join_size_impl(join, probe, out, options, filtered_join_probe_kind::anti);
}

cudf_status_t cudf_filtered_join_left_semi_join_with_output_size_ex(
  cudf_filtered_join_handle join,
  cudf_table_view_handle probe,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  return filtered_join_indices_with_output_size_impl(
    join, probe, output_size, out_indices, options, filtered_join_probe_kind::semi);
}

cudf_status_t cudf_filtered_join_left_anti_join_with_output_size_ex(
  cudf_filtered_join_handle join,
  cudf_table_view_handle probe,
  size_t output_size,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  return filtered_join_indices_with_output_size_impl(
    join, probe, output_size, out_indices, options, filtered_join_probe_kind::anti);
}

}  // extern "C"
