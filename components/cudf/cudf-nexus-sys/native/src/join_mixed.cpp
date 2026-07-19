#include "join_common.hpp"

using namespace cudf_rust::join;

extern "C" {
cudf_status_t cudf_mixed_inner_join_indices_ex(cudf_table_view_handle left_equality,
                                               cudf_table_view_handle right_equality,
                                               cudf_table_view_handle left_conditional,
                                               cudf_table_view_handle right_conditional,
                                               cudf_ast_expression_handle predicate,
                                               cudf_null_equality_t null_equality,
                                               cudf_column_handle* out_left_indices,
                                               cudf_column_handle* out_right_indices,
                                               cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::inner);
}

cudf_status_t cudf_mixed_left_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::left);
}

cudf_status_t cudf_mixed_full_join_indices_ex(cudf_table_view_handle left_equality,
                                              cudf_table_view_handle right_equality,
                                              cudf_table_view_handle left_conditional,
                                              cudf_table_view_handle right_conditional,
                                              cudf_ast_expression_handle predicate,
                                              cudf_null_equality_t null_equality,
                                              cudf_column_handle* out_left_indices,
                                              cudf_column_handle* out_right_indices,
                                              cudf_exec_options_t const* options)
{
  return mixed_pair_join_indices_impl(left_equality,
                                      right_equality,
                                      left_conditional,
                                      right_conditional,
                                      predicate,
                                      null_equality,
                                      out_left_indices,
                                      out_right_indices,
                                      options,
                                      pair_join_kind::full);
}

cudf_status_t cudf_mixed_left_semi_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options)
{
  return mixed_single_side_join_indices_impl(left_equality,
                                             right_equality,
                                             left_conditional,
                                             right_conditional,
                                             predicate,
                                             null_equality,
                                             out_indices,
                                             options,
                                             single_side_join_kind::semi);
}

cudf_status_t cudf_mixed_left_anti_join_indices_ex(cudf_table_view_handle left_equality,
                                                   cudf_table_view_handle right_equality,
                                                   cudf_table_view_handle left_conditional,
                                                   cudf_table_view_handle right_conditional,
                                                   cudf_ast_expression_handle predicate,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_column_handle* out_indices,
                                                   cudf_exec_options_t const* options)
{
  return mixed_single_side_join_indices_impl(left_equality,
                                             right_equality,
                                             left_conditional,
                                             right_conditional,
                                             predicate,
                                             null_equality,
                                             out_indices,
                                             options,
                                             single_side_join_kind::anti);
}

cudf_status_t cudf_mixed_inner_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                       cudf_table_view_handle right_equality,
                                                       cudf_table_view_handle left_conditional,
                                                       cudf_table_view_handle right_conditional,
                                                       cudf_ast_expression_handle predicate,
                                                       cudf_null_equality_t null_equality,
                                                       cudf_mixed_join_size_data_handle* out,
                                                       cudf_exec_options_t const* options)
{
  return mixed_join_size_data_impl(left_equality,
                                   right_equality,
                                   left_conditional,
                                   right_conditional,
                                   predicate,
                                   null_equality,
                                   out,
                                   options,
                                   mixed_join_size_kind::inner);
}

cudf_status_t cudf_mixed_left_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                      cudf_table_view_handle right_equality,
                                                      cudf_table_view_handle left_conditional,
                                                      cudf_table_view_handle right_conditional,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_null_equality_t null_equality,
                                                      cudf_mixed_join_size_data_handle* out,
                                                      cudf_exec_options_t const* options)
{
  return mixed_join_size_data_impl(left_equality,
                                   right_equality,
                                   left_conditional,
                                   right_conditional,
                                   predicate,
                                   null_equality,
                                   out,
                                   options,
                                   mixed_join_size_kind::left);
}

cudf_status_t cudf_mixed_full_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                      cudf_table_view_handle right_equality,
                                                      cudf_table_view_handle left_conditional,
                                                      cudf_table_view_handle right_conditional,
                                                      cudf_ast_expression_handle predicate,
                                                      cudf_null_equality_t null_equality,
                                                      cudf_mixed_join_size_data_handle* out,
                                                      cudf_exec_options_t const* options)
{
  return mixed_join_size_data_impl(left_equality,
                                   right_equality,
                                   left_conditional,
                                   right_conditional,
                                   predicate,
                                   null_equality,
                                   out,
                                   options,
                                   mixed_join_size_kind::full);
}

cudf_status_t cudf_mixed_left_semi_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                           cudf_table_view_handle right_equality,
                                                           cudf_table_view_handle left_conditional,
                                                           cudf_table_view_handle right_conditional,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_null_equality_t null_equality,
                                                           cudf_mixed_join_size_data_handle* out,
                                                           cudf_exec_options_t const* options)
{
  return mixed_join_size_data_impl(left_equality,
                                   right_equality,
                                   left_conditional,
                                   right_conditional,
                                   predicate,
                                   null_equality,
                                   out,
                                   options,
                                   mixed_join_size_kind::semi);
}

cudf_status_t cudf_mixed_left_anti_join_begin_size_data_ex(cudf_table_view_handle left_equality,
                                                           cudf_table_view_handle right_equality,
                                                           cudf_table_view_handle left_conditional,
                                                           cudf_table_view_handle right_conditional,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_null_equality_t null_equality,
                                                           cudf_mixed_join_size_data_handle* out,
                                                           cudf_exec_options_t const* options)
{
  return mixed_join_size_data_impl(left_equality,
                                   right_equality,
                                   left_conditional,
                                   right_conditional,
                                   predicate,
                                   null_equality,
                                   out,
                                   options,
                                   mixed_join_size_kind::anti);
}

cudf_status_t cudf_mixed_join_size_data_destroy(cudf_mixed_join_size_data_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner      = static_cast<mixed_join_size_data_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_mixed_join_size_data_output_size(cudf_mixed_join_size_data_handle handle,
                                                    size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "mixed join size data is null");
  CUDF_RUST_TRY
  {
    auto* owner = static_cast<mixed_join_size_data_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->output_size;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_join_size_data_device_allocated_size(cudf_mixed_join_size_data_handle handle,
                                                              size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "mixed join size data is null");
  CUDF_RUST_TRY
  {
    auto* owner = static_cast<mixed_join_size_data_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->device_allocated_size_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_join_size_data_gather_map_size(cudf_mixed_join_size_data_handle handle,
                                                        size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "mixed join size data is null");
  CUDF_RUST_TRY
  {
    auto* owner = static_cast<mixed_join_size_data_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    auto const map_count =
      (owner->kind == mixed_join_size_kind::semi || owner->kind == mixed_join_size_kind::anti) ? 1 : 2;
    *out = checked_join_gather_map_size_bytes(owner->output_size, map_count);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_join_size_data_materialize_indices_ex(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_left_indices,
  cudf_column_handle* out_right_indices,
  cudf_exec_options_t const* options)
{
  return mixed_join_size_data_materialize_indices_impl(
    size_data, out_left_indices, out_right_indices, options);
}

cudf_status_t cudf_mixed_join_size_data_materialize_single_indices_ex(
  cudf_mixed_join_size_data_handle size_data,
  cudf_column_handle* out_indices,
  cudf_exec_options_t const* options)
{
  return mixed_join_size_data_materialize_single_indices_impl(size_data, out_indices, options);
}

}  // extern "C"
