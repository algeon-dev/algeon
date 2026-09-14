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
  auto* owner     = reinterpret_cast<mixed_join_size_data_owner*>(handle);
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
    auto* owner = reinterpret_cast<mixed_join_size_data_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->output_size;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_join_size_data_device_allocated_size(
  cudf_mixed_join_size_data_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "mixed join size data is null");
  CUDF_RUST_TRY
  {
    auto* owner = reinterpret_cast<mixed_join_size_data_owner*>(handle);
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
    auto* owner = reinterpret_cast<mixed_join_size_data_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    auto const map_count =
      (owner->kind == mixed_join_size_kind::semi || owner->kind == mixed_join_size_kind::anti) ? 1
                                                                                               : 2;
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

namespace {
struct mixed_left_state_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::shared_ptr<void> stream_lifetime;
  std::unique_ptr<statistics_resource_box> observed_mr;
  // The resource reference and predicate must outlive every retained device allocation.
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> observed_mr_ref;
  cudf_ast_expression_owner_ref predicate;
  std::unique_ptr<cudf::mixed_left_join_state> state;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  cudaStream_t stream = nullptr;
  int32_t device = 0;
};

void validate_mixed_left_context(mixed_left_state_owner const& owner, cudf_exec_context const& exec)
{
  CUDF_EXPECTS(owner.device == exec.device && owner.stream == exec.stream.value() &&
                 owner.mr_raw == exec.mr_raw,
               "mixed left state requires its original device, stream and memory resource");
}
}  // namespace

extern "C" {
cudf_status_t cudf_mixed_left_join_state_create_ex(
  cudf_table_view_handle left_equality, cudf_table_view_handle left_conditional,
  cudf_ast_expression_handle predicate, cudf_null_equality_t null_equality,
  cudf_mixed_left_join_state_handle* out, cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left_equality, "left equality is null");
  CUDF_RUST_NULL_CHECK(left_conditional, "left conditional is null");
  CUDF_RUST_NULL_CHECK(predicate, "predicate is null");
  cudf::null_equality nulls;
  auto status = to_null_equality(null_equality, &nulls);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY {
    std::unique_ptr<cudf_exec_context> exec;
    status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto owner = std::make_unique<mixed_left_state_owner>();
    owner->device = exec->device;
    owner->stream = exec->stream.value();
    owner->mr_raw = exec->mr_raw;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime = exec->stream_lifetime;
    owner->observed_mr = std::make_unique<statistics_resource_box>(
      cuda::mr::any_resource<cuda::mr::device_accessible>{*exec->mr});
    owner->observed_mr_ref = std::make_shared<cuda::mr::any_resource<cuda::mr::device_accessible>>(
      rmm::mr::any_device_resource_from_raw(owner->observed_mr.get()));
    owner->predicate = cudf_ast_expression_owner_ref_from_handle(predicate);
    owner->state = std::make_unique<cudf::mixed_left_join_state>(
      *reinterpret_cast<cudf::table_view*>(left_equality),
      *reinterpret_cast<cudf::table_view*>(left_conditional), nulls, exec->stream, *owner->observed_mr_ref);
    *out = reinterpret_cast<cudf_mixed_left_join_state_handle>(owner.release());
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_consume_ex(
  cudf_mixed_left_join_state_handle state, cudf_table_view_handle right_equality,
  cudf_table_view_handle right_conditional, cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_NULL_CHECK(right_equality, "right equality is null");
  CUDF_RUST_NULL_CHECK(right_conditional, "right conditional is null");
  CUDF_RUST_TRY {
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<mixed_left_state_owner*>(state);
    validate_mixed_left_context(owner, *exec);
    owner.state->consume(*reinterpret_cast<cudf::table_view*>(right_equality),
                         *reinterpret_cast<cudf::table_view*>(right_conditional),
                         *owner.predicate->expression, exec->stream, *owner.observed_mr_ref);
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_seal_ex(
  cudf_mixed_left_join_state_handle state, cudf_join_kind_t kind, cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY {
    CUDF_EXPECTS(kind == CUDF_JOIN_KIND_LEFT_SEMI_JOIN || kind == CUDF_JOIN_KIND_LEFT_ANTI_JOIN,
                 "mixed left state only supports semi/anti output");
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<mixed_left_state_owner*>(state);
    validate_mixed_left_context(owner, *exec);
    owner.state->seal(kind == CUDF_JOIN_KIND_LEFT_ANTI_JOIN, exec->stream, *owner.observed_mr_ref);
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_output_size(cudf_mixed_left_join_state_handle state, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY {
    *out = reinterpret_cast<mixed_left_state_owner*>(state)->state->output_size();
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_device_allocated_size(cudf_mixed_left_join_state_handle state, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY {
    *out = reinterpret_cast<mixed_left_state_owner*>(state)->observed_mr->resource().get_bytes_counter().value;
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_materialize_indices_ex(
  cudf_mixed_left_join_state_handle state, cudf_column_handle* out, cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY {
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<mixed_left_state_owner*>(state);
    validate_mixed_left_context(owner, *exec);
    auto indices = owner.state->materialize_indices(exec->stream, *exec->mr);
    *out = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  } CUDF_RUST_CATCH
}

cudf_status_t cudf_mixed_left_join_state_destroy(cudf_mixed_left_join_state_handle state)
{
  if (state == nullptr) { return CUDF_SUCCESS; }
  auto* owner = reinterpret_cast<mixed_left_state_owner*>(state);
  return cudf_destroy_under_provenance(cached_join_owner_provenance(*owner), [owner] { delete owner; });
}
}  // extern "C"
