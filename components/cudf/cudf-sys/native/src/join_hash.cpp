#include "join_common.hpp"

using namespace cudf_rust::join;

extern "C" {
cudf_status_t cudf_inner_join_ex(cudf_table_view_handle left_keys,
                                 cudf_table_view_handle right_keys,
                                 cudf_table_view_handle left_payload,
                                 cudf_table_view_handle right_payload,
                                 cudf_null_equality_t null_equality,
                                 cudf_table_handle* out_left,
                                 cudf_table_handle* out_right,
                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto [left_indices, right_indices] =
        cudf::inner_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);

      *out_left  = cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right = cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto [left_indices, right_indices] = cudf::left_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);

      *out_left  = cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right = cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_full_join_ex(cudf_table_view_handle left_keys,
                                cudf_table_view_handle right_keys,
                                cudf_table_view_handle left_payload,
                                cudf_table_view_handle right_payload,
                                cudf_null_equality_t null_equality,
                                cudf_table_handle* out_left,
                                cudf_table_handle* out_right,
                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left, "out_left is null");
  *out_left = nullptr;
  CUDF_RUST_NULL_CHECK(out_right, "out_right is null");
  *out_right = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  CUDF_RUST_NULL_CHECK(left_payload, "left_payload is null");
  CUDF_RUST_NULL_CHECK(right_payload, "right_payload is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);
    auto* lp = static_cast<cudf::table_view*>(left_payload);
    auto* rp = static_cast<cudf::table_view*>(right_payload);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto [left_indices, right_indices] = cudf::full_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);

      *out_left  = cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right = cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_cross_join_ex(cudf_table_view_handle left,
                                 cudf_table_view_handle right,
                                 cudf_table_handle* out,
                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left, "left is null");
  CUDF_RUST_NULL_CHECK(right, "right is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view  = static_cast<cudf::table_view*>(left);
    auto* right_view = static_cast<cudf::table_view*>(right);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = cudf::cross_join(*left_view, *right_view, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_projected_cross_join_ex(cudf_table_view_handle left_selected,
                                           cudf_table_view_handle right_selected,
                                           size_t left_num_rows,
                                           size_t right_num_rows,
                                           cudf_table_handle* out,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view =
      left_selected == nullptr ? nullptr : static_cast<cudf::table_view*>(left_selected);
    auto* right_view =
      right_selected == nullptr ? nullptr : static_cast<cudf::table_view*>(right_selected);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = projected_cross_join_table(
        left_view, right_view, left_num_rows, right_num_rows, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_projected_cross_join_device_allocated_size_ex(
  cudf_table_view_handle left_selected,
  cudf_table_view_handle right_selected,
  size_t left_num_rows,
  size_t right_num_rows,
  size_t* out,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* left_view =
      left_selected == nullptr ? nullptr : static_cast<cudf::table_view*>(left_selected);
    auto* right_view =
      right_selected == nullptr ? nullptr : static_cast<cudf::table_view*>(right_selected);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto result = projected_cross_join_table(
        left_view, right_view, left_num_rows, right_num_rows, exec->stream, *exec->mr);
      *out = result->alloc_size();
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_inner_join_indices_ex(cudf_table_view_handle left_keys,
                                         cudf_table_view_handle right_keys,
                                         cudf_null_equality_t null_equality,
                                         cudf_column_handle* out_left_indices,
                                         cudf_column_handle* out_right_indices,
                                         cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(cudf::inner_join(*lk, *rk, null_eq, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(cudf::left_join(*lk, *rk, null_eq, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_full_join_indices_ex(cudf_table_view_handle left_keys,
                                        cudf_table_view_handle right_keys,
                                        cudf_null_equality_t null_equality,
                                        cudf_column_handle* out_left_indices,
                                        cudf_column_handle* out_right_indices,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(cudf::full_join(*lk, *rk, null_eq, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_semi_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::filtered_join join{*rk, null_eq, 0.5, exec->stream};
      auto indices = join.semi_join(*lk, exec->stream, *exec->mr);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_left_anti_join_indices_ex(cudf_table_view_handle left_keys,
                                             cudf_table_view_handle right_keys,
                                             cudf_null_equality_t null_equality,
                                             cudf_column_handle* out_indices,
                                             cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_indices, "out_indices is null");
  *out_indices = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left_keys is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right_keys is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* lk = static_cast<cudf::table_view*>(left_keys);
    auto* rk = static_cast<cudf::table_view*>(right_keys);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::filtered_join join{*rk, null_eq, 0.5, exec->stream};
      auto indices = join.anti_join(*lk, exec->stream, *exec->mr);
      *out_indices = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                                cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_distinct_hash_join_create_ex(cudf_table_view_handle build,
                                                cudf_null_equality_t null_equality,
                                                cudf_distinct_hash_join_handle* out,
                                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  cudf::null_equality null_eq;
  cudf_status_t status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* build_view = static_cast<cudf::table_view*>(build);
    auto owner       = std::make_unique<distinct_hash_join_owner>();
    owner->device               = exec->device;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime      = exec->stream_lifetime;
    owner->mr_raw               = exec->mr_raw;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      owner->join =
        std::make_unique<cudf::distinct_hash_join>(*build_view, null_eq, 0.5, exec->stream);
    }
    *out = static_cast<cudf_distinct_hash_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_distinct_hash_join_destroy(cudf_distinct_hash_join_handle handle)
{
  if (handle == nullptr) { return; }
  cudf_clear_error();
  auto* owner      = static_cast<distinct_hash_join_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  (void)cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_distinct_hash_join_left_join_ex(cudf_distinct_hash_join_handle join,
                                                   cudf_table_view_handle probe,
                                                   cudf_column_handle* out,
                                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner         = static_cast<distinct_hash_join_owner*>(join);
    auto* distinct_join = owner->join.get();
    auto* probe_view    = static_cast<cudf::table_view*>(probe);
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto indices = distinct_join->left_join(*probe_view, exec->stream, *exec->mr);
      *out         = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_create_ex(cudf_table_view_handle build,
                                       cudf_nullable_join_t nullable_join,
                                       cudf_null_equality_t null_equality,
                                       double load_factor,
                                       cudf_hash_join_handle* out,
                                       cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(build, "build is null");
  cudf::nullable_join nullable;
  cudf_status_t status = to_nullable_join(nullable_join, &nullable);
  if (status != CUDF_SUCCESS) { return status; }
  cudf::null_equality null_eq;
  status = to_null_equality(null_equality, &null_eq);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* build_view = static_cast<cudf::table_view*>(build);
    auto owner       = std::make_unique<hash_join_owner>();
    owner->device    = exec->device;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime = exec->stream_lifetime;
    owner->mr_raw = exec->mr_raw;
    auto const reservation_size =
      cudf::hash_join::pre_build_reservation_size(*build_view, load_factor);
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      owner->join =
        std::make_unique<cudf::hash_join>(*build_view, nullable, null_eq, load_factor, exec->stream);
      owner->device_allocated_size_bytes = reservation_size;
    }
    *out = static_cast<cudf_hash_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_pre_build_reservation_size_ex(cudf_table_view_handle build,
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

    auto* build_view = static_cast<cudf::table_view*>(build);
    *out            = cudf::hash_join::pre_build_reservation_size(*build_view, load_factor);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_hash_join_destroy(cudf_hash_join_handle handle)
{
  if (handle == nullptr) { return; }
  cudf_clear_error();
  auto* owner      = static_cast<hash_join_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  (void)cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_hash_join_device_allocated_size(cudf_hash_join_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "hash join is null");
  CUDF_RUST_TRY
  {
    auto* owner = static_cast<hash_join_owner*>(handle);
    cudf_device_guard device_guard;
    cudf_status_t const device_status = device_guard.activate(owner->device);
    if (device_status != CUDF_SUCCESS) { return device_status; }
    *out = owner->device_allocated_size_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join_ex(cudf_hash_join_handle join,
                                           cudf_table_view_handle probe,
                                           cudf_column_handle* out_left_indices,
                                           cudf_column_handle* out_right_indices,
                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->inner_join(*probe_view, {}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join_with_output_size_ex(cudf_hash_join_handle join,
                                                            cudf_table_view_handle probe,
                                                            size_t output_size,
                                                            cudf_column_handle* out_left_indices,
                                                            cudf_column_handle* out_right_indices,
                                                            cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->inner_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_inner_join_size_ex(cudf_hash_join_handle join,
                                                cudf_table_view_handle probe,
                                                size_t* out,
                                                cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::inner);
}

cudf_status_t cudf_hash_join_left_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->left_join(*probe_view, {}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_left_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->left_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_left_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::left);
}

cudf_status_t cudf_hash_join_full_join_ex(cudf_hash_join_handle join,
                                          cudf_table_view_handle probe,
                                          cudf_column_handle* out_left_indices,
                                          cudf_column_handle* out_right_indices,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->full_join(*probe_view, {}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_full_join_with_output_size_ex(cudf_hash_join_handle join,
                                                           cudf_table_view_handle probe,
                                                           size_t output_size,
                                                           cudf_column_handle* out_left_indices,
                                                           cudf_column_handle* out_right_indices,
                                                           cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_left_indices, "out_left_indices is null");
  *out_left_indices = nullptr;
  CUDF_RUST_NULL_CHECK(out_right_indices, "out_right_indices is null");
  *out_right_indices = nullptr;
  CUDF_RUST_NULL_CHECK(join, "join is null");
  CUDF_RUST_NULL_CHECK(probe, "probe is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* owner      = static_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = static_cast<cudf::table_view*>(probe);

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      release_join_index_pair(hash_join->full_join(
                                *probe_view, std::optional<std::size_t>{output_size}, exec->stream, *exec->mr),
                              out_left_indices,
                              out_right_indices,
                              cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_full_join_size_ex(cudf_hash_join_handle join,
                                               cudf_table_view_handle probe,
                                               size_t* out,
                                               cudf_exec_options_t const* options)
{
  return hash_join_size_impl(join, probe, out, options, hash_join_probe_kind::full);
}
}  // extern "C"
