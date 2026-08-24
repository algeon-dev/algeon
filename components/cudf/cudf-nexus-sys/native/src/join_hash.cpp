#include "join_common.hpp"

#include <cudf/column/column_device_view.cuh>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace cudf_rust::join;

namespace {
std::size_t checked_metadata_add(std::size_t lhs, std::size_t rhs)
{
  CUDF_EXPECTS(lhs <= std::numeric_limits<std::size_t>::max() - rhs,
               "hash join metadata-only owner preflight size overflow",
               std::overflow_error);
  return lhs + rhs;
}

std::size_t checked_metadata_mul(std::size_t lhs, std::size_t rhs)
{
  CUDF_EXPECTS(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
               "hash join metadata-only owner preflight size overflow",
               std::overflow_error);
  return lhs * rhs;
}

std::uint64_t metadata_mod_mul(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t modulus)
{
#if defined(__SIZEOF_INT128__)
  return static_cast<std::uint64_t>((static_cast<unsigned __int128>(lhs) * rhs) % modulus);
#else
  std::uint64_t result = 0;
  lhs %= modulus;
  while (rhs != 0) {
    if ((rhs & 1) != 0) {
      result = result >= modulus - lhs ? result - (modulus - lhs) : result + lhs;
    }
    lhs = lhs >= modulus - lhs ? lhs - (modulus - lhs) : lhs + lhs;
    rhs >>= 1;
  }
  return result;
#endif
}

std::uint64_t metadata_mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus)
{
  std::uint64_t result = 1;
  base %= modulus;
  while (exponent != 0) {
    if ((exponent & 1) != 0) { result = metadata_mod_mul(result, base, modulus); }
    base = metadata_mod_mul(base, base, modulus);
    exponent >>= 1;
  }
  return result;
}

bool metadata_miller_rabin_test(std::uint64_t value,
                                std::uint64_t witness,
                                std::uint64_t odd_factor,
                                std::uint32_t powers_of_two)
{
  auto residue = metadata_mod_pow(witness % value, odd_factor, value);
  if (residue == 1 || residue == value - 1) { return true; }
  for (std::uint32_t power = 1; power < powers_of_two; ++power) {
    residue = metadata_mod_mul(residue, residue, value);
    if (residue == value - 1) { return true; }
  }
  return false;
}

bool metadata_is_prime(std::uint64_t value)
{
  if (value < 2) { return false; }
  for (auto const small_prime :
       {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
    if (value % small_prime == 0) { return value == small_prime; }
  }

  auto odd_factor             = value - 1;
  std::uint32_t powers_of_two = 0;
  while ((odd_factor & 1) == 0) {
    odd_factor >>= 1;
    ++powers_of_two;
  }
  for (auto const witness :
       {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL}) {
    if (!metadata_miller_rabin_test(value, witness, odd_factor, powers_of_two)) { return false; }
  }
  return true;
}

std::uint64_t metadata_next_prime(std::uint64_t value)
{
  if (value <= 2) { return 2; }
  value |= 1;
  while (!metadata_is_prime(value)) {
    CUDF_EXPECTS(value <= std::numeric_limits<std::uint64_t>::max() - 2,
                 "hash join metadata-only owner preflight prime capacity overflow",
                 std::overflow_error);
    value += 2;
  }
  return value;
}

std::size_t hash_join_slot_storage_reservation_size(cudf::size_type rows, double load_factor)
{
  constexpr auto slot_bytes   = sizeof(cudf::hash_value_type) + sizeof(cudf::size_type);
  constexpr auto stride       = std::size_t{2} * 2;
  auto const requested_double = std::ceil(static_cast<double>(rows) / load_factor);
  CUDF_EXPECTS(std::isfinite(requested_double) &&
                 requested_double < static_cast<double>(std::numeric_limits<std::size_t>::max()),
               "hash join metadata-only owner preflight capacity overflow",
               std::overflow_error);
  auto const requested     = static_cast<std::size_t>(requested_double);
  auto const minimum_slots = std::max<std::size_t>(requested, 1);
  auto const size          = minimum_slots / stride + (minimum_slots % stride != 0);
  auto const capacity_slots =
    checked_metadata_mul(static_cast<std::size_t>(metadata_next_prime(size)), stride);
  constexpr auto alignment   = std::min(std::bit_ceil(slot_bytes * 2), std::size_t{16});
  constexpr auto extra_slots = (alignment - 1) / slot_bytes + 1;
  return checked_metadata_mul(checked_metadata_add(capacity_slots, extra_slots), slot_bytes);
}
}  // namespace

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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);
    auto* lp = reinterpret_cast<cudf::table_view*>(left_payload);
    auto* rp = reinterpret_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] =
        cudf::inner_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);

      *out_left =
        cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right =
        cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);
    auto* lp = reinterpret_cast<cudf::table_view*>(left_payload);
    auto* rp = reinterpret_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] =
        cudf::left_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::DONT_CHECK, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);

      *out_left =
        cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right =
        cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);
    auto* lp = reinterpret_cast<cudf::table_view*>(left_payload);
    auto* rp = reinterpret_cast<cudf::table_view*>(right_payload);

    {
      auto [left_indices, right_indices] =
        cudf::full_join(*lk, *rk, null_eq, exec->stream, *exec->mr);

      auto left_full  = concat_views(*lk, *lp);
      auto right_full = concat_views(*rk, *rp);

      auto left_result = gather_table(
        left_full, left_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);
      auto right_result = gather_table(
        right_full, right_indices, cudf::out_of_bounds_policy::NULLIFY, exec->stream, *exec->mr);

      *out_left =
        cudf_release_table_handle(std::move(left_result), cudf_capture_exec_provenance(*exec));
      *out_right =
        cudf_release_table_handle(std::move(right_result), cudf_capture_exec_provenance(*exec));
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

    auto* left_view  = reinterpret_cast<cudf::table_view*>(left);
    auto* right_view = reinterpret_cast<cudf::table_view*>(right);

    {
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
      left_selected == nullptr ? nullptr : reinterpret_cast<cudf::table_view*>(left_selected);
    auto* right_view =
      right_selected == nullptr ? nullptr : reinterpret_cast<cudf::table_view*>(right_selected);

    {
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
      left_selected == nullptr ? nullptr : reinterpret_cast<cudf::table_view*>(left_selected);
    auto* right_view =
      right_selected == nullptr ? nullptr : reinterpret_cast<cudf::table_view*>(right_selected);

    {
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);

    {
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);

    {
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);

    {
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);

    {
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

    auto* lk = reinterpret_cast<cudf::table_view*>(left_keys);
    auto* rk = reinterpret_cast<cudf::table_view*>(right_keys);

    {
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

    auto* build_view            = reinterpret_cast<cudf::table_view*>(build);
    auto owner                  = std::make_unique<distinct_hash_join_owner>();
    owner->device               = exec->device;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime      = exec->stream_lifetime;
    owner->mr_raw               = exec->mr_raw;
    {
      owner->join =
        std::make_unique<cudf::distinct_hash_join>(*build_view, null_eq, 0.5, exec->stream);
    }
    *out = reinterpret_cast<cudf_distinct_hash_join_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_distinct_hash_join_destroy(cudf_distinct_hash_join_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = reinterpret_cast<distinct_hash_join_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
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

    auto* owner         = reinterpret_cast<distinct_hash_join_owner*>(join);
    auto* distinct_join = owner->join.get();
    auto* probe_view    = reinterpret_cast<cudf::table_view*>(probe);
    {
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

    auto* build_view            = reinterpret_cast<cudf::table_view*>(build);
    auto owner                  = std::make_unique<hash_join_owner>();
    owner->device               = exec->device;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime      = exec->stream_lifetime;
    owner->mr_raw               = exec->mr_raw;
    auto const reservation_size = cudf::hash_join::pre_build_reservation_size(
      *build_view, load_factor, exec->stream, *exec->mr);
    {
      owner->join = std::make_unique<cudf::hash_join>(
        *build_view, nullable, null_eq, load_factor, exec->stream, *exec->mr);
      owner->device_allocated_size_bytes = reservation_size;
    }
    *out = reinterpret_cast<cudf_hash_join_handle>(owner.release());
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

    auto* build_view = reinterpret_cast<cudf::table_view*>(build);
    *out             = cudf::hash_join::pre_build_reservation_size(
      *build_view, load_factor, exec->stream, *exec->mr);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_metadata_only_owner_preflight_ex(
  size_t rows,
  double load_factor,
  size_t flat_device_view_count,
  size_t* hash_table_storage_bytes,
  size_t* retained_key_view_storage_bytes)
{
  CUDF_RUST_NULL_CHECK(hash_table_storage_bytes, "hash_table_storage_bytes is null");
  *hash_table_storage_bytes = 0;
  CUDF_RUST_NULL_CHECK(retained_key_view_storage_bytes, "retained_key_view_storage_bytes is null");
  *retained_key_view_storage_bytes = 0;
  CUDF_RUST_TRY
  {
    CUDF_EXPECTS(rows <= static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()),
                 "hash join metadata-only owner preflight row count exceeds "
                 "cudf::size_type",
                 std::overflow_error);
    CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
                 "Invalid load factor: must be greater than 0 and less than or "
                 "equal to 1.",
                 std::invalid_argument);

    auto const rows_i32         = static_cast<cudf::size_type>(rows);
    auto const hash_table_bytes = hash_join_slot_storage_reservation_size(rows_i32, load_factor);
    auto const key_view_bytes   = checked_metadata_add(
      checked_metadata_mul(flat_device_view_count, sizeof(cudf::column_device_view)),
      alignof(cudf::column_device_view) - 1);
    *hash_table_storage_bytes        = hash_table_bytes;
    *retained_key_view_storage_bytes = key_view_bytes;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_join_destroy(cudf_hash_join_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  auto* owner     = reinterpret_cast<hash_join_owner*>(handle);
  auto provenance = cached_join_owner_provenance(*owner);
  return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
}

cudf_status_t cudf_hash_join_device_allocated_size(cudf_hash_join_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "hash join is null");
  CUDF_RUST_TRY
  {
    auto* owner = reinterpret_cast<hash_join_owner*>(handle);
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
      release_join_index_pair(
        hash_join->inner_join(
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
      release_join_index_pair(
        hash_join->left_join(
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
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

    auto* owner      = reinterpret_cast<hash_join_owner*>(join);
    auto* hash_join  = owner->join.get();
    auto* probe_view = reinterpret_cast<cudf::table_view*>(probe);

    {
      release_join_index_pair(
        hash_join->full_join(
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
