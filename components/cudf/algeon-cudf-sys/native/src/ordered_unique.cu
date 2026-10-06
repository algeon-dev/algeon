#include "cudf_rust/stream_compaction.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <cuda/iterator>
#include <thrust/copy.h>

#include <memory>
#include <vector>

namespace {
template <typename RowEqual>
struct ordered_unique_boundary {
  cudf::size_type const* order;
  cudf::size_type size;
  cudf_duplicate_keep_option_t keep;
  RowEqual equal;

  // CUB unrolls predicate calls; keep runtime-type row comparison out of each expanded call site.
  __device__ __noinline__ bool operator()(cudf::size_type position) const
  {
    if (keep == CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST) {
      return position + 1 == size || !equal(order[position], order[position + 1]);
    }
    auto const first = position == 0 || !equal(order[position - 1], order[position]);
    return keep == CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE
      ? first && (position + 1 == size || !equal(order[position], order[position + 1]))
      : first;
  }
};
}  // namespace

extern "C" cudf_status_t cudf_ordered_unique_indices_ex(cudf_table_view_handle input,
                                                cudf_column_view_handle row_order,
                                                int const* keys,
                                                int num_keys,
                                                cudf_duplicate_keep_option_t keep,
                                                cudf_column_handle* out,
                                                cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(row_order, "row_order is null");
  if (num_keys < 0) {
    set_error("num_keys must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_keys > 0) { CUDF_RUST_NULL_CHECK(keys, "keys is null"); }
  if (keep != CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY && keep != CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST &&
      keep != CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST && keep != CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE) {
    set_error("invalid keep option");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto const& view = *reinterpret_cast<cudf::table_view*>(input);
    auto const& order = *reinterpret_cast<cudf::column_view*>(row_order);
    if (order.type().id() != cudf::type_id::INT32 || order.has_nulls() || order.size() != view.num_rows()) {
      set_error("row_order must be a non-null INT32 permutation of the input rows");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto exec = std::unique_ptr<cudf_exec_context>{};
    auto const status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    if (view.num_rows() == 0 || view.num_columns() == 0 || num_keys == 0) {
      auto result = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }
    auto const key_view = view.select(std::vector<cudf::size_type>(keys, keys + num_keys));
    auto comparator = cudf::detail::row::equality::self_comparator(
      cudf::detail::row::equality::preprocessed_table::create(key_view, exec->stream, *exec->mr));
    auto selected = rmm::device_uvector<cudf::size_type>(order.size(), exec->stream, *exec->mr);
    auto const select = [&](auto equal) {
      auto const end = thrust::copy_if(
        rmm::exec_policy(exec->stream, *exec->mr),
        order.begin<cudf::size_type>(),
        order.end<cudf::size_type>(),
        cuda::counting_iterator<cudf::size_type>{0},
        selected.begin(),
        ordered_unique_boundary<decltype(equal)>{order.begin<cudf::size_type>(), order.size(), keep, equal});
      return static_cast<cudf::size_type>(end - selected.begin());
    };
    auto const selected_size = cudf::detail::has_nested_columns(key_view)
      ? select(comparator.equal_to<true>(cudf::nullate::DYNAMIC{cudf::has_nested_nulls(key_view)}, cudf::null_equality::EQUAL))
      : select(comparator.equal_to<false>(cudf::nullate::DYNAMIC{cudf::has_nested_nulls(key_view)}, cudf::null_equality::EQUAL));
    auto result = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32}, selected_size,
                                                std::move(selected).release(), rmm::device_buffer{}, 0);
    *out = cudf_release_column_handle(std::move(result), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
