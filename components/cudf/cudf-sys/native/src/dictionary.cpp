#include "dictionary.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <new>
#include <vector>

extern "C" {
cudf_status_t cudf_dictionary_encode(cudf_column_view_handle input,
                                     cudf_type_id_t indices_type_id,
                                     cudf_column_handle* out)
{
  return cudf_dictionary_encode_ex(input, indices_type_id, out, nullptr);
}

cudf_status_t cudf_dictionary_encode_ex(cudf_column_view_handle input,
                                        cudf_type_id_t indices_type_id,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto* view                      = static_cast<cudf::column_view*>(input);
    auto indices_type               = cudf::data_type(to_cudf_type_id(indices_type_id));
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto result = cudf::dictionary::encode(*view, indices_type, exec.stream, exec.mr);
      *out        = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_decode(cudf_column_view_handle dict_column, cudf_column_handle* out)
{
  return cudf_dictionary_decode_ex(dict_column, out, nullptr);
}

cudf_status_t cudf_dictionary_decode_ex(cudf_column_view_handle dict_column,
                                        cudf_column_handle* out,
                                        cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto result = cudf::dictionary::decode(dict_view, exec.stream, exec.mr);
      *out        = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_keys(cudf_column_view_handle dict_column,
                                   cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* keys = new cudf::column_view(dict_view.keys());
    *out       = static_cast<cudf_column_view_handle>(keys);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_indices(cudf_column_view_handle dict_column,
                                      cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* indices = new cudf::column_view(dict_view.get_indices_annotated());
    *out          = static_cast<cudf_column_view_handle>(indices);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_set_keys(cudf_column_view_handle dict_column,
                                       cudf_column_view_handle keys,
                                       cudf_column_handle* out)
{
  return cudf_dictionary_set_keys_ex(dict_column, keys, out, nullptr);
}

cudf_status_t cudf_dictionary_set_keys_ex(cudf_column_view_handle dict_column,
                                          cudf_column_view_handle keys,
                                          cudf_column_handle* out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_TRY
  {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* keys_view                 = static_cast<cudf::column_view*>(keys);
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      auto result = cudf::dictionary::set_keys(dict_view, *keys_view, exec.stream, exec.mr);
      *out        = cudf_release_column_handle(std::move(result));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_match_dictionaries(const cudf_column_view_handle* dict_columns,
                                                 int32_t num_columns,
                                                 cudf_column_handle** out_columns,
                                                 int32_t* out_count)
{
  return cudf_dictionary_match_dictionaries_ex(
    dict_columns, num_columns, out_columns, out_count, nullptr);
}

cudf_status_t cudf_dictionary_match_dictionaries_ex(const cudf_column_view_handle* dict_columns,
                                                    int32_t num_columns,
                                                    cudf_column_handle** out_columns,
                                                    int32_t* out_count,
                                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_columns, "out_columns is null");
  *out_columns = nullptr;
  CUDF_RUST_NULL_CHECK(out_count, "out_count is null");
  *out_count = 0;
  CUDF_RUST_NULL_CHECK(dict_columns, "dict_columns is null");
  if (num_columns <= 0) {
    set_error("num_columns must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    std::vector<cudf::dictionary_column_view> dict_views;
    dict_views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(dict_columns[i], "dict column view handle is null");
      auto* view = static_cast<cudf::column_view*>(dict_columns[i]);
      dict_views.emplace_back(*view);
    }
    cudf_exec_context exec          = cudf_make_exec_context();
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    std::vector<std::unique_ptr<cudf::column>> results;
    {
      cudf_current_device_resource_guard current_mr{exec.mr_raw};
      results = cudf::dictionary::match_dictionaries(dict_views, exec.stream, exec.mr);
    }

    auto* columns = new (std::nothrow) cudf_column_handle[num_columns];
    if (columns == nullptr) {
      set_error("failed to allocate dictionary match result column handle array");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < num_columns; ++i) {
      columns[i] = cudf_release_column_handle(std::move(results[i]));
    }

    *out_columns = columns;
    *out_count   = num_columns;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
