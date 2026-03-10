#include "dictionary.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

static cudf::type_id to_cudf_type_id(cudf_type_id_t id) {
  return static_cast<cudf::type_id>(id);
}

extern "C" {

cudf_status_t cudf_dictionary_encode(cudf_column_view_handle input,
                                     cudf_type_id_t indices_type_id,
                                     cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(input);
    auto indices_type = cudf::data_type(to_cudf_type_id(indices_type_id));
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::dictionary::encode(*view, indices_type, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_decode(cudf_column_view_handle dict_column,
                                     cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::dictionary::decode(dict_view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_keys(cudf_column_view_handle dict_column,
                                   cudf_column_view_handle* out) {
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* keys = new cudf::column_view(dict_view.keys());
    *out = static_cast<cudf_column_view_handle>(keys);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_indices(cudf_column_view_handle dict_column,
                                      cudf_column_view_handle* out) {
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* indices = new cudf::column_view(dict_view.get_indices_annotated());
    *out = static_cast<cudf_column_view_handle>(indices);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_set_keys(cudf_column_view_handle dict_column,
                                       cudf_column_view_handle keys,
                                       cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(dict_column, "dict_column is null");
  CUDF_RUST_NULL_CHECK(keys, "keys is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(dict_column);
    cudf::dictionary_column_view dict_view(*view);
    auto* keys_view = static_cast<cudf::column_view*>(keys);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::dictionary::set_keys(dict_view, *keys_view, stream, mr);
    *out = static_cast<cudf_column_handle>(result.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_dictionary_match_dictionaries(
    const cudf_column_view_handle* dict_columns,
    int32_t num_columns,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(dict_columns, "dict_columns is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  if (num_columns <= 0) {
    set_error("num_columns must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (int32_t i = 0; i < num_columns; ++i) {
    out[i] = nullptr;
  }
  CUDF_RUST_TRY {
    std::vector<cudf::dictionary_column_view> dict_views;
    dict_views.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      CUDF_RUST_NULL_CHECK(dict_columns[i], "dict column view handle is null");
      auto* view = static_cast<cudf::column_view*>(dict_columns[i]);
      dict_views.emplace_back(*view);
    }
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto results = cudf::dictionary::match_dictionaries(dict_views, stream, mr);
    for (int32_t i = 0; i < num_columns; ++i) {
      out[i] = static_cast<cudf_column_handle>(results[i].release());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
