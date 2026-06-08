#ifndef CUDF_RUST_INTEROP_COMMON_HPP
#define CUDF_RUST_INTEROP_COMMON_HPP

// Private helpers shared by prepared_import.cpp and owning_handle.cpp.
// Not part of the public C API surface.

#include "shim_common.h"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstdint>
#include <limits>
#include <unordered_map>

namespace cudf_interop {

inline cudf::size_type require_non_negative_i32(int32_t value, char const* field) {
  if (value < 0) { throw cudf::logic_error(field); }
  return static_cast<cudf::size_type>(value);
}

inline cudf::size_type require_size_type_from_size_t(size_t value, char const* field) {
  if (value > static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw cudf::logic_error(field);
  }
  return static_cast<cudf::size_type>(value);
}

inline cudf::data_type interop_data_type(cudf_interop_semantic_column_t const& column) {
  auto const id = static_cast<cudf::type_id>(column.type_id);
  switch (id) {
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return cudf::data_type{id, column.scale};
    default: return cudf::data_type{id};
  }
}

using column_index_t = std::unordered_map<uint32_t, cudf_interop_semantic_column_t const*>;
using binding_index_t = std::unordered_map<uint32_t, cudf_interop_buffer_binding_t const*>;

inline column_index_t build_column_index(
    cudf_interop_semantic_column_t const* columns,
    int32_t num_columns) {
  column_index_t index;
  index.reserve(static_cast<size_t>(num_columns));
  for (int32_t i = 0; i < num_columns; ++i) { index.emplace(columns[i].column_id, &columns[i]); }
  return index;
}

inline binding_index_t build_binding_index(
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings) {
  binding_index_t index;
  index.reserve(static_cast<size_t>(num_bindings));
  for (int32_t i = 0; i < num_bindings; ++i) { index.emplace(bindings[i].buffer_id, &bindings[i]); }
  return index;
}

}  // namespace cudf_interop

#endif  // CUDF_RUST_INTEROP_COMMON_HPP
