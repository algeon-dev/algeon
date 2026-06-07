#include "owning_handle.hpp"

#include "shim_common.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

bool column_parent_stores_direct_element_buffer(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::EMPTY:
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
      return false;
    default:
      return true;
  }
}

std::size_t cudf_column_view_payload_size_bytes(cudf::column_view const& column,
                                                rmm::cuda_stream_view stream) {
  std::size_t total = 0;

  if (column.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view const strings{column};
    total += static_cast<std::size_t>(strings.chars_size(stream));
  } else if (column_parent_stores_direct_element_buffer(column.type().id())) {
    total += static_cast<std::size_t>(column.size()) *
             static_cast<std::size_t>(cudf::size_of(column.type()));
  }

  if (column.nullable()) {
    total += static_cast<std::size_t>(
                 cudf::num_bitmask_words(static_cast<cudf::size_type>(column.size()))) *
             sizeof(cudf::bitmask_type);
  }

  for (cudf::size_type child_index = 0; child_index < column.num_children();
       ++child_index) {
    total += cudf_column_view_payload_size_bytes(column.child(child_index), stream);
  }

  return total;
}

class native_column_handle_state final : public cudf_column_handle_state {
 public:
  explicit native_column_handle_state(std::unique_ptr<cudf::column> column)
      : _column(std::move(column)) {}

  [[nodiscard]] bool is_native() const noexcept override { return true; }

  [[nodiscard]] cudf::size_type size() const override { return _column->size(); }

  [[nodiscard]] cudf::column_view view() const override { return _column->view(); }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(*_column, stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(_column->view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _column->alloc_size();
  }

  std::unique_ptr<cudf::column> release_native_column() override {
    return std::move(_column);
  }

 private:
  std::unique_ptr<cudf::column> _column;
};

class native_table_handle_state final : public cudf_table_handle_state {
 public:
  explicit native_table_handle_state(std::unique_ptr<cudf::table> table)
      : _table(std::move(table)) {}

  [[nodiscard]] bool is_native() const noexcept override { return true; }

  [[nodiscard]] cudf::size_type num_columns() const override {
    return _table->num_columns();
  }

  [[nodiscard]] cudf::size_type num_rows() const override {
    return _table->num_rows();
  }

  [[nodiscard]] cudf::table_view view() const override { return _table->view(); }

  [[nodiscard]] std::unique_ptr<cudf::table> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::table>(_table->view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    std::size_t total = 0;
    for (auto const& column : _table->view()) {
      total += cudf_column_view_payload_size_bytes(column, stream);
    }
    return total;
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _table->alloc_size();
  }

  std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns()
      override {
    auto columns = _table->release();
    std::vector<std::unique_ptr<cudf_column_handle_state>> result;
    result.reserve(columns.size());
    for (auto& column : columns) {
      result.push_back(
          std::make_unique<native_column_handle_state>(std::move(column)));
    }
    return result;
  }

 private:
  std::unique_ptr<cudf::table> _table;
};

struct foreign_column_common {
  cudf::data_type dtype;
  cudf::size_type size;
  cudf::size_type offset;
  cudf::bitmask_type const* null_mask;
  cudf::size_type null_count;
  std::size_t validity_byte_len;
};

cudf::size_type require_non_negative_i32(int32_t value, char const* field) {
  if (value < 0) { throw cudf::logic_error(field); }
  return static_cast<cudf::size_type>(value);
}

cudf::size_type require_size_type_from_size_t(size_t value, char const* field) {
  if (value > static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw cudf::logic_error(field);
  }
  return static_cast<cudf::size_type>(value);
}

cudf::data_type interop_data_type(cudf_interop_semantic_column_t const& column) {
  auto const id = static_cast<cudf::type_id>(column.type_id);
  switch (id) {
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      return cudf::data_type{id, column.scale};
    default:
      return cudf::data_type{id};
  }
}

using column_index_t =
    std::unordered_map<uint32_t, cudf_interop_semantic_column_t const*>;
using binding_index_t =
    std::unordered_map<uint32_t, cudf_interop_buffer_binding_t const*>;

cudf_interop_buffer_binding_t const* require_device_binding(
    binding_index_t const& index,
    uint32_t id,
    char const* missing_message,
    char const* placement_message);

column_index_t build_column_index(cudf_interop_semantic_column_t const* columns,
                                  int32_t num_columns) {
  column_index_t index;
  index.reserve(static_cast<std::size_t>(num_columns));
  for (int32_t i = 0; i < num_columns; ++i) {
    index.emplace(columns[i].column_id, &columns[i]);
  }
  return index;
}

binding_index_t build_binding_index(cudf_interop_buffer_binding_t const* bindings,
                                    int32_t num_bindings) {
  binding_index_t index;
  index.reserve(static_cast<std::size_t>(num_bindings));
  for (int32_t i = 0; i < num_bindings; ++i) {
    index.emplace(bindings[i].buffer_id, &bindings[i]);
  }
  return index;
}

foreign_column_common build_foreign_column_common(
    cudf_interop_semantic_column_t const& column,
    binding_index_t const& binding_index) {
  auto const dtype = interop_data_type(column);
  auto const size = require_non_negative_i32(
      column.len, "foreign owned-transfer column length must be >= 0");
  auto const offset = require_non_negative_i32(
      column.offset, "foreign owned-transfer column offset must be >= 0");

  cudf::bitmask_type const* null_mask = nullptr;
  std::size_t validity_byte_len = 0;
  cudf::size_type null_count = 0;
  switch (column.validity_kind) {
    case CUDF_INTEROP_VALIDITY_NONE:
      null_count = 0;
      break;
    case CUDF_INTEROP_VALIDITY_MASK: {
      auto const* validity = require_device_binding(
          binding_index, column.validity_buffer_id,
          "foreign owned-transfer import referenced a missing validity binding",
          "foreign owned-transfer import requires CUDA-device validity bindings");
      null_mask = static_cast<cudf::bitmask_type const*>(validity->data);
      validity_byte_len = validity->byte_len;
      null_count = column.null_count >= 0 ? static_cast<cudf::size_type>(column.null_count)
                                          : static_cast<cudf::size_type>(-1);
      break;
    }
    default:
      throw cudf::logic_error(
          "foreign owned-transfer import validity kind is unsupported");
  }

  return foreign_column_common{
      dtype, size, offset, null_mask, null_count, validity_byte_len};
}

cudf_interop_buffer_binding_t const* require_device_binding(
    binding_index_t const& index,
    uint32_t id,
    char const* missing_message,
    char const* placement_message) {
  auto const it = index.find(id);
  if (it == index.end()) { throw cudf::logic_error(missing_message); }
  if (it->second->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return it->second;
}

class foreign_fixed_width_column_handle_state final
    : public cudf_column_handle_state {
 public:
  foreign_fixed_width_column_handle_state(foreign_column_common common,
                                          void const* values,
                                          std::size_t values_byte_len)
      : _common(std::move(common)),
        _values(values),
        _values_byte_len(values_byte_len) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override {
    return cudf::column_view(_common.dtype, _common.size, _values,
                             _common.null_mask, _common.null_count,
                             _common.offset);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _values_byte_len + _common.validity_byte_len;
  }

 private:
  foreign_column_common _common;
  void const* _values;
  std::size_t _values_byte_len;
};

class foreign_binary_like_column_handle_state final
    : public cudf_column_handle_state {
 public:
  foreign_binary_like_column_handle_state(foreign_column_common common,
                                          void const* chars,
                                          std::size_t chars_byte_len,
                                          int32_t const* offsets,
                                          std::size_t offsets_byte_len)
      : _common(std::move(common)),
        _chars(chars),
        _chars_byte_len(chars_byte_len),
        _offsets(offsets),
        _offsets_byte_len(offsets_byte_len) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override {
    std::vector<cudf::column_view> children;
    children.reserve(1);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(
            _offsets_byte_len / sizeof(int32_t),
            "foreign owned-transfer string offsets binding exceeds cudf::size_type"),
        _offsets, nullptr, 0, 0);
    return cudf::column_view(_common.dtype, _common.size, _chars,
                             _common.null_mask, _common.null_count,
                             _common.offset, children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _chars_byte_len + _offsets_byte_len + _common.validity_byte_len;
  }

 private:
  foreign_column_common _common;
  void const* _chars;
  std::size_t _chars_byte_len;
  int32_t const* _offsets;
  std::size_t _offsets_byte_len;
};

class foreign_list_column_handle_state final : public cudf_column_handle_state {
 public:
  foreign_list_column_handle_state(
      foreign_column_common common,
      int32_t const* offsets,
      std::size_t offsets_byte_len,
      std::unique_ptr<cudf_column_handle_state> child)
      : _common(std::move(common)),
        _offsets(offsets),
        _offsets_byte_len(offsets_byte_len),
        _child(std::move(child)) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override {
    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(
            _offsets_byte_len / sizeof(int32_t),
            "foreign owned-transfer list offsets binding exceeds cudf::size_type"),
        _offsets, nullptr, 0, 0);
    children.emplace_back(_child->view());
    return cudf::column_view(_common.dtype, _common.size, nullptr,
                             _common.null_mask, _common.null_count,
                             _common.offset, children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _offsets_byte_len + _common.validity_byte_len +
           _child->device_allocated_size();
  }

 private:
  foreign_column_common _common;
  int32_t const* _offsets;
  std::size_t _offsets_byte_len;
  std::unique_ptr<cudf_column_handle_state> _child;
};

class foreign_struct_column_handle_state final
    : public cudf_column_handle_state {
 public:
  foreign_struct_column_handle_state(
      foreign_column_common common,
      std::vector<std::unique_ptr<cudf_column_handle_state>> children)
      : _common(std::move(common)), _children(std::move(children)) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override {
    std::vector<cudf::column_view> children;
    children.reserve(_children.size());
    for (auto const& child : _children) {
      children.emplace_back(child->view());
    }
    return cudf::column_view(_common.dtype, _common.size, nullptr,
                             _common.null_mask, _common.null_count,
                             _common.offset, children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    std::size_t total = _common.validity_byte_len;
    for (auto const& child : _children) {
      total += child->device_allocated_size();
    }
    return total;
  }

 private:
  foreign_column_common _common;
  std::vector<std::unique_ptr<cudf_column_handle_state>> _children;
};

class foreign_dictionary_column_handle_state final
    : public cudf_column_handle_state {
 public:
  foreign_dictionary_column_handle_state(
      foreign_column_common common,
      int32_t const* indices,
      std::size_t indices_byte_len,
      std::unique_ptr<cudf_column_handle_state> keys)
      : _common(std::move(common)),
        _indices(indices),
        _indices_byte_len(indices_byte_len),
        _keys(std::move(keys)) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override {
    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(
            _indices_byte_len / sizeof(int32_t),
            "foreign owned-transfer dictionary indices binding exceeds cudf::size_type"),
        _indices, nullptr, 0, 0);
    children.emplace_back(_keys->view());
    return cudf::column_view(_common.dtype, _common.size, nullptr,
                             _common.null_mask, _common.null_count,
                             _common.offset, children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return _indices_byte_len + _common.validity_byte_len +
           _keys->device_allocated_size();
  }

 private:
  foreign_column_common _common;
  int32_t const* _indices;
  std::size_t _indices_byte_len;
  std::unique_ptr<cudf_column_handle_state> _keys;
};

std::unique_ptr<cudf_column_handle_state> build_foreign_column_handle_state(
    uint32_t root_column_id,
    column_index_t const& column_index,
    binding_index_t const& binding_index) {
  auto const column_it = column_index.find(root_column_id);
  CUDF_EXPECTS(column_it != column_index.end(),
               "foreign owned-transfer import referenced a missing semantic column");
  auto const* column = column_it->second;
  auto const common = build_foreign_column_common(*column, binding_index);

  switch (column->layout_kind) {
    case CUDF_INTEROP_LAYOUT_FIXED_WIDTH: {
      CUDF_EXPECTS(cudf::is_fixed_width(common.dtype),
                   "foreign owned-transfer fixed-width layout requires a fixed-width type");
      auto const* values = require_device_binding(
          binding_index, column->values_buffer_id,
          "foreign owned-transfer import referenced a missing fixed-width values binding",
          "foreign owned-transfer import requires CUDA-device fixed-width value bindings");
      return std::make_unique<foreign_fixed_width_column_handle_state>(
          common, values->data, values->byte_len);
    }

    case CUDF_INTEROP_LAYOUT_BINARY_LIKE: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::STRING,
                   "foreign owned-transfer binary-like layout requires STRING type");
      auto const* offsets = require_device_binding(
          binding_index, column->offsets_buffer_id,
          "foreign owned-transfer import referenced a missing binary-like offsets binding",
          "foreign owned-transfer import requires CUDA-device binary-like offsets bindings");
      CUDF_EXPECTS(offsets->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer binary-like offsets binding must be 32-bit aligned");
      auto const* chars = require_device_binding(
          binding_index, column->data_buffer_id,
          "foreign owned-transfer import referenced a missing binary-like chars binding",
          "foreign owned-transfer import requires CUDA-device binary-like chars bindings");
      return std::make_unique<foreign_binary_like_column_handle_state>(
          common, chars->data, chars->byte_len,
          static_cast<int32_t const*>(offsets->data), offsets->byte_len);
    }

    case CUDF_INTEROP_LAYOUT_LIST: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::LIST,
                   "foreign owned-transfer list layout requires LIST type");
      auto const* offsets = require_device_binding(
          binding_index, column->offsets_buffer_id,
          "foreign owned-transfer import referenced a missing list offsets binding",
          "foreign owned-transfer import requires CUDA-device list offsets bindings");
      CUDF_EXPECTS(offsets->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer list offsets binding must be 32-bit aligned");
      auto child = build_foreign_column_handle_state(
          column->child_column_id, column_index, binding_index);
      return std::make_unique<foreign_list_column_handle_state>(
          common, static_cast<int32_t const*>(offsets->data), offsets->byte_len,
          std::move(child));
    }

    case CUDF_INTEROP_LAYOUT_STRUCT: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::STRUCT,
                   "foreign owned-transfer struct layout requires STRUCT type");
      CUDF_EXPECTS(column->num_child_column_ids > 0,
                   "foreign owned-transfer struct layout must reference at least one child");
      CUDF_EXPECTS(column->child_column_ids != nullptr,
                   "foreign owned-transfer struct child ids pointer is null");
      std::vector<std::unique_ptr<cudf_column_handle_state>> children;
      children.reserve(static_cast<std::size_t>(column->num_child_column_ids));
      for (int32_t index = 0; index < column->num_child_column_ids; ++index) {
        children.emplace_back(build_foreign_column_handle_state(
            column->child_column_ids[index], column_index, binding_index));
      }
      return std::make_unique<foreign_struct_column_handle_state>(
          common, std::move(children));
    }

    case CUDF_INTEROP_LAYOUT_DICTIONARY: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::DICTIONARY32,
                   "foreign owned-transfer dictionary layout requires DICTIONARY32 type");
      auto const* indices = require_device_binding(
          binding_index, column->indices_buffer_id,
          "foreign owned-transfer import referenced a missing dictionary indices binding",
          "foreign owned-transfer import requires CUDA-device dictionary indices bindings");
      CUDF_EXPECTS(indices->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer dictionary indices binding must be 32-bit aligned");
      auto keys = build_foreign_column_handle_state(
          column->keys_column_id, column_index, binding_index);
      return std::make_unique<foreign_dictionary_column_handle_state>(
          common, static_cast<int32_t const*>(indices->data), indices->byte_len,
          std::move(keys));
    }

    default:
      CUDF_FAIL("foreign owned-transfer import layout kind is unsupported");
  }
}

class foreign_table_handle_state final : public cudf_table_handle_state {
 public:
  foreign_table_handle_state(
      cudf::size_type num_rows,
      std::vector<std::unique_ptr<cudf_column_handle_state>> columns)
      : _num_rows(num_rows), _columns(std::move(columns)) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type num_columns() const override {
    return static_cast<cudf::size_type>(_columns.size());
  }

  [[nodiscard]] cudf::size_type num_rows() const override { return _num_rows; }

  [[nodiscard]] cudf::table_view view() const override {
    std::vector<cudf::column_view> views;
    views.reserve(_columns.size());
    for (auto const& column : _columns) {
      views.push_back(column->view());
    }
    return cudf::table_view{views};
  }

  [[nodiscard]] std::unique_ptr<cudf::table> deep_copy(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto table_view = view();
    return std::make_unique<cudf::table>(table_view, stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(
      rmm::cuda_stream_view stream) const override {
    std::size_t total = 0;
    for (auto const& column : _columns) {
      total += column->payload_size(stream);
    }
    return total;
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    std::size_t total = 0;
    for (auto const& column : _columns) {
      total += column->device_allocated_size();
    }
    return total;
  }

  std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns()
      override {
    return std::move(_columns);
  }

 private:
  cudf::size_type _num_rows;
  std::vector<std::unique_ptr<cudf_column_handle_state>> _columns;
};

}  // namespace

std::unique_ptr<cudf::column> cudf_column_handle_state::release_native_column() {
  throw cudf::logic_error("column handle is not native-backed");
}

std::vector<std::unique_ptr<cudf_column_handle_state>>
cudf_table_handle_state::release_columns() {
  throw cudf::logic_error("table handle cannot release columns");
}

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column) {
  return cudf_release_column_handle(
      std::make_unique<native_column_handle_state>(std::move(column)));
}

cudf_column_handle cudf_release_column_handle(
    std::unique_ptr<cudf_column_handle_state> column) {
  return static_cast<cudf_column_handle>(column.release());
}

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table) {
  return cudf_release_table_handle(
      std::make_unique<native_table_handle_state>(std::move(table)));
}

cudf_table_handle cudf_release_table_handle(
    std::unique_ptr<cudf_table_handle_state> table) {
  return static_cast<cudf_table_handle>(table.release());
}

cudf_column_handle_state& cudf_column_state_from_handle(cudf_column_handle handle) {
  return *static_cast<cudf_column_handle_state*>(handle);
}

cudf_table_handle_state& cudf_table_state_from_handle(cudf_table_handle handle) {
  return *static_cast<cudf_table_handle_state*>(handle);
}

std::unique_ptr<cudf_column_handle_state> cudf_take_column_state(
    cudf_column_handle handle) {
  return std::unique_ptr<cudf_column_handle_state>(
      static_cast<cudf_column_handle_state*>(handle));
}

std::unique_ptr<cudf_table_handle_state> cudf_take_table_state(
    cudf_table_handle handle) {
  return std::unique_ptr<cudf_table_handle_state>(
      static_cast<cudf_table_handle_state*>(handle));
}

std::unique_ptr<cudf_column_handle_state> cudf_make_foreign_column_handle_state(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings) {
  auto const column_index =
      build_column_index(semantic_columns, num_semantic_columns);
  auto const binding_index = build_binding_index(bindings, num_bindings);
  return build_foreign_column_handle_state(root_column_id, column_index,
                                           binding_index);
}

std::unique_ptr<cudf_table_handle_state> cudf_make_foreign_table_handle_state(
    cudf_interop_semantic_table_t const* table,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings) {
  CUDF_EXPECTS(table != nullptr, "foreign owned-transfer table descriptor is null");
  CUDF_EXPECTS(table->num_rows >= 0,
               "foreign owned-transfer table row count must be >= 0");
  CUDF_EXPECTS(table->num_columns >= 0,
               "foreign owned-transfer table column count must be >= 0");
  if (table->num_columns > 0) {
    CUDF_EXPECTS(table->column_ids != nullptr,
                 "foreign owned-transfer table column ids pointer is null");
  }

  auto const expected_rows = static_cast<cudf::size_type>(table->num_rows);
  std::vector<std::unique_ptr<cudf_column_handle_state>> columns;
  columns.reserve(static_cast<std::size_t>(table->num_columns));
  for (int32_t index = 0; index < table->num_columns; ++index) {
    auto column = cudf_make_foreign_column_handle_state(
        table->column_ids[index], semantic_columns, num_semantic_columns, bindings,
        num_bindings);
    CUDF_EXPECTS(column->size() == expected_rows,
                 "foreign owned-transfer table row count must match every top-level column length");
    columns.push_back(std::move(column));
  }

  return std::make_unique<foreign_table_handle_state>(expected_rows,
                                                      std::move(columns));
}

std::unique_ptr<cudf_table_handle_state>
cudf_make_foreign_table_handle_state_from_columns(
    cudf::size_type num_rows,
    std::vector<std::unique_ptr<cudf_column_handle_state>> columns) {
  return std::make_unique<foreign_table_handle_state>(num_rows,
                                                      std::move(columns));
}
