#include "owning_handle.hpp"

#include "interop_common.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

using cudf_interop::binding_index_t;
using cudf_interop::build_binding_index;
using cudf_interop::build_column_index;
using cudf_interop::column_index_t;
using cudf_interop::interop_data_type;
using cudf_interop::require_non_negative_i32;
using cudf_interop::require_size_type_from_size_t;

namespace {
bool column_parent_stores_direct_element_buffer(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::EMPTY:
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32: return false;
    default: return true;
  }
}

std::size_t checked_size_add(std::size_t left, std::size_t right, char const* context)
{
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(std::string{"size accounting overflow while "} + context);
  }
  return left + right;
}

std::size_t checked_size_mul(std::size_t left, std::size_t right, char const* context)
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string{"size accounting overflow while "} + context);
  }
  return left * right;
}

std::size_t cudf_column_view_payload_size_bytes(cudf::column_view const& column,
                                                rmm::cuda_stream_view stream)
{
  std::size_t total = 0;

  if (column.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view const strings{column};
    total = checked_size_add(
      total, static_cast<std::size_t>(strings.chars_size(stream)), "sizing string characters");
  } else if (column_parent_stores_direct_element_buffer(column.type().id())) {
    auto const values_bytes =
      checked_size_mul(static_cast<std::size_t>(column.size()),
                       static_cast<std::size_t>(cudf::size_of(column.type())),
                       "sizing fixed-width column values");
    total = checked_size_add(total, values_bytes, "accumulating fixed-width column values");
  }

  if (column.nullable()) {
    auto const mask_bytes = checked_size_mul(static_cast<std::size_t>(cudf::num_bitmask_words(
                                               static_cast<cudf::size_type>(column.size()))),
                                             sizeof(cudf::bitmask_type),
                                             "sizing column validity mask");
    total = checked_size_add(total, mask_bytes, "accumulating column validity mask");
  }

  for (cudf::size_type child_index = 0; child_index < column.num_children(); ++child_index) {
    total = checked_size_add(total,
                             cudf_column_view_payload_size_bytes(column.child(child_index), stream),
                             "accumulating recursive child payload");
  }

  return total;
}

class native_column_handle_state final : public cudf_column_handle_state {
 public:
  native_column_handle_state(std::unique_ptr<cudf::column> column,
                             cudf_handle_provenance provenance)
    : _provenance(std::move(provenance)), _column(std::move(column))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return true; }

  [[nodiscard]] cudf::size_type size() const override { return _column->size(); }

  [[nodiscard]] cudf::column_view view() const override { return _column->view(); }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(*_column, stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(_column->view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override { return _column->alloc_size(); }

  [[nodiscard]] std::optional<cudf_handle_provenance> destroy_provenance() const noexcept override
  {
    return _provenance;
  }

  std::unique_ptr<cudf::column> release_native_column() override { return std::move(_column); }

 private:
  cudf_handle_provenance _provenance;
  std::unique_ptr<cudf::column> _column;
};

class native_table_handle_state final : public cudf_table_handle_state {
 public:
  native_table_handle_state(std::unique_ptr<cudf::table> table, cudf_handle_provenance provenance)
    : _provenance(std::move(provenance)), _table(std::move(table))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return true; }

  [[nodiscard]] cudf::size_type num_columns() const override { return _table->num_columns(); }

  [[nodiscard]] cudf::size_type num_rows() const override { return _table->num_rows(); }

  [[nodiscard]] cudf::table_view view() const override { return _table->view(); }

  [[nodiscard]] std::unique_ptr<cudf::table> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::table>(_table->view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    std::size_t total = 0;
    for (auto const& column : _table->view()) {
      total = checked_size_add(total,
                               cudf_column_view_payload_size_bytes(column, stream),
                               "accumulating table column payload");
    }
    return total;
  }

  [[nodiscard]] std::size_t device_allocated_size() const override { return _table->alloc_size(); }

  [[nodiscard]] std::optional<cudf_handle_provenance> destroy_provenance() const noexcept override
  {
    return _provenance;
  }

  std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns() override
  {
    return cudf_invoke_under_provenance(_provenance, [&] {
      std::vector<std::unique_ptr<cudf_column_handle_state>> result;
      result.reserve(static_cast<std::size_t>(_table->num_columns()));
      auto columns = _table->release();
      try {
        for (auto& column : columns) {
          result.push_back(
            std::make_unique<native_column_handle_state>(std::move(column), _provenance));
        }
      } catch (...) {
        std::vector<std::unique_ptr<cudf::column>> restored_columns;
        restored_columns.reserve(columns.size());
        for (auto& column_state : result) {
          restored_columns.push_back(column_state->release_native_column());
        }
        for (auto& column : columns) {
          if (column != nullptr) { restored_columns.push_back(std::move(column)); }
        }
        _table = std::make_unique<cudf::table>(std::move(restored_columns));
        throw;
      }
      return result;
    });
  }

  void restore_released_columns(
    cudf::size_type, std::vector<std::unique_ptr<cudf_column_handle_state>> columns) override
  {
    cudf_invoke_under_provenance(_provenance, [&] {
      std::vector<std::unique_ptr<cudf::column>> native_columns;
      native_columns.reserve(columns.size());
      for (auto& column : columns) {
        native_columns.push_back(column->release_native_column());
      }
      _table = std::make_unique<cudf::table>(std::move(native_columns));
    });
  }

 private:
  cudf_handle_provenance _provenance;
  std::unique_ptr<cudf::table> _table;
};

// ---------------------------------------------------------------------------
// Column-state-backed handle states.
//
// SAFETY INVARIANT: every foreign_*_column_handle_state stores const pointers
// into GPU buffers whose lifetime is managed by the Rust caller (via
// Arc<dyn Any + Send + Sync> on Column::retained_owner).  These classes must
// NEVER free, reallocate, or write through the stored pointers.
//
// - is_native() returns false.
// - view() synthesises a cudf::column_view from the stored pointers.
// - deep_copy() always allocates fresh device memory via the cudf::column
//   copy constructor -- it must never return an alias of the foreign buffers.
// - release_native_column() inherits the base-class default that throws.
//
// Violating this invariant causes use-after-free on the Rust side.
// ---------------------------------------------------------------------------

struct foreign_column_common {
  cudf::data_type dtype;
  cudf::size_type size;
  cudf::size_type offset;
  cudf::bitmask_type const* null_mask;
  cudf::size_type null_count;
  std::size_t validity_byte_len;
};

cudf_interop_buffer_binding_t const* require_device_binding(binding_index_t const& index,
                                                            uint32_t id,
                                                            char const* missing_message,
                                                            char const* placement_message);

foreign_column_common build_foreign_column_common(cudf_interop_semantic_column_t const& column,
                                                  binding_index_t const& binding_index)
{
  auto const dtype = interop_data_type(column);
  auto const size =
    require_non_negative_i32(column.len, "foreign owned-transfer column length must be >= 0");
  auto const offset =
    require_non_negative_i32(column.offset, "foreign owned-transfer column offset must be >= 0");

  cudf::bitmask_type const* null_mask = nullptr;
  std::size_t validity_byte_len       = 0;
  cudf::size_type null_count          = 0;
  switch (column.validity_kind) {
    case CUDF_INTEROP_VALIDITY_NONE: null_count = 0; break;
    case CUDF_INTEROP_VALIDITY_MASK: {
      auto const* validity = require_device_binding(
        binding_index,
        column.validity_buffer_id,
        "foreign owned-transfer import referenced a missing validity binding",
        "foreign owned-transfer import requires CUDA-device validity bindings");
      null_mask         = static_cast<cudf::bitmask_type const*>(validity->data);
      validity_byte_len = validity->byte_len;
      null_count        = column.null_count >= 0 ? static_cast<cudf::size_type>(column.null_count)
                                                 : static_cast<cudf::size_type>(-1);
      break;
    }
    default: throw cudf::logic_error("foreign owned-transfer import validity kind is unsupported");
  }

  return foreign_column_common{dtype, size, offset, null_mask, null_count, validity_byte_len};
}

cudf_interop_buffer_binding_t const* require_device_binding(binding_index_t const& index,
                                                            uint32_t id,
                                                            char const* missing_message,
                                                            char const* placement_message)
{
  auto const it = index.find(id);
  if (it == index.end()) { throw cudf::logic_error(missing_message); }
  if (it->second->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return it->second;
}

class foreign_fixed_width_column_handle_state final : public cudf_column_handle_state {
 public:
  foreign_fixed_width_column_handle_state(foreign_column_common common,
                                          void const* values,
                                          std::size_t values_byte_len)
    : _common(common), _values(values), _values_byte_len(values_byte_len)
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    return cudf::column_view(
      _common.dtype, _common.size, _values, _common.null_mask, _common.null_count, _common.offset);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    return checked_size_add(
      _values_byte_len, _common.validity_byte_len, "sizing foreign fixed-width allocation");
  }

 private:
  foreign_column_common _common;
  void const* _values;
  std::size_t _values_byte_len;
};

class foreign_binary_like_column_handle_state final : public cudf_column_handle_state {
 public:
  foreign_binary_like_column_handle_state(foreign_column_common common,
                                          void const* chars,
                                          std::size_t chars_byte_len,
                                          int32_t const* offsets,
                                          std::size_t offsets_byte_len)
    : _common(common),
      _chars(chars),
      _chars_byte_len(chars_byte_len),
      _offsets(offsets),
      _offsets_byte_len(offsets_byte_len)
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    std::vector<cudf::column_view> children;
    children.reserve(1);
    children.emplace_back(cudf::data_type{cudf::type_id::INT32},
                          require_size_type_from_size_t(_offsets_byte_len / sizeof(int32_t),
                                                        "foreign owned-transfer string offsets "
                                                        "binding exceeds cudf::size_type"),
                          _offsets,
                          nullptr,
                          0,
                          0);
    return cudf::column_view(_common.dtype,
                             _common.size,
                             _chars,
                             _common.null_mask,
                             _common.null_count,
                             _common.offset,
                             children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    auto total = checked_size_add(
      _chars_byte_len, _offsets_byte_len, "sizing foreign binary-like data and offsets");
    return checked_size_add(
      total, _common.validity_byte_len, "sizing foreign binary-like validity");
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
  foreign_list_column_handle_state(foreign_column_common common,
                                   int32_t const* offsets,
                                   std::size_t offsets_byte_len,
                                   std::unique_ptr<cudf_column_handle_state> child)
    : _common(common),
      _offsets(offsets),
      _offsets_byte_len(offsets_byte_len),
      _child(std::move(child))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(cudf::data_type{cudf::type_id::INT32},
                          require_size_type_from_size_t(_offsets_byte_len / sizeof(int32_t),
                                                        "foreign owned-transfer list offsets "
                                                        "binding exceeds cudf::size_type"),
                          _offsets,
                          nullptr,
                          0,
                          0);
    children.emplace_back(_child->view());
    return cudf::column_view(_common.dtype,
                             _common.size,
                             nullptr,
                             _common.null_mask,
                             _common.null_count,
                             _common.offset,
                             children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    auto total = checked_size_add(
      _offsets_byte_len, _common.validity_byte_len, "sizing foreign list offsets and validity");
    return checked_size_add(
      total, _child->device_allocated_size(), "sizing foreign list child allocation");
  }

 private:
  foreign_column_common _common;
  int32_t const* _offsets;
  std::size_t _offsets_byte_len;
  std::unique_ptr<cudf_column_handle_state> _child;
};

class foreign_struct_column_handle_state final : public cudf_column_handle_state {
 public:
  foreign_struct_column_handle_state(
    foreign_column_common common, std::vector<std::unique_ptr<cudf_column_handle_state>> children)
    : _common(common), _children(std::move(children))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    std::vector<cudf::column_view> children;
    children.reserve(_children.size());
    for (auto const& child : _children) {
      children.emplace_back(child->view());
    }
    return cudf::column_view(_common.dtype,
                             _common.size,
                             nullptr,
                             _common.null_mask,
                             _common.null_count,
                             _common.offset,
                             children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    std::size_t total = _common.validity_byte_len;
    for (auto const& child : _children) {
      total = checked_size_add(
        total, child->device_allocated_size(), "accumulating foreign struct child allocation");
    }
    return total;
  }

 private:
  foreign_column_common _common;
  std::vector<std::unique_ptr<cudf_column_handle_state>> _children;
};

class foreign_dictionary_column_handle_state final : public cudf_column_handle_state {
 public:
  foreign_dictionary_column_handle_state(foreign_column_common common,
                                         int32_t const* indices,
                                         std::size_t indices_byte_len,
                                         std::unique_ptr<cudf_column_handle_state> keys)
    : _common(common),
      _indices(indices),
      _indices_byte_len(indices_byte_len),
      _keys(std::move(keys))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return _common.size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(cudf::data_type{cudf::type_id::INT32},
                          require_size_type_from_size_t(_indices_byte_len / sizeof(int32_t),
                                                        "foreign owned-transfer dictionary indices "
                                                        "binding exceeds cudf::size_type"),
                          _indices,
                          nullptr,
                          0,
                          0);
    children.emplace_back(_keys->view());
    return cudf::column_view(_common.dtype,
                             _common.size,
                             nullptr,
                             _common.null_mask,
                             _common.null_count,
                             _common.offset,
                             children);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    return cudf_column_view_payload_size_bytes(view(), stream);
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    auto total = checked_size_add(_indices_byte_len,
                                  _common.validity_byte_len,
                                  "sizing foreign dictionary indices and validity");
    return checked_size_add(
      total, _keys->device_allocated_size(), "sizing foreign dictionary keys allocation");
  }

 private:
  foreign_column_common _common;
  int32_t const* _indices;
  std::size_t _indices_byte_len;
  std::unique_ptr<cudf_column_handle_state> _keys;
};

std::unique_ptr<cudf_column_handle_state> build_foreign_column_handle_state(
  uint32_t root_column_id, column_index_t const& column_index, binding_index_t const& binding_index)
{
  auto const column_it = column_index.find(root_column_id);
  CUDF_EXPECTS(column_it != column_index.end(),
               "foreign owned-transfer import referenced a missing semantic column");
  auto const* column = column_it->second;
  auto const common  = build_foreign_column_common(*column, binding_index);

  switch (column->layout_kind) {
    case CUDF_INTEROP_LAYOUT_FIXED_WIDTH: {
      CUDF_EXPECTS(cudf::is_fixed_width(common.dtype),
                   "foreign owned-transfer fixed-width layout requires a "
                   "fixed-width type");
      auto const* values = require_device_binding(binding_index,
                                                  column->values_buffer_id,
                                                  "foreign owned-transfer import referenced a "
                                                  "missing fixed-width values binding",
                                                  "foreign owned-transfer import requires "
                                                  "CUDA-device fixed-width value bindings");
      return std::make_unique<foreign_fixed_width_column_handle_state>(
        common, values->data, values->byte_len);
    }

    case CUDF_INTEROP_LAYOUT_BINARY_LIKE: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::STRING,
                   "foreign owned-transfer binary-like layout requires STRING type");
      auto const* offsets = require_device_binding(binding_index,
                                                   column->offsets_buffer_id,
                                                   "foreign owned-transfer import referenced a "
                                                   "missing binary-like offsets binding",
                                                   "foreign owned-transfer import requires "
                                                   "CUDA-device binary-like offsets bindings");
      CUDF_EXPECTS(offsets->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer binary-like offsets binding must be "
                   "32-bit aligned");
      auto const* chars = require_device_binding(binding_index,
                                                 column->data_buffer_id,
                                                 "foreign owned-transfer import referenced a "
                                                 "missing binary-like chars binding",
                                                 "foreign owned-transfer import requires "
                                                 "CUDA-device binary-like chars bindings");
      return std::make_unique<foreign_binary_like_column_handle_state>(
        common,
        chars->data,
        chars->byte_len,
        static_cast<int32_t const*>(offsets->data),
        offsets->byte_len);
    }

    case CUDF_INTEROP_LAYOUT_LIST: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::LIST,
                   "foreign owned-transfer list layout requires LIST type");
      auto const* offsets = require_device_binding(binding_index,
                                                   column->offsets_buffer_id,
                                                   "foreign owned-transfer import referenced a "
                                                   "missing list offsets binding",
                                                   "foreign owned-transfer import requires "
                                                   "CUDA-device list offsets bindings");
      CUDF_EXPECTS(offsets->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer list offsets binding must be 32-bit aligned");
      auto child =
        build_foreign_column_handle_state(column->child_column_id, column_index, binding_index);
      return std::make_unique<foreign_list_column_handle_state>(
        common, static_cast<int32_t const*>(offsets->data), offsets->byte_len, std::move(child));
    }

    case CUDF_INTEROP_LAYOUT_STRUCT: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::STRUCT,
                   "foreign owned-transfer struct layout requires STRUCT type");
      CUDF_EXPECTS(column->num_child_column_ids > 0,
                   "foreign owned-transfer struct layout must reference at least "
                   "one child");
      CUDF_EXPECTS(column->child_column_ids != nullptr,
                   "foreign owned-transfer struct child ids pointer is null");
      std::vector<std::unique_ptr<cudf_column_handle_state>> children;
      children.reserve(static_cast<std::size_t>(column->num_child_column_ids));
      for (int32_t index = 0; index < column->num_child_column_ids; ++index) {
        children.emplace_back(build_foreign_column_handle_state(
          column->child_column_ids[index], column_index, binding_index));
      }
      return std::make_unique<foreign_struct_column_handle_state>(common, std::move(children));
    }

    case CUDF_INTEROP_LAYOUT_DICTIONARY: {
      CUDF_EXPECTS(common.dtype.id() == cudf::type_id::DICTIONARY32,
                   "foreign owned-transfer dictionary layout requires DICTIONARY32 type");
      auto const* indices = require_device_binding(binding_index,
                                                   column->indices_buffer_id,
                                                   "foreign owned-transfer import referenced a "
                                                   "missing dictionary indices binding",
                                                   "foreign owned-transfer import requires "
                                                   "CUDA-device dictionary indices bindings");
      CUDF_EXPECTS(indices->byte_len % sizeof(int32_t) == 0,
                   "foreign owned-transfer dictionary indices binding must be "
                   "32-bit aligned");
      auto keys =
        build_foreign_column_handle_state(column->keys_column_id, column_index, binding_index);
      return std::make_unique<foreign_dictionary_column_handle_state>(
        common, static_cast<int32_t const*>(indices->data), indices->byte_len, std::move(keys));
    }

    default: CUDF_FAIL("foreign owned-transfer import layout kind is unsupported");
  }
}

class column_state_table_handle_state final : public cudf_table_handle_state {
 public:
  column_state_table_handle_state(cudf::size_type num_rows,
                                  std::vector<std::unique_ptr<cudf_column_handle_state>> columns)
    : _num_rows(num_rows), _columns(std::move(columns))
  {
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type num_columns() const override
  {
    return static_cast<cudf::size_type>(_columns.size());
  }

  [[nodiscard]] cudf::size_type num_rows() const override { return _num_rows; }

  [[nodiscard]] cudf::table_view view() const override
  {
    std::vector<cudf::column_view> views;
    views.reserve(_columns.size());
    for (auto const& column : _columns) {
      views.push_back(column->view());
    }
    return cudf::table_view{views};
  }

  [[nodiscard]] std::unique_ptr<cudf::table> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    auto table_view = view();
    return std::make_unique<cudf::table>(table_view, stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view stream) const override
  {
    std::size_t total = 0;
    for (auto const& column : _columns) {
      total = checked_size_add(
        total, column->payload_size(stream), "accumulating foreign table column payload");
    }
    return total;
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    std::size_t total = 0;
    for (auto const& column : _columns) {
      total = checked_size_add(
        total, column->device_allocated_size(), "accumulating foreign table column allocation");
    }
    return total;
  }

  [[nodiscard]] std::optional<cudf_handle_provenance> destroy_provenance() const noexcept override
  {
    for (auto const& column : _columns) {
      if (auto provenance = column->destroy_provenance()) { return provenance; }
    }
    return std::nullopt;
  }

  std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns() override
  {
    return std::move(_columns);
  }

  void restore_released_columns(
    cudf::size_type num_rows,
    std::vector<std::unique_ptr<cudf_column_handle_state>> columns) override
  {
    _num_rows = num_rows;
    _columns  = std::move(columns);
  }

 private:
  cudf::size_type _num_rows;
  std::vector<std::unique_ptr<cudf_column_handle_state>> _columns;
};
}  // namespace

std::unique_ptr<cudf::column> cudf_column_handle_state::release_native_column()
{
  throw cudf::logic_error("column handle is not native-backed");
}

std::optional<cudf_handle_provenance> cudf_column_handle_state::destroy_provenance() const noexcept
{
  return std::nullopt;
}

std::vector<std::unique_ptr<cudf_column_handle_state>> cudf_table_handle_state::release_columns()
{
  throw cudf::logic_error("table handle cannot release columns");
}

void cudf_table_handle_state::restore_released_columns(
  cudf::size_type, std::vector<std::unique_ptr<cudf_column_handle_state>>)
{
  throw cudf::logic_error("table handle cannot restore released columns");
}

std::optional<cudf_handle_provenance> cudf_table_handle_state::destroy_provenance() const noexcept
{
  return std::nullopt;
}

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column)
{
  return cudf_release_column_handle(std::move(column), cudf_capture_current_handle_provenance());
}

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column,
                                              cudf_handle_provenance provenance)
{
  return cudf_release_column_handle(
    std::make_unique<native_column_handle_state>(std::move(column), provenance));
}

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf_column_handle_state> column)
{
  if (cudf::rust::test_support::take_column_handle_release_failure()) { throw std::bad_alloc{}; }
  return reinterpret_cast<cudf_column_handle>(column.release());
}

cudf_column_handle cudf_publish_column_handle(std::unique_ptr<cudf_column_handle_state>& column)
{
  if (cudf::rust::test_support::take_column_handle_release_failure()) { throw std::bad_alloc{}; }
  return reinterpret_cast<cudf_column_handle>(column.release());
}

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table)
{
  return cudf_release_table_handle(std::move(table), cudf_capture_current_handle_provenance());
}

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table,
                                            cudf_handle_provenance provenance)
{
  return cudf_release_table_handle(
    std::make_unique<native_table_handle_state>(std::move(table), provenance));
}

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf_table_handle_state> table)
{
  return reinterpret_cast<cudf_table_handle>(table.release());
}

cudf_status_t cudf_destroy_column_handle_under_provenance(cudf_column_handle handle) noexcept
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  try {
    auto* state = static_cast<cudf_column_handle_state*>(handle);
    if (auto provenance = state->destroy_provenance()) {
      return cudf_destroy_under_provenance(*provenance, [state]() { delete state; });
    }
    delete state;
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_status_t cudf_destroy_table_handle_under_provenance(cudf_table_handle handle) noexcept
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  try {
    auto* state = static_cast<cudf_table_handle_state*>(handle);
    if (auto provenance = state->destroy_provenance()) {
      return cudf_destroy_under_provenance(*provenance, [state]() { delete state; });
    }
    delete state;
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_column_handle_state& cudf_column_state_from_handle(cudf_column_handle handle)
{
  return *static_cast<cudf_column_handle_state*>(handle);
}

cudf_table_handle_state& cudf_table_state_from_handle(cudf_table_handle handle)
{
  return *static_cast<cudf_table_handle_state*>(handle);
}

std::unique_ptr<cudf_column_handle_state> cudf_take_column_state(cudf_column_handle handle)
{
  return std::unique_ptr<cudf_column_handle_state>(static_cast<cudf_column_handle_state*>(handle));
}

std::unique_ptr<cudf_table_handle_state> cudf_take_table_state(cudf_table_handle handle)
{
  return std::unique_ptr<cudf_table_handle_state>(static_cast<cudf_table_handle_state*>(handle));
}

std::unique_ptr<cudf_column_handle_state> cudf_make_foreign_column_handle_state(
  uint32_t root_column_id,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings)
{
  auto const column_index  = build_column_index(semantic_columns, num_semantic_columns);
  auto const binding_index = build_binding_index(bindings, num_bindings);
  return build_foreign_column_handle_state(root_column_id, column_index, binding_index);
}

std::unique_ptr<cudf_table_handle_state> cudf_make_foreign_table_handle_state(
  cudf_interop_semantic_table_t const* table,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings)
{
  CUDF_EXPECTS(table != nullptr, "foreign owned-transfer table descriptor is null");
  CUDF_EXPECTS(table->num_rows >= 0, "foreign owned-transfer table row count must be >= 0");
  CUDF_EXPECTS(table->num_columns >= 0, "foreign owned-transfer table column count must be >= 0");
  if (table->num_columns > 0) {
    CUDF_EXPECTS(table->column_ids != nullptr,
                 "foreign owned-transfer table column ids pointer is null");
  }

  auto const expected_rows = static_cast<cudf::size_type>(table->num_rows);
  std::vector<std::unique_ptr<cudf_column_handle_state>> columns;
  columns.reserve(static_cast<std::size_t>(table->num_columns));
  for (int32_t index = 0; index < table->num_columns; ++index) {
    auto column = cudf_make_foreign_column_handle_state(
      table->column_ids[index], semantic_columns, num_semantic_columns, bindings, num_bindings);
    CUDF_EXPECTS(column->size() == expected_rows,
                 "foreign owned-transfer table row count must match every "
                 "top-level column length");
    columns.push_back(std::move(column));
  }

  return std::make_unique<column_state_table_handle_state>(expected_rows, std::move(columns));
}

std::unique_ptr<cudf_table_handle_state> cudf_make_table_handle_state_from_columns(
  cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns)
{
  return std::make_unique<column_state_table_handle_state>(num_rows, std::move(columns));
}

std::unique_ptr<cudf_table_handle_state> cudf_make_foreign_table_handle_state_from_columns(
  cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns)
{
  return cudf_make_table_handle_state_from_columns(num_rows, std::move(columns));
}
