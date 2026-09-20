#include "owning_handle.hpp"

#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void collect_backing_allocation_pointers(cudf::column_view const& column,
                                         std::vector<void*>& pointers)
{
  auto* data = const_cast<void*>(static_cast<void const*>(column.head<std::uint8_t>()));
  if (data != nullptr) { pointers.push_back(data); }
  auto* null_mask = const_cast<void*>(static_cast<void const*>(column.null_mask()));
  if (null_mask != nullptr) { pointers.push_back(null_mask); }
  for (cudf::size_type index = 0; index < column.num_children(); ++index) {
    collect_backing_allocation_pointers(column.child(index), pointers);
  }
}

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
      auto const num_rows = _table->num_rows();
      auto columns        = _table->release();
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
        _table = std::make_unique<cudf::table>(std::move(restored_columns), num_rows);
        throw;
      }
      return result;
    });
  }

  void restore_released_columns(
    cudf::size_type num_rows,
    std::vector<std::unique_ptr<cudf_column_handle_state>> columns) override
  {
    cudf_invoke_under_provenance(_provenance, [&] {
      std::vector<std::unique_ptr<cudf::column>> native_columns;
      native_columns.reserve(columns.size());
      for (auto& column : columns) {
        native_columns.push_back(column->release_native_column());
      }
      _table = std::make_unique<cudf::table>(std::move(native_columns), num_rows);
    });
  }

 private:
  cudf_handle_provenance _provenance;
  std::unique_ptr<cudf::table> _table;
};

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
    return cudf::table_view{views, _num_rows};
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

std::vector<void*> cudf_column_handle_state::backing_allocation_pointers() const
{
  std::vector<void*> pointers;
  collect_backing_allocation_pointers(view(), pointers);
  std::sort(pointers.begin(), pointers.end());
  pointers.erase(std::unique(pointers.begin(), pointers.end()), pointers.end());
  return pointers;
}

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

std::unique_ptr<cudf_table_handle_state> cudf_make_table_handle_state_from_columns(
  cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns)
{
  return std::make_unique<column_state_table_handle_state>(num_rows, std::move(columns));
}
